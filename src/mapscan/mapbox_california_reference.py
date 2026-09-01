"""Build a hash-pinned California reference from the live Mapbox basemap.

The final MapScan viewer uses ``mapbox/light-v11``, whose composite vector
source is ``mapbox.mapbox-streets-v8``.  This module pins the style, TileJSON,
and every vector tile used for California, then derives three independent
pieces of alignment evidence from the source layers used by that basemap:

* ``water`` polygons decide the Pacific coast, bays, lakes, and islands;
* non-maritime ``admin_level`` 0/1 lines close California's land borders; and
* non-maritime ``admin_level`` 2 lines provide the county validation network.

No previously approved MapScan border, Census geometry, or source-map raster
is an input.  A deterministic flood fill from an interior California seed
selects the state land enclosed by Mapbox water and administrative lines.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image
from pyproj import Transformer
from shapely import affinity, box
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPolygon,
    Polygon,
    shape,
)

from .mapbox_water_reference import (
    DEFAULT_MAP_ID,
    DEFAULT_STYLE_ID,
    WEB_MERCATOR_HALF_WORLD,
    _decode_tile,
    _download,
    _tile_coordinates,
)


DEFAULT_BOUNDS_WGS84 = (-125.0, 31.8, -113.3, 42.3)
DEFAULT_ZOOM = 9
DEFAULT_CALIFORNIA_SEED_WGS84 = (-119.5, 37.2)
DEFAULT_WATER_LAYER = "water"
DEFAULT_ADMIN_LAYER = "admin"
DEFAULT_CONTROL_POINTS = (
    ("San Francisco", -122.4194, 37.7749, True, True, False),
    ("Los Angeles", -118.2437, 34.0522, True, True, False),
    ("Sacramento", -121.4944, 38.5816, True, True, False),
    ("San Diego", -117.1611, 32.7157, True, True, False),
    ("Fresno", -119.7871, 36.7378, True, True, False),
    ("Reno", -119.8138, 39.5296, False, False, False),
    ("Las Vegas", -115.1398, 36.1699, False, False, False),
    ("Pacific Ocean", -123.0, 36.0, False, False, True),
    ("San Francisco Bay", -122.35, 37.85, False, False, True),
    ("Monterey Bay", -121.9, 36.75, False, False, True),
    ("Lake Tahoe", -120.0, 39.1, False, False, True),
    ("Salton Sea", -115.8, 33.3, False, True, True),
    ("Santa Catalina Island", -118.416, 33.383, True, True, False),
    ("San Clemente Island", -118.5, 32.9, True, True, False),
    ("Farallon Islands", -123.0, 37.7, True, True, False),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _aggregate_hash(records: Iterable[Dict[str, object]]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: str(item["path"])):
        digest.update(str(record["path"]).encode())
        digest.update(b"\0")
        digest.update(str(record["sha256"]).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _validate_grid(grid: Dict[str, object]) -> Tuple[int, int, Tuple[float, ...]]:
    if str(grid.get("crs")) != "EPSG:3857":
        raise ValueError("Mapbox California reference requires an EPSG:3857 grid")
    width, height = int(grid["width"]), int(grid["height"])
    if width <= 0 or height <= 0:
        raise ValueError("Mapbox California reference grid dimensions must be positive")
    bounds = tuple(float(value) for value in grid["bounds"])
    if len(bounds) != 4 or bounds[0] >= bounds[2] or bounds[1] >= bounds[3]:
        raise ValueError("Mapbox California reference grid bounds are invalid")
    return width, height, bounds


def _lines(geometry):
    if isinstance(geometry, LineString):
        yield geometry
    elif isinstance(geometry, (MultiLineString, GeometryCollection)):
        for part in geometry.geoms:
            yield from _lines(part)


def _polygons(geometry):
    if isinstance(geometry, Polygon):
        yield geometry
    elif isinstance(geometry, (MultiPolygon, GeometryCollection)):
        for part in geometry.geoms:
            yield from _polygons(part)


def _worldview_matches(properties: Dict[str, object]) -> bool:
    worldview = str(properties.get("worldview", "all"))
    values = {value.strip() for value in worldview.split(",")}
    return bool(values & {"all", "US"})


def _admin_matches(properties: Dict[str, object], levels: Sequence[int]) -> bool:
    try:
        level = int(properties.get("admin_level", -1))
    except (TypeError, ValueError):
        return False
    iso = str(properties.get("iso_3166_1", ""))
    return (
        level in levels
        and "US" in iso.split("-")
        and str(properties.get("maritime", "false")) == "false"
        and str(properties.get("disputed", "false")) == "false"
        and _worldview_matches(properties)
    )


def _save_mask(path: Path, mask: np.ndarray) -> Dict[str, object]:
    Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255).save(path)
    return {
        "path": path.name,
        "sha256": _sha256(path),
        "pixel_count": int(np.count_nonzero(mask)),
    }


def _save_overlay(
    path: Path, mask: np.ndarray, color: Tuple[int, int, int, int]
) -> Dict[str, object]:
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    rgba[np.asarray(mask, dtype=bool)] = color
    Image.fromarray(rgba).save(path)
    return {
        "path": path.name,
        "sha256": _sha256(path),
        "pixel_count": int(np.count_nonzero(mask)),
    }


def _flood_component(passable: np.ndarray, seed: Tuple[int, int]) -> np.ndarray:
    height, width = passable.shape
    x, y = seed
    if not (0 <= x < width and 0 <= y < height) or not passable[y, x]:
        raise ValueError("California seed does not fall on passable Mapbox land")
    labels_count, labels = cv2.connectedComponents(
        np.asarray(passable, dtype=np.uint8), connectivity=4
    )
    if labels_count <= 1:
        raise ValueError("Mapbox barriers leave no California land component")
    return labels == int(labels[y, x])


def _enclosed_islands(
    passable: np.ndarray,
    mainland: np.ndarray,
    *,
    maximum_distance_px: float,
) -> np.ndarray:
    """Retain Pacific-side Mapbox islands near the selected mainland.

    A distance-only rule is insufficient on a regional target grid: detached
    land components in Nevada and Arizona can be closer to California than the
    Channel Islands are.  The state flood fill already identifies the mainland,
    so detached components must also lie west of the local mainland envelope at
    the component's latitude.  This is a topology rule derived only from the
    pinned Mapbox water/admin bytes; it does not use a second California border.
    """

    labels_count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        np.asarray(passable, dtype=np.uint8), connectivity=4
    )
    distance = cv2.distanceTransform((~mainland).astype(np.uint8), cv2.DIST_L2, 5)
    retained = np.zeros_like(mainland)
    height, width = mainland.shape
    mainland_label = int(labels[np.argwhere(mainland)[0][0], np.argwhere(mainland)[0][1]])
    for label in range(1, labels_count):
        if label == mainland_label:
            continue
        left, top, component_width, component_height, _ = stats[label]
        touches_edge = (
            left == 0
            or top == 0
            or left + component_width == width
            or top + component_height == height
        )
        component = labels == label
        if touches_edge or float(distance[component].min()) > maximum_distance_px:
            continue

        # Compare the detached component with the western envelope of the
        # mainland over the same latitude band.  A small vertical pad protects
        # tiny islands whose rasterized row falls just beyond a nearby cape.
        vertical_pad = max(2, int(round(0.01 * height)))
        row_start = max(0, int(top) - vertical_pad)
        row_stop = min(height, int(top + component_height) + vertical_pad)
        local_rows, local_columns = np.nonzero(mainland[row_start:row_stop])
        if local_columns.size == 0:
            continue
        local_western_envelope = float(np.quantile(local_columns, 0.02))
        component_center_x = float(centroids[label, 0])
        side_tolerance = max(2.0, 0.006 * width)
        if component_center_x <= local_western_envelope + side_tolerance:
            retained |= component
    return retained


def _fill_internal_holes(mask: np.ndarray) -> np.ndarray:
    """Fill water/void components not connected to the target-grid exterior."""

    inverse = ~np.asarray(mask, dtype=bool)
    _, labels = cv2.connectedComponents(inverse.astype(np.uint8), connectivity=4)
    exterior_labels = np.unique(
        np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1]))
    )
    exterior = np.isin(labels, exterior_labels)
    return ~exterior


def build_mapbox_california_reference(
    output_root: Path,
    access_token: str,
    grid: Dict[str, object],
    *,
    bounds_wgs84: Tuple[float, float, float, float] = DEFAULT_BOUNDS_WGS84,
    zoom: int = DEFAULT_ZOOM,
    map_id: str = DEFAULT_MAP_ID,
    style_id: str = DEFAULT_STYLE_ID,
    california_seed_wgs84: Tuple[float, float] = DEFAULT_CALIFORNIA_SEED_WGS84,
    water_layer: str = DEFAULT_WATER_LAYER,
    admin_layer: str = DEFAULT_ADMIN_LAYER,
    barrier_width_px: int = 3,
    maximum_island_distance_m: float = 160_000.0,
    validate_controls: bool = True,
    force: bool = False,
    pinned_source_manifest_path: Path | None = None,
) -> Dict[str, object]:
    """Pin Mapbox basemap inputs and derive California alignment evidence."""

    if not access_token and pinned_source_manifest_path is None:
        raise ValueError("A Mapbox access token is required")
    if not 0 <= zoom <= 22:
        raise ValueError("Mapbox California reference zoom must be between zero and 22")
    if barrier_width_px < 1:
        raise ValueError("Mapbox administrative barrier width must be positive")
    if maximum_island_distance_m <= 0:
        raise ValueError("Mapbox island distance must be positive")
    width, height, target_bounds = _validate_grid(grid)
    output_root = output_root.resolve()
    if output_root.exists():
        if not force:
            raise FileExistsError(
                f"Mapbox California reference already exists: {output_root}"
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    source_manifest = None
    source_manifest_path = None
    source_grid_supersampling = 1
    if pinned_source_manifest_path is not None:
        source_manifest_path = Path(pinned_source_manifest_path).resolve()
        source_manifest = json.loads(source_manifest_path.read_text())
        if source_manifest.get("status") != "pinned_reference":
            raise ValueError("Source Mapbox manifest is not a pinned reference")
        source_grid = source_manifest.get("target_grid", {})
        source_width, source_height, source_bounds = _validate_grid(source_grid)
        if source_bounds != target_bounds:
            raise ValueError("Derived Mapbox reference must preserve the source bounds")
        factor_x = (width - 1) / max(source_width - 1, 1)
        factor_y = (height - 1) / max(source_height - 1, 1)
        rounded_factor = round(factor_x)
        if (
            rounded_factor < 1
            or not math.isclose(factor_x, rounded_factor, abs_tol=1e-12)
            or not math.isclose(factor_y, rounded_factor, abs_tol=1e-12)
        ):
            raise ValueError(
                "Derived Mapbox reference grid must be an integer, corner-preserving "
                "supersample of the source grid"
            )
        source_grid_supersampling = int(rounded_factor)
        if int(source_manifest.get("zoom", -1)) != zoom:
            raise ValueError("Derived Mapbox reference must preserve the source zoom")
        source_root = source_manifest_path.parent
        source_style = source_manifest.get("style", {})
        source_tileset = source_manifest.get("tileset", {})
        if str(source_style.get("id")) != style_id:
            raise ValueError("Derived Mapbox reference must preserve the source style")
        if str(source_tileset.get("id")) != map_id:
            raise ValueError("Derived Mapbox reference must preserve the source tileset")

    try:
        owner, style_name = style_id.split("/", 1)
    except ValueError as error:
        raise ValueError("Mapbox style id must be owner/style") from error
    style_url = f"https://api.mapbox.com/styles/v1/{owner}/{style_name}"
    style_path = output_root / "style.json"
    if source_manifest is None:
        style_path.write_bytes(_download(style_url, access_token))
    else:
        source_style_path = source_root / str(source_manifest["style"]["path"])
        if _sha256(source_style_path) != str(source_manifest["style"]["sha256"]):
            raise ValueError("Pinned source style hash does not match its manifest")
        shutil.copy2(source_style_path, style_path)
    style = json.loads(style_path.read_text())
    source_ids = {
        str(layer.get("source"))
        for layer in style.get("layers", [])
        if layer.get("source-layer") in {water_layer, admin_layer}
    }
    source_urls = {
        str(style.get("sources", {}).get(source_id, {}).get("url", ""))
        for source_id in source_ids
    }
    if not any(map_id in url for url in source_urls):
        raise ValueError("Pinned Mapbox style does not use the declared basemap source")

    tilejson_url = f"https://api.mapbox.com/v4/{map_id}.json"
    tilejson_path = output_root / "tilejson.json"
    if source_manifest is None:
        tilejson_path.write_bytes(_download(tilejson_url, access_token))
    else:
        source_tilejson_path = source_root / str(
            source_manifest["tileset"]["tilejson_path"]
        )
        if _sha256(source_tilejson_path) != str(
            source_manifest["tileset"]["tilejson_sha256"]
        ):
            raise ValueError("Pinned source TileJSON hash does not match its manifest")
        shutil.copy2(source_tilejson_path, tilejson_path)
    tilejson = json.loads(tilejson_path.read_text())
    vector_layers = {str(layer.get("id")) for layer in tilejson.get("vector_layers", [])}
    if not {water_layer, admin_layer}.issubset(vector_layers):
        raise ValueError("Mapbox basemap is missing water or administrative vector layers")

    x_range, y_range = _tile_coordinates(bounds_wgs84, zoom)
    records = []
    if source_manifest is None:
        tile_coordinates = [
            (tile_x, tile_y) for tile_x in x_range for tile_y in y_range
        ]
        source_records = {}
    else:
        expected_tile_range = source_manifest.get("tile_range", {})
        actual_tile_range = {
            "minimum_x": x_range.start,
            "maximum_x": x_range.stop - 1,
            "minimum_y": y_range.start,
            "maximum_y": y_range.stop - 1,
        }
        if expected_tile_range != actual_tile_range:
            raise ValueError("Derived Mapbox reference must preserve the source tile range")
        source_records = {
            (int(record["x"]), int(record["y"])): record
            for record in source_manifest.get("tiles", [])
        }
        tile_coordinates = sorted(source_records)
        if len(tile_coordinates) != int(source_manifest.get("tile_count", -1)):
            raise ValueError("Pinned source tile inventory is incomplete")
    for tile_x, tile_y in tile_coordinates:
            relative = Path("tiles") / str(zoom) / str(tile_x) / f"{tile_y}.mvt"
            target = output_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            public_url = (
                f"https://api.mapbox.com/v4/{map_id}/{zoom}/{tile_x}/{tile_y}.mvt"
            )
            if source_manifest is None:
                target.write_bytes(_download(public_url, access_token))
            else:
                source_record = source_records[(tile_x, tile_y)]
                source_tile = source_root / str(source_record["path"])
                if _sha256(source_tile) != str(source_record["sha256"]):
                    raise ValueError(
                        f"Pinned source tile hash mismatch: {source_record['path']}"
                    )
                shutil.copy2(source_tile, target)
            decoded = _decode_tile(target.read_bytes())
            water = decoded.get(water_layer, {})
            admin = decoded.get(admin_layer, {})
            record = {
                    "z": zoom,
                    "x": tile_x,
                    "y": tile_y,
                    "path": str(relative),
                    "sha256": _sha256(target),
                    "byte_count": target.stat().st_size,
                    "url_without_token": public_url,
                    "water_feature_count": len(water.get("features", [])),
                    "admin_feature_count": len(admin.get("features", [])),
                }
            if source_manifest is not None:
                expected_record = source_records[(tile_x, tile_y)]
                if (
                    record["sha256"] != expected_record["sha256"]
                    or record["byte_count"] != expected_record["byte_count"]
                ):
                    raise ValueError(
                        f"Copied tile bytes changed: {expected_record['path']}"
                    )
            records.append(record)

    water_mask = np.zeros((height, width), dtype=np.uint8)
    state_lines = np.zeros((height, width), dtype=np.uint8)
    county_lines = np.zeros((height, width), dtype=np.uint8)
    minimum_x, minimum_y, maximum_x, maximum_y = target_bounds
    pixel_width = (maximum_x - minimum_x) / width
    pixel_height = (maximum_y - minimum_y) / height
    tile_span = WEB_MERCATOR_HALF_WORLD * 2 / (1 << zoom)
    feature_counts = {"water": 0, "state_or_country": 0, "county": 0}

    def pixel_points(coordinates) -> np.ndarray:
        values = np.asarray(coordinates, dtype=np.float64)
        pixel_x = (values[:, 0] - minimum_x) / pixel_width - 0.5
        pixel_y = (maximum_y - values[:, 1]) / pixel_height - 0.5
        return np.rint(np.column_stack((pixel_x, pixel_y))).astype(np.int32)

    for record in records:
        tile_x, tile_y = int(record["x"]), int(record["y"])
        decoded = _decode_tile((output_root / str(record["path"])).read_bytes())
        left = -WEB_MERCATOR_HALF_WORLD + tile_x * tile_span
        top = WEB_MERCATOR_HALF_WORLD - tile_y * tile_span
        tile_bounds = box(left, top - tile_span, left + tile_span, top)
        for layer_name in (water_layer, admin_layer):
            layer = decoded.get(layer_name)
            if not isinstance(layer, dict):
                continue
            extent = int(layer["extent"])
            for feature in layer.get("features", []):
                properties = feature.get("properties", {})
                target_mask = None
                if layer_name == water_layer:
                    target_mask = water_mask
                    feature_counts["water"] += 1
                elif _admin_matches(properties, (0, 1)):
                    target_mask = state_lines
                    feature_counts["state_or_country"] += 1
                elif _admin_matches(properties, (2,)):
                    target_mask = county_lines
                    feature_counts["county"] += 1
                if target_mask is None:
                    continue
                geometry = shape(feature["geometry"])
                geometry = affinity.scale(
                    geometry,
                    xfact=tile_span / extent,
                    yfact=-tile_span / extent,
                    origin=(0, 0),
                )
                geometry = affinity.translate(geometry, xoff=left, yoff=top)
                geometry = geometry.intersection(tile_bounds)
                if geometry.is_empty:
                    continue
                if layer_name == water_layer:
                    for polygon in _polygons(geometry):
                        cv2.fillPoly(
                            target_mask,
                            [pixel_points(polygon.exterior.coords)],
                            1,
                            lineType=cv2.LINE_8,
                        )
                        for interior in polygon.interiors:
                            cv2.fillPoly(
                                target_mask,
                                [pixel_points(interior.coords)],
                                0,
                                lineType=cv2.LINE_8,
                            )
                else:
                    for line in _lines(geometry):
                        cv2.polylines(
                            target_mask,
                            [pixel_points(line.coords)],
                            False,
                            1,
                            thickness=barrier_width_px,
                            lineType=cv2.LINE_8,
                        )

    if not np.any(water_mask) or not np.any(state_lines) or not np.any(county_lines):
        raise ValueError("Mapbox tiles did not yield complete California evidence")
    passable = ~(water_mask.astype(bool) | state_lines.astype(bool))
    seed_x_mercator, seed_y_mercator = Transformer.from_crs(
        "EPSG:4326", "EPSG:3857", always_xy=True
    ).transform(*california_seed_wgs84)
    seed = (
        int(math.floor((seed_x_mercator - minimum_x) / pixel_width)),
        int(math.floor((maximum_y - seed_y_mercator) / pixel_height)),
    )
    mainland = _flood_component(passable, seed)
    maximum_island_distance_px = maximum_island_distance_m / max(
        pixel_width, pixel_height
    )
    islands = _enclosed_islands(
        passable,
        mainland,
        maximum_distance_px=maximum_island_distance_px,
    )
    california_land = mainland | islands
    california_interior = _fill_internal_holes(california_land)
    county_lines = county_lines.astype(bool) & cv2.dilate(
        california_interior.astype(np.uint8), np.ones((3, 3), dtype=np.uint8)
    ).astype(bool)
    state_coast = cv2.morphologyEx(
        california_interior.astype(np.uint8),
        cv2.MORPH_GRADIENT,
        np.ones((3, 3), dtype=np.uint8),
    ).astype(bool)

    control_results = []
    controls_passed = True
    if validate_controls:
        transformer = Transformer.from_crs(
            "EPSG:4326", "EPSG:3857", always_xy=True
        )
        for name, longitude, latitude, expected_land, expected_interior, expected_water in DEFAULT_CONTROL_POINTS:
            world_x, world_y = transformer.transform(longitude, latitude)
            pixel_x = int(math.floor((world_x - minimum_x) / pixel_width))
            pixel_y = int(math.floor((maximum_y - world_y) / pixel_height))
            if not (0 <= pixel_x < width and 0 <= pixel_y < height):
                control_results.append(
                    {
                        "name": name,
                        "wgs84": [longitude, latitude],
                        "status": "outside_target_grid",
                    }
                )
                continue
            observed = {
                "land": bool(california_land[pixel_y, pixel_x]),
                "state_interior": bool(california_interior[pixel_y, pixel_x]),
                "water": bool(water_mask[pixel_y, pixel_x]),
            }
            expected = {
                "land": expected_land,
                "state_interior": expected_interior,
                "water": expected_water,
            }
            passed = observed == expected
            controls_passed &= passed
            control_results.append(
                {
                    "name": name,
                    "wgs84": [longitude, latitude],
                    "pixel": [pixel_x, pixel_y],
                    "expected": expected,
                    "observed": observed,
                    "status": "pass" if passed else "fail",
                }
            )
        if not controls_passed:
            failed = [
                item["name"] for item in control_results if item["status"] == "fail"
            ]
            raise ValueError(
                "Pinned Mapbox California reference failed controls: "
                + ", ".join(failed)
            )
    water_edge = cv2.morphologyEx(
        california_land.astype(np.uint8),
        cv2.MORPH_GRADIENT,
        np.ones((3, 3), dtype=np.uint8),
    ).astype(bool)

    artifacts = {
        "water_mask": _save_mask(output_root / "mapbox-water-mask.png", water_mask),
        "state_land_mask": _save_mask(
            output_root / "california-land-mask.png", california_land
        ),
        "state_interior_mask": _save_mask(
            output_root / "california-state-interior-mask.png", california_interior
        ),
        "state_coast_overlay": _save_overlay(
            output_root / "california-state-coast-overlay.png",
            state_coast,
            (101, 255, 155, 255),
        ),
        "water_edge_overlay": _save_overlay(
            output_root / "california-water-edge-overlay.png",
            water_edge,
            (43, 112, 164, 255),
        ),
        "county_overlay": _save_overlay(
            output_root / "california-county-overlay.png",
            county_lines,
            (58, 209, 255, 255),
        ),
        "admin_barrier_mask": _save_mask(
            output_root / "mapbox-admin-barrier-mask.png", state_lines
        ),
    }
    diagnostic = np.zeros((height, width, 4), dtype=np.uint8)
    diagnostic[water_mask.astype(bool)] = (38, 90, 130, 255)
    diagnostic[california_land] = (45, 55, 48, 255)
    diagnostic[county_lines] = (58, 209, 255, 255)
    diagnostic[state_coast] = (101, 255, 155, 255)
    diagnostic_path = output_root / "mapbox-california-reference.png"
    Image.fromarray(diagnostic).save(diagnostic_path)
    artifacts["diagnostic"] = {
        "path": diagnostic_path.name,
        "sha256": _sha256(diagnostic_path),
    }

    manifest = {
        "schema_version": 2 if source_manifest is not None else 1,
        "reference_id": output_root.name,
        "status": "pinned_reference",
        "kind": "mapbox_california_state_coast_water_counties",
        "provider": "Mapbox",
        "authority": {
            "state_and_county_lines": "Mapbox Streets v8 admin source layer",
            "coast_and_water": "Mapbox Streets v8 water source layer",
            "previous_mapscan_canonical_used": False,
            "county_png_used": False,
            "census_used": False,
        },
        "style": {
            "id": style_id,
            "url_without_token": style_url,
            "path": style_path.name,
            "sha256": _sha256(style_path),
        },
        "tileset": {
            "id": map_id,
            "tilejson_url_without_token": tilejson_url,
            "tilejson_path": tilejson_path.name,
            "tilejson_sha256": _sha256(tilejson_path),
            "source_layers": {"water": water_layer, "admin": admin_layer},
        },
        "zoom": zoom,
        "requested_bounds_wgs84": list(bounds_wgs84),
        "tile_range": {
            "minimum_x": x_range.start,
            "maximum_x": x_range.stop - 1,
            "minimum_y": y_range.start,
            "maximum_y": y_range.stop - 1,
        },
        "tile_count": len(records),
        "tile_bytes": sum(int(record["byte_count"]) for record in records),
        "tile_aggregate_sha256": _aggregate_hash(records),
        "tiles": records,
        "target_grid": {
            "crs": "EPSG:3857",
            "bounds": list(target_bounds),
            "width": width,
            "height": height,
        },
        "selection": {
            "california_seed_wgs84": list(california_seed_wgs84),
            "admin_levels_used_for_state_barrier": [0, 1],
            "admin_level_used_for_counties": 2,
            "worldviews": ["all", "US"],
            "maritime": "false",
            "disputed": "false",
            "barrier_width_px": barrier_width_px,
            "maximum_island_distance_m": maximum_island_distance_m,
            "method": (
                "mapbox_water_and_admin_barrier_four_connected_flood_fill_"
                "with_pacific_side_detached_islands"
            ),
        },
        "feature_counts": feature_counts,
        "topology": {
            "mainland_pixel_count": int(np.count_nonzero(mainland)),
            "island_pixel_count": int(np.count_nonzero(islands)),
            "state_land_pixel_count": int(np.count_nonzero(california_land)),
            "state_interior_pixel_count": int(
                np.count_nonzero(california_interior)
            ),
            "internal_water_pixel_count": int(
                np.count_nonzero(california_interior & ~california_land)
            ),
            "state_coast_pixel_count": int(np.count_nonzero(state_coast)),
            "water_edge_pixel_count": int(np.count_nonzero(water_edge)),
            "county_pixel_count": int(np.count_nonzero(county_lines)),
        },
        "controls": {
            "status": "pass" if validate_controls and controls_passed else "not_run",
            "passed_count": sum(
                item["status"] == "pass" for item in control_results
            ),
            "skipped_count": sum(
                item["status"] == "outside_target_grid" for item in control_results
            ),
            "failed_count": sum(
                item["status"] == "fail" for item in control_results
            ),
            "points": control_results,
        },
        "artifacts": artifacts,
        "pinning_policy": (
            "The raw style, TileJSON, and MVT bytes are the reproducibility authority; "
            "Mapbox Streets v8 is live-updated, so a later fetch is a new reference."
        ),
    }
    if source_manifest is not None:
        copied_raw_hashes = {
            "style_sha256": _sha256(style_path),
            "tilejson_sha256": _sha256(tilejson_path),
            "tile_aggregate_sha256": _aggregate_hash(records),
        }
        source_raw_hashes = {
            "style_sha256": str(source_manifest["style"]["sha256"]),
            "tilejson_sha256": str(
                source_manifest["tileset"]["tilejson_sha256"]
            ),
            "tile_aggregate_sha256": str(
                source_manifest["tile_aggregate_sha256"]
            ),
        }
        if copied_raw_hashes != source_raw_hashes:
            raise ValueError("Derived reference did not preserve pinned Mapbox bytes")
        manifest["derivation"] = {
            "kind": "pacific_side_detached_island_topology_v2",
            "source_reference_id": str(
                source_manifest.get("reference_id", source_manifest_path.parent.name)
            ),
            "source_manifest_path": str(source_manifest_path),
            "source_manifest_sha256": _sha256(source_manifest_path),
            "source_raw_hashes": source_raw_hashes,
            "copied_raw_hashes": copied_raw_hashes,
            "raw_bytes_preserved_exactly": True,
            "only_derived_masks_and_overlays_recomputed": True,
            "target_grid_supersampling": source_grid_supersampling,
            "change": (
                "Detached land components are retained only when they are both "
                "near the selected California mainland and on its Pacific side "
                "at the component latitude."
            ),
        }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def derive_mapbox_california_reference_from_pinned(
    source_manifest_path: Path,
    output_root: Path,
    *,
    target_supersampling: int = 1,
    validate_controls: bool | None = None,
) -> Dict[str, object]:
    """Recompute derived evidence while preserving pinned Mapbox bytes exactly."""

    source_manifest_path = Path(source_manifest_path).resolve()
    source = json.loads(source_manifest_path.read_text())
    selection = source.get("selection", {})
    tileset = source.get("tileset", {})
    if not isinstance(target_supersampling, int) or not 1 <= target_supersampling <= 4:
        raise ValueError("Target supersampling must be an integer from one to four")
    source_grid = dict(source["target_grid"])
    source_grid["width"] = (
        (int(source_grid["width"]) - 1) * target_supersampling + 1
    )
    source_grid["height"] = (
        (int(source_grid["height"]) - 1) * target_supersampling + 1
    )
    return build_mapbox_california_reference(
        Path(output_root),
        "",
        source_grid,
        bounds_wgs84=tuple(float(value) for value in source["requested_bounds_wgs84"]),
        zoom=int(source["zoom"]),
        map_id=str(tileset["id"]),
        style_id=str(source["style"]["id"]),
        california_seed_wgs84=tuple(
            float(value) for value in selection["california_seed_wgs84"]
        ),
        water_layer=str(tileset["source_layers"]["water"]),
        admin_layer=str(tileset["source_layers"]["admin"]),
        barrier_width_px=max(
            1, int(selection["barrier_width_px"]) * target_supersampling
        ),
        maximum_island_distance_m=float(selection["maximum_island_distance_m"]),
        validate_controls=(
            source.get("controls", {}).get("status") == "pass"
            and bool(source.get("controls", {}).get("points"))
            if validate_controls is None
            else validate_controls
        ),
        pinned_source_manifest_path=source_manifest_path,
    )
