"""Pinned Mapbox vector-water reference for basemap-exact land restoration.

Mapbox's live water fill is the final visual authority in the MapScan viewer.
For a categorical cell near a named bay, this module lets extraction use the
same vector source as the viewer instead of treating an offset Census polygon
or a flood-fill approximation as the shoreline itself.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import shutil
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, Tuple

import cv2
import mapbox_vector_tile
import numpy as np
from shapely import affinity, box
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, shape


WEB_MERCATOR_HALF_WORLD = 20037508.342789244
DEFAULT_MAP_ID = "mapbox.mapbox-streets-v8"
DEFAULT_STYLE_ID = "mapbox/light-v11"
DEFAULT_WATER_LAYER = "water"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _aggregate_tile_hash(records: Iterable[Dict[str, object]]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: str(item["path"])):
        digest.update(str(record["path"]).encode())
        digest.update(b"\0")
        digest.update(str(record["sha256"]).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _tile_coordinates(
    bounds_wgs84: Tuple[float, float, float, float], zoom: int
) -> Tuple[range, range]:
    west, south, east, north = bounds_wgs84
    if not (-180 <= west < east <= 180 and -85.051129 <= south < north <= 85.051129):
        raise ValueError("Mapbox water bounds are invalid")
    if not 0 <= zoom <= 22:
        raise ValueError("Mapbox water zoom must be between zero and 22")
    count = 1 << zoom

    def tile_x(longitude: float) -> int:
        return int(math.floor((longitude + 180.0) / 360.0 * count))

    def tile_y(latitude: float) -> int:
        radians = math.radians(latitude)
        return int(
            math.floor(
                (1.0 - math.asinh(math.tan(radians)) / math.pi) / 2.0 * count
            )
        )

    x0 = max(0, min(count - 1, tile_x(west)))
    x1 = max(0, min(count - 1, tile_x(math.nextafter(east, west))))
    y0 = max(0, min(count - 1, tile_y(north)))
    y1 = max(0, min(count - 1, tile_y(math.nextafter(south, north))))
    return range(x0, x1 + 1), range(y0, y1 + 1)


def _download(url: str, access_token: str) -> bytes:
    separator = "&" if "?" in url else "?"
    request = urllib.request.Request(
        f"{url}{separator}access_token={access_token}",
        headers={"User-Agent": "MapScan/0.1"},
    )
    with urllib.request.urlopen(request) as response:
        return response.read()


def _decode_tile(raw: bytes) -> Dict[str, object]:
    try:
        payload = gzip.decompress(raw)
    except gzip.BadGzipFile:
        payload = raw
    return mapbox_vector_tile.decode(
        payload, default_options={"y_coord_down": True}
    )


def fetch_mapbox_water_reference(
    output_root: Path,
    access_token: str,
    *,
    bounds_wgs84: Tuple[float, float, float, float],
    zoom: int = 9,
    map_id: str = DEFAULT_MAP_ID,
    style_id: str = DEFAULT_STYLE_ID,
    water_layer: str = DEFAULT_WATER_LAYER,
    force: bool = False,
) -> Dict[str, object]:
    """Pin the exact Mapbox style and vector tiles needed by a water audit."""

    if not access_token:
        raise ValueError("A Mapbox access token is required")
    output_root = output_root.resolve()
    if output_root.exists():
        if not force:
            raise FileExistsError(f"Mapbox water reference already exists: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    try:
        owner, style_name = style_id.split("/", 1)
    except ValueError as error:
        raise ValueError("Mapbox style id must be owner/style") from error
    style_url = f"https://api.mapbox.com/styles/v1/{owner}/{style_name}"
    style_path = output_root / "style.json"
    style_path.write_bytes(_download(style_url, access_token))
    style = json.loads(style_path.read_text())
    matching_layers = [
        layer
        for layer in style.get("layers", [])
        if layer.get("type") == "fill"
        and layer.get("source-layer") == water_layer
    ]
    if not matching_layers:
        raise ValueError("Pinned Mapbox style exposes no matching water fill")
    source_ids = {str(layer.get("source")) for layer in matching_layers}
    source_urls = [
        str(style.get("sources", {}).get(source_id, {}).get("url", ""))
        for source_id in source_ids
    ]
    if not any(map_id in url for url in source_urls):
        raise ValueError("Pinned Mapbox style does not reference the declared map id")

    x_range, y_range = _tile_coordinates(bounds_wgs84, zoom)
    records = []
    for tile_x in x_range:
        for tile_y in y_range:
            relative = Path("tiles") / str(zoom) / str(tile_x) / f"{tile_y}.mvt"
            target = output_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            public_url = (
                f"https://api.mapbox.com/v4/{map_id}/{zoom}/{tile_x}/{tile_y}.mvt"
            )
            target.write_bytes(_download(public_url, access_token))
            decoded = _decode_tile(target.read_bytes())
            layer = decoded.get(water_layer)
            records.append(
                {
                    "z": zoom,
                    "x": tile_x,
                    "y": tile_y,
                    "path": str(relative),
                    "sha256": _sha256(target),
                    "byte_count": target.stat().st_size,
                    "url_without_token": public_url,
                    "water_feature_count": (
                        len(layer.get("features", []))
                        if isinstance(layer, dict)
                        else 0
                    ),
                }
            )

    count = 1 << zoom
    span = WEB_MERCATOR_HALF_WORLD * 2 / count
    coverage_bounds = [
        -WEB_MERCATOR_HALF_WORLD + x_range.start * span,
        WEB_MERCATOR_HALF_WORLD - y_range.stop * span,
        -WEB_MERCATOR_HALF_WORLD + x_range.stop * span,
        WEB_MERCATOR_HALF_WORLD - y_range.start * span,
    ]
    manifest = {
        "schema_version": 1,
        "status": "pinned_reference",
        "kind": "mapbox_vector_water",
        "provider": "Mapbox",
        "style": {
            "id": style_id,
            "url_without_token": style_url,
            "path": style_path.name,
            "sha256": _sha256(style_path),
            "matching_fill_layer_ids": [
                str(layer.get("id")) for layer in matching_layers
            ],
        },
        "source": {
            "map_id": map_id,
            "source_layer": water_layer,
        },
        "zoom": zoom,
        "requested_bounds_wgs84": list(bounds_wgs84),
        "tile_range": {
            "minimum_x": x_range.start,
            "maximum_x": x_range.stop - 1,
            "minimum_y": y_range.start,
            "maximum_y": y_range.stop - 1,
        },
        "coverage_bounds_web_mercator": coverage_bounds,
        "tile_count": len(records),
        "tile_bytes": sum(int(record["byte_count"]) for record in records),
        "tile_aggregate_sha256": _aggregate_tile_hash(records),
        "tiles": records,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return manifest


def _polygons(geometry):
    if isinstance(geometry, Polygon):
        yield geometry
    elif isinstance(geometry, (MultiPolygon, GeometryCollection)):
        for part in geometry.geoms:
            yield from _polygons(part)


def rasterize_pinned_mapbox_water(
    reference_root: Path,
    grid: Dict[str, object],
    *,
    eligible_mask: np.ndarray,
    supersampling: int = 4,
    minimum_coverage_fraction: float = 1.0,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Rasterize hash-verified Mapbox water only where a caller allows it."""

    reference_root = reference_root.resolve()
    manifest_path = reference_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("status") != "pinned_reference"
        or manifest.get("kind") != "mapbox_vector_water"
    ):
        raise ValueError("Mapbox water reference is not pinned")
    if str(grid.get("crs")) != "EPSG:3857":
        raise ValueError("Mapbox water rasterization requires an EPSG:3857 grid")
    if not 1 <= supersampling <= 8:
        raise ValueError("Mapbox water supersampling must be between one and eight")
    if not 0 < minimum_coverage_fraction <= 1:
        raise ValueError("Mapbox water coverage fraction must be in (0, 1]")
    height, width = int(grid["height"]), int(grid["width"])
    eligible = np.asarray(eligible_mask, dtype=bool)
    if eligible.shape != (height, width):
        raise ValueError("Mapbox water eligibility mask differs from the target grid")
    result = np.zeros_like(eligible)
    if not np.any(eligible):
        raise ValueError("Mapbox water eligibility mask is empty")

    style_record = manifest.get("style", {})
    style_path = reference_root / str(style_record.get("path", ""))
    if not style_path.is_file() or _sha256(style_path) != style_record.get("sha256"):
        raise ValueError("Pinned Mapbox style is missing or stale")
    records = manifest.get("tiles", [])
    if len(records) != int(manifest.get("tile_count", -1)) or not records:
        raise ValueError("Pinned Mapbox tile inventory is incomplete")
    for record in records:
        path = reference_root / str(record.get("path", ""))
        if not path.is_file() or _sha256(path) != record.get("sha256"):
            raise ValueError(f"Pinned Mapbox tile is missing or stale: {path}")
    if _aggregate_tile_hash(records) != manifest.get("tile_aggregate_sha256"):
        raise ValueError("Pinned Mapbox tile aggregate hash changed")

    min_x, min_y, max_x, max_y = (float(value) for value in grid["bounds"])
    pixel_width = (max_x - min_x) / width
    pixel_height = (max_y - min_y) / height
    ys, xs = np.nonzero(eligible)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    coverage_bounds = [float(value) for value in manifest["coverage_bounds_web_mercator"]]
    eligible_world_bounds = [
        min_x + x0 * pixel_width,
        max_y - y1 * pixel_height,
        min_x + x1 * pixel_width,
        max_y - y0 * pixel_height,
    ]
    if (
        eligible_world_bounds[0] < coverage_bounds[0]
        or eligible_world_bounds[1] < coverage_bounds[1]
        or eligible_world_bounds[2] > coverage_bounds[2]
        or eligible_world_bounds[3] > coverage_bounds[3]
    ):
        raise ValueError("Pinned Mapbox tiles do not cover the eligible water region")

    high = np.zeros(
        ((y1 - y0) * supersampling, (x1 - x0) * supersampling),
        dtype=np.uint8,
    )
    zoom = int(manifest["zoom"])
    tile_span = WEB_MERCATOR_HALF_WORLD * 2 / (1 << zoom)
    water_layer = str(manifest["source"]["source_layer"])

    def pixel_points(coordinates) -> np.ndarray:
        values = np.asarray(coordinates, dtype=np.float64)
        pixel_x = (
            (values[:, 0] - min_x) / pixel_width * supersampling
            - x0 * supersampling
            - 0.5
        )
        pixel_y = (
            (max_y - values[:, 1]) / pixel_height * supersampling
            - y0 * supersampling
            - 0.5
        )
        return np.rint(np.column_stack((pixel_x, pixel_y))).astype(np.int32)

    rendered_features = 0
    for record in records:
        tile_x, tile_y = int(record["x"]), int(record["y"])
        path = reference_root / str(record["path"])
        decoded = _decode_tile(path.read_bytes())
        layer = decoded.get(water_layer)
        if not isinstance(layer, dict):
            continue
        extent = int(layer["extent"])
        left = -WEB_MERCATOR_HALF_WORLD + tile_x * tile_span
        top = WEB_MERCATOR_HALF_WORLD - tile_y * tile_span
        tile_bounds = box(left, top - tile_span, left + tile_span, top)
        for feature in layer.get("features", []):
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
            rendered_features += 1
            for polygon in _polygons(geometry):
                cv2.fillPoly(
                    high,
                    [pixel_points(polygon.exterior.coords)],
                    1,
                    lineType=cv2.LINE_8,
                )
                for interior in polygon.interiors:
                    cv2.fillPoly(
                        high,
                        [pixel_points(interior.coords)],
                        0,
                        lineType=cv2.LINE_8,
                    )

    coverage = high.reshape(
        y1 - y0, supersampling, x1 - x0, supersampling
    ).mean(axis=(1, 3))
    result[y0:y1, x0:x1] = coverage >= minimum_coverage_fraction
    result &= eligible
    report = {
        "method": "pinned_mapbox_vector_water_fractional_coverage",
        "reference_manifest": {
            "path": str(manifest_path),
            "sha256": _sha256(manifest_path),
        },
        "provider": manifest["provider"],
        "style_id": style_record["id"],
        "style_sha256": style_record["sha256"],
        "map_id": manifest["source"]["map_id"],
        "source_layer": water_layer,
        "zoom": zoom,
        "tile_count": len(records),
        "tile_aggregate_sha256": manifest["tile_aggregate_sha256"],
        "supersampling": supersampling,
        "minimum_coverage_fraction": minimum_coverage_fraction,
        "eligible_pixel_count": int(np.count_nonzero(eligible)),
        "water_pixel_count": int(np.count_nonzero(result)),
        "rendered_feature_count": rendered_features,
        "coverage_policy": (
            "Only the caller's named-water eligibility region is considered; a "
            "categorical cell is water only at the declared Mapbox subpixel coverage."
        ),
    }
    return result, report


def constrain_named_water_to_mapbox(
    water_seed: np.ndarray,
    eligible_interior: np.ndarray,
    target_grid: Dict[str, object],
    reference_root: Path,
    *,
    maximum_distance_px: float = 40.0,
    supersampling: int = 4,
    minimum_coverage_fraction: float = 1.0,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Use Census names for eligibility and pinned Mapbox geometry for shape."""

    if maximum_distance_px <= 0:
        raise ValueError("Named Mapbox water distance must be positive")
    seed = np.asarray(water_seed, dtype=bool)
    interior = np.asarray(eligible_interior, dtype=bool)
    if seed.shape != interior.shape:
        raise ValueError("Named Mapbox water masks differ")
    distance = cv2.distanceTransform(
        (~seed).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_5
    )
    permitted = interior & (distance <= float(maximum_distance_px))
    water, reference_report = rasterize_pinned_mapbox_water(
        reference_root,
        target_grid,
        eligible_mask=permitted,
        supersampling=supersampling,
        minimum_coverage_fraction=minimum_coverage_fraction,
    )
    overlap = water & seed
    if not np.any(overlap):
        raise ValueError("Named Census water does not overlap pinned Mapbox water")
    report = {
        "method": "named_census_water_constrained_to_pinned_mapbox_water",
        "maximum_distance_px": float(maximum_distance_px),
        "seed_pixel_count": int(np.count_nonzero(seed)),
        "seed_pixel_count_inside_eligible_interior": int(
            np.count_nonzero(seed & interior)
        ),
        "permitted_pixel_count": int(np.count_nonzero(permitted)),
        "water_pixel_count": int(np.count_nonzero(water)),
        "retained_seed_pixel_count": int(np.count_nonzero(overlap)),
        "added_beyond_seed_pixel_count": int(np.count_nonzero(water & ~seed)),
        "restored_landward_seed_pixel_count": int(
            np.count_nonzero(seed & interior & ~water)
        ),
        "semantic_policy": (
            "Census exact names select eligible bays; the pinned Mapbox Streets water "
            "layer used by the live viewer decides land versus water."
        ),
        "mapbox_water": reference_report,
    }
    return water, report
