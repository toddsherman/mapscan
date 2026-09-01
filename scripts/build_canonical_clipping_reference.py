#!/usr/bin/env python3
"""Build the mainland clipping interior without changing the display border.

California's Census state polygon includes territorial Pacific water.  The
clipping mainland is therefore the largest component left after subtracting
only exact-name ``Pacific Ocean`` Area Hydrography polygons.  Interior bays,
rivers, and lakes remain inside this outer clipping contract; dataset-specific
water exclusions continue to decide which of those surfaces are transparent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from pyproj import Transformer
from shapely.geometry import Point, mapping
from shapely.ops import transform as transform_geometry, unary_union

from mapscan.canonical_clip import rasterize_polygon_geometry
from mapscan.reference import load_california
from mapscan.water_reference import _verified_features


CLIPPING_ID = "california-mainland-clipping-v2"
PACIFIC_NAME = "Pacific Ocean"
LAND_CONTROLS = {
    "san_francisco": (-122.4194, 37.7749),
    "san_francisco_international_airport": (-122.3790, 37.6213),
    "san_rafael": (-122.5311, 37.9735),
    "marin_city": (-122.5133, 37.8685),
    "daly_city": (-122.4702, 37.6879),
    "berkeley": (-122.2730, 37.8715),
    "redwood_city": (-122.2364, 37.4852),
}
RETAINED_INTERIOR_WATER_CONTROLS = {
    "san_francisco_bay": (-122.36, 37.72),
    "san_pablo_bay": (-122.40, 38.05),
}
EXTERIOR_CONTROLS = {
    "pacific_west_of_san_francisco": (-123.0, 37.7),
    "nevada_near_reno": (-119.8, 39.5),
    "oregon_north_of_border": (-123.0, 42.2),
    "mexico_south_of_san_diego": (-117.0, 32.4),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def _active_display_reference(pointer_path: Path) -> tuple[dict, Path, dict, Path]:
    pointer = _load(pointer_path)
    manifest_record = pointer["manifest"]
    manifest_path = pointer_path.parent / str(manifest_record["path"])
    if _sha256(manifest_path) != str(manifest_record["sha256"]):
        raise ValueError("Active canonical display-border manifest is stale")
    manifest = _load(manifest_path)
    overlay_record = manifest["artifacts"]["overlay"]
    overlay_path = manifest_path.parent / str(overlay_record["path"])
    if _sha256(overlay_path) != str(overlay_record["sha256"]):
        raise ValueError("Active canonical display-border overlay is stale")
    return pointer, manifest_path, manifest, overlay_path


def _old_mainland_mask(path: Path, grid: dict) -> np.ndarray:
    collection = _load(path)
    feature = collection["features"][0]
    coordinates = feature["geometry"]["coordinates"]
    geometry = {"type": "Polygon", "coordinates": [coordinates]}
    return rasterize_polygon_geometry(geometry, grid)


def _pixel(grid: dict, longitude: float, latitude: float) -> tuple[int, int]:
    transformer = Transformer.from_crs("EPSG:4326", str(grid["crs"]), always_xy=True)
    x, y = transformer.transform(longitude, latitude)
    min_x, min_y, max_x, max_y = (float(value) for value in grid["bounds"])
    column = int(round((x - min_x) / (max_x - min_x) * int(grid["width"]) - 0.5))
    row = int(round((max_y - y) / (max_y - min_y) * int(grid["height"]) - 0.5))
    return column, row


def _probe_records(geometry, mask: np.ndarray, grid: dict) -> list[dict]:
    records = []
    groups = (
        ("required_inside_land", LAND_CONTROLS, True),
        ("required_inside_for_dataset_specific_water_policy", RETAINED_INTERIOR_WATER_CONTROLS, True),
        ("required_outside", EXTERIOR_CONTROLS, False),
    )
    for role, controls, expected in groups:
        for name, (longitude, latitude) in controls.items():
            vector_inside = bool(geometry.covers(Point(longitude, latitude)))
            column, row = _pixel(grid, longitude, latitude)
            on_grid = 0 <= row < mask.shape[0] and 0 <= column < mask.shape[1]
            raster_inside = bool(mask[row, column]) if on_grid else False
            if vector_inside is not expected or raster_inside is not expected:
                raise ValueError(f"Canonical clipping control failed: {name}")
            records.append(
                {
                    "id": name,
                    "role": role,
                    "wgs84": [longitude, latitude],
                    "expected_inside": expected,
                    "vector_inside": vector_inside,
                    "canonical_grid_pixel": [column, row],
                    "canonical_grid_contains_control": on_grid,
                    "canonical_grid_inside": raster_inside,
                }
            )
    return records


def build(args: argparse.Namespace) -> dict:
    boundary_reference = args.boundary_reference.resolve()
    water_reference = args.water_reference.resolve()
    pointer_path = args.active_pointer.resolve()
    old_geojson_path = args.old_mainland_geojson.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    pointer, active_manifest_path, active, active_overlay_path = (
        _active_display_reference(pointer_path)
    )
    grid = dict(active["source_grid"])
    state, _ = load_california(boundary_reference)
    pacific_features = [
        geometry
        for record, geometry in _verified_features(water_reference)
        if str(record.get("FULLNAME") or "").casefold() == PACIFIC_NAME.casefold()
    ]
    if not pacific_features:
        raise ValueError("Pinned Area Hydrography contains no Pacific Ocean features")
    pacific = unary_union(pacific_features)
    components = list(state.difference(pacific).geoms)
    mainland_nad83 = max(components, key=lambda geometry: geometry.area)
    to_wgs84 = Transformer.from_crs("EPSG:4269", "EPSG:4326", always_xy=True)
    mainland = transform_geometry(to_wgs84.transform, mainland_nad83)
    if not mainland.is_valid or mainland.geom_type != "Polygon":
        raise ValueError("Derived California mainland clipping geometry is invalid")

    geojson_path = output / "canonical-mainland-clipping.geojson"
    feature_collection = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "role": "canonical_mainland_clipping_interior",
                    "canonical_clipping_id": CLIPPING_ID,
                    "construction": "largest California component after exact-name Pacific Ocean subtraction",
                },
                "geometry": mapping(mainland),
            }
        ],
    }
    _write_json(geojson_path, feature_collection)

    mask = rasterize_polygon_geometry(feature_collection["features"][0]["geometry"], grid)
    if cv2.connectedComponents(mask.astype(np.uint8), 8)[0] - 1 != 1:
        raise ValueError("Canonical-grid mainland clipping mask is not one component")
    mask_path = output / "canonical-mainland-clipping-mask.png"
    Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path, optimize=True)

    old_mask = _old_mainland_mask(old_geojson_path, grid)
    added = mask & ~old_mask
    removed = old_mask & ~mask
    active_overlay = np.asarray(Image.open(active_overlay_path).convert("RGBA"))
    if active_overlay.shape[:2] != mask.shape:
        raise ValueError("Active display border and clipping diagnostic grids differ")
    diagnostic = np.zeros((*mask.shape, 4), dtype=np.uint8)
    diagnostic[mask] = (232, 232, 228, 255)
    diagnostic[added] = (255, 161, 48, 255)
    diagnostic[removed] = (0, 205, 255, 255)
    active_pixels = active_overlay[..., 3] > 0
    diagnostic[active_pixels] = active_overlay[active_pixels]
    diagnostic_path = output / "canonical-clipping-change-diagnostic.png"
    Image.fromarray(diagnostic).save(diagnostic_path, optimize=True)

    probes = _probe_records(mainland, mask, grid)
    probes_path = output / "canonical-clipping-controls.json"
    _write_json(probes_path, {"schema_version": 1, "controls": probes})

    state_manifest_path = boundary_reference / "manifest.json"
    water_manifest_path = water_reference / "manifest.json"
    manifest = {
        "schema_version": 2,
        "status": "pinned_pipeline_reference",
        "kind": "clipping_interior",
        "canonical_clipping_id": CLIPPING_ID,
        "scope": "california_mainland_outer_clipping_interior",
        "source_grid": grid,
        "authority": {
            "state_geometry": {
                "source": "U.S. Census Bureau TIGER/Line 2025 California state polygon",
                "manifest_path": str(state_manifest_path),
                "manifest_sha256": _sha256(state_manifest_path),
            },
            "pacific_exclusion": {
                "source": "U.S. Census Bureau TIGER/Line 2025 Area Hydrography",
                "manifest_path": str(water_manifest_path),
                "manifest_sha256": _sha256(water_manifest_path),
                "exact_feature_name": PACIFIC_NAME,
                "matched_feature_count": len(pacific_features),
            },
            "display_border": {
                "canonical_boundary_id": active["canonical_boundary_id"],
                "pointer_path": str(pointer_path),
                "pointer_sha256": _sha256(pointer_path),
                "manifest_path": str(active_manifest_path),
                "manifest_sha256": _sha256(active_manifest_path),
                "overlay_path": str(active_overlay_path),
                "overlay_sha256": _sha256(active_overlay_path),
            },
        },
        "construction": {
            "operation": "California state polygon minus exact-name Pacific Ocean polygons",
            "component_selection": "largest remaining polygon",
            "source_crs": "EPSG:4269",
            "artifact_crs": "EPSG:4326",
            "interior_water_policy": "retained for dataset-specific semantic exclusion",
            "offshore_island_policy": "excluded here; four approved county.png islands are added separately",
            "display_border_reconstructed_from_fill": False,
        },
        "geometry": {
            "type": mainland.geom_type,
            "valid": bool(mainland.is_valid),
            "exterior_vertex_count": len(mainland.exterior.coords),
            "interior_ring_count": len(mainland.interiors),
            "canonical_grid_component_count": 1,
            "canonical_grid_pixel_count": int(np.count_nonzero(mask)),
        },
        "old_clipping_comparison": {
            "reference_path": str(old_geojson_path),
            "reference_sha256": _sha256(old_geojson_path),
            "added_pixel_count": int(np.count_nonzero(added)),
            "removed_pixel_count": int(np.count_nonzero(removed)),
            "interpretation": "orange restores authoritative California interior; cyan removes old fill-only overreach",
        },
        "controls": {
            "count": len(probes),
            "all_passed": True,
            "artifact": {
                "path": probes_path.name,
                "sha256": _sha256(probes_path),
            },
        },
        "artifacts": {
            "geojson": {"path": geojson_path.name, "sha256": _sha256(geojson_path)},
            "canonical_grid_mask": {"path": mask_path.name, "sha256": _sha256(mask_path)},
            "change_diagnostic": {
                "path": diagnostic_path.name,
                "sha256": _sha256(diagnostic_path),
            },
        },
        "policy": {
            "publication_clipping_interior": "required",
            "active_lime_display_border": "unchanged and separately required",
            "dataset_internal_water_exclusion": "separate evidence",
        },
    }
    manifest_path = output / "canonical-clipping.json"
    _write_json(manifest_path, manifest)
    return {"manifest": str(manifest_path), "manifest_sha256": _sha256(manifest_path), **manifest}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--boundary-reference", type=Path, default=Path("reference/census-2025"))
    parser.add_argument(
        "--water-reference", type=Path, default=Path("reference/census-2025-areawater")
    )
    parser.add_argument(
        "--active-pointer", type=Path, default=Path("reference/canonical-california-boundary.json")
    )
    parser.add_argument(
        "--old-mainland-geojson",
        type=Path,
        default=Path(
            "reference/canonical-california-boundary-v1/canonical-mainland-boundary.geojson"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reference/canonical-california-clipping-v2"),
    )
    print(json.dumps(build(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
