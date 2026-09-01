"""Export an unapproved extraction as recolorable review-only XYZ masks."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image

from .tile_export import (
    _aggregate_hash,
    _copy_source_image,
    _sample_class_overview,
    _sample_class_tile,
    _sha256,
    _tile_range,
    _wgs84_bounds,
)


def _require_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    if _sha256(path) != expected:
        raise ValueError(f"{label.capitalize()} does not match its declared hash")


def _validate_plan_snapshot(
    run_dir: Path,
    snapshot_path: Path,
    snapshot: Dict[str, object],
    declared: object,
) -> str:
    if not isinstance(declared, dict):
        raise ValueError("Extraction has no declared plan provenance")
    declared_hash = str(declared.get("sha256", ""))
    if not declared_hash:
        raise ValueError("Extraction plan provenance has no hash")
    if _sha256(snapshot_path) == declared_hash:
        return declared_hash
    raw_path = str(declared.get("path", ""))
    if not raw_path:
        raise ValueError("Extraction does not match its plan snapshot")
    source = Path(raw_path)
    candidates = [source] if source.is_absolute() else [Path.cwd() / source]
    if not source.is_absolute():
        candidates.append(run_dir.parent.parent / source)
    plan_path = next((path for path in candidates if path.is_file()), None)
    if plan_path is None or _sha256(plan_path) != declared_hash:
        raise ValueError("Extraction's declared source plan is unavailable or changed")
    if json.loads(plan_path.read_text()) != snapshot:
        raise ValueError("Plan snapshot is not semantically identical to the declared source plan")
    return declared_hash


def _copy_canonical_boundary(
    run_dir: Path,
    output_dir: Path,
    extraction_layer: Dict[str, object],
    bounds: tuple[float, float, float, float],
    width: int,
    height: int,
) -> Dict[str, object] | None:
    clip = extraction_layer.get("canonical_clip")
    if not isinstance(clip, dict):
        return None
    active = clip.get("active_manifest")
    if not isinstance(active, dict):
        return None
    manifest_path = Path(str(active.get("path", "")))
    if not manifest_path.is_absolute():
        manifest_path = (run_dir / manifest_path).resolve()
    expected_manifest_hash = str(active.get("sha256", ""))
    _require_hash(manifest_path, expected_manifest_hash, "canonical boundary manifest")
    manifest = json.loads(manifest_path.read_text())
    grid = manifest.get("source_grid")
    if not isinstance(grid, dict):
        raise ValueError("Canonical boundary manifest has no source grid")
    if int(grid.get("width", 0)) != width or int(grid.get("height", 0)) != height:
        raise ValueError("Canonical boundary grid differs from the extraction grid")
    grid_bounds = tuple(float(value) for value in grid.get("bounds", []))
    if len(grid_bounds) != 4 or not np.allclose(grid_bounds, bounds, atol=1e-6):
        raise ValueError("Canonical boundary bounds differ from extraction bounds")
    overlay = manifest.get("artifacts", {}).get("overlay")
    if not isinstance(overlay, dict):
        raise ValueError("Canonical boundary manifest has no display overlay")
    source = manifest_path.parent / str(overlay.get("path", ""))
    expected_overlay_hash = str(overlay.get("sha256", ""))
    _require_hash(source, expected_overlay_hash, "canonical boundary overlay")
    target = output_dir / "canonical-boundary.png"
    shutil.copy2(source, target)
    publication_count = int(clip.get("valid_pixel_count", 0))
    classified_count = int(extraction_layer.get("web_mercator_classified_pixel_count", 0))
    return {
        "canonical_boundary_id": str(manifest.get("canonical_boundary_id", "")),
        "canonical_manifest_sha256": expected_manifest_hash,
        "continuous_border_component_count": int(
            manifest.get("topology", {}).get("combined_component_count", 0)
        ),
        "expected_boundary_component_count": int(
            manifest.get("topology", {}).get("combined_component_count", 0)
        ),
        "publication_interior_pixel_count": publication_count,
        "colored_pixel_count_outside_boundary": int(
            clip.get("colored_pixel_count_outside_boundary", 0)
        ),
        "unclassified_pixel_count_inside_boundary": max(
            publication_count - classified_count, 0
        ),
        "raster": target.name,
        "raster_sha256": expected_overlay_hash,
        "raster_width": width,
        "raster_height": height,
        "raster_bounds": list(_wgs84_bounds(bounds)),
    }


def _mapbox_water_reference(extraction_layer: Dict[str, object]) -> Dict[str, object] | None:
    clip = extraction_layer.get("canonical_clip")
    if not isinstance(clip, dict):
        return None
    water = clip.get("internal_water_exclusion")
    if not isinstance(water, dict):
        return None
    snap = water.get("canonical_shoreline_snap")
    if not isinstance(snap, dict):
        return None
    mapbox = snap.get("mapbox_water")
    if not isinstance(mapbox, dict):
        return None
    reference = mapbox.get("reference_manifest")
    return {
        "method": snap.get("method"),
        "water_pixel_count": snap.get("water_pixel_count"),
        "reference_manifest_sha256": (
            reference.get("sha256") if isinstance(reference, dict) else None
        ),
        "style_sha256": mapbox.get("style_sha256"),
        "tile_aggregate_sha256": mapbox.get("tile_aggregate_sha256"),
        "map_id": mapbox.get("map_id"),
        "source_layer": mapbox.get("source_layer"),
    }


def export_extraction_preview_tiles(
    run_dir: Path,
    output_dir: Path,
    minimum_zoom: int = 4,
    maximum_zoom: int = 9,
    overview_supersampling: int = 4,
) -> Dict[str, object]:
    """Export a hash-bound extraction for visual review without approving it."""

    run_dir = run_dir.resolve()
    output_dir = output_dir.resolve()
    if not 0 <= minimum_zoom <= maximum_zoom <= 22:
        raise ValueError("Tile zooms must satisfy 0 <= minimum <= maximum <= 22")
    if overview_supersampling < 2 or overview_supersampling > 8:
        raise ValueError("Overview supersampling must be between two and eight")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Preview output directory is not empty: {output_dir}")

    extraction_path = run_dir / "extraction.json"
    plan_path = run_dir / "plan.snapshot.json"
    for required in (extraction_path, plan_path):
        if not required.is_file():
            raise FileNotFoundError(f"Missing extraction preview input: {required}")
    extraction = json.loads(extraction_path.read_text())
    plan = json.loads(plan_path.read_text())
    if extraction.get("status") != "needs_visual_review":
        raise ValueError("Extraction preview input must still need visual review")
    declared_plan_hash = _validate_plan_snapshot(
        run_dir, plan_path, plan, extraction.get("plan")
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    source_image = _copy_source_image(plan, run_dir, output_dir)
    plan_layers = {str(layer["id"]): layer for layer in plan.get("layers", [])}
    layer_manifests = []
    all_wgs84_bounds: list[tuple[float, float, float, float]] = []
    preview_boundary = None
    provenance_layers = []

    for extraction_layer in extraction.get("layers", []):
        if not isinstance(extraction_layer, dict) or extraction_layer.get("kind") != "categorical":
            continue
        layer_id = str(extraction_layer.get("id", ""))
        definition = plan_layers.get(layer_id)
        if not isinstance(definition, dict):
            raise ValueError(f"No plan layer matches extraction layer {layer_id}")
        class_path = run_dir / layer_id / "web-mercator-class-id.png"
        if not class_path.is_file():
            raise FileNotFoundError(f"Missing extracted class raster: {class_path}")
        values = np.asarray(Image.open(class_path), dtype=np.uint8)
        warp = extraction_layer.get("warp")
        if not isinstance(warp, dict):
            raise ValueError(f"Layer {layer_id} has no warp metadata")
        bounds = tuple(float(value) for value in warp.get("bounds", []))
        if len(bounds) != 4:
            raise ValueError(f"Layer {layer_id} has no Web-Mercator bounds")
        if values.shape != (int(warp.get("height", 0)), int(warp.get("width", 0))):
            raise ValueError(f"Layer {layer_id} class raster dimensions differ from metadata")
        categories = list(definition.get("categories", []))
        class_count = len(categories)
        if class_count == 0 or int(values.max()) > class_count:
            raise ValueError(f"Layer {layer_id} has an invalid category raster")
        wgs84_bounds = _wgs84_bounds(bounds)
        all_wgs84_bounds.append(wgs84_bounds)
        category_paths: Dict[int, list[Path]] = {
            index: [] for index in range(1, class_count + 1)
        }

        for zoom in range(minimum_zoom, maximum_zoom + 1):
            x_range, y_range = _tile_range(bounds, zoom)
            for tile_x in x_range:
                for tile_y in y_range:
                    coverage_alpha = None
                    if zoom < maximum_zoom:
                        class_tile, coverage_alpha = _sample_class_overview(
                            values,
                            bounds,
                            zoom,
                            tile_x,
                            tile_y,
                            class_count,
                            overview_supersampling,
                        )
                    else:
                        class_tile = _sample_class_tile(values, bounds, zoom, tile_x, tile_y)
                    for class_id, category in enumerate(categories, 1):
                        category_id = str(category["id"])
                        tile_path = (
                            output_dir
                            / "tiles"
                            / layer_id
                            / category_id
                            / str(zoom)
                            / str(tile_x)
                            / f"{tile_y}.png"
                        )
                        tile_path.parent.mkdir(parents=True, exist_ok=True)
                        if coverage_alpha is None:
                            alpha = np.where(class_tile == class_id, 255, 0).astype(np.uint8)
                        else:
                            alpha = np.where(class_tile == class_id, coverage_alpha, 0).astype(np.uint8)
                        white = np.full(alpha.shape, 255, dtype=np.uint8)
                        Image.fromarray(np.dstack((white, white, white, alpha))).save(
                            tile_path, optimize=True
                        )
                        category_paths[class_id].append(tile_path)

        counts = extraction_layer.get("web_mercator_category_pixel_counts", {})
        category_manifests = []
        for class_id, category in enumerate(categories, 1):
            category_id = str(category["id"])
            paths = category_paths[class_id]
            tilejson_dir = output_dir / "tiles" / layer_id / category_id
            tilejson_path = tilejson_dir / "tilejson.json"
            tilejson_path.write_text(
                json.dumps(
                    {
                        "tilejson": "3.0.0",
                        "name": str(category.get("label", category_id)),
                        "scheme": "xyz",
                        "tiles": ["{z}/{x}/{y}.png"],
                        "minzoom": minimum_zoom,
                        "maxzoom": maximum_zoom,
                        "bounds": list(wgs84_bounds),
                    },
                    indent=2,
                )
                + "\n"
            )
            paths.append(tilejson_path)
            tile_set_hash = _aggregate_hash(paths, output_dir)
            display_rgb = category.get("display_rgb", category.get("legend_rgb", [255, 0, 255]))
            if display_rgb and isinstance(display_rgb[0], list):
                display_rgb = display_rgb[0]
            category_manifests.append(
                {
                    "id": category_id,
                    "class_id": class_id,
                    "label": str(category.get("label", category_id)),
                    "display_rgb": [int(value) for value in display_rgb[:3]],
                    "pixel_count": int(counts.get(category_id, 0)),
                    "tile_template": (
                        f"tiles/{layer_id}/{category_id}/{{z}}/{{x}}/{{y}}.png"
                        f"?v={tile_set_hash[:16]}"
                    ),
                    "tilejson": f"tiles/{layer_id}/{category_id}/tilejson.json",
                    "tile_file_count": len(paths) - 1,
                    "tile_set_sha256": tile_set_hash,
                    "encoding": "rgba_white_mask_with_coverage_overviews",
                }
            )
        layer_manifests.append(
            {
                "id": layer_id,
                "label": layer_id.replace("-", " ").title(),
                "kind": "categorical",
                "bounds": list(wgs84_bounds),
                "categories": category_manifests,
            }
        )
        if preview_boundary is None:
            preview_boundary = _copy_canonical_boundary(
                run_dir,
                output_dir,
                extraction_layer,
                bounds,
                values.shape[1],
                values.shape[0],
            )
        provenance_layers.append(
            {
                "id": layer_id,
                "class_raster_sha256": _sha256(class_path),
                "classified_pixel_count": int(np.count_nonzero(values)),
                "publication_interior_sha256": extraction_layer.get("canonical_clip", {})
                .get("artifacts", {})
                .get("interior", {})
                .get("sha256"),
                "mapbox_water_reference": _mapbox_water_reference(extraction_layer),
            }
        )

    if not layer_manifests:
        raise ValueError("No categorical extraction layer was available for preview")
    west = min(item[0] for item in all_wgs84_bounds)
    south = min(item[1] for item in all_wgs84_bounds)
    east = max(item[2] for item in all_wgs84_bounds)
    north = max(item[3] for item in all_wgs84_bounds)
    provenance = {
        "schema_version": 1,
        "kind": "unapproved_extraction_preview_provenance",
        "publication_approved": False,
        "dataset_id": str(extraction.get("dataset_id", "")),
        "extraction_manifest_sha256": _sha256(extraction_path),
        "declared_plan_sha256": declared_plan_hash,
        "plan_snapshot_sha256": _sha256(plan_path),
        "layers": provenance_layers,
    }
    provenance_path = output_dir / "preview-provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    dataset = {
        "schema_version": 1,
        "status": "needs_visual_review",
        "id": str(extraction.get("dataset_id", "")),
        "title": str(extraction.get("title", extraction.get("dataset_id", ""))),
        "bounds": [west, south, east, north],
        "center": [(west + east) / 2, (south + north) / 2],
        "minimum_zoom": minimum_zoom,
        "maximum_native_zoom": maximum_zoom,
        "overscaling": "nearest",
        "overview": {
            "mode": "dominant_class_with_fractional_coverage",
            "supersampling": overview_supersampling,
            "overview_zooms": [minimum_zoom, maximum_zoom - 1],
            "exact_binary_zoom": maximum_zoom,
        },
        "source_image": source_image,
        "approval": {"status": "not_approved"},
        "provenance": {
            "manifest": provenance_path.name,
            "sha256": _sha256(provenance_path),
        },
        "boundary": preview_boundary,
        "layers": layer_manifests,
    }
    dataset_path = output_dir / "dataset.json"
    dataset_path.write_text(json.dumps(dataset, indent=2) + "\n")
    return {**dataset, "dataset_manifest_sha256": _sha256(dataset_path)}
