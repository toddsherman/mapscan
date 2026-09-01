"""Export approved categorical rasters as recolorable static XYZ masks."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Dict, Iterable, Tuple

import cv2
import numpy as np
from PIL import Image
from pyproj import Transformer


WEB_MERCATOR_HALF_WORLD = 20037508.342789244
TILE_SIZE = 256


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tile_range(
    bounds: Tuple[float, float, float, float], zoom: int
) -> Tuple[range, range]:
    xmin, ymin, xmax, ymax = bounds
    count = 1 << zoom
    span = WEB_MERCATOR_HALF_WORLD * 2 / count
    epsilon = span * 1e-12
    x0 = max(0, int(math.floor((xmin + WEB_MERCATOR_HALF_WORLD) / span)))
    x1 = min(
        count - 1,
        int(math.floor((xmax - epsilon + WEB_MERCATOR_HALF_WORLD) / span)),
    )
    y0 = max(0, int(math.floor((WEB_MERCATOR_HALF_WORLD - ymax) / span)))
    y1 = min(
        count - 1,
        int(math.floor((WEB_MERCATOR_HALF_WORLD - (ymin + epsilon)) / span)),
    )
    return range(x0, x1 + 1), range(y0, y1 + 1)


def _sample_class_tile(
    values: np.ndarray,
    bounds: Tuple[float, float, float, float],
    zoom: int,
    tile_x: int,
    tile_y: int,
    tile_size: int = TILE_SIZE,
) -> np.ndarray:
    xmin, ymin, xmax, ymax = bounds
    count = 1 << zoom
    span = WEB_MERCATOR_HALF_WORLD * 2 / count
    tile_left = -WEB_MERCATOR_HALF_WORLD + tile_x * span
    tile_top = WEB_MERCATOR_HALF_WORLD - tile_y * span
    step = span / tile_size
    world_x = tile_left + (np.arange(tile_size) + 0.5) * step
    world_y = tile_top - (np.arange(tile_size) + 0.5) * step
    source_x = np.floor((world_x - xmin) * values.shape[1] / (xmax - xmin)).astype(
        np.int64
    )
    source_y = np.floor((ymax - world_y) * values.shape[0] / (ymax - ymin)).astype(
        np.int64
    )
    valid_x = (source_x >= 0) & (source_x < values.shape[1])
    valid_y = (source_y >= 0) & (source_y < values.shape[0])
    tile = np.zeros((tile_size, tile_size), dtype=np.uint8)
    if np.any(valid_x) and np.any(valid_y):
        tile[np.ix_(valid_y, valid_x)] = values[
            np.ix_(source_y[valid_y], source_x[valid_x])
        ]
    return tile


def _sample_class_overview(
    values: np.ndarray,
    bounds: Tuple[float, float, float, float],
    zoom: int,
    tile_x: int,
    tile_y: int,
    class_count: int,
    supersampling: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return one dominant class and fractional valid coverage per tile pixel.

    Low zooms cannot display every canonical categorical cell. Sampling only
    the destination-pixel center makes a correctly aligned coastline appear
    displaced by up to half of a coarse tile pixel. Supersampling preserves
    the covered fraction while still assigning exactly one mutually exclusive
    category to each visible overview pixel. The maximum native zoom remains
    an exact binary nearest-neighbor representation.
    """

    if supersampling < 2:
        raise ValueError("Overview supersampling must be at least two")
    samples = _sample_class_tile(
        values,
        bounds,
        zoom,
        tile_x,
        tile_y,
        tile_size=TILE_SIZE * supersampling,
    )
    blocks = samples.reshape(
        TILE_SIZE, supersampling, TILE_SIZE, supersampling
    )
    sample_count = supersampling * supersampling
    coverage_count = np.count_nonzero(blocks, axis=(1, 3))
    class_counts = np.stack(
        [
            np.count_nonzero(blocks == class_id, axis=(1, 3))
            for class_id in range(1, class_count + 1)
        ],
        axis=0,
    )
    dominant = np.argmax(class_counts, axis=0).astype(np.uint8) + 1
    dominant[coverage_count == 0] = 0
    alpha = np.rint(coverage_count * (255.0 / sample_count)).astype(np.uint8)
    return dominant, alpha


def _wgs84_bounds(
    bounds: Tuple[float, float, float, float]
) -> Tuple[float, float, float, float]:
    transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    west, south = transformer.transform(bounds[0], bounds[1])
    east, north = transformer.transform(bounds[2], bounds[3])
    return float(west), float(south), float(east), float(north)


def _aggregate_hash(paths: Iterable[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(_sha256(path).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def estimate_xyz_tile_file_count(
    bounds: Tuple[float, float, float, float],
    minimum_zoom: int,
    maximum_zoom: int,
) -> int:
    """Return the exact number of XYZ image files covering ``bounds``.

    This is intentionally independent of the source raster dimensions. It is
    used before a high-resolution export so the deployment file count can be
    audited without generating a pyramid.
    """

    if not 0 <= minimum_zoom <= maximum_zoom <= 22:
        raise ValueError("Tile zooms must satisfy 0 <= minimum <= maximum <= 22")
    return sum(
        len(x_range) * len(y_range)
        for zoom in range(minimum_zoom, maximum_zoom + 1)
        for x_range, y_range in (_tile_range(bounds, zoom),)
    )


def _write_indexed_class_id_pyramid(
    values: np.ndarray,
    bounds: Tuple[float, float, float, float],
    output_dir: Path,
    layer_id: str,
    category_count: int,
    minimum_zoom: int,
    maximum_zoom: int,
    overview_supersampling: int,
) -> Dict[str, object]:
    """Write one lossless class-id pyramid shared by every category.

    PNG luma stores the uint8 class id and PNG alpha stores coverage. Browsers
    decode grayscale-alpha PNGs to RGBA, so Mapbox GL can read the class id
    through the red-channel ``raster-color-mix`` while retaining NoData and
    fractional low-zoom coastline coverage in alpha.
    """

    if values.ndim != 2 or values.dtype != np.uint8:
        raise ValueError("Indexed categorical tiles require a uint8 2D raster")
    if not 1 <= category_count <= 255:
        raise ValueError("Indexed categorical tiles support 1 to 255 classes")
    if int(values.max(initial=0)) > category_count:
        raise ValueError("Class raster contains an id outside the legend")

    tile_root = output_dir / "tiles" / layer_id / "class-id"
    paths: list[Path] = []
    image_byte_count = 0
    for zoom in range(minimum_zoom, maximum_zoom + 1):
        x_range, y_range = _tile_range(bounds, zoom)
        for tile_x in x_range:
            for tile_y in y_range:
                if zoom < maximum_zoom:
                    class_tile, alpha = _sample_class_overview(
                        values,
                        bounds,
                        zoom,
                        tile_x,
                        tile_y,
                        category_count,
                        overview_supersampling,
                    )
                else:
                    class_tile = _sample_class_tile(
                        values, bounds, zoom, tile_x, tile_y
                    )
                    alpha = np.where(class_tile > 0, 255, 0).astype(np.uint8)
                tile_path = tile_root / str(zoom) / str(tile_x) / f"{tile_y}.png"
                tile_path.parent.mkdir(parents=True, exist_ok=True)
                encoded = np.dstack((class_tile, alpha))
                Image.fromarray(encoded).save(tile_path, optimize=True)
                paths.append(tile_path)
                image_byte_count += tile_path.stat().st_size

    wgs84_bounds = _wgs84_bounds(bounds)
    tilejson_path = tile_root / "tilejson.json"
    tilejson = {
        "tilejson": "3.0.0",
        "name": f"{layer_id} class ids",
        "scheme": "xyz",
        "tiles": ["{z}/{x}/{y}.png"],
        "minzoom": minimum_zoom,
        "maxzoom": maximum_zoom,
        "bounds": list(wgs84_bounds),
    }
    tilejson_path.write_text(json.dumps(tilejson, indent=2) + "\n")
    paths.append(tilejson_path)
    tile_set_hash = _aggregate_hash(paths, output_dir)
    tile_count = len(paths) - 1
    return {
        "encoding": "png_luma_alpha_uint8_class_id_v1",
        "class_id_channel": "red_after_browser_png_decode",
        "coverage_channel": "alpha",
        "nodata_class_id": 0,
        "class_id_range": [1, category_count],
        "raster_color_mix": [1, 0, 0, 0],
        "raster_color_range": [0, 1],
        "tile_template": (
            f"tiles/{layer_id}/class-id/{{z}}/{{x}}/{{y}}.png"
            f"?v={tile_set_hash[:16]}"
        ),
        "tilejson": f"tiles/{layer_id}/class-id/tilejson.json",
        "tile_file_count": tile_count,
        "tile_image_byte_count": image_byte_count,
        "mean_tile_image_byte_count": image_byte_count / max(tile_count, 1),
        "tile_set_sha256": tile_set_hash,
        "mapbox_rendering": "single_raster_source_dynamic_raster_color_step_expression",
    }


def _copy_source_image(plan: Dict[str, object], run_dir: Path, output_dir: Path) -> str | None:
    raw_source = plan.get("source")
    if not isinstance(raw_source, str) or not raw_source:
        return None
    source = Path(raw_source)
    candidates = [source] if source.is_absolute() else [Path.cwd() / source]
    if not source.is_absolute():
        candidates.append(run_dir.parent.parent / source)
    source_path = next((path for path in candidates if path.is_file()), None)
    if source_path is None:
        return None
    suffix = source_path.suffix.lower() or ".img"
    target = output_dir / f"source{suffix}"
    shutil.copy2(source_path, target)
    return target.name


def _validated_boundary_provenance(
    materialized_dir: Path,
    materialization: Dict[str, object],
    approval: Dict[str, object],
) -> Dict[str, object] | None:
    raw_boundary = materialization.get("boundary_clip")
    if raw_boundary is None:
        return None
    if not isinstance(raw_boundary, dict):
        raise ValueError("Materialization boundary clip must be an object")
    raw_audit = raw_boundary.get("audit")
    if not isinstance(raw_audit, dict):
        raise ValueError("Boundary-clipped materialization has no audit")
    audit_path = Path(str(raw_audit.get("path", ""))).resolve()
    audit_hash = str(raw_audit.get("sha256", ""))
    if not audit_path.is_file() or _sha256(audit_path) != audit_hash:
        raise ValueError("Boundary clip audit is missing or hash-mismatched")
    if approval.get("boundary_clip_audit_sha256") != audit_hash:
        raise ValueError("Approval does not match the boundary clip audit")
    border_hash = str(raw_boundary.get("continuous_border_sha256", ""))
    if approval.get("hybrid_border_sha256") != border_hash:
        raise ValueError("Approval does not match the continuous boundary")
    if raw_boundary.get("colored_pixel_count_outside_boundary") != 0:
        raise ValueError("Boundary-clipped materialization still colors outside pixels")
    audit = json.loads(audit_path.read_text())
    boundary = audit.get("boundary", {})
    component_count = int(boundary.get("connected_component_count", 0))
    expected_component_count = int(
        boundary.get("expected_component_count", component_count)
    )
    components = boundary.get("components", [])
    if (
        audit.get("status") != "pass"
        or component_count < 1
        or component_count != expected_component_count
        or (components and len(components) != component_count)
        or not all(layer.get("passed") is True for layer in audit.get("layers", []))
    ):
        raise ValueError("Boundary clip audit did not pass its continuity and clip gates")
    for component in components:
        if (
            component.get("role") == "source_supported_island"
            and int(component.get("observed_source_pixel_count", 0)) < 1
        ):
            raise ValueError("Published island lacks hash-bound observed source evidence")
    canonical_display = boundary.get("canonical_display_border")
    canonical_border = raw_boundary.get("canonical_border")
    author_approved_islands = any(
        component.get("role") == "author_approved_canonical_island"
        for component in components
    )
    if author_approved_islands and not (
        isinstance(canonical_display, dict) and isinstance(canonical_border, dict)
    ):
        raise ValueError("Author-approved islands require a canonical display border")
    canonical_display_source = None
    canonical_display_grid = None
    canonical_display_id = None
    if isinstance(canonical_display, dict):
        canonical_display_source = Path(
            str(canonical_display.get("path", ""))
        ).resolve()
        canonical_display_hash = str(canonical_display.get("sha256", ""))
        canonical_display_id = str(
            canonical_display.get("canonical_boundary_id", "")
        )
        canonical_display_grid = canonical_display.get("grid")
        if (
            not canonical_display_source.is_file()
            or _sha256(canonical_display_source) != canonical_display_hash
        ):
            raise ValueError("Canonical display border is missing or hash-mismatched")
        if canonical_display_hash != border_hash:
            raise ValueError("Canonical display border differs from the approved boundary")
        if approval.get("canonical_display_border_sha256") != canonical_display_hash:
            raise ValueError("Approval does not match the canonical display border")
        if not isinstance(canonical_display_grid, dict):
            raise ValueError("Canonical display border has no grid")
        if isinstance(canonical_border, dict):
            if (
                canonical_border.get("canonical_boundary_id") != canonical_display_id
                or canonical_border.get("display_overlay_sha256")
                != canonical_display_hash
                or approval.get("canonical_boundary_manifest_sha256")
                != canonical_border.get("manifest_sha256")
            ):
                raise ValueError("Canonical boundary provenance does not match approval")
    interior_record = boundary.get("interior", {})
    interior_path = audit_path.parent / str(interior_record.get("path", ""))
    if (
        not interior_path.is_file()
        or _sha256(interior_path) != interior_record.get("sha256")
    ):
        raise ValueError("Publication interior is missing or hash-mismatched")
    interior = np.asarray(Image.open(interior_path).convert("L")) > 0
    public_water = None
    raw_water = raw_boundary.get("internal_water_exclusion")
    if isinstance(raw_water, dict):
        water_artifact = raw_water.get("artifact")
        if not isinstance(water_artifact, dict):
            raise ValueError("Internal-water provenance has no exact mask artifact")
        water_source = materialized_dir / str(water_artifact.get("path", ""))
        water_hash = str(water_artifact.get("sha256", ""))
        if not water_source.is_file() or _sha256(water_source) != water_hash:
            raise ValueError("Internal-water mask is missing or hash-mismatched")
        water = np.asarray(Image.open(water_source).convert("L")) > 0
        canonical_record = boundary.get("canonical_interior", {})
        canonical_path = audit_path.parent / str(canonical_record.get("path", ""))
        if (
            not canonical_path.is_file()
            or _sha256(canonical_path) != canonical_record.get("sha256")
        ):
            raise ValueError("Canonical interior is unavailable for water-mask verification")
        canonical = np.asarray(Image.open(canonical_path).convert("L")) > 0
        if water.shape != interior.shape or not np.array_equal(water, canonical & ~interior):
            raise ValueError("Internal-water mask does not exactly match the publication exclusion")
        shoreline = raw_water.get("canonical_shoreline_snap")
        mapbox = shoreline.get("mapbox_water") if isinstance(shoreline, dict) else None
        reference = raw_water.get("reference_manifest")
        mapbox_reference = mapbox.get("reference_manifest") if isinstance(mapbox, dict) else None
        public_water = {
            "method": raw_water.get("method"),
            "excluded_interior_pixel_count": int(np.count_nonzero(water)),
            "colored_water_policy": raw_water.get("colored_water_policy"),
            "reference_manifest_sha256": (
                reference.get("sha256") if isinstance(reference, dict) else None
            ),
            "canonical_shoreline_snap": (
                {
                    "method": shoreline.get("method"),
                    "feature_names": shoreline.get("feature_names"),
                    "maximum_distance_px": shoreline.get("maximum_distance_px"),
                    "water_pixel_count": shoreline.get("water_pixel_count"),
                    "combined_water_pixel_count": shoreline.get("combined_water_pixel_count"),
                    "semantic_policy": shoreline.get("semantic_policy"),
                    "mapbox_water": (
                        {
                            "provider": mapbox.get("provider"),
                            "style_id": mapbox.get("style_id"),
                            "style_sha256": mapbox.get("style_sha256"),
                            "map_id": mapbox.get("map_id"),
                            "source_layer": mapbox.get("source_layer"),
                            "zoom": mapbox.get("zoom"),
                            "tile_count": mapbox.get("tile_count"),
                            "tile_aggregate_sha256": mapbox.get("tile_aggregate_sha256"),
                            "supersampling": mapbox.get("supersampling"),
                            "minimum_coverage_fraction": mapbox.get(
                                "minimum_coverage_fraction"
                            ),
                            "water_pixel_count": mapbox.get("water_pixel_count"),
                            "coverage_policy": mapbox.get("coverage_policy"),
                            "reference_manifest_sha256": (
                                mapbox_reference.get("sha256")
                                if isinstance(mapbox_reference, dict)
                                else None
                            ),
                        }
                        if isinstance(mapbox, dict)
                        else None
                    ),
                }
                if isinstance(shoreline, dict)
                else None
            ),
            "artifact": {"sha256": water_hash},
            "_source_path": str(water_source),
        }
    total_unclassified_inside = 0
    for layer in materialization.get("layers", []):
        class_record = layer.get("artifacts", {}).get("class_id", {})
        class_path = materialized_dir / str(class_record.get("path", ""))
        if (
            not class_path.is_file()
            or _sha256(class_path) != class_record.get("sha256")
        ):
            raise ValueError("Approved boundary-clipped class raster is stale")
        values = np.asarray(Image.open(class_path), dtype=np.uint8)
        if values.shape != interior.shape:
            raise ValueError("Publication interior and class raster grids differ")
        outside_count = int(np.count_nonzero((values > 0) & ~interior))
        inside_empty_count = int(np.count_nonzero((values == 0) & interior))
        coverage_expectation = str(layer.get("coverage_expectation", "full_state"))
        if coverage_expectation not in {"full_state", "sparse_visible_evidence"}:
            raise ValueError("Approved class raster has an unsupported coverage contract")
        if outside_count or (
            inside_empty_count and coverage_expectation != "sparse_visible_evidence"
        ):
            raise ValueError("Approved class raster no longer satisfies its boundary contract")
        if inside_empty_count != int(
            layer.get("unclassified_pixel_count_inside_boundary", inside_empty_count)
        ):
            raise ValueError("Approved class raster NoData count differs from its manifest")
        total_unclassified_inside += inside_empty_count
    if total_unclassified_inside != int(
        raw_boundary.get("unclassified_pixel_count_inside_boundary", -1)
    ):
        raise ValueError("Boundary-clipped materialization NoData total is stale")
    result = {
        "method": str(audit.get("method", "")),
        "audit_sha256": audit_hash,
        "continuous_border_sha256": border_hash,
        "mainland_interior_sha256": str(
            raw_boundary.get("mainland_interior_sha256", "")
        ),
        "publication_interior_sha256": str(
            raw_boundary.get("publication_interior_sha256", "")
        ),
        "continuous_border_component_count": component_count,
        "expected_boundary_component_count": expected_component_count,
        "mainland_interior_pixel_count": int(
            boundary.get("mainland_interior_pixel_count", 0)
        ),
        "publication_interior_pixel_count": int(
            boundary.get(
                "publication_interior_pixel_count",
                boundary.get("mainland_interior_pixel_count", 0),
            )
        ),
        "components": components,
        "selection_policy": boundary.get("selection_policy"),
        "colored_pixel_count_outside_boundary": 0,
        "unclassified_pixel_count_inside_boundary": total_unclassified_inside,
        "coverage_contract": str(raw_boundary.get("coverage_contract", "full_state")),
    }
    for field in (
        "canonical_interior_pixel_count",
        "publication_interior_exclusion_pixel_count",
        "publication_interior_component_count",
    ):
        if field in boundary:
            result[field] = int(boundary[field])
    if canonical_display_source is not None:
        result["_canonical_display_source_path"] = str(canonical_display_source)
        result["_canonical_display_grid"] = canonical_display_grid
        result["canonical_boundary_id"] = canonical_display_id
        result["canonical_display_border_sha256"] = border_hash
    if public_water is not None:
        result["internal_water_exclusion"] = public_water
    return result


def _copy_canonical_display_boundary(
    output_dir: Path,
    boundary: Dict[str, object],
) -> None:
    """Copy the exact approved display line without leaking local paths."""

    raw_source = boundary.pop("_canonical_display_source_path", None)
    raw_grid = boundary.pop("_canonical_display_grid", None)
    if raw_source is None:
        return
    if not isinstance(raw_grid, dict) or raw_grid.get("crs") != "EPSG:3857":
        raise ValueError("Canonical display border must use an EPSG:3857 grid")
    raw_bounds = raw_grid.get("bounds")
    if not isinstance(raw_bounds, list) or len(raw_bounds) != 4:
        raise ValueError("Canonical display border has invalid bounds")
    width = int(raw_grid.get("width", 0))
    height = int(raw_grid.get("height", 0))
    if width < 1 or height < 1:
        raise ValueError("Canonical display border has invalid dimensions")
    source = Path(str(raw_source))
    target = output_dir / "canonical-boundary.png"
    shutil.copy2(source, target)
    expected_hash = str(boundary.get("canonical_display_border_sha256", ""))
    if _sha256(target) != expected_hash:
        raise ValueError("Copied canonical display border changed bytes")
    with Image.open(target) as image:
        if image.size != (width, height):
            raise ValueError("Canonical display border dimensions differ from its grid")
    boundary.update(
        {
            "raster": target.name,
            "raster_sha256": expected_hash,
            "raster_width": width,
            "raster_height": height,
            "raster_bounds": list(
                _wgs84_bounds(tuple(float(value) for value in raw_bounds))
            ),
        }
    )


def _copy_internal_water_mask(
    output_dir: Path, boundary: Dict[str, object]
) -> None:
    water = boundary.get("internal_water_exclusion")
    if not isinstance(water, dict):
        return
    source_value = water.pop("_source_path", None)
    if source_value is None:
        raise ValueError("Internal-water provenance has no source artifact")
    source = Path(str(source_value))
    target = output_dir / "internal-water-exclusion-mask.png"
    shutil.copy2(source, target)
    expected_hash = str(water.get("artifact", {}).get("sha256", ""))
    if _sha256(target) != expected_hash:
        raise ValueError("Copied internal-water mask changed bytes")
    water["artifact"]["path"] = target.name


def _write_public_provenance(
    output_dir: Path,
    materialization: Dict[str, object],
    approval: Dict[str, object],
    materialization_hash: str,
    approval_hash: str,
    boundary: Dict[str, object] | None,
) -> Tuple[str, str]:
    layer_summaries = []
    for layer in materialization.get("layers", []):
        if not isinstance(layer, dict):
            continue
        artifact_hashes = {
            str(name): str(value.get("sha256", ""))
            for name, value in layer.get("artifacts", {}).items()
            if isinstance(value, dict) and value.get("sha256")
        }
        layer_summaries.append(
            {
                "id": str(layer.get("layer_id", "")),
                "width": int(layer.get("width", 0)),
                "height": int(layer.get("height", 0)),
                "final_classified_pixel_count": int(
                    layer.get("final_classified_pixel_count", 0)
                ),
                "final_pixels_by_class_id": layer.get(
                    "final_pixels_by_class_id", {}
                ),
                "artifacts_sha256": artifact_hashes,
            }
        )
    raw_source_diff = materialization.get("source_diff")
    source_diff = None
    if isinstance(raw_source_diff, dict):
        batch = raw_source_diff.get("batch", {})
        report = raw_source_diff.get("report", {})
        source_diff = {
            "fixed_point_reached": raw_source_diff.get("fixed_point_reached") is True,
            "batch_sha256": batch.get("sha256") if isinstance(batch, dict) else None,
            "report_sha256": report.get("sha256") if isinstance(report, dict) else None,
            "comparison_iterations": [
                {
                    "iteration": item.get("iteration"),
                    "status": item.get("status"),
                    "signature_sha256": item.get("signature_sha256"),
                }
                for item in raw_source_diff.get("comparison_iterations", [])
                if isinstance(item, dict)
            ],
        }
    provenance = {
        "schema_version": 1,
        "kind": "approved_publication_provenance",
        "dataset_id": str(materialization.get("dataset_id", "")),
        "materialization_sha256": materialization_hash,
        "approval": {
            "status": "approved",
            "decision_sha256": approval_hash,
            "reviewed_at": approval.get("reviewed_at"),
            "inspection_confirmed": approval.get("inspection_confirmed") is True,
            "approval_carried_forward": approval.get("approval_carried_forward")
            is True,
        },
        "alignment_sha256": approval.get("alignment_sha256"),
        "boundary": boundary,
        "source_diff": source_diff,
        "layers": layer_summaries,
    }
    path = output_dir / "provenance.json"
    path.write_text(json.dumps(provenance, indent=2) + "\n")
    return path.name, _sha256(path)


def _write_boundary_geojson(
    output_dir: Path,
    materialization: Dict[str, object],
    bounds: Tuple[float, float, float, float],
) -> Tuple[str, str, int, int]:
    raw_boundary = materialization.get("boundary_clip")
    if not isinstance(raw_boundary, dict):
        raise ValueError("Boundary GeoJSON requires a boundary-clipped materialization")
    raw_audit = raw_boundary.get("audit")
    if not isinstance(raw_audit, dict):
        raise ValueError("Boundary GeoJSON requires the clip audit")
    audit_path = Path(str(raw_audit.get("path", ""))).resolve()
    audit = json.loads(audit_path.read_text())
    audit_boundary = audit.get("boundary", {})
    interior_artifact = audit_boundary.get(
        "canonical_interior", audit_boundary.get("interior", {})
    )
    interior_path = audit_path.parent / str(interior_artifact.get("path", ""))
    if (
        not interior_path.is_file()
        or _sha256(interior_path) != interior_artifact.get("sha256")
    ):
        raise ValueError("Boundary interior is missing or hash-mismatched")
    interior = np.asarray(Image.open(interior_path)) > 0
    contours, _ = cv2.findContours(
        interior.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        raise ValueError("Boundary interior has no exterior contour")
    xmin, ymin, xmax, ymax = bounds
    transformer = Transformer.from_crs(
        "EPSG:3857", "EPSG:4326", always_xy=True
    )
    ordered_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    component_records = audit.get("boundary", {}).get("components", [])
    if component_records and len(component_records) != len(ordered_contours):
        raise ValueError("Boundary component provenance does not match its contours")
    features = []
    vertex_count = 0
    for index, raw_contour in enumerate(ordered_contours):
        contour = raw_contour[:, 0, :]
        if not np.array_equal(contour[0], contour[-1]):
            contour = np.vstack((contour, contour[0]))
        world_x = xmin + (contour[:, 0] + 0.5) * (xmax - xmin) / interior.shape[1]
        world_y = ymax - (contour[:, 1] + 0.5) * (ymax - ymin) / interior.shape[0]
        longitude, latitude = transformer.transform(world_x, world_y)
        coordinates = [
            [round(float(x), 8), round(float(y), 8)]
            for x, y in zip(longitude, latitude)
        ]
        component = component_records[index] if component_records else {}
        vertex_count += len(coordinates)
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "role": component.get(
                        "role", "approved_continuous_publication_boundary"
                    ),
                    "component_id": component.get("id", f"component-{index + 1:02d}"),
                    "source_sha256": raw_boundary.get("continuous_border_sha256"),
                    "interior_pixel_count": component.get("interior_pixel_count"),
                    "observed_source_pixel_count": component.get(
                        "observed_source_pixel_count"
                    ),
                },
                "geometry": {"type": "LineString", "coordinates": coordinates},
            }
        )
    feature = {
        "type": "FeatureCollection",
        "features": features,
    }
    path = output_dir / "boundary.geojson"
    path.write_text(json.dumps(feature, separators=(",", ":")) + "\n")
    return path.name, _sha256(path), vertex_count, len(features)


def export_categorical_tiles(
    materialized_dir: Path,
    output_dir: Path,
    minimum_zoom: int = 4,
    maximum_zoom: int = 9,
    overview_supersampling: int = 4,
    tile_encoding: str = "per_class_rgba",
) -> Dict[str, object]:
    """Export recolorable categorical tiles from an approved class raster.

    ``per_class_rgba`` retains the established one-mask-per-category format.
    ``indexed_class_id`` writes one shared class-id pyramid and is intended for
    high-resolution, many-class staging packages.
    """

    materialized_dir = materialized_dir.resolve()
    output_dir = output_dir.resolve()
    materialization_path = materialized_dir / "materialization.json"
    approval_path = materialized_dir / "materialization-review-decision.json"
    for required in (materialization_path, approval_path):
        if not required.exists():
            raise FileNotFoundError(f"Missing tile-export input: {required}")
    if not 0 <= minimum_zoom <= maximum_zoom <= 22:
        raise ValueError("Tile zooms must satisfy 0 <= minimum <= maximum <= 22")
    if overview_supersampling < 2 or overview_supersampling > 8:
        raise ValueError("Overview supersampling must be between two and eight")
    if tile_encoding not in {"per_class_rgba", "indexed_class_id"}:
        raise ValueError("Unsupported categorical tile encoding")

    materialization = json.loads(materialization_path.read_text())
    approval = json.loads(approval_path.read_text())
    materialization_hash = _sha256(materialization_path)
    approval_hash = _sha256(approval_path)
    if approval.get("status") != "approved":
        raise ValueError("Materialized raster must be approved before tile export")
    if approval.get("materialization_sha256") != materialization_hash:
        raise ValueError("Materialization approval does not match the current manifest")
    boundary = _validated_boundary_provenance(
        materialized_dir, materialization, approval
    )

    run_dir = Path(str(materialization["source_run"]))
    plan_path = run_dir / "plan.snapshot.json"
    extraction_path = run_dir / "extraction.json"
    plan = json.loads(plan_path.read_text())
    extraction = json.loads(extraction_path.read_text())
    plan_layers = {str(layer["id"]): layer for layer in plan.get("layers", [])}
    extraction_layers = {
        str(layer["id"]): layer for layer in extraction.get("layers", [])
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    source_image = _copy_source_image(plan, run_dir, output_dir)
    layer_manifests = []
    all_wgs84_bounds = []

    for report in materialization.get("layers", []):
        layer_id = str(report["layer_id"])
        definition = plan_layers.get(layer_id)
        extraction_layer = extraction_layers.get(layer_id)
        if not isinstance(definition, dict) or not isinstance(extraction_layer, dict):
            continue
        class_path = (
            materialized_dir
            / str(report["artifacts"]["class_id"]["path"])
        )
        values = np.asarray(Image.open(class_path), dtype=np.uint8)
        raw_bounds = extraction_layer.get("warp", {}).get("bounds")
        if not isinstance(raw_bounds, list) or len(raw_bounds) != 4:
            raise ValueError(f"Layer {layer_id} has no Web-Mercator bounds")
        bounds = tuple(float(value) for value in raw_bounds)
        wgs84_bounds = _wgs84_bounds(bounds)
        all_wgs84_bounds.append(wgs84_bounds)
        categories = []
        category_paths: Dict[int, list[Path]] = {
            index: [] for index, _ in enumerate(definition.get("categories", []), 1)
        }
        class_count = len(category_paths)

        indexed_raster = None
        if tile_encoding == "indexed_class_id":
            indexed_raster = _write_indexed_class_id_pyramid(
                values,
                bounds,
                output_dir,
                layer_id,
                class_count,
                minimum_zoom,
                maximum_zoom,
                overview_supersampling,
            )
        else:
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
                            class_tile = _sample_class_tile(
                                values, bounds, zoom, tile_x, tile_y
                            )
                        for class_id, category in enumerate(
                            definition.get("categories", []), 1
                        ):
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
                                alpha = np.where(
                                    class_tile == class_id, 255, 0
                                ).astype(np.uint8)
                            else:
                                alpha = np.where(
                                    class_tile == class_id, coverage_alpha, 0
                                ).astype(np.uint8)
                            white = np.full(alpha.shape, 255, dtype=np.uint8)
                            rgba = np.dstack((white, white, white, alpha))
                            Image.fromarray(rgba).save(tile_path, optimize=True)
                            category_paths[class_id].append(tile_path)

        final_counts = report.get("final_pixels_by_class_id", {})
        for class_id, category in enumerate(definition.get("categories", []), 1):
            category_id = str(category["id"])
            display_rgb = category.get(
                "display_rgb", category.get("legend_rgb", [255, 0, 255])
            )
            if display_rgb and isinstance(display_rgb[0], list):
                display_rgb = display_rgb[0]
            category_manifest = {
                "id": category_id,
                "class_id": class_id,
                "label": str(category.get("label", category_id)),
                "display_rgb": [int(value) for value in display_rgb[:3]],
                "pixel_count": int(final_counts.get(str(class_id), 0)),
            }
            if tile_encoding == "per_class_rgba":
                paths = category_paths[class_id]
                tilejson_dir = output_dir / "tiles" / layer_id / category_id
                tilejson_path = tilejson_dir / "tilejson.json"
                tilejson = {
                    "tilejson": "3.0.0",
                    "name": str(category.get("label", category_id)),
                    "scheme": "xyz",
                    "tiles": ["{z}/{x}/{y}.png"],
                    "minzoom": minimum_zoom,
                    "maxzoom": maximum_zoom,
                    "bounds": list(wgs84_bounds),
                }
                tilejson_path.write_text(json.dumps(tilejson, indent=2) + "\n")
                paths.append(tilejson_path)
                tile_set_hash = _aggregate_hash(paths, output_dir)
                category_manifest.update(
                    {
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
            categories.append(category_manifest)
        layer_manifest = {
            "id": layer_id,
            "label": layer_id.replace("-", " ").title(),
            "kind": "categorical",
            "bounds": list(wgs84_bounds),
            "categories": categories,
        }
        if indexed_raster is not None:
            layer_manifest["indexed_raster"] = indexed_raster
        layer_manifests.append(layer_manifest)

    if not layer_manifests:
        raise ValueError("No categorical materialized layer was available for export")
    west = min(bounds[0] for bounds in all_wgs84_bounds)
    south = min(bounds[1] for bounds in all_wgs84_bounds)
    east = max(bounds[2] for bounds in all_wgs84_bounds)
    north = max(bounds[3] for bounds in all_wgs84_bounds)
    if boundary is not None:
        (
            boundary_name,
            boundary_geojson_hash,
            boundary_vertex_count,
            boundary_feature_count,
        ) = (
            _write_boundary_geojson(output_dir, materialization, bounds)
        )
        boundary.update(
            {
                "geojson": boundary_name,
                "geojson_sha256": boundary_geojson_hash,
                "geojson_vertex_count": boundary_vertex_count,
                "geojson_feature_count": boundary_feature_count,
            }
        )
        _copy_canonical_display_boundary(output_dir, boundary)
        _copy_internal_water_mask(output_dir, boundary)
    provenance_name, provenance_hash = _write_public_provenance(
        output_dir,
        materialization,
        approval,
        materialization_hash,
        approval_hash,
        boundary,
    )
    dataset = {
        "schema_version": 1,
        "status": "approved_publication",
        "id": str(materialization["dataset_id"]),
        "title": str(plan.get("title", materialization["dataset_id"])),
        "bounds": [west, south, east, north],
        "center": [(west + east) / 2, (south + north) / 2],
        "minimum_zoom": minimum_zoom,
        "maximum_native_zoom": maximum_zoom,
        "overscaling": "nearest",
        "categorical_tile_encoding": tile_encoding,
        "overview": {
            "mode": "dominant_class_with_fractional_coverage",
            "supersampling": overview_supersampling,
            "overview_zooms": [minimum_zoom, maximum_zoom - 1],
            "exact_binary_zoom": maximum_zoom,
        },
        "source_image": source_image,
        "materialization": {
            "sha256": materialization_hash,
        },
        "approval": {
            "status": "approved",
            "reviewed_at": approval.get("reviewed_at"),
            "sha256": approval_hash,
        },
        "provenance": {
            "manifest": provenance_name,
            "sha256": provenance_hash,
        },
        "boundary": boundary,
        "layers": layer_manifests,
    }
    dataset_path = output_dir / "dataset.json"
    dataset_path.write_text(json.dumps(dataset, indent=2) + "\n")
    result = {
        **dataset,
        "dataset_manifest_sha256": _sha256(dataset_path),
    }
    return result
