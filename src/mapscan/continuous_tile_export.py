"""Export a continuous MapScan raster as deterministic native-color XYZ tiles.

Approved publications retain the strict review-decision gate.  An explicit
review-preview mode produces a clearly non-publishable package so a candidate
can be inspected against the live basemap before author approval.
"""

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


def _load(path: Path) -> Dict[str, object]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _artifact_path(
    run_dir: Path, artifacts: Dict[str, object], name: str
) -> Tuple[Path, Dict[str, object]]:
    raw_record = artifacts.get(name)
    if not isinstance(raw_record, dict):
        raise ValueError(f"Continuous extraction has no {name} artifact")
    path = run_dir / str(raw_record.get("path", ""))
    if not path.is_file() or _sha256(path) != raw_record.get("sha256"):
        raise ValueError(f"Continuous extraction artifact is missing or stale: {name}")
    return path, raw_record


def _aggregate_hash(paths: Iterable[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(_sha256(path).encode())
        digest.update(b"\n")
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


def _wgs84_bounds(
    bounds: Tuple[float, float, float, float]
) -> Tuple[float, float, float, float]:
    transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    west, south = transformer.transform(bounds[0], bounds[1])
    east, north = transformer.transform(bounds[2], bounds[3])
    return float(west), float(south), float(east), float(north)


def _sample_rgba_tile(
    rgb: np.ndarray,
    valid: np.ndarray,
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
    source_x = np.floor((world_x - xmin) * rgb.shape[1] / (xmax - xmin)).astype(
        np.int64
    )
    source_y = np.floor((ymax - world_y) * rgb.shape[0] / (ymax - ymin)).astype(
        np.int64
    )
    valid_x = (source_x >= 0) & (source_x < rgb.shape[1])
    valid_y = (source_y >= 0) & (source_y < rgb.shape[0])
    tile = np.zeros((tile_size, tile_size, 4), dtype=np.uint8)
    if np.any(valid_x) and np.any(valid_y):
        sampled_rgb = rgb[np.ix_(source_y[valid_y], source_x[valid_x])]
        sampled_valid = valid[np.ix_(source_y[valid_y], source_x[valid_x])]
        rows = np.flatnonzero(valid_y)
        columns = np.flatnonzero(valid_x)
        tile[np.ix_(rows, columns, np.arange(3))] = sampled_rgb
        tile[np.ix_(rows, columns, [3])] = (
            sampled_valid.astype(np.uint8)[:, :, None] * 255
        )
    return tile


def _sample_rgba_overview(
    rgb: np.ndarray,
    valid: np.ndarray,
    bounds: Tuple[float, float, float, float],
    zoom: int,
    tile_x: int,
    tile_y: int,
    supersampling: int,
) -> np.ndarray:
    if supersampling < 2:
        raise ValueError("Continuous overview supersampling must be at least two")
    samples = _sample_rgba_tile(
        rgb,
        valid,
        bounds,
        zoom,
        tile_x,
        tile_y,
        tile_size=TILE_SIZE * supersampling,
    )
    blocks = samples.reshape(
        TILE_SIZE, supersampling, TILE_SIZE, supersampling, 4
    )
    sample_valid = blocks[:, :, :, :, 3] > 0
    coverage_count = sample_valid.sum(axis=(1, 3), dtype=np.uint16)
    output = np.zeros((TILE_SIZE, TILE_SIZE, 4), dtype=np.uint8)
    for channel in range(3):
        weighted = np.where(sample_valid, blocks[:, :, :, :, channel], 0)
        channel_sum = weighted.sum(axis=(1, 3), dtype=np.uint32)
        np.divide(
            channel_sum,
            coverage_count,
            out=output[:, :, channel],
            where=coverage_count > 0,
            casting="unsafe",
        )
    output[:, :, 3] = np.rint(
        coverage_count * (255.0 / (supersampling * supersampling))
    ).astype(np.uint8)
    return output


def _continuous_band_definitions(
    ramp_stops: list[Dict[str, object]],
    special_values: list[Dict[str, object]],
) -> list[Dict[str, object]]:
    """Build exhaustive, non-overlapping selection bands from legend stops."""

    ordered = sorted(ramp_stops, key=lambda item: float(item["value"]))
    if not ordered:
        raise ValueError("Continuous band export needs at least one ramp stop")
    values = [float(item["value"]) for item in ordered]
    if len(values) != len(set(values)):
        raise ValueError("Continuous ramp stops must have unique values")

    def number(value: float) -> str:
        return f"{int(value):,}" if value.is_integer() else f"{value:g}"

    first = ordered[0]
    first_value = values[0]
    special_color = (
        list(special_values[0]["display_rgb"])
        if special_values
        else list(first["display_rgb"])
    )
    definitions: list[Dict[str, object]] = [
        {
            "id": f"elevation-below-{number(first_value).replace(',', '')}-m",
            "label": f"Below {number(first_value)} m / depression",
            "lower_bound": None,
            "upper_bound": first_value,
            "lower_inclusive": True,
            "upper_inclusive": False,
            "display_rgb": special_color,
            "special_value_ids": [str(item["id"]) for item in special_values],
        }
    ]
    for index, item in enumerate(ordered):
        lower = float(item["value"])
        upper = values[index + 1] if index + 1 < len(values) else None
        lower_text = number(lower)
        upper_text = number(upper) if upper is not None else None
        definitions.append(
            {
                "id": (
                    f"elevation-{lower_text.replace(',', '')}-m-plus"
                    if upper is None
                    else (
                        f"elevation-{lower_text.replace(',', '')}-to-"
                        f"{upper_text.replace(',', '')}-m"
                    )
                ),
                "label": (
                    f"{lower_text} m+"
                    if upper is None
                    else f"{lower_text}–<{upper_text} m"
                ),
                "lower_bound": lower,
                "upper_bound": upper,
                "lower_inclusive": True,
                "upper_inclusive": False,
                "display_rgb": list(item["display_rgb"]),
                "special_value_ids": [],
            }
        )
    return definitions


def _continuous_band_masks(
    encoded: np.ndarray,
    interior: np.ndarray,
    encoding: Dict[str, object],
    definitions: list[Dict[str, object]],
) -> list[np.ndarray]:
    """Partition every approved value cell into exactly one legend interval."""

    offset = float(encoding["offset"])
    scale = float(encoding["scale"])
    decoded = np.full(encoded.shape, np.nan, dtype=np.float64)
    decoded[interior] = offset + (encoded[interior].astype(np.float64) - 1.0) * scale
    masks = []
    covered = np.zeros(interior.shape, dtype=bool)
    for definition in definitions:
        selected = interior.copy()
        lower = definition.get("lower_bound")
        upper = definition.get("upper_bound")
        if lower is not None:
            selected &= decoded >= float(lower)
        if upper is not None:
            selected &= decoded < float(upper)
        if np.any(covered & selected):
            raise ValueError("Continuous elevation bands overlap")
        masks.append(selected)
        covered |= selected
    if not np.array_equal(covered, interior):
        raise ValueError("Continuous elevation bands do not cover every approved value")
    return masks


def _write_boundary_geojson(
    output_dir: Path,
    canonical_interior: np.ndarray,
    bounds: Tuple[float, float, float, float],
    boundary_sha256: str,
) -> Tuple[str, str, int, int, list[Dict[str, object]]]:
    contours, _ = cv2.findContours(
        canonical_interior.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    ordered = sorted(contours, key=cv2.contourArea, reverse=True)
    if len(ordered) != 5:
        raise ValueError(
            f"Continuous canonical interior has {len(ordered)} components; expected five"
        )
    xmin, ymin, xmax, ymax = bounds
    transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    features = []
    components = []
    vertex_count = 0
    for index, raw_contour in enumerate(ordered):
        contour = raw_contour[:, 0, :]
        if not np.array_equal(contour[0], contour[-1]):
            contour = np.vstack((contour, contour[0]))
        world_x = xmin + (contour[:, 0] + 0.5) * (xmax - xmin) / canonical_interior.shape[1]
        world_y = ymax - (contour[:, 1] + 0.5) * (ymax - ymin) / canonical_interior.shape[0]
        longitude, latitude = transformer.transform(world_x, world_y)
        coordinates = [
            [round(float(x), 8), round(float(y), 8)]
            for x, y in zip(longitude, latitude)
        ]
        component_id = "mainland" if index == 0 else f"canonical-island-{index:02d}"
        role = (
            "canonical_mainland_clipping"
            if index == 0
            else "author_approved_canonical_island"
        )
        component_mask = np.zeros(canonical_interior.shape, dtype=np.uint8)
        cv2.drawContours(component_mask, [raw_contour], -1, 1, thickness=cv2.FILLED)
        component = {
            "id": component_id,
            "role": role,
            "authority": "california-county-detail-border-v2",
            "interior_pixel_count": int(np.count_nonzero(component_mask)),
        }
        components.append(component)
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "role": role,
                    "component_id": component_id,
                    "source_sha256": boundary_sha256,
                    "interior_pixel_count": component["interior_pixel_count"],
                },
                "geometry": {"type": "LineString", "coordinates": coordinates},
            }
        )
        vertex_count += len(coordinates)
    path = output_dir / "boundary.geojson"
    path.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, separators=(",", ":"))
        + "\n"
    )
    return path.name, _sha256(path), vertex_count, len(features), components


def export_continuous_tiles(
    run_dir: Path,
    audit_path: Path,
    output_dir: Path,
    minimum_zoom: int = 4,
    maximum_zoom: int = 9,
    overview_supersampling: int = 4,
    review_preview: bool = False,
) -> Dict[str, object]:
    """Export one native-color surface while retaining the exact 16-bit values."""

    run_dir = run_dir.resolve()
    audit_path = audit_path.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise ValueError("Continuous tile export requires a fresh output directory")
    if minimum_zoom < 0 or maximum_zoom < minimum_zoom:
        raise ValueError("Invalid continuous tile zoom range")

    manifest_path = run_dir / "continuous-extraction.json"
    decision_path = run_dir / "review-decision.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Continuous export needs its extraction manifest")
    if not review_preview and not decision_path.is_file():
        raise FileNotFoundError(
            "Approved continuous export needs its review decision"
        )
    manifest = _load(manifest_path)
    decision = _load(decision_path) if decision_path.is_file() else None
    manifest_hash = _sha256(manifest_path)
    decision_hash = _sha256(decision_path) if decision_path.is_file() else None
    if manifest.get("extraction_kind") != "continuous_color_ramp":
        raise ValueError("Run is not a continuous color-ramp extraction")
    if not review_preview:
        if decision is None or decision.get("status") != "approved":
            raise ValueError("Continuous extraction has not been approved")
        if decision.get("review_manifest_sha256") != manifest_hash:
            raise ValueError("Continuous approval is stale")

    audit = _load(audit_path)
    if (
        audit.get("status") != "pass"
        or int(audit.get("source_different_pixel_count", -1)) != 0
        or int(audit.get("web_different_pixel_count", -1)) != 0
        or any(int(value) != 0 for value in audit.get("evidence_different_pixel_counts", {}).values())
    ):
        raise ValueError("Continuous independent diff audit did not pass")
    audited_run = Path(str(audit.get("run", "")))
    if not audited_run.is_absolute():
        audited_run = (Path.cwd() / audited_run).resolve()
    if audited_run != run_dir:
        raise ValueError("Continuous diff audit belongs to a different run")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("Continuous extraction has no artifacts")
    preview_path, preview_record = _artifact_path(
        run_dir, artifacts, "web_mercator_preview"
    )
    value_path, value_record = _artifact_path(
        run_dir, artifacts, "web_mercator_value"
    )
    interior_path, interior_record = _artifact_path(
        run_dir, artifacts, "web_mercator_publication_interior_mask"
    )
    water_path, water_record = _artifact_path(
        run_dir, artifacts, "web_mercator_internal_water_mask"
    )
    values = np.asarray(Image.open(value_path), dtype=np.uint16)
    preview = np.asarray(Image.open(preview_path).convert("RGBA"), dtype=np.uint8)
    interior = np.asarray(Image.open(interior_path).convert("L"), dtype=np.uint8) > 0
    water = np.asarray(Image.open(water_path).convert("L"), dtype=np.uint8) > 0
    if preview.shape[:2] != values.shape or values.shape != interior.shape or water.shape != values.shape:
        raise ValueError("Continuous publication artifacts use different grids")
    if not np.array_equal(values > 0, interior):
        raise ValueError("Continuous values do not exactly cover the publication interior")
    if np.any(interior & water):
        raise ValueError("Continuous publication still colors internal water")
    if int(manifest.get("target", {}).get("unknown_inside_pixel_count", -1)) != 0:
        raise ValueError("Continuous extraction still has unknown interior pixels")
    if int(manifest.get("target", {}).get("colored_outside_pixel_count", -1)) != 0:
        raise ValueError("Continuous extraction colors outside its approved boundary")

    warp = manifest.get("warp")
    if not isinstance(warp, dict) or warp.get("crs") != "EPSG:3857":
        raise ValueError("Continuous extraction has no EPSG:3857 publication grid")
    raw_bounds = warp.get("bounds")
    if not isinstance(raw_bounds, list) or len(raw_bounds) != 4:
        raise ValueError("Continuous extraction has invalid Web-Mercator bounds")
    bounds = tuple(float(value) for value in raw_bounds)
    wgs84_bounds = _wgs84_bounds(bounds)

    canonical_clip = manifest.get("canonical_clip")
    if not isinstance(canonical_clip, dict):
        raise ValueError("Continuous extraction has no canonical clip")
    active_manifest_record = canonical_clip.get("active_manifest")
    if not isinstance(active_manifest_record, dict):
        raise ValueError("Continuous extraction has no active boundary manifest")
    canonical_manifest_path = Path(str(active_manifest_record.get("path", ""))).resolve()
    if (
        not canonical_manifest_path.is_file()
        or _sha256(canonical_manifest_path) != active_manifest_record.get("sha256")
    ):
        raise ValueError("Continuous canonical boundary manifest is stale")
    canonical_manifest = _load(canonical_manifest_path)
    overlay_record = canonical_manifest.get("artifacts", {}).get("overlay")
    if not isinstance(overlay_record, dict):
        raise ValueError("Canonical boundary has no display overlay")
    overlay_source = canonical_manifest_path.parent / str(overlay_record.get("path", ""))
    if not overlay_source.is_file() or _sha256(overlay_source) != overlay_record.get("sha256"):
        raise ValueError("Canonical display boundary is stale")
    if not review_preview:
        assert decision is not None
        reviewed_boundary = decision.get("canonical_boundary")
        if (
            not isinstance(reviewed_boundary, dict)
            or reviewed_boundary.get("manifest_sha256")
            != active_manifest_record.get("sha256")
            or reviewed_boundary.get("overlay_sha256")
            != overlay_record.get("sha256")
        ):
            raise ValueError(
                "Continuous approval belongs to another canonical boundary"
            )

    output_dir.mkdir(parents=True)
    source_record = manifest.get("source")
    source_name = None
    if isinstance(source_record, dict):
        source_path = Path(str(source_record.get("path", "")))
        if not source_path.is_absolute():
            source_path = (Path.cwd() / source_path).resolve()
        if source_path.is_file() and _sha256(source_path) == source_record.get("sha256"):
            source_name = f"source{source_path.suffix.lower() or '.img'}"
            shutil.copy2(source_path, output_dir / source_name)

    value_target = output_dir / "continuous-values.png"
    interior_target = output_dir / "publication-interior-mask.png"
    water_target = output_dir / "internal-water-exclusion-mask.png"
    boundary_target = output_dir / "canonical-boundary.png"
    shutil.copy2(value_path, value_target)
    shutil.copy2(interior_path, interior_target)
    shutil.copy2(water_path, water_target)
    shutil.copy2(overlay_source, boundary_target)

    rgb = preview[:, :, :3]
    band_definitions = _continuous_band_definitions(
        list(manifest.get("ramp_stops", [])),
        list(manifest.get("special_values", [])),
    )
    band_masks = _continuous_band_masks(
        values,
        interior,
        dict(manifest.get("encoding", {})),
        band_definitions,
    )
    band_records = [
        {**definition, "pixel_count": int(np.count_nonzero(mask))}
        for definition, mask in zip(band_definitions, band_masks)
    ]
    all_tile_paths: list[Path] = []

    def write_tile_set(
        category_id: str,
        label: str,
        valid: np.ndarray,
    ) -> tuple[list[Path], Path, str]:
        tile_root = output_dir / "tiles" / "elevation" / category_id
        paths = []
        for zoom in range(minimum_zoom, maximum_zoom + 1):
            x_range, y_range = _tile_range(bounds, zoom)
            for tile_x in x_range:
                for tile_y in y_range:
                    if zoom < maximum_zoom:
                        tile = _sample_rgba_overview(
                            rgb,
                            valid,
                            bounds,
                            zoom,
                            tile_x,
                            tile_y,
                            overview_supersampling,
                        )
                    else:
                        tile = _sample_rgba_tile(
                            rgb, valid, bounds, zoom, tile_x, tile_y
                        )
                    path = tile_root / str(zoom) / str(tile_x) / f"{tile_y}.png"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(tile, mode="RGBA").save(path, optimize=True)
                    paths.append(path)
        tilejson_path = tile_root / "tilejson.json"
        tilejson_path.write_text(
            json.dumps(
                {
                    "tilejson": "3.0.0",
                    "name": (
                        "California elevation"
                        if category_id == "elevation-surface"
                        else f"California elevation — {label}"
                    ),
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
        all_tile_paths.extend(paths)
        return paths, tilejson_path, _aggregate_hash(
            [*paths, tilejson_path], output_dir
        )

    tile_paths, tilejson_path, tile_set_hash = write_tile_set(
        "elevation-surface", "full surface", interior
    )
    band_tile_sets = [
        write_tile_set(str(definition["id"]), str(definition["label"]), mask)
        for definition, mask in zip(band_definitions, band_masks)
    ]

    canonical_interior = interior | water
    (
        boundary_geojson_name,
        boundary_geojson_hash,
        boundary_vertex_count,
        boundary_feature_count,
        boundary_components,
    ) = _write_boundary_geojson(
        output_dir,
        canonical_interior,
        bounds,
        str(overlay_record["sha256"]),
    )
    source_grid = canonical_manifest.get("source_grid", {})
    canonical_bounds = tuple(float(value) for value in source_grid.get("bounds", []))
    if len(canonical_bounds) != 4:
        raise ValueError("Canonical display boundary has invalid bounds")
    publication_component_count, _ = cv2.connectedComponents(
        interior.astype(np.uint8), connectivity=8
    )
    boundary = {
        "method": (
            "review_preview_continuous_extraction_exact_canonical_clip"
            if review_preview
            else "approved_continuous_extraction_exact_canonical_clip"
        ),
        "continuous_border_component_count": 5,
        "expected_boundary_component_count": 5,
        "mainland_interior_pixel_count": int(boundary_components[0]["interior_pixel_count"]),
        "canonical_interior_pixel_count": int(np.count_nonzero(canonical_interior)),
        "publication_interior_pixel_count": int(np.count_nonzero(interior)),
        "publication_interior_exclusion_pixel_count": int(np.count_nonzero(water)),
        "publication_interior_component_count": int(publication_component_count - 1),
        "components": boundary_components,
        "colored_pixel_count_outside_boundary": 0,
        "unclassified_pixel_count_inside_boundary": 0,
        "coverage_contract": "full_state",
        "canonical_boundary_id": canonical_manifest.get("canonical_boundary_id"),
        "canonical_display_border_sha256": overlay_record["sha256"],
        "internal_water_exclusion": {
            "method": canonical_clip.get("internal_water_exclusion", {}).get("method"),
            "excluded_interior_pixel_count": int(np.count_nonzero(water)),
            "colored_water_policy": "always transparent",
            "artifact": {
                "path": water_target.name,
                "sha256": water_record["sha256"],
            },
        },
        "geojson": boundary_geojson_name,
        "geojson_sha256": boundary_geojson_hash,
        "geojson_vertex_count": boundary_vertex_count,
        "geojson_feature_count": boundary_feature_count,
        "raster": boundary_target.name,
        "raster_sha256": overlay_record["sha256"],
        "raster_width": int(source_grid.get("width", 0)),
        "raster_height": int(source_grid.get("height", 0)),
        "raster_bounds": list(_wgs84_bounds(canonical_bounds)),
    }

    audit_hash = _sha256(audit_path)
    alignment_path = Path(str(manifest.get("alignment", "")))
    if not alignment_path.is_absolute():
        alignment_path = (Path.cwd() / alignment_path).resolve()
    alignment_hash = _sha256(alignment_path) if alignment_path.is_file() else None
    approval_status = "not_approved" if review_preview else "approved"
    reviewed_at = decision.get("reviewed_at") if decision is not None else None
    provenance = {
        "schema_version": 1,
        "kind": (
            "continuous_review_preview_provenance"
            if review_preview
            else "approved_publication_provenance"
        ),
        "dataset_id": manifest["dataset_id"],
        "materialization_sha256": manifest_hash,
        "publication_source_kind": (
            "unapproved_continuous_extraction_review_preview"
            if review_preview
            else "approved_continuous_extraction"
        ),
        "approval": {
            "status": approval_status,
            "decision_sha256": decision_hash,
            "reviewed_at": reviewed_at,
            "inspection_confirmed": not review_preview,
            "approval_carried_forward": False,
        },
        "alignment_sha256": alignment_hash,
        "continuous_diff_audit_sha256": audit_hash,
        "boundary": boundary,
        "continuous": {
            "units": manifest.get("units"),
            "encoding": manifest.get("encoding"),
            "ramp_stops": manifest.get("ramp_stops"),
            "special_values": manifest.get("special_values"),
            "selection_bands": band_records,
            "selection_band_source_sha256": value_record["sha256"],
            "value_raster_sha256": value_record["sha256"],
            "publication_interior_sha256": interior_record["sha256"],
            "preview_sha256": preview_record["sha256"],
        },
    }
    provenance_path = output_dir / "provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    provenance_hash = _sha256(provenance_path)

    category = {
        "id": "elevation-surface",
        "class_id": 1,
        "label": "Full elevation surface",
        "display_rgb": [29, 107, 0],
        "pixel_count": int(np.count_nonzero(interior)),
        "tile_template": (
            "tiles/elevation/elevation-surface/{z}/{x}/{y}.png"
            f"?v={tile_set_hash[:16]}"
        ),
        "tilejson": "tiles/elevation/elevation-surface/tilejson.json",
        "tile_file_count": len(tile_paths),
        "tile_set_sha256": tile_set_hash,
        "encoding": "rgba_native_continuous_with_fractional_coverage_overviews",
        "render_mode": "native_color",
        "color_editable": False,
        "category_role": "continuous_surface",
        "default_enabled": True,
        "units": manifest.get("units"),
        "legend_stops": manifest.get("ramp_stops", []),
        "special_values": manifest.get("special_values", []),
    }
    band_categories = []
    for index, (definition, mask, tile_set) in enumerate(
        zip(band_definitions, band_masks, band_tile_sets), start=2
    ):
        band_paths, band_tilejson_path, band_tile_hash = tile_set
        band_id = str(definition["id"])
        band_categories.append(
            {
                "id": band_id,
                "class_id": index,
                "label": definition["label"],
                "display_rgb": definition["display_rgb"],
                "pixel_count": int(np.count_nonzero(mask)),
                "tile_template": (
                    f"tiles/elevation/{band_id}/{{z}}/{{x}}/{{y}}.png"
                    f"?v={band_tile_hash[:16]}"
                ),
                "tilejson": str(band_tilejson_path.relative_to(output_dir)),
                "tile_file_count": len(band_paths),
                "tile_set_sha256": band_tile_hash,
                "encoding": "rgba_native_continuous_band_with_fractional_coverage_overviews",
                "render_mode": "native_color",
                "color_editable": False,
                "category_role": "continuous_band",
                "default_enabled": False,
                "units": manifest.get("units"),
                "value_range": {
                    "lower_bound": definition["lower_bound"],
                    "upper_bound": definition["upper_bound"],
                    "lower_inclusive": definition["lower_inclusive"],
                    "upper_inclusive": definition["upper_inclusive"],
                    "special_value_ids": definition["special_value_ids"],
                },
            }
        )
    dataset = {
        "schema_version": 1,
        "status": "needs_visual_review" if review_preview else "approved_publication",
        "id": manifest["dataset_id"],
        "title": manifest["title"],
        "bounds": list(wgs84_bounds),
        "center": [
            (wgs84_bounds[0] + wgs84_bounds[2]) / 2,
            (wgs84_bounds[1] + wgs84_bounds[3]) / 2,
        ],
        "minimum_zoom": minimum_zoom,
        "maximum_native_zoom": maximum_zoom,
        "overscaling": "nearest",
        "overview": {
            "mode": "native_color_mean_with_fractional_coverage",
            "supersampling": overview_supersampling,
            "overview_zooms": [minimum_zoom, maximum_zoom - 1],
            "exact_binary_zoom": maximum_zoom,
        },
        "source_image": source_name,
        "materialization": {
            "kind": (
                "continuous_extraction_review_candidate"
                if review_preview
                else "continuous_extraction"
            ),
            "sha256": manifest_hash,
        },
        "approval": {
            "status": approval_status,
            "reviewed_at": reviewed_at,
            "sha256": decision_hash,
        },
        "provenance": {"manifest": provenance_path.name, "sha256": provenance_hash},
        "boundary": boundary,
        "continuous": {
            "units": manifest.get("units"),
            "encoding": manifest.get("encoding"),
            "ramp_stops": manifest.get("ramp_stops"),
            "special_values": manifest.get("special_values"),
            "selection_bands": band_records,
            "value_raster": {
                "path": value_target.name,
                "sha256": value_record["sha256"],
                "width": int(values.shape[1]),
                "height": int(values.shape[0]),
            },
            "publication_interior": {
                "path": interior_target.name,
                "sha256": interior_record["sha256"],
            },
            "diff_audit_sha256": audit_hash,
        },
        "layers": [
            {
                "id": "elevation",
                "label": "Elevation",
                "kind": "continuous",
                "bounds": list(wgs84_bounds),
                "categories": [category, *band_categories],
            }
        ],
    }
    dataset_path = output_dir / "dataset.json"
    dataset_path.write_text(json.dumps(dataset, indent=2) + "\n")
    return {
        **dataset,
        "dataset_manifest_sha256": _sha256(dataset_path),
        "tile_file_count": len(all_tile_paths),
        "tile_set_sha256": tile_set_hash,
    }
