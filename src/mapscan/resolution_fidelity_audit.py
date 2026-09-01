"""Audit restart resolution and plan source-native autonomous comparisons.

The accepted alignment contract maps a pinned Mapbox target grid back to the
original source raster.  Sampling the local Jacobian of that mapping tells us
whether one target pixel collapses multiple source pixels.  This module turns
that measurement into a deterministic supersampling and regional-diff policy;
it does not mutate accepted artifacts or render map-scale diagnostics.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from pyproj import CRS, Transformer


SCHEMA_VERSION = "mapscan.resolution-fidelity-audit.v1"
_FIDELITY_TERMS = (
    "mismatch_fraction",
    "match_fraction",
    "roundtrip",
    "observed_fraction",
    "inferred_fraction",
    "coverage_fraction",
    "iou",
)
_PUBLICATION_ALIASES = {
    "deer": "deer-distribution-california",
    "elevation": "california-topography-elevation",
    "farms-v2": "california-agricultural-land-use",
    "forest": "california-forest-cover",
    "plantzone": "california-plant-hardiness-zones",
    "quake": "earthquake-shaking-potential-california-2025-final-hybrid-v1",
    "rainfall-historical": "california-annual-average-precipitation-1900-1960",
    "rainfall-1981-2010": "california-average-annual-precipitation-1981-2010",
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _reference_pixels_to_web_mercator(
    pixel_x: np.ndarray, pixel_y: np.ndarray, grid: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    minimum_x, minimum_y, maximum_x, maximum_y = map(float, grid["bounds"])
    width, height = int(grid["width"]), int(grid["height"])
    x = minimum_x + pixel_x / max(width - 1, 1) * (maximum_x - minimum_x)
    y = maximum_y - pixel_y / max(height - 1, 1) * (maximum_y - minimum_y)
    return x, y


def _residual_displacement(
    transform: Mapping[str, Any], pixel_x: np.ndarray, pixel_y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    residual = transform["residual_warp"]
    centers = np.asarray(residual["centers_reference_px"], dtype=np.float64)
    coefficients = np.asarray(residual["coefficients_source_px"], dtype=np.float64)
    radius = float(residual["radius_reference_px"])
    points = np.column_stack((pixel_x.ravel(), pixel_y.ravel()))
    distance = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
    scaled = distance / radius
    remaining = np.clip(1.0 - scaled, 0.0, 1.0)
    displacement = (remaining**4 * (4.0 * scaled + 1.0)) @ coefficients
    shape = pixel_x.shape
    return displacement[:, 0].reshape(shape), displacement[:, 1].reshape(shape)


def reference_to_source_points(
    transform: Mapping[str, Any], pixel_x: np.ndarray, pixel_y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Map small target-grid arrays to original source pixels."""

    pixel_x = np.asarray(pixel_x, dtype=np.float64)
    pixel_y = np.asarray(pixel_y, dtype=np.float64)
    if pixel_x.shape != pixel_y.shape:
        raise ValueError("reference point arrays must have the same shape")
    kind = str(transform.get("kind", ""))
    if kind == "regular_global_mapbox_registration":
        matrix = np.asarray(
            transform["reference_pixel_to_source_original_matrix"], dtype=np.float64
        )
        denominator = matrix[2, 0] * pixel_x + matrix[2, 1] * pixel_y + matrix[2, 2]
        mapped_x = (
            matrix[0, 0] * pixel_x + matrix[0, 1] * pixel_y + matrix[0, 2]
        ) / denominator
        mapped_y = (
            matrix[1, 0] * pixel_x + matrix[1, 1] * pixel_y + matrix[1, 2]
        ) / denominator
        return mapped_x, mapped_y
    if kind not in {
        "projection_aware_mapbox_registration",
        "projection_aware_residual_warp_mapbox_registration",
    }:
        raise ValueError(f"unsupported accepted transform kind: {kind}")

    projection = transform["projection"]
    transformer = Transformer.from_crs(
        "EPSG:3857", CRS.from_wkt(projection["crs_wkt"]), always_xy=True
    )
    web_x, web_y = _reference_pixels_to_web_mercator(
        pixel_x, pixel_y, transform["target_grid"]
    )
    projected_x, projected_y = transformer.transform(web_x, web_y, errcheck=True)
    center = np.asarray(projection["normalization_center"], dtype=np.float64)
    scale = float(projection["normalization_scale"])
    normalized_x = (projected_x - center[0]) / scale
    normalized_y = -(projected_y - center[1]) / scale
    matrix = np.asarray(
        transform["candidate_normalized_to_source_original_matrix"], dtype=np.float64
    )
    mapped_x = normalized_x * matrix[0, 0] + normalized_y * matrix[0, 1] + matrix[0, 2]
    mapped_y = normalized_x * matrix[1, 0] + normalized_y * matrix[1, 1] + matrix[1, 2]
    if kind == "projection_aware_residual_warp_mapbox_registration":
        residual_x, residual_y = _residual_displacement(transform, pixel_x, pixel_y)
        mapped_x += residual_x
        mapped_y += residual_y
    return mapped_x, mapped_y


def _recommended_factor(source_pixels_per_target_pixel_p95: float) -> int:
    """Choose a small integer grid factor with a 10% antialiasing margin."""

    required = source_pixels_per_target_pixel_p95 * 1.10
    if required <= 1.0:
        return 1
    return min(4, max(2, int(math.ceil(required))))


def analyze_transform_resolution(
    transform: Mapping[str, Any], *, sample_rows: int = 25, sample_columns: int = 25
) -> dict[str, Any]:
    """Measure local source sampling density across the accepted warp."""

    if sample_rows < 3 or sample_columns < 3:
        raise ValueError("resolution audit requires at least a 3 by 3 sample grid")
    grid = transform["target_grid"]
    width, height = int(grid["width"]), int(grid["height"])
    source_height, source_width = map(int, transform["source_original_shape"])
    sample_x, sample_y = np.meshgrid(
        np.linspace(1.0, max(width - 3, 1), sample_columns),
        np.linspace(1.0, max(height - 3, 1), sample_rows),
    )
    source_x, source_y = reference_to_source_points(transform, sample_x, sample_y)
    source_dx_x, source_dx_y = reference_to_source_points(
        transform, sample_x + 1.0, sample_y
    )
    source_dy_x, source_dy_y = reference_to_source_points(
        transform, sample_x, sample_y + 1.0
    )
    valid = (
        np.isfinite(source_x)
        & np.isfinite(source_y)
        & (source_x >= 0)
        & (source_x < source_width)
        & (source_y >= 0)
        & (source_y < source_height)
    )
    jacobians = np.stack(
        (
            source_dx_x - source_x,
            source_dy_x - source_x,
            source_dx_y - source_y,
            source_dy_y - source_y,
        ),
        axis=-1,
    ).reshape(sample_rows, sample_columns, 2, 2)
    singular_values = np.linalg.svd(jacobians[valid], compute_uv=False)
    if not len(singular_values):
        raise ValueError("accepted transform has no sampled overlap with its source")
    maximum_axis = singular_values[:, 0]
    minimum_axis = singular_values[:, 1]
    p95 = float(np.percentile(maximum_axis, 95))
    factor = _recommended_factor(p95)
    working_scale = float(transform.get("working_scale_from_original", 1.0))
    return {
        "sample_grid": [sample_rows, sample_columns],
        "sample_count": int(valid.size),
        "in_source_sample_count": int(np.count_nonzero(valid)),
        "in_source_sample_fraction": float(np.mean(valid)),
        "source_pixels_per_target_pixel": {
            "median_maximum_axis": float(np.median(maximum_axis)),
            "p95_maximum_axis": p95,
            "maximum_axis": float(np.max(maximum_axis)),
            "median_minimum_axis": float(np.median(minimum_axis)),
        },
        "target_grid_supersampling_factor": factor,
        "target_grid_resolution_loss": bool(factor > 1),
        "equivalent_native_zoom_increment": int(math.ceil(math.log2(factor))),
        "alignment_working_scale_from_original": working_scale,
        "alignment_working_downsampling_factor": 1.0 / working_scale,
        "registration_detail_risk": bool(working_scale < 0.5),
    }


def plan_native_regional_diffs(
    transform: Mapping[str, Any], *, rows: int = 6, columns: int = 6
) -> list[dict[str, Any]]:
    """Plan target cells that must be compared in original-source pixel space."""

    if rows < 1 or columns < 1:
        raise ValueError("regional-diff grid must be positive")
    grid = transform["target_grid"]
    width, height = int(grid["width"]), int(grid["height"])
    source_height, source_width = map(int, transform["source_original_shape"])
    cells: list[dict[str, Any]] = []
    for row in range(rows):
        top, bottom = round(row * height / rows), round((row + 1) * height / rows)
        for column in range(columns):
            left, right = round(column * width / columns), round(
                (column + 1) * width / columns
            )
            xs = np.asarray([left, right - 1, left, right - 1, (left + right) / 2])
            ys = np.asarray([top, top, bottom - 1, bottom - 1, (top + bottom) / 2])
            source_x, source_y = reference_to_source_points(transform, xs, ys)
            valid = (
                (source_x >= 0)
                & (source_x < source_width)
                & (source_y >= 0)
                & (source_y < source_height)
            )
            if not np.any(valid):
                continue
            clipped_x = np.clip(source_x[valid], 0, source_width - 1)
            clipped_y = np.clip(source_y[valid], 0, source_height - 1)
            source_bounds = [
                int(math.floor(float(np.min(clipped_x)))),
                int(math.floor(float(np.min(clipped_y)))),
                int(math.ceil(float(np.max(clipped_x)))) + 1,
                int(math.ceil(float(np.max(clipped_y)))) + 1,
            ]
            cells.append(
                {
                    "id": f"r{row + 1}-c{column + 1}",
                    "target_pixel_bounds": [left, top, right, bottom],
                    "source_original_pixel_bounds": source_bounds,
                    "comparison_space": "original_source_pixels",
                    "required_panels": [
                        "source_original",
                        "extraction_reconstruction_resampled_to_source",
                        "pixel_diff",
                    ],
                    "categorical_resampling": "nearest",
                    "continuous_resampling": "area_or_lanczos",
                }
            )
    return cells


def _accepted_extraction(map_dir: Path) -> tuple[Path | None, dict[str, Any] | None]:
    pointers = sorted(map_dir.glob("*/accepted-extraction.json"))
    if not pointers:
        return None, None
    if len(pointers) > 1:
        raise ValueError(f"multiple accepted extraction pointers under {map_dir}")
    pointer = pointers[0]
    accepted = _read_json(pointer)
    iteration = pointer.parent / str(accepted["accepted_iteration"]) / "iteration.json"
    if not iteration.is_file():
        raise FileNotFoundError(iteration)
    return pointer, _read_json(iteration)


def _numeric_fidelity_leaves(value: Any, path: str = "") -> dict[str, float]:
    results: dict[str, float] = {}
    if isinstance(value, Mapping):
        for key, nested in value.items():
            child = f"{path}.{key}" if path else str(key)
            results.update(_numeric_fidelity_leaves(nested, child))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            results.update(_numeric_fidelity_leaves(nested, f"{path}[{index}]"))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if any(term in path.casefold() for term in _FIDELITY_TERMS):
            results[path] = float(value)
    return results


def _viewer_state(map_id: str, viewer_data_root: Path | None) -> dict[str, Any]:
    if viewer_data_root is None:
        return {"public": False, "staging": False}
    alias = _PUBLICATION_ALIASES.get(map_id)
    public_catalog = viewer_data_root / "catalog.json"
    public_ids: set[str] = set()
    if public_catalog.is_file():
        public_ids = {
            str(item.get("id"))
            for item in _read_json(public_catalog).get("datasets", [])
            if isinstance(item, Mapping)
        }
    staging_ids = {
        str(_read_json(path).get("id"))
        for path in (viewer_data_root / "staging").glob("*/dataset.json")
    }
    return {
        "public": bool(alias and alias in public_ids),
        "public_dataset_id": alias if alias in public_ids else None,
        "staging": bool(
            alias in staging_ids
            or any(map_id in item for item in staging_ids)
            or (map_id == "farms-v2" and any("agricultural" in item for item in staging_ids))
        ),
        "staging_dataset_ids": sorted(
            item
            for item in staging_ids
            if map_id in item
            or item == alias
            or (map_id == "farms-v2" and "agricultural" in item)
        ),
    }


def _priority(record: Mapping[str, Any]) -> tuple[int, str]:
    resolution = record["resolution"]
    if resolution["target_grid_resolution_loss"]:
        return 1, "accepted z9 grid collapses multiple original-source pixels"
    if record["final_status"] != "complete":
        return 2, "map remains blocked or incomplete"
    if record["extraction_fidelity_risk"]["measured_mismatch"]:
        return 3, "accepted extraction retains a nontrivial measured source mismatch"
    if resolution["registration_detail_risk"]:
        return 4, "registration was estimated on a source reduced below 50%"
    if not record["viewer_state"]["staging"]:
        return 5, "accepted output has no autonomous staging package"
    return 6, "rerun native regional source-diff gates before publication"


def audit_restart_run(
    run_root: Path, *, viewer_data_root: Path | None = None
) -> dict[str, Any]:
    """Audit every map directory in a no-human restart run."""

    run_root = run_root.resolve()
    records: list[dict[str, Any]] = []
    for map_dir in sorted(path for path in run_root.iterdir() if path.is_dir()):
        experiment_path = map_dir / "EXPERIMENT.json"
        alignment_path = map_dir / "automatic-alignment" / "accepted-alignment.json"
        if not experiment_path.is_file() or not alignment_path.is_file():
            continue
        experiment = _read_json(experiment_path)
        alignment = _read_json(alignment_path)
        transform = alignment["transform"]
        extraction_pointer, extraction_iteration = _accepted_extraction(map_dir)
        resolution = analyze_transform_resolution(transform)
        target_grid = transform["target_grid"]
        factor = int(resolution["target_grid_supersampling_factor"])
        fidelity_metrics = (
            _numeric_fidelity_leaves(extraction_iteration.get("scores", {}))
            if extraction_iteration
            else {}
        )
        mismatch_values = [
            value
            for key, value in fidelity_metrics.items()
            if key.casefold().endswith("mismatch_fraction")
        ]
        accepted_extraction_count = experiment.get("extraction", {}).get(
            "accepted_automatic_iteration_count"
        )
        record: dict[str, Any] = {
            "map_id": map_dir.name,
            "source": experiment["source"],
            "source_dimensions": {
                "width": int(transform["source_original_shape"][1]),
                "height": int(transform["source_original_shape"][0]),
            },
            "source_extent": (
                "partial"
                if map_dir.name == "farms-v2"
                or bool(
                    extraction_iteration
                    and "missing_source_extent_remains_nodata"
                    in extraction_iteration.get("gates", {})
                )
                else "full"
            ),
            "accepted_alignment_iteration": experiment.get("alignment", {}).get(
                "accepted_automatic_iteration_count"
            ),
            "accepted_extraction_iteration": accepted_extraction_count,
            "accepted_extraction_pointer": (
                str(extraction_pointer) if extraction_pointer else None
            ),
            "final_status": experiment.get("final", {}).get("status", "in_progress"),
            "blocker": experiment.get("final", {}).get("blocker"),
            "target_grid": target_grid,
            "base_reference_zoom": int(experiment["mapbox_reference"]["zoom"]),
            "recommended_target_grid": {
                **target_grid,
                # Preserve the exact outer pixel-center coordinates. Multiplying
                # dimensions directly would add one interval on each axis.
                "width": (int(target_grid["width"]) - 1) * factor + 1,
                "height": (int(target_grid["height"]) - 1) * factor + 1,
                "supersampling_factor": factor,
                "corner_preserving": True,
                "equivalent_minimum_native_zoom": int(
                    experiment["mapbox_reference"]["zoom"]
                )
                + int(resolution["equivalent_native_zoom_increment"]),
            },
            "resolution": resolution,
            "native_regional_diff_policy": {
                "grid": [6, 6],
                "require_every_supported_cell": True,
                "acceptance": (
                    "zero meaningful categorical mismatch; continuous or linear maps "
                    "must pass their source-family tolerance in every cell"
                ),
                "cells": plan_native_regional_diffs(transform),
            },
            "extraction_fidelity_metrics": fidelity_metrics,
            "registration_fidelity_risk": {
                "at_risk": resolution["registration_detail_risk"],
                "reason": (
                    "alignment estimation reduced the source below 50%; rerun at native "
                    "or multi-scale resolution"
                    if resolution["registration_detail_risk"]
                    else None
                ),
            },
            "extraction_fidelity_risk": {
                "at_risk": bool(
                    factor > 1
                    or extraction_iteration is None
                    or any(value > 0.005 for value in mismatch_values)
                ),
                "target_resolution_loss": factor > 1,
                "missing_accepted_extraction": extraction_iteration is None,
                "measured_mismatch": any(value > 0.005 for value in mismatch_values),
                "maximum_measured_mismatch_fraction": max(mismatch_values, default=None),
                "native_source_regional_diff_recorded": False,
                "reason": (
                    "current acceptance crops are target-grid overviews; new runs must "
                    "reconstruct into original-source pixels for every supported 6x6 cell"
                ),
            },
            "viewer_state": _viewer_state(map_dir.name, viewer_data_root),
        }
        priority, reason = _priority(record)
        record["execution_priority"] = priority
        record["execution_reason"] = reason
        records.append(record)
    queue = [
        {
            "priority": item["execution_priority"],
            "map_id": item["map_id"],
            "reason": item["execution_reason"],
            "supersampling_factor": item["resolution"][
                "target_grid_supersampling_factor"
            ],
        }
        for item in sorted(
            records,
            key=lambda item: (item["execution_priority"], item["map_id"]),
        )
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "run_root": str(run_root),
        "policy": {
            "resolution_measure": "p95 local maximum singular value of target-to-source Jacobian",
            "supersampling_margin": 1.10,
            "maximum_supersampling_factor": 4,
            "native_diff_grid": [6, 6],
            "overview_only_diffs_forbidden": True,
            "accepted_artifacts_mutated": False,
        },
        "maps": records,
        "execution_queue": queue,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--viewer-data-root", type=Path)
    arguments = parser.parse_args(argv)
    report = audit_restart_run(
        arguments.run_root, viewer_data_root=arguments.viewer_data_root
    )
    arguments.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
