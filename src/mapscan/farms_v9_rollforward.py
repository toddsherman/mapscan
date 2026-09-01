"""Versioned validation-only roll-forward for the partial farms alignment.

The v8 county validation evidence has been consumed.  This module therefore
never reuses it as a holdout: it becomes training, while the formerly sealed
v8 county-test mask is deterministically divided into fresh validation and a
new sealed final county confirmation mask before any source or candidate is
scored.  Golden Gate and East Bay water confirmation masks are outside this
module's API and cannot be evaluated here.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
from PIL import Image
from pyproj import CRS, Transformer
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

from .automatic_alignment_loop import _sha256
from .automatic_categorical_extraction import (
    _projection_reference_to_source_base,
    _projection_source_to_reference_base,
    _reference_to_source_remap,
    _residual_displacement,
    _source_to_reference_remap,
)
from .elevation_nonrigid_alignment import CompactResidualWarp, fit_compact_residual_warp
from .farms_nonrigid_alignment import (
    EXPECTED_FARMS_SOURCE_SHA256,
    EXPECTED_MAPBOX_V2_MANIFEST_SHA256,
    FarmsNonrigidConfig,
    _balanced_county_validation_report,
    _county_validation_geographic_cells,
    _east_boundary,
    _holdout_gate,
    _holdout_report,
    _load_source,
    _map_points,
    _projection_contexts,
    _regularity_report,
    _render_line,
    _robust_indices,
    _source_observable_extents,
    _supported_coast_gate,
    build_mapbox_farms_semantics,
    derive_county_residual_controls,
    derive_nevada_residual_controls,
    load_pinned_mapbox_reference,
    serialize_farms_nonrigid_transform,
)


EXPECTED_V8_COUNTY_TEST_FILE_SHA256 = (
    "2d7f586c8d2f1150c145e04c7b9cd4df72df64541743fa9aee0cb3e2f50f1210"
)
EXPECTED_V8_COUNTY_TEST_LOGICAL_SHA256 = (
    "7546e4d8da4e4cc095f76c675f5d1888275dd602aa42c525d18dff834c429b70"
)
EXPECTED_V8_COUNTY_VALIDATION_FILE_SHA256 = (
    "35b59a0090c627c52ba5f6ca78bd7ea8f025da5c47247e33be178d25844c7182"
)
EXPECTED_V8_COUNTY_VALIDATION_LOGICAL_SHA256 = (
    "ccc36c98700972d657a7e01ab5565c532543f6c6c3f7497e01f7641f1f45819d"
)
EXPECTED_V8_COUNTY_TRAINING_FILE_SHA256 = (
    "c25005d53fea36be29bbb71708592cfe63b7eb266dba341295efeffadf07f63b"
)
EXPECTED_V8_COUNTY_TRAINING_LOGICAL_SHA256 = (
    "080b7b30fcf6c822b04e9bd02805c28577ddf444e6f51a90a2c52cc3fe53df10"
)
EXPECTED_V8_VALIDATION_REPORT_SHA256 = (
    "977781ed6efd88cd0b5eb3ef57783e1821d19ca29907d87683da148ae0529e20"
)
EXPECTED_V8_FROZEN_CANDIDATE_SHA256 = (
    "59c58d46934b9afaeb09d099f65d244bb66615554b667f16d18cfe2708f51ab0"
)
EXPECTED_V9_PARTITION_RECEIPT_SHA256 = (
    "8fc6814049aada1166dcc2de43aea597a8c8e713ebc2ddacce96794aeb533147"
)
EXPECTED_V9_VALIDATION_REPORT_SHA256 = (
    "85a55784e57b514517860036a82db5d881c9ebfc78f0c16f421a0391fc3ba683"
)
EXPECTED_V9_FROZEN_CANDIDATE_SHA256 = (
    "d92a58a83fe5bb85bd5f73dafc2acefaca8f93a208257d688084737bdf6c1152"
)
EXPECTED_V9_FINAL_COUNTY_FILE_SHA256 = (
    "7d4a69b7c6049d3814b8c312844e168faa12087624629ab9c0456844bcaf60c8"
)
EXPECTED_V9_FINAL_COUNTY_LOGICAL_SHA256 = (
    "b04d345dfaaca0b979183d31c1d192594b9cf935413414ebea69b07373d7fccc"
)
EXPECTED_V8_GOLDEN_GATE_FILE_SHA256 = (
    "1bf763041b0667b0968254d775548a35cc1b86746ff7e5a5af554b590f7bb288"
)
EXPECTED_V8_GOLDEN_GATE_LOGICAL_SHA256 = (
    "9fbb8b6f657082331372bec5692b31657fae72770b552697ce3e15622f31b2f4"
)
EXPECTED_V8_EAST_BAY_FILE_SHA256 = (
    "7dc67794eadf6912338767b819c463bdb7a9a53c7a2d5c58a6db303c027f3f84"
)
EXPECTED_V8_EAST_BAY_LOGICAL_SHA256 = (
    "c66af4c210ec50290c11fa02e6b892568364c81c32bf6f43bb914d2a1869fbf3"
)
CANONICAL_V9_FINAL_OUTPUT_RELATIVE_PATH = (
    "runs/farms-v2-nonrigid-mapbox-v2-one-shot-final-acceptance-v9"
)
CANONICAL_V9_FINAL_OUTPUT_ROOT = (
    Path(__file__).resolve().parents[2] / CANONICAL_V9_FINAL_OUTPUT_RELATIVE_PATH
).resolve()


@dataclass(frozen=True)
class FarmsV9CountySplit:
    """Target-only v9 county roles derived from the untouched v8 test mask."""

    bay_guard: np.ndarray
    eligible: np.ndarray
    validation: np.ndarray
    final_acceptance: np.ndarray
    unused_buffer: np.ndarray
    diagnostics: Mapping[str, Any]


@dataclass(frozen=True)
class FarmsV9Config:
    """Immutable bounded v9 roll-forward parameters."""

    local_reference_bin_px: int = 120
    local_minimum_bin_support: int = 12
    local_maximum_match_working_px: float = 18.0
    local_heldout_exclusion_working_px: float = 24.0
    local_maximum_reference_points: int = 12_000
    residual_radii_reference_px: tuple[float, ...] = (360.0, 440.0, 520.0, 600.0)
    residual_ridges: tuple[float, ...] = (0.5, 1.0, 2.0)
    expected_candidate_count: int = 25


@dataclass(frozen=True)
class FarmsV9PreflightResult:
    status: str
    stop_reason: str
    report_path: Path
    frozen_candidate_path: Path | None
    artifact_paths: tuple[Path, ...]


@dataclass(frozen=True)
class FarmsV9FinalAcceptanceResult:
    status: str
    stop_reason: str
    report_path: Path
    artifact_paths: tuple[Path, ...]


def _mask_sha256(mask: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(mask.astype(np.uint8)).tobytes()
    ).hexdigest()


def _role_cells(
    shape: tuple[int, int],
    *,
    rows: int,
    columns: int,
    validation: bool,
) -> np.ndarray:
    height, width = shape
    result = np.zeros(shape, dtype=np.uint8)
    for row in range(rows):
        for column in range(columns):
            role = (2 * row + column + row // 2) % 2 == 0
            if role != validation:
                continue
            y1, y2 = round(row * height / rows), round((row + 1) * height / rows)
            x1, x2 = round(column * width / columns), round((column + 1) * width / columns)
            result[y1:y2, x1:x2] = 1
    return result.astype(bool)


def _target_geographic_coverage(
    mask: np.ndarray,
    *,
    bounds: tuple[int, int, int, int],
    rows: int = 6,
    columns: int = 6,
) -> dict[str, Any]:
    x0, y0, x1, y1 = bounds
    height, width = mask.shape
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError("Geographic coverage bounds are invalid")
    x_edges = np.linspace(x0, x1, columns + 1).round().astype(int)
    y_edges = np.linspace(y0, y1, rows + 1).round().astype(int)
    cells: list[dict[str, int]] = []
    for row in range(rows):
        for column in range(columns):
            count = int(
                np.count_nonzero(
                    mask[
                        y_edges[row] : y_edges[row + 1],
                        x_edges[column] : x_edges[column + 1],
                    ]
                )
            )
            if count < 20:
                continue
            cells.append({"row": row, "column": column, "pixel_count": count})
    return {
        "grid": [rows, columns],
        "common_bounds_xyxy": [x0, y0, x1, y1],
        "minimum_pixels_per_supported_cell": 20,
        "supported_cell_count": len(cells),
        "supported_rows": sorted({item["row"] for item in cells}),
        "supported_columns": sorted({item["column"] for item in cells}),
        "cells": cells,
    }


def build_farms_v9_county_split(
    old_county_test: np.ndarray,
    *,
    role_cells: tuple[int, int] = (14, 14),
    role_erosion_px: int = 12,
) -> FarmsV9CountySplit:
    """Split the old test mask without accepting source or candidate inputs."""

    if old_county_test.ndim != 2:
        raise ValueError("Old county-test mask must be two-dimensional")
    if old_county_test.shape != (3920, 3398):
        raise ValueError("Farms v9 split requires the exact pinned Mapbox-v2 grid")
    if role_cells != (14, 14) or role_erosion_px != 12:
        raise ValueError("Farms v9 split parameters are immutable")

    # This is the same fixed target-only Bay guard declared before v8 fitting.
    # It fully contains the fixed Golden Gate and East Bay coordinate boxes, so
    # those sealed water masks need not be constructed or read.
    bay_bounds = (380, 1260, 1245, 2170)
    bay_guard = np.zeros(old_county_test.shape, dtype=np.uint8)
    bay_guard[
        bay_bounds[1] : bay_bounds[3] + 1,
        bay_bounds[0] : bay_bounds[2] + 1,
    ] = 1
    bay_guard = cv2.dilate(bay_guard, np.ones((51, 51), np.uint8)).astype(bool)
    eligible = old_county_test.astype(bool) & ~bay_guard

    rows, columns = role_cells
    validation_cells = _role_cells(
        old_county_test.shape,
        rows=rows,
        columns=columns,
        validation=True,
    )
    final_cells = _role_cells(
        old_county_test.shape,
        rows=rows,
        columns=columns,
        validation=False,
    )
    kernel = np.ones((2 * role_erosion_px + 1,) * 2, np.uint8)
    validation = eligible & cv2.erode(
        validation_cells.astype(np.uint8), kernel
    ).astype(bool)
    final_acceptance = eligible & cv2.erode(
        final_cells.astype(np.uint8), kernel
    ).astype(bool)
    unused_buffer = eligible & ~(validation | final_acceptance)

    if np.any(validation & final_acceptance):
        raise ValueError("Fresh v9 validation overlaps sealed final county evidence")
    if np.any((validation | final_acceptance) & bay_guard):
        raise ValueError("Fresh v9 county roles overlap the fixed Bay guard")
    minimum_separation = float(
        min(
            np.min(distance_transform_edt(~final_acceptance)[validation]),
            np.min(distance_transform_edt(~validation)[final_acceptance]),
        )
    )
    if minimum_separation < 25.0:
        raise ValueError("Fresh v9 county roles lack their fixed target buffer")

    eligible_y, eligible_x = np.nonzero(eligible)
    if not len(eligible_x):
        raise ValueError("Fresh v9 county split has no eligible evidence")
    common_eligible_bounds = (
        int(eligible_x.min()),
        int(eligible_y.min()),
        int(eligible_x.max()) + 1,
        int(eligible_y.max()) + 1,
    )
    full_grid_bounds = (0, 0, old_county_test.shape[1], old_county_test.shape[0])
    validation_coverage = _target_geographic_coverage(
        validation, bounds=common_eligible_bounds
    )
    final_coverage = _target_geographic_coverage(
        final_acceptance, bounds=common_eligible_bounds
    )
    validation_full_grid_coverage = _target_geographic_coverage(
        validation, bounds=full_grid_bounds
    )
    final_full_grid_coverage = _target_geographic_coverage(
        final_acceptance, bounds=full_grid_bounds
    )
    for role, coverage in (
        ("validation", validation_coverage),
        ("final_acceptance", final_coverage),
    ):
        if (
            coverage["supported_cell_count"] < 6
            or len(coverage["supported_rows"]) < 3
            or len(coverage["supported_columns"]) < 3
        ):
            raise ValueError(f"Fresh v9 {role} lacks target-only geographic coverage")

    diagnostics = {
        "method": "fixed_14x14_target_cells_two_roles_eroded_12px",
        "authority": {
            "source_pixels_used": False,
            "candidate_transform_used": False,
            "validation_or_acceptance_scores_used": False,
            "old_v8_county_test_is_only_input_evidence": True,
            "golden_gate_or_east_bay_masks_read": False,
        },
        "role_formula": "(2*row + column + row//2) % 2",
        "role_cells": list(role_cells),
        "role_erosion_px": role_erosion_px,
        "minimum_validation_to_final_separation_px": minimum_separation,
        "common_eligible_bounds_xyxy": list(common_eligible_bounds),
        "bay_bounds_before_25px_dilation": list(bay_bounds),
        "golden_gate_bounds_contained_by_bay_guard": [601, 1727, 750, 1853],
        "east_bay_bounds_contained_by_bay_guard": [716, 1719, 881, 1927],
        "pixel_counts": {
            "old_county_test": int(np.count_nonzero(old_county_test)),
            "bay_guard_intersection": int(
                np.count_nonzero(old_county_test & bay_guard)
            ),
            "eligible": int(np.count_nonzero(eligible)),
            "validation": int(np.count_nonzero(validation)),
            "final_acceptance": int(np.count_nonzero(final_acceptance)),
            "unused_buffer": int(np.count_nonzero(unused_buffer)),
        },
        "mask_sha256": {
            "old_county_test": _mask_sha256(old_county_test),
            "bay_guard": _mask_sha256(bay_guard),
            "eligible": _mask_sha256(eligible),
            "validation": _mask_sha256(validation),
            "final_acceptance": _mask_sha256(final_acceptance),
            "unused_buffer": _mask_sha256(unused_buffer),
        },
        "validation_geographic_coverage_common_eligible_bounds": validation_coverage,
        "final_acceptance_geographic_coverage_common_eligible_bounds": final_coverage,
        "validation_geographic_coverage_full_grid": validation_full_grid_coverage,
        "final_acceptance_geographic_coverage_full_grid": final_full_grid_coverage,
    }
    return FarmsV9CountySplit(
        bay_guard=bay_guard,
        eligible=eligible,
        validation=validation,
        final_acceptance=final_acceptance,
        unused_buffer=unused_buffer,
        diagnostics=diagnostics,
    )


def _write_mask(path: Path, mask: np.ndarray) -> dict[str, Any]:
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "pixel_count": int(np.count_nonzero(mask)),
    }


def _load_exact_v8_role(
    path: Path,
    *,
    role: str,
    expected_file_sha256: str,
    expected_logical_sha256: str,
) -> np.ndarray:
    """Load one exact v8 role, rejecting same-shape substitutions."""

    path = path.resolve()
    file_sha256 = _sha256(path)
    if file_sha256 != expected_file_sha256:
        raise ValueError(
            f"Authoritative v8 {role} file hash mismatch: {file_sha256}"
        )
    mask = np.asarray(Image.open(path).convert("L")) > 0
    logical_sha256 = _mask_sha256(mask)
    if logical_sha256 != expected_logical_sha256:
        raise ValueError(
            f"Authoritative v8 {role} logical hash mismatch: {logical_sha256}"
        )
    return mask


def _require_exact_artifact_sha256(
    path: Path, *, label: str, expected_sha256: str
) -> None:
    actual_sha256 = _sha256(path.resolve())
    if actual_sha256 != expected_sha256:
        raise ValueError(f"Authoritative {label} hash mismatch: {actual_sha256}")


def write_farms_v9_partition_receipt(
    old_county_test_path: Path,
    consumed_v8_validation_path: Path,
    output_root: Path,
) -> Path:
    """Persist the target-only split before any validation scoring occurs."""

    old_county_test_path = old_county_test_path.resolve()
    consumed_v8_validation_path = consumed_v8_validation_path.resolve()
    output_root = output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("V9 partition output must be a fresh immutable directory")
    old = _load_exact_v8_role(
        old_county_test_path,
        role="county_test",
        expected_file_sha256=EXPECTED_V8_COUNTY_TEST_FILE_SHA256,
        expected_logical_sha256=EXPECTED_V8_COUNTY_TEST_LOGICAL_SHA256,
    )
    consumed_validation = _load_exact_v8_role(
        consumed_v8_validation_path,
        role="county_validation",
        expected_file_sha256=EXPECTED_V8_COUNTY_VALIDATION_FILE_SHA256,
        expected_logical_sha256=EXPECTED_V8_COUNTY_VALIDATION_LOGICAL_SHA256,
    )
    if consumed_validation.shape != old.shape:
        raise ValueError("Authoritative v8 county roles have different grids")
    output_root.mkdir(parents=True, exist_ok=True)
    split = build_farms_v9_county_split(old)
    artifacts = {
        "bay_guard": _write_mask(output_root / "bay-guard.png", split.bay_guard),
        "eligible": _write_mask(output_root / "eligible.png", split.eligible),
        "validation": _write_mask(output_root / "validation.png", split.validation),
        "final_acceptance": _write_mask(
            output_root / "sealed-final-acceptance.png", split.final_acceptance
        ),
        "unused_buffer": _write_mask(
            output_root / "unused-buffer.png", split.unused_buffer
        ),
        "consumed_v8_validation_now_training": _write_mask(
            output_root / "consumed-v8-validation-now-training.png",
            consumed_validation,
        ),
    }
    receipt = {
        "schema_version": 2,
        "kind": "farms_v9_target_only_county_partition_receipt_v2",
        "status": "partition_frozen_before_scoring",
        "v8_role_ancestry": {
            "old_county_test": {
                "path": str(old_county_test_path),
                "file_sha256": EXPECTED_V8_COUNTY_TEST_FILE_SHA256,
                "logical_sha256": EXPECTED_V8_COUNTY_TEST_LOGICAL_SHA256,
                "v9_role": "source_for_target_only_validation_final_split",
            },
            "old_county_validation": {
                "path": str(consumed_v8_validation_path),
                "file_sha256": EXPECTED_V8_COUNTY_VALIDATION_FILE_SHA256,
                "logical_sha256": EXPECTED_V8_COUNTY_VALIDATION_LOGICAL_SHA256,
                "v9_role": "consumed_training",
            },
        },
        "diagnostics": dict(split.diagnostics),
        "artifacts": artifacts,
        "authority": {
            "source_pixels_used": False,
            "candidate_transform_used": False,
            "validation_scores_used": False,
            "final_acceptance_scores_used": False,
            "golden_gate_or_east_bay_masks_read": False,
        },
    }
    path = output_root / "partition-receipt.json"
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return path


def _load_verified_partition(
    receipt_path: Path,
    old_county_test: np.ndarray,
    consumed_v8_validation: np.ndarray,
) -> tuple[FarmsV9CountySplit, Mapping[str, Any]]:
    if _mask_sha256(old_county_test) != EXPECTED_V8_COUNTY_TEST_LOGICAL_SHA256:
        raise ValueError("V9 partition received substituted v8 county test pixels")
    if (
        _mask_sha256(consumed_v8_validation)
        != EXPECTED_V8_COUNTY_VALIDATION_LOGICAL_SHA256
    ):
        raise ValueError("V9 partition received substituted v8 validation pixels")
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("kind") != "farms_v9_target_only_county_partition_receipt_v2":
        raise ValueError("V9 partition receipt kind is invalid")
    ancestry = receipt.get("v8_role_ancestry", {})
    expected_ancestry = {
        "old_county_test": (
            EXPECTED_V8_COUNTY_TEST_FILE_SHA256,
            EXPECTED_V8_COUNTY_TEST_LOGICAL_SHA256,
        ),
        "old_county_validation": (
            EXPECTED_V8_COUNTY_VALIDATION_FILE_SHA256,
            EXPECTED_V8_COUNTY_VALIDATION_LOGICAL_SHA256,
        ),
    }
    for role, (file_sha256, logical_sha256) in expected_ancestry.items():
        item = ancestry.get(role, {})
        if item.get("file_sha256") != file_sha256:
            raise ValueError(f"V9 partition ancestry file hash mismatch: {role}")
        if item.get("logical_sha256") != logical_sha256:
            raise ValueError(f"V9 partition ancestry logical hash mismatch: {role}")
    rebuilt = build_farms_v9_county_split(old_county_test)
    masks: dict[str, np.ndarray] = {}
    for name, artifact in receipt["artifacts"].items():
        path = Path(artifact["path"])
        if _sha256(path) != artifact["sha256"]:
            raise ValueError(f"V9 partition artifact hash mismatch: {name}")
        masks[name] = np.asarray(Image.open(path).convert("L")) > 0
    expected = {
        "bay_guard": rebuilt.bay_guard,
        "eligible": rebuilt.eligible,
        "validation": rebuilt.validation,
        "final_acceptance": rebuilt.final_acceptance,
        "unused_buffer": rebuilt.unused_buffer,
        "consumed_v8_validation_now_training": consumed_v8_validation,
    }
    for name, mask in expected.items():
        if not np.array_equal(masks[name], mask):
            raise ValueError(f"V9 partition artifact differs from deterministic split: {name}")
    if receipt["diagnostics"] != rebuilt.diagnostics:
        raise ValueError("V9 partition receipt diagnostics are stale")
    return rebuilt, receipt


def _county_heldout_exclusion_authority(
    *,
    heldout: np.ndarray,
    source_shape: tuple[int, int],
    projection: Any,
    grid: Mapping[str, Any],
    seed_matrix: np.ndarray,
    exclusion_radius_working_px: int,
) -> tuple[np.ndarray, np.ndarray, Mapping[str, Any]]:
    """Render the new heldout roles before controls and freeze their exclusion."""

    if exclusion_radius_working_px != 24:
        raise ValueError("V9 county heldout exclusion radius is immutable at 24px")
    rendered = _render_line(
        heldout,
        source_shape,
        projection,
        grid,
        seed_matrix,
        None,
    )
    if not np.any(rendered):
        raise ValueError("Fresh v9 heldout county geometry does not render in source")
    exclusion = cv2.dilate(
        rendered.astype(np.uint8),
        np.ones((2 * exclusion_radius_working_px + 1,) * 2, np.uint8),
    ).astype(bool)
    return rendered, exclusion, {
        "method": "render_fresh_v9_validation_or_final_before_control_construction",
        "heldout_reference_logical_sha256": _mask_sha256(heldout),
        "rendered_heldout_logical_sha256": _mask_sha256(rendered),
        "heldout_exclusion_logical_sha256": _mask_sha256(exclusion),
        "heldout_reference_pixel_count": int(np.count_nonzero(heldout)),
        "rendered_heldout_pixel_count": int(np.count_nonzero(rendered)),
        "heldout_exclusion_pixel_count": int(np.count_nonzero(exclusion)),
        "heldout_exclusion_radius_working_px": exclusion_radius_working_px,
        "heldout_scores_evaluated": False,
        "constructed_before_residual_controls": True,
    }


def _verify_county_assignment_exclusion(
    *,
    source_assignment: np.ndarray,
    rendered_heldout: np.ndarray,
    expected_exclusion: np.ndarray,
    actual_exclusion: np.ndarray,
    required_separation_working_px: float = 24.0,
) -> Mapping[str, Any]:
    """Prove that training assignments remain buffered from fresh heldout ink."""

    shapes = {
        source_assignment.shape,
        rendered_heldout.shape,
        expected_exclusion.shape,
        actual_exclusion.shape,
    }
    if len(shapes) != 1:
        raise ValueError("County assignment/exclusion masks have different shapes")
    if not np.array_equal(expected_exclusion, actual_exclusion):
        raise ValueError("Residual controls did not use the frozen v9 exclusion mask")
    if not np.any(source_assignment):
        raise ValueError("County training assignment is empty")
    overlap = int(np.count_nonzero(source_assignment & actual_exclusion))
    heldout_distance = distance_transform_edt(~rendered_heldout)
    minimum_separation = float(np.min(heldout_distance[source_assignment]))
    if overlap:
        raise ValueError("County training assignment intersects fresh heldout exclusion")
    if minimum_separation < required_separation_working_px:
        raise ValueError("County training assignment lacks required heldout separation")
    return {
        "source_assignment_pixel_count": int(np.count_nonzero(source_assignment)),
        "assignment_exclusion_overlap_pixel_count": overlap,
        "minimum_assignment_to_rendered_heldout_working_px": minimum_separation,
        "required_minimum_separation_working_px": required_separation_working_px,
        "passed": True,
    }


def _native_visible_r1c0_controls(
    *,
    source: Any,
    projection: Any,
    grid: Mapping[str, Any],
    seed_matrix: np.ndarray,
    consumed_r1c0: np.ndarray,
    heldout: np.ndarray,
    config: FarmsV9Config,
) -> tuple[np.ndarray, np.ndarray, Mapping[str, Any], np.ndarray]:
    """Derive explicit consumed-r1c0 controls from visible native ink only."""

    scale = float(source.working_scale)
    original_shape = source.rgb_original.shape[:2]
    scope_original = cv2.resize(
        source.topology.county_scope.astype(np.uint8),
        (original_shape[1], original_shape[0]),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    ink_distance = cv2.distanceTransform(
        source.native_county_topology.native_ink.astype(np.uint8),
        cv2.DIST_L2,
        3,
    )
    positive = ink_distance[source.native_county_topology.native_ink]
    native_radius = int(math.ceil(float(np.quantile(positive, 0.90))))
    hsv = cv2.cvtColor(source.rgb_original, cv2.COLOR_RGB2HSV)
    thematic = (
        scope_original
        & (hsv[:, :, 1] >= 40)
        & (hsv[:, :, 2] >= 40)
    )
    occlusion = cv2.dilate(
        thematic.astype(np.uint8),
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * native_radius + 1,) * 2
        ),
    ).astype(bool) & scope_original

    reference_y, reference_x = np.nonzero(consumed_r1c0)
    reference_points = np.column_stack((reference_x, reference_y)).astype(np.float64)
    if len(reference_points) > config.local_maximum_reference_points:
        indices = np.linspace(
            0, len(reference_points) - 1, config.local_maximum_reference_points
        ).round().astype(int)
        reference_points = reference_points[indices]
    heldout_distance_target = distance_transform_edt(~heldout)
    reference_xy = np.rint(reference_points).astype(np.int32)
    target_safe = heldout_distance_target[
        reference_xy[:, 1], reference_xy[:, 0]
    ] > 7.0
    reference_points = reference_points[target_safe]
    mapped_working = _map_points(
        reference_points, projection, grid, seed_matrix, None
    )
    mapped_original = mapped_working / scale
    rounded = np.rint(mapped_original).astype(np.int32)
    inside = (
        (rounded[:, 0] >= 0)
        & (rounded[:, 0] < original_shape[1])
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < original_shape[0])
    )
    direct = inside.copy()
    direct[inside] &= scope_original[rounded[inside, 1], rounded[inside, 0]]
    direct[inside] &= ~occlusion[rounded[inside, 1], rounded[inside, 0]]

    heldout_y, heldout_x = np.nonzero(heldout)
    heldout_points = np.column_stack((heldout_x, heldout_y)).astype(np.float64)
    if len(heldout_points) > 30_000:
        indices = np.linspace(0, len(heldout_points) - 1, 30_000).round().astype(int)
        heldout_points = heldout_points[indices]
    mapped_heldout_original = _map_points(
        heldout_points, projection, grid, seed_matrix, None
    ) / scale
    heldout_tree = cKDTree(mapped_heldout_original)

    native_y, native_x = np.nonzero(source.native_county_topology.native_ink)
    native_points = np.column_stack((native_x, native_y)).astype(np.float64)
    reference_tree = cKDTree(mapped_original[direct])
    training_distance, _ = reference_tree.query(native_points, workers=1)
    heldout_distance, _ = heldout_tree.query(native_points, workers=1)
    maximum_match_original = config.local_maximum_match_working_px / scale
    heldout_exclusion_original = config.local_heldout_exclusion_working_px / scale
    assignment = (
        (training_distance <= maximum_match_original)
        & (heldout_distance > heldout_exclusion_original)
        & ~occlusion[native_y, native_x]
    )
    assignment_points = native_points[assignment]
    if len(assignment_points) < 100:
        raise ValueError("Native-visible r1c0 source assignment is undersupported")
    assignment_tree = cKDTree(assignment_points)
    distances, nearest = assignment_tree.query(mapped_original[direct], workers=1)
    eligible = distances <= maximum_match_original
    eligible_references = reference_points[direct][eligible]
    eligible_mapped = mapped_original[direct][eligible]
    residual_working = (
        assignment_points[nearest[eligible]] - eligible_mapped
    ) * scale

    bin_size = int(config.local_reference_bin_px)
    bin_ids = np.floor(eligible_references / bin_size).astype(np.int32)
    controls: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    control_rows: list[dict[str, Any]] = []
    for key in sorted({tuple(item) for item in bin_ids.tolist()}):
        indices = np.flatnonzero(np.all(bin_ids == key, axis=1))
        if len(indices) < config.local_minimum_bin_support:
            continue
        robust = _robust_indices(indices, residual_working)
        if len(robust) < config.local_minimum_bin_support:
            continue
        center = np.median(eligible_references[robust], axis=0)
        displacement = np.median(residual_working[robust], axis=0)
        controls.append(center)
        targets.append(displacement)
        control_rows.append(
            {
                "bin": list(key),
                "support": int(len(robust)),
                "center_reference_px": center.tolist(),
                "displacement_working_px": displacement.tolist(),
                "displacement_norm_working_px": float(np.linalg.norm(displacement)),
            }
        )
    if len(controls) < 3:
        raise ValueError("Native-visible r1c0 controls are undersupported")

    assignment_mask = np.zeros(original_shape, dtype=bool)
    assigned_pixels = np.rint(assignment_points).astype(np.int32)
    assignment_mask[assigned_pixels[:, 1], assigned_pixels[:, 0]] = True
    diagnostics = {
        "method": "consumed_v8_r1c0_native_visible_train_only_nearest_ink_controls",
        "native_occlusion_radius_original_px": native_radius,
        "native_occlusion_radius_selection": "ceil_p90_native_ink_positive_distance",
        "reference_point_count": int(len(reference_points)),
        "direct_reference_point_count": int(np.count_nonzero(direct)),
        "source_assignment_pixel_count": int(len(assignment_points)),
        "eligible_match_count": int(np.count_nonzero(eligible)),
        "control_count": len(controls),
        "controls": control_rows,
        "maximum_match_working_px": config.local_maximum_match_working_px,
        "heldout_exclusion_working_px": config.local_heldout_exclusion_working_px,
        "minimum_assignment_to_heldout_original_px": float(
            np.min(heldout_distance[assignment])
        ),
        "minimum_assignment_to_heldout_working_px": float(
            np.min(heldout_distance[assignment]) * scale
        ),
        "new_validation_or_final_scores_used": False,
        "new_validation_and_final_masks_used_only_for_exclusion": True,
        "consumed_v8_validation_used_as_training": True,
        "retained_water_masks_read": False,
    }
    if (
        diagnostics["minimum_assignment_to_heldout_working_px"]
        < config.local_heldout_exclusion_working_px
    ):
        raise ValueError("Native-visible r1c0 assignment lacks heldout separation")
    return (
        np.asarray(controls, dtype=np.float64),
        np.asarray(targets, dtype=np.float64),
        diagnostics,
        assignment_mask,
    )


def _load_hashed_mask(path: Path, sha256: str) -> np.ndarray:
    if _sha256(path) != sha256:
        raise ValueError(f"V8 artifact hash mismatch: {path}")
    return np.asarray(Image.open(path).convert("L")) > 0


def _warp_from_frozen(payload: Mapping[str, Any]) -> CompactResidualWarp:
    item = payload["warp_working"]
    return CompactResidualWarp(
        np.asarray(item["centers_reference_px"], dtype=np.float64),
        np.asarray(item["coefficients_source_px"], dtype=np.float64),
        float(item["radius_reference_px"]),
        float(item["ridge"]),
    )


def run_farms_v9_validation_preflight(
    *,
    source_path: Path,
    reference_manifest_path: Path,
    v8_preflight_root: Path,
    partition_receipt_path: Path,
    output_root: Path,
    config: FarmsV9Config = FarmsV9Config(),
    base_config: FarmsNonrigidConfig = FarmsNonrigidConfig(),
) -> FarmsV9PreflightResult:
    """Fit a bounded v9 shortlist and score only fresh validation evidence."""

    source_path = source_path.resolve()
    reference_manifest_path = reference_manifest_path.resolve()
    v8_preflight_root = v8_preflight_root.resolve()
    partition_receipt_path = partition_receipt_path.resolve()
    output_root = output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("V9 validation output must be a fresh immutable directory")
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "validation-report.json"
    if _sha256(source_path) != EXPECTED_FARMS_SOURCE_SHA256:
        raise ValueError("V9 requires pristine farmsv2.png bytes")
    if _sha256(reference_manifest_path) != EXPECTED_MAPBOX_V2_MANIFEST_SHA256:
        raise ValueError("V9 requires the exact pinned Mapbox-v2 manifest")

    v8_report_path = v8_preflight_root / "validation-report.json"
    v8_frozen_path = v8_preflight_root / "frozen-validation-candidate.json"
    _require_exact_artifact_sha256(
        v8_report_path,
        label="v8 validation report",
        expected_sha256=EXPECTED_V8_VALIDATION_REPORT_SHA256,
    )
    _require_exact_artifact_sha256(
        v8_frozen_path,
        label="v8 frozen candidate",
        expected_sha256=EXPECTED_V8_FROZEN_CANDIDATE_SHA256,
    )
    v8_report = json.loads(v8_report_path.read_text())
    v8_frozen = json.loads(v8_frozen_path.read_text())
    if v8_report.get("status") != "validation_pass":
        raise ValueError("V9 requires the retained v8 validation history")

    county_artifacts = v8_report["county_mask_artifacts"]
    if (
        county_artifacts["county_training"]["sha256"]
        != EXPECTED_V8_COUNTY_TRAINING_FILE_SHA256
    ):
        raise ValueError("V8 report does not pin the authoritative county training")
    old_training = _load_exact_v8_role(
        Path(county_artifacts["county_training"]["path"]),
        role="county_training",
        expected_file_sha256=EXPECTED_V8_COUNTY_TRAINING_FILE_SHA256,
        expected_logical_sha256=EXPECTED_V8_COUNTY_TRAINING_LOGICAL_SHA256,
    )
    if (
        county_artifacts["county_validation"]["sha256"]
        != EXPECTED_V8_COUNTY_VALIDATION_FILE_SHA256
    ):
        raise ValueError("V8 report does not pin the authoritative county validation")
    consumed_v8_validation = _load_exact_v8_role(
        Path(county_artifacts["county_validation"]["path"]),
        role="county_validation",
        expected_file_sha256=EXPECTED_V8_COUNTY_VALIDATION_FILE_SHA256,
        expected_logical_sha256=EXPECTED_V8_COUNTY_VALIDATION_LOGICAL_SHA256,
    )
    old_test_path = Path(county_artifacts["county_test"]["path"])
    if county_artifacts["county_test"]["sha256"] != EXPECTED_V8_COUNTY_TEST_FILE_SHA256:
        raise ValueError("V8 report does not pin the authoritative county test")
    old_test = _load_exact_v8_role(
        old_test_path,
        role="county_test",
        expected_file_sha256=EXPECTED_V8_COUNTY_TEST_FILE_SHA256,
        expected_logical_sha256=EXPECTED_V8_COUNTY_TEST_LOGICAL_SHA256,
    )
    split, partition_receipt = _load_verified_partition(
        partition_receipt_path, old_test, consumed_v8_validation
    )
    expanded_training = old_training | consumed_v8_validation
    heldout = split.validation | split.final_acceptance

    source = _load_source(source_path, base_config)
    source_extents, source_extent_diagnostics = _source_observable_extents(
        source, base_config
    )
    reference = load_pinned_mapbox_reference(reference_manifest_path)
    semantics = build_mapbox_farms_semantics(reference)
    projection_id = str(v8_frozen["projection"])
    projection = next(
        item for item in _projection_contexts(reference) if item.id == projection_id
    )
    seed_matrix = np.asarray(v8_frozen["seed_matrix_working"], dtype=np.float64)

    heldout_exclusion_radius = int(
        math.ceil(base_config.residual_county_maximum_match_px)
    ) + 6
    rendered_new_heldout, frozen_new_heldout_exclusion, exclusion_authority = (
        _county_heldout_exclusion_authority(
            heldout=heldout,
            source_shape=source.rgb_working.shape[:2],
            projection=projection,
            grid=reference.grid,
            seed_matrix=seed_matrix,
            exclusion_radius_working_px=heldout_exclusion_radius,
        )
    )

    county_controls, county_targets, county_diagnostics, assignment_masks = (
        derive_county_residual_controls(
            reference,
            source,
            projection,
            seed_matrix,
            expanded_training,
            split.validation,
            split.final_acceptance,
            base_config,
        )
    )
    exclusion_proof = _verify_county_assignment_exclusion(
        source_assignment=assignment_masks["source_county_training_assignment"],
        rendered_heldout=rendered_new_heldout,
        expected_exclusion=frozen_new_heldout_exclusion,
        actual_exclusion=assignment_masks["source_county_heldout_exclusion"],
        required_separation_working_px=24.0,
    )
    consumed_cells = _county_validation_geographic_cells(
        consumed_v8_validation, base_config
    )
    consumed_r1c0 = consumed_cells["county_validation_r1_c0"]
    local_controls, local_targets, local_diagnostics, local_assignment = (
        _native_visible_r1c0_controls(
            source=source,
            projection=projection,
            grid=reference.grid,
            seed_matrix=seed_matrix,
            consumed_r1c0=consumed_r1c0,
            heldout=heldout,
            config=config,
        )
    )

    _nevada_controls, _nevada_targets, split_points, nevada_diagnostics = (
        derive_nevada_residual_controls(
            reference,
            semantics,
            source,
            projection,
            seed_matrix,
            base_config,
        )
    )
    east = _east_boundary(source.topology)
    validation_points_y, validation_points_x = np.nonzero(split.validation)
    validation_points = np.column_stack(
        (validation_points_x, validation_points_y)
    ).astype(np.float64)
    if len(validation_points) > 20_000:
        indices = np.linspace(
            0, len(validation_points) - 1, 20_000
        ).round().astype(int)
        validation_points = validation_points[indices]
    validation_cells = _county_validation_geographic_cells(
        split.validation, base_config
    )

    # Only the already-consumed v8 water validation masks are loaded for
    # non-degradation.  Golden Gate/East Bay final water files are neither
    # named here nor reachable through this code path.
    named_validation: dict[str, np.ndarray] = {}
    for name in (
        "outer_pacific_validation_microsegments",
        "north_bay_san_pablo_validation_microsegments",
        "south_bay_validation_microsegments",
    ):
        artifact = v8_report["named_pacific_mask_artifacts"][name]
        named_validation[name] = _load_hashed_mask(
            Path(artifact["path"]), artifact["sha256"]
        )
    named_points = {}
    for name, mask in named_validation.items():
        y, x = np.nonzero(mask)
        named_points[name] = np.column_stack((x, y)).astype(np.float64)

    families = {
        "expanded-county": (county_controls, county_targets),
        "expanded-county-plus-native-visible-r1c0": (
            np.concatenate((county_controls, local_controls), axis=0),
            np.concatenate((county_targets, local_targets), axis=0),
        ),
    }
    candidates: list[dict[str, Any]] = []

    def evaluate(candidate_id: str, family: str, warp: CompactResidualWarp) -> None:
        validation_admin = _holdout_report(
            split_points["validation"],
            east,
            projection,
            reference.grid,
            seed_matrix,
            warp,
            observable_extent=source_extents["state_admin"],
        )
        validation_counties = _holdout_report(
            validation_points,
            source.county_topology,
            projection,
            reference.grid,
            seed_matrix,
            warp,
            observable_extent=source_extents["county_lines"],
        )
        balanced = _balanced_county_validation_report(
            validation_cells,
            source.county_topology,
            projection,
            reference.grid,
            seed_matrix,
            warp,
            observable_extent=source_extents["county_lines"],
            config=base_config,
        )
        named_reports = {}
        for name, points in named_points.items():
            source_line = (
                source.native_pacific_coast_edge
                if name.startswith("outer_pacific")
                else source.native_water_edge
            )
            named_reports[name] = _holdout_report(
                points,
                source_line,
                projection,
                reference.grid,
                seed_matrix,
                warp,
                observable_extent=(
                    source_extents["pacific_coast"]
                    if name.startswith("outer_pacific")
                    else source_extents["named_bay"]
                ),
            )
        outer = named_reports["outer_pacific_validation_microsegments"]
        bay_reports = {
            name: report
            for name, report in named_reports.items()
            if not name.startswith("outer_pacific")
        }
        regularity = _regularity_report(
            reference, projection, seed_matrix, warp, base_config
        )
        gates = {
            "regularity": bool(regularity["passed"]),
            "consumed_admin_non_degradation": _holdout_gate(
                validation_admin, base_config
            ),
            "fresh_county_visible_fraction": bool(
                validation_counties["visible_fraction"]
                >= base_config.holdout_minimum_visible_fraction
            ),
            "fresh_county_median": bool(
                validation_counties["median_px"]
                <= base_config.county_median_limit_px
            ),
            "fresh_county_support": bool(
                validation_counties["within_8px_fraction"]
                >= base_config.county_within_8_minimum
            ),
            "fresh_county_geographic_balance": bool(balanced["passed"]),
            "consumed_outer_pacific_non_degradation": _supported_coast_gate(
                outer, base_config
            ),
            "consumed_named_bay_non_degradation": bool(
                all(
                    _supported_coast_gate(report, base_config, bay=True)
                    for report in bay_reports.values()
                )
            ),
        }
        score = float(
            validation_counties["median_px"]
            + 0.20 * validation_counties["p90_px"]
            + 10.0 * (1.0 - validation_counties["within_8px_fraction"])
            + validation_admin["median_px"]
            + 0.20 * validation_admin["p90_px"]
            + outer["median_px"]
            + 0.20 * outer["p90_px"]
            + sum(
                report["median_px"] + 0.20 * report["p90_px"]
                for report in bay_reports.values()
            )
            + 0.02 * regularity["maximum_residual_displacement_working_px"]
        )
        candidates.append(
            {
                "id": candidate_id,
                "family": family,
                "radius_reference_px": float(warp.radius_reference_px),
                "ridge": float(warp.ridge),
                "eligible": bool(all(gates.values())),
                "selection_score": score,
                "gates": gates,
                "fresh_validation_counties": validation_counties,
                "fresh_balanced_validation_counties": balanced,
                "consumed_admin_non_degradation": validation_admin,
                "consumed_outer_pacific_non_degradation": outer,
                "consumed_named_bay_non_degradation": bay_reports,
                "regularity": regularity,
                "warp": warp,
            }
        )

    evaluate("v8-frozen-warp-baseline", "v8-frozen-baseline", _warp_from_frozen(v8_frozen))
    for family, (controls, targets) in families.items():
        for radius in config.residual_radii_reference_px:
            for ridge in config.residual_ridges:
                warp = fit_compact_residual_warp(
                    controls,
                    targets,
                    radius_reference_px=radius,
                    ridge=ridge,
                )
                evaluate(
                    f"{family}-wendland-r{radius:g}-ridge{ridge:g}",
                    family,
                    warp,
                )
    if len(candidates) != config.expected_candidate_count:
        raise ValueError("V9 bounded shortlist candidate count changed")
    eligible = [item for item in candidates if item["eligible"]]
    selected = min(eligible, key=lambda item: (item["selection_score"], item["id"])) if eligible else None

    source_assignment_artifacts = {}
    artifact_paths: list[Path] = []
    rendered_new_heldout_artifact = _write_mask(
        output_root / "source-rendered-fresh-validation-or-final-heldout.png",
        rendered_new_heldout,
    )
    frozen_new_heldout_exclusion_artifact = _write_mask(
        output_root / "source-fresh-heldout-exclusion-24px.png",
        frozen_new_heldout_exclusion,
    )
    artifact_paths.extend(
        (
            Path(rendered_new_heldout_artifact["path"]),
            Path(frozen_new_heldout_exclusion_artifact["path"]),
        )
    )
    for name, mask in assignment_masks.items():
        artifact = _write_mask(output_root / f"source-{name.replace('_', '-')}.png", mask)
        source_assignment_artifacts[name] = artifact
        artifact_paths.append(Path(artifact["path"]))
    local_assignment_artifact = _write_mask(
        output_root / "source-native-visible-r1c0-assignment-original.png",
        local_assignment,
    )
    artifact_paths.append(Path(local_assignment_artifact["path"]))

    serializable_candidates = [
        {key: value for key, value in item.items() if key != "warp"}
        for item in candidates
    ]
    frozen_candidate_path: Path | None = None
    if selected is not None:
        warp = selected["warp"]
        transform = serialize_farms_nonrigid_transform(
            seed_matrix_working=seed_matrix,
            projection=projection,
            warp_working=warp,
            working_scale=source.working_scale,
            source_original_shape=source.rgb_original.shape[:2],
            source_working_shape=source.rgb_working.shape[:2],
            target_grid=reference.grid,
        )
        frozen = {
            "schema_version": 1,
            "kind": "farms_v9_frozen_validation_candidate_v1",
            "source_sha256": _sha256(source_path),
            "reference_manifest_sha256": _sha256(reference_manifest_path),
            "v8_history": {
                "validation_report_path": str(v8_report_path),
                "validation_report_sha256": _sha256(v8_report_path),
                "frozen_candidate_path": str(v8_frozen_path),
                "frozen_candidate_sha256": _sha256(v8_frozen_path),
                "old_validation_role": "consumed_training_in_v9",
            },
            "partition_receipt_path": str(partition_receipt_path),
            "partition_receipt_sha256": _sha256(partition_receipt_path),
            "selected_candidate": {
                key: value for key, value in selected.items() if key != "warp"
            },
            "warp_working": {
                "centers_reference_px": warp.centers_reference_px.tolist(),
                "coefficients_source_px": warp.coefficients_source_px.tolist(),
                "radius_reference_px": float(warp.radius_reference_px),
                "ridge": float(warp.ridge),
            },
            "serialized_transform": transform,
            "authority": {
                "fresh_validation_used_for_model_selection": True,
                "sealed_final_county_scores_evaluated": False,
                "golden_gate_or_east_bay_masks_read_or_scored": False,
                "consumed_v8_validation_used_as_training": True,
                "retained_masks_used_only_for_buffered_exclusion": True,
            },
        }
        frozen_candidate_path = output_root / "frozen-validation-candidate.json"
        frozen_candidate_path.write_text(
            json.dumps(frozen, indent=2, sort_keys=True) + "\n"
        )
        artifact_paths.append(frozen_candidate_path)

    status = "validation_pass" if selected is not None else "blocked"
    stop_reason = (
        "fresh_v9_validation_candidate_frozen_final_acceptance_not_evaluated"
        if selected is not None
        else "no_v9_candidate_passed_fresh_validation_final_acceptance_not_evaluated"
    )
    report = {
        "schema_version": 1,
        "kind": "farms_v9_rollforward_validation_preflight_v1",
        "status": status,
        "stop_reason": stop_reason,
        "inputs": {
            "source_path": str(source_path),
            "source_sha256": _sha256(source_path),
            "reference_manifest_path": str(reference_manifest_path),
            "reference_manifest_sha256": _sha256(reference_manifest_path),
            "v8_validation_report_path": str(v8_report_path),
            "v8_validation_report_sha256": _sha256(v8_report_path),
            "v8_frozen_candidate_path": str(v8_frozen_path),
            "v8_frozen_candidate_sha256": _sha256(v8_frozen_path),
            "partition_receipt_path": str(partition_receipt_path),
            "partition_receipt_sha256": _sha256(partition_receipt_path),
        },
        "authority": {
            "v8_validation_is_consumed_training": True,
            "fresh_v9_validation_used_for_selection": True,
            "sealed_final_county_mask_persisted_but_scores_evaluated": False,
            "golden_gate_or_east_bay_masks_read_or_scored": False,
            "consumed_outer_north_south_validation_used_only_for_non_degradation": True,
            "manual_inputs_used": False,
            "official_iteration_or_extraction_written": False,
        },
        "config": {
            "v9": config.__dict__,
            "base": base_config.__dict__,
        },
        "partition": partition_receipt,
        "source_observable_extent_diagnostics": source_extent_diagnostics,
        "source_county_topology_diagnostics": source.county_topology_diagnostics,
        "training": {
            "expanded_county": county_diagnostics,
            "native_visible_r1c0": local_diagnostics,
            "nevada_admin_not_used_for_controls": nevada_diagnostics,
            "county_control_count": int(len(county_controls)),
            "local_r1c0_control_count": int(len(local_controls)),
            "fresh_heldout_exclusion_authority": exclusion_authority,
            "fresh_heldout_exclusion_proof": exclusion_proof,
        },
        "fresh_heldout_exclusion_artifacts": {
            "rendered_heldout": rendered_new_heldout_artifact,
            "heldout_exclusion_24px": frozen_new_heldout_exclusion_artifact,
        },
        "source_assignment_artifacts": source_assignment_artifacts,
        "local_native_assignment_artifact": local_assignment_artifact,
        "candidate_count": len(candidates),
        "candidates": serializable_candidates,
        "selected_candidate_id": selected["id"] if selected else None,
        "frozen_candidate_path": str(frozen_candidate_path) if frozen_candidate_path else None,
        "frozen_candidate_sha256": _sha256(frozen_candidate_path) if frozen_candidate_path else None,
        "sealed_final_county_scores_evaluated": False,
        "golden_gate_or_east_bay_scores_evaluated": False,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    artifact_paths.append(report_path)
    return FarmsV9PreflightResult(
        status=status,
        stop_reason=stop_reason,
        report_path=report_path,
        frozen_candidate_path=frozen_candidate_path,
        artifact_paths=tuple(artifact_paths),
    )


def farms_v9_final_gate_contract() -> Mapping[str, Any]:
    """Return the immutable one-shot gate contract without opening final masks."""

    config = FarmsNonrigidConfig()
    return {
        "schema_version": 1,
        "kind": "farms_v9_one_shot_final_gate_contract_v1",
        "candidate_policy": "exact_frozen_v9_candidate_only_no_tuning_or_fallback",
        "canonical_output_relative_path": CANONICAL_V9_FINAL_OUTPUT_RELATIVE_PATH,
        "canonical_output_absolute_path": str(CANONICAL_V9_FINAL_OUTPUT_ROOT),
        "expected_sha256": {
            "partition_receipt": EXPECTED_V9_PARTITION_RECEIPT_SHA256,
            "validation_report": EXPECTED_V9_VALIDATION_REPORT_SHA256,
            "frozen_validation_candidate": EXPECTED_V9_FROZEN_CANDIDATE_SHA256,
            "sealed_final_county_file": EXPECTED_V9_FINAL_COUNTY_FILE_SHA256,
            "sealed_final_county_logical": EXPECTED_V9_FINAL_COUNTY_LOGICAL_SHA256,
            "golden_gate_file": EXPECTED_V8_GOLDEN_GATE_FILE_SHA256,
            "golden_gate_logical": EXPECTED_V8_GOLDEN_GATE_LOGICAL_SHA256,
            "east_bay_file": EXPECTED_V8_EAST_BAY_FILE_SHA256,
            "east_bay_logical": EXPECTED_V8_EAST_BAY_LOGICAL_SHA256,
        },
        "final_county": {
            "minimum_visible_count": config.holdout_minimum_visible_count,
            "minimum_visible_fraction": config.holdout_minimum_visible_fraction,
            "maximum_median_working_px": config.county_median_limit_px,
            "minimum_within_8px_fraction": config.county_within_8_minimum,
            "balanced_grid": list(config.county_validation_balanced_cells),
            "balanced_minimum_visible_count_per_cell": (
                config.county_validation_balanced_minimum_visible_count
            ),
            "balanced_minimum_visible_cells": (
                config.county_validation_balanced_minimum_visible_cells
            ),
            "balanced_minimum_visible_rows": (
                config.county_validation_balanced_minimum_visible_rows
            ),
            "balanced_minimum_visible_columns": (
                config.county_validation_balanced_minimum_visible_columns
            ),
            "balanced_maximum_p90_working_px": (
                config.county_validation_balanced_p90_limit_px
            ),
            "balanced_minimum_cell_pass_fraction": (
                config.county_validation_balanced_minimum_cell_pass_fraction
            ),
            "balanced_minimum_axis_pass_fraction": (
                config.county_validation_balanced_minimum_axis_pass_fraction
            ),
        },
        "golden_gate_and_east_bay": {
            "minimum_visible_count": config.holdout_minimum_visible_count,
            "minimum_visible_fraction": config.bay_holdout_minimum_visible_fraction,
            "maximum_median_working_px": config.state_median_limit_px,
            "maximum_p90_working_px": config.bay_holdout_p90_limit_px,
            "minimum_within_8px_fraction": config.bay_holdout_within_8_minimum,
        },
        "regularity": {
            "minimum_jacobian_ratio": config.minimum_jacobian_ratio,
            "maximum_jacobian_ratio": config.maximum_jacobian_ratio,
            "maximum_local_condition_number": config.maximum_local_condition_number,
            "maximum_residual_displacement_working_px": (
                config.maximum_residual_displacement_working_px
            ),
            "positive_triangles_required": True,
            "zero_grid_overlap_required": True,
            "zero_boundary_self_intersections_required": True,
        },
        "consumer_integrity": {
            "sparse_projection_residual_roundtrip_required": True,
            "sparse_convergence_required": True,
            "sparse_maximum_source_roundtrip_error_px": 0.02,
            "sparse_maximum_reference_roundtrip_error_px": 0.02,
            "full_target_reference_to_source_remap_required": True,
            "full_source_to_reference_remap_required": True,
            "full_source_inverse_convergence_required": True,
        },
        "authority": {
            "source_observable_extent_logic": "same_frozen_v9_validation_producer",
            "sealed_county_and_named_water_scores_evaluated_once": True,
            "selection_after_final_scores": False,
            "parameter_tuning_after_final_scores": False,
            "fallback_candidate_after_final_scores": False,
            "official_attempt_written": False,
        },
    }


def _require_canonical_v9_final_output(
    output_root: Path,
) -> Path:
    """Host-enforce one canonical final location so evidence opens only once."""

    expected = CANONICAL_V9_FINAL_OUTPUT_ROOT
    actual = output_root.resolve()
    if actual != expected:
        raise ValueError(
            f"One-shot final output must use canonical path {expected}; got {actual}"
        )
    if actual.exists():
        raise FileExistsError(
            "Canonical one-shot final output already exists; retry is forbidden"
        )
    return expected


def _final_acceptance_gate_decisions(
    *,
    county: Mapping[str, Any],
    balanced_county: Mapping[str, Any],
    golden_gate: Mapping[str, Any],
    east_bay: Mapping[str, Any],
    regularity: Mapping[str, Any],
    serialized_roundtrip: Mapping[str, Any],
    full_consumer_preflight: Mapping[str, Any],
) -> Mapping[str, bool]:
    config = FarmsNonrigidConfig()
    return {
        "final_county_minimum_visible_count": bool(
            int(county["visible_count"]) >= config.holdout_minimum_visible_count
        ),
        "final_county_minimum_visible_fraction": bool(
            float(county["visible_fraction"])
            >= config.holdout_minimum_visible_fraction
        ),
        "final_county_median": bool(
            float(county["median_px"]) <= config.county_median_limit_px
        ),
        "final_county_within_8px_support": bool(
            float(county["within_8px_fraction"])
            >= config.county_within_8_minimum
        ),
        "final_county_geographic_balance": bool(balanced_county["passed"]),
        "golden_gate_named_bay": _supported_coast_gate(
            golden_gate, config, bay=True
        ),
        "east_bay_named_bay": _supported_coast_gate(east_bay, config, bay=True),
        "regularity": bool(regularity["passed"]),
        "serialized_roundtrip": bool(serialized_roundtrip["passed"]),
        "full_consumer_preflight": bool(full_consumer_preflight["passed"]),
    }


def projection_residual_consumer_roundtrip_report(
    transform: Mapping[str, Any], reference_points: np.ndarray
) -> Mapping[str, Any]:
    """Replay the production projection/residual consumer on sparse points."""

    if transform.get("kind") != "projection_aware_residual_warp_mapbox_registration":
        raise ValueError("V9 roundtrip requires a projection-aware residual contract")
    points = np.asarray(reference_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (2,) or not len(points):
        raise ValueError("Roundtrip reference points must have shape (N, 2)")
    projection = transform["projection"]
    crs = CRS.from_wkt(projection["crs_wkt"])
    forward_transformer = Transformer.from_crs(
        "EPSG:3857", crs, always_xy=True
    )
    inverse_transformer = Transformer.from_crs(
        crs, "EPSG:3857", always_xy=True
    )
    reference_x, reference_y = points[:, 0], points[:, 1]
    base_x, base_y = _projection_reference_to_source_base(
        transform, reference_x, reference_y, forward_transformer
    )
    residual_x, residual_y = _residual_displacement(
        transform, reference_x, reference_y
    )
    source_x = base_x + residual_x
    source_y = base_y + residual_y
    estimate_x, estimate_y = _projection_source_to_reference_base(
        transform, source_x, source_y, inverse_transformer
    )
    inverse = transform["inverse_solver"]
    maximum_iterations = int(inverse["maximum_iterations"])
    reference_tolerance = float(inverse["reference_tolerance_px"])
    iterations_used = maximum_iterations
    maximum_update = math.inf
    final_update = np.full(len(points), math.inf, dtype=np.float64)
    for iteration in range(1, maximum_iterations + 1):
        residual_x, residual_y = _residual_displacement(
            transform, estimate_x, estimate_y
        )
        next_x, next_y = _projection_source_to_reference_base(
            transform,
            source_x - residual_x,
            source_y - residual_y,
            inverse_transformer,
        )
        final_update = np.maximum(
            np.abs(next_x - estimate_x), np.abs(next_y - estimate_y)
        )
        maximum_update = float(np.max(final_update))
        estimate_x, estimate_y = next_x, next_y
        if maximum_update <= reference_tolerance:
            iterations_used = iteration
            break
    recovered_x, recovered_y = _projection_reference_to_source_base(
        transform, estimate_x, estimate_y, forward_transformer
    )
    residual_x, residual_y = _residual_displacement(
        transform, estimate_x, estimate_y
    )
    source_error = np.maximum(
        np.abs(recovered_x + residual_x - source_x),
        np.abs(recovered_y + residual_y - source_y),
    )
    reference_error = np.maximum(
        np.abs(estimate_x - reference_x), np.abs(estimate_y - reference_y)
    )
    source_tolerance = float(inverse["source_roundtrip_tolerance_px"])
    reference_roundtrip_tolerance = 0.02
    converged = bool(
        np.all(np.isfinite(final_update)) and maximum_update <= reference_tolerance
    )
    passed = bool(
        converged
        and
        np.all(np.isfinite(source_error))
        and np.all(np.isfinite(reference_error))
        and float(np.max(source_error)) <= source_tolerance
        and float(np.max(reference_error)) <= reference_roundtrip_tolerance
    )
    return {
        "passed": passed,
        "method": "production_projection_residual_inverse_fixed_point_sparse_replay",
        "point_count": int(len(points)),
        "iterations_used": iterations_used,
        "maximum_iterations": maximum_iterations,
        "converged": converged,
        "converged_point_fraction": float(
            np.mean(final_update <= reference_tolerance)
        ),
        "final_maximum_reference_update_px": maximum_update,
        "reference_tolerance_px": reference_tolerance,
        "maximum_source_roundtrip_error_px": float(np.max(source_error)),
        "source_roundtrip_tolerance_px": source_tolerance,
        "maximum_reference_roundtrip_error_px": float(np.max(reference_error)),
        "reference_roundtrip_tolerance_px": reference_roundtrip_tolerance,
        "production_helpers_used": [
            "_projection_reference_to_source_base",
            "_projection_source_to_reference_base",
            "_residual_displacement",
        ],
    }


def projection_residual_full_consumer_preflight(
    transform: Mapping[str, Any], source_shape: tuple[int, int]
) -> Mapping[str, Any]:
    """Exercise both exact dense production remappers before sealed scoring."""

    target_started = time.perf_counter()
    target_x, target_y = _reference_to_source_remap(transform)
    target_runtime = float(time.perf_counter() - target_started)
    target_finite = bool(
        np.all(np.isfinite(target_x)) and np.all(np.isfinite(target_y))
    )
    target_report = {
        "shape": list(target_x.shape),
        "finite": target_finite,
        "x_minimum": float(np.min(target_x)),
        "x_maximum": float(np.max(target_x)),
        "y_minimum": float(np.min(target_y)),
        "y_maximum": float(np.max(target_y)),
        "runtime_seconds": target_runtime,
    }
    del target_x, target_y

    source_diagnostics: dict[str, Any] = {}
    source_started = time.perf_counter()
    source_x, source_y = _source_to_reference_remap(
        transform, source_shape, diagnostics=source_diagnostics
    )
    source_runtime = float(time.perf_counter() - source_started)
    source_finite = bool(
        np.all(np.isfinite(source_x)) and np.all(np.isfinite(source_y))
    )
    source_report = {
        **source_diagnostics,
        "shape": list(source_x.shape),
        "finite": source_finite,
        "x_minimum": float(np.min(source_x)),
        "x_maximum": float(np.max(source_x)),
        "y_minimum": float(np.min(source_y)),
        "y_maximum": float(np.max(source_y)),
        "runtime_seconds": source_runtime,
    }
    del source_x, source_y
    inverse = transform["inverse_solver"]
    passed = bool(
        target_finite
        and source_finite
        and source_report["converged"]
        and source_report["maximum_final_reference_update_px"]
        <= float(inverse["reference_tolerance_px"])
        and source_report["maximum_source_roundtrip_error_px"]
        <= float(inverse["source_roundtrip_tolerance_px"])
    )
    return {
        "passed": passed,
        "method": "exact_production_dense_reference_and_source_remap_preflight",
        "target_reference_to_source": target_report,
        "source_to_reference": source_report,
    }


def run_farms_v9_one_shot_final_acceptance(
    *,
    source_path: Path,
    reference_manifest_path: Path,
    v8_preflight_root: Path,
    partition_receipt_path: Path,
    v9_validation_report_path: Path,
    v9_frozen_candidate_path: Path,
    output_root: Path,
) -> FarmsV9FinalAcceptanceResult:
    """Evaluate the three sealed v9 acceptance channels exactly once."""

    source_path = source_path.resolve()
    reference_manifest_path = reference_manifest_path.resolve()
    v8_preflight_root = v8_preflight_root.resolve()
    partition_receipt_path = partition_receipt_path.resolve()
    v9_validation_report_path = v9_validation_report_path.resolve()
    v9_frozen_candidate_path = v9_frozen_candidate_path.resolve()
    output_root = output_root.resolve()
    output_root = _require_canonical_v9_final_output(output_root)

    _require_exact_artifact_sha256(
        partition_receipt_path,
        label="v9 partition receipt",
        expected_sha256=EXPECTED_V9_PARTITION_RECEIPT_SHA256,
    )
    _require_exact_artifact_sha256(
        v9_validation_report_path,
        label="v9 validation report",
        expected_sha256=EXPECTED_V9_VALIDATION_REPORT_SHA256,
    )
    _require_exact_artifact_sha256(
        v9_frozen_candidate_path,
        label="v9 frozen candidate",
        expected_sha256=EXPECTED_V9_FROZEN_CANDIDATE_SHA256,
    )
    if _sha256(source_path) != EXPECTED_FARMS_SOURCE_SHA256:
        raise ValueError("Final acceptance requires pristine farmsv2.png bytes")
    if _sha256(reference_manifest_path) != EXPECTED_MAPBOX_V2_MANIFEST_SHA256:
        raise ValueError("Final acceptance requires the exact Mapbox-v2 manifest")

    v8_report_path = v8_preflight_root / "validation-report.json"
    v8_frozen_path = v8_preflight_root / "frozen-validation-candidate.json"
    _require_exact_artifact_sha256(
        v8_report_path,
        label="v8 validation report",
        expected_sha256=EXPECTED_V8_VALIDATION_REPORT_SHA256,
    )
    _require_exact_artifact_sha256(
        v8_frozen_path,
        label="v8 frozen candidate",
        expected_sha256=EXPECTED_V8_FROZEN_CANDIDATE_SHA256,
    )
    partition = json.loads(partition_receipt_path.read_text())
    validation = json.loads(v9_validation_report_path.read_text())
    frozen = json.loads(v9_frozen_candidate_path.read_text())
    v8_report = json.loads(v8_report_path.read_text())
    v8_frozen = json.loads(v8_frozen_path.read_text())
    if validation.get("status") != "validation_pass":
        raise ValueError("V9 validation did not pass")
    if validation.get("frozen_candidate_sha256") != EXPECTED_V9_FROZEN_CANDIDATE_SHA256:
        raise ValueError("V9 validation report does not pin the exact frozen candidate")
    if frozen.get("partition_receipt_sha256") != EXPECTED_V9_PARTITION_RECEIPT_SHA256:
        raise ValueError("V9 frozen candidate does not pin the partition receipt")
    if frozen.get("selected_candidate", {}).get("id") != "v8-frozen-warp-baseline":
        raise ValueError("Final acceptance candidate identity is not frozen")
    if frozen.get("source_sha256") != EXPECTED_FARMS_SOURCE_SHA256:
        raise ValueError("V9 frozen candidate source authority changed")
    if frozen.get("reference_manifest_sha256") != EXPECTED_MAPBOX_V2_MANIFEST_SHA256:
        raise ValueError("V9 frozen candidate reference authority changed")

    config = FarmsNonrigidConfig()
    source = _load_source(source_path, config)
    source_extents, source_extent_diagnostics = _source_observable_extents(
        source, config
    )
    reference = load_pinned_mapbox_reference(reference_manifest_path)
    projection = next(
        item
        for item in _projection_contexts(reference)
        if item.id == str(v8_frozen["projection"])
    )
    seed_matrix = np.asarray(v8_frozen["seed_matrix_working"], dtype=np.float64)
    warp = _warp_from_frozen(frozen)
    recomputed_transform = serialize_farms_nonrigid_transform(
        seed_matrix_working=seed_matrix,
        projection=projection,
        warp_working=warp,
        working_scale=source.working_scale,
        source_original_shape=source.rgb_original.shape[:2],
        source_working_shape=source.rgb_working.shape[:2],
        target_grid=reference.grid,
    )
    if recomputed_transform != frozen.get("serialized_transform"):
        raise ValueError("Frozen v9 serialized transform does not reproduce exactly")
    state_y, state_x = np.nonzero(reference.state_land)
    roundtrip_points = np.column_stack((state_x, state_y)).astype(np.float64)
    if len(roundtrip_points) > 1_000:
        indices = np.linspace(0, len(roundtrip_points) - 1, 1_000).round().astype(int)
        roundtrip_points = roundtrip_points[indices]
    roundtrip = projection_residual_consumer_roundtrip_report(
        recomputed_transform, roundtrip_points
    )
    full_consumer_preflight = projection_residual_full_consumer_preflight(
        recomputed_transform, source.rgb_original.shape[:2]
    )
    regularity = _regularity_report(
        reference, projection, seed_matrix, warp, config
    )

    # Freeze the predeclared gates before opening any sealed role.  If the
    # process fails after this point, the nonempty directory is immutable
    # evidence that the one-shot was consumed and cannot be retried in place.
    output_root.mkdir(parents=True, exist_ok=False)
    gate_contract_path = output_root / "predeclared-gate-contract.json"
    gate_contract_path.write_text(
        json.dumps(farms_v9_final_gate_contract(), indent=2, sort_keys=True) + "\n"
    )

    final_artifact = partition["artifacts"]["final_acceptance"]
    if final_artifact.get("sha256") != EXPECTED_V9_FINAL_COUNTY_FILE_SHA256:
        raise ValueError("Partition receipt does not pin the sealed final county file")
    final_county = _load_exact_v8_role(
        Path(final_artifact["path"]),
        role="v9_final_county",
        expected_file_sha256=EXPECTED_V9_FINAL_COUNTY_FILE_SHA256,
        expected_logical_sha256=EXPECTED_V9_FINAL_COUNTY_LOGICAL_SHA256,
    )
    retained_artifacts = v8_report["named_pacific_mask_artifacts"]
    golden_artifact = retained_artifacts["golden_gate_test_microsegments"]
    east_artifact = retained_artifacts["east_bay_test_microsegments"]
    golden_gate_mask = _load_exact_v8_role(
        Path(golden_artifact["path"]),
        role="golden_gate_test_microsegments",
        expected_file_sha256=EXPECTED_V8_GOLDEN_GATE_FILE_SHA256,
        expected_logical_sha256=EXPECTED_V8_GOLDEN_GATE_LOGICAL_SHA256,
    )
    east_bay_mask = _load_exact_v8_role(
        Path(east_artifact["path"]),
        role="east_bay_test_microsegments",
        expected_file_sha256=EXPECTED_V8_EAST_BAY_FILE_SHA256,
        expected_logical_sha256=EXPECTED_V8_EAST_BAY_LOGICAL_SHA256,
    )

    final_y, final_x = np.nonzero(final_county)
    final_points = np.column_stack((final_x, final_y)).astype(np.float64)
    if len(final_points) > 20_000:
        indices = np.linspace(0, len(final_points) - 1, 20_000).round().astype(int)
        final_points = final_points[indices]
    final_county_report = _holdout_report(
        final_points,
        source.county_topology,
        projection,
        reference.grid,
        seed_matrix,
        warp,
        observable_extent=source_extents["county_lines"],
    )
    final_cells = _county_validation_geographic_cells(final_county, config)
    balanced_final = _balanced_county_validation_report(
        final_cells,
        source.county_topology,
        projection,
        reference.grid,
        seed_matrix,
        warp,
        observable_extent=source_extents["county_lines"],
        config=config,
    )

    def named_report(mask: np.ndarray) -> Mapping[str, Any]:
        y, x = np.nonzero(mask)
        return _holdout_report(
            np.column_stack((x, y)).astype(np.float64),
            source.native_water_edge,
            projection,
            reference.grid,
            seed_matrix,
            warp,
            observable_extent=source_extents["named_bay"],
        )

    golden_gate_report = named_report(golden_gate_mask)
    east_bay_report = named_report(east_bay_mask)
    gates = _final_acceptance_gate_decisions(
        county=final_county_report,
        balanced_county=balanced_final,
        golden_gate=golden_gate_report,
        east_bay=east_bay_report,
        regularity=regularity,
        serialized_roundtrip=roundtrip,
        full_consumer_preflight=full_consumer_preflight,
    )
    accepted = bool(all(gates.values()))

    artifact_paths: list[Path] = [gate_contract_path]
    rendered_final = _render_line(
        final_county,
        source.rgb_working.shape[:2],
        projection,
        reference.grid,
        seed_matrix,
        warp,
    )
    rendered_golden = _render_line(
        golden_gate_mask,
        source.rgb_working.shape[:2],
        projection,
        reference.grid,
        seed_matrix,
        warp,
    )
    rendered_east = _render_line(
        east_bay_mask,
        source.rgb_working.shape[:2],
        projection,
        reference.grid,
        seed_matrix,
        warp,
    )
    rendered_artifacts = {
        "final_county": _write_mask(output_root / "rendered-final-county.png", rendered_final),
        "golden_gate": _write_mask(output_root / "rendered-golden-gate.png", rendered_golden),
        "east_bay": _write_mask(output_root / "rendered-east-bay.png", rendered_east),
    }
    artifact_paths.extend(Path(item["path"]) for item in rendered_artifacts.values())
    overlay = source.rgb_working.copy()
    gray = cv2.cvtColor(overlay, cv2.COLOR_RGB2GRAY)
    overlay = np.clip(0.55 * overlay + 0.45 * gray[:, :, None], 0, 255).astype(np.uint8)
    overlay[source.county_topology] = (0, 220, 255)
    overlay[rendered_final] = (255, 230, 0)
    overlay[rendered_golden] = (255, 70, 220)
    overlay[rendered_east] = (255, 145, 0)
    overlay[source.topology.state_coast] = (60, 255, 90)
    overlay_path = output_root / "one-shot-final-overlay.png"
    Image.fromarray(overlay).save(overlay_path)
    overlay_artifact = {"path": str(overlay_path), "sha256": _sha256(overlay_path)}
    artifact_paths.append(overlay_path)

    status = "accepted" if accepted else "rejected"
    stop_reason = (
        "one_shot_v9_final_acceptance_passed_official_promotion_not_written"
        if accepted
        else "one_shot_v9_final_acceptance_failed_no_tuning_or_fallback_permitted"
    )
    report = {
        "schema_version": 1,
        "kind": "farms_v9_one_shot_final_acceptance_v1",
        "status": status,
        "stop_reason": stop_reason,
        "gate_contract": farms_v9_final_gate_contract(),
        "gate_contract_artifact": {
            "path": str(gate_contract_path),
            "sha256": _sha256(gate_contract_path),
        },
        "inputs": {
            "source_path": str(source_path),
            "source_sha256": _sha256(source_path),
            "reference_manifest_path": str(reference_manifest_path),
            "reference_manifest_sha256": _sha256(reference_manifest_path),
            "partition_receipt_path": str(partition_receipt_path),
            "partition_receipt_sha256": _sha256(partition_receipt_path),
            "v9_validation_report_path": str(v9_validation_report_path),
            "v9_validation_report_sha256": _sha256(v9_validation_report_path),
            "v9_frozen_candidate_path": str(v9_frozen_candidate_path),
            "v9_frozen_candidate_sha256": _sha256(v9_frozen_candidate_path),
            "v8_validation_report_path": str(v8_report_path),
            "v8_validation_report_sha256": _sha256(v8_report_path),
            "v8_frozen_candidate_path": str(v8_frozen_path),
            "v8_frozen_candidate_sha256": _sha256(v8_frozen_path),
        },
        "candidate": {
            "id": frozen["selected_candidate"]["id"],
            "serialized_transform_exactly_reproduced": True,
            "serialized_roundtrip": roundtrip,
            "full_consumer_preflight": full_consumer_preflight,
            "regularity": regularity,
        },
        "scores": {
            "sealed_final_county": final_county_report,
            "sealed_final_county_balanced": balanced_final,
            "golden_gate": golden_gate_report,
            "east_bay": east_bay_report,
        },
        "gates": gates,
        "all_gates_passed": accepted,
        "source_observable_extent_diagnostics": source_extent_diagnostics,
        "rendered_artifacts": rendered_artifacts,
        "overlay_artifact": overlay_artifact,
        "authority": {
            "sealed_final_county_scores_evaluated_exactly_once": True,
            "golden_gate_scores_evaluated_exactly_once": True,
            "east_bay_scores_evaluated_exactly_once": True,
            "candidate_count_evaluated": 1,
            "candidate_selected_before_final_scores": True,
            "parameter_tuning_after_final_scores": False,
            "fallback_candidate_after_final_scores": False,
            "manual_inputs_used": False,
            "official_alignment_attempt_written": False,
            "extraction_written": False,
        },
    }
    report_path = output_root / "final-acceptance-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    artifact_paths.append(report_path)
    return FarmsV9FinalAcceptanceResult(
        status=status,
        stop_reason=stop_reason,
        report_path=report_path,
        artifact_paths=tuple(artifact_paths),
    )
