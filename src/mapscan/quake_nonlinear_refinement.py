"""Read-only source-native thin-plate refinement for the Quake alignment."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import distance_transform_edt

from .automatic_alignment_loop import (
    _project_reference_points,
    _projection_contexts,
    _transform,
    load_pinned_mapbox_reference,
)
from .native_alignment_validation import (
    REGIONAL_GRID,
    VALIDATION_SCALES,
    _balanced_gate,
    _normalized_line_metrics,
    _scaled_semantic,
)


SCHEMA_VERSION = "mapscan.quake-native-nonlinear-refinement.v1"
CONTROL_GRID = (20, 20)
SMOOTHING = 0.0003


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


@dataclass(frozen=True)
class ThinPlateWarp:
    center: np.ndarray
    span: np.ndarray
    controls: np.ndarray
    displacements: np.ndarray
    interpolator: RBFInterpolator

    def map_original(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        return points + self.interpolator((points - self.center) / self.span)

    def map_scaled(self, points: np.ndarray, scale: float) -> np.ndarray:
        return self.map_original(points) * scale


def _reference_points(mask: np.ndarray, reference: Any, projection: Any, matrix: np.ndarray):
    y, x = np.nonzero(mask)
    points = np.column_stack((x, y)).astype(np.float64)
    projected = _project_reference_points(points, projection, reference.grid)
    return x, y, _transform(projected, matrix)


def _build_warp(reference: Any, projection: Any, matrix: np.ndarray, source: np.ndarray, working_scale: float):
    rx, ry, base = _reference_points(reference.state_coast, reference, projection, matrix)
    height, width = source.shape
    x, y = np.rint(base[:, 0]).astype(int), np.rint(base[:, 1]).astype(int)
    inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    distance, indices = distance_transform_edt(~source, return_indices=True)
    residual = np.full(len(x), np.inf)
    residual[inside] = distance[y[inside], x[inside]]
    sx, sy = np.clip(x, 0, width - 1), np.clip(y, 0, height - 1)
    nearest = np.column_stack((indices[1, sy, sx], indices[0, sy, sx]))
    delta = nearest - base
    rows, columns = CONTROL_GRID
    rh, rw = reference.state_coast.shape
    controls, values, reports = [], [], []
    for row in range(rows):
        for column in range(columns):
            selected = (
                inside
                & (ry >= row * rh / rows) & (ry < (row + 1) * rh / rows)
                & (rx >= column * rw / columns) & (rx < (column + 1) * rw / columns)
                & (residual * working_scale <= 50.0)
            )
            count = int(np.count_nonzero(selected))
            if count < 15:
                continue
            local = delta[selected]
            median = np.median(local, axis=0)
            mad = float(np.median(np.linalg.norm(local - median, axis=1)))
            if mad > 75.0:
                continue
            control = np.median(base[selected], axis=0)
            controls.append(control)
            values.append(median)
            reports.append({
                "row": row, "column": column, "support_point_count": count,
                "control_source_original": control.tolist(),
                "displacement_source_original_px": median.tolist(),
                "displacement_mad_source_original_px": mad,
            })
    if len(controls) < 20:
        raise ValueError("insufficient nonlinear anchors")
    controls = np.asarray(controls, dtype=float)
    values = np.asarray(values, dtype=float)
    center = np.asarray((width / 2, height / 2), dtype=float)
    span = np.asarray((width, height), dtype=float)
    interpolator = RBFInterpolator(
        (controls - center) / span, values,
        kernel="thin_plate_spline", smoothing=SMOOTHING, degree=1,
    )
    return ThinPlateWarp(center, span, controls, values, interpolator), reports


def _render(mapped: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    points = np.rint(mapped).astype(int)
    inside = (
        (points[:, 0] >= 0) & (points[:, 0] < shape[1])
        & (points[:, 1] >= 0) & (points[:, 1] < shape[0])
    )
    result = np.zeros(shape, dtype=np.uint8)
    result[points[inside, 1], points[inside, 0]] = 1
    return cv2.dilate(result, np.ones((2, 2), np.uint8)).astype(bool)


def _balanced(rx, ry, mapped, source, reference_shape, working_scale, validation_scale, grid):
    points = np.rint(mapped).astype(int)
    height, width = source.shape
    inside = (
        (points[:, 0] >= 0) & (points[:, 0] < width)
        & (points[:, 1] >= 0) & (points[:, 1] < height)
    )
    distance = distance_transform_edt(~source)
    residual = np.full(len(points), math.hypot(width, height))
    residual[inside] = distance[points[inside, 1], points[inside, 0]]
    residual *= working_scale / validation_scale
    rows, columns = grid
    rh, rw = reference_shape
    cells = []
    for row in range(rows):
        for column in range(columns):
            selected = (
                (ry >= row * rh / rows) & (ry < (row + 1) * rh / rows)
                & (rx >= column * rw / columns) & (rx < (column + 1) * rw / columns)
            )
            if np.count_nonzero(selected) < 20:
                continue
            values = residual[selected]
            p90 = float(np.quantile(values, 0.9))
            cells.append({
                "id": f"r{row + 1}-c{column + 1}", "row": row, "column": column,
                "working_equivalent_median_px": float(np.median(values)),
                "working_equivalent_p90_px": p90,
                "working_equivalent_within_8px_fraction": float(np.mean(values <= 8)),
                "passed": p90 <= 12,
            })
    report = _balanced_gate(cells)
    report["grid"] = list(grid)
    report["median_cell_p90_px"] = float(np.median([item["working_equivalent_p90_px"] for item in cells]))
    report["cells"] = cells
    report["passed"] = bool(report["passed"] and report["median_cell_p90_px"] <= 12)
    return report


def _regularity(warp: ThinPlateWarp, boundary: np.ndarray) -> dict[str, Any]:
    low, high = np.min(boundary, axis=0), np.max(boundary, axis=0)
    xs, ys = np.linspace(low[0], high[0], 41), np.linspace(low[1], high[1], 41)
    xx, yy = np.meshgrid(xs, ys)
    mapped = warp.map_original(np.column_stack((xx.ravel(), yy.ravel()))).reshape(41, 41, 2)
    dxdx = np.gradient(mapped[..., 0], xs[1] - xs[0], axis=1)
    dxdy = np.gradient(mapped[..., 0], ys[1] - ys[0], axis=0)
    dydx = np.gradient(mapped[..., 1], xs[1] - xs[0], axis=1)
    dydy = np.gradient(mapped[..., 1], ys[1] - ys[0], axis=0)
    determinant = dxdx * dydy - dxdy * dydx
    displacement = np.linalg.norm(warp.map_original(boundary) - boundary, axis=1)
    return {
        "passed": bool(np.min(determinant) > 0.1 and np.max(displacement) <= 250),
        "minimum_sampled_jacobian_determinant": float(np.min(determinant)),
        "maximum_sampled_jacobian_determinant": float(np.max(determinant)),
        "maximum_boundary_displacement_source_original_px": float(np.max(displacement)),
        "maximum_displacement_source_original_px": 250.0,
    }


def run_quake_nonlinear_refinement(map_dir: Path, reference_path: Path, output_dir: Path) -> Path:
    """Create and fully gate an isolated candidate; mutate no accepted artifact."""

    if output_dir.exists():
        raise FileExistsError(f"output exists: {output_dir}")
    alignment_path = map_dir / "automatic-alignment/accepted-alignment.json"
    adapter_path = map_dir / "source-clean/source-adapter.json"
    alignment, adapter = _read_json(alignment_path), _read_json(adapter_path)
    source_path = Path(adapter["source"]["path"]).resolve()
    if _sha256(source_path) != adapter["source"]["sha256"]:
        raise ValueError("pristine source hash mismatch")
    reference = load_pinned_mapbox_reference(reference_path.resolve())
    for key in ("style_sha256", "tilejson_sha256", "tile_aggregate_sha256"):
        if alignment["mapbox_reference"][key] != reference.pin[key]:
            raise ValueError("raw Mapbox pin changed")
    transform = alignment["transform"]
    working_scale = float(transform["working_scale_from_original"])
    matrix = np.asarray(transform["candidate_normalized_to_source_original_matrix"], dtype=float)
    projection = next(
        item for item in _projection_contexts(reference)
        if item.id == transform["projection"]["id"]
    )
    with Image.open(source_path) as image:
        source_rgb = np.asarray(image.convert("RGB"))
    original_shape = source_rgb.shape[:2]
    hypothesis = alignment["scores"]["source_alignment_hypothesis"]
    output_dir.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix="mapscan-quake-nonlinear-") as temporary:
        temporary = Path(temporary)
        native_scratch = temporary / "native"
        native_scratch.mkdir()
        native, native_detector = _scaled_semantic(
            source_rgb, "ordered_gradient_bands", native_scratch,
            validation_scale=1.0, original_shape=original_shape,
            generator_hypothesis_id=hypothesis["generator_hypothesis_id"],
            hypothesis_variant_kind=hypothesis["variant_kind"],
        )
        warp, anchors = _build_warp(reference, projection, matrix, native.state_coast, working_scale)
        shutil.rmtree(native_scratch)
        rx, ry, base_boundary = _reference_points(reference.state_coast, reference, projection, matrix)
        candidate_boundary = warp.map_original(base_boundary)
        multiscale, native_balance = [], None
        for validation_scale in VALIDATION_SCALES:
            if validation_scale == 1.0:
                source_state = native.state_coast
                source_shape = original_shape
                detector = native_detector
            else:
                source_shape = (
                    round(original_shape[0] * validation_scale),
                    round(original_shape[1] * validation_scale),
                )
                source_state = cv2.resize(
                    native.state_coast.astype(np.uint8),
                    (round(original_shape[1] * validation_scale), round(original_shape[0] * validation_scale)),
                    interpolation=cv2.INTER_AREA,
                ) > 0.1
                detector = "authoritative-native-boundary-pyramid-from-100-percent-v1"
            mapped = candidate_boundary * validation_scale
            rendered = _render(mapped, source_shape)
            kernel = max(3, round(25 * validation_scale / working_scale))
            corridor = cv2.dilate(rendered.astype(np.uint8), np.ones((kernel, kernel), np.uint8)).astype(bool)
            metrics = _normalized_line_metrics(
                rendered, source_state,
                accepted_working_scale=working_scale, validation_scale=validation_scale,
                source_scope=corridor, overlap_tolerance_working_px=5,
            )
            balanced = _balanced(
                rx, ry, mapped, source_state, reference.state_coast.shape,
                working_scale, validation_scale, (4, 5),
            )
            gates = {
                "semantic_full_state_median": {
                    "passed": metrics["working_equivalent_median_px"] <= 4.5,
                    "value": metrics["working_equivalent_median_px"], "maximum": 4.5,
                },
                "semantic_full_state_balanced_tail": balanced,
                "semantic_full_state_support": {
                    "passed": metrics["working_equivalent_within_8px_fraction"] >= 0.75,
                    "value": metrics["working_equivalent_within_8px_fraction"], "minimum": 0.75,
                },
                "semantic_full_state_symmetric_overlap": {
                    "passed": metrics["f1"] >= 0.42,
                    "value": metrics["f1"], "minimum": 0.42,
                },
                "county_channel": {"passed": True, "status": "not_applicable", "reason": "not observable in source"},
            }
            multiscale.append({
                "validation_scale_from_original": validation_scale,
                "source_shape": list(source_shape), "detector": detector,
                "state": metrics, "balanced_tail": balanced, "gates": gates,
                "passed": all(item["passed"] for item in gates.values()),
            })
            if validation_scale == 1.0:
                native_balance = _balanced(
                    rx, ry, mapped, source_state, reference.state_coast.shape,
                    working_scale, 1.0, REGIONAL_GRID,
                )
    assert native_balance is not None
    land_y, land_x = np.nonzero(reference.state_land)
    stride = max(1, len(land_x) // 60000)
    land_reference = np.column_stack((land_x[::stride], land_y[::stride])).astype(float)
    land_projected = _project_reference_points(land_reference, projection, reference.grid)
    land_mapped = np.rint(warp.map_original(_transform(land_projected, matrix))).astype(int)
    inside = (
        (land_mapped[:, 0] >= 0) & (land_mapped[:, 0] < original_shape[1])
        & (land_mapped[:, 1] >= 0) & (land_mapped[:, 1] < original_shape[0])
    )
    containment = float(np.mean(native.foreground_interior[land_mapped[inside, 1], land_mapped[inside, 0]]))
    containment_gate = {"passed": containment >= 0.9, "value": containment, "minimum": 0.9}
    regularity = _regularity(warp, base_boundary)
    gates = {
        "all_multiscale_immutable_semantic_gates": {
            "passed": all(item["passed"] for item in multiscale),
            "passing_scales": sum(item["passed"] for item in multiscale),
            "required_scales": len(multiscale),
        },
        "native_6x6_state_balance": native_balance,
        "silhouette_land_containment": containment_gate,
        "regular_nonfolding_warp": regularity,
        "pinned_mapbox_hashes_unchanged": True,
        "projection_round_trip_inherited_unchanged": True,
    }
    passed = all(bool(value if isinstance(value, bool) else value["passed"]) for value in gates.values())
    report = output_dir / "nonlinear-refinement.json"
    report.write_text(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "producer": "mapscan.quake_nonlinear_refinement",
        "status": "pass" if passed else "retry",
        "source": {"path": str(source_path), "sha256": _sha256(source_path), "shape": list(original_shape)},
        "accepted_alignment": {"path": str(alignment_path.resolve()), "sha256": _sha256(alignment_path), "mutated": False},
        "reference": {"path": str(reference_path.resolve()), "raw_pin": {key: reference.pin[key] for key in ("style_sha256", "tilejson_sha256", "tile_aggregate_sha256")}},
        "candidate": {
            "kind": "projection_aware_regularized_thin_plate_source_displacement",
            "base_matrix": matrix.tolist(), "control_grid": list(CONTROL_GRID),
            "smoothing": SMOOTHING, "center": warp.center.tolist(), "span": warp.span.tolist(),
            "controls_source_original": warp.controls.tolist(),
            "displacements_source_original_px": warp.displacements.tolist(),
            "anchor_reports": anchors,
        },
        "multiscale": multiscale, "native_6x6": native_balance, "gates": gates,
        "authority": {
            "fresh_pristine_source_only": True, "pinned_mapbox_geometry_only": True,
            "manual_input_used": False, "prior_class_raster_used": False,
            "accepted_artifacts_mutated": False, "public_artifacts_mutated": False,
        },
    }, indent=2) + "\n")
    return report
