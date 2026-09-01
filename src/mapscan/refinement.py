"""Fit conservative alignment refinements from author-drawn review arrows."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
from scipy.spatial import ConvexHull, QhullError


MODEL_ORDER = ("translation", "similarity", "affine", "projective")
MINIMUM_POINTS = {"translation": 1, "similarity": 2, "affine": 3, "projective": 4}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fit_matrix(model: str, current: np.ndarray, target: np.ndarray) -> np.ndarray:
    if len(current) < MINIMUM_POINTS[model]:
        raise ValueError(f"{model} needs at least {MINIMUM_POINTS[model]} points")
    if model == "translation":
        delta = np.median(target - current, axis=0)
        return np.asarray(
            [[1.0, 0.0, delta[0]], [0.0, 1.0, delta[1]], [0.0, 0.0, 1.0]]
        )
    rows = []
    values = []
    if model == "similarity":
        for (x, y), (target_x, target_y) in zip(current, target):
            rows.extend(((x, -y, 1.0, 0.0), (y, x, 0.0, 1.0)))
            values.extend((target_x, target_y))
        a, b, offset_x, offset_y = np.linalg.lstsq(
            np.asarray(rows), np.asarray(values), rcond=None
        )[0]
        return np.asarray(
            [[a, -b, offset_x], [b, a, offset_y], [0.0, 0.0, 1.0]]
        )
    if model == "affine":
        for (x, y), (target_x, target_y) in zip(current, target):
            rows.extend(
                (
                    (x, y, 1.0, 0.0, 0.0, 0.0),
                    (0.0, 0.0, 0.0, x, y, 1.0),
                )
            )
            values.extend((target_x, target_y))
        parameters = np.linalg.lstsq(
            np.asarray(rows), np.asarray(values), rcond=None
        )[0]
        return np.asarray(
            [
                [parameters[0], parameters[1], parameters[2]],
                [parameters[3], parameters[4], parameters[5]],
                [0.0, 0.0, 1.0],
            ]
        )
    if model == "projective":
        matrix, _ = cv2.findHomography(current, target, method=0)
        if matrix is None:
            raise ValueError("Correction points do not define a projective transform")
        return matrix / matrix[2, 2]
    raise ValueError(f"Unknown correction model: {model}")


def _apply_matrix(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack((points, np.ones(len(points)))) @ matrix.T
    if np.any(np.abs(homogeneous[:, 2]) < 1e-9):
        raise ValueError("Correction transform maps a point to infinity")
    return homogeneous[:, :2] / homogeneous[:, 2, None]


def _residual_summary(residuals: np.ndarray) -> Dict[str, float]:
    return {
        "median_px": float(np.median(residuals)),
        "p90_px": float(np.quantile(residuals, 0.9)),
        "max_px": float(np.max(residuals)),
        "rms_px": float(np.sqrt(np.mean(np.square(residuals)))),
    }


def _evaluate_model(
    model: str, current: np.ndarray, target: np.ndarray
) -> Tuple[np.ndarray, Dict[str, object]]:
    matrix = _fit_matrix(model, current, target)
    training = np.linalg.norm(_apply_matrix(matrix, current) - target, axis=1)
    held_out = []
    if len(current) - 1 >= MINIMUM_POINTS[model]:
        for index in range(len(current)):
            keep = np.arange(len(current)) != index
            held_out_matrix = _fit_matrix(model, current[keep], target[keep])
            prediction = _apply_matrix(
                held_out_matrix, current[index : index + 1]
            )[0]
            held_out.append(float(np.linalg.norm(prediction - target[index])))
    else:
        held_out = training.tolist()
    linear_determinant = float(np.linalg.det(matrix[:2, :2]))
    return matrix, {
        "model": model,
        "matrix_current_to_target_pixels": matrix.tolist(),
        "training": _residual_summary(training),
        "leave_one_out": _residual_summary(np.asarray(held_out)),
        "linear_determinant": linear_determinant,
    }


def _normalized_matrix(matrix: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = np.asarray(
        [[max(width - 1, 1), 0.0, 0.0], [0.0, max(height - 1, 1), 0.0], [0.0, 0.0, 1.0]]
    )
    return np.linalg.inv(scale) @ matrix @ scale


def _correction_points(
    record: Dict[str, object], reverse_declared_direction: bool = False
) -> Tuple[np.ndarray, np.ndarray, str]:
    current = []
    target = []
    direction = record.get("direction")
    if direction == "authoritative_reference_to_current_warped_source":
        for item in record.get("corrections", []):
            current.append(
                (float(item["source"]["pixel"]["x"]), float(item["source"]["pixel"]["y"]))
            )
            target.append(
                (float(item["reference"]["pixel"]["x"]), float(item["reference"]["pixel"]["y"]))
            )
        interpretation = "source_endpoint_to_reference_start"
    elif direction == "current_warped_source_to_authoritative_target":
        for item in record.get("corrections", []):
            current.append(
                (float(item["current"]["pixel"]["x"]), float(item["current"]["pixel"]["y"]))
            )
            target.append(
                (float(item["target"]["pixel"]["x"]), float(item["target"]["pixel"]["y"]))
            )
        interpretation = "declared_current_to_target"
        if reverse_declared_direction:
            current, target = target, current
            interpretation = "reversed_legacy_reference_start_to_source_endpoint"
    else:
        raise ValueError("Unsupported correction direction")
    return (
        np.asarray(current, dtype=np.float64),
        np.asarray(target, dtype=np.float64),
        interpretation,
    )


def fit_review_corrections(
    alignment_path: Path,
    corrections_path: Path,
    output_dir: Path,
    max_leave_one_out_p90_px: float = 4.0,
    max_leave_one_out_max_px: float = 8.0,
    minimum_axis_coverage: float = 0.5,
    minimum_hull_coverage: float = 0.08,
    reverse_declared_direction: bool = False,
) -> Dict[str, object]:
    alignment = json.loads(alignment_path.read_text())
    correction_record = json.loads(corrections_path.read_text())
    grid = correction_record["grid"]
    if grid.get("crs") != "EPSG:3857":
        raise ValueError("Review corrections must use an EPSG:3857 grid")
    current, target, direction_interpretation = _correction_points(
        correction_record, reverse_declared_direction=reverse_declared_direction
    )
    if len(current) < 3:
        raise ValueError("At least three distributed review corrections are required")
    width, height = int(grid["width"]), int(grid["height"])
    coverage = {
        "x_fraction": float(np.ptp(current[:, 0]) / max(width - 1, 1)),
        "y_fraction": float(np.ptp(current[:, 1]) / max(height - 1, 1)),
    }
    normalized_current = current / np.asarray(
        [max(width - 1, 1), max(height - 1, 1)]
    )
    try:
        coverage["convex_hull_fraction"] = float(ConvexHull(normalized_current).volume)
    except QhullError:
        coverage["convex_hull_fraction"] = 0.0
    if (
        min(coverage["x_fraction"], coverage["y_fraction"])
        < minimum_axis_coverage
        or coverage["convex_hull_fraction"] < minimum_hull_coverage
    ):
        raise ValueError(
            "Correction points are too concentrated for a global transform; "
            "spread them across both axes and across substantial map area"
        )

    candidates = []
    selected_model = None
    selected_matrix = None
    for model in MODEL_ORDER:
        if len(current) - 1 < MINIMUM_POINTS[model]:
            candidates.append(
                {
                    "model": model,
                    "evaluated": False,
                    "passes_residual_gate": False,
                    "reason": "Insufficient points for leave-one-out validation",
                }
            )
            continue
        matrix, report = _evaluate_model(model, current, target)
        report["evaluated"] = True
        report["passes_residual_gate"] = bool(
            report["leave_one_out"]["p90_px"] <= max_leave_one_out_p90_px
            and report["leave_one_out"]["max_px"] <= max_leave_one_out_max_px
            and report["linear_determinant"] > 0
        )
        candidates.append(report)
        if selected_model is None and report["passes_residual_gate"]:
            selected_model, selected_matrix = model, matrix
    if selected_model is None or selected_matrix is None:
        raise ValueError("No conservative correction model passes held-out residual gates")

    normalized_current_to_target = _normalized_matrix(
        selected_matrix, width, height
    )
    incremental_target_to_parent = np.linalg.inv(normalized_current_to_target)
    incremental_target_to_parent /= incremental_target_to_parent[2, 2]
    parent_correction = alignment.get("web_mercator_correction")
    if isinstance(parent_correction, dict):
        parent_grid = parent_correction["grid"]
        if (
            int(parent_grid["width"]) != width
            or int(parent_grid["height"]) != height
            or not np.allclose(parent_grid["bounds"], grid["bounds"])
        ):
            raise ValueError("Parent and new correction grids do not match")
        parent_target_to_base = np.asarray(
            parent_correction["target_to_current_normalized_matrix"],
            dtype=np.float64,
        )
        target_to_base = parent_target_to_base @ incremental_target_to_parent
        composition_depth = int(parent_correction.get("composition_depth", 1)) + 1
    else:
        target_to_base = incremental_target_to_parent
        composition_depth = 1
    target_to_base /= target_to_base[2, 2]
    selected_report = next(
        candidate for candidate in candidates if candidate["model"] == selected_model
    )

    refined = json.loads(json.dumps(alignment))
    refined["schema_version"] = max(int(refined.get("schema_version", 1)), 2)
    parent_transform_model = alignment.get("transform_model")
    if not isinstance(parent_transform_model, str):
        best = alignment.get("best")
        parent_transform_model = (
            best.get("transform_model") if isinstance(best, dict) else None
        )
        if (
            not isinstance(parent_transform_model, str)
            and isinstance(best, dict)
            and isinstance(best.get("parameters"), dict)
        ):
            # Early automatic-alignment artifacts predate the explicit field,
            # but their six-parameter model is the affine-like family.
            required = {
                "center_x_fraction",
                "center_y_fraction",
                "state_height_fraction",
                "x_scale_ratio",
                "rotation_degrees",
                "x_shear",
            }
            if required.issubset(best["parameters"]):
                parent_transform_model = "affine_like"
    if not isinstance(parent_transform_model, str):
        raise ValueError("Parent alignment has no transform model")
    refined["transform_model"] = (
        f"{parent_transform_model}+web_mercator_{selected_model}_review_correction"
    )
    refined["parent_alignment"] = {
        "path": str(alignment_path),
        "sha256": _sha256(alignment_path),
    }
    refined["web_mercator_correction"] = {
        "model": selected_model,
        "direction": "desired_target_to_pre_correction_sampling_grid",
        "grid": grid,
        "target_to_current_normalized_matrix": target_to_base.tolist(),
        "incremental_target_to_parent_normalized_matrix": incremental_target_to_parent.tolist(),
        "current_to_target_pixel_matrix": selected_matrix.tolist(),
        "composition_depth": composition_depth,
        "direction_interpretation": direction_interpretation,
        "corrections_path": str(corrections_path),
        "corrections_sha256": _sha256(corrections_path),
        "coverage": coverage,
        "fit": selected_report,
    }
    refined["metrics"] = {
        **alignment.get("metrics", {}),
        "review_correction_training_median_px": selected_report["training"]["median_px"],
        "review_correction_leave_one_out_median_px": selected_report["leave_one_out"]["median_px"],
        "review_correction_leave_one_out_p90_px": selected_report["leave_one_out"]["p90_px"],
        "review_correction_leave_one_out_max_px": selected_report["leave_one_out"]["max_px"],
    }
    refined["warning"] = (
        "This assisted refinement incorporates author-drawn displacement vectors. "
        "It remains diagnostic until the regenerated full-resolution output is reviewed."
    )
    report = {
        "schema_version": 1,
        "status": "diagnostic_only",
        "alignment": str(alignment_path),
        "corrections": str(corrections_path),
        "direction_interpretation": direction_interpretation,
        "point_count": len(current),
        "coverage": coverage,
        "selection_policy": {
            "model_order": list(MODEL_ORDER),
            "max_leave_one_out_p90_px": max_leave_one_out_p90_px,
            "max_leave_one_out_max_px": max_leave_one_out_max_px,
        },
        "selected_model": selected_model,
        "candidates": candidates,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "alignment.json").write_text(json.dumps(refined, indent=2) + "\n")
    (output_dir / "correction-fit.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def _wendland_c2(distances: np.ndarray, radius: float) -> np.ndarray:
    radial = distances / radius
    one_minus = np.maximum(1.0 - radial, 0.0)
    return np.power(one_minus, 4) * (4.0 * radial + 1.0)


def _local_displacement(
    points: np.ndarray,
    centers: np.ndarray,
    coefficients: np.ndarray,
    radius_px: float,
) -> np.ndarray:
    output = np.zeros((len(points), 2), dtype=np.float64)
    for center, coefficient in zip(centers, coefficients):
        distances = np.linalg.norm(points - center, axis=1)
        output += _wendland_c2(distances, radius_px)[:, None] * coefficient
    return output


def fit_local_review_corrections(
    alignment_path: Path,
    corrections_path: Path,
    output_dir: Path,
    radius_px: float = 500.0,
) -> Dict[str, object]:
    """Fit an exact compact correction that decays to zero outside local neighborhoods."""

    if radius_px <= 0:
        raise ValueError("Local correction radius must be positive")
    alignment = json.loads(alignment_path.read_text())
    correction_record = json.loads(corrections_path.read_text())
    grid = correction_record["grid"]
    if grid.get("crs") != "EPSG:3857":
        raise ValueError("Review corrections must use an EPSG:3857 grid")
    current, target, direction_interpretation = _correction_points(correction_record)
    if len(current) < 3:
        raise ValueError("At least three local review corrections are required")
    # A desired output location samples the parent at the current source location.
    sampling_delta = current - target
    distances = np.linalg.norm(target[:, None, :] - target[None, :, :], axis=2)
    kernel = _wendland_c2(distances, radius_px)
    condition = float(np.linalg.cond(kernel))
    if not np.isfinite(condition) or condition > 1e8:
        raise ValueError("Local correction controls are numerically ill-conditioned")
    coefficients = np.linalg.solve(kernel, sampling_delta)
    fitted = target + _local_displacement(target, target, coefficients, radius_px)
    residuals = np.linalg.norm(fitted - current, axis=1)

    width, height = int(grid["width"]), int(grid["height"])
    normalized_targets = target / np.asarray(
        [max(width - 1, 1), max(height - 1, 1)]
    )
    try:
        convex_hull_fraction = float(ConvexHull(normalized_targets).volume)
    except QhullError:
        convex_hull_fraction = 0.0
    sample_x = np.linspace(0, width - 1, 81)
    sample_y = np.linspace(0, height - 1, 81)
    mesh_x, mesh_y = np.meshgrid(sample_x, sample_y)
    samples = np.column_stack((mesh_x.ravel(), mesh_y.ravel()))
    mapped = samples + _local_displacement(samples, target, coefficients, radius_px)
    mapped_x = mapped[:, 0].reshape(mesh_x.shape)
    mapped_y = mapped[:, 1].reshape(mesh_y.shape)
    derivative_x_x = np.gradient(mapped_x, sample_x, axis=1)
    derivative_x_y = np.gradient(mapped_x, sample_y, axis=0)
    derivative_y_x = np.gradient(mapped_y, sample_x, axis=1)
    derivative_y_y = np.gradient(mapped_y, sample_y, axis=0)
    jacobian = derivative_x_x * derivative_y_y - derivative_x_y * derivative_y_x
    if float(np.min(jacobian)) <= 0:
        raise ValueError("Local correction introduces a foldover")

    local_operation = {
        "type": "compact_wendland_c2_displacement",
        "grid": grid,
        "radius_px": float(radius_px),
        "centers_pixel": target.tolist(),
        "coefficients_pixel": coefficients.tolist(),
        "sampling_delta_at_controls_px": sampling_delta.tolist(),
    }
    parent_correction = alignment.get("web_mercator_correction")
    if isinstance(parent_correction, dict):
        parent_grid = parent_correction["grid"]
        if (
            int(parent_grid["width"]) != width
            or int(parent_grid["height"]) != height
            or not np.allclose(parent_grid["bounds"], grid["bounds"])
        ):
            raise ValueError("Parent and local correction grids do not match")
        parent_operations = parent_correction.get("operations")
        if not isinstance(parent_operations, list):
            parent_operations = [
                {
                    "type": "matrix",
                    "matrix": parent_correction["target_to_current_normalized_matrix"],
                }
            ]
        composition_depth = int(parent_correction.get("composition_depth", 1)) + 1
    else:
        parent_operations = []
        composition_depth = 1

    refined = json.loads(json.dumps(alignment))
    refined["schema_version"] = max(int(refined.get("schema_version", 1)), 2)
    parent_transform_model = alignment.get("transform_model")
    if not isinstance(parent_transform_model, str):
        best = alignment.get("best")
        parent_transform_model = (
            best.get("transform_model") if isinstance(best, dict) else None
        )
    if not isinstance(parent_transform_model, str):
        raise ValueError("Parent alignment has no transform model")
    refined["transform_model"] = (
        f"{parent_transform_model}+web_mercator_compact_local_review_correction"
    )
    refined["parent_alignment"] = {
        "path": str(alignment_path),
        "sha256": _sha256(alignment_path),
    }
    refined["web_mercator_correction"] = {
        "model": "compact_wendland_c2",
        "grid": grid,
        "operations": [local_operation, *parent_operations],
        "composition_depth": composition_depth,
        "direction_interpretation": direction_interpretation,
        "corrections_path": str(corrections_path),
        "corrections_sha256": _sha256(corrections_path),
        "local_fit": {
            "control_count": len(current),
            "radius_px": float(radius_px),
            "control_convex_hull_fraction": convex_hull_fraction,
            "kernel_condition": condition,
            "control_residuals": _residual_summary(residuals),
            "sampled_jacobian_min": float(np.min(jacobian)),
            "sampled_jacobian_max": float(np.max(jacobian)),
        },
        # Retained for older readers; operations are authoritative.
        "target_to_current_normalized_matrix": (
            parent_correction["target_to_current_normalized_matrix"]
            if isinstance(parent_correction, dict)
            else np.eye(3).tolist()
        ),
    }
    refined["metrics"] = {
        **alignment.get("metrics", {}),
        "local_correction_control_count": len(current),
        "local_correction_radius_px": float(radius_px),
        "local_correction_control_hull_fraction": convex_hull_fraction,
        "local_correction_control_max_px": float(np.max(residuals)),
        "local_correction_jacobian_min": float(np.min(jacobian)),
        "local_correction_jacobian_max": float(np.max(jacobian)),
    }
    refined["warning"] = (
        "This assisted local refinement uses compact-support author corrections that "
        "decay exactly to the parent alignment outside the recorded neighborhoods."
    )
    report = {
        "schema_version": 1,
        "status": "diagnostic_only",
        "alignment": str(alignment_path),
        "corrections": str(corrections_path),
        "direction_interpretation": direction_interpretation,
        "model": "compact_wendland_c2",
        "control_count": len(current),
        "radius_px": float(radius_px),
        "control_convex_hull_fraction": convex_hull_fraction,
        "kernel_condition": condition,
        "control_residuals": _residual_summary(residuals),
        "sampled_jacobian_min": float(np.min(jacobian)),
        "sampled_jacobian_max": float(np.max(jacobian)),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "alignment.json").write_text(json.dumps(refined, indent=2) + "\n")
    (output_dir / "local-correction-fit.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report
