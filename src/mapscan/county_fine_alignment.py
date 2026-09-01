"""Fine registration against the user-supplied high-resolution county raster.

This stage deliberately ignores coastline, ocean, terrain, thematic color
boundaries, text, and generic Canny edges while fitting.  It uses only
multi-arm interior county-line junctions from the registered ``county.png``
mask and narrow dark-stroke evidence in the warped source.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

from .auto_refinement import (
    _alignment_transform,
    _line_patch_segments,
    _native_grid,
    _perimeter_segments,
    _registered_reference_masks,
    _residual_summary,
    _warp_evidence,
    _zero_shift_boundary_metric,
)
from .reference import load_california
from .refinement import _evaluate_model, fit_review_corrections


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _odd(value: float, minimum: int = 3) -> int:
    result = max(minimum, round(value))
    return result if result % 2 else result + 1


def _administrative_stroke_response(
    rgb: np.ndarray, valid: np.ndarray
) -> np.ndarray:
    """Emphasize narrow dark cartographic strokes instead of arbitrary edges."""

    lightness = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[:, :, 0]
    scale = rgb.shape[0] / 2400.0
    responses = []
    for nominal_size in (5, 9, 15, 23):
        size = _odd(nominal_size * scale)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        responses.append(
            cv2.morphologyEx(lightness, cv2.MORPH_BLACKHAT, kernel).astype(
                np.float32
            )
        )
    response = np.maximum.reduce(responses)
    blur_size = _odd(3 * scale)
    response = cv2.GaussianBlur(response, (blur_size, blur_size), 0)
    response[~valid] = 0.0
    return response


def _junction_template(
    segment: Dict[str, object], radius: int, scale: float
) -> np.ndarray:
    center = np.asarray(segment["center"], dtype=np.float64)
    template = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
    points = np.rint(
        np.asarray(segment["points"], dtype=np.float64)
        - (center - np.asarray([radius, radius]))
    ).astype(np.int32)
    inside = (
        (points[:, 0] >= 0)
        & (points[:, 0] < template.shape[1])
        & (points[:, 1] >= 0)
        & (points[:, 1] < template.shape[0])
    )
    template[points[inside, 1], points[inside, 0]] = 255
    blur = _odd(5 * scale)
    return cv2.GaussianBlur(template.astype(np.float32), (blur, blur), 0)


def _template_matches(
    response: np.ndarray,
    valid: np.ndarray,
    county_mask: np.ndarray,
    *,
    maximum_count: int = 56,
) -> tuple[list[Dict[str, object]], list[Dict[str, object]]]:
    """Match distinctive county junction topology to dark source strokes."""

    scale = response.shape[0] / 2400.0
    radius = max(24, round(44 * scale))
    search = max(18, round(32 * scale))
    segments = _line_patch_segments(
        county_mask,
        valid,
        maximum_count=maximum_count,
        patch_radius_px=radius,
        id_prefix="county_png",
        family="county_png_junction",
    )
    matches: list[Dict[str, object]] = []
    for segment in segments:
        center = np.rint(np.asarray(segment["center"])).astype(np.int32)
        x1 = int(center[0] - radius - search)
        x2 = int(center[0] + radius + search + 1)
        y1 = int(center[1] - radius - search)
        y2 = int(center[1] + radius + search + 1)
        if x1 < 0 or y1 < 0 or x2 > response.shape[1] or y2 > response.shape[0]:
            continue
        template = _junction_template(segment, radius, scale)
        scores = cv2.matchTemplate(
            response[y1:y2, x1:x2], template, cv2.TM_CCOEFF_NORMED
        )
        _, peak, _, location = cv2.minMaxLoc(scores)
        separated = scores.copy()
        separation = max(5, round(7 * scale))
        cv2.circle(separated, location, separation, -1.0, -1)
        second = float(np.max(separated))
        shift = np.asarray(
            [location[0] - search, location[1] - search], dtype=np.float64
        )
        margin = float(peak - second)
        accepted = bool(
            peak >= 0.27
            and margin >= 0.035
            and abs(shift[0]) < search - max(2, round(2 * scale))
            and abs(shift[1]) < search - max(2, round(2 * scale))
        )
        match = {
            "id": segment["id"],
            "family": segment["family"],
            "reference_pixel": np.asarray(segment["center"]).tolist(),
            "source_pixel": (np.asarray(segment["center"]) + shift).tolist(),
            "shift_px": shift.tolist(),
            "shift_magnitude_px": float(np.linalg.norm(shift)),
            "peak_correlation": float(peak),
            "second_correlation": second,
            "uniqueness_margin": margin,
            "accepted_by_local_evidence": accepted,
            "accepted_by_global_consistency": False,
        }
        matches.append(match)

    accepted = [item for item in matches if item["accepted_by_local_evidence"]]
    if len(accepted) < 12:
        return matches, []
    current = np.asarray([item["source_pixel"] for item in accepted], dtype=np.float64)
    target = np.asarray([item["reference_pixel"] for item in accepted], dtype=np.float64)
    ransac_threshold = max(1.75, 2.5 * scale)
    _, inliers = cv2.estimateAffine2D(
        current,
        target,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_threshold,
        maxIters=10000,
        confidence=0.999,
        refineIters=50,
    )
    if inliers is None:
        return matches, []
    consistent = []
    for item, is_inlier in zip(accepted, inliers.ravel() > 0):
        item["accepted_by_global_consistency"] = bool(is_inlier)
        if is_inlier:
            consistent.append(item)
    return matches, consistent


def _model_reports(matches: list[Dict[str, object]]) -> list[Dict[str, object]]:
    current = np.asarray([item["source_pixel"] for item in matches], dtype=np.float64)
    target = np.asarray([item["reference_pixel"] for item in matches], dtype=np.float64)
    reports = []
    for model in ("translation", "similarity", "affine", "projective"):
        _, report = _evaluate_model(model, current, target)
        reports.append(report)
    return reports


def _fit_matches_for_coverage(
    matches: list[Dict[str, object]], coverage_model: str
) -> list[Dict[str, object]]:
    """Return the evidence subset that is permitted to determine the transform.

    Partial-state maps expose fewer independent geographic neighborhoods and are
    more vulnerable to an otherwise consistent relief or thematic edge. Their
    fit therefore uses the same stricter local-evidence floor as the independent
    visible-quadrant audit. Full-state maps retain the broader RANSAC-consistent
    set because their distributed perimeter veto supplies additional coverage.
    """

    if coverage_model != "partial_state":
        return list(matches)
    return [
        item
        for item in matches
        if float(item["peak_correlation"]) >= 0.30
        and float(item["uniqueness_margin"]) >= 0.08
    ]


def _project_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack((points, np.ones(len(points)))) @ matrix.T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def _partial_fold_gate(
    after_summary: Dict[str, float],
    improved_fraction: float,
    relative_p90_improvement: float,
    scale: float,
) -> Dict[str, bool]:
    """Allow one bounded outlier without hiding a region-wide mismatch."""

    strict = bool(
        after_summary["p90_px"] <= 2.1 * scale
        and after_summary["max_px"] <= 4.0 * scale
        and improved_fraction >= 0.75
        and relative_p90_improvement >= 0.20
    )
    isolated_outlier = bool(
        not strict
        and after_summary["median_px"] <= 1.5 * scale
        and after_summary["p90_px"] <= 3.0 * scale
        and after_summary["max_px"] <= 5.0 * scale
        and improved_fraction >= 0.90
        and relative_p90_improvement >= 0.50
    )
    return {
        "strict_passed": strict,
        "isolated_outlier_passed": isolated_outlier,
        "passed": strict or isolated_outlier,
    }


def _spatial_holdout_audit(
    matches: list[Dict[str, object]],
    working_shape: Tuple[int, int],
    *,
    coverage_model: str = "full_or_most_state",
) -> Dict[str, object]:
    """Cross-validate on geographic bands excluded from model selection.

    Local template matching is independent for every junction.  This audit then
    withholds each fixed north-to-south macroregion from RANSAC and fitting, so
    a match cannot both determine and validate the transform in its fold.
    """

    locally_accepted = [
        item for item in matches if item["accepted_by_local_evidence"]
    ]
    if coverage_model == "partial_state":
        locally_accepted = [
            item
            for item in locally_accepted
            if float(item["peak_correlation"]) >= 0.30
            and float(item["uniqueness_margin"]) >= 0.08
        ]
        if len(locally_accepted) < 20:
            return {
                "method": "four_visible_extent_quadrants_excluded_from_training",
                "coverage_model": coverage_model,
                "locally_accepted_match_count": len(locally_accepted),
                "folds": [],
                "passed": False,
                "reason": "insufficient_high_confidence_partial_state_evidence",
            }
        current = np.asarray(
            [item["source_pixel"] for item in locally_accepted], dtype=np.float64
        )
        target = np.asarray(
            [item["reference_pixel"] for item in locally_accepted], dtype=np.float64
        )
        middle_x = float(np.median(target[:, 0]))
        middle_y = float(np.median(target[:, 1]))
        quadrant = (target[:, 0] >= middle_x).astype(np.uint8) + 2 * (
            target[:, 1] >= middle_y
        ).astype(np.uint8)
        folds = []
        aggregate_before = []
        aggregate_after = []
        scale = working_shape[0] / 2400.0
        for index, name in enumerate(
            ("northwest", "northeast", "southwest", "southeast")
        ):
            validation = np.flatnonzero(quadrant == index)
            training = np.flatnonzero(quadrant != index)
            fold: Dict[str, object] = {
                "region": name,
                "training_count": int(len(training)),
                "validation_count": int(len(validation)),
                "validation_ids": [
                    locally_accepted[item]["id"] for item in validation
                ],
            }
            if len(validation) < 4 or len(training) < 12:
                fold["passed"] = False
                fold["reason"] = "insufficient_spatially_separated_evidence"
                folds.append(fold)
                continue
            cv2.setRNGSeed(0)
            matrix, inliers = cv2.findHomography(
                current[training],
                target[training],
                cv2.RANSAC,
                2.5 * scale,
                maxIters=10000,
                confidence=0.999,
            )
            if (
                matrix is None
                or inliers is None
                or int(np.count_nonzero(inliers)) < 12
            ):
                fold["passed"] = False
                fold["reason"] = "training_model_not_robust"
                folds.append(fold)
                continue
            before = np.linalg.norm(current[validation] - target[validation], axis=1)
            predicted = _project_points(matrix, current[validation])
            after = np.linalg.norm(predicted - target[validation], axis=1)
            before_summary = _residual_summary(before)
            after_summary = _residual_summary(after)
            improved_fraction = float(np.mean(after < before))
            relative_p90_improvement = float(
                (before_summary["p90_px"] - after_summary["p90_px"])
                / max(before_summary["p90_px"], 1e-6)
            )
            gate = _partial_fold_gate(
                after_summary,
                improved_fraction,
                relative_p90_improvement,
                scale,
            )
            fold.update(
                {
                    "training_ransac_inlier_count": int(np.count_nonzero(inliers)),
                    "before": before_summary,
                    "after": after_summary,
                    "improved_fraction": improved_fraction,
                    "relative_p90_improvement": relative_p90_improvement,
                    **gate,
                }
            )
            folds.append(fold)
            aggregate_before.extend(before.tolist())
            aggregate_after.extend(after.tolist())
        target_span = np.ptp(target, axis=0)
        strict_pass_count = sum(bool(item.get("strict_passed")) for item in folds)
        return {
            "method": "four_visible_extent_quadrants_excluded_from_training",
            "coverage_model": coverage_model,
            "local_evidence_floor": {
                "peak_correlation": 0.30,
                "uniqueness_margin": 0.08,
            },
            "locally_accepted_match_count": len(locally_accepted),
            "visible_extent_midpoint": [middle_x, middle_y],
            "visible_extent_span_px": target_span.tolist(),
            "folds": folds,
            "strict_pass_count": strict_pass_count,
            "isolated_outlier_pass_count": sum(
                bool(item.get("isolated_outlier_passed")) for item in folds
            ),
            "aggregate_before": _residual_summary(
                np.asarray(aggregate_before, dtype=np.float64)
            ),
            "aggregate_after": _residual_summary(
                np.asarray(aggregate_after, dtype=np.float64)
            ),
            "passed": bool(
                len(folds) == 4
                and strict_pass_count >= 3
                and all(item.get("passed") for item in folds)
            ),
        }
    height, _ = working_shape
    current = np.asarray(
        [item["source_pixel"] for item in locally_accepted], dtype=np.float64
    )
    target = np.asarray(
        [item["reference_pixel"] for item in locally_accepted], dtype=np.float64
    )
    normalized_y = target[:, 1] / max(height - 1, 1)
    bands = (
        ("north", 0.00, 0.25),
        ("north_central", 0.25, 0.45),
        ("south_central", 0.45, 0.65),
        ("south", 0.65, 1.01),
    )
    folds = []
    aggregate_before = []
    aggregate_after = []
    for name, lower, upper in bands:
        validation = np.flatnonzero(
            (normalized_y >= lower) & (normalized_y < upper)
        )
        training = np.flatnonzero(
            ~((normalized_y >= lower) & (normalized_y < upper))
        )
        fold: Dict[str, object] = {
            "region": name,
            "normalized_y_range": [lower, min(upper, 1.0)],
            "training_count": int(len(training)),
            "validation_count": int(len(validation)),
            "validation_ids": [locally_accepted[index]["id"] for index in validation],
        }
        if len(validation) < 4 or len(training) < 12:
            fold["passed"] = False
            fold["reason"] = "insufficient_spatially_separated_evidence"
            folds.append(fold)
            continue
        cv2.setRNGSeed(0)
        matrix, inliers = cv2.findHomography(
            current[training],
            target[training],
            cv2.RANSAC,
            2.5 * (height / 2400.0),
            maxIters=10000,
            confidence=0.999,
        )
        if matrix is None or inliers is None or int(np.count_nonzero(inliers)) < 12:
            fold["passed"] = False
            fold["reason"] = "training_model_not_robust"
            folds.append(fold)
            continue
        before = np.linalg.norm(current[validation] - target[validation], axis=1)
        predicted = _project_points(matrix, current[validation])
        after = np.linalg.norm(predicted - target[validation], axis=1)
        before_summary = _residual_summary(before)
        after_summary = _residual_summary(after)
        improved_fraction = float(np.mean(after < before))
        relative_p90_improvement = float(
            (before_summary["p90_px"] - after_summary["p90_px"])
            / max(before_summary["p90_px"], 1e-6)
        )
        passed = bool(
            after_summary["p90_px"] <= 2.1 * (height / 2400.0)
            and after_summary["max_px"] <= 4.0 * (height / 2400.0)
            and improved_fraction >= 0.75
            and relative_p90_improvement >= 0.20
        )
        fold.update(
            {
                "training_ransac_inlier_count": int(np.count_nonzero(inliers)),
                "before": before_summary,
                "after": after_summary,
                "improved_fraction": improved_fraction,
                "relative_p90_improvement": relative_p90_improvement,
                "passed": passed,
            }
        )
        folds.append(fold)
        aggregate_before.extend(before.tolist())
        aggregate_after.extend(after.tolist())
    return {
        "method": "four_fixed_geographic_bands_excluded_from_training",
        "coverage_model": coverage_model,
        "locally_accepted_match_count": len(locally_accepted),
        "folds": folds,
        "aggregate_before": _residual_summary(
            np.asarray(aggregate_before, dtype=np.float64)
        ),
        "aggregate_after": _residual_summary(
            np.asarray(aggregate_after, dtype=np.float64)
        ),
        "passed": bool(len(folds) == 4 and all(item.get("passed") for item in folds)),
    }


def _transform_regularity_audit(
    matrix: np.ndarray, native_grid: Dict[str, object]
) -> Dict[str, object]:
    """Check the fine correction for foldovers and unstable local scaling."""

    width = int(native_grid["width"])
    height = int(native_grid["height"])
    determinants = []
    minimum_scales = []
    maximum_scales = []
    displacements = []
    a, b, c, d, e, f, g, h, i = matrix.ravel()
    for y in np.linspace(0, height - 1, 31):
        for x in np.linspace(0, width - 1, 31):
            denominator = g * x + h * y + i
            numerator_x = a * x + b * y + c
            numerator_y = d * x + e * y + f
            jacobian = np.asarray(
                [
                    [
                        (a * denominator - g * numerator_x) / denominator**2,
                        (b * denominator - h * numerator_x) / denominator**2,
                    ],
                    [
                        (d * denominator - g * numerator_y) / denominator**2,
                        (e * denominator - h * numerator_y) / denominator**2,
                    ],
                ]
            )
            singular_values = np.linalg.svd(jacobian, compute_uv=False)
            determinants.append(float(np.linalg.det(jacobian)))
            minimum_scales.append(float(np.min(singular_values)))
            maximum_scales.append(float(np.max(singular_values)))
            projected = _project_points(matrix, np.asarray([[x, y]]))[0]
            displacements.append(float(np.linalg.norm(projected - [x, y])))
    report = {
        "sample_grid": [31, 31],
        "jacobian_determinant": {
            "minimum": min(determinants),
            "maximum": max(determinants),
        },
        "local_scale": {
            "minimum": min(minimum_scales),
            "maximum": max(maximum_scales),
        },
        "full_grid_displacement_px": _residual_summary(
            np.asarray(displacements, dtype=np.float64)
        ),
    }
    report["passed"] = bool(
        report["jacobian_determinant"]["minimum"] > 0.95
        and report["jacobian_determinant"]["maximum"] < 1.05
        and report["local_scale"]["minimum"] > 0.97
        and report["local_scale"]["maximum"] < 1.04
    )
    return report


def _native_correction_record(
    matches: list[Dict[str, object]],
    working_shape: Tuple[int, int],
    native_grid: Dict[str, object],
) -> Dict[str, object]:
    working_height, working_width = working_shape
    scale = np.asarray(
        [
            max(int(native_grid["width"]) - 1, 1) / max(working_width - 1, 1),
            max(int(native_grid["height"]) - 1, 1) / max(working_height - 1, 1),
        ]
    )
    return {
        "schema_version": 1,
        "direction": "current_warped_source_to_authoritative_target",
        "evidence": "county_png_interior_multi_arm_junctions_only",
        "grid": native_grid,
        "corrections": [
            {
                "id": item["id"],
                "current": {
                    "pixel": {
                        "x": float(item["source_pixel"][0] * scale[0]),
                        "y": float(item["source_pixel"][1] * scale[1]),
                    }
                },
                "target": {
                    "pixel": {
                        "x": float(item["reference_pixel"][0] * scale[0]),
                        "y": float(item["reference_pixel"][1] * scale[1]),
                    }
                },
                "working_grid_evidence": {
                    "peak_correlation": item["peak_correlation"],
                    "uniqueness_margin": item["uniqueness_margin"],
                    "shift_px": item["shift_px"],
                },
            }
            for item in matches
        ],
    }


def _diagnostic(
    warped: np.ndarray,
    state_mask: np.ndarray,
    county_mask: np.ndarray,
    matches: list[Dict[str, object]],
) -> np.ndarray:
    output = warped.copy()
    state_line = cv2.dilate(state_mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    county_line = cv2.dilate(county_mask.astype(np.uint8), np.ones((2, 2), np.uint8)) > 0
    output[state_line] = np.asarray([0, 225, 255], dtype=np.uint8)
    output[county_line] = np.asarray([255, 0, 225], dtype=np.uint8)
    for item in matches:
        reference = tuple(np.rint(item["reference_pixel"]).astype(int))
        source = tuple(np.rint(item["source_pixel"]).astype(int))
        color = (40, 225, 40) if item["accepted_by_global_consistency"] else (255, 170, 0)
        cv2.arrowedLine(output, reference, source, color, 2, cv2.LINE_AA, tipLength=0.25)
        cv2.circle(output, reference, 4, (255, 0, 225), 1, cv2.LINE_AA)
    return output


def _transparent_overlay(
    mask: np.ndarray, color: Sequence[int]
) -> np.ndarray:
    output = np.zeros((*mask.shape, 4), dtype=np.uint8)
    output[mask, :3] = np.asarray(color, dtype=np.uint8)
    output[mask, 3] = 255
    return output


def fine_align_to_county_reference(
    image_path: Path,
    alignment_path: Path,
    county_reference_path: Path,
    reference_root: Path,
    output_dir: Path,
    *,
    working_height: int = 2400,
) -> Dict[str, object]:
    """Fit a robust projective fine correction using only county.png junctions."""

    output_dir.mkdir(parents=True, exist_ok=True)
    rgb = np.asarray(Image.open(image_path).convert("RGB"))
    state, _ = load_california(reference_root)
    parent_alignment = json.loads(alignment_path.read_text())
    county_reference = json.loads(county_reference_path.read_text())
    if county_reference.get("status") != "pass":
        raise ValueError("county.png reference has not passed registration QA")
    source_path = Path(str(county_reference["source"]["path"])).resolve()
    if _sha256(source_path) != county_reference["source"]["sha256"]:
        raise ValueError("county.png source hash does not match its registration")

    parent_transform = _alignment_transform(parent_alignment, state)
    before_warped, before_valid, grid, before_evidence = _warp_evidence(
        rgb, state, parent_transform, working_height
    )
    state_mask, county_mask = _registered_reference_masks(
        county_reference,
        county_reference_path,
        grid["bounds"],
        before_warped.shape[:2],
    )
    before_response = _administrative_stroke_response(before_warped, before_valid)
    before_matches, before_consistent = _template_matches(
        before_response, before_valid, county_mask
    )
    if len(before_consistent) < 12:
        raise ValueError(
            "Fewer than 12 distributed county.png junctions passed robust matching"
        )
    coverage_model = str(
        parent_alignment.get("best", {}).get("coverage_model", "full_or_most_state")
    )
    holdout_audit = _spatial_holdout_audit(
        before_matches,
        before_warped.shape[:2],
        coverage_model=coverage_model,
    )
    if not holdout_audit["passed"]:
        raise ValueError(
            "The county.png correction failed independent geographic holdouts"
        )
    fit_matches = _fit_matches_for_coverage(before_consistent, coverage_model)
    if len(fit_matches) < 12:
        raise ValueError(
            "Fewer than 12 coverage-appropriate county.png junctions may enter the fit"
        )
    before_models = _model_reports(fit_matches)
    native_grid = _native_grid(parent_transform, state, rgb.shape[:2])
    correction_record = _native_correction_record(
        fit_matches, before_warped.shape[:2], native_grid
    )
    corrections_path = output_dir / "automatic-county-corrections.json"
    corrections_path.write_text(json.dumps(correction_record, indent=2) + "\n")

    native_scale = float(native_grid["height"]) / float(before_warped.shape[0])
    fit_report = fit_review_corrections(
        alignment_path,
        corrections_path,
        output_dir,
        max_leave_one_out_p90_px=1.25 * native_scale,
        max_leave_one_out_max_px=1.6 * native_scale,
        minimum_axis_coverage=0.30,
        minimum_hull_coverage=0.06,
    )
    if fit_report["selected_model"] not in {
        "translation",
        "similarity",
        "affine",
        "projective",
    }:
        raise ValueError("The fine-registration evidence did not justify a safe model")
    selected_fit = next(
        item
        for item in fit_report["candidates"]
        if item["model"] == fit_report["selected_model"]
    )
    fine_matrix = np.asarray(
        selected_fit["matrix_current_to_target_pixels"], dtype=np.float64
    )
    regularity_audit = _transform_regularity_audit(fine_matrix, native_grid)
    if not regularity_audit["passed"]:
        raise ValueError("The county.png correction failed transform regularity QA")
    candidate_alignment_path = output_dir / "alignment.json"
    candidate_alignment = json.loads(candidate_alignment_path.read_text())
    candidate_alignment["automatic_fine_alignment"] = {
        "method": "county_png_junction_template_matching_least_flexible_model",
        "selected_model": fit_report["selected_model"],
        "fit_evidence": "interior_county_junctions_only",
        "excluded_evidence": [
            "coastline_and_ocean",
            "terrain_and_hillshade",
            "thematic_color_boundaries",
            "text_and_city_symbols",
            "generic_nearest_canny_edges",
        ],
        "county_reference": {
            "path": str(county_reference_path),
            "sha256": _sha256(county_reference_path),
            "source_path": str(source_path),
            "source_sha256": county_reference["source"]["sha256"],
        },
        "corrections": {
            "path": str(corrections_path),
            "sha256": _sha256(corrections_path),
        },
    }
    candidate_alignment["warning"] = (
        "Automatic county.png-primary fine registration. Coastline, ocean, terrain, "
        "thematic bands, labels, and generic edges were excluded from fitting. "
        "The candidate still requires author visual approval."
    )
    candidate_alignment_path.write_text(json.dumps(candidate_alignment, indent=2) + "\n")

    candidate_transform = _alignment_transform(candidate_alignment, state)
    after_warped, after_valid, after_grid, after_evidence = _warp_evidence(
        rgb, state, candidate_transform, working_height
    )
    if after_grid != grid:
        raise AssertionError("Fine-registration candidate changed the working grid")
    after_state_mask, after_county_mask = _registered_reference_masks(
        county_reference,
        county_reference_path,
        after_grid["bounds"],
        after_warped.shape[:2],
    )
    after_response = _administrative_stroke_response(after_warped, after_valid)
    after_matches, after_consistent = _template_matches(
        after_response, after_valid, after_county_mask
    )
    if len(after_consistent) < 12:
        raise ValueError("The corrected warp lost county.png junction support")
    after_models = _model_reports(after_consistent)
    before_shift = np.asarray(
        [item["shift_magnitude_px"] for item in before_consistent], dtype=np.float64
    )
    after_shift = np.asarray(
        [item["shift_magnitude_px"] for item in after_consistent], dtype=np.float64
    )
    before_summary = _residual_summary(before_shift)
    after_summary = _residual_summary(after_shift)
    fixed_point = bool(
        after_summary["median_px"] <= 1.0
        and after_summary["p90_px"] <= 2.0
        and after_summary["p90_px"] < before_summary["p90_px"]
    )
    perimeter_before = _perimeter_segments(
        state, grid["bounds"], before_warped.shape[:2], 32
    )
    perimeter_after = _perimeter_segments(
        state, after_grid["bounds"], after_warped.shape[:2], 32
    )
    _, before_distance, before_nearest, before_gradient_x, before_gradient_y = (
        before_evidence
    )
    _, after_distance, after_nearest, after_gradient_x, after_gradient_y = (
        after_evidence
    )
    before_state_boundary = _zero_shift_boundary_metric(
        perimeter_before,
        before_valid,
        before_distance,
        before_nearest,
        before_gradient_x,
        before_gradient_y,
    )
    after_state_boundary = _zero_shift_boundary_metric(
        perimeter_after,
        after_valid,
        after_distance,
        after_nearest,
        after_gradient_x,
        after_gradient_y,
    )
    minimum_state_boundary_segments = 6 if coverage_model == "partial_state" else 24
    state_boundary_veto = bool(
        after_state_boundary["count"] >= minimum_state_boundary_segments
        and after_state_boundary["median_px"]
        <= before_state_boundary["median_px"] + 0.5
        and after_state_boundary["p90_px"]
        <= before_state_boundary["p90_px"] + 0.5
        and after_state_boundary["max_px"]
        <= before_state_boundary["max_px"] + 1.0
    )
    if not state_boundary_veto:
        raise ValueError(
            "The county.png correction worsened the independent state-boundary veto"
        )

    Image.fromarray(before_warped, mode="RGB").save(
        output_dir / "working-source-before.jpg", quality=94, subsampling=0
    )
    Image.fromarray(after_warped, mode="RGB").save(
        output_dir / "working-source-after.jpg", quality=94, subsampling=0
    )
    Image.fromarray(
        _diagnostic(before_warped, state_mask, county_mask, before_matches), mode="RGB"
    ).save(output_dir / "working-diagnostic-before.jpg", quality=94, subsampling=0)
    Image.fromarray(
        _diagnostic(after_warped, after_state_mask, after_county_mask, after_matches),
        mode="RGB",
    ).save(output_dir / "working-diagnostic-after.jpg", quality=94, subsampling=0)
    response_scale = max(float(np.quantile(before_response[before_valid], 0.99)), 1.0)
    Image.fromarray(
        np.clip(before_response / response_scale * 255, 0, 255).astype(np.uint8),
        mode="L",
    ).save(output_dir / "administrative-stroke-response.png", optimize=True)

    full_before, _, full_grid, _ = _warp_evidence(
        rgb, state, parent_transform, int(native_grid["height"])
    )
    full_after, _, full_after_grid, _ = _warp_evidence(
        rgb, state, candidate_transform, int(native_grid["height"])
    )
    if full_grid != full_after_grid:
        raise AssertionError("Fine-registration candidate changed the native grid")
    full_state, full_county = _registered_reference_masks(
        county_reference,
        county_reference_path,
        full_grid["bounds"],
        full_after.shape[:2],
    )
    Image.fromarray(full_before, mode="RGB").save(
        output_dir / "web-mercator-source-before.jpg", quality=95, subsampling=0
    )
    Image.fromarray(full_after, mode="RGB").save(
        output_dir / "web-mercator-source-after.jpg", quality=95, subsampling=0
    )
    Image.fromarray(
        _transparent_overlay(full_state, (0, 225, 255)), mode="RGBA"
    ).save(output_dir / "web-mercator-county-png-state-overlay.png", optimize=True)
    Image.fromarray(
        _transparent_overlay(full_county, (255, 0, 225)), mode="RGBA"
    ).save(output_dir / "web-mercator-county-png-county-overlay.png", optimize=True)

    artifact_names = [
        "alignment.json",
        "correction-fit.json",
        "automatic-county-corrections.json",
        "working-source-before.jpg",
        "working-source-after.jpg",
        "working-diagnostic-before.jpg",
        "working-diagnostic-after.jpg",
        "administrative-stroke-response.png",
        "web-mercator-source-before.jpg",
        "web-mercator-source-after.jpg",
        "web-mercator-county-png-state-overlay.png",
        "web-mercator-county-png-county-overlay.png",
    ]
    report = {
        "schema_version": 1,
        "status": "needs_author_review" if fixed_point else "needs_iteration",
        "method": "county_png_primary_automatic_fine_registration",
        "source": {"path": str(image_path), "sha256": _sha256(image_path)},
        "parent_alignment": {
            "path": str(alignment_path),
            "sha256": _sha256(alignment_path),
        },
        "candidate_alignment": {
            "path": str(candidate_alignment_path),
            "sha256": _sha256(candidate_alignment_path),
        },
        "county_reference": candidate_alignment["automatic_fine_alignment"][
            "county_reference"
        ],
        "evidence_policy": candidate_alignment["automatic_fine_alignment"],
        "working_grid": grid,
        "native_grid": native_grid,
        "before": {
            "candidate_match_count": len(before_matches),
            "locally_accepted_count": int(
                sum(item["accepted_by_local_evidence"] for item in before_matches)
            ),
            "globally_consistent_count": len(before_consistent),
            "fit_eligible_count": len(fit_matches),
            "fit_eligible_ids": [item["id"] for item in fit_matches],
            "shift_residual": before_summary,
            "models": before_models,
            "matches": before_matches,
        },
        "fit": fit_report,
        "independent_spatial_holdouts": holdout_audit,
        "transform_regularity": regularity_audit,
        "state_boundary_veto": {
            "role": "veto_only_not_fit_evidence",
            "before": before_state_boundary,
            "after": after_state_boundary,
            "minimum_visible_segment_count": minimum_state_boundary_segments,
            "passed": state_boundary_veto,
        },
        "after": {
            "candidate_match_count": len(after_matches),
            "locally_accepted_count": int(
                sum(item["accepted_by_local_evidence"] for item in after_matches)
            ),
            "globally_consistent_count": len(after_consistent),
            "shift_residual": after_summary,
            "models": after_models,
            "matches": after_matches,
        },
        "fixed_point_gate": {
            "passed": fixed_point,
            "maximum_median_px": 1.0,
            "maximum_p90_px": 2.0,
            "requires_p90_improvement": True,
        },
        "artifacts": {
            name: {"path": name, "sha256": _sha256(output_dir / name)}
            for name in artifact_names
        },
        "publication_allowed": False,
        "warning": (
            "This is an automatic alignment candidate, not an approved extraction. "
            "Todd must visually confirm the county.png state and county registration."
        ),
    }
    report_path = output_dir / "county-fine-alignment.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report
