"""Automatic, Census-anchored fine registration for the lower Colorado edge.

The parent county fit remains authoritative for the interior county network.
This stage corrects only the residual corroborated by three independent image
channels along the lower Colorado River.  The Mexico border is a no-change
validation region because ``county.png`` is measurably displaced there while
the source already agrees with the Census state boundary.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Iterable, Sequence

import cv2
import numpy as np
from PIL import Image

from .auto_refinement import (
    _alignment_transform,
    _native_grid,
    _registered_reference_masks,
    _residual_summary,
    _warp_evidence,
)
from .county_fine_alignment import _transparent_overlay
from .extraction import _target_state_mask
from .reference import load_california


BASE_WIDTH = 3398
BASE_HEIGHT = 3920
LOWER_COLORADO_SPANS = tuple((start, start + 50) for start in range(3450, 3800, 50))
UPPER_EAST_SPANS = tuple((start, start + 50) for start in range(2500, 3050, 50))
SOUTH_BORDER_SPANS = tuple((start, start + 50) for start in range(2550, 3200, 50))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _smoothstep(value: np.ndarray | float) -> np.ndarray | float:
    clipped = np.clip(value, 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def _image_evidence(rgb: np.ndarray) -> Dict[str, np.ndarray]:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lightness = lab[:, :, 0].astype(np.uint8)
    chroma = np.hypot(lab[:, :, 1] - 128.0, lab[:, :, 2] - 128.0)
    saturation = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[:, :, 1].astype(np.float32)
    blackhat = np.maximum.reduce(
        [
            cv2.morphologyEx(
                lightness,
                cv2.MORPH_BLACKHAT,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)),
            ).astype(np.float32)
            for size in (7, 13)
        ]
    )
    return {"blackhat": blackhat, "saturation": saturation, "chroma": chroma}


def _scaled_spans(
    spans: Iterable[tuple[int, int]], scale: float
) -> list[tuple[int, int]]:
    return [(round(start * scale), round(end * scale)) for start, end in spans]


def _best_shift(
    evidence: Dict[str, np.ndarray],
    samples: Sequence[tuple[int, int]],
    *,
    axis: str,
    maximum_shift_px: int,
) -> tuple[Dict[str, int], Dict[str, float]]:
    scores: Dict[str, list[tuple[int, float]]] = {
        "blackhat": [],
        "saturation": [],
        "chroma": [],
    }
    height, width = evidence["blackhat"].shape
    for shift in range(-maximum_shift_px, maximum_shift_px + 1):
        blackhat_values = []
        saturation_values = []
        chroma_values = []
        for first, second in samples:
            x, y = (second, first) if axis == "vertical" else (first, second)
            if axis == "vertical":
                x += shift
                if not (5 <= x < width - 5 and 0 <= y < height):
                    continue
                inside = (slice(y, y + 1), slice(x - 4, x - 1))
                outside = (slice(y, y + 1), slice(x + 2, x + 5))
            else:
                y += shift
                if not (5 <= y < height - 5 and 0 <= x < width):
                    continue
                inside = (slice(y - 4, y - 1), slice(x, x + 1))
                outside = (slice(y + 2, y + 5), slice(x, x + 1))
            blackhat_values.append(float(evidence["blackhat"][y, x]))
            saturation_values.append(
                float(np.mean(evidence["saturation"][inside]))
                - float(np.mean(evidence["saturation"][outside]))
            )
            chroma_values.append(
                float(np.mean(evidence["chroma"][inside]))
                - float(np.mean(evidence["chroma"][outside]))
            )
        if not blackhat_values:
            continue
        scores["blackhat"].append((shift, float(np.median(blackhat_values))))
        scores["saturation"].append((shift, float(np.median(saturation_values))))
        scores["chroma"].append((shift, float(np.median(chroma_values))))
    shifts = {}
    strengths = {}
    for key, candidates in scores.items():
        shift, strength = max(candidates, key=lambda item: item[1])
        shifts[key] = int(shift)
        strengths[key] = float(strength)
    return shifts, strengths


def _vertical_profiles(
    rgb: np.ndarray,
    state_mask: np.ndarray,
    spans: Iterable[tuple[int, int]],
    *,
    maximum_shift_px: int = 12,
) -> list[Dict[str, object]]:
    evidence = _image_evidence(rgb)
    profiles = []
    for index, (start, end) in enumerate(spans):
        samples = []
        for y in range(max(0, start), min(state_mask.shape[0], end)):
            xs = np.flatnonzero(state_mask[y])
            if len(xs) and xs[-1] > 0.80 * state_mask.shape[1]:
                samples.append((y, int(xs[-1])))
        if len(samples) < max(20, round(0.60 * (end - start))):
            continue
        shifts, strengths = _best_shift(
            evidence,
            samples,
            axis="vertical",
            maximum_shift_px=maximum_shift_px,
        )
        values = np.asarray(list(shifts.values()), dtype=np.float64)
        accepted = bool(
            np.ptp(values) <= 3
            and strengths["blackhat"] >= 25.0
            and strengths["saturation"] >= 20.0
            and strengths["chroma"] >= 8.0
            and np.max(np.abs(values)) < maximum_shift_px
        )
        profiles.append(
            {
                "id": f"east_{index:02d}",
                "span": [int(start), int(end)],
                "center": [
                    float(np.median([item[1] for item in samples])),
                    float(np.median([item[0] for item in samples])),
                ],
                "estimator_shifts_px": shifts,
                "estimator_strengths": strengths,
                "consensus_shift_px": float(np.median(values)),
                "estimator_range_px": float(np.ptp(values)),
                "accepted": accepted,
            }
        )
    return profiles


def _horizontal_profiles(
    rgb: np.ndarray,
    state_mask: np.ndarray,
    spans: Iterable[tuple[int, int]],
    *,
    maximum_shift_px: int = 12,
) -> list[Dict[str, object]]:
    evidence = _image_evidence(rgb)
    profiles = []
    for index, (start, end) in enumerate(spans):
        samples = []
        for x in range(max(0, start), min(state_mask.shape[1], end)):
            ys = np.flatnonzero(state_mask[:, x])
            if len(ys) and ys[-1] > 0.80 * state_mask.shape[0]:
                samples.append((x, int(ys[-1])))
        if len(samples) < max(20, round(0.60 * (end - start))):
            continue
        shifts, strengths = _best_shift(
            evidence,
            samples,
            axis="horizontal",
            maximum_shift_px=maximum_shift_px,
        )
        values = np.asarray(list(shifts.values()), dtype=np.float64)
        profiles.append(
            {
                "id": f"south_{index:02d}",
                "span": [int(start), int(end)],
                "center": [
                    float(np.median([item[0] for item in samples])),
                    float(np.median([item[1] for item in samples])),
                ],
                "estimator_shifts_px": shifts,
                "estimator_strengths": strengths,
                "consensus_shift_px": float(np.median(values)),
                "estimator_range_px": float(np.ptp(values)),
                "accepted": bool(np.ptp(values) <= 3),
            }
        )
    return profiles


def _profile_summary(profiles: Sequence[Dict[str, object]]) -> Dict[str, object]:
    accepted = [item for item in profiles if item["accepted"]]
    consensus = np.asarray(
        [float(item["consensus_shift_px"]) for item in accepted],
        dtype=np.float64,
    )
    result: Dict[str, object] = {
        "window_count": len(profiles),
        "accepted_count": len(accepted),
        "estimator_agreement_within_3_fraction": float(
            np.mean([float(item["estimator_range_px"]) <= 3.0 for item in profiles])
        )
        if profiles
        else 0.0,
        "signed_median_px": float(np.median(consensus)) if len(consensus) else None,
        "residual": _residual_summary(np.abs(consensus)),
        "profiles": list(profiles),
    }
    for key in ("blackhat", "saturation", "chroma"):
        values = np.asarray(
            [float(item["estimator_shifts_px"][key]) for item in accepted],
            dtype=np.float64,
        )
        result[key] = {
            "signed_median_px": float(np.median(values)) if len(values) else None,
            "residual": _residual_summary(np.abs(values)),
        }
    return result


def _operation_weight(
    x: float,
    y: float,
    parameters: Dict[str, float],
) -> float:
    return float(
        _smoothstep(
            (x - parameters["start_x_px"]) / parameters["ramp_width_px"]
        )
        * _smoothstep(
            (y - parameters["start_y_px"]) / parameters["ramp_height_px"]
        )
    )


def _infer_amplitude(
    profiles: Sequence[Dict[str, object]],
    fit_indexes: set[int],
    parameters: Dict[str, float],
) -> Dict[str, object]:
    candidates = []
    used = []
    for index, item in enumerate(profiles):
        if index not in fit_indexes or not item["accepted"]:
            continue
        x, y = (float(value) for value in item["center"])
        weight = _operation_weight(x, y, parameters)
        if weight < 0.40:
            continue
        for key, shift in item["estimator_shifts_px"].items():
            amplitude = -float(shift) / weight
            candidates.append(amplitude)
            used.append(
                {
                    "window": item["id"],
                    "estimator": key,
                    "observed_shift_px": shift,
                    "operation_weight": weight,
                    "amplitude_candidate_px": amplitude,
                }
            )
    values = np.asarray(candidates, dtype=np.float64)
    if len(values) < 9:
        raise ValueError("Insufficient corroborated lower-Colorado fit evidence")
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    amplitude = int(math.floor(median + 0.5))
    return {
        "candidate_count": len(values),
        "median_amplitude_px": median,
        "amplitude_mad_px": mad,
        "selected_source_to_target_amplitude_x_px": amplitude,
        "evidence": used,
    }


def _operation_parameters(grid: Dict[str, object]) -> Dict[str, float]:
    x_scale = (int(grid["width"]) - 1) / (BASE_WIDTH - 1)
    y_scale = (int(grid["height"]) - 1) / (BASE_HEIGHT - 1)
    return {
        "start_x_px": 2600.0 * x_scale,
        "ramp_width_px": 600.0 * x_scale,
        "start_y_px": 3250.0 * y_scale,
        "ramp_height_px": 350.0 * y_scale,
    }


def _append_operation(
    parent_alignment: Dict[str, object],
    parent_alignment_path: Path,
    grid: Dict[str, object],
    parameters: Dict[str, float],
    amplitude: int,
) -> Dict[str, object]:
    candidate = json.loads(json.dumps(parent_alignment))
    parent_correction = parent_alignment["web_mercator_correction"]
    parent_operations = parent_correction.get("operations")
    if not isinstance(parent_operations, list):
        parent_operations = [
            {
                "type": "matrix",
                "matrix": parent_correction["target_to_current_normalized_matrix"],
            }
        ]
    operation = {
        "type": "lower_colorado_smoothstep_x",
        "grid": grid,
        **parameters,
        "source_to_target_amplitude_x_px": float(amplitude),
        "target_to_parent_sampling_amplitude_x_px": float(-amplitude),
        "fit_target": "Census_2025_state_boundary_lower_Colorado",
    }
    candidate["schema_version"] = max(int(candidate.get("schema_version", 1)), 2)
    candidate["transform_model"] = (
        f"{parent_alignment['transform_model']}+automatic_lower_colorado_scalar_bump"
    )
    candidate["parent_alignment"] = {
        "path": str(parent_alignment_path),
        "sha256": _sha256(parent_alignment_path),
    }
    candidate["web_mercator_correction"] = {
        **parent_correction,
        "model": "county_projective_plus_lower_colorado_smoothstep_x",
        "operations": [operation, *parent_operations],
        "composition_depth": int(parent_correction.get("composition_depth", 1)) + 1,
    }
    candidate["automatic_lower_colorado_refinement"] = {
        "method": "three_channel_Census_anchored_scalar_smoothstep_x",
        "operation": operation,
        "reference_priority": {
            "state_boundary": "Census_2025_authoritative",
            "interior_counties": "registered_county_png_supplemental",
            "conflict_policy": "never_move_a_Census_aligned_state_edge_to_county_png",
        },
    }
    candidate["warning"] = (
        "Automatic lower-Colorado candidate only. The Mexico border was held fixed "
        "as a validation region because it already matches Census while county.png "
        "is regionally displaced there. Author review is required."
    )
    return candidate


def _sampled_regularity(
    grid: Dict[str, object], parameters: Dict[str, float], amplitude: float
) -> Dict[str, object]:
    x = np.linspace(0.0, int(grid["width"]) - 1, 121)
    y = np.linspace(0.0, int(grid["height"]) - 1, 121)
    mesh_x, mesh_y = np.meshgrid(x, y)
    weights = _smoothstep(
        (mesh_x - parameters["start_x_px"]) / parameters["ramp_width_px"]
    ) * _smoothstep(
        (mesh_y - parameters["start_y_px"]) / parameters["ramp_height_px"]
    )
    mapped_x = mesh_x - amplitude * weights
    mapped_y = mesh_y
    dx_dx = np.gradient(mapped_x, x, axis=1)
    dx_dy = np.gradient(mapped_x, y, axis=0)
    dy_dx = np.gradient(mapped_y, x, axis=1)
    dy_dy = np.gradient(mapped_y, y, axis=0)
    determinant = dx_dx * dy_dy - dx_dy * dy_dx
    return {
        "sampling_jacobian_min": float(np.min(determinant)),
        "sampling_jacobian_max": float(np.max(determinant)),
        "maximum_shear": float(np.max(np.abs(dx_dy))),
        "maximum_displacement_px": float(np.max(np.abs(amplitude * weights))),
        "passed": bool(
            np.min(determinant) >= 0.98
            and np.max(determinant) <= 1.02
            and np.max(np.abs(dx_dy)) <= 0.03
            and abs(amplitude) <= 8.0
        ),
    }


def _state_border(mask: np.ndarray) -> np.ndarray:
    eroded = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    return mask & ~eroded


def _diagnostic(
    rgb: np.ndarray,
    authoritative_state: np.ndarray,
    county_state: np.ndarray,
    county: np.ndarray,
    profiles: Sequence[Dict[str, object]],
) -> np.ndarray:
    result = rgb.astype(np.float32)
    for mask, color, opacity in (
        (county, (255, 0, 225), 0.72),
        (county_state, (0, 225, 255), 0.78),
        (authoritative_state, (255, 220, 0), 0.94),
    ):
        result[mask] = (1.0 - opacity) * result[mask] + opacity * np.asarray(color)
    result = np.clip(result, 0, 255).astype(np.uint8)
    for index, profile in enumerate(profiles):
        x, y = (round(float(value)) for value in profile["center"])
        color = (40, 225, 90) if index % 2 == 0 else (40, 150, 255)
        cv2.circle(result, (x, y), 12, color, 3, cv2.LINE_AA)
        cv2.putText(
            result,
            "fit" if index % 2 == 0 else "holdout",
            (x - 45, y - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return result


def _run_once(
    fine_run: Path,
    image_path: Path,
    county_reference_path: Path,
    reference_root: Path,
    output_dir: Path,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    parent_report_path = fine_run / "county-fine-alignment.json"
    parent_report = json.loads(parent_report_path.read_text())
    if parent_report.get("status") != "needs_author_review":
        raise ValueError("Parent county fine alignment has not passed automatic QA")
    parent_alignment_path = fine_run / "alignment.json"
    if _sha256(parent_alignment_path) != parent_report["candidate_alignment"]["sha256"]:
        raise ValueError("Parent county fine alignment hash does not match")
    county_reference = json.loads(county_reference_path.read_text())
    if county_reference.get("status") != "pass":
        raise ValueError("county.png reference has not passed registration QA")

    rgb = np.asarray(Image.open(image_path).convert("RGB"))
    state, _ = load_california(reference_root)
    parent_alignment = json.loads(parent_alignment_path.read_text())
    parent_transform = _alignment_transform(parent_alignment, state)
    grid = _native_grid(parent_transform, state, rgb.shape[:2])
    before, before_valid, rendered_grid, _ = _warp_evidence(
        rgb, state, parent_transform, int(grid["height"])
    )
    if rendered_grid["bounds"] != grid["bounds"]:
        raise AssertionError("Parent review grid changed")
    state_mask = _target_state_mask(state, grid["bounds"], before.shape[:2])
    authoritative_border = _state_border(state_mask)
    county_state, county_mask = _registered_reference_masks(
        county_reference, county_reference_path, grid["bounds"], before.shape[:2]
    )

    y_scale = (int(grid["height"]) - 1) / (BASE_HEIGHT - 1)
    x_scale = (int(grid["width"]) - 1) / (BASE_WIDTH - 1)
    lower_spans = _scaled_spans(LOWER_COLORADO_SPANS, y_scale)
    upper_spans = _scaled_spans(UPPER_EAST_SPANS, y_scale)
    south_spans = _scaled_spans(SOUTH_BORDER_SPANS, x_scale)
    before_lower = _vertical_profiles(before, state_mask, lower_spans)
    if sum(bool(item["accepted"]) for item in before_lower) < 6:
        raise ValueError("Lower-Colorado evidence did not agree in at least six windows")
    parameters = _operation_parameters(grid)
    fit_indexes = {0, 2, 4, 6}
    amplitude_fit = _infer_amplitude(before_lower, fit_indexes, parameters)
    amplitude = int(amplitude_fit["selected_source_to_target_amplitude_x_px"])
    if not (
        2 <= amplitude <= 8
        and float(amplitude_fit["amplitude_mad_px"]) <= 1.5
    ):
        raise ValueError("Lower-Colorado scalar displacement is unstable or unsafe")

    candidate_alignment = _append_operation(
        parent_alignment,
        parent_alignment_path,
        grid,
        parameters,
        amplitude,
    )
    candidate_alignment_path = output_dir / "alignment.json"
    candidate_alignment_path.write_text(json.dumps(candidate_alignment, indent=2) + "\n")
    candidate_transform = _alignment_transform(candidate_alignment, state)
    after, after_valid, after_grid, _ = _warp_evidence(
        rgb, state, candidate_transform, int(grid["height"])
    )
    if after_grid != rendered_grid:
        raise AssertionError("Lower-Colorado refinement changed the review grid")

    after_lower = _vertical_profiles(after, state_mask, lower_spans)
    before_upper = _vertical_profiles(before, state_mask, upper_spans)
    after_upper = _vertical_profiles(after, state_mask, upper_spans)
    before_south = _horizontal_profiles(before, state_mask, south_spans)
    after_south = _horizontal_profiles(after, state_mask, south_spans)
    fit_before = _profile_summary(
        [item for index, item in enumerate(before_lower) if index in fit_indexes]
    )
    fit_after = _profile_summary(
        [item for index, item in enumerate(after_lower) if index in fit_indexes]
    )
    holdout_before = _profile_summary(
        [item for index, item in enumerate(before_lower) if index not in fit_indexes]
    )
    holdout_after = _profile_summary(
        [item for index, item in enumerate(after_lower) if index not in fit_indexes]
    )
    lower_all_after = _profile_summary(after_lower)
    fixed_point_fit = _infer_amplitude(after_lower, fit_indexes, parameters)
    fixed_point_amplitude = int(
        fixed_point_fit["selected_source_to_target_amplitude_x_px"]
    )
    edge_gate = bool(
        holdout_after["accepted_count"] == holdout_before["accepted_count"]
        and holdout_after["accepted_count"] >= 3
        and holdout_after["residual"]["median_px"] <= 1.0
        and holdout_after["residual"]["p90_px"] <= 2.0
        and holdout_after["residual"]["max_px"] <= 3.0
        and lower_all_after["accepted_count"] >= 6
    )
    fixed_point_gate = bool(abs(fixed_point_amplitude) <= 1)
    if not edge_gate or not fixed_point_gate:
        raise ValueError("Lower-Colorado candidate failed held-out or fixed-point QA")

    before_upper_summary = _profile_summary(before_upper)
    after_upper_summary = _profile_summary(after_upper)
    before_south_summary = _profile_summary(before_south)
    after_south_summary = _profile_summary(after_south)
    unchanged_regions_pass = bool(
        after_upper_summary["residual"]["p90_px"]
        <= before_upper_summary["residual"]["p90_px"] + 0.5
        and after_south_summary["residual"]["p90_px"]
        <= before_south_summary["residual"]["p90_px"] + 0.5
    )
    if not unchanged_regions_pass:
        raise ValueError("Lower-Colorado correction degraded an authoritative no-change region")

    native_per_working_x = (int(grid["width"]) - 1) / (
        int(parent_report["working_grid"]["width"]) - 1
    )
    accepted_county_x = [
        float(item["reference_pixel"][0]) * native_per_working_x
        for item in parent_report["after"]["matches"]
        if item["accepted_by_global_consistency"]
    ]
    anchor_max_x = max(accepted_county_x)
    unaffected_prefix = max(0, int(math.floor(parameters["start_x_px"])) - 2)
    unchanged_pixels = bool(
        np.array_equal(before[:, :unaffected_prefix], after[:, :unaffected_prefix])
        and np.array_equal(
            before_valid[:, :unaffected_prefix], after_valid[:, :unaffected_prefix]
        )
    )
    county_veto = bool(
        anchor_max_x < parameters["start_x_px"] and unchanged_pixels
    )
    if not county_veto:
        raise ValueError("Lower-Colorado correction touched accepted county controls")

    regularity = _sampled_regularity(grid, parameters, amplitude)
    if not regularity["passed"]:
        raise ValueError("Lower-Colorado correction failed transform regularity QA")

    Image.fromarray(before, mode="RGB").save(
        output_dir / "web-mercator-source-before.jpg", quality=95, subsampling=0
    )
    Image.fromarray(after, mode="RGB").save(
        output_dir / "web-mercator-source-after.jpg", quality=95, subsampling=0
    )
    Image.fromarray(
        _transparent_overlay(authoritative_border, (255, 220, 0)), mode="RGBA"
    ).save(output_dir / "web-mercator-census-state-overlay.png", optimize=True)
    Image.fromarray(
        _transparent_overlay(county_state, (0, 225, 255)), mode="RGBA"
    ).save(output_dir / "web-mercator-county-png-state-overlay.png", optimize=True)
    Image.fromarray(
        _transparent_overlay(county_mask, (255, 0, 225)), mode="RGBA"
    ).save(output_dir / "web-mercator-county-png-county-overlay.png", optimize=True)
    Image.fromarray(
        _diagnostic(before, authoritative_border, county_state, county_mask, before_lower),
        mode="RGB",
    ).save(output_dir / "lower-colorado-diagnostic-before.jpg", quality=95, subsampling=0)
    Image.fromarray(
        _diagnostic(after, authoritative_border, county_state, county_mask, after_lower),
        mode="RGB",
    ).save(output_dir / "lower-colorado-diagnostic-after.jpg", quality=95, subsampling=0)

    artifacts = [
        "alignment.json",
        "web-mercator-source-before.jpg",
        "web-mercator-source-after.jpg",
        "web-mercator-census-state-overlay.png",
        "web-mercator-county-png-state-overlay.png",
        "web-mercator-county-png-county-overlay.png",
        "lower-colorado-diagnostic-before.jpg",
        "lower-colorado-diagnostic-after.jpg",
    ]
    reference_manifest = reference_root / "manifest.json"
    report = {
        "schema_version": 1,
        "status": "needs_author_review",
        "method": "automatic_Census_anchored_lower_Colorado_scalar_refinement",
        "source": {"path": str(image_path), "sha256": _sha256(image_path)},
        "parent_fine_alignment": {
            "path": str(parent_report_path),
            "sha256": _sha256(parent_report_path),
        },
        "candidate_alignment": {
            "path": str(candidate_alignment_path),
            "sha256": _sha256(candidate_alignment_path),
        },
        "authoritative_state_reference": {
            "kind": "Census_2025_state_boundary",
            "manifest": str(reference_manifest),
            "manifest_sha256": _sha256(reference_manifest),
        },
        "county_reference": parent_report["county_reference"],
        "reference_conflict": {
            "decision": "Census state boundary overrides county.png state outline",
            "county_png_role": "interior_county_network_and_supplemental_state_diagnostic",
            "mexico_border_source_vs_Census_p90_px": before_south_summary["residual"]["p90_px"],
            "correction_applied_to_mexico_border_normal": False,
        },
        "grid": grid,
        "working_grid": grid,
        "fit_evidence": {
            "partition": "alternating_precommitted_50px_windows_0_2_4_6",
            "before": fit_before,
            "after": fit_after,
            "amplitude_fit": amplitude_fit,
        },
        "independent_edge_holdouts": {
            "partition": "alternating_precommitted_50px_windows_1_3_5",
            "before": holdout_before,
            "after": holdout_after,
            "passed": edge_gate,
        },
        "fixed_point_gate": {
            "second_pass_proposed_amplitude_px": fixed_point_amplitude,
            "passed": fixed_point_gate,
        },
        "unchanged_region_veto": {
            "upper_east": {
                "before": before_upper_summary,
                "after": after_upper_summary,
            },
            "mexico_border": {
                "before": before_south_summary,
                "after": after_south_summary,
            },
            "passed": unchanged_regions_pass,
        },
        "interior_county_veto": {
            "accepted_anchor_count": len(accepted_county_x),
            "maximum_accepted_anchor_x_native_px": anchor_max_x,
            "operation_zero_until_x_native_px": parameters["start_x_px"],
            "unaffected_prefix_byte_identical": unchanged_pixels,
            "passed": county_veto,
        },
        "transform_regularity": regularity,
        "evidence_policy": candidate_alignment["automatic_lower_colorado_refinement"],
        "artifacts": {
            name: {"path": name, "sha256": _sha256(output_dir / name)}
            for name in artifacts
        },
        "publication_allowed": False,
        "warning": "Automatic candidate only; Todd must visually review it.",
    }
    report_path = output_dir / "lower-colorado-refinement.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def refine_lower_colorado(
    fine_run: Path,
    image_path: Path,
    county_reference_path: Path,
    reference_root: Path,
    output_dir: Path,
) -> Dict[str, object]:
    """Build twice, require byte-identical outputs, and leave an unpublished candidate."""

    first = _run_once(
        fine_run, image_path, county_reference_path, reference_root, output_dir
    )
    first_hashes = {
        name: _sha256(output_dir / name)
        for name in [*first["artifacts"], "lower-colorado-refinement.json"]
    }
    second = _run_once(
        fine_run, image_path, county_reference_path, reference_root, output_dir
    )
    second_hashes = {
        name: _sha256(output_dir / name)
        for name in [*second["artifacts"], "lower-colorado-refinement.json"]
    }
    passed = first_hashes == second_hashes
    audit = {
        "schema_version": 1,
        "passed": passed,
        "method": "two_complete_same_path_rebuilds",
        "first": first_hashes,
        "second": second_hashes,
    }
    (output_dir / "determinism-audit.json").write_text(
        json.dumps(audit, indent=2) + "\n"
    )
    if not passed:
        raise ValueError("Lower-Colorado refinement is not byte deterministic")
    return second
