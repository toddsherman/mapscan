"""Automatic compact refinement for California's south and southeast edge.

The interior county fit is retained as the parent.  This stage adds only a
small compact-support correction where the source's saturated thematic fill
transitions to pale exterior page/background along the authoritative border.
It never searches generic edges or ocean shadows.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Sequence

import cv2
import numpy as np
from PIL import Image

from .auto_refinement import (
    _alignment_transform,
    _native_grid,
    _perimeter_segments,
    _registered_reference_masks,
    _residual_summary,
    _warp_evidence,
)
from .county_fine_alignment import (
    _administrative_stroke_response,
    _diagnostic,
    _template_matches,
    _transparent_overlay,
)
from .reference import load_california
from .refinement import fit_local_review_corrections


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _signed_land_background_match(
    rgb: np.ndarray,
    segment: Dict[str, object],
    *,
    maximum_normal_shift_px: int = 12,
    validate_halves: bool = True,
) -> Dict[str, object]:
    """Locate the real land-to-pale-background transition along one segment."""

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lightness = lab[:, :, 0]
    chroma = np.hypot(lab[:, :, 1] - 128.0, lab[:, :, 2] - 128.0)
    height, width = chroma.shape
    points = np.asarray(segment["points"], dtype=np.float64)
    normals = np.asarray(segment["normals"], dtype=np.float64)
    candidates = []
    for step in range(-maximum_normal_shift_px, maximum_normal_shift_px + 1):
        base = points + normals * step
        samples = []
        valid = np.ones(len(points), dtype=bool)
        for offset in (2, 4):
            inside = np.rint(base + normals * offset).astype(np.int32)
            outside = np.rint(base - normals * offset).astype(np.int32)
            valid &= (
                (inside[:, 0] >= 0)
                & (inside[:, 0] < width)
                & (inside[:, 1] >= 0)
                & (inside[:, 1] < height)
                & (outside[:, 0] >= 0)
                & (outside[:, 0] < width)
                & (outside[:, 1] >= 0)
                & (outside[:, 1] < height)
            )
            samples.append((inside, outside))
        if np.count_nonzero(valid) < max(50, round(0.70 * len(points))):
            continue
        chroma_differences = []
        lightness_differences = []
        for inside, outside in samples:
            chroma_differences.append(
                chroma[inside[valid, 1], inside[valid, 0]]
                - chroma[outside[valid, 1], outside[valid, 0]]
            )
            lightness_differences.append(
                lightness[outside[valid, 1], outside[valid, 0]]
                - lightness[inside[valid, 1], inside[valid, 0]]
            )
        chroma_difference = np.median(np.stack(chroma_differences), axis=0)
        lightness_difference = np.median(
            np.stack(lightness_differences), axis=0
        )
        signed_transition = chroma_difference + 0.25 * lightness_difference
        score = float(np.quantile(signed_transition, 0.35))
        support = float(
            np.mean((chroma_difference > 8.0) & (lightness_difference > -10.0))
        )
        candidates.append(
            {
                "normal_shift_px": step,
                "score": score,
                "support_fraction": support,
                "median_chroma_difference": float(np.median(chroma_difference)),
                "median_lightness_difference": float(
                    np.median(lightness_difference)
                ),
            }
        )
    candidates.sort(key=lambda item: item["score"], reverse=True)
    best = candidates[0]
    separated = [
        item
        for item in candidates[1:]
        if abs(item["normal_shift_px"] - best["normal_shift_px"]) >= 3
    ]
    runner_up = separated[0] if separated else best
    uniqueness_margin = float(best["score"] - runner_up["score"])
    accepted = bool(
        best["score"] >= 15.0
        and best["support_fraction"] >= 0.50
        and uniqueness_margin >= 8.0
        and abs(best["normal_shift_px"]) < maximum_normal_shift_px
    )
    half_validation = []
    if validate_halves and len(points) >= 90:
        split = len(points) // 2
        for suffix, indices in (
            ("first", slice(0, split + 1)),
            ("second", slice(split, len(points))),
        ):
            half_segment = dict(segment)
            half_segment["id"] = f"{segment['id']}_{suffix}_half"
            half_segment["points"] = points[indices]
            half_segment["normals"] = normals[indices]
            half_match = _signed_land_background_match(
                rgb,
                half_segment,
                maximum_normal_shift_px=maximum_normal_shift_px,
                validate_halves=False,
            )
            half_validation.append(
                {
                    "half": suffix,
                    "normal_shift_px": half_match["normal_shift_px"],
                    "score": half_match["best"]["score"],
                    "support_fraction": half_match["best"]["support_fraction"],
                    "uniqueness_margin": half_match["uniqueness_margin"],
                    "accepted": half_match["accepted_by_local_evidence"],
                }
            )
        accepted = bool(
            accepted
            and all(item["accepted"] for item in half_validation)
            and max(
                abs(item["normal_shift_px"] - best["normal_shift_px"])
                for item in half_validation
            )
            <= 2
        )
    center = np.asarray(segment["center"], dtype=np.float64)
    normal = np.asarray(segment["normal"], dtype=np.float64)
    source = center + normal * best["normal_shift_px"]
    return {
        "id": segment["id"],
        "family": "signed_south_southeast_land_background_transition",
        "reference_pixel": center.tolist(),
        "source_pixel": source.tolist(),
        "normal": normal.tolist(),
        "normal_shift_px": int(best["normal_shift_px"]),
        "shift_magnitude_px": float(abs(best["normal_shift_px"])),
        "best": best,
        "runner_up_separated_by_3px": runner_up,
        "uniqueness_margin": uniqueness_margin,
        "independent_half_validation": half_validation,
        "accepted_by_local_evidence": accepted,
        "accepted_by_global_consistency": accepted,
    }


def _edge_segments(state, grid: Dict[str, object], shape: tuple[int, int]):
    segments = _perimeter_segments(state, grid["bounds"], shape, 48, 101)
    height, width = shape
    return [
        segment
        for segment in segments
        if (
            segment["center"][0] / width > 0.92
            and segment["center"][1] / height > 0.74
        )
        or (
            segment["center"][0] / width > 0.72
            and segment["center"][1] / height > 0.975
        )
        or (
            segment["center"][0] / width > 0.35
            and segment["center"][1] / height > 0.79
        )
    ]


def _correction_record(
    matches: Sequence[Dict[str, object]], grid: Dict[str, object]
) -> Dict[str, object]:
    return {
        "schema_version": 1,
        "direction": "current_warped_source_to_authoritative_target",
        "evidence": "signed_south_southeast_land_to_pale_background_transition",
        "grid": grid,
        "corrections": [
            {
                "id": item["id"],
                "current": {
                    "pixel": {
                        "x": item["source_pixel"][0],
                        "y": item["source_pixel"][1],
                    }
                },
                "target": {
                    "pixel": {
                        "x": item["reference_pixel"][0],
                        "y": item["reference_pixel"][1],
                    }
                },
                "automatic_evidence": {
                    "normal_shift_px": item["normal_shift_px"],
                    "transition_score": item["best"]["score"],
                    "support_fraction": item["best"]["support_fraction"],
                    "uniqueness_margin": item["uniqueness_margin"],
                },
            }
            for item in matches
        ],
    }


def _half_segment(
    segment: Dict[str, object], *, first: bool
) -> Dict[str, object]:
    points = np.asarray(segment["points"], dtype=np.float64)
    normals = np.asarray(segment["normals"], dtype=np.float64)
    split = len(points) // 2
    indices = slice(0, split + 1) if first else slice(split, len(points))
    half_points = points[indices]
    half_normals = normals[indices]
    normal = np.mean(half_normals, axis=0)
    normal /= max(float(np.linalg.norm(normal)), 1e-9)
    result = dict(segment)
    result["id"] = f"{segment['id']}_{'fit' if first else 'holdout'}"
    result["points"] = half_points
    result["normals"] = half_normals
    result["center"] = np.mean(half_points, axis=0)
    result["normal"] = normal
    return result


def _paired_segment_matches(
    rgb: np.ndarray, segments: Sequence[Dict[str, object]]
) -> tuple[list[Dict[str, object]], list[Dict[str, object]], list[Dict[str, object]]]:
    """Use one contiguous half to fit and the other as untouched evidence."""

    fit_matches = []
    holdout_matches = []
    pairs = []
    for segment in segments:
        fit = _signed_land_background_match(
            rgb, _half_segment(segment, first=True), validate_halves=False
        )
        holdout = _signed_land_background_match(
            rgb, _half_segment(segment, first=False), validate_halves=False
        )
        accepted = bool(
            fit["accepted_by_local_evidence"]
            and holdout["accepted_by_local_evidence"]
            and abs(fit["normal_shift_px"] - holdout["normal_shift_px"]) <= 2
        )
        pairs.append(
            {
                "segment_id": segment["id"],
                "accepted": accepted,
                "fit_shift_px": fit["normal_shift_px"],
                "holdout_shift_px": holdout["normal_shift_px"],
                "shift_difference_px": abs(
                    fit["normal_shift_px"] - holdout["normal_shift_px"]
                ),
                "fit_evidence_passed": fit["accepted_by_local_evidence"],
                "holdout_evidence_passed": holdout[
                    "accepted_by_local_evidence"
                ],
            }
        )
        if accepted:
            fit_matches.append(fit)
            holdout_matches.append(holdout)
    return fit_matches, holdout_matches, pairs


def _match_summary(
    matches: Sequence[Dict[str, object]],
    *,
    fixed_ids: set[str] | None = None,
) -> Dict[str, object]:
    accepted = [
        item
        for item in matches
        if (
            item["accepted_by_local_evidence"]
            if fixed_ids is None
            else item["id"] in fixed_ids and item["best"]["score"] >= 12.0
        )
    ]
    return {
        "candidate_count": len(matches),
        "accepted_count": len(accepted),
        "residual": _residual_summary(
            np.asarray(
                [item["shift_magnitude_px"] for item in accepted],
                dtype=np.float64,
            )
        ),
        "matches": list(matches),
    }


def refine_southern_edge(
    fine_run: Path,
    image_path: Path,
    county_reference_path: Path,
    reference_root: Path,
    output_dir: Path,
    *,
    radius_px: float = 520.0,
) -> Dict[str, object]:
    """Add a bounded automatic correction to south and southeast only."""

    output_dir.mkdir(parents=True, exist_ok=True)
    fine_report_path = fine_run / "county-fine-alignment.json"
    fine_report = json.loads(fine_report_path.read_text())
    if fine_report.get("status") != "needs_author_review":
        raise ValueError("Parent county fine alignment has not passed automatic QA")
    alignment_path = fine_run / "alignment.json"
    if _sha256(alignment_path) != fine_report["candidate_alignment"]["sha256"]:
        raise ValueError("Parent county fine alignment hash does not match")
    county_reference = json.loads(county_reference_path.read_text())
    if county_reference.get("status") != "pass":
        raise ValueError("county.png reference has not passed registration QA")

    rgb = np.asarray(Image.open(image_path).convert("RGB"))
    state, _ = load_california(reference_root)
    parent_alignment = json.loads(alignment_path.read_text())
    parent_transform = _alignment_transform(parent_alignment, state)
    native_grid = _native_grid(parent_transform, state, rgb.shape[:2])
    before, before_valid, grid, _ = _warp_evidence(
        rgb, state, parent_transform, int(native_grid["height"])
    )
    state_mask, county_mask = _registered_reference_masks(
        county_reference, county_reference_path, grid["bounds"], before.shape[:2]
    )
    segments = _edge_segments(state, grid, before.shape[:2])
    accepted, before_holdouts, evidence_pairs = _paired_segment_matches(
        before, segments
    )
    if len(accepted) < 6:
        raise ValueError("Fewer than six reliable south/southeast anchors were found")

    corrections_path = output_dir / "automatic-southern-edge-corrections.json"
    corrections_path.write_text(
        json.dumps(_correction_record(accepted, native_grid), indent=2) + "\n"
    )
    local_fit = fit_local_review_corrections(
        alignment_path,
        corrections_path,
        output_dir,
        radius_px=radius_px,
    )
    if not (
        local_fit["sampled_jacobian_min"] > 0.95
        and local_fit["sampled_jacobian_max"] < 1.05
    ):
        raise ValueError("Southern-edge correction failed Jacobian regularity QA")

    candidate_alignment_path = output_dir / "alignment.json"
    candidate_alignment = json.loads(candidate_alignment_path.read_text())
    candidate_alignment["automatic_southern_edge_refinement"] = {
        "method": "signed_land_to_pale_background_compact_wendland_c2",
        "parent_fine_run": {
            "path": str(fine_report_path),
            "sha256": _sha256(fine_report_path),
        },
        "fit_evidence": "south_and_southeast_border_transition_only",
        "excluded_evidence": [
            "generic_canny_edges",
            "ocean_drop_shadows",
            "islands_and_lakes",
            "terrain_and_hillshade",
            "hazard_band_boundaries",
            "labels_and_city_symbols",
        ],
        "corrections": {
            "path": str(corrections_path),
            "sha256": _sha256(corrections_path),
        },
    }
    candidate_alignment["warning"] = (
        "Automatic compact south/southeast edge refinement. The accepted county "
        "projective alignment remains the parent and the correction decays exactly "
        "to zero outside the southern neighborhoods. Author review is required."
    )
    candidate_alignment_path.write_text(
        json.dumps(candidate_alignment, indent=2) + "\n"
    )

    candidate_transform = _alignment_transform(candidate_alignment, state)
    after, after_valid, after_grid, _ = _warp_evidence(
        rgb, state, candidate_transform, int(native_grid["height"])
    )
    if after_grid != grid:
        raise AssertionError("Southern refinement changed the review grid")
    after_fit_matches = [
        _signed_land_background_match(
            after, _half_segment(segment, first=True), validate_halves=False
        )
        for segment in segments
        if f"{segment['id']}_fit" in {item["id"] for item in accepted}
    ]
    after_holdouts = [
        _signed_land_background_match(
            after, _half_segment(segment, first=False), validate_halves=False
        )
        for segment in segments
        if f"{segment['id']}_holdout" in {
            item["id"] for item in before_holdouts
        }
    ]
    fit_ids = {item["id"] for item in accepted}
    holdout_ids = {item["id"] for item in before_holdouts}
    before_summary = _match_summary(accepted, fixed_ids=fit_ids)
    after_summary = _match_summary(after_fit_matches, fixed_ids=fit_ids)
    before_holdout_summary = _match_summary(
        before_holdouts, fixed_ids=holdout_ids
    )
    after_holdout_summary = _match_summary(after_holdouts, fixed_ids=holdout_ids)
    edge_gate = bool(
        after_holdout_summary["accepted_count"] >= 6
        and after_holdout_summary["residual"]["median_px"] <= 1.0
        and after_holdout_summary["residual"]["p90_px"] <= 2.0
        and after_holdout_summary["residual"]["p90_px"]
        < before_holdout_summary["residual"]["p90_px"]
    )
    if not edge_gate:
        raise ValueError("Southern-edge correction did not reach a stable fixed point")

    working_height = int(fine_report["working_grid"]["height"])
    working_before, working_before_valid, working_grid, _ = _warp_evidence(
        rgb, state, parent_transform, working_height
    )
    working_after, working_after_valid, _, _ = _warp_evidence(
        rgb, state, candidate_transform, working_height
    )
    _, working_county = _registered_reference_masks(
        county_reference,
        county_reference_path,
        working_grid["bounds"],
        working_before.shape[:2],
    )
    before_response = _administrative_stroke_response(
        working_before, working_before_valid
    )
    after_response = _administrative_stroke_response(
        working_after, working_after_valid
    )
    before_county_matches, before_county_consistent = _template_matches(
        before_response, working_before_valid, working_county
    )
    after_county_matches, after_county_consistent = _template_matches(
        after_response, working_after_valid, working_county
    )
    before_county = _residual_summary(
        np.asarray(
            [item["shift_magnitude_px"] for item in before_county_consistent]
        )
    )
    after_county = _residual_summary(
        np.asarray(
            [item["shift_magnitude_px"] for item in after_county_consistent]
        )
    )
    county_veto = bool(
        len(after_county_consistent) >= 20
        and after_county["median_px"] <= before_county["median_px"] + 0.25
        and after_county["p90_px"] <= before_county["p90_px"] + 0.5
    )
    if not county_veto:
        raise ValueError("Southern refinement degraded the interior county fit")

    Image.fromarray(before, mode="RGB").save(
        output_dir / "web-mercator-source-before.jpg", quality=95, subsampling=0
    )
    Image.fromarray(after, mode="RGB").save(
        output_dir / "web-mercator-source-after.jpg", quality=95, subsampling=0
    )
    Image.fromarray(
        _transparent_overlay(state_mask, (0, 225, 255)), mode="RGBA"
    ).save(output_dir / "web-mercator-county-png-state-overlay.png", optimize=True)
    Image.fromarray(
        _transparent_overlay(county_mask, (255, 0, 225)), mode="RGBA"
    ).save(output_dir / "web-mercator-county-png-county-overlay.png", optimize=True)
    Image.fromarray(
        _diagnostic(before, state_mask, county_mask, accepted), mode="RGB"
    ).save(output_dir / "southern-edge-diagnostic-before.jpg", quality=95)
    Image.fromarray(
        _diagnostic(after, state_mask, county_mask, after_fit_matches), mode="RGB"
    ).save(output_dir / "southern-edge-diagnostic-after.jpg", quality=95)

    artifacts = [
        "alignment.json",
        "local-correction-fit.json",
        "automatic-southern-edge-corrections.json",
        "web-mercator-source-before.jpg",
        "web-mercator-source-after.jpg",
        "web-mercator-county-png-state-overlay.png",
        "web-mercator-county-png-county-overlay.png",
        "southern-edge-diagnostic-before.jpg",
        "southern-edge-diagnostic-after.jpg",
    ]
    report = {
        "schema_version": 1,
        "status": "needs_author_review",
        "method": "automatic_compact_south_southeast_edge_refinement",
        "source": {"path": str(image_path), "sha256": _sha256(image_path)},
        "parent_fine_alignment": {
            "path": str(fine_report_path),
            "sha256": _sha256(fine_report_path),
        },
        "candidate_alignment": {
            "path": str(candidate_alignment_path),
            "sha256": _sha256(candidate_alignment_path),
        },
        "county_reference": fine_report["county_reference"],
        "grid": native_grid,
        "before": before_summary,
        "after": after_summary,
        "independent_edge_holdouts": {
            "method": "second_contiguous_half_never_used_for_fit",
            "evidence_pairs": evidence_pairs,
            "before": before_holdout_summary,
            "after": after_holdout_summary,
            "passed": edge_gate,
        },
        "edge_fixed_point_gate": {
            "passed": edge_gate,
            "validation_source": "independent_edge_holdouts",
        },
        "interior_county_veto": {
            "before": before_county,
            "after": after_county,
            "passed": county_veto,
        },
        "local_fit": local_fit,
        "evidence_policy": candidate_alignment[
            "automatic_southern_edge_refinement"
        ],
        "artifacts": {
            name: {"path": name, "sha256": _sha256(output_dir / name)}
            for name in artifacts
        },
        "publication_allowed": False,
        "warning": "Automatic candidate only; Todd must review the southern edge.",
    }
    report_path = output_dir / "southern-edge-refinement.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report
