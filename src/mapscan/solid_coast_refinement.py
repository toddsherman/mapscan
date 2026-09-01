"""Apply a compact west-coast correction while preserving other components.

The stage is intentionally evidence-specific: it consumes the largest connected
component of exact solid thematic colors, compares only its western envelope to
the approved canonical mainland line, and adds fixed pins for every retained
offshore component.  The correction therefore decays before the eastern border
and is explicitly constrained to leave already-aligned islands in place.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Sequence

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

from .auto_refinement import _alignment_transform
from .extraction import warp_classified_to_web_mercator
from .reference import load_california
from .refinement import fit_local_review_corrections
from .solid_mask_alignment import _exact_palette_mask, _external_boundary


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _line_mask(path: Path) -> np.ndarray:
    rgba = np.asarray(Image.open(path).convert("RGBA"))
    return rgba[..., 3] > 0


def _component_records(mask: np.ndarray, maximum_components: int) -> tuple[np.ndarray, list[dict]]:
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    records = []
    for label in range(1, count):
        x, y, width, height, area = stats[label]
        records.append(
            {
                "label": int(label),
                "area": int(area),
                "bbox": [int(x), int(y), int(width), int(height)],
                "centroid": [float(value) for value in centroids[label]],
            }
        )
    records.sort(key=lambda item: item["area"], reverse=True)
    return labels, records[:maximum_components]


def _largest_external_boundary(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        raise ValueError("Thematic mainland mask has no external boundary")
    boundary = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(
        boundary,
        [max(contours, key=cv2.contourArea)],
        -1,
        1,
        1,
        cv2.LINE_8,
    )
    return boundary.astype(bool)


def _western_rows(
    source_boundary: np.ndarray,
    target_boundary: np.ndarray,
    *,
    start_y: int,
    end_y: int,
    maximum_x_fraction: float = 0.72,
) -> np.ndarray:
    rows = []
    maximum_x = source_boundary.shape[1] * maximum_x_fraction
    for y in range(max(0, start_y), min(source_boundary.shape[0], end_y)):
        source_x = np.flatnonzero(source_boundary[y])
        target_x = np.flatnonzero(target_boundary[y])
        if not len(source_x) or not len(target_x):
            continue
        source_west, target_west = int(source_x.min()), int(target_x.min())
        if source_west >= maximum_x or target_west >= maximum_x:
            continue
        rows.append((y, source_west, target_west, target_west - source_west))
    if not rows:
        raise ValueError("No shared western-coast rows were found")
    return np.asarray(rows, dtype=np.float64)


def _west_gap_summary(rows: np.ndarray) -> Dict[str, float]:
    """Measure only the visible white gap where source remains east of target."""

    gaps = np.maximum(-rows[:, 3], 0.0)
    return {
        "median_px": float(np.median(gaps)),
        "p90_px": float(np.quantile(gaps, 0.9)),
        "max_px": float(np.max(gaps)),
        "fraction_with_gap": float(np.mean(gaps > 0.0)),
        "sample_count": int(len(gaps)),
    }


def _eastern_row_drift(
    before: np.ndarray,
    after: np.ndarray,
    *,
    minimum_x_fraction: float = 0.4,
) -> Dict[str, float]:
    """Verify that a west-side stretch leaves the eastern envelope stationary."""

    minimum_x = before.shape[1] * minimum_x_fraction
    drift = []
    for y in range(before.shape[0]):
        before_x = np.flatnonzero(before[y])
        after_x = np.flatnonzero(after[y])
        if not len(before_x) or not len(after_x):
            continue
        before_east, after_east = int(before_x.max()), int(after_x.max())
        if before_east < minimum_x or after_east < minimum_x:
            continue
        drift.append(float(after_east - before_east))
    if not drift:
        raise ValueError("No shared eastern-border rows were found")
    values = np.asarray(drift, dtype=np.float64)
    absolute = np.abs(values)
    return {
        "median_px": float(np.median(values)),
        "p90_absolute_px": float(np.quantile(absolute, 0.9)),
        "max_absolute_px": float(np.max(absolute)),
        "unchanged_fraction": float(np.mean(values == 0.0)),
        "sample_count": int(len(values)),
    }


def _coast_controls(
    rows: np.ndarray,
    *,
    count: int,
    smoothing_half_window: int,
    minimum_left_shift_px: float,
    maximum_left_shift_px: float,
) -> list[dict]:
    if count < 3:
        raise ValueError("At least three coast controls are required")
    if minimum_left_shift_px <= 0 or maximum_left_shift_px < minimum_left_shift_px:
        raise ValueError("Left-shift limits are invalid")
    y_targets = np.linspace(rows[:, 0].min(), rows[:, 0].max(), count)
    controls = []
    used_y = set()
    for requested_y in y_targets:
        nearest_index = int(np.argmin(np.abs(rows[:, 0] - requested_y)))
        y, source_x = rows[nearest_index, :2]
        y_int = int(y)
        if y_int in used_y:
            continue
        used_y.add(y_int)
        local = rows[np.abs(rows[:, 0] - y) <= smoothing_half_window, 3]
        measured = float(np.median(local))
        desired = max(
            -maximum_left_shift_px,
            min(-minimum_left_shift_px, measured),
        )
        controls.append(
            {
                "id": f"west_coast_{len(controls):02d}",
                "current": [float(source_x), float(y)],
                "target": [float(source_x + desired), float(y)],
                "measured_target_minus_source_x_px": measured,
                "applied_target_minus_source_x_px": desired,
            }
        )
    return controls


def _centroid(mask: np.ndarray) -> np.ndarray:
    y, x = np.nonzero(mask)
    if not len(x):
        raise ValueError("Cannot measure an empty component")
    return np.asarray([float(np.mean(x)), float(np.mean(y))])


def _component_pin_points(mask: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Return a center and four boundary extrema to hold a component fixed."""

    y, x = np.nonzero(mask)
    if not len(x):
        raise ValueError("Cannot pin an empty component")
    points = np.column_stack((x, y)).astype(np.float64)
    center = np.asarray([float(np.mean(x)), float(np.mean(y))])

    def centered_extreme(axis: int, value: float) -> np.ndarray:
        candidates = points[points[:, axis] == value]
        other_axis = 1 - axis
        return candidates[
            int(np.argmin(np.abs(candidates[:, other_axis] - center[other_axis])))
        ]

    candidates = [
        ("center", center),
        ("west", centered_extreme(0, float(np.min(x)))),
        ("east", centered_extreme(0, float(np.max(x)))),
        ("north", centered_extreme(1, float(np.min(y)))),
        ("south", centered_extreme(1, float(np.max(y)))),
    ]
    result = []
    seen = set()
    for label, point in candidates:
        key = tuple(np.rint(point).astype(int))
        if key in seen:
            continue
        seen.add(key)
        result.append((label, point))
    return result


def _distance_summary(source: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    distance = distance_transform_edt(~target)
    values = distance[source]
    return {
        "median_px": float(np.median(values)),
        "p90_px": float(np.quantile(values, 0.9)),
        "mean_px": float(np.mean(values)),
        "within_3px_fraction": float(np.mean(values <= 3.0)),
        "sample_count": int(len(values)),
    }


def _render_diagnostic(
    target: np.ndarray, before: np.ndarray, after: np.ndarray
) -> np.ndarray:
    output = np.full((*target.shape, 3), 18, dtype=np.uint8)
    output[target] = (255, 50, 210)
    output[before] = (0, 225, 240)
    output[after] = (90, 255, 80)
    output[target & after] = (255, 255, 255)
    return output


def refine_solid_west_coast(
    image_path: Path,
    alignment_path: Path,
    reference_root: Path,
    canonical_boundary_manifest_path: Path,
    output_dir: Path,
    colors: Sequence[Sequence[int]],
    *,
    maximum_components: int = 5,
    coast_control_count: int = 13,
    radius_px: float = 360.0,
    minimum_left_shift_px: float = 4.0,
    maximum_left_shift_px: float = 20.0,
    smoothing_half_window: int = 45,
    start_y_fraction: float = 0.04,
    end_y_fraction: float = 0.95,
) -> Dict[str, object]:
    """Create a compact, island-pinned west-coast child alignment."""

    image_path = image_path.resolve()
    alignment_path = alignment_path.resolve()
    reference_root = reference_root.resolve()
    canonical_boundary_manifest_path = canonical_boundary_manifest_path.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rgb = np.asarray(Image.open(image_path).convert("RGB"))
    exact = _exact_palette_mask(rgb, colors)
    labels, components = _component_records(exact, maximum_components)
    if len(components) < 2:
        raise ValueError("A mainland and at least one offshore component are required")
    alignment = json.loads(alignment_path.read_text())
    state, _ = load_california(reference_root)
    parent_transform = _alignment_transform(alignment, state)
    canonical = json.loads(canonical_boundary_manifest_path.read_text())
    grid = canonical["source_grid"]
    target_shape = (int(grid["height"]), int(grid["width"]))
    mainland_path = canonical_boundary_manifest_path.parent / str(
        canonical["artifacts"]["mainland"]["path"]
    )
    target_mainland = _line_mask(mainland_path)
    if target_mainland.shape != target_shape:
        raise ValueError("Canonical mainland overlay does not match its declared grid")

    def warp_source_component(label: int, transform: dict) -> np.ndarray:
        warped, report = warp_classified_to_web_mercator(
            (labels == label).astype(np.uint8),
            state,
            transform,
            rgb.shape[:2],
            target_height=target_shape[0],
            clip_to_state=False,
        )
        if warped.shape != target_shape or not np.allclose(report["bounds"], grid["bounds"]):
            raise ValueError("Alignment and canonical boundary use different grids")
        return warped > 0

    parent_component_masks = [
        warp_source_component(int(item["label"]), parent_transform)
        for item in components
    ]
    parent_mainland_boundary = _largest_external_boundary(parent_component_masks[0])
    rows_before = _western_rows(
        parent_mainland_boundary,
        target_mainland,
        start_y=round(target_shape[0] * start_y_fraction),
        end_y=round(target_shape[0] * end_y_fraction),
    )
    coast_controls = _coast_controls(
        rows_before,
        count=coast_control_count,
        smoothing_half_window=smoothing_half_window,
        minimum_left_shift_px=minimum_left_shift_px,
        maximum_left_shift_px=maximum_left_shift_px,
    )
    fixed_controls = []
    for index, component_mask in enumerate(parent_component_masks[1:], 1):
        for point_label, point in _component_pin_points(component_mask):
            fixed_controls.append(
                {
                    "id": f"island_pin_{index:02d}_{point_label}",
                    "current": point.tolist(),
                    "target": point.tolist(),
                    "source_component_area": int(components[index]["area"]),
                }
            )
    controls = [*coast_controls, *fixed_controls]
    corrections = {
        "schema_version": 1,
        "direction": "authoritative_reference_to_current_warped_source",
        "generated_by": "mapscan.solid_coast_refinement",
        "grid": grid,
        "corrections": [
            {
                "id": item["id"],
                "reference": {
                    "pixel": {"x": item["target"][0], "y": item["target"][1]}
                },
                "source": {
                    "pixel": {"x": item["current"][0], "y": item["current"][1]}
                },
                "evidence": {
                    key: value
                    for key, value in item.items()
                    if key not in {"id", "current", "target"}
                },
            }
            for item in controls
        ],
    }
    corrections_path = output_dir / "coast-corrections.json"
    corrections_path.write_text(json.dumps(corrections, indent=2) + "\n")

    fit_dir = output_dir / "fit"
    fit_report = fit_local_review_corrections(
        alignment_path, corrections_path, fit_dir, radius_px=radius_px
    )
    candidate_alignment_path = fit_dir / "alignment.json"
    candidate_alignment = json.loads(candidate_alignment_path.read_text())
    candidate_transform = _alignment_transform(candidate_alignment, state)
    candidate_component_masks = [
        warp_source_component(int(item["label"]), candidate_transform)
        for item in components
    ]
    candidate_mainland_boundary = _largest_external_boundary(candidate_component_masks[0])
    rows_after = _western_rows(
        candidate_mainland_boundary,
        target_mainland,
        start_y=round(target_shape[0] * start_y_fraction),
        end_y=round(target_shape[0] * end_y_fraction),
    )

    before_median = float(np.median(rows_before[:, 3]))
    after_median = float(np.median(rows_after[:, 3]))
    before_gap = _west_gap_summary(rows_before)
    after_gap = _west_gap_summary(rows_after)
    if (
        after_gap["p90_px"] >= before_gap["p90_px"]
        or after_gap["fraction_with_gap"] >= before_gap["fraction_with_gap"]
    ):
        raise ValueError("Coast correction did not reduce the visible western white gap")
    eastern_drift = _eastern_row_drift(
        parent_mainland_boundary, candidate_mainland_boundary
    )
    if (
        eastern_drift["p90_absolute_px"] > 1.0
        or eastern_drift["unchanged_fraction"] < 0.98
    ):
        raise ValueError("West-coast stretch moved too much of the eastern border")
    island_drift = []
    for index, (before_mask, after_mask) in enumerate(
        zip(parent_component_masks[1:], candidate_component_masks[1:]), 1
    ):
        before_center, after_center = _centroid(before_mask), _centroid(after_mask)
        drift = float(np.linalg.norm(after_center - before_center))
        island_drift.append(
            {
                "component": index,
                "before_centroid": before_center.tolist(),
                "after_centroid": after_center.tolist(),
                "centroid_drift_px": drift,
            }
        )
    maximum_island_drift = max(item["centroid_drift_px"] for item in island_drift)
    if maximum_island_drift > 0.5:
        raise ValueError("Coast correction moves an island by more than half a pixel")

    combined_before = _external_boundary(
        np.logical_or.reduce(parent_component_masks), maximum_components
    )
    combined_after = _external_boundary(
        np.logical_or.reduce(candidate_component_masks), maximum_components
    )
    target_all_path = canonical_boundary_manifest_path.parent / str(
        canonical["artifacts"]["overlay"]["path"]
    )
    target_all = _line_mask(target_all_path)
    Image.fromarray(
        _render_diagnostic(target_all, combined_before, combined_after), mode="RGB"
    ).save(output_dir / "coast-correction-diagnostic.png", optimize=True)
    Image.fromarray(combined_after.astype(np.uint8) * 255, mode="L").save(
        output_dir / "candidate-boundary.png", optimize=True
    )

    report = {
        "schema_version": 1,
        "status": "needs_visual_review",
        "source": {"path": str(image_path), "sha256": _sha256(image_path)},
        "parent_alignment": {
            "path": str(alignment_path),
            "sha256": _sha256(alignment_path),
        },
        "candidate_alignment": {
            "path": str(candidate_alignment_path),
            "sha256": _sha256(candidate_alignment_path),
        },
        "policy": {
            "mainland_west_coast": "stretch left to eliminate the canonical-line white gap",
            "offshore_components": "fixed center and four-extrema pins",
            "eastern_border": "verified stationary by row-envelope comparison",
            "canonical_border": "unchanged",
        },
        "controls": {
            "coast": coast_controls,
            "fixed": fixed_controls,
            "radius_px": radius_px,
        },
        "coast": {
            "before_target_minus_source_x_median_px": before_median,
            "after_target_minus_source_x_median_px": after_median,
            "before_target_minus_source_x_p10_p90_px": [
                float(value) for value in np.quantile(rows_before[:, 3], [0.1, 0.9])
            ],
            "after_target_minus_source_x_p10_p90_px": [
                float(value) for value in np.quantile(rows_after[:, 3], [0.1, 0.9])
            ],
            "white_gap_before": before_gap,
            "white_gap_after": after_gap,
        },
        "eastern_border_drift": eastern_drift,
        "islands": island_drift,
        "maximum_island_centroid_drift_px": maximum_island_drift,
        "boundary_distance_before": _distance_summary(combined_before, target_all),
        "boundary_distance_after": _distance_summary(combined_after, target_all),
        "local_fit": fit_report,
        "artifacts": {
            "corrections": {
                "path": corrections_path.name,
                "sha256": _sha256(corrections_path),
            },
            "diagnostic": {
                "path": "coast-correction-diagnostic.png",
                "sha256": _sha256(output_dir / "coast-correction-diagnostic.png"),
            },
            "candidate_boundary": {
                "path": "candidate-boundary.png",
                "sha256": _sha256(output_dir / "candidate-boundary.png"),
            },
        },
    }
    (output_dir / "coast-refinement-report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report
