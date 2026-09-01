import cv2
import numpy as np

from mapscan.county_fine_alignment import (
    _administrative_stroke_response,
    _fit_matches_for_coverage,
    _partial_fold_gate,
    _spatial_holdout_audit,
    _template_matches,
    _transform_regularity_audit,
)


def _junction_mask(shape=(520, 520)):
    mask = np.zeros(shape, dtype=np.uint8)
    centers = []
    for y in (70, 190, 310, 430):
        for x in (70, 190, 310, 430):
            centers.append((x, y))
            cv2.line(mask, (x - 30, y), (x + 30, y), 1, 1)
            cv2.line(mask, (x, y - 30), (x, y + 30), 1, 1)
            cv2.line(mask, (x, y), (x + 24, y + 24), 1, 1)
    return mask > 0, centers


def test_administrative_stroke_response_prefers_narrow_dark_lines():
    rgb = np.full((240, 240, 3), 190, dtype=np.uint8)
    cv2.line(rgb, (20, 120), (220, 120), (45, 45, 45), 2)
    cv2.circle(rgb, (120, 120), 58, (105, 135, 170), -1)
    valid = np.ones(rgb.shape[:2], dtype=bool)
    response = _administrative_stroke_response(rgb, valid)
    assert float(np.mean(response[118:123, 30:210])) > float(
        np.mean(response[70:90, 70:170])
    )


def test_template_matching_rejects_large_ocean_artifact_and_recovers_shift():
    county, _ = _junction_mask()
    shifted = np.zeros_like(county, dtype=np.uint8)
    matrix = np.asarray([[1, 0, 5], [0, 1, 4]], dtype=np.float32)
    shifted = cv2.warpAffine(
        county.astype(np.uint8), matrix, (county.shape[1], county.shape[0])
    )
    rgb = np.full((*county.shape, 3), 205, dtype=np.uint8)
    rgb[shifted > 0] = (35, 35, 35)
    cv2.ellipse(rgb, (210, 210), (150, 80), 18, 0, 360, (75, 120, 175), -1)
    rgb[shifted > 0] = (35, 35, 35)
    valid = np.ones(county.shape, dtype=bool)
    response = _administrative_stroke_response(rgb, valid)
    matches, consistent = _template_matches(
        response, valid, county, maximum_count=40
    )
    assert len(matches) >= 12
    assert len(consistent) >= 12
    shifts = np.asarray([item["shift_px"] for item in consistent])
    assert np.median(shifts[:, 0]) == 5
    assert np.median(shifts[:, 1]) == 4


def test_ocean_artifact_without_county_strokes_is_insufficient_evidence():
    county, _ = _junction_mask()
    rgb = np.full((*county.shape, 3), 205, dtype=np.uint8)
    cv2.ellipse(rgb, (260, 260), (210, 110), 18, 0, 360, (35, 85, 145), 4)
    valid = np.ones(county.shape, dtype=bool)
    response = _administrative_stroke_response(rgb, valid)
    _, consistent = _template_matches(response, valid, county, maximum_count=40)
    assert consistent == []


def test_spatial_holdouts_never_use_the_validation_region_for_fitting():
    target = []
    for y in (180, 720, 1200, 1800, 2220):
        for x in (200, 700, 1200, 1700):
            target.append((x, y))
    target = np.asarray(target, dtype=np.float64)
    current_to_target = np.asarray(
        [
            [0.998, -0.003, -5.0],
            [0.002, 0.997, -4.0],
            [-0.000002, 0.000001, 1.0],
        ]
    )
    target_to_current = np.linalg.inv(current_to_target)
    homogeneous = np.column_stack((target, np.ones(len(target)))) @ target_to_current.T
    current = homogeneous[:, :2] / homogeneous[:, 2:3]
    matches = [
        {
            "id": f"junction_{index:02d}",
            "source_pixel": source.tolist(),
            "reference_pixel": reference.tolist(),
            "accepted_by_local_evidence": True,
        }
        for index, (source, reference) in enumerate(zip(current, target))
    ]
    audit = _spatial_holdout_audit(matches, (2400, 2080))
    assert audit["passed"] is True
    assert len(audit["folds"]) == 4
    assert all(fold["validation_count"] >= 4 for fold in audit["folds"])
    assert audit["aggregate_after"]["p90_px"] < 0.01


def test_partial_state_holdouts_use_visible_extent_quadrants():
    target = np.asarray(
        [
            (x, y)
            for y in (550, 850, 1250, 1750)
            for x in (320, 540, 760, 980, 1200)
        ],
        dtype=np.float64,
    )
    current_to_target = np.asarray(
        [
            [0.997, -0.004, -8.0],
            [0.003, 0.998, -5.0],
            [-0.000002, 0.000001, 1.0],
        ]
    )
    target_to_current = np.linalg.inv(current_to_target)
    homogeneous = np.column_stack((target, np.ones(len(target)))) @ target_to_current.T
    current = homogeneous[:, :2] / homogeneous[:, 2:3]
    matches = [
        {
            "id": f"junction_{index:02d}",
            "source_pixel": source.tolist(),
            "reference_pixel": reference.tolist(),
            "accepted_by_local_evidence": True,
            "peak_correlation": 0.5,
            "uniqueness_margin": 0.2,
        }
        for index, (source, reference) in enumerate(zip(current, target))
    ]

    audit = _spatial_holdout_audit(
        matches, (2400, 2080), coverage_model="partial_state"
    )

    assert audit["passed"] is True
    assert audit["method"] == "four_visible_extent_quadrants_excluded_from_training"
    assert len(audit["folds"]) == 4
    assert all(fold["validation_count"] >= 4 for fold in audit["folds"])
    assert audit["aggregate_after"]["p90_px"] < 0.01


def test_partial_state_fit_uses_only_the_holdout_quality_evidence_floor():
    matches = [
        {
            "id": "strong",
            "peak_correlation": 0.42,
            "uniqueness_margin": 0.12,
        },
        {
            "id": "weak_peak",
            "peak_correlation": 0.29,
            "uniqueness_margin": 0.12,
        },
        {
            "id": "ambiguous",
            "peak_correlation": 0.42,
            "uniqueness_margin": 0.079,
        },
    ]

    assert [
        item["id"] for item in _fit_matches_for_coverage(matches, "partial_state")
    ] == ["strong"]
    assert _fit_matches_for_coverage(matches, "full_or_most_state") == matches


def test_partial_fold_gate_allows_only_a_bounded_isolated_outlier():
    accepted = _partial_fold_gate(
        {"median_px": 0.5, "p90_px": 2.85, "max_px": 4.55},
        improved_fraction=1.0,
        relative_p90_improvement=0.67,
        scale=1.0,
    )
    assert accepted == {
        "strict_passed": False,
        "isolated_outlier_passed": True,
        "passed": True,
    }
    assert _partial_fold_gate(
        {"median_px": 0.5, "p90_px": 2.85, "max_px": 5.1},
        improved_fraction=1.0,
        relative_p90_improvement=0.67,
        scale=1.0,
    )["passed"] is False


def test_transform_regularity_rejects_foldover():
    grid = {"width": 2080, "height": 2400}
    assert _transform_regularity_audit(np.eye(3), grid)["passed"] is True
    foldover = np.asarray([[-1.0, 0.0, 2080.0], [0.0, 1.0, 0.0], [0, 0, 1]])
    assert _transform_regularity_audit(foldover, grid)["passed"] is False
