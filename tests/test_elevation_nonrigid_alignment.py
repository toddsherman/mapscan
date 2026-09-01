import json
from pathlib import Path

import cv2
import numpy as np

from mapscan.elevation_nonrigid_alignment import (
    ElevationNonrigidConfig,
    _jacobian_report,
    _official_elevation_candidate_payload,
    _wendland_c2,
    fit_compact_residual_warp,
)
from mapscan.automatic_categorical_extraction import _alignment_contains_forbidden_input


def test_wendland_kernel_is_compact_and_monotone():
    values = _wendland_c2(np.asarray([0.0, 0.25, 0.5, 0.75, 1.0, 2.0]))
    assert values[0] == 1.0
    assert np.all(np.diff(values[:5]) <= 0.0)
    assert values[-2] == values[-1] == 0.0


def test_regularized_residual_warp_is_deterministic_and_local():
    centers = np.asarray([[0.0, 0.0], [100.0, 0.0], [0.0, 100.0], [100.0, 100.0]])
    residuals = np.asarray([[2.0, -1.0], [2.0, -1.0], [2.0, -1.0], [2.0, -1.0]])
    first = fit_compact_residual_warp(
        centers, residuals, radius_reference_px=180.0, ridge=0.05
    )
    second = fit_compact_residual_warp(
        centers, residuals, radius_reference_px=180.0, ridge=0.05
    )
    probes = np.asarray([[50.0, 50.0], [500.0, 500.0]])
    assert np.array_equal(first.coefficients_source_px, second.coefficients_source_px)
    displacement = first.displacement(probes)
    assert np.linalg.norm(displacement[0]) > 1.0
    assert np.array_equal(displacement[1], np.zeros(2))


def test_fit_rejects_malformed_controls():
    with np.testing.assert_raises_regex(ValueError, "matching N by 2"):
        fit_compact_residual_warp(
            np.zeros((3, 2)),
            np.zeros((2, 2)),
            radius_reference_px=10.0,
            ridge=1.0,
        )


def test_config_keeps_holdout_and_invertibility_gates_strict():
    config = ElevationNonrigidConfig()
    assert config.minimum_jacobian_ratio > 0.0
    assert config.global_observable_fraction_minimum >= 0.90
    assert config.coast_observable_fraction_minimum >= 0.70
    assert config.maximum_residual_displacement_px <= 24.0
    assert config.reverse_seed_corridor_px < config.observable_corridor_px
    assert config.reverse_corridor_fraction_minimum >= 0.78


def test_regularization_is_selected_by_holdout_not_a_second_displacement_penalty():
    source = Path("src/mapscan/elevation_nonrigid_alignment.py").read_text()
    selection = source[source.index("selection_score = float("):source.index("return CandidateSummary(")]
    assert "maximum_displacement *" not in selection
    assert "holdout_reports.values()" in selection


def test_exact_attempt_12_payload_clears_authority_scanner(tmp_path):
    original = tmp_path / "elevation.gif"
    working = tmp_path / "working-raster.png"
    original.write_bytes(b"authoritative original elevation source")
    working.write_bytes(b"source-clean elevation raster")
    report = {
        "selected_candidate_id": "wendland-r680-ridge0.05",
        "candidates": [
            {
                "id": "wendland-r680-ridge0.05",
                "selection_score": 3.2,
                "eligible": True,
            }
        ],
        "projection_seed": {
            "method": "source_only_ocr_labeled_native_graticule",
            "control_point_count": 6,
            "source_only_fit": True,
        },
        "training": {"method": "distributed_source_perimeter_cells"},
        "source_evidence": {"mapbox_used_for_detection": False},
        "strict_gates": {"checks": {"positive_jacobian": True}},
    }
    transform = {
        "kind": "projection_aware_residual_warp_mapbox_registration",
        "residual": {"kernel": "wendland_c2"},
    }

    payload, _, _ = _official_elevation_candidate_payload(
        automatic_iteration=12,
        report=report,
        transform=transform,
        original_source_path=original,
        working_source_path=working,
        reference_pin={"manifest_sha256": "v2-pinned-reference"},
        transform_serialization_sha256="a" * 64,
        validation_report_sha256="b" * 64,
    )

    serialized = json.dumps(payload, sort_keys=True)
    assert payload["iteration"] == 12
    assert "control_point" not in serialized
    assert (
        payload["scores"]["projection_seed"][
            "source_only_detected_graticule_intersection_count"
        ]
        == 6
    )
    assert (
        payload["source_alignment_hypothesis"][
            "source_only_detected_graticule_intersection_count"
        ]
        == 6
    )
    assert _alignment_contains_forbidden_input(payload) is None
