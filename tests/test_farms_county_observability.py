from __future__ import annotations

import inspect

import cv2
import numpy as np

from mapscan.farms_county_observability import (
    CountyObservabilityConfig,
    ThematicOcclusionConfig,
    derive_county_observability,
    derive_thematic_occlusion,
    evaluate_mapped_county_geography,
    render_county_observability_overlay,
)


def _synthetic_source() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb = np.full((120, 160, 3), 230, dtype=np.uint8)
    scope = np.ones((120, 160), dtype=bool)
    scope[:, :12] = False
    ink = np.zeros_like(scope)
    cv2.line(ink.view(np.uint8), (24, 28), (145, 28), 1, 1, cv2.LINE_8)
    cv2.line(ink.view(np.uint8), (78, 18), (78, 105), 1, 1, cv2.LINE_8)
    rgb[ink] = (140, 140, 140)
    # An opaque patch whose color is indistinguishable from the county ink.
    rgb[65:100, 105:145] = (140, 140, 140)
    return rgb, scope, ink


def test_observability_is_source_only_and_conservatively_calibrated():
    rgb, scope, ink = _synthetic_source()

    evidence = derive_county_observability(rgb, scope, ink)

    assert evidence.diagnostics["observed_ink_self_retention"] >= 0.985
    assert evidence.observable[45, 45]
    assert not evidence.observable[82, 125]
    assert not evidence.observable[45, 5]
    assert evidence.explicit_exclusion[45, 5]
    authority = evidence.diagnostics["authority"]
    assert authority["mapbox_geometry_used"] is False
    assert authority["candidate_transform_used"] is False
    assert authority["reference_to_source_residual_distance_used"] is False


def test_evaluation_distance_never_changes_source_observability():
    rgb, scope, ink = _synthetic_source()
    evidence = derive_county_observability(rgb, scope, ink)
    reference = np.asarray([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    mapped = np.asarray([[78.0, 28.0], [40.0, 40.0], [125.0, 82.0]])

    narrow = evaluate_mapped_county_geography(
        reference,
        mapped,
        evidence,
        ink,
        config=CountyObservabilityConfig(evaluation_distance_px=2.0),
    )
    wide = evaluate_mapped_county_geography(
        reference,
        mapped,
        evidence,
        ink,
        config=CountyObservabilityConfig(evaluation_distance_px=20.0),
    )

    assert narrow["source_observable_count"] == wide["source_observable_count"]
    assert narrow["source_explicitly_unobservable_count"] == 1
    assert narrow["near_detected_source_topology_count"] == 1
    assert wide["near_detected_source_topology_count"] == 2


def test_observability_api_cannot_receive_reference_or_residual_inputs():
    parameters = inspect.signature(derive_county_observability).parameters

    assert set(parameters) == {
        "rgb_source",
        "county_scope",
        "observed_county_ink",
        "config",
    }
    assert "reference" not in parameters
    assert "residual" not in parameters
    assert "mapbox" not in parameters


def test_observability_overlay_marks_each_source_evidence_class():
    rgb, scope, ink = _synthetic_source()
    evidence = derive_county_observability(rgb, scope, ink)

    overlay = render_county_observability_overlay(rgb, evidence, ink)

    assert np.any(np.all(overlay == (255, 0, 220), axis=2))
    assert np.any(np.all(overlay == (255, 150, 0), axis=2))
    assert not np.array_equal(overlay[45, 5], rgb[45, 5])


def test_thematic_occlusion_is_bounded_and_independent_of_observed_ink():
    rgb = np.full((80, 100, 3), 230, dtype=np.uint8)
    scope = np.ones((80, 100), dtype=bool)
    ink = np.zeros_like(scope)
    rgb[25:60, 30:75] = (220, 40, 60)
    cv2.line(ink.view(np.uint8), (20, 42), (85, 42), 1, 1, cv2.LINE_8)
    rgb[ink] = (52, 52, 52)

    evidence = derive_thematic_occlusion(
        rgb,
        scope,
        config=ThematicOcclusionConfig(printed_stroke_radius_px=2),
    )

    assert evidence.occluded[32, 50]
    assert evidence.occluded[42, 50]
    assert not evidence.observable[42, 50]
    assert evidence.observable[10, 10]
    assert evidence.diagnostics["authority"][
        "diagnostic_only_not_an_acceptance_omission"
    ] is True
    assert evidence.diagnostics["config"]["printed_stroke_radius_px"] == 2


def test_thematic_occlusion_rejects_expansion_beyond_printed_stroke():
    rgb = np.full((20, 20, 3), 230, dtype=np.uint8)
    scope = np.ones((20, 20), dtype=bool)
    ink = np.zeros_like(scope)

    try:
        derive_thematic_occlusion(
            rgb,
            scope,
            config=ThematicOcclusionConfig(printed_stroke_radius_px=4),
        )
    except ValueError as error:
        assert "stroke" in str(error).lower()
    else:
        raise AssertionError("Thematic expansion beyond stroke radius must fail")
