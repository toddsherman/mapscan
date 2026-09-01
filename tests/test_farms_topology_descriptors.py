from __future__ import annotations

import cv2
import numpy as np

from mapscan.farms_topology_descriptors import (
    TopologyDescriptorConfig,
    descriptor_audit_provenance,
    match_training_topology,
    summarize_match_consistency,
    topology_descriptor_fields,
)


def _line(mask: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> None:
    cv2.line(mask.view(np.uint8), start, end, 1, 1, cv2.LINE_8)


def test_tangent_descriptor_rejects_closer_perpendicular_alias():
    rendered = np.zeros((80, 100), dtype=bool)
    source = np.zeros_like(rendered)
    _line(rendered, (10, 30), (90, 30))
    _line(source, (10, 34), (90, 34))  # true same-orientation line
    _line(source, (50, 20), (50, 32))  # closer false perpendicular line

    reference = np.asarray([[50.0, 30.0], [51.0, 30.0], [52.0, 30.0]])
    mapped = reference.copy()
    config = TopologyDescriptorConfig(minimum_bin_support=1)
    nearest = match_training_topology(
        reference, mapped, rendered, source, method="nearest", config=config
    )
    descriptor = match_training_topology(
        reference, mapped, rendered, source, method="descriptor", config=config
    )

    assert nearest.source_points[0].tolist() == [50.0, 30.0]
    assert np.all(descriptor.source_points[:, 1] == 34.0)
    assert np.all(descriptor.orientation_error_degrees < 5.0)


def test_descriptor_fields_mark_crossing_as_junction_not_ordinary_segment():
    mask = np.zeros((80, 80), dtype=bool)
    _line(mask, (10, 40), (70, 40))
    _line(mask, (40, 10), (40, 70))

    fields = topology_descriptor_fields(mask)

    assert fields.junction_distance[40, 40] == 0
    assert fields.junction_distance[40, 15] > 6
    assert fields.coherence[40, 15] > fields.coherence[40, 40]


def test_consistency_summary_rejects_scattered_control_bin():
    rendered = np.zeros((100, 120), dtype=bool)
    source = np.zeros_like(rendered)
    _line(rendered, (10, 30), (110, 30))
    _line(source, (10, 34), (55, 34))
    _line(source, (56, 42), (110, 42))
    xs = np.arange(10, 111, dtype=np.float64)
    reference = np.column_stack((xs, np.full_like(xs, 30.0)))
    mapped = reference.copy()
    config = TopologyDescriptorConfig(
        bin_reference_px=260,
        minimum_bin_support=20,
        maximum_bin_residual_mad_px=2.0,
    )

    matches = match_training_topology(
        reference, mapped, rendered, source, method="descriptor", config=config
    )
    report = summarize_match_consistency(matches, config=config)

    assert report["bin_count"] == 1
    assert report["accepted_control_count"] == 0
    assert report["bins"][0]["residual_median_absolute_deviation_px"] > 2.0


def test_descriptor_api_provenance_cannot_accept_validation_or_test_masks():
    rendered = np.zeros((32, 32), dtype=bool)
    source = np.zeros_like(rendered)
    _line(rendered, (4, 12), (28, 12))
    _line(source, (4, 14), (28, 14))

    provenance = descriptor_audit_provenance(
        rendered, source, TopologyDescriptorConfig()
    )

    assert provenance["validation_or_retained_acceptance_inputs"] == []
    assert "validation" not in TopologyDescriptorConfig.__dataclass_fields__
    assert "test" not in TopologyDescriptorConfig.__dataclass_fields__
    assert len(provenance["rendered_training_mask_sha256"]) == 64
    assert len(provenance["source_training_mask_sha256"]) == 64


def test_matcher_rejects_unregistered_matching_method():
    mask = np.zeros((32, 32), dtype=bool)
    _line(mask, (4, 12), (28, 12))
    points = np.asarray([[12.0, 12.0]])

    try:
        match_training_topology(
            points,
            points,
            mask,
            mask,
            method="validation",  # type: ignore[arg-type]
            config=TopologyDescriptorConfig(minimum_bin_support=1),
        )
    except ValueError as error:
        assert "method" in str(error).lower()
    else:
        raise AssertionError("Unregistered matcher method must fail closed")
