from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from mapscan.farms_native_county_topology import (
    compare_native_and_working_topology,
    derive_farms_native_county_topology,
)
from mapscan.farms_nonrigid_alignment import FarmsNonrigidConfig, _load_source


def test_native_county_topology_recovers_thin_line_before_rgb_reduction():
    native = np.full((120, 100, 3), 220, dtype=np.uint8)
    # Pure black is furniture and must not become the anchor.
    native[4:116, 3] = 0
    # A one-native-pixel administrative line disappears after RGB area
    # reduction, but its flat neutral core remains recoverable as a mask.
    native[12:108, 47] = 52
    native[75, 25:76] = 52
    scope = np.ones((20, 16), dtype=bool)
    state = np.zeros_like(scope)

    result = derive_farms_native_county_topology(
        native,
        county_scope_working=scope,
        state_coast_working=state,
    )

    assert result.diagnostics["anchor_gray"] == 52
    assert result.diagnostics["mapbox_or_reference_geometry_used"] is False
    assert result.diagnostics["candidate_transform_used"] is False
    assert np.count_nonzero(result.native_ink) >= 140
    assert np.count_nonzero(result.working_ink) >= 20


def test_native_county_topology_respects_source_scope_and_state_corridor():
    native = np.full((200, 200, 3), 220, dtype=np.uint8)
    native[20:180, 40] = 52
    native[20:180, 100] = 52
    scope = np.ones((40, 40), dtype=bool)
    scope[:, :8] = False
    state = np.zeros_like(scope)
    state[:, 20] = True

    result = derive_farms_native_county_topology(
        native,
        county_scope_working=scope,
        state_coast_working=state,
    )

    assert not np.any(result.working_ink[:, :8])
    assert not np.any(result.working_ink[:, 17:24])


def test_real_farms_native_county_topology_adds_source_supported_geometry():
    source = _load_source("examples/farmsv2.png", FarmsNonrigidConfig())
    result = derive_farms_native_county_topology(
        source.rgb_original,
        county_scope_working=source.topology.county_scope,
        state_coast_working=source.topology.state_coast,
    )
    comparison = compare_native_and_working_topology(
        result.working_ink,
        source.topology.internal_topology,
        corridor_px=2,
    )

    assert result.diagnostics["native_source_shape"] == [5500, 4250]
    assert result.diagnostics["working_shape"] == [900, 695]
    assert result.diagnostics["anchor_gray"] == 52
    assert result.diagnostics["retained_component_count"] >= 15
    assert result.diagnostics["working_pixel_count"] > 9_000
    assert comparison["native_pixels_missing_from_working_rgb_count"] > 1_000
    assert comparison["working_rgb_supported_fraction"] > 0.80


def test_real_source_combines_native_primary_with_only_nearby_scoped_secondary():
    source = _load_source("examples/farmsv2.png", FarmsNonrigidConfig())
    native = source.native_county_topology.working_ink
    scoped_working = (
        source.topology.internal_topology & source.topology.county_scope
    )
    native_corridor = cv2.dilate(
        native.astype(np.uint8), np.ones((5, 5), np.uint8)
    ).astype(bool)
    expected_secondary = scoped_working & native_corridor
    expected_combined = native | expected_secondary

    assert np.array_equal(
        source.working_rgb_scoped_county_topology, scoped_working
    )
    assert np.array_equal(
        source.working_rgb_secondary_county_topology, expected_secondary
    )
    assert np.array_equal(source.county_topology, expected_combined)
    assert not np.any(
        source.county_topology & scoped_working & ~native_corridor
    )
    assert source.semantic.counties is source.county_topology
    assert source.county_topology_diagnostics[
        "unsupported_working_rgb_pixels_raw_union_forbidden"
    ] is True


def test_topology_comparison_is_symmetric_and_auditable():
    first = np.zeros((10, 10), dtype=bool)
    second = np.zeros_like(first)
    first[2:8, 2] = True
    second[2:8, 3] = True

    exact = compare_native_and_working_topology(first, second, corridor_px=0)
    tolerant = compare_native_and_working_topology(first, second, corridor_px=1)

    assert exact["native_pixels_missing_from_working_rgb_count"] == 6
    assert exact["working_rgb_pixels_unsupported_by_native_count"] == 6
    assert tolerant["native_pixels_missing_from_working_rgb_count"] == 0
    assert tolerant["working_rgb_pixels_unsupported_by_native_count"] == 0
