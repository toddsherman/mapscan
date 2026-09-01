from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from mapscan.automatic_alignment_loop import _dominant_neutral_pacific, _resize_working
from mapscan.farms_partial_topology import (
    derive_farms_partial_topology,
    render_farms_county_scope_overlay,
)


def _synthetic_partial_farms_source() -> tuple[np.ndarray, np.ndarray]:
    height, width = 300, 220
    rgb = np.full((height, width, 3), 225, dtype=np.uint8)

    pacific = np.zeros((height, width), dtype=bool)
    coast = np.asarray(
        [(0, 80), (34, 80), (47, 125), (42, 175), (69, 245), (82, 299)],
        dtype=np.int32,
    )
    ocean_polygon = np.vstack(
        [np.asarray([(0, 299), (0, 80)], dtype=np.int32), coast[1:]]
    )
    cv2.fillPoly(pacific.view(np.uint8), [ocean_polygon], 1)
    rgb[pacific] = (208, 207, 212)

    # A flat Nevada panel split into a vertical component and a triangle by the
    # white legend.  Only its west edge is California state-boundary evidence.
    rgb[12:103, 135:151] = (146, 146, 146)
    triangle = np.asarray([(151, 92), (219, 145), (219, 195)], dtype=np.int32)
    cv2.fillPoly(rgb, [triangle], (146, 146, 146))

    # Upper-right legend/UI, deliberately adjacent to the Nevada fill.
    rgb[12:91, 154:216] = (255, 255, 255)

    # Long neutral internal topology and one chromatic thematic patch.
    cv2.line(rgb, (76, 35), (102, 230), (52, 52, 52), 2, cv2.LINE_8)
    cv2.line(rgb, (52, 150), (170, 150), (52, 52, 52), 2, cv2.LINE_8)
    # Source-observed U.S./Mexico border: Pacific anchored and ending at the
    # right inset edge. Dense thematic pixels may interrupt it in real maps,
    # but the synthetic keeps the topology explicit.
    cv2.line(rgb, (70, 250), (216, 250), (52, 52, 52), 2, cv2.LINE_8)
    # Unambiguous California-side line and an ambiguous Arizona-side component
    # that terminates at the right frame after the Nevada trace disappears.
    cv2.line(rgb, (92, 205), (142, 225), (52, 52, 52), 2, cv2.LINE_8)
    cv2.line(rgb, (176, 205), (216, 232), (52, 52, 52), 2, cv2.LINE_8)
    # Mexico linework must be clipped pixel-locally south of the border.
    cv2.line(rgb, (104, 274), (170, 274), (52, 52, 52), 2, cv2.LINE_8)
    rgb[170:205, 90:125] = (205, 40, 85)
    return rgb, pacific


def test_partial_farms_adapter_isolates_state_edges_from_internal_network():
    rgb, pacific = _synthetic_partial_farms_source()

    result = derive_farms_partial_topology(rgb, pacific)

    assert result.state_coast.shape == pacific.shape
    assert result.internal_topology.shape == pacific.shape
    assert np.count_nonzero(result.state_coast) > 200
    assert np.count_nonzero(result.internal_topology) > 100
    assert result.state_coast[60, 135]
    assert np.any(result.state_coast[138:151, 180:195])
    assert not result.state_coast[150, 100]
    assert result.internal_topology[150, 100]


def test_partial_farms_adapter_excludes_ui_and_neighbor_but_keeps_thematic_support():
    rgb, pacific = _synthetic_partial_farms_source()

    result = derive_farms_partial_topology(rgb, pacific)

    assert result.layout_exclusion[30, 180]
    assert result.neighboring_region[60, 140]
    assert not result.foreground_interior[60, 140]
    assert not result.foreground_interior[30, 180]
    assert result.foreground_interior[180, 105]
    assert result.diagnostics["thematic_support_fraction"] == 1.0


def test_partial_farms_adapter_omits_adjacent_and_south_of_border_topology():
    rgb, pacific = _synthetic_partial_farms_source()

    result = derive_farms_partial_topology(rgb, pacific)

    assert result.internal_topology[215, 116]
    assert not np.any(result.internal_topology[205:235, 195:217])
    assert not np.any(result.internal_topology[270:279, 100:175])
    assert np.any(result.ambiguous_topology_exclusion[205:235, 195:217])
    assert np.any(result.ambiguous_topology_exclusion[270:279, 100:175])
    assert not np.any(result.county_scope[255:])
    scope = result.diagnostics["county_scope"]
    assert scope["colorado_boundary_inferred_as_geometry"] is False
    assert scope["omission_envelope_is_not_alignment_evidence"] is True
    assert scope["ambiguous_lower_right_omitted_with_warning"] is True
    assert len(scope["county_scope_sha256"]) == 64
    assert len(scope["retained_internal_topology_sha256"]) == 64


def test_partial_farms_adapter_fails_closed_without_southern_border():
    rgb, pacific = _synthetic_partial_farms_source()
    rgb[244:257] = (225, 225, 225)

    try:
        derive_farms_partial_topology(rgb, pacific)
    except ValueError as error:
        assert "southern" in str(error).lower()
    else:
        raise AssertionError("Missing southern boundary must fail closed")


def test_partial_farms_scope_overlay_distinguishes_retained_and_omitted_lines():
    rgb, pacific = _synthetic_partial_farms_source()
    result = derive_farms_partial_topology(rgb, pacific)

    overlay = render_farms_county_scope_overlay(rgb, result)

    assert np.any(np.all(overlay == (255, 0, 220), axis=2))
    assert np.any(np.all(overlay == (255, 55, 45), axis=2))
    assert np.any(np.all(overlay == (255, 225, 0), axis=2))


def test_real_farms_scope_detects_partial_border_and_omits_neighbor_networks():
    original = np.asarray(Image.open(Path("examples/farmsv2.png")).convert("RGB"))
    working, _scale = _resize_working(original, 900)
    pacific_result = _dominant_neutral_pacific(working)
    assert pacific_result is not None
    pacific, _diagnostics = pacific_result

    result = derive_farms_partial_topology(working, pacific)

    assert 742 <= result.diagnostics["southern_border"]["row_y"] <= 748
    assert result.diagnostics["east_boundary_last_row"] == 447
    assert result.diagnostics["raw_internal_topology_px"] == 10_271
    assert 7_000 <= result.diagnostics["internal_topology_px"] <= 7_700
    scope = result.diagnostics["county_scope"]
    assert scope["ambiguous_adjacent_topology_pixels"] >= 2_500
    assert scope["pacific_anchored_southern_core_east_x"] < 400
    south_row = result.diagnostics["southern_border"]["row_y"]
    assert not np.any(result.internal_topology[south_row - 3 :])


def test_partial_farms_adapter_provenance_is_strictly_source_only():
    rgb, pacific = _synthetic_partial_farms_source()

    result = derive_farms_partial_topology(rgb, pacific)

    authority = result.diagnostics["authority"]
    assert authority == {
        "manual_inputs_used": False,
        "prior_run_artifacts_used": False,
        "county_png_used": False,
        "mapbox_used_for_hypothesis_construction": False,
        "mapbox_required_for_acceptance": True,
    }
    assert result.diagnostics["neighboring_region"][
        "mapbox_used_for_neighbor_selection"
    ] is False
