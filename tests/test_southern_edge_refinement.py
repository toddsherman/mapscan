import numpy as np

from mapscan.southern_edge_refinement import (
    _paired_segment_matches,
    _signed_land_background_match,
)


def _horizontal_segment(y=100):
    points = np.column_stack((np.arange(20, 181), np.full(161, y)))
    normals = np.tile(np.asarray([0.0, -1.0]), (len(points), 1))
    return {
        "id": "south_border",
        "points": points,
        "normals": normals,
        "center": np.asarray([100.0, float(y)]),
        "normal": np.asarray([0.0, -1.0]),
    }


def test_signed_boundary_uses_land_side_and_ignores_parallel_shadow():
    rgb = np.full((220, 220, 3), 232, dtype=np.uint8)
    rgb[:103, :] = (185, 42, 38)
    rgb[111:115, :] = (155, 155, 155)
    match = _signed_land_background_match(rgb, _horizontal_segment())
    assert abs(match["normal_shift_px"] + 3) <= 2
    assert match["normal_shift_px"] > -9


def test_split_evidence_rejects_inconsistent_edge_halves():
    rgb = np.full((220, 220, 3), 232, dtype=np.uint8)
    rgb[:103, :101] = (185, 42, 38)
    rgb[:109, 101:] = (185, 42, 38)
    fit, holdout, pairs = _paired_segment_matches(
        rgb, [_horizontal_segment()]
    )
    assert fit == []
    assert holdout == []
    assert pairs[0]["accepted"] is False
    assert pairs[0]["shift_difference_px"] >= 3
