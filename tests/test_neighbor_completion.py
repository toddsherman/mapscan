from __future__ import annotations

import numpy as np

from mapscan.neighbor_completion import (
    closed_line_component_interiors,
    fill_unknown_from_neighbors,
    partition_completion_pixels,
)


def test_closed_line_component_interiors_preserves_separate_islands():
    line = np.zeros((30, 40), dtype=np.uint8)
    line[3:10, 4] = 1
    line[3:10, 11] = 1
    line[3, 4:12] = 1
    line[9, 4:12] = 1
    line[17:27, 23] = 1
    line[17:27, 34] = 1
    line[17, 23:35] = 1
    line[26, 23:35] = 1

    filled = closed_line_component_interiors(line, expected_components=2)

    assert filled[6, 7]
    assert filled[21, 29]
    assert not filled[14, 18]


def test_neighbor_fill_uses_component_boundary_and_preserves_known_values():
    seed = np.zeros((9, 11), dtype=np.uint8)
    seed[2:7, 2] = 2
    seed[2:7, 8] = 7
    unknown = np.zeros_like(seed, dtype=bool)
    unknown[2:7, 3:8] = True

    output, diagnostics, metrics = fill_unknown_from_neighbors(
        seed, unknown, np.zeros_like(seed, dtype=bool), neighbors=4
    )

    assert np.all(output[unknown] > 0)
    assert np.array_equal(output[seed > 0], seed[seed > 0])
    assert np.all(output[3:6, 3:5] == 2)
    assert np.all(output[3:6, 6:8] == 7)
    assert metrics["component_count"] == 1
    assert not diagnostics["fallback"].any()


def test_authored_stamp_weight_can_change_neighbor_choice():
    seed = np.zeros((7, 7), dtype=np.uint8)
    seed[2, 2] = 4
    seed[2, 4] = 9
    seed[4, 2] = 4
    seed[4, 4] = 9
    unknown = np.zeros_like(seed, dtype=bool)
    unknown[3, 3] = True
    manual = np.zeros_like(seed, dtype=bool)
    manual[2, 4] = True

    output, diagnostics, metrics = fill_unknown_from_neighbors(
        seed, unknown, manual, neighbors=4, manual_weight=3.0
    )

    assert output[3, 3] == 9
    assert diagnostics["manual_neighbor"][3, 3]
    assert diagnostics["manual_changed_choice"][3, 3]
    assert metrics["manual_weight_changed_choice_pixel_count"] == 1


def test_unknown_outside_is_not_passed_to_neighbor_fill():
    seed = np.zeros((6, 8), dtype=np.uint8)
    seed[1:5, 1] = 3
    unknown = np.zeros_like(seed, dtype=bool)
    unknown[2:4, 2:7] = True
    valid = np.zeros_like(seed, dtype=bool)
    valid[:, :5] = True

    output, _, _ = fill_unknown_from_neighbors(
        seed, unknown & valid, np.zeros_like(seed, dtype=bool), neighbors=3
    )
    output[~valid] = 0

    assert np.all(output[unknown & valid] == 3)
    assert np.all(output[unknown & ~valid] == 0)


def test_completion_partition_includes_untracked_interior_nodata():
    seed = np.array(
        [
            [0, 0, 0, 0],
            [0, 2, 0, 0],
            [0, 0, 4, 0],
        ],
        dtype=np.uint8,
    )
    audited = np.zeros_like(seed, dtype=bool)
    audited[1, 2] = True
    audited[2, 3] = True
    valid = np.zeros_like(seed, dtype=bool)
    valid[1:, 1:3] = True

    inside, outside, additional = partition_completion_pixels(
        seed, audited, valid
    )

    assert inside[1, 2]
    assert inside[2, 1]
    assert not inside[2, 3]
    assert outside[2, 3]
    assert additional[2, 1]
    assert not additional[1, 2]
