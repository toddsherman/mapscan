import numpy as np

from mapscan.solid_mask_alignment import (
    _apply,
    _exact_palette_mask,
    _external_boundary,
    _retain_components,
    _similarity_matrix,
)


def test_exact_palette_mask_does_not_accept_nearby_basemap_color():
    rgb = np.asarray(
        [[[10, 20, 30], [10, 20, 31]], [[40, 50, 60], [255, 255, 255]]],
        dtype=np.uint8,
    )
    result = _exact_palette_mask(rgb, [[10, 20, 30], [40, 50, 60]])
    assert result.tolist() == [[True, False], [True, False]]


def test_component_filter_and_external_boundary_are_deterministic():
    mask = np.zeros((30, 30), dtype=bool)
    mask[3:20, 2:17] = True
    mask[8:11, 8:11] = False
    mask[23:28, 23:28] = True
    mask[0, 29] = True
    retained, components = _retain_components(
        mask, minimum_area=10, maximum_components=2
    )
    assert [item["area"] for item in components] == [246, 25]
    assert not retained[0, 29]
    boundary = _external_boundary(retained, maximum_components=2)
    assert boundary[3, 2]
    assert not boundary[8, 8]


def test_similarity_matrix_uses_grid_center_and_expected_direction():
    matrix = _similarity_matrix([0.0, 0.0, 4.0, -3.0], 101, 81)
    points = np.asarray([[50.0, 40.0], [0.0, 0.0]])
    moved = _apply(matrix, points)
    assert np.allclose(moved, points + [4.0, -3.0])
