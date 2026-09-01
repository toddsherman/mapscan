import numpy as np

from mapscan.enclosed_fill import fill_small_enclosed_holes


def test_fills_only_small_hole_with_one_surrounding_class():
    values = np.full((9, 9), 2, dtype=np.uint8)
    values[3:6, 3:6] = 0
    fill_values, fill_mask, report = fill_small_enclosed_holes(values, 50)
    assert np.all(fill_mask[3:6, 3:6])
    assert np.all(fill_values[3:6, 3:6] == 2)
    assert report["filled_component_count"] == 1
    assert report["filled_pixel_count"] == 9


def test_rejects_mixed_boundary_edge_limit_and_protected_zeros():
    mixed = np.ones((9, 9), dtype=np.uint8)
    mixed[3:6, 3:6] = 0
    mixed[2, 4] = 2
    assert not np.any(fill_small_enclosed_holes(mixed, 50)[1])

    edge = np.ones((9, 9), dtype=np.uint8)
    edge[0, 3] = 0
    assert not np.any(fill_small_enclosed_holes(edge, 50)[1])

    at_limit = np.ones((14, 14), dtype=np.uint8)
    at_limit[2:7, 2:12] = 0
    assert not np.any(fill_small_enclosed_holes(at_limit, 50)[1])

    protected = np.ones((9, 9), dtype=np.uint8)
    protected[4, 4] = 0
    protected_mask = np.zeros_like(protected, dtype=bool)
    protected_mask[4, 4] = True
    assert not np.any(
        fill_small_enclosed_holes(
            protected, 50, protected_zero_mask=protected_mask
        )[1]
    )
