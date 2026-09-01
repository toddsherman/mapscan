import numpy as np

from mapscan.hybrid_perimeter import (
    _closed_mainland_geometry,
    _coast_window,
    _connect_southern_seam,
    _hybrid_mainland_interior,
    _hybrid_masks,
    _unified_overlay,
    _web_mercator_pixel,
)


def test_coast_window_requires_agreement_between_independent_channels():
    height, width = 60, 80
    targets = [(y, 30) for y in range(10, 50)]
    evidence = {
        "blackhat": np.zeros((height, width), dtype=np.float32),
        "saturation": np.zeros((height, width), dtype=np.float32),
        "chroma": np.zeros((height, width), dtype=np.float32),
    }
    evidence["blackhat"][:, 33] = 40
    evidence["saturation"][:, 34:] = 80
    evidence["chroma"][:, 34:] = 30

    result = _coast_window(evidence, targets, maximum_shift_px=10)

    assert result["accepted"] is True
    assert abs(result["consensus_shift_px"] - 3) <= 1
    assert result["estimator_range_px"] <= 3


def test_hybrid_mask_uses_county_coast_and_census_land_border():
    mainland = np.zeros((10, 200), dtype=bool)
    mainland[:, 50:151] = True
    census = np.zeros_like(mainland)
    census[:, 50] = True
    census[:, 150] = True
    census[0, 50:151] = True
    census[9, 50:151] = True
    county = np.zeros_like(mainland)
    county[:, 49] = True
    county[:, 150] = True
    county[5, 70:91] = True

    coast, land = _hybrid_masks(
        mainland,
        census,
        county,
        coast_start_y=1,
        coast_end_y=9,
    )

    assert coast[5, 49]
    assert coast[5, 80]  # Preserve a deep bay/cape excursion, not one x per row.
    assert not coast[5, 150]
    assert not land[5, 50]
    assert land[5, 150]
    assert land[0, 50]


def test_hybrid_mask_uses_county_png_for_tahoe_hinge():
    mainland = np.zeros((12, 220), dtype=bool)
    mainland[:, 40:181] = True
    census = np.zeros_like(mainland)
    census[:, 40] = True
    census[:, 180] = True
    county = np.zeros_like(mainland)
    county[:, 39] = True
    county[:, 176] = True

    county_authority, census_authority = _hybrid_masks(
        mainland,
        census,
        county,
        coast_start_y=1,
        coast_end_y=11,
        tahoe_region=(140, 3, 205, 9),
    )

    assert county_authority[5, 176]
    assert not census_authority[5, 180]
    assert census_authority[1, 180]


def test_unified_overlay_is_one_color_over_the_exact_hybrid_union():
    coast = np.zeros((8, 9), dtype=bool)
    land = np.zeros_like(coast)
    coast[1:7, 2] = True
    land[6, 2:8] = True

    overlay = _unified_overlay(coast, land)
    expected = coast | land

    assert np.array_equal(overlay[..., 3] > 0, expected)
    assert set(map(tuple, overlay[expected, :3])) == {(80, 255, 120)}
    assert np.count_nonzero(overlay[~expected]) == 0


def test_closed_mainland_geometry_discards_offshore_components_and_closes_ring():
    interior = np.zeros((30, 40), dtype=bool)
    interior[3:28, 8:34] = True
    interior[12:18, 3:8] = True  # detailed coastal peninsula
    interior[20:23, 1:4] = True  # disconnected offshore artifact

    filled, border = _closed_mainland_geometry(interior)

    assert filled[15, 4]
    assert not filled[21, 2]
    assert np.count_nonzero(border & ~filled) == 0
    import cv2

    assert cv2.connectedComponents(border.astype(np.uint8), 8)[0] - 1 == 1


def test_hybrid_mainland_interior_uses_county_coast_and_census_land_surface():
    mainland = np.zeros((20, 40), dtype=bool)
    mainland[2:19, 10:34] = True
    county = mainland.copy()
    county[8:13, 7:10] = True
    census_border = np.zeros_like(mainland)
    census_border[2:19, 33] = True

    hybrid = _hybrid_mainland_interior(
        mainland,
        county,
        census_border,
        coast_start_y=2,
        coast_end_y=19,
    )

    assert hybrid[10, 8]  # county.png coastal detail survives.
    assert hybrid[10, 33]  # Census land-border side remains inside.


def test_web_mercator_pixel_maps_zero_to_grid_center():
    assert _web_mercator_pixel(0, 0, (-100, -100, 100, 100), (101, 201)) == (
        100,
        50,
    )


def test_southern_seam_closes_only_the_small_crop_edge_gap():
    county = np.zeros((12, 20), dtype=bool)
    census = np.zeros_like(county)
    county[11, 5] = True
    census[11, 12] = True

    connector, report = _connect_southern_seam(county, census, (9.0, 10.8))

    assert report["passed"] is True
    assert report["county_png_endpoint_px"] == [5, 11]
    assert report["Census_endpoint_px"] == [12, 11]
    assert report["endpoint_gap_px"] == 7.0
    assert report["connector_pixel_count"] == 8
    assert np.all(connector[11, 5:13])
    assert np.count_nonzero(connector) == 8
