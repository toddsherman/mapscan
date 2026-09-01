import numpy as np

from mapscan.categorical_comparison import (
    _observed_source_masks,
    _palette,
    _render,
    region_pixel_bounds,
)
from mapscan.extraction import (
    _restore_observed_class_evidence_from_external_water,
    _source_context_exclusion,
)


def test_render_preserves_exact_legend_colors_and_white_nodata():
    categories = [
        {"legend_rgb": [255, 255, 255]},
        {"legend_rgb": [224, 0, 0]},
    ]
    values = np.asarray([[0, 1, 2]], dtype=np.uint8)
    rendered = _render(values, _palette(categories))
    assert rendered.tolist() == [[[255, 255, 255], [255, 255, 255], [224, 0, 0]]]


def test_blue_source_context_excludes_water_but_not_dark_red():
    rgb = np.asarray([[[0, 128, 255], [160, 0, 0], [80, 80, 80]]], dtype=np.uint8)
    mask, report = _source_context_exclusion(
        rgb,
        np.ones((1, 3), dtype=bool),
        {"blue_dominant_water": {}},
    )
    assert mask.tolist() == [[True, False, False]]
    assert report["source_pixel_count"] == 1


def test_direct_class_evidence_outranks_external_water_when_enabled():
    interior = np.asarray([[True, False, False]], dtype=bool)
    water = np.asarray([[False, True, True]], dtype=bool)
    classes = np.asarray([[1, 0, 7]], dtype=np.uint8)
    restored_interior, remaining_water, restored = (
        _restore_observed_class_evidence_from_external_water(
            interior, water, classes, True
        )
    )
    assert restored.tolist() == [[False, False, True]]
    assert restored_interior.tolist() == [[True, False, True]]
    assert remaining_water.tolist() == [[False, True, False]]


def test_external_water_remains_authoritative_without_explicit_opt_in():
    interior = np.asarray([[True, False]], dtype=bool)
    water = np.asarray([[False, True]], dtype=bool)
    classes = np.asarray([[1, 7]], dtype=np.uint8)
    restored_interior, remaining_water, restored = (
        _restore_observed_class_evidence_from_external_water(
            interior, water, classes, False
        )
    )
    assert np.array_equal(restored_interior, interior)
    assert np.array_equal(remaining_water, water)
    assert not np.any(restored)


def test_comparison_detects_observed_source_pixels_hidden_by_water_mask():
    preclip = np.asarray([[1, 2, 0]], dtype=np.uint8)
    interior = np.asarray([[True, False, False]], dtype=bool)
    water = np.asarray([[False, True, True]], dtype=bool)
    observed, retained, removed = _observed_source_masks(preclip, interior, water)
    assert observed.tolist() == [[True, True, False]]
    assert retained.tolist() == [[True, False, False]]
    assert removed.tolist() == [[False, True, False]]


def test_geographic_review_region_converts_to_native_pixel_window():
    grid = {
        "crs": "EPSG:4326",
        "bounds": [0.0, 0.0, 10.0, 10.0],
        "width": 100,
        "height": 100,
    }
    region = {"id": "sample", "bounds_wgs84": [2.0, 3.0, 5.0, 7.0]}
    assert region_pixel_bounds(region, grid) == (20, 30, 50, 70)
