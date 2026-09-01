import numpy as np

from mapscan.dither_texture_classifier import (
    build_dither_texture_model,
    classify_dither_texture,
)


def _pattern(primary, secondary, *, offset=0):
    values = np.empty((18, 24, 3), dtype=np.uint8)
    values[:] = primary
    yy, xx = np.indices(values.shape[:2])
    values[(xx + 2 * yy + offset) % 5 >= 3] = secondary
    return values


def test_texture_signatures_preserve_rows_with_the_same_median_rgb():
    primary = (180, 120, 90)
    first = _pattern(primary, (220, 150, 100))
    second = _pattern(primary, (130, 90, 70), offset=1)
    # Both rows have the same per-channel median despite different mixtures.
    assert tuple(np.median(first.reshape(-1, 3), axis=0)) == primary
    assert tuple(np.median(second.reshape(-1, 3), axis=0)) == primary
    rgb = np.zeros((44, 52, 3), dtype=np.uint8)
    rgb[2:20, 2:26] = first
    rgb[24:42, 2:26] = second
    model = build_dither_texture_model(
        rgb,
        ((2, 2, 24, 18), (2, 24, 24, 18)),
    )

    assert model.is_distinguishable
    domain = np.zeros(rgb.shape[:2], dtype=bool)
    domain[5:17, 5:23] = True
    domain[27:39, 5:23] = True
    observed, nearest, _ = classify_dither_texture(rgb, domain, model, 12.0, 1.0)

    assert np.mean(nearest[5:17, 5:23] == 1) > 0.95
    assert np.mean(nearest[27:39, 5:23] == 2) > 0.95
    assert np.count_nonzero(observed) / np.count_nonzero(domain) > 0.90


def test_texture_model_rejects_rows_collapsed_by_the_source_raster():
    pattern = _pattern((180, 120, 90), (220, 150, 100))
    rgb = np.zeros((44, 52, 3), dtype=np.uint8)
    rgb[2:20, 2:26] = pattern
    rgb[24:42, 2:26] = pattern

    model = build_dither_texture_model(
        rgb,
        ((2, 2, 24, 18), (2, 24, 24, 18)),
    )

    assert not model.is_distinguishable
    assert model.ambiguous_pairs == ((1, 2, 0.0),)
