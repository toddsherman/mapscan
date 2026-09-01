import numpy as np
import pytest

from mapscan.native_regional_diff import _binary_scores, source_to_reference_points


def _transform(scale=2.0):
    forward = np.asarray([[scale, 0, 3], [0, scale, 5], [0, 0, 1]], dtype=float)
    return {
        "kind": "regular_global_mapbox_registration",
        "source_original_to_reference_pixel_matrix": forward.tolist(),
    }


def test_source_points_use_the_serialized_accepted_transform():
    x, y = source_to_reference_points(
        _transform(), np.asarray([[0.0, 2.0]]), np.asarray([[1.0, 3.0]])
    )

    assert x.tolist() == [[3.0, 7.0]]
    assert y.tolist() == [[7.0, 11.0]]


def test_binary_scores_allow_one_pixel_registration_tolerance():
    source = np.zeros((9, 9), dtype=np.uint8)
    reconstructed = np.zeros_like(source)
    source[2:7, 4] = 1
    reconstructed[2:7, 5] = 1

    score = _binary_scores(source, reconstructed)

    assert score["exact_iou"] == 0.0
    assert score["tolerant_precision"] == pytest.approx(1.0)
    assert score["tolerant_recall"] == pytest.approx(1.0)
    assert score["tolerant_f1"] == pytest.approx(1.0)
