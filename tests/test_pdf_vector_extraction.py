from __future__ import annotations

import numpy as np

from mapscan.pdf_vector_extraction import _color_key, _path_contours


def test_color_key_normalizes_native_pdf_colors() -> None:
    assert _color_key((0.10001, 0.2, 0.3, 0.0)) == (0.1, 0.2, 0.3, 0.0)
    assert _color_key(None) is None


def test_path_contours_supports_closed_lines_and_cubic_curves() -> None:
    path = [
        ("m", (1, 1)),
        ("l", (4, 1)),
        ("c", (5, 1), (5, 4), (4, 4)),
        ("l", (1, 4)),
        ("h",),
    ]
    contours, unsupported = _path_contours(path, 2.0, 3.0)
    assert not unsupported
    assert len(contours) == 1
    points = contours[0].reshape((-1, 2))
    assert np.array_equal(points[0], [2, 3])
    assert np.array_equal(points[-1], [2, 12])
    assert len(points) > 5


def test_path_contours_reports_unsupported_operators() -> None:
    contours, unsupported = _path_contours([("q", (1, 1))], 1.0, 1.0)
    assert contours == []
    assert unsupported == {"q"}


def test_path_contours_supports_pdf_cubic_shorthand() -> None:
    path = [
        ("m", (0, 0)),
        ("v", (2, 0), (2, 2)),
        ("y", (0, 2), (0, 0)),
        ("h",),
    ]
    contours, unsupported = _path_contours(path, 1.0, 1.0)
    assert not unsupported
    assert len(contours) == 1
    assert len(contours[0]) >= 9
