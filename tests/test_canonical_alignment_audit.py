import numpy as np

from mapscan.canonical_alignment_audit import (
    _dark_cartographic_strokes,
    _regional_reports,
    _summary,
)


def test_dark_strokes_exclude_darkest_population_red():
    rgb = np.asarray([[[0, 0, 0], [160, 0, 0], [70, 70, 70], [0, 128, 255]]])
    selected = _dark_cartographic_strokes(rgb, np.ones((1, 4), dtype=bool))
    assert selected.tolist() == [[True, False, True, False]]


def test_summary_reports_alignment_thresholds():
    report = _summary(np.asarray([0.0, 1.0, 2.0, 9.0]))
    assert report["median_px"] == 1.5
    assert report["within_3px_fraction"] == 0.75
    assert report["within_8px_fraction"] == 0.75


def test_regional_reports_cover_all_line_pixels():
    line = np.zeros((41, 41), dtype=bool)
    line[2:39, 2] = True
    line[2:39, 38] = True
    line[2, 2:39] = True
    line[38, 2:39] = True
    distance = np.zeros(line.shape, dtype=float)
    reports = _regional_reports(line, distance)
    assert len(reports) == 8
    assert sum(item["count"] for item in reports) == int(np.count_nonzero(line))
    assert all(item["p90_px"] == 0.0 for item in reports)
