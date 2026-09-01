import numpy as np
from scipy.interpolate import RBFInterpolator

from mapscan.quake_nonlinear_refinement import ThinPlateWarp, _render


def test_warp_scales_after_mapping_original_space() -> None:
    controls = np.asarray([[0, 0], [10, 0], [0, 10], [10, 10]], dtype=float)
    values = np.tile(np.asarray([[2.0, -1.0]]), (4, 1))
    center, span = np.asarray([5.0, 5.0]), np.asarray([10.0, 10.0])
    rbf = RBFInterpolator((controls - center) / span, values, kernel="thin_plate_spline", smoothing=0.0001, degree=1)
    warp = ThinPlateWarp(center, span, controls, values, rbf)
    points = np.asarray([[3.0, 4.0], [8.0, 7.0]])
    assert np.allclose(warp.map_scaled(points, 0.5), (points + [2, -1]) * 0.5)


def test_render_clips_outside_points() -> None:
    rendered = _render(np.asarray([[2.0, 3.0], [-5.0, 1.0]]), (8, 8))
    assert rendered[3, 2]
    assert not rendered[1, 0]
