import numpy as np

from mapscan.east_anchored_refinement import (
    _east_anchored_parameters,
    fit_east_anchored_horizontal_scale,
)
from mapscan.extraction import _normalized_to_source


def test_east_anchored_scale_preserves_mean_east_x_and_every_y_coordinate():
    parameters = {
        "center_x_fraction": 0.54,
        "center_y_fraction": 0.49,
        "state_height_fraction": 0.84,
        "x_scale_ratio": 1.01,
        "rotation_degrees": 0.25,
        "x_shear": 0.027,
    }
    eastern = np.asarray(
        [[0.18, -0.45], [0.20, -0.1], [0.28, 0.2], [0.40, 0.45]],
        dtype=np.float64,
    )
    shape = (1200, 1117)

    fitted, report = _east_anchored_parameters(parameters, shape, eastern, 0.99)
    before = _normalized_to_source(eastern, parameters, shape)
    after = _normalized_to_source(eastern, fitted, shape)

    assert np.isclose(np.mean(after[:, 0]), np.mean(before[:, 0]))
    assert np.allclose(after[:, 1], before[:, 1])
    assert np.isclose(fitted["x_scale_ratio"], parameters["x_scale_ratio"] * 0.99)
    assert fitted["center_y_fraction"] == parameters["center_y_fraction"]
    assert fitted["state_height_fraction"] == parameters["state_height_fraction"]
    assert fitted["rotation_degrees"] == parameters["rotation_degrees"]
    assert fitted["x_shear"] == parameters["x_shear"]
    assert report["source_y_displacement_px"] == 0.0


def test_required_width_reduction_rejects_a_range_that_can_widen(tmp_path):
    try:
        fit_east_anchored_horizontal_scale(
            tmp_path / "unused.png",
            tmp_path / "unused.json",
            tmp_path / "reference",
            tmp_path / "canonical.json",
            tmp_path / "output",
            minimum_multiplier=0.99,
            maximum_multiplier=1.01,
            candidate_count=3,
            require_rendered_width_reduction=True,
        )
    except ValueError as error:
        assert "greater than 1" in str(error)
    else:
        raise AssertionError("A widening-capable source-reset range was accepted")
