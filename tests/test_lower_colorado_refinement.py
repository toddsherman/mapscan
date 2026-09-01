import numpy as np

from mapscan.extraction import _correct_target_web_coordinates
from mapscan.lower_colorado_refinement import _infer_amplitude, _smoothstep


def test_lower_colorado_operation_is_bounded_and_uses_sampling_direction():
    transform = {
        "web_mercator_correction": {
            "grid": {"bounds": [0.0, 0.0, 100.0, 100.0], "width": 101, "height": 101},
            "operations": [
                {
                    "type": "lower_colorado_smoothstep_x",
                    "grid": {"bounds": [0.0, 0.0, 100.0, 100.0], "width": 101, "height": 101},
                    "start_x_px": 50.0,
                    "ramp_width_px": 20.0,
                    "start_y_px": 50.0,
                    "ramp_height_px": 20.0,
                    "target_to_parent_sampling_amplitude_x_px": -6.0,
                }
            ],
        }
    }
    x = np.asarray([40.0, 80.0, 80.0])
    # Pixel y is measured down from max_y, so web y 20 corresponds to pixel y 80.
    y = np.asarray([20.0, 60.0, 20.0])
    corrected_x, corrected_y = _correct_target_web_coordinates(x, y, transform)
    assert np.allclose(corrected_x, [40.0, 80.0, 74.0])
    assert np.allclose(corrected_y, y)


def test_three_channel_amplitude_fit_recovers_six_pixels():
    parameters = {
        "start_x_px": 2600.0,
        "ramp_width_px": 600.0,
        "start_y_px": 3250.0,
        "ramp_height_px": 350.0,
    }
    profiles = []
    for index, (x, y) in enumerate(
        [(3269.0, 3475.0), (3262.0, 3525.0), (3221.0, 3575.0), (3212.0, 3625.0)]
    ):
        weight = float(
            _smoothstep((x - 2600.0) / 600.0)
            * _smoothstep((y - 3250.0) / 350.0)
        )
        observed = -round(6.0 * weight)
        profiles.append(
            {
                "id": f"east_{index:02d}",
                "center": [x, y],
                "accepted": True,
                "estimator_shifts_px": {
                    "blackhat": observed,
                    "saturation": observed,
                    "chroma": observed,
                },
            }
        )
    result = _infer_amplitude(profiles, {0, 1, 2, 3}, parameters)
    assert result["selected_source_to_target_amplitude_x_px"] == 6
    assert result["amplitude_mad_px"] <= 1.0
