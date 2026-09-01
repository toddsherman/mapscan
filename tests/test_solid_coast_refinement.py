import numpy as np

from mapscan.solid_coast_refinement import (
    _coast_controls,
    _component_pin_points,
    _eastern_row_drift,
    _west_gap_summary,
    _western_rows,
)


def test_western_rows_and_controls_force_only_leftward_correction():
    source = np.zeros((120, 100), dtype=bool)
    target = np.zeros_like(source)
    source[10:110, 30] = True
    target[10:110, 23] = True
    rows = _western_rows(source, target, start_y=10, end_y=110)
    controls = _coast_controls(
        rows,
        count=5,
        smoothing_half_window=5,
        minimum_left_shift_px=4.0,
        maximum_left_shift_px=20.0,
    )
    assert len(controls) == 5
    assert all(item["applied_target_minus_source_x_px"] == -7.0 for item in controls)
    assert all(item["target"][0] < item["current"][0] for item in controls)


def test_leftward_instruction_overrides_small_opposite_measurement():
    rows = np.asarray(
        [[float(y), 20.0, 22.0, 2.0] for y in range(20, 100)], dtype=float
    )
    controls = _coast_controls(
        rows,
        count=4,
        smoothing_half_window=4,
        minimum_left_shift_px=3.0,
        maximum_left_shift_px=8.0,
    )
    assert all(item["applied_target_minus_source_x_px"] == -3.0 for item in controls)


def test_component_pins_hold_center_and_four_extrema():
    mask = np.zeros((20, 30), dtype=bool)
    mask[5:15, 8:22] = True
    pins = _component_pin_points(mask)
    assert {name for name, _ in pins} == {"center", "west", "east", "north", "south"}


def test_white_gap_score_ignores_safe_leftward_overreach():
    rows = np.asarray(
        [
            [10.0, 30.0, 25.0, -5.0],
            [11.0, 24.0, 25.0, 1.0],
            [12.0, 20.0, 25.0, 5.0],
        ]
    )
    summary = _west_gap_summary(rows)
    assert summary["median_px"] == 0.0
    assert summary["max_px"] == 5.0
    assert summary["fraction_with_gap"] == 1 / 3


def test_eastern_drift_reports_a_stationary_envelope():
    before = np.zeros((20, 30), dtype=bool)
    after = np.zeros_like(before)
    before[:, 24] = True
    after[:, 24] = True
    after[5, 23] = True
    summary = _eastern_row_drift(before, after)
    assert summary["p90_absolute_px"] == 0.0
    assert summary["unchanged_fraction"] == 1.0
