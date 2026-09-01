import numpy as np

from mapscan.native_alignment_validation import (
    _balanced_gate,
    _candidate_translation,
    _normalized_line_metrics,
)


def test_normalized_metrics_preserve_accepted_working_pixel_units() -> None:
    source = np.zeros((20, 20), dtype=bool)
    rendered = np.zeros((20, 20), dtype=bool)
    source[10, 4:16] = True
    rendered[12, 4:16] = True
    metrics = _normalized_line_metrics(
        rendered,
        source,
        accepted_working_scale=0.25,
        validation_scale=0.5,
        source_scope=np.ones_like(source),
        overlap_tolerance_working_px=5.0,
    )
    assert metrics["native_scale_median_px"] == 2.0
    assert metrics["working_equivalent_median_px"] == 1.0


def test_balanced_gate_rejects_one_failed_axis() -> None:
    reports = []
    for row in range(3):
        for column in range(3):
            reports.append(
                {
                    "row": row,
                    "column": column,
                    "passed": column != 2,
                }
            )
    gate = _balanced_gate(reports)
    assert gate["passed"] is False
    assert gate["column_pass_fractions"]["2"] == 0.0


def test_failed_candidate_is_read_only_translation_seed() -> None:
    source = np.zeros((40, 40), dtype=bool)
    rendered = np.zeros((40, 40), dtype=bool)
    source[10:35, 12] = True
    rendered[10:35, 10] = True
    candidate = _candidate_translation(rendered, source, 1.0)
    # Tiny test support is intentionally below the production confidence floor.
    assert candidate["status"] == "unavailable"
