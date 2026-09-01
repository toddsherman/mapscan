from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mapscan.automatic_categorical_extraction import OCRWord
from mapscan.automatic_continuous_numeric_extraction import (
    ContinuousNumericConfig,
    ContinuousNumericLegend,
    NumericRampStop,
    _attempt_contains_forbidden_authority,
    _geographic_roundtrip_metrics,
    _numeric_column,
    _prior_automatic_extraction_state,
    _project_pixels_to_ramp,
    _quantized_band_ids,
    _validate_inputs,
    _values_to_ramp_rgb,
    classify_continuous_numeric,
    detect_continuous_numeric_legend,
    run_continuous_numeric_source_audit,
)
from mapscan.experiment_log import NoHumanExperimentLog


ROOT = Path(__file__).resolve().parents[1]


def _word(text: str, confidence: float, left: int, top: int) -> OCRWord:
    return OCRWord(text, confidence, left, top, max(7, len(text) * 7), 12, (1, 1, top))


def test_numeric_column_uses_meters_not_neighbouring_feet_and_best_psm_duplicate() -> None:
    header = [_word("Elevation", 97, 899, 40), _word("above", 96, 970, 40), _word("sea", 96, 1018, 40), _word("level", 96, 1048, 40)]
    words = list(header)
    meter_values = (5000, 4000, 2000, 1000, 500, 250, 100, 0)
    for index, value in enumerate(meter_values):
        words.append(_word(str(value), 92, 1030, 80 + index * 21))
    # Lower-confidence alternate PSM misreads the 250 row.
    words.append(_word("350", 61, 1031, 80 + 5 * 21))
    # A valid but shorter feet column must not displace the complete sequence.
    for index, value in enumerate((14000, 8000, 4000, 2000, 1000, 500, 200, 0)):
        words.append(_word(str(value), 95, 920, 95 + index * 17))
    selected = _numeric_column(words, header)
    assert [int(word.text) for word in selected] == list(meter_values)
    assert all(word.left >= 1030 for word in selected)


def _synthetic_legend() -> ContinuousNumericLegend:
    samples = np.asarray([0, 50, 100, 150, 200], dtype=np.float32)
    colors = np.asarray(
        [(20, 130, 20), (100, 175, 10), (220, 190, 0), (235, 100, 0), (190, 30, 20)],
        dtype=np.uint8,
    )
    stops = tuple(
        NumericRampStop(float(value), float(100 - index * 20), str(value), 99.0, (0, 0, 10, 10), tuple(int(channel) for channel in color))
        for index, (value, color) in enumerate(zip((200, 100, 0), colors[[4, 2, 0]]))
    )
    return ContinuousNumericLegend(
        "Elevation above sea level",
        99.0,
        "meters",
        stops,
        (0, 0, 20, 100),
        samples,
        colors,
        (0, 110, 20, 20),
        (0, 70, 60),
        99.0,
        "test",
        (),
    )


def test_projection_is_chunked_and_preserves_continuous_and_quantized_semantics() -> None:
    legend = _synthetic_legend()
    rgb = np.tile(legend.sample_rgb, (8, 3, 1))
    values, residual = _project_pixels_to_ramp(
        rgb, legend, luminance_weight=0.25, chunk_pixels=7
    )
    assert np.max(residual) < 1e-5
    assert np.allclose(values[0], np.tile(legend.sample_values_m, 3), atol=1e-3)
    ids, intervals = _quantized_band_ids(
        values, np.zeros(values.shape, dtype=bool), legend
    )
    assert intervals == ((1, 0.0, 100.0), (2, 100.0, 200.0))
    assert set(np.unique(ids)) == {1, 2}


def test_classification_keeps_observed_inferred_occluded_disjoint() -> None:
    legend = _synthetic_legend()
    rgb = np.empty((40, 60, 3), dtype=np.uint8)
    rgb[:, :20] = legend.sample_rgb[0]
    rgb[:, 20:40] = legend.sample_rgb[2]
    rgb[:, 40:] = legend.sample_rgb[4]
    rgb[7:15, 5:55] = (90, 90, 90)  # anti-aliased fringe
    rgb[8:14, 5:55] = (5, 5, 5)  # cartographic ink
    rgb[20:24, 5:55] = (20, 90, 180)  # hydrography
    rgb[30:36, 20:32] = legend.depression_rgb
    domain = np.ones(rgb.shape[:2], dtype=bool)
    classification, scores = classify_continuous_numeric(
        rgb,
        domain,
        legend,
        (),
        config=ContinuousNumericConfig(
            minimum_stop_count=3,
            maximum_ramp_residual=4.0,
            maximum_special_residual=4.0,
            projection_chunk_pixels=113,
        ),
    )
    assert not np.any(classification.observed & classification.inferred)
    assert not np.any(classification.observed & classification.occluded)
    assert not np.any(classification.inferred & classification.occluded)
    assert np.all(
        classification.observed
        | classification.inferred
        | classification.occluded
    )
    assert np.all(classification.occluded[8:14, 5:55])
    assert np.all(classification.occluded[7:15, 5:55])
    assert scores["source_evidenced_antialias_fringe_pixel_count"] > 0
    assert np.all(classification.occluded[20:24, 5:55])
    assert np.any(classification.observed_depression[30:36, 20:32])
    assert np.all(
        classification.completed_depression
        | np.isfinite(classification.completed_values_m)
    )


def test_geographic_roundtrip_reports_supported_cells_and_exact_match() -> None:
    domain = np.ones((12, 12), dtype=bool)
    matched = domain.copy()
    yy, xx = np.indices(domain.shape, dtype=np.float32)
    reports = _geographic_roundtrip_metrics(
        domain, matched, (xx, yy), domain.shape, 3, 3
    )
    assert len(reports) == 9
    assert all(report["source_match_fraction"] == 1.0 for report in reports)


def test_real_elevation_legend_recovers_eight_meter_stops_and_depression(tmp_path: Path) -> None:
    source = ROOT / "examples/elevation.gif"
    rgb = np.asarray(Image.open(source).convert("RGB"))
    legend = detect_continuous_numeric_legend(source, rgb, tmp_path)
    assert [stop.value_m for stop in legend.stops] == [
        5000.0,
        4000.0,
        2000.0,
        1000.0,
        500.0,
        250.0,
        100.0,
        0.0,
    ]
    assert all(stop.ocr_confidence >= 75.0 for stop in legend.stops)
    assert legend.header_confidence >= 90.0
    assert legend.depression_ocr_confidence >= 90.0
    left, top, width, height = legend.ramp_bbox
    assert 969 <= left <= 971
    assert 157 <= top <= 159
    assert 38 <= width <= 40
    assert 152 <= height <= 154
    assert len(legend.sample_values_m) >= 60
    representatives = {stop.value_m: stop.representative_rgb for stop in legend.stops}
    # Horizontal ticks used to contaminate these rows with dark brown/green.
    # Stop colors must now come from adjacent clean cross-column ramp samples.
    assert representatives[500.0][0] >= 200
    assert representatives[250.0][0] >= 180
    assert representatives[100.0][1] >= 160
    assert min(representatives[4000.0]) >= 210
    payload = json.loads((tmp_path / "legend/continuous-numeric-legend.json").read_text())
    assert payload["special_semantics"][0]["numeric_value"] is None

    # A sparse real-source sample must survive numeric -> legend RGB -> numeric
    # reconstruction without the old nonmonotonic/tick-contaminated failures.
    sampled = rgb[::4, ::4]
    projected, residual = _project_pixels_to_ramp(
        sampled,
        legend,
        luminance_weight=ContinuousNumericConfig().luminance_weight,
        chunk_pixels=200_000,
    )
    direct = residual <= ContinuousNumericConfig().maximum_ramp_residual
    reconstructed = _values_to_ramp_rgb(
        projected,
        np.zeros(projected.shape, dtype=bool),
        legend,
        luminance_weight=ContinuousNumericConfig().luminance_weight,
    )
    roundtrip, _ = _project_pixels_to_ramp(
        reconstructed,
        legend,
        luminance_weight=ContinuousNumericConfig().luminance_weight,
        chunk_pixels=200_000,
    )
    assert np.mean(
        np.abs(roundtrip[direct] - projected[direct])
        > ContinuousNumericConfig().maximum_continuous_roundtrip_error_m
    ) <= ContinuousNumericConfig().maximum_semantic_mismatch_fraction


def test_source_only_audit_never_claims_official_extraction(tmp_path: Path) -> None:
    report = run_continuous_numeric_source_audit(
        ROOT / "examples/elevation.gif", tmp_path / "audit"
    )
    assert report["status"] == "source_only_ready_alignment_required"
    assert report["official_extraction_attempt_created"] is False
    assert report["source_only_direct_evidence_pixel_count"] > 180_000
    assert len(report["known_source_ambiguities"]) == 3


def test_official_elevation_log_cannot_extract_without_accepted_alignment() -> None:
    run = ROOT / "runs/mapbox-autonomous-restart-v1/elevation"
    log = NoHumanExperimentLog.load(run / "EXPERIMENT.json")
    # Exercise the rejected precondition without coupling the test to the
    # mutable official run's current alignment phase.
    log.data["alignment"]["accepted_automatic_iteration_count"] = None
    log.data["extraction"]["iterations"] = []
    log.data["extraction"]["accepted_automatic_iteration_count"] = None
    log.data["final"] = {"status": "in_progress", "blocker": None}
    with pytest.raises(ValueError, match="requires accepted alignment"):
        _validate_inputs(
            run / "source-clean/source-adapter.json",
            run / "automatic-alignment/accepted-alignment.json",
            ROOT / "reference/mapbox-light-v11-california-z9-v1/manifest.json",
            log,
        )


def test_blocked_automatic_extraction_history_resumes_at_next_contiguous_ordinal(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.png"
    Image.new("RGB", (2, 2), (0, 0, 0)).save(source)
    log = NoHumanExperimentLog(
        "continuous-retry",
        source,
        mapbox_reference={"reference_id": "test"},
        source_type="continuous_numeric_ramp",
    )
    log.data["alignment"]["accepted_automatic_iteration_count"] = 1
    log.data["extraction"]["iterations"] = [
        {
            "automatic_iteration": 1,
            "counts_toward_automatic_iteration_count": True,
            "decision": "retry",
        },
        {
            "automatic_iteration": 2,
            "counts_toward_automatic_iteration_count": True,
            "decision": "blocked",
        },
    ]
    log.data["final"] = {"status": "blocked", "blocker": "old projector mismatch"}

    assert _prior_automatic_extraction_state(log) == (2, "blocked")
    log.resume_automatic_blocked(reason="corrected projector", producer="test")
    appended = log.record_extraction_iteration(
        scores={"mismatch": 0.0},
        gates={"strict": True, "fixed_point": False},
        decision="retry",
        provenance={
            "actor_kind": "automated",
            "producer": "test",
            "input_kinds": ["authoritative_original_source_pixels"],
            "manual_arrows": False,
            "manual_stamps": False,
            "human_approval": False,
        },
        method="corrected deterministic replay",
    )
    assert appended["automatic_iteration"] == 3
    assert [
        item["automatic_iteration"]
        for item in log.data["extraction"]["iterations"]
    ] == [1, 2, 3]


def test_exact_extraction_payload_preflight_allows_only_false_negative_attestations() -> None:
    payload = {
        "scores": {"semantic_mismatch": 0.0},
        "gates": {"strict": True},
        "provenance": {
            "actor_kind": "automated",
            "producer": "test",
            "input_kinds": ["authoritative_original_source_pixels"],
            "manual_arrows": False,
            "manual_stamps": False,
            "human_approval": False,
        },
        "method": "source-only automatic reconstruction",
        "artifacts": [],
    }
    assert _attempt_contains_forbidden_authority(payload) is None

    payload["provenance"]["manual_arrows"] = True
    assert _attempt_contains_forbidden_authority(payload) == "provenance.manual_arrows"
    payload["provenance"]["manual_arrows"] = False
    payload["scores"]["control_point_count"] = 8
    assert _attempt_contains_forbidden_authority(payload) == "scores.control_point_count"


def test_interrupted_preflight_can_continue_the_same_recorded_resumption(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.png"
    Image.new("RGB", (2, 2), (0, 0, 0)).save(source)
    log = NoHumanExperimentLog(
        "continuous-preflight-continuation",
        source,
        mapbox_reference={"reference_id": "test"},
        source_type="continuous_numeric_ramp",
    )
    log.data["extraction"]["iterations"] = [
        {
            "automatic_iteration": 1,
            "counts_toward_automatic_iteration_count": True,
            "decision": "blocked",
        }
    ]
    log.data["final"] = {"status": "blocked", "blocker": "old algorithm"}
    log.resume_automatic_blocked(
        reason="corrected automatic algorithm",
        producer="mapscan.automatic_continuous_numeric_extraction",
    )
    assert _prior_automatic_extraction_state(log) == (1, None)


def test_continuous_retry_rejects_nonautomatic_or_noncontiguous_history(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.png"
    Image.new("RGB", (2, 2), (0, 0, 0)).save(source)
    log = NoHumanExperimentLog(
        "continuous-invalid-retry",
        source,
        mapbox_reference={"reference_id": "test"},
        source_type="continuous_numeric_ramp",
    )
    log.data["extraction"]["iterations"] = [
        {
            "automatic_iteration": 2,
            "counts_toward_automatic_iteration_count": True,
            "decision": "blocked",
        }
    ]
    log.data["final"] = {"status": "blocked", "blocker": "invalid history"}
    with pytest.raises(ValueError, match="not contiguous"):
        _prior_automatic_extraction_state(log)

    log.data["extraction"]["iterations"][0]["automatic_iteration"] = None
    log.data["extraction"]["iterations"][0][
        "counts_toward_automatic_iteration_count"
    ] = False
    with pytest.raises(ValueError, match="ineligible attempts"):
        _prior_automatic_extraction_state(log)
