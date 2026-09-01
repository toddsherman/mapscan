import json

import pytest

from mapscan.experiment_log import (
    NoHumanExperimentLog,
    automatic_provenance,
)


REFERENCE = {
    "id": "mapbox-california-reference-v1",
    "state_sha256": "a" * 64,
    "county_sha256": "b" * 64,
    "water_sha256": "c" * 64,
}


def _log(tmp_path):
    source = tmp_path / "source.png"
    source.write_bytes(b"source-image")
    return NoHumanExperimentLog(
        "sample-map",
        source,
        mapbox_reference=REFERENCE,
        created_at="2026-08-30T12:00:00+00:00",
    )


def _automatic(*inputs):
    return automatic_provenance("mapscan-test", inputs)


def test_log_counts_only_automatic_iterations_and_renders_required_fields(tmp_path):
    log = _log(tmp_path)
    first = log.record_alignment_iteration(
        scores={"perimeter_p90_px": 4.2},
        gates={"held_out_regions": {"passed": False, "value": 5, "minimum": 7}},
        decision="retry",
        provenance=_automatic("mapbox_state", "source_image_pixels"),
        method="affine candidate",
    )
    second = log.record_alignment_iteration(
        scores={"perimeter_p90_px": 1.1},
        gates={"held_out_regions": {"passed": True, "value": 7, "minimum": 7}},
        decision="accept",
        provenance=_automatic(
            "mapbox_state", "mapbox_counties", "source_image_pixels"
        ),
        method="thin-plate residual candidate",
    )
    extraction = log.record_extraction_iteration(
        scores={"removed_observed_pixels": 0, "changed_observed_pixels": 0},
        gates={"source_fidelity": True, "regional_comparisons": True},
        decision="accept",
        provenance=_automatic("aligned_source", "legend_pixels", "source_diff"),
        method="Lab classification and deterministic completion",
    )
    log.finalize("complete")
    written = log.write(tmp_path / "experiment.md")

    assert first["automatic_iteration"] == 1
    assert second["automatic_iteration"] == 2
    assert extraction["automatic_iteration"] == 1
    assert log.data["alignment"]["accepted_automatic_iteration_count"] == 2
    assert log.data["extraction"]["accepted_automatic_iteration_count"] == 1
    markdown = (tmp_path / "experiment.md").read_text()
    assert "SHA-256" in markdown
    assert "image/png" in markdown
    assert "Accepted automatic iteration count: **2**" in markdown
    assert "Accepted automatic iteration count: **1**" in markdown
    assert "perimeter_p90_px" in markdown
    assert "source_fidelity" in markdown
    assert "Status: **complete**" in markdown
    assert written["markdown_sha256"]
    assert written["json_sha256"]


@pytest.mark.parametrize(
    "provenance,reason",
    [
        (
            {
                "actor_kind": "automated",
                "producer": "mapscan-test",
                "input_kinds": ["mapbox_state", "manual_arrow"],
                "manual_arrows": True,
                "manual_stamps": False,
                "human_approval": False,
            },
            "manual arrows",
        ),
        (
            {
                "actor_kind": "automated",
                "producer": "mapscan-test",
                "input_kinds": ["aligned_source", "clone_stamp"],
                "manual_arrows": False,
                "manual_stamps": True,
                "human_approval": False,
            },
            "manual stamps",
        ),
        (
            {
                "actor_kind": "human",
                "producer": "reviewer",
                "input_kinds": ["visual_approval"],
                "manual_arrows": False,
                "manual_stamps": False,
                "human_approval": True,
            },
            "actor_kind",
        ),
    ],
)
def test_manual_attempt_is_logged_but_never_counted_or_accepted(
    tmp_path, provenance, reason
):
    log = _log(tmp_path)
    item = log.record_alignment_iteration(
        scores={"visual_fit": 1.0},
        gates={"visual_review": True},
        decision="accept",
        provenance=provenance,
        method="historical intervention",
    )

    assert item["counts_toward_automatic_iteration_count"] is False
    assert item["automatic_iteration"] is None
    assert reason in item["automatic_eligibility_reason"]
    assert log.data["alignment"]["accepted_automatic_iteration_count"] is None
    with pytest.raises(ValueError, match="accepted automatic alignment"):
        log.record_extraction_iteration(
            scores={"diff": 0},
            gates={"diff": True},
            decision="accept",
            provenance=_automatic("aligned_source", "legend_pixels"),
            method="classification",
        )


def test_failed_gates_cannot_be_automatically_accepted(tmp_path):
    log = _log(tmp_path)

    with pytest.raises(ValueError, match="failed gates"):
        log.record_alignment_iteration(
            scores={"p90": 9.0},
            gates={"alignment": False},
            decision="accept",
            provenance=_automatic("mapbox_state", "source_image_pixels"),
            method="bad candidate",
        )


def test_complete_requires_both_automatic_phase_acceptances(tmp_path):
    log = _log(tmp_path)
    log.record_alignment_iteration(
        scores={"p90": 1.0},
        gates={"alignment": True},
        decision="accept",
        provenance=_automatic("mapbox_state", "source_image_pixels"),
        method="accepted candidate",
    )

    with pytest.raises(ValueError, match="alignment and extraction"):
        log.finalize("complete")
    with pytest.raises(ValueError, match="requires a blocker"):
        log.finalize("blocked")
    log.finalize("blocked", "Legend could not be read automatically")
    assert log.data["final"] == {
        "status": "blocked",
        "blocker": "Legend could not be read automatically",
    }


def test_json_log_can_be_resumed_without_changing_source_evidence(tmp_path):
    log = _log(tmp_path)
    original_hash = log.data["source"]["sha256"]
    log.record_alignment_iteration(
        scores={"p90": 1.0},
        gates={"alignment": True},
        decision="accept",
        provenance=_automatic("mapbox_state", "source_image_pixels"),
        method="accepted candidate",
        recorded_at="2026-08-30T12:01:00+00:00",
    )
    log.write(tmp_path / "experiment.md")

    resumed = NoHumanExperimentLog.load(tmp_path / "experiment.json")

    assert resumed.data["source"]["sha256"] == original_hash
    assert resumed.data["alignment"]["accepted_automatic_iteration_count"] == 1
    assert json.loads((tmp_path / "experiment.json").read_text())["schema_version"].endswith("v1")


def test_blocked_log_can_be_reopened_automatically_without_changing_counts(tmp_path):
    log = _log(tmp_path)
    log.record_alignment_iteration(
        scores={"p90": 40.0},
        gates={"alignment": False},
        decision="blocked",
        provenance=_automatic("mapbox_state", "source_image_pixels"),
        method="exhausted first automatic ensemble",
    )
    log.finalize("blocked", "first projection ensemble exhausted")

    event = log.resume_automatic_blocked(
        reason="added source-channel observability classifier",
        producer="automatic-aligner-v2",
        recorded_at="2026-08-30T12:02:00+00:00",
    )

    assert event["previous_blocker"] == "first projection ensemble exhausted"
    assert log.data["final"] == {"status": "in_progress", "blocker": None}
    assert len(log.data["alignment"]["iterations"]) == 1
    assert log.data["alignment"]["accepted_automatic_iteration_count"] is None
    assert "added source-channel observability classifier" in log.render_markdown()
    with pytest.raises(ValueError, match="Only a blocked experiment"):
        log.resume_automatic_blocked(reason="again", producer="automatic-aligner-v2")


def test_failed_log_has_a_separate_explicit_automatic_resumption_path(tmp_path):
    log = _log(tmp_path)
    log.finalize("failed", "automatic OCR adapter raised on a readable swatch")

    event = log.resume_automatic_failed(
        reason="fixed deterministic legend grouping",
        producer="automatic-extractor-v2",
        recorded_at="2026-08-30T12:03:00+00:00",
    )

    assert event["previous_status"] == "failed"
    assert event["previous_blocker"].startswith("automatic OCR")
    assert log.data["final"]["status"] == "in_progress"
    with pytest.raises(ValueError, match="Only a failed experiment"):
        log.resume_automatic_failed(reason="again", producer="automatic-extractor-v2")
