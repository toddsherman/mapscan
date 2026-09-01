import hashlib
import json

from mapscan.experiment_log import NoHumanExperimentLog, automatic_provenance
from mapscan.restart_failure_report import build_restart_failure_report


REFERENCE = {"id": "mapbox-test-v1", "manifest_sha256": "a" * 64}


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot(run_root, maps):
    path = run_root / "restart-manifest.snapshot.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "mapscan.no-human-restart.v1",
                "mapbox_reference": REFERENCE,
                "maps": maps,
            }
        )
    )
    return path


def _semantic(state_f1, state_p90, county_f1, county_p90):
    def metric(f1, p90):
        return {
            "reference_to_source": {
                "median_px": p90 / 2,
                "p90_px": p90,
                "within_8px_fraction": 0.9,
            },
            "precision": f1,
            "recall": f1,
            "f1": f1,
        }

    return {"state_coast": metric(state_f1, state_p90), "counties": metric(county_f1, county_p90)}


def _iteration(
    log,
    map_dir,
    number,
    projection,
    model,
    decision,
    state_f1,
    state_p90,
    county_f1,
    county_p90,
    failed_gate,
):
    attempt = map_dir / f"automatic-alignment/alignment-{number:02d}-{projection}-{model}"
    attempt.mkdir(parents=True)
    candidate = attempt / "candidate.json"
    candidate.write_text(
        json.dumps(
            {
                "projection": projection,
                "model": model,
                "decision": decision,
            }
        )
    )
    diagnostic = attempt / "semantic-full-line-validation.png"
    diagnostic.write_bytes(b"diagnostic")
    log.record_alignment_iteration(
        scores={
            "objective": 20 - number,
            "projection": {"id": projection},
            "semantic_full_line": _semantic(
                state_f1, state_p90, county_f1, county_p90
            ),
        },
        gates=(
            {failed_gate: True, "projection_round_trip": True}
            if decision == "accept"
            else {
                failed_gate: {
                    "passed": False,
                    "value": state_p90,
                    "maximum": 12.0,
                },
                "projection_round_trip": True,
            }
        ),
        decision=decision,
        provenance=automatic_provenance(
            "report-test", ["source_pixels", "pinned_mapbox_state"]
        ),
        method=f"geographically balanced {projection} {model} Mapbox registration",
        artifacts=[
            {"path": str(candidate), "sha256": _sha256(candidate)},
            {"path": str(diagnostic), "sha256": _sha256(diagnostic)},
        ],
    )


def test_report_summarizes_candidates_best_metrics_gates_and_artifacts(tmp_path):
    run_root = tmp_path / "run"
    run_root.mkdir()
    blocked_source = tmp_path / "blocked.png"
    blocked_source.write_bytes(b"blocked")
    complete_source = tmp_path / "complete.png"
    complete_source.write_bytes(b"complete")
    idle_source = tmp_path / "idle.png"
    idle_source.write_bytes(b"idle")
    _snapshot(
        run_root,
        [
            {
                "id": "blocked",
                "title": "Blocked map",
                "source_type": "categorical_full_state",
            },
            {
                "id": "complete",
                "title": "Complete map",
                "source_type": "categorical_full_state",
            },
            {
                "id": "idle",
                "title": "Idle map",
                "source_type": "continuous_numeric_ramp",
            },
        ],
    )
    blocked_dir = run_root / "blocked"
    blocked = NoHumanExperimentLog(
        "blocked", blocked_source, mapbox_reference=REFERENCE
    )
    _iteration(
        blocked,
        blocked_dir,
        1,
        "web_mercator",
        "similarity",
        "retry",
        0.70,
        9.0,
        0.90,
        4.0,
        "semantic_full_state_support",
    )
    _iteration(
        blocked,
        blocked_dir,
        2,
        "california_albers",
        "regular_affine",
        "blocked",
        0.85,
        6.0,
        0.75,
        8.0,
        "semantic_full_county_support",
    )
    blocked.finalize("blocked", "projection ensemble exhausted")
    blocked.write(blocked_dir / "EXPERIMENT.md")

    complete_dir = run_root / "complete"
    complete = NoHumanExperimentLog(
        "complete", complete_source, mapbox_reference=REFERENCE
    )
    _iteration(
        complete,
        complete_dir,
        1,
        "geographic",
        "similarity",
        "accept",
        0.95,
        2.0,
        0.96,
        2.2,
        "unused_failed_gate",
    )
    complete.record_extraction_iteration(
        scores={"source_diff": 0.0},
        gates={"source_diff": True},
        decision="accept",
        provenance=automatic_provenance(
            "report-test", ["aligned_source", "legend", "source_diff"]
        ),
        method="automatic extraction",
    )
    complete.finalize("complete")
    complete.write(complete_dir / "EXPERIMENT.md")

    idle_dir = run_root / "idle"
    idle = NoHumanExperimentLog("idle", idle_source, mapbox_reference=REFERENCE)
    idle.write(idle_dir / "EXPERIMENT.md")
    source_hashes = {
        path: _sha256(path)
        for path in run_root.rglob("*.json")
    }

    result = build_restart_failure_report(run_root)

    assert result.map_count == 3
    assert result.alignment_candidate_count == 3
    assert result.accepted_alignment_candidate_count == 1
    assert result.blocked_alignment_candidate_count == 1
    assert all(_sha256(path) == digest for path, digest in source_hashes.items())
    report = result.report_path.read_text()
    assert "Blocked map" in report
    assert "#2 california_albers/regular_affine; F1 0.85" in report
    assert "#1 web_mercator/similarity; F1 0.9" in report
    assert "semantic_full_county_support (value=6, max=12)" in report
    assert "semantic-full-line-validation.png" in report
    assert "No automatic alignment candidate has been recorded." in report
    assert "| 1 | 1 | 0 | 1 |" in report


def test_report_marks_missing_experiment_without_mutating_snapshot(tmp_path):
    run_root = tmp_path / "run"
    run_root.mkdir()
    snapshot = _snapshot(
        run_root,
        [{"id": "missing", "title": "Missing map", "source_type": "categorical_sparse"}],
    )
    before = snapshot.read_bytes()
    output = tmp_path / "reports" / "failures.md"

    result = build_restart_failure_report(run_root, output)

    assert result.report_path == output.resolve()
    assert snapshot.read_bytes() == before
    report = output.read_text()
    assert "Missing map" in report
    assert "EXPERIMENT.json is missing" in report


def test_diagnostic_links_never_escape_the_inspected_run(tmp_path):
    run_root = tmp_path / "run"
    run_root.mkdir()
    source = tmp_path / "source.png"
    source.write_bytes(b"source")
    outside = tmp_path / "candidate.json"
    outside.write_text("{}")
    _snapshot(
        run_root,
        [{"id": "sample", "title": "Sample", "source_type": "categorical_full_state"}],
    )
    log = NoHumanExperimentLog("sample", source, mapbox_reference=REFERENCE)
    log.record_alignment_iteration(
        scores={"objective": 1.0},
        gates={"fit": False},
        decision="blocked",
        provenance=automatic_provenance(
            "report-test", ["source_pixels", "pinned_mapbox_state"]
        ),
        method="automatic candidate",
        artifacts=[{"path": str(outside), "sha256": _sha256(outside)}],
    )
    log.finalize("blocked", "no match")
    log.write(run_root / "sample/EXPERIMENT.md")

    report = build_restart_failure_report(run_root).report_path.read_text()

    assert str(outside) not in report
    assert "| — |" in report
