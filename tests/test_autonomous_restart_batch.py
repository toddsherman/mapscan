import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from mapscan.autonomous_restart_batch import (
    AutonomousRestartBatchConfig,
    AutonomousRestartBatchRunner,
    _contains_forbidden_evidence,
)
from mapscan.experiment_log import NoHumanExperimentLog, automatic_provenance
from mapscan.restart_registry import SCHEMA_VERSION as RESTART_SCHEMA
from mapscan.source_working_raster import prepare_source_working_raster


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_accepted_candidate_preflight_rejects_old_control_point_key_only():
    old_payload = {
        "scores": {
            "source_alignment_hypothesis": {
                "source_graticule_control_point_count": 0,
            }
        }
    }
    corrected_payload = {
        "scores": {
            "source_alignment_hypothesis": {
                "detected_graticule_intersection_count": 0,
            }
        }
    }

    assert _contains_forbidden_evidence(old_payload) == (
        "scores.source_alignment_hypothesis.source_graticule_control_point_count"
    )
    assert _contains_forbidden_evidence(corrected_payload) is None


def _fixture(tmp_path, source_type="categorical_full_state"):
    source = tmp_path / "source.png"
    Image.new("RGB", (30, 40), (120, 180, 80)).save(source)
    reference_root = tmp_path / "reference"
    reference_root.mkdir()
    reference_manifest = reference_root / "manifest.json"
    reference_manifest.write_text('{"pinned":"synthetic-mapbox"}\n')
    restart = tmp_path / "restart.json"
    restart.write_text(
        json.dumps(
            {
                "schema_version": RESTART_SCHEMA,
                "mapbox_reference": {
                    "id": "mapbox-synthetic-v1",
                    "provider": "mapbox",
                    "root": str(reference_root),
                    "manifest_sha256": _sha256(reference_manifest),
                },
                "maps": [
                    {
                        "id": "sample",
                        "title": "Sample",
                        "source": str(source),
                        "source_type": source_type,
                    }
                ],
            }
        )
    )
    return source, reference_manifest, restart, tmp_path / "run"


def test_default_source_family_extractors_are_registered(tmp_path):
    _, _, restart, run_root = _fixture(tmp_path)
    runner = AutonomousRestartBatchRunner(restart, run_root)

    assert set(runner.specialized_extraction_runners) == {
        "continuous_numeric_ramp",
        "named_linear_and_polygon_features_without_legend",
        "native_pdf_vector_categorical",
        "ordered_gradient_bands",
        "overlapping_chromatic_and_grayscale",
        "overlapping_feature_and_categorical",
    }
    assert (
        runner.specialized_extraction_runners[
            "named_linear_and_polygon_features_without_legend"
        ].__module__
        == "mapscan.river_source_extraction"
    )
    assert (
        runner.specialized_extraction_runners[
            "overlapping_chromatic_and_grayscale"
        ].__module__
        == "mapscan.storm_layer_extraction"
    )
    assert (
        runner.specialized_extraction_runners["ordered_gradient_bands"].__module__
        == "mapscan.automatic_ordered_band_extraction"
    )


def test_accepted_alignment_may_pin_original_or_decoded_working_source(tmp_path):
    source, _, restart, run_root = _fixture(tmp_path)
    working = prepare_source_working_raster(source, tmp_path / "source-clean")
    runner = AutonomousRestartBatchRunner(restart, run_root)
    accepted = tmp_path / "accepted-alignment.json"

    for source_sha256 in (working.source_sha256, working.working_raster_sha256):
        accepted.write_text(
            json.dumps({"decision": "accept", "source_sha256": source_sha256})
        )
        runner._validate_accepted_alignment(accepted, working)


def test_accepted_extraction_is_found_in_later_immutable_run_by_alignment_hash(
    tmp_path,
):
    source, _, restart, run_root = _fixture(tmp_path)
    working = prepare_source_working_raster(source, tmp_path / "source-clean")
    runner = AutonomousRestartBatchRunner(restart, run_root)
    alignment = tmp_path / "accepted-alignment.json"
    alignment.write_text('{"decision":"accept"}\n')
    first = tmp_path / "map/extraction-run-01"
    first.mkdir(parents=True)
    (first / "iteration-only.json").write_text('{"decision":"blocked"}\n')
    second = tmp_path / "map/extraction-run-02"
    second.mkdir()
    pointer = second / "accepted-extraction.json"
    pointer.write_text(
        json.dumps(
            {
                "status": "accepted",
                "source": {"sha256": working.source_sha256},
                "alignment": {
                    "path": str(alignment),
                    "sha256": _sha256(alignment),
                },
            }
        )
    )

    assert runner._accepted_extraction(tmp_path / "map", working, alignment) == pointer


def _native_pdf_fixture(tmp_path):
    source = tmp_path / "geologic.pdf"
    Image.new("RGB", (36, 48), (220, 205, 175)).save(source, "PDF", resolution=72)
    reference_root = tmp_path / "reference"
    reference_root.mkdir()
    reference_manifest = reference_root / "manifest.json"
    reference_manifest.write_text('{"pinned":"synthetic-mapbox"}\n')
    restart = tmp_path / "restart.json"
    restart.write_text(
        json.dumps(
            {
                "schema_version": RESTART_SCHEMA,
                "mapbox_reference": {
                    "id": "mapbox-synthetic-v1",
                    "provider": "mapbox",
                    "root": str(reference_root),
                    "manifest_sha256": _sha256(reference_manifest),
                },
                "maps": [
                    {
                        "id": "geologic",
                        "title": "Geologic",
                        "source": str(source),
                        "source_type": "native_pdf_vector_categorical",
                    }
                ],
            }
        )
    )
    return source, reference_manifest, restart, tmp_path / "run"


def _alignment_runner(calls):
    def run(source, reference, output, log, **_kwargs):
        calls.append(("alignment", Path(source), Path(reference)))
        output.mkdir(parents=True, exist_ok=True)
        attempt = output / "alignment-01-web_mercator-similarity"
        attempt.mkdir()
        payload = {
            "schema_version": 1,
            "decision": "accept",
            "source_sha256": _sha256(Path(source)),
            "mapbox_reference": {"manifest_sha256": _sha256(Path(reference))},
            "transform": {"kind": "projection_aware_mapbox_registration"},
        }
        candidate = attempt / "candidate.json"
        candidate.write_text(json.dumps(payload))
        (output / "accepted-alignment.json").write_text(json.dumps(payload))
        log.record_alignment_iteration(
            scores={"state_coast_p90_px": 2.0},
            gates={"mapbox_holdout": True},
            decision="accept",
            provenance=automatic_provenance(
                "test-alignment",
                ["original_source_image_pixels", "pinned_mapbox_state_coast"],
            ),
            method="automatic synthetic alignment",
            artifacts=[{"path": str(candidate), "sha256": _sha256(candidate)}],
        )
        return SimpleNamespace(
            status="pass", stop_reason="alignment gates passed", accepted=object()
        )

    return run


def _extraction_runner(calls):
    def run(
        source,
        alignment,
        reference,
        output,
        log,
        _markdown,
        _json,
        **_kwargs,
    ):
        calls.append(("extraction", Path(source), Path(reference)))
        output.mkdir(parents=True, exist_ok=True)
        first = output / "extraction-01"
        first.mkdir()
        first_report = first / "iteration.json"
        first_report.write_text('{"decision":"retry"}\n')
        log.record_extraction_iteration(
            scores={"source_diff": 0.10},
            gates={"source_diff": False},
            decision="retry",
            provenance=automatic_provenance(
                "test-extraction", ["source_clean_pixels", "source_diff"]
            ),
            method="automatic categorical extraction",
            artifacts=[{"path": str(first_report), "sha256": _sha256(first_report)}],
        )
        log.write(_markdown, _json)
        second = output / "extraction-02"
        second.mkdir()
        second_report = second / "iteration.json"
        second_report.write_text('{"decision":"accept"}\n')
        log.record_extraction_iteration(
            scores={"source_diff": 0.0},
            gates={"source_diff": True},
            decision="accept",
            provenance=automatic_provenance(
                "test-extraction", ["source_clean_pixels", "source_diff"]
            ),
            method="automatic categorical extraction",
            artifacts=[{"path": str(second_report), "sha256": _sha256(second_report)}],
        )
        log.write(_markdown, _json)
        accepted = {
            "status": "accepted",
            "source": {"path": str(source), "sha256": _sha256(Path(source))},
            "alignment": {
                "path": str(alignment),
                "sha256": _sha256(Path(alignment)),
            },
        }
        (output / "accepted-extraction.json").write_text(json.dumps(accepted))
        log.finalize("complete")
        log.write(_markdown, _json)
        return SimpleNamespace(stop_reason="extraction gates passed")

    return run


def test_batch_uses_source_clean_raster_and_checkpoints_every_attempt(
    tmp_path, monkeypatch
):
    source, reference, restart, run_root = _fixture(tmp_path)
    calls = []
    refreshes = []
    from mapscan.restart_registry import NoHumanRestartRegistry

    original_refresh = NoHumanRestartRegistry.refresh_index

    def refresh(registry):
        refreshes.append(True)
        return original_refresh(registry)

    monkeypatch.setattr(NoHumanRestartRegistry, "refresh_index", refresh)
    result = AutonomousRestartBatchRunner(
        restart,
        run_root,
        alignment_runner=_alignment_runner(calls),
        extraction_runner=_extraction_runner(calls),
    ).run()

    assert [call[0] for call in calls] == ["alignment", "extraction"]
    assert all(call[1].name == "working-raster.png" for call in calls)
    assert all(call[2] == reference.resolve() for call in calls)
    assert len(refreshes) >= 6  # initialize, three attempts, finalize, final refresh
    log = json.loads((run_root / "sample/EXPERIMENT.json").read_text())
    assert log["source"]["sha256"] == _sha256(source)
    assert log["alignment"]["accepted_automatic_iteration_count"] == 1
    assert log["extraction"]["accepted_automatic_iteration_count"] == 2
    assert log["final"]["status"] == "complete"
    assert all(
        item["counts_toward_automatic_iteration_count"]
        for phase in ("alignment", "extraction")
        for item in log[phase]["iterations"]
    )
    assert (run_root / "sample/source-clean/source-adapter.json").is_file()
    assert "| 1 | 1 | 2 | 2 | complete |" in result.index_path.read_text()


def test_unsupported_data_model_aligns_then_blocks_without_extraction(tmp_path):
    _, _, restart, run_root = _fixture(tmp_path, "unsupported_numeric_model")
    calls = []

    def extraction_must_not_run(*_args, **_kwargs):
        raise AssertionError("unsupported data model reached categorical extraction")

    result = AutonomousRestartBatchRunner(
        restart,
        run_root,
        alignment_runner=_alignment_runner(calls),
        extraction_runner=extraction_must_not_run,
    ).run()

    assert [call[0] for call in calls] == ["alignment"]
    outcome = result.outcomes[0]
    assert outcome.status == "blocked"
    assert outcome.phase == "extraction"
    assert "Unsupported autonomous extraction data model" in outcome.message
    log = json.loads((run_root / "sample/EXPERIMENT.json").read_text())
    assert log["alignment"]["accepted_automatic_iteration_count"] == 1
    assert log["extraction"]["iterations"] == []


def test_source_specific_extractor_uses_original_source_log_and_manifest(tmp_path):
    source, reference, restart, run_root = _fixture(
        tmp_path, "overlapping_feature_and_categorical"
    )
    calls = []

    def specialized(
        source_manifest,
        alignment,
        mapbox_manifest,
        output,
        log,
        markdown,
        json_path,
    ):
        calls.append(
            (
                Path(source_manifest),
                Path(alignment),
                Path(mapbox_manifest),
                log.data["source"]["sha256"],
            )
        )
        assert Path(source_manifest).name == "source-adapter.json"
        assert log.data["source"]["sha256"] == _sha256(source)
        output.mkdir(parents=True)
        iteration = output / "extraction-01"
        iteration.mkdir()
        report = iteration / "iteration.json"
        report.write_text('{"decision":"accept"}\n')
        log.record_extraction_iteration(
            scores={"source_roundtrip": 1.0},
            gates={"source_roundtrip": True},
            decision="accept",
            provenance=automatic_provenance(
                "test-specialized-extraction",
                ["authoritative_original_pixels", "source_reconstruction_diff"],
            ),
            method="source-specific overlapping-layer extraction",
            artifacts=[{"path": str(report), "sha256": _sha256(report)}],
        )
        accepted = {
            "status": "accepted",
            "source": {"path": str(source), "sha256": _sha256(source)},
            "alignment": {
                "path": str(alignment),
                "sha256": _sha256(Path(alignment)),
            },
        }
        (output / "accepted-extraction.json").write_text(json.dumps(accepted))
        log.finalize("complete")
        log.write(markdown, json_path)
        return SimpleNamespace(stop_reason="specialized extraction gates passed")

    result = AutonomousRestartBatchRunner(
        restart,
        run_root,
        alignment_runner=_alignment_runner([]),
        extraction_runner=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("generic extraction should not run")
        ),
        specialized_extraction_runners={
            "overlapping_feature_and_categorical": specialized
        },
    ).run()

    assert len(calls) == 1
    assert calls[0][0] == (run_root / "sample/source-clean/source-adapter.json").resolve()
    assert calls[0][2] == reference.resolve()
    assert result.outcomes[0].status == "complete"
    log = json.loads((run_root / "sample/EXPERIMENT.json").read_text())
    assert log["extraction"]["accepted_automatic_iteration_count"] == 1
    artifacts = log["extraction"]["iterations"][0]["artifacts"]
    assert any(item.get("kind") == "source_clean_adapter_manifest" for item in artifacts)


def test_alignment_block_stops_before_extraction_and_is_logged(tmp_path):
    _, _, restart, run_root = _fixture(tmp_path)
    calls = []

    def blocked_alignment(source, reference, output, log, **_kwargs):
        calls.append("alignment")
        output.mkdir(parents=True)
        log.record_alignment_iteration(
            scores={"p90": 90.0},
            gates={"mapbox_holdout": False},
            decision="blocked",
            provenance=automatic_provenance(
                "test-alignment", ["source_clean_pixels", "pinned_mapbox_state"]
            ),
            method="automatic alignment",
        )
        return SimpleNamespace(
            status="blocked", stop_reason="ensemble exhausted", accepted=None
        )

    result = AutonomousRestartBatchRunner(
        restart,
        run_root,
        alignment_runner=blocked_alignment,
        extraction_runner=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("extraction should not run")
        ),
    ).run()

    assert calls == ["alignment"]
    assert result.outcomes[0].status == "blocked"
    assert result.outcomes[0].phase == "alignment"
    assert result.outcomes[0].alignment_automatic_iterations == 1


def test_explicit_algorithm_retry_reopens_blocked_map_and_preserves_attempts(tmp_path):
    _, _, restart, run_root = _fixture(tmp_path)

    def first_blocked(source, reference, output, log, **_kwargs):
        output.mkdir(parents=True)
        log.record_alignment_iteration(
            scores={"p90": 90.0},
            gates={"mapbox_holdout": False},
            decision="blocked",
            provenance=automatic_provenance(
                "test-alignment-v1", ["source_clean_pixels", "pinned_mapbox_state"]
            ),
            method="first automatic ensemble",
        )
        return SimpleNamespace(
            status="blocked", stop_reason="first ensemble exhausted", accepted=None
        )

    first = AutonomousRestartBatchRunner(
        restart,
        run_root,
        alignment_runner=first_blocked,
        extraction_runner=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("extraction should not run before alignment")
        ),
    ).run()
    assert first.outcomes[0].status == "blocked"

    calls = []

    def improved_alignment(source, reference, output, log, **_kwargs):
        calls.append("alignment-v2")
        attempt = output / "alignment-02-web_mercator-similarity"
        attempt.mkdir()
        payload = {
            "schema_version": 1,
            "decision": "accept",
            "source_sha256": _sha256(Path(source)),
            "mapbox_reference": {"manifest_sha256": _sha256(Path(reference))},
            "transform": {"kind": "projection_aware_mapbox_registration"},
        }
        candidate = attempt / "candidate.json"
        candidate.write_text(json.dumps(payload))
        (output / "accepted-alignment.json").write_text(json.dumps(payload))
        log.record_alignment_iteration(
            scores={"p90": 2.0},
            gates={"mapbox_holdout": True},
            decision="accept",
            provenance=automatic_provenance(
                "test-alignment-v2", ["source_clean_pixels", "pinned_mapbox_state"]
            ),
            method="improved automatic ensemble",
        )
        return SimpleNamespace(
            status="pass", stop_reason="alignment gates passed", accepted=object()
        )

    second = AutonomousRestartBatchRunner(
        restart,
        run_root,
        config=AutonomousRestartBatchConfig(
            retry_blocked_map_ids=frozenset({"sample"}),
            retry_reason="added automatic source observability model",
            retry_producer="test-alignment-v2",
        ),
        alignment_runner=improved_alignment,
        extraction_runner=_extraction_runner(calls),
    ).run()

    assert second.outcomes[0].status == "complete"
    persisted = json.loads((run_root / "sample/EXPERIMENT.json").read_text())
    assert [
        item["automatic_iteration"] for item in persisted["alignment"]["iterations"]
    ] == [1, 2]
    assert persisted["automatic_resumptions"][0]["previous_blocker"].endswith(
        "first ensemble exhausted"
    )
    assert "added automatic source observability model" in (
        run_root / "sample/EXPERIMENT.md"
    ).read_text()


def test_explicit_code_fix_retry_reopens_failed_extraction_only(tmp_path):
    _, _, restart, run_root = _fixture(tmp_path)
    calls = []

    def extraction_bug(*_args, **_kwargs):
        raise ValueError("readable legend swatch was not grouped")

    first = AutonomousRestartBatchRunner(
        restart,
        run_root,
        alignment_runner=_alignment_runner(calls),
        extraction_runner=extraction_bug,
    ).run()
    assert first.outcomes[0].status == "failed"
    assert [call[0] for call in calls] == ["alignment"]

    second = AutonomousRestartBatchRunner(
        restart,
        run_root,
        config=AutonomousRestartBatchConfig(
            retry_failed_map_ids=frozenset({"sample"}),
            retry_reason="fixed deterministic legend grouping",
            retry_producer="test-extraction-v2",
        ),
        alignment_runner=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("accepted alignment should be retained")
        ),
        extraction_runner=_extraction_runner(calls),
    ).run()

    assert second.outcomes[0].status == "complete"
    persisted = json.loads((run_root / "sample/EXPERIMENT.json").read_text())
    assert len(persisted["alignment"]["iterations"]) == 1
    assert persisted["automatic_resumptions"][0]["previous_status"] == "failed"
    assert [call[0] for call in calls] == ["alignment", "extraction"]


def test_complete_map_resumes_without_rerunning_any_stage(tmp_path):
    _, _, restart, run_root = _fixture(tmp_path)
    calls = []
    first = AutonomousRestartBatchRunner(
        restart,
        run_root,
        alignment_runner=_alignment_runner(calls),
        extraction_runner=_extraction_runner(calls),
    ).run()
    assert first.outcomes[0].status == "complete"

    def must_not_run(*_args, **_kwargs):
        raise AssertionError("terminal map stage was rerun")

    resumed = AutonomousRestartBatchRunner(
        restart,
        run_root,
        alignment_runner=must_not_run,
        extraction_runner=must_not_run,
    ).run()

    assert resumed.outcomes[0].status == "complete"
    assert resumed.outcomes[0].phase == "terminal"
    assert calls == [
        ("alignment", run_root / "sample/source-clean/working-raster.png", first.mapbox_manifest_path),
        ("extraction", run_root / "sample/source-clean/working-raster.png", first.mapbox_manifest_path),
    ]


def test_resume_continues_automatic_ordinals_without_overwriting_prior_attempts(
    tmp_path,
):
    _, _, restart, run_root = _fixture(tmp_path)
    from mapscan.restart_registry import NoHumanRestartRegistry

    registry = NoHumanRestartRegistry(restart, run_root)
    registry.initialize()
    log_path = run_root / "sample/EXPERIMENT.json"
    log = NoHumanExperimentLog.load(log_path)
    log.record_alignment_iteration(
        scores={"p90": 12.0},
        gates={"mapbox_holdout": False},
        decision="retry",
        provenance=automatic_provenance(
            "earlier-automatic-run", ["source_clean_pixels", "pinned_mapbox_state"]
        ),
        method="first automatic candidate",
    )
    log.write(run_root / "sample/EXPERIMENT.md", log_path)
    calls = []

    result = AutonomousRestartBatchRunner(
        restart,
        run_root,
        alignment_runner=_alignment_runner(calls),
        extraction_runner=_extraction_runner(calls),
    ).run()

    assert result.outcomes[0].status == "complete"
    resumed_log = json.loads(log_path.read_text())
    assert [
        item["automatic_iteration"] for item in resumed_log["alignment"]["iterations"]
    ] == [1, 2]
    assert resumed_log["alignment"]["accepted_automatic_iteration_count"] == 2


def test_manual_attempt_is_never_consumed_by_batch_runner(tmp_path):
    _, _, restart, run_root = _fixture(tmp_path)
    from mapscan.restart_registry import NoHumanRestartRegistry

    registry = NoHumanRestartRegistry(restart, run_root)
    registry.initialize()
    log_path = run_root / "sample/EXPERIMENT.json"
    log = NoHumanExperimentLog.load(log_path)
    log.record_alignment_iteration(
        scores={"visual": 1.0},
        gates={"visual": True},
        decision="accept",
        provenance={
            "actor_kind": "human",
            "producer": "reviewer",
            "input_kinds": ["manual_arrow"],
            "manual_arrows": True,
            "manual_stamps": False,
            "human_approval": True,
        },
        method="historical manual correction",
    )
    log.write(run_root / "sample/EXPERIMENT.md", log_path)

    result = AutonomousRestartBatchRunner(
        restart,
        run_root,
        alignment_runner=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("manual attempt reached aligner")
        ),
        extraction_runner=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("manual attempt reached extractor")
        ),
    ).run()

    assert result.outcomes[0].status == "failed"
    assert "ineligible or manual evidence" in result.outcomes[0].message
    persisted = json.loads(log_path.read_text())
    assert persisted["alignment"]["iterations"][0][
        "counts_toward_automatic_iteration_count"
    ] is False


def test_native_pdf_dispatch_uses_source_clean_manifest_and_resumes_safely(tmp_path):
    source, reference, restart, run_root = _native_pdf_fixture(tmp_path)
    calls = []
    geologic_config = object()

    def generic_must_not_run(*_args, **_kwargs):
        raise AssertionError("native PDF reached the generic raster aligner")

    def geologic_alignment(source_manifest, pinned_reference, output, log, **kwargs):
        from mapscan.source_working_raster import load_working_raster_artifact

        source_manifest = Path(source_manifest)
        working = load_working_raster_artifact(source_manifest)
        calls.append(
            (
                "geologic-alignment",
                source_manifest,
                Path(pinned_reference),
                kwargs.get("config"),
            )
        )
        assert source_manifest == run_root / "geologic/source-clean/source-adapter.json"
        assert working.source_path == source.resolve()
        assert working.manifest["authority"] == {
            "original_source_authoritative": True,
            "prior_alignment_used": False,
            "prior_extraction_used": False,
            "manual_input_used": False,
        }
        output.mkdir(parents=True)
        attempt = (
            output
            / "alignment-01-hypothesis-01-baseline-semantic-native-pdf-graticule-california-albers"
        )
        attempt.mkdir()
        payload = {
            "schema_version": 1,
            "decision": "accept",
            "source_sha256": working.working_raster_sha256,
            "mapbox_reference": {"manifest_sha256": _sha256(Path(pinned_reference))},
            "transform": {
                "kind": "projection_aware_mapbox_registration",
                "source_pixel_space": "original_raster",
                "reference_pixel_space": "pinned_mapbox_target_grid",
            },
            "exact_transform_provenance": {
                "fit_evidence": "original_pdf_native_vector_graticule_only",
                "source_adapter_sha256": _sha256(source_manifest),
            },
        }
        candidate = attempt / "candidate.json"
        candidate.write_text(json.dumps(payload))
        (output / "accepted-alignment.json").write_text(json.dumps(payload))
        log.record_alignment_iteration(
            scores={"native_graticule_p90_px": 0.1, "mapbox_state_p90_px": 1.0},
            gates={"native_graticule": True, "mapbox_state_coast": True},
            decision="accept",
            provenance=automatic_provenance(
                "mapscan.geologic_pdf_alignment",
                [
                    "original_pdf_native_vector_graticule",
                    "source_clean_pdf_working_raster",
                    "pinned_mapbox_state_coast",
                ],
            ),
            method="native PDF graticule with independent Mapbox state/coast gate",
            artifacts=[{"path": str(candidate), "sha256": _sha256(candidate)}],
        )
        return SimpleNamespace(
            status="pass", stop_reason="native PDF gates passed", accepted=candidate
        )

    def extraction_must_not_run(*_args, **_kwargs):
        raise AssertionError("unsupported PDF extraction was invented")

    first = AutonomousRestartBatchRunner(
        restart,
        run_root,
        config=AutonomousRestartBatchConfig(
            pdf_dpi=72,
            geologic_alignment_config=geologic_config,
        ),
        alignment_runner=generic_must_not_run,
        geologic_alignment_runner=geologic_alignment,
        extraction_runner=extraction_must_not_run,
        specialized_extraction_runners={},
    ).run()

    assert calls == [
        (
            "geologic-alignment",
            run_root / "geologic/source-clean/source-adapter.json",
            reference.resolve(),
            geologic_config,
        )
    ]
    outcome = first.outcomes[0]
    assert outcome.status == "blocked"
    assert outcome.phase == "extraction"
    assert outcome.alignment_automatic_iterations == 1
    assert outcome.extraction_automatic_iterations == 0
    assert "native_pdf_vector_categorical" in outcome.message
    persisted = json.loads((run_root / "geologic/EXPERIMENT.json").read_text())
    assert persisted["source"]["sha256"] == _sha256(source)
    assert persisted["alignment"]["accepted_automatic_iteration_count"] == 1
    assert persisted["extraction"]["iterations"] == []
    iteration = persisted["alignment"]["iterations"][0]
    assert iteration["counts_toward_automatic_iteration_count"] is True
    assert all(
        iteration["provenance"][key] is False
        for key in ("manual_arrows", "manual_stamps", "human_approval")
    )
    kinds = {artifact.get("kind") for artifact in iteration["artifacts"]}
    assert "source_clean_adapter_manifest" in kinds
    assert "source_clean_working_raster" in kinds
    assert not any(
        token in " ".join(iteration["provenance"]["input_kinds"]).lower()
        for token in ("manual", "arrow", "stamp", "county.png", "census")
    )

    # Simulate a crash after the checkpointed acceptance but before its root
    # pointer was durably retained. Resume must recover the one immutable
    # current-run candidate without invoking either alignment implementation.
    accepted_pointer = run_root / "geologic/automatic-alignment/accepted-alignment.json"
    accepted_pointer.unlink()

    resumed = AutonomousRestartBatchRunner(
        restart,
        run_root,
        config=AutonomousRestartBatchConfig(pdf_dpi=72),
        alignment_runner=generic_must_not_run,
        geologic_alignment_runner=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("accepted geologic alignment was rerun")
        ),
        extraction_runner=extraction_must_not_run,
        specialized_extraction_runners={},
    ).run()

    assert resumed.outcomes[0].phase == "terminal"
    assert resumed.outcomes[0].status == "blocked"
    assert resumed.outcomes[0].alignment_automatic_iterations == 1
    assert resumed.outcomes[0].extraction_automatic_iterations == 0
    assert accepted_pointer.is_file()
    assert len(calls) == 1
