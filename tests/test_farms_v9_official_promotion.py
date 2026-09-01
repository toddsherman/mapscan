from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

import mapscan.farms_v9_official_promotion as promotion
from mapscan.automatic_categorical_extraction import (
    _alignment_contains_forbidden_input,
    _load_accepted_alignment,
)
from mapscan.experiment_log import NoHumanExperimentLog
from mapscan.farms_nonrigid_alignment import load_pinned_mapbox_reference
from mapscan.farms_v9_official_promotion import (
    EXPECTED_FINAL_GATE_CONTRACT_SHA256,
    EXPECTED_FINAL_REPORT_SHA256,
    EXPECTED_OFFICIAL_EXPERIMENT_JSON_SHA256,
    EXPECTED_OFFICIAL_EXPERIMENT_MARKDOWN_SHA256,
    EXPECTED_V9_FROZEN_CANDIDATE_SHA256,
    OFFICIAL_AUTOMATIC_ITERATION,
    _validate_official_checkpoint,
    farms_v9_official_promotion_contract,
    prepare_farms_v9_official_promotion,
    promote_farms_v9_alignment_officially,
)


EXPECTED_PROPOSED_CANDIDATE_SHA256 = (
    "21ad14602078051c130942c701a570b59edcde1525d1ab7bbd282252a59dc7ed"
)
EXPECTED_POST_EXTRACTION_EXPERIMENT_JSON_SHA256 = (
    "c195f54e725efdcf4b106d489b61599627a45c27828929f27a94460499b652cf"
)
EXPECTED_POST_EXTRACTION_EXPERIMENT_MARKDOWN_SHA256 = (
    "3f1438bfff0be392dd20bda132043c58a850d16d541097e8bf789718ff2b54d5"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pre_promotion_log_fixture() -> NoHumanExperimentLog:
    """Derive a semantic ten-attempt fixture without requiring canonical old bytes.

    The immutable first ten attempts remain useful test evidence after the
    exactly-once canonical publication.  Rewinding only the promotion-owned
    fields gives the pure pre-write state that the publisher is required to
    accept, while the production checkpoint hashes remain unchanged.
    """

    current = NoHumanExperimentLog.load(
        promotion.CANONICAL_OFFICIAL_MAP_ROOT / "EXPERIMENT.json"
    )
    data = copy.deepcopy(current.data)
    assert len(data["alignment"]["iterations"]) == OFFICIAL_AUTOMATIC_ITERATION
    accepted = data["alignment"]["iterations"][-1]
    assert accepted["automatic_iteration"] == OFFICIAL_AUTOMATIC_ITERATION
    assert accepted["decision"] == "accept"
    data["alignment"]["iterations"] = data["alignment"]["iterations"][:10]
    data["alignment"]["accepted_automatic_iteration_count"] = None
    data["automatic_resumptions"] = []
    data["extraction"] = {
        "iterations": [],
        "accepted_automatic_iteration_count": None,
    }
    data["final"] = {
        "status": "blocked",
        "blocker": (
            "Automatic alignment blocked: "
            "projection_and_regular_model_sequence_exhausted"
        ),
    }
    fixture = NoHumanExperimentLog.__new__(NoHumanExperimentLog)
    fixture.data = data
    return fixture


def _write_pre_promotion_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Write a self-contained pre-write checkpoint and pin its fixture bytes."""

    copied_root = tmp_path / "runs/mapbox-autonomous-restart-v1/farms-v2"
    fixture = _pre_promotion_log_fixture()
    fixture.write(copied_root / "EXPERIMENT.md", copied_root / "EXPERIMENT.json")
    monkeypatch.setattr(
        promotion, "CANONICAL_OFFICIAL_MAP_ROOT", copied_root.resolve()
    )
    monkeypatch.setattr(
        promotion,
        "EXPECTED_OFFICIAL_EXPERIMENT_JSON_SHA256",
        _sha256(copied_root / "EXPERIMENT.json"),
    )
    monkeypatch.setattr(
        promotion,
        "EXPECTED_OFFICIAL_EXPERIMENT_MARKDOWN_SHA256",
        _sha256(copied_root / "EXPERIMENT.md"),
    )
    return copied_root


def test_farms_v9_promotion_contract_pins_every_authority_and_checkpoint():
    contract = farms_v9_official_promotion_contract()
    assert contract["official_automatic_iteration"] == 11
    assert contract["expected_sha256"] == {
        "source": promotion.EXPECTED_FARMS_SOURCE_SHA256,
        "reference_manifest": promotion.EXPECTED_MAPBOX_V2_MANIFEST_SHA256,
        "final_report": EXPECTED_FINAL_REPORT_SHA256,
        "final_gate_contract": EXPECTED_FINAL_GATE_CONTRACT_SHA256,
        "frozen_candidate": EXPECTED_V9_FROZEN_CANDIDATE_SHA256,
        "official_experiment_json_before_promotion": (
            EXPECTED_OFFICIAL_EXPERIMENT_JSON_SHA256
        ),
        "official_experiment_markdown_before_promotion": (
            EXPECTED_OFFICIAL_EXPERIMENT_MARKDOWN_SHA256
        ),
        "proposed_official_candidate": EXPECTED_PROPOSED_CANDIDATE_SHA256,
    }
    assert contract["required_official_checkpoint"] == {
        "map_id": "farms-v2",
        "final_status": "blocked",
        "alignment_attempt_count": 10,
        "automatic_alignment_ordinals": list(range(1, 11)),
        "accepted_alignment_automatic_iteration": None,
        "extraction_attempt_count": 0,
        "accepted_extraction_automatic_iteration": None,
        "accepted_alignment_pointer_absent": True,
        "attempt_11_directory_absent": True,
    }
    assert contract["publication"]["retry_or_overwrite_allowed"] is False
    assert contract["publication"]["official_index_updated"] is False
    assert contract["publication"]["extraction_started"] is False
    assert contract["provenance"]["statewide_source_coverage_claimed"] is False
    assert contract["provenance"]["unobservable_geography_policy"] == (
        "omitted_with_warning_not_guessed"
    )


def test_farms_v9_read_only_plan_has_stable_extraction_safe_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    copied_root = _write_pre_promotion_fixture(tmp_path, monkeypatch)
    experiment_json = copied_root / "EXPERIMENT.json"
    experiment_markdown = copied_root / "EXPERIMENT.md"
    before = (_sha256(experiment_json), _sha256(experiment_markdown))
    plan = prepare_farms_v9_official_promotion(official_map_root=copied_root)

    assert plan.candidate_sha256 == EXPECTED_PROPOSED_CANDIDATE_SHA256
    assert plan.candidate_payload["iteration"] == OFFICIAL_AUTOMATIC_ITERATION
    assert plan.candidate_payload["decision"] == "accept"
    assert plan.candidate_payload["source_alignment_hypothesis"] == {
        "id": "source-only-partial-california-native-county-water-v9",
        "source_coverage": "partial_california",
        "statewide_source_coverage_claimed": False,
        "unobservable_geography_policy": "omitted_with_warning_not_guessed",
        "county_evidence": (
            "source_only_native_dark_neutral_ink_with_geographic_scope"
        ),
        "water_evidence": "source_only_pacific_and_internal_water_edges",
        "mapbox_role": "training_validation_and_final_geometry_authority",
    }
    assert _alignment_contains_forbidden_input(plan.candidate_payload) is None
    assert all(plan.gates.values())
    assert plan.provenance["actor_kind"] == "automated"
    assert plan.provenance["human_approval"] is False
    assert not plan.candidate_directory.exists()
    assert not plan.accepted_pointer_path.exists()
    assert before == (_sha256(experiment_json), _sha256(experiment_markdown))


def test_farms_v9_checkpoint_rejects_gap_acceptance_or_extraction():
    log = _pre_promotion_log_fixture()
    reference = load_pinned_mapbox_reference(
        promotion.CANONICAL_REFERENCE_MANIFEST_PATH
    )
    changed = NoHumanExperimentLog.__new__(NoHumanExperimentLog)
    changed.data = copy.deepcopy(log.data)
    changed.data["alignment"]["iterations"][4]["automatic_iteration"] = 6
    with pytest.raises(ValueError, match="contiguous rejected"):
        _validate_official_checkpoint(
            changed,
            source_path=promotion.CANONICAL_SOURCE_PATH,
            reference_pin=reference.pin,
        )

    changed.data = copy.deepcopy(log.data)
    changed.data["extraction"]["iterations"].append({"attempt": 1})
    with pytest.raises(ValueError, match="zero extraction"):
        _validate_official_checkpoint(
            changed,
            source_path=promotion.CANONICAL_SOURCE_PATH,
            reference_pin=reference.pin,
        )


def test_farms_v9_official_promotion_uses_log_semantics_and_refuses_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    copied_root = _write_pre_promotion_fixture(tmp_path, monkeypatch)

    result = promote_farms_v9_alignment_officially(
        official_map_root=copied_root
    )
    assert result.automatic_iteration == 11
    assert result.candidate_sha256 == EXPECTED_PROPOSED_CANDIDATE_SHA256
    assert result.candidate_path.read_bytes() == result.accepted_pointer_path.read_bytes()
    assert _sha256(result.candidate_path) == EXPECTED_PROPOSED_CANDIDATE_SHA256

    log = NoHumanExperimentLog.load(result.experiment_json_path)
    assert len(log.data["alignment"]["iterations"]) == 11
    assert log.data["alignment"]["accepted_automatic_iteration_count"] == 11
    accepted = log.data["alignment"]["iterations"][-1]
    assert accepted["attempt"] == 11
    assert accepted["automatic_iteration"] == 11
    assert accepted["decision"] == "accept"
    assert accepted["counts_toward_automatic_iteration_count"] is True
    assert accepted["all_gates_passed"] is True
    assert accepted["provenance"]["producer"] == promotion.OFFICIAL_PRODUCER
    assert log.data["extraction"]["iterations"] == []
    assert log.data["extraction"]["accepted_automatic_iteration_count"] is None
    assert log.data["final"] == {"status": "in_progress", "blocker": None}
    assert len(log.data["automatic_resumptions"]) == 1

    reference = load_pinned_mapbox_reference(
        promotion.CANONICAL_REFERENCE_MANIFEST_PATH
    )
    loaded = _load_accepted_alignment(
        result.accepted_pointer_path,
        promotion.CANONICAL_SOURCE_PATH,
        reference.grid,
        reference_pin=reference.pin,
        accepted_iteration_count=11,
        alignment_iterations=log.data["alignment"]["iterations"],
    )
    assert loaded["iteration"] == 11
    assert loaded["transform"]["kind"] == (
        "projection_aware_residual_warp_mapbox_registration"
    )

    with pytest.raises((ValueError, FileExistsError), match="hash mismatch|retry"):
        promote_farms_v9_alignment_officially(official_map_root=copied_root)


def test_farms_v9_promotion_rejects_changed_official_bytes_before_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    copied_root = _write_pre_promotion_fixture(tmp_path, monkeypatch)
    experiment = copied_root / "EXPERIMENT.json"
    parsed = json.loads(experiment.read_text())
    parsed["final"]["blocker"] = "changed"
    experiment.write_text(json.dumps(parsed, indent=2) + "\n")
    with pytest.raises(ValueError, match="EXPERIMENT.json pre-promotion hash mismatch"):
        prepare_farms_v9_official_promotion(official_map_root=copied_root)
    assert not (copied_root / "automatic-alignment/accepted-alignment.json").exists()
    assert not (
        copied_root
        / "automatic-alignment"
        / promotion.OFFICIAL_CANDIDATE_DIRECTORY_NAME
    ).exists()


def test_canonical_completed_state_preserves_accepted_a11_and_accepted_e4():
    root = promotion.CANONICAL_OFFICIAL_MAP_ROOT
    experiment_json = root / "EXPERIMENT.json"
    experiment_markdown = root / "EXPERIMENT.md"
    candidate = (
        root
        / "automatic-alignment"
        / promotion.OFFICIAL_CANDIDATE_DIRECTORY_NAME
        / "candidate.json"
    )
    pointer = root / "automatic-alignment/accepted-alignment.json"

    assert _sha256(experiment_json) == EXPECTED_POST_EXTRACTION_EXPERIMENT_JSON_SHA256
    assert (
        _sha256(experiment_markdown)
        == EXPECTED_POST_EXTRACTION_EXPERIMENT_MARKDOWN_SHA256
    )
    assert candidate.read_bytes() == pointer.read_bytes()
    assert _sha256(candidate) == EXPECTED_PROPOSED_CANDIDATE_SHA256

    log = NoHumanExperimentLog.load(experiment_json)
    assert len(log.data["alignment"]["iterations"]) == 11
    assert log.data["alignment"]["accepted_automatic_iteration_count"] == 11
    accepted = log.data["alignment"]["iterations"][-1]
    assert accepted["attempt"] == 11
    assert accepted["automatic_iteration"] == 11
    assert accepted["counts_toward_automatic_iteration_count"] is True
    assert accepted["automatic_eligibility_reason"] == "eligible automatic iteration"
    assert accepted["decision"] == "accept"
    assert accepted["all_gates_passed"] is True
    assert len(accepted["gates"]) == 22
    assert all(accepted["gates"].values())
    assert accepted["provenance"]["producer"] == promotion.OFFICIAL_PRODUCER
    assert accepted["provenance"]["manual_arrows"] is False
    assert accepted["provenance"]["manual_stamps"] is False
    assert accepted["provenance"]["human_approval"] is False
    assert len(log.data["automatic_resumptions"]) == 1
    assert len(log.data["extraction"]["iterations"]) == 4
    assert log.data["extraction"]["accepted_automatic_iteration_count"] == 4
    extracted = log.data["extraction"]["iterations"][-1]
    assert extracted["attempt"] == 4
    assert extracted["automatic_iteration"] == 4
    assert extracted["counts_toward_automatic_iteration_count"] is True
    assert extracted["automatic_eligibility_reason"] == (
        "eligible automatic iteration"
    )
    assert extracted["decision"] == "accept"
    assert extracted["all_gates_passed"] is True
    assert extracted["scores"]["source_observed_fraction"] == pytest.approx(
        0.8032296441192851
    )
    assert extracted["scores"]["source_inferred_fraction"] == pytest.approx(
        0.19677035588071487
    )
    assert extracted["scores"]["meaningful_source_mismatch_pixel_count"] == 0
    extent = extracted["scores"]["classification_domain"][
        "aligned_source_extent"
    ]
    assert extent["covered_fraction"] == pytest.approx(0.48707748051548116)
    assert extent["partial_extent"] is True
    assert extent["missing_extent_policy"] == "preserve_as_nodata_never_infer"
    assert log.data["final"] == {"status": "complete", "blocker": None}
    assert (root / "automatic-extraction/extraction-04/iteration.json").is_file()

    reference = load_pinned_mapbox_reference(
        promotion.CANONICAL_REFERENCE_MANIFEST_PATH
    )
    loaded = _load_accepted_alignment(
        pointer,
        promotion.CANONICAL_SOURCE_PATH,
        reference.grid,
        reference_pin=reference.pin,
        accepted_iteration_count=11,
        alignment_iterations=log.data["alignment"]["iterations"],
    )
    assert loaded["iteration"] == 11
    assert loaded["decision"] == "accept"


def test_canonical_post_promotion_retry_is_refused_without_mutation():
    root = promotion.CANONICAL_OFFICIAL_MAP_ROOT
    tracked = (
        root / "EXPERIMENT.json",
        root / "EXPERIMENT.md",
        root
        / "automatic-alignment"
        / promotion.OFFICIAL_CANDIDATE_DIRECTORY_NAME
        / "candidate.json",
        root / "automatic-alignment/accepted-alignment.json",
    )
    before = {path: _sha256(path) for path in tracked}
    with pytest.raises(
        (ValueError, FileExistsError), match="pre-promotion hash mismatch|retry"
    ):
        promote_farms_v9_alignment_officially(official_map_root=root)
    assert before == {path: _sha256(path) for path in tracked}
