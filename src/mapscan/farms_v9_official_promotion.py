"""Fail-closed official promotion for the accepted farms-v9 alignment.

The alignment itself was selected and evaluated in immutable validation and
one-shot final-acceptance runs.  This module does not fit, tune, or score an
alignment.  It only verifies those exact bytes, verifies the untouched official
ten-attempt checkpoint, constructs the deterministic automatic attempt 11, and
publishes it through :class:`NoHumanExperimentLog`.

Call :func:`prepare_farms_v9_official_promotion` for the required independent
pre-write audit.  The mutating function is deliberately separate and refuses a
retry, an alternate official root, or any changed input/history byte.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .automatic_categorical_extraction import (
    _alignment_contains_forbidden_input,
    _load_accepted_alignment,
)
from .experiment_log import NoHumanExperimentLog, automatic_provenance
from .farms_nonrigid_alignment import (
    EXPECTED_FARMS_SOURCE_SHA256,
    EXPECTED_MAPBOX_V2_MANIFEST_SHA256,
    load_pinned_mapbox_reference,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_SOURCE_PATH = (PROJECT_ROOT / "examples/farmsv2.png").resolve()
CANONICAL_REFERENCE_MANIFEST_PATH = (
    PROJECT_ROOT / "reference/mapbox-light-v11-california-z9-v2/manifest.json"
).resolve()
CANONICAL_FINAL_ROOT = (
    PROJECT_ROOT / "runs/farms-v2-nonrigid-mapbox-v2-one-shot-final-acceptance-v9"
).resolve()
CANONICAL_FINAL_REPORT_PATH = CANONICAL_FINAL_ROOT / "final-acceptance-report.json"
CANONICAL_FINAL_GATE_CONTRACT_PATH = (
    CANONICAL_FINAL_ROOT / "predeclared-gate-contract.json"
)
CANONICAL_V9_FROZEN_CANDIDATE_PATH = (
    PROJECT_ROOT
    / "runs/farms-v2-nonrigid-mapbox-v2-validation-preflight-rollforward-v9"
    / "frozen-validation-candidate.json"
).resolve()
CANONICAL_OFFICIAL_MAP_ROOT = (
    PROJECT_ROOT / "runs/mapbox-autonomous-restart-v1/farms-v2"
).resolve()

EXPECTED_FINAL_REPORT_SHA256 = (
    "3471cf63068000e2703234d4229823585a9c19eeef584ee315fae5b2109296e0"
)
EXPECTED_FINAL_GATE_CONTRACT_SHA256 = (
    "edc7ad237c142aec2719ffc9dab490a71a72c5a86685b15678aa4351870ffbaa"
)
EXPECTED_V9_FROZEN_CANDIDATE_SHA256 = (
    "d92a58a83fe5bb85bd5f73dafc2acefaca8f93a208257d688084737bdf6c1152"
)
EXPECTED_OFFICIAL_EXPERIMENT_JSON_SHA256 = (
    "2529bb551d53cba7d7ef69ccd41de84227ed1ab30d0321de8d0859391aaab64c"
)
EXPECTED_OFFICIAL_EXPERIMENT_MARKDOWN_SHA256 = (
    "c7135a4bd86ec157188a5f2d4964185b91457620b315b2b4cc814a118cb4906c"
)
EXPECTED_PROPOSED_OFFICIAL_CANDIDATE_SHA256 = (
    "21ad14602078051c130942c701a570b59edcde1525d1ab7bbd282252a59dc7ed"
)

OFFICIAL_AUTOMATIC_ITERATION = 11
OFFICIAL_CANDIDATE_DIRECTORY_NAME = (
    "alignment-11-california_albers-wendland-c2-partial-source-v9"
)
OFFICIAL_PRODUCER = "mapscan.farms_v9_official_promotion.v1"
OFFICIAL_RESUMPTION_REASON = (
    "frozen partial-source farms-v9 projection-aware residual alignment passed "
    "fresh validation and exactly-once county and named-water acceptance gates"
)


@dataclass(frozen=True)
class FarmsV9OfficialPromotionPlan:
    """Deterministic, read-only official attempt-11 publication plan."""

    candidate_payload: Mapping[str, Any]
    candidate_bytes: bytes
    candidate_sha256: str
    scores: Mapping[str, Any]
    gates: Mapping[str, Any]
    provenance: Mapping[str, Any]
    candidate_directory: Path
    candidate_path: Path
    accepted_pointer_path: Path
    experiment_json_path: Path
    experiment_markdown_path: Path


@dataclass(frozen=True)
class FarmsV9OfficialPromotionResult:
    """Paths and hashes written by the one permitted official promotion."""

    automatic_iteration: int
    candidate_path: Path
    candidate_sha256: str
    accepted_pointer_path: Path
    experiment_json_path: Path
    experiment_markdown_path: Path
    experiment_json_sha256: str
    experiment_markdown_sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"{label} hash mismatch: expected {expected}, got {actual}")


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _artifact(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {"path": str(resolved), "sha256": _sha256(resolved)}


def farms_v9_official_promotion_contract() -> Mapping[str, Any]:
    """Return the frozen pre-write authority and publication schema."""

    return {
        "schema_version": 1,
        "kind": "farms_v9_official_alignment_promotion_contract_v1",
        "official_automatic_iteration": OFFICIAL_AUTOMATIC_ITERATION,
        "official_candidate_directory_name": OFFICIAL_CANDIDATE_DIRECTORY_NAME,
        "canonical_paths": {
            "source": str(CANONICAL_SOURCE_PATH),
            "reference_manifest": str(CANONICAL_REFERENCE_MANIFEST_PATH),
            "final_report": str(CANONICAL_FINAL_REPORT_PATH),
            "final_gate_contract": str(CANONICAL_FINAL_GATE_CONTRACT_PATH),
            "frozen_candidate": str(CANONICAL_V9_FROZEN_CANDIDATE_PATH),
            "official_map_root": str(CANONICAL_OFFICIAL_MAP_ROOT),
        },
        "expected_sha256": {
            "source": EXPECTED_FARMS_SOURCE_SHA256,
            "reference_manifest": EXPECTED_MAPBOX_V2_MANIFEST_SHA256,
            "final_report": EXPECTED_FINAL_REPORT_SHA256,
            "final_gate_contract": EXPECTED_FINAL_GATE_CONTRACT_SHA256,
            "frozen_candidate": EXPECTED_V9_FROZEN_CANDIDATE_SHA256,
            "official_experiment_json_before_promotion": (
                EXPECTED_OFFICIAL_EXPERIMENT_JSON_SHA256
            ),
            "official_experiment_markdown_before_promotion": (
                EXPECTED_OFFICIAL_EXPERIMENT_MARKDOWN_SHA256
            ),
            "proposed_official_candidate": (
                EXPECTED_PROPOSED_OFFICIAL_CANDIDATE_SHA256
            ),
        },
        "required_official_checkpoint": {
            "map_id": "farms-v2",
            "final_status": "blocked",
            "alignment_attempt_count": 10,
            "automatic_alignment_ordinals": list(range(1, 11)),
            "accepted_alignment_automatic_iteration": None,
            "extraction_attempt_count": 0,
            "accepted_extraction_automatic_iteration": None,
            "accepted_alignment_pointer_absent": True,
            "attempt_11_directory_absent": True,
        },
        "publication": {
            "candidate_and_pointer_byte_identical": True,
            "experiment_log_api": "NoHumanExperimentLog",
            "experiment_json_and_markdown_atomic_replace": True,
            "official_index_updated": False,
            "extraction_started": False,
            "retry_or_overwrite_allowed": False,
        },
        "provenance": {
            "producer": OFFICIAL_PRODUCER,
            "actor_kind": "automated",
            "input_kinds": [
                "authoritative_original_source_pixels",
                "source_only_partial_map_observable_extent",
                "source_only_native_county_ink",
                "source_only_pacific_and_internal_water_edges",
                "pinned_mapbox_v2_state_coast_counties_and_water",
                "deterministic_target_only_validation_partition",
                "frozen_v9_validation_candidate",
                "sealed_one_shot_final_acceptance_evidence",
            ],
            "statewide_source_coverage_claimed": False,
            "unobservable_geography_policy": "omitted_with_warning_not_guessed",
        },
    }


def _validate_official_checkpoint(
    log: NoHumanExperimentLog,
    *,
    source_path: Path,
    reference_pin: Mapping[str, Any],
) -> None:
    data = log.data
    alignment = data.get("alignment", {})
    extraction = data.get("extraction", {})
    iterations = alignment.get("iterations", [])
    if data.get("map_id") != "farms-v2":
        raise ValueError("Official checkpoint is not farms-v2")
    if data.get("source", {}).get("sha256") != _sha256(source_path):
        raise ValueError("Official checkpoint source hash disagrees")
    logged_reference = data.get("mapbox_reference", {})
    # The restart registry stores the richer, human-facing v2 reference record,
    # while alignment consumers use the manifest's compact cryptographic pin.
    # Bind the shared bytes explicitly instead of pretending those two schemas
    # are identical.
    if (
        logged_reference.get("id") != "mapbox-light-v11-california-z9-v2"
        or logged_reference.get("manifest_sha256")
        != reference_pin.get("manifest_sha256")
        or logged_reference.get("style_sha256") != reference_pin.get("style_sha256")
        or logged_reference.get("tilejson_sha256")
        != reference_pin.get("tilejson_sha256")
        or logged_reference.get("tiles_sha256")
        != reference_pin.get("tile_aggregate_sha256")
    ):
        raise ValueError("Official checkpoint does not pin exact Mapbox-v2 authority")
    if data.get("final") != {
        "status": "blocked",
        "blocker": "Automatic alignment blocked: projection_and_regular_model_sequence_exhausted",
    }:
        raise ValueError("Official checkpoint is not the exact blocked pre-v9 state")
    if data.get("automatic_resumptions") != []:
        raise ValueError("Official checkpoint already contains a resumption")
    if len(iterations) != 10 or alignment.get(
        "accepted_automatic_iteration_count"
    ) is not None:
        raise ValueError("Official checkpoint must have ten unaccepted alignment attempts")
    if extraction.get("iterations") != [] or extraction.get(
        "accepted_automatic_iteration_count"
    ) is not None:
        raise ValueError("Official checkpoint must have zero extraction attempts")
    for expected, item in enumerate(iterations, 1):
        if (
            item.get("attempt") != expected
            or item.get("automatic_iteration") != expected
            or item.get("counts_toward_automatic_iteration_count") is not True
            or item.get("automatic_eligibility_reason")
            != "eligible automatic iteration"
            or item.get("decision") not in {"retry", "reject", "blocked"}
            or item.get("all_gates_passed") is not False
        ):
            raise ValueError(
                f"Official alignment attempt {expected} is not a contiguous rejected automatic attempt"
            )


def _safe_partial_observability(
    diagnostics: Mapping[str, Any],
) -> Mapping[str, Any]:
    masks = diagnostics.get("masks", {})
    required_masks = (
        "map_panel",
        "california_positive_support",
        "county_lines",
        "pacific_coast",
        "named_bay",
        "state_admin",
    )
    if any(name not in masks for name in required_masks):
        raise ValueError("Final report lacks required partial-source extent masks")
    if diagnostics.get("ambiguous_lower_right_county_evidence_policy") != (
        "omitted_with_warning"
    ):
        raise ValueError("Final report does not explicitly omit ambiguous county evidence")
    if diagnostics.get("mapbox_used") is not False:
        raise ValueError("Source observability extent must be source-only")
    return {
        "source_scope": "partial_california_map",
        "extent_method": diagnostics["method"],
        "county_line_visibility_method": diagnostics[
            "county_line_visibility_method"
        ],
        "ambiguous_county_evidence_pixel_count": int(
            diagnostics["ambiguous_lower_right_county_evidence_pixels"]
        ),
        "ambiguous_county_evidence_policy": "omitted_with_warning_not_guessed",
        "neighboring_geography_role": "observed_negative_evidence",
        "statewide_source_coverage_claimed": False,
        "working_shape": list(diagnostics["working_shape"]),
        "source_only_extent_masks": {
            name: dict(masks[name]) for name in required_masks
        },
    }


def _build_candidate_payload(
    *,
    final_report: Mapping[str, Any],
    final_gate_contract: Mapping[str, Any],
    frozen: Mapping[str, Any],
    reference_pin: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if final_report.get("status") != "accepted":
        raise ValueError("Official promotion requires an accepted one-shot final report")
    if final_report.get("all_gates_passed") is not True or not all(
        value is True for value in final_report.get("gates", {}).values()
    ):
        raise ValueError("One-shot final acceptance gates did not all pass")
    if final_report.get("gate_contract") != final_gate_contract:
        raise ValueError("Final report gate contract differs from its pinned artifact")
    gate_artifact = final_report.get("gate_contract_artifact", {})
    if gate_artifact.get("sha256") != EXPECTED_FINAL_GATE_CONTRACT_SHA256:
        raise ValueError("Final report does not pin the exact gate contract")
    inputs = final_report.get("inputs", {})
    expected_inputs = {
        "source_sha256": EXPECTED_FARMS_SOURCE_SHA256,
        "reference_manifest_sha256": EXPECTED_MAPBOX_V2_MANIFEST_SHA256,
        "v9_frozen_candidate_sha256": EXPECTED_V9_FROZEN_CANDIDATE_SHA256,
    }
    for key, expected in expected_inputs.items():
        if inputs.get(key) != expected:
            raise ValueError(f"Final report input {key} is not the exact frozen authority")
    authority = final_report.get("authority", {})
    required_authority = {
        "sealed_final_county_scores_evaluated_exactly_once": True,
        "golden_gate_scores_evaluated_exactly_once": True,
        "east_bay_scores_evaluated_exactly_once": True,
        "candidate_count_evaluated": 1,
        "candidate_selected_before_final_scores": True,
        "parameter_tuning_after_final_scores": False,
        "fallback_candidate_after_final_scores": False,
        "official_alignment_attempt_written": False,
        "extraction_written": False,
    }
    if any(authority.get(key) != value for key, value in required_authority.items()):
        raise ValueError("One-shot final authority is not eligible for promotion")
    if frozen.get("source_sha256") != EXPECTED_FARMS_SOURCE_SHA256:
        raise ValueError("Frozen candidate source hash disagrees")
    if frozen.get("reference_manifest_sha256") != EXPECTED_MAPBOX_V2_MANIFEST_SHA256:
        raise ValueError("Frozen candidate reference hash disagrees")
    frozen_authority = frozen.get("authority", {})
    if (
        frozen_authority.get("sealed_final_county_scores_evaluated") is not False
        or frozen_authority.get("golden_gate_or_east_bay_masks_read_or_scored")
        is not False
    ):
        raise ValueError("Frozen validation candidate was not sealed before final scoring")
    transform = frozen.get("serialized_transform")
    if not isinstance(transform, Mapping):
        raise ValueError("Frozen candidate lacks the serialized transform")
    selected = frozen.get("selected_candidate", {})
    if selected.get("eligible") is not True or not all(
        value is True for value in selected.get("gates", {}).values()
    ):
        raise ValueError("Frozen candidate did not pass validation-only gates")
    final_candidate = final_report.get("candidate", {})
    if (
        final_candidate.get("id") != selected.get("id")
        or final_candidate.get("serialized_transform_exactly_reproduced") is not True
        or final_candidate.get("regularity") != selected.get("regularity")
    ):
        raise ValueError("Final report candidate differs from the frozen validation candidate")

    observability = _safe_partial_observability(
        final_report["source_observable_extent_diagnostics"]
    )
    scores = {
        "objective": selected["selection_score"],
        "fresh_v9_validation_counties": selected["fresh_validation_counties"],
        "fresh_v9_validation_counties_balanced": selected[
            "fresh_balanced_validation_counties"
        ],
        "consumed_evidence_non_degradation": {
            "state_admin": selected["consumed_admin_non_degradation"],
            "outer_pacific": selected["consumed_outer_pacific_non_degradation"],
            "named_bay": selected["consumed_named_bay_non_degradation"],
        },
        "one_shot_final_acceptance": final_report["scores"],
        "regularity": final_candidate["regularity"],
        "serialized_roundtrip": final_candidate["serialized_roundtrip"],
        "full_consumer_preflight": final_candidate["full_consumer_preflight"],
        "partial_source_observability": observability,
        "authority_sha256": {
            "final_report": EXPECTED_FINAL_REPORT_SHA256,
            "final_gate_contract": EXPECTED_FINAL_GATE_CONTRACT_SHA256,
            "frozen_validation_candidate": EXPECTED_V9_FROZEN_CANDIDATE_SHA256,
        },
    }
    gates = {
        **{
            f"validation_{key}": bool(value)
            for key, value in selected["gates"].items()
        },
        **{
            f"acceptance_{key}": bool(value)
            for key, value in final_report["gates"].items()
        },
        "frozen_candidate_exact": True,
        "source_and_reference_exact": True,
        "partial_source_observability_explicit": True,
        "one_shot_final_authority_exact": True,
    }
    payload = {
        "schema_version": 1,
        "iteration": OFFICIAL_AUTOMATIC_ITERATION,
        "model": "projection_aware_wendland_c2_partial_source_v9",
        "projection": "california_albers",
        "scores": scores,
        "gates": gates,
        "decision": "accept",
        "source_sha256": EXPECTED_FARMS_SOURCE_SHA256,
        "authoritative_source_sha256": EXPECTED_FARMS_SOURCE_SHA256,
        "mapbox_reference": dict(reference_pin),
        "source_alignment_hypothesis": {
            "id": "source-only-partial-california-native-county-water-v9",
            "source_coverage": "partial_california",
            "statewide_source_coverage_claimed": False,
            "unobservable_geography_policy": "omitted_with_warning_not_guessed",
            "county_evidence": "source_only_native_dark_neutral_ink_with_geographic_scope",
            "water_evidence": "source_only_pacific_and_internal_water_edges",
            "mapbox_role": "training_validation_and_final_geometry_authority",
        },
        "transform": dict(transform),
    }
    forbidden = _alignment_contains_forbidden_input(payload)
    if forbidden:
        raise ValueError(f"Official farms payload contains forbidden evidence at {forbidden}")
    return payload, scores, gates


def _validate_final_report_artifacts(final_report: Mapping[str, Any]) -> None:
    """Resolve every artifact the official log will cite against report bytes."""

    expected: list[tuple[str, Mapping[str, Any]]] = [
        ("final gate contract", final_report.get("gate_contract_artifact", {})),
        ("one-shot final overlay", final_report.get("overlay_artifact", {})),
    ]
    rendered = final_report.get("rendered_artifacts", {})
    for name in ("final_county", "golden_gate", "east_bay"):
        expected.append((f"rendered {name}", rendered.get(name, {})))
    for label, artifact in expected:
        path_value = artifact.get("path")
        digest = artifact.get("sha256")
        if not isinstance(path_value, str) or not isinstance(digest, str):
            raise ValueError(f"Final report lacks {label} artifact authority")
        _require_hash(Path(path_value).resolve(), digest, label)


def _preflight_extraction_contract(
    payload: Mapping[str, Any],
    *,
    source_path: Path,
    reference_pin: Mapping[str, Any],
    reference_grid: Mapping[str, Any],
) -> None:
    candidate_bytes = _canonical_json_bytes(payload)
    with tempfile.TemporaryDirectory(prefix="mapscan-farms-v9-promotion-") as directory:
        candidate_path = Path(directory) / "candidate.json"
        candidate_path.write_bytes(candidate_bytes)
        _load_accepted_alignment(
            candidate_path,
            source_path,
            reference_grid,
            reference_pin=reference_pin,
            accepted_iteration_count=OFFICIAL_AUTOMATIC_ITERATION,
        )


def prepare_farms_v9_official_promotion(
    *, official_map_root: Path = CANONICAL_OFFICIAL_MAP_ROOT
) -> FarmsV9OfficialPromotionPlan:
    """Verify the complete checkpoint and return a read-only attempt-11 plan."""

    official_map_root = official_map_root.resolve()
    if official_map_root != CANONICAL_OFFICIAL_MAP_ROOT:
        raise ValueError("Farms-v9 promotion must use the canonical official map root")
    _require_hash(
        CANONICAL_SOURCE_PATH, EXPECTED_FARMS_SOURCE_SHA256, "pristine farms source"
    )
    _require_hash(
        CANONICAL_REFERENCE_MANIFEST_PATH,
        EXPECTED_MAPBOX_V2_MANIFEST_SHA256,
        "Mapbox-v2 manifest",
    )
    _require_hash(
        CANONICAL_FINAL_REPORT_PATH,
        EXPECTED_FINAL_REPORT_SHA256,
        "one-shot final report",
    )
    _require_hash(
        CANONICAL_FINAL_GATE_CONTRACT_PATH,
        EXPECTED_FINAL_GATE_CONTRACT_SHA256,
        "one-shot final gate contract",
    )
    _require_hash(
        CANONICAL_V9_FROZEN_CANDIDATE_PATH,
        EXPECTED_V9_FROZEN_CANDIDATE_SHA256,
        "v9 frozen candidate",
    )
    experiment_json_path = official_map_root / "EXPERIMENT.json"
    experiment_markdown_path = official_map_root / "EXPERIMENT.md"
    _require_hash(
        experiment_json_path,
        EXPECTED_OFFICIAL_EXPERIMENT_JSON_SHA256,
        "official farms EXPERIMENT.json pre-promotion",
    )
    _require_hash(
        experiment_markdown_path,
        EXPECTED_OFFICIAL_EXPERIMENT_MARKDOWN_SHA256,
        "official farms EXPERIMENT.md pre-promotion",
    )

    reference = load_pinned_mapbox_reference(CANONICAL_REFERENCE_MANIFEST_PATH)
    log = NoHumanExperimentLog.load(experiment_json_path)
    _validate_official_checkpoint(
        log, source_path=CANONICAL_SOURCE_PATH, reference_pin=reference.pin
    )
    final_report = json.loads(CANONICAL_FINAL_REPORT_PATH.read_text())
    final_gate_contract = json.loads(
        CANONICAL_FINAL_GATE_CONTRACT_PATH.read_text()
    )
    frozen = json.loads(CANONICAL_V9_FROZEN_CANDIDATE_PATH.read_text())
    _validate_final_report_artifacts(final_report)
    payload, scores, gates = _build_candidate_payload(
        final_report=final_report,
        final_gate_contract=final_gate_contract,
        frozen=frozen,
        reference_pin=reference.pin,
    )
    if payload["transform"].get("target_grid") != dict(reference.grid):
        raise ValueError("Frozen farms transform does not target exact Mapbox-v2 grid")
    _preflight_extraction_contract(
        payload,
        source_path=CANONICAL_SOURCE_PATH,
        reference_pin=reference.pin,
        reference_grid=reference.grid,
    )

    candidate_directory = (
        official_map_root / "automatic-alignment" / OFFICIAL_CANDIDATE_DIRECTORY_NAME
    )
    candidate_path = candidate_directory / "candidate.json"
    accepted_pointer_path = (
        official_map_root / "automatic-alignment" / "accepted-alignment.json"
    )
    if candidate_directory.exists() or accepted_pointer_path.exists():
        raise FileExistsError("Official farms attempt 11 already exists; retry is forbidden")
    candidate_bytes = _canonical_json_bytes(payload)
    candidate_sha256 = hashlib.sha256(candidate_bytes).hexdigest()
    if candidate_sha256 != EXPECTED_PROPOSED_OFFICIAL_CANDIDATE_SHA256:
        raise ValueError(
            "Proposed official farms candidate differs from the independently "
            "auditable attempt-11 bytes: " + candidate_sha256
        )
    return FarmsV9OfficialPromotionPlan(
        candidate_payload=payload,
        candidate_bytes=candidate_bytes,
        candidate_sha256=candidate_sha256,
        scores=scores,
        gates=gates,
        provenance=automatic_provenance(
            OFFICIAL_PRODUCER,
            farms_v9_official_promotion_contract()["provenance"]["input_kinds"],
        ),
        candidate_directory=candidate_directory,
        candidate_path=candidate_path,
        accepted_pointer_path=accepted_pointer_path,
        experiment_json_path=experiment_json_path,
        experiment_markdown_path=experiment_markdown_path,
    )


def promote_farms_v9_alignment_officially(
    *, official_map_root: Path = CANONICAL_OFFICIAL_MAP_ROOT
) -> FarmsV9OfficialPromotionResult:
    """Publish immutable automatic attempt 11 exactly once.

    This function intentionally performs no extraction and no index/document
    update.  It must only be called after an independent audit of the plan.
    """

    plan = prepare_farms_v9_official_promotion(official_map_root=official_map_root)
    log = NoHumanExperimentLog.load(plan.experiment_json_path)

    # Create the immutable candidate and its convenience pointer first.  If a
    # process failure occurs later, their presence fail-closes any retry rather
    # than allowing attempt 11 to be scored or published twice.
    plan.candidate_directory.mkdir(parents=True, exist_ok=False)
    candidate_tmp = plan.candidate_path.with_name(".candidate.json.tmp")
    candidate_tmp.write_bytes(plan.candidate_bytes)
    os.replace(candidate_tmp, plan.candidate_path)
    pointer_tmp = plan.accepted_pointer_path.with_name(".accepted-alignment.json.tmp")
    pointer_tmp.write_bytes(plan.candidate_bytes)
    os.replace(pointer_tmp, plan.accepted_pointer_path)

    timestamp = None
    log.resume_automatic_blocked(
        reason=OFFICIAL_RESUMPTION_REASON,
        producer=OFFICIAL_PRODUCER,
        recorded_at=timestamp,
    )
    iteration = log.record_alignment_iteration(
        scores=plan.scores,
        gates=plan.gates,
        decision="accept",
        provenance=plan.provenance,
        method=(
            "frozen projection-aware California Albers plus regularized "
            "Wendland-C2 residual for a source-derived partial California extent, "
            "with fresh geographic county validation and one-shot county and "
            "named-water acceptance"
        ),
        artifacts=[
            _artifact(plan.candidate_path),
            _artifact(CANONICAL_FINAL_REPORT_PATH),
            _artifact(CANONICAL_FINAL_GATE_CONTRACT_PATH),
            _artifact(CANONICAL_V9_FROZEN_CANDIDATE_PATH),
            _artifact(CANONICAL_FINAL_ROOT / "one-shot-final-overlay.png"),
        ],
        note=(
            "The source is a partial California map. Geography outside the "
            "automatically detected source-observable extent is omitted with a "
            "warning and is not guessed; extraction has not started."
        ),
        recorded_at=timestamp,
    )
    if (
        iteration.get("attempt") != OFFICIAL_AUTOMATIC_ITERATION
        or iteration.get("automatic_iteration") != OFFICIAL_AUTOMATIC_ITERATION
        or log.data["alignment"].get("accepted_automatic_iteration_count")
        != OFFICIAL_AUTOMATIC_ITERATION
        or log.data["extraction"].get("iterations") != []
        or log.data["final"] != {"status": "in_progress", "blocker": None}
    ):
        raise RuntimeError("NoHumanExperimentLog did not produce exact official attempt 11")
    written = log.write(plan.experiment_markdown_path, plan.experiment_json_path)
    if plan.accepted_pointer_path.read_bytes() != plan.candidate_path.read_bytes():
        raise RuntimeError("Accepted pointer is not byte-identical to immutable candidate")
    return FarmsV9OfficialPromotionResult(
        automatic_iteration=OFFICIAL_AUTOMATIC_ITERATION,
        candidate_path=plan.candidate_path,
        candidate_sha256=plan.candidate_sha256,
        accepted_pointer_path=plan.accepted_pointer_path,
        experiment_json_path=plan.experiment_json_path,
        experiment_markdown_path=plan.experiment_markdown_path,
        experiment_json_sha256=written["json_sha256"],
        experiment_markdown_sha256=written["markdown_sha256"],
    )
