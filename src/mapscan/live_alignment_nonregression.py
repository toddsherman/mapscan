"""Fail-closed audit for the alignments backing the live MapScan catalog."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


COUNTY_GATE_NAMES = frozenset(
    {
        "county_holdout_median",
        "county_holdout_support",
        "geographically_balanced_counties",
        "semantic_full_county_median",
        "semantic_full_county_support",
        "semantic_full_county_symmetric_overlap",
        "source_county_channel_observability",
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _gate_passed(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return bool(value.get("passed")) if isinstance(value, Mapping) else False


def _county_required(payload: Mapping[str, Any]) -> bool:
    return bool(
        payload.get("scores", {})
        .get("semantic_full_line", {})
        .get("source_capabilities", {})
        .get("counties", {})
        .get("required_for_acceptance", True)
    )


def _effective_failed_gates(
    payload: Mapping[str, Any], *, county_required: bool
) -> list[str]:
    return sorted(
        name
        for name, value in payload.get("gates", {}).items()
        if not _gate_passed(value)
        and not (not county_required and name in COUNTY_GATE_NAMES)
    )


def _semantic_metrics(payload: Mapping[str, Any]) -> dict[str, float] | None:
    scores = payload.get("scores", {})
    state = scores.get("semantic_full_line", {}).get("state_coast")
    silhouette = scores.get("silhouette")
    if not isinstance(state, Mapping) or not isinstance(silhouette, Mapping):
        return None
    reference_to_source = state.get("reference_to_source", {})
    try:
        median = float(reference_to_source["median_px"])
        p90 = float(reference_to_source["p90_px"])
        f1 = float(state["f1"])
        containment = float(silhouette["land_containment_fraction"])
    except (KeyError, TypeError, ValueError):
        return None
    return {
        "state_median_px": median,
        "state_p90_px": p90,
        "state_f1": f1,
        "land_containment_fraction": containment,
        "semantic_alignment_score": (
            p90 + median + 10.0 * (1.0 - f1) + 10.0 * (1.0 - containment)
        ),
    }


def _candidate_key(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    metrics = candidate["metrics"]
    return (
        metrics["semantic_alignment_score"],
        metrics["state_p90_px"],
        metrics["state_median_px"],
        -metrics["state_f1"],
        -metrics["land_containment_fraction"],
        float(candidate["objective"]),
        str(candidate["projection"]),
        str(candidate["model"]),
        str(candidate["path"]),
    )


def _alignment_record(extraction: Mapping[str, Any]) -> Mapping[str, Any] | None:
    alignment = extraction.get("alignment")
    if isinstance(alignment, Mapping):
        return alignment
    accepted = extraction.get("accepted_alignment")
    return accepted if isinstance(accepted, Mapping) else None


def audit_live_alignment_nonregression(
    registry_path: Path, output_path: Path
) -> dict[str, Any]:
    """Audit every live extraction's bound alignment and same-run alternatives."""

    registry_path = registry_path.resolve()
    repository_root = registry_path.parent.parent
    registry = json.loads(registry_path.read_text())
    dataset_reports: list[dict[str, Any]] = []
    for dataset in registry["datasets"]:
        extraction_path = _resolve(repository_root, dataset["accepted_extraction"])
        extraction = json.loads(extraction_path.read_text())
        extraction_hash_matches = _sha256(extraction_path) == dataset.get(
            "accepted_extraction_sha256"
        )
        alignment_record = _alignment_record(extraction)
        if alignment_record is None:
            dataset_reports.append(
                {
                    "public_directory": dataset["public_directory"],
                    "status": "fail",
                    "reason": "accepted extraction has no bound alignment",
                    "extraction_hash_matches_registry": extraction_hash_matches,
                }
            )
            continue
        alignment_path = _resolve(repository_root, str(alignment_record["path"]))
        alignment = json.loads(alignment_path.read_text())
        alignment_hash_matches = _sha256(alignment_path) == alignment_record.get(
            "sha256"
        )
        accepted_metrics = _semantic_metrics(alignment)
        candidate_files = sorted(alignment_path.parent.glob("alignment-*/candidate.json"))
        if accepted_metrics is None or not candidate_files:
            failed = _effective_failed_gates(alignment, county_required=True)
            status_value = str(alignment.get("status", alignment.get("decision", "")))
            accepted_status = status_value in {"pass", "accept", "accepted"}
            passed = (
                extraction_hash_matches
                and alignment_hash_matches
                and accepted_status
                and not failed
            )
            dataset_reports.append(
                {
                    "public_directory": dataset["public_directory"],
                    "mode": "specialized_alignment_gate_audit",
                    "status": "pass" if passed else "fail",
                    "accepted_alignment": str(alignment_path),
                    "accepted_status": status_value,
                    "failed_gates": failed,
                    "extraction_hash_matches_registry": extraction_hash_matches,
                    "alignment_hash_matches_extraction": alignment_hash_matches,
                }
            )
            continue

        county_required = _county_required(alignment)
        candidates = []
        for candidate_path in candidate_files:
            candidate_payload = json.loads(candidate_path.read_text())
            metrics = _semantic_metrics(candidate_payload)
            if (
                metrics is None
                or candidate_payload.get("source_sha256")
                != alignment.get("source_sha256")
            ):
                continue
            failed = _effective_failed_gates(
                candidate_payload, county_required=county_required
            )
            candidates.append(
                {
                    "path": str(candidate_path),
                    "iteration": candidate_payload.get("iteration"),
                    "projection": candidate_payload.get("projection"),
                    "model": candidate_payload.get("model"),
                    "objective": float(
                        candidate_payload.get("scores", {}).get("objective", float("inf"))
                    ),
                    "metrics": metrics,
                    "failed_gates": failed,
                    "passes_effective_gates": not failed,
                }
            )
        passing = sorted(
            (candidate for candidate in candidates if candidate["passes_effective_gates"]),
            key=_candidate_key,
        )
        best = passing[0] if passing else None
        accepted_key = (
            accepted_metrics["semantic_alignment_score"],
            accepted_metrics["state_p90_px"],
            accepted_metrics["state_median_px"],
            -accepted_metrics["state_f1"],
            -accepted_metrics["land_containment_fraction"],
            float(alignment.get("scores", {}).get("objective", float("inf"))),
            str(alignment.get("projection")),
            str(alignment.get("model")),
        )
        best_key = _candidate_key(best)[:-1] if best is not None else None
        accepted_is_global_best = best_key is not None and accepted_key <= best_key
        passed = (
            extraction_hash_matches
            and alignment_hash_matches
            and not _effective_failed_gates(
                alignment, county_required=county_required
            )
            and accepted_is_global_best
        )
        dataset_reports.append(
            {
                "public_directory": dataset["public_directory"],
                "mode": "complete_candidate_ensemble_nonregression",
                "status": "pass" if passed else "fail",
                "accepted_alignment": str(alignment_path),
                "county_channel_required_for_acceptance": county_required,
                "comparable_candidate_count": len(candidates),
                "passing_candidate_count": len(passing),
                "accepted_metrics": accepted_metrics,
                "best_passing_candidate": best,
                "accepted_is_global_best": accepted_is_global_best,
                "accepted_failed_gates": _effective_failed_gates(
                    alignment, county_required=county_required
                ),
                "extraction_hash_matches_registry": extraction_hash_matches,
                "alignment_hash_matches_extraction": alignment_hash_matches,
            }
        )

    failures = [item["public_directory"] for item in dataset_reports if item["status"] != "pass"]
    report = {
        "schema_version": 1,
        "kind": "mapscan_live_alignment_nonregression_audit",
        "registry": str(registry_path),
        "selection_policy": "global_best_passing_semantic_alignment_candidate",
        "dataset_count": len(dataset_reports),
        "failure_count": len(failures),
        "failed_public_directories": failures,
        "status": "pass" if not failures else "fail",
        "datasets": dataset_reports,
    }
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    return report
