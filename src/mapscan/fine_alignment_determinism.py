"""Determinism audit for repeated automatic county fine-alignment runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict


CORE_ARTIFACTS = (
    "automatic-county-corrections.json",
    "working-source-before.jpg",
    "working-source-after.jpg",
    "working-diagnostic-before.jpg",
    "working-diagnostic-after.jpg",
    "administrative-stroke-response.png",
    "web-mercator-source-before.jpg",
    "web-mercator-source-after.jpg",
    "web-mercator-county-png-state-overlay.png",
    "web-mercator-county-png-county-overlay.png",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selected_matrix(run_dir: Path) -> tuple[str, object]:
    fit = json.loads((run_dir / "correction-fit.json").read_text())
    selected = str(fit["selected_model"])
    candidate = next(item for item in fit["candidates"] if item["model"] == selected)
    return selected, candidate["matrix_current_to_target_pixels"]


def audit_fine_alignment_determinism(
    first_run: Path, second_run: Path, output_path: Path
) -> Dict[str, object]:
    first_run = first_run.resolve()
    second_run = second_run.resolve()
    first_report_path = first_run / "county-fine-alignment.json"
    second_report_path = second_run / "county-fine-alignment.json"
    first = json.loads(first_report_path.read_text())
    second = json.loads(second_report_path.read_text())
    first_model, first_matrix = _selected_matrix(first_run)
    second_model, second_matrix = _selected_matrix(second_run)

    provenance_keys = (
        ("source", "sha256"),
        ("parent_alignment", "sha256"),
        ("county_reference", "source_sha256"),
    )
    provenance = {}
    provenance_identical = True
    for section, key in provenance_keys:
        first_value = first[section][key]
        second_value = second[section][key]
        identical = first_value == second_value
        provenance[f"{section}.{key}"] = {
            "first": first_value,
            "second": second_value,
            "identical": identical,
        }
        provenance_identical = provenance_identical and identical

    artifact_hashes = {}
    for name in CORE_ARTIFACTS:
        first_path = first_run / name
        second_path = second_run / name
        first_hash = _sha256(first_path)
        second_hash = _sha256(second_path)
        artifact_hashes[name] = {
            "first": first_hash,
            "second": second_hash,
            "identical": first_hash == second_hash,
        }

    matrix_identical = first_model == second_model and first_matrix == second_matrix
    holdouts_identical = (
        first["independent_spatial_holdouts"]
        == second["independent_spatial_holdouts"]
    )
    after_metrics_identical = first["after"] == second["after"]
    vetoes_identical = (
        first["state_boundary_veto"] == second["state_boundary_veto"]
        and first["transform_regularity"] == second["transform_regularity"]
    )
    passed = bool(
        provenance_identical
        and matrix_identical
        and holdouts_identical
        and after_metrics_identical
        and vetoes_identical
        and all(item["identical"] for item in artifact_hashes.values())
    )
    report: Dict[str, object] = {
        "schema_version": 1,
        "audit_kind": "fine_alignment_repeat_determinism",
        "first_run": {
            "path": str(first_run),
            "report_sha256": _sha256(first_report_path),
        },
        "second_run": {
            "path": str(second_run),
            "report_sha256": _sha256(second_report_path),
        },
        "provenance": provenance,
        "provenance_identical": provenance_identical,
        "selected_model": {"first": first_model, "second": second_model},
        "matrix_identical": matrix_identical,
        "holdouts_identical": holdouts_identical,
        "after_metrics_identical": after_metrics_identical,
        "vetoes_identical": vetoes_identical,
        "artifact_hashes": artifact_hashes,
        "passed": passed,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    return report
