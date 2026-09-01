"""Audit every live MapScan dataset against its immutable source authority."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _bound_path(record: object, base: Path) -> Path | None:
    if not isinstance(record, Mapping):
        return None
    raw = record.get("path")
    digest = record.get("sha256")
    if not isinstance(raw, str) or not isinstance(digest, str):
        return None
    path = Path(raw)
    candidates = [path] if path.is_absolute() else [base / path, Path.cwd() / path]
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file() and _sha256(resolved) == digest:
            return resolved
    return None


def _gate_passed(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, Mapping) and "passed" in value:
        return bool(value["passed"])
    return True


def audit_live_dataset_readiness(
    registry_path: Path,
    catalog_path: Path,
    public_root: Path,
    alignment_audit_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Bind publication, extraction, fixed-point, and coverage evidence."""

    registry_path = registry_path.resolve()
    repository_root = registry_path.parents[1]
    catalog_path = catalog_path.resolve()
    public_root = public_root.resolve()
    alignment_audit_path = alignment_audit_path.resolve()
    output_path = output_path.resolve()
    registry = _json(registry_path)
    catalog = _json(catalog_path)
    alignment_audit = _json(alignment_audit_path)
    alignment_by_directory = {
        item["public_directory"]: item
        for item in alignment_audit.get("datasets", [])
        if isinstance(item, Mapping) and isinstance(item.get("public_directory"), str)
    }
    catalog_by_manifest = {
        item["manifest"]: item
        for item in catalog.get("datasets", [])
        if isinstance(item, Mapping) and isinstance(item.get("manifest"), str)
    }

    datasets: list[dict[str, Any]] = []
    for authority in registry.get("datasets", []):
        public_directory = str(authority["public_directory"])
        pointer_path = (repository_root / str(authority["accepted_extraction"])).resolve()
        public_directory_path = public_root / public_directory
        public_manifest_path = public_directory_path / "dataset.json"
        provenance_path = public_directory_path / "provenance.json"
        manifest_relative = f"datasets/{public_directory}/dataset.json"
        pointer = _json(pointer_path)
        public_manifest = _json(public_manifest_path)
        provenance = _json(provenance_path)

        accepted_iteration = pointer.get("accepted_iteration")
        iteration_report_path = (
            pointer_path.parent / str(accepted_iteration) / "iteration.json"
            if accepted_iteration
            else None
        )
        extraction_report = (
            _json(iteration_report_path)
            if iteration_report_path is not None and iteration_report_path.is_file()
            else pointer
        )
        gates = extraction_report.get("gates", {})
        failed_gates = [
            str(name)
            for name, value in gates.items()
            if not _gate_passed(value)
        ] if isinstance(gates, Mapping) else ["invalid_gate_payload"]
        deterministic_gates = [
            str(name)
            for name, value in gates.items()
            if ("fixed_point" in str(name) or "deterministic" in str(name))
            and _gate_passed(value)
        ] if isinstance(gates, Mapping) else []

        source_path = _bound_path(pointer.get("source"), pointer_path.parent)
        alignment_record = pointer.get("alignment") or pointer.get("accepted_alignment")
        alignment_path = _bound_path(alignment_record, pointer_path.parent)
        public_accepted = provenance.get("accepted_extraction", {})
        coverage = provenance.get("publication_coverage", {})
        layers = coverage.get("layers", []) if isinstance(coverage, Mapping) else []
        outside_pixels = sum(
            int(layer.get("colored_pixel_count_outside_state", -1))
            for layer in layers
            if isinstance(layer, Mapping)
        )
        catalog_entry = catalog_by_manifest.get(manifest_relative)
        alignment_entry = alignment_by_directory.get(public_directory)
        checks = {
            "registry_pointer_hash_matches": _sha256(pointer_path)
            == authority.get("accepted_extraction_sha256"),
            "accepted_extraction_status": pointer.get("status") == "accepted",
            "source_hash_resolves": source_path is not None,
            "alignment_hash_resolves": alignment_path is not None,
            "all_extraction_gates_pass": not failed_gates,
            "deterministic_or_fixed_point_evidence": bool(deterministic_gates),
            "live_alignment_nonregression_passes": bool(alignment_entry)
            and alignment_entry.get("status") == "pass",
            "public_provenance_binds_extraction": public_accepted.get("sha256")
            == authority.get("accepted_extraction_sha256"),
            "public_package_approved": provenance.get("publication_approved") is True
            and provenance.get("status") == "approved_publication"
            and public_manifest.get("status") == "approved_publication",
            "catalog_binds_public_manifest": catalog_entry is not None,
            "publication_layers_declared": bool(layers),
            "publication_exterior_is_empty": outside_pixels == 0,
        }
        failed_checks = [name for name, passed in checks.items() if not passed]
        datasets.append(
            {
                "public_directory": public_directory,
                "dataset_id": public_manifest.get("id"),
                "schema_version": pointer.get("schema_version"),
                "accepted_extraction": str(pointer_path),
                "accepted_extraction_sha256": _sha256(pointer_path),
                "source": str(source_path) if source_path else None,
                "alignment": str(alignment_path) if alignment_path else None,
                "automatic_iteration_count": pointer.get("automatic_iteration_count")
                or pointer.get("automatic_extraction_iteration_count"),
                "accepted_iteration": accepted_iteration,
                "passed_determinism_gates": deterministic_gates,
                "failed_extraction_gates": failed_gates,
                "publication_layer_count": len(layers),
                "colored_pixel_count_outside_state": outside_pixels,
                "checks": checks,
                "failed_checks": failed_checks,
                "status": "pass" if not failed_checks else "fail",
            }
        )

    failures = [item for item in datasets if item["status"] != "pass"]
    report = {
        "schema_version": 1,
        "kind": "mapscan_live_dataset_readiness_audit",
        "policy": (
            "Every live map must bind its accepted source and alignment, pass its "
            "schema-specific extraction and fixed-point gates, remain approved in "
            "the catalog, and publish zero colored pixels outside California."
        ),
        "registry": str(registry_path),
        "registry_sha256": _sha256(registry_path),
        "catalog": str(catalog_path),
        "catalog_sha256": _sha256(catalog_path),
        "alignment_audit": str(alignment_audit_path),
        "alignment_audit_sha256": _sha256(alignment_audit_path),
        "dataset_count": len(datasets),
        "passing_dataset_count": len(datasets) - len(failures),
        "failure_count": len(failures),
        "failed_public_directories": [item["public_directory"] for item in failures],
        "status": "pass" if not failures else "fail",
        "datasets": datasets,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    return report
