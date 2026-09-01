"""Manifest-driven orchestration for source-clean, no-human MapScan restarts."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping

from .experiment_log import NoHumanExperimentLog


SCHEMA_VERSION = "mapscan.no-human-restart.v1"
_MAP_ID = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_TOP_LEVEL_FIELDS = {"schema_version", "mapbox_reference", "maps"}
_MAP_FIELDS = {"id", "title", "source", "source_type"}
_DEPRECATED_SOURCE_NAMES = {"county.png", "farms.png"}
_FORBIDDEN_FIELD_PARTS = {
    "alignment_path",
    "alignment_dir",
    "approval",
    "approved",
    "arrow",
    "candidate_path",
    "control_point",
    "correction",
    "manual",
    "materialization",
    "paint",
    "stamp",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _escape_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _forbidden_field(record: Mapping[str, Any]) -> str | None:
    """Locate legacy intervention/materialization fields recursively."""

    for key, value in record.items():
        normalized = str(key).lower().replace("-", "_").replace(" ", "_")
        if any(part in normalized for part in _FORBIDDEN_FIELD_PARTS):
            return str(key)
        if isinstance(value, Mapping):
            nested = _forbidden_field(value)
            if nested is not None:
                return f"{key}.{nested}"
        elif isinstance(value, list):
            for index, item in enumerate(value):
                if isinstance(item, Mapping):
                    nested = _forbidden_field(item)
                    if nested is not None:
                        return f"{key}[{index}].{nested}"
    return None


def _find_deprecated_source(value: Any) -> str | None:
    if isinstance(value, str) and Path(value).name.casefold() in _DEPRECATED_SOURCE_NAMES:
        return Path(value).name
    if isinstance(value, Mapping):
        for nested in value.values():
            found = _find_deprecated_source(nested)
            if found is not None:
                return found
    if isinstance(value, list):
        for nested in value:
            found = _find_deprecated_source(nested)
            if found is not None:
                return found
    return None


def _validate_reference(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("restart manifest requires one pinned Mapbox reference")
    reference = dict(value)
    reference_id = str(reference.get("id", ""))
    provider = str(reference.get("provider", ""))
    if "mapbox" not in reference_id.lower() and provider.lower() != "mapbox":
        raise ValueError("reference must identify Mapbox as its provider")
    hashes = {
        key: candidate
        for key, candidate in reference.items()
        if str(key).lower().endswith("sha256")
    }
    if not hashes:
        raise ValueError("pinned Mapbox reference requires at least one SHA-256")
    for key, candidate in hashes.items():
        if not isinstance(candidate, str) or re.fullmatch(r"[0-9a-fA-F]{64}", candidate) is None:
            raise ValueError(f"Mapbox reference field {key} is not a SHA-256")
    return reference


class NoHumanRestartRegistry:
    """Initialize/resume per-map logs and summarize their automatic progress."""

    def __init__(self, manifest_path: Path, run_root: Path) -> None:
        self.manifest_path = manifest_path.resolve()
        self.run_root = run_root.resolve()
        self.reference: dict[str, Any] = {}
        self.maps: list[dict[str, Any]] = []
        self.logs: dict[str, NoHumanExperimentLog] = {}
        self._read_manifest()

    def _read_manifest(self) -> None:
        if not self.manifest_path.is_file():
            raise FileNotFoundError(self.manifest_path)
        raw = json.loads(self.manifest_path.read_text())
        if not isinstance(raw, Mapping):
            raise ValueError("restart manifest must contain a JSON object")
        unknown_top = set(raw) - _TOP_LEVEL_FIELDS
        if unknown_top:
            raise ValueError(
                f"restart manifest has unsupported fields: {sorted(unknown_top)}"
            )
        version = raw.get("schema_version", SCHEMA_VERSION)
        if version != SCHEMA_VERSION:
            raise ValueError("unsupported restart-manifest schema")
        forbidden = _forbidden_field(raw)
        if forbidden is not None:
            raise ValueError(
                f"restart manifest contains legacy/manual field {forbidden!r}"
            )
        deprecated = _find_deprecated_source(raw)
        if deprecated is not None:
            raise ValueError(
                f"deprecated restart input {deprecated!r}; use the current source instead"
            )
        self.reference = _validate_reference(raw.get("mapbox_reference"))
        map_records = raw.get("maps")
        if not isinstance(map_records, list) or not map_records:
            raise ValueError("restart manifest requires a non-empty maps list")

        seen: set[str] = set()
        maps: list[dict[str, Any]] = []
        for index, item in enumerate(map_records):
            if not isinstance(item, Mapping):
                raise ValueError(f"maps[{index}] must be an object")
            unknown = set(item) - _MAP_FIELDS
            if unknown:
                raise ValueError(
                    f"maps[{index}] has unsupported restart fields: {sorted(unknown)}"
                )
            map_id = str(item.get("id", ""))
            identity = map_id.casefold()
            if identity in seen:
                raise ValueError(f"duplicate map id {map_id!r}")
            if _MAP_ID.fullmatch(map_id) is None or map_id in {".", ".."}:
                raise ValueError(f"maps[{index}] has invalid id {map_id!r}")
            seen.add(identity)
            source_value = item.get("source")
            if not isinstance(source_value, str) or not source_value.strip():
                raise ValueError(f"maps[{index}] requires a source path")
            source = Path(source_value)
            if not source.is_absolute():
                source = self.manifest_path.parent / source
            source = source.resolve()
            if not source.is_file():
                raise FileNotFoundError(f"missing source for {map_id}: {source}")
            if source.name.casefold() in _DEPRECATED_SOURCE_NAMES:
                raise ValueError(f"deprecated restart input {source.name!r}")
            maps.append(
                {
                    "id": map_id,
                    "title": str(item.get("title") or map_id),
                    "source": str(source),
                    "source_type": item.get("source_type"),
                    "source_sha256": _sha256(source),
                }
            )
        self.maps = maps

    def _snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "manifest_path": str(self.manifest_path),
            "mapbox_reference": self.reference,
            "maps": self.maps,
        }

    def _write_or_verify_snapshot(self) -> None:
        path = self.run_root / "restart-manifest.snapshot.json"
        expected = self._snapshot()
        if path.exists():
            if json.loads(path.read_text()) != expected:
                raise ValueError("restart manifest or source hashes changed for this run root")
            return
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(json.dumps(expected, indent=2, ensure_ascii=False) + "\n")
        os.replace(temporary, path)

    def initialize(self) -> dict[str, Any]:
        """Create or resume every map log, then render the aggregate index."""

        self.run_root.mkdir(parents=True, exist_ok=True)
        self._write_or_verify_snapshot()
        logs: dict[str, NoHumanExperimentLog] = {}
        for record in self.maps:
            map_dir = self.run_root / record["id"]
            markdown_path = map_dir / "EXPERIMENT.md"
            json_path = map_dir / "EXPERIMENT.json"
            if markdown_path.exists() and not json_path.exists():
                raise ValueError(
                    f"cannot resume {record['id']}: EXPERIMENT.json is missing"
                )
            if json_path.exists():
                log = NoHumanExperimentLog.load(json_path)
                if log.data.get("map_id") != record["id"]:
                    raise ValueError(f"experiment-log id mismatch for {record['id']}")
                source = log.data.get("source", {})
                if (
                    source.get("path") != record["source"]
                    or source.get("sha256") != record["source_sha256"]
                ):
                    raise ValueError(f"experiment-log source mismatch for {record['id']}")
                if log.data.get("mapbox_reference") != self.reference:
                    raise ValueError(
                        f"experiment-log Mapbox reference mismatch for {record['id']}"
                    )
            else:
                log = NoHumanExperimentLog(
                    record["id"],
                    Path(record["source"]),
                    mapbox_reference=self.reference,
                    source_type=record.get("source_type"),
                )
            log.write(markdown_path, json_path)
            logs[record["id"]] = log
        self.logs = logs
        index = self.refresh_index()
        return {
            "run_root": str(self.run_root),
            "map_count": len(self.maps),
            "index": str(index),
            "logs": {
                map_id: str(self.run_root / map_id / "EXPERIMENT.md")
                for map_id in logs
            },
        }

    def _load_current_logs(self) -> None:
        logs: dict[str, NoHumanExperimentLog] = {}
        for record in self.maps:
            path = self.run_root / record["id"] / "EXPERIMENT.json"
            if not path.is_file():
                raise FileNotFoundError(f"missing experiment log for {record['id']}: {path}")
            log = NoHumanExperimentLog.load(path)
            if log.data.get("map_id") != record["id"]:
                raise ValueError(f"experiment-log id mismatch for {record['id']}")
            source = log.data.get("source", {})
            if (
                source.get("path") != record["source"]
                or source.get("sha256") != record["source_sha256"]
            ):
                raise ValueError(f"experiment-log source mismatch for {record['id']}")
            if log.data.get("mapbox_reference") != self.reference:
                raise ValueError(
                    f"experiment-log Mapbox reference mismatch for {record['id']}"
                )
            logs[record["id"]] = log
        self.logs = logs

    def render_index(self) -> str:
        lines = [
            "# No-human MapScan restart",
            "",
            "> Machine-generated aggregate. Counts include only iterations whose provenance is automatic and excludes arrows, control points, stamps, painting, and human approvals.",
            "",
            f"Mapbox reference: `{json.dumps(self.reference, sort_keys=True, separators=(',', ':'))}`",
            "",
            "| Map | Source SHA-256 | Alignment automatic iterations | Alignment accepted at | Extraction automatic iterations | Extraction accepted at | Status | Blocker | Log |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |",
        ]
        for record in self.maps:
            log = self.logs[record["id"]]
            alignment = log.data["alignment"]
            extraction = log.data["extraction"]
            alignment_count = sum(
                bool(item["counts_toward_automatic_iteration_count"])
                for item in alignment["iterations"]
            )
            extraction_count = sum(
                bool(item["counts_toward_automatic_iteration_count"])
                for item in extraction["iterations"]
            )
            final = log.data["final"]
            relative = f"{record['id']}/EXPERIMENT.md"
            lines.append(
                "| {title} | `{source_hash}` | {align_count} | {align_accept} | {extract_count} | {extract_accept} | {status} | {blocker} | [experiment]({relative}) |".format(
                    title=_escape_cell(record["title"]),
                    source_hash=record["source_sha256"],
                    align_count=alignment_count,
                    align_accept=alignment["accepted_automatic_iteration_count"]
                    if alignment["accepted_automatic_iteration_count"] is not None
                    else "—",
                    extract_count=extraction_count,
                    extract_accept=extraction["accepted_automatic_iteration_count"]
                    if extraction["accepted_automatic_iteration_count"] is not None
                    else "—",
                    status=_escape_cell(final["status"]),
                    blocker=_escape_cell(final.get("blocker") or "—"),
                    relative=relative,
                )
            )
        lines.append("")
        return "\n".join(lines)

    def refresh_index(self) -> Path:
        """Reload agent-updated per-map logs and atomically rebuild ``INDEX.md``."""

        self._write_or_verify_snapshot()
        self._load_current_logs()
        path = self.run_root / "INDEX.md"
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(self.render_index())
        os.replace(temporary, path)
        return path
