"""Read-only diagnostics for a no-human MapScan restart run.

The analyzer reads only the restart snapshot, experiment logs, and immutable
candidate artifacts.  Its sole write is the requested Markdown report; it does
not update experiment logs, counters, statuses, approvals, or ``INDEX.md``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "mapscan.restart-failure-report.v1"
DEFAULT_REPORT_NAME = "FAILURE_REPORT.md"
DIAGNOSTIC_FILENAMES = {
    "aligned-source-mapbox-overlay.png",
    "candidate.json",
    "semantic-full-line-validation.png",
    "source-county-evidence.png",
    "source-edges.png",
    "source-mapbox-evidence.png",
    "source-state-coast-evidence.png",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _number(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return _escape(value)
    if abs(numeric) >= 1000:
        return f"{numeric:,.1f}"
    return f"{numeric:.{digits}f}".rstrip("0").rstrip(".")


def _gate_passed(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, Mapping) and isinstance(value.get("passed"), bool):
        return bool(value["passed"])
    return None


def _failed_gate_text(name: str, value: Any) -> str:
    if isinstance(value, Mapping):
        parts = [name]
        if "value" in value:
            parts.append(f"value={_number(value['value'])}")
        if "minimum" in value:
            parts.append(f"min={_number(value['minimum'])}")
        if "maximum" in value:
            parts.append(f"max={_number(value['maximum'])}")
        if value.get("reason"):
            parts.append(str(value["reason"]))
        return " (".join((parts[0], ", ".join(parts[1:]))) + (")" if len(parts) > 1 else "")
    return name


def _failed_gates(iteration: Mapping[str, Any]) -> list[str]:
    return [
        _failed_gate_text(str(name), value)
        for name, value in iteration.get("gates", {}).items()
        if _gate_passed(value) is False
    ]


def _semantic_metrics(iteration: Mapping[str, Any], kind: str) -> dict[str, float] | None:
    semantic = iteration.get("scores", {}).get("semantic_full_line")
    if not isinstance(semantic, Mapping):
        return None
    record = semantic.get(kind)
    if not isinstance(record, Mapping):
        return None
    forward = record.get("reference_to_source")
    if not isinstance(forward, Mapping):
        return None
    values = {
        "median_px": forward.get("median_px"),
        "p90_px": forward.get("p90_px"),
        "within_8px_fraction": forward.get("within_8px_fraction"),
        "f1": record.get("f1"),
        "precision": record.get("precision"),
        "recall": record.get("recall"),
    }
    if not any(value is not None for value in values.values()):
        return None
    return {
        key: float(value)
        for key, value in values.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


def _candidate_artifact_payload(
    iteration: Mapping[str, Any],
    run_root: Path,
    map_dir: Path,
) -> Mapping[str, Any] | None:
    for artifact in iteration.get("artifacts", []):
        path = _resolve_artifact_path(artifact, run_root, map_dir)
        if path is None or path.name != "candidate.json":
            continue
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, Mapping):
            return payload
    return None


def _projection_model(
    iteration: Mapping[str, Any],
    payload: Mapping[str, Any] | None,
) -> tuple[str, str]:
    scores = iteration.get("scores", {})
    projection_record = scores.get("projection") if isinstance(scores, Mapping) else None
    projection = (
        str(projection_record.get("id"))
        if isinstance(projection_record, Mapping) and projection_record.get("id")
        else None
    )
    model = None
    if payload:
        projection = projection or (
            str(payload.get("projection")) if payload.get("projection") else None
        )
        model = str(payload.get("model")) if payload.get("model") else None
    method = str(iteration.get("method", ""))
    if model is None:
        match = re.search(r"\b(similarity|regular_affine)\b", method)
        model = match.group(1) if match else "unspecified"
    if projection is None:
        match = re.search(
            r"geographically balanced ([a-z0-9_]+) (?:similarity|regular_affine)",
            method,
        )
        projection = match.group(1) if match else "unspecified"
    return projection, model


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _resolve_artifact_path(
    artifact: Mapping[str, Any], run_root: Path, map_dir: Path
) -> Path | None:
    raw = artifact.get("path")
    if not isinstance(raw, str) or not raw.strip():
        return None
    value = Path(raw)
    candidates = [value] if value.is_absolute() else [map_dir / value, run_root / value]
    for candidate in candidates:
        path = candidate.resolve()
        if path.is_file() and _is_within(path, run_root):
            return path
    return None


def _diagnostic_links(
    iteration: Mapping[str, Any],
    run_root: Path,
    map_dir: Path,
    report_parent: Path,
) -> list[str]:
    links = []
    seen: set[Path] = set()
    for artifact in iteration.get("artifacts", []):
        path = _resolve_artifact_path(artifact, run_root, map_dir)
        if path is None or path in seen or path.name not in DIAGNOSTIC_FILENAMES:
            continue
        seen.add(path)
        relative = os.path.relpath(path, report_parent)
        links.append(f"[{path.name}]({Path(relative).as_posix()})")
    return links


def _automatic_iterations(record: Mapping[str, Any], phase: str) -> list[Mapping[str, Any]]:
    return [
        item
        for item in record.get(phase, {}).get("iterations", [])
        if item.get("counts_toward_automatic_iteration_count") is True
    ]


def _best_candidate(
    candidates: Sequence[Mapping[str, Any]],
    kind: str,
) -> tuple[Mapping[str, Any], Mapping[str, float]] | None:
    available = []
    for item in candidates:
        metrics = _semantic_metrics(item, kind)
        if metrics is None:
            continue
        available.append((item, metrics))
    if not available:
        return None
    return max(
        available,
        key=lambda pair: (
            pair[1].get("f1", -1.0),
            pair[1].get("within_8px_fraction", -1.0),
            -pair[1].get("p90_px", 1e12),
            -pair[1].get("median_px", 1e12),
        ),
    )


def _metric_summary(
    best: tuple[Mapping[str, Any], Mapping[str, float]] | None,
    run_root: Path,
    map_dir: Path,
) -> str:
    if best is None:
        return "not available"
    item, metrics = best
    payload = _candidate_artifact_payload(item, run_root, map_dir)
    projection, model = _projection_model(item, payload)
    return (
        f"#{item.get('automatic_iteration') or item.get('attempt')} "
        f"{projection}/{model}; F1 {_number(metrics.get('f1'))}, "
        f"median {_number(metrics.get('median_px'))} px, "
        f"p90 {_number(metrics.get('p90_px'))} px, "
        f"within 8 px {_number(metrics.get('within_8px_fraction'))}"
    )


def _candidate_row(
    item: Mapping[str, Any],
    run_root: Path,
    map_dir: Path,
    report_parent: Path,
) -> str:
    payload = _candidate_artifact_payload(item, run_root, map_dir)
    projection, model = _projection_model(item, payload)
    state = _semantic_metrics(item, "state_coast")
    counties = _semantic_metrics(item, "counties")
    failed = _failed_gates(item)
    diagnostics = _diagnostic_links(item, run_root, map_dir, report_parent)

    def metric(value: Mapping[str, float] | None) -> str:
        if value is None:
            return "—"
        return (
            f"med {_number(value.get('median_px'))}; "
            f"p90 {_number(value.get('p90_px'))}; "
            f"F1 {_number(value.get('f1'))}"
        )

    return (
        "| {number} | {projection} | {model} | {decision} | {objective} | "
        "{state} | {counties} | {failed} | {diagnostics} |"
    ).format(
        number=item.get("automatic_iteration") or item.get("attempt") or "—",
        projection=_escape(projection),
        model=_escape(model),
        decision=_escape(item.get("decision", "—")),
        objective=_number(item.get("scores", {}).get("objective")),
        state=_escape(metric(state)),
        counties=_escape(metric(counties)),
        failed=_escape("; ".join(failed) if failed else "none"),
        diagnostics=" · ".join(diagnostics) if diagnostics else "—",
    )


@dataclass(frozen=True)
class FailureReportResult:
    report_path: Path
    report_sha256: str
    map_count: int
    alignment_candidate_count: int
    accepted_alignment_candidate_count: int
    blocked_alignment_candidate_count: int


def build_restart_failure_report(
    run_root: Path,
    output_path: Path | None = None,
) -> FailureReportResult:
    """Summarize automatic candidates without modifying the run being inspected."""

    run_root = run_root.resolve()
    snapshot_path = run_root / "restart-manifest.snapshot.json"
    if not snapshot_path.is_file():
        raise FileNotFoundError(snapshot_path)
    snapshot = json.loads(snapshot_path.read_text())
    maps = snapshot.get("maps")
    if not isinstance(maps, list) or not maps:
        raise ValueError("restart snapshot has no maps")
    output_path = (output_path or run_root / DEFAULT_REPORT_NAME).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_parent = output_path.parent

    records: list[tuple[Mapping[str, Any], Mapping[str, Any] | None, str | None]] = []
    total_candidates = accepted_candidates = blocked_candidates = 0
    for map_record in maps:
        map_id = str(map_record.get("id", ""))
        experiment_path = run_root / map_id / "EXPERIMENT.json"
        if not experiment_path.is_file():
            records.append((map_record, None, "EXPERIMENT.json is missing"))
            continue
        try:
            experiment = json.loads(experiment_path.read_text())
        except json.JSONDecodeError as error:
            records.append((map_record, None, f"EXPERIMENT.json is invalid: {error}"))
            continue
        candidates = _automatic_iterations(experiment, "alignment")
        total_candidates += len(candidates)
        accepted_candidates += sum(item.get("decision") == "accept" for item in candidates)
        blocked_candidates += sum(item.get("decision") == "blocked" for item in candidates)
        records.append((map_record, experiment, None))

    lines = [
        "# Autonomous MapScan failure report",
        "",
        "> Read-only diagnostic generated from the restart snapshot, per-map experiment logs, and immutable automatic candidate artifacts. It does not alter acceptance state or iteration counts.",
        "",
        "## Run summary",
        "",
        f"- Report schema: `{SCHEMA_VERSION}`",
        f"- Run: `{run_root}`",
        f"- Mapbox reference: `{snapshot.get('mapbox_reference', {}).get('id', 'unknown')}`",
        f"- Maps: **{len(maps)}**",
        f"- Automatic alignment candidates: **{total_candidates}**",
        f"- Accepted alignment candidates: **{accepted_candidates}**",
        f"- Blocked alignment candidates: **{blocked_candidates}**",
        "",
        "| Map | Data model | Alignment candidates | Accepted | Blocked | Accepted at | Extraction candidates | Extraction accepted at | Best state/coast | Best counties | Final status |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for map_record, experiment, error in records:
        map_id = str(map_record.get("id", ""))
        map_dir = run_root / map_id
        if experiment is None:
            lines.append(
                f"| {_escape(map_record.get('title') or map_id)} | {_escape(map_record.get('source_type') or 'unspecified')} | 0 | 0 | 0 | — | 0 | — | not available | not available | {_escape(error)} |"
            )
            continue
        candidates = _automatic_iterations(experiment, "alignment")
        extraction = _automatic_iterations(experiment, "extraction")
        alignment = experiment.get("alignment", {})
        extraction_record = experiment.get("extraction", {})
        final = experiment.get("final", {})
        best_state = _metric_summary(
            _best_candidate(candidates, "state_coast"), run_root, map_dir
        )
        best_counties = _metric_summary(
            _best_candidate(candidates, "counties"), run_root, map_dir
        )
        lines.append(
            "| {title} | {source_type} | {count} | {accepted} | {blocked} | {accepted_at} | {extract_count} | {extract_at} | {state} | {counties} | {status} |".format(
                title=_escape(map_record.get("title") or map_id),
                source_type=_escape(map_record.get("source_type") or "unspecified"),
                count=len(candidates),
                accepted=sum(item.get("decision") == "accept" for item in candidates),
                blocked=sum(item.get("decision") == "blocked" for item in candidates),
                accepted_at=alignment.get("accepted_automatic_iteration_count") or "—",
                extract_count=len(extraction),
                extract_at=extraction_record.get("accepted_automatic_iteration_count") or "—",
                state=_escape(best_state),
                counties=_escape(best_counties),
                status=_escape(final.get("status", "unknown")),
            )
        )

    for map_record, experiment, error in records:
        map_id = str(map_record.get("id", ""))
        title = str(map_record.get("title") or map_id)
        map_dir = run_root / map_id
        lines.extend(["", f"## {title}", ""])
        if experiment is None:
            lines.extend([f"- Status: **unavailable**", f"- Error: {_escape(error)}", ""])
            continue
        candidates = _automatic_iterations(experiment, "alignment")
        extraction = _automatic_iterations(experiment, "extraction")
        final = experiment.get("final", {})
        lines.extend(
            [
                f"- Map ID: `{map_id}`",
                f"- Data model: `{map_record.get('source_type') or 'unspecified'}`",
                f"- Final status: **{final.get('status', 'unknown')}**",
                f"- Blocker: {final.get('blocker') or 'none'}",
                f"- Automatic alignment candidates: **{len(candidates)}**; accepted **{sum(item.get('decision') == 'accept' for item in candidates)}**; blocked **{sum(item.get('decision') == 'blocked' for item in candidates)}**; accepted at **{experiment.get('alignment', {}).get('accepted_automatic_iteration_count') or 'not accepted'}**",
                f"- Automatic extraction candidates: **{len(extraction)}**; accepted at **{experiment.get('extraction', {}).get('accepted_automatic_iteration_count') or 'not accepted'}**",
                f"- Best state/coast semantic result: {_metric_summary(_best_candidate(candidates, 'state_coast'), run_root, map_dir)}",
                f"- Best county semantic result: {_metric_summary(_best_candidate(candidates, 'counties'), run_root, map_dir)}",
                "",
            ]
        )
        if not candidates:
            lines.extend(["No automatic alignment candidate has been recorded.", ""])
            continue
        lines.extend(
            [
                "| Automatic # | Projection | Model | Decision | Objective | State/coast med-p90-F1 | County med-p90-F1 | Failed gates | Diagnostics |",
                "| ---: | --- | --- | --- | ---: | --- | --- | --- | --- |",
            ]
        )
        lines.extend(
            _candidate_row(item, run_root, map_dir, report_parent)
            for item in candidates
        )
        lines.append("")

    content = "\n".join(lines)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    try:
        temporary.write_text(content)
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return FailureReportResult(
        output_path,
        _sha256(output_path),
        len(maps),
        total_candidates,
        accepted_candidates,
        blocked_candidates,
    )
