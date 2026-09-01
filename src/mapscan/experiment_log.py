"""Auditable Markdown logs for fully automatic MapScan experiments.

The log intentionally distinguishes an automatic iteration from an attempt that
contains human-authored alignment arrows, clone stamps, or an approval decision.
Those attempts can be retained as historical evidence, but they never increment
the automatic iteration counts and can never satisfy an automatic phase gate.
"""

from __future__ import annotations

import hashlib
import json
import mimetypes
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "mapscan.no-human-experiment-log.v1"
VALID_DECISIONS = {"accept", "retry", "reject", "blocked"}
MANUAL_INPUT_TOKENS = {
    "approval",
    "arrow",
    "brush",
    "click",
    "control_point",
    "human",
    "manual",
    "paint",
    "stamp",
}

DEFAULT_APPROACH = (
    "Pin the Mapbox California state, coast/water, and county vector geometry by "
    "identifier and content hash.",
    "Align the unmodified source image to the Mapbox state perimeter, using "
    "county geometry as secondary evidence and geographically withheld validation "
    "only when a county-like source network is automatically observable.",
    "Render and score the complete bounded candidate-warp ensemble against the "
    "original source, apply capability-aware automatic gates, and accept only the "
    "globally best passing candidate after every alternative has been ranked.",
    "Lock the accepted transform before extracting thematic data.",
    "Read the legend and classify pixels from the aligned source; retain observed "
    "and automatically completed pixels as separate evidence.",
    "Reconstruct the extraction in legend colors, compare it with the aligned "
    "source globally and in native-resolution regions, and apply source-diff gates.",
    "Repeat extraction from the locked alignment until the automatic source-diff "
    "and fixed-point gates pass.",
    "Mark the map complete only when both phases have an automatically accepted "
    "iteration; manual arrows, stamps, painting, and approvals never count.",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _source_record(path: Path, source_type: str | None = None) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    media_type = source_type or mimetypes.guess_type(path.name)[0]
    return {
        "path": str(path),
        "filename": path.name,
        "source_type": media_type or "application/octet-stream",
        "suffix": path.suffix.lower(),
        "byte_count": path.stat().st_size,
        "sha256": _sha256(path),
    }


def automatic_provenance(
    producer: str,
    input_kinds: Sequence[str],
) -> dict[str, Any]:
    """Return explicit provenance for an attempt performed without human input."""

    if not producer.strip():
        raise ValueError("Automatic provenance requires a producer")
    if not input_kinds:
        raise ValueError("Automatic provenance requires declared input kinds")
    return {
        "actor_kind": "automated",
        "producer": producer,
        "input_kinds": [str(value) for value in input_kinds],
        "manual_arrows": False,
        "manual_stamps": False,
        "human_approval": False,
    }


def _contains_manual_token(value: str) -> bool:
    normalized = value.lower().replace("-", "_").replace(" ", "_")
    pieces = set(normalized.split("_"))
    return any(token in pieces or token in normalized for token in MANUAL_INPUT_TOKENS)


def _automatic_eligibility(provenance: Mapping[str, Any]) -> tuple[bool, str]:
    if provenance.get("actor_kind") != "automated":
        return False, "actor_kind is not automated"
    if provenance.get("manual_arrows") is True:
        return False, "manual arrows were used"
    if provenance.get("manual_stamps") is True:
        return False, "manual stamps or painting were used"
    if provenance.get("human_approval") is True:
        return False, "a human approval was used"
    inputs = provenance.get("input_kinds")
    if not isinstance(inputs, list) or not inputs:
        return False, "automatic input provenance is missing"
    manual_inputs = [str(value) for value in inputs if _contains_manual_token(str(value))]
    if manual_inputs:
        return False, f"manual input provenance: {', '.join(manual_inputs)}"
    if not str(provenance.get("producer", "")).strip():
        return False, "automatic producer is missing"
    return True, "eligible automatic iteration"


def _gate_passed(gate: Any) -> bool:
    if isinstance(gate, bool):
        return gate
    if isinstance(gate, Mapping) and isinstance(gate.get("passed"), bool):
        return bool(gate["passed"])
    raise ValueError("Each gate must be a boolean or an object with boolean 'passed'")


def _all_gates_pass(gates: Mapping[str, Any]) -> bool:
    if not gates:
        raise ValueError("An iteration requires at least one gate")
    return all(_gate_passed(value) for value in gates.values())


def _json_value(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _escape_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


class NoHumanExperimentLog:
    """Accumulate and render one no-human alignment/extraction experiment."""

    def __init__(
        self,
        map_id: str,
        source_path: Path,
        *,
        mapbox_reference: Mapping[str, Any],
        source_type: str | None = None,
        approach: Sequence[str] = DEFAULT_APPROACH,
        created_at: str | None = None,
    ) -> None:
        if not map_id.strip():
            raise ValueError("map_id is required")
        if not mapbox_reference:
            raise ValueError("A pinned Mapbox reference record is required")
        if not approach:
            raise ValueError("The experiment approach cannot be empty")
        self.data: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "map_id": map_id,
            "created_at": created_at or _utc_now(),
            "updated_at": created_at or _utc_now(),
            "automation_policy": {
                "human_intervention_allowed": False,
                "excluded_from_automatic_counts": [
                    "manual alignment arrows or control points",
                    "manual clone stamps, brushes, or painted pixels",
                    "human approval as a phase gate",
                ],
            },
            "approach": [str(step) for step in approach],
            "source": _source_record(source_path, source_type),
            "mapbox_reference": dict(mapbox_reference),
            "alignment": {
                "iterations": [],
                "accepted_automatic_iteration_count": None,
            },
            "extraction": {
                "iterations": [],
                "accepted_automatic_iteration_count": None,
            },
            "automatic_resumptions": [],
            "final": {"status": "in_progress", "blocker": None},
        }

    @classmethod
    def load(cls, path: Path) -> "NoHumanExperimentLog":
        """Resume a JSON experiment log previously written by :meth:`write`."""

        raw = json.loads(path.read_text())
        if raw.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported experiment-log schema")
        # ``automatic_resumptions`` was added as a backward-compatible audit
        # field.  Existing v1 logs remain valid and acquire an empty history.
        raw.setdefault("automatic_resumptions", [])
        instance = cls.__new__(cls)
        instance.data = raw
        return instance

    def _record_iteration(
        self,
        phase: str,
        *,
        scores: Mapping[str, Any],
        gates: Mapping[str, Any],
        decision: str,
        provenance: Mapping[str, Any],
        method: str,
        artifacts: Sequence[Mapping[str, Any]] = (),
        note: str | None = None,
        recorded_at: str | None = None,
    ) -> dict[str, Any]:
        if phase not in {"alignment", "extraction"}:
            raise ValueError("phase must be alignment or extraction")
        if decision not in VALID_DECISIONS:
            raise ValueError(f"decision must be one of {sorted(VALID_DECISIONS)}")
        if not method.strip():
            raise ValueError("Iteration method is required")
        if not scores:
            raise ValueError("An iteration requires at least one score")
        passed = _all_gates_pass(gates)
        eligible, eligibility_reason = _automatic_eligibility(provenance)
        phase_record = self.data[phase]
        if phase_record["accepted_automatic_iteration_count"] is not None:
            raise ValueError(f"The automatic {phase} phase is already accepted")
        if phase == "extraction" and self.data["alignment"][
            "accepted_automatic_iteration_count"
        ] is None:
            raise ValueError(
                "Extraction cannot start before an accepted automatic alignment"
            )
        if decision == "accept" and eligible and not passed:
            raise ValueError("An automatic iteration cannot be accepted with failed gates")

        automatic_count = sum(
            bool(item["counts_toward_automatic_iteration_count"])
            for item in phase_record["iterations"]
        )
        automatic_ordinal = automatic_count + 1 if eligible else None
        iteration = {
            "attempt": len(phase_record["iterations"]) + 1,
            "automatic_iteration": automatic_ordinal,
            "counts_toward_automatic_iteration_count": eligible,
            "automatic_eligibility_reason": eligibility_reason,
            "recorded_at": recorded_at or _utc_now(),
            "method": method,
            "scores": dict(scores),
            "gates": dict(gates),
            "all_gates_passed": passed,
            "decision": decision,
            "provenance": dict(provenance),
            "artifacts": [dict(value) for value in artifacts],
            "note": note,
        }
        phase_record["iterations"].append(iteration)
        if decision == "accept" and eligible:
            phase_record["accepted_automatic_iteration_count"] = automatic_ordinal
        self.data["updated_at"] = recorded_at or _utc_now()
        return iteration

    def record_alignment_iteration(self, **kwargs: Any) -> dict[str, Any]:
        return self._record_iteration("alignment", **kwargs)

    def record_extraction_iteration(self, **kwargs: Any) -> dict[str, Any]:
        return self._record_iteration("extraction", **kwargs)

    def finalize(self, status: str, blocker: str | None = None) -> None:
        if status not in {"complete", "blocked", "failed"}:
            raise ValueError("Final status must be complete, blocked, or failed")
        if status == "complete" and (
            self.data["alignment"]["accepted_automatic_iteration_count"] is None
            or self.data["extraction"]["accepted_automatic_iteration_count"] is None
        ):
            raise ValueError("Completion requires accepted automatic alignment and extraction")
        if status in {"blocked", "failed"} and not (blocker or "").strip():
            raise ValueError(f"A {status} experiment requires a blocker")
        self.data["final"] = {"status": status, "blocker": blocker}
        self.data["updated_at"] = _utc_now()

    def _resume_automatic_terminal(
        self,
        *,
        allowed_status: str,
        reason: str,
        producer: str,
        recorded_at: str | None = None,
    ) -> dict[str, Any]:
        """Reopen one explicit automatic terminal state without erasing history.

        A resumption is not an alignment or extraction iteration and therefore
        does not alter either automatic iteration count.  It records why a new
        algorithm version is allowed to append fresh attempts after an earlier
        exhaustive run blocked.
        """

        final = self.data.get("final", {})
        if final.get("status") != allowed_status:
            raise ValueError(
                f"Only a {allowed_status} experiment can be automatically resumed"
            )
        if not reason.strip():
            raise ValueError("Automatic resumption requires a reason")
        if not producer.strip():
            raise ValueError("Automatic resumption requires a producer")
        for phase in ("alignment", "extraction"):
            if any(
                not item.get("counts_toward_automatic_iteration_count", False)
                for item in self.data[phase]["iterations"]
            ):
                raise ValueError(
                    "Cannot automatically resume a log containing ineligible attempts"
                )
        timestamp = recorded_at or _utc_now()
        event = {
            "resumption": len(self.data.setdefault("automatic_resumptions", [])) + 1,
            "recorded_at": timestamp,
            "producer": producer,
            "reason": reason,
            "previous_status": allowed_status,
            "previous_blocker": final.get("blocker"),
        }
        self.data["automatic_resumptions"].append(event)
        self.data["final"] = {"status": "in_progress", "blocker": None}
        self.data["updated_at"] = timestamp
        return event

    def resume_automatic_blocked(
        self,
        *,
        reason: str,
        producer: str,
        recorded_at: str | None = None,
    ) -> dict[str, Any]:
        """Reopen an automatically blocked experiment without erasing history."""

        return self._resume_automatic_terminal(
            allowed_status="blocked",
            reason=reason,
            producer=producer,
            recorded_at=recorded_at,
        )

    def resume_automatic_failed(
        self,
        *,
        reason: str,
        producer: str,
        recorded_at: str | None = None,
    ) -> dict[str, Any]:
        """Reopen an explicitly selected automatic failure after a code fix."""

        return self._resume_automatic_terminal(
            allowed_status="failed",
            reason=reason,
            producer=producer,
            recorded_at=recorded_at,
        )

    def _phase_markdown(self, phase: str) -> list[str]:
        title = phase.capitalize()
        record = self.data[phase]
        accepted = record["accepted_automatic_iteration_count"]
        lines = [
            f"## {title} iterations",
            "",
            f"Accepted automatic iteration count: **{accepted if accepted is not None else 'not accepted'}**",
            "",
            "| Attempt | Automatic # | Counts | Method | Scores | Gates | Decision |",
            "| ---: | ---: | :---: | --- | --- | --- | --- |",
        ]
        for item in record["iterations"]:
            auto = item["automatic_iteration"]
            gate_summary = {
                key: _gate_passed(value) for key, value in item["gates"].items()
            }
            lines.append(
                "| {attempt} | {auto} | {counts} | {method} | `{scores}` | `{gates}` | {decision} |".format(
                    attempt=item["attempt"],
                    auto=auto if auto is not None else "—",
                    counts="yes" if item["counts_toward_automatic_iteration_count"] else "no",
                    method=_escape_cell(item["method"]),
                    scores=_escape_cell(_json_value(item["scores"])),
                    gates=_escape_cell(_json_value(gate_summary)),
                    decision=_escape_cell(item["decision"]),
                )
            )
        if not record["iterations"]:
            lines.append("| — | — | — | No iterations recorded | `{}` | `{}` | — |")
        lines.append("")
        for item in record["iterations"]:
            lines.extend(
                [
                    f"### {title} attempt {item['attempt']}",
                    "",
                    f"- Automatic eligibility: **{'counts' if item['counts_toward_automatic_iteration_count'] else 'excluded'}** — {item['automatic_eligibility_reason']}",
                    f"- Decision: **{item['decision']}**",
                    f"- All gates passed: **{'yes' if item['all_gates_passed'] else 'no'}**",
                    f"- Scores: `{_json_value(item['scores'])}`",
                    f"- Gates: `{_json_value(item['gates'])}`",
                    f"- Provenance: `{_json_value(item['provenance'])}`",
                ]
            )
            if item.get("artifacts"):
                lines.append(f"- Artifacts: `{_json_value(item['artifacts'])}`")
            if item.get("note"):
                lines.append(f"- Note: {_escape_cell(item['note'])}")
            lines.append("")
        return lines

    def render_markdown(self) -> str:
        source = self.data["source"]
        final = self.data["final"]
        lines = [
            f"# No-human MapScan experiment: {self.data['map_id']}",
            "",
            "> Machine-generated. Manual arrows, control points, stamps, painting, and human approvals are retained only as excluded evidence and never count as automatic iterations.",
            "",
            "## Source",
            "",
            f"- File: `{source['filename']}`",
            f"- Type: `{source['source_type']}`",
            f"- Bytes: `{source['byte_count']}`",
            f"- SHA-256: `{source['sha256']}`",
            f"- Path: `{source['path']}`",
            "",
            "## Mapbox reference",
            "",
            f"`{_json_value(self.data['mapbox_reference'])}`",
            "",
            "## Approach",
            "",
        ]
        lines.extend(
            f"{number}. {step}" for number, step in enumerate(self.data["approach"], 1)
        )
        lines.append("")
        lines.extend(self._phase_markdown("alignment"))
        lines.extend(self._phase_markdown("extraction"))
        resumptions = self.data.get("automatic_resumptions", [])
        lines.extend(["## Automatic resumptions", ""])
        if resumptions:
            for item in resumptions:
                lines.extend(
                    [
                        f"### Resumption {item['resumption']}",
                        "",
                        f"- Recorded at: `{item['recorded_at']}`",
                        f"- Producer: `{_escape_cell(item['producer'])}`",
                        f"- Reason: {_escape_cell(item['reason'])}",
                        f"- Previous status: `{item.get('previous_status', 'blocked')}`",
                        f"- Previous blocker: {_escape_cell(item['previous_blocker'] or 'none')}",
                        "",
                    ]
                )
        else:
            lines.extend(["No automatic resumptions recorded.", ""])
        lines.extend(
            [
                "## Final status",
                "",
                f"- Status: **{final['status']}**",
                f"- Blocker: {final['blocker'] or 'none'}",
                "",
            ]
        )
        return "\n".join(lines)

    def write(self, markdown_path: Path, json_path: Path | None = None) -> dict[str, str]:
        """Atomically write the human-readable log and its machine-readable state."""

        markdown_path = markdown_path.resolve()
        json_path = (json_path or markdown_path.with_suffix(".json")).resolve()
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_tmp = markdown_path.with_name(f".{markdown_path.name}.tmp")
        json_tmp = json_path.with_name(f".{json_path.name}.tmp")
        markdown_tmp.write_text(self.render_markdown())
        json_tmp.write_text(json.dumps(self.data, indent=2, ensure_ascii=False) + "\n")
        os.replace(markdown_tmp, markdown_path)
        os.replace(json_tmp, json_path)
        return {
            "markdown": str(markdown_path),
            "markdown_sha256": _sha256(markdown_path),
            "json": str(json_path),
            "json_sha256": _sha256(json_path),
        }
