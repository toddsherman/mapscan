"""Manifest-driven batch orchestration for the no-human MapScan restart.

This runner is intentionally a thin coordinator.  It gives the isolated
alignment and extraction loops only a source-clean working raster and the
pinned Mapbox manifest.  It never searches historical run directories or
accepts manual controls, painted pixels, prior alignments, or legacy reference
artifacts.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .experiment_log import NoHumanExperimentLog
from .restart_registry import NoHumanRestartRegistry
from .source_working_raster import (
    MANIFEST_NAME as SOURCE_ADAPTER_MANIFEST,
    WorkingRasterArtifact,
    load_working_raster_artifact,
    prepare_source_working_raster,
)


SCHEMA_VERSION = "mapscan.autonomous-restart-batch.v1"
SUPPORTED_CATEGORICAL_SOURCE_TYPES = frozenset(
    {
        "categorical_full_state",
        "categorical_sparse",
        "dithered_categorical_full_state",
    }
)
FORBIDDEN_EVIDENCE_TOKENS = {
    "approval",
    "arrow",
    "brush",
    "canonical_boundary",
    "control_point",
    "county.png",
    "farms.png",
    "human",
    "manual",
    "paint",
    "stamp",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_without_overwrite(path: Path, content: bytes) -> None:
    if path.exists():
        if path.is_file() and path.read_bytes() == content:
            return
        raise FileExistsError(f"refusing to overwrite a different artifact: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_bytes(content)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _contains_forbidden_evidence(value: Any, path: str = "") -> str | None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            current = f"{path}.{key}" if path else str(key)
            normalized = str(key).lower().replace("-", "_").replace(" ", "_")
            if any(token.replace(".", "_") in normalized for token in FORBIDDEN_EVIDENCE_TOKENS):
                return current
            found = _contains_forbidden_evidence(child, current)
            if found:
                return found
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            found = _contains_forbidden_evidence(child, f"{path}[{index}]")
            if found:
                return found
    elif isinstance(value, str):
        normalized = value.lower().replace("-", "_").replace(" ", "_")
        if any(token.replace(".", "_") in normalized for token in FORBIDDEN_EVIDENCE_TOKENS):
            return path or "value"
    return None


def _validate_log_is_automatic(log: NoHumanExperimentLog) -> None:
    for phase in ("alignment", "extraction"):
        for item in log.data[phase]["iterations"]:
            if not item.get("counts_toward_automatic_iteration_count", False):
                raise ValueError(
                    f"{phase} attempt {item.get('attempt')} contains ineligible or manual evidence"
                )
            provenance = item.get("provenance", {})
            if (
                provenance.get("actor_kind") != "automated"
                or provenance.get("manual_arrows") is not False
                or provenance.get("manual_stamps") is not False
                or provenance.get("human_approval") is not False
            ):
                raise ValueError(
                    f"{phase} attempt {item.get('attempt')} is not fully automatic"
                )
            forbidden = _contains_forbidden_evidence(provenance.get("input_kinds", []))
            if forbidden:
                raise ValueError(
                    f"{phase} attempt {item.get('attempt')} references forbidden evidence at {forbidden}"
                )


def _resolve_mapbox_manifest(registry: NoHumanRestartRegistry) -> Path:
    reference = registry.reference
    configured = reference.get("manifest_path") or reference.get("root")
    if not isinstance(configured, str) or not configured.strip():
        raise ValueError("pinned Mapbox reference requires root or manifest_path")
    raw = Path(configured)
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend(
            [
                registry.manifest_path.parent / raw,
                Path.cwd() / raw,
                registry.manifest_path.parent.parent / raw,
            ]
        )
    resolved: list[Path] = []
    for candidate in candidates:
        path = candidate / "manifest.json" if candidate.is_dir() else candidate
        path = path.resolve()
        if path.is_file() and path not in resolved:
            resolved.append(path)
    if not resolved:
        raise FileNotFoundError(f"pinned Mapbox manifest not found for {configured!r}")
    expected = reference.get("manifest_sha256")
    matches = [path for path in resolved if _sha256(path) == expected]
    if len(matches) != 1:
        raise ValueError(
            "pinned Mapbox manifest path is ambiguous or does not match manifest_sha256"
        )
    return matches[0]


@dataclass(frozen=True)
class AutonomousRestartBatchConfig:
    """Pinned runner policy; no interactive or legacy inputs are representable."""

    pdf_page_number: int = 1
    pdf_dpi: int = 150
    inspect_pdf_vectors: bool = True
    categorical_source_types: frozenset[str] = SUPPORTED_CATEGORICAL_SOURCE_TYPES
    alignment_config: Any = None
    geologic_alignment_config: Any = None
    extraction_config: Any = None
    retry_blocked_map_ids: frozenset[str] = frozenset()
    retry_failed_map_ids: frozenset[str] = frozenset()
    retry_reason: str = "automatic algorithm capability update"
    retry_producer: str = "mapscan-autonomous-restart-batch-v1"

    def __post_init__(self) -> None:
        if self.pdf_page_number < 1:
            raise ValueError("pdf_page_number must be positive")
        if self.pdf_dpi < 72 or self.pdf_dpi > 1200:
            raise ValueError("pdf_dpi must be between 72 and 1200")
        if not self.categorical_source_types:
            raise ValueError("categorical_source_types cannot be empty")
        retry_ids = self.retry_blocked_map_ids | self.retry_failed_map_ids
        if retry_ids and not self.retry_reason.strip():
            raise ValueError("retry_reason is required when retrying terminal maps")
        if retry_ids and not self.retry_producer.strip():
            raise ValueError("retry_producer is required when retrying terminal maps")
        overlap = self.retry_blocked_map_ids & self.retry_failed_map_ids
        if overlap:
            raise ValueError(
                f"map ids cannot be both blocked and failed retries: {sorted(overlap)}"
            )


@dataclass(frozen=True)
class MapBatchOutcome:
    map_id: str
    status: str
    phase: str
    message: str
    alignment_automatic_iterations: int
    extraction_automatic_iterations: int


@dataclass(frozen=True)
class AutonomousRestartBatchResult:
    run_root: Path
    index_path: Path
    mapbox_manifest_path: Path
    outcomes: tuple[MapBatchOutcome, ...]


class _CheckpointingStageLog:
    """Bridge a working-raster stage log into the authoritative source log."""

    def __init__(
        self,
        stage: NoHumanExperimentLog,
        authoritative: NoHumanExperimentLog,
        checkpoint: Callable[[], None],
        source_artifacts: Sequence[Mapping[str, Any]],
    ) -> None:
        self.stage = stage
        self.authoritative = authoritative
        self.data = stage.data
        self._checkpoint = checkpoint
        self._source_artifacts = [dict(item) for item in source_artifacts]

    def _with_source_provenance(self, kwargs: Mapping[str, Any]) -> dict[str, Any]:
        copied = dict(kwargs)
        provenance = dict(copied.get("provenance", {}))
        inputs = [str(value) for value in provenance.get("input_kinds", [])]
        for value in (
            "authoritative_original_source",
            "source_clean_working_raster",
        ):
            if value not in inputs:
                inputs.append(value)
        provenance["input_kinds"] = inputs
        copied["provenance"] = provenance
        copied["artifacts"] = [
            *[dict(item) for item in copied.get("artifacts", [])],
            *self._source_artifacts,
        ]
        return copied

    def record_alignment_iteration(self, **kwargs: Any) -> dict[str, Any]:
        copied = self._with_source_provenance(kwargs)
        stage_item = self.stage.record_alignment_iteration(**copied)
        self.authoritative.record_alignment_iteration(**copied)
        self._checkpoint()
        return stage_item

    def record_extraction_iteration(self, **kwargs: Any) -> dict[str, Any]:
        copied = self._with_source_provenance(kwargs)
        stage_item = self.stage.record_extraction_iteration(**copied)
        self.authoritative.record_extraction_iteration(**copied)
        self._checkpoint()
        return stage_item

    def finalize(self, status: str, blocker: str | None = None) -> None:
        self.stage.finalize(status, blocker)
        self.authoritative.finalize(status, blocker)
        self._checkpoint()

    def write(self, *_args: Any, **_kwargs: Any) -> dict[str, str]:
        self._checkpoint()
        return {}


class _CheckpointingCanonicalLog:
    """Checkpoint a source-specific extractor against the original-source log.

    Generic raster extraction uses a temporary log whose source is the decoded
    working raster.  Native-PDF and overlapping-layer extractors deliberately
    validate the original source hash instead, so they need a thin wrapper over
    the canonical log rather than a working-raster stage log.
    """

    def __init__(
        self,
        canonical: NoHumanExperimentLog,
        checkpoint: Callable[[], None],
        source_artifacts: Sequence[Mapping[str, Any]],
    ) -> None:
        self.canonical = canonical
        self.data = canonical.data
        self._checkpoint = checkpoint
        self._source_artifacts = [dict(item) for item in source_artifacts]

    def record_extraction_iteration(self, **kwargs: Any) -> dict[str, Any]:
        copied = dict(kwargs)
        provenance = dict(copied.get("provenance", {}))
        inputs = [str(value) for value in provenance.get("input_kinds", [])]
        for value in ("authoritative_original_source", "source_clean_working_raster"):
            if value not in inputs:
                inputs.append(value)
        provenance["input_kinds"] = inputs
        copied["provenance"] = provenance
        copied["artifacts"] = [
            *[dict(item) for item in copied.get("artifacts", [])],
            *self._source_artifacts,
        ]
        item = self.canonical.record_extraction_iteration(**copied)
        self._checkpoint()
        return item

    def finalize(self, status: str, blocker: str | None = None) -> None:
        self.canonical.finalize(status, blocker)
        self._checkpoint()

    def write(self, *_args: Any, **_kwargs: Any) -> dict[str, str]:
        self._checkpoint()
        return {}


class AutonomousRestartBatchRunner:
    """Run or safely resume every map declared by a no-human restart manifest."""

    def __init__(
        self,
        manifest_path: Path,
        run_root: Path,
        *,
        config: AutonomousRestartBatchConfig = AutonomousRestartBatchConfig(),
        source_adapter: Callable[..., WorkingRasterArtifact] = prepare_source_working_raster,
        alignment_runner: Callable[..., Any] | None = None,
        geologic_alignment_runner: Callable[..., Any] | None = None,
        extraction_runner: Callable[..., Any] | None = None,
        specialized_extraction_runners: Mapping[str, Callable[..., Any]] | None = None,
    ) -> None:
        self.manifest_path = manifest_path.resolve()
        self.run_root = run_root.resolve()
        self.config = config
        self.source_adapter = source_adapter
        if alignment_runner is None:
            from .automatic_alignment_loop import run_automatic_alignment_loop

            alignment_runner = run_automatic_alignment_loop
        if extraction_runner is None:
            from .automatic_categorical_extraction import (
                run_automatic_categorical_extraction,
            )

            extraction_runner = run_automatic_categorical_extraction
        if geologic_alignment_runner is None:
            from .geologic_pdf_alignment import run_geologic_pdf_alignment

            geologic_alignment_runner = run_geologic_pdf_alignment
        self.alignment_runner = alignment_runner
        self.geologic_alignment_runner = geologic_alignment_runner
        self.extraction_runner = extraction_runner
        if specialized_extraction_runners is None:
            from .automatic_continuous_numeric_extraction import (
                run_automatic_continuous_numeric_extraction,
            )
            from .automatic_ordered_band_extraction import (
                run_automatic_ordered_band_extraction,
            )
            from .fire_layer_extraction import run_fire_overlapping_layer_extraction
            from .geologic_pdf_extraction import run_geologic_pdf_vector_extraction
            from .river_source_extraction import run_named_hydrography_extraction
            from .storm_layer_extraction import run_storm_overlapping_layer_extraction

            specialized_extraction_runners = {
                "continuous_numeric_ramp": (
                    run_automatic_continuous_numeric_extraction
                ),
                "ordered_gradient_bands": run_automatic_ordered_band_extraction,
                "native_pdf_vector_categorical": run_geologic_pdf_vector_extraction,
                "named_linear_and_polygon_features_without_legend": (
                    run_named_hydrography_extraction
                ),
                "overlapping_feature_and_categorical": (
                    run_fire_overlapping_layer_extraction
                ),
                "overlapping_chromatic_and_grayscale": (
                    run_storm_overlapping_layer_extraction
                ),
            }
        self.specialized_extraction_runners = dict(specialized_extraction_runners)
        self.registry: NoHumanRestartRegistry | None = None
        self.mapbox_manifest_path: Path | None = None

    def _checkpoint(
        self,
        log: NoHumanExperimentLog,
        markdown_path: Path,
        json_path: Path,
    ) -> None:
        if self.registry is None:
            raise RuntimeError("batch registry is not initialized")
        log.write(markdown_path, json_path)
        self.registry.refresh_index()

    def _working_raster(
        self,
        record: Mapping[str, Any],
        map_dir: Path,
    ) -> WorkingRasterArtifact:
        source_path = Path(str(record["source"])).resolve()
        adapter_dir = map_dir / "source-clean"
        manifest_path = adapter_dir / SOURCE_ADAPTER_MANIFEST
        if manifest_path.exists():
            artifact = load_working_raster_artifact(manifest_path)
        else:
            arguments: dict[str, Any] = {}
            if source_path.suffix.lower() == ".pdf":
                arguments = {
                    "pdf_page_number": self.config.pdf_page_number,
                    "pdf_dpi": self.config.pdf_dpi,
                    "inspect_pdf_vectors": self.config.inspect_pdf_vectors,
                }
            artifact = self.source_adapter(source_path, adapter_dir, **arguments)
        if artifact.source_path != source_path:
            raise ValueError("source-clean adapter targets a different original source")
        if artifact.source_sha256 != record["source_sha256"]:
            raise ValueError("source-clean adapter original hash differs from restart manifest")
        return artifact

    @staticmethod
    def _source_artifacts(artifact: WorkingRasterArtifact) -> list[dict[str, Any]]:
        return [
            {
                "path": str(artifact.manifest_path),
                "sha256": _sha256(artifact.manifest_path),
                "kind": "source_clean_adapter_manifest",
            },
            {
                "path": str(artifact.working_raster_path),
                "sha256": artifact.working_raster_sha256,
                "decoded_rgb_sha256": artifact.decoded_rgb_sha256,
                "kind": "source_clean_working_raster",
            },
        ]

    def _stage_log(
        self,
        canonical: NoHumanExperimentLog,
        working: WorkingRasterArtifact,
        *,
        phase: str,
        checkpoint: Callable[[], None],
    ) -> _CheckpointingStageLog:
        if self.registry is None:
            raise RuntimeError("batch registry is not initialized")
        stage = NoHumanExperimentLog(
            str(canonical.data["map_id"]),
            working.working_raster_path,
            mapbox_reference=self.registry.reference,
        )
        stage.data["alignment"] = copy.deepcopy(canonical.data["alignment"])
        if phase == "alignment":
            stage.data["extraction"] = copy.deepcopy(canonical.data["extraction"])
        elif phase == "extraction":
            # A crashed extraction restarts in a new immutable run directory;
            # canonical counts continue through the bridge without rewriting old attempts.
            stage.data["extraction"] = {
                "iterations": [],
                "accepted_automatic_iteration_count": None,
            }
        else:
            raise ValueError("phase must be alignment or extraction")
        stage.data["final"] = {"status": "in_progress", "blocker": None}
        return _CheckpointingStageLog(
            stage,
            canonical,
            checkpoint,
            self._source_artifacts(working),
        )

    @staticmethod
    def _validate_accepted_alignment(path: Path, working: WorkingRasterArtifact) -> None:
        payload = json.loads(path.read_text())
        if payload.get("decision") != "accept":
            raise ValueError("accepted-alignment artifact does not contain an acceptance")
        if payload.get("source_sha256") not in {
            working.source_sha256,
            working.working_raster_sha256,
        }:
            raise ValueError("accepted alignment targets a different source")
        forbidden = _contains_forbidden_evidence(payload)
        if forbidden:
            raise ValueError(f"accepted alignment contains forbidden evidence at {forbidden}")

    def _accepted_alignment(
        self,
        log: NoHumanExperimentLog,
        alignment_root: Path,
        working: WorkingRasterArtifact,
    ) -> Path:
        accepted = alignment_root / "accepted-alignment.json"
        ordinal = log.data["alignment"]["accepted_automatic_iteration_count"]
        if ordinal is None:
            raise ValueError("experiment log has no accepted automatic alignment")
        if not accepted.exists():
            candidates = sorted(alignment_root.glob(f"alignment-{ordinal:02d}-*/candidate.json"))
            valid = []
            for candidate in candidates:
                try:
                    self._validate_accepted_alignment(candidate, working)
                except (ValueError, json.JSONDecodeError):
                    continue
                valid.append(candidate)
            if len(valid) != 1:
                raise ValueError("cannot recover a unique accepted alignment from current-run candidates")
            _write_without_overwrite(accepted, valid[0].read_bytes())
        self._validate_accepted_alignment(accepted, working)
        return accepted

    @staticmethod
    def _accepted_extraction(
        map_dir: Path,
        working: WorkingRasterArtifact,
        alignment_path: Path,
    ) -> Path:
        candidates = sorted(
            {
                *map_dir.glob("extraction-run-*/accepted-extraction.json"),
                *map_dir.glob("automatic-extraction*/accepted-extraction.json"),
            }
        )
        valid: list[Path] = []
        for path in candidates:
            try:
                payload = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            source = payload.get("source", {})
            alignment = payload.get("alignment", {})
            if (
                payload.get("status") != "accepted"
                or source.get("sha256")
                not in {working.source_sha256, working.working_raster_sha256}
                or alignment.get("sha256") != _sha256(alignment_path)
                or _contains_forbidden_evidence(payload)
            ):
                continue
            valid.append(path)
        if len(valid) != 1:
            raise ValueError(
                "accepted extraction log does not have one matching current-run manifest"
            )
        return valid[0]

    @staticmethod
    def _counts(log: NoHumanExperimentLog) -> tuple[int, int]:
        return tuple(
            sum(
                bool(item.get("counts_toward_automatic_iteration_count"))
                for item in log.data[phase]["iterations"]
            )
            for phase in ("alignment", "extraction")
        )  # type: ignore[return-value]

    def _outcome(
        self,
        record: Mapping[str, Any],
        log: NoHumanExperimentLog,
        phase: str,
        message: str,
    ) -> MapBatchOutcome:
        alignment_count, extraction_count = self._counts(log)
        return MapBatchOutcome(
            str(record["id"]),
            str(log.data["final"]["status"]),
            phase,
            message,
            alignment_count,
            extraction_count,
        )

    def _run_map(self, record: Mapping[str, Any]) -> MapBatchOutcome:
        if self.registry is None or self.mapbox_manifest_path is None:
            raise RuntimeError("batch runner is not initialized")
        map_id = str(record["id"])
        map_dir = self.run_root / map_id
        markdown_path = map_dir / "EXPERIMENT.md"
        json_path = map_dir / "EXPERIMENT.json"
        log = self.registry.logs[map_id]

        def checkpoint() -> None:
            self._checkpoint(log, markdown_path, json_path)

        try:
            _validate_log_is_automatic(log)
            working = self._working_raster(record, map_dir)
            terminal = str(log.data["final"]["status"])
            if terminal == "blocked" and map_id in self.config.retry_blocked_map_ids:
                log.resume_automatic_blocked(
                    reason=self.config.retry_reason,
                    producer=self.config.retry_producer,
                )
                checkpoint()
                terminal = "in_progress"
            elif terminal == "failed" and map_id in self.config.retry_failed_map_ids:
                log.resume_automatic_failed(
                    reason=self.config.retry_reason,
                    producer=self.config.retry_producer,
                )
                checkpoint()
                terminal = "in_progress"
            if terminal in {"complete", "blocked", "failed"}:
                if log.data["alignment"]["accepted_automatic_iteration_count"] is not None:
                    alignment_path = self._accepted_alignment(
                        log, map_dir / "automatic-alignment", working
                    )
                    if terminal == "complete":
                        self._accepted_extraction(map_dir, working, alignment_path)
                return self._outcome(record, log, "terminal", "safely resumed terminal map")

            alignment_root = map_dir / "automatic-alignment"
            if log.data["alignment"]["accepted_automatic_iteration_count"] is None:
                existing_pointer = alignment_root / "accepted-alignment.json"
                if existing_pointer.exists():
                    existing_payload = json.loads(existing_pointer.read_text())
                    if existing_payload.get("decision") == "accept":
                        raise ValueError(
                            "unlogged accepted alignment exists; refusing to consume "
                            "ambiguous current-run state"
                        )
                stage_log = self._stage_log(
                    log, working, phase="alignment", checkpoint=checkpoint
                )
                source_type = str(record.get("source_type") or "unspecified")
                is_native_pdf = source_type == "native_pdf_vector_categorical"
                alignment_runner = (
                    self.geologic_alignment_runner
                    if is_native_pdf
                    else self.alignment_runner
                )
                alignment_input = (
                    working.manifest_path
                    if is_native_pdf
                    else working.alignment_input_path
                )
                selected_config = (
                    self.config.geologic_alignment_config
                    if is_native_pdf
                    else self.config.alignment_config
                )
                arguments: dict[str, Any] = {}
                if selected_config is not None:
                    arguments["config"] = selected_config
                if not is_native_pdf:
                    arguments["source_family"] = source_type
                result = alignment_runner(
                    alignment_input,
                    self.mapbox_manifest_path,
                    alignment_root,
                    stage_log,
                    **arguments,
                )
                if result.accepted is None or result.status != "pass":
                    blocker = f"Automatic alignment blocked: {result.stop_reason}"
                    log.finalize("blocked", blocker)
                    checkpoint()
                    return self._outcome(record, log, "alignment", blocker)

            alignment_path = self._accepted_alignment(log, alignment_root, working)
            source_type = str(record.get("source_type") or "unspecified")
            specialized_runner = self.specialized_extraction_runners.get(source_type)
            if (
                source_type not in self.config.categorical_source_types
                and specialized_runner is None
            ):
                blocker = (
                    f"Unsupported autonomous extraction data model: {source_type}; "
                    "alignment is retained but categorical extraction was not run"
                )
                log.finalize("blocked", blocker)
                checkpoint()
                return self._outcome(record, log, "extraction", blocker)

            if log.data["extraction"]["accepted_automatic_iteration_count"] is not None:
                self._accepted_extraction(map_dir, working, alignment_path)
                if log.data["final"]["status"] != "complete":
                    log.finalize("complete")
                    checkpoint()
                return self._outcome(
                    record, log, "extraction", "safely resumed accepted extraction"
                )

            prior_runs = sorted(
                {
                    *map_dir.glob("extraction-run-*"),
                    *map_dir.glob("automatic-extraction*"),
                }
            )
            extraction_root = map_dir / f"extraction-run-{len(prior_runs) + 1:02d}"
            if specialized_runner is None:
                stage_log: Any = self._stage_log(
                    log, working, phase="extraction", checkpoint=checkpoint
                )
                arguments = {}
                if self.config.extraction_config is not None:
                    arguments["config"] = self.config.extraction_config
                extraction_input = working.alignment_input_path
                selected_runner = self.extraction_runner
            else:
                stage_log = _CheckpointingCanonicalLog(
                    log, checkpoint, self._source_artifacts(working)
                )
                arguments = {}
                extraction_input = working.manifest_path
                selected_runner = specialized_runner
            result = selected_runner(
                extraction_input,
                alignment_path,
                self.mapbox_manifest_path,
                extraction_root,
                stage_log,
                markdown_path,
                json_path,
                **arguments,
            )
            accepted_ordinal = log.data["extraction"][
                "accepted_automatic_iteration_count"
            ]
            final_status = log.data["final"]["status"]
            if accepted_ordinal is None and final_status == "in_progress":
                log.finalize(
                    "blocked",
                    f"Automatic categorical extraction blocked: {result.stop_reason}",
                )
            elif accepted_ordinal is not None:
                if final_status != "complete":
                    raise ValueError(
                        "extraction accepted an iteration without completing the experiment log"
                    )
                self._accepted_extraction(map_dir, working, alignment_path)
            checkpoint()
            return self._outcome(record, log, "extraction", str(result.stop_reason))
        except Exception as error:
            blocker = f"{type(error).__name__}: {error}"
            log.finalize("failed", blocker)
            checkpoint()
            return self._outcome(record, log, "failed", blocker)

    def run(self, map_ids: Sequence[str] | None = None) -> AutonomousRestartBatchResult:
        self.registry = NoHumanRestartRegistry(self.manifest_path, self.run_root)
        self.registry.initialize()
        self.mapbox_manifest_path = _resolve_mapbox_manifest(self.registry)
        selected = set(map_ids) if map_ids is not None else None
        known = {str(record["id"]) for record in self.registry.maps}
        unknown_retries = (
            set(self.config.retry_blocked_map_ids)
            | set(self.config.retry_failed_map_ids)
        ) - known
        if unknown_retries:
            raise ValueError(f"unknown blocked retry map ids: {sorted(unknown_retries)}")
        if selected is not None:
            unknown = selected - known
            if unknown:
                raise ValueError(f"unknown restart map ids: {sorted(unknown)}")
        outcomes = []
        for record in self.registry.maps:
            if selected is None or record["id"] in selected:
                outcomes.append(self._run_map(record))
        index = self.registry.refresh_index()
        return AutonomousRestartBatchResult(
            self.run_root,
            index,
            self.mapbox_manifest_path,
            tuple(outcomes),
        )
