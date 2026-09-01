"""Non-counting audit for immutable Mapbox reference derivation migrations."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
from pyproj import CRS, Transformer

from .automatic_alignment_loop import (
    AlignmentLoopConfig,
    ProjectionContext,
    SourceSemanticEvidence,
    _normalizer,
    _semantic_full_line_validation,
    load_pinned_mapbox_reference,
)
from .geologic_pdf_alignment import GeologicPdfAlignmentConfig, _state_coast_validation
from .experiment_log import NoHumanExperimentLog
from .restart_registry import NoHumanRestartRegistry


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_non_counting_reference_migration_receipt(
    *,
    map_id: str,
    alignment_path: Path,
    accepted_alignment_reference: Mapping[str, Any],
    current_reference: Mapping[str, Any],
    accepted_automatic_iteration_count: int,
    reference_revisions: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Verify that an immutable old-pin alignment was audited against the current pin.

    This is deliberately stronger than comparing the active experiment pin.  It
    binds the old and new pins, unchanged raw Mapbox bytes, accepted alignment
    bytes and ordinal, and the strict per-map v2 validation result to one hashed
    non-counting audit.  Historical accepted pointers remain immutable.
    """

    alignment_path = alignment_path.resolve()
    alignment_hash = _sha256(alignment_path)
    matches: list[Mapping[str, Any]] = []
    required_authority = {
        "counts_as_alignment_iteration": False,
        "changes_existing_iteration_counts": False,
        "manual_input_used": False,
        "old_transform_reoptimized": False,
        "strict_current_semantic_gates_used": True,
    }
    raw_keys = ("style_sha256", "tilejson_sha256", "tile_aggregate_sha256")
    for revision in reference_revisions:
        if (
            revision.get("producer") != "mapscan.mapbox_reference_migration"
            or revision.get("raw_mapbox_bytes_preserved_exactly") is not True
            or revision.get("automatic_iteration_count_changed") is not False
            or revision.get("source_manifest_sha256")
            != accepted_alignment_reference.get("manifest_sha256")
        ):
            continue
        previous = revision.get("previous_reference", {})
        current = revision.get("current_reference", {})
        if (
            previous.get("manifest_sha256")
            != accepted_alignment_reference.get("manifest_sha256")
            or current.get("manifest_sha256") != current_reference.get("manifest_sha256")
        ):
            continue
        receipt = revision.get("non_counting_audit", {})
        audit_path_value = receipt.get("path")
        audit_hash = receipt.get("sha256")
        if not isinstance(audit_path_value, str) or not isinstance(audit_hash, str):
            continue
        audit_path = Path(audit_path_value).resolve()
        if not audit_path.is_file() or _sha256(audit_path) != audit_hash:
            continue
        try:
            audit = json.loads(audit_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if (
            audit.get("kind") != "non_counting_mapbox_reference_migration_audit"
            or audit.get("status") != "pass"
            or int(audit.get("resume_count", -1)) != 0
            or audit.get("authority") != required_authority
            or audit.get("old_reference") != dict(accepted_alignment_reference)
            or audit.get("new_reference") != dict(current_reference)
        ):
            continue
        old_pin = audit["old_reference"]
        new_pin = audit["new_reference"]
        if any(old_pin.get(key) != new_pin.get(key) for key in raw_keys):
            continue
        revision_raw = revision.get("raw_hashes", {})
        if any(
            revision_raw.get(
                "tiles_sha256" if key == "tile_aggregate_sha256" else key
            )
            != new_pin.get(key)
            for key in raw_keys
        ):
            continue
        records = [
            record
            for record in audit.get("records", [])
            if record.get("map_id") == map_id
        ]
        if len(records) != 1:
            continue
        record = records[0]
        try:
            recorded_alignment_path = Path(
                str(record.get("accepted_alignment_path", ""))
            ).resolve()
        except (OSError, RuntimeError):
            continue
        gates = record.get("new_semantic_gates", {})
        if (
            record.get("status") != "retain_original_acceptance"
            or record.get("counts_as_alignment_iteration") is not False
            or record.get("failed_new_gates") != []
            or record.get("old_reference") != old_pin
            or record.get("new_reference") != new_pin
            or recorded_alignment_path != alignment_path
            or record.get("accepted_alignment_sha256") != alignment_hash
            or record.get("original_accepted_automatic_iteration_count")
            != accepted_automatic_iteration_count
            or not isinstance(gates, Mapping)
            or not gates
            or not all(
                value is True
                or (isinstance(value, Mapping) and value.get("passed") is True)
                for value in gates.values()
            )
        ):
            continue
        matches.append(
            {
                "revision": revision,
                "audit_path": str(audit_path),
                "audit_sha256": audit_hash,
                "record": record,
            }
        )
    if len(matches) != 1:
        raise ValueError(
            "Accepted alignment old Mapbox pin lacks one exact passing non-counting migration receipt"
        )
    return matches[0]


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-mapbox-reference-migration")
    try:
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _mask(path: Path) -> np.ndarray:
    image = np.asarray(Image.open(path))
    if image.ndim == 3:
        image = image[:, :, 3] if image.shape[2] == 4 else np.max(image, axis=2)
    return image > 0


def _serialized_projection(transform: Mapping[str, Any]) -> ProjectionContext:
    payload = transform["projection"]
    crs = CRS.from_wkt(str(payload["crs_wkt"]))
    wkt = crs.to_wkt(version="WKT2_2019", pretty=False)
    expected_wkt_hash = str(payload["crs_wkt_sha256"])
    if hashlib.sha256(wkt.encode()).hexdigest() != expected_wkt_hash:
        # PROJ may normalize equivalent WKT during parsing. The serialized text
        # remains the authority when the normalized form differs by version.
        serialized_wkt = str(payload["crs_wkt"])
        if hashlib.sha256(serialized_wkt.encode()).hexdigest() != expected_wkt_hash:
            raise ValueError("Accepted projection WKT hash is invalid")
        wkt = serialized_wkt
    return ProjectionContext(
        id=str(payload["id"]),
        crs=crs,
        crs_wkt=wkt,
        crs_wkt_sha256=expected_wkt_hash,
        reference_to_candidate=Transformer.from_crs(
            "EPSG:3857", crs, always_xy=True
        ),
        candidate_to_reference=Transformer.from_crs(
            crs, "EPSG:3857", always_xy=True
        ),
        normalization_center=np.asarray(
            payload["normalization_center"], dtype=np.float64
        ),
        normalization_scale=float(payload["normalization_scale"]),
    )


def _accepted_candidate_directory(alignment_root: Path, iteration: int) -> Path:
    matches = []
    for candidate_path in alignment_root.glob(
        f"alignment-{iteration:02d}-*/candidate.json"
    ):
        payload = json.loads(candidate_path.read_text())
        if payload.get("decision") == "accept" and int(payload["iteration"]) == iteration:
            matches.append(candidate_path.parent)
    if len(matches) != 1:
        raise ValueError(
            f"Expected one retained accepted candidate for iteration {iteration}: "
            f"{alignment_root}"
        )
    return matches[0]


def _accepted_transform_reference_normalizer(
    old_reference: Any, new_reference: Any
) -> tuple[np.ndarray, float]:
    """Return the v1 normalizer embedded in an unchanged accepted transform.

    Reference-derived geometry can change during a migration, but the accepted
    matrix must not.  Non-projection fallbacks therefore continue to interpret
    that matrix with the old reference center/height while testing new lines.
    """

    if old_reference.grid != new_reference.grid:
        raise ValueError("Reference migration changed the target grid")
    center, height, _, _ = _normalizer(old_reference.state_coast)
    return center, height


def _semantic_evidence(candidate_root: Path, accepted: Mapping[str, Any]) -> tuple[SourceSemanticEvidence, dict[str, Any]]:
    state_path = candidate_root / "source-state-coast-evidence.png"
    county_path = candidate_root / "source-county-evidence.png"
    if not state_path.is_file() or not county_path.is_file():
        raise ValueError(f"Accepted candidate lacks retained semantic masks: {candidate_root}")
    state = _mask(state_path)
    counties = _mask(county_path)
    county_gate = accepted.get("gates", {}).get(
        "source_county_channel_observability", {}
    )
    county_absent = county_gate.get("required_for_acceptance") is False
    empty = np.zeros_like(state)
    return (
        SourceSemanticEvidence(
            state_coast=state,
            counties=counties,
            dark_cartographic_ink=empty,
            border_connected_water=empty,
            foreground_interior=empty,
            foreground_boundary=state,
            county_observability_override="absent" if county_absent else None,
            source_adapter_id=(
                "retained_accepted_candidate_semantic_channel"
                if county_absent
                else None
            ),
        ),
        {
            "state_coast": {
                "path": str(state_path.resolve()),
                "sha256": _sha256(state_path),
                "pixel_count": int(np.count_nonzero(state)),
            },
            "counties": {
                "path": str(county_path.resolve()),
                "sha256": _sha256(county_path),
                "pixel_count": int(np.count_nonzero(counties)),
            },
        },
    )


def _write_overlay(
    path: Path,
    semantic: SourceSemanticEvidence,
    old_rendered: Mapping[str, np.ndarray],
    new_rendered: Mapping[str, np.ndarray],
) -> None:
    height, width = semantic.state_coast.shape
    image = np.zeros((height, width, 4), dtype=np.uint8)
    image[semantic.state_coast] = (255, 255, 255, 255)
    image[old_rendered["rendered_state"]] = (255, 181, 66, 255)
    image[new_rendered["rendered_state"]] = (61, 214, 255, 255)
    overlap = old_rendered["rendered_state"] & new_rendered["rendered_state"]
    image[overlap] = (155, 255, 115, 255)
    Image.fromarray(image).save(path)


def audit_accepted_transform_reference_migration(
    run_root: Path,
    old_reference_manifest_path: Path,
    new_reference_manifest_path: Path,
    output_root: Path,
    *,
    config: AlignmentLoopConfig | None = None,
) -> dict[str, Any]:
    """Revalidate retained transforms without writing experiment-log attempts.

    Only candidates retaining the aligner's standard semantic evidence artifacts
    are audited here. Specialized PDF/non-rigid adapters must supply their own
    migration validator rather than being approximated as regular transforms.
    """

    config = config or AlignmentLoopConfig()
    run_root = Path(run_root).resolve()
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    old_reference = load_pinned_mapbox_reference(old_reference_manifest_path)
    new_reference = load_pinned_mapbox_reference(new_reference_manifest_path)
    if old_reference.grid != new_reference.grid:
        raise ValueError("Reference migration changed the target grid")
    if (
        old_reference.pin["style_sha256"] != new_reference.pin["style_sha256"]
        or old_reference.pin["tilejson_sha256"]
        != new_reference.pin["tilejson_sha256"]
        or old_reference.pin["tile_aggregate_sha256"]
        != new_reference.pin["tile_aggregate_sha256"]
    ):
        raise ValueError("Reference migration changed pinned raw Mapbox bytes")

    records = []
    skipped_specialized = []
    for accepted_path in sorted(
        run_root.glob("*/automatic-alignment/accepted-alignment.json")
    ):
        map_id = accepted_path.parents[1].name
        accepted = json.loads(accepted_path.read_text())
        iteration = int(accepted["iteration"])
        candidate_root = _accepted_candidate_directory(
            accepted_path.parent, iteration
        )
        state_path = candidate_root / "source-state-coast-evidence.png"
        county_path = candidate_root / "source-county-evidence.png"
        if not state_path.is_file() or not county_path.is_file():
            skipped_specialized.append(
                {
                    "map_id": map_id,
                    "iteration": iteration,
                    "reason": "specialized_adapter_requires_native_migration_validator",
                    "accepted_alignment_path": str(accepted_path.resolve()),
                }
            )
            continue
        semantic, semantic_artifacts = _semantic_evidence(candidate_root, accepted)
        matrix = np.asarray(
            accepted["scores"]["normalized_reference_to_working_source_matrix"],
            dtype=np.float64,
        )
        projection = _serialized_projection(accepted["transform"])
        old_center, old_height = _accepted_transform_reference_normalizer(
            old_reference, new_reference
        )
        old_scores, old_gates, old_rendered = _semantic_full_line_validation(
            matrix,
            old_reference,
            semantic,
            old_center,
            old_height,
            config,
            projection,
        )
        new_scores, new_gates, new_rendered = _semantic_full_line_validation(
            matrix,
            new_reference,
            semantic,
            old_center,
            old_height,
            config,
            projection,
        )
        failed = sorted(
            name for name, gate in new_gates.items() if gate.get("passed") is False
        )
        map_root = output_root / map_id
        map_root.mkdir()
        overlay_path = map_root / "v1-v2-state-line-audit.png"
        _write_overlay(overlay_path, semantic, old_rendered, new_rendered)
        record = {
            "map_id": map_id,
            "original_accepted_automatic_iteration_count": iteration,
            "counts_as_alignment_iteration": False,
            "accepted_alignment_path": str(accepted_path.resolve()),
            "accepted_alignment_sha256": _sha256(accepted_path),
            "candidate_root": str(candidate_root.resolve()),
            "accepted_transform_interpretation": {
                "reference_normalizer_source": "old_reference_v1",
                "reference_center": old_center.tolist(),
                "reference_state_height": float(old_height),
                "matrix_reoptimized": False,
                "matrix_renormalized": False,
            },
            "source_semantic_artifacts": semantic_artifacts,
            "old_reference": dict(old_reference.pin),
            "new_reference": dict(new_reference.pin),
            "old_semantic_full_line": old_scores,
            "new_semantic_full_line": new_scores,
            "old_semantic_gates": old_gates,
            "new_semantic_gates": new_gates,
            "failed_new_gates": failed,
            "status": "retain_original_acceptance" if not failed else "resume_counted_iterations",
            "artifact": {
                "path": str(overlay_path),
                "sha256": _sha256(overlay_path),
                "legend": {
                    "source_state": "white",
                    "v1_reference_state": "orange",
                    "v2_reference_state": "cyan",
                    "v1_v2_overlap": "green",
                },
            },
        }
        record_path = map_root / "reference-migration-audit.json"
        record_path.write_text(json.dumps(record, indent=2) + "\n")
        records.append(record)

    aggregate = {
        "schema_version": 1,
        "kind": "non_counting_mapbox_reference_migration_audit",
        "authority": {
            "counts_as_alignment_iteration": False,
            "changes_existing_iteration_counts": False,
            "manual_input_used": False,
            "old_transform_reoptimized": False,
            "strict_current_semantic_gates_used": True,
        },
        "old_reference": dict(old_reference.pin),
        "new_reference": dict(new_reference.pin),
        "audited_count": len(records),
        "retained_count": sum(
            record["status"] == "retain_original_acceptance" for record in records
        ),
        "resume_count": sum(
            record["status"] == "resume_counted_iterations" for record in records
        ),
        "records": records,
        "skipped_specialized": skipped_specialized,
        "status": (
            "pass"
            if records
            and all(record["status"] == "retain_original_acceptance" for record in records)
            else "requires_counted_retries"
        ),
    }
    aggregate_path = output_root / "reference-migration-audit.json"
    aggregate_path.write_text(json.dumps(aggregate, indent=2) + "\n")
    return aggregate


def audit_geologic_reference_migration(
    run_root: Path,
    old_reference_manifest_path: Path,
    new_reference_manifest_path: Path,
    output_root: Path,
    *,
    config: GeologicPdfAlignmentConfig | None = None,
) -> dict[str, Any]:
    """Append the native geologic adapter's non-counting v2 audit."""

    config = config or GeologicPdfAlignmentConfig()
    run_root = Path(run_root).resolve()
    output_root = Path(output_root).resolve()
    aggregate_path = output_root / "reference-migration-audit.json"
    aggregate = json.loads(aggregate_path.read_text())
    accepted_path = (
        run_root / "geologic/automatic-alignment/accepted-alignment.json"
    )
    accepted = json.loads(accepted_path.read_text())
    old_reference = load_pinned_mapbox_reference(old_reference_manifest_path)
    new_reference = load_pinned_mapbox_reference(new_reference_manifest_path)
    context = _serialized_projection(accepted["transform"])
    original_matrix = np.asarray(
        accepted["transform"]["candidate_normalized_to_source_original_matrix"],
        dtype=np.float64,
    )
    working_raster_path = Path(
        accepted["exact_transform_provenance"]["working_raster"]["path"]
    )
    expected_raster_hash = str(
        accepted["exact_transform_provenance"]["working_raster"]["sha256"]
    )
    if _sha256(working_raster_path) != expected_raster_hash:
        raise ValueError("Geologic source-clean working raster hash changed")
    rgb = np.asarray(Image.open(working_raster_path).convert("RGB"))
    old_scores, old_gates, old_rendered, _, old_scale = _state_coast_validation(
        rgb, old_reference, context, original_matrix, config
    )
    new_scores, new_gates, new_rendered, _, new_scale = _state_coast_validation(
        rgb, new_reference, context, original_matrix, config
    )
    if old_scale != new_scale:
        raise ValueError("Geologic migration changed validation raster scale")
    failed = sorted(
        name for name, gate in new_gates.items() if gate.get("passed") is False
    )
    map_root = output_root / "geologic"
    map_root.mkdir(exist_ok=False)
    overlay_path = map_root / "v1-v2-native-state-line-audit.png"
    height, width = new_rendered["source_edges"].shape
    overlay = np.zeros((height, width, 4), dtype=np.uint8)
    overlay[new_rendered["source_edges"]] = (255, 255, 255, 255)
    overlay[old_rendered["rendered_state"]] = (255, 181, 66, 255)
    overlay[new_rendered["rendered_state"]] = (61, 214, 255, 255)
    overlap = old_rendered["rendered_state"] & new_rendered["rendered_state"]
    overlay[overlap] = (155, 255, 115, 255)
    Image.fromarray(overlay).save(overlay_path)
    record = {
        "map_id": "geologic",
        "adapter": "native_pdf_graticule_affine",
        "original_accepted_automatic_iteration_count": int(accepted["iteration"]),
        "counts_as_alignment_iteration": False,
        "accepted_alignment_path": str(accepted_path),
        "accepted_alignment_sha256": _sha256(accepted_path),
        "working_raster": {
            "path": str(working_raster_path),
            "sha256": expected_raster_hash,
        },
        "old_reference": dict(old_reference.pin),
        "new_reference": dict(new_reference.pin),
        "unchanged_native_graticule_scores": accepted["scores"]["native_graticule"],
        "unchanged_native_graticule_gates": {
            name: value
            for name, value in accepted["gates"].items()
            if name.startswith("native_graticule_")
        },
        "old_semantic_full_line": old_scores,
        "new_semantic_full_line": new_scores,
        "old_semantic_gates": old_gates,
        "new_semantic_gates": new_gates,
        "failed_new_gates": failed,
        "status": "retain_original_acceptance" if not failed else "resume_counted_iterations",
        "artifact": {
            "path": str(overlay_path),
            "sha256": _sha256(overlay_path),
            "legend": {
                "source_edges": "white",
                "v1_reference_state": "orange",
                "v2_reference_state": "cyan",
                "v1_v2_overlap": "green",
            },
        },
    }
    record_path = map_root / "reference-migration-audit.json"
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    aggregate["records"].append(record)
    aggregate["records"] = sorted(
        aggregate["records"], key=lambda item: item["map_id"]
    )
    aggregate["skipped_specialized"] = [
        item
        for item in aggregate.get("skipped_specialized", [])
        if item.get("map_id") != "geologic"
    ]
    aggregate["audited_count"] = len(aggregate["records"])
    aggregate["retained_count"] = sum(
        item["status"] == "retain_original_acceptance"
        for item in aggregate["records"]
    )
    aggregate["resume_count"] = sum(
        item["status"] == "resume_counted_iterations"
        for item in aggregate["records"]
    )
    aggregate["status"] = (
        "pass" if aggregate["resume_count"] == 0 else "requires_counted_retries"
    )
    aggregate_path.write_text(json.dumps(aggregate, indent=2) + "\n")
    return record


def migrate_autonomous_restart_reference_after_audit(
    restart_manifest_path: Path,
    run_root: Path,
    new_reference_manifest_path: Path,
    audit_path: Path,
) -> dict[str, Any]:
    """Move the active run pin to v2 after a passing non-counting audit.

    Historical attempt payloads remain byte-for-byte untouched.  Only the
    active manifest/snapshot and each log's top-level current reference are
    advanced; every log receives the same non-counting migration record.
    """

    restart_manifest_path = Path(restart_manifest_path).resolve()
    run_root = Path(run_root).resolve()
    new_reference_manifest_path = Path(new_reference_manifest_path).resolve()
    audit_path = Path(audit_path).resolve()
    audit = json.loads(audit_path.read_text())
    if audit.get("status") != "pass" or int(audit.get("resume_count", -1)) != 0:
        raise ValueError("Reference pin migration requires a passing zero-retry audit")
    manifest = json.loads(restart_manifest_path.read_text())
    old_reference = dict(manifest["mapbox_reference"])
    if old_reference.get("manifest_sha256") != audit["old_reference"][
        "manifest_sha256"
    ]:
        raise ValueError("Restart manifest does not match the audited old reference")
    new_manifest = json.loads(new_reference_manifest_path.read_text())
    new_manifest_hash = _sha256(new_reference_manifest_path)
    if new_manifest_hash != audit["new_reference"]["manifest_sha256"]:
        raise ValueError("New reference does not match the audited manifest")
    artifacts = new_manifest["artifacts"]
    new_reference = {
        **old_reference,
        "id": str(new_manifest.get("reference_id", new_reference_manifest_path.parent.name)),
        "root": str(new_reference_manifest_path.parent.relative_to(restart_manifest_path.parent.parent)),
        "manifest_sha256": new_manifest_hash,
        "style_sha256": str(new_manifest["style"]["sha256"]),
        "tilejson_sha256": str(new_manifest["tileset"]["tilejson_sha256"]),
        "tiles_sha256": str(new_manifest["tile_aggregate_sha256"]),
        "state_sha256": str(artifacts["state_land_mask"]["sha256"]),
        "state_coast_sha256": str(artifacts["state_coast_overlay"]["sha256"]),
        "county_sha256": str(artifacts["county_overlay"]["sha256"]),
        "water_sha256": str(artifacts["water_mask"]["sha256"]),
    }

    snapshot_path = run_root / "restart-manifest.snapshot.json"
    snapshot = json.loads(snapshot_path.read_text())
    if snapshot.get("mapbox_reference") != old_reference:
        raise ValueError("Run snapshot does not match the audited old reference")
    backup_path = run_root / "restart-manifest.snapshot.mapbox-v1.json"
    if backup_path.exists():
        if backup_path.read_bytes() != snapshot_path.read_bytes():
            raise ValueError("Existing v1 snapshot backup does not match")
    else:
        shutil.copy2(snapshot_path, backup_path)

    recorded_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    migration_record = {
        "kind": "non_counting_pinned_mapbox_reference_migration",
        "recorded_at": recorded_at,
        "counts_as_alignment_iteration": False,
        "changes_existing_iteration_counts": False,
        "old_reference": old_reference,
        "new_reference": new_reference,
        "audit": {"path": str(audit_path), "sha256": _sha256(audit_path)},
        "reason": "Pacific-side detached-island topology correction; raw Mapbox bytes unchanged",
    }

    # Prepare log payloads before changing the central manifest/snapshot.
    logs: list[tuple[NoHumanExperimentLog, Path, Path]] = []
    already_migrated_logs: list[str] = []
    for map_record in manifest["maps"]:
        map_id = str(map_record["id"])
        json_path = run_root / map_id / "EXPERIMENT.json"
        markdown_path = run_root / map_id / "EXPERIMENT.md"
        log = NoHumanExperimentLog.load(json_path)
        if log.data.get("mapbox_reference") == new_reference:
            already_migrated_logs.append(map_id)
            continue
        if log.data.get("mapbox_reference") != old_reference:
            raise ValueError(f"Experiment log {map_id} is not pinned to audited v1")
        log.data["mapbox_reference"] = dict(new_reference)
        revisions = log.data.setdefault("mapbox_reference_revisions", [])
        revisions.append(
            {
                "revision": len(revisions) + 1,
                "recorded_at": recorded_at,
                "producer": "mapscan.mapbox_reference_migration",
                "reason": migration_record["reason"],
                "previous_reference": old_reference,
                "current_reference": new_reference,
                "raw_mapbox_bytes_preserved_exactly": True,
                "raw_hashes": {
                    "style_sha256": new_reference["style_sha256"],
                    "tilejson_sha256": new_reference["tilejson_sha256"],
                    "tiles_sha256": new_reference["tiles_sha256"],
                },
                "source_manifest_sha256": old_reference["manifest_sha256"],
                "automatic_iteration_count_changed": False,
                "non_counting_audit": migration_record["audit"],
            }
        )
        logs.append((log, markdown_path, json_path))

    manifest["mapbox_reference"] = dict(new_reference)
    snapshot["mapbox_reference"] = dict(new_reference)
    _atomic_json(restart_manifest_path, manifest)
    _atomic_json(snapshot_path, snapshot)
    for log, markdown_path, json_path in logs:
        log.write(markdown_path, json_path)

    registry = NoHumanRestartRegistry(restart_manifest_path, run_root)
    registry.initialize()
    result = {
        **migration_record,
        "status": "complete",
        "restart_manifest": {
            "path": str(restart_manifest_path),
            "sha256": _sha256(restart_manifest_path),
        },
        "active_snapshot": {
            "path": str(snapshot_path),
            "sha256": _sha256(snapshot_path),
        },
        "preserved_v1_snapshot": {
            "path": str(backup_path),
            "sha256": _sha256(backup_path),
        },
        "migrated_log_count": len(logs),
        "already_migrated_log_ids": already_migrated_logs,
        "index": str(registry.refresh_index()),
    }
    result_path = audit_path.parent / "official-reference-pin-migration.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    return result
