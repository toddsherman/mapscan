"""Provenance-safe migration of reviewed categorical corrections between alignments."""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Sequence

import cv2
import numpy as np
from PIL import Image

from .extraction import warp_classified_to_web_mercator
from .manual_stamp import apply_clone_stamp_operations
from .reference import load_california


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text())


def _alignment_transform(alignment: Dict[str, object]) -> Dict[str, object]:
    if alignment.get("alignment_mode") == "assisted":
        transform: Dict[str, object] = {
            "projection": "assisted_reference_crs",
            "projection_crs": alignment["reference"]["crs"],
            "transform_model": alignment["transform_model"],
            "reference_to_source_matrix": alignment["reference_to_source_matrix"],
        }
    else:
        transform = dict(alignment["best"])
    if "web_mercator_correction" in alignment:
        transform["web_mercator_correction"] = alignment[
            "web_mercator_correction"
        ]
    return transform


def _grid_from_manifest(manifest: Dict[str, object]) -> Dict[str, object]:
    layers = manifest.get("layers", [])
    if not isinstance(layers, list) or not layers:
        raise ValueError("Extraction manifest has no layers")
    grid = layers[0].get("warp")
    if not isinstance(grid, dict):
        raise ValueError("Extraction manifest layer has no warp grid")
    return grid


def _same_grid(first: Dict[str, object], second: Dict[str, object]) -> bool:
    return (
        str(first.get("crs")) == str(second.get("crs"))
        and int(first.get("width", -1)) == int(second.get("width", -2))
        and int(first.get("height", -1)) == int(second.get("height", -2))
        and np.allclose(
            np.asarray(first.get("bounds", []), dtype=float),
            np.asarray(second.get("bounds", []), dtype=float),
            rtol=0,
            atol=1e-6,
        )
    )


def _project_point(point: Sequence[float], matrix: np.ndarray) -> list[float]:
    projected = cv2.perspectiveTransform(
        np.asarray(point, dtype=np.float64).reshape((1, 1, 2)), matrix
    ).reshape(2)
    return [round(float(projected[0]), 3), round(float(projected[1]), 3)]


def _smoothstep(value: float) -> float:
    clipped = min(1.0, max(0.0, value))
    return clipped * clipped * (3.0 - 2.0 * clipped)


def _invert_lower_colorado_sampling(
    point: Sequence[float], operation: Dict[str, object]
) -> list[float]:
    """Map a parent-grid feature into the child grid of a smoothstep correction."""

    parent_x, parent_y = (float(value) for value in point)
    start_x = float(operation["start_x_px"])
    ramp_width = float(operation["ramp_width_px"])
    start_y = float(operation["start_y_px"])
    ramp_height = float(operation["ramp_height_px"])
    sampling_amplitude = float(
        operation["target_to_parent_sampling_amplitude_x_px"]
    )
    y_weight = _smoothstep((parent_y - start_y) / ramp_height)
    target_x = parent_x
    # The forward sampling rule is parent_x = target_x + amplitude * weight.
    # Its derivative is tightly bounded by the refinement gate, so fixed-point
    # inversion converges rapidly without approximating the bump as a matrix.
    for _ in range(32):
        x_weight = _smoothstep((target_x - start_x) / ramp_width)
        next_x = parent_x - sampling_amplitude * x_weight * y_weight
        if abs(next_x - target_x) <= 1e-9:
            target_x = next_x
            break
        target_x = next_x
    return [round(target_x, 3), round(parent_y, 3)]


def _alignment_chain(
    source_alignment_path: Path,
    target_alignment_path: Path,
) -> list[tuple[Path, Dict[str, object]]]:
    """Return verified child alignments ordered from source to target."""

    source_hash = _sha256(source_alignment_path)
    current_path = target_alignment_path
    reverse_chain: list[tuple[Path, Dict[str, object]]] = []
    visited = set()
    while _sha256(current_path) != source_hash:
        current_hash = _sha256(current_path)
        if current_hash in visited:
            raise ValueError("Alignment parent chain contains a cycle")
        visited.add(current_hash)
        current = _load(current_path)
        parent = current.get("parent_alignment")
        if not isinstance(parent, dict):
            raise ValueError(
                "Target alignment is not a hash-linked descendant of the reviewed "
                "source alignment"
            )
        reverse_chain.append((current_path, current))
        if parent.get("sha256") == source_hash:
            current_path = source_alignment_path
            continue
        parent_path = Path(str(parent.get("path", ""))).resolve()
        if not parent_path.is_file() or _sha256(parent_path) != parent.get("sha256"):
            raise ValueError(
                "Alignment parent path or hash is stale; target is not a "
                "hash-linked child or descendant"
            )
        current_path = parent_path
    return list(reversed(reverse_chain))


def _map_parent_point_to_child(
    point: Sequence[float],
    child: Dict[str, object],
) -> tuple[list[float], Dict[str, object]]:
    local = child.get("automatic_lower_colorado_refinement")
    if isinstance(local, dict) and isinstance(local.get("operation"), dict):
        operation = local["operation"]
        if operation.get("type") != "lower_colorado_smoothstep_x":
            raise ValueError("Unsupported automatic local correction for stamp migration")
        return _invert_lower_colorado_sampling(point, operation), {
            "type": "inverse_lower_colorado_smoothstep_x",
            "source_to_target_amplitude_x_px": operation[
                "source_to_target_amplitude_x_px"
            ],
        }

    correction = child.get("web_mercator_correction")
    if not isinstance(correction, dict):
        raise ValueError("Child alignment has no Web-Mercator correction")
    matrix = np.asarray(correction.get("current_to_target_pixel_matrix"), dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError("Child alignment lacks a 3 by 3 parent-to-target matrix")
    return _project_point(point, matrix), {
        "type": "projective_parent_to_child",
        "matrix_current_parent_to_target_pixels": matrix.tolist(),
    }


def migrate_stamp_corrections(
    source_run: Path,
    target_run: Path,
) -> Dict[str, object]:
    """Move observed stamp sources through child alignments without moving targets.

    Target pixels are geographic locations in the shared Web-Mercator grid and stay
    fixed. Observed-source pixels follow every verified incremental transform in
    the target's hash-linked alignment chain. Composite/manual sources stay fixed
    because their inputs were authored in target-grid coordinates by earlier
    operations.
    """

    source_run = source_run.resolve()
    target_run = target_run.resolve()
    source_extraction_path = source_run / "extraction.json"
    target_extraction_path = target_run / "extraction.json"
    source_stamp_path = source_run / "stamp-corrections.json"
    for required in (
        source_extraction_path,
        target_extraction_path,
        source_stamp_path,
    ):
        if not required.exists():
            raise FileNotFoundError(required)

    source_extraction = _load(source_extraction_path)
    target_extraction = _load(target_extraction_path)
    stamps = _load(source_stamp_path)
    if stamps.get("extraction_manifest_sha256") != _sha256(source_extraction_path):
        raise ValueError("Source stamp corrections do not match the source extraction")
    if source_extraction["source"]["sha256"] != target_extraction["source"]["sha256"]:
        raise ValueError("Correction migration requires the identical source image")

    source_alignment_path = Path(source_extraction["alignment"]["path"]).resolve()
    target_alignment_path = Path(target_extraction["alignment"]["path"]).resolve()
    source_alignment = _load(source_alignment_path)
    target_alignment = _load(target_alignment_path)
    source_alignment_hash = _sha256(source_alignment_path)
    target_alignment_hash = _sha256(target_alignment_path)
    chain = _alignment_chain(source_alignment_path, target_alignment_path)

    source_grid = _grid_from_manifest(source_extraction)
    target_grid = _grid_from_manifest(target_extraction)
    if not _same_grid(source_grid, target_grid):
        raise ValueError("Correction migration requires an unchanged output grid")
    if chain:
        correction = target_alignment.get("web_mercator_correction")
        if not isinstance(correction, dict):
            raise ValueError("Target alignment has no Web-Mercator correction")
        correction_grid = correction.get("grid")
        if not isinstance(correction_grid, dict):
            raise ValueError("Target correction has no declared grid")
        if not _same_grid(target_grid, correction_grid):
            raise ValueError("Correction migration requires an unchanged output grid")

    width = int(target_grid["width"])
    height = int(target_grid["height"])
    migrated_operations = []
    displacements = []
    mapping_reports = []
    observed_count = 0
    composite_count = 0
    for operation in stamps.get("operations", []):
        migrated = copy.deepcopy(operation)
        source_mode = str(operation.get("source_mode", "observed"))
        if source_mode == "observed":
            old_source = [float(value) for value in operation["source"]]
            new_source = old_source
            operation_mappings = []
            for child_path, child in chain:
                new_source, mapping = _map_parent_point_to_child(new_source, child)
                operation_mappings.append(
                    {
                        "alignment_path": str(child_path),
                        "alignment_sha256": _sha256(child_path),
                        **mapping,
                    }
                )
            if not mapping_reports:
                mapping_reports = operation_mappings
            if not (0 <= new_source[0] < width and 0 <= new_source[1] < height):
                raise ValueError("Migrated observed stamp source falls outside the grid")
            migrated["source"] = new_source
            displacements.append(float(np.hypot(
                new_source[0] - old_source[0], new_source[1] - old_source[1]
            )))
            observed_count += 1
        elif source_mode == "composite_at_operation_time":
            composite_count += 1
        else:
            raise ValueError(f"Unsupported source mode: {source_mode}")
        migrated_operations.append(migrated)

    now = datetime.now(timezone.utc).isoformat()
    migration_provenance = {
        "schema_version": 1,
        "status": "needs_visual_review",
        "approval_carried_forward": False,
        "source_run": str(source_run),
        "target_run": str(target_run),
        "source_extraction": {
            "path": str(source_extraction_path),
            "sha256": _sha256(source_extraction_path),
        },
        "target_extraction": {
            "path": str(target_extraction_path),
            "sha256": _sha256(target_extraction_path),
        },
        "source_stamp_corrections": {
            "path": str(source_stamp_path),
            "sha256": _sha256(source_stamp_path),
        },
        "source_alignment": {
            "path": str(source_alignment_path),
            "sha256": source_alignment_hash,
        },
        "target_alignment": {
            "path": str(target_alignment_path),
            "sha256": target_alignment_hash,
            "alignment_chain_length": len(chain),
        },
        "grid": target_grid,
        "strategy": {
            "observed_sources": "parent pixels transformed into child target pixels",
            "composite_sources": "unchanged authored target-grid pixels",
            "targets": "unchanged geographic target-grid pixels",
            "radii": "unchanged target-grid pixels",
            "incremental_mappings": mapping_reports,
        },
        "operation_count": len(migrated_operations),
        "observed_source_operation_count": observed_count,
        "composite_source_operation_count": composite_count,
        "observed_source_displacement_px": {
            "median": float(np.median(displacements)) if displacements else 0.0,
            "p90": float(np.percentile(displacements, 90)) if displacements else 0.0,
            "maximum": max(displacements, default=0.0),
        },
        "warning": (
            "The previous approval covers only the previous alignment. This migrated "
            "candidate requires a new visual review before publication."
        ),
    }
    migrated_stamps = copy.deepcopy(stamps)
    migrated_stamps.update(
        {
            "dataset_id": target_extraction["dataset_id"],
            "saved_at": now,
            "extraction_manifest_sha256": _sha256(target_extraction_path),
            "inference_manifest_sha256": None,
            "operations": migrated_operations,
            "migration": migration_provenance,
        }
    )
    stamp_output = target_run / "stamp-corrections.json"
    stamp_output.write_text(json.dumps(migrated_stamps, indent=2) + "\n")

    optional_outputs: Dict[str, Dict[str, object]] = {}
    source_inference_path = source_run / "inference-selection.json"
    if source_inference_path.exists():
        inference_selection = {
            "schema_version": 1,
            "enabled": False,
            "migration": {
                "source": str(source_inference_path),
                "source_sha256": _sha256(source_inference_path),
                "status": "retained_disabled",
            },
        }
        target_inference_path = target_run / "inference-selection.json"
        target_inference_path.write_text(json.dumps(inference_selection, indent=2) + "\n")
        optional_outputs["inference_selection"] = {
            "path": str(target_inference_path),
            "sha256": _sha256(target_inference_path),
        }
    source_enclosed_path = source_run / "enclosed-hole-fill-selection.json"
    if source_enclosed_path.exists():
        enclosed_selection = _load(source_enclosed_path)
        enclosed_selection["migration"] = {
            "source": str(source_enclosed_path),
            "source_sha256": _sha256(source_enclosed_path),
            "status": "rules_recomputed_on_child_candidate",
        }
        target_enclosed_path = target_run / "enclosed-hole-fill-selection.json"
        target_enclosed_path.write_text(json.dumps(enclosed_selection, indent=2) + "\n")
        optional_outputs["enclosed_hole_fill_selection"] = {
            "path": str(target_enclosed_path),
            "sha256": _sha256(target_enclosed_path),
        }

    migration_provenance["outputs"] = {
        "stamp_corrections": {
            "path": str(stamp_output),
            "sha256": _sha256(stamp_output),
        },
        **optional_outputs,
    }
    report_path = target_run / "stamp-correction-migration.json"
    report_path.write_text(json.dumps(migration_provenance, indent=2) + "\n")
    return migration_provenance


def audit_alignment_application(
    source_run: Path,
    target_run: Path,
    output_dir: Path,
) -> Dict[str, object]:
    """Prove the target baseline is an exact application of its child alignment."""

    source_run = source_run.resolve()
    target_run = target_run.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_manifest = _load(source_run / "extraction.json")
    target_manifest = _load(target_run / "extraction.json")
    layer_id = str(target_manifest["layers"][0]["id"])
    old_stored_path = source_run / layer_id / "web-mercator-class-id.png"
    new_stored_path = target_run / layer_id / "web-mercator-class-id.png"
    new_source_path = target_run / layer_id / "source-class-id.png"
    old_stored = np.asarray(Image.open(old_stored_path), dtype=np.uint8)
    new_stored = np.asarray(Image.open(new_stored_path), dtype=np.uint8)
    new_source = np.asarray(Image.open(new_source_path), dtype=np.uint8)
    if old_stored.shape != new_stored.shape:
        raise ValueError("Old and new extraction grids differ")

    old_alignment_path = Path(source_manifest["alignment"]["path"])
    new_alignment_path = Path(target_manifest["alignment"]["path"])
    old_alignment = _load(old_alignment_path)
    new_alignment = _load(new_alignment_path)
    plan = _load(target_run / "plan.snapshot.json")
    state, _ = load_california(Path(plan.get("reference", "reference/census-2025")))
    source_shape = (
        int(target_manifest["source"]["height"]),
        int(target_manifest["source"]["width"]),
    )
    expected_new, new_grid = warp_classified_to_web_mercator(
        new_source,
        state,
        _alignment_transform(new_alignment),
        source_shape,
    )
    expected_parent, parent_grid = warp_classified_to_web_mercator(
        new_source,
        state,
        _alignment_transform(old_alignment),
        source_shape,
    )
    mismatch = expected_new != new_stored
    parent_difference = expected_parent != new_stored
    stored_difference = old_stored != new_stored
    expected_target_path = output_dir / "recomputed-target-class-id.png"
    Image.fromarray(expected_new).save(expected_target_path, optimize=True)
    artifacts = {
        "recomputed-target-class-id.png": {
            "path": expected_target_path.name,
            "sha256": _sha256(expected_target_path),
        }
    }
    for name, mask in (
        ("target-recompute-mismatch-mask.png", mismatch),
        ("target-vs-parent-recompute-difference-mask.png", parent_difference),
        ("stored-old-vs-new-difference-mask.png", stored_difference),
    ):
        path = output_dir / name
        Image.fromarray(mask.astype(np.uint8) * 255).save(path, optimize=True)
        artifacts[name] = {"path": name, "sha256": _sha256(path)}
    source_alignment_sha256 = _sha256(old_alignment_path)
    target_alignment_sha256 = _sha256(new_alignment_path)
    validated_noop = source_alignment_sha256 == target_alignment_sha256
    correction = new_alignment.get("web_mercator_correction")
    if correction is not None:
        current_to_target_pixel_matrix = correction[
            "current_to_target_pixel_matrix"
        ]
    elif validated_noop:
        current_to_target_pixel_matrix = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    else:
        raise ValueError(
            "Target alignment has no Web-Mercator correction and is not a "
            "hash-identical validated no-op"
        )
    exact_recompute = not np.any(mismatch)
    applied_change = bool(np.any(parent_difference))
    audit_passes = exact_recompute and (applied_change or validated_noop)
    result = {
        "schema_version": 1,
        "status": "pass" if audit_passes else "fail",
        "application_mode": "validated_noop" if validated_noop else "applied_change",
        "layer_id": layer_id,
        "source_alignment": {
            "path": str(old_alignment_path),
            "sha256": source_alignment_sha256,
        },
        "target_alignment": {
            "path": str(new_alignment_path),
            "sha256": target_alignment_sha256,
            "parent": new_alignment.get("parent_alignment"),
            "current_to_target_pixel_matrix": current_to_target_pixel_matrix,
        },
        "new_grid": new_grid,
        "parent_grid": parent_grid,
        "stored_parent_baseline": {
            "path": str(old_stored_path),
            "sha256": _sha256(old_stored_path),
        },
        "stored_target_baseline": {
            "path": str(new_stored_path),
            "sha256": _sha256(new_stored_path),
        },
        "recomputed_target_baseline_sha256": _sha256(expected_target_path),
        "target_recompute_mismatch_pixel_count": int(np.count_nonzero(mismatch)),
        "target_vs_parent_recompute_difference_pixel_count": int(
            np.count_nonzero(parent_difference)
        ),
        "stored_parent_vs_target_difference_pixel_count": int(
            np.count_nonzero(stored_difference)
        ),
        "stored_parent_vs_target_difference_fraction": float(np.mean(stored_difference)),
        "artifacts": artifacts,
        "conclusion": (
            "The stored target is an exact pixel-for-pixel recomputation under the "
            "hash-identical validated no-op alignment."
            if validated_noop
            else
            "The stored target is an exact pixel-for-pixel recomputation under the "
            "declared target alignment and differs from the parent-alignment recomputation."
        ),
    }
    (output_dir / "alignment-application-audit.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


def compare_materialized_candidates(
    approved_dir: Path,
    candidate_dir: Path,
    output_dir: Path,
) -> Dict[str, object]:
    """Write an auditable class transition summary without granting approval."""

    approved_dir = approved_dir.resolve()
    candidate_dir = candidate_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    approved_manifest = _load(approved_dir / "materialization.json")
    candidate_manifest = _load(candidate_dir / "materialization.json")
    approved_layer = approved_manifest["layers"][0]
    candidate_layer = candidate_manifest["layers"][0]
    layer_id = str(candidate_layer["layer_id"])

    def artifact(root: Path, layer: Dict[str, object], name: str) -> Path:
        return root / layer["artifacts"][name]["path"]

    old_class_path = artifact(approved_dir, approved_layer, "class_id")
    new_class_path = artifact(candidate_dir, candidate_layer, "class_id")
    old_values = np.asarray(Image.open(old_class_path), dtype=np.uint8)
    new_values = np.asarray(Image.open(new_class_path), dtype=np.uint8)
    if old_values.shape != new_values.shape:
        raise ValueError("Materialized candidate grids differ")
    changed = old_values != new_values
    max_class = int(max(old_values.max(initial=0), new_values.max(initial=0)))
    transitions = {
        f"{old}->{new}": int(np.count_nonzero((old_values == old) & (new_values == new)))
        for old in range(max_class + 1)
        for new in range(max_class + 1)
        if old != new and np.any((old_values == old) & (new_values == new))
    }
    old_manual_mask_path = artifact(approved_dir, approved_layer, "manual_mask")
    new_manual_mask_path = artifact(candidate_dir, candidate_layer, "manual_mask")
    old_manual_values_path = artifact(approved_dir, approved_layer, "manual_values")
    new_manual_values_path = artifact(candidate_dir, candidate_layer, "manual_values")
    old_manual_mask = np.asarray(Image.open(old_manual_mask_path)) > 0
    new_manual_mask = np.asarray(Image.open(new_manual_mask_path)) > 0
    old_manual_values = np.asarray(Image.open(old_manual_values_path), dtype=np.uint8)
    new_manual_values = np.asarray(Image.open(new_manual_values_path), dtype=np.uint8)
    change_mask_path = output_dir / "approved-vs-candidate-change-mask.png"
    Image.fromarray(changed.astype(np.uint8) * 255).save(change_mask_path, optimize=True)
    change_overlay_path = output_dir / "approved-vs-candidate-change-overlay.png"
    change_overlay = np.zeros((*changed.shape, 4), dtype=np.uint8)
    change_overlay[changed] = [255, 0, 255, 190]
    Image.fromarray(change_overlay).save(change_overlay_path, optimize=True)
    result = {
        "schema_version": 1,
        "status": "needs_visual_review",
        "approval_carried_forward": False,
        "layer_id": layer_id,
        "approved": {
            "path": str(approved_dir),
            "materialization_sha256": _sha256(approved_dir / "materialization.json"),
            "review_decision": (
                _load(approved_dir / "materialization-review-decision.json")
                if (approved_dir / "materialization-review-decision.json").exists()
                else None
            ),
            "class_id_sha256": _sha256(old_class_path),
        },
        "candidate": {
            "path": str(candidate_dir),
            "materialization_sha256": _sha256(candidate_dir / "materialization.json"),
            "class_id_sha256": _sha256(new_class_path),
        },
        "grid": {"width": old_values.shape[1], "height": old_values.shape[0]},
        "changed_pixel_count": int(np.count_nonzero(changed)),
        "changed_pixel_fraction": float(np.mean(changed)),
        "old_nonzero_pixel_count": int(np.count_nonzero(old_values)),
        "new_nonzero_pixel_count": int(np.count_nonzero(new_values)),
        "manual_target_mask_identical": bool(np.array_equal(old_manual_mask, new_manual_mask)),
        "manual_target_pixel_count": int(np.count_nonzero(new_manual_mask)),
        "manual_value_changed_pixel_count": int(
            np.count_nonzero(
                old_manual_mask & new_manual_mask & (old_manual_values != new_manual_values)
            )
        ),
        "class_transitions": transitions,
        "artifacts": {
            "change_mask": {
                "path": change_mask_path.name,
                "sha256": _sha256(change_mask_path),
            },
            "change_overlay": {
                "path": change_overlay_path.name,
                "sha256": _sha256(change_overlay_path),
            },
        },
        "warning": (
            "Differences include the accepted alignment movement, the migrated stamp "
            "source sampling, and recomputed enclosed-hole fill. Visual review is still required."
        ),
    }
    (output_dir / "approved-vs-candidate-comparison.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


def audit_stamp_migration(
    source_run: Path,
    target_run: Path,
    output_dir: Path,
    *,
    approved_materialized_dir: Path | None = None,
    candidate_materialized_dir: Path | None = None,
) -> Dict[str, object]:
    """Compare migrated replay with the unchanged-coordinate counterfactual."""

    source_run = source_run.resolve()
    target_run = target_run.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_manifest = _load(source_run / "extraction.json")
    target_manifest = _load(target_run / "extraction.json")
    layer_id = str(target_manifest["layers"][0]["id"])
    approved_dir = (
        approved_materialized_dir.resolve()
        if approved_materialized_dir is not None
        else source_run / "materialized-v4"
    )
    candidate_dir = (
        candidate_materialized_dir.resolve()
        if candidate_materialized_dir is not None
        else target_run / "materialized-v1"
    )
    old_manual_values = np.asarray(
        Image.open(
            approved_dir / layer_id / "web-mercator-manual-values.png"
        ),
        dtype=np.uint8,
    )
    old_manual_mask = np.asarray(
        Image.open(
            approved_dir / layer_id / "web-mercator-manual-override-mask.png"
        )
    ) > 0
    new_observed = np.asarray(
        Image.open(target_run / layer_id / "web-mercator-class-id.png"),
        dtype=np.uint8,
    )
    unchanged_operations = _load(source_run / "stamp-corrections.json")[
        "operations"
    ]
    migrated_operations = _load(target_run / "stamp-corrections.json")[
        "operations"
    ]
    unchanged_values, unchanged_mask = apply_clone_stamp_operations(
        new_observed, unchanged_operations
    )
    migrated_values, migrated_mask = apply_clone_stamp_operations(
        new_observed, migrated_operations
    )
    stored_candidate_values = np.asarray(
        Image.open(
            candidate_dir / layer_id / "web-mercator-manual-values.png"
        ),
        dtype=np.uint8,
    )
    stored_candidate_mask = np.asarray(
        Image.open(
            candidate_dir / layer_id / "web-mercator-manual-override-mask.png"
        )
    ) > 0
    migrated_difference = old_manual_mask & (migrated_values != old_manual_values)
    unchanged_difference = old_manual_mask & (unchanged_values != old_manual_values)
    migrated_count = int(np.count_nonzero(migrated_difference))
    unchanged_count = int(np.count_nonzero(unchanged_difference))
    manual_count = int(np.count_nonzero(old_manual_mask))
    masks_identical = (
        np.array_equal(old_manual_mask, migrated_mask)
        and np.array_equal(old_manual_mask, unchanged_mask)
        and np.array_equal(old_manual_mask, stored_candidate_mask)
    )
    replay_exact = np.array_equal(migrated_values, stored_candidate_values)
    migration_record = _load(target_run / "stamp-correction-migration.json")
    observed_source_count = int(
        migration_record.get("observed_source_operation_count", 0)
    )
    comparison_passed = (
        migrated_count < unchanged_count
        if observed_source_count > 0
        else migrated_count == unchanged_count
    )
    artifacts = {}
    for name, mask in (
        ("migrated-vs-approved-manual-value-difference-mask.png", migrated_difference),
        ("unchanged-source-vs-approved-manual-value-difference-mask.png", unchanged_difference),
    ):
        path = output_dir / name
        Image.fromarray(mask.astype(np.uint8) * 255).save(path, optimize=True)
        artifacts[name] = {"path": name, "sha256": _sha256(path)}
    result = {
        "schema_version": 1,
        "status": (
            "pass"
            if masks_identical and replay_exact and comparison_passed
            else "fail"
        ),
        "source_dataset_id": source_manifest["dataset_id"],
        "target_dataset_id": target_manifest["dataset_id"],
        "layer_id": layer_id,
        "manual_target_mask_identical": masks_identical,
        "migrated_replay_matches_stored_candidate": replay_exact,
        "manual_target_pixel_count": manual_count,
        "observed_source_operation_count": observed_source_count,
        "migrated_source_changed_value_pixel_count": migrated_count,
        "migrated_source_changed_value_fraction": migrated_count / max(manual_count, 1),
        "unchanged_source_changed_value_pixel_count": unchanged_count,
        "unchanged_source_changed_value_fraction": unchanged_count / max(manual_count, 1),
        "changed_value_reduction_pixel_count": unchanged_count - migrated_count,
        "changed_value_reduction_fraction": (
            (unchanged_count - migrated_count) / max(unchanged_count, 1)
        ),
        "artifacts": artifacts,
        "conclusion": (
            "Migrated replay matches the stored candidate and preserves every authored "
            "target. Observed stamp sources must improve over stale coordinates; a "
            "composite-only correction history must remain exactly unchanged."
        ),
    }
    (output_dir / "stamp-migration-audit.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result
