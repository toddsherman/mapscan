"""Promote an exactly reviewed extraction into an exportable materialization.

This path is intentionally narrower than correction materialization: it accepts
no inference or manual edits.  It can only package the byte-identical class
raster that already passed both alignment and category review, while carrying
the extraction's canonical clipping and derived-pixel masks forward.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
from PIL import Image

from .canonical_boundary import ACTIVE_CANONICAL_POINTER, load_active_canonical_border
from .canonical_clip import (
    canonical_publication_interior,
    close_west_coast_clipping_seam,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text())


def _resolve_declared_path(value: object, root: Path) -> Path:
    path = Path(str(value))
    candidates = [path] if path.is_absolute() else [Path.cwd() / path, root / path]
    resolved = next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)
    if resolved is None:
        raise FileNotFoundError(path)
    return resolved


def _verified_artifact(root: Path, record: object, label: str) -> Path:
    if not isinstance(record, dict):
        raise ValueError(f"Missing {label} artifact record")
    path = root / str(record.get("path", ""))
    if not path.is_file() or _sha256(path) != record.get("sha256"):
        raise ValueError(f"{label} artifact is missing or hash-mismatched")
    return path


def _copy_artifact(source: Path, target: Path) -> Dict[str, object]:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    if _sha256(source) != _sha256(target):
        raise ValueError(f"Copied artifact changed bytes: {source}")
    return {"path": str(target), "sha256": _sha256(target)}


def _source_diff_provenance(
    batch_path: Path, case_id: str, extraction_hash: str, class_hashes: set[str]
) -> Dict[str, object]:
    batch_path = batch_path.resolve()
    batch = _load(batch_path)
    case = next(
        (
            item
            for item in batch.get("cases", [])
            if isinstance(item, dict) and item.get("id") == case_id
        ),
        None,
    )
    if (
        batch.get("status") != "pass"
        or not isinstance(case, dict)
        or case.get("status") != "pass"
        or case.get("fixed_point_reached") is not True
    ):
        raise ValueError("Source-diff case has not reached a passing fixed point")
    report_path = batch_path.parent / str(case.get("report", ""))
    report = _load(report_path)
    if (
        report.get("status") != "pass"
        or report.get("manifest_sha256") != extraction_hash
    ):
        raise ValueError("Source-diff report does not match the reviewed extraction")
    audited_hashes = {
        str(layer.get("artifacts", {}).get("audited_class_id", {}).get("sha256", ""))
        for layer in report.get("layers", [])
        if isinstance(layer, dict)
    }
    if audited_hashes != class_hashes:
        raise ValueError("Source-diff fixed point is not byte-identical to the reviewed classes")
    return {
        "fixed_point_reached": True,
        "batch": {"path": str(batch_path), "sha256": _sha256(batch_path)},
        "report": {"path": str(report_path), "sha256": _sha256(report_path)},
        "comparison_iterations": case.get("comparison_iterations", []),
    }


def promote_reviewed_extraction(
    run_dir: Path,
    output_dir: Path,
    *,
    author_statement: str,
    source_diff_batch_path: Path,
    source_diff_case_id: str,
    canonical_pointer_path: Path = ACTIVE_CANONICAL_POINTER,
) -> Dict[str, object]:
    """Create a reviewed materialization without changing a single class cell."""

    statement = author_statement.strip()
    if not statement:
        raise ValueError("An author approval statement is required")
    run_dir = run_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise ValueError("Reviewed extraction promotion requires a fresh output directory")

    extraction_path = run_dir / "extraction.json"
    plan_snapshot_path = run_dir / "plan.snapshot.json"
    alignment_decision_path = run_dir / "review-decision.json"
    classification_decision_path = run_dir / "classification-review-decision.json"
    extraction = _load(extraction_path)
    plan = _load(plan_snapshot_path)
    extraction_hash = _sha256(extraction_path)
    alignment_decision = _load(alignment_decision_path)
    classification_decision = _load(classification_decision_path)
    for label, decision in (
        ("alignment", alignment_decision),
        ("classification", classification_decision),
    ):
        if (
            decision.get("status") != "approved"
            or decision.get("extraction_manifest_sha256") != extraction_hash
        ):
            raise ValueError(f"{label.title()} approval does not match the extraction")

    declared_plan = _resolve_declared_path(extraction["plan"]["path"], run_dir)
    if _sha256(declared_plan) != extraction["plan"]["sha256"]:
        raise ValueError("Reviewed extraction plan is stale")
    if _load(plan_snapshot_path) != _load(declared_plan):
        raise ValueError("Plan snapshot differs semantically from the reviewed plan")

    active_manifest_path, active, pointer = load_active_canonical_border(
        canonical_pointer_path
    )
    active_pointer_path = canonical_pointer_path.resolve()
    active_overlay_path = active_manifest_path.parent / str(
        active["artifacts"]["overlay"]["path"]
    )
    if _sha256(active_overlay_path) != active["artifacts"]["overlay"]["sha256"]:
        raise ValueError("Active canonical display border is stale")
    expected_components = int(active["topology"]["combined_component_count"])

    plan_layers = {str(item["id"]): item for item in plan.get("layers", [])}
    layer_reports = []
    class_hashes: set[str] = set()
    output_dir.mkdir(parents=True)
    shared_valid: np.ndarray | None = None
    shared_grid: Dict[str, object] | None = None
    shared_canonical_clip: Dict[str, object] | None = None
    direct_masks: Dict[str, np.ndarray] = {}

    for layer in extraction.get("layers", []):
        if layer.get("kind") not in {"categorical", "patterned_categorical"}:
            continue
        layer_id = str(layer["id"])
        definition = plan_layers.get(layer_id)
        if not isinstance(definition, dict):
            raise ValueError(f"Missing plan definition for {layer_id}")
        canonical_clip = layer.get("canonical_clip", {})
        active_record = canonical_clip.get("active_manifest", {})
        if (
            active_record.get("canonical_boundary_id") != active["canonical_boundary_id"]
            or active_record.get("sha256") != _sha256(active_manifest_path)
            or int(canonical_clip.get("component_count", 0)) != expected_components
        ):
            raise ValueError("Extraction does not use the active canonical boundary")

        class_source = run_dir / layer_id / "web-mercator-class-id.png"
        preview_source = run_dir / layer_id / "web-mercator-preview.png"
        if not class_source.is_file() or not preview_source.is_file():
            raise FileNotFoundError(f"Missing reviewed categorical artifacts for {layer_id}")
        values = np.asarray(Image.open(class_source), dtype=np.uint8)
        valid_source = _verified_artifact(
            run_dir,
            canonical_clip.get("artifacts", {}).get("interior"),
            f"{layer_id} canonical interior",
        )
        valid = np.asarray(Image.open(valid_source).convert("L")) > 0
        if values.shape != valid.shape:
            raise ValueError("Reviewed classes and canonical interior use different grids")
        outside_count = int(np.count_nonzero((values > 0) & ~valid))
        inside_empty_count = int(np.count_nonzero((values == 0) & valid))
        coverage_expectation = str(
            definition.get("coverage_expectation", "full_state")
        )
        if coverage_expectation not in {"full_state", "sparse_visible_evidence"}:
            raise ValueError(f"Unsupported coverage expectation: {coverage_expectation}")
        if outside_count or (
            inside_empty_count and coverage_expectation != "sparse_visible_evidence"
        ):
            raise ValueError("Reviewed classes do not satisfy their canonical coverage contract")
        if shared_valid is None:
            shared_valid = valid
            shared_grid = dict(layer["warp"])
            shared_canonical_clip = dict(canonical_clip)
        elif not np.array_equal(valid, shared_valid):
            raise ValueError("Categorical layers use different publication interiors")

        completion_record = layer.get("completion_artifacts", {}).get(
            "web_mercator_target_completion_mask"
        )
        if isinstance(completion_record, dict):
            completion_source = _verified_artifact(
                run_dir, completion_record, f"{layer_id} completion mask"
            )
            completion = np.asarray(Image.open(completion_source).convert("L")) > 0
        else:
            completion_source = None
            completion = np.zeros(values.shape, dtype=bool)
        if completion.shape != values.shape or np.any(completion & ~valid):
            raise ValueError("Completion evidence lies outside the canonical interior")
        direct = (values > 0) & ~completion
        direct_masks[layer_id] = direct

        speck = layer.get("speck_suppression", {})
        speck_mask_source = None
        speck_original_source = None
        if isinstance(speck, dict) and speck.get("artifacts"):
            speck_mask_source = _verified_artifact(
                run_dir, speck["artifacts"].get("mask"), f"{layer_id} speck mask"
            )
            speck_original_source = _verified_artifact(
                run_dir,
                speck["artifacts"].get("original_class_id"),
                f"{layer_id} speck originals",
            )
            speck_mask = np.asarray(Image.open(speck_mask_source).convert("L")) > 0
            if speck_mask.shape != values.shape or np.any(speck_mask & ~valid):
                raise ValueError("Speck reassignment evidence lies outside the canonical interior")

        target_dir = output_dir / layer_id
        target_dir.mkdir(parents=True)
        artifacts: Dict[str, Dict[str, object]] = {}

        def copy(name: str, source: Path, filename: str) -> None:
            record = _copy_artifact(source, target_dir / filename)
            record["path"] = str(Path(layer_id) / filename)
            artifacts[name] = record

        copy("class_id", class_source, "web-mercator-class-id-final.png")
        copy("preview", preview_source, "web-mercator-preview-final.png")
        class_hashes.add(artifacts["class_id"]["sha256"])

        def save(name: str, filename: str, array: np.ndarray) -> None:
            path = target_dir / filename
            Image.fromarray(array).save(path, optimize=True)
            artifacts[name] = {
                "path": str(Path(layer_id) / filename),
                "sha256": _sha256(path),
            }

        save("observed_mask", "web-mercator-observed-retained-mask.png", direct.astype(np.uint8) * 255)
        save("inference_mask", "web-mercator-inference-retained-mask.png", np.zeros(values.shape, dtype=np.uint8))
        save("manual_mask", "web-mercator-manual-override-mask.png", np.zeros(values.shape, dtype=np.uint8))
        save("manual_values", "web-mercator-manual-values.png", np.zeros(values.shape, dtype=np.uint8))
        if completion_source is not None:
            copy("target_completion_mask", completion_source, "web-mercator-target-completion-mask.png")
        if speck_mask_source is not None and speck_original_source is not None:
            copy("speck_reassignment_mask", speck_mask_source, "web-mercator-speck-reassignment-mask.png")
            copy("speck_original_class_id", speck_original_source, "web-mercator-speck-original-class-id.png")
        removed_record = canonical_clip.get("artifacts", {}).get("removed")
        if isinstance(removed_record, dict):
            copy(
                "boundary_removed_mask",
                _verified_artifact(run_dir, removed_record, f"{layer_id} boundary removal"),
                "web-mercator-boundary-removed-mask.png",
            )

        categories = definition.get("categories", [])
        layer_reports.append(
            {
                "layer_id": layer_id,
                "coverage_expectation": coverage_expectation,
                "width": int(values.shape[1]),
                "height": int(values.shape[0]),
                "observed_retained_pixel_count": int(np.count_nonzero(direct)),
                "target_completion_pixel_count": int(np.count_nonzero(completion)),
                "speck_reassignment_pixel_count": int(
                    speck.get("reassigned_pixel_count", 0) if isinstance(speck, dict) else 0
                ),
                "manual_override_pixel_count": 0,
                "inference_retained_pixel_count": 0,
                "final_classified_pixel_count": int(np.count_nonzero(values)),
                "final_pixels_by_class_id": {
                    str(class_id): int(np.count_nonzero(values == class_id))
                    for class_id in range(1, len(categories) + 1)
                },
                "colored_pixel_count_outside_boundary": outside_count,
                "unclassified_pixel_count_inside_boundary": inside_empty_count,
                "artifacts": artifacts,
            }
        )

    if (
        not layer_reports
        or shared_valid is None
        or shared_grid is None
        or shared_canonical_clip is None
    ):
        raise ValueError("No reviewed categorical layers were available")
    source_diff = _source_diff_provenance(
        source_diff_batch_path,
        source_diff_case_id,
        extraction_hash,
        class_hashes,
    )

    canonical_valid = shared_valid
    internal_water = shared_canonical_clip.get("internal_water_exclusion")
    internal_water_mask: np.ndarray | None = None
    internal_water_provenance: Dict[str, object] | None = None
    reconstructed_coastal_seam = None
    if isinstance(internal_water, dict):
        mainland_record = shared_canonical_clip.get("mainland_manifest", {})
        if not isinstance(mainland_record, dict):
            raise ValueError("Water-aware extraction has no canonical mainland record")
        mainland_path = _resolve_declared_path(mainland_record.get("path"), run_dir)
        canonical_valid, _ = canonical_publication_interior(
            shared_grid,
            mainland_manifest_path=mainland_path,
            active_pointer_path=canonical_pointer_path,
        )
        declared_coastal_seam = shared_canonical_clip.get("coastal_seam")
        if declared_coastal_seam is not None:
            if not isinstance(declared_coastal_seam, dict):
                raise ValueError("Reviewed coastal seam provenance is invalid")
            canonical_valid, reconstructed_coastal_seam, _ = (
                close_west_coast_clipping_seam(
                    canonical_valid,
                    shared_grid,
                    maximum_gap_px=int(
                        declared_coastal_seam.get("maximum_gap_px", 50)
                    ),
                    maximum_x_fraction=float(
                        declared_coastal_seam.get("maximum_x_fraction", 0.72)
                    ),
                    start_y_fraction=float(
                        declared_coastal_seam.get("start_y_fraction", 0.04)
                    ),
                    end_y_fraction=float(
                        declared_coastal_seam.get("end_y_fraction", 0.95)
                    ),
                    active_pointer_path=canonical_pointer_path,
                )
            )
            seam_fields = (
                "method",
                "maximum_gap_px",
                "maximum_x_fraction",
                "start_y_fraction",
                "end_y_fraction",
                "accepted_row_count",
                "canonical_line_row_count",
                "interpolated_line_row_count",
                "added_pixel_count",
                "accepted_gap_median_px",
                "accepted_gap_max_px",
                "large_opening_policy",
            )
            if any(
                declared_coastal_seam.get(field)
                != reconstructed_coastal_seam.get(field)
                for field in seam_fields
            ):
                raise ValueError("Reviewed coastal seam does not reconstruct exactly")
            declared_active = declared_coastal_seam.get("active_manifest", {})
            reconstructed_active = reconstructed_coastal_seam.get(
                "active_manifest", {}
            )
            if declared_active != reconstructed_active:
                raise ValueError("Reviewed coastal seam uses a stale canonical line")
        if np.any(shared_valid & ~canonical_valid):
            raise ValueError("Publication interior extends beyond the canonical boundary")
        expected_base_count = int(
            shared_canonical_clip.get("base_valid_pixel_count", -1)
        )
        expected_publication_count = int(
            shared_canonical_clip.get("valid_pixel_count", -1)
        )
        expected_excluded_count = int(
            internal_water.get("excluded_interior_pixel_count", -1)
        )
        excluded_count = int(np.count_nonzero(canonical_valid & ~shared_valid))
        if (
            expected_base_count != int(np.count_nonzero(canonical_valid))
            or expected_publication_count != int(np.count_nonzero(shared_valid))
            or expected_excluded_count != excluded_count
        ):
            raise ValueError("Internal-water publication counts are stale")
        water_artifact = internal_water.get("artifact")
        if isinstance(water_artifact, dict):
            water_source = _verified_artifact(
                run_dir, water_artifact, "reviewed internal-water mask"
            )
            internal_water_mask = np.asarray(
                Image.open(water_source).convert("L")
            ) > 0
            expected_water = canonical_valid & ~shared_valid
            if internal_water_mask.shape != shared_valid.shape or not np.array_equal(
                internal_water_mask, expected_water
            ):
                raise ValueError(
                    "Reviewed internal-water mask does not exactly reconstruct the publication exclusion"
                )
            shoreline = internal_water.get("canonical_shoreline_snap")
            internal_water_provenance = {
                "method": internal_water.get("method"),
                "reference_manifest": internal_water.get("reference_manifest"),
                "excluded_interior_pixel_count": excluded_count,
                "colored_water_policy": internal_water.get("colored_water_policy"),
                "canonical_shoreline_snap": shoreline,
            }

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        canonical_valid.astype(np.uint8), 8
    )
    component_ids = sorted(
        range(1, component_count),
        key=lambda component: int(stats[component, cv2.CC_STAT_AREA]),
        reverse=True,
    )
    if len(component_ids) != expected_components:
        raise ValueError("Canonical publication interior component count changed")
    components = []
    direct_union = np.zeros(shared_valid.shape, dtype=bool)
    for direct in direct_masks.values():
        direct_union |= direct
    for index, component in enumerate(component_ids):
        selected = labels == component
        components.append(
            {
                "id": "mainland" if index == 0 else f"canonical-island-{index:02d}",
                "role": (
                    "canonical_mainland_clipping"
                    if index == 0
                    else "author_approved_canonical_island"
                ),
                "authority": active["canonical_boundary_id"],
                "interior_pixel_count": int(np.count_nonzero(selected)),
                "observed_source_pixel_count": int(np.count_nonzero(selected & direct_union)),
            }
        )

    publication_interior_path = output_dir / "publication-interior-mask.png"
    Image.fromarray(shared_valid.astype(np.uint8) * 255).save(
        publication_interior_path, optimize=True
    )
    canonical_interior_path = output_dir / "canonical-publication-interior-mask.png"
    Image.fromarray(canonical_valid.astype(np.uint8) * 255).save(
        canonical_interior_path, optimize=True
    )
    if internal_water_mask is not None and internal_water_provenance is not None:
        internal_water_path = output_dir / "internal-water-exclusion-mask.png"
        Image.fromarray(internal_water_mask.astype(np.uint8) * 255).save(
            internal_water_path, optimize=True
        )
        internal_water_provenance["artifact"] = {
            "path": internal_water_path.name,
            "sha256": _sha256(internal_water_path),
        }
    mainland = (labels == component_ids[0]) & shared_valid
    mainland_path = output_dir / "publication-mainland-interior-mask.png"
    Image.fromarray(mainland.astype(np.uint8) * 255).save(mainland_path, optimize=True)
    integrity_overlay = np.zeros((*shared_valid.shape, 4), dtype=np.uint8)
    contours, _ = cv2.findContours(
        canonical_valid.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    cv2.drawContours(integrity_overlay, contours, -1, (40, 255, 110, 255), 1)
    integrity_path = output_dir / "publication-integrity-boundary-overlay.png"
    Image.fromarray(integrity_overlay).save(integrity_path, optimize=True)

    total_unclassified_inside = sum(
        int(layer["unclassified_pixel_count_inside_boundary"])
        for layer in layer_reports
    )
    boundary_audit = {
        "schema_version": 1,
        "status": "pass",
        "method": "reviewed_extraction_exact_canonical_clip",
        "boundary": {
            "connected_component_count": expected_components,
            "expected_component_count": expected_components,
            "mainland_interior_pixel_count": int(np.count_nonzero(mainland)),
            "publication_interior_pixel_count": int(np.count_nonzero(shared_valid)),
            "canonical_interior_pixel_count": int(np.count_nonzero(canonical_valid)),
            "publication_interior_exclusion_pixel_count": int(
                np.count_nonzero(canonical_valid & ~shared_valid)
            ),
            "publication_interior_component_count": int(
                cv2.connectedComponents(shared_valid.astype(np.uint8), 8)[0] - 1
            ),
            "selection_policy": {
                "mainland": (
                    "approved canonical mainland clipping interior plus exact reviewed "
                    "lime-bounded coastal seam"
                    if reconstructed_coastal_seam is not None
                    else "approved canonical mainland clipping interior"
                ),
                "islands": "four author-approved county.png canonical outlines",
                "display_border_reconstructed_from_fill": False,
                "publication_interior": (
                    "canonical interior minus exact reviewed internal exclusions"
                    if isinstance(internal_water, dict)
                    else "canonical interior"
                ),
            },
            "components": components,
            "interior": {
                "path": publication_interior_path.name,
                "sha256": _sha256(publication_interior_path),
            },
            "canonical_interior": {
                "path": canonical_interior_path.name,
                "sha256": _sha256(canonical_interior_path),
            },
            "integrity_border": {
                "path": integrity_path.name,
                "sha256": _sha256(integrity_path),
            },
            "canonical_display_border": {
                "path": str(active_overlay_path),
                "sha256": _sha256(active_overlay_path),
                "canonical_boundary_id": active["canonical_boundary_id"],
                "grid": active["source_grid"],
            },
            "coastal_seam": reconstructed_coastal_seam,
            "internal_water_exclusion": internal_water_provenance,
        },
        "layers": [
            {
                "layer_id": layer["layer_id"],
                "passed": True,
                "coverage_expectation": layer["coverage_expectation"],
                "colored_pixel_count_outside_boundary": 0,
                "unclassified_pixel_count_inside_boundary": layer[
                    "unclassified_pixel_count_inside_boundary"
                ],
            }
            for layer in layer_reports
        ],
    }
    boundary_audit_path = output_dir / "boundary-clip-audit.json"
    boundary_audit_path.write_text(json.dumps(boundary_audit, indent=2) + "\n")

    alignment_path = _resolve_declared_path(plan["alignment"], run_dir)
    materialization = {
        "schema_version": 1,
        "status": "approved_reviewed_extraction_materialization",
        "dataset_id": extraction["dataset_id"],
        "source_run": str(run_dir),
        "extraction_manifest_sha256": extraction_hash,
        "reviewed_extraction": {
            "alignment_decision_sha256": _sha256(alignment_decision_path),
            "classification_decision_sha256": _sha256(classification_decision_path),
            "class_raster_policy": "byte_identical_to_dual_reviewed_extraction",
        },
        "source_diff": source_diff,
        "boundary_clip": {
            "audit": {
                "path": str(boundary_audit_path),
                "sha256": _sha256(boundary_audit_path),
            },
            "continuous_border_sha256": _sha256(active_overlay_path),
            "mainland_interior_sha256": _sha256(mainland_path),
            "publication_interior_sha256": _sha256(publication_interior_path),
            "canonical_interior_sha256": _sha256(canonical_interior_path),
            "publication_interior_exclusion_pixel_count": int(
                np.count_nonzero(canonical_valid & ~shared_valid)
            ),
            "publication_interior_component_count": int(
                cv2.connectedComponents(shared_valid.astype(np.uint8), 8)[0] - 1
            ),
            "boundary_component_count": expected_components,
            "expected_boundary_component_count": expected_components,
            "colored_pixel_count_outside_boundary": 0,
            "unclassified_pixel_count_inside_boundary": total_unclassified_inside,
            "coverage_contract": (
                "sparse_visible_evidence"
                if any(
                    layer["coverage_expectation"] == "sparse_visible_evidence"
                    for layer in layer_reports
                )
                else "full_state"
            ),
            "canonical_border": {
                "canonical_boundary_id": active["canonical_boundary_id"],
                "manifest_sha256": _sha256(active_manifest_path),
                "display_overlay_sha256": _sha256(active_overlay_path),
            },
            "coastal_seam": reconstructed_coastal_seam,
            "internal_water_exclusion": internal_water_provenance,
        },
        "precedence": [
            "aligned_legend_classification",
            "neighbor_derived_cartographic_completion",
            "tiny_isolated_speck_reassignment",
            "canonical_boundary_clip",
        ],
        "warning": (
            "Completion and isolated-speck decisions remain separately masked; "
            "neither is relabeled as direct source observation."
        ),
        "layers": layer_reports,
    }
    materialization_path = output_dir / "materialization.json"
    materialization_path.write_text(json.dumps(materialization, indent=2) + "\n")
    decision = {
        "schema_version": 1,
        "scope": "byte_identical_reviewed_extraction_materialization",
        "status": "approved",
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "materialization_sha256": _sha256(materialization_path),
        "materialization_path": str(materialization_path),
        "alignment_sha256": _sha256(alignment_path),
        "alignment_review_decision_sha256": _sha256(alignment_decision_path),
        "classification_review_decision_sha256": _sha256(classification_decision_path),
        "canonical_boundary_manifest_sha256": _sha256(active_manifest_path),
        "canonical_display_border_sha256": _sha256(active_overlay_path),
        "boundary_clip_audit_sha256": _sha256(boundary_audit_path),
        "hybrid_border_sha256": _sha256(active_overlay_path),
        "source_diff_batch_sha256": _sha256(source_diff_batch_path.resolve()),
        "author_statement": statement,
        "inspection_confirmed": True,
        "approval_carried_forward": False,
        "approval_basis": (
            "The materialized class raster is byte-identical to the extraction "
            "approved in both alignment and per-category review."
        ),
        "canonical_pointer_sha256": _sha256(active_pointer_path),
        "canonical_pointer_status": pointer["status"],
    }
    decision_path = output_dir / "materialization-review-decision.json"
    decision_path.write_text(json.dumps(decision, indent=2) + "\n")
    return {
        "status": "approved",
        "materialization": {
            "path": str(materialization_path),
            "sha256": _sha256(materialization_path),
        },
        "decision": {"path": str(decision_path), "sha256": _sha256(decision_path)},
        "class_raster_hashes": sorted(class_hashes),
        "boundary_component_count": expected_components,
        "colored_pixel_count_outside_boundary": 0,
        "unclassified_pixel_count_inside_boundary": total_unclassified_inside,
    }
