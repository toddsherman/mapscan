"""Bind a reviewed materialization to exact displayed publication boundaries."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
from PIL import Image

from .extraction import _fill_indexed_nodata_in_mask, _preview
from .review_safety import require_fresh_review_output


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text())


def _save(path: Path, values: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(values).save(path, optimize=True)
    return _sha256(path)


def _load_mainland_boundary(
    audit_path: Path,
) -> tuple[Dict[str, object], Path, Path, str | None]:
    audit = _load(audit_path)
    canonical_boundary_id = audit.get("canonical_boundary_id")
    if canonical_boundary_id is not None:
        topology = audit.get("topology", {})
        if (
            audit.get("status") != "pass"
            or topology.get("interior_component_count") != 1
            or topology.get("border_component_count") != 1
            or topology.get("interior_is_exact_fill_of_displayed_border") is not True
        ):
            raise ValueError("Canonical mainland boundary raster has not passed")
        mask_record = audit["artifacts"]["interior"]
        border_record = audit["artifacts"]["border"]
    else:
        if audit.get("status") != "pass_no_additional_warp":
            raise ValueError("Hybrid perimeter audit has not passed")
        topology = audit.get("unified_border", {})
        if (
            topology.get("passed") is not True
            or topology.get("connected_component_count") != 1
            or topology.get("interior_is_exact_fill_of_displayed_border") is not True
        ):
            raise ValueError("Hybrid perimeter is not one closed clipping boundary")
        artifacts = audit["artifacts"]
        mask_record = artifacts[
            "web-mercator-authoritative-mainland-interior-mask.png"
        ]
        border_record = artifacts[
            "web-mercator-authoritative-unified-border-overlay.png"
        ]
    mask_path = audit_path.parent / str(mask_record["path"])
    border_path = audit_path.parent / str(border_record["path"])
    if (
        _sha256(mask_path) != mask_record["sha256"]
        or _sha256(border_path) != border_record["sha256"]
    ):
        raise ValueError("Mainland boundary artifacts are missing or stale")
    return audit, mask_path, border_path, (
        str(canonical_boundary_id) if canonical_boundary_id is not None else None
    )


def clip_materialization_to_boundary(
    materialized_dir: Path,
    perimeter_audit_path: Path,
    output_dir: Path,
    *,
    component_audit_path: Path | None = None,
) -> Dict[str, object]:
    """Clip every layer artifact to a hash-bound mainland and selected islands.

    Newly exposed sub-pixel coastal slivers are completed with the same
    deterministic nearest-class rule used by the fixed-point source diff. The
    completion footprint remains a distinct evidence layer.
    """

    materialized_dir = materialized_dir.resolve()
    perimeter_audit_path = perimeter_audit_path.resolve()
    output_dir = output_dir.resolve()
    component_audit_path = (
        component_audit_path.resolve() if component_audit_path is not None else None
    )
    if output_dir == materialized_dir:
        raise ValueError("Boundary clipping requires a new output directory")
    require_fresh_review_output(output_dir, "Boundary clipping")
    manifest_path = materialized_dir / "materialization.json"
    manifest = _load(manifest_path)
    _, mask_path, border_path, canonical_boundary_id = _load_mainland_boundary(
        perimeter_audit_path
    )
    mainland_valid = np.asarray(Image.open(mask_path).convert("L")) > 0
    displayed_border = np.asarray(Image.open(border_path).convert("RGBA"))[..., 3] > 0
    valid = mainland_valid.copy()
    component_audit = None
    component_records = [
        {
            "id": "mainland",
            "role": "required_canonical_mainland",
            "authority": (
                canonical_boundary_id or "legacy_regional_hybrid_perimeter"
            ),
            "interior_pixel_count": int(np.count_nonzero(mainland_valid)),
            "border_pixel_count": int(np.count_nonzero(displayed_border)),
        }
    ]
    expected_component_count = 1
    if component_audit_path is not None:
        component_audit = _load(component_audit_path)
        if component_audit.get("status") != "pass":
            raise ValueError("Source-supported boundary-component audit has not passed")
        if component_audit.get("extraction", {}).get("sha256") != manifest.get(
            "extraction_manifest_sha256"
        ):
            raise ValueError("Boundary-component audit does not match the materialization")
        component_artifacts = component_audit.get("artifacts", {})
        island_interior_record = component_artifacts.get("island_interior", {})
        island_border_record = component_artifacts.get("island_border", {})
        island_interior_path = component_audit_path.parent / str(
            island_interior_record.get("path", "")
        )
        island_border_path = component_audit_path.parent / str(
            island_border_record.get("path", "")
        )
        if (
            not island_interior_path.is_file()
            or _sha256(island_interior_path) != island_interior_record.get("sha256")
            or not island_border_path.is_file()
            or _sha256(island_border_path) != island_border_record.get("sha256")
        ):
            raise ValueError("Source-supported island artifacts are missing or stale")
        island_valid = np.asarray(Image.open(island_interior_path).convert("L")) > 0
        island_border = (
            np.asarray(Image.open(island_border_path).convert("RGBA"))[..., 3] > 0
        )
        if island_valid.shape != mainland_valid.shape or island_border.shape != valid.shape:
            raise ValueError("Island and mainland boundary grids differ")
        if np.any(mainland_valid & island_valid):
            raise ValueError("Selected island interiors overlap the mainland")
        selected_records = [
            item
            for item in component_audit.get("islands", [])
            if isinstance(item, dict) and item.get("selected") is True
        ]
        expected_component_count += len(selected_records)
        island_interior_count = cv2.connectedComponents(
            island_valid.astype(np.uint8), 8
        )[0] - 1
        island_border_count = cv2.connectedComponents(
            island_border.astype(np.uint8), 8
        )[0] - 1
        if (
            island_interior_count != len(selected_records)
            or island_border_count != len(selected_records)
            or any(int(item.get("observed_source_pixel_count", 0)) < 1 for item in selected_records)
        ):
            raise ValueError("Selected island topology or observed evidence is invalid")
        valid |= island_valid
        displayed_border |= island_border
        component_records.extend(selected_records)
    interior_component_count = cv2.connectedComponents(valid.astype(np.uint8), 8)[0] - 1
    border_component_count = cv2.connectedComponents(
        displayed_border.astype(np.uint8), 8
    )[0] - 1
    if (
        interior_component_count != expected_component_count
        or border_component_count != expected_component_count
    ):
        raise ValueError("Publication boundary component topology changed")

    source_run = Path(str(manifest["source_run"])).resolve()
    extraction_path = source_run / "extraction.json"
    if _sha256(extraction_path) != manifest["extraction_manifest_sha256"]:
        raise ValueError("Materialization extraction manifest is stale")
    extraction = _load(extraction_path)
    plan_path = Path(str(extraction["plan"]["path"])).resolve()
    if _sha256(plan_path) != extraction["plan"]["sha256"]:
        raise ValueError("Extraction plan is stale")
    definitions = {str(item["id"]): item for item in _load(plan_path)["layers"]}

    promoted = copy.deepcopy(manifest)
    promoted["status"] = "needs_visual_review"
    promoted["warning"] = (
        "This unpublished review candidate is clipped to the exact filled "
        "interior of the displayed canonical mainland and any separately audited, "
        "source-supported island borders. Pixels "
        "outside are forced to NoData; boundary-only completion remains "
        "separately masked."
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    layer_reports = []

    for layer in promoted["layers"]:
        layer_id = str(layer["layer_id"])
        definition = definitions[layer_id]
        categories = definition["categories"]
        original_class_record = layer["artifacts"]["class_id"]
        original_class_path = materialized_dir / str(original_class_record["path"])
        if _sha256(original_class_path) != original_class_record["sha256"]:
            raise ValueError(f"Materialized class raster is stale: {layer_id}")
        original = np.asarray(Image.open(original_class_path), dtype=np.uint8)
        if original.shape != valid.shape:
            raise ValueError("Boundary mask and materialization grids differ")
        removed = (original > 0) & ~valid
        clipped = original.copy()
        clipped[~valid] = 0
        completed, _ = _fill_indexed_nodata_in_mask(clipped, valid)
        completion = valid & (clipped == 0) & (completed > 0)

        for key, artifact in list(layer["artifacts"].items()):
            relative = Path(str(artifact["path"]))
            source = materialized_dir / relative
            if _sha256(source) != artifact["sha256"]:
                raise ValueError(f"Materialized artifact is stale: {source}")
            destination = output_dir / relative
            if key == "class_id":
                values = completed.astype(np.uint8)
            elif key == "preview":
                values = _preview(completed, categories)
            else:
                values = np.asarray(Image.open(source))
                if values.shape[:2] != valid.shape:
                    raise ValueError(f"Layer artifact uses a different grid: {source}")
                values = values.copy()
                values[~valid] = 0
            artifact["sha256"] = _save(destination, values)

        completion_relative = Path(layer_id) / "web-mercator-boundary-completion-mask.png"
        removed_relative = Path(layer_id) / "web-mercator-boundary-removed-mask.png"
        layer["artifacts"]["boundary_completion_mask"] = {
            "path": str(completion_relative),
            "sha256": _save(
                output_dir / completion_relative, completion.astype(np.uint8) * 255
            ),
        }
        layer["artifacts"]["boundary_removed_mask"] = {
            "path": str(removed_relative),
            "sha256": _save(output_dir / removed_relative, removed.astype(np.uint8) * 255),
        }
        final_count = int(np.count_nonzero(completed))
        outside_after = int(np.count_nonzero((completed > 0) & ~valid))
        empty_inside = int(np.count_nonzero((completed == 0) & valid))
        layer["final_classified_pixel_count"] = final_count
        layer["final_pixels_by_class_id"] = {
            str(class_id): int(np.count_nonzero(completed == class_id))
            for class_id in range(1, len(categories) + 1)
        }
        layer["boundary_removed_pixel_count"] = int(np.count_nonzero(removed))
        layer["boundary_completion_pixel_count"] = int(np.count_nonzero(completion))
        layer["colored_pixel_count_outside_boundary"] = outside_after
        layer["unclassified_pixel_count_after"] = empty_inside
        retained_fields = {
            "observed_mask": "observed_retained_pixel_count",
            "inference_mask": "inference_retained_pixel_count",
            "manual_mask": "manual_override_pixel_count",
            "enclosed_fill_mask": "enclosed_fill_pixel_count",
        }
        for artifact_key, field in retained_fields.items():
            artifact = layer["artifacts"].get(artifact_key)
            if isinstance(artifact, dict):
                layer[field] = int(
                    np.count_nonzero(
                        np.asarray(Image.open(output_dir / str(artifact["path"])))
                    )
                )
        manual_mask_artifact = layer["artifacts"].get("manual_mask")
        manual_values_artifact = layer["artifacts"].get("manual_values")
        if isinstance(manual_mask_artifact, dict) and isinstance(
            manual_values_artifact, dict
        ):
            manual_mask = np.asarray(
                Image.open(output_dir / str(manual_mask_artifact["path"]))
            ) > 0
            manual_values = np.asarray(
                Image.open(output_dir / str(manual_values_artifact["path"]))
            )
            layer["manual_nonzero_pixel_count"] = int(
                np.count_nonzero(manual_mask & (manual_values > 0))
            )
            layer["manual_zero_pixel_count"] = int(
                np.count_nonzero(manual_mask & (manual_values == 0))
            )
        completion_artifact = layer["artifacts"].get("source_diff_completion_mask")
        if isinstance(completion_artifact, dict):
            layer["source_diff_completion_pixel_count"] = int(
                np.count_nonzero(
                    np.asarray(
                        Image.open(output_dir / str(completion_artifact["path"]))
                    )
                )
            )
        if outside_after != 0 or empty_inside != 0 or final_count != int(valid.sum()):
            raise AssertionError("Boundary clipping did not produce an exact filled interior")
        layer_reports.append(
            {
                "layer_id": layer_id,
                "classified_pixel_count_before": int(np.count_nonzero(original)),
                "mainland_interior_pixel_count": int(valid.sum()),
                "removed_outside_pixel_count": int(np.count_nonzero(removed)),
                "completed_inside_pixel_count": int(np.count_nonzero(completion)),
                "classified_pixel_count_after": final_count,
                "colored_pixel_count_outside_boundary": outside_after,
                "unclassified_pixel_count_inside_boundary": empty_inside,
                "passed": outside_after == 0 and empty_inside == 0,
            }
        )

    boundary_dir = output_dir / "boundary"
    boundary_mask_copy = boundary_dir / "web-mercator-authoritative-publication-interior-mask.png"
    boundary_border_copy = boundary_dir / "web-mercator-authoritative-publication-border-overlay.png"
    boundary_mask_hash = _save(boundary_mask_copy, valid.astype(np.uint8) * 255)
    combined_border_rgba = np.zeros((*valid.shape, 4), dtype=np.uint8)
    combined_border_rgba[displayed_border] = [80, 255, 120, 255]
    boundary_border_hash = _save(boundary_border_copy, combined_border_rgba)
    report = {
        "schema_version": 1,
        "status": "pass" if all(item["passed"] for item in layer_reports) else "fail",
        "method": "exact_displayed_border_fill_clip_with_separate_boundary_completion",
        "input_materialization": {"path": str(manifest_path), "sha256": _sha256(manifest_path)},
        "mainland_boundary_audit": {
            "path": str(perimeter_audit_path),
            "sha256": _sha256(perimeter_audit_path),
            "canonical_boundary_id": canonical_boundary_id,
        },
        "perimeter_audit": {
            "path": str(perimeter_audit_path),
            "sha256": _sha256(perimeter_audit_path),
        },
        "component_audit": (
            {"path": str(component_audit_path), "sha256": _sha256(component_audit_path)}
            if component_audit_path is not None
            else None
        ),
        "boundary": {
            "interior": {"path": str(boundary_mask_copy.relative_to(output_dir)), "sha256": boundary_mask_hash},
            "border": {"path": str(boundary_border_copy.relative_to(output_dir)), "sha256": boundary_border_hash},
            "connected_component_count": expected_component_count,
            "expected_component_count": expected_component_count,
            "mainland_interior_pixel_count": int(mainland_valid.sum()),
            "publication_interior_pixel_count": int(valid.sum()),
            "components": component_records,
            "selection_policy": (
                component_audit.get("selection_policy")
                if isinstance(component_audit, dict)
                else {
                    "qualifying_evidence": "mainland_only",
                    "manual_or_inferred_pixels_can_select_component": False,
                }
            ),
        },
        "layers": layer_reports,
        "publication_allowed": False,
        "author_review_required": True,
    }
    report_path = output_dir / "boundary-clip-audit.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    promoted["boundary_clip"] = {
        "audit": {"path": str(report_path), "sha256": _sha256(report_path)},
        "continuous_border_sha256": boundary_border_hash,
        "mainland_interior_sha256": _sha256(mask_path),
        "publication_interior_sha256": boundary_mask_hash,
        "boundary_component_count": expected_component_count,
        "expected_boundary_component_count": expected_component_count,
        "canonical_boundary_id": canonical_boundary_id,
        "components": component_records,
        "colored_pixel_count_outside_boundary": 0,
        "unclassified_pixel_count_inside_boundary": 0,
    }
    materialization_path = output_dir / "materialization.json"
    materialization_path.write_text(json.dumps(promoted, indent=2) + "\n")
    return promoted
