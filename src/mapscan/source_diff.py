"""Source-to-output diff gates for extraction review candidates."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Dict, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

from .extraction import (
    _fill_indexed_nodata_in_mask,
    _preview,
    _state_mask_in_source,
    _target_state_mask,
    warp_classified_to_web_mercator,
)
from .reference import load_california
from .continuous_extraction import audit_continuous_source_diff
from .pdf_vector_extraction import audit_pdf_vector_diff


FULL_STATE = "full_state"
SPARSE_EVIDENCE = "sparse_visible_evidence"
VALID_COVERAGE_EXPECTATIONS = {FULL_STATE, SPARSE_EVIDENCE}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_mask(path: Path, mask: np.ndarray) -> None:
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(path, optimize=True)


def _promote_final_extraction_artifacts(
    report: Dict[str, object], iteration_output: Path, case_output: Path
) -> None:
    """Make a fixed-point case report self-contained outside iteration folders."""

    for layer in report.get("layers", []):
        for artifact in layer.get("artifacts", {}).values():
            relative = Path(str(artifact["path"]))
            source = iteration_output / relative
            destination = case_output / relative
            if not source.is_file() or _sha256(source) != artifact["sha256"]:
                raise ValueError("Fixed-point extraction artifact is missing or stale")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            if _sha256(destination) != artifact["sha256"]:
                raise AssertionError("Promoted fixed-point artifact hash changed")


def _candidate_layer_path(
    run_dir: Path, candidate_dir: Path | None, layer_id: str
) -> Path:
    base = candidate_dir if candidate_dir is not None else run_dir
    choices = (
        base / layer_id / "web-mercator-class-id-final.png",
        base / layer_id / "web-mercator-class-id-audited.png",
        base / layer_id / "web-mercator-class-id.png",
    )
    for path in choices:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"No Web-Mercator class raster found for {layer_id} under {base}"
    )


def _audit_layer_arrays(
    candidate: np.ndarray,
    target_state_mask: np.ndarray,
    visible_source_evidence: np.ndarray,
    coverage_expectation: str,
    repair_full_state: bool,
) -> Tuple[np.ndarray, Dict[str, object], Dict[str, np.ndarray]]:
    """Diff one candidate against state coverage and source-derived evidence."""

    if coverage_expectation not in VALID_COVERAGE_EXPECTATIONS:
        raise ValueError(
            f"coverage_expectation must be one of {sorted(VALID_COVERAGE_EXPECTATIONS)}"
        )
    if candidate.shape != target_state_mask.shape:
        raise ValueError("Candidate and target state mask shapes differ")
    if visible_source_evidence.shape != candidate.shape:
        raise ValueError("Visible source evidence and candidate shapes differ")

    candidate = candidate.astype(np.uint8, copy=False)
    target_state_mask = target_state_mask.astype(bool, copy=False)
    visible_source_evidence = visible_source_evidence.astype(bool, copy=False)
    unclassified_before = target_state_mask & (candidate == 0)
    dropped_evidence = target_state_mask & visible_source_evidence & (candidate == 0)
    repaired = candidate.copy()
    completion_mask = np.zeros(candidate.shape, dtype=bool)
    if coverage_expectation == FULL_STATE and repair_full_state:
        repaired, _ = _fill_indexed_nodata_in_mask(repaired, target_state_mask)
        completion_mask = target_state_mask & (candidate == 0) & (repaired > 0)
    unclassified_after = target_state_mask & (repaired == 0)

    state_pixels = int(np.count_nonzero(target_state_mask))
    before_count = int(np.count_nonzero(unclassified_before))
    after_count = int(np.count_nonzero(unclassified_after))
    dropped_count = int(np.count_nonzero(dropped_evidence))
    if coverage_expectation == FULL_STATE:
        passed = after_count == 0
        gate = "no internal NoData after deterministic completion"
    else:
        passed = dropped_count == 0
        gate = "no visible source-derived evidence dropped"
    report = {
        "coverage_expectation": coverage_expectation,
        "gate": gate,
        "status": "pass" if passed else "fail",
        "target_state_pixel_count": state_pixels,
        "classified_pixel_count_before": state_pixels - before_count,
        "unclassified_pixel_count_before": before_count,
        "unclassified_fraction_before": before_count / max(state_pixels, 1),
        "visible_source_evidence_pixel_count": int(
            np.count_nonzero(target_state_mask & visible_source_evidence)
        ),
        "dropped_visible_source_evidence_pixel_count": dropped_count,
        "repair_full_state_enabled": bool(repair_full_state),
        "completed_pixel_count": int(np.count_nonzero(completion_mask)),
        "unclassified_pixel_count_after": after_count,
        "unclassified_fraction_after": after_count / max(state_pixels, 1),
    }
    masks = {
        "unclassified_before": unclassified_before,
        "dropped_visible_source_evidence": dropped_evidence,
        "completion": completion_mask,
        "unclassified_after": unclassified_after,
    }
    return repaired, report, masks


def audit_extraction_source_diff(
    run_dir: Path,
    output_dir: Path,
    candidate_dir: Path | None = None,
    repair_full_state: bool = True,
) -> Dict[str, object]:
    """Audit every extraction layer using an explicit coverage expectation."""

    manifest_path = run_dir / "extraction.json"
    manifest = json.loads(manifest_path.read_text())
    plan_path = Path(manifest["plan"]["path"])
    plan = json.loads(plan_path.read_text())
    definitions = {str(layer["id"]): layer for layer in plan["layers"]}
    reference_root = Path(plan.get("reference", "reference/census-2025"))
    state, _ = load_california(reference_root)
    source_state = np.asarray(Image.open(run_dir / "source-state-mask.png")) > 0
    output_dir.mkdir(parents=True, exist_ok=True)
    layer_reports = []

    for layer_manifest in manifest["layers"]:
        layer_id = str(layer_manifest["id"])
        definition = definitions[layer_id]
        coverage = str(definition.get("coverage_expectation", ""))
        if coverage not in VALID_COVERAGE_EXPECTATIONS:
            raise ValueError(
                f"Layer {layer_id} requires an explicit coverage_expectation"
            )
        layer_output = output_dir / layer_id
        layer_output.mkdir(parents=True, exist_ok=True)
        source_class_path = run_dir / layer_id / "source-class-id.png"
        baseline_path = run_dir / layer_id / "web-mercator-class-id.png"
        candidate_path = _candidate_layer_path(run_dir, candidate_dir, layer_id)
        source_class = np.asarray(Image.open(source_class_path))
        baseline = np.asarray(Image.open(baseline_path))
        candidate = np.asarray(Image.open(candidate_path))
        canonical_clip = layer_manifest.get("canonical_clip")
        if isinstance(canonical_clip, dict):
            interior_record = canonical_clip.get("artifacts", {}).get("interior", {})
            interior_path = run_dir / str(interior_record.get("path", ""))
            if (
                not interior_path.is_file()
                or _sha256(interior_path) != interior_record.get("sha256")
            ):
                raise ValueError(
                    f"Layer {layer_id} canonical publication interior is stale"
                )
            target_state = np.asarray(Image.open(interior_path).convert("L")) > 0
            if target_state.shape != candidate.shape:
                raise ValueError(
                    f"Layer {layer_id} canonical publication interior uses another grid"
                )
            target_mask_kind = "canonical_publication_interior"
        else:
            target_state = _target_state_mask(
                state, layer_manifest["warp"]["bounds"], candidate.shape
            )
            target_mask_kind = "legacy_census_state_mask"
        repaired, report, masks = _audit_layer_arrays(
            candidate,
            target_state,
            baseline > 0,
            coverage,
            repair_full_state,
        )

        source_unclassified = source_state & (source_class == 0)
        report.update(
            {
                "id": layer_id,
                "kind": str(layer_manifest["kind"]),
                "target_mask_kind": target_mask_kind,
                "source_state_pixel_count": int(np.count_nonzero(source_state)),
                "source_unclassified_pixel_count": int(
                    np.count_nonzero(source_unclassified)
                ),
                "source_unclassified_fraction": float(
                    np.count_nonzero(source_unclassified)
                    / max(np.count_nonzero(source_state), 1)
                ),
                "candidate": {
                    "path": str(candidate_path),
                    "sha256": _sha256(candidate_path),
                },
                "baseline_visible_evidence": {
                    "path": str(baseline_path),
                    "sha256": _sha256(baseline_path),
                },
            }
        )
        audited_path = layer_output / "web-mercator-class-id-audited.png"
        preview_path = layer_output / "web-mercator-preview-audited.png"
        Image.fromarray(repaired, mode="L").save(audited_path, optimize=True)
        Image.fromarray(_preview(repaired, definition["categories"]), mode="RGBA").save(
            preview_path, optimize=True
        )
        artifact_paths = {
            "unclassified_before": layer_output
            / "web-mercator-unclassified-before-mask.png",
            "dropped_visible_source_evidence": layer_output
            / "web-mercator-dropped-source-evidence-mask.png",
            "completion": layer_output / "web-mercator-diff-completion-mask.png",
            "unclassified_after": layer_output
            / "web-mercator-unclassified-after-mask.png",
        }
        for key, path in artifact_paths.items():
            _save_mask(path, masks[key])
        report["artifacts"] = {
            "audited_class_id": {
                "path": str(audited_path.relative_to(output_dir)),
                "sha256": _sha256(audited_path),
            },
            "audited_preview": {
                "path": str(preview_path.relative_to(output_dir)),
                "sha256": _sha256(preview_path),
            },
            **{
                key: {
                    "path": str(path.relative_to(output_dir)),
                    "sha256": _sha256(path),
                }
                for key, path in artifact_paths.items()
            },
        }
        layer_reports.append(report)

    result = {
        "schema_version": 1,
        "audit_kind": "classified_source_diff",
        "status": (
            "pass" if all(item["status"] == "pass" for item in layer_reports) else "fail"
        ),
        "dataset_id": manifest["dataset_id"],
        "run": str(run_dir),
        "candidate": str(candidate_dir) if candidate_dir is not None else None,
        "manifest_sha256": _sha256(manifest_path),
        "plan": str(plan_path),
        "plan_sha256": _sha256(plan_path),
        "layers": layer_reports,
    }
    (output_dir / "source-diff-audit.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


def _alignment_transform(alignment: Dict[str, object]) -> Dict[str, object]:
    if alignment.get("alignment_mode") == "assisted":
        transform = {
            "projection": "assisted_reference_crs",
            "projection_crs": alignment["reference"]["crs"],
            "transform_model": alignment["transform_model"],
            "reference_to_source_matrix": alignment["reference_to_source_matrix"],
        }
    else:
        transform = dict(alignment["best"])
    if "web_mercator_correction" in alignment:
        transform["web_mercator_correction"] = alignment["web_mercator_correction"]
    return transform


def audit_feature_source_diff(run_dir: Path, output_dir: Path) -> Dict[str, object]:
    """Recompute the declared feature-ink gate and diff both stored masks."""

    manifest_path = run_dir / "feature-extraction.json"
    manifest = json.loads(manifest_path.read_text())
    plan_path = Path(manifest["plan"])
    plan = json.loads(plan_path.read_text())
    source_path = Path(plan["source"])
    alignment = json.loads(Path(plan["alignment"]).read_text())
    transform = _alignment_transform(alignment)
    reference_root = Path(plan.get("reference", "reference/census-2025"))
    state, _ = load_california(reference_root)
    rgb = np.asarray(Image.open(source_path).convert("RGB"))
    source_state = _state_mask_in_source(
        state, str(transform["projection_crs"]), transform, rgb.shape[:2]
    )
    gate = plan["observed_ink"]["initial_hsv_gate_opencv"]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    expected_source = (
        (hsv[:, :, 0] >= int(gate["hue_min"]))
        & (hsv[:, :, 0] <= int(gate["hue_max"]))
        & (hsv[:, :, 1] >= int(gate["saturation_min"]))
        & (hsv[:, :, 2] >= int(gate["value_min"]))
        & source_state
    )
    stored_source_path = run_dir / "source-observed-ink.png"
    stored_source = np.asarray(Image.open(stored_source_path)) > 0
    expected_web, _ = warp_classified_to_web_mercator(
        expected_source.astype(np.uint8), state, transform, rgb.shape[:2]
    )
    expected_web = expected_web > 0
    stored_web_path = run_dir / "web-mercator-observed-ink.png"
    stored_web = np.asarray(Image.open(stored_web_path)) > 0
    source_missing = expected_source & ~stored_source
    source_extra = stored_source & ~expected_source
    web_missing = expected_web & ~stored_web
    web_extra = stored_web & ~expected_web
    passed = not any(
        np.any(mask) for mask in (source_missing, source_extra, web_missing, web_extra)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    masks = {
        "source_missing": source_missing,
        "source_extra": source_extra,
        "web_missing": web_missing,
        "web_extra": web_extra,
    }
    artifacts = {}
    for name, mask in masks.items():
        path = output_dir / f"{name.replace('_', '-')}-mask.png"
        _save_mask(path, mask)
        artifacts[name] = {"path": path.name, "sha256": _sha256(path)}
    result = {
        "schema_version": 1,
        "audit_kind": "feature_source_diff",
        "status": "pass" if passed else "fail",
        "dataset_id": manifest["dataset_id"],
        "run": str(run_dir),
        "source_expected_pixel_count": int(np.count_nonzero(expected_source)),
        "source_stored_pixel_count": int(np.count_nonzero(stored_source)),
        "source_missing_pixel_count": int(np.count_nonzero(source_missing)),
        "source_extra_pixel_count": int(np.count_nonzero(source_extra)),
        "web_expected_pixel_count": int(np.count_nonzero(expected_web)),
        "web_stored_pixel_count": int(np.count_nonzero(stored_web)),
        "web_missing_pixel_count": int(np.count_nonzero(web_missing)),
        "web_extra_pixel_count": int(np.count_nonzero(web_extra)),
        "artifacts": artifacts,
        "semantic_limit": (
            "This proves lossless retention of declared blue-ink evidence only; "
            "text-versus-geometry interpretation remains a separate review gate."
        ),
    }
    (output_dir / "source-diff-audit.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


def audit_source_diff_batch(config_path: Path, output_dir: Path) -> Dict[str, object]:
    """Run source-diff audits to a clean, repeatable fixed point."""

    config = json.loads(config_path.read_text())
    output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for item in config["cases"]:
        case_output = output_dir / str(item["id"])
        iteration_reports = []
        previous_signature = None
        stable = False
        candidate_dir = Path(item["candidate"]) if item.get("candidate") else None
        maximum_iterations = int(item.get("maximum_iterations", 4))
        report = None
        for iteration_index in range(maximum_iterations):
            iteration_output = case_output / f"iteration-{iteration_index + 1:02d}"
            if item["kind"] == "extraction":
                report = audit_extraction_source_diff(
                    Path(item["run"]),
                    iteration_output,
                    candidate_dir=candidate_dir,
                    repair_full_state=bool(item.get("repair_full_state", True)),
                )
                signature = tuple(
                    (
                        layer["id"],
                        layer["status"],
                        layer["dropped_visible_source_evidence_pixel_count"],
                        layer["unclassified_pixel_count_after"],
                        layer["artifacts"]["audited_class_id"]["sha256"],
                    )
                    for layer in report["layers"]
                )
                # The next comparison consumes the just-rendered audited
                # candidate, not the original, so deterministic completion is
                # independently checked rather than merely assumed.
                candidate_dir = iteration_output
            elif item["kind"] == "feature":
                report = audit_feature_source_diff(Path(item["run"]), iteration_output)
                signature = (
                    report["status"],
                    report["source_missing_pixel_count"],
                    report["source_extra_pixel_count"],
                    report["web_missing_pixel_count"],
                    report["web_extra_pixel_count"],
                    tuple(
                        artifact["sha256"]
                        for artifact in report["artifacts"].values()
                    ),
                )
            elif item["kind"] == "continuous":
                report = audit_continuous_source_diff(
                    Path(item["run"]), iteration_output
                )
                signature = (
                    report["status"],
                    report["source_different_pixel_count"],
                    report["web_different_pixel_count"],
                    tuple(
                        artifact["sha256"]
                        for artifact in report["artifacts"].values()
                    ),
                )
            elif item["kind"] == "pdf_vector":
                report = audit_pdf_vector_diff(Path(item["run"]), iteration_output)
                signature = (
                    report["status"],
                    report["rendered_visible_fill_pixel_count"],
                    report["dropped_rendered_visible_fill_pixel_count"],
                    report["artifact"]["sha256"],
                )
            else:
                raise ValueError(f"Unsupported source-diff case kind: {item['kind']}")
            iteration_reports.append(
                {
                    "iteration": iteration_index + 1,
                    "status": report["status"],
                    "signature_sha256": hashlib.sha256(
                        repr(signature).encode("utf-8")
                    ).hexdigest(),
                    "report": str(
                        (iteration_output / "source-diff-audit.json").relative_to(
                            output_dir
                        )
                    ),
                }
            )
            if report["status"] == "pass" and signature == previous_signature:
                stable = True
                break
            previous_signature = signature
        assert report is not None
        final_report_path = case_output / "source-diff-audit.json"
        if item["kind"] == "extraction":
            _promote_final_extraction_artifacts(report, iteration_output, case_output)
        final_report_path.write_text(json.dumps(report, indent=2) + "\n")
        reports.append(
            {
                "id": item["id"],
                "kind": item["kind"],
                "status": "pass" if report["status"] == "pass" and stable else "fail",
                "fixed_point_reached": stable,
                "comparison_iterations": iteration_reports,
                "report": str(final_report_path.relative_to(output_dir)),
            }
        )
    result = {
        "schema_version": 1,
        "audit_kind": "source_diff_batch",
        "status": "pass" if all(item["status"] == "pass" for item in reports) else "fail",
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "cases": reports,
        "pending_semantic_extraction": config.get("pending_semantic_extraction", []),
        "pending_semantic_validation": config.get("pending_semantic_validation", []),
    }
    (output_dir / "source-diff-batch.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result
