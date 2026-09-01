"""Promote a stable source-diff result into a self-contained review candidate."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image

from .review_safety import require_fresh_review_output


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text())


def _copy_verified(source: Path, destination: Path, expected_hash: str) -> str:
    if not source.is_file() or _sha256(source) != expected_hash:
        raise ValueError(f"Source-diff materialization input is missing or stale: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    copied_hash = _sha256(destination)
    if copied_hash != expected_hash:
        raise AssertionError("Copied source-diff artifact hash changed")
    return copied_hash


def _first_verified_path(candidates: list[Path], expected_hash: str) -> Path:
    """Resolve one hash-bound artifact across current and legacy batch layouts."""

    checked: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in checked:
            continue
        checked.add(resolved)
        if resolved.is_file() and _sha256(resolved) == expected_hash:
            return resolved
    rendered = ", ".join(str(path) for path in checked)
    raise ValueError(
        "Source-diff materialization input is missing or stale; checked: "
        f"{rendered}"
    )


def promote_source_diff_materialization(
    materialized_dir: Path,
    source_diff_batch_dir: Path,
    case_id: str,
    output_dir: Path,
) -> Dict[str, object]:
    """Replace a materialization's class surface with its stable audited result."""

    materialized_dir = materialized_dir.resolve()
    source_diff_batch_dir = source_diff_batch_dir.resolve()
    output_dir = output_dir.resolve()
    require_fresh_review_output(output_dir, "Source-diff promotion")
    manifest_path = materialized_dir / "materialization.json"
    batch_path = source_diff_batch_dir / "source-diff-batch.json"
    manifest = _load(manifest_path)
    batch = _load(batch_path)
    matching = [item for item in batch.get("cases", []) if item.get("id") == case_id]
    if batch.get("status") != "pass" or len(matching) != 1:
        raise ValueError("Source-diff batch or requested case has not passed")
    case = matching[0]
    if case.get("status") != "pass" or case.get("fixed_point_reached") is not True:
        raise ValueError("Source-diff case has not reached a passing fixed point")
    case_dir = source_diff_batch_dir / case_id
    final_report_path = source_diff_batch_dir / str(case["report"])
    final_report = _load(final_report_path)
    if final_report.get("manifest_sha256") != manifest.get(
        "extraction_manifest_sha256"
    ):
        raise ValueError("Source-diff report does not match the materialized extraction")
    iterations = case.get("comparison_iterations", [])
    if len(iterations) < 2:
        raise ValueError("Source-diff promotion requires repeated comparison iterations")
    first_report_path = source_diff_batch_dir / str(iterations[0]["report"])
    first_report = _load(first_report_path)
    final_layers = {str(item["id"]): item for item in final_report["layers"]}
    first_layers = {str(item["id"]): item for item in first_report["layers"]}

    promoted = copy.deepcopy(manifest)
    promoted["status"] = "needs_visual_review"
    promoted["source_diff"] = {
        "batch": {"path": str(batch_path), "sha256": _sha256(batch_path)},
        "case_id": case_id,
        "report": {
            "path": str(final_report_path),
            "sha256": _sha256(final_report_path),
        },
        "fixed_point_reached": True,
        "comparison_iterations": iterations,
    }
    promoted["warning"] = (
        "This review candidate includes deterministic full-state source-diff "
        "completion. Completion remains separately masked and does not become "
        "observed or manually authored evidence."
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    for layer in promoted["layers"]:
        layer_id = str(layer["layer_id"])
        final_layer = final_layers[layer_id]
        first_layer = first_layers[layer_id]
        for key, artifact in list(layer["artifacts"].items()):
            if key in {"class_id", "preview"}:
                continue
            relative = Path(str(artifact["path"]))
            _copy_verified(
                materialized_dir / relative,
                output_dir / relative,
                str(artifact["sha256"]),
            )

        replacements = {
            "class_id": final_layer["artifacts"]["audited_class_id"],
            "preview": final_layer["artifacts"]["audited_preview"],
        }
        for key, source_artifact in replacements.items():
            destination_relative = Path(str(layer["artifacts"][key]["path"]))
            artifact_relative = Path(str(source_artifact["path"]))
            source_candidates = [case_dir / artifact_relative]
            candidate_record = final_layer.get("candidate")
            if isinstance(candidate_record, dict) and candidate_record.get("path"):
                candidate_path = Path(str(candidate_record["path"]))
                if not candidate_path.is_absolute():
                    candidate_path = Path.cwd() / candidate_path
                source_candidates.append(
                    candidate_path
                    if key == "class_id"
                    else candidate_path.parent / artifact_relative.name
                )
            # Older source-diff batches kept all stable artifacts in their
            # iteration directories instead of copying them to the case root.
            # Every candidate is still accepted only when its recorded hash
            # matches, so layout compatibility does not weaken provenance.
            for iteration in reversed(iterations):
                iteration_report = source_diff_batch_dir / str(iteration["report"])
                source_candidates.append(iteration_report.parent / artifact_relative)
            source_path = _first_verified_path(
                source_candidates, str(source_artifact["sha256"])
            )
            copied_hash = _copy_verified(
                source_path,
                output_dir / destination_relative,
                str(source_artifact["sha256"]),
            )
            layer["artifacts"][key] = {
                "path": str(destination_relative),
                "sha256": copied_hash,
            }

        completion_artifact = first_layer["artifacts"]["completion"]
        completion_relative = Path(layer_id) / "web-mercator-source-diff-completion-mask.png"
        completion_hash = _copy_verified(
            first_report_path.parent / str(completion_artifact["path"]),
            output_dir / completion_relative,
            str(completion_artifact["sha256"]),
        )
        layer["artifacts"]["source_diff_completion_mask"] = {
            "path": str(completion_relative),
            "sha256": completion_hash,
        }
        final_values = np.asarray(
            Image.open(output_dir / layer["artifacts"]["class_id"]["path"]),
            dtype=np.uint8,
        )
        layer["final_classified_pixel_count"] = int(np.count_nonzero(final_values))
        layer["final_pixels_by_class_id"] = {
            str(class_id): int(np.count_nonzero(final_values == class_id))
            for class_id in range(1, int(final_values.max(initial=0)) + 1)
        }
        layer["source_diff_completion_pixel_count"] = int(
            first_layer["completed_pixel_count"]
        )
        layer["dropped_visible_source_evidence_pixel_count"] = int(
            final_layer["dropped_visible_source_evidence_pixel_count"]
        )
        layer["unclassified_pixel_count_after"] = int(
            final_layer["unclassified_pixel_count_after"]
        )

    output_manifest_path = output_dir / "materialization.json"
    output_manifest_path.write_text(json.dumps(promoted, indent=2) + "\n")
    return promoted
