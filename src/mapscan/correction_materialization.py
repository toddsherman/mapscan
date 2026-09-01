"""Materialize reviewed inference and manual corrections into a final candidate."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from PIL import Image

from .enclosed_fill import fill_small_enclosed_holes
from .manual_stamp import apply_clone_stamp_operations, apply_inference_exclusions


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _latest_inference_root(run_dir: Path) -> Path:
    candidates = [
        path
        for path in run_dir.glob("inference*")
        if path.is_dir() and (path / "inference.json").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"No inference artifact exists under {run_dir}")

    def version(path: Path) -> tuple[int, int]:
        match = re.fullmatch(r"inference-v(\d+)", path.name)
        return (1, int(match.group(1))) if match else (0, 0)

    return max(candidates, key=version)


def _display_rgb(category: Dict[str, object]) -> list[int]:
    color = category.get("display_rgb", category.get("legend_rgb", [255, 0, 255]))
    if isinstance(color, list) and color and isinstance(color[0], list):
        color = color[0]
    if not isinstance(color, list) or len(color) < 3:
        return [255, 0, 255]
    return [int(color[0]), int(color[1]), int(color[2])]


def _preview(values: np.ndarray, categories: list[Dict[str, object]]) -> np.ndarray:
    palette = np.zeros((256, 4), dtype=np.uint8)
    for class_id, category in enumerate(categories, 1):
        palette[class_id, :3] = _display_rgb(category)
        palette[class_id, 3] = 255
    return palette[values]


def _load_optional_json(path: Path) -> Optional[Dict[str, object]]:
    return json.loads(path.read_text()) if path.exists() else None


def materialize_review_corrections(
    run_dir: Path,
    output_dir: Path,
    inference_dir: Optional[Path] = None,
    include_inference: Optional[bool] = None,
) -> Dict[str, object]:
    """Apply audited corrections with manual override as the highest precedence."""

    run_dir = run_dir.resolve()
    output_dir = output_dir.resolve()
    manifest_path = run_dir / "extraction.json"
    plan_path = run_dir / "plan.snapshot.json"
    stamp_path = run_dir / "stamp-corrections.json"
    exclusion_path = run_dir / "inference-exclusions.json"
    for required in (manifest_path, plan_path, stamp_path):
        if not required.exists():
            raise FileNotFoundError(f"Missing correction input: {required}")

    manifest = json.loads(manifest_path.read_text())
    plan = json.loads(plan_path.read_text())
    stamps = json.loads(stamp_path.read_text())
    selection = _load_optional_json(run_dir / "inference-selection.json")
    enclosed_selection = _load_optional_json(
        run_dir / "enclosed-hole-fill-selection.json"
    )
    include_enclosed_fill = bool(
        isinstance(enclosed_selection, dict)
        and enclosed_selection.get("enabled") is True
    )
    enclosed_maximum_area = int(
        enclosed_selection.get("maximum_area_exclusive", 50)
        if isinstance(enclosed_selection, dict)
        else 50
    )
    if include_inference is None:
        include_inference = not (
            isinstance(selection, dict) and selection.get("enabled") is False
        )
    exclusions = _load_optional_json(exclusion_path) if include_inference else None
    extraction_hash = _sha256(manifest_path)
    inference_root = None
    inference_manifest_path = None
    inference_hash = None
    if include_inference:
        inference_root = (
            inference_dir.resolve()
            if inference_dir is not None
            else _latest_inference_root(run_dir)
        )
        inference_manifest_path = inference_root / "inference.json"
        if not inference_manifest_path.exists():
            raise FileNotFoundError(
                f"Missing inference manifest: {inference_manifest_path}"
            )
        inference_hash = _sha256(inference_manifest_path)
    if int(stamps.get("schema_version", 0)) not in {2, 3}:
        raise ValueError(
            "Solid manual overrides require stamp correction schema version 2 or 3"
        )
    if stamps.get("extraction_manifest_sha256") != extraction_hash:
        raise ValueError("Stamp corrections do not match the extraction manifest")
    if include_inference and stamps.get("inference_manifest_sha256") != inference_hash:
        raise ValueError("Stamp corrections do not match the selected inference artifact")
    if exclusions is not None:
        if exclusions.get("extraction_manifest_sha256") != extraction_hash:
            raise ValueError("Inference exclusions do not match the extraction manifest")
        if exclusions.get("inference_manifest_sha256") != inference_hash:
            raise ValueError("Inference exclusions do not match the selected inference artifact")

    plan_layers = {str(item["id"]): item for item in plan.get("layers", [])}
    stamp_operations = stamps.get("operations", [])
    exclusion_operations = exclusions.get("operations", []) if exclusions else []
    output_dir.mkdir(parents=True, exist_ok=True)
    reports = []

    for layer_report in manifest.get("layers", []):
        layer_id = str(layer_report["id"])
        definition = plan_layers.get(layer_id)
        if not isinstance(definition, dict):
            continue
        observed_path = run_dir / layer_id / "web-mercator-class-id.png"
        if not observed_path.exists():
            continue

        observed = np.asarray(Image.open(observed_path), dtype=np.uint8)
        if include_inference:
            assert inference_root is not None
            inferred_path = (
                inference_root / layer_id / "web-mercator-class-id-inferred.png"
            )
            inference_mask_path = (
                inference_root / layer_id / "web-mercator-inference-mask.png"
            )
            if not inferred_path.exists() or not inference_mask_path.exists():
                continue
            inferred = np.asarray(Image.open(inferred_path), dtype=np.uint8)
            inference_mask = np.asarray(Image.open(inference_mask_path)) > 0
            if inferred.shape != observed.shape or inference_mask.shape != observed.shape:
                raise ValueError(
                    f"Correction rasters have mismatched shapes for {layer_id}"
                )
        else:
            inferred = np.zeros_like(observed)
            inference_mask = np.zeros_like(observed, dtype=bool)

        layer_stamps = [
            item for item in stamp_operations if item.get("layer_id") == layer_id
        ]
        layer_exclusions = [
            item for item in exclusion_operations if item.get("layer_id") == layer_id
        ]
        manual_values, manual_mask = apply_clone_stamp_operations(
            observed, layer_stamps
        )
        exclusion_mask = apply_inference_exclusions(
            inference_mask, layer_exclusions
        )
        retained_inference = inference_mask & ~exclusion_mask & ~manual_mask
        retained_observed = (observed > 0) & ~manual_mask
        final = observed.copy()
        final[retained_inference] = inferred[retained_inference]
        final[manual_mask] = manual_values[manual_mask]
        enclosed_values = np.zeros_like(observed)
        enclosed_mask = np.zeros_like(observed, dtype=bool)
        enclosed_report: Dict[str, object] = {
            "filled_component_count": 0,
            "filled_pixel_count": 0,
        }
        if include_enclosed_fill:
            enclosed_values, enclosed_mask, enclosed_report = (
                fill_small_enclosed_holes(
                    final,
                    maximum_area_exclusive=enclosed_maximum_area,
                    protected_zero_mask=manual_mask & (manual_values == 0),
                )
            )
            final[enclosed_mask] = enclosed_values[enclosed_mask]

        layer_dir = output_dir / layer_id
        layer_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            "class_id": layer_dir / "web-mercator-class-id-final.png",
            "preview": layer_dir / "web-mercator-preview-final.png",
            "observed_mask": layer_dir / "web-mercator-observed-retained-mask.png",
            "inference_mask": layer_dir / "web-mercator-inference-retained-mask.png",
            "manual_mask": layer_dir / "web-mercator-manual-override-mask.png",
            "manual_values": layer_dir / "web-mercator-manual-values.png",
        }
        if include_enclosed_fill:
            paths["enclosed_fill_mask"] = (
                layer_dir / "web-mercator-enclosed-fill-mask.png"
            )
            paths["enclosed_fill_values"] = (
                layer_dir / "web-mercator-enclosed-fill-values.png"
            )
        Image.fromarray(final).save(paths["class_id"], optimize=True)
        Image.fromarray(_preview(final, definition.get("categories", []))).save(
            paths["preview"], optimize=True
        )
        for key, mask in (
            ("observed_mask", retained_observed),
            ("inference_mask", retained_inference),
            ("manual_mask", manual_mask),
        ):
            Image.fromarray(mask.astype(np.uint8) * 255).save(
                paths[key], optimize=True
            )
        Image.fromarray(manual_values).save(
            paths["manual_values"], optimize=True
        )
        if include_enclosed_fill:
            Image.fromarray(enclosed_mask.astype(np.uint8) * 255).save(
                paths["enclosed_fill_mask"], optimize=True
            )
            Image.fromarray(enclosed_values).save(
                paths["enclosed_fill_values"], optimize=True
            )

        class_count = len(definition.get("categories", []))
        reports.append(
            {
                "layer_id": layer_id,
                "width": int(observed.shape[1]),
                "height": int(observed.shape[0]),
                "stamp_operation_count": len(layer_stamps),
                "exclusion_operation_count": len(layer_exclusions),
                "observed_input_pixel_count": int(np.count_nonzero(observed)),
                "observed_retained_pixel_count": int(np.count_nonzero(retained_observed)),
                "inference_input_pixel_count": int(np.count_nonzero(inference_mask)),
                "inference_excluded_pixel_count": int(np.count_nonzero(exclusion_mask)),
                "inference_retained_pixel_count": int(np.count_nonzero(retained_inference)),
                "manual_override_pixel_count": int(np.count_nonzero(manual_mask)),
                "manual_nonzero_pixel_count": int(
                    np.count_nonzero(manual_mask & (manual_values > 0))
                ),
                "manual_zero_pixel_count": int(
                    np.count_nonzero(manual_mask & (manual_values == 0))
                ),
                "observed_changed_by_manual_pixel_count": int(
                    np.count_nonzero(manual_mask & (observed != manual_values))
                ),
                "enclosed_fill_component_count": int(
                    enclosed_report["filled_component_count"]
                ),
                "enclosed_fill_pixel_count": int(
                    enclosed_report["filled_pixel_count"]
                ),
                "final_classified_pixel_count": int(np.count_nonzero(final)),
                "final_pixels_by_class_id": {
                    str(class_id): int(np.count_nonzero(final == class_id))
                    for class_id in range(1, class_count + 1)
                },
                "artifacts": {
                    key: {
                        "path": str(path.relative_to(output_dir)),
                        "sha256": _sha256(path),
                    }
                    for key, path in paths.items()
                },
            }
        )

    if not reports:
        raise ValueError("No compatible categorical layers were materialized")
    result = {
        "schema_version": 1,
        "status": "needs_visual_review",
        "dataset_id": manifest["dataset_id"],
        "source_run": str(run_dir),
        "extraction_manifest_sha256": extraction_hash,
        "inference": (
            {
                "path": str(inference_manifest_path),
                "sha256": inference_hash,
            }
            if include_inference
            else None
        ),
        "stamp_corrections": {
            "path": str(stamp_path),
            "sha256": _sha256(stamp_path),
            "schema_version": stamps["schema_version"],
        },
        "inference_exclusions": (
            {"path": str(exclusion_path), "sha256": _sha256(exclusion_path)}
            if exclusions is not None
            else None
        ),
        "enclosed_hole_fill": (
            {
                "selection_path": str(
                    run_dir / "enclosed-hole-fill-selection.json"
                ),
                "maximum_area_exclusive": enclosed_maximum_area,
                "connectivity": 8,
                "boundary": "exactly_one_nonzero_class",
                "manual_zero_pixels": "never_fill",
            }
            if include_enclosed_fill
            else None
        ),
        "precedence": [
            "observed_classification",
            *(
                ["retained_automatic_inference"]
                if include_inference
                else []
            ),
            *(
                ["small_enclosed_zero_fill"]
                if include_enclosed_fill
                else []
            ),
            "manual_override_patch",
        ],
        "warning": (
            "This is a materialized review candidate. Manual overrides are inferred "
            "author edits, not observed source pixels, and remain separately masked."
        ),
        "layers": reports,
    }
    (output_dir / "materialization.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result
