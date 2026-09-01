"""Audit categorical color extraction under conservative threshold perturbations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.distance import cdist

from .extraction import _source_context_exclusion, classify_categorical


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> Dict[str, object]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _resolve_file(value: object, root: Path) -> Path:
    path = Path(str(value))
    candidates = [path] if path.is_absolute() else [Path.cwd() / path, root / path]
    resolved = next((item.resolve() for item in candidates if item.is_file()), None)
    if resolved is None:
        raise FileNotFoundError(path)
    return resolved


def _save_mask(path: Path, mask: np.ndarray) -> Dict[str, object]:
    Image.fromarray(mask.astype(np.uint8) * 255).save(path, optimize=True)
    return {"path": path.name, "sha256": _sha256(path)}


def _category_preview(values: np.ndarray, categories: list[dict]) -> np.ndarray:
    result = np.full((*values.shape, 3), 20, dtype=np.uint8)
    for class_id, category in enumerate(categories, 1):
        raw_color = category.get("display_rgb", category.get("legend_rgb"))
        color = raw_color[0] if isinstance(raw_color[0], list) else raw_color
        result[values == class_id] = np.asarray(color[:3], dtype=np.uint8)
    return result


def _component_summary(mask: np.ndarray) -> Dict[str, object]:
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    areas = stats[1:, cv2.CC_STAT_AREA] if count > 1 else np.asarray([], dtype=int)
    return {
        "component_count": int(len(areas)),
        "single_pixel_component_count": int(np.count_nonzero(areas == 1)),
        "component_count_at_most_3px": int(np.count_nonzero(areas <= 3)),
        "maximum_component_area": int(areas.max()) if len(areas) else 0,
    }


def audit_categorical_fidelity(
    run_dir: Path,
    output_dir: Path,
    *,
    distance_perturbation: float = 4.0,
    margin_perturbation: float = 1.0,
) -> Dict[str, object]:
    """Recompute a reviewed categorical extraction across a threshold ensemble."""

    run_dir = run_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise ValueError("Categorical fidelity audit requires a fresh output directory")
    if distance_perturbation <= 0 or margin_perturbation <= 0:
        raise ValueError("Fidelity perturbations must be positive")

    extraction_path = run_dir / "extraction.json"
    plan_snapshot_path = run_dir / "plan.snapshot.json"
    extraction = _load(extraction_path)
    plan = _load(plan_snapshot_path)
    declared_plan = _resolve_file(extraction["plan"]["path"], run_dir)
    if _sha256(declared_plan) != extraction["plan"]["sha256"]:
        raise ValueError("Extraction plan hash is stale")
    if _load(declared_plan) != plan:
        raise ValueError("Plan snapshot differs from the declared plan")
    source_path = _resolve_file(extraction["source"]["path"], run_dir)
    if _sha256(source_path) != extraction["source"]["sha256"]:
        raise ValueError("Extraction source hash is stale")

    source_rgb = np.asarray(Image.open(source_path).convert("RGB"))
    state_mask = np.asarray(Image.open(run_dir / "source-state-mask.png").convert("L")) > 0
    if state_mask.shape != source_rgb.shape[:2]:
        raise ValueError("Source and source-state mask shapes differ")
    source_context, _ = _source_context_exclusion(
        source_rgb, state_mask, plan.get("source_context_exclusion")
    )
    plan_layers = {str(layer["id"]): layer for layer in plan.get("layers", [])}
    extraction_layers = {
        str(layer["id"]): layer for layer in extraction.get("layers", [])
    }
    output_dir.mkdir(parents=True)
    layer_reports = []

    for layer_id, definition in plan_layers.items():
        if definition.get("kind") != "categorical":
            continue
        report = extraction_layers.get(layer_id)
        if not isinstance(report, dict):
            raise ValueError(f"Extraction is missing categorical layer {layer_id}")
        categories = definition.get("categories", [])
        if not isinstance(categories, list) or not categories:
            raise ValueError(f"Layer {layer_id} has no categories")
        maximum_distance = float(definition.get("max_distance", 24.0))
        minimum_margin = float(definition.get("min_margin", 3.0))
        baseline_path = run_dir / layer_id / "source-class-id.png"
        web_path = run_dir / layer_id / "web-mercator-class-id.png"
        interior_path = run_dir / layer_id / "web-mercator-publication-interior-mask.png"
        baseline = np.asarray(Image.open(baseline_path), dtype=np.uint8)
        web = np.asarray(Image.open(web_path), dtype=np.uint8)
        interior = np.asarray(Image.open(interior_path).convert("L")) > 0
        if baseline.shape != state_mask.shape or web.shape != interior.shape:
            raise ValueError(f"Layer {layer_id} artifact shapes differ")

        distances = sorted(
            {
                max(0.0, maximum_distance - distance_perturbation),
                maximum_distance,
                maximum_distance + distance_perturbation,
            }
        )
        margins = sorted(
            {
                max(0.0, minimum_margin - margin_perturbation),
                minimum_margin,
                minimum_margin + margin_perturbation,
            }
        )
        variants: dict[tuple[float, float], np.ndarray] = {}
        variant_reports = []
        for max_distance in distances:
            for min_margin in margins:
                values, _ = classify_categorical(
                    source_rgb,
                    state_mask,
                    categories,
                    max_distance=max_distance,
                    min_margin=min_margin,
                    exclusion_mask=source_context,
                )
                variants[(max_distance, min_margin)] = values
                variant_reports.append(
                    {
                        "maximum_lab_distance": max_distance,
                        "minimum_lab_margin": min_margin,
                        "classified_pixel_count": int(np.count_nonzero(values)),
                        "added_vs_baseline": int(
                            np.count_nonzero((values > 0) & (baseline == 0))
                        ),
                        "dropped_vs_baseline": int(
                            np.count_nonzero((values == 0) & (baseline > 0))
                        ),
                        "different_nonzero_class_count": int(
                            np.count_nonzero(
                                (values > 0) & (baseline > 0) & (values != baseline)
                            )
                        ),
                    }
                )
        recomputed = variants[(maximum_distance, minimum_margin)]
        if not np.array_equal(recomputed, baseline):
            raise ValueError(f"Stored source classes do not recompute for {layer_id}")
        strict = variants[(min(distances), max(margins))]
        relaxed = variants[(max(distances), min(margins))]
        strict_drop = (baseline > 0) & (strict == 0)
        relaxed_addition = (baseline == 0) & (relaxed > 0)
        variant_stack = np.stack(list(variants.values()), axis=0)
        nonzero_min = np.where(variant_stack > 0, variant_stack, 255).min(axis=0)
        nonzero_max = variant_stack.max(axis=0)
        semantic_change = (nonzero_max > 0) & (nonzero_min != nonzero_max)
        source_spread = source_rgb.max(axis=2).astype(int) - source_rgb.min(axis=2).astype(int)
        pale_neutral = (source_rgb.min(axis=2) >= 240) & (source_spread <= 20)
        pale_relaxed = relaxed_addition & pale_neutral

        legend_rgb = np.asarray(
            [
                category["legend_rgb"][0]
                if isinstance(category["legend_rgb"][0], list)
                else category["legend_rgb"]
                for category in categories
            ],
            dtype=np.uint8,
        )
        legend_lab = cv2.cvtColor(legend_rgb[None, :, :], cv2.COLOR_RGB2LAB)[
            0
        ].astype(np.float32)
        pairwise = cdist(legend_lab, legend_lab)
        minimum_separation = float(
            pairwise[np.triu_indices(len(categories), 1)].min()
        ) if len(categories) > 1 else None

        layer_dir = output_dir / layer_id
        category_dir = layer_dir / "categories"
        category_dir.mkdir(parents=True)
        artifacts = {
            "strict_drop": _save_mask(layer_dir / "strict-drop-mask.png", strict_drop),
            "relaxed_addition": _save_mask(
                layer_dir / "relaxed-addition-mask.png", relaxed_addition
            ),
            "pale_neutral_relaxed_addition": _save_mask(
                layer_dir / "pale-neutral-relaxed-addition-mask.png", pale_relaxed
            ),
            "semantic_change": _save_mask(
                layer_dir / "semantic-class-change-mask.png", semantic_change
            ),
        }
        category_reports = []
        for class_id, category in enumerate(categories, 1):
            source_mask = baseline == class_id
            web_mask = web == class_id
            source_artifact = _save_mask(
                category_dir / f"{category['id']}-source-mask.png", source_mask
            )
            web_artifact = _save_mask(
                category_dir / f"{category['id']}-web-mask.png", web_mask
            )
            source_artifact["path"] = str(
                (category_dir / f"{category['id']}-source-mask.png").relative_to(
                    layer_dir
                )
            )
            web_artifact["path"] = str(
                (category_dir / f"{category['id']}-web-mask.png").relative_to(
                    layer_dir
                )
            )
            category_reports.append(
                {
                    "id": category["id"],
                    "class_id": class_id,
                    "visible_source_status": (
                        "observed" if np.any(source_mask) else "absent"
                    ),
                    "source_pixel_count": int(np.count_nonzero(source_mask)),
                    "web_pixel_count": int(np.count_nonzero(web_mask)),
                    "relaxed_addition_pixel_count": int(
                        np.count_nonzero(relaxed_addition & (relaxed == class_id))
                    ),
                    "source_components": _component_summary(source_mask),
                    "web_components": _component_summary(web_mask),
                    "artifacts": {"source_mask": source_artifact, "web_mask": web_artifact},
                }
            )

        faded_source = np.rint(source_rgb.astype(np.float32) * 0.55).astype(np.uint8)
        faded_source[~state_mask] = 12
        evidence = faded_source.copy()
        evidence[baseline > 0] = _category_preview(baseline, categories)[baseline > 0]
        evidence[strict_drop] = [255, 181, 71]
        evidence[relaxed_addition] = [52, 208, 255]
        evidence[semantic_change] = [255, 56, 96]
        preview = _category_preview(baseline, categories)
        montage = np.concatenate((source_rgb, preview, evidence), axis=1)
        montage_path = layer_dir / "source-fidelity-diagnostic.jpg"
        Image.fromarray(montage).save(montage_path, quality=94, optimize=True)
        artifacts["diagnostic"] = {
            "path": montage_path.name,
            "sha256": _sha256(montage_path),
            "panels": [
                "original source",
                "stored categorical extraction",
                "baseline colors with strict drops amber, relaxed additions cyan, semantic changes red",
            ],
        }

        outside_web = int(np.count_nonzero((web > 0) & ~interior))
        empty_categories = [
            item["id"] for item in category_reports if item["web_pixel_count"] == 0
        ]
        relaxed_count = int(np.count_nonzero(relaxed_addition))
        pale_count = int(np.count_nonzero(pale_relaxed))
        layer_reports.append(
            {
                "id": layer_id,
                "status": "pass",
                "baseline_recomputed_exactly": True,
                "stored_source_class_sha256": _sha256(baseline_path),
                "stored_web_class_sha256": _sha256(web_path),
                "maximum_lab_distance": maximum_distance,
                "minimum_lab_margin": minimum_margin,
                "legend_minimum_pairwise_lab_separation": minimum_separation,
                "variant_count": len(variants),
                "variants": variant_reports,
                "strict_drop_pixel_count": int(np.count_nonzero(strict_drop)),
                "relaxed_addition_pixel_count": relaxed_count,
                "pale_neutral_relaxed_addition_pixel_count": pale_count,
                "pale_neutral_fraction_of_relaxed_additions": (
                    pale_count / relaxed_count if relaxed_count else 0.0
                ),
                "semantic_class_change_pixel_count": int(
                    np.count_nonzero(semantic_change)
                ),
                "web_colored_pixel_count_outside_canonical_interior": outside_web,
                "empty_categories": empty_categories,
                "empty_category_policy": (
                    "preserve_legend_entry_as_zero_coverage"
                    if empty_categories
                    else "all_legend_categories_observed"
                ),
                "empty_categories_require_visual_review": bool(empty_categories),
                "threshold_decision": "retain_reviewed_conservative_thresholds",
                "threshold_decision_reason": (
                    "Relaxation admits pixels nearest to legend colors without proving that "
                    "they encode data; pale neutral additions are especially confusable with "
                    "the uncolored map background. Strict and relaxed evidence remain separate."
                ),
                "categories": category_reports,
                "artifacts": artifacts,
            }
        )

    if not layer_reports:
        raise ValueError("Run contains no simple categorical layer to audit")
    if any(
        layer["semantic_class_change_pixel_count"] != 0
        or layer["web_colored_pixel_count_outside_canonical_interior"] != 0
        for layer in layer_reports
    ):
        raise ValueError("Categorical fidelity gate failed")
    result = {
        "schema_version": 1,
        "audit_kind": "categorical_threshold_fidelity",
        "status": "pass",
        "run": str(run_dir),
        "extraction_manifest_sha256": _sha256(extraction_path),
        "plan_sha256": _sha256(declared_plan),
        "source_sha256": _sha256(source_path),
        "distance_perturbation": distance_perturbation,
        "margin_perturbation": margin_perturbation,
        "layers": layer_reports,
        "publication_approved": False,
    }
    report_path = output_dir / "categorical-fidelity-audit.json"
    report_path.write_text(json.dumps(result, indent=2) + "\n")
    return {**result, "audit_sha256": _sha256(report_path)}
