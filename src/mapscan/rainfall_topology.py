#!/usr/bin/env python3
"""Conservative optional topology stage for confusable rainfall palettes.

This stage never changes the accepted categorical candidate.  It preserves
three different evidence states in separate rasters:

* directly classified, non-confusable color evidence;
* visible membership in a confusable legend-palette family; and
* a class assignment inferred only where a palette-family component has a
  dominant boundary against the immediately adjacent readable-color class.

Everything else remains explicitly unresolved.  In particular, a contour is
not treated as a numeric label and topology never invents a middle ordinal class
without a semantic anchor.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

from .extraction import warp_classified_to_web_mercator
from .reference import load_california


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _confusable_groups(pairs: Sequence[Dict[str, object]]) -> List[List[str]]:
    parent: Dict[str, str] = {}

    def find(item: str) -> str:
        parent.setdefault(item, item)
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(first: str, second: str) -> None:
        root_first, root_second = find(first), find(second)
        if root_first != root_second:
            parent[root_second] = root_first

    for pair in pairs:
        union(str(pair["first_category"]), str(pair["second_category"]))
    grouped: Dict[str, List[str]] = defaultdict(list)
    for item in parent:
        grouped[find(item)].append(item)
    return [sorted(items) for items in grouped.values() if len(items) > 1]


def _sample_histogram(
    rgb: np.ndarray, rect: Sequence[int]
) -> Tuple[Dict[Tuple[int, int, int], float], int]:
    x0, y0, x1, y1 = (int(value) for value in rect)
    pixels = rgb[y0:y1, x0:x1].reshape((-1, 3))
    colors, counts = np.unique(pixels, axis=0, return_counts=True)
    total = int(len(pixels))
    return {
        tuple(int(channel) for channel in color): float(count) / total
        for color, count in zip(colors, counts)
    }, total


def _total_variation(
    first: Dict[Tuple[int, int, int], float],
    second: Dict[Tuple[int, int, int], float],
) -> float:
    colors = set(first) | set(second)
    return 0.5 * sum(abs(first.get(color, 0.0) - second.get(color, 0.0)) for color in colors)


def _shared_palette_colors(
    histograms: Sequence[Dict[Tuple[int, int, int], float]],
    minimum_fraction: float = 0.005,
) -> List[Tuple[int, int, int]]:
    shared = set(histograms[0])
    for histogram in histograms[1:]:
        shared &= set(histogram)
    return sorted(
        color
        for color in shared
        if min(histogram.get(color, 0.0) for histogram in histograms)
        >= minimum_fraction
    )


def _credible_numeric_labels(
    rgb: np.ndarray,
    state_mask: np.ndarray,
    expected_labels: Iterable[str],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Run two sparse OCR layouts and retain only strict expected-value tokens."""

    expected = {label.strip() for label in expected_labels}
    expected |= {label.removesuffix(".0") for label in expected if label.endswith(".0")}
    all_tokens: List[Dict[str, object]] = []
    credible: List[Dict[str, object]] = []
    masked = np.full_like(rgb, 255)
    masked[state_mask] = rgb[state_mask]
    with tempfile.TemporaryDirectory(prefix="rainfall-topology-ocr-") as temp_dir:
        image_path = Path(temp_dir) / "state-only.png"
        Image.fromarray(masked).save(image_path)
        for psm in (11, 12):
            process = subprocess.run(
                ["tesseract", str(image_path), "stdout", "--psm", str(psm), "tsv"],
                check=True,
                capture_output=True,
                text=True,
            )
            for row in csv.DictReader(process.stdout.splitlines(), delimiter="\t"):
                token = row.get("text", "").strip()
                if not token:
                    continue
                try:
                    confidence = float(row.get("conf", "-1"))
                    left, top = int(row["left"]), int(row["top"])
                    width, height = int(row["width"]), int(row["height"])
                except (TypeError, ValueError):
                    continue
                normalized = token.replace(",", ".").strip("()[]{}:;")
                record = {
                    "psm": psm,
                    "text": token,
                    "normalized": normalized,
                    "confidence": confidence,
                    "box": [left, top, width, height],
                }
                if re.fullmatch(r"\d+(?:\.\d+)?", normalized):
                    all_tokens.append(record)
                if normalized not in expected:
                    continue
                threshold = 85.0 if "." not in normalized else 70.0
                if confidence < threshold:
                    continue
                center_x = min(max(left + width // 2, 0), state_mask.shape[1] - 1)
                center_y = min(max(top + height // 2, 0), state_mask.shape[0] - 1)
                if not state_mask[center_y, center_x]:
                    continue
                credible.append(record)
    # The two layouts may return the same box; preserve only distinct observations.
    unique = {}
    for record in credible:
        key = (record["normalized"], tuple(record["box"]))
        unique[key] = record
    return sorted(unique.values(), key=lambda item: (item["box"][1], item["box"][0])), all_tokens


def _save_class(path: Path, values: np.ndarray) -> None:
    maximum = int(np.max(values)) if values.size else 0
    dtype = np.uint16 if maximum > 255 else np.uint8
    Image.fromarray(values.astype(dtype)).save(path, optimize=True)


def _save_mask(path: Path, mask: np.ndarray) -> None:
    Image.fromarray(mask.astype(np.uint8) * 255).save(path, optimize=True)


def _infer_family_variant(
    visible: np.ndarray,
    gray: np.ndarray,
    source_class: np.ndarray,
    direct_known: np.ndarray,
    indices: Sequence[int],
    category_count: int,
    dark_threshold: int,
    boundary_radius: int,
) -> Tuple[np.ndarray, List[Dict[str, object]]]:
    """Infer endpoint classes for one reasonable segmentation parameter set."""

    dark = gray < dark_threshold
    sealed_dark = cv2.dilate(
        dark.astype(np.uint8), np.ones((3, 3), dtype=np.uint8)
    ) > 0
    work = visible & ~sealed_dark
    component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        work.astype(np.uint8), connectivity=8
    )
    low_anchor = indices[0] - 1 if indices[0] > 1 else None
    high_anchor = indices[-1] + 1 if indices[-1] < category_count else None
    kernel = np.ones((2 * boundary_radius + 1, 2 * boundary_radius + 1), dtype=np.uint8)
    inferred = np.zeros(source_class.shape, dtype=np.uint8)
    component_reports = []
    for component_id in range(1, component_count):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area < 50:
            continue
        component = labels == component_id
        ring = (cv2.dilate(component.astype(np.uint8), kernel) > 0) & ~component
        values, counts = np.unique(source_class[ring & direct_known], return_counts=True)
        supports = {int(value): int(count) for value, count in zip(values, counts)}
        total_support = sum(supports.values())
        low_support = supports.get(low_anchor, 0) if low_anchor is not None else 0
        high_support = supports.get(high_anchor, 0) if high_anchor is not None else 0
        assignment = None
        reason = "no_dominant_immediate_ordinal_anchor"
        if low_anchor is not None and low_support >= 25:
            dominance = low_support / max(high_support, 1)
            share = low_support / max(total_support, 1)
            if dominance >= 4.0 and share >= 0.5:
                assignment = indices[0]
                reason = "dominant_immediate_lower_anchor"
        if high_anchor is not None and high_support >= 25:
            dominance = high_support / max(low_support, 1)
            share = high_support / max(total_support, 1)
            if dominance >= 4.0 and share >= 0.5:
                if assignment is None:
                    assignment = indices[-1]
                    reason = "dominant_immediate_upper_anchor"
                else:
                    assignment = None
                    reason = "conflicting_immediate_anchors"
        if assignment is not None:
            inferred[component] = assignment
        component_reports.append(
            {
                "id": component_id,
                "area": area,
                "centroid": [
                    float(centroids[component_id, 0]),
                    float(centroids[component_id, 1]),
                ],
                "direct_boundary_support_by_index": {
                    str(index): count for index, count in supports.items()
                },
                "assigned_index": assignment,
                "decision": reason,
            }
        )
    return inferred, component_reports


def extract_rainfall_topology(
    plan_path: Path, extraction_dir: Path, output_dir: Path
) -> Dict[str, object]:
    """Write a separate, non-mutating topology interpretation of rainfall evidence."""

    resolved_extraction = extraction_dir.resolve()
    resolved_output = output_dir.resolve()
    if resolved_output == resolved_extraction or resolved_extraction in resolved_output.parents:
        raise ValueError(
            "Rainfall topology output must be separate from the accepted extraction run"
        )
    plan = json.loads(plan_path.read_text())
    manifest_path = extraction_dir / "extraction.json"
    manifest = json.loads(manifest_path.read_text())
    source_path = Path(plan["source"])
    alignment_path = Path(plan["alignment"])
    alignment = json.loads(alignment_path.read_text())
    plan_sha256 = _sha256(plan_path)
    source_sha256 = _sha256(source_path)
    if manifest["plan"]["sha256"] != plan_sha256:
        raise ValueError("Extraction manifest does not belong to the supplied rainfall plan")
    if manifest["source"]["sha256"] != source_sha256:
        raise ValueError("Extraction source hash does not match the supplied rainfall plan")
    rgb = np.asarray(Image.open(source_path).convert("RGB"))
    state_mask = np.asarray(Image.open(extraction_dir / "source-state-mask.png")) > 0
    layer = plan["layers"][0]
    layer_dir = extraction_dir / str(layer["id"])
    source_class = np.asarray(Image.open(layer_dir / "source-class-id.png"))
    completion_mask = np.asarray(Image.open(layer_dir / "source-completion-mask.png")) > 0
    categories = list(layer["categories"])
    category_by_id = {str(category["id"]): category for category in categories}
    category_index = {
        str(category["id"]): index for index, category in enumerate(categories, 1)
    }
    extraction_layer = manifest["layers"][0]
    groups = _confusable_groups(extraction_layer["extraction"]["confusable_legend_pairs"])
    output_dir.mkdir(parents=True, exist_ok=True)

    color_evidence = source_class.copy()
    color_evidence[completion_mask] = 0
    family_ids = np.zeros(source_class.shape, dtype=np.uint8)
    topology_class = np.zeros(source_class.shape, dtype=np.uint8)
    topology_mask = np.zeros(source_class.shape, dtype=bool)
    group_reports: List[Dict[str, object]] = []
    legend_pair_reports: List[Dict[str, object]] = []
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    direct_known = (source_class > 0) & ~completion_mask

    for family_id, ids in enumerate(groups, 1):
        indices = sorted(category_index[item] for item in ids)
        for index in indices:
            color_evidence[color_evidence == index] = 0
        histograms = []
        sample_pixel_counts = []
        for item in ids:
            histogram, pixel_count = _sample_histogram(
                rgb, category_by_id[item]["sample_rect"]
            )
            histograms.append(histogram)
            sample_pixel_counts.append(pixel_count)
        for first_position in range(len(ids)):
            for second_position in range(first_position + 1, len(ids)):
                legend_pair_reports.append(
                    {
                        "first": ids[first_position],
                        "second": ids[second_position],
                        "total_variation_distance": _total_variation(
                            histograms[first_position], histograms[second_position]
                        ),
                        "information_identical": histograms[first_position]
                        == histograms[second_position],
                    }
                )
        colors = _shared_palette_colors(histograms)
        visible = np.zeros(state_mask.shape, dtype=bool)
        for color in colors:
            visible |= np.all(rgb == np.asarray(color, dtype=np.uint8), axis=2)
        visible &= state_mask
        family_ids[visible] = family_id
        variants = []
        base_component_reports = []
        for dark_threshold in (120, 160, 200):
            for boundary_radius in (2, 3, 4):
                variant, component_reports = _infer_family_variant(
                    visible,
                    gray,
                    source_class,
                    direct_known,
                    indices,
                    len(categories),
                    dark_threshold,
                    boundary_radius,
                )
                variants.append(
                    {
                        "dark_luminance_threshold": dark_threshold,
                        "boundary_radius": boundary_radius,
                        "values": variant,
                        "assigned_pixel_count": int(np.count_nonzero(variant)),
                    }
                )
                if dark_threshold == 160 and boundary_radius == 3:
                    base_component_reports = component_reports
        variant_stack = np.stack([item["values"] for item in variants], axis=0)
        stable_values = variant_stack[0]
        stable = (stable_values > 0) & np.all(
            variant_stack == stable_values[None, :, :], axis=0
        )
        topology_class[stable] = stable_values[stable]
        topology_mask[stable] = True
        inferred_by_class = {
            str(categories[index - 1]["id"]): int(
                np.count_nonzero(stable & (stable_values == index))
            )
            for index in indices
            if np.any(stable & (stable_values == index))
        }
        component_reports = []
        for component in base_component_reports:
            converted = dict(component)
            converted["direct_boundary_support_by_class_id"] = {
                str(categories[int(index) - 1]["id"]): count
                for index, count in converted.pop(
                    "direct_boundary_support_by_index"
                ).items()
                if 1 <= int(index) <= len(categories)
            }
            assignment = converted.pop("assigned_index")
            converted["assigned_class_id_in_base_variant"] = (
                str(categories[assignment - 1]["id"])
                if assignment is not None
                else None
            )
            component_reports.append(converted)
        inferred = topology_mask & (family_ids == family_id)
        group_reports.append(
            {
                "family_id": family_id,
                "category_ids": ids,
                "category_indices": indices,
                "sample_pixel_counts": sample_pixel_counts,
                "shared_gif_colors": [list(color) for color in colors],
                "visible_palette_family_pixel_count": int(np.count_nonzero(visible)),
                "topology_inferred_pixel_count": int(np.count_nonzero(inferred)),
                "unresolved_visible_pixel_count": int(np.count_nonzero(visible & ~inferred)),
                "topology_inferred_by_class": dict(inferred_by_class),
                "eligible_component_count": len(component_reports),
                "stability_gate": {
                    "required_agreement_across_all_variants": True,
                    "dark_luminance_thresholds": [120, 160, 200],
                    "boundary_radii": [2, 3, 4],
                    "variant_assigned_pixel_counts": [
                        {
                            "dark_luminance_threshold": item[
                                "dark_luminance_threshold"
                            ],
                            "boundary_radius": item["boundary_radius"],
                            "assigned_pixel_count": item["assigned_pixel_count"],
                        }
                        for item in variants
                    ],
                },
                "component_decisions": component_reports,
            }
        )

    expected_labels = [str(category["label"]).split()[0] for category in categories]
    numeric_labels, all_numeric_ocr_tokens = _credible_numeric_labels(
        rgb, state_mask, expected_labels
    )
    unresolved = (family_ids > 0) & ~topology_mask
    _save_class(
        output_dir / "source-observed-nonconfusable-class-id.png", color_evidence
    )
    _save_class(output_dir / "source-palette-family-id.png", family_ids)
    _save_class(output_dir / "source-topology-inferred-class-id.png", topology_class)
    _save_mask(output_dir / "source-topology-inference-mask.png", topology_mask)
    _save_mask(output_dir / "source-unresolved-confusable-palette-mask.png", unresolved)

    display_palette = np.zeros((len(categories) + 1, 3), dtype=np.uint8)
    for index, category in enumerate(categories, 1):
        display_palette[index] = np.asarray(category["display_rgb"], dtype=np.uint8)
    source_topology_preview = display_palette[topology_class]
    Image.fromarray(source_topology_preview).save(
        output_dir / "source-topology-inferred-preview.png", optimize=True
    )

    overlay = rgb.copy()
    overlay[topology_mask] = np.rint(
        overlay[topology_mask].astype(np.float32) * 0.2
        + np.asarray([0, 238, 180], dtype=np.float32) * 0.8
    ).astype(np.uint8)
    overlay[unresolved] = np.rint(
        overlay[unresolved].astype(np.float32) * 0.25
        + np.asarray([255, 0, 180], dtype=np.float32) * 0.75
    ).astype(np.uint8)
    Image.fromarray(overlay).save(
        output_dir / "source-topology-review.png", optimize=True
    )

    reference_root = Path(plan.get("reference", "reference/census-2025"))
    state, _ = load_california(reference_root)
    transform = _alignment_transform(alignment)
    web_artifacts = {}
    web_topology_class = None
    for name, source_values in (
        ("observed-nonconfusable-class-id", color_evidence),
        ("palette-family-id", family_ids),
        ("topology-inferred-class-id", topology_class),
        ("topology-inference-mask", topology_mask.astype(np.uint8)),
        ("unresolved-confusable-palette-mask", unresolved.astype(np.uint8)),
    ):
        warped, warp_report = warp_classified_to_web_mercator(
            source_values, state, transform, rgb.shape[:2]
        )
        path = output_dir / f"web-mercator-{name}.png"
        if name.endswith("mask"):
            _save_mask(path, warped > 0)
        else:
            _save_class(path, warped)
        if name == "topology-inferred-class-id":
            web_topology_class = warped
        web_artifacts[name] = {"path": path.name, "sha256": _sha256(path)}
    assert web_topology_class is not None
    web_topology_preview_path = output_dir / "web-mercator-topology-inferred-preview.png"
    Image.fromarray(display_palette[web_topology_class]).save(
        web_topology_preview_path, optimize=True
    )
    web_artifacts["topology-inferred-preview"] = {
        "path": web_topology_preview_path.name,
        "sha256": _sha256(web_topology_preview_path),
    }

    artifacts = {}
    for path in sorted(output_dir.glob("source-*.png")):
        artifacts[path.stem.removeprefix("source-")] = {
            "path": path.name,
            "sha256": _sha256(path),
        }
    result = {
        "schema_version": 1,
        "status": "partial_topology_inference_only",
        "dataset_id": plan["dataset_id"],
        "source": {"path": str(source_path), "sha256": source_sha256},
        "plan": {"path": str(plan_path), "sha256": plan_sha256},
        "extraction": {
            "path": str(extraction_dir),
            "manifest_path": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
            "source_class_id_sha256": _sha256(layer_dir / "source-class-id.png"),
            "source_completion_mask_sha256": _sha256(
                layer_dir / "source-completion-mask.png"
            ),
        },
        "accepted_candidate_unchanged": True,
        "legend_information": {
            "confusable_groups": groups,
            "pair_comparisons": legend_pair_reports,
        },
        "numeric_isohyet_labels": {
            "credible_expected_value_label_count": len(numeric_labels),
            "credible_labels": numeric_labels,
            "all_numeric_ocr_tokens_for_audit": all_numeric_ocr_tokens,
            "interpretation": (
                "No semantic numeric anchor was used. OCR observations must match a "
                "legend value, lie inside the registered state, and pass a strict "
                "confidence threshold; contours alone never supply a numeric value."
            ),
        },
        "groups": group_reports,
        "source_pixel_totals": {
            "direct_nonconfusable_color_evidence": int(np.count_nonzero(color_evidence)),
            "visible_confusable_palette_family_evidence": int(
                np.count_nonzero(family_ids)
            ),
            "topology_inferred": int(np.count_nonzero(topology_mask)),
            "unresolved_visible_confusable_palette": int(np.count_nonzero(unresolved)),
        },
        "artifacts": artifacts,
        "web_mercator_artifacts": web_artifacts,
        "warp": warp_report,
        "semantic_limit": (
            "Identical GIF palettes erase per-pixel class identity. This prototype "
            "recovers only endpoint classes where region topology has a dominant "
            "immediately adjacent readable-color anchor. Middle classes and all "
            "weak or conflicting regions remain unresolved rather than guessed."
        ),
    }
    report_path = output_dir / "rainfall-topology-report.json"
    report_path.write_text(json.dumps(result, indent=2) + "\n")
    return result
