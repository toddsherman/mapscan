"""Conservative, explicitly marked inference for small categorical occlusions."""

from __future__ import annotations

import hashlib
import csv
import json
import re
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

from .extraction import warp_classified_to_web_mercator
from .reference import load_california


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def infer_small_categorical_gaps(
    values: np.ndarray,
    valid_mask: np.ndarray,
    class_count: int,
    max_gap_radius_px: int = 6,
    max_component_area_px: int = 1_200,
    max_component_dimension_px: int = 120,
    ring_radius_px: int = 3,
    minimum_ring_pixels: int = 12,
    minimum_dominance: float = 0.98,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Fill only narrow zero-valued gaps locally enclosed by one observed class.

    The original values remain authoritative. The returned inference mask marks
    every filled pixel so downstream products can disclose or disable them.
    """

    if values.ndim != 2 or valid_mask.shape != values.shape:
        raise ValueError("Categorical values and valid mask must be matching 2D arrays")
    if max_gap_radius_px < 1 or ring_radius_px < 1:
        raise ValueError("Gap and ring radii must be positive")
    if not 0.5 < minimum_dominance <= 1.0:
        raise ValueError("Minimum dominance must be greater than 0.5 and at most 1")

    gap_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * max_gap_radius_px + 1, 2 * max_gap_radius_px + 1),
    )
    ring_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * ring_radius_px + 1, 2 * ring_radius_px + 1),
    )
    proposals = np.zeros(values.shape, dtype=np.uint8)
    conflicts = np.zeros(values.shape, dtype=bool)
    accepted_components = {str(class_id): 0 for class_id in range(1, class_count + 1)}
    proposed_pixels = {str(class_id): 0 for class_id in range(1, class_count + 1)}
    rejected = {
        "too_large": 0,
        "too_wide": 0,
        "insufficient_ring": 0,
        "mixed_surroundings": 0,
    }

    nodata = (values == 0) & valid_mask.astype(bool)
    for class_id in range(1, class_count + 1):
        observed = (values == class_id).astype(np.uint8)
        closed = cv2.morphologyEx(observed, cv2.MORPH_CLOSE, gap_kernel) > 0
        candidates = (closed & nodata).astype(np.uint8)
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
            candidates, connectivity=8
        )
        for component_id in range(1, component_count):
            area = int(stats[component_id, cv2.CC_STAT_AREA])
            x = int(stats[component_id, cv2.CC_STAT_LEFT])
            y = int(stats[component_id, cv2.CC_STAT_TOP])
            width = int(stats[component_id, cv2.CC_STAT_WIDTH])
            height = int(stats[component_id, cv2.CC_STAT_HEIGHT])
            if area > max_component_area_px:
                rejected["too_large"] += 1
                continue
            if max(width, height) > max_component_dimension_px:
                rejected["too_wide"] += 1
                continue
            x0 = max(0, x - ring_radius_px)
            y0 = max(0, y - ring_radius_px)
            x1 = min(values.shape[1], x + width + ring_radius_px)
            y1 = min(values.shape[0], y + height + ring_radius_px)
            window = np.s_[y0:y1, x0:x1]
            component = labels[window] == component_id
            ring = (
                cv2.dilate(component.astype(np.uint8), ring_kernel).astype(bool)
                & ~component
                & valid_mask[window].astype(bool)
            )
            neighbors = values[window][ring]
            neighbors = neighbors[neighbors > 0]
            if len(neighbors) < minimum_ring_pixels:
                rejected["insufficient_ring"] += 1
                continue
            same_fraction = float(np.mean(neighbors == class_id))
            if same_fraction < minimum_dominance:
                rejected["mixed_surroundings"] += 1
                continue
            proposal_window = proposals[window]
            conflict_window = conflicts[window]
            occupied = component & (proposal_window > 0)
            conflict_window |= occupied & (proposal_window != class_id)
            proposal_window[component & ~occupied] = class_id
            accepted_components[str(class_id)] += 1
            proposed_pixels[str(class_id)] += area

    inference_mask = (proposals > 0) & ~conflicts
    inferred = values.copy().astype(np.uint8)
    inferred[inference_mask] = proposals[inference_mask]
    pixels_by_class = {
        str(class_id): int(np.count_nonzero(inference_mask & (proposals == class_id)))
        for class_id in range(1, class_count + 1)
    }
    report = {
        "method": "single_class_morphological_closure_with_dominant_ring_gate",
        "parameters": {
            "max_gap_radius_px": max_gap_radius_px,
            "max_component_area_px": max_component_area_px,
            "max_component_dimension_px": max_component_dimension_px,
            "ring_radius_px": ring_radius_px,
            "minimum_ring_pixels": minimum_ring_pixels,
            "minimum_dominance": minimum_dominance,
        },
        "inferred_pixel_count": int(np.count_nonzero(inference_mask)),
        "conflict_pixel_count": int(np.count_nonzero(conflicts)),
        "inferred_pixels_by_class_id": pixels_by_class,
        "accepted_components_by_class_id": accepted_components,
        "proposed_pixels_by_class_id_before_conflict_rejection": proposed_pixels,
        "rejected_component_counts": rejected,
        "warning": (
            "Inferred pixels are not observed source data. They fill only small narrow gaps "
            "surrounded almost entirely by one class and remain separately masked."
        ),
    }
    return inferred, inference_mask, report


def infer_ocr_label_occlusions(
    values: np.ndarray,
    valid_mask: np.ndarray,
    ocr_tsv_path: Path,
    minimum_confidence: float = 85.0,
    label_padding_px: int = 14,
    context_radius_px: int = 12,
    minimum_context_pixels: int = 40,
    minimum_dominance: float = 0.94,
    maximum_label_dimension_px: int = 260,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Infer full OCR label neighborhoods when one observed class surrounds them."""

    if not ocr_tsv_path.exists():
        raise FileNotFoundError(f"Missing OCR TSV: {ocr_tsv_path}")
    if not 0.5 < minimum_dominance <= 1.0:
        raise ValueError("OCR inference dominance must be greater than 0.5 and at most 1")
    inferred = values.copy().astype(np.uint8)
    inference_mask = np.zeros(values.shape, dtype=bool)
    accepted = []
    rejected = {
        "low_confidence_or_nonword": 0,
        "too_large": 0,
        "outside_valid_map": 0,
        "no_nodata": 0,
        "insufficient_context": 0,
        "mixed_context": 0,
    }
    with ocr_tsv_path.open(newline="") as handle:
        rows = csv.DictReader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in rows:
            text = str(row.get("text", "")).strip()
            try:
                confidence = float(row.get("conf", "-1"))
                x = int(row.get("left", "0"))
                y = int(row.get("top", "0"))
                width = int(row.get("width", "0"))
                height = int(row.get("height", "0"))
            except ValueError:
                rejected["low_confidence_or_nonword"] += 1
                continue
            letters = re.sub(r"[^A-Za-z]", "", text)
            if confidence < minimum_confidence or len(letters) < 3:
                rejected["low_confidence_or_nonword"] += 1
                continue
            if max(width, height) > maximum_label_dimension_px or width <= 0 or height <= 0:
                rejected["too_large"] += 1
                continue
            x0 = max(0, x - label_padding_px)
            y0 = max(0, y - label_padding_px)
            x1 = min(values.shape[1], x + width + label_padding_px)
            y1 = min(values.shape[0], y + height + label_padding_px)
            valid_window = valid_mask[y0:y1, x0:x1].astype(bool)
            if not np.any(valid_window):
                rejected["outside_valid_map"] += 1
                continue
            target = (values[y0:y1, x0:x1] == 0) & valid_window
            if not np.any(target):
                rejected["no_nodata"] += 1
                continue
            context_x0 = max(0, x0 - context_radius_px)
            context_y0 = max(0, y0 - context_radius_px)
            context_x1 = min(values.shape[1], x1 + context_radius_px)
            context_y1 = min(values.shape[0], y1 + context_radius_px)
            context = values[context_y0:context_y1, context_x0:context_x1]
            context_valid = valid_mask[context_y0:context_y1, context_x0:context_x1]
            observed = context[(context > 0) & context_valid]
            if len(observed) < minimum_context_pixels:
                rejected["insufficient_context"] += 1
                continue
            counts = np.bincount(observed, minlength=256)
            counts[0] = 0
            class_id = int(np.argmax(counts))
            dominance = float(counts[class_id] / len(observed))
            if dominance < minimum_dominance:
                rejected["mixed_context"] += 1
                continue
            output_window = inferred[y0:y1, x0:x1]
            mask_window = inference_mask[y0:y1, x0:x1]
            newly_inferred = target & ~mask_window
            output_window[newly_inferred] = class_id
            mask_window[newly_inferred] = True
            accepted.append(
                {
                    "text": text,
                    "confidence": confidence,
                    "bbox": [x, y, width, height],
                    "padded_bbox": [x0, y0, x1 - x0, y1 - y0],
                    "class_id": class_id,
                    "dominance": dominance,
                    "inferred_pixel_count": int(np.count_nonzero(newly_inferred)),
                }
            )
    report = {
        "method": "ocr_label_bbox_with_dominant_class_context",
        "ocr_tsv": str(ocr_tsv_path),
        "parameters": {
            "minimum_confidence": minimum_confidence,
            "label_padding_px": label_padding_px,
            "context_radius_px": context_radius_px,
            "minimum_context_pixels": minimum_context_pixels,
            "minimum_dominance": minimum_dominance,
            "maximum_label_dimension_px": maximum_label_dimension_px,
        },
        "accepted_label_count": len(accepted),
        "inferred_pixel_count": int(np.count_nonzero(inference_mask)),
        "accepted_labels": accepted,
        "rejected_label_counts": rejected,
        "warning": (
            "OCR boxes are semantic occlusion proposals, not observed data. Whole padded "
            "label neighborhoods are filled only when one observed class dominates context."
        ),
    }
    return inferred, inference_mask, report


def infer_ocr_label_occlusions_by_nearest_class(
    values: np.ndarray,
    valid_mask: np.ndarray,
    ocr_tsv_path: Path,
    minimum_confidence: float = 75.0,
    label_padding_px: int = 18,
    context_radius_px: int = 32,
    maximum_propagation_distance_px: float = 28.0,
    minimum_distance_margin_px: float = 1.5,
    maximum_label_dimension_px: int = 260,
    additional_ocr_tsv_paths: Sequence[Path] = (),
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Propagate nearby observed classes through OCR label occlusions.

    Unlike a dominant-class fill, this method can preserve a class boundary that
    passes underneath a city name. Each missing pixel independently inherits its
    nearest observed class, and pixels too far from evidence or nearly equidistant
    between competing classes remain unfilled.
    """

    ocr_tsv_paths = [ocr_tsv_path, *additional_ocr_tsv_paths]
    for path in ocr_tsv_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing OCR TSV: {path}")
    if context_radius_px < 1 or maximum_propagation_distance_px <= 0:
        raise ValueError("OCR propagation context and maximum distance must be positive")
    if minimum_distance_margin_px < 0:
        raise ValueError("OCR propagation distance margin cannot be negative")

    proposals = np.zeros(values.shape, dtype=np.uint8)
    conflicts = np.zeros(values.shape, dtype=bool)
    accepted = []
    rejected = {
        "low_confidence_or_nonword": 0,
        "too_large": 0,
        "outside_valid_map": 0,
        "no_nodata": 0,
        "no_observed_context": 0,
        "no_safe_pixels": 0,
    }

    for current_ocr_tsv_path in ocr_tsv_paths:
        with current_ocr_tsv_path.open(newline="") as handle:
            rows = csv.DictReader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
            for row in rows:
                text = str(row.get("text", "")).strip()
                try:
                    confidence = float(row.get("conf", "-1"))
                    x = int(row.get("left", "0"))
                    y = int(row.get("top", "0"))
                    width = int(row.get("width", "0"))
                    height = int(row.get("height", "0"))
                except ValueError:
                    rejected["low_confidence_or_nonword"] += 1
                    continue
                letters = re.sub(r"[^A-Za-z]", "", text)
                if confidence < minimum_confidence or len(letters) < 3:
                    rejected["low_confidence_or_nonword"] += 1
                    continue
                if (
                    max(width, height) > maximum_label_dimension_px
                    or width <= 0
                    or height <= 0
                ):
                    rejected["too_large"] += 1
                    continue

                x0 = max(0, x - label_padding_px)
                y0 = max(0, y - label_padding_px)
                x1 = min(values.shape[1], x + width + label_padding_px)
                y1 = min(values.shape[0], y + height + label_padding_px)
                target_valid = valid_mask[y0:y1, x0:x1].astype(bool)
                if not np.any(target_valid):
                    rejected["outside_valid_map"] += 1
                    continue
                target = (values[y0:y1, x0:x1] == 0) & target_valid
                if not np.any(target):
                    rejected["no_nodata"] += 1
                    continue

                context_x0 = max(0, x0 - context_radius_px)
                context_y0 = max(0, y0 - context_radius_px)
                context_x1 = min(values.shape[1], x1 + context_radius_px)
                context_y1 = min(values.shape[0], y1 + context_radius_px)
                context = values[context_y0:context_y1, context_x0:context_x1]
                class_ids = np.unique(context[context > 0])
                if len(class_ids) == 0:
                    rejected["no_observed_context"] += 1
                    continue

                distances = np.stack(
                    [
                        distance_transform_edt(context != class_id)
                        for class_id in class_ids
                    ],
                    axis=0,
                )
                best_index = np.argmin(distances, axis=0)
                best_distance = np.take_along_axis(
                    distances, best_index[np.newaxis, ...], axis=0
                )[0]
                if len(class_ids) == 1:
                    distance_margin = np.full(context.shape, np.inf, dtype=np.float64)
                else:
                    nearest_two = np.partition(distances, 1, axis=0)[:2]
                    distance_margin = nearest_two[1] - nearest_two[0]
                nearest_class = class_ids[best_index]

                candidate = np.zeros(context.shape, dtype=bool)
                target_y0 = y0 - context_y0
                target_x0 = x0 - context_x0
                candidate[
                    target_y0 : target_y0 + target.shape[0],
                    target_x0 : target_x0 + target.shape[1],
                ] = target
                safe = (
                    candidate
                    & (best_distance <= maximum_propagation_distance_px)
                    & (distance_margin >= minimum_distance_margin_px)
                )
                if not np.any(safe):
                    rejected["no_safe_pixels"] += 1
                    continue

                proposal_window = proposals[
                    context_y0:context_y1, context_x0:context_x1
                ]
                conflict_window = conflicts[
                    context_y0:context_y1, context_x0:context_x1
                ]
                occupied = safe & (proposal_window > 0)
                conflict_window |= occupied & (proposal_window != nearest_class)
                newly_proposed = safe & ~occupied
                proposal_window[newly_proposed] = nearest_class[newly_proposed]
                if not np.any(newly_proposed):
                    continue
                accepted.append(
                    {
                        "text": text,
                        "ocr_tsv": str(current_ocr_tsv_path),
                        "confidence": confidence,
                        "bbox": [x, y, width, height],
                        "padded_bbox": [x0, y0, x1 - x0, y1 - y0],
                        "context_class_ids": [
                            int(class_id) for class_id in class_ids
                        ],
                        "proposed_pixel_count": int(
                            np.count_nonzero(newly_proposed)
                        ),
                        "ambiguous_pixel_count": int(
                            np.count_nonzero(candidate & ~safe)
                        ),
                    }
                )

    inference_mask = (proposals > 0) & ~conflicts
    inferred = values.copy().astype(np.uint8)
    inferred[inference_mask] = proposals[inference_mask]
    report = {
        "method": "ocr_label_bbox_nearest_observed_class_with_ambiguity_gate",
        "ocr_tsvs": [str(path) for path in ocr_tsv_paths],
        "parameters": {
            "minimum_confidence": minimum_confidence,
            "label_padding_px": label_padding_px,
            "context_radius_px": context_radius_px,
            "maximum_propagation_distance_px": maximum_propagation_distance_px,
            "minimum_distance_margin_px": minimum_distance_margin_px,
            "maximum_label_dimension_px": maximum_label_dimension_px,
        },
        "accepted_label_count": len(accepted),
        "inferred_pixel_count": int(np.count_nonzero(inference_mask)),
        "conflict_pixel_count": int(np.count_nonzero(conflicts)),
        "inferred_pixels_by_class_id": {
            str(int(class_id)): int(
                np.count_nonzero(inference_mask & (proposals == class_id))
            )
            for class_id in np.unique(proposals[proposals > 0])
        },
        "accepted_labels": accepted,
        "rejected_label_counts": rejected,
        "warning": (
            "OCR boxes identify likely label and nearby city-symbol occlusions. Filled "
            "pixels inherit the nearest observed class; distant and boundary-ambiguous "
            "pixels remain transparent, and every inferred pixel remains separately masked."
        ),
    }
    return inferred, inference_mask, report


def _preview(values: np.ndarray, categories: Sequence[Dict[str, object]]) -> np.ndarray:
    rgba = np.zeros((*values.shape, 4), dtype=np.uint8)
    for class_id, category in enumerate(categories, 1):
        color = category.get("display_rgb", category.get("legend_rgb", [255, 0, 255]))
        if isinstance(color, list) and color and isinstance(color[0], list):
            color = color[0]
        selected = values == class_id
        rgba[selected, :3] = np.asarray(color[:3], dtype=np.uint8)
        rgba[selected, 3] = 230
    return rgba


def infer_categorical_run(
    run_dir: Path,
    output_dir: Optional[Path] = None,
    max_gap_radius_px: int = 6,
    max_component_area_px: int = 1_200,
    max_component_dimension_px: int = 120,
    ring_radius_px: int = 3,
    minimum_ring_pixels: int = 12,
    minimum_dominance: float = 0.98,
    ocr_tsv_path: Optional[Path] = None,
    ocr_minimum_confidence: float = 75.0,
    ocr_label_padding_px: int = 18,
    ocr_context_radius_px: int = 32,
    ocr_maximum_propagation_distance_px: float = 28.0,
    ocr_minimum_distance_margin_px: float = 1.5,
    ocr_additional_tsv_paths: Sequence[Path] = (),
) -> Dict[str, object]:
    """Create separate inferred source/Web-Mercator artifacts for a reviewed run."""

    run_dir = run_dir.resolve()
    output_dir = (output_dir or (run_dir / "inference")).resolve()
    manifest_path = run_dir / "extraction.json"
    plan_path = run_dir / "plan.snapshot.json"
    if not manifest_path.exists() or not plan_path.exists():
        raise FileNotFoundError("Inference needs extraction.json and plan.snapshot.json")
    manifest = json.loads(manifest_path.read_text())
    plan = json.loads(plan_path.read_text())
    alignment_path = Path(plan["alignment"])
    alignment = json.loads(alignment_path.read_text())
    if alignment.get("alignment_mode") == "assisted":
        transform = {
            "projection": "assisted_reference_crs",
            "projection_crs": alignment["reference"]["crs"],
            "transform_model": alignment["transform_model"],
            "reference_to_source_matrix": alignment["reference_to_source_matrix"],
        }
        if "web_mercator_correction" in alignment:
            transform["web_mercator_correction"] = alignment["web_mercator_correction"]
    else:
        transform = alignment["best"]
    reference_root = Path(plan.get("reference", "reference/census-2025"))
    state, _ = load_california(reference_root)
    valid_mask = np.asarray(Image.open(run_dir / "source-state-mask.png")) > 0
    plan_layers = {str(layer["id"]): layer for layer in plan.get("layers", [])}
    output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    supported_kinds = {"categorical", "grayscale_categorical", "gradient_categorical"}

    for layer_report in manifest.get("layers", []):
        layer_id = str(layer_report["id"])
        definition = plan_layers.get(layer_id)
        if not isinstance(definition, dict) or definition.get("kind") not in supported_kinds:
            continue
        source_path = run_dir / layer_id / "source-class-id.png"
        if not source_path.exists():
            continue
        source_values = np.asarray(Image.open(source_path), dtype=np.uint8)
        categories = definition.get("categories", [])
        inferred, inference_mask, report = infer_small_categorical_gaps(
            source_values,
            valid_mask,
            len(categories),
            max_gap_radius_px=max_gap_radius_px,
            max_component_area_px=max_component_area_px,
            max_component_dimension_px=max_component_dimension_px,
            ring_radius_px=ring_radius_px,
            minimum_ring_pixels=minimum_ring_pixels,
            minimum_dominance=minimum_dominance,
        )
        morphology_mask = inference_mask.copy()
        text_report = None
        text_mask = np.zeros(source_values.shape, dtype=bool)
        if ocr_tsv_path is not None:
            ocr_inferred, text_mask, text_report = (
                infer_ocr_label_occlusions_by_nearest_class(
                source_values,
                valid_mask,
                ocr_tsv_path,
                minimum_confidence=ocr_minimum_confidence,
                label_padding_px=ocr_label_padding_px,
                context_radius_px=ocr_context_radius_px,
                maximum_propagation_distance_px=(
                    ocr_maximum_propagation_distance_px
                ),
                minimum_distance_margin_px=ocr_minimum_distance_margin_px,
                additional_ocr_tsv_paths=ocr_additional_tsv_paths,
                )
            )
            new_text_mask = text_mask & ~inference_mask
            inferred[new_text_mask] = ocr_inferred[new_text_mask]
            inference_mask |= text_mask
            report["morphological_inferred_pixel_count"] = int(
                np.count_nonzero(morphology_mask)
            )
            report["method"] = (
                "morphological_closure_plus_ocr_label_nearest_observed_class"
            )
            report["ocr_label_inference"] = text_report
            report["inferred_pixel_count"] = int(np.count_nonzero(inference_mask))
            report["warning"] = (
                "Inferred pixels are not observed source data. Narrow gaps require a "
                "dominant surrounding class; OCR label gaps inherit the nearest observed "
                "class while boundary-ambiguous pixels remain transparent. Every inferred "
                "pixel remains separately masked."
            )
        layer_dir = output_dir / layer_id
        layer_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(inferred, mode="L").save(
            layer_dir / "source-class-id-inferred.png", optimize=True
        )
        Image.fromarray(inference_mask.astype(np.uint8) * 255, mode="L").save(
            layer_dir / "source-inference-mask.png", optimize=True
        )
        Image.fromarray(morphology_mask.astype(np.uint8) * 255, mode="L").save(
            layer_dir / "source-inference-mask-morphological.png", optimize=True
        )
        if ocr_tsv_path is not None:
            Image.fromarray(text_mask.astype(np.uint8) * 255, mode="L").save(
                layer_dir / "source-inference-mask-ocr-labels.png", optimize=True
            )
        Image.fromarray(_preview(inferred, categories), mode="RGBA").save(
            layer_dir / "source-preview-inferred.png", optimize=True
        )
        warped, warp_report = warp_classified_to_web_mercator(
            inferred, state, transform, source_values.shape
        )
        warped_mask, _ = warp_classified_to_web_mercator(
            inference_mask.astype(np.uint8), state, transform, source_values.shape
        )
        warped_text_mask = None
        if ocr_tsv_path is not None:
            warped_text_mask, _ = warp_classified_to_web_mercator(
                text_mask.astype(np.uint8), state, transform, source_values.shape
            )
        Image.fromarray(warped, mode="L").save(
            layer_dir / "web-mercator-class-id-inferred.png", optimize=True
        )
        Image.fromarray((warped_mask > 0).astype(np.uint8) * 255, mode="L").save(
            layer_dir / "web-mercator-inference-mask.png", optimize=True
        )
        if warped_text_mask is not None:
            Image.fromarray((warped_text_mask > 0).astype(np.uint8) * 255, mode="L").save(
                layer_dir / "web-mercator-inference-mask-ocr-labels.png", optimize=True
            )
        Image.fromarray(_preview(warped, categories), mode="RGBA").save(
            layer_dir / "web-mercator-preview-inferred.png", optimize=True
        )
        report.update(
            {
                "layer_id": layer_id,
                "source_class_id_sha256": _sha256(source_path),
                "warp": warp_report,
            }
        )
        reports.append(report)

    result = {
        "schema_version": 1,
        "status": "needs_visual_review",
        "dataset_id": manifest["dataset_id"],
        "source_run": str(run_dir),
        "extraction_manifest_sha256": _sha256(manifest_path),
        "alignment": str(alignment_path),
        "alignment_sha256": _sha256(alignment_path),
        "layers": reports,
        "warning": (
            "Observed class rasters remain authoritative. Every inferred pixel is stored "
            "in a separate mask and must be disclosed or disabled in downstream products."
        ),
    }
    (output_dir / "inference.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
