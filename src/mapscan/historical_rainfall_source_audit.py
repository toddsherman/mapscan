"""Source-only distinguishability audit for the historical rainfall GIF.

The historical precipitation map has a 35-cell numeric legend, but an indexed
GIF can collapse several original colors onto the same palette indices.  This
audit examines the pristine GIF representation itself: container structure,
raw palette indices, translation-invariant dither topology, OCR, and connected
polygon topology.  It never assigns semantic classes.  If two legend meanings
remain observationally equivalent, the correct result is a durable blocker.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.stats import chi2_contingency

from .automatic_categorical_extraction import (
    OCRWord,
    _automatic_swatch_sequence,
    _detect_complete_legend,
    _run_tesseract_ocr,
)
from .dither_texture_classifier import build_dither_texture_model


SCHEMA_VERSION = "mapscan.historical-rainfall-source-ambiguity-audit.v1"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _skip_subblocks(data: bytes, position: int) -> tuple[int, int]:
    payload_bytes = 0
    while True:
        if position >= len(data):
            raise ValueError("truncated GIF sub-block sequence")
        length = data[position]
        position += 1
        if length == 0:
            return position, payload_bytes
        if position + length > len(data):
            raise ValueError("truncated GIF sub-block payload")
        payload_bytes += length
        position += length


def inspect_gif_container(data: bytes) -> dict[str, Any]:
    """Parse enough of GIF89a to expose every possible metadata channel."""

    if len(data) < 14 or data[:6] not in {b"GIF87a", b"GIF89a"}:
        raise ValueError("source is not a complete GIF stream")
    packed = data[10]
    global_palette_size = 2 ** ((packed & 0x07) + 1) if packed & 0x80 else 0
    position = 13 + global_palette_size * 3
    image_count = 0
    extensions: list[dict[str, int]] = []
    compressed_image_bytes = 0
    while position < len(data):
        marker = data[position]
        if marker == 0x3B:
            if position != len(data) - 1:
                raise ValueError("GIF contains trailing bytes after its trailer")
            return {
                "version": data[:6].decode("ascii"),
                "global_palette_size": global_palette_size,
                "image_descriptor_count": image_count,
                "extension_blocks": extensions,
                "compressed_image_payload_bytes": compressed_image_bytes,
                "trailer_offset": position,
            }
        if marker == 0x21:
            if position + 1 >= len(data):
                raise ValueError("truncated GIF extension")
            label = data[position + 1]
            position, payload_bytes = _skip_subblocks(data, position + 2)
            extensions.append(
                {"label": int(label), "payload_bytes": int(payload_bytes)}
            )
            continue
        if marker == 0x2C:
            if position + 10 > len(data):
                raise ValueError("truncated GIF image descriptor")
            descriptor_packed = data[position + 9]
            position += 10
            if descriptor_packed & 0x80:
                local_size = 2 ** ((descriptor_packed & 0x07) + 1)
                position += local_size * 3
            if position >= len(data):
                raise ValueError("GIF image lacks an LZW code size")
            position += 1
            position, payload_bytes = _skip_subblocks(data, position)
            image_count += 1
            compressed_image_bytes += payload_bytes
            continue
        raise ValueError(f"unexpected GIF block marker 0x{marker:02x}")
    raise ValueError("GIF trailer is missing")


def _index_counts(values: np.ndarray, alphabet: Sequence[int]) -> np.ndarray:
    return np.asarray([np.count_nonzero(values == item) for item in alphabet], dtype=np.int64)


def _patch_support(values: np.ndarray, size: int) -> set[bytes]:
    if min(values.shape) < size:
        return set()
    return {
        values[y : y + size, x : x + size].tobytes()
        for y in range(values.shape[0] - size + 1)
        for x in range(values.shape[1] - size + 1)
    }


def _pairwise_patch_overlap(
    class_ids: Sequence[int], patches: Sequence[np.ndarray], size: int
) -> list[dict[str, Any]]:
    supports = [_patch_support(patch, size) for patch in patches]
    reports: list[dict[str, Any]] = []
    for first in range(len(supports)):
        for second in range(first + 1, len(supports)):
            intersection = supports[first] & supports[second]
            union = supports[first] | supports[second]
            reports.append(
                {
                    "first_class_id": int(class_ids[first]),
                    "second_class_id": int(class_ids[second]),
                    "first_support_size": len(supports[first]),
                    "second_support_size": len(supports[second]),
                    "intersection_size": len(intersection),
                    "jaccard": len(intersection) / max(len(union), 1),
                    "first_covered_by_second": len(intersection)
                    / max(len(supports[first]), 1),
                    "second_covered_by_first": len(intersection)
                    / max(len(supports[second]), 1),
                }
            )
    return reports


def _numeric_ocr_candidates(
    words: Iterable[OCRWord], legend_boxes: Sequence[tuple[int, int, int, int]]
) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for word in words:
        if not re.search(r"\d", word.text):
            continue
        intersects_legend = any(
            word.right > x
            and word.left < x + width
            and word.top + word.height > y
            and word.top < y + height
            for x, y, width, height in legend_boxes
        )
        normalized = word.text.strip().replace(",", "")
        exact_numeric = bool(re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", normalized))
        reports.append(
            {
                "text": word.text,
                "confidence": word.confidence,
                "bbox": [word.left, word.top, word.width, word.height],
                "intersects_automatically_detected_legend": intersects_legend,
                "exact_numeric_token": exact_numeric,
                "qualifies_as_high_confidence_nonlegend_contour_value": bool(
                    exact_numeric and word.confidence >= 75.0 and not intersects_legend
                ),
            }
        )
    return reports


def _write_swatch_preview(
    output_path: Path,
    raw_indices: np.ndarray,
    palette: np.ndarray,
    entries: Sequence[Any],
    class_ids: Sequence[int],
) -> None:
    scale = 8
    pad = 10
    cell_width = 34 * scale + 2 * pad
    cell_height = 18 * scale + 42
    canvas = Image.new("RGB", (cell_width * len(class_ids), cell_height), "white")
    draw = ImageDraw.Draw(canvas)
    for column, class_id in enumerate(class_ids):
        entry = entries[class_id - 1]
        x, y, width, height = entry.swatch_bbox
        patch = palette[raw_indices[y : y + height, x : x + width]]
        rendered = Image.fromarray(patch.astype(np.uint8), mode="RGB").resize(
            (width * scale, height * scale), Image.Resampling.NEAREST
        )
        left = column * cell_width + pad
        canvas.paste(rendered, (left, 30))
        draw.text((left, 8), f"class {class_id}: {entry.label}", fill="black")
        draw.text(
            (left, 30 + height * scale + 6),
            "/".join(str(value) for value in sorted(np.unique(raw_indices[y:y+height, x:x+width]))),
            fill="black",
        )
    canvas.save(output_path, optimize=True)


def _write_topology_preview(
    output_path: Path,
    source_rgb: np.ndarray,
    ambiguous_mask: np.ndarray,
    lower_mask: np.ndarray,
    upper_mask: np.ndarray,
    bridge_components: Sequence[Mapping[str, Any]],
    labels: np.ndarray,
) -> None:
    preview = source_rgb.copy()
    preview[ambiguous_mask] = (
        0.25 * preview[ambiguous_mask] + 0.75 * np.asarray((255, 0, 220))
    ).astype(np.uint8)
    preview[lower_mask] = (0, 255, 255)
    preview[upper_mask] = (0, 255, 80)
    for item in bridge_components:
        component = labels == int(item["component_id"])
        contours, _ = cv2.findContours(
            component.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(preview, contours, -1, (255, 230, 0), 3)
    Image.fromarray(preview, mode="RGB").save(output_path, optimize=True)


def run_historical_rainfall_source_ambiguity_audit(
    source_path: Path, output_dir: Path
) -> dict[str, Any]:
    """Write a deterministic source-only blocker or distinguishability report."""

    source_path = source_path.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise ValueError("historical rainfall audit requires a fresh output directory")
    output_dir.mkdir(parents=True)
    source_bytes = source_path.read_bytes()
    container = inspect_gif_container(source_bytes)
    with Image.open(source_path) as image:
        if image.mode != "P" or int(getattr(image, "n_frames", 1)) != 1:
            raise ValueError("historical rainfall audit requires one indexed GIF frame")
        raw_indices = np.asarray(image).copy()
        palette_values = image.getpalette()
        if palette_values is None:
            raise ValueError("indexed rainfall GIF has no palette")
        palette = np.asarray(palette_values, dtype=np.uint8).reshape((-1, 3))
        source_rgb = palette[raw_indices]

    words, tsv_text, tesseract_version = _run_tesseract_ocr(source_path)
    swatches, swatch_diagnostics = _automatic_swatch_sequence(source_rgb)
    entries, axis, step, label_diagnostics = _detect_complete_legend(
        source_rgb, swatches, words, output_dir
    )
    if len(entries) != 35:
        raise ValueError("source-only audit did not recover all 35 legend semantics")
    numeric_labels = [float(entry.label) for entry in entries]
    if len(set(numeric_labels)) != len(numeric_labels):
        raise ValueError("historical rainfall legend contains duplicate OCR semantics")

    boxes = [tuple(map(int, entry.swatch_bbox)) for entry in entries]
    patches = [
        raw_indices[y : y + height, x : x + width]
        for x, y, width, height in boxes
    ]
    texture_model = build_dither_texture_model(source_rgb, boxes)
    center_distances = np.asarray(texture_model.pairwise_distances)
    class_records: list[dict[str, Any]] = []
    alphabet_groups: dict[frozenset[int], list[int]] = defaultdict(list)
    for class_id, (entry, patch) in enumerate(zip(entries, patches), 1):
        values, counts = np.unique(patch, return_counts=True)
        alphabet = frozenset(int(value) for value in values)
        alphabet_groups[alphabet].append(class_id)
        class_records.append(
            {
                "class_id": class_id,
                "label": entry.label,
                "swatch_bbox": list(entry.swatch_bbox),
                "raw_index_crop_sha256": _sha256_bytes(patch.tobytes()),
                "raw_palette_indices": [
                    {
                        "index": int(value),
                        "rgb": [int(channel) for channel in palette[value]],
                        "count": int(count),
                        "fraction": int(count) / patch.size,
                    }
                    for value, count in zip(values, counts)
                ],
            }
        )

    collision_groups: list[dict[str, Any]] = []
    indistinguishable_groups: list[dict[str, Any]] = []
    for alphabet, class_ids in sorted(
        alphabet_groups.items(), key=lambda item: min(item[1])
    ):
        if len(class_ids) < 2:
            continue
        selected = [patches[class_id - 1] for class_id in class_ids]
        sorted_alphabet = sorted(alphabet)
        contingency = np.asarray(
            [_index_counts(patch, sorted_alphabet) for patch in selected]
        )
        statistic, p_value, degrees, expected = chi2_contingency(contingency)
        pair_centers = [
            float(center_distances[first - 1, second - 1])
            for offset, first in enumerate(class_ids)
            for second in class_ids[offset + 1 :]
        ]
        overlap_2 = _pairwise_patch_overlap(class_ids, selected, 2)
        overlap_3 = _pairwise_patch_overlap(class_ids, selected, 3)
        record = {
            "class_ids": class_ids,
            "labels": [entries[class_id - 1].label for class_id in class_ids],
            "shared_raw_palette_alphabet": sorted_alphabet,
            "raw_index_contingency": contingency.tolist(),
            "multinomial_equal_distribution_chi_square": float(statistic),
            "multinomial_equal_distribution_degrees_of_freedom": int(degrees),
            "multinomial_equal_distribution_p_value": float(p_value),
            "maximum_lab_texture_center_distance": max(pair_centers),
            "two_by_two_patch_support_overlap": overlap_2,
            "three_by_three_patch_support_overlap": overlap_3,
            "source_indistinguishable": bool(
                p_value >= 0.05
                and max(pair_centers) < 1.0
                and min(item["jaccard"] for item in overlap_2) >= 0.95
            ),
        }
        collision_groups.append(record)
        if record["source_indistinguishable"]:
            indistinguishable_groups.append(record)

    triple = next(
        (
            group
            for group in indistinguishable_groups
            if len(group["class_ids"]) >= 3
        ),
        None,
    )
    topology: dict[str, Any] = {
        "status": "not_applicable",
        "bridge_component_count": 0,
        "bridge_components": [],
    }
    topology_preview_path: Path | None = None
    if triple is not None:
        class_ids = list(triple["class_ids"])
        lower_id = min(class_ids) - 1
        upper_id = max(class_ids) + 1
        ambiguous_indices = list(triple["shared_raw_palette_alphabet"])
        legend_mask = np.zeros(raw_indices.shape, dtype=bool)
        for x, y, width, height in boxes:
            legend_mask[y : y + height, x : x + width] = True
        ambiguous_mask = np.isin(raw_indices, ambiguous_indices) & ~legend_mask
        lower_indices_all = [
            item["index"] for item in class_records[lower_id - 1]["raw_palette_indices"]
        ]
        upper_indices_all = [
            item["index"] for item in class_records[upper_id - 1]["raw_palette_indices"]
        ]
        other_lower_indices = {
            item["index"]
            for class_index, record in enumerate(class_records, 1)
            if class_index != lower_id
            for item in record["raw_palette_indices"]
        }
        other_upper_indices = {
            item["index"]
            for class_index, record in enumerate(class_records, 1)
            if class_index != upper_id
            for item in record["raw_palette_indices"]
        }
        # Anchor contact is counted only through palette indices unique to that
        # legend semantic.  This prevents a shared dither color (for example
        # index 227 in both class 3 and classes 1/2) from manufacturing a
        # topology constraint.
        lower_indices = sorted(set(lower_indices_all) - other_lower_indices)
        upper_indices = sorted(set(upper_indices_all) - other_upper_indices)
        if not lower_indices or not upper_indices:
            raise ValueError("ordered topology anchors lack source-exclusive indices")
        lower_mask = np.isin(raw_indices, lower_indices) & ~legend_mask
        upper_mask = np.isin(raw_indices, upper_indices) & ~legend_mask
        component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(
            ambiguous_mask.astype(np.uint8), 8
        )
        dilating = np.ones((7, 7), dtype=np.uint8)
        components: list[dict[str, Any]] = []
        bridges: list[dict[str, Any]] = []
        for component_id in range(1, component_count):
            area = int(stats[component_id, cv2.CC_STAT_AREA])
            if area < 50:
                continue
            component = labels == component_id
            ring = (
                cv2.dilate(component.astype(np.uint8), dilating) > 0
            ) & ~component
            lower_count = int(np.count_nonzero(ring & lower_mask))
            upper_count = int(np.count_nonzero(ring & upper_mask))
            item = {
                "component_id": component_id,
                "area": area,
                "centroid": [float(value) for value in centroids[component_id]],
                "lower_anchor_class_id": lower_id,
                "lower_anchor_contact_pixel_count": lower_count,
                "upper_anchor_class_id": upper_id,
                "upper_anchor_contact_pixel_count": upper_count,
                "touches_both_order_anchors": lower_count >= 5 and upper_count >= 5,
            }
            components.append(item)
            if item["touches_both_order_anchors"]:
                bridges.append(item)
        topology = {
            "status": "nonunique",
            "method": (
                "connected raw-index components after excluding every automatic legend "
                "swatch; three-pixel source-only contact ring"
            ),
            "ambiguous_class_ids": class_ids,
            "ambiguous_labels": [entries[value - 1].label for value in class_ids],
            "lower_anchor_class_id": lower_id,
            "lower_anchor_label": entries[lower_id - 1].label,
            "lower_anchor_source_exclusive_palette_indices": lower_indices,
            "upper_anchor_class_id": upper_id,
            "upper_anchor_label": entries[upper_id - 1].label,
            "upper_anchor_source_exclusive_palette_indices": upper_indices,
            "component_count_at_least_50_pixels": len(components),
            "bridge_component_count": len(bridges),
            "bridge_components": bridges,
            "conclusion": (
                "At least one single connected source appearance touches both ordered "
                "anchors; without a contour value label, topology cannot choose which "
                "intermediate semantic value it represents."
            ),
        }
        topology_preview_path = output_dir / "source-only-topology-ambiguity.png"
        _write_topology_preview(
            topology_preview_path,
            source_rgb,
            ambiguous_mask,
            lower_mask,
            upper_mask,
            bridges,
            labels,
        )

    legend_boxes = [
        tuple(map(int, entry.swatch_bbox)) for entry in entries
    ] + [tuple(map(int, entry.label_bbox)) for entry in entries]
    ocr_candidates = _numeric_ocr_candidates(words, legend_boxes)
    high_confidence_contours = [
        item
        for item in ocr_candidates
        if item["qualifies_as_high_confidence_nonlegend_contour_value"]
    ]

    tsv_path = output_dir / "source-full-page-ocr.tsv"
    tsv_path.write_text(tsv_text)
    swatch_preview_path = output_dir / "ambiguous-legend-raw-index-crops.png"
    preview_classes = sorted(
        {class_id for group in indistinguishable_groups for class_id in group["class_ids"]}
    )
    _write_swatch_preview(
        swatch_preview_path, raw_indices, palette, entries, preview_classes
    )
    ocr_path = output_dir / "source-numeric-ocr-candidates.json"
    ocr_path.write_text(json.dumps(ocr_candidates, indent=2) + "\n")

    all_distinguishable = not indistinguishable_groups
    topology_resolves = bool(
        all_distinguishable
        or (
            topology.get("status") == "unique"
            and not topology.get("bridge_component_count")
        )
    )
    contour_labels_resolve = bool(high_confidence_contours)
    passed = all_distinguishable or (topology_resolves and contour_labels_resolve)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "source_distinguishable" if passed else "blocked_source_indistinguishable",
        "source": {
            "path": str(source_path),
            "sha256": _sha256(source_path),
            "byte_count": len(source_bytes),
            "width": int(raw_indices.shape[1]),
            "height": int(raw_indices.shape[0]),
            "mode": "P",
        },
        "gif_container": container,
        "source_authority": {
            "original_source_pixels_only": True,
            "manual_inputs_used": False,
            "prior_extraction_used": False,
            "mapbox_used": False,
        },
        "legend": {
            "automatic_swatch_selection": swatch_diagnostics,
            "swatch_axis_x": axis,
            "row_step_px": step,
            "tesseract_version": tesseract_version,
            "class_count": len(entries),
            "numeric_labels": numeric_labels,
            "label_diagnostics": label_diagnostics,
            "classes": class_records,
        },
        "raw_palette_collision_groups": collision_groups,
        "source_indistinguishable_groups": [
            {
                key: value
                for key, value in group.items()
                if key
                in {
                    "class_ids",
                    "labels",
                    "shared_raw_palette_alphabet",
                    "multinomial_equal_distribution_p_value",
                    "maximum_lab_texture_center_distance",
                    "two_by_two_patch_support_overlap",
                    "three_by_three_patch_support_overlap",
                    "source_indistinguishable",
                }
            }
            for group in indistinguishable_groups
        ],
        "source_only_order_topology": topology,
        "numeric_contour_ocr": {
            "candidate_count": len(ocr_candidates),
            "high_confidence_nonlegend_contour_value_count": len(
                high_confidence_contours
            ),
            "high_confidence_nonlegend_contour_values": high_confidence_contours,
            "artifact": ocr_path.name,
        },
        "gates": {
            "all_35_legend_semantics_readable": len(entries) == 35,
            "single_indexed_source_frame": (
                container["image_descriptor_count"] == 1
                and not container["extension_blocks"]
            ),
            "every_semantic_class_source_distinguishable": all_distinguishable,
            "ordered_topology_uniquely_resolves_collapsed_classes": topology_resolves,
            "source_contour_values_resolve_collapsed_classes": contour_labels_resolve,
        },
        "official_extraction_attempt_allowed": passed,
        "blocker": (
            None
            if passed
            else (
                "The pristine indexed GIF contains observationally equivalent legend "
                "semantics. Raw palette indices, local dither topology/frequency, OCR, "
                "and order topology do not uniquely distinguish every one of 35 classes."
            )
        ),
        "artifacts": [],
    }
    report_path = output_dir / "source-ambiguity-audit.json"
    report["artifacts"] = [
        {
            "path": str(path),
            "sha256": _sha256(path),
        }
        for path in (
            tsv_path,
            ocr_path,
            swatch_preview_path,
            *([topology_preview_path] if topology_preview_path is not None else []),
        )
    ]
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    markdown_path = output_dir / "SOURCE_AMBIGUITY_AUDIT.md"
    markdown_path.write_text(
        "# Historical rainfall source ambiguity audit\n\n"
        f"- Source SHA-256: `{report['source']['sha256']}`\n"
        f"- Automatically read legend semantics: `{len(entries)}`\n"
        f"- Indexed GIF frames: `{container['image_descriptor_count']}`\n"
        f"- GIF extension/metadata blocks: `{len(container['extension_blocks'])}`\n"
        f"- Source-indistinguishable groups: "
        f"`{[group['labels'] for group in indistinguishable_groups]}`\n"
        f"- Order-topology bridge components: `{topology.get('bridge_component_count', 0)}`\n"
        f"- High-confidence nonlegend numeric contour values: "
        f"`{len(high_confidence_contours)}`\n"
        f"- Official extraction attempt allowed: `{passed}`\n\n"
        f"{report['blocker'] or 'Every class is source-distinguishable.'}\n"
    )
    return report
