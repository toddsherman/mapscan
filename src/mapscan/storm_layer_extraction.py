"""Strict no-human extraction for the ARkStorm overlapping-layer map.

The source encodes one mutually-exclusive grayscale precipitation layer and
three independent chromatic overlays.  Chromatic pixels obscure the grayscale
base; the extractor therefore records those base pixels as occluded/unknown
and never propagates a neighboring precipitation class beneath an overlay.

Only the pristine source adapter, an accepted automatic Mapbox alignment, and
the pinned Mapbox land/water reference are accepted.  Every channel is warped
to Mapbox, reconstructed back to source, checked globally and geographically,
and replayed a second time byte-for-byte before acceptance.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .automatic_alignment_loop import load_pinned_mapbox_reference
from .automatic_categorical_extraction import (
    OCRWord,
    _load_accepted_alignment,
    _reference_to_source_remap,
    _run_tesseract_ocr,
    _source_data_mask,
    _source_to_reference,
    _source_to_reference_remap,
)
from .experiment_log import NoHumanExperimentLog, automatic_provenance
from .extraction import classify_grayscale, infer_sparse_chroma_overlays
from .source_working_raster import WorkingRasterArtifact, load_working_raster_artifact


SCHEMA_VERSION = "mapscan.storm-overlapping-layer-extraction.v1"
PRODUCER = "mapscan.storm_layer_extraction"
FORBIDDEN_PATH_TOKENS = (
    "automatic-alignment-orphaned-race",
    "county.png",
    "census",
    "manual",
    "legacy",
    "landslide-extract",
)
PRECIPITATION_IDS = ("4-to-8-inches", "8-to-16-inches", "16-to-25-7-inches")
OVERLAY_IDS = (
    "landslide-susceptibility",
    "maximum-wind-speed",
    "predicted-flooding",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "sha256": _sha256(path),
        "byte_count": path.stat().st_size,
    }


def _save_rgb(path: Path, values: np.ndarray) -> None:
    Image.fromarray(values.astype(np.uint8), mode="RGB").save(path, optimize=True)


def _save_mask(path: Path, values: np.ndarray) -> None:
    Image.fromarray(values.astype(np.uint8) * 255, mode="L").save(path, optimize=True)


def _save_ids(path: Path, values: np.ndarray) -> None:
    Image.fromarray(values.astype(np.uint8), mode="L").save(path, optimize=True)


def _assert_clean_path(path: Path, kind: str) -> None:
    normalized = str(path.resolve()).lower()
    matched = next((token for token in FORBIDDEN_PATH_TOKENS if token in normalized), None)
    if matched:
        raise ValueError(f"{kind} path contains forbidden no-human evidence: {matched}")


@dataclass(frozen=True)
class StormLegendEntry:
    layer_id: str
    label: str
    rgb: tuple[int, int, int]
    swatch_bbox: tuple[int, int, int, int]
    label_bbox: tuple[int, int, int, int]
    ocr_confidence: float

    @property
    def gray(self) -> int:
        return int(round(0.299 * self.rgb[0] + 0.587 * self.rgb[1] + 0.114 * self.rgb[2]))


@dataclass(frozen=True)
class StormExtractionConfig:
    required_replay_count: int = 2
    minimum_legend_confidence: float = 65.0
    overlay_minimum_chroma: float = 10.0
    overlay_coefficient_threshold: float = 0.32
    overlay_maximum_residual: float = 6.0
    overlay_complexity_penalty: float = 1.4
    precipitation_smoothing_radius: int = 5
    precipitation_maximum_gray_distance: float = 20.0
    precipitation_maximum_chroma: float = 13.0
    precipitation_background_cutoff: int = 245
    minimum_precipitation_roundtrip_fraction: float = 0.94
    minimum_overlay_roundtrip_iou: float = 0.80
    minimum_occluded_roundtrip_iou: float = 0.80
    minimum_composite_roundtrip_fraction: float = 0.90
    geographic_rows: int = 4
    geographic_columns: int = 3
    minimum_geographic_match_fraction: float = 0.88
    minimum_passing_geographic_cells: int = 8

    def __post_init__(self) -> None:
        if self.required_replay_count != 2:
            raise ValueError("storm extraction requires exactly two fixed-point passes")
        for value in (
            self.minimum_precipitation_roundtrip_fraction,
            self.minimum_overlay_roundtrip_iou,
            self.minimum_occluded_roundtrip_iou,
            self.minimum_composite_roundtrip_fraction,
            self.minimum_geographic_match_fraction,
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError("fraction gates must be between zero and one")
        if self.minimum_passing_geographic_cells < 1:
            raise ValueError("minimum_passing_geographic_cells must be positive")


@dataclass(frozen=True)
class StormExtractionIteration:
    iteration: int
    decision: str
    scores: Mapping[str, Any]
    gates: Mapping[str, Any]
    report_path: Path
    artifact_paths: tuple[Path, ...]


@dataclass(frozen=True)
class StormExtractionResult:
    status: str
    stop_reason: str
    iterations: tuple[StormExtractionIteration, ...]
    accepted_extraction_path: Path | None
    artifact_paths: tuple[Path, ...]

    @property
    def accepted(self) -> Path | None:
        return self.accepted_extraction_path


def _validate_inputs(
    source_adapter_manifest_path: Path,
    accepted_alignment_path: Path,
    mapbox_manifest_path: Path,
    experiment_log: NoHumanExperimentLog,
) -> tuple[WorkingRasterArtifact, Any, dict[str, Any]]:
    for path, kind in (
        (source_adapter_manifest_path, "source adapter"),
        (accepted_alignment_path, "accepted alignment"),
        (mapbox_manifest_path, "Mapbox manifest"),
    ):
        _assert_clean_path(path, kind)
    working = load_working_raster_artifact(source_adapter_manifest_path)
    if working.source_path.suffix.lower() != ".png":
        raise ValueError("storm overlapping-layer extraction requires the source PNG")
    authority = working.manifest.get("authority", {})
    if authority != {
        "manual_input_used": False,
        "original_source_authoritative": True,
        "prior_alignment_used": False,
        "prior_extraction_used": False,
    }:
        raise ValueError("source-clean authority record is not pristine")
    if experiment_log.data.get("map_id") != "landslide":
        raise ValueError("storm extractor requires the landslide experiment log")
    if experiment_log.data["source"].get("sha256") != working.source_sha256:
        raise ValueError("experiment and source-clean original hashes disagree")
    if experiment_log.data["source"].get("source_type") != "overlapping_chromatic_and_grayscale":
        raise ValueError("storm source data model is not overlapping_chromatic_and_grayscale")
    if experiment_log.data["extraction"]["iterations"]:
        raise ValueError("storm extraction log already contains attempts")

    reference = load_pinned_mapbox_reference(mapbox_manifest_path)
    accepted_count = experiment_log.data["alignment"]["accepted_automatic_iteration_count"]
    if accepted_count is None:
        raise ValueError("extraction requires an accepted automatic alignment")
    alignment = _load_accepted_alignment(
        accepted_alignment_path,
        working.working_raster_path,
        reference.grid,
        reference.pin,
        accepted_iteration_count=int(accepted_count),
        map_id=str(experiment_log.data["map_id"]),
        reference_revisions=experiment_log.data.get("mapbox_reference_revisions", []),
        alignment_iterations=experiment_log.data["alignment"]["iterations"],
    )
    return working, reference, alignment


def _line_tokens(words: Sequence[OCRWord]) -> tuple[str, ...]:
    text = " ".join(word.text for word in sorted(words, key=lambda value: value.left))
    return tuple(
        value.replace(",", ".")
        for value in re.findall(r"[a-z]+|\d+(?:[.,]\d+)?", text.lower())
    )


def _contains_all(tokens: Sequence[str], required: Sequence[str]) -> bool:
    available = set(tokens)
    return all(value in available for value in required)


def _legend_matchers() -> tuple[
    tuple[str, str, re.Pattern[str], Callable[[Sequence[str]], bool]], ...
]:
    return (
        (
            PRECIPITATION_IDS[0],
            "4-8 in (102-204 mm)",
            re.compile(r"^4\D*8$"),
            lambda tokens: _contains_all(tokens, ("4", "8", "in", "102", "204", "mm")),
        ),
        (
            PRECIPITATION_IDS[1],
            "8-16 in (204-408 mm)",
            re.compile(r"^8\D*16$"),
            lambda tokens: _contains_all(tokens, ("8", "16", "in", "204", "408", "mm")),
        ),
        (
            PRECIPITATION_IDS[2],
            "16-25.7 in (408-652 mm)",
            re.compile(r"^16\D*25[.,]7$"),
            lambda tokens: (
                _contains_all(tokens, ("16", "in", "408", "652", "mm"))
                and any(value.startswith("25") for value in tokens)
            ),
        ),
        (
            OVERLAY_IDS[0],
            "Landslide susceptibility > 9",
            re.compile(r"^landslide$", re.IGNORECASE),
            lambda tokens: _contains_all(tokens, ("landslide", "susceptibility", "9")),
        ),
        (
            OVERLAY_IDS[1],
            "Maximum wind speed > 60 mile/h (96.6 km/h)",
            re.compile(r"^maximum$", re.IGNORECASE),
            lambda tokens: _contains_all(tokens, ("maximum", "wind", "speed", "60")),
        ),
        (
            OVERLAY_IDS[2],
            "Predicted flooding",
            re.compile(r"^predicted$", re.IGNORECASE),
            lambda tokens: _contains_all(tokens, ("predicted", "flooding")),
        ),
    )


def _cluster_physical_lines(
    candidates: Sequence[tuple[tuple[int, int, int, int], float]],
) -> list[list[tuple[tuple[int, int, int, int], float]]]:
    clusters: list[list[tuple[tuple[int, int, int, int], float]]] = []
    for item in sorted(candidates, key=lambda value: value[0][1]):
        bbox = item[0]
        center = bbox[1] + bbox[3] / 2
        for cluster in clusters:
            other = cluster[0][0]
            other_center = other[1] + other[3] / 2
            if abs(center - other_center) <= 14 and abs(bbox[0] - other[0]) <= 15:
                cluster.append(item)
                break
        else:
            clusters.append([item])
    return clusters


def _swatch_from_line(
    source_rgb: np.ndarray, bbox: tuple[int, int, int, int]
) -> tuple[tuple[int, int, int], tuple[int, int, int, int]]:
    left, top, _, height = bbox
    center_y = top + height / 2
    search_left = max(0, left - 32)
    search_right = max(search_left + 1, left - 5)
    search_top = max(0, int(round(center_y - 11)))
    search_bottom = min(source_rgb.shape[0], int(round(center_y + 11)))
    crop = source_rgb[search_top:search_bottom, search_left:search_right]
    if not crop.size:
        raise ValueError("legend swatch search window is empty")
    distance_from_white = np.linalg.norm(crop.astype(np.int16) - 255, axis=2)
    candidate = distance_from_white >= 25.0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate.astype(np.uint8), 8)
    components = []
    for component_id in range(1, count):
        x, y, width, component_height, area = (int(v) for v in stats[component_id])
        if width >= 8 and component_height >= 8 and area >= 50:
            components.append((area, component_id, x, y, width, component_height))
    if not components:
        raise ValueError("legend line has no rectangular swatch to its left")
    _, component_id, x, y, width, component_height = max(components)
    component = labels == component_id
    # Erode one pixel to prevent anti-aliased white edges from shifting color.
    core = cv2.erode(component.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    if not np.any(core):
        core = component
    rgb = tuple(int(value) for value in np.median(crop[core], axis=0).round())
    return rgb, (
        search_left + x,
        search_top + y,
        width,
        component_height,
    )


def _detect_semantic_legend(
    source_rgb: np.ndarray,
    words: Sequence[OCRWord],
    *,
    minimum_confidence: float,
) -> tuple[StormLegendEntry, ...]:
    grouped: dict[tuple[int, int, int], list[OCRWord]] = {}
    for word in words:
        grouped.setdefault(word.line_key, []).append(word)
    entries: list[StormLegendEntry] = []
    for layer_id, label, anchor_pattern, matcher in _legend_matchers():
        candidates = []
        for line_words in grouped.values():
            if not matcher(_line_tokens(line_words)):
                continue
            anchors = [
                word
                for word in line_words
                if anchor_pattern.fullmatch(word.text.strip()) is not None
            ]
            if not anchors:
                continue
            anchor = min(anchors, key=lambda value: value.left)
            anchor_center = anchor.top + anchor.height / 2
            ordered = sorted(
                (
                    word
                    for word in line_words
                    if word.left >= anchor.left
                    and abs((word.top + word.height / 2) - anchor_center) <= 20
                ),
                key=lambda value: value.left,
            )
            semantic_words = [ordered[0]]
            for word in ordered[1:]:
                previous = semantic_words[-1]
                if word.left - (previous.left + previous.width) >= 50:
                    break
                semantic_words.append(word)
            ordered = semantic_words
            left = min(value.left for value in ordered)
            top = min(value.top for value in ordered)
            right = max(value.left + value.width for value in ordered)
            bottom = max(value.top + value.height for value in ordered)
            confidence = float(np.mean([value.confidence for value in ordered]))
            candidates.append(((left, top, right - left, bottom - top), confidence))
        clusters = _cluster_physical_lines(candidates)
        if len(clusters) != 1:
            raise ValueError(
                f"expected exactly one physical OCR legend line for {label!r}; found {len(clusters)}"
            )
        bbox, confidence = max(clusters[0], key=lambda value: value[1])
        if confidence < minimum_confidence:
            raise ValueError(f"legend OCR confidence is too low for {label!r}")
        rgb, swatch_bbox = _swatch_from_line(source_rgb, bbox)
        entries.append(
            StormLegendEntry(layer_id, label, rgb, swatch_bbox, bbox, confidence)
        )

    precipitation = entries[:3]
    overlays = entries[3:]
    if any(max(entry.rgb) - min(entry.rgb) > 14 for entry in precipitation):
        raise ValueError("precipitation legend swatches are not grayscale")
    if any(max(entry.rgb) - min(entry.rgb) < 25 for entry in overlays):
        raise ValueError("impact overlay legend swatches are not chromatic")
    if len({entry.rgb for entry in entries}) != len(entries):
        raise ValueError("storm legend swatches are not unique")
    return tuple(entries)


def _warp_target_to_source(
    target: np.ndarray, source_to_target_remap: tuple[np.ndarray, np.ndarray]
) -> np.ndarray:
    return cv2.remap(
        target,
        source_to_target_remap[0],
        source_to_target_remap[1],
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection = int(np.count_nonzero(left & right))
    union = int(np.count_nonzero(left | right))
    return intersection / max(union, 1)


def _render_precipitation(
    ids: np.ndarray, entries: Sequence[StormLegendEntry]
) -> np.ndarray:
    rgb = np.zeros((*ids.shape, 3), dtype=np.uint8)
    for class_id, entry in enumerate(entries, 1):
        rgb[ids == class_id] = entry.rgb
    return rgb


def _render_overlay_composite(
    masks: Sequence[np.ndarray], entries: Sequence[StormLegendEntry]
) -> np.ndarray:
    rgb = np.zeros((*masks[0].shape, 3), dtype=np.uint8)
    count = np.zeros(masks[0].shape, dtype=np.uint8)
    total = np.zeros((*masks[0].shape, 3), dtype=np.uint16)
    for mask, entry in zip(masks, entries):
        total[mask] += np.asarray(entry.rgb, dtype=np.uint16)
        count[mask] += 1
    present = count > 0
    rgb[present] = np.rint(total[present] / count[present, None]).astype(np.uint8)
    return rgb


def _render_composite(
    precipitation_ids: np.ndarray,
    precipitation_entries: Sequence[StormLegendEntry],
    overlay_masks: Sequence[np.ndarray],
    overlay_entries: Sequence[StormLegendEntry],
) -> np.ndarray:
    base = _render_precipitation(precipitation_ids, precipitation_entries)
    overlay = _render_overlay_composite(overlay_masks, overlay_entries)
    present = np.logical_or.reduce(overlay_masks)
    base[present] = overlay[present]
    return base


def _semantic_signature(
    precipitation_ids: np.ndarray,
    occluded: np.ndarray,
    overlay_masks: Sequence[np.ndarray],
) -> np.ndarray:
    signature = precipitation_ids.astype(np.uint16)
    signature[occluded] |= np.uint16(1 << 5)
    for index, mask in enumerate(overlay_masks):
        signature[mask] |= np.uint16(1 << (index + 2))
    return signature


def _regional_metrics(
    source_signature: np.ndarray,
    reconstructed_signature: np.ndarray,
    source_to_target_remap: tuple[np.ndarray, np.ndarray],
    target_shape: tuple[int, int],
    rows: int,
    columns: int,
) -> list[dict[str, Any]]:
    map_x, map_y = source_to_target_remap
    target_height, target_width = target_shape
    support = source_signature > 0
    reports = []
    for row in range(rows):
        for column in range(columns):
            cell = (
                (map_x >= column * target_width / columns)
                & (map_x < (column + 1) * target_width / columns)
                & (map_y >= row * target_height / rows)
                & (map_y < (row + 1) * target_height / rows)
                & support
            )
            expected_count = int(np.count_nonzero(cell))
            if expected_count == 0:
                continue
            matched = int(np.count_nonzero(cell & (source_signature == reconstructed_signature)))
            reports.append(
                {
                    "id": f"r{row + 1}-c{column + 1}",
                    "expected_pixel_count": expected_count,
                    "matched_pixel_count": matched,
                    "match_fraction": matched / expected_count,
                }
            )
    return reports


def _diagnostic(
    aligned_source: np.ndarray,
    precipitation_rgb: np.ndarray,
    overlay_rgb: np.ndarray,
    composite_rgb: np.ndarray,
    occluded: np.ndarray,
    *,
    rows: int,
    columns: int,
    maximum_height: int = 1500,
) -> np.ndarray:
    review = cv2.addWeighted(aligned_source, 0.5, composite_rgb, 0.5, 0.0)
    review[occluded] = cv2.addWeighted(
        review[occluded].reshape(-1, 1, 3),
        0.35,
        np.full((np.count_nonzero(occluded), 1, 3), (255, 210, 0), dtype=np.uint8),
        0.65,
        0.0,
    ).reshape(-1, 3)
    scale = min(1.0, maximum_height / aligned_source.shape[0])
    if scale < 1.0:
        size = (
            max(1, round(aligned_source.shape[1] * scale)),
            max(1, round(aligned_source.shape[0] * scale)),
        )
        panels = [
            cv2.resize(value, size, interpolation=cv2.INTER_AREA)
            for value in (aligned_source, precipitation_rgb, overlay_rgb, review)
        ]
    else:
        panels = [aligned_source, precipitation_rgb, overlay_rgb, review]
    output = np.concatenate(panels, axis=1)
    canvas = Image.fromarray(output)
    draw = ImageDraw.Draw(canvas)
    labels = (
        "aligned pristine source",
        "observed precipitation; overlay base left unknown",
        "three independent impact overlays",
        "50/50 source-composite; gold means base occluded",
    )
    panel_width, panel_height = panels[0].shape[1], panels[0].shape[0]
    for index, label in enumerate(labels):
        x = index * panel_width + 12
        draw.rectangle((x - 4, 8, x + min(panel_width - 20, 390), 36), fill=(0, 0, 0))
        draw.text((x, 12), label, fill=(255, 255, 255))
        for column in range(1, columns):
            grid_x = index * panel_width + round(column * panel_width / columns)
            draw.line((grid_x, 0, grid_x, panel_height), fill=(255, 255, 255), width=1)
        for row in range(1, rows):
            grid_y = round(row * panel_height / rows)
            draw.line(
                (index * panel_width, grid_y, (index + 1) * panel_width, grid_y),
                fill=(255, 255, 255),
                width=1,
            )
    return np.asarray(canvas)


def _legend_payload(
    entries: Sequence[StormLegendEntry],
    working: WorkingRasterArtifact,
    tesseract_version: str,
) -> dict[str, Any]:
    precipitation, overlays = entries[:3], entries[3:]
    serialize = lambda entry: {
        "id": entry.layer_id,
        "label": entry.label,
        "rgb": list(entry.rgb),
        "swatch_bbox": list(entry.swatch_bbox),
        "label_bbox": list(entry.label_bbox),
        "ocr_confidence": entry.ocr_confidence,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "semantics_established",
        "source": {"path": str(working.source_path), "sha256": working.source_sha256},
        "tesseract_version": tesseract_version,
        "precipitation_classes": [serialize(entry) for entry in precipitation],
        "independent_overlays": [serialize(entry) for entry in overlays],
        "occlusion_contract": (
            "Any visible impact-overlay pixel makes the underlying precipitation "
            "class occluded/unknown. No neighboring precipitation class is inferred "
            "beneath an overlay."
        ),
    }


def run_storm_overlapping_layer_extraction(
    source_adapter_manifest_path: Path,
    accepted_alignment_path: Path,
    mapbox_manifest_path: Path,
    output_dir: Path,
    experiment_log: NoHumanExperimentLog,
    experiment_markdown_path: Path,
    experiment_json_path: Path,
    *,
    config: StormExtractionConfig = StormExtractionConfig(),
) -> StormExtractionResult:
    """Extract precipitation plus independent landslide/wind/flood overlays."""

    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise ValueError("storm extraction requires a fresh output directory")
    working, reference, alignment = _validate_inputs(
        source_adapter_manifest_path.resolve(),
        accepted_alignment_path.resolve(),
        mapbox_manifest_path.resolve(),
        experiment_log,
    )
    output_dir.mkdir(parents=True)
    source_rgb = np.asarray(Image.open(working.working_raster_path).convert("RGB"))
    transform = alignment["transform"]
    source_domain = _source_data_mask(
        reference.state_land, reference.water, transform, source_rgb.shape[:2]
    )
    target_domain = reference.state_land & ~reference.water
    source_to_target = _source_to_reference_remap(transform, source_rgb.shape[:2])
    reference_to_source = _reference_to_source_remap(transform)

    words, combined_tsv, tesseract_version = _run_tesseract_ocr(
        working.working_raster_path
    )
    entries = _detect_semantic_legend(
        source_rgb, words, minimum_confidence=config.minimum_legend_confidence
    )
    precipitation_entries, overlay_entries = entries[:3], entries[3:]
    legend_dir = output_dir / "legend"
    legend_dir.mkdir()
    legend_path = legend_dir / "semantic-layers.json"
    legend_path.write_text(
        json.dumps(_legend_payload(entries, working, tesseract_version), indent=2) + "\n"
    )
    ocr_path = legend_dir / "ocr.tsv"
    ocr_path.write_text(combined_tsv)

    overlay_categories = [{"legend_rgb": list(entry.rgb)} for entry in overlay_entries]
    source_overlay_masks, overlay_report = infer_sparse_chroma_overlays(
        source_rgb,
        source_domain,
        overlay_categories,
        config.overlay_minimum_chroma,
        config.overlay_coefficient_threshold,
        config.overlay_maximum_residual,
        config.overlay_complexity_penalty,
    )
    source_overlay_masks = [mask.astype(bool) for mask in source_overlay_masks]
    source_overlay_union = np.logical_or.reduce(source_overlay_masks)
    source_overlay_membership_count = np.sum(np.stack(source_overlay_masks), axis=0)
    source_multi_overlay = source_overlay_membership_count > 1
    source_overlay_observed = [
        mask & ~source_multi_overlay for mask in source_overlay_masks
    ]
    source_overlay_inferred = [
        mask & source_multi_overlay for mask in source_overlay_masks
    ]
    source_lab = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    source_lab_chroma = np.linalg.norm(source_lab[:, :, 1:3] - 128.0, axis=2)
    source_overlay_ambiguous = (
        source_domain
        & (source_lab_chroma >= config.overlay_minimum_chroma)
        & ~source_overlay_union
    )
    # The chromatic evidence says an overlay or other source ink is visible, but
    # not which supported overlay it belongs to.  Treat the base precipitation
    # as unknowable here; never let the grayscale classifier fill underneath it.
    source_occluded = source_overlay_union | source_overlay_ambiguous
    precipitation_categories = [
        {"legend_gray": entry.gray} for entry in precipitation_entries
    ]
    source_precipitation_ids, precipitation_report = classify_grayscale(
        source_rgb,
        source_domain,
        precipitation_categories,
        config.precipitation_smoothing_radius,
        config.precipitation_maximum_gray_distance,
        config.precipitation_maximum_chroma,
        exclusion_mask=source_occluded,
        adaptive_centers=False,
        background_cutoff=config.precipitation_background_cutoff,
    )
    source_precipitation_ids[source_occluded] = 0
    source_precipitation_observed = source_precipitation_ids > 0
    source_precipitation_inferred = np.zeros(source_domain.shape, dtype=bool)
    aligned_source = _source_to_reference(
        source_rgb,
        transform,
        cv2.INTER_LINEAR,
        (255, 255, 255),
        reference_to_source,
    )
    previous_source_signature: np.ndarray | None = None
    previous_target_signature: np.ndarray | None = None
    iterations: list[StormExtractionIteration] = []
    all_artifacts: list[Path] = [legend_path, ocr_path]
    accepted_path: Path | None = None

    for iteration_number in range(1, config.required_replay_count + 1):
        iteration_dir = output_dir / f"extraction-{iteration_number:02d}"
        iteration_dir.mkdir()
        target_precipitation_ids = _source_to_reference(
            source_precipitation_ids,
            transform,
            cv2.INTER_NEAREST,
            0,
            reference_to_source,
        )
        target_precipitation_observed = _source_to_reference(
            source_precipitation_observed.astype(np.uint8),
            transform,
            cv2.INTER_NEAREST,
            0,
            reference_to_source,
        ) > 0
        target_precipitation_inferred = _source_to_reference(
            source_precipitation_inferred.astype(np.uint8),
            transform,
            cv2.INTER_NEAREST,
            0,
            reference_to_source,
        ) > 0
        target_occluded = _source_to_reference(
            source_occluded.astype(np.uint8),
            transform,
            cv2.INTER_NEAREST,
            0,
            reference_to_source,
        ) > 0
        target_overlay_ambiguous = _source_to_reference(
            source_overlay_ambiguous.astype(np.uint8),
            transform,
            cv2.INTER_NEAREST,
            0,
            reference_to_source,
        ) > 0
        target_overlay_masks = [
            _source_to_reference(
                mask.astype(np.uint8),
                transform,
                cv2.INTER_NEAREST,
                0,
                reference_to_source,
            )
            > 0
            for mask in source_overlay_masks
        ]
        target_overlay_inferred = [
            _source_to_reference(
                mask.astype(np.uint8),
                transform,
                cv2.INTER_NEAREST,
                0,
                reference_to_source,
            )
            > 0
            for mask in source_overlay_inferred
        ]
        target_overlay_observed_masks = [
            _source_to_reference(
                mask.astype(np.uint8),
                transform,
                cv2.INTER_NEAREST,
                0,
                reference_to_source,
            )
            > 0
            for mask in source_overlay_observed
        ]
        target_precipitation_ids[~target_domain] = 0
        target_precipitation_observed &= target_domain
        target_precipitation_inferred &= target_domain
        target_occluded &= target_domain
        target_overlay_ambiguous &= target_domain
        for index in range(3):
            target_overlay_masks[index] &= target_domain
            target_overlay_inferred[index] &= target_domain
            target_overlay_observed_masks[index] &= target_domain
        target_precipitation_ids[target_occluded] = 0
        target_precipitation_observed &= ~target_occluded
        target_precipitation_inferred &= ~target_occluded

        reconstructed_precipitation_ids = _warp_target_to_source(
            target_precipitation_ids, source_to_target
        )
        reconstructed_occluded = _warp_target_to_source(
            target_occluded.astype(np.uint8), source_to_target
        ) > 0
        reconstructed_overlay_masks = [
            _warp_target_to_source(mask.astype(np.uint8), source_to_target) > 0
            for mask in target_overlay_masks
        ]
        precip_expected = source_precipitation_ids > 0
        precip_matches = precip_expected & (
            reconstructed_precipitation_ids == source_precipitation_ids
        )
        precipitation_roundtrip = float(
            np.count_nonzero(precip_matches) / max(np.count_nonzero(precip_expected), 1)
        )
        overlay_ious = [
            _iou(source_mask, reconstructed_mask)
            for source_mask, reconstructed_mask in zip(
                source_overlay_masks, reconstructed_overlay_masks
            )
        ]
        occluded_iou = _iou(source_occluded, reconstructed_occluded)
        source_signature = _semantic_signature(
            source_precipitation_ids, source_occluded, source_overlay_masks
        )
        target_signature = _semantic_signature(
            target_precipitation_ids, target_occluded, target_overlay_masks
        )
        reconstructed_signature = _warp_target_to_source(
            target_signature, source_to_target
        )
        support = source_signature > 0
        composite_roundtrip = float(
            np.count_nonzero(support & (source_signature == reconstructed_signature))
            / max(np.count_nonzero(support), 1)
        )
        regional = _regional_metrics(
            source_signature,
            reconstructed_signature,
            source_to_target,
            target_domain.shape,
            config.geographic_rows,
            config.geographic_columns,
        )
        passing_cells = sum(
            report["match_fraction"] >= config.minimum_geographic_match_fraction
            for report in regional
        )
        stable = (
            previous_source_signature is not None
            and previous_target_signature is not None
            and np.array_equal(previous_source_signature, source_signature)
            and np.array_equal(previous_target_signature, target_signature)
        )
        class_ids = {int(value) for value in np.unique(target_precipitation_ids) if value > 0}
        overlay_nonempty = [bool(np.any(mask)) for mask in target_overlay_masks]
        exterior_empty = not bool(
            np.any(target_precipitation_ids[~target_domain] > 0)
            or np.any(target_occluded[~target_domain])
            or np.any(target_overlay_ambiguous[~target_domain])
            or any(np.any(mask[~target_domain]) for mask in target_overlay_masks)
        )
        gates: dict[str, Any] = {
            "all_six_legend_semantics_established": len(entries) == 6,
            "legend_ocr_confidence": all(
                entry.ocr_confidence >= config.minimum_legend_confidence for entry in entries
            ),
            "three_mutually_exclusive_precipitation_classes": class_ids == {1, 2, 3},
            "three_independent_overlays_nonempty": all(overlay_nonempty),
            "no_precipitation_inference_under_overlays": not bool(
                np.any(target_precipitation_ids[target_occluded] > 0)
                or np.any(target_precipitation_inferred)
            ),
            "overlay_observed_inferred_partition": all(
                not np.any(observed & inferred)
                and np.array_equal(observed | inferred, complete)
                for observed, inferred, complete in zip(
                    target_overlay_observed_masks,
                    target_overlay_inferred,
                    target_overlay_masks,
                )
            ),
            "precipitation_source_roundtrip": {
                "passed": precipitation_roundtrip
                >= config.minimum_precipitation_roundtrip_fraction,
                "value": precipitation_roundtrip,
                "minimum": config.minimum_precipitation_roundtrip_fraction,
            },
            "overlay_source_roundtrip_iou": {
                "passed": all(
                    value >= config.minimum_overlay_roundtrip_iou for value in overlay_ious
                ),
                "values": overlay_ious,
                "minimum": config.minimum_overlay_roundtrip_iou,
            },
            "occluded_source_roundtrip_iou": {
                "passed": occluded_iou >= config.minimum_occluded_roundtrip_iou,
                "value": occluded_iou,
                "minimum": config.minimum_occluded_roundtrip_iou,
            },
            "source_composite_roundtrip": {
                "passed": composite_roundtrip
                >= config.minimum_composite_roundtrip_fraction,
                "value": composite_roundtrip,
                "minimum": config.minimum_composite_roundtrip_fraction,
            },
            "geographic_source_diff": {
                "passed": passing_cells >= config.minimum_passing_geographic_cells,
                "value": passing_cells,
                "minimum": config.minimum_passing_geographic_cells,
                "supported_cells": len(regional),
            },
            "mapbox_water_and_exterior_empty": exterior_empty,
            "successive_all_channel_fixed_point": stable,
        }
        all_gates_passed = all(
            value if isinstance(value, bool) else bool(value["passed"])
            for value in gates.values()
        )
        decision = (
            "accept"
            if all_gates_passed
            else ("retry" if iteration_number < config.required_replay_count else "blocked")
        )

        source_precip_path = iteration_dir / "source-precipitation-class-id.png"
        source_observed_path = iteration_dir / "source-precipitation-observed-mask.png"
        source_inferred_path = iteration_dir / "source-precipitation-inferred-mask.png"
        source_occluded_path = iteration_dir / "source-precipitation-occluded-mask.png"
        source_overlay_ambiguous_path = iteration_dir / "source-overlay-ambiguous-mask.png"
        target_precip_path = iteration_dir / "mapbox-precipitation-class-id.png"
        target_observed_path = iteration_dir / "mapbox-precipitation-observed-mask.png"
        target_inferred_path = iteration_dir / "mapbox-precipitation-inferred-mask.png"
        target_occluded_path = iteration_dir / "mapbox-precipitation-occluded-mask.png"
        target_overlay_ambiguous_path = iteration_dir / "mapbox-overlay-ambiguous-mask.png"
        source_diff_path = iteration_dir / "source-composite-roundtrip-diff-mask.png"
        source_composite_path = iteration_dir / "source-composite-reconstruction.png"
        target_precip_rgb_path = iteration_dir / "mapbox-precipitation-reconstruction.png"
        target_overlay_rgb_path = iteration_dir / "mapbox-overlays-reconstruction.png"
        target_composite_path = iteration_dir / "mapbox-composite-reconstruction.png"
        aligned_source_path = iteration_dir / "mapbox-aligned-source.png"
        diagnostic_path = iteration_dir / "source-extraction-diagnostic.png"
        source_overlay_total_paths = [
            iteration_dir / f"source-{entry.layer_id}-mask.png"
            for entry in overlay_entries
        ]
        source_overlay_paths = [
            iteration_dir / f"source-{entry.layer_id}-observed-mask.png"
            for entry in overlay_entries
        ]
        source_overlay_inferred_paths = [
            iteration_dir / f"source-{entry.layer_id}-inferred-mask.png"
            for entry in overlay_entries
        ]
        target_overlay_paths = [
            iteration_dir / f"mapbox-{entry.layer_id}-mask.png"
            for entry in overlay_entries
        ]
        target_overlay_observed_paths = [
            iteration_dir / f"mapbox-{entry.layer_id}-observed-mask.png"
            for entry in overlay_entries
        ]
        target_overlay_inferred_paths = [
            iteration_dir / f"mapbox-{entry.layer_id}-inferred-mask.png"
            for entry in overlay_entries
        ]
        source_precip_rgb = _render_precipitation(
            source_precipitation_ids, precipitation_entries
        )
        source_composite_rgb = _render_composite(
            source_precipitation_ids,
            precipitation_entries,
            source_overlay_masks,
            overlay_entries,
        )
        target_precip_rgb = _render_precipitation(
            target_precipitation_ids, precipitation_entries
        )
        target_overlay_rgb = _render_overlay_composite(
            target_overlay_masks, overlay_entries
        )
        target_composite_rgb = _render_composite(
            target_precipitation_ids,
            precipitation_entries,
            target_overlay_masks,
            overlay_entries,
        )
        _save_ids(source_precip_path, source_precipitation_ids)
        _save_mask(source_observed_path, source_precipitation_observed)
        _save_mask(source_inferred_path, source_precipitation_inferred)
        _save_mask(source_occluded_path, source_occluded)
        _save_mask(source_overlay_ambiguous_path, source_overlay_ambiguous)
        for path, mask in zip(source_overlay_total_paths, source_overlay_masks):
            _save_mask(path, mask)
        for path, mask in zip(source_overlay_paths, source_overlay_observed):
            _save_mask(path, mask)
        for path, mask in zip(source_overlay_inferred_paths, source_overlay_inferred):
            _save_mask(path, mask)
        _save_ids(target_precip_path, target_precipitation_ids)
        _save_mask(target_observed_path, target_precipitation_observed)
        _save_mask(target_inferred_path, target_precipitation_inferred)
        _save_mask(target_occluded_path, target_occluded)
        _save_mask(target_overlay_ambiguous_path, target_overlay_ambiguous)
        for path, mask in zip(target_overlay_paths, target_overlay_masks):
            _save_mask(path, mask)
        for path, mask in zip(target_overlay_observed_paths, target_overlay_observed_masks):
            _save_mask(path, mask)
        for path, mask in zip(target_overlay_inferred_paths, target_overlay_inferred):
            _save_mask(path, mask)
        _save_mask(source_diff_path, support & (source_signature != reconstructed_signature))
        _save_rgb(source_composite_path, source_composite_rgb)
        _save_rgb(target_precip_rgb_path, target_precip_rgb)
        _save_rgb(target_overlay_rgb_path, target_overlay_rgb)
        _save_rgb(target_composite_path, target_composite_rgb)
        _save_rgb(aligned_source_path, aligned_source)
        _save_rgb(
            diagnostic_path,
            _diagnostic(
                aligned_source,
                target_precip_rgb,
                target_overlay_rgb,
                target_composite_rgb,
                target_occluded,
                rows=config.geographic_rows,
                columns=config.geographic_columns,
            ),
        )
        artifact_paths = tuple(
            [
                source_precip_path,
                source_observed_path,
                source_inferred_path,
                source_occluded_path,
                source_overlay_ambiguous_path,
                *source_overlay_total_paths,
                *source_overlay_paths,
                *source_overlay_inferred_paths,
                source_diff_path,
                source_composite_path,
                target_precip_path,
                target_observed_path,
                target_inferred_path,
                target_occluded_path,
                target_overlay_ambiguous_path,
                *target_overlay_paths,
                *target_overlay_observed_paths,
                *target_overlay_inferred_paths,
                target_precip_rgb_path,
                target_overlay_rgb_path,
                target_composite_path,
                aligned_source_path,
                diagnostic_path,
            ]
        )
        scores = {
            "legend_labels": [entry.label for entry in entries],
            "legend_rgb": {entry.layer_id: list(entry.rgb) for entry in entries},
            "legend_ocr_confidences": [entry.ocr_confidence for entry in entries],
            "source_precipitation_observed_pixel_count": int(
                np.count_nonzero(source_precipitation_observed)
            ),
            "source_precipitation_inferred_pixel_count": 0,
            "source_precipitation_occluded_pixel_count": int(
                np.count_nonzero(source_occluded)
            ),
            "source_overlay_ambiguous_pixel_count": int(
                np.count_nonzero(source_overlay_ambiguous)
            ),
            "mapbox_precipitation_observed_pixel_count": int(
                np.count_nonzero(target_precipitation_observed)
            ),
            "mapbox_precipitation_inferred_pixel_count": 0,
            "mapbox_precipitation_occluded_pixel_count": int(
                np.count_nonzero(target_occluded)
            ),
            "mapbox_overlay_ambiguous_pixel_count": int(
                np.count_nonzero(target_overlay_ambiguous)
            ),
            "source_precipitation_class_pixel_counts": {
                entry.layer_id: int(np.count_nonzero(source_precipitation_ids == class_id))
                for class_id, entry in enumerate(precipitation_entries, 1)
            },
            "mapbox_precipitation_class_pixel_counts": {
                entry.layer_id: int(np.count_nonzero(target_precipitation_ids == class_id))
                for class_id, entry in enumerate(precipitation_entries, 1)
            },
            "source_overlay_observed_pixel_counts": {
                entry.layer_id: int(np.count_nonzero(mask))
                for entry, mask in zip(overlay_entries, source_overlay_observed)
            },
            "source_overlay_inferred_pixel_counts": {
                entry.layer_id: int(np.count_nonzero(mask))
                for entry, mask in zip(overlay_entries, source_overlay_inferred)
            },
            "mapbox_overlay_observed_pixel_counts": {
                entry.layer_id: int(np.count_nonzero(mask))
                for entry, mask in zip(overlay_entries, target_overlay_observed_masks)
            },
            "mapbox_overlay_inferred_pixel_counts": {
                entry.layer_id: int(np.count_nonzero(mask))
                for entry, mask in zip(overlay_entries, target_overlay_inferred)
            },
            "source_multi_overlay_pixel_count": int(
                np.count_nonzero(source_multi_overlay)
            ),
            "precipitation_source_roundtrip_fraction": precipitation_roundtrip,
            "overlay_source_roundtrip_ious": {
                entry.layer_id: value for entry, value in zip(overlay_entries, overlay_ious)
            },
            "occluded_source_roundtrip_iou": occluded_iou,
            "source_composite_roundtrip_fraction": composite_roundtrip,
            "geographic_cells": regional,
            "overlay_unmixing": overlay_report,
            "precipitation_classification": precipitation_report,
            "successive_source_signature_equal": stable,
            "successive_mapbox_signature_equal": stable,
        }
        report_path = iteration_dir / "iteration.json"
        report_path.write_text(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "iteration": iteration_number,
                    "decision": decision,
                    "scores": scores,
                    "gates": gates,
                    "provenance": {
                        "original_source": {
                            "path": str(working.source_path),
                            "sha256": working.source_sha256,
                        },
                        "source_adapter": {
                            "path": str(working.manifest_path),
                            "sha256": _sha256(working.manifest_path),
                        },
                        "accepted_alignment": {
                            "path": str(accepted_alignment_path.resolve()),
                            "sha256": _sha256(accepted_alignment_path.resolve()),
                        },
                        "pinned_mapbox_manifest": {
                            "path": str(mapbox_manifest_path.resolve()),
                            "sha256": _sha256(mapbox_manifest_path.resolve()),
                        },
                        "prior_run_artifacts_used": False,
                        "manual_inputs_used": False,
                    },
                    "artifacts": [_artifact(path, output_dir) for path in artifact_paths],
                },
                indent=2,
            )
            + "\n"
        )
        complete_artifacts = (*artifact_paths, report_path)
        experiment_log.record_extraction_iteration(
            scores=scores,
            gates=gates,
            decision=decision,
            provenance=automatic_provenance(
                PRODUCER,
                [
                    "authoritative_original_png_pixels",
                    "automatic_six_entry_ocr_and_swatch_legend",
                    "accepted_automatic_mapbox_alignment",
                    "pinned_mapbox_land_and_water",
                    "source_channel_and_composite_reconstruction_diff",
                    "geographically_partitioned_source_diff",
                    "deterministic_overlapping_layer_fixed_point_replay",
                ],
            ),
            method=(
                "automatic three-class grayscale precipitation extraction with "
                "explicit overlay occlusion plus independent landslide, maximum-wind, "
                "and predicted-flooding chroma masks, Mapbox clipping, source/composite "
                "reconstruction, geographic diff, and fixed-point replay"
            ),
            artifacts=[
                {"path": str(path), "sha256": _sha256(path)}
                for path in (*complete_artifacts, legend_path)
            ],
        )
        experiment_log.write(experiment_markdown_path, experiment_json_path)
        iterations.append(
            StormExtractionIteration(
                iteration_number,
                decision,
                scores,
                gates,
                report_path,
                tuple(complete_artifacts),
            )
        )
        all_artifacts.extend(complete_artifacts)
        previous_source_signature = source_signature.copy()
        previous_target_signature = target_signature.copy()
        if decision == "accept":
            accepted_path = output_dir / "accepted-extraction.json"
            accepted_path.write_text(
                json.dumps(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "status": "accepted",
                        "automatic_iteration_count": iteration_number,
                        "source": {
                            "path": str(working.source_path),
                            "sha256": working.source_sha256,
                        },
                        "source_adapter": {
                            "path": str(working.manifest_path),
                            "sha256": _sha256(working.manifest_path),
                        },
                        "alignment": {
                            "path": str(accepted_alignment_path.resolve()),
                            "sha256": _sha256(accepted_alignment_path.resolve()),
                        },
                        "legend": {
                            "path": "legend/semantic-layers.json",
                            "sha256": _sha256(legend_path),
                        },
                        "accepted_iteration": f"extraction-{iteration_number:02d}",
                        "layers": {
                            "maximum_daily_precipitation": {
                                "kind": "mutually_exclusive_categorical",
                                "class_ids": f"extraction-{iteration_number:02d}/mapbox-precipitation-class-id.png",
                                "observed_mask": f"extraction-{iteration_number:02d}/mapbox-precipitation-observed-mask.png",
                                "inferred_mask": f"extraction-{iteration_number:02d}/mapbox-precipitation-inferred-mask.png",
                                "occluded_unknown_mask": f"extraction-{iteration_number:02d}/mapbox-precipitation-occluded-mask.png",
                            },
                            **{
                                entry.layer_id: {
                                    "kind": "independent_binary_overlay",
                                    "mask": f"extraction-{iteration_number:02d}/mapbox-{entry.layer_id}-mask.png",
                                    "observed_mask": f"extraction-{iteration_number:02d}/mapbox-{entry.layer_id}-observed-mask.png",
                                    "inferred_mask": f"extraction-{iteration_number:02d}/mapbox-{entry.layer_id}-inferred-mask.png",
                                }
                                for entry in overlay_entries
                            },
                            "unresolved_chromatic_evidence": {
                                "kind": "occluded_unknown",
                                "mask": f"extraction-{iteration_number:02d}/mapbox-overlay-ambiguous-mask.png",
                            },
                        },
                    },
                    indent=2,
                )
                + "\n"
            )
            all_artifacts.append(accepted_path)
            experiment_log.finalize("complete")
            experiment_log.write(experiment_markdown_path, experiment_json_path)
            break

    if accepted_path is not None:
        return StormExtractionResult(
            "accepted",
            "legend, channel reconstruction, geographic, occlusion, and fixed-point gates passed",
            tuple(iterations),
            accepted_path,
            tuple(all_artifacts),
        )
    blocker = (
        "Storm overlapping-layer extraction did not pass every semantic, channel "
        "reconstruction, geographic, occlusion, and fixed-point gate after two passes"
    )
    experiment_log.finalize("blocked", blocker)
    experiment_log.write(experiment_markdown_path, experiment_json_path)
    return StormExtractionResult(
        "blocked", blocker, tuple(iterations), None, tuple(all_artifacts)
    )
