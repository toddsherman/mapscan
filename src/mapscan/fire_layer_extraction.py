"""No-human extraction for the fire hazard map's overlapping data model.

The authoritative image contains two different thematic encodings:

* three mutually exclusive solid hazard colors (Moderate, High, Very High), and
* a dark, regularly repeated dot symbol for Local Responsibility Area (LRA).

The dot layer is deliberately kept separate from the hazard class raster.  The
same dark ink is also used for county borders, labels, and city marks, so color
alone is insufficient: LRA pixels must belong to small dot-like components in
a locally repeated two-dimensional pattern derived from the legend swatch.

Only the source-clean working raster, its accepted automatic alignment, and the
pinned Mapbox land/water masks are accepted as inputs.  A second byte-identical
pass is required before the extraction may be accepted.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .automatic_alignment_loop import load_pinned_mapbox_reference
from .automatic_categorical_extraction import (
    LegendEntry,
    OCRWord,
    _classify,
    _load_accepted_alignment,
    _palette_lab,
    _reference_to_source_remap,
    _run_tesseract_ocr,
    _source_data_mask,
    _source_to_reference,
    _source_to_reference_remap,
    detect_legend,
)
from .experiment_log import NoHumanExperimentLog, automatic_provenance
from .source_working_raster import WorkingRasterArtifact, load_working_raster_artifact


SCHEMA_VERSION = "mapscan.fire-overlapping-layer-extraction.v1"
PRODUCER = "mapscan.fire_layer_extraction"
FORBIDDEN_PATH_TOKENS = (
    "automatic-alignment-orphaned-race",
    "county.png",
    "census",
    "manual",
    "legacy",
)
EXPECTED_HAZARD_LABELS = ("moderate", "high", "very high")


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


def _normalize_label(value: str) -> str:
    return " ".join(re.findall(r"[a-z]+", value.lower()))


@dataclass(frozen=True)
class LRALegendEvidence:
    label: str
    label_bbox: tuple[int, int, int, int]
    label_confidence: float
    dot_rgb: tuple[int, int, int]
    swatch_bbox: tuple[int, int, int, int]
    dot_component_count: int
    median_dot_area: float
    median_dot_width: float
    median_dot_height: float
    median_column_spacing: float
    median_row_spacing: float


@dataclass(frozen=True)
class LRADotEvidence:
    mask: np.ndarray
    dark_distance_threshold: float
    candidate_component_count: int
    repeated_component_count: int
    selected_pixel_count: int
    local_window_px: int
    minimum_local_component_count: int
    mapbox_line_exclusion_pixel_count: int
    mapbox_line_buffer_target_px: int


@dataclass(frozen=True)
class FireExtractionConfig:
    required_replay_count: int = 2
    maximum_hazard_lab_distance: float = 64.0
    minimum_hazard_margin: float = 0.5
    minimum_legend_confidence: float = 90.0
    minimum_lra_legend_dots: int = 30
    minimum_lra_map_components: int = 500
    maximum_inferred_hazard_fraction: float = 0.12
    minimum_hazard_roundtrip_fraction: float = 0.94
    minimum_lra_roundtrip_iou: float = 0.80
    geographic_rows: int = 4
    geographic_columns: int = 3
    minimum_geographic_match_fraction: float = 0.90
    minimum_passing_geographic_cells: int = 8

    def __post_init__(self) -> None:
        if self.required_replay_count != 2:
            raise ValueError("fire extraction requires exactly two fixed-point passes")
        if self.minimum_lra_legend_dots < 4:
            raise ValueError("minimum_lra_legend_dots must be at least four")
        if self.minimum_lra_map_components < 1:
            raise ValueError("minimum_lra_map_components must be positive")
        for value in (
            self.maximum_inferred_hazard_fraction,
            self.minimum_hazard_roundtrip_fraction,
            self.minimum_lra_roundtrip_iou,
            self.minimum_geographic_match_fraction,
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError("fraction gates must be between zero and one")


@dataclass(frozen=True)
class FireExtractionIteration:
    iteration: int
    decision: str
    scores: Mapping[str, Any]
    gates: Mapping[str, Any]
    report_path: Path
    artifact_paths: tuple[Path, ...]


@dataclass(frozen=True)
class FireExtractionResult:
    status: str
    stop_reason: str
    iterations: tuple[FireExtractionIteration, ...]
    accepted_extraction_path: Path | None
    artifact_paths: tuple[Path, ...]

    @property
    def accepted(self) -> Path | None:
        return self.accepted_extraction_path


def _assert_clean_path(path: Path, kind: str) -> None:
    normalized = str(path.resolve()).lower()
    matched = next((token for token in FORBIDDEN_PATH_TOKENS if token in normalized), None)
    if matched:
        raise ValueError(f"{kind} path contains forbidden no-human evidence: {matched}")


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
    if working.source_path.suffix.lower() != ".webp":
        raise ValueError("fire overlapping-layer extraction requires the source WEBP")
    authority = working.manifest.get("authority", {})
    if authority != {
        "manual_input_used": False,
        "original_source_authoritative": True,
        "prior_alignment_used": False,
        "prior_extraction_used": False,
    }:
        raise ValueError("source-clean authority record is not pristine")
    if experiment_log.data.get("map_id") != "fire":
        raise ValueError("fire extractor requires the fire experiment log")
    if experiment_log.data["source"].get("sha256") != working.source_sha256:
        raise ValueError("experiment and source-clean original hashes disagree")
    if experiment_log.data["source"].get("source_type") != "overlapping_feature_and_categorical":
        raise ValueError("fire source data model is not overlapping_feature_and_categorical")
    if experiment_log.data["extraction"]["iterations"]:
        raise ValueError("fire extraction log already contains attempts")

    reference = load_pinned_mapbox_reference(mapbox_manifest_path)
    accepted_count = experiment_log.data["alignment"][
        "accepted_automatic_iteration_count"
    ]
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


def _find_label_line(
    words: Sequence[OCRWord], expected: str, minimum_confidence: float
) -> tuple[tuple[int, int, int, int], float]:
    target = _normalize_label(expected)
    grouped: dict[tuple[int, int, int], list[OCRWord]] = defaultdict(list)
    for word in words:
        if word.confidence >= minimum_confidence:
            grouped[word.line_key].append(word)
    matches = []
    for line_words in grouped.values():
        ordered = sorted(line_words, key=lambda value: value.left)
        normalized = [_normalize_label(value.text) for value in ordered]
        normalized = [value for value in normalized if value]
        for start in range(len(normalized)):
            for stop in range(start + 1, len(normalized) + 1):
                if " ".join(normalized[start:stop]) != target:
                    continue
                selected = ordered[start:stop]
                left = min(value.left for value in selected)
                top = min(value.top for value in selected)
                right = max(value.right for value in selected)
                bottom = max(value.top + value.height for value in selected)
                matches.append(
                    (
                        (left, top, right - left, bottom - top),
                        min(value.confidence for value in selected),
                    )
                )
    # The two pinned Tesseract layout passes can report the same physical line.
    # Collapse only identical geometry; two labels in different places remain
    # an ambiguity and reject the run.
    deduplicated: dict[tuple[int, int, int, int], float] = {}
    for bbox, confidence in matches:
        deduplicated[bbox] = max(confidence, deduplicated.get(bbox, -1.0))
    matches = list(deduplicated.items())
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one readable OCR line {expected!r}; found {len(matches)}"
        )
    bbox, confidence = matches[0]
    return bbox, confidence


def _median_positive_spacing(values: np.ndarray) -> float:
    unique = np.unique(np.rint(values).astype(np.int32))
    differences = np.diff(unique)
    differences = differences[differences >= 3]
    if not len(differences):
        raise ValueError("legend dot grid does not expose repeated spacing")
    return float(np.median(differences))


def _detect_lra_legend(
    source_rgb: np.ndarray,
    words: Sequence[OCRWord],
    *,
    minimum_confidence: float,
    minimum_dot_count: int,
) -> LRALegendEvidence:
    label_bbox, confidence = _find_label_line(
        words, "Local Responsibility Area", minimum_confidence
    )
    left, top, width, height = label_bbox
    # The symbol is on the same OCR row immediately to the label's left.  The
    # generous window is then reduced to dark, dot-like connected components.
    swatch_search_width = round(
        max(source_rgb.shape[1] * 0.11, max(height, 16) * 7.0)
    )
    swatch_left = max(0, left - swatch_search_width)
    swatch_right = max(swatch_left + 1, left - round(source_rgb.shape[1] * 0.01))
    swatch_top = max(0, top - round(max(height, 16) * 1.7))
    swatch_bottom = min(source_rgb.shape[0], top + height + round(max(height, 16) * 1.7))
    crop = source_rgb[swatch_top:swatch_bottom, swatch_left:swatch_right]
    if not crop.size:
        raise ValueError("LRA legend symbol search window is empty")

    dark = np.max(crop, axis=2) < 120
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        dark.astype(np.uint8), 8
    )
    candidates = []
    for component_id in range(1, count):
        x, y, box_width, box_height, area = (
            int(value) for value in stats[component_id]
        )
        if 3 <= box_width <= 12 and 3 <= box_height <= 12 and 8 <= area <= 100:
            extent = area / (box_width * box_height)
            if extent >= 0.45:
                candidates.append(component_id)
    if len(candidates) < minimum_dot_count:
        raise ValueError("LRA legend does not contain a sufficiently repeated dot grid")

    component_pixels = np.isin(labels, candidates)
    colors = crop[component_pixels]
    dot_rgb = tuple(int(value) for value in np.median(colors, axis=0).round())
    selected_stats = stats[candidates]
    selected_centroids = centroids[candidates]
    column_spacing = _median_positive_spacing(selected_centroids[:, 0])
    row_spacing = _median_positive_spacing(selected_centroids[:, 1])
    return LRALegendEvidence(
        label="Local Responsibility Area",
        label_bbox=label_bbox,
        label_confidence=float(confidence),
        dot_rgb=dot_rgb,
        swatch_bbox=(
            swatch_left,
            swatch_top,
            swatch_right - swatch_left,
            swatch_bottom - swatch_top,
        ),
        dot_component_count=len(candidates),
        median_dot_area=float(np.median(selected_stats[:, cv2.CC_STAT_AREA])),
        median_dot_width=float(np.median(selected_stats[:, cv2.CC_STAT_WIDTH])),
        median_dot_height=float(np.median(selected_stats[:, cv2.CC_STAT_HEIGHT])),
        median_column_spacing=column_spacing,
        median_row_spacing=row_spacing,
    )


def _detect_lra_source_dots(
    source_rgb: np.ndarray,
    source_domain: np.ndarray,
    legend: LRALegendEvidence,
    known_mapbox_lines: np.ndarray | None = None,
    *,
    mapbox_line_buffer_target_px: int = 25,
) -> LRADotEvidence:
    color = np.asarray(legend.dot_rgb, dtype=np.int16)
    distance = np.linalg.norm(source_rgb.astype(np.int16) - color, axis=2)
    # The decoded WebP makes a map-scale dot much smaller than the legend dot
    # and gives its anti-aliased edge a wider color range.  A broad threshold
    # makes continuous border ink form long connected components (which are
    # rejected below), while retaining each isolated circular dot as one small
    # component.  Pinned Mapbox state/county lines are an independent exclusion
    # channel so compression breaks in those lines cannot masquerade as dots.
    legend_core_threshold = max(12.0, min(24.0, legend.median_dot_area * 0.48))
    threshold = min(90.0, legend_core_threshold * 6.0)
    line_exclusion = (
        np.zeros(source_domain.shape, dtype=bool)
        if known_mapbox_lines is None
        else known_mapbox_lines.astype(bool)
    )
    if line_exclusion.shape != source_domain.shape:
        raise ValueError("Mapbox line exclusion mask shape disagrees with source")
    dark = (distance <= threshold) & source_domain & ~line_exclusion
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        dark.astype(np.uint8), 8
    )
    maximum_width = max(2, int(round(legend.median_dot_width)))
    maximum_height = max(2, int(round(legend.median_dot_height)))
    maximum_area = max(4, int(round(legend.median_dot_area * 1.10)))
    candidate_ids = []
    for component_id in range(1, count):
        width = int(stats[component_id, cv2.CC_STAT_WIDTH])
        height = int(stats[component_id, cv2.CC_STAT_HEIGHT])
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        aspect = max(width, height) / max(min(width, height), 1)
        if (
            width <= maximum_width
            and height <= maximum_height
            and area <= maximum_area
            and aspect <= 1.5
        ):
            candidate_ids.append(component_id)
    if not candidate_ids:
        raise ValueError("no map-scale LRA dot candidates were found")

    centers = np.zeros(source_domain.shape, dtype=np.uint8)
    center_xy = np.rint(centroids[candidate_ids]).astype(np.int32)
    center_xy[:, 0] = np.clip(center_xy[:, 0], 0, source_domain.shape[1] - 1)
    center_xy[:, 1] = np.clip(center_xy[:, 1], 0, source_domain.shape[0] - 1)
    centers[center_xy[:, 1], center_xy[:, 0]] = 1
    local_window = max(
        13,
        int(
            round(
                2.1
                * max(legend.median_column_spacing, legend.median_row_spacing)
            )
        ),
    )
    if local_window % 2 == 0:
        local_window += 1
    local_count = cv2.boxFilter(
        centers, cv2.CV_16U, (local_window, local_window), normalize=False
    )
    minimum_local_count = 2
    repeated_ids = [
        component_id
        for component_id, (x, y) in zip(candidate_ids, center_xy)
        if int(local_count[y, x]) >= minimum_local_count
    ]
    if not repeated_ids:
        raise ValueError("map-scale dark components do not form a repeated LRA pattern")

    mask = np.isin(labels, repeated_ids) & source_domain & ~line_exclusion
    return LRADotEvidence(
        mask=mask,
        dark_distance_threshold=threshold,
        candidate_component_count=len(candidate_ids),
        repeated_component_count=len(repeated_ids),
        selected_pixel_count=int(np.count_nonzero(mask)),
        local_window_px=local_window,
        minimum_local_component_count=minimum_local_count,
        mapbox_line_exclusion_pixel_count=int(np.count_nonzero(line_exclusion)),
        mapbox_line_buffer_target_px=mapbox_line_buffer_target_px,
    )


def _infer_overprinted_hazard(
    observed_ids: np.ndarray,
    source_rgb: np.ndarray,
    source_domain: np.ndarray,
    dark_rgb: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Infer only dark overprint pixels with strong local hazard agreement."""

    color = np.asarray(dark_rgb, dtype=np.int16)
    dark_distance = np.linalg.norm(source_rgb.astype(np.int16) - color, axis=2)
    overprint = source_domain & (observed_ids == 0) & (dark_distance <= 60.0)
    known = observed_ids > 0
    if not np.any(known):
        raise ValueError("fire source contains no observed hazard pixels")
    distance_to_hazard = cv2.distanceTransform((~known).astype(np.uint8), cv2.DIST_L2, 5)
    counts = np.stack(
        [
            cv2.boxFilter(
                (observed_ids == class_id).astype(np.uint16),
                cv2.CV_32S,
                (17, 17),
                normalize=False,
            )
            for class_id in range(1, 4)
        ],
        axis=2,
    )
    total = np.sum(counts, axis=2)
    winner = np.argmax(counts, axis=2).astype(np.uint8) + 1
    winning_count = np.max(counts, axis=2)
    dominance = winning_count / np.maximum(total, 1)
    inferred = (
        overprint
        & (distance_to_hazard <= 8.0)
        & (total >= 12)
        & (dominance >= 0.72)
    )
    complete = observed_ids.copy()
    complete[inferred] = winner[inferred]
    return complete, inferred


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


def _regional_roundtrip_metrics(
    source_hazard_expected: np.ndarray,
    source_hazard_matched: np.ndarray,
    source_lra_expected: np.ndarray,
    source_lra_matched: np.ndarray,
    source_to_target_remap: tuple[np.ndarray, np.ndarray],
    target_shape: tuple[int, int],
    rows: int,
    columns: int,
) -> list[dict[str, Any]]:
    map_x, map_y = source_to_target_remap
    target_height, target_width = target_shape
    reports = []
    for row in range(rows):
        for column in range(columns):
            left, right = column * target_width / columns, (column + 1) * target_width / columns
            top, bottom = row * target_height / rows, (row + 1) * target_height / rows
            cell = (
                (map_x >= left)
                & (map_x < right)
                & (map_y >= top)
                & (map_y < bottom)
            )
            hazard_expected = cell & source_hazard_expected
            lra_expected = cell & source_lra_expected
            hazard_count = int(np.count_nonzero(hazard_expected))
            lra_count = int(np.count_nonzero(lra_expected))
            if hazard_count + lra_count == 0:
                continue
            hazard_match = int(np.count_nonzero(hazard_expected & source_hazard_matched))
            lra_match = int(np.count_nonzero(lra_expected & source_lra_matched))
            reports.append(
                {
                    "id": f"r{row + 1}-c{column + 1}",
                    "hazard_expected_pixel_count": hazard_count,
                    "hazard_match_pixel_count": hazard_match,
                    "lra_expected_pixel_count": lra_count,
                    "lra_match_pixel_count": lra_match,
                    "combined_match_fraction": (hazard_match + lra_match)
                    / (hazard_count + lra_count),
                }
            )
    return reports


def _render_hazard(ids: np.ndarray, entries: Sequence[LegendEntry]) -> np.ndarray:
    rgb = np.zeros((*ids.shape, 3), dtype=np.uint8)
    for entry in entries:
        rgb[ids == entry.class_id] = entry.rgb
    return rgb


def _diagnostic(
    aligned_source: np.ndarray,
    hazard_rgb: np.ndarray,
    lra_mask: np.ndarray,
    inferred: np.ndarray,
    lra_rgb: Sequence[int],
    *,
    maximum_height: int = 1500,
) -> np.ndarray:
    composite = hazard_rgb.copy()
    composite[lra_mask] = tuple(int(value) for value in lra_rgb)
    review = cv2.addWeighted(aligned_source, 0.5, composite, 0.5, 0.0)
    review[inferred] = (0, 255, 255)
    scale = min(1.0, maximum_height / aligned_source.shape[0])
    if scale < 1.0:
        size = (
            max(1, round(aligned_source.shape[1] * scale)),
            max(1, round(aligned_source.shape[0] * scale)),
        )
        panels = [
            cv2.resize(value, size, interpolation=cv2.INTER_AREA)
            for value in (aligned_source, hazard_rgb, composite, review)
        ]
    else:
        panels = [aligned_source, hazard_rgb, composite, review]
    output = np.concatenate(panels, axis=1)
    canvas = Image.fromarray(output)
    draw = ImageDraw.Draw(canvas)
    labels = (
        "aligned source",
        "three hazard classes",
        "hazard + independent LRA dots",
        "50/50 source-extraction; cyan inferred",
    )
    panel_width = panels[0].shape[1]
    for index, label in enumerate(labels):
        x = index * panel_width + 12
        draw.rectangle((x - 4, 8, x + 285, 36), fill=(0, 0, 0))
        draw.text((x, 12), label, fill=(255, 255, 255))
    return np.asarray(canvas)


def _legend_payload(
    hazard_entries: Sequence[LegendEntry],
    lra: LRALegendEvidence,
    working: WorkingRasterArtifact,
    tesseract_version: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "semantics_established",
        "source": {"path": str(working.source_path), "sha256": working.source_sha256},
        "tesseract_version": tesseract_version,
        "hazard_classes": [
            {
                "class_id": entry.class_id,
                "label": entry.label,
                "rgb": list(entry.rgb),
                "swatch_bbox": list(entry.swatch_bbox),
                "label_bbox": list(entry.label_bbox),
                "ocr_confidence": entry.ocr_confidence,
            }
            for entry in hazard_entries
        ],
        "overlapping_layers": [
            {
                "id": "local-responsibility-area",
                "label": lra.label,
                "kind": "binary_repeated_dot_symbol",
                "rgb": list(lra.dot_rgb),
                "label_bbox": list(lra.label_bbox),
                "swatch_bbox": list(lra.swatch_bbox),
                "ocr_confidence": lra.label_confidence,
                "legend_dot_component_count": lra.dot_component_count,
                "semantic_contract": (
                    "small dark components must match the legend ink and belong "
                    "to a locally repeated dot pattern; borders and text are excluded"
                ),
            }
        ],
    }


def run_fire_overlapping_layer_extraction(
    source_adapter_manifest_path: Path,
    accepted_alignment_path: Path,
    mapbox_manifest_path: Path,
    output_dir: Path,
    experiment_log: NoHumanExperimentLog,
    experiment_markdown_path: Path,
    experiment_json_path: Path,
    *,
    config: FireExtractionConfig = FireExtractionConfig(),
) -> FireExtractionResult:
    """Extract the solid hazard classes and independent LRA dot layer."""

    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise ValueError("fire extraction requires a fresh output directory")
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
    source_to_target = _source_to_reference_remap(transform, source_rgb.shape[:2])
    mapbox_line_buffer_target_px = 25
    target_known_lines = cv2.dilate(
        (reference.counties | reference.state_coast).astype(np.uint8),
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (mapbox_line_buffer_target_px, mapbox_line_buffer_target_px),
        ),
    )
    source_known_lines = _warp_target_to_source(
        target_known_lines, source_to_target
    ) > 0

    hazard_legend = detect_legend(
        working.working_raster_path, source_rgb, source_domain, output_dir
    )
    hazard_entries = tuple(
        sorted(hazard_legend.entries, key=lambda value: value.class_id)
    )
    labels = tuple(_normalize_label(entry.label) for entry in hazard_entries)
    if labels != EXPECTED_HAZARD_LABELS:
        raise ValueError(
            f"fire hazard legend must be {EXPECTED_HAZARD_LABELS}; found {labels}"
        )
    if any(entry.ocr_confidence < config.minimum_legend_confidence for entry in hazard_entries):
        raise ValueError("fire hazard legend OCR confidence is below the strict gate")
    words, combined_tsv, tesseract_version = _run_tesseract_ocr(
        working.working_raster_path
    )
    lra_legend = _detect_lra_legend(
        source_rgb,
        words,
        minimum_confidence=config.minimum_legend_confidence,
        minimum_dot_count=config.minimum_lra_legend_dots,
    )
    lra_evidence = _detect_lra_source_dots(
        source_rgb,
        source_domain,
        lra_legend,
        source_known_lines,
        mapbox_line_buffer_target_px=mapbox_line_buffer_target_px,
    )
    if lra_evidence.repeated_component_count < config.minimum_lra_map_components:
        raise ValueError("too few repeated map-scale components for an LRA layer")

    combined_legend_path = output_dir / "legend" / "semantic-layers.json"
    combined_legend_path.write_text(
        json.dumps(
            _legend_payload(
                hazard_entries, lra_legend, working, tesseract_version
            ),
            indent=2,
        )
        + "\n"
    )
    lra_tsv_path = output_dir / "legend" / "lra-ocr.tsv"
    lra_tsv_path.write_text(combined_tsv)

    palette_lab = _palette_lab(hazard_entries)
    source_observed_ids, _, _ = _classify(
        source_rgb,
        source_domain,
        palette_lab,
        config.maximum_hazard_lab_distance,
        config.minimum_hazard_margin,
    )
    source_complete_ids, source_inferred = _infer_overprinted_hazard(
        source_observed_ids, source_rgb, source_domain, lra_legend.dot_rgb
    )
    target_domain = reference.state_land & ~reference.water
    reference_to_source = _reference_to_source_remap(transform)
    aligned_source = _source_to_reference(
        source_rgb,
        transform,
        cv2.INTER_LINEAR,
        (255, 255, 255),
        reference_to_source,
    )

    previous_source_ids: np.ndarray | None = None
    previous_target_ids: np.ndarray | None = None
    previous_lra: np.ndarray | None = None
    iterations: list[FireExtractionIteration] = []
    all_artifacts: list[Path] = [
        *hazard_legend.artifacts,
        combined_legend_path,
        lra_tsv_path,
    ]
    accepted_path: Path | None = None

    for iteration_number in range(1, config.required_replay_count + 1):
        iteration_dir = output_dir / f"extraction-{iteration_number:02d}"
        iteration_dir.mkdir()
        target_complete_ids = _source_to_reference(
            source_complete_ids,
            transform,
            cv2.INTER_NEAREST,
            0,
            reference_to_source,
        )
        target_observed = _source_to_reference(
            (source_observed_ids > 0).astype(np.uint8),
            transform,
            cv2.INTER_NEAREST,
            0,
            reference_to_source,
        ) > 0
        target_inferred = _source_to_reference(
            source_inferred.astype(np.uint8),
            transform,
            cv2.INTER_NEAREST,
            0,
            reference_to_source,
        ) > 0
        target_lra = _source_to_reference(
            lra_evidence.mask.astype(np.uint8),
            transform,
            cv2.INTER_NEAREST,
            0,
            reference_to_source,
        ) > 0
        target_complete_ids[~target_domain] = 0
        target_observed &= target_domain
        target_inferred &= target_domain
        target_lra &= target_domain
        target_inferred &= ~target_observed

        reconstructed_source_ids = _warp_target_to_source(
            target_complete_ids, source_to_target
        )
        reconstructed_source_lra = _warp_target_to_source(
            target_lra.astype(np.uint8), source_to_target
        ) > 0
        source_expected = source_complete_ids > 0
        source_hazard_match = source_expected & (
            reconstructed_source_ids == source_complete_ids
        )
        hazard_roundtrip = float(
            np.count_nonzero(source_hazard_match)
            / max(np.count_nonzero(source_expected), 1)
        )
        lra_intersection = int(
            np.count_nonzero(reconstructed_source_lra & lra_evidence.mask)
        )
        lra_union = int(
            np.count_nonzero(reconstructed_source_lra | lra_evidence.mask)
        )
        lra_iou = lra_intersection / max(lra_union, 1)
        inferred_fraction = float(
            np.count_nonzero(source_inferred)
            / max(np.count_nonzero(source_complete_ids > 0), 1)
        )
        regional = _regional_roundtrip_metrics(
            source_expected,
            source_hazard_match,
            lra_evidence.mask,
            lra_evidence.mask & reconstructed_source_lra,
            source_to_target,
            target_domain.shape,
            config.geographic_rows,
            config.geographic_columns,
        )
        passing_cells = sum(
            report["combined_match_fraction"]
            >= config.minimum_geographic_match_fraction
            for report in regional
        )
        stable = (
            previous_source_ids is not None
            and previous_target_ids is not None
            and previous_lra is not None
            and np.array_equal(previous_source_ids, source_complete_ids)
            and np.array_equal(previous_target_ids, target_complete_ids)
            and np.array_equal(previous_lra, target_lra)
        )
        observed_class_ids = {
            int(value) for value in np.unique(target_complete_ids) if value > 0
        }
        gates: dict[str, Any] = {
            "exact_three_hazard_legend_classes": labels == EXPECTED_HAZARD_LABELS,
            "hazard_legend_ocr_confidence": all(
                entry.ocr_confidence >= config.minimum_legend_confidence
                for entry in hazard_entries
            ),
            "all_hazard_classes_preserved": observed_class_ids == {1, 2, 3},
            "lra_semantics_established": (
                lra_legend.label_confidence >= config.minimum_legend_confidence
                and lra_legend.dot_component_count >= config.minimum_lra_legend_dots
            ),
            "lra_repeated_map_components": {
                "passed": lra_evidence.repeated_component_count
                >= config.minimum_lra_map_components,
                "value": lra_evidence.repeated_component_count,
                "minimum": config.minimum_lra_map_components,
            },
            "hazard_inferred_fraction": {
                "passed": inferred_fraction <= config.maximum_inferred_hazard_fraction,
                "value": inferred_fraction,
                "maximum": config.maximum_inferred_hazard_fraction,
            },
            "hazard_source_roundtrip": {
                "passed": hazard_roundtrip
                >= config.minimum_hazard_roundtrip_fraction,
                "value": hazard_roundtrip,
                "minimum": config.minimum_hazard_roundtrip_fraction,
            },
            "lra_source_roundtrip_iou": {
                "passed": lra_iou >= config.minimum_lra_roundtrip_iou,
                "value": lra_iou,
                "minimum": config.minimum_lra_roundtrip_iou,
            },
            "geographic_source_diff": {
                "passed": passing_cells >= config.minimum_passing_geographic_cells,
                "value": passing_cells,
                "minimum": config.minimum_passing_geographic_cells,
                "supported_cells": len(regional),
            },
            "mapbox_water_and_exterior_empty": not bool(
                np.any(target_complete_ids[~target_domain] > 0)
                or np.any(target_lra[~target_domain])
            ),
            "observed_and_inferred_hazard_disjoint": not bool(
                np.any(target_observed & target_inferred)
            ),
            "successive_overlapping_layer_fixed_point": stable,
        }
        all_gates_pass = all(
            value if isinstance(value, bool) else bool(value["passed"])
            for value in gates.values()
        )
        decision = (
            "accept"
            if all_gates_pass
            else (
                "retry"
                if iteration_number < config.required_replay_count
                else "blocked"
            )
        )

        source_ids_path = iteration_dir / "source-hazard-class-id.png"
        source_observed_path = iteration_dir / "source-hazard-observed-mask.png"
        source_inferred_path = iteration_dir / "source-hazard-inferred-mask.png"
        source_lra_path = iteration_dir / "source-lra-dot-mask.png"
        source_diff_path = iteration_dir / "source-hazard-roundtrip-diff-mask.png"
        source_lra_diff_path = iteration_dir / "source-lra-roundtrip-diff-mask.png"
        target_ids_path = iteration_dir / "mapbox-hazard-class-id.png"
        target_observed_path = iteration_dir / "mapbox-hazard-observed-mask.png"
        target_inferred_path = iteration_dir / "mapbox-hazard-inferred-mask.png"
        target_lra_path = iteration_dir / "mapbox-lra-dot-mask.png"
        target_hazard_path = iteration_dir / "mapbox-hazard-reconstruction.png"
        target_composite_path = iteration_dir / "mapbox-composite-reconstruction.png"
        aligned_source_path = iteration_dir / "mapbox-aligned-source.png"
        diagnostic_path = iteration_dir / "source-extraction-diagnostic.png"
        hazard_rgb = _render_hazard(target_complete_ids, hazard_entries)
        composite = hazard_rgb.copy()
        composite[target_lra] = lra_legend.dot_rgb
        _save_ids(source_ids_path, source_complete_ids)
        _save_mask(source_observed_path, source_observed_ids > 0)
        _save_mask(source_inferred_path, source_inferred)
        _save_mask(source_lra_path, lra_evidence.mask)
        _save_mask(source_diff_path, source_expected & ~source_hazard_match)
        _save_mask(
            source_lra_diff_path,
            reconstructed_source_lra ^ lra_evidence.mask,
        )
        _save_ids(target_ids_path, target_complete_ids)
        _save_mask(target_observed_path, target_observed)
        _save_mask(target_inferred_path, target_inferred)
        _save_mask(target_lra_path, target_lra)
        _save_rgb(target_hazard_path, hazard_rgb)
        _save_rgb(target_composite_path, composite)
        _save_rgb(aligned_source_path, aligned_source)
        _save_rgb(
            diagnostic_path,
            _diagnostic(
                aligned_source,
                hazard_rgb,
                target_lra,
                target_inferred,
                lra_legend.dot_rgb,
            ),
        )
        artifact_paths = (
            source_ids_path,
            source_observed_path,
            source_inferred_path,
            source_lra_path,
            source_diff_path,
            source_lra_diff_path,
            target_ids_path,
            target_observed_path,
            target_inferred_path,
            target_lra_path,
            target_hazard_path,
            target_composite_path,
            aligned_source_path,
            diagnostic_path,
        )
        scores = {
            "hazard_legend_class_count": len(hazard_entries),
            "hazard_labels": [entry.label for entry in hazard_entries],
            "hazard_legend_ocr_confidences": [
                entry.ocr_confidence for entry in hazard_entries
            ],
            "lra_label_ocr_confidence": lra_legend.label_confidence,
            "lra_legend_dot_component_count": lra_legend.dot_component_count,
            "lra_candidate_map_component_count": lra_evidence.candidate_component_count,
            "lra_repeated_map_component_count": lra_evidence.repeated_component_count,
            "source_lra_pixel_count": lra_evidence.selected_pixel_count,
            "source_mapbox_line_exclusion_pixel_count": (
                lra_evidence.mapbox_line_exclusion_pixel_count
            ),
            "mapbox_lra_pixel_count": int(np.count_nonzero(target_lra)),
            "source_hazard_observed_pixel_count": int(
                np.count_nonzero(source_observed_ids > 0)
            ),
            "source_hazard_inferred_pixel_count": int(np.count_nonzero(source_inferred)),
            "source_hazard_inferred_fraction": inferred_fraction,
            "mapbox_hazard_observed_pixel_count": int(np.count_nonzero(target_observed)),
            "mapbox_hazard_inferred_pixel_count": int(np.count_nonzero(target_inferred)),
            "hazard_source_roundtrip_fraction": hazard_roundtrip,
            "lra_source_roundtrip_iou": lra_iou,
            "class_source_observed_pixel_counts": {
                _normalize_label(entry.label): int(
                    np.count_nonzero(source_observed_ids == entry.class_id)
                )
                for entry in hazard_entries
            },
            "class_mapbox_pixel_counts": {
                _normalize_label(entry.label): int(
                    np.count_nonzero(target_complete_ids == entry.class_id)
                )
                for entry in hazard_entries
            },
            "geographic_cells": regional,
            "successive_source_hazard_equal": stable,
            "successive_mapbox_hazard_equal": stable,
            "successive_mapbox_lra_equal": stable,
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
                    "lra_detection": {
                        "legend": {
                            "rgb": list(lra_legend.dot_rgb),
                            "swatch_bbox": list(lra_legend.swatch_bbox),
                            "median_dot_area": lra_legend.median_dot_area,
                            "median_dot_width": lra_legend.median_dot_width,
                            "median_dot_height": lra_legend.median_dot_height,
                            "median_column_spacing": lra_legend.median_column_spacing,
                            "median_row_spacing": lra_legend.median_row_spacing,
                        },
                        "map": {
                            "dark_distance_threshold": lra_evidence.dark_distance_threshold,
                            "local_window_px": lra_evidence.local_window_px,
                            "minimum_local_component_count": (
                                lra_evidence.minimum_local_component_count
                            ),
                            "mapbox_line_buffer_target_px": (
                                lra_evidence.mapbox_line_buffer_target_px
                            ),
                            "mapbox_line_exclusion_pixel_count": (
                                lra_evidence.mapbox_line_exclusion_pixel_count
                            ),
                        },
                    },
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
                    "authoritative_original_webp_pixels",
                    "automatic_hazard_swatch_and_ocr_legend",
                    "automatic_lra_label_and_repeated_dot_symbol",
                    "accepted_automatic_mapbox_alignment",
                    "pinned_mapbox_land_and_water",
                    "source_reconstruction_diff",
                    "deterministic_overlapping_layer_fixed_point_replay",
                ],
            ),
            method=(
                "automatic three-class hazard color extraction plus independent "
                "legend-derived repeated LRA dot layer, Mapbox land/water clipping, "
                "source reconstruction, geographic diff, and fixed-point replay"
            ),
            artifacts=[
                {"path": str(path), "sha256": _sha256(path)}
                for path in (*complete_artifacts, combined_legend_path)
            ],
        )
        experiment_log.write(experiment_markdown_path, experiment_json_path)
        iterations.append(
            FireExtractionIteration(
                iteration_number,
                decision,
                scores,
                gates,
                report_path,
                tuple(complete_artifacts),
            )
        )
        all_artifacts.extend(complete_artifacts)
        previous_source_ids = source_complete_ids.copy()
        previous_target_ids = target_complete_ids.copy()
        previous_lra = target_lra.copy()
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
                            "sha256": _sha256(combined_legend_path),
                        },
                        "accepted_iteration": f"extraction-{iteration_number:02d}",
                        "layers": {
                            "hazard_level": {
                                "kind": "mutually_exclusive_categorical",
                                "class_ids": f"extraction-{iteration_number:02d}/mapbox-hazard-class-id.png",
                                "observed_mask": f"extraction-{iteration_number:02d}/mapbox-hazard-observed-mask.png",
                                "inferred_mask": f"extraction-{iteration_number:02d}/mapbox-hazard-inferred-mask.png",
                            },
                            "local_responsibility_area": {
                                "kind": "independent_binary_dot_symbol",
                                "mask": f"extraction-{iteration_number:02d}/mapbox-lra-dot-mask.png",
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
        return FireExtractionResult(
            "accepted",
            "hazard, LRA, source-diff, geographic, and fixed-point gates passed",
            tuple(iterations),
            accepted_path,
            tuple(all_artifacts),
        )
    blocker = (
        "Fire overlapping-layer extraction did not pass every semantic, "
        "source-diff, geographic, and fixed-point gate after two deterministic passes"
    )
    experiment_log.finalize("blocked", blocker)
    experiment_log.write(experiment_markdown_path, experiment_json_path)
    return FireExtractionResult(
        "blocked", blocker, tuple(iterations), None, tuple(all_artifacts)
    )
