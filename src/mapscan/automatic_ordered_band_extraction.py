"""No-human extraction for ordered, shaded thematic color bands.

This adapter handles maps whose legend defines an ordered set of color
manifolds rather than one flat RGB value per class.  It was introduced for the
California earthquake shaking-potential map, where six adjacent legend rows
(VI through XI) form one continuous yellow-to-red ramp and the same ramp is
modulated by hillshade in the map body.

The adapter accepts only the source-clean manifest, an accepted automatic
alignment, and the pinned Mapbox reference.  It does not expose control points,
paint, stamps, palette overrides, or review decisions.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .automatic_alignment_loop import load_pinned_mapbox_reference
from .automatic_categorical_extraction import (
    _artifact,
    _load_accepted_alignment,
    _nearest_completion,
    _parse_tesseract_tsv,
    _reference_to_source_remap,
    _run_tesseract_tsv,
    _save_ids,
    _save_mask,
    _save_rgb,
    _source_data_mask,
    _source_to_reference,
    _source_to_reference_remap,
    _tesseract_version,
)
from .experiment_log import NoHumanExperimentLog, automatic_provenance
from .source_working_raster import WorkingRasterArtifact, load_working_raster_artifact


SCHEMA_VERSION = "mapscan.automatic-ordered-band-extraction.v1"
PRODUCER = "mapscan.automatic_ordered_band_extraction"
EXPECTED_SOURCE_TYPE = "ordered_gradient_bands"

# The semantic order is part of the source data model.  The image must still
# prove every descriptor through OCR; position alone is never sufficient.
ORDERED_SEMANTICS = (
    ("VI", "STRONG SHAKING"),
    ("VII", "VERY STRONG SHAKING"),
    ("VIII", "SEVERE SHAKING"),
    ("IX", "VIOLENT SHAKING"),
    ("X", "EXTREME SHAKING"),
    ("XI", "DEVASTATING SHAKING"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _all_gates_pass(gates: Mapping[str, Any]) -> bool:
    return all(
        value if isinstance(value, bool) else bool(value["passed"])
        for value in gates.values()
    )


@dataclass(frozen=True)
class OrderedBandConfig:
    required_replay_count: int = 2
    minimum_descriptor_ocr_confidence: float = 90.0
    minimum_observed_fraction: float = 0.72
    maximum_inferred_fraction: float = 0.28
    maximum_semantic_mismatch_fraction: float = 0.01
    minimum_source_roundtrip_fraction: float = 0.94
    minimum_pixel_saturation: int = 40
    minimum_pixel_value: int = 40
    geographic_rows: int = 4
    geographic_columns: int = 3
    minimum_geographic_match_fraction: float = 0.90
    minimum_passing_geographic_cells: int = 8

    def __post_init__(self) -> None:
        if self.required_replay_count != 2:
            raise ValueError("ordered-band extraction requires exactly two replays")
        if not 0 <= self.minimum_pixel_saturation <= 255:
            raise ValueError("minimum_pixel_saturation must be a byte")
        if not 0 <= self.minimum_pixel_value <= 255:
            raise ValueError("minimum_pixel_value must be a byte")
        for value in (
            self.minimum_observed_fraction,
            self.maximum_inferred_fraction,
            self.maximum_semantic_mismatch_fraction,
            self.minimum_source_roundtrip_fraction,
            self.minimum_geographic_match_fraction,
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError("ordered-band fraction gates must be in [0, 1]")


@dataclass(frozen=True)
class OrderedBandEntry:
    class_id: int
    roman: str
    label: str
    swatch_bbox: tuple[int, int, int, int]
    representative_rgb: tuple[int, int, int]
    hue_minimum: float
    hue_maximum: float
    hue_center: float
    descriptor_ocr_confidence: float
    descriptor_ocr_text: str


@dataclass(frozen=True)
class OrderedBandLegend:
    entries: tuple[OrderedBandEntry, ...]
    tesseract_version: str
    row_step_px: float
    common_swatch_x_range: tuple[int, int]
    artifact_paths: tuple[Path, ...]


@dataclass(frozen=True)
class OrderedBandIteration:
    iteration: int
    decision: str
    scores: Mapping[str, Any]
    gates: Mapping[str, Any]
    report_path: Path
    artifact_paths: tuple[Path, ...]


@dataclass(frozen=True)
class AutomaticOrderedBandExtractionResult:
    status: str
    stop_reason: str
    legend: OrderedBandLegend
    iterations: tuple[OrderedBandIteration, ...]
    accepted_extraction_path: Path | None
    artifact_paths: tuple[Path, ...]

    @property
    def accepted(self) -> Path | None:
        return self.accepted_extraction_path


def _signed_warm_hue(hsv: np.ndarray) -> np.ndarray:
    """Map OpenCV's red wrap to a monotone red-to-yellow coordinate."""

    hue = hsv[..., 0].astype(np.float32)
    hue[hue >= 150.0] -= 180.0
    return hue


def _warm_support(rgb: np.ndarray, minimum_saturation: int, minimum_value: int) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue = _signed_warm_hue(hsv)
    return (
        (hue >= -30.0)
        & (hue <= 40.0)
        & (hsv[..., 1] >= minimum_saturation)
        & (hsv[..., 2] >= minimum_value)
    )


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    changes = np.diff(np.pad(mask.astype(np.int8), (1, 1)))
    return list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))


def _candidate_row_segments(
    warm_layout: np.ndarray,
) -> list[tuple[int, int, tuple[int, int]]]:
    """Find long, filled warm rectangles without assuming legend coordinates."""

    height, width = warm_layout.shape
    row_support = warm_layout.sum(axis=1).astype(np.float32)
    row_support = cv2.blur(row_support[:, None], (1, 5)).ravel()
    row_mask = row_support >= max(20.0, width * 0.08)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        row_mask[:, None].astype(np.uint8), 8
    )
    candidates: list[tuple[int, int, tuple[int, int]]] = []
    minimum_height = max(8, round(height * 0.015))
    maximum_height = max(minimum_height + 1, round(height * 0.06))
    for component in range(1, component_count):
        top = int(stats[component, cv2.CC_STAT_TOP])
        band_height = int(stats[component, cv2.CC_STAT_HEIGHT])
        if not minimum_height <= band_height <= maximum_height:
            continue
        bottom = top + band_height
        trim = max(2, round(band_height * 0.08))
        stable_columns = warm_layout[top + trim : bottom - trim].mean(axis=0) >= 0.85
        horizontal_runs = sorted(
            _runs(stable_columns), key=lambda value: value[1] - value[0], reverse=True
        )
        if not horizontal_runs:
            continue
        left, right = horizontal_runs[0]
        if right - left < width * 0.08:
            continue
        candidates.append((top, bottom, (int(left), int(right))))
    return candidates


def _select_regular_six(
    candidates: Sequence[tuple[int, int, tuple[int, int]]]
) -> tuple[tuple[int, int, tuple[int, int]], ...]:
    """Select the most regular six consecutive rows with a common x span."""

    ordered = sorted(candidates)
    if len(ordered) < 6:
        raise ValueError("No complete six-row ordered gradient legend was found")
    scored = []
    for start in range(len(ordered) - 5):
        group = ordered[start : start + 6]
        centers = np.asarray([(top + bottom) / 2.0 for top, bottom, _ in group])
        steps = np.diff(centers)
        heights = np.asarray([bottom - top for top, bottom, _ in group], dtype=float)
        lefts = np.asarray([span[0] for _, _, span in group], dtype=float)
        rights = np.asarray([span[1] for _, _, span in group], dtype=float)
        common_left, common_right = int(np.max(lefts)), int(np.min(rights))
        common_width = common_right - common_left
        median_width = float(np.median(rights - lefts))
        if common_width <= 0 or common_width < 0.85 * median_width:
            continue
        step_cv = float(np.std(steps) / max(np.mean(steps), 1.0))
        height_cv = float(np.std(heights) / max(np.mean(heights), 1.0))
        x_cv = float((np.std(lefts) + np.std(rights)) / max(median_width, 1.0))
        gap_ratio = float(np.mean(steps) / max(np.mean(heights), 1.0))
        if not 0.90 <= gap_ratio <= 1.40:
            continue
        score = step_cv + height_cv + x_cv
        scored.append((score, start, group))
    if not scored:
        raise ValueError("Warm rectangles do not form one regular six-row legend")
    scored.sort(key=lambda value: (value[0], value[1]))
    if scored[0][0] > 0.18:
        raise ValueError("Six-row gradient legend regularity is below the strict gate")
    return tuple(scored[0][2])


def _descriptor_ocr(
    source_rgb: np.ndarray,
    bbox: tuple[int, int, int, int],
    expected_descriptor: str,
) -> tuple[str, float]:
    left, top, width, height = bbox
    crop = source_rgb[top : top + height, left : left + width]
    enlarged = cv2.resize(crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(enlarged, cv2.COLOR_RGB2GRAY)
    # Colored type and the pale diamond outlines trade places in contrast
    # across the ramp.  Require semantics in either deterministic OCR view and
    # use the best confidence per required token.
    passes = [
        _parse_tesseract_tsv(_run_tesseract_tsv(enlarged, psm=6)),
        _parse_tesseract_tsv(_run_tesseract_tsv(gray, psm=6)),
    ]
    retained = [word for words in passes for word in words if word.confidence >= 20.0]
    text = " | ".join(
        " ".join(word.text for word in words if word.confidence >= 20.0)
        for words in passes
    )
    normalized = re.sub(r"[^A-Z]+", " ", text.upper()).strip()
    expected_tokens = expected_descriptor.split()
    matched = []
    for token in expected_tokens:
        token_matches = [
            word for word in retained if re.sub(r"[^A-Z]", "", word.text.upper()) == token
        ]
        if not token_matches:
            raise ValueError(
                f"Ordered-band legend OCR did not establish {expected_descriptor!r}: {text!r}"
            )
        matched.append(max(token_matches, key=lambda value: value.confidence).confidence)
    return normalized, float(min(matched))


def detect_ordered_band_legend(
    source_path: Path,
    source_rgb: np.ndarray,
    source_land_domain: np.ndarray,
    output_dir: Path,
    *,
    config: OrderedBandConfig = OrderedBandConfig(),
) -> OrderedBandLegend:
    """Recover six gradient rows, OCR semantics, and their hue manifolds."""

    legend_dir = output_dir / "legend"
    legend_dir.mkdir(parents=True, exist_ok=True)
    warm = _warm_support(
        source_rgb, config.minimum_pixel_saturation, config.minimum_pixel_value
    )
    rows = _select_regular_six(_candidate_row_segments(warm & ~source_land_domain))
    common_left = max(row[2][0] for row in rows)
    common_right = min(row[2][1] for row in rows)
    if common_right - common_left < source_rgb.shape[1] * 0.08:
        raise ValueError("Ordered-band legend has no common filled swatch interval")

    executable = shutil.which("tesseract")
    if executable is None:
        raise RuntimeError("Ordered-band legend semantics require Tesseract")
    version = _tesseract_version(executable)
    entries: list[OrderedBandEntry] = []
    # Image order is XI -> VI; semantic ids are VI=1 -> XI=6.
    image_semantics = tuple(reversed(ORDERED_SEMANTICS))
    swatch_width = common_right - common_left
    descriptor_left = max(0, common_right - round(swatch_width * 0.05))
    descriptor_right = min(
        source_rgb.shape[1], common_right + round(swatch_width * 0.95)
    )
    for image_index, ((top, bottom, _), (roman, descriptor)) in enumerate(
        zip(rows, image_semantics)
    ):
        trim = max(3, round((bottom - top) * 0.06))
        pixels = source_rgb[
            top + trim : bottom - trim,
            common_left + 4 : common_right - 4,
        ]
        pixel_hsv = cv2.cvtColor(pixels, cv2.COLOR_RGB2HSV)
        pixel_hue = _signed_warm_hue(pixel_hsv)
        valid = (
            (pixel_hsv[..., 1] >= config.minimum_pixel_saturation)
            & (pixel_hsv[..., 2] >= config.minimum_pixel_value)
            & (pixel_hue >= -30.0)
            & (pixel_hue <= 40.0)
        )
        hue_values = pixel_hue[valid]
        if hue_values.size < 100:
            raise ValueError(f"Ordered-band swatch {roman} has too little color evidence")
        hue_minimum, hue_center, hue_maximum = np.quantile(
            hue_values, [0.01, 0.5, 0.99]
        )
        representative = tuple(
            int(value) for value in np.median(pixels[valid], axis=0).round()
        )
        ocr_bbox = (
            descriptor_left,
            top,
            descriptor_right - descriptor_left,
            bottom - top,
        )
        ocr_text, confidence = _descriptor_ocr(source_rgb, ocr_bbox, descriptor)
        class_id = len(ORDERED_SEMANTICS) - image_index
        entries.append(
            OrderedBandEntry(
                class_id=class_id,
                roman=roman,
                label=f"{roman} {descriptor.title()}",
                swatch_bbox=(common_left, top, common_right - common_left, bottom - top),
                representative_rgb=representative,
                hue_minimum=float(hue_minimum),
                hue_maximum=float(hue_maximum),
                hue_center=float(hue_center),
                descriptor_ocr_confidence=confidence,
                descriptor_ocr_text=ocr_text,
            )
        )
    entries.sort(key=lambda value: value.class_id)
    if any(
        entry.descriptor_ocr_confidence < config.minimum_descriptor_ocr_confidence
        for entry in entries
    ):
        raise ValueError("Ordered-band legend descriptor OCR is below the strict gate")

    centers = np.asarray([(top + bottom) / 2.0 for top, bottom, _ in rows])
    row_step = float(np.median(np.diff(centers)))
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "semantics_established",
        "source": {"path": str(source_path), "sha256": _sha256(source_path)},
        "tesseract_version": version,
        "row_step_px": row_step,
        "common_swatch_x_range": [common_left, common_right],
        "classes": [
            {
                "class_id": entry.class_id,
                "roman": entry.roman,
                "label": entry.label,
                "swatch_bbox": list(entry.swatch_bbox),
                "representative_rgb": list(entry.representative_rgb),
                "hue_manifold": {
                    "minimum": entry.hue_minimum,
                    "center": entry.hue_center,
                    "maximum": entry.hue_maximum,
                    "coordinate": "opencv_hue_with_red_wrap_shifted_by_minus_180",
                },
                "descriptor_ocr_text": entry.descriptor_ocr_text,
                "descriptor_ocr_confidence": entry.descriptor_ocr_confidence,
            }
            for entry in entries
        ],
    }
    legend_path = legend_dir / "ordered-band-legend.json"
    legend_path.write_text(json.dumps(payload, indent=2) + "\n")
    preview = Image.fromarray(source_rgb.copy())
    draw = ImageDraw.Draw(preview)
    for entry in entries:
        left, top, width, height = entry.swatch_bbox
        draw.rectangle((left, top, left + width, top + height), outline=(0, 255, 255), width=4)
        draw.text((left + 8, top + 8), entry.label, fill=(0, 0, 0))
    preview_path = legend_dir / "ordered-band-detection.png"
    preview.save(preview_path, optimize=True)
    return OrderedBandLegend(
        tuple(entries), version, row_step, (common_left, common_right),
        (legend_path, preview_path),
    )


def _classify_ordered_hue(
    source_rgb: np.ndarray,
    domain: np.ndarray,
    entries: Sequence[OrderedBandEntry],
    config: OrderedBandConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Classify thematic pixels by distance to the six legend hue manifolds."""

    hsv = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2HSV)
    hue = _signed_warm_hue(hsv)
    warm = (
        domain
        & (hsv[..., 1] >= config.minimum_pixel_saturation)
        & (hsv[..., 2] >= config.minimum_pixel_value)
        & (hue >= min(entry.hue_minimum for entry in entries) - 5.0)
        & (hue <= max(entry.hue_maximum for entry in entries) + 8.0)
    )
    distances = []
    for entry in entries:
        distances.append(
            np.maximum(
                np.maximum(entry.hue_minimum - hue, hue - entry.hue_maximum), 0.0
            )
        )
    stacked = np.stack(distances, axis=-1)
    nearest_index = np.argmin(stacked, axis=-1)
    observed_ids = np.zeros(domain.shape, dtype=np.uint8)
    observed_ids[warm] = nearest_index[warm].astype(np.uint8) + 1
    complete_ids, inferred = _nearest_completion(observed_ids, domain)
    return observed_ids, complete_ids, inferred, warm


def _render_classes(ids: np.ndarray, entries: Sequence[OrderedBandEntry]) -> np.ndarray:
    output = np.zeros((*ids.shape, 3), dtype=np.uint8)
    for entry in entries:
        output[ids == entry.class_id] = entry.representative_rgb
    return output


def _warp_target_to_source(
    target: np.ndarray, source_to_target: tuple[np.ndarray, np.ndarray]
) -> np.ndarray:
    return cv2.remap(
        target,
        source_to_target[0],
        source_to_target[1],
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _geographic_metrics(
    expected: np.ndarray,
    matched: np.ndarray,
    source_to_target: tuple[np.ndarray, np.ndarray],
    target_shape: tuple[int, int],
    rows: int,
    columns: int,
) -> list[dict[str, Any]]:
    map_x, map_y = source_to_target
    target_height, target_width = target_shape
    reports = []
    for row in range(rows):
        for column in range(columns):
            cell = (
                (map_x >= column * target_width / columns)
                & (map_x < (column + 1) * target_width / columns)
                & (map_y >= row * target_height / rows)
                & (map_y < (row + 1) * target_height / rows)
            )
            cell_expected = cell & expected
            count = int(np.count_nonzero(cell_expected))
            if not count:
                continue
            match_count = int(np.count_nonzero(cell_expected & matched))
            reports.append(
                {
                    "id": f"r{row + 1}-c{column + 1}",
                    "source_expected_pixel_count": count,
                    "source_match_pixel_count": match_count,
                    "source_match_fraction": match_count / count,
                }
            )
    return reports


def _diagnostic(
    aligned_source: np.ndarray,
    reconstruction: np.ndarray,
    observed: np.ndarray,
    inferred: np.ndarray,
    *,
    maximum_height: int = 1500,
) -> np.ndarray:
    overlay = cv2.addWeighted(aligned_source, 0.5, reconstruction, 0.5, 0.0)
    overlay[inferred] = (0, 255, 255)
    scale = min(1.0, maximum_height / aligned_source.shape[0])
    if scale < 1.0:
        size = (
            max(1, round(aligned_source.shape[1] * scale)),
            max(1, round(aligned_source.shape[0] * scale)),
        )
        panels = [
            cv2.resize(panel, size, interpolation=cv2.INTER_AREA)
            for panel in (aligned_source, reconstruction, overlay)
        ]
    else:
        panels = [aligned_source, reconstruction, overlay]
    canvas = Image.fromarray(np.concatenate(panels, axis=1))
    draw = ImageDraw.Draw(canvas)
    for index, label in enumerate(
        ("aligned original", "VI-XI class reconstruction", "50/50; cyan inferred")
    ):
        x = index * panels[0].shape[1] + 10
        draw.rectangle((x - 4, 7, x + 275, 35), fill=(0, 0, 0))
        draw.text((x, 11), label, fill=(255, 255, 255))
    return np.asarray(canvas)


def _write_geographic_crops(
    iteration_dir: Path,
    aligned_source: np.ndarray,
    reconstruction: np.ndarray,
    inferred: np.ndarray,
    target_domain: np.ndarray,
    rows: int,
    columns: int,
) -> list[Path]:
    """Persist every supported geographic cell for visual source auditing."""

    root = iteration_dir / "geographic-crops"
    root.mkdir()
    height, width = target_domain.shape
    paths = []
    for row in range(rows):
        top, bottom = round(row * height / rows), round((row + 1) * height / rows)
        for column in range(columns):
            left, right = (
                round(column * width / columns),
                round((column + 1) * width / columns),
            )
            domain_crop = target_domain[top:bottom, left:right]
            if not np.any(domain_crop):
                continue
            source_crop = aligned_source[top:bottom, left:right]
            reconstruction_crop = reconstruction[top:bottom, left:right]
            overlay = cv2.addWeighted(source_crop, 0.5, reconstruction_crop, 0.5, 0.0)
            overlay[inferred[top:bottom, left:right]] = (0, 255, 255)
            path = root / f"r{row + 1}-c{column + 1}-source-extraction.png"
            _save_rgb(path, np.concatenate((source_crop, reconstruction_crop, overlay), axis=1))
            paths.append(path)
    return paths


def _validate_inputs(
    source_adapter_manifest_path: Path,
    accepted_alignment_path: Path,
    mapbox_manifest_path: Path,
    experiment_log: NoHumanExperimentLog,
) -> tuple[WorkingRasterArtifact, Any, dict[str, Any]]:
    working = load_working_raster_artifact(source_adapter_manifest_path)
    authority = working.manifest.get("authority", {})
    if authority != {
        "manual_input_used": False,
        "original_source_authoritative": True,
        "prior_alignment_used": False,
        "prior_extraction_used": False,
    }:
        raise ValueError("ordered-band source-clean authority is not pristine")
    if experiment_log.data["source"].get("sha256") != working.source_sha256:
        raise ValueError("experiment and source-clean original hashes disagree")
    if experiment_log.data["source"].get("source_type") != EXPECTED_SOURCE_TYPE:
        raise ValueError("ordered-band extractor requires ordered_gradient_bands")
    if experiment_log.data["extraction"]["iterations"]:
        raise ValueError("ordered-band extraction log already contains attempts")
    reference = load_pinned_mapbox_reference(mapbox_manifest_path)
    accepted_count = experiment_log.data["alignment"]["accepted_automatic_iteration_count"]
    if accepted_count is None:
        raise ValueError("ordered-band extraction requires accepted alignment")
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


def run_automatic_ordered_band_extraction(
    source_adapter_manifest_path: Path,
    accepted_alignment_path: Path,
    mapbox_manifest_path: Path,
    output_dir: Path,
    experiment_log: NoHumanExperimentLog,
    experiment_markdown_path: Path,
    experiment_json_path: Path,
    *,
    config: OrderedBandConfig = OrderedBandConfig(),
) -> AutomaticOrderedBandExtractionResult:
    """Extract six ordered bands and require source, geographic, and replay gates."""

    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise ValueError("ordered-band extraction requires a fresh output directory")
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
    source_domain_path = output_dir / "source-mapbox-land-minus-water-mask.png"
    _save_mask(source_domain_path, source_domain)
    legend = detect_ordered_band_legend(
        working.working_raster_path,
        source_rgb,
        source_domain,
        output_dir,
        config=config,
    )
    source_observed_ids, source_complete_ids, source_inferred, meaningful = (
        _classify_ordered_hue(source_rgb, source_domain, legend.entries, config)
    )
    source_reconstruction = _render_classes(source_complete_ids, legend.entries)
    reconstructed_ids, _, _, _ = _classify_ordered_hue(
        source_reconstruction, source_domain, legend.entries, config
    )
    semantic_mismatch = meaningful & (reconstructed_ids != source_observed_ids)

    target_domain = reference.state_land & ~reference.water
    reference_to_source = _reference_to_source_remap(transform)
    source_to_target = _source_to_reference_remap(transform, source_rgb.shape[:2])
    aligned_source = _source_to_reference(
        source_rgb, transform, cv2.INTER_LINEAR, (255, 255, 255), reference_to_source
    )
    base_target_ids = _source_to_reference(
        source_complete_ids, transform, cv2.INTER_NEAREST, 0, reference_to_source
    )
    base_target_observed = _source_to_reference(
        (source_observed_ids > 0).astype(np.uint8),
        transform,
        cv2.INTER_NEAREST,
        0,
        reference_to_source,
    ) > 0
    base_target_ids[~target_domain] = 0
    if np.any(target_domain & (base_target_ids == 0)):
        base_target_ids, _ = _nearest_completion(base_target_ids, target_domain)
    base_target_observed &= target_domain
    base_target_inferred = target_domain & ~base_target_observed

    observed_count = int(np.count_nonzero(source_observed_ids))
    domain_count = int(np.count_nonzero(source_domain))
    inferred_count = int(np.count_nonzero(source_inferred))
    meaningful_count = int(np.count_nonzero(meaningful))
    mismatch_count = int(np.count_nonzero(semantic_mismatch))
    observed_fraction = observed_count / max(domain_count, 1)
    inferred_fraction = inferred_count / max(domain_count, 1)
    mismatch_fraction = mismatch_count / max(meaningful_count, 1)
    class_observed_counts = {
        entry.label: int(np.count_nonzero(source_observed_ids == entry.class_id))
        for entry in legend.entries
    }

    previous_source_ids: np.ndarray | None = None
    previous_target_ids: np.ndarray | None = None
    iterations: list[OrderedBandIteration] = []
    all_artifacts: list[Path] = [source_domain_path, *legend.artifact_paths]
    accepted_path: Path | None = None
    for iteration_number in range(1, config.required_replay_count + 1):
        iteration_dir = output_dir / f"extraction-{iteration_number:02d}"
        iteration_dir.mkdir()
        target_ids = base_target_ids.copy()
        target_observed = base_target_observed.copy()
        target_inferred = base_target_inferred.copy()
        target_reconstruction = _render_classes(target_ids, legend.entries)
        source_roundtrip_ids = _warp_target_to_source(target_ids, source_to_target)
        source_expected = source_domain & (source_complete_ids > 0)
        source_matched = source_expected & (source_roundtrip_ids == source_complete_ids)
        source_roundtrip_fraction = int(np.count_nonzero(source_matched)) / max(
            int(np.count_nonzero(source_expected)), 1
        )
        regional = _geographic_metrics(
            source_expected,
            source_matched,
            source_to_target,
            target_domain.shape,
            config.geographic_rows,
            config.geographic_columns,
        )
        passing_cells = sum(
            cell["source_match_fraction"] >= config.minimum_geographic_match_fraction
            for cell in regional
        )
        stable = (
            previous_source_ids is not None
            and previous_target_ids is not None
            and np.array_equal(previous_source_ids, source_complete_ids)
            and np.array_equal(previous_target_ids, target_ids)
        )
        target_classes = {int(value) for value in np.unique(target_ids) if value}
        gates: dict[str, Any] = {
            "exact_six_ordered_legend_semantics": [
                (entry.roman, entry.label.split(" ", 1)[1].upper())
                for entry in legend.entries
            ]
            == list(ORDERED_SEMANTICS),
            "descriptor_ocr_confidence": {
                "passed": all(
                    entry.descriptor_ocr_confidence
                    >= config.minimum_descriptor_ocr_confidence
                    for entry in legend.entries
                ),
                "minimum": config.minimum_descriptor_ocr_confidence,
                "values": [
                    entry.descriptor_ocr_confidence for entry in legend.entries
                ],
            },
            "all_six_classes_preserved": target_classes == set(range(1, 7)),
            "observed_source_coverage": {
                "passed": observed_fraction >= config.minimum_observed_fraction,
                "value": observed_fraction,
                "minimum": config.minimum_observed_fraction,
            },
            "inferred_source_fraction": {
                "passed": inferred_fraction <= config.maximum_inferred_fraction,
                "value": inferred_fraction,
                "maximum": config.maximum_inferred_fraction,
            },
            "meaningful_source_reconstruction_mismatch": {
                "passed": mismatch_fraction <= config.maximum_semantic_mismatch_fraction,
                "value": mismatch_fraction,
                "maximum": config.maximum_semantic_mismatch_fraction,
            },
            "source_domain_complete": bool(np.all(source_complete_ids[source_domain] > 0)),
            "source_layout_empty": not bool(np.any(source_complete_ids[~source_domain] > 0)),
            "mapbox_water_and_exterior_empty": not bool(np.any(target_ids[~target_domain] > 0)),
            "target_domain_complete": bool(np.all(target_ids[target_domain] > 0)),
            "observed_and_inferred_disjoint": not bool(
                np.any(target_observed & target_inferred)
            ),
            "source_alignment_roundtrip": {
                "passed": source_roundtrip_fraction
                >= config.minimum_source_roundtrip_fraction,
                "value": source_roundtrip_fraction,
                "minimum": config.minimum_source_roundtrip_fraction,
            },
            "geographic_source_diff": {
                "passed": passing_cells >= config.minimum_passing_geographic_cells,
                "value": passing_cells,
                "minimum": config.minimum_passing_geographic_cells,
                "supported_cells": len(regional),
            },
            "successive_ordered_band_fixed_point": stable,
        }
        decision = "accept" if _all_gates_pass(gates) else (
            "retry" if iteration_number < config.required_replay_count else "blocked"
        )

        source_ids_path = iteration_dir / "source-class-id.png"
        source_observed_path = iteration_dir / "source-observed-mask.png"
        source_inferred_path = iteration_dir / "source-inferred-mask.png"
        source_occluded_path = iteration_dir / "source-occluded-nonthematic-mask.png"
        source_reconstruction_path = iteration_dir / "source-reconstruction.png"
        source_diff_path = iteration_dir / "source-semantic-diff-mask.png"
        target_ids_path = iteration_dir / "mapbox-class-id.png"
        target_observed_path = iteration_dir / "mapbox-observed-mask.png"
        target_inferred_path = iteration_dir / "mapbox-inferred-mask.png"
        target_reconstruction_path = iteration_dir / "mapbox-reconstruction.png"
        aligned_source_path = iteration_dir / "mapbox-aligned-source.png"
        diagnostic_path = iteration_dir / "source-extraction-comparison.png"
        source_roundtrip_diff_path = iteration_dir / "source-roundtrip-diff-mask.png"
        _save_ids(source_ids_path, source_complete_ids)
        _save_mask(source_observed_path, source_observed_ids > 0)
        _save_mask(source_inferred_path, source_inferred)
        _save_mask(source_occluded_path, source_domain & ~(source_observed_ids > 0))
        _save_rgb(source_reconstruction_path, source_reconstruction)
        _save_mask(source_diff_path, semantic_mismatch)
        _save_ids(target_ids_path, target_ids)
        _save_mask(target_observed_path, target_observed)
        _save_mask(target_inferred_path, target_inferred)
        _save_rgb(target_reconstruction_path, target_reconstruction)
        _save_rgb(aligned_source_path, aligned_source)
        _save_mask(source_roundtrip_diff_path, source_expected & ~source_matched)
        _save_rgb(
            diagnostic_path,
            _diagnostic(
                aligned_source,
                target_reconstruction,
                target_observed,
                target_inferred,
            ),
        )
        crop_paths = _write_geographic_crops(
            iteration_dir,
            aligned_source,
            target_reconstruction,
            target_inferred,
            target_domain,
            config.geographic_rows,
            config.geographic_columns,
        )
        class_mask_paths = []
        class_mask_dir = iteration_dir / "class-masks"
        class_mask_dir.mkdir()
        for entry in legend.entries:
            mask_path = class_mask_dir / f"{entry.class_id:02d}-{entry.roman.lower()}.png"
            _save_mask(mask_path, target_ids == entry.class_id)
            class_mask_paths.append(mask_path)
        artifact_paths = (
            source_ids_path,
            source_observed_path,
            source_inferred_path,
            source_occluded_path,
            source_reconstruction_path,
            source_diff_path,
            source_roundtrip_diff_path,
            target_ids_path,
            target_observed_path,
            target_inferred_path,
            target_reconstruction_path,
            aligned_source_path,
            diagnostic_path,
            *class_mask_paths,
            *crop_paths,
        )
        scores = {
            "legend_class_count": len(legend.entries),
            "legend_labels": [entry.label for entry in legend.entries],
            "legend_descriptor_ocr_confidences": [
                entry.descriptor_ocr_confidence for entry in legend.entries
            ],
            "source_domain_pixel_count": domain_count,
            "source_observed_pixel_count": observed_count,
            "source_inferred_pixel_count": inferred_count,
            "source_occluded_nonthematic_pixel_count": inferred_count,
            "source_observed_fraction": observed_fraction,
            "source_inferred_fraction": inferred_fraction,
            "source_occluded_nonthematic_fraction": inferred_fraction,
            "meaningful_source_pixel_count": meaningful_count,
            "meaningful_source_mismatch_pixel_count": mismatch_count,
            "meaningful_source_mismatch_fraction": mismatch_fraction,
            "source_alignment_roundtrip_fraction": source_roundtrip_fraction,
            "legend_class_observed_pixel_counts": class_observed_counts,
            "mapbox_class_pixel_counts": {
                entry.label: int(np.count_nonzero(target_ids == entry.class_id))
                for entry in legend.entries
            },
            "geographic_cells": regional,
            "successive_source_equal": stable,
            "successive_mapbox_equal": stable,
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
                    "authoritative_original_source_pixels",
                    "automatic_six_row_gradient_legend_geometry",
                    "automatic_legend_descriptor_ocr",
                    "legend_gradient_color_manifolds",
                    "accepted_automatic_mapbox_alignment",
                    "pinned_mapbox_land_and_water",
                    "source_reconstruction_diff",
                    "deterministic_two_pass_fixed_point_replay",
                ],
            ),
            method=(
                "automatic VI-XI ordered gradient-manifold classification, dark ink and "
                "water exclusion, nearest thematic completion, Mapbox clipping, source "
                "roundtrip and geographic diff, and two-pass fixed-point replay"
            ),
            artifacts=[
                {"path": str(path), "sha256": _sha256(path)}
                for path in (*complete_artifacts, *legend.artifact_paths)
            ],
        )
        experiment_log.write(experiment_markdown_path, experiment_json_path)
        iterations.append(
            OrderedBandIteration(
                iteration_number, decision, scores, gates, report_path,
                tuple(complete_artifacts)
            )
        )
        all_artifacts.extend(complete_artifacts)
        previous_source_ids = source_complete_ids.copy()
        previous_target_ids = target_ids.copy()
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
                            "path": "legend/ordered-band-legend.json",
                            "sha256": _sha256(legend.artifact_paths[0]),
                        },
                        "accepted_iteration": f"extraction-{iteration_number:02d}",
                        "layers": {
                            entry.roman.lower(): {
                                "kind": "mutually_exclusive_ordered_band",
                                "class_id": entry.class_id,
                                "label": entry.label,
                                "mask": (
                                    f"extraction-{iteration_number:02d}/class-masks/"
                                    f"{entry.class_id:02d}-{entry.roman.lower()}.png"
                                ),
                            }
                            for entry in legend.entries
                        },
                        "observed_mask": (
                            f"extraction-{iteration_number:02d}/mapbox-observed-mask.png"
                        ),
                        "inferred_mask": (
                            f"extraction-{iteration_number:02d}/mapbox-inferred-mask.png"
                        ),
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
        return AutomaticOrderedBandExtractionResult(
            "accepted",
            "VI-XI semantics, source diff, geographic, and fixed-point gates passed",
            legend,
            tuple(iterations),
            accepted_path,
            tuple(all_artifacts),
        )
    blocker = (
        "Ordered-band extraction did not pass every semantic, source-diff, "
        "geographic, and fixed-point gate after two deterministic passes"
    )
    experiment_log.finalize("blocked", blocker)
    experiment_log.write(experiment_markdown_path, experiment_json_path)
    return AutomaticOrderedBandExtractionResult(
        "blocked", blocker, legend, tuple(iterations), None, tuple(all_artifacts)
    )
