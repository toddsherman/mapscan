"""No-human extraction for continuous numeric color-ramp maps.

The adapter is intentionally isolated from MapScan's historical, configured
continuous extractor.  Its only authorities are a pristine source raster, an
accepted automatic alignment, automatically OCRed legend geometry, and the
pinned Mapbox land/water reference.  There are no plan values, control points,
paint, stamps, or inherited extraction pixels.

The first supported source family is the California elevation GIF.  Its land
legend contains a continuous/quantized 0--5000 metre ramp and a separate
``Depression`` swatch.  The swatch is preserved as a categorical semantic: the
source gives no numeric depth, so this module never invents one.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
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
    OCRWord,
    _alignment_contains_forbidden_input,
    _artifact,
    _load_accepted_alignment,
    _parse_tesseract_tsv,
    _reference_to_source_remap,
    _run_tesseract_ocr,
    _run_tesseract_tsv,
    _save_ids,
    _save_mask,
    _save_rgb,
    _source_data_mask,
    _source_to_reference,
    _source_to_reference_remap,
)
from .experiment_log import NoHumanExperimentLog, automatic_provenance
from .source_working_raster import WorkingRasterArtifact, load_working_raster_artifact


SCHEMA_VERSION = "mapscan.automatic-continuous-numeric-extraction.v1"
PRODUCER = "mapscan.automatic_continuous_numeric_extraction"
EXPECTED_SOURCE_TYPE = "continuous_numeric_ramp"


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


def _rgb_to_lab(colors: np.ndarray) -> np.ndarray:
    shaped = np.asarray(colors, dtype=np.float32).reshape((-1, 1, 3)) / 255.0
    return cv2.cvtColor(shaped, cv2.COLOR_RGB2LAB).reshape((-1, 3))


@dataclass(frozen=True)
class ContinuousNumericConfig:
    required_replay_count: int = 2
    minimum_stop_count: int = 7
    minimum_numeric_ocr_confidence: float = 75.0
    minimum_header_ocr_confidence: float = 90.0
    minimum_special_ocr_confidence: float = 90.0
    maximum_ramp_residual: float = 12.0
    maximum_special_residual: float = 8.0
    luminance_weight: float = 0.25
    projection_chunk_pixels: int = 200_000
    minimum_observed_fraction: float = 0.72
    maximum_inferred_fraction: float = 0.28
    maximum_semantic_mismatch_fraction: float = 0.01
    maximum_continuous_roundtrip_error_m: float = 35.0
    minimum_source_roundtrip_fraction: float = 0.94
    geographic_rows: int = 4
    geographic_columns: int = 3
    minimum_geographic_match_fraction: float = 0.90
    minimum_passing_geographic_cells: int = 8

    def __post_init__(self) -> None:
        if self.required_replay_count != 2:
            raise ValueError("continuous extraction requires exactly two replays")
        if self.minimum_stop_count < 3:
            raise ValueError("a continuous ramp requires at least three numeric stops")
        if self.projection_chunk_pixels < 1:
            raise ValueError("projection_chunk_pixels must be positive")
        if not 0.0 < self.luminance_weight <= 1.0:
            raise ValueError("luminance_weight must be in (0, 1]")
        for value in (
            self.minimum_observed_fraction,
            self.maximum_inferred_fraction,
            self.maximum_semantic_mismatch_fraction,
            self.minimum_source_roundtrip_fraction,
            self.minimum_geographic_match_fraction,
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError("continuous extraction fractions must be in [0, 1]")


@dataclass(frozen=True)
class NumericRampStop:
    value_m: float
    center_y: float
    ocr_text: str
    ocr_confidence: float
    ocr_bbox: tuple[int, int, int, int]
    representative_rgb: tuple[int, int, int]


@dataclass(frozen=True)
class ContinuousNumericLegend:
    header_text: str
    header_confidence: float
    units: str
    stops: tuple[NumericRampStop, ...]
    ramp_bbox: tuple[int, int, int, int]
    sample_values_m: np.ndarray
    sample_rgb: np.ndarray
    depression_bbox: tuple[int, int, int, int]
    depression_rgb: tuple[int, int, int]
    depression_ocr_confidence: float
    tesseract_version: str
    artifact_paths: tuple[Path, ...]

    @property
    def minimum_m(self) -> float:
        return float(min(stop.value_m for stop in self.stops))

    @property
    def maximum_m(self) -> float:
        return float(max(stop.value_m for stop in self.stops))


@dataclass(frozen=True)
class ContinuousClassification:
    observed_values_m: np.ndarray
    completed_values_m: np.ndarray
    observed_depression: np.ndarray
    completed_depression: np.ndarray
    observed: np.ndarray
    inferred: np.ndarray
    occluded: np.ndarray
    residual: np.ndarray


@dataclass(frozen=True)
class ContinuousNumericIteration:
    iteration: int
    decision: str
    scores: Mapping[str, Any]
    gates: Mapping[str, Any]
    report_path: Path
    artifact_paths: tuple[Path, ...]


@dataclass(frozen=True)
class AutomaticContinuousNumericExtractionResult:
    status: str
    stop_reason: str
    legend: ContinuousNumericLegend
    iterations: tuple[ContinuousNumericIteration, ...]
    accepted_extraction_path: Path | None
    artifact_paths: tuple[Path, ...]

    @property
    def accepted(self) -> Path | None:
        return self.accepted_extraction_path


def _normalized_integer(text: str) -> int | None:
    normalized = text.strip().replace(",", "").replace("−", "-").replace("—", "-")
    if not re.fullmatch(r"-?\d{1,6}", normalized):
        return None
    return int(normalized)


def _best_duplicate_words(words: Sequence[OCRWord]) -> list[OCRWord]:
    """Collapse PSM duplicates by their nearly identical raster boxes."""

    ordered = sorted(words, key=lambda word: word.confidence, reverse=True)
    retained: list[OCRWord] = []
    for word in ordered:
        if any(
            abs(word.left - other.left) <= 3
            and abs(word.top - other.top) <= 3
            and abs(word.width - other.width) <= 4
            and abs(word.height - other.height) <= 4
            for other in retained
        ):
            continue
        retained.append(word)
    return sorted(retained, key=lambda word: (word.top, word.left))


def _header_words(words: Sequence[OCRWord]) -> tuple[list[OCRWord], float]:
    required = ("ELEVATION", "ABOVE", "SEA", "LEVEL")
    matches: list[OCRWord] = []
    for token in required:
        options = [
            word
            for word in words
            if re.sub(r"[^A-Z]", "", word.text.upper()) == token
        ]
        if not options:
            raise ValueError("Numeric ramp legend header was not established by OCR")
        matches.append(max(options, key=lambda word: word.confidence))
    tops = np.asarray([word.top for word in matches], dtype=float)
    if float(np.ptp(tops)) > max(8.0, np.median([word.height for word in matches])):
        raise ValueError("Numeric ramp legend header tokens do not share one line")
    return matches, float(min(word.confidence for word in matches))


def _numeric_column(words: Sequence[OCRWord], header: Sequence[OCRWord]) -> tuple[OCRWord, ...]:
    """Select the longest regular, strictly descending nonnegative stop series."""

    header_top = min(word.top for word in header)
    header_right = max(word.right for word in header)
    candidates = [
        word
        for word in words
        if _normalized_integer(word.text) is not None
        and _normalized_integer(word.text) >= 0
        and word.top > header_top + 12
        and word.left >= header_right - 70
    ]
    candidates = _best_duplicate_words(candidates)
    if len(candidates) < 3:
        raise ValueError("Numeric ramp legend has too few OCR number candidates")

    # Enumerating columns is cheap (legends contain tens, not millions, of OCR
    # boxes) and avoids silently borrowing the neighbouring feet column.
    scored: list[tuple[tuple[float, ...], tuple[OCRWord, ...]]] = []
    for axis_word in candidates:
        column = sorted(
            [word for word in candidates if abs(word.left - axis_word.left) <= 6],
            key=lambda word: word.center_y,
        )
        for length in range(len(column), 2, -1):
            for start in range(0, len(column) - length + 1):
                sequence = tuple(column[start : start + length])
                values = [_normalized_integer(word.text) for word in sequence]
                assert all(value is not None for value in values)
                if not all(a > b for a, b in zip(values, values[1:])):
                    continue
                steps = np.diff([word.center_y for word in sequence])
                if np.any(steps < 8.0) or np.any(steps > 40.0):
                    continue
                step_cv = float(np.std(steps) / max(np.mean(steps), 1.0))
                if step_cv > 0.22:
                    continue
                confidence = float(np.mean([word.confidence for word in sequence]))
                ends_at_zero = float(values[-1] == 0)
                scored.append(
                    (
                        (
                            float(length),
                            ends_at_zero,
                            -step_cv,
                            confidence / 100.0,
                        ),
                        sequence,
                    )
                )
    if not scored:
        raise ValueError("No regular descending OCR numeric ramp column was found")
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1]


def _dark_column_runs(
    source_rgb: np.ndarray,
    left: int,
    right: int,
    top: int,
    bottom: int,
) -> list[tuple[int, int, float]]:
    dark = np.max(source_rgb[top:bottom, left:right], axis=2) <= 55
    coverage = dark.mean(axis=0)
    mask = coverage >= 0.72
    changes = np.diff(np.pad(mask.astype(np.int8), (1, 1)))
    runs = list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))
    return [
        (left + int(start), left + int(end), float(np.max(coverage[start:end])))
        for start, end in runs
    ]


def _ramp_bounds(
    source_rgb: np.ndarray, sequence: Sequence[OCRWord]
) -> tuple[int, int, int, int]:
    numeric_left = int(round(np.median([word.left for word in sequence])))
    top = max(0, int(math.floor(sequence[0].center_y - 4)))
    bottom = min(source_rgb.shape[0], int(math.ceil(sequence[-1].center_y + 3)))
    search_left = max(0, numeric_left - round(source_rgb.shape[1] * 0.15))
    search_right = max(search_left + 1, numeric_left - 4)
    runs = _dark_column_runs(source_rgb, search_left, search_right, top, bottom)
    pairs = []
    for first, second in itertools.combinations(runs, 2):
        left = first[1] - 1
        right = second[0]
        width = right - left
        if source_rgb.shape[1] * 0.02 <= width <= source_rgb.shape[1] * 0.08:
            pairs.append((first[2] + second[2], left, right))
    if not pairs:
        raise ValueError("OCR numeric stops have no enclosed vertical ramp borders")
    _, left, right = max(pairs)
    return left, top, right - left + 1, bottom - top


def _interpolate_stop_value(y: float, stops: Sequence[NumericRampStop]) -> float:
    ys = np.asarray([stop.center_y for stop in stops], dtype=np.float32)
    values = np.asarray([stop.value_m for stop in stops], dtype=np.float32)
    return float(np.interp(y, ys, values))


def _sample_ramp(
    source_rgb: np.ndarray,
    ramp_bbox: tuple[int, int, int, int],
    stops: Sequence[NumericRampStop],
) -> tuple[np.ndarray, np.ndarray]:
    left, top, width, height = ramp_bbox
    # Stay well inside the vertical outline.  The scan has a thick black box
    # and horizontal tick strokes that enter the ramp; sampling close to either
    # side makes the median sensitive to those non-data pixels.
    inset = max(5, width // 5)
    x0, x1 = left + inset, left + width - inset
    sample_values: list[float] = []
    sample_rgb: list[np.ndarray] = []
    for y in range(top + 2, top + height - 2):
        # Every OCR stop is crossed by a horizontal black tick.  Those rows are
        # legend structure, not ramp color.  Recover stop representatives from
        # adjacent clean rows rather than allowing a single tick row to create
        # a dark-brown branch in the continuous color manifold.
        if min(abs(y - stop.center_y) for stop in stops) <= 4.0:
            continue
        row = source_rgb[y, x0:x1]
        luminance = cv2.cvtColor(row[None, :, :], cv2.COLOR_RGB2GRAY)[0]
        upper_quartile = float(np.quantile(luminance, 0.75))
        # Relative rejection preserves legitimately dark green at the bottom
        # of the ramp while removing darker scan/tick contamination within a
        # row.  The cross-column median then remains robust to isolated texture.
        retained = row[
            (luminance >= max(35.0, upper_quartile - 28.0))
            & (np.max(row, axis=1) > 55)
        ]
        if retained.shape[0] < max(6, row.shape[0] // 2):
            continue
        sample_values.append(_interpolate_stop_value(y, stops))
        sample_rgb.append(np.median(retained, axis=0))
    if len(sample_values) < 20:
        raise ValueError("Numeric ramp contains too few clean color samples")
    values = np.asarray(sample_values, dtype=np.float32)
    colors = np.asarray(sample_rgb, dtype=np.float32).round().clip(0, 255).astype(np.uint8)
    order = np.argsort(values)
    return values[order], colors[order]


def _depression_swatch(
    source_rgb: np.ndarray,
    ramp_bbox: tuple[int, int, int, int],
) -> tuple[tuple[int, int, int, int], tuple[int, int, int], float]:
    left, top, width, height = ramp_bbox
    ramp_bottom = top + height
    search_top = min(source_rgb.shape[0] - 1, ramp_bottom + 2)
    search_bottom = min(source_rgb.shape[0], ramp_bottom + max(35, height // 3))
    strip = source_rgb[search_top:search_bottom, left : left + width]
    dark_fraction = (np.max(strip, axis=2) <= 55).mean(axis=1)
    border_rows = np.flatnonzero(dark_fraction >= 0.70)
    if border_rows.size < 2:
        raise ValueError("Depression legend swatch borders were not detected")
    runs = []
    changes = np.diff(np.pad((dark_fraction >= 0.70).astype(np.int8), (1, 1)))
    for start, end in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)):
        runs.append((int(start), int(end)))
    if len(runs) < 2:
        raise ValueError("Depression legend swatch has no enclosing horizontal borders")
    selected = None
    for first, second in zip(runs, runs[1:]):
        gap = second[0] - first[1]
        if 8 <= gap <= 35:
            selected = (first, second)
            break
    if selected is None:
        raise ValueError("Depression swatch border spacing is not plausible")
    first, second = selected
    swatch_top = search_top + first[0]
    swatch_bottom = search_top + second[1]
    bbox = (left, swatch_top, width, swatch_bottom - swatch_top)
    pixels = source_rgb[swatch_top + 3 : swatch_bottom - 3, left + 3 : left + width - 3]
    chromatic = pixels[(pixels.max(axis=2) - pixels.min(axis=2) >= 20) & (pixels.max(axis=2) > 40)]
    if chromatic.shape[0] < 20:
        raise ValueError("Depression swatch has too little chromatic evidence")
    colors, counts = np.unique(chromatic.reshape(-1, 3), axis=0, return_counts=True)
    representative = tuple(int(value) for value in colors[int(np.argmax(counts))])

    crop_left = max(0, left + width - 2)
    crop_right = min(source_rgb.shape[1], crop_left + max(100, width * 3))
    crop_top = max(0, swatch_top - 5)
    crop_bottom = min(source_rgb.shape[0], swatch_bottom + 5)
    crop = source_rgb[crop_top:crop_bottom, crop_left:crop_right]
    enlarged = cv2.resize(crop, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    candidates = []
    for psm in (6, 7, 11, 13):
        candidates.extend(_parse_tesseract_tsv(_run_tesseract_tsv(enlarged, psm=psm)))
    matches = [
        word
        for word in candidates
        if re.sub(r"[^A-Z]", "", word.text.upper()) == "DEPRESSION"
    ]
    if not matches:
        raise ValueError("Depression legend semantic was not established by OCR")
    confidence = float(max(word.confidence for word in matches))
    return bbox, representative, confidence


def detect_continuous_numeric_legend(
    source_path: Path,
    source_rgb: np.ndarray,
    output_dir: Path,
    *,
    config: ContinuousNumericConfig = ContinuousNumericConfig(),
) -> ContinuousNumericLegend:
    """Recover numeric stops, ramp colors, units, and the depression semantic."""

    legend_dir = output_dir / "legend"
    legend_dir.mkdir(parents=True, exist_ok=True)
    words, combined_tsv, version = _run_tesseract_ocr(source_path)
    header, header_confidence = _header_words(words)
    if header_confidence < config.minimum_header_ocr_confidence:
        raise ValueError("Numeric ramp legend header OCR is below the strict gate")
    sequence = _numeric_column(words, header)
    if len(sequence) < config.minimum_stop_count:
        raise ValueError("Numeric ramp has fewer OCR stops than the strict gate")
    if any(word.confidence < config.minimum_numeric_ocr_confidence for word in sequence):
        raise ValueError("Numeric ramp stop OCR is below the strict gate")
    values = [_normalized_integer(word.text) for word in sequence]
    if values[-1] != 0 or values[0] <= values[-1]:
        raise ValueError("Above-sea-level ramp must descend to a numeric zero stop")

    ramp_bbox = _ramp_bounds(source_rgb, sequence)
    provisional = []
    for value, word in zip(values, sequence):
        assert value is not None
        provisional.append(
            NumericRampStop(
                float(value),
                word.center_y,
                word.text,
                word.confidence,
                (word.left, word.top, word.width, word.height),
                (0, 0, 0),
            )
        )
    sample_values, sample_rgb = _sample_ramp(source_rgb, ramp_bbox, provisional)
    stops = []
    for stop in provisional:
        index = int(np.argmin(np.abs(sample_values - stop.value_m)))
        stops.append(
            NumericRampStop(
                stop.value_m,
                stop.center_y,
                stop.ocr_text,
                stop.ocr_confidence,
                stop.ocr_bbox,
                tuple(int(value) for value in sample_rgb[index]),
            )
        )
    depression_bbox, depression_rgb, depression_confidence = _depression_swatch(
        source_rgb, ramp_bbox
    )
    if depression_confidence < config.minimum_special_ocr_confidence:
        raise ValueError("Depression OCR is below the strict gate")

    tsv_path = legend_dir / "tesseract-combined.tsv"
    tsv_path.write_text(combined_tsv)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "numeric_semantics_established",
        "source": {"path": str(source_path), "sha256": _sha256(source_path)},
        "header": {
            "text": "Elevation above sea level",
            "ocr_confidence": header_confidence,
            "units": "meters",
        },
        "ramp_bbox": list(ramp_bbox),
        "numeric_stops": [
            {
                "value_m": stop.value_m,
                "center_y": stop.center_y,
                "ocr_text": stop.ocr_text,
                "ocr_confidence": stop.ocr_confidence,
                "ocr_bbox": list(stop.ocr_bbox),
                "representative_rgb": list(stop.representative_rgb),
            }
            for stop in stops
        ],
        "dense_ramp_samples": [
            {"value_m": float(value), "rgb": [int(channel) for channel in rgb]}
            for value, rgb in zip(sample_values, sample_rgb)
        ],
        "special_semantics": [
            {
                "label": "Depression",
                "numeric_value": None,
                "reason": "legend supplies a category but no depth",
                "bbox": list(depression_bbox),
                "representative_rgb": list(depression_rgb),
                "ocr_confidence": depression_confidence,
            }
        ],
        "tesseract_version": version,
    }
    legend_path = legend_dir / "continuous-numeric-legend.json"
    legend_path.write_text(json.dumps(payload, indent=2) + "\n")

    preview = Image.fromarray(source_rgb.copy())
    draw = ImageDraw.Draw(preview)
    left, top, width, height = ramp_bbox
    draw.rectangle((left, top, left + width, top + height), outline=(0, 255, 255), width=3)
    for stop in stops:
        draw.line((left - 5, stop.center_y, left + width + 5, stop.center_y), fill=(255, 0, 255), width=2)
        draw.text((left - 65, stop.center_y - 6), f"{stop.value_m:g} m", fill=(255, 0, 255))
    dl, dt, dw, dh = depression_bbox
    draw.rectangle((dl, dt, dl + dw, dt + dh), outline=(0, 255, 255), width=3)
    preview_path = legend_dir / "continuous-numeric-legend-detection.png"
    preview.save(preview_path, optimize=True)
    return ContinuousNumericLegend(
        "Elevation above sea level",
        header_confidence,
        "meters",
        tuple(stops),
        ramp_bbox,
        sample_values,
        sample_rgb,
        depression_bbox,
        depression_rgb,
        depression_confidence,
        version,
        (legend_path, preview_path, tsv_path),
    )


def _project_pixels_to_ramp(
    rgb: np.ndarray,
    legend: ContinuousNumericLegend,
    *,
    luminance_weight: float,
    chunk_pixels: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Project pixels onto the OCR-valued dense Lab curve with bounded memory.

    The implementation holds only one pixel chunk and one curve segment at a
    time.  Memory is therefore O(chunk + image), never O(image * ramp samples),
    which matters for high-resolution scans.
    """

    flat = np.asarray(rgb, dtype=np.uint8).reshape((-1, 3))
    sample_lab = _rgb_to_lab(legend.sample_rgb)
    weights = np.asarray([luminance_weight, 1.0, 1.0], dtype=np.float32)
    curve = sample_lab * weights
    starts = curve[:-1]
    deltas = curve[1:] - curve[:-1]
    denominators = np.maximum(np.sum(deltas * deltas, axis=1), 1e-9)
    value_starts = legend.sample_values_m[:-1]
    value_deltas = np.diff(legend.sample_values_m)

    values = np.empty(flat.shape[0], dtype=np.float32)
    residual = np.empty(flat.shape[0], dtype=np.float32)
    for chunk_start in range(0, flat.shape[0], chunk_pixels):
        chunk_end = min(flat.shape[0], chunk_start + chunk_pixels)
        pixels = _rgb_to_lab(flat[chunk_start:chunk_end]) * weights
        best_distance = np.full(pixels.shape[0], np.inf, dtype=np.float32)
        best_value = np.zeros(pixels.shape[0], dtype=np.float32)
        for segment in range(starts.shape[0]):
            delta = deltas[segment]
            fraction = np.clip(
                ((pixels - starts[segment]) @ delta) / denominators[segment],
                0.0,
                1.0,
            )
            difference = pixels - (starts[segment] + fraction[:, None] * delta)
            distance = np.sum(difference * difference, axis=1)
            selected = distance < best_distance
            best_distance[selected] = distance[selected]
            best_value[selected] = (
                value_starts[segment] + fraction[selected] * value_deltas[segment]
            )
        values[chunk_start:chunk_end] = best_value
        residual[chunk_start:chunk_end] = np.sqrt(best_distance)
    return values.reshape(rgb.shape[:2]), residual.reshape(rgb.shape[:2])


def _depression_distance(
    rgb: np.ndarray,
    representative_rgb: tuple[int, int, int],
    luminance_weight: float,
    chunk_pixels: int,
) -> np.ndarray:
    flat = np.asarray(rgb, dtype=np.uint8).reshape((-1, 3))
    target = _rgb_to_lab(np.asarray([representative_rgb]))[0]
    weights = np.asarray([luminance_weight, 1.0, 1.0], dtype=np.float32)
    target *= weights
    output = np.empty(flat.shape[0], dtype=np.float32)
    for start in range(0, flat.shape[0], chunk_pixels):
        end = min(flat.shape[0], start + chunk_pixels)
        lab = _rgb_to_lab(flat[start:end]) * weights
        output[start:end] = np.linalg.norm(lab - target, axis=1)
    return output.reshape(rgb.shape[:2])


def _ocr_occlusion_mask(
    shape: tuple[int, int], words: Sequence[OCRWord], domain: np.ndarray
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    for word in words:
        if word.confidence < 20.0:
            continue
        left = max(0, word.left - 2)
        top = max(0, word.top - 2)
        right = min(shape[1], word.right + 2)
        bottom = min(shape[0], word.top + word.height + 2)
        if right <= left or bottom <= top:
            continue
        # Page text and legend labels outside the accepted Mapbox land domain
        # are irrelevant; avoid expanding their boxes into the state.
        if np.any(domain[top:bottom, left:right]):
            mask[top:bottom, left:right] = 1
    return (mask > 0) & domain


def _cartographic_occlusions(
    source_rgb: np.ndarray, domain: np.ndarray, ocr_mask: np.ndarray
) -> tuple[np.ndarray, Mapping[str, int]]:
    channels = source_rgb.astype(np.int16)
    blue = (
        domain
        & (channels[..., 2] - channels[..., 0] >= 15)
        & (channels[..., 2] - channels[..., 1] >= 5)
    )
    chroma = channels.max(axis=2) - channels.min(axis=2)
    gray = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2GRAY)
    neutral_dark = domain & (gray <= 75) & (chroma <= 22)
    # Thin labels/lines may be colored.  A local black-hat captures only ink
    # darker than its neighbourhood rather than excluding broad shaded relief.
    blackhat = cv2.morphologyEx(
        gray,
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    )
    local_ink = domain & (blackhat >= 45) & (gray <= 125)
    base = domain & (blue | neutral_dark | local_ink | ocr_mask)
    # Text, dotted geomorphic boundaries, graticules, and city symbols have
    # anti-aliased fringes just outside the darkest core/OCR rectangle.  Expand
    # by only one pixel and require independent local-darkness evidence, so the
    # source comparison cannot turn ordinary terrain into occlusion merely
    # because it reconstructed poorly.
    broad_blackhat = cv2.morphologyEx(
        gray,
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
    )
    fringe_support = (
        ((np.maximum(blackhat, broad_blackhat) >= 20) & (gray <= 190))
        | ((gray <= 100) & (chroma <= 50))
    )
    adjacent = cv2.dilate(
        base.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    ) > 0
    antialias_fringe = domain & ~base & adjacent & fringe_support
    combined = base | antialias_fringe
    return combined, {
        "blue_hydrography_pixel_count": int(np.count_nonzero(blue)),
        "neutral_dark_ink_pixel_count": int(np.count_nonzero(neutral_dark)),
        "local_dark_ink_pixel_count": int(np.count_nonzero(local_ink)),
        "ocr_label_pixel_count": int(np.count_nonzero(ocr_mask)),
        "source_evidenced_antialias_fringe_pixel_count": int(
            np.count_nonzero(antialias_fringe)
        ),
        "combined_occluded_pixel_count": int(np.count_nonzero(combined)),
    }


def _nearest_continuous_completion(
    observed_values_m: np.ndarray,
    observed_depression: np.ndarray,
    domain: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    observed_numeric = np.isfinite(observed_values_m) & domain
    observed_any = observed_numeric | (observed_depression & domain)
    if not np.any(observed_any):
        raise ValueError("Continuous source domain contains no direct legend evidence")
    # DIST_LABEL_PIXEL gives each zero (observed) pixel a stable label.  This
    # avoids scipy's two int64 coordinate planes and bounds high-resolution
    # memory to two 32-bit image planes plus the outputs.
    _distance, labels = cv2.distanceTransformWithLabels(
        (~observed_any).astype(np.uint8),
        cv2.DIST_L2,
        cv2.DIST_MASK_5,
        labelType=cv2.DIST_LABEL_PIXEL,
    )
    maximum_label = int(labels.max(initial=0))
    value_by_label = np.full(maximum_label + 1, np.nan, dtype=np.float32)
    depression_by_label = np.zeros(maximum_label + 1, dtype=bool)
    value_by_label[labels[observed_numeric]] = observed_values_m[observed_numeric]
    depression_by_label[labels[observed_depression & domain]] = True
    completed_values = value_by_label[labels]
    completed_depression = depression_by_label[labels]
    completed_values[completed_depression | ~domain] = np.nan
    completed_depression &= domain
    if np.any(domain & ~completed_depression & ~np.isfinite(completed_values)):
        raise ValueError("Nearest continuous completion left an unclassified land pixel")
    return completed_values, completed_depression


def classify_continuous_numeric(
    source_rgb: np.ndarray,
    domain: np.ndarray,
    legend: ContinuousNumericLegend,
    words: Sequence[OCRWord],
    *,
    config: ContinuousNumericConfig = ContinuousNumericConfig(),
) -> tuple[ContinuousClassification, Mapping[str, int]]:
    """Classify direct ramp/special evidence and complete occlusions spatially."""

    if source_rgb.shape[:2] != domain.shape:
        raise ValueError("Continuous source image and domain differ in shape")
    projected_values, residual = _project_pixels_to_ramp(
        source_rgb,
        legend,
        luminance_weight=config.luminance_weight,
        chunk_pixels=config.projection_chunk_pixels,
    )
    depression_residual = _depression_distance(
        source_rgb,
        legend.depression_rgb,
        config.luminance_weight,
        config.projection_chunk_pixels,
    )
    ocr_mask = _ocr_occlusion_mask(domain.shape, words, domain)
    occluded, occlusion_scores = _cartographic_occlusions(source_rgb, domain, ocr_mask)
    depression_candidate = (
        domain
        & (depression_residual <= config.maximum_special_residual)
        & (depression_residual + 1.0 < residual)
    )
    protected_depression_count = int(np.count_nonzero(occluded & depression_candidate))
    occluded &= ~depression_candidate
    occlusion_scores = {
        **occlusion_scores,
        "legend_semantic_depression_protected_pixel_count": protected_depression_count,
        "combined_occluded_pixel_count": int(np.count_nonzero(occluded)),
    }
    observed_depression = depression_candidate
    observed_numeric = (
        domain
        & ~occluded
        & ~observed_depression
        & (residual <= config.maximum_ramp_residual)
    )
    observed_values = np.full(domain.shape, np.nan, dtype=np.float32)
    observed_values[observed_numeric] = np.clip(
        projected_values[observed_numeric], legend.minimum_m, legend.maximum_m
    )
    completed_values, completed_depression = _nearest_continuous_completion(
        observed_values, observed_depression, domain
    )
    observed = observed_numeric | observed_depression
    # Inferred and occluded are deliberately disjoint provenance channels.
    # Both receive nearest-source completion, but known ink/water is not
    # silently reported as ordinary model inference.
    inferred = domain & ~observed & ~occluded
    occluded = domain & ~observed & occluded
    return (
        ContinuousClassification(
            observed_values,
            completed_values,
            observed_depression,
            completed_depression,
            observed,
            inferred,
            occluded,
            residual,
        ),
        occlusion_scores,
    )


def _values_to_ramp_rgb(
    values_m: np.ndarray,
    depression: np.ndarray,
    legend: ContinuousNumericLegend,
    *,
    luminance_weight: float = 0.25,
) -> np.ndarray:
    # Build a compact inverse of the *actual* Lab projector rather than using
    # the OCR/Y value associated with the nearest raw sample.  Scan texture and
    # repeated colors (notably the white 4000--5000 m plateau) make the raw
    # sample curve non-injective.  Densely interpolating its RGB segments and
    # then projecting those candidates through the same forward model yields a
    # deterministic palette whose keys are reachable semantic values.
    source_colors = legend.sample_rgb.astype(np.float32)
    inverse_colors = []
    subdivisions = 16
    for index in range(len(source_colors) - 1):
        start = source_colors[index]
        delta = source_colors[index + 1] - start
        for fraction in np.linspace(0.0, 1.0, subdivisions, endpoint=False):
            inverse_colors.append(
                np.rint(start + delta * fraction).clip(0, 255).astype(np.uint8)
            )
    inverse_colors.append(source_colors[-1].round().clip(0, 255).astype(np.uint8))
    inverse_rgb = np.unique(np.asarray(inverse_colors, dtype=np.uint8), axis=0)
    inverse_values, _inverse_residual = _project_pixels_to_ramp(
        inverse_rgb.reshape((1, -1, 3)),
        legend,
        luminance_weight=luminance_weight,
        chunk_pixels=max(len(inverse_rgb), 1),
    )
    inverse_values = inverse_values.ravel()
    order = np.argsort(inverse_values, kind="stable")
    inverse_values = inverse_values[order]
    inverse_rgb = inverse_rgb[order]

    output = np.zeros((*values_m.shape, 3), dtype=np.uint8)
    valid = np.isfinite(values_m) & ~depression
    if np.any(valid):
        values = values_m[valid]
        indices = np.searchsorted(inverse_values, values, side="left")
        indices = np.clip(indices, 0, len(inverse_values) - 1)
        previous = np.clip(indices - 1, 0, len(inverse_values) - 1)
        choose_previous = np.abs(values - inverse_values[previous]) < np.abs(
            values - inverse_values[indices]
        )
        indices[choose_previous] = previous[choose_previous]
        output[valid] = inverse_rgb[indices]
    output[depression] = legend.depression_rgb
    return output


def _quantized_band_ids(
    values_m: np.ndarray, depression: np.ndarray, legend: ContinuousNumericLegend
) -> tuple[np.ndarray, tuple[tuple[int, float, float], ...]]:
    ascending = sorted({float(stop.value_m) for stop in legend.stops})
    intervals = tuple(
        (index + 1, ascending[index], ascending[index + 1])
        for index in range(len(ascending) - 1)
    )
    ids = np.zeros(values_m.shape, dtype=np.uint8)
    valid = np.isfinite(values_m) & ~depression
    for class_id, minimum, maximum in intervals:
        selected = valid & (values_m >= minimum) & (
            values_m < maximum if maximum < ascending[-1] else values_m <= maximum
        )
        ids[selected] = class_id
    return ids, intervals


def _encode_meters(values_m: np.ndarray, depression: np.ndarray) -> np.ndarray:
    encoded = np.zeros(values_m.shape, dtype=np.uint16)
    valid = np.isfinite(values_m) & ~depression
    # 0 is nodata/depression.  A one-based metre encoding preserves numeric 0.
    encoded[valid] = np.clip(np.rint(values_m[valid]) + 1, 1, 65535).astype(np.uint16)
    return encoded


def _save_u16(path: Path, values: np.ndarray) -> None:
    Image.fromarray(values.astype(np.uint16), mode="I;16").save(path, optimize=True)


def _source_only_audit_domain(
    source_rgb: np.ndarray, legend: ContinuousNumericLegend
) -> np.ndarray:
    """Conservative source-only panel for pre-alignment legend/reconstruction audit.

    This is not a publication domain.  It excludes the right-side legend/UI and
    obvious Pacific blue, but intentionally does not claim that white pixels are
    California land.  The accepted Mapbox state mask remains mandatory for the
    official extractor.
    """

    left, ramp_top, _width, ramp_height = legend.ramp_bbox
    domain = np.zeros(source_rgb.shape[:2], dtype=bool)
    channels = source_rgb.astype(np.int16)
    blue = (
        (channels[..., 2] - channels[..., 0] >= 15)
        & (channels[..., 2] - channels[..., 1] >= 5)
    )
    # The source's explanatory UI starts as a long near-white run on the title
    # row.  Its irregular lower edge can touch white Sierra pixels, so taking a
    # component hull would erase valid map data.  The source-only audit instead
    # keeps the conservative map panel left of that automatically detected run.
    # Publication still uses the accepted Mapbox state domain, not this crop.
    chroma = channels.max(axis=2) - channels.min(axis=2)
    pale = (channels.min(axis=2) >= 220) & (chroma <= 22)
    probe_y = max(2, ramp_top - round(ramp_height * 0.85))
    probe = cv2.morphologyEx(
        pale[max(0, probe_y - 2) : probe_y + 3].any(axis=0).astype(np.uint8)[None, :],
        cv2.MORPH_CLOSE,
        np.ones((1, 21), dtype=np.uint8),
    )[0] > 0
    changes = np.diff(np.pad(probe.astype(np.int8), (1, 1)))
    runs = list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))
    right_runs = [
        (int(start), int(end))
        for start, end in runs
        if end >= source_rgb.shape[1] - 3 and end - start >= source_rgb.shape[1] * 0.20
    ]
    layout_left = min((start for start, _end in right_runs), default=left - 12)
    domain[:, : max(1, min(left - 12, layout_left))] = True
    return domain & ~blue


def run_continuous_numeric_source_audit(
    source_path: Path,
    output_dir: Path,
    *,
    config: ContinuousNumericConfig = ContinuousNumericConfig(),
) -> Mapping[str, Any]:
    """Write a no-log, source-only readiness audit before alignment is accepted."""

    source_path = source_path.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise ValueError("Continuous source audit requires a fresh output directory")
    output_dir.mkdir(parents=True)
    source_rgb = np.asarray(Image.open(source_path).convert("RGB"))
    legend = detect_continuous_numeric_legend(source_path, source_rgb, output_dir, config=config)
    domain = _source_only_audit_domain(source_rgb, legend)
    values, residual = _project_pixels_to_ramp(
        source_rgb,
        legend,
        luminance_weight=config.luminance_weight,
        chunk_pixels=config.projection_chunk_pixels,
    )
    depression_residual = _depression_distance(
        source_rgb,
        legend.depression_rgb,
        config.luminance_weight,
        config.projection_chunk_pixels,
    )
    chroma = source_rgb.max(axis=2).astype(np.int16) - source_rgb.min(axis=2).astype(np.int16)
    direct_depression = domain & (depression_residual <= config.maximum_special_residual) & (
        depression_residual + 1.0 < residual
    )
    # White/neutral source pixels are deliberately omitted from this pre-
    # alignment audit because white high terrain is visually identical to page
    # background.  The official Mapbox land domain resolves that ambiguity.
    direct_numeric = (
        domain
        & ~direct_depression
        & (residual <= config.maximum_ramp_residual)
        & ((chroma >= 12) | (source_rgb.min(axis=2) < 245))
    )
    audited_values = np.full(domain.shape, np.nan, dtype=np.float32)
    audited_values[direct_numeric] = values[direct_numeric]
    reconstruction = _values_to_ramp_rgb(
        audited_values,
        direct_depression,
        legend,
        luminance_weight=config.luminance_weight,
    )
    observed = direct_numeric | direct_depression
    residual_preview = np.clip(residual / max(config.maximum_ramp_residual, 1e-6) * 255, 0, 255).astype(np.uint8)

    domain_path = output_dir / "source-only-nonauthoritative-panel-mask.png"
    observed_path = output_dir / "source-only-direct-evidence-mask.png"
    depression_path = output_dir / "source-only-depression-evidence-mask.png"
    reconstruction_path = output_dir / "source-only-reconstruction.png"
    residual_path = output_dir / "source-only-ramp-residual.png"
    comparison_path = output_dir / "source-only-comparison.png"
    _save_mask(domain_path, domain)
    _save_mask(observed_path, observed)
    _save_mask(depression_path, direct_depression)
    _save_rgb(reconstruction_path, reconstruction)
    Image.fromarray(residual_preview, mode="L").save(residual_path, optimize=True)
    overlay = cv2.addWeighted(source_rgb, 0.5, reconstruction, 0.5, 0.0)
    overlay[~observed] = source_rgb[~observed]
    _save_rgb(comparison_path, np.concatenate((source_rgb, reconstruction, overlay), axis=1))
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "source_only_ready_alignment_required",
        "source": {"path": str(source_path), "sha256": _sha256(source_path)},
        "numeric_stop_values_m": [stop.value_m for stop in legend.stops],
        "numeric_stop_ocr_confidences": [stop.ocr_confidence for stop in legend.stops],
        "depression_ocr_confidence": legend.depression_ocr_confidence,
        "dense_ramp_sample_count": int(len(legend.sample_values_m)),
        "source_only_direct_evidence_pixel_count": int(np.count_nonzero(observed)),
        "source_only_numeric_pixel_count": int(np.count_nonzero(direct_numeric)),
        "source_only_depression_pixel_count": int(np.count_nonzero(direct_depression)),
        "source_only_panel_pixel_count": int(np.count_nonzero(domain)),
        "known_source_ambiguities": [
            "white high-elevation pixels are identical to white page/layout pixels until the accepted Mapbox land mask supplies the domain",
            "Depression is a readable category but has no numeric depth in the source legend; it is preserved as a separate mask",
            "blue hydrography is cartographic occlusion, not a numeric elevation",
        ],
        "official_extraction_attempt_created": False,
        "artifacts": [
            _artifact(path, output_dir)
            for path in (
                *legend.artifact_paths,
                domain_path,
                observed_path,
                depression_path,
                reconstruction_path,
                residual_path,
                comparison_path,
            )
        ],
    }
    report_path = output_dir / "source-audit.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


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


def _decode_meters(encoded: np.ndarray, depression: np.ndarray) -> np.ndarray:
    values = np.full(encoded.shape, np.nan, dtype=np.float32)
    valid = (encoded > 0) & ~depression
    values[valid] = encoded[valid].astype(np.float32) - 1.0
    return values


def _geographic_roundtrip_metrics(
    expected_domain: np.ndarray,
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
            expected = cell & expected_domain
            count = int(np.count_nonzero(expected))
            if not count:
                continue
            match_count = int(np.count_nonzero(expected & matched))
            reports.append(
                {
                    "id": f"r{row + 1}-c{column + 1}",
                    "source_expected_pixel_count": count,
                    "source_match_pixel_count": match_count,
                    "source_match_fraction": match_count / count,
                }
            )
    return reports


def _write_geographic_crops(
    root: Path,
    aligned_source: np.ndarray,
    reconstruction: np.ndarray,
    inferred: np.ndarray,
    occluded: np.ndarray,
    target_domain: np.ndarray,
    rows: int,
    columns: int,
) -> list[Path]:
    root.mkdir()
    height, width = target_domain.shape
    paths = []
    for row in range(rows):
        top, bottom = round(row * height / rows), round((row + 1) * height / rows)
        for column in range(columns):
            left, right = round(column * width / columns), round((column + 1) * width / columns)
            if not np.any(target_domain[top:bottom, left:right]):
                continue
            source_crop = aligned_source[top:bottom, left:right]
            reconstruction_crop = reconstruction[top:bottom, left:right]
            overlay = cv2.addWeighted(source_crop, 0.5, reconstruction_crop, 0.5, 0.0)
            overlay[inferred[top:bottom, left:right]] = (0, 255, 255)
            overlay[occluded[top:bottom, left:right]] = (255, 0, 255)
            path = root / f"r{row + 1}-c{column + 1}-source-extraction.png"
            _save_rgb(path, np.concatenate((source_crop, reconstruction_crop, overlay), axis=1))
            paths.append(path)
    return paths


def _diagnostic(
    aligned_source: np.ndarray,
    reconstruction: np.ndarray,
    inferred: np.ndarray,
    occluded: np.ndarray,
    maximum_height: int = 1500,
) -> np.ndarray:
    overlay = cv2.addWeighted(aligned_source, 0.5, reconstruction, 0.5, 0.0)
    overlay[inferred] = (0, 255, 255)
    overlay[occluded] = (255, 0, 255)
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
    panel_width = panels[0].shape[1]
    for index, label in enumerate(
        ("aligned original", "continuous elevation reconstruction", "50/50; cyan inferred; magenta occluded")
    ):
        draw.rectangle((index * panel_width, 0, (index + 1) * panel_width, 30), fill=(0, 0, 0))
        draw.text((index * panel_width + 8, 7), label, fill=(255, 255, 255))
    return np.asarray(canvas)


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
        raise ValueError("continuous numeric source-clean authority is not pristine")
    if experiment_log.data["source"].get("sha256") != working.source_sha256:
        raise ValueError("experiment and source-clean original hashes disagree")
    if experiment_log.data["source"].get("source_type") != EXPECTED_SOURCE_TYPE:
        raise ValueError("continuous extractor requires continuous_numeric_ramp")
    _prior_automatic_extraction_state(experiment_log)
    reference = load_pinned_mapbox_reference(mapbox_manifest_path)
    accepted_count = experiment_log.data["alignment"]["accepted_automatic_iteration_count"]
    if accepted_count is None:
        raise ValueError("continuous numeric extraction requires accepted alignment")
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


def _prior_automatic_extraction_state(
    experiment_log: NoHumanExperimentLog,
) -> tuple[int, str | None]:
    """Validate append-only retry history and return ``(count, resume_status)``.

    A fresh run begins at automatic extraction iteration one.  A corrected
    deterministic algorithm may also continue an explicitly blocked automatic
    phase, but it must preserve every previous counted attempt and append the
    next contiguous ordinal.  In-progress, accepted, or mixed-authority
    histories are deliberately not guessed at or silently replayed.
    """

    phase = experiment_log.data["extraction"]
    if phase.get("accepted_automatic_iteration_count") is not None:
        raise ValueError("continuous numeric extraction is already accepted")
    iterations = list(phase.get("iterations", []))
    if any(
        not bool(item.get("counts_toward_automatic_iteration_count", False))
        for item in iterations
    ):
        raise ValueError("continuous numeric extraction history contains ineligible attempts")
    ordinals = [item.get("automatic_iteration") for item in iterations]
    expected = list(range(1, len(iterations) + 1))
    if ordinals != expected:
        raise ValueError("continuous numeric extraction history is not contiguous")
    if not iterations:
        if experiment_log.data.get("final", {}).get("status") != "in_progress":
            raise ValueError("fresh continuous numeric extraction log is not in progress")
        return 0, False
    if iterations[-1].get("decision") != "blocked":
        raise ValueError("blocked continuous numeric history lacks a terminal blocked attempt")
    final_status = experiment_log.data.get("final", {}).get("status")
    if final_status == "blocked":
        return len(iterations), "blocked"
    if final_status == "in_progress":
        resumptions = experiment_log.data.get("automatic_resumptions", [])
        if not resumptions or resumptions[-1].get("producer") != PRODUCER:
            raise ValueError("in-progress extraction history lacks this producer's resumption")
        return len(iterations), None
    raise ValueError("existing continuous numeric attempts require a blocked final state")


def _attempt_contains_forbidden_authority(payload: Mapping[str, Any]) -> str | None:
    """Scan an exact extraction payload while honoring negative attestations.

    ``automatic_provenance`` deliberately includes keys such as
    ``manual_arrows: false``.  Those fields prove the absence of manual input;
    the alignment scanner otherwise rejects the key name without considering
    the false value.  Remove only the three explicitly false attestations, then
    scan every other exact payload field with the shared strict scanner.
    """

    checked = dict(payload)
    provenance = dict(checked.get("provenance", {}))
    for key in ("manual_arrows", "manual_stamps", "human_approval"):
        if provenance.get(key) is not False:
            return f"provenance.{key}"
        provenance.pop(key)
    checked["provenance"] = provenance
    return _alignment_contains_forbidden_input(checked)


def run_automatic_continuous_numeric_extraction(
    source_adapter_manifest_path: Path,
    accepted_alignment_path: Path,
    mapbox_manifest_path: Path,
    output_dir: Path,
    experiment_log: NoHumanExperimentLog,
    experiment_markdown_path: Path,
    experiment_json_path: Path,
    *,
    config: ContinuousNumericConfig = ContinuousNumericConfig(),
) -> AutomaticContinuousNumericExtractionResult:
    """Extract numeric elevation plus a nonnumeric depression mask after alignment."""

    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise ValueError("continuous numeric extraction requires a fresh output directory")
    working, reference, alignment = _validate_inputs(
        source_adapter_manifest_path.resolve(),
        accepted_alignment_path.resolve(),
        mapbox_manifest_path.resolve(),
        experiment_log,
    )
    prior_automatic_count, resume_status = _prior_automatic_extraction_state(
        experiment_log
    )
    if resume_status == "blocked":
        experiment_log.resume_automatic_blocked(
            reason=(
                "cleaned OCR ramp sampling, projector-consistent inverse palette, "
                "and independently source-evidenced cartographic-ink occlusion"
            ),
            producer=PRODUCER,
        )
        experiment_log.write(experiment_markdown_path, experiment_json_path)
    output_dir.mkdir(parents=True)
    source_rgb = np.asarray(Image.open(working.working_raster_path).convert("RGB"))
    transform = alignment["transform"]
    source_domain = _source_data_mask(
        reference.state_land, reference.water, transform, source_rgb.shape[:2]
    )
    target_domain = reference.state_land & ~reference.water
    source_domain_path = output_dir / "source-mapbox-land-minus-water-mask.png"
    _save_mask(source_domain_path, source_domain)

    legend = detect_continuous_numeric_legend(
        working.working_raster_path, source_rgb, output_dir, config=config
    )
    words, _combined_tsv, _version = _run_tesseract_ocr(working.working_raster_path)
    classification, occlusion_scores = classify_continuous_numeric(
        source_rgb, source_domain, legend, words, config=config
    )
    source_reconstruction = _values_to_ramp_rgb(
        classification.completed_values_m,
        classification.completed_depression,
        legend,
        luminance_weight=config.luminance_weight,
    )
    reconstructed_values, reconstructed_residual = _project_pixels_to_ramp(
        source_reconstruction,
        legend,
        luminance_weight=config.luminance_weight,
        chunk_pixels=config.projection_chunk_pixels,
    )
    reconstructed_depression_residual = _depression_distance(
        source_reconstruction,
        legend.depression_rgb,
        config.luminance_weight,
        config.projection_chunk_pixels,
    )
    reconstructed_depression = (
        reconstructed_depression_residual <= config.maximum_special_residual
    ) & (reconstructed_depression_residual + 1.0 < reconstructed_residual)
    direct_numeric = np.isfinite(classification.observed_values_m)
    semantic_mismatch = source_domain & (
        (
            direct_numeric
            & (
                reconstructed_depression
                | (
                    np.abs(reconstructed_values - classification.observed_values_m)
                    > config.maximum_continuous_roundtrip_error_m
                )
            )
        )
        | (classification.observed_depression & ~reconstructed_depression)
    )

    reference_to_source = _reference_to_source_remap(transform)
    source_to_target = _source_to_reference_remap(transform, source_rgb.shape[:2])
    aligned_source = _source_to_reference(
        source_rgb, transform, cv2.INTER_LINEAR, (255, 255, 255), reference_to_source
    )
    source_encoded = _encode_meters(
        classification.completed_values_m, classification.completed_depression
    )
    target_encoded = _source_to_reference(
        source_encoded, transform, cv2.INTER_NEAREST, 0, reference_to_source
    ).astype(np.uint16)
    target_depression = _source_to_reference(
        classification.completed_depression.astype(np.uint8),
        transform,
        cv2.INTER_NEAREST,
        0,
        reference_to_source,
    ) > 0
    target_observed = _source_to_reference(
        classification.observed.astype(np.uint8),
        transform,
        cv2.INTER_NEAREST,
        0,
        reference_to_source,
    ) > 0
    target_occluded = _source_to_reference(
        classification.occluded.astype(np.uint8),
        transform,
        cv2.INTER_NEAREST,
        0,
        reference_to_source,
    ) > 0
    target_encoded[~target_domain] = 0
    target_depression &= target_domain
    target_observed &= target_domain
    target_occluded &= target_domain & ~target_observed
    target_values = _decode_meters(target_encoded, target_depression)
    incomplete = target_domain & ~target_depression & ~np.isfinite(target_values)
    if np.any(incomplete):
        observed_target_values = target_values.copy()
        completed_values, completed_depression = _nearest_continuous_completion(
            observed_target_values, target_depression, target_domain
        )
        target_values = completed_values
        target_depression = completed_depression
        target_encoded = _encode_meters(target_values, target_depression)
    target_inferred = target_domain & ~target_observed & ~target_occluded
    target_reconstruction = _values_to_ramp_rgb(
        target_values,
        target_depression,
        legend,
        luminance_weight=config.luminance_weight,
    )
    target_band_ids, intervals = _quantized_band_ids(
        target_values, target_depression, legend
    )

    domain_count = int(np.count_nonzero(source_domain))
    observed_count = int(np.count_nonzero(classification.observed))
    inferred_count = int(np.count_nonzero(classification.inferred))
    occluded_count = int(np.count_nonzero(classification.occluded))
    observed_fraction = observed_count / max(domain_count, 1)
    inferred_fraction = inferred_count / max(domain_count, 1)
    occluded_fraction = occluded_count / max(domain_count, 1)
    meaningful_count = observed_count
    mismatch_count = int(np.count_nonzero(semantic_mismatch))
    mismatch_fraction = mismatch_count / max(meaningful_count, 1)

    previous_source_encoded: np.ndarray | None = None
    previous_target_encoded: np.ndarray | None = None
    previous_target_depression: np.ndarray | None = None
    iterations: list[ContinuousNumericIteration] = []
    all_artifacts: list[Path] = [source_domain_path, *legend.artifact_paths]
    accepted_path: Path | None = None
    for replay_index in range(1, config.required_replay_count + 1):
        iteration_number = prior_automatic_count + replay_index
        iteration_dir = output_dir / f"extraction-{iteration_number:02d}"
        iteration_dir.mkdir()
        source_roundtrip_encoded = _warp_target_to_source(target_encoded, source_to_target)
        source_roundtrip_depression = _warp_target_to_source(
            target_depression.astype(np.uint8), source_to_target
        ) > 0
        source_roundtrip_values = _decode_meters(
            source_roundtrip_encoded, source_roundtrip_depression
        )
        source_numeric = source_domain & ~classification.completed_depression
        source_match = source_domain & (
            (
                classification.completed_depression
                & source_roundtrip_depression
            )
            | (
                source_numeric
                & ~source_roundtrip_depression
                & np.isfinite(source_roundtrip_values)
                & (
                    np.abs(
                        source_roundtrip_values - classification.completed_values_m
                    )
                    <= config.maximum_continuous_roundtrip_error_m
                )
            )
        )
        source_roundtrip_fraction = int(np.count_nonzero(source_match)) / max(
            domain_count, 1
        )
        geographic = _geographic_roundtrip_metrics(
            source_domain,
            source_match,
            source_to_target,
            target_domain.shape,
            config.geographic_rows,
            config.geographic_columns,
        )
        passing_cells = sum(
            cell["source_match_fraction"] >= config.minimum_geographic_match_fraction
            for cell in geographic
        )
        stable = (
            previous_source_encoded is not None
            and previous_target_encoded is not None
            and previous_target_depression is not None
            and np.array_equal(previous_source_encoded, source_encoded)
            and np.array_equal(previous_target_encoded, target_encoded)
            and np.array_equal(previous_target_depression, target_depression)
        )
        band_ids_present = {int(value) for value in np.unique(target_band_ids) if value}
        gates: dict[str, Any] = {
            "numeric_legend_semantics": bool(
                len(legend.stops) >= config.minimum_stop_count
                and legend.stops[-1].value_m == 0
                and all(
                    first.value_m > second.value_m
                    for first, second in zip(legend.stops, legend.stops[1:])
                )
            ),
            "numeric_stop_ocr_confidence": {
                "passed": all(
                    stop.ocr_confidence >= config.minimum_numeric_ocr_confidence
                    for stop in legend.stops
                ),
                "minimum": config.minimum_numeric_ocr_confidence,
                "values": [stop.ocr_confidence for stop in legend.stops],
            },
            "depression_semantic_readable_and_nonnumeric": {
                "passed": legend.depression_ocr_confidence
                >= config.minimum_special_ocr_confidence,
                "minimum": config.minimum_special_ocr_confidence,
                "value": legend.depression_ocr_confidence,
            },
            "all_numeric_intervals_preserved": band_ids_present
            == set(range(1, len(intervals) + 1)),
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
            "source_provenance_masks_disjoint_complete": bool(
                not np.any(classification.observed & classification.inferred)
                and not np.any(classification.observed & classification.occluded)
                and not np.any(classification.inferred & classification.occluded)
                and np.all(
                    classification.observed[source_domain]
                    | classification.inferred[source_domain]
                    | classification.occluded[source_domain]
                )
            ),
            "meaningful_source_reconstruction_mismatch": {
                "passed": mismatch_fraction
                <= config.maximum_semantic_mismatch_fraction,
                "value": mismatch_fraction,
                "maximum": config.maximum_semantic_mismatch_fraction,
            },
            "source_domain_complete": bool(
                np.all(
                    classification.completed_depression[source_domain]
                    | np.isfinite(classification.completed_values_m[source_domain])
                )
            ),
            "source_layout_empty": bool(
                not np.any(classification.completed_depression[~source_domain])
                and not np.any(np.isfinite(classification.completed_values_m[~source_domain]))
            ),
            "mapbox_water_and_exterior_empty": bool(
                not np.any(target_encoded[~target_domain] > 0)
                and not np.any(target_depression[~target_domain])
            ),
            "target_domain_complete": bool(
                np.all(target_depression[target_domain] | (target_encoded[target_domain] > 0))
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
                "supported_cells": len(geographic),
            },
            "successive_continuous_fixed_point": stable,
        }
        decision = (
            "accept"
            if _all_gates_pass(gates)
            else "retry" if replay_index < config.required_replay_count else "blocked"
        )

        source_values_path = iteration_dir / "source-elevation-meters-plus-one.png"
        source_bands_path = iteration_dir / "source-quantized-band-id.png"
        source_observed_path = iteration_dir / "source-observed-mask.png"
        source_inferred_path = iteration_dir / "source-inferred-mask.png"
        source_occluded_path = iteration_dir / "source-occluded-mask.png"
        source_depression_path = iteration_dir / "source-depression-mask.png"
        source_reconstruction_path = iteration_dir / "source-reconstruction.png"
        source_diff_path = iteration_dir / "source-semantic-diff-mask.png"
        source_roundtrip_diff_path = iteration_dir / "source-roundtrip-diff-mask.png"
        target_values_path = iteration_dir / "mapbox-elevation-meters-plus-one.png"
        target_bands_path = iteration_dir / "mapbox-quantized-band-id.png"
        target_observed_path = iteration_dir / "mapbox-observed-mask.png"
        target_inferred_path = iteration_dir / "mapbox-inferred-mask.png"
        target_occluded_path = iteration_dir / "mapbox-occluded-mask.png"
        target_depression_path = iteration_dir / "mapbox-depression-mask.png"
        target_reconstruction_path = iteration_dir / "mapbox-reconstruction.png"
        aligned_source_path = iteration_dir / "mapbox-aligned-source.png"
        diagnostic_path = iteration_dir / "source-extraction-comparison.png"
        source_band_ids, _source_intervals = _quantized_band_ids(
            classification.completed_values_m,
            classification.completed_depression,
            legend,
        )
        _save_u16(source_values_path, source_encoded)
        _save_ids(source_bands_path, source_band_ids)
        _save_mask(source_observed_path, classification.observed)
        _save_mask(source_inferred_path, classification.inferred)
        _save_mask(source_occluded_path, classification.occluded)
        _save_mask(source_depression_path, classification.completed_depression)
        _save_rgb(source_reconstruction_path, source_reconstruction)
        _save_mask(source_diff_path, semantic_mismatch)
        _save_mask(source_roundtrip_diff_path, source_domain & ~source_match)
        _save_u16(target_values_path, target_encoded)
        _save_ids(target_bands_path, target_band_ids)
        _save_mask(target_observed_path, target_observed)
        _save_mask(target_inferred_path, target_inferred)
        _save_mask(target_occluded_path, target_occluded)
        _save_mask(target_depression_path, target_depression)
        _save_rgb(target_reconstruction_path, target_reconstruction)
        _save_rgb(aligned_source_path, aligned_source)
        _save_rgb(
            diagnostic_path,
            _diagnostic(
                aligned_source,
                target_reconstruction,
                target_inferred,
                target_occluded,
            ),
        )
        class_mask_dir = iteration_dir / "quantized-band-masks"
        class_mask_dir.mkdir()
        class_mask_paths = []
        for class_id, minimum, maximum in intervals:
            path = class_mask_dir / f"{class_id:02d}-{minimum:g}-to-{maximum:g}-meters.png"
            _save_mask(path, target_band_ids == class_id)
            class_mask_paths.append(path)
        crop_paths = _write_geographic_crops(
            iteration_dir / "geographic-crops",
            aligned_source,
            target_reconstruction,
            target_inferred,
            target_occluded,
            target_domain,
            config.geographic_rows,
            config.geographic_columns,
        )
        artifact_paths = (
            source_values_path,
            source_bands_path,
            source_observed_path,
            source_inferred_path,
            source_occluded_path,
            source_depression_path,
            source_reconstruction_path,
            source_diff_path,
            source_roundtrip_diff_path,
            target_values_path,
            target_bands_path,
            target_observed_path,
            target_inferred_path,
            target_occluded_path,
            target_depression_path,
            target_reconstruction_path,
            aligned_source_path,
            diagnostic_path,
            *class_mask_paths,
            *crop_paths,
        )
        scores = {
            "numeric_stop_count": len(legend.stops),
            "numeric_stop_values_m": [stop.value_m for stop in legend.stops],
            "numeric_stop_ocr_confidences": [stop.ocr_confidence for stop in legend.stops],
            "depression_ocr_confidence": legend.depression_ocr_confidence,
            "source_domain_pixel_count": domain_count,
            "source_observed_pixel_count": observed_count,
            "source_inferred_pixel_count": inferred_count,
            "source_occluded_pixel_count": occluded_count,
            "source_observed_fraction": observed_fraction,
            "source_inferred_fraction": inferred_fraction,
            "source_occluded_fraction": occluded_fraction,
            "meaningful_source_pixel_count": meaningful_count,
            "meaningful_source_mismatch_pixel_count": mismatch_count,
            "meaningful_source_mismatch_fraction": mismatch_fraction,
            "source_alignment_roundtrip_fraction": source_roundtrip_fraction,
            "source_depression_pixel_count": int(
                np.count_nonzero(classification.completed_depression)
            ),
            "mapbox_depression_pixel_count": int(np.count_nonzero(target_depression)),
            "quantized_intervals_m": [
                {"class_id": class_id, "minimum": minimum, "maximum": maximum}
                for class_id, minimum, maximum in intervals
            ],
            "mapbox_quantized_band_pixel_counts": {
                str(class_id): int(np.count_nonzero(target_band_ids == class_id))
                for class_id, _minimum, _maximum in intervals
            },
            "geographic_cells": geographic,
            "successive_source_equal": stable,
            "successive_mapbox_equal": stable,
            "occlusion_components": dict(occlusion_scores),
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
        provenance = automatic_provenance(
            PRODUCER,
            [
                "authoritative_original_source_pixels",
                "automatic_numeric_ramp_stop_ocr",
                "automatic_dense_legend_color_curve",
                "automatic_depression_semantic_ocr",
                "accepted_automatic_mapbox_alignment",
                "pinned_mapbox_land_and_water",
                "source_reconstruction_diff",
                "deterministic_two_pass_fixed_point_replay",
            ],
        )
        method = (
            "automatic OCR-valued dense Lab ramp projection, separate nonnumeric "
            "Depression semantic, cartographic occlusion recovery, Mapbox clipping, "
            "source reconstruction/roundtrip geographic diff, and two-pass replay"
        )
        recorded_artifacts = [
            {"path": str(path), "sha256": _sha256(path)}
            for path in (*complete_artifacts, *legend.artifact_paths)
        ]
        exact_attempt_payload = {
            "scores": scores,
            "gates": gates,
            "decision": decision,
            "provenance": provenance,
            "method": method,
            "artifacts": recorded_artifacts,
        }
        forbidden = _attempt_contains_forbidden_authority(exact_attempt_payload)
        if forbidden:
            raise ValueError(
                f"Continuous extraction attempt contains forbidden authority at {forbidden}"
            )
        recorded_iteration = experiment_log.record_extraction_iteration(
            scores=scores,
            gates=gates,
            decision=decision,
            provenance=provenance,
            method=method,
            artifacts=recorded_artifacts,
        )
        if recorded_iteration["automatic_iteration"] != iteration_number:
            raise ValueError("continuous extraction automatic ordinal changed while appending")
        experiment_log.write(experiment_markdown_path, experiment_json_path)
        iterations.append(
            ContinuousNumericIteration(
                iteration_number,
                decision,
                scores,
                gates,
                report_path,
                tuple(complete_artifacts),
            )
        )
        all_artifacts.extend(complete_artifacts)
        previous_source_encoded = source_encoded.copy()
        previous_target_encoded = target_encoded.copy()
        previous_target_depression = target_depression.copy()
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
                            "path": "legend/continuous-numeric-legend.json",
                            "sha256": _sha256(legend.artifact_paths[0]),
                        },
                        "accepted_iteration": f"extraction-{iteration_number:02d}",
                        "layers": {
                            "elevation_meters": {
                                "kind": "continuous_numeric",
                                "units": "meters",
                                "encoding": "uint16 value_plus_one; zero_is_nodata",
                                "raster": f"extraction-{iteration_number:02d}/mapbox-elevation-meters-plus-one.png",
                            },
                            "depression": {
                                "kind": "nonnumeric_special_semantic",
                                "numeric_depth": None,
                                "mask": f"extraction-{iteration_number:02d}/mapbox-depression-mask.png",
                            },
                            "quantized_elevation_bands": {
                                "kind": "mutually_exclusive_numeric_intervals",
                                "class_id_raster": f"extraction-{iteration_number:02d}/mapbox-quantized-band-id.png",
                                "intervals_m": [
                                    {
                                        "class_id": class_id,
                                        "minimum": minimum,
                                        "maximum": maximum,
                                        "mask": (
                                            f"extraction-{iteration_number:02d}/quantized-band-masks/"
                                            f"{class_id:02d}-{minimum:g}-to-{maximum:g}-meters.png"
                                        ),
                                    }
                                    for class_id, minimum, maximum in intervals
                                ],
                            },
                        },
                        "observed_mask": f"extraction-{iteration_number:02d}/mapbox-observed-mask.png",
                        "inferred_mask": f"extraction-{iteration_number:02d}/mapbox-inferred-mask.png",
                        "occluded_mask": f"extraction-{iteration_number:02d}/mapbox-occluded-mask.png",
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
        return AutomaticContinuousNumericExtractionResult(
            "accepted",
            "numeric ramp, depression, source diff, geographic, and fixed-point gates passed",
            legend,
            tuple(iterations),
            accepted_path,
            tuple(all_artifacts),
        )
    blocker = (
        "Continuous numeric extraction did not pass every OCR, source-diff, "
        "geographic, provenance, and fixed-point gate after two deterministic passes"
    )
    experiment_log.finalize("blocked", blocker)
    experiment_log.write(experiment_markdown_path, experiment_json_path)
    return AutomaticContinuousNumericExtractionResult(
        "blocked", blocker, legend, tuple(iterations), None, tuple(all_artifacts)
    )
