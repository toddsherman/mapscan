"""Source-only canvas, legend, and thematic-support hypotheses.

This module deliberately stops before geographic registration.  It turns one
unaltered source raster into several deterministic source-space hypotheses that
an aligner can test against pinned reference geometry later.  No prior MapScan
alignment, extraction, authored correction, or approval is an input.

The important design choice is to preserve ambiguity.  A screenshot containing
a web-map canvas and a properties sidebar yields both the full canvas and one or
more cropped canvas hypotheses.  Likewise, every credible repeated legend-
swatch group produces a separate palette/support hypothesis.  Downstream gates,
not this source-only stage, decide which hypothesis is geographic.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from PIL import Image


SCHEMA_VERSION = 1
MANIFEST_NAME = "source-alignment-hypotheses.json"


@dataclass(frozen=True)
class SourceHypothesisConfig:
    working_max_dimension: int = 1200
    color_quantization_step: int = 16
    minimum_swatch_area_px: int = 12
    minimum_swatch_count: int = 3
    maximum_quantized_colors: int = 512
    maximum_legend_groups: int = 6
    palette_lab_tolerance: float = 14.0
    minimum_support_fraction: float = 0.002
    maximum_support_fraction: float = 0.92
    maximum_hypotheses: int = 24


@dataclass(frozen=True)
class LegendSwatch:
    box_working: tuple[int, int, int, int]
    box_original: tuple[int, int, int, int]
    rgb: tuple[int, int, int]
    quantized_code: int
    fill_fraction: float
    detection_kind: str


@dataclass(frozen=True)
class LegendGroup:
    id: str
    orientation: str
    score: float
    box_working: tuple[int, int, int, int]
    box_original: tuple[int, int, int, int]
    swatches: tuple[LegendSwatch, ...]


@dataclass(frozen=True)
class CanvasHypothesis:
    id: str
    kind: str
    score: float
    box_working: tuple[int, int, int, int]
    box_original: tuple[int, int, int, int]
    diagnostics: dict[str, float | int | str]


@dataclass(frozen=True)
class SourceAlignmentHypothesis:
    id: str
    score: float
    canvas_id: str
    support_kind: str
    legend_group_id: str | None
    roi_working: tuple[int, int, int, int]
    roi_original: tuple[int, int, int, int]
    palette_rgb: tuple[tuple[int, int, int], ...]
    diagnostics: dict[str, float | int | str]
    artifacts: dict[str, str]


@dataclass(frozen=True)
class SourceAlignmentHypothesisResult:
    manifest_path: Path
    layout_diagnostic_path: Path
    legend_diagnostic_path: Path
    legend_groups: tuple[LegendGroup, ...]
    canvases: tuple[CanvasHypothesis, ...]
    hypotheses: tuple[SourceAlignmentHypothesis, ...]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        frames = int(getattr(image, "n_frames", 1))
        if frames != 1:
            raise ValueError("Source alignment hypotheses require a single-frame image")
        return np.asarray(image.convert("RGB"))


def _resize_working(
    rgb: np.ndarray, maximum_dimension: int
) -> tuple[np.ndarray, float]:
    height, width = rgb.shape[:2]
    scale = min(1.0, float(maximum_dimension) / max(height, width))
    if scale == 1.0:
        return rgb.copy(), 1.0
    resized = cv2.resize(
        rgb,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def _original_box(
    box: Sequence[int], scale: float, original_shape: tuple[int, int]
) -> tuple[int, int, int, int]:
    height, width = original_shape
    x1, y1, x2, y2 = box
    working_width = max(1, round(width * scale))
    working_height = max(1, round(height * scale))
    return (
        max(0, min(width, round(x1 / scale))),
        max(0, min(height, round(y1 / scale))),
        width if x2 >= working_width else max(0, min(width, round(x2 / scale))),
        height if y2 >= working_height else max(0, min(height, round(y2 / scale))),
    )


def _rgb_to_code(rgb: np.ndarray, step: int) -> np.ndarray:
    levels = max(2, math.ceil(256 / step))
    quantized = np.minimum(rgb.astype(np.int32) // step, levels - 1)
    return (
        quantized[..., 0] * levels * levels
        + quantized[..., 1] * levels
        + quantized[..., 2]
    ).astype(np.int32)


def _swatch_components(
    rgb: np.ndarray, config: SourceHypothesisConfig
) -> list[dict[str, Any]]:
    """Find small, solid, rectangular components in commonly used colors."""

    height, width = rgb.shape[:2]
    codes = _rgb_to_code(rgb, config.color_quantization_step)
    values, counts = np.unique(codes, return_counts=True)
    eligible = [
        (int(count), int(value))
        for value, count in zip(values, counts)
        if count >= config.minimum_swatch_area_px
    ]
    eligible.sort(reverse=True)
    maximum_area = max(config.minimum_swatch_area_px, round(height * width * 0.006))
    maximum_width = max(6, round(width * 0.13))
    maximum_height = max(6, round(height * 0.08))
    result: list[dict[str, Any]] = []
    for _, code in eligible[: config.maximum_quantized_colors]:
        count, components, stats, _ = cv2.connectedComponentsWithStats(
            (codes == code).astype(np.uint8), 8
        )
        for component in range(1, count):
            x, y, component_width, component_height, area = map(
                int, stats[component]
            )
            if not (
                config.minimum_swatch_area_px <= area <= maximum_area
                and 4 <= component_width <= maximum_width
                and 4 <= component_height <= maximum_height
            ):
                continue
            aspect = component_width / max(component_height, 1)
            fill = area / max(component_width * component_height, 1)
            # JPEG ringing and a one-pixel dark outline can split the uniform
            # interior of an otherwise obvious legend rectangle.  Alignment,
            # repeated size, and color diversity are the stronger group-level
            # tests, so the component-level fill threshold is intentionally
            # permissive.
            if not (0.32 <= aspect <= 6.0 and fill >= 0.55):
                continue
            interior = rgb[
                y : y + component_height, x : x + component_width
            ][components[y : y + component_height, x : x + component_width] == component]
            if not len(interior):
                continue
            color = tuple(int(value) for value in np.median(interior, axis=0))
            result.append(
                {
                    "box": (x, y, x + component_width, y + component_height),
                    "width": component_width,
                    "height": component_height,
                    "area": area,
                    "fill": float(fill),
                    "code": code,
                    "rgb": color,
                    "center_x": x + component_width / 2.0,
                    "center_y": y + component_height / 2.0,
                }
            )
    return result


def _color_spread(colors: np.ndarray) -> float:
    if len(colors) < 2:
        return 0.0
    lab = cv2.cvtColor(colors.reshape(1, -1, 3).astype(np.uint8), cv2.COLOR_RGB2LAB)[
        0
    ].astype(np.float32)
    distances = np.linalg.norm(lab[:, None, :] - lab[None, :, :], axis=2)
    return float(np.quantile(distances, 0.75))


def _best_vertical_run(
    column: Sequence[dict[str, Any]], image_height: int
) -> list[dict[str, Any]]:
    """Keep one locally coherent legend run, excluding distant false boxes."""

    if len(column) < 2:
        return list(column)
    median_height = float(np.median([item["height"] for item in column]))
    split_gap = max(4.0 * median_height, image_height * 0.035)
    runs: list[list[dict[str, Any]]] = [[]]
    for item in column:
        if runs[-1] and item["center_y"] - runs[-1][-1]["center_y"] > split_gap:
            runs.append([])
        runs[-1].append(item)
    runs.sort(
        key=lambda run: (
            -len(run),
            -len({item["code"] for item in run}),
            run[-1]["center_y"] - run[0]["center_y"],
        )
    )
    return runs[0]


def _recover_vertical_swatch_gaps(
    rgb: np.ndarray, column: Sequence[dict[str, Any]], step: int
) -> list[dict[str, Any]]:
    """Recover a non-uniform swatch occupying an otherwise regular legend slot.

    Some source legends use a gradient inside one swatch.  Exact/quantized
    connected components then find every solid swatch except that one.  A gap
    is recovered only when its geometry is implied by at least four aligned
    peers and the expected rectangle is predominantly chromatic, non-white
    source material.
    """

    result = list(column)
    if len(result) < 4:
        return result
    result.sort(key=lambda item: item["center_y"])
    gaps = np.diff([item["center_y"] for item in result])
    if not len(gaps):
        return result
    typical = float(np.median(gaps))
    if typical <= 2.0:
        return result
    width = max(4, round(float(np.median([item["width"] for item in result]))))
    height = max(4, round(float(np.median([item["height"] for item in result]))))
    center_x = float(np.median([item["center_x"] for item in result]))
    levels = max(2, math.ceil(256 / step))
    additions: list[dict[str, Any]] = []
    for first, second in zip(result, result[1:]):
        gap = float(second["center_y"] - first["center_y"])
        slots = int(round(gap / typical))
        if slots < 2 or slots > 3 or abs(gap / slots - typical) > typical * 0.30:
            continue
        for slot in range(1, slots):
            center_y = first["center_y"] + gap * slot / slots
            x1 = max(0, round(center_x - width / 2))
            x2 = min(rgb.shape[1], x1 + width)
            y1 = max(0, round(center_y - height / 2))
            y2 = min(rgb.shape[0], y1 + height)
            patch = rgb[y1:y2, x1:x2]
            if patch.shape[0] < height * 0.75 or patch.shape[1] < width * 0.75:
                continue
            chroma = patch.max(axis=2).astype(np.int16) - patch.min(axis=2).astype(np.int16)
            nonwhite = np.min(patch, axis=2) < 238
            evidence_fraction = float(np.mean((chroma >= 18) & nonwhite))
            if evidence_fraction < 0.70:
                continue
            color = tuple(int(value) for value in np.median(patch.reshape(-1, 3), axis=0))
            quantized = np.minimum(np.asarray(color) // step, levels - 1)
            code = int(quantized[0] * levels * levels + quantized[1] * levels + quantized[2])
            additions.append(
                {
                    "box": (x1, y1, x2, y2),
                    "width": x2 - x1,
                    "height": y2 - y1,
                    "area": int((x2 - x1) * (y2 - y1)),
                    "fill": evidence_fraction,
                    "code": code,
                    "rgb": color,
                    "center_x": (x1 + x2) / 2.0,
                    "center_y": (y1 + y2) / 2.0,
                    "detection_kind": "recovered_regular_gap",
                }
            )
    result.extend(additions)
    result.sort(key=lambda item: item["center_y"])
    return result


def detect_repeated_legend_swatches(
    rgb: np.ndarray,
    *,
    scale_from_original: float = 1.0,
    original_shape: tuple[int, int] | None = None,
    config: SourceHypothesisConfig | None = None,
) -> tuple[LegendGroup, ...]:
    """Return aligned, size-consistent, multi-color swatch groups."""

    config = config or SourceHypothesisConfig()
    original_shape = original_shape or rgb.shape[:2]
    components = _swatch_components(rgb, config)
    if not components:
        return ()
    width = rgb.shape[1]
    x_tolerance = max(5.0, width * 0.012)
    used_keys: set[tuple[tuple[int, int, int, int], ...]] = set()
    groups: list[LegendGroup] = []
    for seed in sorted(components, key=lambda item: (item["center_x"], item["center_y"])):
        aligned = [
            item
            for item in components
            if abs(item["center_x"] - seed["center_x"]) <= x_tolerance
            and 0.58 <= item["width"] / max(seed["width"], 1) <= 1.72
            and 0.58 <= item["height"] / max(seed["height"], 1) <= 1.72
        ]
        aligned.sort(key=lambda item: item["center_y"])
        # Keep at most one rectangle at a given vertical position.  This avoids
        # duplicate quantized fragments from the same anti-aliased swatch.
        column: list[dict[str, Any]] = []
        for item in aligned:
            if column and abs(item["center_y"] - column[-1]["center_y"]) < max(
                2.0, min(item["height"], column[-1]["height"]) * 0.55
            ):
                if item["fill"] * item["area"] > column[-1]["fill"] * column[-1]["area"]:
                    column[-1] = item
                continue
            column.append(item)
        column = _best_vertical_run(column, rgb.shape[0])
        column = _recover_vertical_swatch_gaps(
            rgb, column, config.color_quantization_step
        )
        unique_codes = {item["code"] for item in column}
        if len(column) < config.minimum_swatch_count or len(unique_codes) < 3:
            continue
        colors = np.asarray([item["rgb"] for item in column], dtype=np.uint8)
        spread = _color_spread(colors)
        if spread < 12.0:
            continue
        widths = np.asarray([item["width"] for item in column], dtype=np.float64)
        heights = np.asarray([item["height"] for item in column], dtype=np.float64)
        centers = np.asarray([item["center_x"] for item in column])
        size_consistency = math.exp(
            -float(np.std(widths) / max(np.mean(widths), 1.0))
            -float(np.std(heights) / max(np.mean(heights), 1.0))
        )
        alignment = math.exp(-float(np.std(centers)) / max(x_tolerance, 1.0))
        count_score = min(1.0, len(column) / 8.0)
        spread_score = min(1.0, spread / 55.0)
        center_y = np.asarray([item["center_y"] for item in column], dtype=np.float64)
        gaps = np.diff(center_y)
        spacing_consistency = (
            math.exp(-float(np.std(gaps)) / max(float(np.mean(gaps)), 1.0))
            if len(gaps)
            else 0.0
        )
        color_uniqueness = len(unique_codes) / len(column)
        score = float(
            0.18 * count_score
            + 0.18 * size_consistency
            + 0.18 * alignment
            + 0.16 * spread_score
            + 0.16 * spacing_consistency
            + 0.14 * color_uniqueness
        )
        boxes = tuple(item["box"] for item in column)
        key = tuple(boxes)
        if key in used_keys:
            continue
        used_keys.add(key)
        x1 = min(box[0] for box in boxes)
        y1 = min(box[1] for box in boxes)
        x2 = max(box[2] for box in boxes)
        y2 = max(box[3] for box in boxes)
        swatches = tuple(
            LegendSwatch(
                box_working=item["box"],
                box_original=_original_box(
                    item["box"], scale_from_original, original_shape
                ),
                rgb=item["rgb"],
                quantized_code=item["code"],
                fill_fraction=item["fill"],
                detection_kind=item.get("detection_kind", "uniform_component"),
            )
            for item in column
        )
        groups.append(
            LegendGroup(
                id="",
                orientation="vertical",
                score=score,
                box_working=(x1, y1, x2, y2),
                box_original=_original_box(
                    (x1, y1, x2, y2), scale_from_original, original_shape
                ),
                swatches=swatches,
            )
        )
    groups.sort(
        key=lambda group: (
            -group.score,
            -len(group.swatches),
            group.box_working[0],
            group.box_working[1],
        )
    )
    # Highly overlapping groups are alternative seeds for the same legend.
    selected: list[LegendGroup] = []
    for group in groups:
        colors = {swatch.quantized_code for swatch in group.swatches}
        duplicate_index: int | None = None
        for index, prior in enumerate(selected):
            prior_colors = {swatch.quantized_code for swatch in prior.swatches}
            common = len(colors & prior_colors) / max(min(len(colors), len(prior_colors)), 1)
            center_gap = abs(
                (group.box_working[0] + group.box_working[2]) / 2
                - (prior.box_working[0] + prior.box_working[2]) / 2
            )
            if common >= 0.75 and center_gap <= x_tolerance:
                duplicate_index = index
                break
        if duplicate_index is None:
            selected.append(group)
        else:
            prior = selected[duplicate_index]
            if (
                len(group.swatches) > len(prior.swatches)
                and group.score >= prior.score - 0.08
            ):
                selected[duplicate_index] = group
        if len(selected) >= config.maximum_legend_groups:
            break
    selected.sort(key=lambda group: (-group.score, -len(group.swatches), group.box_working))
    return tuple(
        LegendGroup(
            id=f"legend-{index:02d}",
            orientation=group.orientation,
            score=group.score,
            box_working=group.box_working,
            box_original=group.box_original,
            swatches=group.swatches,
        )
        for index, group in enumerate(selected, start=1)
    )


def _suffix_ui_score(rgb: np.ndarray, x: int) -> float:
    suffix = rgb[:, x:]
    if not suffix.size:
        return 0.0
    lab = cv2.cvtColor(suffix, cv2.COLOR_RGB2LAB)
    chroma = suffix.max(axis=2).astype(np.int16) - suffix.min(axis=2).astype(np.int16)
    bright_neutral = (lab[..., 0] >= 205) & (chroma <= 18)
    return float(np.mean(bright_neutral))


def _canvas_base_score(rgb: np.ndarray, box: Sequence[int]) -> float:
    x1, y1, x2, y2 = map(int, box)
    crop = rgb[y1:y2, x1:x2]
    if not crop.size:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    chroma = crop.max(axis=2).astype(np.int16) - crop.min(axis=2).astype(np.int16)
    entropy_hist = np.bincount((gray // 16).ravel(), minlength=16).astype(np.float64)
    entropy_hist /= max(entropy_hist.sum(), 1.0)
    entropy = -float(np.sum(entropy_hist * np.log2(np.maximum(entropy_hist, 1e-12)))) / 4.0
    edge = cv2.Canny(gray, 70, 150)
    edge_score = min(1.0, float(np.mean(edge > 0)) / 0.12)
    chroma_score = min(1.0, float(np.mean(chroma >= 18)) / 0.35)
    return float(0.38 * entropy + 0.30 * edge_score + 0.32 * chroma_score)


def detect_map_canvas_hypotheses(
    rgb: np.ndarray,
    legend_groups: Sequence[LegendGroup],
    *,
    scale_from_original: float = 1.0,
    original_shape: tuple[int, int] | None = None,
) -> tuple[CanvasHypothesis, ...]:
    """Emit full-canvas and evidence-backed UI/header exclusion candidates."""

    height, width = rgb.shape[:2]
    original_shape = original_shape or rgb.shape[:2]
    panel_candidates: list[tuple[int, float, str]] = []
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    vertical_change = np.linalg.norm(lab[:, 1:, :] - lab[:, :-1, :], axis=2).mean(axis=0)
    for group in legend_groups:
        group_x1 = group.box_working[0]
        group_center = (group.box_working[0] + group.box_working[2]) / 2.0
        if group_center < width * 0.68:
            continue
        swatch_width = float(np.median([s.box_working[2] - s.box_working[0] for s in group.swatches]))
        search_start = max(round(width * 0.50), round(group_x1 - width * 0.18))
        search_end = min(width - 2, round(group_x1 - max(4.0, 1.35 * swatch_width)))
        if search_end <= search_start:
            continue
        possible: list[tuple[int, float]] = []
        edge_floor = float(np.quantile(vertical_change[search_start:search_end], 0.70))
        for x in range(search_start, search_end + 1):
            ui = _suffix_ui_score(rgb, x)
            if ui < 0.72 or vertical_change[x] < edge_floor:
                continue
            # Prefer a separator close to the first swatch, once the right
            # suffix has convincingly become bright neutral UI.
            proximity = 1.0 - (group_x1 - x) / max(group_x1 - search_start, 1)
            edge_score = min(1.0, float(vertical_change[x]) / 16.0)
            score = float(0.58 * ui + 0.24 * proximity + 0.18 * edge_score)
            possible.append((x + 1, score))
        possible.sort(key=lambda item: (-item[1], -item[0]))
        for right, score in possible[:2]:
            panel_candidates.append((right, score, group.id))

    # A full-width title/header is another common non-map layout region.
    horizontal_change = np.linalg.norm(lab[1:, :, :] - lab[:-1, :, :], axis=2).mean(axis=1)
    header_candidates: list[tuple[int, float]] = []
    search_end_y = max(2, round(height * 0.22))
    if search_end_y > 2:
        edge_floor = float(np.quantile(horizontal_change[:search_end_y], 0.86))
        for y in range(max(2, round(height * 0.025)), search_end_y):
            top = rgb[:y]
            top_chroma = top.max(axis=2).astype(np.int16) - top.min(axis=2).astype(np.int16)
            uniform = float(np.mean(top_chroma < 14))
            if horizontal_change[y] >= edge_floor and uniform >= 0.78:
                edge_score = min(1.0, float(horizontal_change[y]) / 18.0)
                header_candidates.append((y + 1, 0.65 * uniform + 0.35 * edge_score))
        header_candidates.sort(key=lambda item: (-item[1], item[0]))
        header_candidates = header_candidates[:1]

    strongest_panel = max((item[1] for item in panel_candidates), default=0.0)
    full_box = (0, 0, width, height)
    candidates: list[tuple[str, str, tuple[int, int, int, int], float, dict[str, Any]]] = [
        (
            "full-canvas",
            "full_canvas",
            full_box,
            max(0.0, _canvas_base_score(rgb, full_box) - 0.18 * strongest_panel),
            {"excluded_right_ui_confidence": 0.0, "excluded_header_confidence": 0.0},
        )
    ]
    seen_right: list[int] = []
    for right, confidence, legend_id in sorted(
        panel_candidates, key=lambda item: (-item[1], -item[0])
    ):
        if any(abs(right - prior) < max(5, round(width * 0.025)) for prior in seen_right):
            continue
        seen_right.append(right)
        box = (0, 0, right, height)
        candidates.append(
            (
                f"exclude-right-ui-{len(seen_right):02d}",
                "exclude_right_ui",
                box,
                min(1.0, _canvas_base_score(rgb, box) + 0.22 * confidence),
                {
                    "excluded_right_ui_confidence": float(confidence),
                    "excluded_header_confidence": 0.0,
                    "trigger_legend_group": legend_id,
                    "excluded_width_fraction": float((width - right) / width),
                },
            )
        )
        if len(seen_right) >= 2:
            break
    for index, (top, confidence) in enumerate(header_candidates, start=1):
        box = (0, top, width, height)
        candidates.append(
            (
                f"below-header-{index:02d}",
                "exclude_header",
                box,
                min(1.0, _canvas_base_score(rgb, box) + 0.15 * confidence),
                {
                    "excluded_right_ui_confidence": 0.0,
                    "excluded_header_confidence": float(confidence),
                    "excluded_height_fraction": float(top / height),
                },
            )
        )
        for right in seen_right[:1]:
            box = (0, top, right, height)
            candidates.append(
                (
                    "exclude-right-ui-and-header-01",
                    "exclude_right_ui_and_header",
                    box,
                    min(1.0, _canvas_base_score(rgb, box) + 0.20 * confidence),
                    {
                        "excluded_right_ui_confidence": float(strongest_panel),
                        "excluded_header_confidence": float(confidence),
                        "excluded_width_fraction": float((width - right) / width),
                        "excluded_height_fraction": float(top / height),
                    },
                )
            )
    candidates.sort(key=lambda item: (-item[3], item[0]))
    return tuple(
        CanvasHypothesis(
            id=item[0],
            kind=item[1],
            score=float(item[3]),
            box_working=item[2],
            box_original=_original_box(item[2], scale_from_original, original_shape),
            diagnostics=item[4],
        )
        for item in candidates
    )


def _palette_mask(
    rgb: np.ndarray,
    roi: Sequence[int],
    colors: Sequence[Sequence[int]],
    tolerance: float,
    legend_groups: Sequence[LegendGroup],
) -> tuple[np.ndarray, np.ndarray]:
    x1, y1, x2, y2 = map(int, roi)
    result = np.zeros(rgb.shape[:2], dtype=bool)
    assignments = np.full(rgb.shape[:2], -1, dtype=np.int16)
    if not colors or x2 <= x1 or y2 <= y1:
        return result, assignments
    palette = np.asarray(colors, dtype=np.uint8)
    # A near-white swatch is indistinguishable from page/UI background without
    # additional geographic evidence.  Omitting it is safer than inventing a
    # huge false state support component.
    palette_chroma = palette.max(axis=1).astype(np.int16) - palette.min(axis=1).astype(np.int16)
    keep = ~((palette.min(axis=1) >= 238) & (palette_chroma <= 8))
    palette = palette[keep]
    if not len(palette):
        return result, assignments
    crop = rgb[y1:y2, x1:x2]
    crop_lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB).astype(np.float32)
    palette_lab = cv2.cvtColor(
        palette.reshape(1, -1, 3), cv2.COLOR_RGB2LAB
    )[0].astype(np.float32)
    best_distance = np.full(crop.shape[:2], np.inf, dtype=np.float32)
    best_index = np.full(crop.shape[:2], -1, dtype=np.int16)
    for index, color in enumerate(palette_lab):
        distance = np.linalg.norm(crop_lab - color, axis=2)
        update = distance < best_distance
        best_distance[update] = distance[update]
        best_index[update] = index
    accepted = best_distance <= tolerance
    result[y1:y2, x1:x2] = accepted
    assignments[y1:y2, x1:x2][accepted] = best_index[accepted]
    # Legend marks are data definitions, never geographic support, even when a
    # legend is overlaid inside the map canvas.
    for group in legend_groups:
        for swatch in group.swatches:
            sx1, sy1, sx2, sy2 = swatch.box_working
            pad = 2
            result[max(0, sy1 - pad) : min(rgb.shape[0], sy2 + pad), max(0, sx1 - pad) : min(rgb.shape[1], sx2 + pad)] = False
            assignments[max(0, sy1 - pad) : min(rgb.shape[0], sy2 + pad), max(0, sx1 - pad) : min(rgb.shape[1], sx2 + pad)] = -1
    return result, assignments


def _foreground_mask(rgb: np.ndarray, roi: Sequence[int]) -> np.ndarray:
    """Conservative source-only fallback when no legend is detectable."""

    x1, y1, x2, y2 = map(int, roi)
    crop = rgb[y1:y2, x1:x2]
    result = np.zeros(rgb.shape[:2], dtype=bool)
    if not crop.size:
        return result
    codes = _rgb_to_code(crop, 24)
    background = np.zeros(codes.shape, dtype=bool)
    values, counts = np.unique(codes, return_counts=True)
    minimum_area = max(24, round(codes.size * 0.0015))
    for value, total in zip(values, counts):
        if total < minimum_area:
            continue
        count, components, stats, _ = cv2.connectedComponentsWithStats(
            (codes == value).astype(np.uint8), 8
        )
        for component in range(1, count):
            bx, by, width, height, area = map(int, stats[component])
            if area < minimum_area:
                continue
            if bx == 0 or by == 0 or bx + width == codes.shape[1] or by + height == codes.shape[0]:
                background |= components == component
    foreground = ~background
    foreground = cv2.morphologyEx(
        foreground.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
    ).astype(bool)
    result[y1:y2, x1:x2] = foreground
    return result


def _fill_small_holes(mask: np.ndarray, roi: Sequence[int]) -> np.ndarray:
    x1, y1, x2, y2 = map(int, roi)
    crop = mask[y1:y2, x1:x2].copy()
    inverse = ~crop
    count, components, stats, _ = cv2.connectedComponentsWithStats(
        inverse.astype(np.uint8), 8
    )
    maximum_hole = max(8, round(crop.size * 0.0012))
    for component in range(1, count):
        bx, by, width, height, area = map(int, stats[component])
        touches = bx == 0 or by == 0 or bx + width == crop.shape[1] or by + height == crop.shape[0]
        if not touches and area <= maximum_hole:
            crop[components == component] = True
    result = mask.copy()
    result[y1:y2, x1:x2] = crop
    return result


def _select_geographic_support(
    raw: np.ndarray, roi: Sequence[int]
) -> tuple[np.ndarray, dict[str, float | int]]:
    x1, y1, x2, y2 = map(int, roi)
    crop = raw[y1:y2, x1:x2].astype(np.uint8)
    if not np.any(crop):
        return np.zeros_like(raw), {
            "raw_support_pixel_count": 0,
            "selected_support_pixel_count": 0,
            "component_count": 0,
            "component_coherence": 0.0,
        }
    closed = cv2.morphologyEx(crop, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    count, components, stats, _ = cv2.connectedComponentsWithStats(closed, 8)
    if count <= 1:
        return np.zeros_like(raw), {
            "raw_support_pixel_count": int(np.count_nonzero(raw)),
            "selected_support_pixel_count": 0,
            "component_count": 0,
            "component_coherence": 0.0,
        }
    order = sorted(range(1, count), key=lambda index: int(stats[index, cv2.CC_STAT_AREA]), reverse=True)
    main = order[0]
    main_area = int(stats[main, cv2.CC_STAT_AREA])
    mx, my, mw, mh, _ = map(int, stats[main])
    main_mask = components == main
    distance_from_main = cv2.distanceTransform(
        (~main_mask).astype(np.uint8), cv2.DIST_L2, 5
    )
    keep = [main]
    minimum_satellite = max(5, round(main_area * 0.00015))
    for component in order[1:]:
        area = int(stats[component, cv2.CC_STAT_AREA])
        minimum_distance = float(np.min(distance_from_main[components == component]))
        maximum_satellite_gap = max(7.0, max(mw, mh) * 0.04)
        if area >= minimum_satellite and minimum_distance <= maximum_satellite_gap:
            keep.append(component)
    selected_crop = np.isin(components, keep)
    selected = np.zeros_like(raw)
    selected[y1:y2, x1:x2] = selected_crop
    selected = _fill_small_holes(selected, roi)
    selected_count = int(np.count_nonzero(selected))
    raw_count = int(np.count_nonzero(raw[y1:y2, x1:x2]))
    return selected, {
        "raw_support_pixel_count": raw_count,
        "selected_support_pixel_count": selected_count,
        "component_count": int(count - 1),
        "selected_component_count": int(len(keep)),
        "largest_component_pixel_count": main_area,
        "component_coherence": float(min(1.0, main_area / max(raw_count, 1))),
    }


def _boundary(mask: np.ndarray) -> np.ndarray:
    return cv2.morphologyEx(
        mask.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)
    ).astype(bool)


def _support_score(
    canvas: CanvasHypothesis,
    legend: LegendGroup | None,
    support: np.ndarray,
    assignments: np.ndarray | None,
    component_diagnostics: dict[str, float | int],
) -> tuple[float, dict[str, float | int | str]]:
    x1, y1, x2, y2 = canvas.box_working
    roi_area = max((x2 - x1) * (y2 - y1), 1)
    support_count = int(np.count_nonzero(support[y1:y2, x1:x2]))
    support_fraction = support_count / roi_area
    coherence = float(component_diagnostics.get("component_coherence", 0.0))
    if assignments is not None:
        observed = assignments[support]
        observed = observed[observed >= 0]
        palette_size = len(legend.swatches) if legend is not None else 0
        used = len(np.unique(observed)) if len(observed) else 0
        palette_usage = used / max(palette_size, 1)
    else:
        used = 0
        palette_size = 0
        palette_usage = 0.0
    fraction_score = min(1.0, support_fraction / 0.16)
    if support_fraction > 0.78:
        fraction_score *= max(0.0, (0.92 - support_fraction) / 0.14)
    legend_score = legend.score if legend is not None else 0.28
    usage_score = palette_usage if legend is not None else 0.35
    score = float(
        0.22 * canvas.score
        + 0.25 * legend_score
        + 0.20 * min(1.0, coherence)
        + 0.18 * min(1.0, usage_score)
        + 0.15 * fraction_score
    )
    return score, {
        **component_diagnostics,
        "roi_area_px": int(roi_area),
        "support_fraction_of_roi": float(support_fraction),
        "palette_color_count": int(palette_size),
        "observed_palette_color_count": int(used),
        "palette_usage_fraction": float(palette_usage),
        "canvas_score": float(canvas.score),
        "legend_score": float(legend.score) if legend is not None else 0.0,
    }


def _save_mask(path: Path, mask: np.ndarray) -> None:
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)


def _artifact_relative(path: Path, root: Path) -> str:
    return str(path.relative_to(root))


def _write_layout_diagnostic(
    path: Path,
    rgb: np.ndarray,
    canvases: Sequence[CanvasHypothesis],
) -> None:
    diagnostic = rgb.copy()
    colors = [(75, 255, 130), (255, 190, 50), (80, 195, 255), (255, 90, 190)]
    for index, canvas in enumerate(canvases):
        x1, y1, x2, y2 = canvas.box_working
        color = colors[index % len(colors)]
        cv2.rectangle(diagnostic, (x1, y1), (max(x1, x2 - 1), max(y1, y2 - 1)), color, 2)
        cv2.putText(
            diagnostic,
            f"{canvas.id} {canvas.score:.3f}",
            (x1 + 6, min(y2 - 7, y1 + 22 + 20 * index)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            color,
            1,
            cv2.LINE_AA,
        )
    Image.fromarray(diagnostic).save(path)


def _write_legend_diagnostic(
    path: Path, rgb: np.ndarray, groups: Sequence[LegendGroup]
) -> None:
    diagnostic = rgb.copy()
    colors = [(255, 80, 190), (80, 220, 255), (255, 210, 55), (90, 255, 130)]
    for index, group in enumerate(groups):
        color = colors[index % len(colors)]
        for swatch in group.swatches:
            x1, y1, x2, y2 = swatch.box_working
            cv2.rectangle(diagnostic, (x1, y1), (x2 - 1, y2 - 1), color, 2)
        x1, y1, _, _ = group.box_working
        cv2.putText(
            diagnostic,
            f"{group.id} n={len(group.swatches)} score={group.score:.3f}",
            (max(2, x1 - 4), max(14, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            color,
            1,
            cv2.LINE_AA,
        )
    Image.fromarray(diagnostic).save(path)


def generate_source_alignment_hypotheses(
    source_path: Path,
    output_root: Path,
    *,
    config: SourceHypothesisConfig | None = None,
) -> SourceAlignmentHypothesisResult:
    """Generate auditable source-space alignment hypotheses and diagnostics."""

    config = config or SourceHypothesisConfig()
    source_path = source_path.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Output root already exists: {output_root}")
    output_root.mkdir(parents=True)
    rgb_original = _load_rgb(source_path)
    rgb, scale = _resize_working(rgb_original, config.working_max_dimension)
    original_shape = rgb_original.shape[:2]
    groups = detect_repeated_legend_swatches(
        rgb,
        scale_from_original=scale,
        original_shape=original_shape,
        config=config,
    )
    canvases = detect_map_canvas_hypotheses(
        rgb,
        groups,
        scale_from_original=scale,
        original_shape=original_shape,
    )
    layout_path = output_root / "layout-hypotheses.png"
    legend_path = output_root / "legend-swatches.png"
    _write_layout_diagnostic(layout_path, rgb, canvases)
    _write_legend_diagnostic(legend_path, rgb, groups)

    generated: list[tuple[SourceAlignmentHypothesis, np.ndarray, np.ndarray]] = []
    for canvas in canvases:
        for group in groups:
            colors = tuple(swatch.rgb for swatch in group.swatches)
            raw, assignments = _palette_mask(
                rgb,
                canvas.box_working,
                colors,
                config.palette_lab_tolerance,
                groups,
            )
            support, component_diagnostics = _select_geographic_support(
                raw, canvas.box_working
            )
            score, diagnostics = _support_score(
                canvas, group, support, assignments, component_diagnostics
            )
            support_fraction = float(diagnostics["support_fraction_of_roi"])
            if not (
                config.minimum_support_fraction
                <= support_fraction
                <= config.maximum_support_fraction
            ):
                continue
            hypothesis_id = f"{canvas.id}--{group.id}-palette"
            generated.append(
                (
                    SourceAlignmentHypothesis(
                        id=hypothesis_id,
                        score=score,
                        canvas_id=canvas.id,
                        support_kind="legend_palette",
                        legend_group_id=group.id,
                        roi_working=canvas.box_working,
                        roi_original=canvas.box_original,
                        palette_rgb=colors,
                        diagnostics=diagnostics,
                        artifacts={},
                    ),
                    support,
                    _boundary(support),
                )
            )
        fallback_raw = _foreground_mask(rgb, canvas.box_working)
        fallback, component_diagnostics = _select_geographic_support(
            fallback_raw, canvas.box_working
        )
        score, diagnostics = _support_score(
            canvas, None, fallback, None, component_diagnostics
        )
        support_fraction = float(diagnostics["support_fraction_of_roi"])
        if (
            config.minimum_support_fraction
            <= support_fraction
            <= config.maximum_support_fraction
        ):
            generated.append(
                (
                    SourceAlignmentHypothesis(
                        id=f"{canvas.id}--border-foreground",
                        score=score,
                        canvas_id=canvas.id,
                        support_kind="border_connected_foreground",
                        legend_group_id=None,
                        roi_working=canvas.box_working,
                        roi_original=canvas.box_original,
                        palette_rgb=(),
                        diagnostics=diagnostics,
                        artifacts={},
                    ),
                    fallback,
                    _boundary(fallback),
                )
            )
    generated.sort(key=lambda item: (-item[0].score, item[0].id))
    generated = generated[: config.maximum_hypotheses]

    hypotheses: list[SourceAlignmentHypothesis] = []
    for index, (hypothesis, support, boundary) in enumerate(generated, start=1):
        prefix = f"hypothesis-{index:02d}-{hypothesis.id}"
        support_path = output_root / f"{prefix}-support.png"
        boundary_path = output_root / f"{prefix}-state-boundary.png"
        overlay_path = output_root / f"{prefix}-overlay.png"
        _save_mask(support_path, support)
        _save_mask(boundary_path, boundary)
        overlay = rgb.copy()
        tint = np.asarray([255, 70, 200], dtype=np.float32)
        overlay[support] = np.clip(
            0.68 * overlay[support].astype(np.float32) + 0.32 * tint, 0, 255
        ).astype(np.uint8)
        overlay[boundary] = (80, 255, 145)
        x1, y1, x2, y2 = hypothesis.roi_working
        cv2.rectangle(overlay, (x1, y1), (max(x1, x2 - 1), max(y1, y2 - 1)), (255, 210, 55), 2)
        Image.fromarray(overlay).save(overlay_path)
        artifacts = {
            "support_mask": _artifact_relative(support_path, output_root),
            "state_boundary_mask": _artifact_relative(boundary_path, output_root),
            "overlay": _artifact_relative(overlay_path, output_root),
        }
        hypotheses.append(
            SourceAlignmentHypothesis(
                id=hypothesis.id,
                score=hypothesis.score,
                canvas_id=hypothesis.canvas_id,
                support_kind=hypothesis.support_kind,
                legend_group_id=hypothesis.legend_group_id,
                roi_working=hypothesis.roi_working,
                roi_original=hypothesis.roi_original,
                palette_rgb=hypothesis.palette_rgb,
                diagnostics=hypothesis.diagnostics,
                artifacts=artifacts,
            )
        )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "source_alignment_hypotheses",
        "authority": {
            "original_source_pixels_only": True,
            "legacy_alignment_used": False,
            "legacy_extraction_used": False,
            "manual_input_used": False,
        },
        "source": {
            "path": str(source_path),
            "sha256": _sha256(source_path),
            "original_shape": [int(value) for value in original_shape],
            "working_shape": [int(value) for value in rgb.shape[:2]],
            "working_scale_from_original": float(scale),
        },
        "config": asdict(config),
        "legend_groups": [asdict(group) for group in groups],
        "canvas_hypotheses": [asdict(canvas) for canvas in canvases],
        "hypotheses": [asdict(hypothesis) for hypothesis in hypotheses],
        "diagnostics": {
            "layout": layout_path.name,
            "legend_swatches": legend_path.name,
        },
    }
    manifest_path = output_root / MANIFEST_NAME
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n")
    return SourceAlignmentHypothesisResult(
        manifest_path=manifest_path,
        layout_diagnostic_path=layout_path,
        legend_diagnostic_path=legend_path,
        legend_groups=groups,
        canvases=canvases,
        hypotheses=tuple(hypotheses),
    )
