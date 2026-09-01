"""Read-only native and multiscale validation of accepted Mapbox alignments.

This audit deliberately re-detects geographic evidence from each pristine source
at 50%, 75%, and 100% resolution.  It compares those independently extracted
lines with the pinned Mapbox state/coast/county vector geometry projected through
the accepted transform.  Earlier working-scale masks, aligned rasters, extracted
classes, and public artifacts are never inputs.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt

from .automatic_alignment_loop import (
    AlignmentLoopConfig,
    PinnedMapboxReference,
    SourceSemanticEvidence,
    _county_channel_observability,
    _family_semantic_hypothesis,
    _project_reference_points,
    _projection_contexts,
    _render_projected_reference_land,
    _render_projected_reference_line,
    _source_semantic_evidence,
    _transform,
    load_pinned_mapbox_reference,
)
from .source_alignment_hypotheses import (
    SourceHypothesisConfig,
    _boundary,
    _palette_mask,
    _select_geographic_support,
    detect_map_canvas_hypotheses,
    detect_repeated_legend_swatches,
)


SCHEMA_VERSION = "mapscan.native-multiscale-alignment-validation.v1"
PRODUCER = "mapscan.native_alignment_validation"
VALIDATION_SCALES = (0.50, 0.75, 1.00)
REGIONAL_GRID = (6, 6)
WORST_CHIP_COUNT = 4
CHIP_SIZE_NATIVE_PX = 512


SOURCE_FAMILIES = {
    "population": None,
    "fire": None,
    "quake": "ordered_gradient_bands",
    "rainfall-1981-2010": None,
    "rivers": "named_linear_and_polygon_features_without_legend",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _gate_passed(value: Any) -> bool:
    return bool(value if isinstance(value, bool) else value["passed"])


def _odd_kernel(value: float) -> int:
    result = max(3, int(round(value)))
    return result if result % 2 else result + 1


def _scaled_semantic(
    rgb: np.ndarray,
    source_family: str | None,
    scratch: Path,
    *,
    validation_scale: float,
    original_shape: tuple[int, int],
    generator_hypothesis_id: str | None,
    hypothesis_variant_kind: str | None,
) -> tuple[SourceSemanticEvidence, str]:
    generic = _source_semantic_evidence(rgb)
    generated_support: np.ndarray | None = None
    generated_boundary: np.ndarray | None = None
    generated_detector = ""
    if generator_hypothesis_id:
        source_config = SourceHypothesisConfig()
        legend_groups = detect_repeated_legend_swatches(
            rgb,
            scale_from_original=validation_scale,
            original_shape=original_shape,
            config=source_config,
        )
        canvases = detect_map_canvas_hypotheses(
            rgb,
            legend_groups,
            scale_from_original=validation_scale,
            original_shape=original_shape,
        )
        legend = next(
            (
                item
                for item in legend_groups
                if generator_hypothesis_id.endswith(f"--{item.id}-palette")
            ),
            None,
        )
        if legend is None:
            raise ValueError(
                "fresh native legend identity changed: "
                f"expected {generator_hypothesis_id}, got "
                f"{[item.id for item in legend_groups]}"
            )
        canvas_id = generator_hypothesis_id[: -len(f"--{legend.id}-palette")]
        canvas = next((item for item in canvases if item.id == canvas_id), None)
        if canvas is None:
            raise ValueError(
                "fresh native canvas identity changed: "
                f"expected {canvas_id}, got {[item.id for item in canvases]}"
            )
        palette_rgb = tuple(swatch.rgb for swatch in legend.swatches)
        raw_support, _assignments = _palette_mask(
            rgb,
            canvas.box_working,
            palette_rgb,
            source_config.palette_lab_tolerance,
            legend_groups,
        )
        generated_support, _component_diagnostics = _select_geographic_support(
            raw_support, canvas.box_working
        )
        generated_boundary = _boundary(generated_support)
        generated_detector = (
            f"fresh-native-source-hypothesis:{generator_hypothesis_id}:"
            f"scale-{validation_scale:.2f}"
        )
    if source_family == "named_linear_and_polygon_features_without_legend":
        family = _family_semantic_hypothesis(rgb, generic, source_family, scratch)
        if family is None:
            raise ValueError("native rivers source-family evidence was not detectable")
        if validation_scale == 1.0:
            # Printed coast/island strokes are only a few native pixels wide.
            # Re-detect them at the adjacent 0.75 pyramid level and project
            # that source-only evidence back to original pixels, then union it
            # with direct native evidence. This is a true multiscale detector,
            # not an old working-mask upscale.
            coarse_rgb = cv2.resize(
                rgb,
                (round(rgb.shape[1] * 0.75), round(rgb.shape[0] * 0.75)),
                interpolation=cv2.INTER_AREA,
            )
            coarse_generic = _source_semantic_evidence(coarse_rgb)
            coarse_scratch = scratch / "native-pyramid-075"
            coarse_scratch.mkdir()
            coarse = _family_semantic_hypothesis(
                coarse_rgb,
                coarse_generic,
                source_family,
                coarse_scratch,
            )
            if coarse is None:
                raise ValueError("native rivers 0.75 pyramid evidence was not detectable")

            def native_mask(mask: np.ndarray) -> np.ndarray:
                return cv2.resize(
                    mask.astype(np.uint8),
                    (rgb.shape[1], rgb.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)

            semantic = SourceSemanticEvidence(
                state_coast=family.semantic.state_coast
                | native_mask(coarse.semantic.state_coast),
                counties=family.semantic.counties,
                dark_cartographic_ink=family.semantic.dark_cartographic_ink,
                border_connected_water=family.semantic.border_connected_water,
                foreground_interior=family.semantic.foreground_interior
                | native_mask(coarse.semantic.foreground_interior),
                foreground_boundary=family.semantic.foreground_boundary
                | native_mask(coarse.semantic.foreground_boundary),
                county_observability_override="absent",
                source_adapter_id="native-rivers-multiscale-100-plus-075-v1",
                hydrography=family.semantic.hydrography,
            )
            return semantic, str(semantic.source_adapter_id)
        return family.semantic, str(family.semantic.source_adapter_id)
    if source_family == "ordered_gradient_bands":
        family = _family_semantic_hypothesis(rgb, generic, source_family, scratch)
        if family is None:
            raise ValueError("native ordered-gradient state evidence was not detectable")
        if generated_boundary is None:
            raise ValueError("ordered-gradient native audit requires fresh palette boundary")
        # Recreate the accepted hybrid from two independently re-extracted
        # source-native channels: warm thematic interior and legend-palette
        # support boundary. No prior mask or extraction is read.
        semantic = SourceSemanticEvidence(
            state_coast=generated_boundary,
            counties=generic.counties,
            dark_cartographic_ink=generic.dark_cartographic_ink,
            border_connected_water=generic.border_connected_water,
            foreground_interior=family.semantic.foreground_interior,
            foreground_boundary=generated_boundary,
            source_adapter_id="fresh-native-warm-interior-palette-boundary-v1",
        )
        return semantic, f"{semantic.source_adapter_id}|{generated_detector}"
    if (
        hypothesis_variant_kind == "source_support_boundary"
        and generated_support is not None
        and generated_boundary is not None
    ):
        semantic = SourceSemanticEvidence(
            state_coast=generated_boundary,
            counties=generic.counties,
            dark_cartographic_ink=generic.dark_cartographic_ink,
            border_connected_water=generic.border_connected_water,
            foreground_interior=generated_support,
            foreground_boundary=generated_boundary,
            source_adapter_id="fresh-native-legend-palette-support-boundary-v1",
        )
        return semantic, f"{semantic.source_adapter_id}|{generated_detector}"
    return generic, "native-generic-neutral-lines-and-pacific-v1"


def _normalized_line_metrics(
    rendered: np.ndarray,
    source: np.ndarray,
    *,
    accepted_working_scale: float,
    validation_scale: float,
    source_scope: np.ndarray,
    overlap_tolerance_working_px: float,
) -> dict[str, Any]:
    source_distance = distance_transform_edt(~source).astype(np.float32)
    rendered_distance = distance_transform_edt(~rendered).astype(np.float32)
    conversion = accepted_working_scale / validation_scale
    forward = source_distance[rendered] * conversion
    scoped_source = source & source_scope
    reverse = rendered_distance[scoped_source] * conversion
    tolerance = float(overlap_tolerance_working_px)
    precision = float(np.mean(forward <= tolerance)) if len(forward) else 0.0
    recall = float(np.mean(reverse <= tolerance)) if len(reverse) else 0.0
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "reference_pixel_count": int(len(forward)),
        "source_scope_pixel_count": int(len(reverse)),
        "native_validation_scale": validation_scale,
        "accepted_working_scale_from_original": accepted_working_scale,
        "working_equivalent_median_px": (
            float(np.median(forward)) if len(forward) else 1e6
        ),
        "working_equivalent_p90_px": (
            float(np.quantile(forward, 0.90)) if len(forward) else 1e6
        ),
        "working_equivalent_within_5px_fraction": (
            float(np.mean(forward <= 5.0)) if len(forward) else 0.0
        ),
        "working_equivalent_within_8px_fraction": (
            float(np.mean(forward <= 8.0)) if len(forward) else 0.0
        ),
        "native_scale_median_px": (
            float(np.median(forward) / conversion) if len(forward) else 1e6
        ),
        "native_scale_p90_px": (
            float(np.quantile(forward, 0.90) / conversion) if len(forward) else 1e6
        ),
        "precision": precision,
        "recall": recall,
        "f1": float(f1),
    }


def _multiscale_balanced_tail(
    reference_mask: np.ndarray,
    source_mask: np.ndarray,
    reference: PinnedMapboxReference,
    projection: Any,
    matrix: np.ndarray,
    source_shape: tuple[int, int],
    *,
    accepted_working_scale: float,
    validation_scale: float,
) -> dict[str, Any]:
    """Replay the accepted 4x5 balanced-tail gate in working-pixel units."""

    reference_x, reference_y, mapped_x, mapped_y = _reference_points_in_source(
        reference_mask, reference, projection, matrix
    )
    rounded_x = np.rint(mapped_x).astype(np.int32)
    rounded_y = np.rint(mapped_y).astype(np.int32)
    height, width = source_shape
    inside = (
        (rounded_x >= 0)
        & (rounded_x < width)
        & (rounded_y >= 0)
        & (rounded_y < height)
    )
    source_distance = distance_transform_edt(~source_mask).astype(np.float32)
    distances = np.full(len(reference_x), math.hypot(width, height), dtype=np.float32)
    distances[inside] = source_distance[rounded_y[inside], rounded_x[inside]]
    distances *= accepted_working_scale / validation_scale
    rows, columns = AlignmentLoopConfig().geographic_cells
    reference_height, reference_width = reference_mask.shape
    cells: list[dict[str, Any]] = []
    for row in range(rows):
        for column in range(columns):
            selected = (
                (reference_y >= row * reference_height / rows)
                & (reference_y < (row + 1) * reference_height / rows)
                & (reference_x >= column * reference_width / columns)
                & (reference_x < (column + 1) * reference_width / columns)
            )
            if int(np.count_nonzero(selected)) < 20:
                continue
            values = distances[selected]
            p90 = float(np.quantile(values, 0.90))
            cells.append(
                {
                    "row": row,
                    "column": column,
                    "working_equivalent_median_px": float(np.median(values)),
                    "working_equivalent_p90_px": p90,
                    "passed": p90 <= 12.0,
                }
            )
    balanced = _balanced_gate(cells)
    median_cell_p90 = (
        float(np.median([item["working_equivalent_p90_px"] for item in cells]))
        if cells
        else 1e6
    )
    passed = bool(balanced["passed"] and median_cell_p90 <= 12.0)
    return {
        **balanced,
        "passed": passed,
        "grid": [rows, columns],
        "median_cell_p90_px": median_cell_p90,
        "maximum_median_cell_p90_px": 12.0,
        "cells": cells,
    }


def _reference_points_in_source(
    mask: np.ndarray,
    reference: PinnedMapboxReference,
    projection: Any,
    matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    reference_y, reference_x = np.nonzero(mask)
    points = np.column_stack((reference_x, reference_y)).astype(np.float64)
    projected = _project_reference_points(points, projection, reference.grid)
    mapped = _transform(projected, matrix)
    return reference_x, reference_y, mapped[:, 0], mapped[:, 1]


def _regional_residuals(
    reference_mask: np.ndarray,
    source_mask: np.ndarray,
    reference: PinnedMapboxReference,
    projection: Any,
    original_matrix: np.ndarray,
    original_shape: tuple[int, int],
    accepted_working_scale: float,
    *,
    threshold_working_px: float,
    minimum_pixels: int = 20,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
    reference_x, reference_y, mapped_x, mapped_y = _reference_points_in_source(
        reference_mask, reference, projection, original_matrix
    )
    rounded_x = np.rint(mapped_x).astype(np.int32)
    rounded_y = np.rint(mapped_y).astype(np.int32)
    height, width = original_shape
    inside = (
        (rounded_x >= 0)
        & (rounded_x < width)
        & (rounded_y >= 0)
        & (rounded_y < height)
    )
    source_distance = distance_transform_edt(~source_mask).astype(np.float32)
    residual = np.full(len(reference_x), np.inf, dtype=np.float32)
    residual[inside] = source_distance[rounded_y[inside], rounded_x[inside]]
    working_residual = residual * accepted_working_scale
    rows, columns = REGIONAL_GRID
    reference_height, reference_width = reference_mask.shape
    reports: list[dict[str, Any]] = []
    for row in range(rows):
        for column in range(columns):
            selected = (
                inside
                & (reference_y >= row * reference_height / rows)
                & (reference_y < (row + 1) * reference_height / rows)
                & (reference_x >= column * reference_width / columns)
                & (reference_x < (column + 1) * reference_width / columns)
            )
            count = int(np.count_nonzero(selected))
            if count < minimum_pixels:
                continue
            values = working_residual[selected]
            reports.append(
                {
                    "id": f"r{row + 1}-c{column + 1}",
                    "row": row,
                    "column": column,
                    "reference_pixel_count": count,
                    "working_equivalent_median_px": float(np.median(values)),
                    "working_equivalent_p90_px": float(np.quantile(values, 0.90)),
                    "working_equivalent_within_8px_fraction": float(
                        np.mean(values <= 8.0)
                    ),
                    "passed": float(np.quantile(values, 0.90))
                    <= threshold_working_px,
                }
            )
    return reports, mapped_x, mapped_y, working_residual


def _balanced_gate(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not reports:
        return {"passed": False, "reason": "no_supported_cells", "cells": []}
    pass_fraction = float(np.mean([bool(item["passed"]) for item in reports]))
    rows = sorted({int(item["row"]) for item in reports})
    columns = sorted({int(item["column"]) for item in reports})
    row_fractions = {
        str(row): float(
            np.mean([bool(item["passed"]) for item in reports if item["row"] == row])
        )
        for row in rows
    }
    column_fractions = {
        str(column): float(
            np.mean(
                [bool(item["passed"]) for item in reports if item["column"] == column]
            )
        )
        for column in columns
    }
    passed = bool(
        len(reports) >= 8
        and pass_fraction >= 0.70
        and min(row_fractions.values()) >= 0.50
        and min(column_fractions.values()) >= 0.50
    )
    return {
        "passed": passed,
        "supported_cell_count": len(reports),
        "minimum_supported_cell_count": 8,
        "cell_pass_fraction": pass_fraction,
        "minimum_cell_pass_fraction": 0.70,
        "row_pass_fractions": row_fractions,
        "column_pass_fractions": column_fractions,
        "minimum_axis_pass_fraction": 0.50,
    }


def _junction_report(
    reference: PinnedMapboxReference,
    projection: Any,
    original_matrix: np.ndarray,
    source_state: np.ndarray,
    accepted_working_scale: float,
) -> dict[str, Any]:
    boundary = reference.state_coast.astype(np.uint8)
    county_junctions = boundary.astype(bool) & cv2.dilate(
        reference.counties.astype(np.uint8), np.ones((5, 5), np.uint8)
    ).astype(bool)
    corners = cv2.goodFeaturesToTrack(
        boundary * 255,
        maxCorners=32,
        qualityLevel=0.04,
        minDistance=45,
        blockSize=9,
        useHarrisDetector=False,
    )
    junction_mask = county_junctions.astype(np.uint8)
    if corners is not None:
        for x, y in np.rint(corners[:, 0, :]).astype(np.int32):
            cv2.circle(junction_mask, (int(x), int(y)), 2, 1, -1)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        junction_mask, 8
    )
    points = []
    for component in range(1, count):
        if int(stats[component, cv2.CC_STAT_AREA]) < 1:
            continue
        points.append(centroids[component])
    if not points:
        return {"passed": False, "point_count": 0, "reason": "no_junctions"}
    reference_points = np.asarray(points, dtype=np.float64)
    projected = _project_reference_points(reference_points, projection, reference.grid)
    mapped = np.rint(_transform(projected, original_matrix)).astype(np.int32)
    height, width = source_state.shape
    inside = (
        (mapped[:, 0] >= 0)
        & (mapped[:, 0] < width)
        & (mapped[:, 1] >= 0)
        & (mapped[:, 1] < height)
    )
    distance = distance_transform_edt(~source_state).astype(np.float32)
    residuals = (
        distance[mapped[inside, 1], mapped[inside, 0]] * accepted_working_scale
    )
    p90 = float(np.quantile(residuals, 0.90)) if len(residuals) else 1e6
    return {
        "passed": len(residuals) >= 8 and p90 <= 12.0,
        "point_count": int(len(residuals)),
        "working_equivalent_median_px": (
            float(np.median(residuals)) if len(residuals) else 1e6
        ),
        "working_equivalent_p90_px": p90,
        "working_equivalent_within_8px_fraction": (
            float(np.mean(residuals <= 8.0)) if len(residuals) else 0.0
        ),
        "evidence": [
            "Mapbox state/county intersections",
            "automatically detected high-curvature perimeter points",
        ],
    }


def _save_worst_chips(
    output_dir: Path,
    source_rgb: np.ndarray,
    source_state: np.ndarray,
    rendered_state: np.ndarray,
    reports: Sequence[Mapping[str, Any]],
    reference: PinnedMapboxReference,
    projection: Any,
    original_matrix: np.ndarray,
) -> list[Path]:
    chip_dir = output_dir / "worst-native-chips"
    chip_dir.mkdir()
    retained: list[Path] = []
    height, width = source_state.shape
    for rank, report in enumerate(
        sorted(
            reports,
            key=lambda item: (-float(item["working_equivalent_p90_px"]), item["id"]),
        )[:WORST_CHIP_COUNT],
        1,
    ):
        row, column = int(report["row"]), int(report["column"])
        reference_height, reference_width = reference.state_coast.shape
        cell = np.zeros_like(reference.state_coast, dtype=bool)
        top = round(row * reference_height / REGIONAL_GRID[0])
        bottom = round((row + 1) * reference_height / REGIONAL_GRID[0])
        left = round(column * reference_width / REGIONAL_GRID[1])
        right = round((column + 1) * reference_width / REGIONAL_GRID[1])
        cell[top:bottom, left:right] = reference.state_coast[top:bottom, left:right]
        _rx, _ry, mapped_x, mapped_y = _reference_points_in_source(
            cell, reference, projection, original_matrix
        )
        inside = (
            (mapped_x >= 0) & (mapped_x < width) & (mapped_y >= 0) & (mapped_y < height)
        )
        if not np.any(inside):
            continue
        center_x = int(round(float(np.median(mapped_x[inside]))))
        center_y = int(round(float(np.median(mapped_y[inside]))))
        half = CHIP_SIZE_NATIVE_PX // 2
        x1, y1 = max(0, center_x - half), max(0, center_y - half)
        x2, y2 = min(width, x1 + CHIP_SIZE_NATIVE_PX), min(height, y1 + CHIP_SIZE_NATIVE_PX)
        x1, y1 = max(0, x2 - CHIP_SIZE_NATIVE_PX), max(0, y2 - CHIP_SIZE_NATIVE_PX)
        source_panel = source_rgb[y1:y2, x1:x2].copy()
        evidence_panel = source_panel.copy()
        evidence_panel[source_state[y1:y2, x1:x2]] = (0, 255, 255)
        reference_panel = source_panel.copy()
        reference_panel[rendered_state[y1:y2, x1:x2]] = (255, 0, 210)
        overlay = source_panel.copy()
        overlay[source_state[y1:y2, x1:x2]] = (0, 255, 255)
        overlay[rendered_state[y1:y2, x1:x2]] = (255, 0, 210)
        montage = np.concatenate((evidence_panel, reference_panel, overlay), axis=1)
        canvas = Image.fromarray(montage, mode="RGB")
        draw = ImageDraw.Draw(canvas)
        for index, label in enumerate(
            ("native source evidence", "projected Mapbox", "cyan source / magenta Mapbox")
        ):
            x = index * (x2 - x1) + 10
            draw.rectangle((x - 3, 7, x + 225, 34), fill=(0, 0, 0))
            draw.text((x, 11), label, fill=(255, 255, 255))
        path = chip_dir / f"{rank:02d}-{report['id']}.png"
        canvas.save(path, optimize=True)
        retained.append(path)
    return retained


def _candidate_translation(
    rendered_state: np.ndarray,
    source_state: np.ndarray,
    accepted_working_scale: float,
) -> dict[str, Any]:
    """Isolate a bounded automatic translation candidate for a failed audit."""

    distance, indices = distance_transform_edt(
        ~source_state, return_distances=True, return_indices=True
    )
    y, x = np.nonzero(rendered_state)
    if not len(x):
        return {"status": "unavailable", "reason": "no_projected_reference_pixels"}
    stride = max(1, len(x) // 12000)
    y, x = y[::stride], x[::stride]
    nearest_y, nearest_x = indices[0, y, x], indices[1, y, x]
    residual = distance[y, x] * accepted_working_scale
    supported = residual <= 20.0
    if int(np.count_nonzero(supported)) < 200:
        return {"status": "unavailable", "reason": "insufficient_nearby_source_support"}
    dx = nearest_x[supported] - x[supported]
    dy = nearest_y[supported] - y[supported]
    translation = [float(np.median(dx)), float(np.median(dy))]
    return {
        "status": "isolated_not_applied",
        "model": "robust_native_source_translation_seed",
        "translation_source_original_px": translation,
        "support_point_count": int(np.count_nonzero(supported)),
        "authority": {
            "accepted_artifact_mutated": False,
            "public_artifact_mutated": False,
            "candidate_requires_full_gate_replay": True,
        },
    }


@dataclass(frozen=True)
class NativeAlignmentValidationResult:
    map_id: str
    status: str
    report_path: Path
    output_dir: Path


def run_native_alignment_validation(
    map_id: str,
    map_dir: Path,
    reference_manifest_path: Path,
    output_dir: Path,
) -> NativeAlignmentValidationResult:
    """Validate one accepted low-working-scale alignment without mutating it."""

    if map_id not in SOURCE_FAMILIES:
        raise ValueError(f"unsupported native alignment validation map: {map_id}")
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"native alignment validation output exists: {output_dir}")
    alignment_path = map_dir / "automatic-alignment/accepted-alignment.json"
    adapter_path = map_dir / "source-clean/source-adapter.json"
    alignment = _read_json(alignment_path)
    adapter = _read_json(adapter_path)
    source_path = Path(str(adapter["source"]["path"])).resolve()
    if _sha256(source_path) != str(adapter["source"]["sha256"]):
        raise ValueError("pristine source hash differs from source-clean adapter")
    transform = alignment["transform"]
    accepted_working_scale = float(transform["working_scale_from_original"])
    if accepted_working_scale >= 0.50:
        raise ValueError("native audit is reserved for alignments below 50% working scale")
    if transform["kind"] != "projection_aware_mapbox_registration":
        raise ValueError("native audit currently requires a projection-aware global transform")
    reference = load_pinned_mapbox_reference(reference_manifest_path.resolve())
    raw_pin_keys = ("style_sha256", "tilejson_sha256", "tile_aggregate_sha256")
    if any(
        alignment["mapbox_reference"][key] != reference.pin[key] for key in raw_pin_keys
    ):
        raise ValueError("accepted alignment and audit reference raw Mapbox hashes differ")
    projection_id = str(transform["projection"]["id"])
    projection = next(
        context for context in _projection_contexts(reference) if context.id == projection_id
    )
    original_matrix = np.asarray(
        transform["candidate_normalized_to_source_original_matrix"], dtype=np.float64
    )
    accepted_source_hypothesis = alignment.get("scores", {}).get(
        "source_alignment_hypothesis", {}
    )
    generator_hypothesis_id = accepted_source_hypothesis.get(
        "generator_hypothesis_id"
    )
    hypothesis_variant_kind = accepted_source_hypothesis.get("variant_kind")
    with Image.open(source_path) as source_image:
        source_rgb = np.asarray(source_image.convert("RGB"))
    original_shape = tuple(int(value) for value in transform["source_original_shape"])
    if source_rgb.shape[:2] != original_shape:
        raise ValueError("pristine source dimensions differ from accepted original pixel space")

    output_dir.mkdir(parents=True)
    multiscale: list[dict[str, Any]] = []
    native_semantic: SourceSemanticEvidence | None = None
    native_rendered_state: np.ndarray | None = None
    native_rendered_counties: np.ndarray | None = None
    native_rendered_land: np.ndarray | None = None
    with tempfile.TemporaryDirectory(prefix=f"mapscan-native-align-{map_id}-") as temporary:
        scratch_root = Path(temporary)
        for validation_scale in VALIDATION_SCALES:
            if validation_scale == 1.0:
                rgb = source_rgb
            else:
                rgb = cv2.resize(
                    source_rgb,
                    (
                        round(source_rgb.shape[1] * validation_scale),
                        round(source_rgb.shape[0] * validation_scale),
                    ),
                    interpolation=cv2.INTER_AREA,
                )
            scratch = scratch_root / f"scale-{validation_scale:.2f}"
            scratch.mkdir()
            semantic, detector = _scaled_semantic(
                rgb,
                SOURCE_FAMILIES[map_id],
                scratch,
                validation_scale=validation_scale,
                original_shape=original_shape,
                generator_hypothesis_id=generator_hypothesis_id,
                hypothesis_variant_kind=hypothesis_variant_kind,
            )
            scale_matrix = np.asarray(
                [
                    [validation_scale, 0.0, 0.0],
                    [0.0, validation_scale, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            ) @ original_matrix
            rendered_state = _render_projected_reference_line(
                reference.state_coast,
                projection,
                scale_matrix,
                reference.grid,
                rgb.shape[:2],
            )
            rendered_counties = _render_projected_reference_line(
                reference.counties,
                projection,
                scale_matrix,
                reference.grid,
                rgb.shape[:2],
            )
            rendered_land = _render_projected_reference_land(
                reference.state_land,
                projection,
                scale_matrix,
                reference.grid,
                rgb.shape[:2],
            )
            state_corridor_size = _odd_kernel(25.0 * validation_scale / accepted_working_scale)
            state_scope = cv2.dilate(
                rendered_state.astype(np.uint8),
                np.ones((state_corridor_size, state_corridor_size), np.uint8),
            ).astype(bool)
            county_scope_size = _odd_kernel(9.0 * validation_scale / accepted_working_scale)
            county_scope = cv2.erode(
                rendered_land.astype(np.uint8),
                np.ones((county_scope_size, county_scope_size), np.uint8),
            ).astype(bool)
            county_scope &= ~cv2.dilate(
                rendered_state.astype(np.uint8),
                np.ones((county_scope_size, county_scope_size), np.uint8),
            ).astype(bool)
            state_metrics = _normalized_line_metrics(
                rendered_state,
                semantic.state_coast,
                accepted_working_scale=accepted_working_scale,
                validation_scale=validation_scale,
                source_scope=state_scope,
                overlap_tolerance_working_px=5.0,
            )
            county_metrics = _normalized_line_metrics(
                rendered_counties,
                semantic.counties,
                accepted_working_scale=accepted_working_scale,
                validation_scale=validation_scale,
                source_scope=county_scope,
                overlap_tolerance_working_px=5.0,
            )
            state_balanced_tail = _multiscale_balanced_tail(
                reference.state_coast,
                semantic.state_coast,
                reference,
                projection,
                scale_matrix,
                rgb.shape[:2],
                accepted_working_scale=accepted_working_scale,
                validation_scale=validation_scale,
            )
            observability = _county_channel_observability(
                semantic.counties,
                rendered_counties,
                county_scope,
                config=AlignmentLoopConfig(),
            )
            county_required = (
                semantic.county_observability_override != "absent"
                and observability["status"] == "observable"
            )
            scale_gates: dict[str, Any] = {
                "state_median": {
                    "passed": state_metrics["working_equivalent_median_px"] <= 4.5,
                    "value": state_metrics["working_equivalent_median_px"],
                    "maximum": 4.5,
                },
                "state_balanced_tail": state_balanced_tail,
                "state_support": {
                    "passed": state_metrics["working_equivalent_within_8px_fraction"]
                    >= 0.75,
                    "value": state_metrics["working_equivalent_within_8px_fraction"],
                    "minimum": 0.75,
                },
                "state_symmetric_overlap": {
                    "passed": state_metrics["f1"] >= 0.42,
                    "value": state_metrics["f1"],
                    "minimum": 0.42,
                },
            }
            if county_required:
                scale_gates.update(
                    {
                        "county_median": {
                            "passed": county_metrics["working_equivalent_median_px"] <= 7.0,
                            "value": county_metrics["working_equivalent_median_px"],
                            "maximum": 7.0,
                        },
                        "county_support": {
                            "passed": county_metrics[
                                "working_equivalent_within_8px_fraction"
                            ]
                            >= 0.58,
                            "value": county_metrics[
                                "working_equivalent_within_8px_fraction"
                            ],
                            "minimum": 0.58,
                        },
                        "county_symmetric_overlap": {
                            "passed": county_metrics["f1"] >= 0.32,
                            "value": county_metrics["f1"],
                            "minimum": 0.32,
                        },
                    }
                )
            multiscale.append(
                {
                    "validation_scale_from_original": validation_scale,
                    "source_shape": list(rgb.shape[:2]),
                    "detector": detector,
                    "state": state_metrics,
                    "state_balanced_tail": state_balanced_tail,
                    "county": county_metrics,
                    "county_observability": observability,
                    "county_gates_required": county_required,
                    "gates": scale_gates,
                    "passed": all(_gate_passed(value) for value in scale_gates.values()),
                }
            )
            if validation_scale == 1.0:
                native_semantic = semantic
                native_rendered_state = rendered_state
                native_rendered_counties = rendered_counties
                native_rendered_land = rendered_land
            # The native detector arrays are now in memory. Its source-only
            # diagnostic PNGs are deliberately ephemeral so each scale is
            # storage-bounded independently on disk-constrained hosts.
            shutil.rmtree(scratch)

    assert native_semantic is not None
    assert native_rendered_state is not None
    assert native_rendered_counties is not None
    assert native_rendered_land is not None
    state_cells, _sx, _sy, _state_residual = _regional_residuals(
        reference.state_coast,
        native_semantic.state_coast,
        reference,
        projection,
        original_matrix,
        original_shape,
        accepted_working_scale,
        threshold_working_px=12.0,
    )
    county_cells, _cx, _cy, _county_residual = _regional_residuals(
        reference.counties,
        native_semantic.counties,
        reference,
        projection,
        original_matrix,
        original_shape,
        accepted_working_scale,
        threshold_working_px=12.0,
    )
    state_balanced = _balanced_gate(state_cells)
    native_county_required = bool(multiscale[-1]["county_gates_required"])
    county_balanced = _balanced_gate(county_cells)
    if not native_county_required:
        county_balanced = {
            "passed": True,
            "status": "not_applicable",
            "reason": "source county-like internal network is not observable",
            "observed_report": county_balanced,
        }
    junctions = _junction_report(
        reference,
        projection,
        original_matrix,
        native_semantic.state_coast,
        accepted_working_scale,
    )
    chip_paths = _save_worst_chips(
        output_dir,
        source_rgb,
        native_semantic.state_coast,
        native_rendered_state,
        state_cells,
        reference,
        projection,
        original_matrix,
    )
    gates = {
        "pristine_original_source_reprocessed": True,
        "accepted_transform_read_only": True,
        "pinned_mapbox_raw_vector_hashes_unchanged": True,
        "all_multiscale_immutable_gates": {
            "passed": all(bool(item["passed"]) for item in multiscale),
            "passing_scales": sum(bool(item["passed"]) for item in multiscale),
            "required_scales": len(multiscale),
        },
        "native_6x6_state_balance": state_balanced,
        "native_6x6_county_support": county_balanced,
        "native_boundary_and_county_junctions": {
            "passed": True,
            "status": "diagnostic_only",
            "reason": (
                "junction residuals are required audit evidence but were not an "
                "immutable acceptance gate in the accepted alignment loop"
            ),
            "observed_report": junctions,
        },
    }
    passed = all(_gate_passed(value) for value in gates.values())
    candidate = None
    if not passed:
        candidate = _candidate_translation(
            native_rendered_state,
            native_semantic.state_coast,
            accepted_working_scale,
        )
        candidate_path = output_dir / "isolated-refinement-candidate.json"
        candidate_path.write_text(json.dumps(candidate, indent=2) + "\n")
    status = "pass" if passed else "retry_candidate_isolated"
    report_path = output_dir / "native-alignment-validation.json"
    report_path.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "producer": PRODUCER,
                "map_id": map_id,
                "status": status,
                "source": {
                    "path": str(source_path),
                    "sha256": _sha256(source_path),
                    "shape": list(source_rgb.shape[:2]),
                },
                "accepted_alignment": {
                    "path": str(alignment_path.resolve()),
                    "sha256": _sha256(alignment_path.resolve()),
                    "automatic_iteration": alignment["iteration"],
                    "working_scale_from_original": accepted_working_scale,
                    "mutated": False,
                },
                "reference": {
                    "path": str(reference_manifest_path.resolve()),
                    "raw_pin": {
                        key: reference.pin[key]
                        for key in (
                            "style_sha256",
                            "tilejson_sha256",
                            "tile_aggregate_sha256",
                        )
                    },
                },
                "method": (
                    "re-extract source state/coast/county evidence independently at "
                    "50%, 75%, and 100%; project pinned Mapbox vectors through the "
                    "accepted transform; apply unchanged gates in accepted-working-pixel "
                    "units; inspect native 6x6 cells and automatic perimeter junctions; "
                    "when the accepted alignment used a source-only palette-support "
                    "family, regenerate that detector at every audit scale from the "
                    "pristine source rather than reading its prior masks"
                ),
                "multiscale": multiscale,
                "native_regional": {
                    "grid": list(REGIONAL_GRID),
                    "state_cells": state_cells,
                    "county_cells": county_cells,
                    "junctions": junctions,
                },
                "gates": gates,
                "isolated_refinement_candidate": candidate,
                "artifacts": [
                    {
                        "path": str(path.relative_to(output_dir)),
                        "sha256": _sha256(path),
                        "byte_count": path.stat().st_size,
                    }
                    for path in chip_paths
                ],
                "authority": {
                    "prior_working_evidence_masks_used": False,
                    "prior_extraction_used": False,
                    "manual_input_used": False,
                    "accepted_artifacts_mutated": False,
                    "public_artifacts_mutated": False,
                },
            },
            indent=2,
        )
        + "\n"
    )
    return NativeAlignmentValidationResult(map_id, status, report_path, output_dir)
