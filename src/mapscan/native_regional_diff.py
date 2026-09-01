"""Storage-bounded native-source regional fidelity evaluation.

Accepted target-grid extractions are sampled back into original-source pixels.
Scores are computed for a fixed geographic 6x6 grid, but only the worst few
visual panels are persisted.  Existing accepted/public artifacts are read-only.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw
from pyproj import CRS, Transformer

from .resolution_fidelity_audit import plan_native_regional_diffs


SCHEMA_VERSION = "mapscan.native-regional-source-diff.v1"
GRID = (6, 6)

_CONFIGS: dict[str, dict[str, Any]] = {
    "population": {
        "family": "categorical",
        "source_id": "source-class-id.png",
        "target_id": "web-mercator-class-id.png",
        "source_diff": "source-semantic-diff-mask.png",
        "target_rgb": "web-mercator-reconstruction.png",
        "global_semantic_max": 0.002,
        "cell_semantic_max": 0.010,
    },
    "rainfall-1981-2010": {
        "family": "categorical",
        "source_id": "source-class-id.png",
        "target_id": "web-mercator-class-id.png",
        "source_diff": "source-semantic-diff-mask.png",
        "target_rgb": "web-mercator-reconstruction.png",
        "global_semantic_max": 0.002,
        "cell_semantic_max": 0.010,
    },
    "deer": {
        "family": "categorical",
        "source_id": "source-class-id.png",
        "target_id": "web-mercator-class-id.png",
        "source_diff": "source-semantic-diff-mask.png",
        "target_rgb": "web-mercator-reconstruction.png",
        "global_semantic_max": 0.002,
        "cell_semantic_max": 0.010,
    },
    "forest": {
        "family": "categorical",
        "source_id": "source-class-id.png",
        "target_id": "web-mercator-class-id.png",
        "source_diff": "source-semantic-diff-mask.png",
        "target_rgb": "web-mercator-reconstruction.png",
        "global_semantic_max": 0.002,
        "cell_semantic_max": 0.010,
    },
    "plantzone": {
        "family": "categorical",
        "source_id": "source-class-id.png",
        "target_id": "web-mercator-class-id.png",
        "source_diff": "source-semantic-diff-mask.png",
        "target_rgb": "web-mercator-reconstruction.png",
        "global_semantic_max": 0.002,
        "cell_semantic_max": 0.010,
    },
    "quake": {
        "family": "categorical",
        "source_id": "source-class-id.png",
        "target_id": "mapbox-class-id.png",
        "source_diff": "source-semantic-diff-mask.png",
        "target_rgb": "mapbox-reconstruction.png",
        "exclude": ["source-occluded-nonthematic-mask.png"],
        "global_semantic_max": 0.005,
        "cell_semantic_max": 0.020,
    },
    "elevation": {
        "family": "continuous",
        "source_id": "source-quantized-band-id.png",
        "target_id": "mapbox-quantized-band-id.png",
        "source_diff": "source-semantic-diff-mask.png",
        "target_rgb": "mapbox-reconstruction.png",
        "exclude": ["source-occluded-mask.png"],
        "global_semantic_max": 0.005,
        "cell_semantic_max": 0.020,
    },
    "fire": {
        "family": "overlapping",
        "target_rgb": "mapbox-composite-reconstruction.png",
        "classes": [
            {
                "name": "hazard",
                "source": "source-hazard-class-id.png",
                "target": "mapbox-hazard-class-id.png",
            }
        ],
        "binary": [
            {
                "name": "local-responsibility-area",
                "source": "source-lra-dot-mask.png",
                "target": "mapbox-lra-dot-mask.png",
                "global_f1_min": 0.95,
                "cell_f1_min": 0.80,
            }
        ],
        "source_diff": [
            "source-hazard-roundtrip-diff-mask.png",
            "source-lra-roundtrip-diff-mask.png",
        ],
        "global_diff_max": 0.005,
        "cell_diff_max": 0.030,
    },
    "landslide": {
        "family": "overlapping",
        "target_rgb": "mapbox-composite-reconstruction.png",
        "classes": [
            {
                "name": "precipitation",
                "source": "source-precipitation-class-id.png",
                "target": "mapbox-precipitation-class-id.png",
                "exclude": [
                    "source-precipitation-occluded-mask.png",
                    "source-overlay-ambiguous-mask.png",
                ],
            }
        ],
        "binary": [
            {
                "name": "landslide-susceptibility",
                "source": "source-landslide-susceptibility-mask.png",
                "target": "mapbox-landslide-susceptibility-mask.png",
                "global_f1_min": 0.95,
                "cell_f1_min": 0.80,
            },
            {
                "name": "maximum-wind-speed",
                "source": "source-maximum-wind-speed-mask.png",
                "target": "mapbox-maximum-wind-speed-mask.png",
                "global_f1_min": 0.95,
                "cell_f1_min": 0.80,
            },
            {
                "name": "predicted-flooding",
                "source": "source-predicted-flooding-mask.png",
                "target": "mapbox-predicted-flooding-mask.png",
                "global_f1_min": 0.95,
                "cell_f1_min": 0.80,
            },
        ],
        "source_diff": ["source-composite-roundtrip-diff-mask.png"],
        "global_diff_max": 0.010,
        "cell_diff_max": 0.050,
    },
    "rivers": {
        "family": "linear",
        "binary": [
            {
                "name": name,
                "source": f"source-{name}.png",
                "target": f"mapbox-{name}.png",
                "global_f1_min": 0.85,
                "cell_f1_min": 0.60,
                "optional_empty": name == "line-reconnection-inferred",
            }
            for name in (
                "river-or-stream-observed",
                "lake-or-reservoir-outline-observed",
                "lake-or-reservoir-interior-inferred",
                "dry-streambed-observed",
                "dry-lake-outline-observed",
                "dry-lake-interior-inferred",
                "line-reconnection-inferred",
            )
        ],
        "source_diff": ["source-semantic-roundtrip-diff.png"],
        "global_diff_max": 0.050,
        "cell_diff_max": 0.150,
    },
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _load_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.copy())


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def _web_mercator_to_reference(
    web_x: np.ndarray, web_y: np.ndarray, grid: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    minimum_x, minimum_y, maximum_x, maximum_y = map(float, grid["bounds"])
    width, height = int(grid["width"]), int(grid["height"])
    pixel_x = (web_x - minimum_x) / (maximum_x - minimum_x) * max(width - 1, 1)
    pixel_y = (maximum_y - web_y) / (maximum_y - minimum_y) * max(height - 1, 1)
    return pixel_x, pixel_y


def _residual_displacement(
    transform: Mapping[str, Any], pixel_x: np.ndarray, pixel_y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    residual = transform["residual_warp"]
    centers = np.asarray(residual["centers_reference_px"], dtype=np.float64)
    coefficients = np.asarray(residual["coefficients_source_px"], dtype=np.float64)
    radius = float(residual["radius_reference_px"])
    points = np.column_stack((pixel_x.ravel(), pixel_y.ravel()))
    result = np.zeros((len(points), 2), dtype=np.float64)
    for start in range(0, len(points), 32768):
        stop = min(start + 32768, len(points))
        scaled = (
            np.linalg.norm(points[start:stop, None, :] - centers[None, :, :], axis=2)
            / radius
        )
        remaining = np.clip(1.0 - scaled, 0.0, 1.0)
        result[start:stop] = (remaining**4 * (4.0 * scaled + 1.0)) @ coefficients
    shape = pixel_x.shape
    return result[:, 0].reshape(shape), result[:, 1].reshape(shape)


def _projection_source_to_reference_base(
    transform: Mapping[str, Any], source_x: np.ndarray, source_y: np.ndarray,
    transformer: Transformer,
) -> tuple[np.ndarray, np.ndarray]:
    projection = transform["projection"]
    matrix = np.asarray(
        transform["source_original_to_candidate_normalized_matrix"], dtype=np.float64
    )
    normalized_x = source_x * matrix[0, 0] + source_y * matrix[0, 1] + matrix[0, 2]
    normalized_y = source_x * matrix[1, 0] + source_y * matrix[1, 1] + matrix[1, 2]
    center = np.asarray(projection["normalization_center"], dtype=np.float64)
    scale = float(projection["normalization_scale"])
    projected_x = normalized_x * scale + center[0]
    projected_y = -normalized_y * scale + center[1]
    web_x, web_y = transformer.transform(projected_x, projected_y, errcheck=True)
    return _web_mercator_to_reference(web_x, web_y, transform["target_grid"])


def source_to_reference_points(
    transform: Mapping[str, Any], source_x: np.ndarray, source_y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Map a bounded original-source crop to accepted target-grid coordinates."""

    source_x = np.asarray(source_x, dtype=np.float64)
    source_y = np.asarray(source_y, dtype=np.float64)
    kind = str(transform["kind"])
    if kind == "regular_global_mapbox_registration":
        matrix = np.asarray(
            transform["source_original_to_reference_pixel_matrix"], dtype=np.float64
        )
        denominator = matrix[2, 0] * source_x + matrix[2, 1] * source_y + matrix[2, 2]
        return (
            (matrix[0, 0] * source_x + matrix[0, 1] * source_y + matrix[0, 2])
            / denominator,
            (matrix[1, 0] * source_x + matrix[1, 1] * source_y + matrix[1, 2])
            / denominator,
        )
    projection = transform["projection"]
    transformer = Transformer.from_crs(
        CRS.from_wkt(projection["crs_wkt"]), "EPSG:3857", always_xy=True
    )
    mapped_x, mapped_y = _projection_source_to_reference_base(
        transform, source_x, source_y, transformer
    )
    if kind != "projection_aware_residual_warp_mapbox_registration":
        return mapped_x, mapped_y
    inverse = transform["inverse_solver"]
    for _ in range(int(inverse["maximum_iterations"])):
        residual_x, residual_y = _residual_displacement(transform, mapped_x, mapped_y)
        next_x, next_y = _projection_source_to_reference_base(
            transform, source_x - residual_x, source_y - residual_y, transformer
        )
        delta = np.maximum(np.abs(next_x - mapped_x), np.abs(next_y - mapped_y))
        mapped_x, mapped_y = next_x, next_y
        if float(np.max(delta)) <= float(inverse["reference_tolerance_px"]):
            return mapped_x, mapped_y
    raise ValueError("residual-warp source-to-reference crop did not converge")


def _accepted_paths(map_dir: Path) -> tuple[dict[str, Any], Path]:
    alignment = _read_json(map_dir / "automatic-alignment/accepted-alignment.json")
    pointers = sorted(map_dir.glob("*/accepted-extraction.json"))
    if len(pointers) != 1:
        raise ValueError(f"expected one accepted extraction pointer for {map_dir.name}")
    pointer = _read_json(pointers[0])
    iteration_dir = pointers[0].parent / str(pointer["accepted_iteration"])
    return alignment["transform"], iteration_dir


def _sample_target(target: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    values = target
    if values.dtype not in (np.uint8, np.uint16, np.float32):
        values = values.astype(np.float32)
    return cv2.remap(
        values,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _crop_mapping(
    transform: Mapping[str, Any], cell: Mapping[str, Any]
) -> tuple[tuple[int, int, int, int], np.ndarray, np.ndarray, np.ndarray]:
    source_height, source_width = map(int, transform["source_original_shape"])
    left, top, right, bottom = map(int, cell["source_original_pixel_bounds"])
    left, top = max(0, left - 3), max(0, top - 3)
    right, bottom = min(source_width, right + 3), min(source_height, bottom + 3)
    source_y, source_x = np.mgrid[top:bottom, left:right]
    map_x, map_y = source_to_reference_points(transform, source_x, source_y)
    target_left, target_top, target_right, target_bottom = map(
        float, cell["target_pixel_bounds"]
    )
    geographic = (
        (map_x >= target_left)
        & (map_x < target_right)
        & (map_y >= target_top)
        & (map_y < target_bottom)
    )
    return (left, top, right, bottom), map_x, map_y, geographic


def _binary_scores(source: np.ndarray, reconstructed: np.ndarray) -> dict[str, float]:
    source = source.astype(bool)
    reconstructed = reconstructed.astype(bool)
    union = int(np.count_nonzero(source | reconstructed))
    intersection = int(np.count_nonzero(source & reconstructed))
    kernel = np.ones((3, 3), dtype=np.uint8)
    source_near = cv2.dilate(source.astype(np.uint8), kernel).astype(bool)
    reconstructed_near = cv2.dilate(reconstructed.astype(np.uint8), kernel).astype(bool)
    precision = float(np.count_nonzero(reconstructed & source_near)) / max(
        int(np.count_nonzero(reconstructed)), 1
    )
    recall = float(np.count_nonzero(source & reconstructed_near)) / max(
        int(np.count_nonzero(source)), 1
    )
    return {
        "source_pixel_count": int(np.count_nonzero(source)),
        "reconstructed_pixel_count": int(np.count_nonzero(reconstructed)),
        "exact_iou": intersection / max(union, 1),
        "tolerant_precision": precision,
        "tolerant_recall": recall,
        "tolerant_f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
    }


def _evaluate_categorical_cell(
    inputs: Mapping[str, Any],
    bounds: tuple[int, int, int, int],
    map_x: np.ndarray,
    map_y: np.ndarray,
    geographic: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    left, top, right, bottom = bounds
    source_ids = inputs["source_ids"][top:bottom, left:right]
    reconstructed_ids = _sample_target(inputs["target_ids"], map_x, map_y)
    domain = geographic & (source_ids > 0)
    for exclusion in inputs["exclusions"]:
        domain &= exclusion[top:bottom, left:right] == 0
    source_diff = inputs["source_diff"][top:bottom, left:right] > 0
    roundtrip_diff = domain & (reconstructed_ids != source_ids)
    semantic_diff = domain & source_diff
    count = int(np.count_nonzero(domain))
    score = {
        "source_supported_pixel_count": count,
        "roundtrip_mismatch_pixel_count": int(np.count_nonzero(roundtrip_diff)),
        "roundtrip_mismatch_fraction": float(np.count_nonzero(roundtrip_diff))
        / max(count, 1),
        "semantic_mismatch_pixel_count": int(np.count_nonzero(semantic_diff)),
        "semantic_mismatch_fraction": float(np.count_nonzero(semantic_diff))
        / max(count, 1),
    }
    return score, roundtrip_diff | semantic_diff


def _render_panel(
    source_rgb: np.ndarray,
    target_rgb: np.ndarray,
    bounds: tuple[int, int, int, int],
    map_x: np.ndarray,
    map_y: np.ndarray,
    diff: np.ndarray,
    label: str,
) -> np.ndarray:
    left, top, right, bottom = bounds
    source = source_rgb[top:bottom, left:right].copy()
    reconstruction = _sample_target(target_rgb, map_x, map_y)
    overlay = cv2.addWeighted(source, 0.5, reconstruction, 0.5, 0.0)
    overlay[diff] = (255, 0, 255)
    canvas = Image.fromarray(np.concatenate((source, reconstruction, overlay), axis=1))
    draw = ImageDraw.Draw(canvas)
    panel_width = source.shape[1]
    for index, title in enumerate(("source", "target reconstructed native", label)):
        x = index * panel_width
        draw.rectangle((x, 0, x + min(panel_width, 300), 24), fill=(0, 0, 0))
        draw.text((x + 5, 5), title, fill=(255, 255, 255))
    return np.asarray(canvas)


def _source_rgb(map_dir: Path, transform: Mapping[str, Any]) -> np.ndarray:
    source = _load_rgb(map_dir / "source-clean/working-raster.png")
    expected = tuple(map(int, transform["source_original_shape"]))
    if source.shape[:2] != expected:
        raise ValueError(f"source-clean dimensions disagree for {map_dir.name}")
    return source


def _categorical_report(
    map_dir: Path,
    iteration_dir: Path,
    transform: Mapping[str, Any],
    config: Mapping[str, Any],
    output_dir: Path,
    worst_panel_count: int,
) -> dict[str, Any]:
    cells = plan_native_regional_diffs(transform, rows=GRID[0], columns=GRID[1])
    inputs = {
        "source_ids": _load_image(iteration_dir / config["source_id"]),
        "target_ids": _load_image(iteration_dir / config["target_id"]),
        "source_diff": _load_image(iteration_dir / config["source_diff"]),
        "exclusions": [
            _load_image(iteration_dir / name) for name in config.get("exclude", [])
        ],
    }
    reports: list[dict[str, Any]] = []
    for cell in cells:
        bounds, map_x, map_y, geographic = _crop_mapping(transform, cell)
        score, _ = _evaluate_categorical_cell(
            inputs, bounds, map_x, map_y, geographic
        )
        if score["source_supported_pixel_count"]:
            reports.append({**cell, **score})
    total = sum(item["source_supported_pixel_count"] for item in reports)
    roundtrip = sum(item["roundtrip_mismatch_pixel_count"] for item in reports)
    semantic = sum(item["semantic_mismatch_pixel_count"] for item in reports)
    global_roundtrip = roundtrip / max(total, 1)
    global_semantic = semantic / max(total, 1)
    worst_roundtrip = max((item["roundtrip_mismatch_fraction"] for item in reports), default=0)
    worst_semantic = max((item["semantic_mismatch_fraction"] for item in reports), default=0)
    gates = {
        "global_roundtrip_mismatch": global_roundtrip <= 0.002,
        "worst_cell_roundtrip_mismatch": worst_roundtrip <= 0.010,
        "global_semantic_mismatch": global_semantic
        <= float(config["global_semantic_max"]),
        "worst_cell_semantic_mismatch": worst_semantic
        <= float(config["cell_semantic_max"]),
        "all_supported_cells_measured": len(reports) > 0,
    }
    ranked = sorted(
        reports,
        key=lambda item: max(
            item["roundtrip_mismatch_fraction"], item["semantic_mismatch_fraction"]
        ),
        reverse=True,
    )[:worst_panel_count]
    source_rgb = _source_rgb(map_dir, transform)
    target_rgb = _load_rgb(iteration_dir / config["target_rgb"])
    panels = []
    for item in ranked:
        bounds, map_x, map_y, geographic = _crop_mapping(transform, item)
        _, diff = _evaluate_categorical_cell(
            inputs, bounds, map_x, map_y, geographic
        )
        path = output_dir / f"{item['id']}-native-diff.png"
        Image.fromarray(
            _render_panel(
                source_rgb,
                target_rgb,
                bounds,
                map_x,
                map_y,
                diff,
                "50/50; magenta mismatch",
            )
        ).save(path, optimize=True)
        panels.append(str(path))
    return {
        "source_family": config["family"],
        "scores": {
            "source_supported_pixel_count": total,
            "roundtrip_mismatch_fraction": global_roundtrip,
            "semantic_mismatch_fraction": global_semantic,
            "worst_cell_roundtrip_mismatch_fraction": worst_roundtrip,
            "worst_cell_semantic_mismatch_fraction": worst_semantic,
        },
        "thresholds": {
            "global_roundtrip_mismatch_maximum": 0.002,
            "worst_cell_roundtrip_mismatch_maximum": 0.010,
            "global_semantic_mismatch_maximum": config["global_semantic_max"],
            "worst_cell_semantic_mismatch_maximum": config["cell_semantic_max"],
        },
        "gates": gates,
        "decision": "pass" if all(gates.values()) else "retry",
        "cells": reports,
        "worst_crop_panels": panels,
    }


def _prepare_multi_inputs(
    iteration_dir: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    inputs: dict[str, Any] = {"classes": [], "binary": []}
    for channel in config.get("classes", []):
        inputs["classes"].append(
            {
                **channel,
                "source_values": _load_image(iteration_dir / channel["source"]),
                "target_values": _load_image(iteration_dir / channel["target"]),
                "exclusions": [
                    _load_image(iteration_dir / name)
                    for name in channel.get("exclude", [])
                ],
            }
        )
    for channel in config.get("binary", []):
        inputs["binary"].append(
            {
                **channel,
                "source_values": _load_image(iteration_dir / channel["source"]) > 0,
                "target_values": _load_image(iteration_dir / channel["target"]) > 0,
            }
        )
    inputs["source_diff"] = [
        _load_image(iteration_dir / name) > 0 for name in config.get("source_diff", [])
    ]
    return inputs


def _evaluate_multi_cell(
    inputs: Mapping[str, Any],
    bounds: tuple[int, int, int, int],
    map_x: np.ndarray,
    map_y: np.ndarray,
    geographic: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    left, top, right, bottom = bounds
    diff = np.zeros(geographic.shape, dtype=bool)
    support = np.zeros_like(diff)
    class_scores: dict[str, Any] = {}
    for channel in inputs["classes"]:
        source = channel["source_values"][top:bottom, left:right]
        reconstructed = _sample_target(channel["target_values"], map_x, map_y)
        domain = geographic & (source > 0)
        for exclusion in channel["exclusions"]:
            domain &= exclusion[top:bottom, left:right] == 0
        mismatch = domain & (source != reconstructed)
        count = int(np.count_nonzero(domain))
        class_scores[channel["name"]] = {
            "source_supported_pixel_count": count,
            "mismatch_pixel_count": int(np.count_nonzero(mismatch)),
            "mismatch_fraction": float(np.count_nonzero(mismatch)) / max(count, 1),
        }
        support |= domain
        diff |= mismatch
    binary_scores: dict[str, Any] = {}
    for channel in inputs["binary"]:
        source = channel["source_values"][top:bottom, left:right] & geographic
        reconstructed = (
            _sample_target(channel["target_values"].astype(np.uint8), map_x, map_y) > 0
        ) & geographic
        if not np.any(source | reconstructed):
            continue
        binary_scores[channel["name"]] = _binary_scores(source, reconstructed)
        support |= source | reconstructed
        diff |= source ^ reconstructed
    source_family_diff = np.zeros_like(diff)
    for mask in inputs["source_diff"]:
        source_family_diff |= mask[top:bottom, left:right]
    family_domain = geographic & support
    family_diff = family_domain & source_family_diff
    diff |= family_diff
    family_count = int(np.count_nonzero(family_domain))
    return (
        {
            "class_channels": class_scores,
            "binary_channels": binary_scores,
            "source_family_supported_pixel_count": family_count,
            "source_family_diff_pixel_count": int(np.count_nonzero(family_diff)),
            "source_family_diff_fraction": float(np.count_nonzero(family_diff))
            / max(family_count, 1),
        },
        diff,
    )


def _synthesized_linear_target_rgb(
    inputs: Mapping[str, Any], target_shape: tuple[int, int]
) -> np.ndarray:
    output = np.full((*target_shape, 3), 245, dtype=np.uint8)
    palette = [
        (30, 90, 210),
        (0, 130, 210),
        (120, 210, 245),
        (170, 100, 45),
        (190, 120, 55),
        (235, 205, 160),
        (110, 65, 190),
    ]
    for channel, color in zip(inputs["binary"], palette):
        output[channel["target_values"]] = color
    return output


def _multi_report(
    map_dir: Path,
    iteration_dir: Path,
    transform: Mapping[str, Any],
    config: Mapping[str, Any],
    output_dir: Path,
    worst_panel_count: int,
) -> dict[str, Any]:
    inputs = _prepare_multi_inputs(iteration_dir, config)
    cells = plan_native_regional_diffs(transform, rows=GRID[0], columns=GRID[1])
    reports: list[dict[str, Any]] = []
    for cell in cells:
        bounds, map_x, map_y, geographic = _crop_mapping(transform, cell)
        score, _ = _evaluate_multi_cell(inputs, bounds, map_x, map_y, geographic)
        if score["source_family_supported_pixel_count"]:
            reports.append({**cell, **score})

    class_global: dict[str, Any] = {}
    for channel in inputs["classes"]:
        name = channel["name"]
        supported = sum(
            item["class_channels"].get(name, {}).get("source_supported_pixel_count", 0)
            for item in reports
        )
        mismatch = sum(
            item["class_channels"].get(name, {}).get("mismatch_pixel_count", 0)
            for item in reports
        )
        worst = max(
            (
                item["class_channels"].get(name, {}).get("mismatch_fraction", 0.0)
                for item in reports
            ),
            default=0.0,
        )
        class_global[name] = {
            "source_supported_pixel_count": supported,
            "mismatch_pixel_count": mismatch,
            "mismatch_fraction": mismatch / max(supported, 1),
            "worst_cell_mismatch_fraction": worst,
        }

    binary_global: dict[str, Any] = {}
    for channel in inputs["binary"]:
        name = channel["name"]
        scores = [
            item["binary_channels"][name]
            for item in reports
            if name in item["binary_channels"]
        ]
        weight = sum(max(item["source_pixel_count"], 1) for item in scores)
        optional_empty = bool(channel.get("optional_empty"))
        weighted_f1 = (
            1.0
            if not scores and optional_empty
            else sum(
                item["tolerant_f1"] * max(item["source_pixel_count"], 1)
                for item in scores
            )
            / max(weight, 1)
        )
        binary_global[name] = {
            "supported_cell_count": len(scores),
            "optional_empty": optional_empty,
            "weighted_tolerant_f1": weighted_f1,
            "worst_cell_tolerant_f1": min(
                (item["tolerant_f1"] for item in scores), default=1.0
            ),
        }

    family_count = sum(item["source_family_supported_pixel_count"] for item in reports)
    family_diff = sum(item["source_family_diff_pixel_count"] for item in reports)
    family_fraction = family_diff / max(family_count, 1)
    worst_family = max(
        (item["source_family_diff_fraction"] for item in reports), default=0.0
    )
    gates: dict[str, bool] = {
        "source_family_global_diff": family_fraction
        <= float(config["global_diff_max"]),
        "source_family_worst_cell_diff": worst_family
        <= float(config["cell_diff_max"]),
        "all_supported_cells_measured": bool(reports),
    }
    for name, score in class_global.items():
        gates[f"{name}_global_roundtrip"] = score["mismatch_fraction"] <= 0.002
        gates[f"{name}_worst_cell_roundtrip"] = (
            score["worst_cell_mismatch_fraction"] <= 0.010
        )
    for channel in inputs["binary"]:
        score = binary_global[channel["name"]]
        gates[f"{channel['name']}_global_tolerant_f1"] = (
            score["weighted_tolerant_f1"] >= float(channel["global_f1_min"])
        )
        gates[f"{channel['name']}_worst_cell_tolerant_f1"] = (
            score["worst_cell_tolerant_f1"] >= float(channel["cell_f1_min"])
        )

    def severity(item: Mapping[str, Any]) -> float:
        values = [
            item["source_family_diff_fraction"] / max(float(config["cell_diff_max"]), 1e-9)
        ]
        values.extend(
            score["mismatch_fraction"] / 0.010
            for score in item["class_channels"].values()
        )
        values.extend(1.0 - score["tolerant_f1"] for score in item["binary_channels"].values())
        return max(values, default=0.0)

    ranked = sorted(reports, key=severity, reverse=True)[:worst_panel_count]
    source_rgb = _source_rgb(map_dir, transform)
    if "target_rgb" in config:
        target_rgb = _load_rgb(iteration_dir / config["target_rgb"])
    else:
        grid = transform["target_grid"]
        target_rgb = _synthesized_linear_target_rgb(
            inputs, (int(grid["height"]), int(grid["width"]))
        )
    panels = []
    for item in ranked:
        bounds, map_x, map_y, geographic = _crop_mapping(transform, item)
        _, diff = _evaluate_multi_cell(inputs, bounds, map_x, map_y, geographic)
        path = output_dir / f"{item['id']}-native-diff.png"
        Image.fromarray(
            _render_panel(
                source_rgb,
                target_rgb,
                bounds,
                map_x,
                map_y,
                diff,
                "50/50; magenta channel diff",
            )
        ).save(path, optimize=True)
        panels.append(str(path))
    return {
        "source_family": config["family"],
        "scores": {
            "class_channels": class_global,
            "binary_channels": binary_global,
            "source_family_diff_fraction": family_fraction,
            "worst_cell_source_family_diff_fraction": worst_family,
        },
        "thresholds": {
            "class_global_mismatch_maximum": 0.002,
            "class_worst_cell_mismatch_maximum": 0.010,
            "source_family_global_diff_maximum": config["global_diff_max"],
            "source_family_worst_cell_diff_maximum": config["cell_diff_max"],
            "binary_channels": {
                item["name"]: {
                    "global_tolerant_f1_minimum": item["global_f1_min"],
                    "worst_cell_tolerant_f1_minimum": item["cell_f1_min"],
                }
                for item in inputs["binary"]
            },
        },
        "gates": gates,
        "decision": "pass" if all(gates.values()) else "retry",
        "cells": reports,
        "worst_crop_panels": panels,
    }


def evaluate_map(
    run_root: Path, map_id: str, output_root: Path, *, worst_panel_count: int = 3
) -> dict[str, Any]:
    """Evaluate one accepted map without altering its official artifacts."""

    if map_id not in _CONFIGS:
        raise ValueError(f"unsupported map id: {map_id}")
    if not 1 <= worst_panel_count <= 5:
        raise ValueError("worst panel count must be between one and five")
    map_dir = run_root.resolve() / map_id
    transform, iteration_dir = _accepted_paths(map_dir)
    output_dir = output_root.resolve() / map_id
    output_dir.mkdir(parents=True, exist_ok=True)
    config = _CONFIGS[map_id]
    if config["family"] in {"categorical", "continuous"}:
        evaluation = _categorical_report(
            map_dir,
            iteration_dir,
            transform,
            config,
            output_dir,
            worst_panel_count,
        )
    else:
        evaluation = _multi_report(
            map_dir,
            iteration_dir,
            transform,
            config,
            output_dir,
            worst_panel_count,
        )
    report = {
        "schema_version": SCHEMA_VERSION,
        "map_id": map_id,
        "accepted_alignment_path": str(
            map_dir / "automatic-alignment/accepted-alignment.json"
        ),
        "accepted_extraction_iteration_dir": str(iteration_dir),
        "grid": list(GRID),
        "comparison_space": "original_source_pixels",
        "known_exclusions": [
            "zero/non-data source class",
            "source layout and legend outside the supported thematic domain",
            "accepted source-family occlusion masks",
            "Mapbox exterior/water already excluded by accepted target extraction",
        ],
        "accepted_artifacts_mutated": False,
        **evaluation,
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("map_ids", nargs="+")
    parser.add_argument("--worst-panels", type=int, default=3)
    args = parser.parse_args(argv)
    results = [
        evaluate_map(
            args.run_root, map_id, args.output_root, worst_panel_count=args.worst_panels
        )
        for map_id in args.map_ids
    ]
    (args.output_root / "batch-report.json").write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "maps": [
                    {
                        "map_id": item["map_id"],
                        "decision": item["decision"],
                        "scores": item["scores"],
                        "report": str(args.output_root / item["map_id"] / "report.json"),
                    }
                    for item in results
                ],
            },
            indent=2,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
