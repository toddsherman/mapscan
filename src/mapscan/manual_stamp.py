"""Auditable clone-stamp corrections for missing categorical map pixels."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


def validate_stamp_operations(
    operations: Iterable[Dict[str, object]],
    width: int,
    height: int,
    layer_ids: Sequence[str],
    maximum_operations: int = 10_000,
    maximum_radius_px: int = 250,
) -> List[Dict[str, object]]:
    """Validate and normalize browser-recorded source-to-target stamp operations."""

    items = list(operations)
    if len(items) > maximum_operations:
        raise ValueError(f"At most {maximum_operations} stamp operations may be saved")
    allowed_layers = set(layer_ids)
    normalized: List[Dict[str, object]] = []
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Stamp operation {index} must be an object")
        layer_id = str(item.get("layer_id", ""))
        if layer_id not in allowed_layers:
            raise ValueError(f"Stamp operation {index} has an unknown layer")
        source = item.get("source")
        target = item.get("target")
        if (
            not isinstance(source, list)
            or not isinstance(target, list)
            or len(source) != 2
            or len(target) != 2
        ):
            raise ValueError(f"Stamp operation {index} needs source and target points")
        try:
            source_x, source_y = float(source[0]), float(source[1])
            target_x, target_y = float(target[0]), float(target[1])
            radius = float(item.get("radius_px", 0))
        except (TypeError, ValueError):
            raise ValueError(f"Stamp operation {index} contains non-numeric values")
        if not all(
            math.isfinite(value)
            for value in (source_x, source_y, target_x, target_y, radius)
        ):
            raise ValueError(f"Stamp operation {index} contains non-finite values")
        if not (0 <= source_x < width and 0 <= target_x < width):
            raise ValueError(f"Stamp operation {index} x coordinate is outside the raster")
        if not (0 <= source_y < height and 0 <= target_y < height):
            raise ValueError(f"Stamp operation {index} y coordinate is outside the raster")
        if not 1 <= radius <= maximum_radius_px:
            raise ValueError(
                f"Stamp operation {index} radius must be from 1 to {maximum_radius_px} pixels"
            )
        source_mode = str(item.get("source_mode", "observed"))
        if source_mode not in {"observed", "composite_at_operation_time"}:
            raise ValueError(
                f"Stamp operation {index} has an unsupported source mode"
            )
        normalized.append(
            {
                "layer_id": layer_id,
                "source": [round(source_x, 3), round(source_y, 3)],
                "target": [round(target_x, 3), round(target_y, 3)],
                "radius_px": round(radius, 3),
                "source_mode": source_mode,
            }
        )
    return normalized


def apply_clone_stamp_operations(
    observed: np.ndarray,
    operations: Iterable[Dict[str, object]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Replay solid clone patches without mutating the observed raster."""

    if observed.ndim != 2:
        raise ValueError("Clone stamps require a two-dimensional class-ID raster")
    manual = np.zeros(observed.shape, dtype=np.uint8)
    mask = np.zeros(observed.shape, dtype=bool)
    height, width = observed.shape
    for operation in operations:
        source_x, source_y = (int(round(float(value))) for value in operation["source"])
        target_x, target_y = (int(round(float(value))) for value in operation["target"])
        radius = max(1, int(round(float(operation["radius_px"]))))
        source_mode = str(operation.get("source_mode", "observed"))
        if source_mode not in {"observed", "composite_at_operation_time"}:
            raise ValueError("Clone stamp has an unsupported source mode")
        diameter = radius * 2 + 1
        source_patch = np.zeros((diameter, diameter), dtype=np.uint8)
        source_valid = np.zeros((diameter, diameter), dtype=bool)
        for offset_y in range(-radius, radius + 1):
            for offset_x in range(-radius, radius + 1):
                if offset_x * offset_x + offset_y * offset_y > radius * radius:
                    continue
                sample_x = source_x + offset_x
                sample_y = source_y + offset_y
                if not (0 <= sample_x < width and 0 <= sample_y < height):
                    continue
                patch_y = offset_y + radius
                patch_x = offset_x + radius
                class_id = int(observed[sample_y, sample_x])
                if source_mode == "composite_at_operation_time" and mask[
                    sample_y, sample_x
                ]:
                    class_id = int(manual[sample_y, sample_x])
                source_patch[patch_y, patch_x] = class_id
                source_valid[patch_y, patch_x] = True
        for offset_y in range(-radius, radius + 1):
            for offset_x in range(-radius, radius + 1):
                patch_y = offset_y + radius
                patch_x = offset_x + radius
                if not source_valid[patch_y, patch_x]:
                    continue
                output_x = target_x + offset_x
                output_y = target_y + offset_y
                if not (0 <= output_x < width and 0 <= output_y < height):
                    continue
                class_id = int(source_patch[patch_y, patch_x])
                manual[output_y, output_x] = class_id
                mask[output_y, output_x] = True
    return manual, mask


def validate_inference_exclusions(
    operations: Iterable[Dict[str, object]],
    width: int,
    height: int,
    layer_ids: Sequence[str],
    maximum_operations: int = 10_000,
    maximum_radius_px: int = 250,
) -> List[Dict[str, object]]:
    """Validate circular author rejections of automatic inference pixels."""

    items = list(operations)
    if len(items) > maximum_operations:
        raise ValueError(f"At most {maximum_operations} exclusion operations may be saved")
    allowed_layers = set(layer_ids)
    normalized: List[Dict[str, object]] = []
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Exclusion operation {index} must be an object")
        layer_id = str(item.get("layer_id", ""))
        center = item.get("center")
        if layer_id not in allowed_layers:
            raise ValueError(f"Exclusion operation {index} has an unknown layer")
        if not isinstance(center, list) or len(center) != 2:
            raise ValueError(f"Exclusion operation {index} needs a center point")
        try:
            center_x, center_y = float(center[0]), float(center[1])
            radius = float(item.get("radius_px", 0))
        except (TypeError, ValueError):
            raise ValueError(f"Exclusion operation {index} contains non-numeric values")
        if not all(math.isfinite(value) for value in (center_x, center_y, radius)):
            raise ValueError(f"Exclusion operation {index} contains non-finite values")
        if not (0 <= center_x < width and 0 <= center_y < height):
            raise ValueError(f"Exclusion operation {index} center is outside the raster")
        if not 1 <= radius <= maximum_radius_px:
            raise ValueError(
                f"Exclusion operation {index} radius must be from 1 to "
                f"{maximum_radius_px} pixels"
            )
        normalized.append(
            {
                "layer_id": layer_id,
                "center": [round(center_x, 3), round(center_y, 3)],
                "radius_px": round(radius, 3),
            }
        )
    return normalized


def apply_inference_exclusions(
    inference_mask: np.ndarray,
    operations: Iterable[Dict[str, object]],
) -> np.ndarray:
    """Replay circular rejections, clipping them to inferred pixels only."""

    if inference_mask.ndim != 2:
        raise ValueError("Inference exclusions require a two-dimensional mask")
    excluded = np.zeros(inference_mask.shape, dtype=bool)
    height, width = inference_mask.shape
    for operation in operations:
        center_x, center_y = (
            int(round(float(value))) for value in operation["center"]
        )
        radius = max(1, int(round(float(operation["radius_px"]))))
        for offset_y in range(-radius, radius + 1):
            for offset_x in range(-radius, radius + 1):
                if offset_x * offset_x + offset_y * offset_y > radius * radius:
                    continue
                x = center_x + offset_x
                y = center_y + offset_y
                if 0 <= x < width and 0 <= y < height and inference_mask[y, x]:
                    excluded[y, x] = True
    return excluded
