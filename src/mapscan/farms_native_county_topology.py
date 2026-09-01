"""Recover thin farms county ink before alignment-scale downsampling.

``farmsv2.png`` is a 4250x5500 cartographic raster, while alignment runs on a
695x900 working canvas.  The printed county boundaries contain a narrow,
flat dark-neutral core which can disappear when RGB is area-reduced first.
This module identifies that core on the pristine source and only then performs
an any-positive binary reduction.  The API is deliberately source-only: it
accepts source-derived scope/state masks and has no reference-map, transform,
validation, or retained-acceptance inputs.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping

import cv2
import numpy as np


@dataclass(frozen=True)
class FarmsNativeCountyTopology:
    """Native-resolution county ink and its working-canvas reduction."""

    native_ink: np.ndarray
    working_ink: np.ndarray
    diagnostics: Mapping[str, Any]


def _mask_sha256(mask: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(mask.astype(np.uint8)).tobytes()
    ).hexdigest()


def _resize_mask_nearest(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    return cv2.resize(
        mask.astype(np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)


def _binary_any_reduce(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Area-reduce a mask while preserving every positive native pixel."""

    height, width = shape
    return cv2.resize(
        mask.astype(np.float32),
        (width, height),
        interpolation=cv2.INTER_AREA,
    ) > 0


def derive_farms_native_county_topology(
    rgb_original: np.ndarray,
    *,
    county_scope_working: np.ndarray,
    state_coast_working: np.ndarray,
) -> FarmsNativeCountyTopology:
    """Recover long, flat dark-neutral county strokes at native resolution.

    The ink color is selected from the source itself.  Black is excluded to
    avoid the inset frame and text; the dominant neutral tone in the remaining
    dark interval is the printed administrative-line core.  Long-component
    filtering rejects isolated relief shadows and glyph fragments.  Finally,
    source-derived county scope and state/coast corridors remove layout,
    neighboring geography, and the state perimeter.
    """

    if rgb_original.ndim != 3 or rgb_original.shape[2] != 3:
        raise ValueError("rgb_original must have shape (height, width, 3)")
    if county_scope_working.shape != state_coast_working.shape:
        raise ValueError("Working scope and state/coast masks must share a shape")
    if county_scope_working.ndim != 2:
        raise ValueError("Working masks must be two-dimensional")

    working_shape = county_scope_working.shape
    native_shape = rgb_original.shape[:2]
    scope_native = _resize_mask_nearest(county_scope_working, native_shape)

    gray = cv2.cvtColor(rgb_original, cv2.COLOR_RGB2GRAY)
    rgb16 = rgb_original.astype(np.int16)
    chroma = rgb16.max(axis=2) - rgb16.min(axis=2)

    # The printed frame/text is pure black.  The administrative network uses a
    # distinct dark-neutral tone above black; find its mode rather than
    # hard-coding the farms-v2 palette value.
    anchor_domain = (
        scope_native
        & (gray >= 24)
        & (gray <= 110)
        & (chroma <= 4)
    )
    values = gray[anchor_domain]
    minimum_anchor_support = max(
        50, round(np.count_nonzero(scope_native) * 0.00001)
    )
    if values.size < minimum_anchor_support:
        raise ValueError("Native source lacks a supported dark-neutral ink mode")
    histogram = np.bincount(values, minlength=256)
    anchor_gray = int(np.argmax(histogram[24:111]) + 24)
    anchor_count = int(histogram[anchor_gray])
    if anchor_count < minimum_anchor_support:
        raise ValueError("Native dark-neutral ink mode is undersupported")

    ink_candidate = (
        scope_native
        & (np.abs(gray.astype(np.int16) - anchor_gray) <= 2)
        & (chroma <= 4)
    )
    # Bridge only sub-stroke antialiasing gaps at native resolution.  This is
    # intentionally much smaller than an alignment working pixel (~6 native
    # pixels for farms-v2), so it cannot invent missing geographic spans.
    closed = cv2.morphologyEx(
        ink_candidate.astype(np.uint8),
        cv2.MORPH_CLOSE,
        np.ones((3, 3), np.uint8),
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(closed, 8)
    minimum_extent = max(24, round(min(native_shape) * 0.020))
    native_ink = np.zeros(native_shape, dtype=bool)
    retained_components: list[dict[str, int]] = []
    for label in range(1, count):
        x, y, width, height, area = map(int, stats[label])
        if max(width, height) < minimum_extent or area < 24:
            continue
        native_ink |= labels == label
        retained_components.append(
            {
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                "pixel_count": area,
            }
        )
    if not retained_components:
        raise ValueError("Native dark-neutral topology lacks long components")

    working_ink = _binary_any_reduce(native_ink, working_shape)
    state_corridor = cv2.dilate(
        state_coast_working.astype(np.uint8),
        np.ones((7, 7), np.uint8),
    ).astype(bool)
    working_ink &= county_scope_working.astype(bool)
    working_ink &= ~state_corridor

    diagnostics = {
        "method": "native_resolution_source_dark_neutral_mode_long_components_then_any_positive_reduction",
        "native_source_shape": list(native_shape),
        "working_shape": list(working_shape),
        "anchor_gray": anchor_gray,
        "anchor_rgb": [anchor_gray, anchor_gray, anchor_gray],
        "anchor_mode_pixel_count": anchor_count,
        "minimum_anchor_mode_support": minimum_anchor_support,
        "anchor_gray_tolerance": 2,
        "anchor_chroma_limit": 4,
        "black_frame_and_text_excluded_below_gray": 24,
        "minimum_native_component_extent_px": minimum_extent,
        "candidate_native_pixel_count": int(np.count_nonzero(ink_candidate)),
        "retained_native_pixel_count": int(np.count_nonzero(native_ink)),
        "retained_component_count": len(retained_components),
        "retained_components": retained_components,
        "working_pixel_count": int(np.count_nonzero(working_ink)),
        "native_ink_sha256": _mask_sha256(native_ink),
        "working_ink_sha256": _mask_sha256(working_ink),
        "binary_reduction": "float32_cv2_inter_area_then_any_positive",
        "mapbox_or_reference_geometry_used": False,
        "candidate_transform_used": False,
        "validation_or_retained_acceptance_inputs_supported": False,
    }
    return FarmsNativeCountyTopology(
        native_ink=native_ink,
        working_ink=working_ink,
        diagnostics=diagnostics,
    )


def compare_native_and_working_topology(
    native_working_ink: np.ndarray,
    working_rgb_topology: np.ndarray,
    *,
    corridor_px: int = 2,
) -> dict[str, Any]:
    """Quantify source-only agreement without declaring either mask truth."""

    if native_working_ink.shape != working_rgb_topology.shape:
        raise ValueError("Topology masks must share a working canvas")
    if corridor_px < 0:
        raise ValueError("corridor_px must be nonnegative")
    size = 2 * corridor_px + 1
    kernel = np.ones((size, size), np.uint8)
    native = native_working_ink.astype(bool)
    working = working_rgb_topology.astype(bool)
    native_corridor = cv2.dilate(native.astype(np.uint8), kernel).astype(bool)
    working_corridor = cv2.dilate(working.astype(np.uint8), kernel).astype(bool)
    native_supported_working = working & native_corridor
    working_supported_native = native & working_corridor
    return {
        "corridor_px": corridor_px,
        "native_working_pixel_count": int(np.count_nonzero(native)),
        "working_rgb_topology_pixel_count": int(np.count_nonzero(working)),
        "working_rgb_pixels_supported_by_native_count": int(
            np.count_nonzero(native_supported_working)
        ),
        "working_rgb_pixels_unsupported_by_native_count": int(
            np.count_nonzero(working & ~native_corridor)
        ),
        "native_pixels_supported_by_working_rgb_count": int(
            np.count_nonzero(working_supported_native)
        ),
        "native_pixels_missing_from_working_rgb_count": int(
            np.count_nonzero(native & ~working_corridor)
        ),
        "working_rgb_supported_fraction": (
            float(np.count_nonzero(native_supported_working) / np.count_nonzero(working))
            if np.any(working)
            else None
        ),
        "native_supported_fraction": (
            float(np.count_nonzero(working_supported_native) / np.count_nonzero(native))
            if np.any(native)
            else None
        ),
        "interpretation": (
            "unsupported counts are source-only disagreement diagnostics, not "
            "validation labels or ground-truth error classifications"
        ),
    }
