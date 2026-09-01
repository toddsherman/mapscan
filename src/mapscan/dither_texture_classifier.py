"""Deterministic local-texture classification for dithered map legends.

Some indexed or scanned maps encode a semantic class as a repeatable mixture of
colors rather than one RGB value.  A median-color palette silently merges such
classes.  This module models the local Lab mean and standard deviation of each
legend swatch and refuses a model when two rows are not distinguishable in the
source raster itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class TextureClassSignature:
    class_id: int
    swatch_box: tuple[int, int, int, int]
    center: tuple[float, ...]
    robust_radius: float
    top_rgb_distribution: tuple[tuple[tuple[int, int, int], float], ...]


@dataclass(frozen=True)
class DitherTextureModel:
    window_size: int
    standard_deviation_weight: float
    signatures: tuple[TextureClassSignature, ...]
    pairwise_distances: tuple[tuple[float, ...], ...]
    ambiguous_pairs: tuple[tuple[int, int, float], ...]
    minimum_center_distance: float

    @property
    def is_distinguishable(self) -> bool:
        return not self.ambiguous_pairs

    def diagnostics(self) -> dict[str, Any]:
        return {
            "method": "local-lab-mean-and-standard-deviation-texture-signatures",
            "window_size": self.window_size,
            "standard_deviation_weight": self.standard_deviation_weight,
            "minimum_center_distance": self.minimum_center_distance,
            "minimum_observed_pairwise_distance": min(
                (
                    value
                    for row_index, row in enumerate(self.pairwise_distances)
                    for column_index, value in enumerate(row)
                    if column_index > row_index
                ),
                default=None,
            ),
            "ambiguous_pairs": [
                {
                    "first_class_id": first,
                    "second_class_id": second,
                    "center_distance": distance,
                }
                for first, second, distance in self.ambiguous_pairs
            ],
            "classes": [
                {
                    "class_id": item.class_id,
                    "swatch_box": list(item.swatch_box),
                    "feature_center": list(item.center),
                    "robust_radius": item.robust_radius,
                    "top_rgb_distribution": [
                        {"rgb": list(rgb), "fraction": fraction}
                        for rgb, fraction in item.top_rgb_distribution
                    ],
                }
                for item in self.signatures
            ],
        }


def local_texture_features(
    rgb: np.ndarray,
    *,
    window_size: int = 5,
    standard_deviation_weight: float = 0.5,
) -> np.ndarray:
    """Return translation-tolerant Lab distribution moments per source pixel."""

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("texture features require an RGB image")
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError("texture window size must be an odd integer of at least three")
    lab = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    mean = cv2.boxFilter(
        lab,
        -1,
        (window_size, window_size),
        normalize=True,
        borderType=cv2.BORDER_REFLECT,
    )
    squared_mean = cv2.boxFilter(
        lab * lab,
        -1,
        (window_size, window_size),
        normalize=True,
        borderType=cv2.BORDER_REFLECT,
    )
    deviation = np.sqrt(np.maximum(squared_mean - mean * mean, 0.0))
    return np.concatenate((mean, deviation * standard_deviation_weight), axis=2)


def _top_rgb_distribution(
    patch: np.ndarray, maximum_colors: int = 16
) -> tuple[tuple[tuple[int, int, int], float], ...]:
    colors, counts = np.unique(patch.reshape(-1, 3), axis=0, return_counts=True)
    order = sorted(
        range(len(colors)),
        key=lambda index: (-int(counts[index]), tuple(int(value) for value in colors[index])),
    )[:maximum_colors]
    total = max(int(np.sum(counts)), 1)
    return tuple(
        (
            tuple(int(value) for value in colors[index]),
            float(counts[index]) / total,
        )
        for index in order
    )


def build_dither_texture_model(
    rgb: np.ndarray,
    swatch_boxes: Sequence[tuple[int, int, int, int]],
    *,
    window_size: int = 5,
    standard_deviation_weight: float = 0.5,
    minimum_center_distance: float = 1.0,
) -> DitherTextureModel:
    """Build one signature per legend row and audit pairwise separability.

    The distance floor is deliberately absolute.  A sub-unit separation in
    8-bit Lab texture space is too small to distinguish reliably after a map is
    warped or resampled, even if the original swatch arrays are not byte-equal.
    """

    if len(swatch_boxes) < 2:
        raise ValueError("a dither texture model requires at least two swatches")
    features = local_texture_features(
        rgb,
        window_size=window_size,
        standard_deviation_weight=standard_deviation_weight,
    )
    radius = window_size // 2
    signatures: list[TextureClassSignature] = []
    for class_id, box in enumerate(swatch_boxes, 1):
        x, y, width, height = map(int, box)
        if width <= 2 * radius or height <= 2 * radius:
            raise ValueError("legend swatch is too small for texture analysis")
        patch_features = features[
            y + radius : y + height - radius,
            x + radius : x + width - radius,
        ].reshape(-1, 6)
        patch_rgb = rgb[y : y + height, x : x + width]
        if not len(patch_features) or not patch_rgb.size:
            raise ValueError("legend swatch texture region is empty")
        center = np.median(patch_features, axis=0)
        distances = np.linalg.norm(patch_features - center[None, :], axis=1)
        median_distance = float(np.median(distances))
        median_absolute_deviation = float(
            np.median(np.abs(distances - median_distance))
        )
        signatures.append(
            TextureClassSignature(
                class_id=class_id,
                swatch_box=(x, y, width, height),
                center=tuple(float(value) for value in center),
                robust_radius=median_distance + 3.0 * median_absolute_deviation,
                top_rgb_distribution=_top_rgb_distribution(patch_rgb),
            )
        )

    centers = np.asarray([item.center for item in signatures], dtype=np.float32)
    pairwise = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    ambiguous: list[tuple[int, int, float]] = []
    for first in range(len(signatures)):
        for second in range(first + 1, len(signatures)):
            distance = float(pairwise[first, second])
            if distance < minimum_center_distance:
                ambiguous.append((first + 1, second + 1, distance))
    return DitherTextureModel(
        window_size=window_size,
        standard_deviation_weight=standard_deviation_weight,
        signatures=tuple(signatures),
        pairwise_distances=tuple(tuple(float(value) for value in row) for row in pairwise),
        ambiguous_pairs=tuple(ambiguous),
        minimum_center_distance=minimum_center_distance,
    )


def classify_dither_texture(
    rgb: np.ndarray,
    domain: np.ndarray,
    model: DitherTextureModel,
    maximum_distance: float,
    minimum_margin: float,
    *,
    chunk_size: int = 100_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Classify pixels by their local swatch-distribution signature."""

    if not model.is_distinguishable:
        raise ValueError("dither texture signatures contain ambiguous legend rows")
    if domain.shape != rgb.shape[:2]:
        raise ValueError("texture classification domain shape differs from the image")
    features = local_texture_features(
        rgb,
        window_size=model.window_size,
        standard_deviation_weight=model.standard_deviation_weight,
    ).reshape(-1, 6)
    centers = np.asarray([item.center for item in model.signatures], dtype=np.float32)
    indices = np.flatnonzero(domain.ravel())
    class_ids = np.zeros(domain.shape, dtype=np.uint8)
    nearest_ids = np.zeros(domain.shape, dtype=np.uint8)
    nearest_distances = np.full(domain.shape, np.inf, dtype=np.float32)
    flat_class = class_ids.ravel()
    flat_nearest = nearest_ids.ravel()
    flat_distance = nearest_distances.ravel()
    for start in range(0, len(indices), chunk_size):
        selected = indices[start : start + chunk_size]
        distances = np.linalg.norm(
            features[selected, None, :] - centers[None, :, :], axis=2
        )
        best = np.argmin(distances, axis=1)
        best_distance = distances[np.arange(len(selected)), best]
        second = np.partition(distances, 1, axis=1)[:, 1]
        accepted = (best_distance <= maximum_distance) & (
            second - best_distance >= minimum_margin
        )
        flat_nearest[selected] = best + 1
        flat_distance[selected] = best_distance
        flat_class[selected[accepted]] = best[accepted] + 1
    return class_ids, nearest_ids, nearest_distances
