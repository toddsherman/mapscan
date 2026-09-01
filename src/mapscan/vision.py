"""Image preparation and edge evidence for geographic registration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt


@dataclass(frozen=True)
class PreparedImage:
    rgb: np.ndarray
    gray: np.ndarray
    edges: np.ndarray
    distance: np.ndarray
    nearest_edge_y: np.ndarray
    nearest_edge_x: np.ndarray
    gradient_x: np.ndarray
    gradient_y: np.ndarray
    color_evidence: np.ndarray
    eastern_straight_edges: np.ndarray
    eastern_straight_distance: np.ndarray
    eastern_border_hinge: tuple[float, float] | None
    scale_from_original: float
    original_width: int
    original_height: int


def _connected_eastern_border_evidence(
    lines: np.ndarray | None,
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, tuple[float, float] | None]:
    """Select the connected vertical/diagonal California-Nevada line pair.

    On north-up California maps the Nevada border has a distinctive hinge: a
    nearly vertical northern segment meets a southeast-running diagonal near
    Lake Tahoe. Requiring that junction prevents unrelated parcel lines, map
    frames, and legend columns from being pooled into one permissive channel.
    """

    height, width = image_shape
    minimum_dimension = min(height, width)
    evidence = np.zeros(image_shape, dtype=np.uint8)
    if lines is None:
        return evidence, None

    verticals = []
    diagonals = []
    for x1, y1, x2, y2 in lines[:, 0]:
        first = np.array((float(x1), float(y1)))
        second = np.array((float(x2), float(y2)))
        delta = second - first
        length = float(np.linalg.norm(delta))
        angle = abs(float(np.degrees(np.arctan2(delta[1], delta[0]))))
        folded_angle = min(angle, 180.0 - angle)
        midpoint = (first + second) / 2.0

        if folded_angle >= 80.0 and length >= minimum_dimension * 0.12:
            top, bottom = sorted((first, second), key=lambda point: point[1])
            if (
                width * 0.48 <= midpoint[0] <= width * 0.82
                and top[1] <= height * 0.32
                and bottom[1] >= height * 0.18
            ):
                verticals.append((length, top, bottom))

        # Image y increases downward, so the southeast segment has dx and dy
        # with the same sign regardless of Hough endpoint ordering.
        if (
            28.0 <= folded_angle <= 65.0
            and delta[0] * delta[1] > 0
            and length >= minimum_dimension * 0.18
        ):
            upper_left, lower_right = sorted(
                (first, second), key=lambda point: point[1]
            )
            if (
                midpoint[0] >= width * 0.58
                and upper_left[1] <= height * 0.50
                and lower_right[0] >= upper_left[0] + width * 0.15
                and lower_right[1] >= upper_left[1] + height * 0.12
            ):
                diagonals.append((length, upper_left, lower_right))

    pairs = []
    for vertical_length, top, bottom in verticals:
        for diagonal_length, upper_left, lower_right in diagonals:
            junction_gap = float(np.linalg.norm(bottom - upper_left))
            if junction_gap > minimum_dimension * 0.07:
                continue
            # Prefer the longest coherent pair after junction proximity. This
            # distinguishes the state line from short legend-column pairs.
            score = junction_gap / minimum_dimension - 0.08 * (
                (vertical_length + diagonal_length) / (2.0 * minimum_dimension)
            )
            pairs.append((score, top, bottom, upper_left, lower_right))

    if not pairs:
        return evidence, None

    _, top, bottom, upper_left, lower_right = min(pairs, key=lambda item: item[0])
    # The detected Hough segments may stop a few pixels short of the actual
    # junction. Use their infinite-line intersection to make the hinge explicit.
    vertical_delta = bottom - top
    diagonal_delta = lower_right - upper_left
    cross = vertical_delta[0] * diagonal_delta[1] - vertical_delta[1] * diagonal_delta[0]
    if abs(cross) > 1e-6:
        offset = upper_left - top
        t = (offset[0] * diagonal_delta[1] - offset[1] * diagonal_delta[0]) / cross
        hinge = top + t * vertical_delta
    else:
        hinge = (bottom + upper_left) / 2.0

    top_point = tuple(np.rint(top).astype(int))
    hinge_point = tuple(np.rint(hinge).astype(int))
    lower_right_point = tuple(np.rint(lower_right).astype(int))
    cv2.line(evidence, top_point, hinge_point, 1, 3, cv2.LINE_AA)
    cv2.line(evidence, hinge_point, lower_right_point, 1, 3, cv2.LINE_AA)
    return evidence, (float(hinge[0]), float(hinge[1]))


def prepare_image(path: Path, max_dimension: int = 900) -> PreparedImage:
    """Downsample an image and produce conservative multi-scale line evidence."""

    with Image.open(path) as source:
        original = np.asarray(source.convert("RGB"))
    original_height, original_width = original.shape[:2]
    scale = min(1.0, max_dimension / max(original_height, original_width))
    if scale < 1.0:
        rgb = cv2.resize(
            original,
            (round(original_width * scale), round(original_height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    else:
        rgb = original

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    # A state line may be thin and dark (forest) or gray over hillshade (farms).
    # Combining two blur scales keeps both while discarding much of the JPEG noise.
    fine = cv2.GaussianBlur(gray, (3, 3), 0)
    coarse = cv2.GaussianBlur(gray, (7, 7), 0)
    median = float(np.median(fine))
    lower = max(18, int(0.45 * median))
    upper = max(lower + 20, int(0.9 * median))
    edges_fine = cv2.Canny(fine, lower, upper, L2gradient=True)
    edges_coarse = cv2.Canny(coarse, max(12, lower // 2), max(35, upper // 2), L2gradient=True)
    edges = (edges_fine > 0) | (edges_coarse > 0)

    # Ignore the literal image frame, which otherwise creates an attractive but false fit.
    # Page borders and map neatlines are common in downloaded figures and can
    # look more attractive to a partial-outline matcher than real geography.
    # Treat the outer 3% as layout evidence, not registration evidence. A true
    # geographic feature cropped this close to the frame is incomplete anyway.
    margin = max(3, round(min(rgb.shape[:2]) * 0.03))
    edges[:margin, :] = False
    edges[-margin:, :] = False
    edges[:, :margin] = False
    edges[:, -margin:] = False
    distance, nearest = distance_transform_edt(~edges, return_indices=True)

    gradient_x = cv2.Sobel(fine, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(fine, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(gradient_x, gradient_y)
    gradient_x = np.divide(
        gradient_x, magnitude, out=np.zeros_like(gradient_x), where=magnitude > 1e-6
    )
    gradient_y = np.divide(
        gradient_y, magnitude, out=np.zeros_like(gradient_y), where=magnitude > 1e-6
    )

    # Data colors are useful global evidence when edge clutter admits a locally
    # convincing but geographically absurd match. Highly saturated colors are
    # retained, except for dominant colors connected to the outer image border
    # (for example the blue Pacific in the forest map).
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    chroma = rgb.max(axis=2).astype(np.int16) - rgb.min(axis=2).astype(np.int16)
    color_evidence = (hsv[:, :, 1] >= 50) & (hsv[:, :, 2] >= 45) & (chroma >= 35)
    quantized = (rgb[:, :, 0] // 16).astype(np.int32) * 256
    quantized += (rgb[:, :, 1] // 16).astype(np.int32) * 16
    quantized += (rgb[:, :, 2] // 16).astype(np.int32)
    border_width = max(4, round(min(rgb.shape[:2]) * 0.012))
    border_mask = np.zeros(rgb.shape[:2], dtype=bool)
    border_mask[:border_width, :] = True
    border_mask[-border_width:, :] = True
    border_mask[:, :border_width] = True
    border_mask[:, -border_width:] = True
    global_counts = np.bincount(quantized.ravel(), minlength=4096)
    border_counts = np.bincount(quantized[border_mask], minlength=4096)
    dominant_border_codes = np.flatnonzero(
        (global_counts > rgb.shape[0] * rgb.shape[1] * 0.004)
        & (border_counts > int(border_mask.sum()) * 0.008)
    )
    if len(dominant_border_codes):
        color_evidence &= ~np.isin(quantized, dominant_border_codes)

    # California's eastern border contains a vertical segment joined to a long
    # southeast diagonal. Preserve only a connected pair with that geometry;
    # pooling every long line admits crop parcels and legend elements.
    hough_lines = cv2.HoughLinesP(
        edges.astype(np.uint8) * 255,
        1,
        np.pi / 720,
        threshold=max(28, round(min(rgb.shape[:2]) * 0.05)),
        minLineLength=max(55, round(min(rgb.shape[:2]) * 0.10)),
        maxLineGap=max(10, round(min(rgb.shape[:2]) * 0.025)),
    )
    eastern_straight_edges, eastern_border_hinge = _connected_eastern_border_evidence(
        hough_lines, edges.shape
    )
    eastern_straight_edges = eastern_straight_edges > 0
    eastern_straight_distance = distance_transform_edt(~eastern_straight_edges)
    return PreparedImage(
        rgb=rgb,
        gray=gray,
        edges=edges,
        distance=distance.astype(np.float32),
        nearest_edge_y=nearest[0].astype(np.int32),
        nearest_edge_x=nearest[1].astype(np.int32),
        gradient_x=gradient_x,
        gradient_y=gradient_y,
        color_evidence=color_evidence,
        eastern_straight_edges=eastern_straight_edges,
        eastern_straight_distance=eastern_straight_distance.astype(np.float32),
        eastern_border_hinge=eastern_border_hinge,
        scale_from_original=scale,
        original_width=original_width,
        original_height=original_height,
    )
