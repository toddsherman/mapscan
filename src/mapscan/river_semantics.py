"""Auditable text-versus-geometry separation for named hydrography maps."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

from .extraction import warp_classified_to_web_mercator
from .feature_extraction import _alignment_transform
from .reference import load_california


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rotate_page(
    page: np.ndarray, angle_degrees: float, scale: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Rotate and scale a page without clipping, returning source-to-page affine."""

    height, width = page.shape
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_degrees, scale)
    output_width = max(
        1, int(math.ceil(height * abs(matrix[0, 1]) + width * abs(matrix[0, 0])))
    )
    output_height = max(
        1, int(math.ceil(height * abs(matrix[0, 0]) + width * abs(matrix[0, 1])))
    )
    matrix[0, 2] += output_width / 2.0 - center[0]
    matrix[1, 2] += output_height / 2.0 - center[1]
    rotated = cv2.warpAffine(
        page,
        matrix,
        (output_width, output_height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )
    return rotated, matrix


def _transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack((points, np.ones(len(points), dtype=np.float64)))
    return homogeneous @ matrix.T


def _angle_distance(first: float, second: float) -> float:
    difference = abs((first - second) % 180.0)
    return min(difference, 180.0 - difference)


def _candidate_consensus_counts(candidates: Sequence[Dict[str, object]]) -> list[int]:
    """Count distinct nearby OCR rotations supporting each source-space proposal."""

    counts = []
    for index, candidate in enumerate(candidates):
        supporting_angles = {int(candidate["ocr_angle_degrees"])}
        center = np.asarray(candidate["source_center"], dtype=np.float64)
        length = float(candidate["source_word_length_px"])
        orientation = float(candidate["source_orientation_degrees"])
        for other_index, other in enumerate(candidates):
            if index == other_index:
                continue
            other_orientation = float(other["source_orientation_degrees"])
            if _angle_distance(orientation, other_orientation) > 22.0:
                continue
            other_center = np.asarray(other["source_center"], dtype=np.float64)
            other_length = float(other["source_word_length_px"])
            maximum_distance = max(10.0, 0.38 * max(length, other_length))
            if float(np.linalg.norm(center - other_center)) <= maximum_distance:
                supporting_angles.add(int(other["ocr_angle_degrees"]))
        counts.append(len(supporting_angles))
    return counts


def select_text_detections(
    candidates: Sequence[Dict[str, object]],
    high_confidence: float = 65.0,
    consensus_confidence: float = 42.0,
    minimum_consensus_rotations: int = 2,
) -> Tuple[list[Dict[str, object]], list[Dict[str, object]]]:
    """Select strong OCR words while retaining all rejected proposals for review."""

    consensus_counts = _candidate_consensus_counts(candidates)
    accepted = []
    ambiguous = []
    for candidate, support in zip(candidates, consensus_counts):
        record = dict(candidate)
        record["consensus_rotation_count"] = support
        confidence = float(record["confidence"])
        if confidence >= high_confidence:
            record["acceptance_reason"] = "high_confidence_rotation_aware_ocr"
            accepted.append(record)
        elif confidence >= consensus_confidence and support >= minimum_consensus_rotations:
            record["acceptance_reason"] = "multi_rotation_spatial_consensus"
            accepted.append(record)
        else:
            record["rejection_reason"] = "insufficient_confidence_or_rotation_consensus"
            ambiguous.append(record)
    return accepted, ambiguous


def _regions_from_detections(
    shape: Tuple[int, int], detections: Iterable[Dict[str, object]]
) -> np.ndarray:
    regions = np.zeros(shape, dtype=np.uint8)
    for detection in detections:
        polygon = np.rint(np.asarray(detection["source_polygon"], dtype=np.float64)).astype(
            np.int32
        )
        cv2.fillConvexPoly(regions, polygon, 1)
    return regions.astype(bool)


def _representative_detections(
    detections: Sequence[Dict[str, object]],
) -> list[Dict[str, object]]:
    """Suppress near-duplicate rotation hits before morphology-based word expansion."""

    representatives: list[Dict[str, object]] = []
    for detection in sorted(detections, key=lambda item: -float(item["confidence"])):
        center = np.asarray(detection["source_center"], dtype=np.float64)
        length = float(detection["source_word_length_px"])
        orientation = float(detection["source_orientation_degrees"])
        duplicate = False
        for kept in representatives:
            kept_center = np.asarray(kept["source_center"], dtype=np.float64)
            kept_length = float(kept["source_word_length_px"])
            if _angle_distance(
                orientation, float(kept["source_orientation_degrees"])
            ) > 18.0:
                continue
            if float(np.linalg.norm(center - kept_center)) <= max(
                8.0, 0.3 * max(length, kept_length)
            ):
                duplicate = True
                break
        if not duplicate:
            representatives.append(detection)
    return representatives


def extract_confirmed_glyph_components(
    observed_ink: np.ndarray,
    detections: Sequence[Dict[str, object]],
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Keep compact components inside OCR boxes while rejecting connected linework."""

    observed = observed_ink.astype(bool)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        observed.astype(np.uint8), connectivity=8
    )
    accepted_ids: set[int] = set()
    rejected_ids: set[int] = set()
    for detection in detections:
        region = _regions_from_detections(observed.shape, [detection])
        word_height = float(detection["source_word_height_px"])
        for component_id in np.unique(labels[region]):
            component_id = int(component_id)
            if component_id <= 0 or component_id >= count:
                continue
            width = int(stats[component_id, cv2.CC_STAT_WIDTH])
            height = int(stats[component_id, cv2.CC_STAT_HEIGHT])
            area = int(stats[component_id, cv2.CC_STAT_AREA])
            maximum_dimension = max(18.0, 3.2 * word_height)
            maximum_area = max(120.0, 22.0 * word_height)
            aspect = max(width, height) / max(min(width, height), 1)
            extent = area / max(width * height, 1)
            if (
                max(width, height) > maximum_dimension
                or area > maximum_area
                or aspect > 8.0
                or extent < 0.055
            ):
                rejected_ids.add(component_id)
                continue
            component = labels == component_id
            overlap_fraction = float(
                np.count_nonzero(component & region) / max(area, 1)
            )
            if overlap_fraction < 0.55:
                rejected_ids.add(component_id)
                continue
            accepted_ids.add(component_id)
    mask = np.isin(labels, list(accepted_ids)) if accepted_ids else np.zeros_like(observed)
    return mask, {
        "accepted_glyph_component_count": len(accepted_ids),
        "rejected_line_or_large_component_count": len(rejected_ids - accepted_ids),
    }


def expanded_detection_corridors(
    shape: Tuple[int, int],
    detections: Sequence[Dict[str, object]],
    maximum_extension_px: float = 72.0,
) -> np.ndarray:
    """Return conservative unresolved-ink corridors around confirmed partial words."""

    corridors = np.zeros(shape, dtype=np.uint8)
    for detection in _representative_detections(detections):
        center = np.asarray(detection["source_center"], dtype=np.float64)
        length = float(detection["source_word_length_px"])
        height = float(detection["source_word_height_px"])
        orientation = float(detection["source_orientation_degrees"])
        radians = math.radians(orientation)
        along = np.asarray([math.cos(radians), math.sin(radians)])
        across = np.asarray([-along[1], along[0]])
        extension = min(maximum_extension_px, max(14.0, 1.8 * length))
        half_length = length / 2.0 + extension
        half_height = max(5.0, 0.9 * height)
        polygon = np.asarray(
            [
                center - along * half_length - across * half_height,
                center + along * half_length - across * half_height,
                center + along * half_length + across * half_height,
                center - along * half_length + across * half_height,
            ]
        )
        polygon[:, 0] = np.clip(polygon[:, 0], 0, shape[1] - 1)
        polygon[:, 1] = np.clip(polygon[:, 1], 0, shape[0] - 1)
        cv2.fillConvexPoly(corridors, np.rint(polygon).astype(np.int32), 1)
    return corridors.astype(bool)


def detect_rotation_aware_text_like_regions(
    observed_ink: np.ndarray,
    angles: Sequence[int] = tuple(range(-80, 91, 10)),
    compact_component_maximum_area_px: int = 240,
    compact_component_maximum_dimension_px: int = 32,
    thick_core_distance_px: float = 2.8,
    horizontal_closing_width_px: int = 25,
    minimum_compact_components_per_region: int = 3,
    longitudinal_padding_px: int = 24,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Find unresolved word-like ink without declaring nearby linework to be text.

    Compact connected components supply glyph evidence. Rotation-normalized closing
    then finds thicker, partly connected words such as curved river labels. The
    resulting strips are review regions: their observed ink is withheld from the
    geometry candidate, but is not promoted to confirmed text.
    """

    observed = observed_ink.astype(bool)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        observed.astype(np.uint8), connectivity=8
    )
    area = stats[:, cv2.CC_STAT_AREA]
    width = stats[:, cv2.CC_STAT_WIDTH]
    height = stats[:, cv2.CC_STAT_HEIGHT]
    aspect = np.maximum(width, height) / np.maximum(np.minimum(width, height), 1)
    extent = area / np.maximum(width * height, 1)
    compact_ids = np.where(
        (area >= 3)
        & (area <= compact_component_maximum_area_px)
        & (np.maximum(width, height) <= compact_component_maximum_dimension_px)
        & (aspect <= 6.0)
        & (extent >= 0.08)
    )[0]
    compact = np.isin(labels, compact_ids)

    distance = cv2.distanceTransform(observed.astype(np.uint8), cv2.DIST_L2, 5)
    thick_core = distance >= thick_core_distance_px
    review_regions = np.zeros(observed.shape, dtype=np.uint8)
    accepted_region_count = 0
    candidate_region_count = 0
    closing_kernel = np.ones((3, horizontal_closing_width_px), dtype=np.uint8)
    for angle in angles:
        rotated, matrix = _rotate_page(
            np.where(thick_core, 0, 255).astype(np.uint8), float(angle), 1.0
        )
        rotated_core = rotated < 128
        closed = cv2.morphologyEx(
            rotated_core.astype(np.uint8), cv2.MORPH_CLOSE, closing_kernel
        )
        component_count, rotated_labels, rotated_stats, _ = (
            cv2.connectedComponentsWithStats(closed, connectivity=8)
        )
        inverse = cv2.invertAffineTransform(matrix)
        for component_id in range(1, component_count):
            x = int(rotated_stats[component_id, cv2.CC_STAT_LEFT])
            y = int(rotated_stats[component_id, cv2.CC_STAT_TOP])
            component_width = int(rotated_stats[component_id, cv2.CC_STAT_WIDTH])
            component_height = int(rotated_stats[component_id, cv2.CC_STAT_HEIGHT])
            core_pixel_count = int(
                np.count_nonzero(rotated_core & (rotated_labels == component_id))
            )
            if (
                component_width < 22
                or component_width > 190
                or component_height < 2
                or component_height > 24
                or component_width / max(component_height, 1) < 2.0
                or core_pixel_count < 8
            ):
                continue
            candidate_region_count += 1
            transverse_padding = max(3, round(component_height * 0.5))
            corners = np.asarray(
                [
                    [x - longitudinal_padding_px, y - transverse_padding],
                    [
                        x + component_width + longitudinal_padding_px,
                        y - transverse_padding,
                    ],
                    [
                        x + component_width + longitudinal_padding_px,
                        y + component_height + transverse_padding,
                    ],
                    [
                        x - longitudinal_padding_px,
                        y + component_height + transverse_padding,
                    ],
                ],
                dtype=np.float64,
            )
            polygon = _transform_points(corners, inverse)
            polygon[:, 0] = np.clip(polygon[:, 0], 0, observed.shape[1] - 1)
            polygon[:, 1] = np.clip(polygon[:, 1], 0, observed.shape[0] - 1)
            region = np.zeros(observed.shape, dtype=np.uint8)
            cv2.fillConvexPoly(region, np.rint(polygon).astype(np.int32), 1)
            supporting_ids = set(np.unique(labels[(region > 0) & compact]).tolist())
            supporting_ids.discard(0)
            if len(supporting_ids) < minimum_compact_components_per_region:
                continue
            cv2.fillConvexPoly(
                review_regions, np.rint(polygon).astype(np.int32), 1
            )
            accepted_region_count += 1

    review_regions_bool = review_regions.astype(bool)
    text_like_pixels = compact | (observed & review_regions_bool)
    return text_like_pixels, review_regions_bool, {
        "method": "multi_rotation_thick_core_closing_validated_by_compact_glyph_components",
        "compact_component_count": int(len(compact_ids)),
        "compact_component_pixel_count": int(np.count_nonzero(compact)),
        "thick_core_pixel_count": int(np.count_nonzero(thick_core)),
        "candidate_rotated_region_count": candidate_region_count,
        "accepted_rotated_region_count": accepted_region_count,
        "review_region_pixel_count": int(np.count_nonzero(review_regions_bool)),
        "observed_text_like_pixel_count": int(np.count_nonzero(text_like_pixels)),
        "parameters": {
            "angles_degrees": list(angles),
            "compact_component_maximum_area_px": compact_component_maximum_area_px,
            "compact_component_maximum_dimension_px": compact_component_maximum_dimension_px,
            "thick_core_distance_px": thick_core_distance_px,
            "horizontal_closing_width_px": horizontal_closing_width_px,
            "minimum_compact_components_per_region": minimum_compact_components_per_region,
            "longitudinal_padding_px": longitudinal_padding_px,
        },
    }


def expand_confirmed_text_regions(
    observed_ink: np.ndarray,
    core_regions: np.ndarray,
    detections: Sequence[Dict[str, object]],
    maximum_extension_px: float = 72.0,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Grow partial OCR tokens only to nearby small, glyph-like ink components."""

    observed = observed_ink.astype(bool)
    expanded_regions = core_regions.astype(bool).copy()
    representative_count = 0
    expanded_detection_count = 0
    accepted_component_ids: set[int] = set()
    rejected_long_or_large_components: set[int] = set()
    component_count, component_labels, stats, centroids = cv2.connectedComponentsWithStats(
        observed.astype(np.uint8), connectivity=8
    )
    for detection in _representative_detections(detections):
        representative_count += 1
        center = np.asarray(detection["source_center"], dtype=np.float64)
        length = float(detection["source_word_length_px"])
        height = float(detection["source_word_height_px"])
        orientation = float(detection["source_orientation_degrees"])
        radians = math.radians(orientation)
        along = np.asarray([math.cos(radians), math.sin(radians)])
        across = np.asarray([-along[1], along[0]])
        extension = min(maximum_extension_px, max(14.0, 1.8 * length))
        half_length = length / 2.0 + extension
        half_height = max(5.0, 0.9 * height)
        polygon = np.asarray(
            [
                center - along * half_length - across * half_height,
                center + along * half_length - across * half_height,
                center + along * half_length + across * half_height,
                center - along * half_length + across * half_height,
            ]
        )
        polygon[:, 0] = np.clip(polygon[:, 0], 0, observed.shape[1] - 1)
        polygon[:, 1] = np.clip(polygon[:, 1], 0, observed.shape[0] - 1)
        corridor = np.zeros(observed.shape, dtype=np.uint8)
        cv2.fillConvexPoly(corridor, np.rint(polygon).astype(np.int32), 1)
        selected_ids = []
        for component_id in np.unique(component_labels[corridor > 0]):
            component_id = int(component_id)
            if component_id <= 0 or component_id >= component_count:
                continue
            component_width = int(stats[component_id, cv2.CC_STAT_WIDTH])
            component_height = int(stats[component_id, cv2.CC_STAT_HEIGHT])
            component_area = int(stats[component_id, cv2.CC_STAT_AREA])
            maximum_glyph_dimension = max(18.0, 3.2 * height)
            maximum_glyph_area = max(120.0, 22.0 * height)
            if (
                max(component_width, component_height) > maximum_glyph_dimension
                or component_area > maximum_glyph_area
            ):
                rejected_long_or_large_components.add(component_id)
                continue
            component_center = np.asarray(centroids[component_id], dtype=np.float64)
            offset = component_center - center
            longitudinal = abs(float(np.dot(offset, along)))
            perpendicular = abs(float(np.dot(offset, across)))
            if longitudinal > half_length or perpendicular > max(5.0, 0.85 * height):
                continue
            component_mask = component_labels == component_id
            corridor_fraction = float(
                np.count_nonzero(component_mask & (corridor > 0)) / component_area
            )
            if corridor_fraction < 0.8:
                continue
            selected_ids.append(component_id)
        if not selected_ids:
            continue
        selected = np.isin(component_labels, selected_ids)
        before = int(np.count_nonzero(expanded_regions))
        expanded_regions |= selected
        if int(np.count_nonzero(expanded_regions)) > before:
            expanded_detection_count += 1
            accepted_component_ids.update(selected_ids)
    report = {
        "method": "confirmed_ocr_seed_nearby_glyph_component_expansion",
        "representative_detection_count": representative_count,
        "expanded_detection_count": expanded_detection_count,
        "accepted_glyph_component_count": len(accepted_component_ids),
        "rejected_long_or_large_component_count": len(
            rejected_long_or_large_components
        ),
        "maximum_longitudinal_extension_px": maximum_extension_px,
        "region_pixel_count_before": int(np.count_nonzero(core_regions)),
        "region_pixel_count_after": int(np.count_nonzero(expanded_regions)),
    }
    return expanded_regions, report


def infer_reconnections(
    observed_geometry: np.ndarray,
    observed_ink: np.ndarray,
    text_regions: np.ndarray,
    maximum_gap_px: int = 28,
    maximum_component_area_px: int = 240,
    maximum_component_dimension_px: int = 90,
    minimum_contact_pixels: int = 2,
    minimum_contact_span_px: float = 5.0,
) -> Tuple[np.ndarray, list[Dict[str, object]], Dict[str, int]]:
    """Propose straight, locally supported bridges only inside OCR text regions."""

    if observed_geometry.shape != observed_ink.shape or text_regions.shape != observed_ink.shape:
        raise ValueError("River separation masks must share one source grid")
    if maximum_gap_px < 2:
        raise ValueError("maximum_gap_px must be at least two")

    geometry = observed_geometry.astype(bool)
    kernel_size = 2 * maximum_gap_px + 1
    accepted = np.zeros(geometry.shape, dtype=bool)
    records = []
    rejected = {
        "too_large": 0,
        "too_wide": 0,
        "insufficient_geometry_contacts": 0,
        "insufficient_contact_span": 0,
        "orientation_mismatch": 0,
        "insufficient_straight_line_support": 0,
    }
    contact_kernel = np.ones((3, 3), dtype=np.uint8)
    for angle in range(0, 180, 15):
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.uint8)
        radius = maximum_gap_px
        radians = math.radians(angle)
        dx = int(round(radius * math.cos(radians)))
        dy = int(round(radius * math.sin(radians)))
        cv2.line(kernel, (radius - dx, radius - dy), (radius + dx, radius + dy), 1, 1)
        closed = cv2.morphologyEx(
            geometry.astype(np.uint8),
            cv2.MORPH_CLOSE,
            kernel,
            borderType=cv2.BORDER_CONSTANT,
            borderValue=0,
        ) > 0
        proposals = closed & text_regions & ~observed_ink.astype(bool)
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
            proposals.astype(np.uint8), connectivity=8
        )
        for component_id in range(1, component_count):
            area = int(stats[component_id, cv2.CC_STAT_AREA])
            x = int(stats[component_id, cv2.CC_STAT_LEFT])
            y = int(stats[component_id, cv2.CC_STAT_TOP])
            width = int(stats[component_id, cv2.CC_STAT_WIDTH])
            height = int(stats[component_id, cv2.CC_STAT_HEIGHT])
            if area > maximum_component_area_px:
                rejected["too_large"] += 1
                continue
            if max(width, height) > maximum_component_dimension_px:
                rejected["too_wide"] += 1
                continue
            component = labels == component_id
            contacts = (
                cv2.dilate(component.astype(np.uint8), contact_kernel).astype(bool)
                & geometry
            )
            contact_points = np.column_stack(np.nonzero(contacts))
            if len(contact_points) < minimum_contact_pixels:
                rejected["insufficient_geometry_contacts"] += 1
                continue
            difference = contact_points[:, None, :] - contact_points[None, :, :]
            distances = np.sqrt(np.sum(difference.astype(np.float64) ** 2, axis=2))
            endpoint_indices = np.unravel_index(int(np.argmax(distances)), distances.shape)
            contact_span = float(distances[endpoint_indices])
            if contact_span < minimum_contact_span_px:
                rejected["insufficient_contact_span"] += 1
                continue
            first = contact_points[endpoint_indices[0]]
            second = contact_points[endpoint_indices[1]]
            dx_contact = float(second[1] - first[1])
            dy_contact = float(second[0] - first[0])
            contact_orientation = math.degrees(
                math.atan2(dy_contact, dx_contact)
            ) % 180.0
            if _angle_distance(contact_orientation, float(angle)) > 18.0:
                rejected["orientation_mismatch"] += 1
                continue
            line = np.zeros(geometry.shape, dtype=np.uint8)
            cv2.line(
                line,
                (int(first[1]), int(first[0])),
                (int(second[1]), int(second[0])),
                1,
                1,
            )
            line_candidate = (
                (line > 0) & text_regions & ~observed_ink.astype(bool)
            )
            if not np.any(line_candidate):
                rejected["insufficient_straight_line_support"] += 1
                continue
            component_support = cv2.dilate(
                component.astype(np.uint8), np.ones((3, 3), dtype=np.uint8)
            ).astype(bool)
            support_fraction = float(
                np.count_nonzero(line_candidate & component_support)
                / np.count_nonzero(line_candidate)
            )
            if support_fraction < 0.8:
                rejected["insufficient_straight_line_support"] += 1
                continue
            newly_accepted = line_candidate & ~accepted
            if not np.any(newly_accepted):
                continue
            accepted |= line_candidate
            records.append(
                {
                    "source_bbox": [x, y, width, height],
                    "pixel_count": area,
                    "new_pixel_count": int(np.count_nonzero(newly_accepted)),
                    "proposal_angle_degrees": angle,
                    "contact_orientation_degrees": contact_orientation,
                    "straight_line_support_fraction": support_fraction,
                    "geometry_contact_pixel_count": int(len(contact_points)),
                    "contact_span_px": contact_span,
                    "source_contact_endpoints_xy": [
                        [int(first[1]), int(first[0])],
                        [int(second[1]), int(second[0])],
                    ],
                }
            )
    return accepted, records, rejected


def _preview(
    rgb: np.ndarray,
    geometry: np.ndarray,
    text: np.ndarray,
    inference: np.ndarray,
    ambiguous: np.ndarray,
) -> np.ndarray:
    preview = np.rint(rgb.astype(np.float32) * 0.36 + 255.0 * 0.64).astype(np.uint8)
    preview[geometry] = [0, 205, 232]
    preview[ambiguous] = [255, 132, 0]
    preview[text] = [235, 0, 210]
    preview[inference] = [255, 220, 0]
    return preview


def _build_ocr_candidate(
    text: str,
    confidence: float,
    bbox: Tuple[float, float, float, float],
    angle: int,
    inverse: np.ndarray,
    observed: np.ndarray,
    ocr_scale: float,
    minimum_candidate_confidence: float,
    engine: str,
) -> Tuple[Dict[str, object] | None, str | None]:
    """Validate one OCR box and map it back to source-image coordinates."""

    x, y, width, height = bbox
    letters = re.sub(r"[^A-Za-z]", "", text)
    if confidence < minimum_candidate_confidence or len(letters) < 3:
        return None, "confidence_or_nonword"
    if width <= 0 or height <= 0:
        return None, "invalid_dimensions"
    source_word_length = width / ocr_scale
    source_word_height = height / ocr_scale
    if (
        source_word_length > 190
        or source_word_height < 4
        or source_word_height > 46
    ):
        return None, "implausible_source_size"
    padding = 1.5 * ocr_scale
    corners = np.asarray(
        [
            [x - padding, y - padding],
            [x + width + padding, y - padding],
            [x + width + padding, y + height + padding],
            [x - padding, y + height + padding],
        ],
        dtype=np.float64,
    )
    polygon = _transform_points(corners, inverse)
    polygon[:, 0] = np.clip(polygon[:, 0], 0, observed.shape[1] - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, observed.shape[0] - 1)
    region = np.zeros(observed.shape, dtype=np.uint8)
    cv2.fillConvexPoly(region, np.rint(polygon).astype(np.int32), 1)
    region_area = int(np.count_nonzero(region))
    ink_count = int(np.count_nonzero(observed & (region > 0)))
    ink_density = ink_count / max(region_area, 1)
    if ink_count < 6 or ink_density < 0.025:
        return None, "insufficient_ink_density"
    center = _transform_points(
        np.asarray([[x + width / 2.0, y + height / 2.0]]), inverse
    )[0]
    return (
        {
            "text": text,
            "confidence": confidence,
            "engine": engine,
            "ocr_angle_degrees": int(angle),
            "source_orientation_degrees": float((-angle) % 180),
            "source_center": [float(center[0]), float(center[1])],
            "source_word_length_px": float(source_word_length),
            "source_word_height_px": float(source_word_height),
            "source_polygon": polygon.tolist(),
            "observed_ink_pixel_count": ink_count,
            "region_pixel_count": region_area,
            "observed_ink_density": ink_density,
        },
        None,
    )


def extract_river_semantics(
    run_dir: Path,
    output_dir: Path,
    angles: Sequence[int] = tuple(range(-80, 91, 10)),
    ocr_scale: float = 2.0,
    page_segmentation_mode: int = 11,
    minimum_candidate_confidence: float = 30.0,
    high_confidence: float = 65.0,
    consensus_confidence: float = 42.0,
    maximum_gap_px: int = 28,
) -> Dict[str, object]:
    """Separate blue label ink from observed geometry without mutating evidence."""

    run_dir = run_dir.resolve()
    output_dir = output_dir.resolve()
    manifest_path = run_dir / "feature-extraction.json"
    manifest = json.loads(manifest_path.read_text())
    plan_path = Path(manifest["plan"]).resolve()
    plan = json.loads(plan_path.read_text())
    source_path = Path(plan["source"]).resolve()
    alignment_path = Path(plan["alignment"]).resolve()
    alignment = json.loads(alignment_path.read_text())
    transform = _alignment_transform(alignment)
    reference_root = Path(plan.get("reference", "reference/census-2025"))
    state, _ = load_california(reference_root)
    tesseract = shutil.which("tesseract")
    if tesseract is None:
        raise RuntimeError("Tesseract is required for rotation-aware river label detection")

    rgb = np.asarray(Image.open(source_path).convert("RGB"))
    observed_path = run_dir / "source-observed-ink.png"
    web_observed_path = run_dir / "web-mercator-observed-ink.png"
    observed = np.asarray(Image.open(observed_path)) > 0
    page = np.where(observed, 0, 255).astype(np.uint8)
    candidates: list[Dict[str, object]] = []
    raw_word_count = 0
    rejection_counts = {
        "confidence_or_nonword": 0,
        "invalid_dimensions": 0,
        "implausible_source_size": 0,
        "insufficient_ink_density": 0,
    }
    engine_raw_word_counts = {"tesseract": 0, "apple_vision": 0}
    vision_status = "unavailable"

    with tempfile.TemporaryDirectory(prefix="mapscan-river-ocr-") as temporary:
        temporary_dir = Path(temporary)
        vision_binary = None
        swiftc = shutil.which("swiftc")
        vision_script = Path(__file__).resolve().parents[2] / "scripts" / "vision_text.swift"
        if swiftc is not None and vision_script.exists():
            candidate_binary = temporary_dir / "vision-text"
            compilation = subprocess.run(
                [swiftc, str(vision_script), "-o", str(candidate_binary)],
                capture_output=True,
                text=True,
            )
            if compilation.returncode == 0:
                vision_binary = candidate_binary
                vision_status = "enabled"
            else:
                vision_status = "compile_failed"
        for angle in angles:
            rotated, matrix = _rotate_page(page, float(angle), ocr_scale)
            page_path = temporary_dir / f"angle-{angle:+04d}.png"
            Image.fromarray(rotated, mode="L").save(page_path, optimize=True)
            output_base = temporary_dir / f"angle-{angle:+04d}"
            subprocess.run(
                [
                    tesseract,
                    str(page_path),
                    str(output_base),
                    "-l",
                    "eng",
                    "--psm",
                    str(page_segmentation_mode),
                    "-c",
                    "tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.-",
                    "tsv",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            tsv_path = output_base.with_suffix(".tsv")
            inverse = cv2.invertAffineTransform(matrix)
            with tsv_path.open(newline="") as handle:
                rows = csv.DictReader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
                for row in rows:
                    text = str(row.get("text", "")).strip()
                    try:
                        confidence = float(row.get("conf", "-1"))
                        bbox = (
                            float(row.get("left", "0")),
                            float(row.get("top", "0")),
                            float(row.get("width", "0")),
                            float(row.get("height", "0")),
                        )
                    except (TypeError, ValueError):
                        rejection_counts["confidence_or_nonword"] += 1
                        continue
                    if confidence >= 0:
                        raw_word_count += 1
                        engine_raw_word_counts["tesseract"] += 1
                    candidate, rejection = _build_ocr_candidate(
                        text,
                        confidence,
                        bbox,
                        int(angle),
                        inverse,
                        observed,
                        ocr_scale,
                        minimum_candidate_confidence,
                        "tesseract",
                    )
                    if candidate is not None:
                        candidates.append(candidate)
                    elif rejection is not None:
                        rejection_counts[rejection] += 1

            if vision_binary is not None:
                vision = subprocess.run(
                    [str(vision_binary), str(page_path)],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                for record in json.loads(vision.stdout):
                    text = str(record.get("text", "")).strip()
                    confidence = float(record.get("confidence", 0.0)) * 100.0
                    normalized = record["normalized_bbox_bottom_left"]
                    bbox = (
                        float(normalized[0]) * rotated.shape[1],
                        (1.0 - float(normalized[1]) - float(normalized[3]))
                        * rotated.shape[0],
                        float(normalized[2]) * rotated.shape[1],
                        float(normalized[3]) * rotated.shape[0],
                    )
                    raw_word_count += 1
                    engine_raw_word_counts["apple_vision"] += 1
                    candidate, rejection = _build_ocr_candidate(
                        text,
                        confidence,
                        bbox,
                        int(angle),
                        inverse,
                        observed,
                        ocr_scale,
                        minimum_candidate_confidence,
                        "apple_vision",
                    )
                    if candidate is not None:
                        candidates.append(candidate)
                    elif rejection is not None:
                        rejection_counts[rejection] += 1

    accepted, ambiguous = select_text_detections(
        candidates,
        high_confidence=high_confidence,
        consensus_confidence=consensus_confidence,
    )
    core_regions = _regions_from_detections(observed.shape, accepted)
    core_text_mask, core_glyph_report = extract_confirmed_glyph_components(
        observed, accepted
    )
    expanded_text_pixels, expansion_report = expand_confirmed_text_regions(
        observed, core_text_mask, accepted
    )
    morphology_text_like, morphology_review_regions, morphology_report = (
        detect_rotation_aware_text_like_regions(observed, angles=angles)
    )
    text_mask = observed & expanded_text_pixels
    expansion_text_mask = text_mask & ~core_text_mask
    unresolved_regions = morphology_review_regions | morphology_text_like
    ambiguous_mask = morphology_text_like & ~text_mask
    observed_geometry = observed & ~text_mask & ~ambiguous_mask
    inference_mask, inferred_geometries, inference_rejected = infer_reconnections(
        observed_geometry,
        observed,
        core_regions,
        maximum_gap_px=maximum_gap_px,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(observed_path, output_dir / "source-observed-ink-authoritative.png")
    shutil.copyfile(web_observed_path, output_dir / "web-observed-ink-authoritative.png")
    source_masks = {
        "source-text-mask.png": text_mask,
        "source-text-mask-ocr-core.png": core_text_mask,
        "source-text-mask-connected-expansion.png": expansion_text_mask,
        "source-text-region-mask.png": core_regions,
        "source-ambiguous-text-mask.png": ambiguous_mask,
        "source-unresolved-text-mask.png": ambiguous_mask,
        "source-text-review-region-mask.png": unresolved_regions,
        "source-rotation-morphology-region-mask.png": morphology_review_regions,
        "source-observed-geometry-fragments.png": observed_geometry,
        "source-inferred-reconnections.png": inference_mask,
    }
    for name, mask in source_masks.items():
        Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(
            output_dir / name, optimize=True
        )

    warped_masks = {}
    for name, mask in {
        "web-text-mask.png": text_mask,
        "web-text-mask-ocr-core.png": core_text_mask,
        "web-text-mask-connected-expansion.png": expansion_text_mask,
        "web-text-region-mask.png": core_regions,
        "web-ambiguous-text-mask.png": ambiguous_mask,
        "web-unresolved-text-mask.png": ambiguous_mask,
        "web-text-review-region-mask.png": unresolved_regions,
        "web-rotation-morphology-region-mask.png": morphology_review_regions,
        "web-observed-geometry-fragments.png": observed_geometry,
        "web-inferred-reconnections.png": inference_mask,
    }.items():
        warped, _ = warp_classified_to_web_mercator(
            mask.astype(np.uint8),
            state,
            transform,
            observed.shape,
            target_height=int(manifest["warp"]["height"]),
        )
        warped_masks[name] = warped > 0
        Image.fromarray((warped > 0).astype(np.uint8) * 255, mode="L").save(
            output_dir / name, optimize=True
        )

    source_preview = _preview(
        rgb, observed_geometry, text_mask, inference_mask, ambiguous_mask
    )
    Image.fromarray(source_preview, mode="RGB").save(
        output_dir / "source-separation-preview.png", optimize=True
    )
    web_rgb = np.asarray(Image.open(run_dir / "web-mercator-source.jpg").convert("RGB"))
    web_preview = _preview(
        web_rgb,
        warped_masks["web-observed-geometry-fragments.png"],
        warped_masks["web-text-mask.png"],
        warped_masks["web-inferred-reconnections.png"],
        warped_masks["web-ambiguous-text-mask.png"],
    )
    Image.fromarray(web_preview, mode="RGB").save(
        output_dir / "web-separation-preview.png", optimize=True
    )

    detection_path = output_dir / "rotation-aware-ocr-detections.json"
    detection_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "accepted": accepted,
                "ambiguous": ambiguous,
                "rejection_counts_before_semantic_selection": rejection_counts,
            },
            indent=2,
        )
        + "\n"
    )
    geometry_path = output_dir / "inferred-reconnection-geometries.json"
    geometry_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "coordinate_space": "source_image_pixels_xy",
                "geometries": inferred_geometries,
                "rejected_component_counts": inference_rejected,
            },
            indent=2,
        )
        + "\n"
    )

    source_partition_valid = bool(
        not np.any(text_mask & ambiguous_mask)
        and not np.any(text_mask & observed_geometry)
        and not np.any(ambiguous_mask & observed_geometry)
        and np.array_equal(
            text_mask | ambiguous_mask | observed_geometry,
            observed,
        )
    )
    web_observed = np.asarray(Image.open(web_observed_path)) > 0
    web_partition = (
        warped_masks["web-text-mask.png"]
        | warped_masks["web-ambiguous-text-mask.png"]
        | warped_masks["web-observed-geometry-fragments.png"]
    )
    web_partition_valid = bool(
        not np.any(
            warped_masks["web-text-mask.png"]
            & warped_masks["web-ambiguous-text-mask.png"]
        )
        and not np.any(
            warped_masks["web-text-mask.png"]
            & warped_masks["web-observed-geometry-fragments.png"]
        )
        and not np.any(
            warped_masks["web-ambiguous-text-mask.png"]
            & warped_masks["web-observed-geometry-fragments.png"]
        )
        and np.array_equal(web_partition, web_observed)
    )
    result = {
        "schema_version": 1,
        "status": "needs_semantic_review",
        "dataset_id": manifest["dataset_id"],
        "run": str(run_dir),
        "run_manifest_sha256": _sha256(manifest_path),
        "source_observed_ink": {
            "path": str(observed_path),
            "sha256": _sha256(observed_path),
            "pixel_count": int(np.count_nonzero(observed)),
            "authoritative": True,
        },
        "method": "multi_angle_tesseract_and_apple_vision_with_compact_glyph_validation",
        "parameters": {
            "angles_degrees": list(angles),
            "ocr_scale": ocr_scale,
            "page_segmentation_mode": page_segmentation_mode,
            "minimum_candidate_confidence": minimum_candidate_confidence,
            "high_confidence": high_confidence,
            "consensus_confidence": consensus_confidence,
            "maximum_reconnection_gap_px": maximum_gap_px,
        },
        "ocr": {
            "raw_word_count": raw_word_count,
            "raw_word_count_by_engine": engine_raw_word_counts,
            "apple_vision_status": vision_status,
            "candidate_count": len(candidates),
            "accepted_detection_count": len(accepted),
            "ambiguous_detection_count": len(ambiguous),
            "detections": detection_path.name,
            "detections_sha256": _sha256(detection_path),
            "confirmed_glyph_components": core_glyph_report,
            "connected_text_expansion": expansion_report,
            "unresolved_text_morphology": morphology_report,
        },
        "source_partition": {
            "valid": source_partition_valid,
            "observed_text_pixel_count": int(np.count_nonzero(text_mask)),
            "ocr_core_text_pixel_count": int(np.count_nonzero(core_text_mask)),
            "connected_expansion_text_pixel_count": int(
                np.count_nonzero(expansion_text_mask)
            ),
            "observed_geometry_pixel_count": int(np.count_nonzero(observed_geometry)),
            "ambiguous_text_pixel_count": int(np.count_nonzero(ambiguous_mask)),
            "unresolved_text_like_pixel_count": int(
                np.count_nonzero(ambiguous_mask)
            ),
            "text_like_review_region_observed_pixel_count": int(
                np.count_nonzero(observed & unresolved_regions)
            ),
            "text_like_review_region_pixels_left_in_geometry": int(
                np.count_nonzero(observed_geometry & unresolved_regions)
            ),
        },
        "inference": {
            "pixel_count": int(np.count_nonzero(inference_mask)),
            "geometry_count": len(inferred_geometries),
            "overlaps_observed_pixel_count": int(np.count_nonzero(inference_mask & observed)),
            "geometries": geometry_path.name,
            "geometries_sha256": _sha256(geometry_path),
        },
        "web_partition": {
            "valid": web_partition_valid,
            "expected_observed_pixel_count": int(np.count_nonzero(web_observed)),
            "partition_observed_pixel_count": int(np.count_nonzero(web_partition)),
        },
        "warnings": [
            "Observed blue ink remains authoritative and is never overwritten by semantic output.",
            "OCR text masks are automatic proposals; orange unresolved pixels are withheld from the geometry candidate until reviewed.",
            "Yellow reconnections are inference, never observed evidence, and remain independently disableable.",
            "This diagnostic is not a publishable hydrography layer.",
        ],
    }
    (output_dir / "river-semantics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
