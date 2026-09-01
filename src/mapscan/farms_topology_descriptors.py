"""Training-only topology descriptors for the partial farms map.

This module deliberately has no knowledge of validation or retained-acceptance
masks.  It compares a rendered Mapbox *training* line mask with source-derived
county topology in one common raster space.  Local line orientation comes from
a structure tensor; proximity to a skeleton junction supplies a second,
rotation-invariant descriptor.  The two signals reject the most common
nearest-line alias: a nearby but differently oriented county or road-like line.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Literal

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class TopologyDescriptorConfig:
    maximum_match_px: float = 18.0
    candidate_count: int = 12
    tensor_sigma_px: float = 2.0
    minimum_orientation_coherence: float = 0.42
    maximum_orientation_error_degrees: float = 32.0
    orientation_penalty_px: float = 8.0
    junction_radius_px: float = 6.0
    junction_penalty_px: float = 5.0
    bin_reference_px: int = 260
    minimum_bin_support: int = 20
    maximum_bin_residual_mad_px: float = 5.0


@dataclass(frozen=True)
class TopologyDescriptorFields:
    tangent_x: np.ndarray
    tangent_y: np.ndarray
    coherence: np.ndarray
    junction_distance: np.ndarray
    skeleton: np.ndarray
    junctions: np.ndarray


@dataclass(frozen=True)
class TopologyMatches:
    reference_points: np.ndarray
    mapped_points: np.ndarray
    source_points: np.ndarray
    residuals: np.ndarray
    distances: np.ndarray
    orientation_error_degrees: np.ndarray
    reference_coherence: np.ndarray
    source_coherence: np.ndarray
    reference_junction_near: np.ndarray
    source_junction_near: np.ndarray
    eligible: np.ndarray
    method: str


def _sha256_mask(mask: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(mask.astype(np.uint8)).tobytes()
    ).hexdigest()


def _morphological_skeleton(mask: np.ndarray) -> np.ndarray:
    """Return a deterministic OpenCV morphological skeleton."""

    work = mask.astype(np.uint8).copy()
    skeleton = np.zeros_like(work)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while np.any(work):
        eroded = cv2.erode(work, element)
        opened = cv2.dilate(eroded, element)
        skeleton |= work & ~opened
        work = eroded
    return skeleton.astype(bool)


def topology_descriptor_fields(
    mask: np.ndarray, *, sigma_px: float = 2.0
) -> TopologyDescriptorFields:
    """Describe local line tangent, confidence, and junction proximity."""

    if mask.ndim != 2:
        raise ValueError("Topology descriptor input must be a two-dimensional mask")
    if not np.any(mask):
        raise ValueError("Topology descriptor input mask is empty")
    foreground = mask.astype(np.float32)
    gx = cv2.Sobel(foreground, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(foreground, cv2.CV_32F, 0, 1, ksize=3)
    jxx = cv2.GaussianBlur(gx * gx, (0, 0), sigma_px)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), sigma_px)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), sigma_px)
    root = np.sqrt(np.maximum((jxx - jyy) ** 2 + 4.0 * jxy**2, 0.0))
    coherence = root / np.maximum(jxx + jyy, 1e-6)
    # The dominant tensor eigenvector is the line normal. Rotate it by pi/2.
    normal_angle = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy)
    tangent_x = -np.sin(normal_angle)
    tangent_y = np.cos(normal_angle)

    skeleton = _morphological_skeleton(mask)
    # Morphological skeletons can contain one-pixel gaps around an exact cross,
    # so immediate degree alone is brittle.  A junction is the skeleton locus
    # where the local structure tensor has no single dominant orientation.
    # Ordinary bends retain substantially higher coherence at this scale.
    junctions = skeleton & (coherence < 0.35)
    junction_distance = (
        distance_transform_edt(~junctions)
        if np.any(junctions)
        else np.full(mask.shape, np.inf)
    )
    return TopologyDescriptorFields(
        tangent_x=tangent_x.astype(np.float32),
        tangent_y=tangent_y.astype(np.float32),
        coherence=coherence.astype(np.float32),
        junction_distance=junction_distance.astype(np.float32),
        skeleton=skeleton,
        junctions=junctions,
    )


def _sample_fields(
    fields: TopologyDescriptorFields, points_xy: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    height, width = fields.coherence.shape
    rounded = np.rint(points_xy).astype(np.int32)
    x = np.clip(rounded[:, 0], 0, width - 1)
    y = np.clip(rounded[:, 1], 0, height - 1)
    tangent = np.column_stack((fields.tangent_x[y, x], fields.tangent_y[y, x]))
    return tangent, fields.coherence[y, x], fields.junction_distance[y, x], (
        (rounded[:, 0] >= 0)
        & (rounded[:, 0] < width)
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < height)
    )


def match_training_topology(
    reference_points: np.ndarray,
    mapped_points: np.ndarray,
    rendered_training_mask: np.ndarray,
    source_training_mask: np.ndarray,
    *,
    method: Literal["nearest", "descriptor"] = "descriptor",
    config: TopologyDescriptorConfig = TopologyDescriptorConfig(),
) -> TopologyMatches:
    """Match projected training pixels to source topology.

    ``reference_points`` stay in pinned Mapbox target pixels for geographic
    binning; ``mapped_points`` and both masks share the source working canvas.
    No validation/test input exists in this API.
    """

    reference_points = np.asarray(reference_points, dtype=np.float64)
    mapped_points = np.asarray(mapped_points, dtype=np.float64)
    if method not in {"nearest", "descriptor"}:
        raise ValueError(f"Unknown topology matching method: {method}")
    if (
        reference_points.shape != mapped_points.shape
        or reference_points.ndim != 2
        or reference_points.shape[1] != 2
    ):
        raise ValueError("Reference and mapped points must be equal Nx2 arrays")
    if rendered_training_mask.shape != source_training_mask.shape:
        raise ValueError("Rendered and source topology masks must share a canvas")
    source_y, source_x = np.nonzero(source_training_mask)
    if len(source_x) < config.minimum_bin_support:
        raise ValueError("Source training topology is undersupported")
    source_pixels = np.column_stack((source_x, source_y)).astype(np.float64)
    tree = cKDTree(source_pixels)

    reference_fields = topology_descriptor_fields(
        rendered_training_mask, sigma_px=config.tensor_sigma_px
    )
    source_fields = topology_descriptor_fields(
        source_training_mask, sigma_px=config.tensor_sigma_px
    )
    (
        reference_tangent,
        reference_coherence,
        reference_junction_distance,
        inside,
    ) = _sample_fields(reference_fields, mapped_points)

    k = 1 if method == "nearest" else min(config.candidate_count, len(source_pixels))
    query_distance, query_index = tree.query(mapped_points, k=k)
    if k == 1:
        query_distance = query_distance[:, None]
        query_index = query_index[:, None]
    candidates = source_pixels[query_index]
    flat_candidates = candidates.reshape(-1, 2)
    source_tangent, source_coherence, source_junction_distance, _ = _sample_fields(
        source_fields, flat_candidates
    )
    source_tangent = source_tangent.reshape(len(mapped_points), k, 2)
    source_coherence = source_coherence.reshape(len(mapped_points), k)
    source_junction_distance = source_junction_distance.reshape(len(mapped_points), k)

    reference_tangent_expanded = reference_tangent[:, None, :]
    axial_dot = np.abs(np.sum(reference_tangent_expanded * source_tangent, axis=2))
    axial_dot = np.clip(axial_dot, 0.0, 1.0)
    angle_degrees = np.degrees(np.arccos(axial_dot))
    reference_confident = reference_coherence >= config.minimum_orientation_coherence
    source_confident = source_coherence >= config.minimum_orientation_coherence
    comparable = reference_confident[:, None] & source_confident
    orientation_penalty = config.orientation_penalty_px * (1.0 - axial_dot)
    orientation_penalty[~comparable] = config.orientation_penalty_px * 0.5

    reference_junction_near = (
        reference_junction_distance <= config.junction_radius_px
    )
    source_junction_near = source_junction_distance <= config.junction_radius_px
    junction_mismatch = reference_junction_near[:, None] != source_junction_near
    score = query_distance.copy()
    if method == "descriptor":
        score = score + orientation_penalty + config.junction_penalty_px * junction_mismatch
        score[comparable & (angle_degrees > config.maximum_orientation_error_degrees)] = np.inf
    chosen = np.argmin(score, axis=1)
    row = np.arange(len(mapped_points))
    chosen_source = candidates[row, chosen]
    chosen_distance = query_distance[row, chosen]
    chosen_angle = angle_degrees[row, chosen]
    chosen_source_coherence = source_coherence[row, chosen]
    chosen_source_junction = source_junction_near[row, chosen]
    chosen_score = score[row, chosen]
    eligible = (
        inside
        & np.isfinite(chosen_score)
        & (chosen_distance <= config.maximum_match_px)
    )
    residuals = chosen_source - mapped_points
    return TopologyMatches(
        reference_points=reference_points,
        mapped_points=mapped_points,
        source_points=chosen_source,
        residuals=residuals,
        distances=chosen_distance,
        orientation_error_degrees=chosen_angle,
        reference_coherence=reference_coherence,
        source_coherence=chosen_source_coherence,
        reference_junction_near=reference_junction_near,
        source_junction_near=chosen_source_junction,
        eligible=eligible,
        method=method,
    )


def _robust_bin_indices(indices: np.ndarray, residuals: np.ndarray) -> np.ndarray:
    if len(indices) < 3:
        return indices
    values = residuals[indices]
    median = np.median(values, axis=0)
    deviation = np.linalg.norm(values - median, axis=1)
    median_deviation = float(np.median(deviation))
    mad = float(np.median(np.abs(deviation - median_deviation)))
    limit = median_deviation + max(2.5 * 1.4826 * mad, 0.75)
    return indices[deviation <= limit]


def summarize_match_consistency(
    matches: TopologyMatches,
    *,
    config: TopologyDescriptorConfig = TopologyDescriptorConfig(),
) -> dict[str, Any]:
    """Summarize training-only match and residual coherence by fixed bins."""

    eligible_indices = np.flatnonzero(matches.eligible)
    row_bin = (matches.reference_points[:, 1] // config.bin_reference_px).astype(int)
    column_bin = (matches.reference_points[:, 0] // config.bin_reference_px).astype(int)
    bins: list[dict[str, Any]] = []
    accepted_controls = 0
    pooled_deviation: list[float] = []
    for key in sorted(
        set(zip(row_bin[matches.eligible], column_bin[matches.eligible]))
    ):
        indices = np.flatnonzero(
            matches.eligible & (row_bin == key[0]) & (column_bin == key[1])
        )
        robust = _robust_bin_indices(indices, matches.residuals)
        if not len(robust):
            continue
        median_residual = np.median(matches.residuals[robust], axis=0)
        deviation = np.linalg.norm(matches.residuals[robust] - median_residual, axis=1)
        residual_mad = float(np.median(deviation))
        accepted = bool(
            len(robust) >= config.minimum_bin_support
            and residual_mad <= config.maximum_bin_residual_mad_px
        )
        accepted_controls += int(accepted)
        pooled_deviation.extend(deviation.tolist())
        bins.append(
            {
                "bin": [int(key[0]), int(key[1])],
                "eligible_support": int(len(indices)),
                "robust_support": int(len(robust)),
                "median_residual_working_px": median_residual.tolist(),
                "residual_median_absolute_deviation_px": residual_mad,
                "median_orientation_error_degrees": float(
                    np.median(matches.orientation_error_degrees[robust])
                ),
                "junction_class_agreement": float(
                    np.mean(
                        matches.reference_junction_near[robust]
                        == matches.source_junction_near[robust]
                    )
                ),
                "accepted_control": accepted,
            }
        )
    confident = (
        matches.eligible
        & (matches.reference_coherence >= config.minimum_orientation_coherence)
        & (matches.source_coherence >= config.minimum_orientation_coherence)
    )
    return {
        "method": matches.method,
        "reference_point_count": int(len(matches.reference_points)),
        "eligible_match_count": int(len(eligible_indices)),
        "eligible_fraction": float(np.mean(matches.eligible)),
        "median_match_distance_px": (
            float(np.median(matches.distances[eligible_indices]))
            if len(eligible_indices)
            else None
        ),
        "orientation_comparable_count": int(np.count_nonzero(confident)),
        "median_orientation_error_degrees": (
            float(np.median(matches.orientation_error_degrees[confident]))
            if np.any(confident)
            else None
        ),
        "junction_class_agreement": (
            float(
                np.mean(
                    matches.reference_junction_near[eligible_indices]
                    == matches.source_junction_near[eligible_indices]
                )
            )
            if len(eligible_indices)
            else None
        ),
        "pooled_within_bin_residual_median_absolute_deviation_px": (
            float(np.median(pooled_deviation)) if pooled_deviation else None
        ),
        "accepted_control_count": accepted_controls,
        "bin_count": len(bins),
        "bins": bins,
        "authority": {
            "accepted_inputs": [
                "pinned_mapbox_county_training_mask",
                "source_only_scoped_county_topology",
                "candidate_seed_mapping",
            ],
            "validation_or_retained_acceptance_masks_supported_by_api": False,
        },
    }


def descriptor_audit_provenance(
    rendered_training_mask: np.ndarray,
    source_training_mask: np.ndarray,
    config: TopologyDescriptorConfig,
) -> dict[str, Any]:
    return {
        "algorithm": (
            "nearest_k_plus_structure_tensor_tangent_and_"
            "skeleton_junction_proximity_v1"
        ),
        "rendered_training_mask_sha256": _sha256_mask(rendered_training_mask),
        "source_training_mask_sha256": _sha256_mask(source_training_mask),
        "config": config.__dict__,
        "validation_or_retained_acceptance_inputs": [],
    }
