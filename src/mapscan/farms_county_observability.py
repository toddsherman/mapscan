"""Source-only county-ink observability for the partial farms map.

The observability mask is frozen before any Mapbox geometry is mapped into the
source.  It combines the source-derived California county scope with a
conservative local contrast score calibrated only from already detected,
neutral county-like ink.  Reference-to-source distance is an evaluation output
and never participates in the observable/unobservable decision.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt


@dataclass(frozen=True)
class CountyObservabilityConfig:
    gaussian_sigma_px: float = 3.0
    chroma_weight: float = 0.35
    noise_floor: float = 4.0
    observed_ink_calibration_quantile: float = 0.01
    minimum_observed_ink_self_retention: float = 0.985
    evaluation_distance_px: float = 8.0


@dataclass(frozen=True)
class CountyObservabilityEvidence:
    score: np.ndarray
    observable: np.ndarray
    explicit_exclusion: np.ndarray
    low_contrast_exclusion: np.ndarray
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class ThematicOcclusionConfig:
    saturation_minimum: int = 40
    value_minimum: int = 40
    printed_stroke_radius_px: int = 2


@dataclass(frozen=True)
class ThematicOcclusionEvidence:
    thematic_pixels: np.ndarray
    expanded_thematic_pixels: np.ndarray
    occluded: np.ndarray
    observable: np.ndarray
    diagnostics: dict[str, Any]


def _mask_sha256(mask: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(mask.astype(np.uint8)).tobytes()
    ).hexdigest()


def derive_county_observability(
    rgb_source: np.ndarray,
    county_scope: np.ndarray,
    observed_county_ink: np.ndarray,
    *,
    config: CountyObservabilityConfig = CountyObservabilityConfig(),
) -> CountyObservabilityEvidence:
    """Derive a conservative county-ink observability mask from source only.

    ``county_scope`` is the independently derived partial-California support;
    all pixels outside it are explicitly unobservable.  Inside that support,
    the score estimates how distinguishable the source's neutral county-ink
    color would be from local background and texture.  The cutoff is the first
    percentile of scores at already observed source ink, so low contrast is
    declared only where detectability is worse than almost all observed ink.
    """

    if rgb_source.ndim != 3 or rgb_source.shape[2] != 3:
        raise ValueError("County observability requires an RGB source raster")
    shape = rgb_source.shape[:2]
    if county_scope.shape != shape or observed_county_ink.shape != shape:
        raise ValueError("County observability masks must match the source raster")
    county_scope = county_scope.astype(bool)
    observed_county_ink = observed_county_ink.astype(bool) & county_scope
    if np.count_nonzero(county_scope) < 100:
        raise ValueError("Source-derived county scope is undersupported")
    if np.count_nonzero(observed_county_ink) < 20:
        raise ValueError("Observed source county ink is undersupported")
    if not 0.0 < config.observed_ink_calibration_quantile < 0.10:
        raise ValueError("Observed-ink calibration quantile must remain conservative")

    lab = cv2.cvtColor(rgb_source, cv2.COLOR_RGB2LAB).astype(np.float32)
    mean = np.stack(
        [
            cv2.GaussianBlur(
                lab[:, :, channel],
                (0, 0),
                config.gaussian_sigma_px,
                borderType=cv2.BORDER_REFLECT,
            )
            for channel in range(3)
        ],
        axis=2,
    )
    square_mean = np.stack(
        [
            cv2.GaussianBlur(
                lab[:, :, channel] ** 2,
                (0, 0),
                config.gaussian_sigma_px,
                borderType=cv2.BORDER_REFLECT,
            )
            for channel in range(3)
        ],
        axis=2,
    )
    variance = np.maximum(square_mean - mean**2, 0.0)
    ink_lab = np.median(lab[observed_county_ink], axis=0)
    delta = mean - ink_lab[None, None, :]
    contrast = np.sqrt(
        delta[:, :, 0] ** 2
        + config.chroma_weight * (delta[:, :, 1] ** 2 + delta[:, :, 2] ** 2)
    )
    noise = np.sqrt(
        variance[:, :, 0]
        + config.chroma_weight
        * (variance[:, :, 1] + variance[:, :, 2])
    )
    score = contrast / (noise + config.noise_floor)
    calibration = score[observed_county_ink]
    threshold = float(
        np.quantile(calibration, config.observed_ink_calibration_quantile)
    )
    # Positively detected source ink is observable by definition even if the
    # local background shares its color; only unobserved low-contrast pixels
    # remain diagnostic occlusion candidates.
    low_contrast = county_scope & (score < threshold) & ~observed_county_ink
    explicit_exclusion = ~county_scope
    observable = county_scope & ~low_contrast
    self_retention = float(np.mean(observable[observed_county_ink]))
    if self_retention < config.minimum_observed_ink_self_retention:
        raise ValueError(
            "Source-only county observability over-excludes observed county ink"
        )
    diagnostics = {
        "method": (
            "source_only_partial_county_scope_plus_local_lab_"
            "ink_contrast_self_calibration_v1"
        ),
        "county_scope_pixel_count": int(np.count_nonzero(county_scope)),
        "observed_county_ink_pixel_count": int(
            np.count_nonzero(observed_county_ink)
        ),
        "observable_pixel_count": int(np.count_nonzero(observable)),
        "observable_fraction_of_county_scope": float(
            np.mean(observable[county_scope])
        ),
        "low_contrast_exclusion_pixel_count": int(
            np.count_nonzero(low_contrast)
        ),
        "observed_ink_self_retention": self_retention,
        "ink_lab_median": ink_lab.tolist(),
        "score_threshold": threshold,
        "observed_ink_score_percentiles": {
            str(percentile): float(np.percentile(calibration, percentile))
            for percentile in (0, 1, 5, 50, 90)
        },
        "county_scope_sha256": _mask_sha256(county_scope),
        "observed_county_ink_sha256": _mask_sha256(observed_county_ink),
        "observable_sha256": _mask_sha256(observable),
        "low_contrast_exclusion_sha256": _mask_sha256(low_contrast),
        "authority": {
            "source_pixels_used": True,
            "source_derived_county_scope_used": True,
            "source_detected_county_ink_used_for_self_calibration": True,
            "mapbox_geometry_used": False,
            "candidate_transform_used": False,
            "reference_to_source_residual_distance_used": False,
            "manual_or_prior_raster_inputs_used": False,
        },
        "config": config.__dict__,
    }
    return CountyObservabilityEvidence(
        score=score.astype(np.float32),
        observable=observable,
        explicit_exclusion=explicit_exclusion,
        low_contrast_exclusion=low_contrast,
        diagnostics=diagnostics,
    )


def derive_thematic_occlusion(
    rgb_source: np.ndarray,
    county_scope: np.ndarray,
    *,
    config: ThematicOcclusionConfig = ThematicOcclusionConfig(),
) -> ThematicOcclusionEvidence:
    """Derive a diagnostic source-only crop-color occlusion mask.

    The fixed HSV test identifies chromatic thematic pixels, expands them only
    by the measured two-working-pixel printed-stroke radius.  The result is
    *possible* thematic occlusion, never proof that a county boundary is
    absent.  No observed-ink or distance signal changes the mask.  This
    diagnostic cannot relax an alignment gate without separate review.
    """

    if rgb_source.ndim != 3 or rgb_source.shape[2] != 3:
        raise ValueError("Thematic occlusion requires an RGB source raster")
    if county_scope.shape != rgb_source.shape[:2]:
        raise ValueError("County scope must match the source raster")
    radius = config.printed_stroke_radius_px
    if radius < 0 or radius > 3:
        raise ValueError("Thematic expansion may not exceed printed stroke radius")
    hsv = cv2.cvtColor(rgb_source, cv2.COLOR_RGB2HSV)
    thematic = (
        county_scope.astype(bool)
        & (hsv[:, :, 1] >= config.saturation_minimum)
        & (hsv[:, :, 2] >= config.value_minimum)
    )
    kernel = np.ones((2 * radius + 1,) * 2, np.uint8)
    expanded = cv2.dilate(thematic.astype(np.uint8), kernel).astype(bool)
    occluded = county_scope.astype(bool) & expanded
    observable = county_scope.astype(bool) & ~occluded
    diagnostics = {
        "method": "fixed_hsv_thematic_pixels_expanded_by_printed_stroke_radius_v1",
        "thematic_pixel_count": int(np.count_nonzero(thematic)),
        "expanded_thematic_pixel_count": int(np.count_nonzero(expanded)),
        "possible_occlusion_pixel_count": int(np.count_nonzero(occluded)),
        "possible_occlusion_fraction_of_scope": float(
            np.mean(occluded[county_scope.astype(bool)])
        ),
        "observable_sha256": _mask_sha256(observable),
        "occluded_sha256": _mask_sha256(occluded),
        "authority": {
            "source_pixels_used": True,
            "source_derived_county_scope_used": True,
            "positive_or_absent_county_ink_used": False,
            "mapbox_geometry_used": False,
            "candidate_transform_used": False,
            "residual_distance_used": False,
            "diagnostic_only_not_an_acceptance_omission": True,
        },
        "config": config.__dict__,
    }
    return ThematicOcclusionEvidence(
        thematic_pixels=thematic,
        expanded_thematic_pixels=expanded,
        occluded=occluded,
        observable=observable,
        diagnostics=diagnostics,
    )


def evaluate_mapped_county_geography(
    reference_points: np.ndarray,
    mapped_source_points: np.ndarray,
    evidence: CountyObservabilityEvidence,
    source_county_ink: np.ndarray,
    *,
    config: CountyObservabilityConfig = CountyObservabilityConfig(),
) -> dict[str, Any]:
    """Measure a frozen geography against source-only observability evidence."""

    reference_points = np.asarray(reference_points, dtype=np.float64)
    mapped_source_points = np.asarray(mapped_source_points, dtype=np.float64)
    if (
        reference_points.shape != mapped_source_points.shape
        or reference_points.ndim != 2
        or reference_points.shape[1] != 2
    ):
        raise ValueError("Mapped county evaluation requires equal Nx2 arrays")
    if source_county_ink.shape != evidence.observable.shape:
        raise ValueError("Source county ink must match observability evidence")
    height, width = evidence.observable.shape
    rounded = np.rint(mapped_source_points).astype(np.int32)
    inside = (
        (rounded[:, 0] >= 0)
        & (rounded[:, 0] < width)
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < height)
    )
    observable = np.zeros(len(rounded), dtype=bool)
    low_contrast = np.zeros(len(rounded), dtype=bool)
    explicit_exclusion = np.zeros(len(rounded), dtype=bool)
    observable[inside] = evidence.observable[
        rounded[inside, 1], rounded[inside, 0]
    ]
    low_contrast[inside] = evidence.low_contrast_exclusion[
        rounded[inside, 1], rounded[inside, 0]
    ]
    explicit_exclusion[inside] = evidence.explicit_exclusion[
        rounded[inside, 1], rounded[inside, 0]
    ]
    distance = distance_transform_edt(~source_county_ink.astype(bool))
    values = np.full(len(rounded), np.nan, dtype=np.float64)
    values[inside] = distance[rounded[inside, 1], rounded[inside, 0]]
    near = observable & (values <= config.evaluation_distance_px)
    observable_far = observable & (values > config.evaluation_distance_px)
    mapped_unobservable = inside & ~observable
    count = len(rounded)

    def fraction(mask: np.ndarray) -> float:
        return float(np.count_nonzero(mask) / count) if count else 0.0

    observable_values = values[observable]
    return {
        "point_count": count,
        "inside_source_canvas_count": int(np.count_nonzero(inside)),
        "off_canvas_count": int(np.count_nonzero(~inside)),
        "source_observable_count": int(np.count_nonzero(observable)),
        "source_observable_fraction": fraction(observable),
        "source_explicitly_unobservable_count": int(
            np.count_nonzero(mapped_unobservable)
        ),
        "source_explicitly_unobservable_fraction": fraction(mapped_unobservable),
        "low_contrast_count": int(np.count_nonzero(low_contrast)),
        "explicit_scope_exclusion_count": int(
            np.count_nonzero(explicit_exclusion)
        ),
        "near_detected_source_topology_count": int(np.count_nonzero(near)),
        "near_detected_source_topology_fraction": fraction(near),
        "observable_but_far_from_detected_topology_count": int(
            np.count_nonzero(observable_far)
        ),
        "observable_but_far_from_detected_topology_fraction": fraction(
            observable_far
        ),
        "within_distance_fraction_among_observable": (
            float(np.mean(observable_values <= config.evaluation_distance_px))
            if len(observable_values)
            else None
        ),
        "median_distance_px_among_observable": (
            float(np.median(observable_values)) if len(observable_values) else None
        ),
        "p90_distance_px_among_observable": (
            float(np.quantile(observable_values, 0.90))
            if len(observable_values)
            else None
        ),
        "interpretation": {
            "unobservable_points_may_be_explained_by_partial_extent_or_contrast": True,
            "observable_far_points_are_not_explained_by_source_observability": True,
            "observable_far_points_may_be_alignment_or_line_identity_error": True,
            "distance_does_not_change_observability": True,
        },
    }


def render_county_observability_overlay(
    rgb_source: np.ndarray,
    evidence: CountyObservabilityEvidence,
    observed_county_ink: np.ndarray,
) -> np.ndarray:
    """Render explicit exclusions, low-contrast areas, and detected ink."""

    overlay = rgb_source.copy()
    overlay[evidence.explicit_exclusion] = (
        0.65 * overlay[evidence.explicit_exclusion]
        + 0.35 * np.asarray((20, 40, 70))
    ).astype(np.uint8)
    overlay[evidence.low_contrast_exclusion] = (255, 150, 0)
    overlay[observed_county_ink.astype(bool)] = (255, 0, 220)
    return overlay
