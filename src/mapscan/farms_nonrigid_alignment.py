"""Strict, source-clean residual alignment for the partial farms raster.

The farms source contains only part of California.  Its strongest state-side
evidence is the printed Nevada boundary plus a Pacific-connected coastline;
the upper-right legend hides the Tahoe segment.  This adapter starts from the
source-only farms topology hypothesis, fits a fresh buffered training-only
Mapbox seed, and permits one bounded smooth residual warp. County, Pacific,
and Bay evidence are split into model-selection and retained strict acceptance
channels before fitting. Earlier experiments inspected the broad geographies,
so the latter are confirmation gates rather than statistically blind holdouts.

Nothing in this module accepts a prior transform, manual control, ``county.png``
or a historical MapScan artifact.  The only reference authority is the pinned
Mapbox manifest supplied to :func:`run_farms_nonrigid_throwaway`.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
from scipy.optimize import differential_evolution
from scipy.spatial import cKDTree
from shapely.geometry import Polygon
from shapely.ops import unary_union

from .automatic_alignment_loop import (
    AlignmentSourceHypothesis,
    PinnedMapboxReference,
    ProjectionContext,
    SourceSemanticEvidence,
    _dominant_neutral_pacific,
    _load_mask,
    _matrix_from_parameters,
    _project_reference_points,
    _projection_contexts,
    _projection_transform_contract,
    _resize_working,
    _sha256,
    _source_semantic_evidence,
    _symmetric_line_report,
    _transform,
    load_pinned_mapbox_reference,
)
from .elevation_nonrigid_alignment import (
    CompactResidualWarp,
    fit_compact_residual_warp,
)
from .farms_partial_topology import (
    FarmsPartialTopology,
    derive_farms_partial_topology,
    render_farms_county_scope_overlay,
)
from .farms_native_county_topology import (
    FarmsNativeCountyTopology,
    compare_native_and_working_topology,
    derive_farms_native_county_topology,
)


EXPERIMENT_KIND = "throwaway_farms_partial_nonrigid_v3"
SOURCE_ADAPTER_ID = "farms-partial-california-topology-v2"
EXPECTED_FARMS_SOURCE_SHA256 = (
    "ff58a54049c93475959e5308649a8bcba1728f1d15b97a72a106172edd9a3561"
)
EXPECTED_MAPBOX_V2_MANIFEST_SHA256 = (
    "5f3b5269ce40193037084383c06f3b9f7e74abdf80ab3c4bdac643a024141f05"
)
EXPECTED_MAPBOX_V2_GRID = (3920, 3398)


@dataclass(frozen=True)
class FarmsNonrigidConfig:
    working_max_dimension: int = 900
    projection_ids: tuple[str, ...] = (
        "web_mercator",
        "california_albers",
        "california_lambert_conformal_conic",
    )
    seed_scale_modes: tuple[tuple[str, float, float, int, str], ...] = (
        ("partial-low", 0.55, 1.0, 3513, "source_partial_extent"),
        ("partial-high", 1.2, 2.4, 3512, "source_partial_extent"),
        ("canvas-high", 1.2, 2.4, 3512, "canvas_bounds_hypothesis"),
    )
    fine_bin_reference_px: int = 100
    microsegment_reference_px: int = 12
    maximum_training_match_px: float = 35.0
    residual_activation_px: float = 3.0
    terminal_high_run_window_px: float = 180.0
    kernel_radii_reference_px: tuple[float, ...] = (95.0, 100.0, 105.0, 110.0, 115.0)
    ridge_values: tuple[float, ...] = (0.35, 0.375, 0.40, 0.425, 0.45)
    maximum_residual_displacement_working_px: float = 20.0
    jacobian_grid: tuple[int, int] = (60, 70)
    minimum_jacobian_ratio: float = 0.15
    maximum_jacobian_ratio: float = 2.0
    maximum_local_condition_number: float = 4.0
    balanced_cells: tuple[int, int] = (6, 6)
    balanced_p90_limit_px: float = 12.0
    balanced_minimum_cells: int = 8
    balanced_minimum_cell_pass_fraction: float = 0.70
    balanced_minimum_axis_pass_fraction: float = 0.50
    minimum_pixels_per_balanced_cell: int = 20
    state_median_limit_px: float = 4.5
    state_within_8_minimum: float = 0.75
    state_f1_minimum: float = 0.42
    county_median_limit_px: float = 7.0
    county_within_8_minimum: float = 0.58
    county_f1_minimum: float = 0.32
    county_validation_balanced_cells: tuple[int, int] = (6, 6)
    county_validation_balanced_minimum_visible_count: int = 20
    county_validation_balanced_minimum_visible_cells: int = 6
    county_validation_balanced_minimum_visible_rows: int = 3
    county_validation_balanced_minimum_visible_columns: int = 3
    county_validation_balanced_p90_limit_px: float = 12.0
    county_validation_balanced_minimum_cell_pass_fraction: float = 0.70
    county_validation_balanced_minimum_axis_pass_fraction: float = 0.50
    minimum_visible_land_fraction: float = 0.20
    minimum_visible_land_containment: float = 0.90
    admin_holdout_p90_limit_px: float = 12.0
    admin_holdout_within_8_minimum: float = 0.75
    holdout_minimum_visible_count: int = 20
    holdout_minimum_visible_fraction: float = 0.25
    bay_holdout_minimum_visible_fraction: float = 0.60
    bay_holdout_p90_limit_px: float = 12.0
    bay_holdout_within_8_minimum: float = 0.75
    seed_partition_cells: tuple[int, int] = (10, 10)
    seed_partition_boundary_guard_px: int = 7
    seed_minimum_visible_fraction_per_cell: float = 0.25
    seed_minimum_visible_cells: int = 4
    seed_minimum_visible_rows: int = 3
    seed_minimum_visible_columns: int = 3
    seed_county_training_weight: float = 0.60
    seed_water_training_weight: float = 0.25
    seed_water_training_corridor_working_px: float = 80.0
    seed_water_heldout_exclusion_working_px: float = 24.0
    seed_water_minimum_visible_count: int = 20
    seed_water_minimum_visible_fraction: float = 0.60
    seed_water_undersupport_penalty: float = 500.0
    seed_water_visibility_penalty: float = 50.0
    seed_water_training_maximum_points: int = 2500
    seed_water_heldout_maximum_points: int = 8000
    seed_optimizer_iterations: int = 90
    seed_optimizer_population: int = 12
    seed_optimizer_seeds: tuple[int, ...] = (3511, 3512, 3513)
    residual_county_bin_reference_px: int = 260
    residual_county_minimum_bin_support: int = 20
    residual_county_maximum_match_px: float = 18.0
    county_residual_kernel_radii_reference_px: tuple[float, ...] = (
        360.0,
        440.0,
        520.0,
        600.0,
    )
    county_residual_ridge_values: tuple[float, ...] = (0.5, 1.0, 2.0)
    source_extent_line_corridor_working_px: int = 6
    water_training_guard_reference_px: int = 120
    water_training_maximum_match_working_px: float = 18.0
    water_training_bin_reference_px: int = 120
    water_training_minimum_bin_support: int = 12


@dataclass(frozen=True)
class FarmsSourceEvidence:
    rgb_original: np.ndarray
    rgb_working: np.ndarray
    working_scale: float
    pacific: np.ndarray
    native_pacific_coast_edge: np.ndarray
    native_pacific_coast_edge_original: np.ndarray
    native_water: np.ndarray
    native_water_edge: np.ndarray
    native_internal_water: np.ndarray
    native_internal_water_edge: np.ndarray
    native_water_original: np.ndarray
    native_water_edge_original: np.ndarray
    native_internal_water_original: np.ndarray
    native_internal_water_edge_original: np.ndarray
    water_diagnostics: Mapping[str, Any]
    topology: FarmsPartialTopology
    native_county_topology: FarmsNativeCountyTopology
    working_rgb_scoped_county_topology: np.ndarray
    working_rgb_secondary_county_topology: np.ndarray
    county_topology: np.ndarray
    county_topology_diagnostics: Mapping[str, Any]
    semantic: SourceSemanticEvidence
    hypothesis: AlignmentSourceHypothesis


@dataclass(frozen=True)
class MapboxFarmsSemantics:
    california_admin: np.ndarray
    pacific_coast: np.ndarray
    primary: np.ndarray
    unsupported_inland_water: np.ndarray
    diagnostics: Mapping[str, Any]


@dataclass(frozen=True)
class FarmsNonrigidResult:
    status: str
    stop_reason: str
    report_path: Path
    transform_path: Path | None
    artifact_paths: tuple[Path, ...]


@dataclass(frozen=True)
class FarmsValidationPreflightResult:
    status: str
    stop_reason: str
    report_path: Path
    frozen_candidate_path: Path | None
    artifact_paths: tuple[Path, ...]


@dataclass(frozen=True)
class _FarmsValidationState:
    source: FarmsSourceEvidence
    reference: PinnedMapboxReference
    semantics: MapboxFarmsSemantics
    projection: ProjectionContext
    seed_matrix: np.ndarray
    seed_diagnostics: Mapping[str, Any]
    county_partitions: Mapping[str, np.ndarray]
    county_validation_cells: Mapping[str, np.ndarray]
    split_points: Mapping[str, np.ndarray]
    control_diagnostics: Mapping[str, Any]
    source_assignment_masks: Mapping[str, np.ndarray]
    source_observable_extents: Mapping[str, np.ndarray]
    source_observable_extent_diagnostics: Mapping[str, Any]
    named_pacific_masks: Mapping[str, np.ndarray]
    candidates: tuple[Mapping[str, Any], ...]
    selected: Mapping[str, Any] | None


def _json_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _mask_bbox(mask: np.ndarray) -> list[int] | None:
    y, x = np.nonzero(mask)
    if not len(x):
        return None
    return [
        int(x.min()),
        int(y.min()),
        int(x.max() - x.min() + 1),
        int(y.max() - y.min() + 1),
    ]


def _source_observable_extents(
    source: FarmsSourceEvidence,
    config: FarmsNonrigidConfig,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Derive fixed partial-coverage domains from source evidence only.

    The farms raster is a cropped regional map, not a complete California
    silhouette.  A target feature is observable only when its transformed
    point falls in the source-derived map support.  Canvas bounds alone are
    insufficient because the same canvas also contains a legend occlusion and
    page margins.  Nevada and the Pacific remain observed negative evidence.
    The domains here do not depend on Mapbox, a candidate
    transform, prior outputs, or manual coordinates.
    """

    height, width = source.rgb_working.shape[:2]
    margin = max(3, round(min(height, width) * 0.012))
    page = np.ones((height, width), dtype=bool)
    page[:margin] = False
    page[-margin:] = False
    page[:, :margin] = False
    page[:, -margin:] = False
    page &= ~source.topology.layout_exclusion

    corridor_width = int(config.source_extent_line_corridor_working_px)
    if corridor_width < 1:
        raise ValueError("Farms source extent corridor must be positive")
    kernel = np.ones((2 * corridor_width + 1,) * 2, np.uint8)
    state_corridor = cv2.dilate(
        source.topology.state_coast.astype(np.uint8), kernel
    ).astype(bool)
    water_corridor = cv2.dilate(
        source.native_water_edge.astype(np.uint8), kernel
    ).astype(bool)
    internal_corridor = cv2.dilate(
        source.county_topology.astype(np.uint8), kernel
    ).astype(bool)

    # The source's Pacific and adjacent-state panels are observed geography,
    # not missing data.  They remain in the state/coast validation domain so a
    # wrong transform cannot hide drift by mapping California geometry into
    # ocean or Nevada.  County visibility is narrower: below the observed
    # Nevada exit the partial source does not expose a trustworthy Colorado
    # boundary, so only the source-derived, omission-only county scope may
    # authorize Mapbox county evidence.  The omitted lower-right network is
    # logged explicitly rather than becoming an accidental negative match.
    map_panel = page
    county_lines = source.topology.county_scope & page
    california_positive_support = (
        source.topology.foreground_interior | state_corridor | internal_corridor
    ) & page & ~source.pacific
    california_positive_support &= ~source.topology.neighboring_region
    california_positive_support &= ~source.native_internal_water

    masks = {
        "map_panel": map_panel,
        "california_positive_support": california_positive_support,
        "county_lines": county_lines,
        "state_admin": map_panel.copy(),
        "pacific_coast": map_panel.copy(),
        "named_bay": map_panel.copy(),
    }
    diagnostics = {
        "method": "source_only_inset_frame_minus_detected_layout_occlusion",
        "manual_inputs_used": False,
        "mapbox_used": False,
        "prior_run_inputs_used": False,
        "working_shape": [height, width],
        "page_margin_working_px": margin,
        "line_corridor_working_px": corridor_width,
        "layout_exclusion_pixels": int(
            np.count_nonzero(source.topology.layout_exclusion)
        ),
        "neighboring_region_pixels": int(
            np.count_nonzero(source.topology.neighboring_region)
        ),
        "neighboring_region_is_observed_negative_evidence": True,
        "county_line_visibility_method": (
            "source_only_southern_clip_and_ambiguous_lower_right_omission"
        ),
        "ambiguous_lower_right_county_evidence_policy": "omitted_with_warning",
        "ambiguous_lower_right_county_evidence_pixels": int(
            np.count_nonzero(source.topology.ambiguous_topology_exclusion)
        ),
        "pacific_pixels": int(np.count_nonzero(source.pacific)),
        "masks": {
            name: {
                "pixel_count": int(np.count_nonzero(mask)),
                "canvas_fraction": float(np.mean(mask)),
                "bbox_working_px": _mask_bbox(mask),
            }
            for name, mask in masks.items()
        },
    }
    if min(np.count_nonzero(mask) for mask in masks.values()) < 100:
        raise ValueError("Source-derived farms observable extent is undersupported")
    return masks, diagnostics


def _source_extent_membership(
    rounded_source_points: np.ndarray,
    observable_extent: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return canvas membership and source-observable membership separately."""

    if rounded_source_points.ndim != 2 or rounded_source_points.shape[1] != 2:
        raise ValueError("Source points must have shape (N, 2)")
    height, width = observable_extent.shape
    inside_canvas = (
        (rounded_source_points[:, 0] >= 0)
        & (rounded_source_points[:, 0] < width)
        & (rounded_source_points[:, 1] >= 0)
        & (rounded_source_points[:, 1] < height)
    )
    visible = inside_canvas.copy()
    canvas_indices = np.flatnonzero(inside_canvas)
    visible[canvas_indices] &= observable_extent[
        rounded_source_points[canvas_indices, 1],
        rounded_source_points[canvas_indices, 0],
    ]
    return inside_canvas, visible


def _require_pinned_v2_reference(reference: PinnedMapboxReference) -> None:
    """Reject every reference except the immutable corrected Mapbox-v2 pin."""

    manifest_sha256 = _sha256(reference.manifest_path)
    shape = (int(reference.grid["height"]), int(reference.grid["width"]))
    if manifest_sha256 != EXPECTED_MAPBOX_V2_MANIFEST_SHA256:
        raise ValueError(
            "The farms adapter requires the exact corrected Mapbox-v2 manifest"
        )
    if shape != EXPECTED_MAPBOX_V2_GRID:
        raise ValueError("The farms adapter requires the exact corrected Mapbox-v2 grid")
    if reference.pin.get("manifest_sha256") != manifest_sha256:
        raise ValueError("The loaded Mapbox reference pin disagrees with its manifest bytes")


def _topology_preserving_binary_reduce(
    mask: np.ndarray, size: tuple[int, int]
) -> np.ndarray:
    """Area-reduce while retaining every positive native-resolution pixel."""

    return cv2.resize(
        mask.astype(np.float32), size, interpolation=cv2.INTER_AREA
    ) > 0


def _load_source(source_path: Path, config: FarmsNonrigidConfig) -> FarmsSourceEvidence:
    image = Image.open(source_path)
    if getattr(image, "n_frames", 1) != 1:
        raise ValueError("The farms adapter requires a single-frame pristine raster")
    original = np.asarray(image.convert("RGB"))
    working, scale = _resize_working(original, config.working_max_dimension)
    generic = _source_semantic_evidence(working)
    pacific_result = _dominant_neutral_pacific(working)
    if pacific_result is None:
        raise ValueError("No source-only neutral Pacific component was detected")
    pacific, pacific_diagnostics = pacific_result
    pacific_color = np.median(working[pacific].astype(np.float64), axis=0)
    neutral_distance = np.linalg.norm(
        original.astype(np.float64) - pacific_color[None, None, :], axis=2
    )
    native_water_candidate = neutral_distance <= 6.0
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        native_water_candidate.astype(np.uint8), 8
    )
    original_height, original_width = original.shape[:2]
    if component_count <= 1:
        raise ValueError("Native-resolution water has no Pacific component")
    # The printed map has a black neatline, so ocean fill need not literally
    # touch the image edge.  The source-derived anchor color's unique largest
    # connected component is the Pacific, mirroring the working-raster check.
    pacific_label = int(1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    _pacific_area = int(stats[pacific_label, cv2.CC_STAT_AREA])
    pacific_original = labels == pacific_label
    pacific_x, pacific_y, pacific_width, pacific_height, _ = map(
        int, stats[pacific_label]
    )
    native_pacific_coast_edge_original = cv2.morphologyEx(
        pacific_original.astype(np.uint8),
        cv2.MORPH_GRADIENT,
        np.ones((3, 3), np.uint8),
    ).astype(bool)
    # Suppress both the page edge and the printed inset map neatline at the
    # Pacific component's long left/bottom sides. Hole boundaries remain, so
    # detached islands survive even though RETR_EXTERNAL omitted them.
    page_margin = max(4, round(3.0 / max(scale, 1e-12)))
    native_pacific_coast_edge_original[:page_margin, :] = False
    native_pacific_coast_edge_original[-page_margin:, :] = False
    native_pacific_coast_edge_original[:, :page_margin] = False
    native_pacific_coast_edge_original[:, -page_margin:] = False
    component_margin = max(4, round(3.0 / max(scale, 1e-12)))
    native_pacific_coast_edge_original[:, : pacific_x + component_margin] = False
    native_pacific_coast_edge_original[
        pacific_y + pacific_height - component_margin :, :
    ] = False
    pacific_distance = cv2.distanceTransform(
        (~pacific_original).astype(np.uint8), cv2.DIST_L2, 3
    )
    native_water_original = pacific_original.copy()
    native_internal_water_original = np.zeros_like(pacific_original)
    internal_components: list[dict[str, Any]] = []
    minimum_component_pixels = max(25, round(25.0 / max(scale * scale, 1e-12)))
    maximum_coastal_gap = 40.0 / max(scale, 1e-12)
    for label in range(1, component_count):
        if label == pacific_label:
            continue
        x, y, width, height, area = map(int, stats[label])
        if area < minimum_component_pixels:
            continue
        local_labels = labels[y : y + height, x : x + width]
        local_component = local_labels == label
        local_distance = pacific_distance[y : y + height, x : x + width]
        minimum_pacific_distance = float(np.min(local_distance[local_component]))
        coastal = minimum_pacific_distance <= maximum_coastal_gap
        if coastal:
            target = native_water_original[y : y + height, x : x + width]
            target[local_component] = True
            target = native_internal_water_original[y : y + height, x : x + width]
            target[local_component] = True
        internal_components.append(
            {
                "bbox_original": [x, y, width, height],
                "bbox_working": [
                    round(x * scale),
                    round(y * scale),
                    max(1, round(width * scale)),
                    max(1, round(height * scale)),
                ],
                "pixel_count": area,
                "minimum_pacific_distance_original_px": minimum_pacific_distance,
                "minimum_pacific_distance_working_px": minimum_pacific_distance * scale,
                "coastal_internal_water": coastal,
            }
        )
    coastal_components = [
        item for item in internal_components if item["coastal_internal_water"]
    ]
    coastal_span = 0
    if coastal_components:
        minimum_x = min(item["bbox_original"][0] for item in coastal_components)
        maximum_x = max(
            item["bbox_original"][0] + item["bbox_original"][2]
            for item in coastal_components
        )
        coastal_span = round((maximum_x - minimum_x) * scale)
    native_water_edge_original = cv2.morphologyEx(
        native_water_original.astype(np.uint8),
        cv2.MORPH_GRADIENT,
        np.ones((3, 3), np.uint8),
    ).astype(bool)
    native_internal_water_edge_original = cv2.morphologyEx(
        native_internal_water_original.astype(np.uint8),
        cv2.MORPH_GRADIENT,
        np.ones((3, 3), np.uint8),
    ).astype(bool)
    working_size = (working.shape[1], working.shape[0])
    native_pacific_coast_edge = _topology_preserving_binary_reduce(
        native_pacific_coast_edge_original, working_size
    )
    native_water = _topology_preserving_binary_reduce(
        native_water_original, working_size
    )
    native_internal_water = _topology_preserving_binary_reduce(
        native_internal_water_original, working_size
    )
    native_water_edge = cv2.morphologyEx(
        native_water.astype(np.uint8),
        cv2.MORPH_GRADIENT,
        np.ones((3, 3), np.uint8),
    ).astype(bool)
    native_internal_water_edge = cv2.morphologyEx(
        native_internal_water.astype(np.uint8),
        cv2.MORPH_GRADIENT,
        np.ones((3, 3), np.uint8),
    ).astype(bool)
    water_diagnostics = {
        "method": "native_resolution_source_pacific_color_components_then_topology_preserving_binary_reduction",
        "native_source_shape": [original_height, original_width],
        "alignment_working_shape": list(working.shape[:2]),
        "pacific_anchor_rgb": pacific_color.tolist(),
        "rgb_distance_limit": 6.0,
        "minimum_component_pixels_original": minimum_component_pixels,
        "maximum_coastal_gap_original_px": maximum_coastal_gap,
        "internal_components": internal_components,
        "native_pacific_coast_edge_original_pixels": int(
            np.count_nonzero(native_pacific_coast_edge_original)
        ),
        "native_pacific_coast_edge_working_pixels": int(
            np.count_nonzero(native_pacific_coast_edge)
        ),
        "native_pacific_coast_page_margin_original_px": page_margin,
        "native_pacific_component_bbox_original": [
            pacific_x,
            pacific_y,
            pacific_width,
            pacific_height,
        ],
        "native_pacific_component_neatline_margin_original_px": component_margin,
        "inset_left_and_bottom_neatline_suppressed": True,
        "detached_island_hole_boundaries_preserved": True,
        "coastal_internal_component_count": len(coastal_components),
        "coastal_internal_horizontal_span_working_px": coastal_span,
        "san_francisco_bay_observability": (
            "supported" if len(coastal_components) >= 1 else "unsupported"
        ),
        "suisun_delta_observability": "supported_native_resolution_strict",
        "unsupported_remainder_policy": "none_for_native_observed_components",
        "binary_reduction": "float32_cv2_inter_area_then_any_positive_preserves_thin_native_water",
        "classification_frozen_before_mapbox_fit": True,
        "mapbox_or_candidate_pixels_used": False,
    }
    topology = derive_farms_partial_topology(working, pacific)
    native_county_topology = derive_farms_native_county_topology(
        original,
        county_scope_working=topology.county_scope,
        state_coast_working=topology.state_coast,
    )
    working_rgb_scoped_county_topology = (
        topology.internal_topology & topology.county_scope
    )
    native_county_corridor = cv2.dilate(
        native_county_topology.working_ink.astype(np.uint8),
        np.ones((5, 5), np.uint8),
    ).astype(bool)
    working_rgb_secondary_county_topology = (
        working_rgb_scoped_county_topology & native_county_corridor
    )
    county_topology = (
        native_county_topology.working_ink
        | working_rgb_secondary_county_topology
    )
    topology_comparison = compare_native_and_working_topology(
        native_county_topology.working_ink,
        working_rgb_scoped_county_topology,
        corridor_px=2,
    )
    county_topology_diagnostics = {
        "method": (
            "native_resolution_primary_plus_working_rgb_secondary_only_within_"
            "two_working_pixels_of_native_ink"
        ),
        "native_primary_pixel_count": int(
            np.count_nonzero(native_county_topology.working_ink)
        ),
        "working_rgb_scoped_pixel_count": int(
            np.count_nonzero(working_rgb_scoped_county_topology)
        ),
        "working_rgb_secondary_pixel_count": int(
            np.count_nonzero(working_rgb_secondary_county_topology)
        ),
        "combined_county_pixel_count": int(np.count_nonzero(county_topology)),
        "secondary_native_corridor_working_px": 2,
        "unsupported_working_rgb_pixels_raw_union_forbidden": True,
        "native": dict(native_county_topology.diagnostics),
        "comparison": topology_comparison,
        "mapbox_or_candidate_pixels_used": False,
        "validation_or_retained_acceptance_inputs_used": False,
    }
    semantic = SourceSemanticEvidence(
        state_coast=topology.state_coast,
        counties=county_topology,
        dark_cartographic_ink=generic.dark_cartographic_ink,
        border_connected_water=pacific,
        foreground_interior=topology.foreground_interior,
        foreground_boundary=topology.foreground_boundary,
        source_adapter_id=SOURCE_ADAPTER_ID,
    )
    hypothesis = AlignmentSourceHypothesis(
        id="source-family-farms-partial-california-topology-v2",
        variant_kind="source_family_semantics",
        semantic=semantic,
        source_only_score=None,
        roi_working=(0, 0, working.shape[1], working.shape[0]),
        generator_hypothesis_id=None,
        diagnostics={
            "source_adapter_id": SOURCE_ADAPTER_ID,
            "pacific": pacific_diagnostics,
            "topology": dict(topology.diagnostics),
            "native_county_topology": county_topology_diagnostics,
            "mapbox_used_for_source_hypothesis_construction": False,
            "manual_inputs_used": False,
            "prior_run_inputs_used": False,
            "county_png_used": False,
        },
    )
    return FarmsSourceEvidence(
        original,
        working,
        scale,
        pacific,
        native_pacific_coast_edge,
        native_pacific_coast_edge_original,
        native_water,
        native_water_edge,
        native_internal_water,
        native_internal_water_edge,
        native_water_original,
        native_water_edge_original,
        native_internal_water_original,
        native_internal_water_edge_original,
        water_diagnostics,
        topology,
        native_county_topology,
        working_rgb_scoped_county_topology,
        working_rgb_secondary_county_topology,
        county_topology,
        county_topology_diagnostics,
        semantic,
        hypothesis,
    )


def build_mapbox_farms_semantics(reference: PinnedMapboxReference) -> MapboxFarmsSemantics:
    """Separate CA admin and Pacific-connected coast from inland water edges."""

    manifest = json.loads(reference.manifest_path.read_text())
    artifact = manifest.get("artifacts", {}).get("admin_barrier_mask")
    if not isinstance(artifact, Mapping):
        raise ValueError("Pinned Mapbox v2 reference lacks its admin barrier artifact")
    admin = _load_mask(reference.root, artifact, alpha=False)
    # The raw tile crop contains the Oregon/Washington top border.  Only admin
    # strokes colocated with the independently pinned California perimeter are
    # eligible; this selection contains no source pixels.
    state_corridor = cv2.dilate(
        reference.state_coast.astype(np.uint8), np.ones((11, 11), np.uint8)
    ).astype(bool)
    california_admin = admin & state_corridor

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        reference.water.astype(np.uint8), 8
    )
    height, width = reference.water.shape
    border_components: list[tuple[int, int]] = []
    for label in range(1, count):
        x, y, component_width, component_height, area = map(int, stats[label])
        if (
            x == 0
            or y == 0
            or x + component_width == width
            or y + component_height == height
        ):
            border_components.append((area, label))
    if not border_components:
        raise ValueError("Pinned Mapbox water has no border-connected Pacific component")
    _area, pacific_label = max(border_components)
    pacific_water = labels == pacific_label
    pacific_edge = cv2.morphologyEx(
        pacific_water.astype(np.uint8),
        cv2.MORPH_GRADIENT,
        np.ones((3, 3), np.uint8),
    ).astype(bool)
    pacific_coast = reference.state_coast & cv2.dilate(
        pacific_edge.astype(np.uint8), np.ones((11, 11), np.uint8)
    ).astype(bool)
    primary = california_admin | pacific_coast
    unsupported = reference.state_coast & ~primary
    if np.count_nonzero(primary) < 1000:
        raise ValueError("Mapbox farms semantic primary is unexpectedly sparse")
    return MapboxFarmsSemantics(
        california_admin=california_admin,
        pacific_coast=pacific_coast,
        primary=primary,
        unsupported_inland_water=unsupported,
        diagnostics={
            "method": "pinned_ca_admin_plus_largest_border_connected_pacific_coast",
            "california_admin_pixels": int(np.count_nonzero(california_admin)),
            "pacific_coast_pixels": int(np.count_nonzero(pacific_coast)),
            "primary_pixels": int(np.count_nonzero(primary)),
            "unsupported_inland_water_pixels": int(np.count_nonzero(unsupported)),
            "tahoe_policy": "unsupported_omitted_with_warning_not_fit_or_strict_support",
            "bay_suisun_delta_policy": (
                "native_resolution_source_water_supported_and_strictly_validated"
            ),
            "source_pixels_used_for_reference_channel_construction": False,
        },
    )


def _east_boundary(topology: FarmsPartialTopology) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        topology.state_coast.astype(np.uint8), 8
    )
    result = np.zeros(topology.state_coast.shape, dtype=bool)
    width = topology.state_coast.shape[1]
    for label in range(1, count):
        x, _y, _component_width, _component_height, _area = map(int, stats[label])
        if x > width * 0.55:
            result |= labels == label
    if np.count_nonzero(result) < 500:
        raise ValueError("Source-only farms Nevada component is missing")
    return result


def _reference_cell_mask(
    shape: tuple[int, int],
    *,
    cells: tuple[int, int],
    predicate: Any,
) -> np.ndarray:
    rows, columns = cells
    height, width = shape
    result = np.zeros(shape, dtype=bool)
    for row in range(rows):
        for column in range(columns):
            if not predicate(row, column):
                continue
            y1, y2 = round(row * height / rows), round((row + 1) * height / rows)
            x1, x2 = round(column * width / columns), round((column + 1) * width / columns)
            result[y1:y2, x1:x2] = True
    return result


def _sample_mask_points(mask: np.ndarray, maximum: int) -> np.ndarray:
    y, x = np.nonzero(mask)
    points = np.column_stack((x, y)).astype(np.float64)
    if len(points) > maximum:
        indices = np.linspace(0, len(points) - 1, maximum).round().astype(int)
        points = points[indices]
    return points


def _water_channel_source_line(
    source: FarmsSourceEvidence, name: str
) -> tuple[np.ndarray, str]:
    if name == "outer_pacific_training_microsegments":
        return source.native_pacific_coast_edge, "native_pacific_coast_edge"
    if name == "observed_suisun_delta_training_microsegments":
        return source.native_internal_water_edge, "native_internal_water_edge"
    raise ValueError(f"Unknown farms water-training channel: {name}")


def _water_channel_heldout_union(
    named_pacific_masks: Mapping[str, np.ndarray], training_name: str
) -> np.ndarray:
    sample = next(iter(named_pacific_masks.values()))
    heldout = np.zeros(sample.shape, dtype=bool)
    outer = training_name.startswith("outer_pacific")
    for name, mask in named_pacific_masks.items():
        if not (
            name.endswith("validation_microsegments")
            or name.endswith("test_microsegments")
        ):
            continue
        if name.startswith("outer_pacific") == outer:
            heldout |= mask
    if not np.any(heldout):
        raise ValueError(f"Water-training channel has no heldout geometry: {training_name}")
    return heldout


def _prepare_water_seed_training_context(
    reference: PinnedMapboxReference,
    source: FarmsSourceEvidence,
    projection: ProjectionContext,
    named_pacific_masks: Mapping[str, np.ndarray],
    config: FarmsNonrigidConfig,
) -> dict[str, Any]:
    """Freeze train-only water evidence for global seed fitting.

    Validation and retained-test pixels contribute only buffered exclusion
    geometry.  Their residuals and pass/fail metrics are not evaluated here.
    """

    source_extents, _ = _source_observable_extents(source, config)
    channels: list[dict[str, Any]] = []
    training_names = sorted(
        name
        for name in named_pacific_masks
        if name.endswith("training_microsegments")
    )
    for name in training_names:
        source_line, source_line_id = _water_channel_source_line(source, name)
        heldout_mask = _water_channel_heldout_union(named_pacific_masks, name)
        training_points = _sample_mask_points(
            named_pacific_masks[name], config.seed_water_training_maximum_points
        )
        heldout_points = _sample_mask_points(
            heldout_mask, config.seed_water_heldout_maximum_points
        )
        source_y, source_x = np.nonzero(source_line)
        if (
            len(training_points) < config.seed_water_minimum_visible_count
            or len(heldout_points) < config.seed_water_minimum_visible_count
            or len(source_x) < config.seed_water_minimum_visible_count
        ):
            raise ValueError(f"Water seed channel is undersupported: {name}")
        channels.append(
            {
                "name": name,
                "source_line_id": source_line_id,
                "source_line_sha256": hashlib.sha256(
                    np.ascontiguousarray(source_line.astype(np.uint8)).tobytes()
                ).hexdigest(),
                "source_points": np.column_stack((source_x, source_y)).astype(
                    np.float64
                ),
                "training_reference_pixel_count": int(
                    np.count_nonzero(named_pacific_masks[name])
                ),
                "training_reference_mask_sha256": hashlib.sha256(
                    np.ascontiguousarray(
                        named_pacific_masks[name].astype(np.uint8)
                    ).tobytes()
                ).hexdigest(),
                "training_normalized": _project_reference_points(
                    training_points, projection, reference.grid
                ),
                "heldout_reference_pixel_count": int(np.count_nonzero(heldout_mask)),
                "heldout_reference_mask_sha256": hashlib.sha256(
                    np.ascontiguousarray(heldout_mask.astype(np.uint8)).tobytes()
                ).hexdigest(),
                "heldout_normalized": _project_reference_points(
                    heldout_points, projection, reference.grid
                ),
            }
        )
    if {item["name"] for item in channels} != {
        "outer_pacific_training_microsegments",
        "observed_suisun_delta_training_microsegments",
    }:
        raise ValueError("Farms water seed requires exactly outer and Suisun training")
    return {
        "method": "train_only_outer_pacific_and_internal_water_seed_evidence",
        "map_panel": source_extents["map_panel"],
        "source_shape": source.rgb_working.shape[:2],
        "channels": channels,
        "heldout_scores_evaluated": False,
    }


def _water_seed_training_score(
    matrix: np.ndarray,
    context: Mapping[str, Any],
    config: FarmsNonrigidConfig,
) -> tuple[float, dict[str, Any]]:
    """Score seed fit using training water evidence and heldout exclusion only."""

    reports: list[dict[str, Any]] = []
    weighted_score = 0.0
    weights = {
        "outer_pacific_training_microsegments": 0.65,
        "observed_suisun_delta_training_microsegments": 0.35,
    }
    exclusion = float(config.seed_water_heldout_exclusion_working_px)
    training_corridor_radius = float(
        config.seed_water_training_corridor_working_px
    )
    for channel in context["channels"]:
        name = str(channel["name"])
        mapped_heldout = _transform(channel["heldout_normalized"], matrix)
        heldout_tree = cKDTree(mapped_heldout)
        mapped_training = _transform(channel["training_normalized"], matrix)
        training_tree = cKDTree(mapped_training)
        source_points = np.asarray(channel["source_points"], dtype=np.float64)
        source_separation, _ = heldout_tree.query(source_points, workers=1)
        source_training_distance, _ = training_tree.query(source_points, workers=1)
        source_assignment = (
            (source_training_distance <= training_corridor_radius)
            & (source_separation > exclusion)
        )
        assignment_count = int(np.count_nonzero(source_assignment))
        assignment_mask = np.zeros(context["source_shape"], dtype=np.uint8)
        if assignment_count:
            assigned = np.rint(source_points[source_assignment]).astype(np.int32)
            assignment_mask[assigned[:, 1], assigned[:, 0]] = 1
        assignment_sha256 = hashlib.sha256(
            np.ascontiguousarray(assignment_mask).tobytes()
        ).hexdigest()
        rounded = np.rint(mapped_training).astype(np.int32)
        _inside_canvas, visible = _source_extent_membership(
            rounded, context["map_panel"]
        )
        visible_count = int(np.count_nonzero(visible))
        visible_fraction = float(np.mean(visible))
        supported = bool(
            assignment_count >= config.seed_water_minimum_visible_count
            and visible_count >= config.seed_water_minimum_visible_count
            and visible_fraction >= config.seed_water_minimum_visible_fraction
        )
        if supported:
            source_tree = cKDTree(source_points[source_assignment])
            distance, _ = source_tree.query(mapped_training[visible], workers=1)
            median = float(np.median(distance))
            p90 = float(np.quantile(distance, 0.90))
            within = float(np.mean(distance <= 8.0))
            score = float(
                median
                + 0.35 * p90
                + 5.0 * (1.0 - within)
                + config.seed_water_visibility_penalty * (1.0 - visible_fraction)
            )
        else:
            median = p90 = float("inf")
            within = 0.0
            score = float(config.seed_water_undersupport_penalty)
            score += config.seed_water_visibility_penalty * (1.0 - visible_fraction)
        weighted_score += weights[name] * score
        reports.append(
            {
                "channel": name,
                "source_line_id": channel["source_line_id"],
                "source_line_sha256": channel["source_line_sha256"],
                "training_reference_pixel_count": channel[
                    "training_reference_pixel_count"
                ],
                "training_reference_mask_sha256": channel[
                    "training_reference_mask_sha256"
                ],
                "heldout_reference_pixel_count": channel[
                    "heldout_reference_pixel_count"
                ],
                "heldout_reference_mask_sha256": channel[
                    "heldout_reference_mask_sha256"
                ],
                "source_assignment_pixel_count": assignment_count,
                "source_assignment_sha256": assignment_sha256,
                "training_corridor_working_px": training_corridor_radius,
                "minimum_source_assignment_to_heldout_working_px": (
                    float(np.min(source_separation[source_assignment]))
                    if assignment_count
                    else None
                ),
                "visible_training_count": visible_count,
                "visible_training_fraction": visible_fraction,
                "median_px": median,
                "p90_px": p90,
                "within_8px_fraction": within,
                "score": score,
                "weight": weights[name],
                "supported": supported,
            }
        )
    return float(weighted_score), {
        "method": context["method"],
        "score": float(weighted_score),
        "heldout_exclusion_working_px": exclusion,
        "training_corridor_working_px": training_corridor_radius,
        "training_corridor_is_seed_only": True,
        "final_residual_control_corridor_working_px": float(
            config.water_training_maximum_match_working_px
        ),
        "minimum_visible_fraction": config.seed_water_minimum_visible_fraction,
        "heldout_scores_evaluated": False,
        "channels": reports,
        "supported": all(item["supported"] for item in reports),
    }


def fit_partitioned_farms_seed(
    reference: PinnedMapboxReference,
    semantics: MapboxFarmsSemantics,
    source: FarmsSourceEvidence,
    projection: ProjectionContext,
    config: FarmsNonrigidConfig,
    *,
    scale_bounds: tuple[float, float] | None = None,
    optimizer_seeds: Sequence[int] | None = None,
    visibility_policy: str = "source_partial_extent",
    named_pacific_masks: Mapping[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    """Fit global scale using declared county cells while preserving holdouts.

    A cropped vertical-to-diagonal Nevada ray is scale-invariant about its
    bend, so admin-only fitting has multiple equally plausible global scales.
    The seed therefore uses buffered three-way county *training* cells away from the
    Bay.  Complementary county cells and every Pacific/Bay pixel are excluded
    and returned as an independent holdout mask.
    """

    height, width = reference.state_land.shape
    # This fixed Mapbox-pixel rectangle covers San Francisco Bay, Suisun Bay,
    # and the connected Delta in the pinned California v2 grid.  It is defined
    # before any source scoring and is always test-only (with a 25 px guard).
    # The bounds are deliberately recorded in the report rather than inferred
    # from a candidate transform.
    bay_bounds = (380, 1260, 1245, 2170)
    bay_guard = np.zeros(reference.counties.shape, dtype=np.uint8)
    bay_guard[bay_bounds[1] : bay_bounds[3] + 1, bay_bounds[0] : bay_bounds[2] + 1] = 1
    bay_guard = cv2.dilate(bay_guard, np.ones((51, 51), np.uint8)).astype(bool)
    partition_cells = config.seed_partition_cells
    county_training_cells = _reference_cell_mask(
        reference.counties.shape,
        cells=partition_cells,
        predicate=lambda row, column: (2 * row + column + row // 2) % 3 == 0,
    )
    county_validation_cells = _reference_cell_mask(
        reference.counties.shape,
        cells=partition_cells,
        predicate=lambda row, column: (2 * row + column + row // 2) % 3 == 1,
    )
    county_test_cells = _reference_cell_mask(
        reference.counties.shape,
        cells=partition_cells,
        predicate=lambda row, column: (2 * row + column + row // 2) % 3 == 2,
    )
    # Boundary guard exceeds the 5-px overlap tolerance and the 2-px rendered
    # stroke.  No reference pixel can belong to neighboring partitions.
    guard_size = 2 * config.seed_partition_boundary_guard_px + 1
    guard_kernel = np.ones((guard_size, guard_size), np.uint8)
    county_training_cells = cv2.erode(
        county_training_cells.astype(np.uint8), guard_kernel
    ).astype(bool)
    county_validation_cells = cv2.erode(
        county_validation_cells.astype(np.uint8), guard_kernel
    ).astype(bool)
    county_test_cells = cv2.erode(
        county_test_cells.astype(np.uint8), guard_kernel
    ).astype(bool)
    county_training = reference.counties & county_training_cells & ~bay_guard
    county_validation = reference.counties & county_validation_cells & ~bay_guard
    county_test = reference.counties & (county_test_cells | bay_guard)
    if np.any(county_training & county_validation) or np.any(
        county_training & county_test
    ) or np.any(county_validation & county_test):
        raise ValueError("Partitioned Mapbox county evidence is not disjoint")
    if min(map(np.count_nonzero, (county_training, county_validation, county_test))) < 1000:
        raise ValueError("Partitioned Mapbox county evidence is incomplete")

    east = _east_boundary(source.topology)
    source_y, source_x = np.nonzero(east)
    # The terminal 22% contains the bend residual fit and remains absent from
    # the global seed.  Its source rows cannot influence scale or translation.
    cutoff = float(np.quantile(source_y, 0.78))
    source_admin = np.column_stack(
        (source_x[source_y <= cutoff], source_y[source_y <= cutoff])
    ).astype(np.float64)
    if len(source_admin) > 1800:
        indices = np.linspace(0, len(source_admin) - 1, 1800).round().astype(int)
        source_admin = source_admin[indices]

    admin_y, admin_x = np.nonzero(semantics.california_admin)
    # Two buffered target bands are absent from the seed tree and are later
    # used for residual validation/test.  The terminal residual block is also
    # absent.  These fractions are fixed by the source-family contract.
    validation_band = (admin_y >= height * 0.20) & (admin_y < height * 0.24)
    test_band = (admin_y >= height * 0.28) & (admin_y < height * 0.32)
    terminal_band = (admin_y >= height * 0.455) & (admin_y < height * 0.50)
    admin_training = ~(validation_band | test_band | terminal_band)
    admin_points = np.column_stack(
        (admin_x[admin_training], admin_y[admin_training])
    ).astype(np.float64)
    admin_tree = cKDTree(
        _project_reference_points(admin_points, projection, reference.grid)
    )
    source_county_y, source_county_x = np.nonzero(source.county_topology)
    source_county_tree = cKDTree(
        np.column_stack((source_county_x, source_county_y))
    )
    source_shape = source.rgb_working.shape[:2]
    source_extents, _source_extent_diagnostics = _source_observable_extents(
        source, config
    )
    if visibility_policy == "source_partial_extent":
        county_observable_extent = source_extents["county_lines"]
    elif visibility_policy == "canvas_bounds_hypothesis":
        # This is a candidate-generation hypothesis only.  It conservatively
        # treats the whole canvas as observable to retain the large-scale basin
        # discovered before partial-extent validation was corrected.  It has no
        # authority at validation or acceptance, which always uses the
        # source-derived partial extent.
        county_observable_extent = np.ones(source_shape, dtype=bool)
    else:
        raise ValueError("Unknown farms seed visibility policy")
    if named_pacific_masks is None:
        named_pacific_masks = _named_pacific_holdouts(semantics, reference, config)
    water_seed_context = _prepare_water_seed_training_context(
        reference,
        source,
        projection,
        named_pacific_masks,
        config,
    )

    def partition_entries(mask: np.ndarray) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        rows, columns = partition_cells
        mask_y, mask_x = np.nonzero(mask)
        for row in range(rows):
            for column in range(columns):
                selected = (
                    (mask_y >= row * height / rows)
                    & (mask_y < (row + 1) * height / rows)
                    & (mask_x >= column * width / columns)
                    & (mask_x < (column + 1) * width / columns)
                )
                if np.count_nonzero(selected) < config.minimum_pixels_per_balanced_cell:
                    continue
                points = np.column_stack((mask_x[selected], mask_y[selected])).astype(
                    np.float64
                )
                if len(points) > 700:
                    indices = np.linspace(0, len(points) - 1, 700).round().astype(int)
                    points = points[indices]
                entries.append(
                    {
                        "row": row,
                        "column": column,
                        "points": points,
                        "normalized": _project_reference_points(
                            points, projection, reference.grid
                        ),
                    }
                )
        return entries

    training_entries = partition_entries(county_training)
    validation_entries = partition_entries(county_validation)

    def county_partition_score(
        matrix: np.ndarray, entries: Sequence[Mapping[str, Any]]
    ) -> dict[str, Any]:
        visible_cells: list[dict[str, Any]] = []
        for entry in entries:
            mapped = _transform(np.asarray(entry["normalized"]), matrix)
            rounded = np.rint(mapped).astype(np.int32)
            inside_canvas, inside = _source_extent_membership(
                rounded, county_observable_extent
            )
            visible_fraction = float(np.mean(inside))
            if (
                visible_fraction < config.seed_minimum_visible_fraction_per_cell
                or np.count_nonzero(inside) < config.minimum_pixels_per_balanced_cell
            ):
                continue
            distance, _ = source_county_tree.query(mapped[inside], workers=1)
            visible_cells.append(
                {
                    "row": int(entry["row"]),
                    "column": int(entry["column"]),
                        "visible_fraction": visible_fraction,
                        "canvas_visible_fraction": float(np.mean(inside_canvas)),
                        "outside_source_observable_extent_count": int(
                            np.count_nonzero(inside_canvas & ~inside)
                        ),
                        "visible_count": int(np.count_nonzero(inside)),
                    "median_px": float(np.median(distance)),
                    "p90_px": float(np.quantile(distance, 0.90)),
                    "within_8px_fraction": float(np.mean(distance <= 8.0)),
                }
            )
        visible_rows = {item["row"] for item in visible_cells}
        visible_columns = {item["column"] for item in visible_cells}
        supported = bool(
            len(visible_cells) >= config.seed_minimum_visible_cells
            and len(visible_rows) >= config.seed_minimum_visible_rows
            and len(visible_columns) >= config.seed_minimum_visible_columns
        )
        if visible_cells:
            cell_medians = np.asarray([item["median_px"] for item in visible_cells])
            cell_p90s = np.asarray([item["p90_px"] for item in visible_cells])
            cell_support = np.asarray(
                [item["within_8px_fraction"] for item in visible_cells]
            )
            score = float(
                np.median(cell_medians)
                + 0.25 * np.median(cell_p90s)
                + 20.0 * (1.0 - np.mean(cell_support))
            )
        else:
            score = 1_000.0
        if not supported:
            score += 250.0
            score += 25.0 * max(0, config.seed_minimum_visible_cells - len(visible_cells))
            score += 25.0 * max(0, config.seed_minimum_visible_rows - len(visible_rows))
            score += 25.0 * max(
                0, config.seed_minimum_visible_columns - len(visible_columns)
            )
        return {
            "score": score,
            "supported": supported,
            "visible_cell_count": len(visible_cells),
            "visible_rows": sorted(visible_rows),
            "visible_columns": sorted(visible_columns),
            "cells": visible_cells,
        }

    def objective(parameters: Sequence[float]) -> float:
        matrix = _matrix_from_parameters(parameters, "similarity", source_shape)
        inverse = np.linalg.inv(matrix)
        normalized_admin = _transform(source_admin, inverse)
        admin_distance, _ = admin_tree.query(normalized_admin, workers=1)
        scale_px = float(np.mean(np.linalg.svd(matrix[:2, :2], compute_uv=False)))
        admin_values = admin_distance * scale_px
        admin_loss = float(
            np.median(admin_values)
            + 0.35 * np.quantile(admin_values, 0.90)
            + 5.0 * (1.0 - np.mean(admin_values <= 8.0))
        )
        county_loss = county_partition_score(matrix, training_entries)["score"]
        water_loss, _water_report = _water_seed_training_score(
            matrix, water_seed_context, config
        )
        admin_county_loss = float(
            (1.0 - config.seed_county_training_weight) * admin_loss
            + config.seed_county_training_weight * county_loss
        )
        return float(
            (1.0 - config.seed_water_training_weight) * admin_county_loss
            + config.seed_water_training_weight * water_loss
        )

    optimizer_candidates: list[dict[str, Any]] = []
    bounded_scale = scale_bounds or (0.55, 2.40)
    if not (0.55 <= bounded_scale[0] < bounded_scale[1] <= 2.40):
        raise ValueError("Farms seed scale bounds exceed the fixed search domain")
    bounds = (
        (-0.35, 1.25),
        (-0.35, 1.25),
        bounded_scale,
        (-18.0, 18.0),
    )
    effective_optimizer_seeds = tuple(optimizer_seeds or config.seed_optimizer_seeds)
    if not effective_optimizer_seeds:
        raise ValueError("Farms seed optimizer requires at least one fixed seed")
    for optimizer_seed in effective_optimizer_seeds:
        result = differential_evolution(
            objective,
            bounds,
            seed=optimizer_seed,
            popsize=config.seed_optimizer_population,
            maxiter=config.seed_optimizer_iterations,
            polish=True,
            workers=1,
            updating="immediate",
        )
        candidate_matrix = _matrix_from_parameters(result.x, "similarity", source_shape)
        training_report = county_partition_score(candidate_matrix, training_entries)
        validation_report = county_partition_score(candidate_matrix, validation_entries)
        water_training_score, water_training_report = _water_seed_training_score(
            candidate_matrix, water_seed_context, config
        )
        optimizer_candidates.append(
            {
                "optimizer_seed": optimizer_seed,
                "optimizer_objective": float(result.fun),
                "optimizer_iterations": int(result.nit),
                "optimizer_success": bool(result.success),
                "parameters": [float(value) for value in result.x],
                "matrix": candidate_matrix,
                "training_counties": training_report,
                "validation_counties": validation_report,
                "water_training": water_training_report,
                "water_training_score": water_training_score,
                "eligible": bool(
                    training_report["supported"]
                    and validation_report["supported"]
                    and water_training_report["supported"]
                ),
            }
        )
    eligible = [item for item in optimizer_candidates if item["eligible"]]
    selected = min(
        eligible or optimizer_candidates,
        key=lambda item: (
            item["validation_counties"]["score"],
            item["optimizer_objective"],
            item["optimizer_seed"],
        ),
    )
    matrix = np.asarray(selected["matrix"], dtype=np.float64)
    masks = {
        "county_training": county_training,
        "county_validation": county_validation,
        "county_test": county_test,
    }
    return matrix, {
        "method": "buffered_admin_plus_three_way_nonbay_county_training_seed",
        "projection": projection.id,
        "model": "similarity",
        "optimizer_objective": selected["optimizer_objective"],
        "optimizer_iterations": selected["optimizer_iterations"],
        "optimizer_success": selected["optimizer_success"],
        "parameters": selected["parameters"],
        "optimizer_candidate_count": len(optimizer_candidates),
        "scale_bounds": list(bounded_scale),
        "optimizer_seeds": list(effective_optimizer_seeds),
        "visibility_policy": visibility_policy,
        "visibility_policy_is_candidate_generation_only": True,
        "selected_optimizer_seed": selected["optimizer_seed"],
        "optimizer_candidates": [
            {key: value for key, value in item.items() if key != "matrix"}
            for item in optimizer_candidates
        ],
        "admin_source_training_points": int(len(source_admin)),
        "admin_reference_training_points": int(len(admin_points)),
        "county_reference_training_points": int(
            sum(len(item["points"]) for item in training_entries)
        ),
        "county_reference_training_pixels": int(np.count_nonzero(county_training)),
        "county_reference_validation_pixels": int(np.count_nonzero(county_validation)),
        "county_reference_test_pixels": int(np.count_nonzero(county_test)),
        "county_partition_grid": list(partition_cells),
        "county_partition_assignment": "(2_times_row_plus_column_plus_floor_row_half)_modulo_3_train_validation_test",
        "county_partition_revision": "fresh_v3_after_v2_test_consumption",
        "county_partition_boundary_guard_reference_px": config.seed_partition_boundary_guard_px,
        "bay_suisun_delta_test_only_bounds_reference_px": list(bay_bounds),
        "bay_suisun_delta_guard_reference_px": 25,
        "county_training_weight": config.seed_county_training_weight,
        "water_training_weight": config.seed_water_training_weight,
        "outer_pacific_and_suisun_delta_training_used_for_fit": True,
        "suisun_delta_is_train_only_not_independently_certified": True,
        "north_and_south_bay_used_for_validation": False,
        "golden_gate_and_east_bay_used_for_retained_acceptance": False,
        "county_training_partition_used_for_global_seed": True,
        "county_validation_partition_used_only_for_seed_selection": True,
        "county_test_partition_evaluated_during_seed_selection": False,
        "full_county_geometry_used_for_fit": False,
        "terminal_admin_residual_region_used_for_fit": False,
        "validation_admin_band_used_for_fit": False,
        "test_admin_band_used_for_fit": False,
    }, masks


def _robust_indices(indices: np.ndarray, residuals: np.ndarray) -> np.ndarray:
    if len(indices) < 4:
        return np.empty(0, dtype=np.int64)
    values = residuals[indices]
    median = np.median(values, axis=0)
    deviation = np.linalg.norm(values - median, axis=1)
    limit = max(float(np.percentile(deviation, 65)), 0.75)
    return indices[deviation <= limit]


def derive_nevada_residual_controls(
    reference: PinnedMapboxReference,
    semantics: MapboxFarmsSemantics,
    source: FarmsSourceEvidence,
    projection: ProjectionContext,
    seed_matrix: np.ndarray,
    config: FarmsNonrigidConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    """Freeze fresh admin validation/test bands without fitting their pixels."""

    ref_y, ref_x = np.nonzero(semantics.california_admin)
    reference_points = np.column_stack((ref_x, ref_y)).astype(np.float64)
    reference_height, reference_width = semantics.california_admin.shape
    validation_region = (
        (ref_y >= reference_height * 0.20)
        & (ref_y < reference_height * 0.24)
        & (ref_x >= reference_width * 0.40)
    )
    test_region = (
        (ref_y >= reference_height * 0.28)
        & (ref_y < reference_height * 0.32)
        & (ref_x >= reference_width * 0.40)
    )
    validation_indices = np.flatnonzero(validation_region)
    test_indices = np.flatnonzero(test_region)
    if min(len(validation_indices), len(test_indices)) < 20:
        raise ValueError("Fresh Nevada validation/test geography is undersupported")
    concatenated = {
        "fit": np.empty((0, 2), dtype=np.float64),
        "validation": reference_points[validation_indices],
        "test": reference_points[test_indices],
    }
    return (
        np.empty((0, 2), dtype=np.float64),
        np.empty((0, 2), dtype=np.float64),
        concatenated,
        {
            "method": "fresh_predeclared_admin_validation_test_only_no_residual_fit",
            "fit_control_count": 0,
            "validation_point_count": int(len(validation_indices)),
            "test_point_count": int(len(test_indices)),
            "validation_region_fractional_bounds": [0.40, 0.20, 1.0, 0.24],
            "test_region_fractional_bounds": [0.40, 0.28, 1.0, 0.32],
            "partition_revision": "fresh_v3_after_v2_test_consumption",
            "partition_membership_depends_only_on_pinned_mapbox_target_pixels": True,
            "source_or_candidate_used_to_select_test_pixels": False,
            "mapbox_pacific_or_bay_used_for_fit": False,
            "mapbox_counties_used_for_fit": False,
            "admin_pixels_used_for_residual_fit": False,
            "manual_inputs_used": False,
        },
    )


def derive_county_residual_controls(
    reference: PinnedMapboxReference,
    source: FarmsSourceEvidence,
    projection: ProjectionContext,
    seed_matrix: np.ndarray,
    county_training: np.ndarray,
    county_validation: np.ndarray,
    county_test: np.ndarray,
    config: FarmsNonrigidConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    """Derive smooth residual controls from train-only county geometry."""

    rendered_training = _render_line(
        county_training,
        source.rgb_working.shape[:2],
        projection,
        reference.grid,
        seed_matrix,
        None,
    )
    rendered_heldout = _render_line(
        county_validation | county_test,
        source.rgb_working.shape[:2],
        projection,
        reference.grid,
        seed_matrix,
        None,
    )
    training_corridor_radius = int(math.ceil(config.residual_county_maximum_match_px))
    heldout_exclusion_radius = training_corridor_radius + 6
    training_corridor = cv2.dilate(
        rendered_training.astype(np.uint8),
        np.ones((2 * training_corridor_radius + 1,) * 2, np.uint8),
    ).astype(bool)
    heldout_exclusion = cv2.dilate(
        rendered_heldout.astype(np.uint8),
        np.ones((2 * heldout_exclusion_radius + 1,) * 2, np.uint8),
    ).astype(bool)
    source_assignment = (
        source.county_topology & training_corridor & ~heldout_exclusion
    )
    source_y, source_x = np.nonzero(source_assignment)
    if len(source_x) < 100:
        raise ValueError("Mapped-domain train-only source topology is undersupported")
    source_tree = cKDTree(np.column_stack((source_x, source_y)))
    ref_y, ref_x = np.nonzero(county_training)
    reference_points = np.column_stack((ref_x, ref_y)).astype(np.float64)
    if len(reference_points) > 20_000:
        indices = np.linspace(0, len(reference_points) - 1, 20_000).round().astype(int)
        reference_points = reference_points[indices]
        ref_x = reference_points[:, 0].astype(np.int32)
        ref_y = reference_points[:, 1].astype(np.int32)
    mapped = _map_points(
        reference_points, projection, reference.grid, seed_matrix, None
    )
    rounded = np.rint(mapped).astype(np.int32)
    source_extents, _source_extent_diagnostics = _source_observable_extents(
        source, config
    )
    inside_canvas, inside = _source_extent_membership(
        rounded, source_extents["county_lines"]
    )
    distance, nearest = source_tree.query(mapped)
    residual = source_tree.data[nearest] - mapped
    eligible = inside & (distance <= config.residual_county_maximum_match_px)
    row_bin = ref_y // config.residual_county_bin_reference_px
    column_bin = ref_x // config.residual_county_bin_reference_px
    controls: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    reports: list[dict[str, Any]] = []
    for key in sorted(set(zip(row_bin[eligible], column_bin[eligible]))):
        indices = np.flatnonzero(
            eligible & (row_bin == key[0]) & (column_bin == key[1])
        )
        robust = _robust_indices(indices, residual)
        if len(robust) < config.residual_county_minimum_bin_support:
            continue
        center = np.median(reference_points[robust], axis=0)
        displacement = np.median(residual[robust], axis=0)
        controls.append(center)
        targets.append(displacement)
        reports.append(
            {
                "bin": [int(key[0]), int(key[1])],
                "center_reference_px": center.tolist(),
                "support": int(len(robust)),
                "median_residual_working_px": displacement.tolist(),
                "residual_norm_working_px": float(np.linalg.norm(displacement)),
            }
        )
    if len(controls) < 8:
        raise ValueError("Train-only county residual controls are undersupported")
    heldout_distance = distance_transform_edt(~rendered_heldout)
    measured_separation = float(np.min(heldout_distance[source_assignment]))
    return (
        np.asarray(controls, dtype=np.float64),
        np.asarray(targets, dtype=np.float64),
        {
            "method": "fixed_reference_bins_train_only_county_nearest_line_robust_medians",
            "control_count": len(controls),
            "bin_reference_px": config.residual_county_bin_reference_px,
            "maximum_training_match_working_px": config.residual_county_maximum_match_px,
            "minimum_bin_support": config.residual_county_minimum_bin_support,
            "training_corridor_radius_working_px": training_corridor_radius,
            "heldout_exclusion_radius_working_px": heldout_exclusion_radius,
            "measured_minimum_source_assignment_to_heldout_working_px": measured_separation,
            "source_assignment_pixel_count": int(np.count_nonzero(source_assignment)),
            "reference_training_canvas_visible_count": int(
                np.count_nonzero(inside_canvas)
            ),
            "reference_training_source_extent_visible_count": int(
                np.count_nonzero(inside)
            ),
            "reference_training_outside_source_observable_extent_count": int(
                np.count_nonzero(inside_canvas & ~inside)
            ),
            "controls": reports,
            "validation_or_test_counties_used": False,
            "bay_suisun_delta_used": False,
        },
        {
            "source_county_training_corridor": training_corridor,
            "source_county_heldout_exclusion": heldout_exclusion,
            "source_county_training_assignment": source_assignment,
        },
    )


def derive_water_residual_controls(
    reference: PinnedMapboxReference,
    source: FarmsSourceEvidence,
    projection: ProjectionContext,
    seed_matrix: np.ndarray,
    named_pacific_masks: Mapping[str, np.ndarray],
    config: FarmsNonrigidConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    """Derive buffered train-only Pacific/Bay residual controls.

    Validation and retained-acceptance microsegments are used only to exclude
    nearby source assignments.  Their distances and scores are never read.
    """

    training_names = sorted(
        name
        for name in named_pacific_masks
        if name.endswith("training_microsegments")
    )
    if not training_names:
        raise ValueError("Farms water residual training masks are missing")
    maximum_match = float(config.water_training_maximum_match_working_px)
    corridor_radius = int(math.ceil(maximum_match))
    heldout_exclusion_radius = corridor_radius + 6
    if heldout_exclusion_radius != int(
        config.seed_water_heldout_exclusion_working_px
    ):
        raise ValueError("Seed and residual water heldout exclusions must agree")
    source_extents, _source_extent_diagnostics = _source_observable_extents(
        source, config
    )
    controls: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    channel_reports: list[dict[str, Any]] = []
    assignment_masks: dict[str, np.ndarray] = {}
    controls_per_channel: dict[str, int] = {}

    for name in training_names:
        reference_mask = named_pacific_masks[name]
        source_line, source_line_id = _water_channel_source_line(source, name)
        heldout_mask = _water_channel_heldout_union(named_pacific_masks, name)
        rendered_heldout = _render_line(
            heldout_mask,
            source.rgb_working.shape[:2],
            projection,
            reference.grid,
            seed_matrix,
            None,
        )
        heldout_exclusion = cv2.dilate(
            rendered_heldout.astype(np.uint8),
            np.ones((2 * heldout_exclusion_radius + 1,) * 2, np.uint8),
        ).astype(bool)
        heldout_distance = distance_transform_edt(~rendered_heldout)
        assignment_masks[
            f"source_{name}_heldout_exclusion"
        ] = heldout_exclusion
        assignment_masks[f"source_{name}_source_line"] = source_line
        source_line_sha256 = hashlib.sha256(
            np.ascontiguousarray(source_line.astype(np.uint8)).tobytes()
        ).hexdigest()
        heldout_mask_sha256 = hashlib.sha256(
            np.ascontiguousarray(heldout_mask.astype(np.uint8)).tobytes()
        ).hexdigest()
        rendered_training = _render_line(
            reference_mask,
            source.rgb_working.shape[:2],
            projection,
            reference.grid,
            seed_matrix,
            None,
        )
        training_corridor = cv2.dilate(
            rendered_training.astype(np.uint8),
            np.ones((2 * corridor_radius + 1,) * 2, np.uint8),
        ).astype(bool)
        source_assignment = source_line & training_corridor & ~heldout_exclusion
        assignment_masks[f"source_{name}_training_corridor"] = training_corridor
        assignment_masks[f"source_{name}_training_assignment"] = source_assignment
        source_y, source_x = np.nonzero(source_assignment)
        if len(source_x) < config.water_training_minimum_bin_support:
            channel_reports.append(
                {
                    "channel": name,
                    "status": "unsupported",
                    "source_line_id": source_line_id,
                    "source_line_sha256": source_line_sha256,
                    "heldout_reference_mask_sha256": heldout_mask_sha256,
                    "source_assignment_pixel_count": int(len(source_x)),
                    "minimum_source_assignment_to_heldout_working_px": None,
                    "control_count": 0,
                }
            )
            controls_per_channel[name] = 0
            continue
        source_tree = cKDTree(np.column_stack((source_x, source_y)))
        ref_y, ref_x = np.nonzero(reference_mask)
        reference_points = np.column_stack((ref_x, ref_y)).astype(np.float64)
        if len(reference_points) > 12_000:
            indices = np.linspace(
                0, len(reference_points) - 1, 12_000
            ).round().astype(int)
            reference_points = reference_points[indices]
            ref_x = reference_points[:, 0].astype(np.int32)
            ref_y = reference_points[:, 1].astype(np.int32)
        mapped = _map_points(
            reference_points, projection, reference.grid, seed_matrix, None
        )
        rounded = np.rint(mapped).astype(np.int32)
        _inside_canvas, visible = _source_extent_membership(
            rounded, source_extents["map_panel"]
        )
        distance, nearest = source_tree.query(mapped)
        residual = source_tree.data[nearest] - mapped
        eligible = visible & (distance <= maximum_match)
        row_bin = ref_y // config.water_training_bin_reference_px
        column_bin = ref_x // config.water_training_bin_reference_px
        report_controls: list[dict[str, Any]] = []
        before = len(controls)
        for key in sorted(set(zip(row_bin[eligible], column_bin[eligible]))):
            indices = np.flatnonzero(
                eligible & (row_bin == key[0]) & (column_bin == key[1])
            )
            robust = _robust_indices(indices, residual)
            if len(robust) < config.water_training_minimum_bin_support:
                continue
            center = np.median(reference_points[robust], axis=0)
            displacement = np.median(residual[robust], axis=0)
            controls.append(center)
            targets.append(displacement)
            report_controls.append(
                {
                    "bin": [int(key[0]), int(key[1])],
                    "center_reference_px": center.tolist(),
                    "support": int(len(robust)),
                    "median_residual_working_px": displacement.tolist(),
                    "residual_norm_working_px": float(np.linalg.norm(displacement)),
                }
            )
        channel_count = len(controls) - before
        controls_per_channel[name] = channel_count
        channel_reports.append(
            {
                "channel": name,
                "status": "ready" if channel_count else "unsupported",
                "source_line_id": source_line_id,
                "source_line_sha256": source_line_sha256,
                "heldout_reference_mask_sha256": heldout_mask_sha256,
                "reference_training_pixel_count": int(
                    np.count_nonzero(reference_mask)
                ),
                "source_assignment_pixel_count": int(len(source_x)),
                "minimum_source_assignment_to_heldout_working_px": float(
                    np.min(heldout_distance[source_assignment])
                ),
                "eligible_match_count": int(np.count_nonzero(eligible)),
                "control_count": channel_count,
                "controls": report_controls,
            }
        )

    outer_count = controls_per_channel.get(
        "outer_pacific_training_microsegments", 0
    )
    suisun_count = controls_per_channel.get(
        "observed_suisun_delta_training_microsegments", 0
    )
    if len(controls) < 4 or outer_count < 2 or suisun_count < 1:
        raise ValueError(
            "Buffered farms Pacific/Suisun training controls are undersupported: "
            f"total={len(controls)}, outer={outer_count}, suisun={suisun_count}; "
            f"channels={channel_reports}"
        )
    return (
        np.asarray(controls, dtype=np.float64),
        np.asarray(targets, dtype=np.float64),
        {
            "method": "buffered_channel_specific_mapbox_water_training_to_source_native_water",
            "control_count": len(controls),
            "outer_pacific_control_count": outer_count,
            "observed_suisun_delta_control_count": suisun_count,
            "maximum_match_working_px": maximum_match,
            "bin_reference_px": config.water_training_bin_reference_px,
            "minimum_bin_support": config.water_training_minimum_bin_support,
            "heldout_exclusion_radius_working_px": heldout_exclusion_radius,
            "validation_or_test_scores_used": False,
            "validation_and_test_masks_used_only_for_buffered_exclusion": True,
            "suisun_delta_source_line_is_internal_water_only": True,
            "suisun_delta_is_train_only_not_independently_certified": True,
            "channels": channel_reports,
        },
        assignment_masks,
    )


def _map_points(
    points: np.ndarray,
    projection: ProjectionContext,
    grid: Mapping[str, Any],
    seed_matrix: np.ndarray,
    warp: CompactResidualWarp | None,
) -> np.ndarray:
    base = _transform(_project_reference_points(points, projection, grid), seed_matrix)
    return base if warp is None else base + warp.displacement(points)


def _render_line(
    mask: np.ndarray,
    source_shape: tuple[int, int],
    projection: ProjectionContext,
    grid: Mapping[str, Any],
    seed_matrix: np.ndarray,
    warp: CompactResidualWarp | None,
) -> np.ndarray:
    y, x = np.nonzero(mask)
    mapped = np.rint(
        _map_points(np.column_stack((x, y)), projection, grid, seed_matrix, warp)
    ).astype(np.int32)
    inside = (
        (mapped[:, 0] >= 0)
        & (mapped[:, 0] < source_shape[1])
        & (mapped[:, 1] >= 0)
        & (mapped[:, 1] < source_shape[0])
    )
    rendered = np.zeros(source_shape, np.uint8)
    rendered[mapped[inside, 1], mapped[inside, 0]] = 1
    return cv2.dilate(rendered, np.ones((2, 2), np.uint8)).astype(bool)


def _render_land(
    reference: PinnedMapboxReference,
    source_shape: tuple[int, int],
    projection: ProjectionContext,
    seed_matrix: np.ndarray,
    warp: CompactResidualWarp | None,
) -> np.ndarray:
    contours, _ = cv2.findContours(
        reference.state_land.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    rendered = np.zeros(source_shape, np.uint8)
    for contour in contours:
        points = contour[:, 0, :].astype(np.float64)
        mapped = np.rint(
            _map_points(points, projection, reference.grid, seed_matrix, warp)
        ).astype(np.int32)
        if len(mapped) >= 3:
            cv2.fillPoly(rendered, [mapped], 1, cv2.LINE_8)
    return rendered.astype(bool)


def _balanced_tail(
    reference_mask: np.ndarray,
    source_mask: np.ndarray,
    mapped: np.ndarray,
    config: FarmsNonrigidConfig,
    *,
    observable_extent: np.ndarray,
) -> dict[str, Any]:
    y, x = np.nonzero(reference_mask)
    rounded = np.rint(mapped).astype(np.int32)
    source_height, source_width = source_mask.shape
    if observable_extent.shape != source_mask.shape:
        raise ValueError("Balanced-tail observable extent must match source mask")
    inside_canvas, inside = _source_extent_membership(
        rounded, observable_extent
    )
    distance = distance_transform_edt(~source_mask)
    values = np.full(len(mapped), math.hypot(source_width, source_height))
    values[inside] = distance[rounded[inside, 1], rounded[inside, 0]]
    rows, columns = config.balanced_cells
    height, width = reference_mask.shape
    cells = []
    for row in range(rows):
        for column in range(columns):
            selected = (
                (y >= row * height / rows)
                & (y < (row + 1) * height / rows)
                & (x >= column * width / columns)
                & (x < (column + 1) * width / columns)
                & inside
            )
            if np.count_nonzero(selected) < config.minimum_pixels_per_balanced_cell:
                continue
            local = values[selected]
            p90 = float(np.quantile(local, 0.90))
            cells.append(
                {
                    "row": row,
                    "column": column,
                    "pixel_count": int(np.count_nonzero(selected)),
                    "median_px": float(np.median(local)),
                    "p90_px": p90,
                    "within_8px_fraction": float(np.mean(local <= 8.0)),
                    "passed": p90 <= config.balanced_p90_limit_px,
                }
            )
    row_pass = {
        str(row): float(np.mean([item["passed"] for item in cells if item["row"] == row]))
        for row in sorted({item["row"] for item in cells})
    }
    column_pass = {
        str(column): float(
            np.mean([item["passed"] for item in cells if item["column"] == column])
        )
        for column in sorted({item["column"] for item in cells})
    }
    fraction = float(np.mean([item["passed"] for item in cells])) if cells else 0.0
    passed = bool(
        len(cells) >= config.balanced_minimum_cells
        and fraction >= config.balanced_minimum_cell_pass_fraction
        and row_pass
        and column_pass
        and min(row_pass.values()) >= config.balanced_minimum_axis_pass_fraction
        and min(column_pass.values()) >= config.balanced_minimum_axis_pass_fraction
    )
    return {
        "passed": passed,
        "grid": list(config.balanced_cells),
        "pixel_limit_px": config.balanced_p90_limit_px,
        "canvas_visible_count": int(np.count_nonzero(inside_canvas)),
        "source_extent_visible_count": int(np.count_nonzero(inside)),
        "outside_source_observable_extent_count": int(
            np.count_nonzero(inside_canvas & ~inside)
        ),
        "off_canvas_count": int(np.count_nonzero(~inside_canvas)),
        "cell_count": len(cells),
        "cell_pass_fraction": fraction,
        "row_pass_fractions": row_pass,
        "column_pass_fractions": column_pass,
        "cells": cells,
        "failed_cells": [item for item in cells if not item["passed"]],
    }


def _named_pacific_holdouts(
    semantics: MapboxFarmsSemantics,
    reference: PinnedMapboxReference,
    config: FarmsNonrigidConfig = FarmsNonrigidConfig(),
) -> dict[str, np.ndarray]:
    """Return predeclared Pacific-connected geographic test channels.

    Coordinates are pinned Mapbox-v2 target-grid pixels and never depend on a
    source candidate.  Phase one is an unused geographic guard between the
    validation and one-shot test phases.
    """

    height, width = semantics.pacific_coast.shape
    if (height, width) != EXPECTED_MAPBOX_V2_GRID or (
        int(reference.grid["height"]), int(reference.grid["width"])
    ) != (height, width):
        raise ValueError("Farms v3 named masks require the exact pinned Mapbox-v2 grid")

    def bounded(x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
        box = np.zeros((height, width), dtype=bool)
        box[max(0, y1) : min(height, y2), max(0, x1) : min(width, x2)] = True
        return semantics.pacific_coast & box

    # Named Bay boxes are fixed EPSG:3857 target pixels derived from declared
    # lon/lat envelopes on the exact v2 grid.  Overlap is removed in a frozen
    # east-to-west order so every source-supported coastline pixel belongs to
    # at most one named geography.
    named_raw = {
        "observed_suisun_delta": bounded(733, 1572, 1111, 1844),
        "north_bay_san_pablo": bounded(601, 1593, 783, 1770),
        "golden_gate": bounded(601, 1727, 750, 1853),
        "east_bay": bounded(716, 1719, 881, 1927),
        "south_bay": bounded(634, 1818, 832, 2092),
    }
    named_supported: dict[str, np.ndarray] = {}
    used = np.zeros((height, width), dtype=bool)
    for name, mask in named_raw.items():
        named_supported[name] = mask & ~used
        used |= mask
    yy = np.indices((height, width))[0]
    # Fixed roles are derived from target pixels alone.  The farms image is a
    # partial California map, so the previous interleaved phase-one training
    # fragments could all land outside the observed source panel.  Use three
    # predeclared, statewide-coordinate coast bands instead: the validation,
    # training, and retained-test bands have 150-px gaps and the training band
    # is more than 120 target pixels from either outer-coast holdout.  The
    # compact Suisun/Delta geography cannot support three independent bands,
    # so it is explicitly train-only.  Disjoint North/South Bay remain
    # validation and Golden Gate/East Bay remain retained acceptance.  No
    # validation or test scores select these roles.
    outer = semantics.pacific_coast & ~used
    result = {
        "outer_pacific_validation_microsegments": (
            outer & (yy >= 1550) & (yy < 1900)
        ),
        "outer_pacific_test_microsegments": (
            outer & (yy >= 2600) & (yy < 2900)
        ),
        "north_bay_san_pablo_validation_microsegments": named_supported[
            "north_bay_san_pablo"
        ],
        "south_bay_validation_microsegments": named_supported["south_bay"],
        "golden_gate_test_microsegments": named_supported["golden_gate"],
        "east_bay_test_microsegments": named_supported["east_bay"],
    }
    guard_radius = int(config.water_training_guard_reference_px)
    if guard_radius < 1:
        raise ValueError("Water training guard must be positive")
    outer_heldout = (
        result["outer_pacific_validation_microsegments"]
        | result["outer_pacific_test_microsegments"]
    )
    result["outer_pacific_training_microsegments"] = (
        outer & (yy >= 2050) & (yy < 2450)
    )
    if (
        distance_transform_edt(~outer_heldout)[
            result["outer_pacific_training_microsegments"]
        ].min()
        <= guard_radius
    ):
        raise ValueError("Outer Pacific training band violates its fixed target guard")
    bay_heldout = (
        named_supported["north_bay_san_pablo"]
        | named_supported["south_bay"]
        | named_supported["golden_gate"]
        | named_supported["east_bay"]
    )
    result["observed_suisun_delta_training_microsegments"] = named_supported[
        "observed_suisun_delta"
    ] & (distance_transform_edt(~bay_heldout) > guard_radius)
    validation_union = np.zeros((height, width), dtype=bool)
    test_union = np.zeros((height, width), dtype=bool)
    for name, mask in result.items():
        if name.endswith("validation_microsegments"):
            validation_union |= mask
        elif name.endswith("test_microsegments"):
            test_union |= mask
    if np.any(validation_union & test_union):
        raise ValueError("Named Pacific validation and test microsegments overlap")
    if any(np.count_nonzero(mask) < 20 for mask in result.values()):
        raise ValueError("Pinned Mapbox named Pacific channel is undersupported")
    training_union = np.zeros((height, width), dtype=bool)
    for name, mask in result.items():
        if name.endswith("training_microsegments"):
            training_union |= mask
    if np.any(training_union & (validation_union | test_union)):
        raise ValueError("Water training microsegments overlap heldout geometry")
    return result


def _holdout_gate(
    report: Mapping[str, Any],
    config: FarmsNonrigidConfig,
    *,
    bay: bool = False,
) -> bool:
    minimum_visible_fraction = (
        config.bay_holdout_minimum_visible_fraction
        if bay
        else config.holdout_minimum_visible_fraction
    )
    p90_limit = (
        config.bay_holdout_p90_limit_px
        if bay
        else config.admin_holdout_p90_limit_px
    )
    within_minimum = (
        config.bay_holdout_within_8_minimum
        if bay
        else config.admin_holdout_within_8_minimum
    )
    return bool(
        int(report["visible_count"]) >= config.holdout_minimum_visible_count
        and float(report["visible_fraction"]) >= minimum_visible_fraction
        and float(report["p90_px"]) <= p90_limit
        and float(report["within_8px_fraction"]) >= within_minimum
    )


def _supported_coast_gate(
    report: Mapping[str, Any], config: FarmsNonrigidConfig, *, bay: bool = False
) -> bool:
    """Strictly gate source-supported coast detail including its p90 tail."""

    return bool(
        int(report["visible_count"]) >= config.holdout_minimum_visible_count
        and float(report["visible_fraction"])
        >= (
            config.bay_holdout_minimum_visible_fraction
            if bay
            else config.holdout_minimum_visible_fraction
        )
        and float(report["median_px"]) <= config.state_median_limit_px
        and float(report["p90_px"])
        <= (
            config.bay_holdout_p90_limit_px
            if bay
            else config.admin_holdout_p90_limit_px
        )
        and float(report["within_8px_fraction"])
        >= (
            config.bay_holdout_within_8_minimum
            if bay
            else config.state_within_8_minimum
        )
    )


def _holdout_report(
    points: np.ndarray,
    source_mask: np.ndarray,
    projection: ProjectionContext,
    grid: Mapping[str, Any],
    seed_matrix: np.ndarray,
    warp: CompactResidualWarp,
    *,
    observable_extent: np.ndarray,
) -> dict[str, Any]:
    mapped = _map_points(points, projection, grid, seed_matrix, warp)
    rounded = np.rint(mapped).astype(np.int32)
    if observable_extent.shape != source_mask.shape:
        raise ValueError("Observable extent must match the source mask")
    inside_canvas, inside = _source_extent_membership(
        rounded, observable_extent
    )
    distance = distance_transform_edt(~source_mask)
    values = distance[rounded[inside, 1], rounded[inside, 0]]
    visible_count = int(np.count_nonzero(inside))
    visible_fraction = float(np.mean(inside)) if len(points) else 0.0
    return {
        "point_count": int(len(points)),
        "visible_count": visible_count,
        "visible_fraction": visible_fraction,
        "canvas_visible_count": int(np.count_nonzero(inside_canvas)),
        "canvas_visible_fraction": (
            float(np.mean(inside_canvas)) if len(points) else 0.0
        ),
        "outside_source_observable_extent_count": int(
            np.count_nonzero(inside_canvas & ~inside)
        ),
        "off_canvas_count": int(np.count_nonzero(~inside_canvas)),
        "median_px": float(np.median(values)) if len(values) else math.inf,
        "p90_px": float(np.quantile(values, 0.90)) if len(values) else math.inf,
        "within_8px_fraction": float(np.mean(values <= 8.0)) if len(values) else 0.0,
    }


def _county_validation_geographic_cells(
    validation_mask: np.ndarray,
    config: FarmsNonrigidConfig,
) -> dict[str, np.ndarray]:
    """Split the pinned Mapbox county validation mask into fixed geography.

    The grid is computed solely from the bounding box of the predeclared
    Mapbox validation partition.  It is therefore independent of the source,
    candidate transform, validation scores, and retained acceptance masks.
    Empty cells remain persisted so the geographic contract is auditable.
    """

    rows, columns = config.county_validation_balanced_cells
    if rows < 1 or columns < 1:
        raise ValueError("County validation geographic grid must be positive")
    y, x = np.nonzero(validation_mask)
    if not len(x):
        raise ValueError("County validation mask is empty")
    x0, x1 = int(x.min()), int(x.max()) + 1
    y0, y1 = int(y.min()), int(y.max()) + 1
    x_edges = np.linspace(x0, x1, columns + 1).round().astype(int)
    y_edges = np.linspace(y0, y1, rows + 1).round().astype(int)
    result: dict[str, np.ndarray] = {}
    for row in range(rows):
        for column in range(columns):
            cell = np.zeros_like(validation_mask, dtype=bool)
            cell[
                y_edges[row] : y_edges[row + 1],
                x_edges[column] : x_edges[column + 1],
            ] = True
            result[f"county_validation_r{row}_c{column}"] = (
                validation_mask & cell
            )
    union = np.zeros_like(validation_mask, dtype=bool)
    for mask in result.values():
        if np.any(union & mask):
            raise ValueError("County validation geographic cells overlap")
        union |= mask
    if not np.array_equal(union, validation_mask):
        raise ValueError("County validation geographic cells are incomplete")
    return result


def _balanced_county_validation_report(
    cell_masks: Mapping[str, np.ndarray],
    source_mask: np.ndarray,
    projection: ProjectionContext,
    grid: Mapping[str, Any],
    seed_matrix: np.ndarray,
    warp: CompactResidualWarp,
    *,
    observable_extent: np.ndarray,
    config: FarmsNonrigidConfig,
) -> dict[str, Any]:
    """Validate county alignment across visible geographic cells.

    A global support fraction can be dominated by one dense, nearly perfect
    county cluster.  This report scores each fixed Mapbox cell independently,
    requires visible coverage across rows and columns, and applies the same
    county median/support thresholds plus a strict p90 tail threshold.
    """

    point_groups: list[np.ndarray] = []
    group_names: list[str] = []
    group_ranges: list[tuple[int, int]] = []
    cursor = 0
    for name, mask in sorted(cell_masks.items()):
        y, x = np.nonzero(mask)
        points = np.column_stack((x, y)).astype(np.float64)
        # Keep the diagnostic bounded without changing geographic coverage.
        if len(points) > 4_000:
            indices = np.linspace(0, len(points) - 1, 4_000).round().astype(int)
            points = points[indices]
        point_groups.append(points)
        group_names.append(name)
        group_ranges.append((cursor, cursor + len(points)))
        cursor += len(points)
    points = (
        np.concatenate(point_groups, axis=0)
        if point_groups
        else np.empty((0, 2), dtype=np.float64)
    )
    mapped = _map_points(points, projection, grid, seed_matrix, warp)
    rounded = np.rint(mapped).astype(np.int32)
    inside_canvas, visible = _source_extent_membership(
        rounded, observable_extent
    )
    distance = distance_transform_edt(~source_mask)
    values = np.full(len(points), np.nan, dtype=np.float64)
    values[visible] = distance[
        rounded[visible, 1], rounded[visible, 0]
    ]
    cells: list[dict[str, Any]] = []
    minimum_count = config.county_validation_balanced_minimum_visible_count
    for name, (start, stop) in zip(group_names, group_ranges):
        row_column = name.rsplit("_r", 1)[1]
        row_text, column_text = row_column.split("_c", 1)
        local_visible = visible[start:stop]
        local_canvas = inside_canvas[start:stop]
        local_values = values[start:stop][local_visible]
        supported = bool(len(local_values) >= minimum_count)
        median = float(np.median(local_values)) if supported else None
        p90 = float(np.quantile(local_values, 0.90)) if supported else None
        within = float(np.mean(local_values <= 8.0)) if supported else None
        passed = bool(
            supported
            and median is not None
            and median <= config.county_median_limit_px
            and p90 is not None
            and p90 <= config.county_validation_balanced_p90_limit_px
            and within is not None
            and within >= config.county_within_8_minimum
        )
        cells.append(
            {
                "id": name,
                "row": int(row_text),
                "column": int(column_text),
                "point_count": stop - start,
                "canvas_visible_count": int(np.count_nonzero(local_canvas)),
                "visible_count": int(np.count_nonzero(local_visible)),
                "supported": supported,
                "median_px": median,
                "p90_px": p90,
                "within_8px_fraction": within,
                "passed": passed,
            }
        )
    supported_cells = [cell for cell in cells if cell["supported"]]
    visible_rows = sorted({cell["row"] for cell in supported_cells})
    visible_columns = sorted({cell["column"] for cell in supported_cells})
    cell_pass_fraction = (
        float(np.mean([cell["passed"] for cell in supported_cells]))
        if supported_cells
        else 0.0
    )
    row_pass_fractions = {
        str(row): float(
            np.mean([cell["passed"] for cell in supported_cells if cell["row"] == row])
        )
        for row in visible_rows
    }
    column_pass_fractions = {
        str(column): float(
            np.mean(
                [cell["passed"] for cell in supported_cells if cell["column"] == column]
            )
        )
        for column in visible_columns
    }
    passed = bool(
        len(supported_cells)
        >= config.county_validation_balanced_minimum_visible_cells
        and len(visible_rows)
        >= config.county_validation_balanced_minimum_visible_rows
        and len(visible_columns)
        >= config.county_validation_balanced_minimum_visible_columns
        and cell_pass_fraction
        >= config.county_validation_balanced_minimum_cell_pass_fraction
        and row_pass_fractions
        and min(row_pass_fractions.values())
        >= config.county_validation_balanced_minimum_axis_pass_fraction
        and column_pass_fractions
        and min(column_pass_fractions.values())
        >= config.county_validation_balanced_minimum_axis_pass_fraction
    )
    return {
        "passed": passed,
        "grid": list(config.county_validation_balanced_cells),
        "minimum_visible_count_per_cell": minimum_count,
        "minimum_visible_cells": config.county_validation_balanced_minimum_visible_cells,
        "minimum_visible_rows": config.county_validation_balanced_minimum_visible_rows,
        "minimum_visible_columns": config.county_validation_balanced_minimum_visible_columns,
        "median_limit_px": config.county_median_limit_px,
        "p90_limit_px": config.county_validation_balanced_p90_limit_px,
        "within_8px_minimum": config.county_within_8_minimum,
        "minimum_cell_pass_fraction": (
            config.county_validation_balanced_minimum_cell_pass_fraction
        ),
        "minimum_axis_pass_fraction": (
            config.county_validation_balanced_minimum_axis_pass_fraction
        ),
        "supported_cell_count": len(supported_cells),
        "visible_row_count": len(visible_rows),
        "visible_column_count": len(visible_columns),
        "cell_pass_fraction": cell_pass_fraction,
        "row_pass_fractions": row_pass_fractions,
        "column_pass_fractions": column_pass_fractions,
        "cells": cells,
        "failed_supported_cells": [
            cell for cell in supported_cells if not cell["passed"]
        ],
    }


def _source_observed_line_report(
    source_mask: np.ndarray, rendered_reference_mask: np.ndarray
) -> dict[str, Any]:
    values = distance_transform_edt(~rendered_reference_mask)[source_mask]
    return {
        "source_observed_pixel_count": int(np.count_nonzero(source_mask)),
        "median_px": float(np.median(values)) if len(values) else math.inf,
        "p90_px": float(np.quantile(values, 0.90)) if len(values) else math.inf,
        "within_8px_fraction": float(np.mean(values <= 8.0)) if len(values) else 0.0,
    }


def _boundary_self_intersections(points: np.ndarray) -> int:
    """Count proper intersections on a closed sampled boundary polygon."""

    def orient(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        return float(np.cross(b - a, c - a))

    total = 0
    count = len(points)
    for first in range(count):
        a, b = points[first], points[(first + 1) % count]
        for second in range(first + 2, count):
            if first == 0 and second == count - 1:
                continue
            c, d = points[second], points[(second + 1) % count]
            if orient(a, b, c) * orient(a, b, d) < 0 and orient(c, d, a) * orient(c, d, b) < 0:
                total += 1
    return total


def _regularity_report(
    reference: PinnedMapboxReference,
    projection: ProjectionContext,
    seed_matrix: np.ndarray,
    warp: CompactResidualWarp,
    config: FarmsNonrigidConfig,
) -> dict[str, Any]:
    y, x = np.nonzero(reference.state_land)
    x_values = np.linspace(x.min(), x.max(), config.jacobian_grid[1])
    y_values = np.linspace(y.min(), y.max(), config.jacobian_grid[0])
    xx, yy = np.meshgrid(x_values, y_values)
    points = np.column_stack((xx.ravel(), yy.ravel()))
    base = _map_points(points, projection, reference.grid, seed_matrix, warp)
    mapped_grid = base.reshape(len(y_values), len(x_values), 2)
    dx = _map_points(points + (1, 0), projection, reference.grid, seed_matrix, warp) - base
    dy = _map_points(points + (0, 1), projection, reference.grid, seed_matrix, warp) - base
    determinant = dx[:, 0] * dy[:, 1] - dx[:, 1] * dy[:, 0]
    median = float(np.median(determinant))
    ratio = determinant / max(median, 1e-12)
    matrices = np.stack((dx, dy), axis=2)
    singular = np.linalg.svd(matrices, compute_uv=False)
    condition = singular[:, 0] / np.maximum(singular[:, 1], 1e-12)

    # Treat the sampled grid as a piecewise-linear triangulation.  Every cell
    # must retain positive signed area in both triangles.  In addition, the
    # union area must equal the sum of cell polygon areas: any interior grid
    # crossing or overlap makes that ratio non-zero.  Together with a simple
    # boundary this is an explicit no-fold/injectivity check for the sampled
    # disk, not merely a point-Jacobian heuristic.
    top_left = mapped_grid[:-1, :-1]
    top_right = mapped_grid[:-1, 1:]
    bottom_right = mapped_grid[1:, 1:]
    bottom_left = mapped_grid[1:, :-1]

    def signed_double_area(
        first: np.ndarray, second: np.ndarray, third: np.ndarray
    ) -> np.ndarray:
        return (second[..., 0] - first[..., 0]) * (
            third[..., 1] - first[..., 1]
        ) - (second[..., 1] - first[..., 1]) * (
            third[..., 0] - first[..., 0]
        )

    triangle_a = signed_double_area(top_left, top_right, bottom_right)
    triangle_b = signed_double_area(top_left, bottom_right, bottom_left)
    minimum_triangle_double_area = float(min(np.min(triangle_a), np.min(triangle_b)))
    polygons: list[Polygon] = []
    invalid_cell_count = 0
    for row in range(len(y_values) - 1):
        for column in range(len(x_values) - 1):
            polygon = Polygon(
                (
                    mapped_grid[row, column],
                    mapped_grid[row, column + 1],
                    mapped_grid[row + 1, column + 1],
                    mapped_grid[row + 1, column],
                )
            )
            if not polygon.is_valid or polygon.area <= 0:
                invalid_cell_count += 1
            polygons.append(polygon)
    summed_cell_area = float(sum(item.area for item in polygons))
    union = unary_union(polygons)
    union_area = float(union.area)
    overlap_area = max(0.0, summed_cell_area - union_area)
    overlap_fraction = overlap_area / max(summed_cell_area, 1e-12)
    top = np.column_stack((x_values, np.full_like(x_values, y_values[0])))
    right = np.column_stack((np.full_like(y_values[1:], x_values[-1]), y_values[1:]))
    bottom = np.column_stack((x_values[-2::-1], np.full_like(x_values[-2::-1], y_values[-1])))
    left = np.column_stack((np.full_like(y_values[-2:0:-1], x_values[0]), y_values[-2:0:-1]))
    boundary = np.concatenate((top, right, bottom, left))
    mapped_boundary = _map_points(boundary, projection, reference.grid, seed_matrix, warp)
    intersections = _boundary_self_intersections(mapped_boundary)
    maximum_displacement = float(
        np.max(np.linalg.norm(warp.displacement(points), axis=1))
    )
    passed = bool(
        np.all(determinant > 0)
        and float(np.min(ratio)) >= config.minimum_jacobian_ratio
        and float(np.max(ratio)) <= config.maximum_jacobian_ratio
        and float(np.max(condition)) <= config.maximum_local_condition_number
        and minimum_triangle_double_area > 0
        and invalid_cell_count == 0
        and overlap_fraction <= 1e-8
        and intersections == 0
        and maximum_displacement <= config.maximum_residual_displacement_working_px
    )
    return {
        "passed": passed,
        "sample_count": int(len(points)),
        "determinant_minimum": float(np.min(determinant)),
        "determinant_median": median,
        "determinant_maximum": float(np.max(determinant)),
        "minimum_to_median_ratio": float(np.min(ratio)),
        "maximum_to_median_ratio": float(np.max(ratio)),
        "positive_fraction": float(np.mean(determinant > 0)),
        "maximum_local_condition_number": float(np.max(condition)),
        "triangulated_cell_count": int((len(y_values) - 1) * (len(x_values) - 1)),
        "minimum_triangle_signed_area": minimum_triangle_double_area / 2.0,
        "invalid_cell_polygon_count": invalid_cell_count,
        "summed_cell_area": summed_cell_area,
        "grid_union_area": union_area,
        "interior_grid_overlap_area": overlap_area,
        "interior_grid_overlap_fraction": overlap_fraction,
        "injectivity_basis": "positive_piecewise_linear_triangles_simple_boundary_and_zero_cell_union_overlap",
        "boundary_self_intersection_count": intersections,
        "maximum_residual_displacement_working_px": maximum_displacement,
    }


def serialize_farms_nonrigid_transform(
    *,
    seed_matrix_working: np.ndarray,
    projection: ProjectionContext,
    warp_working: CompactResidualWarp,
    working_scale: float,
    source_original_shape: tuple[int, int],
    source_working_shape: tuple[int, int],
    target_grid: Mapping[str, Any],
) -> dict[str, Any]:
    contract = _projection_transform_contract(
        seed_matrix_working,
        projection=projection,
        working_scale=working_scale,
        source_original_shape=source_original_shape,
        source_working_shape=source_working_shape,
        target_grid=target_grid,
    )
    residual = {
        "kind": "wendland_c2_reference_pixel_to_source_original_displacement",
        "coordinate_domain": "pinned_mapbox_target_grid_pixels",
        "displacement_range": "source_original_pixels",
        "centers_reference_px": warp_working.centers_reference_px.tolist(),
        "coefficients_source_px": (
            warp_working.coefficients_source_px / working_scale
        ).tolist(),
        "radius_reference_px": float(warp_working.radius_reference_px),
        "ridge": float(warp_working.ridge),
        "kernel": "max(1-r,0)^4*(4r+1)",
    }
    residual["sha256"] = _json_hash(residual)
    contract["kind"] = "projection_aware_residual_warp_mapbox_registration"
    contract["residual_warp"] = residual
    contract["inverse_solver"] = {
        "kind": "base_projection_fixed_point",
        "maximum_iterations": 20,
        "reference_tolerance_px": 1e-4,
        "source_roundtrip_tolerance_px": 0.02,
        "failure_policy": "reject_nonconverged_in_domain_points",
    }
    contract["reference_to_source_steps"].append(
        "add serialized Wendland-C2 source-pixel displacement evaluated in reference pixels"
    )
    contract["source_to_reference_steps"] = [
        "initialize with inverse projection-aware affine seed",
        "fixed-point iterate: subtract serialized reference-domain residual from source pixel",
        *contract["source_to_reference_steps"],
        "verify source-pixel round trip and reject nonconvergence",
    ]
    return contract


def serialized_roundtrip_report(
    transform: Mapping[str, Any], reference_points: np.ndarray
) -> dict[str, Any]:
    """Exercise the serialized Web-Mercator consumer contract on samples."""

    points = np.asarray(reference_points, dtype=np.float64)
    forward = np.asarray(transform["reference_pixel_to_source_original_matrix"])
    inverse = np.asarray(transform["source_original_to_reference_pixel_matrix"])
    residual = transform["residual_warp"]
    warp = CompactResidualWarp(
        np.asarray(residual["centers_reference_px"], dtype=np.float64),
        np.asarray(residual["coefficients_source_px"], dtype=np.float64),
        float(residual["radius_reference_px"]),
        float(residual["ridge"]),
    )
    homogeneous = np.column_stack((points, np.ones(len(points))))
    source = (homogeneous @ forward.T)[:, :2] + warp.displacement(points)
    estimate = (np.column_stack((source, np.ones(len(source)))) @ inverse.T)[:, :2]
    converged = np.zeros(len(points), dtype=bool)
    iterations = np.zeros(len(points), dtype=np.int16)
    tolerance = float(transform["inverse_solver"]["reference_tolerance_px"])
    for iteration in range(1, int(transform["inverse_solver"]["maximum_iterations"]) + 1):
        adjusted = source - warp.displacement(estimate)
        updated = (np.column_stack((adjusted, np.ones(len(adjusted)))) @ inverse.T)[:, :2]
        delta = np.linalg.norm(updated - estimate, axis=1)
        newly = (~converged) & (delta <= tolerance)
        iterations[newly] = iteration
        converged |= newly
        estimate = updated
    source_roundtrip = (
        np.column_stack((estimate, np.ones(len(estimate)))) @ forward.T
    )[:, :2] + warp.displacement(estimate)
    source_error = np.linalg.norm(source_roundtrip - source, axis=1)
    reference_error = np.linalg.norm(estimate - points, axis=1)
    passed = bool(
        np.all(converged)
        and float(np.max(source_error))
        <= float(transform["inverse_solver"]["source_roundtrip_tolerance_px"])
        and float(np.max(reference_error)) <= 0.02
    )
    return {
        "passed": passed,
        "sample_count": int(len(points)),
        "converged_fraction": float(np.mean(converged)),
        "maximum_iterations_used": int(np.max(iterations)),
        "maximum_source_roundtrip_error_px": float(np.max(source_error)),
        "maximum_reference_roundtrip_error_px": float(np.max(reference_error)),
    }


def _overlay(
    rgb: np.ndarray,
    topology: FarmsPartialTopology,
    rendered_state: np.ndarray,
    rendered_counties: np.ndarray,
    output: Path,
) -> None:
    canvas = cv2.cvtColor(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
    canvas[topology.state_coast] = (50, 230, 95)
    canvas[topology.internal_topology] = (45, 180, 225)
    canvas[rendered_state] = (255, 210, 40)
    canvas[rendered_counties] = (255, 75, 210)
    Image.fromarray(canvas).save(output)


def _water_overlay(
    rgb: np.ndarray,
    source: FarmsSourceEvidence,
    rendered_pacific: np.ndarray,
    output: Path,
    crop_output: Path,
) -> dict[str, Any]:
    """Persist a direct visual audit of native water against Mapbox coast."""

    canvas = cv2.cvtColor(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
    canvas[source.native_pacific_coast_edge] = (30, 205, 255)
    canvas[source.native_internal_water_edge] = (70, 255, 105)
    canvas[rendered_pacific] = (255, 75, 210)
    overlap = rendered_pacific & (
        source.native_pacific_coast_edge | source.native_internal_water_edge
    )
    canvas[overlap] = (255, 255, 255)
    Image.fromarray(canvas).save(output)

    internal_y, internal_x = np.nonzero(source.native_internal_water_edge)
    if not len(internal_x):
        raise ValueError("Native internal-water overlay has no observable source component")
    padding = 45
    x1 = max(0, int(internal_x.min()) - padding)
    y1 = max(0, int(internal_y.min()) - padding)
    x2 = min(canvas.shape[1], int(internal_x.max()) + padding + 1)
    y2 = min(canvas.shape[0], int(internal_y.max()) + padding + 1)
    Image.fromarray(canvas[y1:y2, x1:x2]).save(crop_output)
    return {
        "legend": {
            "source_native_pacific_coast_edge": [30, 205, 255],
            "source_native_internal_water_edge": [70, 255, 105],
            "rendered_mapbox_pacific_coast": [255, 75, 210],
            "overlap": [255, 255, 255],
        },
        "internal_water_crop_bounds_working_px": [x1, y1, x2, y2],
    }


def _validation_state(
    source_path: Path,
    reference_manifest_path: Path,
    config: FarmsNonrigidConfig,
) -> _FarmsValidationState:
    """Fit/select with training and validation evidence; never evaluate tests."""

    if _sha256(source_path) != EXPECTED_FARMS_SOURCE_SHA256:
        raise ValueError("The farms adapter requires the pristine farmsv2.png bytes")
    source = _load_source(source_path, config)
    source_extents, source_extent_diagnostics = _source_observable_extents(
        source, config
    )
    reference = load_pinned_mapbox_reference(reference_manifest_path)
    _require_pinned_v2_reference(reference)
    semantics = build_mapbox_farms_semantics(reference)
    projection = next(
        item for item in _projection_contexts(reference) if item.id == "web_mercator"
    )
    seed_matrix, seed_diagnostics, county_partitions = fit_partitioned_farms_seed(
        reference, semantics, source, projection, config
    )
    _unused_controls, _unused_targets, split_points, nevada_diagnostics = (
        derive_nevada_residual_controls(
            reference, semantics, source, projection, seed_matrix, config
        )
    )
    county_controls, county_targets, county_control_diagnostics, source_assignment_masks = (
        derive_county_residual_controls(
            reference,
            source,
            projection,
            seed_matrix,
            county_partitions["county_training"],
            county_partitions["county_validation"],
            county_partitions["county_test"],
            config,
        )
    )
    control_diagnostics = {
        "method": "train_only_county_controls_with_predeclared_heldout_exclusion",
        "nevada_admin": nevada_diagnostics,
        "county_training": county_control_diagnostics,
        "combined_control_count": int(len(county_controls)),
        "nevada_admin_controls_used": False,
        "validation_or_test_scores_used_for_control_target_derivation": False,
        "validation_and_test_masks_used_for_source_assignment_exclusion": True,
    }
    east = _east_boundary(source.topology)
    validation_y, validation_x = np.nonzero(county_partitions["county_validation"])
    county_validation_cells = _county_validation_geographic_cells(
        county_partitions["county_validation"], config
    )
    validation_county_points = np.column_stack((validation_x, validation_y)).astype(
        np.float64
    )
    if len(validation_county_points) > 20_000:
        indices = np.linspace(
            0, len(validation_county_points) - 1, 20_000
        ).round().astype(int)
        validation_county_points = validation_county_points[indices]
    named_pacific_masks = _named_pacific_holdouts(semantics, reference, config)
    named_validation_points: dict[str, np.ndarray] = {}
    for name, mask in named_pacific_masks.items():
        if not name.endswith("validation_microsegments"):
            continue
        mask_y, mask_x = np.nonzero(mask)
        named_validation_points[name] = np.column_stack((mask_x, mask_y)).astype(
            np.float64
        )
    candidates: list[dict[str, Any]] = []
    for radius in config.county_residual_kernel_radii_reference_px:
        for ridge in config.county_residual_ridge_values:
            warp = fit_compact_residual_warp(
                county_controls,
                county_targets,
                radius_reference_px=radius,
                ridge=ridge,
            )
            validation_admin = _holdout_report(
                split_points["validation"],
                east,
                projection,
                reference.grid,
                seed_matrix,
                warp,
                observable_extent=source_extents["state_admin"],
            )
            validation_counties = _holdout_report(
                validation_county_points,
                source.county_topology,
                projection,
                reference.grid,
                seed_matrix,
                warp,
                observable_extent=source_extents["county_lines"],
            )
            balanced_validation_counties = _balanced_county_validation_report(
                county_validation_cells,
                source.county_topology,
                projection,
                reference.grid,
                seed_matrix,
                warp,
                observable_extent=source_extents["county_lines"],
                config=config,
            )
            validation_named: dict[str, Any] = {}
            for name, points in named_validation_points.items():
                source_holdout = (
                    source.native_pacific_coast_edge
                    if name.startswith("outer_pacific")
                    else source.native_water_edge
                )
                validation_named[name] = _holdout_report(
                    points,
                    source_holdout,
                    projection,
                    reference.grid,
                    seed_matrix,
                    warp,
                    observable_extent=(
                        source_extents["pacific_coast"]
                        if name.startswith("outer_pacific")
                        else source_extents["named_bay"]
                    ),
                )
            outer = validation_named["outer_pacific_validation_microsegments"]
            bay_reports = {
                name: report
                for name, report in validation_named.items()
                if not name.startswith("outer_pacific")
            }
            regularity = _regularity_report(
                reference, projection, seed_matrix, warp, config
            )
            eligible = bool(
                regularity["passed"]
                and _holdout_gate(validation_admin, config)
                and validation_counties["visible_fraction"]
                >= config.holdout_minimum_visible_fraction
                and validation_counties["median_px"] <= config.county_median_limit_px
                and validation_counties["within_8px_fraction"]
                >= config.county_within_8_minimum
                and balanced_validation_counties["passed"]
                and _supported_coast_gate(outer, config)
                and all(
                    _supported_coast_gate(report, config, bay=True)
                    for report in bay_reports.values()
                )
            )
            selection_score = float(
                validation_admin["median_px"]
                + 0.35 * validation_admin["p90_px"]
                + validation_counties["median_px"]
                + 0.20 * validation_counties["p90_px"]
                + 10.0 * (1.0 - validation_counties["within_8px_fraction"])
                + outer["median_px"]
                + 0.20 * outer["p90_px"]
                + sum(
                    report["median_px"] + 0.20 * report["p90_px"]
                    for report in bay_reports.values()
                )
                + 0.02 * regularity["maximum_residual_displacement_working_px"]
            )
            candidates.append(
                {
                    "id": f"wendland-r{radius:g}-ridge{ridge:g}",
                    "radius_reference_px": radius,
                    "ridge": ridge,
                    "eligible": eligible,
                    "selection_score": selection_score,
                    "validation_admin_segments": validation_admin,
                    "validation_counties": validation_counties,
                    "balanced_validation_counties": balanced_validation_counties,
                    "validation_outer_pacific": outer,
                    "validation_named_bay_segments": bay_reports,
                    "regularity": regularity,
                    "warp": warp,
                }
            )
    eligible_candidates = [item for item in candidates if item["eligible"]]
    selected = (
        min(
            eligible_candidates,
            key=lambda item: (item["selection_score"], item["id"]),
        )
        if eligible_candidates
        else None
    )
    return _FarmsValidationState(
        source=source,
        reference=reference,
        semantics=semantics,
        projection=projection,
        seed_matrix=seed_matrix,
        seed_diagnostics=seed_diagnostics,
        county_partitions=county_partitions,
        county_validation_cells=county_validation_cells,
        split_points=split_points,
        control_diagnostics=control_diagnostics,
        source_assignment_masks=source_assignment_masks,
        source_observable_extents=source_extents,
        source_observable_extent_diagnostics=source_extent_diagnostics,
        named_pacific_masks=named_pacific_masks,
        candidates=tuple(candidates),
        selected=selected,
    )


def _retained_seed_modes(seed_diagnostics: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Keep the best representative of each of the two distinct scale modes."""

    ranked = sorted(
        seed_diagnostics["optimizer_candidates"],
        key=lambda item: (
            item["validation_counties"]["score"],
            item["optimizer_objective"],
            item["optimizer_seed"],
        ),
    )
    retained: list[Mapping[str, Any]] = []
    for candidate in ranked:
        scale = float(candidate["parameters"][2])
        if all(
            abs(math.log(scale / float(existing["parameters"][2]))) >= 0.35
            for existing in retained
        ):
            retained.append(candidate)
        if len(retained) == 2:
            break
    if len(retained) != 2:
        raise ValueError("The farms seed did not expose two distinct bounded scale modes")
    return retained


def _require_seed_inside_scale_interval(
    candidate: Mapping[str, Any], scale_bounds: tuple[float, float]
) -> float:
    scale = float(candidate["parameters"][2])
    if not scale_bounds[0] <= scale <= scale_bounds[1]:
        raise ValueError("A farms seed escaped its predeclared scale interval")
    return scale


def _bounded_validation_state(
    source_path: Path,
    reference_manifest_path: Path,
    config: FarmsNonrigidConfig,
) -> _FarmsValidationState:
    """Evaluate exactly three projections by two scale modes on validation only."""

    if _sha256(source_path) != EXPECTED_FARMS_SOURCE_SHA256:
        raise ValueError("The farms adapter requires the pristine farmsv2.png bytes")
    source = _load_source(source_path, config)
    source_extents, source_extent_diagnostics = _source_observable_extents(
        source, config
    )
    reference = load_pinned_mapbox_reference(reference_manifest_path)
    _require_pinned_v2_reference(reference)
    semantics = build_mapbox_farms_semantics(reference)
    projection_by_id = {item.id: item for item in _projection_contexts(reference)}
    if tuple(config.projection_ids) != (
        "web_mercator",
        "california_albers",
        "california_lambert_conformal_conic",
    ):
        raise ValueError("The farms projection shortlist must remain fixed")
    if not set(config.projection_ids).issubset(projection_by_id):
        raise ValueError("A pinned farms projection is unavailable")

    east = _east_boundary(source.topology)
    named_pacific_masks = _named_pacific_holdouts(semantics, reference, config)
    named_validation_points: dict[str, np.ndarray] = {}
    for name, mask in named_pacific_masks.items():
        if not name.endswith("validation_microsegments"):
            continue
        y, x = np.nonzero(mask)
        named_validation_points[name] = np.column_stack((x, y)).astype(np.float64)

    all_candidates: list[dict[str, Any]] = []
    seed_reports: dict[str, Any] = {}
    control_reports: dict[str, Any] = {}
    source_assignment_masks: dict[str, np.ndarray] = {}
    common_partitions: Mapping[str, np.ndarray] | None = None
    common_validation_cells: Mapping[str, np.ndarray] | None = None
    split_points_by_mode: dict[str, Mapping[str, np.ndarray]] = {}
    projection_for_mode: dict[str, ProjectionContext] = {}
    matrix_for_mode: dict[str, np.ndarray] = {}

    for projection_id in config.projection_ids:
        projection = projection_by_id[projection_id]
        if tuple(config.seed_scale_modes) != (
            ("partial-low", 0.55, 1.0, 3513, "source_partial_extent"),
            ("partial-high", 1.2, 2.4, 3512, "source_partial_extent"),
            ("canvas-high", 1.2, 2.4, 3512, "canvas_bounds_hypothesis"),
        ):
            raise ValueError("The farms scale-mode shortlist must remain fixed")
        mode_definitions = tuple(
            (label, (minimum, maximum), seed, visibility_policy)
            for label, minimum, maximum, seed, visibility_policy in config.seed_scale_modes
        )
        projection_mode_reports: list[dict[str, Any]] = []
        for (
            mode_label,
            mode_scale_bounds,
            optimizer_seed,
            visibility_policy,
        ) in mode_definitions:
            _selected_seed, seed_diagnostics, county_partitions = (
                fit_partitioned_farms_seed(
                    reference,
                    semantics,
                    source,
                    projection,
                    config,
                    scale_bounds=mode_scale_bounds,
                    optimizer_seeds=(optimizer_seed,),
                    visibility_policy=visibility_policy,
                    named_pacific_masks=named_pacific_masks,
                )
            )
            if common_partitions is None:
                common_partitions = county_partitions
                common_validation_cells = _county_validation_geographic_cells(
                    county_partitions["county_validation"], config
                )
            else:
                for name in common_partitions:
                    if not np.array_equal(
                        common_partitions[name], county_partitions[name]
                    ):
                        raise ValueError(
                            "Projection changed the predeclared county partitions"
                        )
            validation_y, validation_x = np.nonzero(
                county_partitions["county_validation"]
            )
            validation_county_points = np.column_stack(
                (validation_x, validation_y)
            ).astype(np.float64)
            if len(validation_county_points) > 20_000:
                indices = np.linspace(
                    0, len(validation_county_points) - 1, 20_000
                ).round().astype(int)
                validation_county_points = validation_county_points[indices]

            mode = seed_diagnostics["optimizer_candidates"][0]
            optimized_scale = _require_seed_inside_scale_interval(
                mode, mode_scale_bounds
            )
            projection_mode_reports.append(
                {
                    "mode": mode_label,
                    "scale_bounds": list(mode_scale_bounds),
                    "optimizer_seed": optimizer_seed,
                    "visibility_policy": visibility_policy,
                    "optimized_scale": optimized_scale,
                    "diagnostics": seed_diagnostics,
                }
            )
            mode_id = f"{projection_id}-scale-{mode_label}-seed-{optimizer_seed}"
            seed_matrix = _matrix_from_parameters(
                mode["parameters"], "similarity", source.rgb_working.shape[:2]
            )
            matrix_for_mode[mode_id] = seed_matrix
            projection_for_mode[mode_id] = projection
            _unused_controls, _unused_targets, split_points, nevada_diagnostics = (
                derive_nevada_residual_controls(
                    reference, semantics, source, projection, seed_matrix, config
                )
            )
            split_points_by_mode[mode_id] = split_points
            try:
                county_controls, county_targets, county_diagnostics, assignment_masks = (
                    derive_county_residual_controls(
                        reference,
                        source,
                        projection,
                        seed_matrix,
                        county_partitions["county_training"],
                        county_partitions["county_validation"],
                        county_partitions["county_test"],
                        config,
                    )
                )
                water_controls, water_targets, water_diagnostics, water_assignment_masks = (
                    derive_water_residual_controls(
                        reference,
                        source,
                        projection,
                        seed_matrix,
                        named_pacific_masks,
                        config,
                    )
                )
            except Exception as error:
                control_reports[mode_id] = {
                    "status": "blocked",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "nevada_admin": nevada_diagnostics,
                }
                for radius in config.county_residual_kernel_radii_reference_px:
                    for ridge in config.county_residual_ridge_values:
                        all_candidates.append(
                            {
                                "id": (
                                    f"{mode_id}-wendland-r{radius:g}-ridge{ridge:g}"
                                ),
                                "projection_id": projection_id,
                                "seed_mode_id": mode_id,
                                "seed_optimizer_seed": mode["optimizer_seed"],
                                "seed_parameters": mode["parameters"],
                                "seed_matrix_working": seed_matrix.tolist(),
                                "radius_reference_px": radius,
                                "ridge": ridge,
                                "eligible": False,
                                "selection_score": None,
                                "stop_reason": "train_only_control_derivation_failed",
                                "error": str(error),
                            }
                        )
                continue
            control_reports[mode_id] = {
                "status": "ready",
                "nevada_admin": nevada_diagnostics,
                "county_training": county_diagnostics,
                "water_training": water_diagnostics,
                "county_control_count": int(len(county_controls)),
                "water_control_count": int(len(water_controls)),
                "control_count": int(len(county_controls) + len(water_controls)),
                "acceptance_scores_used_for_control_derivation": False,
            }
            for name, mask in assignment_masks.items():
                source_assignment_masks[f"{mode_id}-{name}"] = mask
            for name, mask in water_assignment_masks.items():
                source_assignment_masks[f"{mode_id}-{name}"] = mask

            combined_controls = np.concatenate(
                (county_controls, water_controls), axis=0
            )
            combined_targets = np.concatenate(
                (county_targets, water_targets), axis=0
            )

            for radius in config.county_residual_kernel_radii_reference_px:
                for ridge in config.county_residual_ridge_values:
                    warp = fit_compact_residual_warp(
                        combined_controls,
                        combined_targets,
                        radius_reference_px=radius,
                        ridge=ridge,
                    )
                    validation_admin = _holdout_report(
                        split_points["validation"],
                        east,
                        projection,
                        reference.grid,
                        seed_matrix,
                        warp,
                        observable_extent=source_extents["state_admin"],
                    )
                    validation_counties = _holdout_report(
                        validation_county_points,
                        source.county_topology,
                        projection,
                        reference.grid,
                        seed_matrix,
                        warp,
                        observable_extent=source_extents["county_lines"],
                    )
                    if common_validation_cells is None:
                        raise ValueError("County validation cells are unavailable")
                    balanced_validation_counties = (
                        _balanced_county_validation_report(
                            common_validation_cells,
                            source.county_topology,
                            projection,
                            reference.grid,
                            seed_matrix,
                            warp,
                            observable_extent=source_extents["county_lines"],
                            config=config,
                        )
                    )
                    validation_named: dict[str, Any] = {}
                    for name, points in named_validation_points.items():
                        source_holdout = (
                            source.native_pacific_coast_edge
                            if name.startswith("outer_pacific")
                            else source.native_water_edge
                        )
                        validation_named[name] = _holdout_report(
                            points,
                            source_holdout,
                            projection,
                            reference.grid,
                            seed_matrix,
                            warp,
                            observable_extent=(
                                source_extents["pacific_coast"]
                                if name.startswith("outer_pacific")
                                else source_extents["named_bay"]
                            ),
                        )
                    outer = validation_named[
                        "outer_pacific_validation_microsegments"
                    ]
                    bay_reports = {
                        name: report
                        for name, report in validation_named.items()
                        if not name.startswith("outer_pacific")
                    }
                    regularity = _regularity_report(
                        reference, projection, seed_matrix, warp, config
                    )
                    eligible = bool(
                        regularity["passed"]
                        and _holdout_gate(validation_admin, config)
                        and validation_counties["visible_fraction"]
                        >= config.holdout_minimum_visible_fraction
                        and validation_counties["median_px"]
                        <= config.county_median_limit_px
                        and validation_counties["within_8px_fraction"]
                        >= config.county_within_8_minimum
                        and balanced_validation_counties["passed"]
                        and _supported_coast_gate(outer, config)
                        and all(
                            _supported_coast_gate(report, config, bay=True)
                            for report in bay_reports.values()
                        )
                    )
                    selection_score = float(
                        validation_admin["median_px"]
                        + 0.35 * validation_admin["p90_px"]
                        + validation_counties["median_px"]
                        + 0.20 * validation_counties["p90_px"]
                        + 10.0
                        * (1.0 - validation_counties["within_8px_fraction"])
                        + outer["median_px"]
                        + 0.20 * outer["p90_px"]
                        + sum(
                            report["median_px"] + 0.20 * report["p90_px"]
                            for report in bay_reports.values()
                        )
                        + 0.02
                        * regularity["maximum_residual_displacement_working_px"]
                    )
                    all_candidates.append(
                        {
                            "id": f"{mode_id}-wendland-r{radius:g}-ridge{ridge:g}",
                            "projection_id": projection_id,
                            "seed_mode_id": mode_id,
                            "seed_optimizer_seed": mode["optimizer_seed"],
                            "seed_parameters": mode["parameters"],
                            "seed_matrix_working": seed_matrix.tolist(),
                            "radius_reference_px": radius,
                            "ridge": ridge,
                            "eligible": eligible,
                            "selection_score": selection_score,
                            "validation_admin_segments": validation_admin,
                            "validation_counties": validation_counties,
                            "balanced_validation_counties": (
                                balanced_validation_counties
                            ),
                            "validation_outer_pacific": outer,
                            "validation_named_bay_segments": bay_reports,
                            "regularity": regularity,
                            "warp": warp,
                        }
                    )
        seed_reports[projection_id] = {
            "method": "two_predeclared_scale_intervals_plus_canvas_basin_hypothesis",
            "modes": projection_mode_reports,
        }

    expected_candidate_count = (
        len(config.projection_ids)
        * len(config.seed_scale_modes)
        * len(config.county_residual_kernel_radii_reference_px)
        * len(config.county_residual_ridge_values)
    )
    if expected_candidate_count != 108 or len(all_candidates) != expected_candidate_count:
        raise ValueError("The fixed farms validation shortlist did not emit 108 candidates")
    candidates_with_warp = [item for item in all_candidates if "warp" in item]
    if (
        not candidates_with_warp
        or common_partitions is None
        or common_validation_cells is None
    ):
        raise ValueError("No bounded farms projection/scale candidate could be evaluated")
    eligible = [item for item in candidates_with_warp if item["eligible"]]
    selected = (
        min(eligible, key=lambda item: (item["selection_score"], item["id"]))
        if eligible
        else None
    )
    representative = selected or min(
        candidates_with_warp,
        key=lambda item: (item["selection_score"], item["id"]),
    )
    representative_mode = str(representative["seed_mode_id"])
    return _FarmsValidationState(
        source=source,
        reference=reference,
        semantics=semantics,
        projection=projection_for_mode[representative_mode],
        seed_matrix=matrix_for_mode[representative_mode],
        seed_diagnostics={
            "method": "bounded_three_projection_three_seed_hypothesis_shortlist",
            "projection_ids": list(config.projection_ids),
            "maximum_candidate_count": 108,
            "projection_reports": seed_reports,
        },
        county_partitions=common_partitions,
        county_validation_cells=common_validation_cells,
        split_points=split_points_by_mode[representative_mode],
        control_diagnostics={
            "method": "per_projection_scale_mode_train_only_county_controls",
            "modes": control_reports,
        },
        source_assignment_masks=source_assignment_masks,
        source_observable_extents=source_extents,
        source_observable_extent_diagnostics=source_extent_diagnostics,
        named_pacific_masks=named_pacific_masks,
        candidates=tuple(all_candidates),
        selected=selected,
    )


def _write_mask_artifacts(
    output_root: Path, prefix: str, masks: Mapping[str, np.ndarray]
) -> tuple[dict[str, Any], tuple[Path, ...]]:
    artifacts: dict[str, Any] = {}
    paths: list[Path] = []
    for name, mask in masks.items():
        path = output_root / f"{prefix}-{name.replace('_', '-')}.png"
        Image.fromarray(mask.astype(np.uint8) * 255).save(path)
        role = None
        if name.endswith("validation_microsegments"):
            role = "validation_model_selection"
        elif name.endswith("training_microsegments"):
            role = "train_only_residual_control_geometry"
        elif name.endswith("test_microsegments") or name == "county_test":
            role = "retained_strict_acceptance_mask_not_evaluated"
        artifacts[name] = {
            "path": str(path),
            "sha256": _sha256(path),
            "pixel_count": int(np.count_nonzero(mask)),
            **({"role": role} if role else {}),
        }
        paths.append(path)
    return artifacts, tuple(paths)


def run_farms_nonrigid_validation_preflight(
    source_path: Path,
    reference_manifest_path: Path,
    output_root: Path,
    *,
    config: FarmsNonrigidConfig = FarmsNonrigidConfig(),
) -> FarmsValidationPreflightResult:
    """Select/freeze without reading retained strict acceptance scores."""

    source_path = source_path.resolve()
    reference_manifest_path = reference_manifest_path.resolve()
    output_root = output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("Validation preflight output must be a fresh directory")
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "validation-report.json"
    try:
        state = _bounded_validation_state(source_path, reference_manifest_path, config)
    except Exception as error:
        failure = {
            "schema_version": 1,
            "kind": "farms_nonrigid_validation_preflight_v1",
            "status": "error",
            "stop_reason": (
                "validation_preflight_exception_before_retained_acceptance_evaluation"
            ),
            "error_type": type(error).__name__,
            "error": str(error),
            "retained_acceptance_scores_evaluated": False,
            "inputs": {
                "source_path": str(source_path),
                "source_sha256": _sha256(source_path),
                "reference_manifest_path": str(reference_manifest_path),
                "reference_manifest_sha256": _sha256(reference_manifest_path),
            },
        }
        report_path.write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n")
        return FarmsValidationPreflightResult(
            status="error",
            stop_reason=failure["stop_reason"],
            report_path=report_path,
            frozen_candidate_path=None,
            artifact_paths=(report_path,),
        )

    county_artifacts, county_paths = _write_mask_artifacts(
        output_root, "mapbox", state.county_partitions
    )
    county_cell_artifacts, county_cell_paths = _write_mask_artifacts(
        output_root, "mapbox", state.county_validation_cells
    )
    assignment_artifacts, assignment_paths = _write_mask_artifacts(
        output_root, "source", state.source_assignment_masks
    )
    extent_artifacts, extent_paths = _write_mask_artifacts(
        output_root, "source-observable-extent", state.source_observable_extents
    )
    topology_artifacts, topology_paths = _write_mask_artifacts(
        output_root,
        "source-topology",
        {
            "county_scope": state.source.topology.county_scope,
            "working_rgb_raw_internal_topology": (
                state.source.topology.internal_topology
            ),
            "working_rgb_scoped_county_topology": (
                state.source.working_rgb_scoped_county_topology
            ),
            "native_working_county_ink": (
                state.source.native_county_topology.working_ink
            ),
            "working_rgb_secondary_near_native": (
                state.source.working_rgb_secondary_county_topology
            ),
            "combined_county_topology": state.source.county_topology,
            "ambiguous_topology_exclusion": (
                state.source.topology.ambiguous_topology_exclusion
            ),
            "southern_border_evidence": (
                state.source.topology.southern_border_evidence
            ),
        },
    )
    native_county_artifacts, native_county_paths = _write_mask_artifacts(
        output_root,
        "source-native-county",
        {
            "original_ink": state.source.native_county_topology.native_ink,
        },
    )
    topology_overlay_path = output_root / "source-topology-county-scope-overlay.png"
    Image.fromarray(
        render_farms_county_scope_overlay(
            state.source.rgb_working, state.source.topology
        )
    ).save(topology_overlay_path)
    topology_overlay = {
        "path": str(topology_overlay_path),
        "sha256": _sha256(topology_overlay_path),
        "role": "source_only_retained_vs_omitted_county_evidence_diagnostic",
    }
    named_artifacts, named_paths = _write_mask_artifacts(
        output_root, "mapbox", state.named_pacific_masks
    )
    serializable_candidates = [
        {key: value for key, value in candidate.items() if key != "warp"}
        for candidate in state.candidates
    ]
    frozen_candidate_path: Path | None = None
    frozen_sha256: str | None = None
    if state.selected is not None:
        selected_warp = state.selected["warp"]
        transform = serialize_farms_nonrigid_transform(
            seed_matrix_working=state.seed_matrix,
            projection=state.projection,
            warp_working=selected_warp,
            working_scale=state.source.working_scale,
            source_original_shape=state.source.rgb_original.shape[:2],
            source_working_shape=state.source.rgb_working.shape[:2],
            target_grid=state.reference.grid,
        )
        frozen = {
            "schema_version": 1,
            "kind": "farms_nonrigid_frozen_validation_candidate_v1",
            "source_sha256": _sha256(source_path),
            "reference_manifest_sha256": _sha256(reference_manifest_path),
            "config": asdict(config),
            "projection": state.projection.id,
            "seed_matrix_working": state.seed_matrix.tolist(),
            "selected_candidate": {
                key: value for key, value in state.selected.items() if key != "warp"
            },
            "warp_working": {
                "centers_reference_px": selected_warp.centers_reference_px.tolist(),
                "coefficients_source_px": selected_warp.coefficients_source_px.tolist(),
                "radius_reference_px": selected_warp.radius_reference_px,
                "ridge": selected_warp.ridge,
            },
            "serialized_transform": transform,
            "mask_sha256": {
                "county": {name: item["sha256"] for name, item in county_artifacts.items()},
                "county_validation_cells": {
                    name: item["sha256"]
                    for name, item in county_cell_artifacts.items()
                },
                "source_assignment": {
                    name: item["sha256"] for name, item in assignment_artifacts.items()
                },
                "source_observable_extent": {
                    name: item["sha256"] for name, item in extent_artifacts.items()
                },
                "source_topology": {
                    name: item["sha256"]
                    for name, item in topology_artifacts.items()
                },
                "source_native_county": {
                    name: item["sha256"]
                    for name, item in native_county_artifacts.items()
                },
                "named_pacific": {
                    name: item["sha256"] for name, item in named_artifacts.items()
                },
            },
            "authority": {
                "training_controls": "county_plus_buffered_outer_pacific_and_suisun_delta_training_only",
                "suisun_delta_is_fit_evidence_not_independently_certified": True,
                "county_visibility_uses_source_only_county_scope": True,
                "county_matching_uses_native_primary_and_two_pixel_supported_secondary": True,
                "unsupported_working_rgb_county_pixels_raw_union_forbidden": True,
                "county_topology_diagnostics": dict(
                    state.source.county_topology_diagnostics
                ),
                "ambiguous_lower_right_county_evidence_omitted_with_warning": True,
                "validation_used_for_model_selection": True,
                "retained_acceptance_scores_evaluated": False,
                "manual_or_prior_run_inputs_used": False,
            },
        }
        frozen_candidate_path = output_root / "frozen-validation-candidate.json"
        frozen_candidate_path.write_text(
            json.dumps(frozen, indent=2, sort_keys=True) + "\n"
        )
        frozen_sha256 = _sha256(frozen_candidate_path)
    status = "validation_pass" if state.selected is not None else "blocked"
    stop_reason = (
        "validation_candidate_frozen_retained_acceptance_not_evaluated"
        if state.selected is not None
        else "no_candidate_passed_validation_retained_acceptance_not_evaluated"
    )
    report = {
        "schema_version": 1,
        "kind": "farms_nonrigid_validation_preflight_v1",
        "status": status,
        "stop_reason": stop_reason,
        "inputs": {
            "source_path": str(source_path),
            "source_sha256": _sha256(source_path),
            "reference_manifest_path": str(reference_manifest_path),
            "reference_manifest_sha256": _sha256(reference_manifest_path),
            "mapbox_reference": dict(state.reference.pin),
        },
        "authority": {
            "manual_inputs_used": False,
            "prior_run_inputs_used": False,
            "county_png_used": False,
            "census_used": False,
            "county_training_used_for_seed_and_residual_fit": True,
            "buffered_outer_pacific_and_suisun_delta_used_for_residual_fit": True,
            "outer_pacific_and_suisun_delta_used_for_global_seed_fit": True,
            "suisun_delta_is_fit_evidence_not_independently_certified": True,
            "north_south_bay_used_for_validation_only": True,
            "golden_gate_east_bay_retained_acceptance_not_evaluated": True,
            "county_and_named_pacific_validation_used_for_model_selection": True,
            "line_visibility_uses_source_only_partial_map_extent": True,
            "county_visibility_uses_source_only_county_scope": True,
            "county_matching_uses_native_primary_and_two_pixel_supported_secondary": True,
            "unsupported_working_rgb_county_pixels_raw_union_forbidden": True,
            "ambiguous_lower_right_county_evidence_omitted_with_warning": True,
            "pacific_and_neighboring_state_remain_observed_negative_evidence": True,
            "retained_acceptance_masks_persisted_but_scores_evaluated": False,
            "acceptance_history": (
                "broad_geographies_were_seen_in_prior_throwaways; retained masks "
                "are fixed strict confirmation gates, not blind statistical tests"
            ),
        },
        "config": asdict(config),
        "source_native_water_diagnostics": dict(state.source.water_diagnostics),
        "source_native_county_diagnostics": dict(
            state.source.native_county_topology.diagnostics
        ),
        "source_county_topology_diagnostics": dict(
            state.source.county_topology_diagnostics
        ),
        "source_observable_extent_diagnostics": dict(
            state.source_observable_extent_diagnostics
        ),
        "source_topology_diagnostics": dict(state.source.topology.diagnostics),
        "seed": state.seed_diagnostics,
        "training": state.control_diagnostics,
        "candidates": serializable_candidates,
        "selected_candidate_id": (
            state.selected["id"] if state.selected is not None else None
        ),
        "county_mask_artifacts": county_artifacts,
        "county_validation_cell_artifacts": county_cell_artifacts,
        "source_assignment_artifacts": assignment_artifacts,
        "source_observable_extent_artifacts": extent_artifacts,
        "source_topology_artifacts": topology_artifacts,
        "source_native_county_artifacts": native_county_artifacts,
        "source_topology_overlay": topology_overlay,
        "named_pacific_mask_artifacts": named_artifacts,
        "frozen_candidate_path": (
            str(frozen_candidate_path) if frozen_candidate_path else None
        ),
        "frozen_candidate_sha256": frozen_sha256,
        "retained_acceptance_scores_evaluated": False,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    paths = [
        *county_paths,
        *county_cell_paths,
        *assignment_paths,
        *extent_paths,
        *topology_paths,
        *native_county_paths,
        topology_overlay_path,
        *named_paths,
    ]
    if frozen_candidate_path is not None:
        paths.append(frozen_candidate_path)
    paths.append(report_path)
    return FarmsValidationPreflightResult(
        status=status,
        stop_reason=stop_reason,
        report_path=report_path,
        frozen_candidate_path=frozen_candidate_path,
        artifact_paths=tuple(paths),
    )


def run_farms_nonrigid_throwaway(
    source_path: Path,
    reference_manifest_path: Path,
    output_root: Path,
    *,
    config: FarmsNonrigidConfig = FarmsNonrigidConfig(),
) -> FarmsNonrigidResult:
    """Run a source-clean farms residual experiment without official writes."""

    source_path = source_path.resolve()
    reference_manifest_path = reference_manifest_path.resolve()
    output_root = output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("Throwaway output must be a fresh immutable directory")
    output_root.mkdir(parents=True, exist_ok=True)
    if _sha256(source_path) != EXPECTED_FARMS_SOURCE_SHA256:
        raise ValueError("The farms adapter requires the pristine farmsv2.png bytes")
    source = _load_source(source_path, config)
    source_extents, source_extent_diagnostics = _source_observable_extents(
        source, config
    )
    reference = load_pinned_mapbox_reference(reference_manifest_path)
    _require_pinned_v2_reference(reference)
    semantics = build_mapbox_farms_semantics(reference)
    projection = next(
        item for item in _projection_contexts(reference) if item.id == "web_mercator"
    )
    seed_matrix, seed_diagnostics, county_partitions = fit_partitioned_farms_seed(
        reference, semantics, source, projection, config
    )
    nevada_controls, nevada_targets, split_points, nevada_diagnostics = derive_nevada_residual_controls(
        reference, semantics, source, projection, seed_matrix, config
    )
    county_controls, county_targets, county_control_diagnostics, source_assignment_masks = (
        derive_county_residual_controls(
            reference,
            source,
            projection,
            seed_matrix,
            county_partitions["county_training"],
            county_partitions["county_validation"],
            county_partitions["county_test"],
            config,
        )
    )
    controls = county_controls
    targets = county_targets
    control_diagnostics = {
        "method": "train_only_county_controls_with_predeclared_heldout_exclusion",
        "nevada_admin": nevada_diagnostics,
        "county_training": county_control_diagnostics,
        "combined_control_count": int(len(controls)),
        "nevada_admin_controls_used": False,
        "validation_or_test_scores_used_for_control_target_derivation": False,
        "validation_and_test_masks_used_for_source_assignment_exclusion": True,
    }
    east = _east_boundary(source.topology)
    validation_y, validation_x = np.nonzero(county_partitions["county_validation"])
    validation_county_points = np.column_stack((validation_x, validation_y)).astype(
        np.float64
    )
    if len(validation_county_points) > 20_000:
        indices = np.linspace(
            0, len(validation_county_points) - 1, 20_000
        ).round().astype(int)
        validation_county_points = validation_county_points[indices]
    named_pacific_masks = _named_pacific_holdouts(semantics, reference, config)
    named_validation_points: dict[str, np.ndarray] = {}
    for name, mask in named_pacific_masks.items():
        if not name.endswith("validation_microsegments"):
            continue
        mask_y, mask_x = np.nonzero(mask)
        named_validation_points[name] = np.column_stack((mask_x, mask_y)).astype(
            np.float64
        )
    candidates = []
    for radius in config.county_residual_kernel_radii_reference_px:
        for ridge in config.county_residual_ridge_values:
            warp = fit_compact_residual_warp(
                controls,
                targets,
                radius_reference_px=radius,
                ridge=ridge,
            )
            validation = _holdout_report(
                split_points["validation"],
                east,
                projection,
                reference.grid,
                seed_matrix,
                warp,
                observable_extent=source_extents["state_admin"],
            )
            validation_counties = _holdout_report(
                validation_county_points,
                source.county_topology,
                projection,
                reference.grid,
                seed_matrix,
                warp,
                observable_extent=source_extents["county_lines"],
            )
            validation_named_pacific: dict[str, Any] = {}
            for name, points in named_validation_points.items():
                source_holdout = (
                    source.native_pacific_coast_edge
                    if name.startswith("outer_pacific")
                    else source.native_water_edge
                )
                validation_named_pacific[name] = _holdout_report(
                    points,
                    source_holdout,
                    projection,
                    reference.grid,
                    seed_matrix,
                    warp,
                    observable_extent=(
                        source_extents["pacific_coast"]
                        if name.startswith("outer_pacific")
                        else source_extents["named_bay"]
                    ),
                )
            validation_outer_pacific = validation_named_pacific[
                "outer_pacific_validation_microsegments"
            ]
            validation_bay_reports = {
                name: report
                for name, report in validation_named_pacific.items()
                if not name.startswith("outer_pacific")
            }
            regularity = _regularity_report(
                reference, projection, seed_matrix, warp, config
            )
            eligible = bool(
                regularity["passed"]
                and _holdout_gate(validation, config)
                and validation_counties["visible_fraction"]
                >= config.holdout_minimum_visible_fraction
                and validation_counties["median_px"] <= config.county_median_limit_px
                and validation_counties["within_8px_fraction"]
                >= config.county_within_8_minimum
                and _supported_coast_gate(validation_outer_pacific, config)
                and all(
                    _supported_coast_gate(report, config, bay=True)
                    for report in validation_bay_reports.values()
                )
            )
            selection_score = float(
                validation["median_px"]
                + 0.35 * validation["p90_px"]
                + validation_counties["median_px"]
                + 0.20 * validation_counties["p90_px"]
                + 10.0 * (1.0 - validation_counties["within_8px_fraction"])
                + validation_outer_pacific["median_px"]
                + 0.20 * validation_outer_pacific["p90_px"]
                + sum(
                    report["median_px"] + 0.20 * report["p90_px"]
                    for report in validation_bay_reports.values()
                )
                + 0.02 * regularity["maximum_residual_displacement_working_px"]
            )
            candidates.append(
                {
                    "id": f"wendland-r{radius:g}-ridge{ridge:g}",
                    "radius_reference_px": radius,
                    "ridge": ridge,
                    "eligible": eligible,
                    "selection_score": selection_score,
                    "validation_admin_segments": validation,
                    "validation_counties": validation_counties,
                    "validation_outer_pacific": validation_outer_pacific,
                    "validation_named_bay_segments": validation_bay_reports,
                    "regularity": regularity,
                    "warp": warp,
                }
            )
    eligible = [item for item in candidates if item["eligible"]]
    if not eligible:
        raise ValueError(
            "No farms residual candidate passed validation; independent tests were not evaluated"
        )
    selected = min(
        eligible, key=lambda item: (item["selection_score"], item["id"])
    )
    warp = selected["warp"]
    test_admin = _holdout_report(
        split_points["test"],
        east,
        projection,
        reference.grid,
        seed_matrix,
        warp,
        observable_extent=source_extents["state_admin"],
    )
    named_pacific_reports: dict[str, Any] = {}
    for name, mask in named_pacific_masks.items():
        if not name.endswith("test_microsegments"):
            continue
        holdout_y, holdout_x = np.nonzero(mask)
        source_holdout = (
            source.topology.state_coast
            if name.startswith("outer_pacific")
            else source.native_water_edge
        )
        named_pacific_reports[name] = _holdout_report(
            np.column_stack((holdout_x, holdout_y)).astype(np.float64),
            source_holdout,
            projection,
            reference.grid,
            seed_matrix,
            warp,
            observable_extent=(
                source_extents["pacific_coast"]
                if name.startswith("outer_pacific")
                else source_extents["named_bay"]
            ),
        )
    rendered_state = _render_line(
        reference.state_coast,
        source.rgb_working.shape[:2],
        projection,
        reference.grid,
        seed_matrix,
        warp,
    )
    rendered_state &= source_extents["map_panel"]
    rendered_counties = _render_line(
        reference.counties,
        source.rgb_working.shape[:2],
        projection,
        reference.grid,
        seed_matrix,
        warp,
    )
    rendered_counties &= source_extents["county_lines"]
    rendered_pacific = _render_line(
        semantics.pacific_coast,
        source.rgb_working.shape[:2],
        projection,
        reference.grid,
        seed_matrix,
        warp,
    )
    rendered_pacific &= source_extents["pacific_coast"]
    observed_internal_water = _source_observed_line_report(
        source.native_internal_water_edge, rendered_pacific
    )
    rendered_land = _render_land(
        reference, source.rgb_working.shape[:2], projection, seed_matrix, warp
    )
    rendered_land &= source_extents["map_panel"]
    state_scope = cv2.dilate(
        rendered_state.astype(np.uint8), np.ones((25, 25), np.uint8)
    ).astype(bool)
    county_scope = cv2.erode(
        rendered_land.astype(np.uint8), np.ones((9, 9), np.uint8)
    ).astype(bool)
    county_scope &= ~cv2.dilate(
        rendered_state.astype(np.uint8), np.ones((9, 9), np.uint8)
    ).astype(bool)
    state = _symmetric_line_report(
        rendered_state,
        source.topology.state_coast,
        source_scope=state_scope,
        tolerance=5.0,
    )
    counties = _symmetric_line_report(
        rendered_counties,
        source.county_topology,
        source_scope=county_scope,
        tolerance=5.0,
    )
    partition_reports: dict[str, Any] = {}
    partition_rendered: dict[str, np.ndarray] = {}
    for name, mask in county_partitions.items():
        rendered = _render_line(
            mask,
            source.rgb_working.shape[:2],
            projection,
            reference.grid,
            seed_matrix,
            warp,
        )
        rendered &= source_extents["county_lines"]
        partition_rendered[name] = rendered
        local_scope = cv2.dilate(
            rendered.astype(np.uint8), np.ones((25, 25), np.uint8)
        ).astype(bool)
        partition_reports[name] = _symmetric_line_report(
            rendered,
            source.county_topology,
            source_scope=local_scope,
            tolerance=5.0,
        )
    primary_y, primary_x = np.nonzero(semantics.primary)
    primary_points = np.column_stack((primary_x, primary_y)).astype(np.float64)
    primary_mapped = _map_points(
        primary_points, projection, reference.grid, seed_matrix, warp
    )
    balanced = _balanced_tail(
        semantics.primary,
        source.topology.state_coast,
        primary_mapped,
        config,
        observable_extent=source_extents["map_panel"],
    )
    land_y, land_x = np.nonzero(reference.state_land)
    indices = np.linspace(0, len(land_x) - 1, min(len(land_x), 10000)).round().astype(int)
    land_points = np.column_stack((land_x[indices], land_y[indices])).astype(np.float64)
    mapped_land = np.rint(
        _map_points(land_points, projection, reference.grid, seed_matrix, warp)
    ).astype(np.int32)
    visible_canvas, visible = _source_extent_membership(
        mapped_land, source_extents["map_panel"]
    )
    containment = float(
        np.mean(
            source_extents["california_positive_support"][
                mapped_land[visible, 1], mapped_land[visible, 0]
            ]
        )
    )
    transform = serialize_farms_nonrigid_transform(
        seed_matrix_working=seed_matrix,
        projection=projection,
        warp_working=warp,
        working_scale=source.working_scale,
        source_original_shape=source.rgb_original.shape[:2],
        source_working_shape=source.rgb_working.shape[:2],
        target_grid=reference.grid,
    )
    roundtrip_indices = np.linspace(0, len(land_points) - 1, min(256, len(land_points))).round().astype(int)
    roundtrip = serialized_roundtrip_report(transform, land_points[roundtrip_indices])
    state_forward = state["reference_to_source"]
    county_forward = counties["reference_to_source"]
    validation_county_forward = partition_reports["county_validation"]["reference_to_source"]
    test_county_forward = partition_reports["county_test"]["reference_to_source"]
    gates = {
        "nonrigid_validation_candidate": bool(selected["eligible"]),
        "positive_jacobian_no_fold_condition": bool(selected["regularity"]["passed"]),
        "independent_admin_test_tail": _holdout_gate(test_admin, config),
        "independent_outer_pacific_test_microsegments": _supported_coast_gate(
            named_pacific_reports["outer_pacific_test_microsegments"], config
        ),
        "independent_named_bay_test_microsegments": bool(
            all(
                _supported_coast_gate(report, config, bay=True)
                for name, report in named_pacific_reports.items()
                if not name.startswith("outer_pacific")
            )
        ),
        "observed_internal_bay_suisun_delta_subset": bool(
            observed_internal_water["source_observed_pixel_count"]
            >= config.holdout_minimum_visible_count
            and observed_internal_water["p90_px"] <= config.admin_holdout_p90_limit_px
            and observed_internal_water["within_8px_fraction"]
            >= config.bay_holdout_within_8_minimum
        ),
        "full_mapbox_state_median": state_forward["median_px"] <= config.state_median_limit_px,
        "full_mapbox_state_support": state_forward["within_8px_fraction"] >= config.state_within_8_minimum,
        "full_mapbox_state_symmetric_overlap": state["f1"] >= config.state_f1_minimum,
        "pacific_admin_geographic_tail": bool(balanced["passed"]),
        "independent_full_mapbox_county_median": county_forward["median_px"] <= config.county_median_limit_px,
        "independent_full_mapbox_county_support": county_forward["within_8px_fraction"] >= config.county_within_8_minimum,
        "independent_full_mapbox_county_symmetric_overlap": counties["f1"] >= config.county_f1_minimum,
        "county_validation_median": validation_county_forward["median_px"] <= config.county_median_limit_px,
        "county_validation_support": validation_county_forward["within_8px_fraction"] >= config.county_within_8_minimum,
        "county_test_median": test_county_forward["median_px"] <= config.county_median_limit_px,
        "county_test_support": test_county_forward["within_8px_fraction"] >= config.county_within_8_minimum,
        "partial_source_visible_land_support": float(np.mean(visible)) >= config.minimum_visible_land_fraction,
        "visible_land_containment": containment >= config.minimum_visible_land_containment,
        "serialized_bidirectional_roundtrip": bool(roundtrip["passed"]),
    }
    for name, report in named_pacific_reports.items():
        if name.startswith("outer_pacific"):
            continue
        gates[f"independent_{name}"] = _supported_coast_gate(
            report, config, bay=True
        )
    strict_pass = bool(all(gates.values()))

    source_evidence_path = output_root / "source-evidence.png"
    evidence = np.zeros_like(source.rgb_working)
    evidence[source.pacific] = (90, 90, 90)
    evidence[source.native_internal_water] = (90, 170, 255)
    evidence[source.native_internal_water_edge] = (255, 255, 255)
    evidence[source.topology.state_coast] = (60, 240, 90)
    evidence[source.county_topology] = (45, 180, 225)
    Image.fromarray(evidence).save(source_evidence_path)
    source_extent_artifacts, source_extent_paths = _write_mask_artifacts(
        output_root, "source-observable-extent", source_extents
    )
    source_water_paths: list[Path] = []
    source_water_hashes: dict[str, Any] = {}
    for name, mask in {
        "source_native_water": source.native_water,
        "source_native_water_edge": source.native_water_edge,
        "source_native_internal_water": source.native_internal_water,
        "source_native_internal_water_edge": source.native_internal_water_edge,
        "source_native_water_original": source.native_water_original,
        "source_native_water_edge_original": source.native_water_edge_original,
        "source_native_internal_water_original": source.native_internal_water_original,
        "source_native_internal_water_edge_original": source.native_internal_water_edge_original,
    }.items():
        path = output_root / f"{name.replace('_', '-')}.png"
        Image.fromarray(mask.astype(np.uint8) * 255).save(path)
        source_water_paths.append(path)
        source_water_hashes[name] = {
            "path": str(path),
            "sha256": _sha256(path),
            "pixel_count": int(np.count_nonzero(mask)),
        }
    reference_semantics_path = output_root / "mapbox-source-semantic-primary.png"
    reference_image = np.zeros((*semantics.primary.shape, 3), np.uint8)
    reference_image[semantics.california_admin] = (255, 75, 210)
    reference_image[semantics.pacific_coast] = (255, 210, 40)
    reference_image[semantics.unsupported_inland_water] = (80, 80, 80)
    Image.fromarray(reference_image).save(reference_semantics_path)
    partition_paths: list[Path] = []
    partition_hashes: dict[str, Any] = {}
    for name, mask in county_partitions.items():
        path = output_root / f"mapbox-{name.replace('_', '-')}.png"
        Image.fromarray(mask.astype(np.uint8) * 255).save(path)
        partition_paths.append(path)
        partition_hashes[name] = {
            "path": str(path),
            "sha256": _sha256(path),
            "pixel_count": int(np.count_nonzero(mask)),
        }
    source_assignment_hashes: dict[str, Any] = {}
    for name, mask in source_assignment_masks.items():
        path = output_root / f"{name.replace('_', '-')}.png"
        Image.fromarray(mask.astype(np.uint8) * 255).save(path)
        partition_paths.append(path)
        source_assignment_hashes[name] = {
            "path": str(path),
            "sha256": _sha256(path),
            "pixel_count": int(np.count_nonzero(mask)),
        }
    named_mask_hashes: dict[str, Any] = {}
    for name, mask in named_pacific_masks.items():
        path = output_root / f"mapbox-{name.replace('_', '-')}.png"
        Image.fromarray(mask.astype(np.uint8) * 255).save(path)
        partition_paths.append(path)
        named_mask_hashes[name] = {
            "path": str(path),
            "sha256": _sha256(path),
            "pixel_count": int(np.count_nonzero(mask)),
            "role": (
                "validation_model_selection"
                if name.endswith("validation_microsegments")
                else "one_shot_test_after_model_selection"
            ),
        }
    overlay_path = output_root / "selected-overlay.png"
    _overlay(
        source.rgb_working,
        source.topology,
        rendered_state,
        rendered_counties,
        overlay_path,
    )
    water_overlay_path = output_root / "selected-water-overlay.png"
    internal_water_overlay_path = output_root / "selected-internal-water-overlay.png"
    water_overlay_diagnostics = _water_overlay(
        source.rgb_working,
        source,
        rendered_pacific,
        water_overlay_path,
        internal_water_overlay_path,
    )
    water_overlay_diagnostics.update(
        {
            "full_path": str(water_overlay_path),
            "full_sha256": _sha256(water_overlay_path),
            "internal_crop_path": str(internal_water_overlay_path),
            "internal_crop_sha256": _sha256(internal_water_overlay_path),
        }
    )
    transform_path = output_root / "selected-transform.json"
    transform_path.write_text(json.dumps(transform, indent=2, sort_keys=True) + "\n")

    serializable_candidates = []
    for item in candidates:
        serializable_candidates.append({key: value for key, value in item.items() if key != "warp"})
    report = {
        "schema_version": 1,
        "kind": EXPERIMENT_KIND,
        "status": "strict_pass" if strict_pass else "blocked",
        "stop_reason": "all_strict_gates_passed" if strict_pass else "strict_gate_failure",
        "inputs": {
            "source_path": str(source_path),
            "source_sha256": _sha256(source_path),
            "reference_manifest_path": str(reference_manifest_path),
            "reference_manifest_sha256": _sha256(reference_manifest_path),
            "mapbox_reference": dict(reference.pin),
        },
        "authority": {
            "original_source_authoritative": True,
            "pinned_mapbox_authoritative": True,
            "manual_inputs_used": False,
            "prior_run_inputs_used": False,
            "county_png_used": False,
            "census_used": False,
            "pacific_and_bay_used_for_control_target_derivation": False,
            "pacific_and_named_bay_validation_used_for_model_selection": True,
            "pacific_and_named_bay_test_used_for_model_selection": False,
            "observed_suisun_delta_subset_evaluated_only_after_model_selection": True,
            "county_training_partition_used_for_global_seed": True,
            "county_validation_partition_used_only_for_model_selection": True,
            "county_test_partition_used_once_after_full_model_selection": True,
            "heldout_partition_masks_used_only_for_source_assignment_exclusion": True,
            "line_visibility_uses_source_only_partial_map_extent": True,
            "pacific_and_neighboring_state_remain_observed_negative_evidence": True,
        },
        "config": asdict(config),
        "source": {
            "original_shape": list(source.rgb_original.shape[:2]),
            "working_shape": list(source.rgb_working.shape[:2]),
            "working_scale_from_original": source.working_scale,
            "diagnostics": dict(source.hypothesis.diagnostics),
            "native_water_diagnostics": dict(source.water_diagnostics),
            "native_water_artifacts": source_water_hashes,
            "observable_extent_diagnostics": source_extent_diagnostics,
            "observable_extent_artifacts": source_extent_artifacts,
        },
        "mapbox_semantics": dict(semantics.diagnostics),
        "seed": seed_diagnostics,
        "training": control_diagnostics,
        "candidates": serializable_candidates,
        "selected_candidate_id": selected["id"],
        "selected_candidate": {key: value for key, value in selected.items() if key != "warp"},
        "independent_test_admin_segments": test_admin,
        "strict_validation": {
            "state_coast": state,
            "geographically_balanced_primary": balanced,
            "named_pacific_holdouts": named_pacific_reports,
            "named_pacific_holdout_artifacts": named_mask_hashes,
            "observed_native_internal_bay_suisun_delta": observed_internal_water,
            "observed_native_internal_water_overlay": water_overlay_diagnostics,
            "suisun_delta_policy": {
                "status": "supported_native_resolution_strict",
                "source_native_component": "strict_source_to_mapbox_gate_after_selection",
                "mapbox_named_microsegments": (
                    "separate_validation_selection_and_one_shot_test_gates"
                ),
                "unsupported_detail_remainder": None,
            },
            "counties": counties,
            "county_partitions": partition_reports,
            "county_partition_artifacts": partition_hashes,
            "source_assignment_artifacts": source_assignment_hashes,
            "visible_land_fraction": float(np.mean(visible)),
            "visible_land_containment": containment,
            "serialized_roundtrip": roundtrip,
        },
        "strict_gates": {"passed": strict_pass, "checks": gates},
        "selected_transform": transform,
    }
    report_path = output_root / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return FarmsNonrigidResult(
        status=report["status"],
        stop_reason=report["stop_reason"],
        report_path=report_path,
        transform_path=transform_path,
        artifact_paths=(
            source_evidence_path,
            *source_extent_paths,
            *source_water_paths,
            reference_semantics_path,
            *partition_paths,
            overlay_path,
            water_overlay_path,
            internal_water_overlay_path,
            transform_path,
            report_path,
        ),
    )
