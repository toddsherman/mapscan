"""Deterministic, no-human alignment against a pinned Mapbox reference.

This module is intentionally isolated from MapScan's historical alignment
pipeline.  It accepts only an original categorical raster and the rasterized,
hash-pinned Mapbox reference contract.  It has no inputs for a prior transform,
manual arrows, ``county.png``, Census geometry, or the former canonical border.

The optimizer fits a small sequence of regular global transforms.  California's
state/coast perimeter is the primary signal.  Mapbox county lines have a lower
training weight and remain an independent holdout gate.  Samples are balanced by
geographic cell so dense urban linework cannot dominate the result.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import re
import shutil
import subprocess
from datetime import datetime, timezone
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image
from pyproj import CRS, Transformer, __version__ as pyproj_version, proj_version_str
from scipy.ndimage import distance_transform_edt
from scipy.optimize import differential_evolution
from scipy.spatial import cKDTree

from .experiment_log import NoHumanExperimentLog, automatic_provenance
from .farms_partial_topology import derive_farms_partial_topology
from .mapbox_water_reference import WEB_MERCATOR_HALF_WORLD, _decode_tile
from .source_alignment_hypotheses import (
    SourceHypothesisConfig,
    generate_source_alignment_hypotheses,
)


SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
REFERENCE_KIND = "mapbox_california_state_coast_water_counties"
REFERENCE_STATUS = "pinned_reference"
TRANSFORM_MODELS = ("similarity", "regular_affine")
PROJECTION_CANDIDATES = (
    ("web_mercator", "EPSG:3857"),
    ("california_albers", "EPSG:3310"),
    ("geographic", "EPSG:4326"),
    ("conus_albers", "EPSG:5070"),
    (
        "california_lambert_conformal_conic",
        "+proj=lcc +lat_1=33 +lat_2=45 +lat_0=39 +lon_0=-120 "
        "+datum=NAD83 +units=m +no_defs +type=crs",
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class PinnedMapboxReference:
    """Validated masks and pin metadata consumed by the automatic loop."""

    root: Path
    manifest_path: Path
    grid: Mapping[str, Any]
    state_coast: np.ndarray
    counties: np.ndarray
    state_land: np.ndarray
    water: np.ndarray
    waterways: np.ndarray | None
    pin: Mapping[str, Any]


@dataclass(frozen=True)
class AlignmentLoopConfig:
    """Conservative defaults for full or mostly-full categorical maps."""

    working_max_dimension: int = 900
    geographic_cells: tuple[int, int] = (4, 5)
    primary_samples_per_cell: int = 400
    county_samples_per_cell: int = 120
    county_training_weight: float = 0.12
    hydrography_training_weight: float = 0.02
    seed: int = 3407
    global_iterations: int = 120
    global_population: int = 16
    local_iterations: int = 450
    primary_median_limit_px: float = 3.5
    primary_p90_limit_px: float = 9.0
    primary_within_5px_minimum: float = 0.68
    county_median_limit_px: float = 6.0
    county_within_8px_minimum: float = 0.58
    minimum_primary_holdout_cells: int = 8
    minimum_county_holdout_cells: int = 5
    semantic_primary_median_limit_px: float = 4.5
    semantic_primary_p90_limit_px: float = 12.0
    semantic_primary_within_8px_minimum: float = 0.75
    semantic_primary_f1_minimum: float = 0.42
    semantic_state_tail_minimum_cell_pass_fraction: float = 0.70
    semantic_state_tail_minimum_axis_pass_fraction: float = 0.50
    semantic_state_tail_minimum_pixels_per_cell: int = 20
    semantic_county_median_limit_px: float = 7.0
    semantic_county_within_8px_minimum: float = 0.58
    semantic_county_f1_minimum: float = 0.32
    semantic_overlap_tolerance_px: float = 5.0
    county_observability_minimum_line_ratio: float = 0.10
    county_observability_minimum_occupied_cells: int = 5
    county_observability_connectivity_radius_px: int = 1
    county_observability_minimum_network_axis_span_fraction: float = 0.20
    county_observability_minimum_network_source_fraction: float = 0.10
    county_observability_minimum_long_strand_axis_fraction: float = 0.65
    county_observability_minimum_long_strand_cross_axis_fraction: float = 0.025
    county_observability_minimum_long_strand_count: int = 2
    source_hypothesis_generation_limit: int = 8
    source_hypothesis_shortlist_size: int = 4
    source_hypothesis_coarse_iterations: int = 24
    source_hypothesis_coarse_population: int = 8
    maximum_anisotropy: float = 1.30
    maximum_shear: float = 0.18
    maximum_rotation_degrees: float = 18.0
    minimum_scale_fraction: float = 0.55
    maximum_scale_fraction: float = 1.35
    minimum_relative_improvement: float = 0.0075
    source_coverage: str = "full_or_most_state"

    def __post_init__(self) -> None:
        if self.source_coverage != "full_or_most_state":
            raise ValueError(
                "The automatic Mapbox alignment loop currently supports only "
                "full_or_most_state categorical raster sources"
            )
        if self.working_max_dimension < 128:
            raise ValueError("working_max_dimension must be at least 128")
        if not 0.0 <= self.county_training_weight <= 0.30:
            raise ValueError("county_training_weight must be between 0 and 0.30")
        if not 0.0 <= self.hydrography_training_weight <= 0.20:
            raise ValueError(
                "hydrography_training_weight must be between 0 and 0.20"
            )
        if self.county_training_weight + self.hydrography_training_weight >= 1.0:
            raise ValueError("Auxiliary alignment weights must leave primary weight")
        if self.global_iterations < 1 or self.global_population < 4:
            raise ValueError("Global optimizer settings are invalid")
        if not 0.5 <= self.semantic_state_tail_minimum_cell_pass_fraction <= 1.0:
            raise ValueError("State-tail cell coverage must be between 0.5 and 1.0")
        if not 0.5 <= self.semantic_state_tail_minimum_axis_pass_fraction <= 1.0:
            raise ValueError("State-tail axis coverage must be between 0.5 and 1.0")
        if not 0.0 < self.county_observability_minimum_line_ratio < 1.0:
            raise ValueError("County observability line ratio must be between 0 and 1")
        if self.county_observability_connectivity_radius_px < 0:
            raise ValueError("County observability connectivity radius cannot be negative")
        if not (
            0.0
            < self.county_observability_minimum_network_axis_span_fraction
            < 1.0
        ):
            raise ValueError(
                "County observability network axis span must be between 0 and 1"
            )
        if not (
            0.0
            < self.county_observability_minimum_network_source_fraction
            < 1.0
        ):
            raise ValueError(
                "County observability network source fraction must be between 0 and 1"
            )
        if not (
            0.0
            < self.county_observability_minimum_long_strand_axis_fraction
            < 1.0
        ):
            raise ValueError(
                "County observability long-strand axis fraction must be between 0 and 1"
            )
        if not (
            0.0
            < self.county_observability_minimum_long_strand_cross_axis_fraction
            < 1.0
        ):
            raise ValueError(
                "County observability long-strand cross-axis fraction must be between 0 and 1"
            )
        if self.county_observability_minimum_long_strand_count < 2:
            raise ValueError("County observability requires at least two long strands")
        if not 1 <= self.source_hypothesis_generation_limit <= 24:
            raise ValueError("Source hypothesis generation limit must be between 1 and 24")
        if not 1 <= self.source_hypothesis_shortlist_size <= 8:
            raise ValueError("Source hypothesis shortlist size must be between 1 and 8")
        if self.source_hypothesis_coarse_iterations < 1:
            raise ValueError("Source hypothesis coarse iterations must be positive")
        if self.source_hypothesis_coarse_population < 4:
            raise ValueError("Source hypothesis coarse population must be at least 4")


@dataclass(frozen=True)
class AlignmentCandidate:
    iteration: int
    model: str
    normalized_reference_to_working_source_matrix: np.ndarray
    reference_pixel_to_source_original_matrix: np.ndarray | None
    source_original_to_reference_pixel_matrix: np.ndarray | None
    target_grid: Mapping[str, Any]
    objective: float
    scores: Mapping[str, Any]
    gates: Mapping[str, Any]
    regularity: Mapping[str, float | bool]
    status: str
    artifact_paths: tuple[Path, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class AutomaticAlignmentResult:
    status: str
    stop_reason: str
    candidates: tuple[AlignmentCandidate, ...]
    accepted: AlignmentCandidate | None


@dataclass(frozen=True)
class SourceSemanticEvidence:
    """Source-side geographic signals kept separate from thematic edges."""

    state_coast: np.ndarray
    counties: np.ndarray
    dark_cartographic_ink: np.ndarray
    border_connected_water: np.ndarray
    foreground_interior: np.ndarray
    foreground_boundary: np.ndarray
    county_observability_override: str | None = None
    source_adapter_id: str | None = None
    hydrography: np.ndarray | None = None


@dataclass(frozen=True)
class AlignmentSourceHypothesis:
    """Full-canvas source evidence variant eligible for Mapbox testing."""

    id: str
    variant_kind: str
    semantic: SourceSemanticEvidence
    source_only_score: float | None
    roi_working: tuple[int, int, int, int]
    generator_hypothesis_id: str | None
    diagnostics: Mapping[str, Any]
    artifact_paths: tuple[Path, ...] = field(default_factory=tuple)
    graticule_lonlat: tuple[tuple[float, float], ...] = field(default_factory=tuple)
    graticule_source_points: tuple[tuple[float, float], ...] = field(
        default_factory=tuple
    )


@dataclass(frozen=True)
class ProjectionContext:
    id: str
    crs: CRS
    crs_wkt: str
    crs_wkt_sha256: str
    reference_to_candidate: Transformer
    candidate_to_reference: Transformer
    normalization_center: np.ndarray
    normalization_scale: float


def _load_mask(root: Path, artifact: Mapping[str, Any], *, alpha: bool) -> np.ndarray:
    path = (root / str(artifact["path"])).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if _sha256(path) != str(artifact["sha256"]):
        raise ValueError(f"Pinned Mapbox artifact hash mismatch: {path}")
    image = np.asarray(Image.open(path))
    if alpha:
        if image.ndim != 3 or image.shape[2] != 4:
            raise ValueError(f"Expected RGBA Mapbox overlay: {path}")
        return image[:, :, 3] > 0
    if image.ndim == 3:
        image = image[:, :, 0]
    return image > 0


def _pinned_mapbox_waterway_mask(
    root: Path,
    manifest: Mapping[str, Any],
    grid: Mapping[str, Any],
    state_land: np.ndarray,
) -> np.ndarray | None:
    """Rasterize label-free waterways from the already pinned Streets-v8 tiles.

    Older synthetic/test reference manifests do not carry tile records, so the
    channel is explicitly unavailable there.  Every official tile used here is
    hash-checked against the immutable manifest before decoding.
    """

    records = manifest.get("tiles")
    if not isinstance(records, list) or not records:
        return None
    zoom = int(manifest.get("zoom", -1))
    if zoom < 0:
        return None
    width, height = int(grid["width"]), int(grid["height"])
    minimum_x, minimum_y, maximum_x, maximum_y = map(float, grid["bounds"])
    pixel_width = (maximum_x - minimum_x) / width
    pixel_height = (maximum_y - minimum_y) / height
    tile_span = WEB_MERCATOR_HALF_WORLD * 2.0 / (1 << zoom)
    waterways = np.zeros((height, width), dtype=np.uint8)

    def pixel_points(coordinates: Sequence[Sequence[float]], *, left: float, top: float, extent: int) -> np.ndarray:
        values = np.asarray(coordinates, dtype=np.float64)
        world_x = left + values[:, 0] / extent * tile_span
        world_y = top - values[:, 1] / extent * tile_span
        pixel_x = (world_x - minimum_x) / pixel_width - 0.5
        pixel_y = (maximum_y - world_y) / pixel_height - 0.5
        return np.rint(np.column_stack((pixel_x, pixel_y))).astype(np.int32)

    for record in records:
        relative = Path(str(record.get("path", "")))
        tile_path = (root / relative).resolve()
        if not tile_path.is_file() or _sha256(tile_path) != str(record.get("sha256")):
            raise ValueError(f"Pinned Mapbox tile hash mismatch: {tile_path}")
        decoded = _decode_tile(tile_path.read_bytes())
        layer = decoded.get("waterway")
        if not isinstance(layer, dict):
            continue
        tile_x, tile_y = int(record["x"]), int(record["y"])
        left = -WEB_MERCATOR_HALF_WORLD + tile_x * tile_span
        top = WEB_MERCATOR_HALF_WORLD - tile_y * tile_span
        extent = int(layer.get("extent", 4096))
        for feature in layer.get("features", []):
            properties = feature.get("properties", {})
            kind = str(properties.get("class", properties.get("type", "")))
            if kind not in {"river", "stream", "canal"}:
                continue
            geometry = feature.get("geometry", {})
            geometry_type = geometry.get("type")
            coordinates = geometry.get("coordinates", [])
            lines = [coordinates] if geometry_type == "LineString" else coordinates
            if geometry_type not in {"LineString", "MultiLineString"}:
                continue
            for line in lines:
                if len(line) >= 2:
                    cv2.polylines(
                        waterways,
                        [pixel_points(line, left=left, top=top, extent=extent)],
                        False,
                        1,
                        thickness=2,
                        lineType=cv2.LINE_8,
                    )
    if not np.any(waterways):
        return None
    california_scope = cv2.dilate(
        state_land.astype(np.uint8), np.ones((15, 15), dtype=np.uint8)
    ).astype(bool)
    waterways &= california_scope.astype(np.uint8)
    return waterways.astype(bool) if np.any(waterways) else None


def load_pinned_mapbox_reference(manifest_path: Path) -> PinnedMapboxReference:
    """Load Mapbox masks only after validating the no-legacy authority contract."""

    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") not in {1, 2}:
        raise ValueError("Unsupported Mapbox California reference schema")
    if manifest.get("schema_version") == 2:
        derivation = manifest.get("derivation", {})
        if (
            derivation.get("kind")
            != "pacific_side_detached_island_topology_v2"
            or derivation.get("raw_bytes_preserved_exactly") is not True
            or derivation.get("only_derived_masks_and_overlays_recomputed") is not True
        ):
            raise ValueError("Mapbox reference v2 derivation contract is incomplete")
    if manifest.get("status") != REFERENCE_STATUS or manifest.get("kind") != REFERENCE_KIND:
        raise ValueError("The alignment loop requires a pinned Mapbox California reference")
    authority = manifest.get("authority", {})
    forbidden = {
        "previous_mapscan_canonical_used": authority.get(
            "previous_mapscan_canonical_used"
        ),
        "county_png_used": authority.get("county_png_used"),
        "census_used": authority.get("census_used"),
    }
    if any(value is not False for value in forbidden.values()):
        raise ValueError(
            "Mapbox reference authority must explicitly exclude the old canonical "
            "boundary, county.png, and Census geometry"
        )
    grid = manifest.get("target_grid", {})
    if grid.get("crs") != "EPSG:3857":
        raise ValueError("Mapbox alignment reference must use EPSG:3857")
    width, height = int(grid.get("width", 0)), int(grid.get("height", 0))
    if width <= 0 or height <= 0:
        raise ValueError("Mapbox alignment reference grid is invalid")
    artifacts = manifest.get("artifacts", {})
    required = {"state_coast_overlay", "county_overlay", "state_land_mask", "water_mask"}
    if not required.issubset(artifacts):
        raise ValueError("Mapbox reference is missing required alignment artifacts")
    root = manifest_path.parent
    state_coast = _load_mask(root, artifacts["state_coast_overlay"], alpha=True)
    counties = _load_mask(root, artifacts["county_overlay"], alpha=True)
    state_land = _load_mask(root, artifacts["state_land_mask"], alpha=False)
    water = _load_mask(root, artifacts["water_mask"], alpha=False)
    shapes = {value.shape for value in (state_coast, counties, state_land, water)}
    if shapes != {(height, width)}:
        raise ValueError("Mapbox reference artifact dimensions disagree with target_grid")
    if not all(np.any(value) for value in (state_coast, counties, state_land, water)):
        raise ValueError("Mapbox reference contains an empty required evidence mask")
    waterways = _pinned_mapbox_waterway_mask(root, manifest, grid, state_land)
    style = manifest.get("style", {})
    tileset = manifest.get("tileset", {})
    pin = {
        "id": f"{style.get('id', 'mapbox-style')}@z{manifest.get('zoom')}",
        "style_sha256": style.get("sha256"),
        "tilejson_sha256": tileset.get("tilejson_sha256"),
        "tile_aggregate_sha256": manifest.get("tile_aggregate_sha256"),
        "manifest_sha256": _sha256(manifest_path),
    }
    if not all(isinstance(pin[key], str) and pin[key] for key in (
        "style_sha256", "tilejson_sha256", "tile_aggregate_sha256"
    )):
        raise ValueError("Mapbox reference pin hashes are incomplete")
    return PinnedMapboxReference(
        root=root,
        manifest_path=manifest_path,
        grid=grid,
        state_coast=state_coast,
        counties=counties,
        state_land=state_land,
        water=water,
        waterways=waterways,
        pin=pin,
    )


def _load_categorical_raster(path: Path) -> np.ndarray:
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError(
            "Automatic alignment currently supports categorical raster images only "
            f"({', '.join(sorted(SUPPORTED_SUFFIXES))})"
        )
    image = Image.open(path)
    if getattr(image, "n_frames", 1) != 1:
        raise ValueError("Multi-frame raster sources are not supported")
    if image.mode not in {"RGB", "RGBA", "P", "L"}:
        raise ValueError(f"Unsupported categorical raster mode: {image.mode}")
    return np.asarray(image.convert("RGB"))


def _resize_working(rgb: np.ndarray, maximum: int) -> tuple[np.ndarray, float]:
    height, width = rgb.shape[:2]
    scale = min(1.0, maximum / max(height, width))
    if scale == 1.0:
        return rgb, scale
    resized = cv2.resize(
        rgb,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def _source_edges(rgb: np.ndarray) -> np.ndarray:
    """Legacy diagnostic Canny mask; never used as alignment authority."""

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    fine = cv2.GaussianBlur(gray, (3, 3), 0)
    coarse = cv2.GaussianBlur(gray, (7, 7), 0)
    median = float(np.median(fine))
    lower = max(16, round(0.35 * median))
    upper = max(lower + 24, round(0.90 * median))
    edges = (cv2.Canny(fine, lower, upper, L2gradient=True) > 0) | (
        cv2.Canny(coarse, max(10, lower // 2), max(32, upper // 2), L2gradient=True)
        > 0
    )
    # Crop borders are layout, never geographic evidence.
    margin = max(3, round(min(edges.shape) * 0.008))
    edges[:margin] = False
    edges[-margin:] = False
    edges[:, :margin] = False
    edges[:, -margin:] = False
    if np.count_nonzero(edges) < 100:
        raise ValueError("The source has insufficient automatic boundary evidence")
    return edges


def _remove_small_components(mask: np.ndarray, minimum_pixels: int) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), 8
    )
    kept = np.zeros(mask.shape, dtype=bool)
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= minimum_pixels:
            kept |= labels == label
    return kept


def _border_connected_blue_water(rgb: np.ndarray) -> np.ndarray:
    """Select saturated blue water connected to an image border.

    This is deliberately conservative.  A false negative merely leaves the
    dark cartographic outline as state/coast evidence, while a false positive
    would make a thematic patch look like a coastline.
    """

    red = rgb[:, :, 0].astype(np.int16)
    green = rgb[:, :, 1].astype(np.int16)
    blue = rgb[:, :, 2].astype(np.int16)
    candidate = (
        (blue >= 95)
        & (blue - red >= 35)
        & (blue - green >= 18)
        & (green - red >= 8)
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidate.astype(np.uint8), 8
    )
    height, width = candidate.shape
    minimum_area = max(24, round(height * width * 0.0015))
    water = np.zeros(candidate.shape, dtype=bool)
    for label in range(1, count):
        x, y, component_width, component_height, area = map(int, stats[label])
        touches_border = (
            x == 0
            or y == 0
            or x + component_width == width
            or y + component_height == height
        )
        if touches_border and area >= minimum_area:
            water |= labels == label
    return water


def _border_connected_foreground(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a deterministic color-component foreground and its boundary."""

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    samples = lab.reshape(-1, 3).astype(np.float32)
    cv2.setRNGSeed(3407)
    _, labels, _ = cv2.kmeans(
        samples,
        10,
        None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5),
        4,
        cv2.KMEANS_PP_CENTERS,
    )
    labels = labels.reshape(lab.shape[:2])
    height, width = labels.shape
    background = np.zeros(labels.shape, dtype=bool)
    for cluster in range(10):
        count, components, stats, _ = cv2.connectedComponentsWithStats(
            (labels == cluster).astype(np.uint8), 8
        )
        for component in range(1, count):
            x, y, component_width, component_height, area = map(
                int, stats[component]
            )
            if area <= 200:
                continue
            if (
                x == 0
                or y == 0
                or x + component_width == width
                or y + component_height == height
            ):
                background |= components == component
    foreground = cv2.morphologyEx(
        (~background).astype(np.uint8),
        cv2.MORPH_CLOSE,
        np.ones((9, 9), np.uint8),
    ) > 0
    boundary = cv2.morphologyEx(
        foreground.astype(np.uint8),
        cv2.MORPH_GRADIENT,
        np.ones((3, 3), np.uint8),
    ) > 0
    margin = max(3, round(min(rgb.shape[:2]) * 0.008))
    boundary[:margin] = False
    boundary[-margin:] = False
    boundary[:, :margin] = False
    boundary[:, -margin:] = False
    return foreground, boundary


def _source_semantic_evidence(rgb: np.ndarray) -> SourceSemanticEvidence:
    """Derive geographic line evidence without admitting arbitrary Canny edges.

    County and state boundaries in the supported map family are neutral dark
    cartographic strokes.  Strongly chromatic thematic boundaries are excluded.
    The coastline additionally admits the edge of a large border-connected blue
    water body, which covers maps that encode land/water only by color.
    """

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    maximum = np.max(rgb, axis=2)
    minimum = np.min(rgb, axis=2)
    chroma = maximum.astype(np.int16) - minimum.astype(np.int16)
    local_background = cv2.GaussianBlur(gray, (9, 9), 0)
    contrast = local_background.astype(np.int16) - gray.astype(np.int16)
    ink = ((maximum <= 98) | ((gray <= 155) & (contrast >= 22))) & (chroma <= 58)

    # Single compression flecks and halftone dots are not line evidence.
    ink = _remove_small_components(ink, max(3, round(min(rgb.shape[:2]) * 0.004)))

    margin = max(3, round(min(rgb.shape[:2]) * 0.008))
    ink[:margin] = False
    ink[-margin:] = False
    ink[:, :margin] = False
    ink[:, -margin:] = False

    water = _border_connected_blue_water(rgb)
    foreground, foreground_boundary = _border_connected_foreground(rgb)
    water_edge = cv2.morphologyEx(
        water.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)
    ) > 0
    water_edge[:margin] = False
    water_edge[-margin:] = False
    water_edge[:, :margin] = False
    water_edge[:, -margin:] = False

    state_coast = ink | water_edge | foreground_boundary
    counties = ink & ~cv2.dilate(water_edge.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    if np.count_nonzero(state_coast) < 100:
        raise ValueError("The source has insufficient automatic state/coast evidence")
    if np.count_nonzero(counties) < 100:
        raise ValueError("The source has insufficient automatic county-line evidence")
    return SourceSemanticEvidence(
        state_coast=state_coast,
        counties=counties,
        dark_cartographic_ink=ink,
        border_connected_water=water,
        foreground_interior=foreground,
        foreground_boundary=foreground_boundary,
    )


def _exterior_mask_boundary(mask: np.ndarray) -> np.ndarray:
    """Return only the exterior contour, excluding text/island holes."""

    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    boundary = np.zeros(mask.shape, dtype=np.uint8)
    if contours:
        cv2.drawContours(boundary, contours, -1, 1, 2, cv2.LINE_8)
    margin = max(3, round(min(mask.shape) * 0.008))
    boundary[:margin] = 0
    boundary[-margin:] = 0
    boundary[:, :margin] = 0
    boundary[:, -margin:] = 0
    return boundary.astype(bool)


def _large_pale_blue_water(rgb: np.ndarray) -> np.ndarray:
    """Find a broad Pacific-ocean component in pale printed map palettes."""

    red = rgb[:, :, 0].astype(np.int16)
    green = rgb[:, :, 1].astype(np.int16)
    blue = rgb[:, :, 2].astype(np.int16)
    candidate = (
        (blue >= 110)
        & (blue - red >= 18)
        & (blue - green >= 7)
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidate.astype(np.uint8), 8
    )
    height, width = candidate.shape
    minimum_area = height * width * 0.02
    water = np.zeros(candidate.shape, dtype=np.uint8)
    for label in range(1, count):
        x, _y, _component_width, _component_height, area = map(
            int, stats[label]
        )
        if area >= minimum_area and x < width * 0.25:
            water[labels == label] = 1
    water = cv2.morphologyEx(
        water,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
    )
    return water.astype(bool)


def _dominant_neutral_pacific(
    rgb: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]] | None:
    """Find a flat neutral Pacific component in a grayscale thematic map.

    The selected color/component is inferred from the source alone.  It must
    be a large, low-chroma component adjacent to the left side of the map but
    cannot be white page background.  This intentionally reports its vertical
    coverage so a partial coast remains auditable and cannot masquerade as a
    full-state silhouette.
    """

    height, width = rgb.shape[:2]
    step = 8
    levels = math.ceil(256 / step)
    quantized = (rgb.astype(np.int32) // step).clip(0, levels - 1)
    codes = (
        quantized[:, :, 0] * levels * levels
        + quantized[:, :, 1] * levels
        + quantized[:, :, 2]
    )
    values, counts = np.unique(codes, return_counts=True)
    candidates: list[tuple[int, int, np.ndarray, tuple[int, ...]]] = []
    for code in values[np.argsort(counts)[-32:]]:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            (codes == int(code)).astype(np.uint8), 8
        )
        for label in range(1, count):
            x, y, component_width, component_height, area = map(
                int, stats[label]
            )
            if (
                area < height * width * 0.025
                or x > width * 0.06
                or component_height < height * 0.25
            ):
                continue
            component = labels == label
            color = np.median(rgb[component], axis=0)
            brightness = float(np.mean(color))
            chroma = float(np.max(color) - np.min(color))
            if not (150.0 <= brightness <= 235.0 and chroma <= 15.0):
                continue
            candidates.append(
                (
                    area,
                    label,
                    component,
                    (x, y, component_width, component_height),
                )
            )
    if not candidates:
        return None
    area, _label, seed, box = max(candidates, key=lambda item: item[0])
    color = np.median(rgb[seed], axis=0).astype(np.float32)
    distance = np.linalg.norm(rgb.astype(np.float32) - color, axis=2)
    tolerance_mask = distance <= 18.0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        tolerance_mask.astype(np.uint8), 8
    )
    overlapping = [
        label
        for label in range(1, count)
        if np.count_nonzero(seed & (labels == label))
    ]
    if not overlapping:
        return None
    selected_label = max(overlapping, key=lambda label: int(stats[label, 4]))
    pacific = labels == selected_label
    ys, xs = np.nonzero(pacific)
    return pacific, {
        "method": "source_only_dominant_left_adjacent_neutral_component",
        "mapbox_used_for_pacific_selection": False,
        "seed_area_px": int(area),
        "expanded_area_px": int(np.count_nonzero(pacific)),
        "median_rgb": [float(value) for value in color],
        "box": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
        "vertical_coverage_fraction": float((ys.max() - ys.min() + 1) / height),
    }


def _long_line_components(
    mask: np.ndarray,
    *,
    minimum_axis_fraction: float = 0.08,
) -> np.ndarray:
    """Discard glyph-sized components while retaining long boundary strokes."""

    closed = cv2.morphologyEx(
        mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(closed, 8)
    height, width = mask.shape
    kept = np.zeros(mask.shape, dtype=bool)
    for label in range(1, count):
        _x, _y, component_width, component_height, area = map(
            int, stats[label]
        )
        if area < 12:
            continue
        if (
            component_width >= width * minimum_axis_fraction
            or component_height >= height * minimum_axis_fraction
        ):
            kept |= labels == label
    return kept


def _saturated_blue_linear_network(rgb: np.ndarray) -> np.ndarray:
    """Recover printed blue hydrography while excluding neutral map linework."""

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    blue = (
        (hsv[:, :, 0] >= 85)
        & (hsv[:, :, 0] <= 125)
        # The pale Pacific in printed maps can have a blue hue but low
        # saturation.  Requiring a modestly saturated stroke keeps the
        # hydrography channel on printed rivers/lakes instead of turning the
        # ocean fill into a giant line component.
        & (hsv[:, :, 1] >= 60)
        & (hsv[:, :, 2] >= 55)
    )
    return _long_line_components(blue, minimum_axis_fraction=0.035)


def _closed_california_region_from_state_scale_strokes(
    rgb: np.ndarray,
    pacific: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]] | None:
    """Close a source-side California region from state-scale printed strokes.

    This adapter is deliberately source-only.  It finds the long northern,
    northeastern, diagonal, and southern neutral border segments, then follows
    the saturated-blue Colorado connection between the diagonal and southern
    segments.  The selected flood component must be a non-edge, north-south
    elongated region adjacent to the Pacific.  Mapbox is only allowed to rank
    and validate the resulting hypothesis later.
    """

    height, width = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    chroma = rgb.max(axis=2).astype(np.int16) - rgb.min(axis=2).astype(np.int16)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    blue = (
        (hsv[:, :, 0] >= 85)
        & (hsv[:, :, 0] <= 125)
        & (hsv[:, :, 1] > 35)
        & (hsv[:, :, 2] > 55)
    ).astype(np.uint8)
    blue_zone = cv2.dilate(blue, np.ones((5, 5), dtype=np.uint8))
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 35, 100)
    edges[(chroma > 50) | blue_zone.astype(bool)] = 0
    raw_lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 900.0,
        threshold=25,
        minLineLength=50,
        maxLineGap=25,
    )
    if raw_lines is None:
        return None
    segments: list[dict[str, Any]] = []
    for x1, y1, x2, y2 in raw_lines[:, 0]:
        length = math.hypot(x2 - x1, y2 - y1)
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        if angle > 90.0:
            angle -= 180.0
        if angle < -90.0:
            angle += 180.0
        segments.append(
            {
                "p1": (int(x1), int(y1)),
                "p2": (int(x2), int(y2)),
                "length": float(length),
                "angle": float(angle),
            }
        )

    pacific_near = cv2.dilate(
        pacific.astype(np.uint8), np.ones((31, 31), dtype=np.uint8)
    ).astype(bool)

    def near_pacific(point: tuple[int, int]) -> bool:
        return bool(pacific_near[point[1], point[0]])

    top_candidates = [
        segment
        for segment in segments
        if abs(segment["angle"]) < 12.0
        and segment["length"] > 0.15 * width
        and 0.03 * height
        < min(segment["p1"][1], segment["p2"][1])
        < 0.30 * height
        and (near_pacific(segment["p1"]) or near_pacific(segment["p2"]))
    ]
    if not top_candidates:
        return None
    top = max(top_candidates, key=lambda item: item["length"])
    top_east = max((top["p1"], top["p2"]), key=lambda point: point[0])
    top_west = min((top["p1"], top["p2"]), key=lambda point: point[0])

    def endpoint_distance(segment: Mapping[str, Any], point: tuple[int, int]) -> float:
        return min(
            math.hypot(endpoint[0] - point[0], endpoint[1] - point[1])
            for endpoint in (segment["p1"], segment["p2"])
        )

    vertical_candidates = [
        segment
        for segment in segments
        if abs(segment["angle"]) > 72.0
        and segment["length"] > 0.12 * height
        and max(segment["p1"][0], segment["p2"][0]) < 0.60 * width
    ]
    if not vertical_candidates:
        return None
    vertical = min(
        vertical_candidates,
        key=lambda item: (endpoint_distance(item, top_east), -item["length"]),
    )
    vertical_south = max(
        (vertical["p1"], vertical["p2"]), key=lambda point: point[1]
    )
    diagonal_candidates = [
        segment
        for segment in segments
        if 25.0 < segment["angle"] < 65.0
        and segment["length"] > 0.20 * min(height, width)
    ]
    if not diagonal_candidates:
        return None
    diagonal = min(
        diagonal_candidates,
        key=lambda item: (
            endpoint_distance(item, vertical_south),
            -item["length"],
        ),
    )
    diagonal_southeast = max(
        (diagonal["p1"], diagonal["p2"]),
        key=lambda point: point[0] + point[1],
    )

    def pacific_fraction(segment: Mapping[str, Any]) -> float:
        xs = np.linspace(segment["p1"][0], segment["p2"][0], 100)
        ys = np.linspace(segment["p1"][1], segment["p2"][1], 100)
        return float(
            np.mean(pacific[np.rint(ys).astype(int), np.rint(xs).astype(int)])
        )

    south_candidates = [
        segment
        for segment in segments
        if abs(segment["angle"]) < 20.0
        and segment["length"] > 0.10 * width
        and max(segment["p1"][1], segment["p2"][1]) > 0.68 * height
        and pacific_fraction(segment) < 0.20
    ]
    if not south_candidates:
        return None
    south = max(south_candidates, key=lambda item: item["length"])
    south_east = max((south["p1"], south["p2"]), key=lambda point: point[0])
    south_west = min((south["p1"], south["p2"]), key=lambda point: point[0])

    minimum_x = max(0, min(diagonal_southeast[0], south_east[0]) - 40)
    maximum_x = min(width, max(diagonal_southeast[0], south_east[0]) + 70)
    minimum_y = max(0, min(diagonal_southeast[1], south_east[1]) - 25)
    maximum_y = min(height, max(diagonal_southeast[1], south_east[1]) + 25)
    local_blue = blue[minimum_y:maximum_y, minimum_x:maximum_x]
    if not local_blue.size:
        return None
    near_blue = cv2.dilate(local_blue, np.ones((9, 9), dtype=np.uint8))
    path_cost = np.where(local_blue, 1.0, np.where(near_blue, 3.0, 30.0))
    start = (
        diagonal_southeast[1] - minimum_y,
        diagonal_southeast[0] - minimum_x,
    )
    goal = (south_east[1] - minimum_y, south_east[0] - minimum_x)
    local_height, local_width = path_cost.shape
    if not (
        0 <= start[0] < local_height
        and 0 <= start[1] < local_width
        and 0 <= goal[0] < local_height
        and 0 <= goal[1] < local_width
    ):
        return None
    distances = np.full((local_height, local_width), np.inf)
    previous = np.full((local_height, local_width, 2), -1, dtype=np.int32)
    distances[start] = 0.0
    queue: list[tuple[float, int, int]] = [(0.0, start[0], start[1])]
    neighbors = (
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    )
    while queue:
        distance, y, x = heapq.heappop(queue)
        if distance != distances[y, x]:
            continue
        if (y, x) == goal:
            break
        for delta_y, delta_x in neighbors:
            next_y, next_x = y + delta_y, x + delta_x
            if not (0 <= next_y < local_height and 0 <= next_x < local_width):
                continue
            next_distance = distance + path_cost[next_y, next_x] * (
                math.sqrt(2.0) if delta_x and delta_y else 1.0
            )
            if next_distance < distances[next_y, next_x]:
                distances[next_y, next_x] = next_distance
                previous[next_y, next_x] = (y, x)
                heapq.heappush(queue, (next_distance, next_y, next_x))
    if not np.isfinite(distances[goal]):
        return None
    path: list[tuple[int, int]] = []
    current = goal
    while current != start:
        path.append((current[1] + minimum_x, current[0] + minimum_y))
        prior = previous[current]
        if prior[0] < 0:
            return None
        current = (int(prior[0]), int(prior[1]))
    path.append((start[1] + minimum_x, start[0] + minimum_y))
    blue_fraction = float(
        np.mean([blue[y, x] for x, y in path])
    )
    if blue_fraction < 0.55:
        return None

    barrier = np.zeros((height, width), dtype=np.uint8)
    for segment in (top, vertical, diagonal, south):
        cv2.line(barrier, segment["p1"], segment["p2"], 1, 5, cv2.LINE_8)

    def nearest(point: tuple[int, int], options: Sequence[tuple[int, int]]) -> tuple[int, int]:
        return min(
            options,
            key=lambda candidate: math.hypot(
                candidate[0] - point[0], candidate[1] - point[1]
            ),
        )

    cv2.line(
        barrier,
        top_east,
        nearest(top_east, (vertical["p1"], vertical["p2"])),
        1,
        5,
        cv2.LINE_8,
    )
    cv2.line(
        barrier,
        vertical_south,
        nearest(vertical_south, (diagonal["p1"], diagonal["p2"])),
        1,
        5,
        cv2.LINE_8,
    )
    cv2.polylines(
        barrier, [np.asarray(path, dtype=np.int32)], False, 1, 5, cv2.LINE_8
    )
    cv2.line(
        barrier,
        south_east,
        nearest(south_east, path),
        1,
        5,
        cv2.LINE_8,
    )
    pacific_y, pacific_x = np.nonzero(pacific)
    if not len(pacific_x):
        return None
    for endpoint in (top_west, south_west):
        index = int(
            np.argmin(
                (pacific_x - endpoint[0]) ** 2
                + (pacific_y - endpoint[1]) ** 2
            )
        )
        cv2.line(
            barrier,
            endpoint,
            (int(pacific_x[index]), int(pacific_y[index])),
            1,
            5,
            cv2.LINE_8,
        )

    passable = ~(pacific.astype(bool) | barrier.astype(bool))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        passable.astype(np.uint8), 8
    )
    candidates: list[tuple[float, float, float, int]] = []
    for label in range(1, count):
        x, y, component_width, component_height, area = map(int, stats[label])
        touches_edge = (
            x == 0
            or y == 0
            or x + component_width == width
            or y + component_height == height
        )
        if touches_edge or area < height * width * 0.01:
            continue
        component = (labels == label).astype(np.uint8)
        adjacency = int(
            np.count_nonzero(
                cv2.dilate(component, np.ones((15, 15), dtype=np.uint8))
                & pacific.astype(np.uint8)
            )
        )
        vertical_fraction = component_height / height
        if adjacency and vertical_fraction > 0.60 and x < width * 0.35:
            candidates.append(
                (adjacency / area, vertical_fraction, area / (height * width), label)
            )
    if not candidates:
        return None
    selected_label = max(candidates)[-1]
    selected = (labels == selected_label).astype(np.uint8)
    contours, _ = cv2.findContours(
        selected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    interior = np.zeros_like(selected)
    cv2.drawContours(interior, contours, -1, 1, -1, cv2.LINE_8)
    boundary = cv2.morphologyEx(
        interior, cv2.MORPH_GRADIENT, np.ones((3, 3), dtype=np.uint8)
    )
    return (
        interior.astype(bool),
        boundary.astype(bool),
        barrier.astype(bool),
        {
            "method": "source_only_state_scale_strokes_pacific_and_colorado_flood",
            "mapbox_used_for_region_construction": False,
            "blue_path_fraction": blue_fraction,
            "selected_component_area_fraction": float(np.mean(interior)),
            "selected_component_vertical_fraction": max(candidates)[1],
            "top_segment": [*top["p1"], *top["p2"]],
            "vertical_segment": [*vertical["p1"], *vertical["p2"]],
            "diagonal_segment": [*diagonal["p1"], *diagonal["p2"]],
            "south_segment": [*south["p1"], *south["p2"]],
        },
    )


def _source_pacific_island_evidence(
    pacific: np.ndarray,
    state_interior: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Recover detached island land components from the source Pacific mask.

    Large title text in the ocean also cuts holes in a color-derived water
    mask.  The retained components therefore must be compact, south of the
    state midpoint, and geographically close to the source-only mainland
    silhouette.  No reference geometry participates in the selection.
    """

    height, width = pacific.shape
    non_water = (~pacific).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(non_water, 8)
    distance_to_mainland = cv2.distanceTransform(
        (~state_interior).astype(np.uint8), cv2.DIST_L2, 5
    )
    islands = np.zeros_like(pacific, dtype=bool)
    component_reports: list[dict[str, Any]] = []
    for label in range(1, count):
        x, y, component_width, component_height, area = map(int, stats[label])
        if (
            x == 0
            or y == 0
            or x + component_width == width
            or y + component_height == height
        ):
            continue
        component = labels == label
        minimum_distance = float(np.min(distance_to_mainland[component]))
        aspect = component_width / max(component_height, 1)
        retained = bool(
            20 <= area <= height * width * 0.002
            and y >= height * 0.62
            and x >= width * 0.25
            and minimum_distance <= width * 0.11
            and 0.20 <= aspect <= 8.0
        )
        if retained:
            islands |= component
        if retained or (y >= height * 0.62 and area >= 20):
            component_reports.append(
                {
                    "box": [x, y, component_width, component_height],
                    "area_px": area,
                    "aspect": float(aspect),
                    "minimum_mainland_distance_px": minimum_distance,
                    "retained": retained,
                }
            )
    boundary = cv2.morphologyEx(
        islands.astype(np.uint8),
        cv2.MORPH_GRADIENT,
        np.ones((3, 3), dtype=np.uint8),
    ).astype(bool)
    return islands, boundary, {
        "method": "source_only_detached_pacific_land_components",
        "mapbox_used_for_island_selection": False,
        "retained_component_count": int(
            sum(item["retained"] for item in component_reports)
        ),
        "retained_area_px": int(np.count_nonzero(islands)),
        "components": component_reports,
    }


def _tesseract_words(image_path: Path, *, page_segmentation_mode: int) -> list[dict[str, Any]]:
    """Return OCR words with boxes, or an empty list when OCR is unavailable."""

    executable = shutil.which("tesseract")
    if executable is None:
        return []
    completed = subprocess.run(
        [
            executable,
            str(image_path),
            "stdout",
            "--psm",
            str(page_segmentation_mode),
            "tsv",
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )
    if completed.returncode != 0:
        return []
    words: list[dict[str, Any]] = []
    lines = completed.stdout.splitlines()
    if not lines:
        return words
    for line in lines[1:]:
        fields = line.split("\t")
        if len(fields) < 12 or fields[0] != "5" or not fields[11].strip():
            continue
        try:
            words.append(
                {
                    "left": int(fields[6]),
                    "top": int(fields[7]),
                    "width": int(fields[8]),
                    "height": int(fields[9]),
                    "confidence": float(fields[10]),
                    "text": fields[11].strip(),
                }
            )
        except ValueError:
            continue
    return words


def _normalized_degree_label(text: str) -> tuple[float, str, str | None] | None:
    """Parse a compact OCR degree label and log one common digit correction."""

    compact = re.sub(r"\s+", "", text.upper())
    match = re.search(r"(\d{2,3})[^A-Z0-9]*([NWES])", compact)
    if match is None:
        return None
    raw = int(match.group(1))
    hemisphere = match.group(2)
    value = raw
    correction = None
    limit = 180 if hemisphere in {"E", "W"} else 90
    if value > limit and len(str(raw)) == 3 and str(raw).startswith("4"):
        corrected = int("1" + str(raw)[1:])
        if corrected <= limit:
            value = corrected
            correction = f"ocr_leading_4_normalized_to_1:{raw}->{corrected}"
    if value > limit:
        return None
    signed = -float(value) if hemisphere in {"W", "S"} else float(value)
    return signed, hemisphere, correction


def _labeled_lambert_graticule_controls(
    rgb: np.ndarray,
    hypothesis_root: Path,
    *,
    adapter_id: str,
) -> tuple[
    tuple[tuple[float, float], ...],
    tuple[tuple[float, float], ...],
    dict[str, Any],
    tuple[Path, ...],
]:
    """OCR edge labels and intersect them with long printed meridians.

    Longitude labels are connected to their source meridians by a deterministic
    Hough-line association at the lower map frame. Latitude labels supply the
    printed parallel ordinates. Their cross product produces source-only
    lon/lat controls for a native Lambert candidate; Mapbox is not consulted.
    """

    height, width = rgb.shape[:2]
    artifacts: list[Path] = []
    scale = 2
    bottom_y = round(height * 0.88)
    left_width = max(32, round(width * 0.12))
    bottom_path = hypothesis_root / f"{adapter_id}-longitude-labels.png"
    left_path = hypothesis_root / f"{adapter_id}-latitude-labels.png"
    projection_path = hypothesis_root / f"{adapter_id}-projection-label.png"
    Image.fromarray(rgb[bottom_y:, :]).resize(
        (width * scale, (height - bottom_y) * scale), Image.Resampling.LANCZOS
    ).save(bottom_path)
    Image.fromarray(rgb[:, :left_width]).resize(
        (left_width * scale, height * scale), Image.Resampling.LANCZOS
    ).save(left_path)
    projection_x = round(width * 0.48)
    projection_y2 = round(height * 0.16)
    Image.fromarray(rgb[:projection_y2, projection_x:]).resize(
        ((width - projection_x) * scale, projection_y2 * scale),
        Image.Resampling.LANCZOS,
    ).save(projection_path)
    artifacts.extend((bottom_path, left_path, projection_path))

    longitude_words = _tesseract_words(
        bottom_path, page_segmentation_mode=6
    )
    latitude_words = _tesseract_words(left_path, page_segmentation_mode=6)
    projection_words = _tesseract_words(
        projection_path, page_segmentation_mode=6
    )
    projection_text = " ".join(word["text"] for word in projection_words)
    projection_label_detected = all(
        token in projection_text.lower()
        for token in ("lambert", "conic", "conformal")
    )
    corrections: list[str] = []
    longitude_labels: list[tuple[float, float]] = []
    for word in longitude_words:
        parsed = _normalized_degree_label(str(word["text"]))
        if parsed is None or parsed[1] not in {"E", "W"}:
            continue
        value, _hemisphere, correction = parsed
        if correction:
            corrections.append(correction)
        source_x = (word["left"] + word["width"] / 2.0) / scale
        longitude_labels.append((value, source_x))
    latitude_labels: list[tuple[float, float]] = []
    for word in latitude_words:
        parsed = _normalized_degree_label(str(word["text"]))
        if parsed is None or parsed[1] not in {"N", "S"}:
            continue
        value, _hemisphere, correction = parsed
        if correction:
            corrections.append(correction)
        source_y = (word["top"] + word["height"] / 2.0) / scale
        latitude_labels.append((value, source_y))
    longitude_labels = sorted(set(longitude_labels), key=lambda item: item[1])
    latitude_labels = sorted(set(latitude_labels), key=lambda item: item[1])

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 30, 100)
    raw_lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 1800.0,
        threshold=max(80, round(height * 0.10)),
        minLineLength=max(80, round(height * 0.30)),
        maxLineGap=max(12, round(height * 0.03)),
    )
    line_candidates: list[dict[str, float]] = []
    if raw_lines is not None:
        for x1, y1, x2, y2 in raw_lines[:, 0]:
            dx = float(x2 - x1)
            dy = float(y2 - y1)
            if abs(dy) < 1.0:
                continue
            angle = abs(math.degrees(math.atan2(dy, dx)))
            if angle < 80.0:
                continue
            slope = dx / dy
            intercept = float(x1) - slope * float(y1)
            line_candidates.append(
                {
                    "slope_x_per_y": slope,
                    "intercept_x": intercept,
                    "x_at_bottom": intercept + slope * (height - 1),
                    "length": math.hypot(dx, dy),
                    "absolute_dx": abs(dx),
                }
            )

    selected_meridians: list[tuple[float, float, float]] = []
    for longitude, label_x in longitude_labels:
        eligible = [
            item
            for item in line_candidates
            if abs(item["x_at_bottom"] - label_x) <= width * 0.06
        ]
        if not eligible:
            continue
        if label_x < width * 0.10:
            eligible.sort(
                key=lambda item: (
                    item["absolute_dx"] < width * 0.015,
                    -item["length"],
                    abs(item["x_at_bottom"] - label_x),
                )
            )
        else:
            eligible.sort(
                key=lambda item: (
                    abs(item["x_at_bottom"] - label_x),
                    -item["length"],
                )
            )
        selected = eligible[0]
        selected_meridians.append(
            (
                longitude,
                selected["slope_x_per_y"],
                selected["intercept_x"],
            )
        )

    lonlat: list[tuple[float, float]] = []
    source_points: list[tuple[float, float]] = []
    for latitude, source_y in latitude_labels:
        for longitude, slope, intercept in selected_meridians:
            source_x = intercept + slope * source_y
            if 0.0 <= source_x < width:
                lonlat.append((longitude, latitude))
                source_points.append((source_x, source_y))

    diagnostic_path = hypothesis_root / f"{adapter_id}-graticule-controls.png"
    diagnostic = rgb.copy()
    for _longitude, slope, intercept in selected_meridians:
        cv2.line(
            diagnostic,
            (round(intercept), 0),
            (round(intercept + slope * (height - 1)), height - 1),
            (255, 65, 190),
            2,
            cv2.LINE_AA,
        )
    for source_x, source_y in source_points:
        cv2.circle(
            diagnostic,
            (round(source_x), round(source_y)),
            4,
            (70, 255, 140),
            -1,
            cv2.LINE_AA,
        )
    Image.fromarray(diagnostic).save(diagnostic_path)
    artifacts.append(diagnostic_path)
    diagnostics = {
        "projection_label_detected": projection_label_detected,
        "projection_ocr_text": projection_text,
        "longitude_labels": [
            {"longitude": value, "source_x": position}
            for value, position in longitude_labels
        ],
        "latitude_labels": [
            {"latitude": value, "source_y": position}
            for value, position in latitude_labels
        ],
        "selected_meridian_count": len(selected_meridians),
        "control_point_count": len(lonlat),
        "ocr_corrections": sorted(set(corrections)),
        "mapbox_used_for_control_detection": False,
    }
    manifest_path = hypothesis_root / f"{adapter_id}-graticule-controls.json"
    manifest_path.write_text(json.dumps(diagnostics, indent=2) + "\n")
    artifacts.append(manifest_path)
    if not projection_label_detected or len(selected_meridians) < 3 or len(latitude_labels) < 2:
        return (), (), diagnostics, tuple(artifacts)
    return (
        tuple(lonlat),
        tuple(source_points),
        diagnostics,
        tuple(artifacts),
    )


def _warm_ordered_gradient_state(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Recover a filled red/orange/yellow statewide gradient map panel."""

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    warm = (
        (hsv[:, :, 0] <= 35)
        & (hsv[:, :, 1] >= 55)
        & (hsv[:, :, 2] >= 80)
    )
    closed = cv2.morphologyEx(
        warm.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)),
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(closed, 8)
    height, width = warm.shape
    qualifying: list[tuple[int, int]] = []
    for label in range(1, count):
        x, _y, _component_width, _component_height, area = map(
            int, stats[label]
        )
        if area >= height * width * 0.05 and x < width * 0.50:
            qualifying.append((area, label))
    if not qualifying:
        return np.zeros(warm.shape, dtype=bool), np.zeros(warm.shape, dtype=bool)
    _, selected_label = max(qualifying)
    selected = (labels == selected_label).astype(np.uint8)
    contours, _ = cv2.findContours(
        selected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    interior = np.zeros(selected.shape, dtype=np.uint8)
    cv2.drawContours(interior, contours, -1, 1, -1, cv2.LINE_8)
    boundary = cv2.morphologyEx(
        interior, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)
    )
    return interior.astype(bool), boundary.astype(bool)


def _family_semantic_hypothesis(
    rgb: np.ndarray,
    generic: SourceSemanticEvidence,
    source_family: str | None,
    hypothesis_root: Path,
) -> AlignmentSourceHypothesis | None:
    """Build one auditable source-family channel without consulting Mapbox."""

    graticule_lonlat: tuple[tuple[float, float], ...] = ()
    graticule_source_points: tuple[tuple[float, float], ...] = ()
    extra_artifacts: tuple[Path, ...] = ()
    if source_family == "continuous_numeric_ramp":
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        purple = (
            (hsv[:, :, 0] >= 125)
            & (hsv[:, :, 0] <= 175)
            & (hsv[:, :, 1] >= 90)
            & (hsv[:, :, 2] <= 190)
        )
        dark_red = (
            ((hsv[:, :, 0] <= 8) | (hsv[:, :, 0] >= 175))
            & (hsv[:, :, 1] >= 120)
            & (hsv[:, :, 2] <= 170)
        )
        water = _large_pale_blue_water(rgb)
        coastline = _exterior_mask_boundary(water)
        labeled_boundaries = _long_line_components(purple | dark_red)
        adapter_id = "labeled-lambert-boundary-and-pale-pacific-v1"
        state_coast = coastline | labeled_boundaries
        (
            graticule_lonlat,
            graticule_source_points,
            graticule_diagnostics,
            graticule_artifacts,
        ) = _labeled_lambert_graticule_controls(
            rgb,
            hypothesis_root,
            adapter_id=adapter_id,
        )
        extra_artifacts = graticule_artifacts
        semantic = SourceSemanticEvidence(
            state_coast=state_coast,
            counties=generic.counties,
            dark_cartographic_ink=generic.dark_cartographic_ink,
            border_connected_water=water,
            foreground_interior=~water,
            foreground_boundary=coastline,
            county_observability_override="absent",
            source_adapter_id=adapter_id,
        )
        diagnostics: dict[str, Any] = {
            "source_family": source_family,
            "evidence_channels": [
                "printed_purple_state_boundary",
                "printed_dark_red_international_boundary",
                "pale_pacific_exterior_contour",
            ],
            "projection_label": "Lambert Conic Conformal Projection",
            "native_graticule": graticule_diagnostics,
            "county_channel": "absent",
        }
    elif source_family == "categorical_sparse":
        neutral_pacific = _dominant_neutral_pacific(rgb)
        if neutral_pacific is None:
            return None
        water, pacific_diagnostics = neutral_pacific
        try:
            partial = derive_farms_partial_topology(rgb, water)
        except ValueError:
            return None
        adapter_id = "farms-partial-california-topology-v2"
        semantic = SourceSemanticEvidence(
            state_coast=partial.state_coast,
            counties=partial.internal_topology,
            dark_cartographic_ink=(
                partial.state_coast | partial.internal_topology
            ),
            border_connected_water=water,
            foreground_interior=partial.foreground_interior,
            foreground_boundary=partial.foreground_boundary,
            source_adapter_id=adapter_id,
        )
        neighbor_path = hypothesis_root / f"{adapter_id}-neighbor-exclusion.png"
        topology_path = hypothesis_root / f"{adapter_id}-internal-topology.png"
        Image.fromarray(partial.neighboring_region.astype(np.uint8) * 255).save(
            neighbor_path
        )
        Image.fromarray(partial.internal_topology.astype(np.uint8) * 255).save(
            topology_path
        )
        extra_artifacts = (neighbor_path, topology_path)
        diagnostics = {
            "source_family": source_family,
            "evidence_channels": [
                "dominant_neutral_pacific_coast",
                "flat_nevada_panel_california_facing_edge",
                "neutral_internal_topology_clipped_to_source_california_support",
            ],
            "pacific": pacific_diagnostics,
            "partial_topology": partial.diagnostics,
            "layout_exclusion_working": partial.diagnostics[
                "layout_exclusion_working"
            ],
            "source_perimeter_capability": (
                "partial"
                if pacific_diagnostics["vertical_coverage_fraction"] < 0.80
                else "most_or_full_state"
            ),
            "county_channel": "automatically_observed",
        }
    elif source_family == "ordered_gradient_bands":
        interior, boundary = _warm_ordered_gradient_state(rgb)
        if np.count_nonzero(boundary) < 100:
            return None
        adapter_id = "warm-ordered-gradient-map-panel-v1"
        semantic = SourceSemanticEvidence(
            state_coast=boundary,
            counties=generic.counties,
            dark_cartographic_ink=generic.dark_cartographic_ink,
            border_connected_water=generic.border_connected_water,
            foreground_interior=interior,
            foreground_boundary=boundary,
            source_adapter_id=adapter_id,
        )
        diagnostics = {
            "source_family": source_family,
            "evidence_channels": [
                "largest_warm_ordered_gradient_component",
                "filled_source_state_interior",
            ],
            "county_channel": "automatically_observed",
        }
    elif source_family == "named_linear_and_polygon_features_without_legend":
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        chroma = rgb.max(axis=2).astype(np.int16) - rgb.min(axis=2).astype(np.int16)
        neutral_boundary = _long_line_components((gray < 135) & (chroma < 42))
        water = _large_pale_blue_water(rgb)
        coastline = _exterior_mask_boundary(water)
        closed_region = _closed_california_region_from_state_scale_strokes(
            rgb,
            water,
        )
        if closed_region is None:
            state_coast = coastline | neutral_boundary
            foreground_interior = ~water
            foreground_boundary = coastline
            adapter_id = "neutral-state-perimeter-and-hydrography-separated-v1"
            region_diagnostics: dict[str, Any] = {
                "method": "source_only_closed_region_unavailable",
                "mapbox_used_for_region_construction": False,
            }
            island_diagnostics: dict[str, Any] = {
                "method": "not_available_without_closed_mainland_region",
                "mapbox_used_for_island_selection": False,
            }
            region_evidence_channels: list[str] = []
            hydrography_support = ~water
        else:
            (
                foreground_interior,
                closed_boundary,
                closed_barrier,
                region_diagnostics,
            ) = closed_region
            # The flood interior supplies state-scale topology while the pale
            # Pacific contour retains source coast detail.  The printed neutral
            # boundary is allowed only inside a source-derived corridor around
            # that closed region.  This recovers the detailed Bay outline
            # without admitting unrelated layout/terrain ink elsewhere.
            coastline_scope = cv2.dilate(
                foreground_interior.astype(np.uint8),
                np.ones((9, 9), dtype=np.uint8),
            ).astype(bool)
            island_interior, island_boundary, island_diagnostics = (
                _source_pacific_island_evidence(water, foreground_interior)
            )
            boundary_scope = cv2.dilate(
                closed_boundary.astype(np.uint8),
                np.ones((101, 101), dtype=np.uint8),
            ).astype(bool)
            scoped_printed_boundary = _long_line_components(
                (gray < 165) & (chroma < 42),
                minimum_axis_fraction=0.05,
            ) & boundary_scope
            # California's southeastern boundary follows the Colorado River.
            # Keep saturated-blue linework only where it intersects the
            # automatically closed southeast perimeter corridor; coastal and
            # interior rivers remain hydrography rather than state evidence.
            source_height, source_width = closed_boundary.shape
            southeast = np.zeros_like(closed_boundary, dtype=bool)
            southeast[
                round(source_height * 0.35) :,
                round(source_width * 0.55) :,
            ] = True
            colorado_scope = cv2.dilate(
                closed_barrier.astype(np.uint8),
                np.ones((31, 31), dtype=np.uint8),
            ).astype(bool)
            colorado_border = (
                _saturated_blue_linear_network(rgb)
                & colorado_scope
                & southeast
            )
            state_coast = (
                (coastline & coastline_scope)
                | closed_boundary
                | island_boundary
                | scoped_printed_boundary
                | colorado_border
            )
            foreground_interior |= island_interior
            foreground_boundary = closed_boundary
            adapter_id = "closed-california-region-and-hydrography-v3"
            barrier_path = hypothesis_root / f"{adapter_id}-barrier.png"
            Image.fromarray(closed_barrier.astype(np.uint8) * 255).save(
                barrier_path
            )
            extra_artifacts = (barrier_path,)
            region_evidence_channels = [
                "neutral_linework_scoped_to_closed_state_perimeter",
                "saturated_blue_colorado_scoped_to_southeast_perimeter",
            ]
            hydrography_support = cv2.erode(
                foreground_interior.astype(np.uint8),
                np.ones((3, 3), dtype=np.uint8),
            ).astype(bool)
        hydrography = _saturated_blue_linear_network(rgb) & hydrography_support
        semantic = SourceSemanticEvidence(
            state_coast=state_coast,
            counties=generic.counties,
            dark_cartographic_ink=generic.dark_cartographic_ink,
            border_connected_water=water,
            foreground_interior=foreground_interior,
            foreground_boundary=foreground_boundary,
            county_observability_override="absent",
            source_adapter_id=adapter_id,
            hydrography=hydrography,
        )
        diagnostics = {
            "source_family": source_family,
            "evidence_channels": [
                "long_neutral_state_perimeter",
                "pale_pacific_exterior_contour",
                "saturated_blue_hydrography_auxiliary",
                *region_evidence_channels,
            ],
            "excluded_channel": "blue_hydrography_and_labels",
            "county_channel": "absent",
            "closed_region": region_diagnostics,
            "pacific_islands": island_diagnostics,
        }
    elif source_family == "overlapping_chromatic_and_grayscale":
        adapter_id = "thematic-province-lines-not-counties-v1"
        semantic = replace(
            generic,
            county_observability_override="absent",
            source_adapter_id=adapter_id,
        )
        diagnostics = {
            "source_family": source_family,
            "evidence_channels": ["generic_state_and_coast_perimeter"],
            "excluded_channel": "labeled_geomorphic_province_curves",
            "county_channel": "absent",
        }
    else:
        return None

    if np.count_nonzero(semantic.state_coast) < 100:
        return None
    boundary_path = hypothesis_root / f"{adapter_id}-state-boundary.png"
    foreground_path = hypothesis_root / f"{adapter_id}-foreground.png"
    Image.fromarray(semantic.state_coast.astype(np.uint8) * 255).save(boundary_path)
    Image.fromarray(semantic.foreground_interior.astype(np.uint8) * 255).save(
        foreground_path
    )
    family_artifacts = [boundary_path, foreground_path]
    if semantic.hydrography is not None:
        hydrography_path = hypothesis_root / f"{adapter_id}-hydrography.png"
        Image.fromarray(semantic.hydrography.astype(np.uint8) * 255).save(
            hydrography_path
        )
        family_artifacts.append(hydrography_path)
    return AlignmentSourceHypothesis(
        id=f"source-family--{adapter_id}",
        variant_kind="source_family_semantic_adapter",
        semantic=semantic,
        source_only_score=None,
        roi_working=(0, 0, rgb.shape[1], rgb.shape[0]),
        generator_hypothesis_id=None,
        diagnostics=diagnostics,
        artifact_paths=(*family_artifacts, *extra_artifacts),
        graticule_lonlat=graticule_lonlat,
        graticule_source_points=graticule_source_points,
    )


def _mask_digest(*masks: np.ndarray) -> str:
    digest = hashlib.sha256()
    for mask in masks:
        values = np.ascontiguousarray(mask, dtype=np.uint8)
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest()


def _hypothesis_artifact_record(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def _source_hypothesis_payload(
    hypothesis: AlignmentSourceHypothesis,
) -> dict[str, Any]:
    return {
        "id": hypothesis.id,
        "variant_kind": hypothesis.variant_kind,
        "generator_hypothesis_id": hypothesis.generator_hypothesis_id,
        "source_only_score": hypothesis.source_only_score,
        "roi_working": list(hypothesis.roi_working),
        "diagnostics": dict(hypothesis.diagnostics),
        "detected_graticule_intersection_count": len(
            hypothesis.graticule_lonlat
        ),
        "artifacts": [
            _hypothesis_artifact_record(path) for path in hypothesis.artifact_paths
        ],
        "authority": {
            "source_only_score_used_for_acceptance": False,
            "strict_pinned_mapbox_gates_required": True,
            "full_source_canvas_coordinates_preserved": True,
        },
    }


def _generate_alignment_source_hypotheses(
    source_path: Path,
    hypothesis_root: Path,
    rgb: np.ndarray,
    semantic: SourceSemanticEvidence,
    *,
    working_scale: float,
    config: AlignmentLoopConfig,
    source_family: str | None = None,
) -> tuple[AlignmentSourceHypothesis, ...]:
    """Create bounded full-canvas evidence variants at the aligner's scale."""

    generated = generate_source_alignment_hypotheses(
        source_path,
        hypothesis_root,
        config=SourceHypothesisConfig(
            working_max_dimension=config.working_max_dimension,
            maximum_hypotheses=config.source_hypothesis_generation_limit,
        ),
    )
    manifest = json.loads(generated.manifest_path.read_text())
    source_record = manifest.get("source", {})
    if source_record.get("working_shape") != [int(value) for value in rgb.shape[:2]]:
        raise ValueError(
            "Source alignment hypotheses were not generated at the aligner working shape"
        )
    if not math.isclose(
        float(source_record.get("working_scale_from_original", -1.0)),
        working_scale,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "Source alignment hypotheses were not generated at the aligner working scale"
        )

    height, width = rgb.shape[:2]
    common_artifacts = (
        generated.manifest_path,
        generated.layout_diagnostic_path,
        generated.legend_diagnostic_path,
    )
    variants: list[AlignmentSourceHypothesis] = [
        AlignmentSourceHypothesis(
            id="baseline-semantic",
            variant_kind="baseline_semantic",
            semantic=semantic,
            source_only_score=None,
            roi_working=(0, 0, width, height),
            generator_hypothesis_id=None,
            diagnostics={"baseline": True},
            artifact_paths=common_artifacts,
        )
    ]
    seen = {
        (
            _mask_digest(
            semantic.state_coast,
            semantic.counties,
            semantic.foreground_interior,
            ),
            semantic.county_observability_override,
        )
    }

    family_hypothesis = _family_semantic_hypothesis(
        rgb, semantic, source_family, hypothesis_root
    )
    if family_hypothesis is not None:
        family_signature = (
            _mask_digest(
                family_hypothesis.semantic.state_coast,
                family_hypothesis.semantic.counties,
                family_hypothesis.semantic.foreground_interior,
            ),
            family_hypothesis.semantic.county_observability_override,
        )
        if family_signature not in seen:
            seen.add(family_signature)
            variants.append(family_hypothesis)

    for canvas in generated.canvases:
        x1, y1, x2, y2 = canvas.box_working
        roi = np.zeros((height, width), dtype=bool)
        roi[y1:y2, x1:x2] = True
        candidate = SourceSemanticEvidence(
            state_coast=semantic.state_coast & roi,
            counties=semantic.counties & roi,
            dark_cartographic_ink=semantic.dark_cartographic_ink & roi,
            border_connected_water=semantic.border_connected_water & roi,
            foreground_interior=semantic.foreground_interior & roi,
            foreground_boundary=semantic.foreground_boundary & roi,
            county_observability_override=semantic.county_observability_override,
            source_adapter_id=semantic.source_adapter_id,
        )
        signature = (
            _mask_digest(
                candidate.state_coast,
                candidate.counties,
                candidate.foreground_interior,
            ),
            candidate.county_observability_override,
        )
        if signature in seen or np.count_nonzero(candidate.state_coast) < 100:
            continue
        seen.add(signature)
        roi_path = hypothesis_root / f"aligner-{canvas.id}-roi-mask.png"
        Image.fromarray(roi.astype(np.uint8) * 255).save(roi_path)
        variants.append(
            AlignmentSourceHypothesis(
                id=f"roi-cartographic--{canvas.id}",
                variant_kind="roi_clipped_cartographic_evidence",
                semantic=candidate,
                source_only_score=float(canvas.score),
                roi_working=canvas.box_working,
                generator_hypothesis_id=None,
                diagnostics={
                    "canvas_id": canvas.id,
                    "canvas_kind": canvas.kind,
                    **dict(canvas.diagnostics),
                },
                artifact_paths=(*common_artifacts, roi_path),
            )
        )

    for hypothesis in generated.hypotheses:
        support_path = hypothesis_root / hypothesis.artifacts["support_mask"]
        boundary_path = hypothesis_root / hypothesis.artifacts["state_boundary_mask"]
        overlay_path = hypothesis_root / hypothesis.artifacts["overlay"]
        support = np.asarray(Image.open(support_path).convert("L")) > 0
        boundary = np.asarray(Image.open(boundary_path).convert("L")) > 0
        if support.shape != (height, width) or boundary.shape != (height, width):
            raise ValueError("Source hypothesis masks do not preserve the full canvas")
        x1, y1, x2, y2 = hypothesis.roi_working
        roi = np.zeros((height, width), dtype=bool)
        roi[y1:y2, x1:x2] = True
        candidate = SourceSemanticEvidence(
            state_coast=boundary,
            counties=semantic.counties & roi,
            dark_cartographic_ink=semantic.dark_cartographic_ink & roi,
            border_connected_water=semantic.border_connected_water & roi,
            foreground_interior=support,
            foreground_boundary=boundary,
            county_observability_override=semantic.county_observability_override,
            source_adapter_id=semantic.source_adapter_id,
        )
        signature = (
            _mask_digest(
                candidate.state_coast,
                candidate.counties,
                candidate.foreground_interior,
            ),
            candidate.county_observability_override,
        )
        if signature in seen or np.count_nonzero(candidate.state_coast) < 100:
            continue
        seen.add(signature)
        variants.append(
            AlignmentSourceHypothesis(
                id=f"support-boundary--{hypothesis.id}",
                variant_kind="source_support_boundary",
                semantic=candidate,
                source_only_score=float(hypothesis.score),
                roi_working=hypothesis.roi_working,
                generator_hypothesis_id=hypothesis.id,
                diagnostics=dict(hypothesis.diagnostics),
                artifact_paths=(
                    *common_artifacts,
                    support_path,
                    boundary_path,
                    overlay_path,
                ),
            )
        )

    if source_family == "ordered_gradient_bands" and family_hypothesis is not None:
        # Gradient panels provide an unusually reliable filled-state interior,
        # while their thresholded exterior can smooth away bays and other
        # coastline detail.  Conversely, the source-only palette/support
        # hypotheses often recover the printed coastline well but contain only
        # the populated thematic bands, so their land-containment score is not
        # meaningful.  Combine the two independent source-side channels before
        # Mapbox ranking.  The shortlist and unchanged strict gates remain the
        # only authority for choosing or accepting one of these bounded hybrids.
        support_variants = sorted(
            (
                item
                for item in variants
                if item.variant_kind == "source_support_boundary"
            ),
            key=lambda item: (
                -(item.source_only_score or 0.0),
                item.id,
            ),
        )[:3]
        for support in support_variants:
            adapter_id = "warm-gradient-interior-with-source-boundary-v1"
            hybrid_semantic = SourceSemanticEvidence(
                state_coast=support.semantic.state_coast,
                counties=support.semantic.counties,
                dark_cartographic_ink=support.semantic.dark_cartographic_ink,
                border_connected_water=support.semantic.border_connected_water,
                foreground_interior=family_hypothesis.semantic.foreground_interior,
                foreground_boundary=support.semantic.state_coast,
                county_observability_override=(
                    support.semantic.county_observability_override
                ),
                source_adapter_id=adapter_id,
            )
            signature = (
                _mask_digest(
                    hybrid_semantic.state_coast,
                    hybrid_semantic.counties,
                    hybrid_semantic.foreground_interior,
                ),
                hybrid_semantic.county_observability_override,
            )
            if signature in seen:
                continue
            seen.add(signature)
            variants.append(
                AlignmentSourceHypothesis(
                    id=(
                        "source-family-hybrid--warm-gradient-interior--"
                        f"{support.generator_hypothesis_id or support.id}"
                    ),
                    variant_kind="source_family_boundary_hybrid",
                    semantic=hybrid_semantic,
                    source_only_score=support.source_only_score,
                    roi_working=support.roi_working,
                    generator_hypothesis_id=support.generator_hypothesis_id,
                    diagnostics={
                        "source_family": source_family,
                        "evidence_channels": [
                            "filled_warm_gradient_state_interior",
                            "source_palette_support_boundary",
                        ],
                        "boundary_hypothesis_id": support.id,
                        "mapbox_used_for_hybrid_construction": False,
                        "county_channel": "automatically_observed",
                    },
                    artifact_paths=tuple(
                        dict.fromkeys(
                            (
                                *family_hypothesis.artifact_paths,
                                *support.artifact_paths,
                            )
                        )
                    ),
                )
            )
    return tuple(variants)


def _balanced_split(
    mask: np.ndarray,
    cells: tuple[int, int],
    maximum_per_cell: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return train/holdout points with both sets represented in every usable cell."""

    rows, columns = cells
    height, width = mask.shape
    train: list[np.ndarray] = []
    holdout: list[np.ndarray] = []
    train_cells: list[int] = []
    holdout_cells: list[int] = []
    for row in range(rows):
        y1, y2 = round(row * height / rows), round((row + 1) * height / rows)
        for column in range(columns):
            x1, x2 = round(column * width / columns), round((column + 1) * width / columns)
            ys, xs = np.nonzero(mask[y1:y2, x1:x2])
            if len(xs) < 4:
                continue
            points = np.column_stack((xs + x1, ys + y1)).astype(np.float64)
            order = np.lexsort((points[:, 0], points[:, 1]))
            points = points[order]
            if len(points) > maximum_per_cell:
                indices = np.linspace(0, len(points) - 1, maximum_per_cell).round().astype(int)
                points = points[indices]
            cell_id = row * columns + column
            cell_train = points[::2]
            cell_holdout = points[1::2]
            if len(cell_train) and len(cell_holdout):
                train.append(cell_train)
                holdout.append(cell_holdout)
                train_cells.extend([cell_id] * len(cell_train))
                holdout_cells.extend([cell_id] * len(cell_holdout))
    if not train or not holdout:
        raise ValueError("Reference evidence cannot be geographically balanced")
    return (
        np.concatenate(train),
        np.concatenate(holdout),
        np.asarray(train_cells, dtype=np.int16),
        np.asarray(holdout_cells, dtype=np.int16),
    )


def _normalizer(mask: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    ys, xs = np.nonzero(mask)
    if len(xs) < 10:
        raise ValueError("Mapbox state/coast evidence is too small")
    center_x = (float(xs.min()) + float(xs.max())) / 2.0
    center_y = (float(ys.min()) + float(ys.max())) / 2.0
    height = max(float(ys.max() - ys.min()), 1.0)
    return np.asarray([center_x, center_y]), height, float(xs.min()), float(xs.max())


def _normalize_points(points: np.ndarray, center: np.ndarray, state_height: float) -> np.ndarray:
    return (points - center) / state_height


def _reference_pixels_to_web_mercator(
    points: np.ndarray, grid: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    minimum_x, minimum_y, maximum_x, maximum_y = map(float, grid["bounds"])
    width, height = int(grid["width"]), int(grid["height"])
    x = minimum_x + points[:, 0] / max(width - 1, 1) * (maximum_x - minimum_x)
    y = maximum_y - points[:, 1] / max(height - 1, 1) * (maximum_y - minimum_y)
    return x, y


def _web_mercator_to_reference_pixels(
    x: np.ndarray, y: np.ndarray, grid: Mapping[str, Any]
) -> np.ndarray:
    minimum_x, minimum_y, maximum_x, maximum_y = map(float, grid["bounds"])
    width, height = int(grid["width"]), int(grid["height"])
    pixel_x = (x - minimum_x) / (maximum_x - minimum_x) * max(width - 1, 1)
    pixel_y = (maximum_y - y) / (maximum_y - minimum_y) * max(height - 1, 1)
    return np.column_stack((pixel_x, pixel_y))


def _projection_contexts(reference: PinnedMapboxReference) -> tuple[ProjectionContext, ...]:
    ys, xs = np.nonzero(reference.state_coast)
    if not len(xs):
        raise ValueError("Mapbox state/coast mask is empty")
    sample_indices = np.linspace(0, len(xs) - 1, min(len(xs), 12000)).round().astype(int)
    reference_points = np.column_stack((xs[sample_indices], ys[sample_indices])).astype(
        np.float64
    )
    web_x, web_y = _reference_pixels_to_web_mercator(reference_points, reference.grid)
    contexts = []
    for projection_id, definition in PROJECTION_CANDIDATES:
        crs = CRS.from_user_input(definition)
        forward = Transformer.from_crs("EPSG:3857", crs, always_xy=True)
        inverse = Transformer.from_crs(crs, "EPSG:3857", always_xy=True)
        projected_x, projected_y = forward.transform(web_x, web_y)
        projected = np.column_stack((projected_x, projected_y)).astype(np.float64)
        finite = np.all(np.isfinite(projected), axis=1)
        if np.count_nonzero(finite) != len(projected):
            raise ValueError(f"Projection {projection_id} produced non-finite California points")
        minimum = np.min(projected, axis=0)
        maximum = np.max(projected, axis=0)
        scale = max(float(maximum[1] - minimum[1]), 1e-12)
        wkt = crs.to_wkt(version="WKT2_2019", pretty=False)
        contexts.append(
            ProjectionContext(
                id=projection_id,
                crs=crs,
                crs_wkt=wkt,
                crs_wkt_sha256=hashlib.sha256(wkt.encode("utf-8")).hexdigest(),
                reference_to_candidate=forward,
                candidate_to_reference=inverse,
                normalization_center=(minimum + maximum) / 2.0,
                normalization_scale=scale,
            )
        )
    return tuple(contexts)


def _project_reference_points(
    points: np.ndarray,
    context: ProjectionContext,
    grid: Mapping[str, Any],
) -> np.ndarray:
    web_x, web_y = _reference_pixels_to_web_mercator(points, grid)
    projected_x, projected_y = context.reference_to_candidate.transform(web_x, web_y)
    projected = np.column_stack((projected_x, projected_y)).astype(np.float64)
    if not np.all(np.isfinite(projected)):
        raise ValueError(f"Projection {context.id} produced non-finite coordinates")
    return _projected_to_candidate_normalized(projected, context)


def _projected_to_candidate_normalized(
    projected: np.ndarray, context: ProjectionContext
) -> np.ndarray:
    """Convert projected east/north coordinates to normalized image axes."""

    normalized = (
        np.asarray(projected, dtype=np.float64) - context.normalization_center
    ) / context.normalization_scale
    # Source rasters use image coordinates (y increases downward), whereas all
    # projected CRSs use northing (y increases upward).
    normalized[:, 1] *= -1.0
    return normalized


def _candidate_normalized_to_projected(
    normalized: np.ndarray, context: ProjectionContext
) -> np.ndarray:
    """Invert normalized image axes back to projected east/north coordinates."""

    projected = np.asarray(normalized, dtype=np.float64).copy()
    projected[:, 1] *= -1.0
    projected *= context.normalization_scale
    projected += context.normalization_center
    return projected


def _projection_evidence(
    pixel_evidence: Mapping[str, np.ndarray],
    context: ProjectionContext,
    grid: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    result = {}
    for name, values in pixel_evidence.items():
        result[name] = (
            values
            if name.endswith("_cells") or name.endswith("_source_mask")
            else _project_reference_points(values, context, grid)
        )
    return result


def _projection_round_trip(
    context: ProjectionContext,
    matrix: np.ndarray,
    grid: Mapping[str, Any],
) -> dict[str, Any]:
    width, height = int(grid["width"]), int(grid["height"])
    xs = np.linspace(0, width - 1, 7)
    ys = np.linspace(0, height - 1, 8)
    reference = np.asarray([(x, y) for y in ys for x in xs], dtype=np.float64)
    normalized = _project_reference_points(reference, context, grid)
    source = _transform(normalized, matrix)
    homogeneous = np.column_stack((source, np.ones(len(source))))
    recovered_normalized = (homogeneous @ np.linalg.inv(matrix).T)[:, :2]
    projected = _candidate_normalized_to_projected(recovered_normalized, context)
    web_x, web_y = context.candidate_to_reference.transform(
        projected[:, 0], projected[:, 1]
    )
    recovered = _web_mercator_to_reference_pixels(
        np.asarray(web_x), np.asarray(web_y), grid
    )
    error = np.linalg.norm(recovered - reference, axis=1)
    finite = bool(np.all(np.isfinite(source)) and np.all(np.isfinite(recovered)))
    return {
        "sample_count": int(len(reference)),
        "finite": finite,
        "maximum_error_px": float(np.max(error)) if finite else 1e6,
    }


def _transform_contract(
    normalized_reference_to_working_source: np.ndarray,
    *,
    reference_center: np.ndarray,
    reference_state_height: float,
    working_scale: float,
    source_original_shape: tuple[int, int],
    source_working_shape: tuple[int, int],
    target_grid: Mapping[str, Any],
) -> dict[str, Any]:
    """Return reusable original-source <-> Mapbox-reference pixel transforms."""

    reference_pixel_to_normalized = np.asarray(
        [
            [1.0 / reference_state_height, 0.0, -reference_center[0] / reference_state_height],
            [0.0, 1.0 / reference_state_height, -reference_center[1] / reference_state_height],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    reference_to_working = (
        normalized_reference_to_working_source @ reference_pixel_to_normalized
    )
    original_to_working = np.diag([working_scale, working_scale, 1.0])
    reference_to_original = np.linalg.inv(original_to_working) @ reference_to_working
    original_to_reference = np.linalg.inv(reference_to_original)
    return {
        "kind": "regular_global_mapbox_registration",
        "source_pixel_space": "original_raster",
        "reference_pixel_space": "pinned_mapbox_target_grid",
        "source_original_shape": list(map(int, source_original_shape)),
        "source_working_shape": list(map(int, source_working_shape)),
        "working_scale_from_original": float(working_scale),
        "reference_pixel_to_source_original_matrix": reference_to_original.tolist(),
        "source_original_to_reference_pixel_matrix": original_to_reference.tolist(),
        "target_grid": {
            "crs": str(target_grid["crs"]),
            "bounds": [float(value) for value in target_grid["bounds"]],
            "width": int(target_grid["width"]),
            "height": int(target_grid["height"]),
        },
        "resampling_contract": {
            "source_diagnostic": "linear",
            "categorical_class_ids": "nearest",
        },
    }


def _projection_transform_contract(
    candidate_normalized_to_working_source: np.ndarray,
    *,
    projection: ProjectionContext,
    working_scale: float,
    source_original_shape: tuple[int, int],
    source_working_shape: tuple[int, int],
    target_grid: Mapping[str, Any],
) -> dict[str, Any]:
    """Serialize an exact nonlinear source <-> Mapbox mapping as steps."""

    original_to_working = np.diag([working_scale, working_scale, 1.0])
    candidate_to_original = (
        np.linalg.inv(original_to_working) @ candidate_normalized_to_working_source
    )
    original_to_candidate = np.linalg.inv(candidate_to_original)
    contract: dict[str, Any] = {
        "kind": "projection_aware_mapbox_registration",
        "source_pixel_space": "original_raster",
        "reference_pixel_space": "pinned_mapbox_target_grid",
        "source_original_shape": list(map(int, source_original_shape)),
        "source_working_shape": list(map(int, source_working_shape)),
        "working_scale_from_original": float(working_scale),
        "projection": {
            "id": projection.id,
            "always_xy": True,
            "crs_wkt": projection.crs_wkt,
            "crs_wkt_sha256": projection.crs_wkt_sha256,
            "normalization_center": projection.normalization_center.tolist(),
            "normalization_scale": float(projection.normalization_scale),
            "projected_axis_order": ["easting_right", "northing_up"],
            "candidate_normalized_axis_order": ["image_x_right", "image_y_down"],
            "candidate_normalized_to_projected_operations": [
                "negate normalized y",
                "multiply x and y by normalization_scale",
                "add normalization_center",
            ],
            "pyproj_version": pyproj_version,
            "proj_version": proj_version_str,
        },
        "candidate_normalized_to_source_original_matrix": candidate_to_original.tolist(),
        "source_original_to_candidate_normalized_matrix": original_to_candidate.tolist(),
        "reference_to_source_steps": [
            "reference pixel to EPSG:3857 using target_grid bounds",
            f"EPSG:3857 to {projection.id} using pinned WKT and always_xy",
            "subtract normalization_center, invert northing to image-y, and divide by normalization_scale",
            "apply candidate_normalized_to_source_original_matrix",
        ],
        "source_to_reference_steps": [
            "apply source_original_to_candidate_normalized_matrix",
            "multiply by normalization_scale, invert image-y to northing, and add normalization_center",
            f"{projection.id} to EPSG:3857 using pinned WKT and always_xy",
            "EPSG:3857 to reference pixel using target_grid bounds",
        ],
        "target_grid": {
            "crs": str(target_grid["crs"]),
            "bounds": [float(value) for value in target_grid["bounds"]],
            "width": int(target_grid["width"]),
            "height": int(target_grid["height"]),
        },
        "resampling_contract": {
            "source_diagnostic": "linear",
            "categorical_class_ids": "nearest",
        },
    }
    # EPSG:3857 is affine in target-grid pixels, so retain exact compatibility
    # matrices for existing consumers. Other candidates intentionally omit them.
    if projection.id == "web_mercator":
        width, height = int(target_grid["width"]), int(target_grid["height"])
        minimum_x, minimum_y, maximum_x, maximum_y = map(float, target_grid["bounds"])
        pixel_to_projected = np.asarray(
            [
                [(maximum_x - minimum_x) / max(width - 1, 1), 0.0, minimum_x],
                [0.0, -(maximum_y - minimum_y) / max(height - 1, 1), maximum_y],
                [0.0, 0.0, 1.0],
            ]
        )
        projected_to_normalized = np.asarray(
            [
                [1.0 / projection.normalization_scale, 0.0, -projection.normalization_center[0] / projection.normalization_scale],
                [0.0, -1.0 / projection.normalization_scale, projection.normalization_center[1] / projection.normalization_scale],
                [0.0, 0.0, 1.0],
            ]
        )
        reference_to_original = (
            candidate_to_original @ projected_to_normalized @ pixel_to_projected
        )
        contract["reference_pixel_to_source_original_matrix"] = reference_to_original.tolist()
        contract["source_original_to_reference_pixel_matrix"] = np.linalg.inv(
            reference_to_original
        ).tolist()
    return contract


def _matrix_from_parameters(
    parameters: Sequence[float], model: str, source_shape: tuple[int, int]
) -> np.ndarray:
    height, width = source_shape
    if model == "similarity":
        center_x, center_y, scale_fraction, rotation_degrees = map(float, parameters)
        x_ratio, shear = 1.0, 0.0
    elif model == "regular_affine":
        center_x, center_y, scale_fraction, x_ratio, rotation_degrees, shear = map(
            float, parameters
        )
    else:
        raise ValueError(f"Unsupported regular transform model: {model}")
    theta = math.radians(rotation_degrees)
    cosine, sine = math.cos(theta), math.sin(theta)
    scale = scale_fraction * height
    linear = scale * np.asarray(
        [
            [x_ratio * cosine + shear * sine, -x_ratio * sine + shear * cosine],
            [sine, cosine],
        ],
        dtype=np.float64,
    )
    return np.asarray(
        [
            [linear[0, 0], linear[0, 1], center_x * width],
            [linear[1, 0], linear[1, 1], center_y * height],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack((points, np.ones(len(points), dtype=np.float64)))
    return (homogeneous @ matrix.T)[:, :2]


def _regularity(
    matrix: np.ndarray,
    *,
    model: str,
    config: AlignmentLoopConfig,
    source_shape: tuple[int, int],
) -> dict[str, float | bool]:
    linear = matrix[:2, :2]
    determinant = float(np.linalg.det(linear))
    singular = np.linalg.svd(linear, compute_uv=False)
    anisotropy = float(singular.max() / max(singular.min(), 1e-12))
    rotation = math.degrees(math.atan2(linear[1, 0], linear[0, 0]))
    normalized = linear / max(float(np.linalg.norm(linear[:, 1])), 1e-12)
    shear = float(abs(np.dot(normalized[:, 0], normalized[:, 1])))
    source_height = max(float(source_shape[0]), 1.0)
    minimum_scale_fraction = float(singular.min() / source_height)
    maximum_scale_fraction = float(singular.max() / source_height)
    passed = bool(
        determinant > 0
        and anisotropy <= config.maximum_anisotropy
        and shear <= config.maximum_shear
        and abs(rotation) <= config.maximum_rotation_degrees
        and minimum_scale_fraction >= config.minimum_scale_fraction
        and maximum_scale_fraction <= config.maximum_scale_fraction
        and model in TRANSFORM_MODELS
    )
    return {
        "passed": passed,
        "determinant": determinant,
        "anisotropy": anisotropy,
        "shear": shear,
        "rotation_degrees": float(rotation),
        "minimum_scale_fraction": minimum_scale_fraction,
        "maximum_scale_fraction": maximum_scale_fraction,
    }


def _distances(
    normalized_points: np.ndarray,
    matrix: np.ndarray,
    distance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    transformed = _transform(normalized_points, matrix)
    height, width = distance.shape
    x = np.rint(transformed[:, 0]).astype(int)
    y = np.rint(transformed[:, 1]).astype(int)
    inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    values = np.full(len(normalized_points), math.hypot(width, height) * 0.12)
    values[inside] = distance[y[inside], x[inside]]
    return values, inside


def _summary(values: np.ndarray, inside: np.ndarray, cells: np.ndarray) -> dict[str, Any]:
    visible = values[inside]
    if not len(visible):
        return {
            "median_px": 1e6,
            "p90_px": 1e6,
            "within_5px_fraction": 0.0,
            "within_8px_fraction": 0.0,
            "visible_fraction": 0.0,
            "visible_cell_count": 0,
        }
    return {
        "median_px": float(np.median(visible)),
        "p90_px": float(np.quantile(visible, 0.90)),
        "within_5px_fraction": float(np.mean(visible <= 5.0)),
        "within_8px_fraction": float(np.mean(visible <= 8.0)),
        "visible_fraction": float(np.mean(inside)),
        "visible_cell_count": int(len(np.unique(cells[inside]))),
    }


def _score_candidate(
    matrix: np.ndarray,
    model: str,
    distance: np.ndarray | Mapping[str, np.ndarray],
    evidence: Mapping[str, np.ndarray],
    config: AlignmentLoopConfig,
) -> tuple[float, dict[str, Any], dict[str, Any], dict[str, float | bool]]:
    distance_shape = (
        distance["primary"].shape if isinstance(distance, Mapping) else distance.shape
    )
    regularity = _regularity(
        matrix, model=model, config=config, source_shape=distance_shape
    )
    summaries: dict[str, dict[str, Any]] = {}
    raw: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in ("primary_train", "primary_holdout", "county_train", "county_holdout"):
        if isinstance(distance, Mapping):
            channel = distance["county" if name.startswith("county") else "primary"]
        else:
            channel = distance
        values, inside = _distances(evidence[name], matrix, channel)
        raw[name] = (values, inside)
        summaries[name] = _summary(values, inside, evidence[f"{name}_cells"])
    hydrography_active = bool(
        isinstance(distance, Mapping)
        and "hydrography" in distance
        and "hydrography_train" in evidence
    )
    if hydrography_active:
        for name in ("hydrography_train", "hydrography_holdout"):
            values, inside = _distances(
                evidence[name], matrix, distance["hydrography"]
            )
            raw[name] = (values, inside)
            summaries[name] = _summary(
                values, inside, evidence[f"{name}_cells"]
            )
    primary_values, primary_inside = raw["primary_train"]
    county_values, county_inside = raw["county_train"]
    silhouette_containment = None
    silhouette_visible = None
    if "state_land_train" in evidence and "foreground_source_mask" in evidence:
        transformed_land = _transform(evidence["state_land_train"], matrix)
        height, width = distance_shape
        land_x = np.rint(transformed_land[:, 0]).astype(int)
        land_y = np.rint(transformed_land[:, 1]).astype(int)
        visible_land = (
            (land_x >= 0) & (land_x < width) & (land_y >= 0) & (land_y < height)
        )
        contained_land = np.zeros(len(transformed_land), dtype=bool)
        foreground = evidence["foreground_source_mask"].astype(bool)
        contained_land[visible_land] = foreground[
            land_y[visible_land], land_x[visible_land]
        ]
        silhouette_visible = float(np.mean(visible_land))
        silhouette_containment = float(np.mean(contained_land))
    primary_loss = float(
        np.mean(np.minimum(primary_values, 20.0))
        + 0.35 * min(float(np.quantile(primary_values, 0.90)), 100.0)
        + 30.0 * (1.0 - float(np.mean(primary_inside)))
        + (
            20.0 * (1.0 - silhouette_containment)
            if silhouette_containment is not None
            else 0.0
        )
    )
    county_loss = float(
        np.mean(np.minimum(county_values, 14.0))
        + 0.15 * min(float(np.quantile(county_values, 0.90)), 18.0)
        + 12.0 * (1.0 - float(np.mean(county_inside)))
    )
    hydrography_weight = (
        config.hydrography_training_weight if hydrography_active else 0.0
    )
    hydrography_loss = 0.0
    if hydrography_active:
        hydro_values, hydro_inside = raw["hydrography_train"]
        hydrography_loss = float(
            np.mean(np.minimum(hydro_values, 14.0))
            + 0.15 * min(float(np.quantile(hydro_values, 0.90)), 22.0)
            + 12.0 * (1.0 - float(np.mean(hydro_inside)))
        )
    objective = (
        (1.0 - config.county_training_weight - hydrography_weight) * primary_loss
        + config.county_training_weight * county_loss
        + hydrography_weight * hydrography_loss
    )
    if not regularity["passed"]:
        objective += 10.0
    primary = summaries["primary_holdout"]
    county = summaries["county_holdout"]
    gates = {
        "regular_transform": bool(regularity["passed"]),
        # Foreground perimeter fit is a coarse optimizer diagnostic. It is not
        # an acceptance gate because legends can occlude a large section of the
        # source state outline. The rendered semantic full-line gates below are
        # the authoritative state/coast validation.
        "geographically_balanced_primary": {
            "passed": primary["visible_cell_count"] >= config.minimum_primary_holdout_cells,
            "value": primary["visible_cell_count"],
            "minimum": config.minimum_primary_holdout_cells,
        },
        "county_holdout_median": {
            "passed": county["median_px"] <= config.county_median_limit_px,
            "value": county["median_px"],
            "maximum": config.county_median_limit_px,
        },
        "county_holdout_support": {
            "passed": county["within_8px_fraction"] >= config.county_within_8px_minimum,
            "value": county["within_8px_fraction"],
            "minimum": config.county_within_8px_minimum,
        },
        "geographically_balanced_counties": {
            "passed": county["visible_cell_count"] >= config.minimum_county_holdout_cells,
            "value": county["visible_cell_count"],
            "minimum": config.minimum_county_holdout_cells,
        },
    }
    if silhouette_containment is not None:
        gates["silhouette_land_containment"] = {
            "passed": silhouette_containment >= 0.90,
            "value": silhouette_containment,
            "minimum": 0.90,
        }
        summaries["silhouette"] = {
            "land_containment_fraction": silhouette_containment,
            "visible_fraction": silhouette_visible,
            "sample_count": int(len(evidence["state_land_train"])),
        }
    return float(objective), summaries, gates, regularity


def _fit_candidate(
    model: str,
    source_shape: tuple[int, int],
    distance: np.ndarray | Mapping[str, np.ndarray],
    evidence: Mapping[str, np.ndarray],
    config: AlignmentLoopConfig,
) -> tuple[np.ndarray, float, dict[str, Any], dict[str, Any], dict[str, float | bool]]:
    if model == "similarity":
        bounds = (
            (-0.1, 1.1),
            (-0.1, 1.1),
            (config.minimum_scale_fraction, config.maximum_scale_fraction),
            (-18.0, 18.0),
        )
    elif model == "regular_affine":
        bounds = (
            (-0.1, 1.1),
            (-0.1, 1.1),
            (config.minimum_scale_fraction, config.maximum_scale_fraction),
            (0.78, 1.22),
            (-18.0, 18.0),
            (-0.16, 0.16),
        )
    else:
        raise ValueError(f"Unsupported regular transform model: {model}")

    objective_config = (
        replace(config, county_training_weight=0.0)
        if model == "similarity"
        else config
    )

    def objective(parameters: Sequence[float]) -> float:
        matrix = _matrix_from_parameters(parameters, model, source_shape)
        score, _, _, _ = _score_candidate(
            matrix, model, distance, evidence, objective_config
        )
        return score

    global_result = differential_evolution(
        objective,
        bounds,
        seed=config.seed,
        popsize=config.global_population,
        maxiter=config.global_iterations,
        polish=True,
        workers=1,
        updating="immediate",
    )
    parameters = global_result.x
    matrix = _matrix_from_parameters(parameters, model, source_shape)
    score, summaries, gates, regularity = _score_candidate(
        matrix, model, distance, evidence, config
    )
    return matrix, score, summaries, gates, regularity


def _reference_to_source_matrix(
    normalized_reference_to_source: np.ndarray,
    reference_center: np.ndarray,
    reference_state_height: float,
) -> np.ndarray:
    return normalized_reference_to_source @ np.asarray(
        [
            [1.0 / reference_state_height, 0.0, -reference_center[0] / reference_state_height],
            [0.0, 1.0 / reference_state_height, -reference_center[1] / reference_state_height],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _warp_reference_mask(
    mask: np.ndarray,
    reference_to_source: np.ndarray,
    source_shape: tuple[int, int],
) -> np.ndarray:
    height, width = source_shape
    return cv2.warpPerspective(
        mask.astype(np.uint8),
        reference_to_source,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ) > 0


def _render_projected_reference_line(
    mask: np.ndarray,
    context: ProjectionContext,
    matrix: np.ndarray,
    grid: Mapping[str, Any],
    source_shape: tuple[int, int],
) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    points = np.column_stack((xs, ys)).astype(np.float64)
    projected = _project_reference_points(points, context, grid)
    transformed = np.rint(_transform(projected, matrix)).astype(np.int32)
    height, width = source_shape
    inside = (
        (transformed[:, 0] >= 0)
        & (transformed[:, 0] < width)
        & (transformed[:, 1] >= 0)
        & (transformed[:, 1] < height)
    )
    rendered = np.zeros(source_shape, dtype=np.uint8)
    rendered[transformed[inside, 1], transformed[inside, 0]] = 1
    return cv2.dilate(rendered, np.ones((2, 2), np.uint8)) > 0


def _render_projected_reference_land(
    mask: np.ndarray,
    context: ProjectionContext,
    matrix: np.ndarray,
    grid: Mapping[str, Any],
    source_shape: tuple[int, int],
) -> np.ndarray:
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    rendered = np.zeros(source_shape, dtype=np.uint8)
    for contour in contours:
        points = contour[:, 0, :].astype(np.float64)
        projected = _project_reference_points(points, context, grid)
        transformed = np.rint(_transform(projected, matrix)).astype(np.int32)
        if len(transformed) >= 3:
            cv2.fillPoly(rendered, [transformed], 1, cv2.LINE_8)
    return rendered > 0


def _line_distance_summary(line: np.ndarray, distance: np.ndarray) -> dict[str, Any]:
    values = distance[line]
    if not len(values):
        return {
            "pixel_count": 0,
            "median_px": 1e6,
            "p90_px": 1e6,
            "within_5px_fraction": 0.0,
            "within_8px_fraction": 0.0,
        }
    return {
        "pixel_count": int(len(values)),
        "median_px": float(np.median(values)),
        "p90_px": float(np.quantile(values, 0.90)),
        "within_5px_fraction": float(np.mean(values <= 5.0)),
        "within_8px_fraction": float(np.mean(values <= 8.0)),
    }


def _symmetric_line_report(
    rendered_reference: np.ndarray,
    source_semantic: np.ndarray,
    *,
    source_scope: np.ndarray,
    tolerance: float,
) -> dict[str, Any]:
    source_distance = distance_transform_edt(~source_semantic).astype(np.float32)
    reference_distance = distance_transform_edt(~rendered_reference).astype(np.float32)
    forward = _line_distance_summary(rendered_reference, source_distance)
    scoped_source = source_semantic & source_scope
    reverse = _line_distance_summary(scoped_source, reference_distance)
    precision = float(np.mean(source_distance[rendered_reference] <= tolerance)) if np.any(rendered_reference) else 0.0
    recall = float(np.mean(reference_distance[scoped_source] <= tolerance)) if np.any(scoped_source) else 0.0
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "reference_to_source": forward,
        "source_to_reference": reverse,
        "overlap_tolerance_px": float(tolerance),
        "precision": precision,
        "recall": recall,
        "f1": float(f1),
        "source_scope_pixel_count": int(np.count_nonzero(scoped_source)),
    }


def _reference_points_to_source(
    points: np.ndarray,
    *,
    matrix: np.ndarray,
    reference_center: np.ndarray,
    reference_state_height: float,
    projection: ProjectionContext | None,
    grid: Mapping[str, Any],
) -> np.ndarray:
    if projection is None:
        normalized = _normalize_points(
            np.asarray(points, dtype=np.float64),
            reference_center,
            reference_state_height,
        )
    else:
        normalized = _project_reference_points(points, projection, grid)
    return _transform(normalized, matrix)


def _geographically_balanced_state_tail(
    reference_state: np.ndarray,
    source_state: np.ndarray,
    *,
    matrix: np.ndarray,
    reference_center: np.ndarray,
    reference_state_height: float,
    projection: ProjectionContext | None,
    grid: Mapping[str, Any],
    config: AlignmentLoopConfig,
    visible_reference_only: bool = False,
) -> dict[str, Any]:
    """Validate state tails per geographic cell at the unchanged pixel limit.

    A global p90 can be dominated by one unsupported island, generalized bay,
    or stylized boundary segment.  This report applies the same p90 threshold
    independently to geographically balanced cells, then requires broad row,
    column, and total coverage so coherent regional drift still fails.
    """

    ys, xs = np.nonzero(reference_state)
    reference_points = np.column_stack((xs, ys)).astype(np.float64)
    transformed = np.rint(
        _reference_points_to_source(
            reference_points,
            matrix=matrix,
            reference_center=reference_center,
            reference_state_height=reference_state_height,
            projection=projection,
            grid=grid,
        )
    ).astype(np.int32)
    source_distance = distance_transform_edt(~source_state).astype(np.float32)
    source_height, source_width = source_state.shape
    inside = (
        (transformed[:, 0] >= 0)
        & (transformed[:, 0] < source_width)
        & (transformed[:, 1] >= 0)
        & (transformed[:, 1] < source_height)
    )
    distances = np.full(len(reference_points), math.hypot(source_width, source_height))
    distances[inside] = source_distance[
        transformed[inside, 1], transformed[inside, 0]
    ]
    rows, columns = config.geographic_cells
    reference_height, reference_width = reference_state.shape
    cells = []
    for row in range(rows):
        for column in range(columns):
            selected = (
                (ys >= row * reference_height / rows)
                & (ys < (row + 1) * reference_height / rows)
                & (xs >= column * reference_width / columns)
                & (xs < (column + 1) * reference_width / columns)
            )
            reference_pixel_count = int(np.count_nonzero(selected))
            if visible_reference_only:
                selected &= inside
            pixel_count = int(np.count_nonzero(selected))
            if pixel_count < config.semantic_state_tail_minimum_pixels_per_cell:
                continue
            values = distances[selected]
            p90 = float(np.quantile(values, 0.90))
            cells.append(
                {
                    "row": row,
                    "column": column,
                    "pixel_count": pixel_count,
                    "reference_pixel_count": reference_pixel_count,
                    "visible_fraction": float(
                        pixel_count / max(reference_pixel_count, 1)
                    ),
                    "median_px": float(np.median(values)),
                    "p90_px": p90,
                    "passed": p90 <= config.semantic_primary_p90_limit_px,
                }
            )
    if not cells:
        return {
            "passed": False,
            "reason": "no_geographically_supported_state_cells",
            "cell_count": 0,
            "cells": [],
        }
    cell_pass_fraction = float(np.mean([item["passed"] for item in cells]))
    row_pass_fractions = {
        str(row): float(
            np.mean([item["passed"] for item in cells if item["row"] == row])
        )
        for row in sorted({item["row"] for item in cells})
    }
    column_pass_fractions = {
        str(column): float(
            np.mean(
                [item["passed"] for item in cells if item["column"] == column]
            )
        )
        for column in sorted({item["column"] for item in cells})
    }
    median_cell_p90 = float(np.median([item["p90_px"] for item in cells]))
    passed = bool(
        len(cells) >= config.minimum_primary_holdout_cells
        and median_cell_p90 <= config.semantic_primary_p90_limit_px
        and cell_pass_fraction
        >= config.semantic_state_tail_minimum_cell_pass_fraction
        and min(row_pass_fractions.values())
        >= config.semantic_state_tail_minimum_axis_pass_fraction
        and min(column_pass_fractions.values())
        >= config.semantic_state_tail_minimum_axis_pass_fraction
    )
    return {
        "passed": passed,
        "visibility_mode": (
            "rendered_reference_inside_partial_source_canvas"
            if visible_reference_only
            else "complete_reference"
        ),
        "reference_visible_fraction": float(np.mean(inside)),
        "pixel_limit_px": config.semantic_primary_p90_limit_px,
        "cell_count": len(cells),
        "minimum_cell_count": config.minimum_primary_holdout_cells,
        "median_cell_p90_px": median_cell_p90,
        "cell_pass_fraction": cell_pass_fraction,
        "minimum_cell_pass_fraction": config.semantic_state_tail_minimum_cell_pass_fraction,
        "row_pass_fractions": row_pass_fractions,
        "column_pass_fractions": column_pass_fractions,
        "minimum_axis_pass_fraction": config.semantic_state_tail_minimum_axis_pass_fraction,
        "cells": cells,
    }


def _county_channel_observability(
    source_counties: np.ndarray,
    rendered_counties: np.ndarray,
    county_scope: np.ndarray,
    *,
    config: AlignmentLoopConfig,
) -> dict[str, Any]:
    """Classify whether a connected county-like internal network is observable.

    Pixel density and geographic occupancy alone mistake distributed labels,
    city dots, logos, and perimeter remnants for county boundaries.  County
    evidence must additionally contain either a lightly bridged connected
    component spanning both axes or multiple long, distributed network strands.
    Each must carry a material fraction of observed source-line pixels.  The
    one-pixel bridge joins minor antialiasing/compression gaps without merging
    separate words or symbols.
    """

    scoped_source = source_counties & county_scope
    source_pixel_count = int(np.count_nonzero(scoped_source))
    reference_pixel_count = int(np.count_nonzero(rendered_counties))
    line_ratio = source_pixel_count / max(reference_pixel_count, 1)
    ys, xs = np.nonzero(county_scope)
    occupied_cells = 0
    cell_reports = []
    topology = {
        "connectivity_radius_px": config.county_observability_connectivity_radius_px,
        "component_count": 0,
        "qualifying_component_count": 0,
        "long_strand_component_count": 0,
        "minimum_axis_span_fraction": (
            config.county_observability_minimum_network_axis_span_fraction
        ),
        "minimum_source_pixel_fraction": (
            config.county_observability_minimum_network_source_fraction
        ),
        "minimum_long_strand_axis_fraction": (
            config.county_observability_minimum_long_strand_axis_fraction
        ),
        "minimum_long_strand_cross_axis_fraction": (
            config.county_observability_minimum_long_strand_cross_axis_fraction
        ),
        "minimum_long_strand_count": config.county_observability_minimum_long_strand_count,
        "largest_component": None,
    }
    if len(xs):
        minimum_x, maximum_x = int(xs.min()), int(xs.max()) + 1
        minimum_y, maximum_y = int(ys.min()), int(ys.max()) + 1
        scope_width = max(maximum_x - minimum_x, 1)
        scope_height = max(maximum_y - minimum_y, 1)
        rows, columns = config.geographic_cells
        for row in range(rows):
            for column in range(columns):
                y1 = round(minimum_y + row * (maximum_y - minimum_y) / rows)
                y2 = round(minimum_y + (row + 1) * (maximum_y - minimum_y) / rows)
                x1 = round(minimum_x + column * (maximum_x - minimum_x) / columns)
                x2 = round(minimum_x + (column + 1) * (maximum_x - minimum_x) / columns)
                scoped_pixels = int(np.count_nonzero(scoped_source[y1:y2, x1:x2]))
                scope_pixels = int(np.count_nonzero(county_scope[y1:y2, x1:x2]))
                minimum_pixels = max(3, round(scope_pixels * 0.0005))
                occupied = scoped_pixels >= minimum_pixels
                occupied_cells += int(occupied)
                if scope_pixels:
                    cell_reports.append(
                        {
                            "row": row,
                            "column": column,
                            "source_line_pixels": scoped_pixels,
                            "minimum_pixels": minimum_pixels,
                            "occupied": occupied,
                        }
                    )
        radius = config.county_observability_connectivity_radius_px
        if radius:
            kernel_size = 2 * radius + 1
            connected_source = cv2.dilate(
                scoped_source.astype(np.uint8),
                np.ones((kernel_size, kernel_size), np.uint8),
            )
        else:
            connected_source = scoped_source.astype(np.uint8)
        component_count, component_labels, component_stats, _ = (
            cv2.connectedComponentsWithStats(connected_source, 8)
        )
        component_reports = []
        for component in range(1, component_count):
            x, y, width, height, connected_pixels = map(
                int, component_stats[component]
            )
            source_pixels = int(
                np.count_nonzero(scoped_source & (component_labels == component))
            )
            width_fraction = width / scope_width
            height_fraction = height / scope_height
            source_fraction = source_pixels / max(source_pixel_count, 1)
            qualifies = bool(
                width_fraction
                >= config.county_observability_minimum_network_axis_span_fraction
                and height_fraction
                >= config.county_observability_minimum_network_axis_span_fraction
                and source_fraction
                >= config.county_observability_minimum_network_source_fraction
            )
            long_strand = bool(
                max(width_fraction, height_fraction)
                >= config.county_observability_minimum_long_strand_axis_fraction
                and min(width_fraction, height_fraction)
                >= config.county_observability_minimum_long_strand_cross_axis_fraction
                and source_fraction
                >= config.county_observability_minimum_network_source_fraction
            )
            component_reports.append(
                {
                    "connected_pixel_count": connected_pixels,
                    "source_line_pixel_count": source_pixels,
                    "source_line_pixel_fraction": float(source_fraction),
                    "width_fraction_of_scope": float(width_fraction),
                    "height_fraction_of_scope": float(height_fraction),
                    "qualifies_as_network": qualifies,
                    "qualifies_as_long_strand": long_strand,
                }
            )
        component_reports.sort(
            key=lambda item: (
                item["source_line_pixel_count"], item["connected_pixel_count"]
            ),
            reverse=True,
        )
        topology.update(
            {
                "component_count": len(component_reports),
                "qualifying_component_count": sum(
                    item["qualifies_as_network"] for item in component_reports
                ),
                "long_strand_component_count": sum(
                    item["qualifies_as_long_strand"] for item in component_reports
                ),
                "largest_component": component_reports[0]
                if component_reports
                else None,
            }
        )
    network_observed = bool(
        topology["qualifying_component_count"] > 0
        or topology["long_strand_component_count"]
        >= config.county_observability_minimum_long_strand_count
    )
    observable = bool(
        line_ratio >= config.county_observability_minimum_line_ratio
        and occupied_cells >= config.county_observability_minimum_occupied_cells
        and network_observed
    )
    return {
        "status": "observable" if observable else "absent",
        "required_for_acceptance": observable,
        "reason": (
            "county_like_internal_line_network_observed"
            if observable
            else "insufficient_county_like_internal_line_network"
        ),
        "source_line_pixel_count": source_pixel_count,
        "rendered_reference_line_pixel_count": reference_pixel_count,
        "source_to_reference_line_ratio": float(line_ratio),
        "minimum_line_ratio": config.county_observability_minimum_line_ratio,
        "occupied_cell_count": occupied_cells,
        "minimum_occupied_cell_count": config.county_observability_minimum_occupied_cells,
        "connected_internal_network_observed": network_observed,
        "topology": topology,
        "cells": cell_reports,
    }


def _not_applicable_gate(gate: Mapping[str, Any], reason: str) -> dict[str, Any]:
    return {
        "passed": True,
        "status": "not_applicable",
        "reason": reason,
        "observed_gate": dict(gate),
    }


def _semantic_full_line_validation(
    matrix: np.ndarray,
    reference: PinnedMapboxReference,
    semantic: SourceSemanticEvidence,
    reference_center: np.ndarray,
    reference_state_height: float,
    config: AlignmentLoopConfig,
    projection: ProjectionContext | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, np.ndarray]]:
    """Render complete reference lines and require bidirectional agreement."""

    source_shape = semantic.state_coast.shape
    if projection is None:
        reference_to_source = _reference_to_source_matrix(
            matrix, reference_center, reference_state_height
        )
        rendered_state = _warp_reference_mask(
            reference.state_coast, reference_to_source, source_shape
        )
        rendered_counties = _warp_reference_mask(
            reference.counties, reference_to_source, source_shape
        )
        rendered_land = _warp_reference_mask(
            reference.state_land, reference_to_source, source_shape
        )
    else:
        rendered_state = _render_projected_reference_line(
            reference.state_coast,
            projection,
            matrix,
            reference.grid,
            source_shape,
        )
        rendered_counties = _render_projected_reference_line(
            reference.counties,
            projection,
            matrix,
            reference.grid,
            source_shape,
        )
        rendered_land = _render_projected_reference_land(
            reference.state_land,
            projection,
            matrix,
            reference.grid,
            source_shape,
        )

    state_corridor = cv2.dilate(
        rendered_state.astype(np.uint8), np.ones((25, 25), np.uint8)
    ) > 0
    county_scope = cv2.erode(
        rendered_land.astype(np.uint8), np.ones((9, 9), np.uint8)
    ) > 0
    county_scope &= ~cv2.dilate(
        rendered_state.astype(np.uint8), np.ones((9, 9), np.uint8)
    ).astype(bool)
    state = _symmetric_line_report(
        rendered_state,
        semantic.state_coast,
        source_scope=state_corridor,
        tolerance=config.semantic_overlap_tolerance_px,
    )
    counties = _symmetric_line_report(
        rendered_counties,
        semantic.counties,
        source_scope=county_scope,
        tolerance=config.semantic_overlap_tolerance_px,
    )
    state_tail = _geographically_balanced_state_tail(
        reference.state_coast,
        semantic.state_coast,
        matrix=matrix,
        reference_center=reference_center,
        reference_state_height=reference_state_height,
        projection=projection,
        grid=reference.grid,
        config=config,
        visible_reference_only=(
            semantic.source_adapter_id
            == "farms-partial-california-topology-v2"
        ),
    )
    observed_county_capability = _county_channel_observability(
        semantic.counties,
        rendered_counties,
        county_scope,
        config=config,
    )
    if semantic.county_observability_override == "absent":
        county_capability = {
            "status": "absent",
            "reason": "source_family_semantics_exclude_county_network",
            "required_for_acceptance": False,
            "source_adapter_id": semantic.source_adapter_id,
            "observed_classifier": observed_county_capability,
        }
    else:
        county_capability = observed_county_capability
    state_forward = state["reference_to_source"]
    county_forward = counties["reference_to_source"]
    gates = {
        "semantic_full_state_median": {
            "passed": state_forward["median_px"] <= config.semantic_primary_median_limit_px,
            "value": state_forward["median_px"],
            "maximum": config.semantic_primary_median_limit_px,
        },
        "semantic_full_state_balanced_tail": {
            "passed": state_tail["passed"],
            "value": state_tail.get("median_cell_p90_px", 1e6),
            "maximum": config.semantic_primary_p90_limit_px,
            "cell_pass_fraction": state_tail.get("cell_pass_fraction", 0.0),
            "minimum_cell_pass_fraction": config.semantic_state_tail_minimum_cell_pass_fraction,
            "minimum_axis_pass_fraction": config.semantic_state_tail_minimum_axis_pass_fraction,
        },
        "semantic_full_state_support": {
            "passed": state_forward["within_8px_fraction"] >= config.semantic_primary_within_8px_minimum,
            "value": state_forward["within_8px_fraction"],
            "minimum": config.semantic_primary_within_8px_minimum,
        },
        "semantic_full_state_symmetric_overlap": {
            "passed": state["f1"] >= config.semantic_primary_f1_minimum,
            "value": state["f1"],
            "minimum": config.semantic_primary_f1_minimum,
        },
        "semantic_full_county_median": {
            "passed": county_forward["median_px"] <= config.semantic_county_median_limit_px,
            "value": county_forward["median_px"],
            "maximum": config.semantic_county_median_limit_px,
        },
        "semantic_full_county_support": {
            "passed": county_forward["within_8px_fraction"] >= config.semantic_county_within_8px_minimum,
            "value": county_forward["within_8px_fraction"],
            "minimum": config.semantic_county_within_8px_minimum,
        },
        "semantic_full_county_symmetric_overlap": {
            "passed": counties["f1"] >= config.semantic_county_f1_minimum,
            "value": counties["f1"],
            "minimum": config.semantic_county_f1_minimum,
        },
        "source_county_channel_observability": {
            "passed": True,
            "value": county_capability["status"],
            "required_for_acceptance": county_capability[
                "required_for_acceptance"
            ],
            "reason": county_capability["reason"],
        },
    }
    if not county_capability["required_for_acceptance"]:
        for name in (
            "semantic_full_county_median",
            "semantic_full_county_support",
            "semantic_full_county_symmetric_overlap",
        ):
            gates[name] = _not_applicable_gate(
                gates[name], county_capability["reason"]
            )
    return (
        {
            "state_coast": state,
            "state_geographically_balanced_tail": state_tail,
            "counties": counties,
            "source_capabilities": {"counties": county_capability},
        },
        gates,
        {
            "rendered_state": rendered_state,
            "rendered_counties": rendered_counties,
            "rendered_land": rendered_land,
        },
    )


def _graticule_seed_matrix(
    hypothesis: AlignmentSourceHypothesis,
    projection: ProjectionContext,
    grid: Mapping[str, Any],
    *,
    model: str,
) -> tuple[np.ndarray, dict[str, Any]] | None:
    """Fit a regular transform from source-only OCR/graticule controls."""

    if (
        projection.id != "california_lambert_conformal_conic"
        or len(hypothesis.graticule_lonlat) < 4
        or len(hypothesis.graticule_lonlat)
        != len(hypothesis.graticule_source_points)
    ):
        return None
    lonlat = np.asarray(hypothesis.graticule_lonlat, dtype=np.float64)
    source = np.asarray(hypothesis.graticule_source_points, dtype=np.float64)
    to_web_mercator = Transformer.from_crs(
        "EPSG:4326", "EPSG:3857", always_xy=True
    )
    web_x, web_y = to_web_mercator.transform(lonlat[:, 0], lonlat[:, 1])
    reference_pixels = _web_mercator_to_reference_pixels(
        np.asarray(web_x), np.asarray(web_y), grid
    )
    candidate = _project_reference_points(reference_pixels, projection, grid)
    if model == "regular_affine":
        design = np.column_stack((candidate, np.ones(len(candidate))))
        coefficients = np.linalg.lstsq(design, source, rcond=None)[0]
        matrix = np.vstack((coefficients.T, [0.0, 0.0, 1.0]))
    elif model == "similarity":
        candidate_center = np.mean(candidate, axis=0)
        source_center = np.mean(source, axis=0)
        candidate_centered = candidate - candidate_center
        source_centered = source - source_center
        left, singular, right = np.linalg.svd(
            candidate_centered.T @ source_centered
        )
        rotation = right.T @ left.T
        if np.linalg.det(rotation) <= 0:
            right[-1, :] *= -1.0
            rotation = right.T @ left.T
        scale = float(
            np.sum(singular)
            / max(float(np.sum(candidate_centered**2)), 1e-12)
        )
        linear = scale * rotation
        translation = source_center - linear @ candidate_center
        matrix = np.asarray(
            [
                [linear[0, 0], linear[0, 1], translation[0]],
                [linear[1, 0], linear[1, 1], translation[1]],
                [0.0, 0.0, 1.0],
            ]
        )
    else:
        return None
    residual = np.linalg.norm(_transform(candidate, matrix) - source, axis=1)
    return matrix, {
        "method": "source_only_ocr_labeled_native_graticule",
        "projection": projection.id,
        "control_point_count": int(len(source)),
        "median_residual_px": float(np.median(residual)),
        "maximum_residual_px": float(np.max(residual)),
        "mapbox_geometry_used_for_fit": False,
        "mapbox_strict_gates_required_after_fit": True,
    }


def _deterministic_mask_points(
    mask: np.ndarray,
    *,
    maximum: int,
) -> np.ndarray:
    """Return deterministic x/y samples without favoring scanline prefixes."""

    y, x = np.nonzero(mask)
    points = np.column_stack((x, y)).astype(np.float64)
    if len(points) <= maximum:
        return points
    indices = np.linspace(0, len(points) - 1, maximum).round().astype(int)
    return points[indices]


def _farms_partial_registration_seed(
    hypothesis: AlignmentSourceHypothesis,
    projection: ProjectionContext,
    reference: PinnedMapboxReference,
    *,
    model: str,
    source_shape: tuple[int, int],
    config: AlignmentLoopConfig,
) -> tuple[np.ndarray, dict[str, Any]] | None:
    """Fit observed farms boundaries to Mapbox without a full-state inside bias.

    The farms raster is a partial geographic crop.  A reference-to-source
    objective penalizes unseen reference boundaries and therefore finds a
    deceptively small California.  This seed reverses that correspondence:
    each *observed* source coast/Nevada component must map back onto the full
    pinned Mapbox state line, with source internal topology as a lower-weight
    discriminator.  It cannot accept a transform; the unchanged full rendered
    Mapbox gates run immediately afterward.
    """

    semantic = hypothesis.semantic
    if semantic.source_adapter_id != "farms-partial-california-topology-v2":
        return None
    if model not in TRANSFORM_MODELS:
        return None

    reference_state_y, reference_state_x = np.nonzero(reference.state_coast)
    reference_county_y, reference_county_x = np.nonzero(reference.counties)
    if not len(reference_state_x) or not len(reference_county_x):
        return None
    reference_state = np.column_stack(
        (reference_state_x, reference_state_y)
    ).astype(np.float64)
    reference_counties = np.column_stack(
        (reference_county_x, reference_county_y)
    ).astype(np.float64)
    state_normalized = _project_reference_points(
        reference_state, projection, reference.grid
    )
    county_normalized = _project_reference_points(
        reference_counties, projection, reference.grid
    )
    state_tree = cKDTree(state_normalized)
    county_tree = cKDTree(county_normalized)

    component_count, component_labels, component_stats, _ = (
        cv2.connectedComponentsWithStats(
            semantic.state_coast.astype(np.uint8), 8
        )
    )
    minimum_component = max(80, round(np.prod(source_shape) * 0.00015))
    state_components = []
    state_component_reports = []
    for label in range(1, component_count):
        x, y, width, height, area = map(int, component_stats[label])
        if area < minimum_component:
            continue
        points = _deterministic_mask_points(
            component_labels == label,
            maximum=1200,
        )
        state_components.append(points)
        state_component_reports.append(
            {"box": [x, y, width, height], "area_px": area, "samples": len(points)}
        )
    if len(state_components) < 2:
        return None
    county_source = _deterministic_mask_points(
        semantic.counties,
        maximum=1800,
    )
    if len(county_source) < 100:
        return None

    if model == "similarity":
        bounds = (
            (-0.35, 1.25),
            (-0.35, 1.25),
            (config.minimum_scale_fraction, config.maximum_scale_fraction),
            (-18.0, 18.0),
        )
    else:
        bounds = (
            (-0.35, 1.25),
            (-0.35, 1.25),
            (config.minimum_scale_fraction, config.maximum_scale_fraction),
            (0.78, 1.22),
            (-18.0, 18.0),
            (-0.16, 0.16),
        )

    source_height = float(source_shape[0])

    def reverse_distances(
        points: np.ndarray,
        matrix: np.ndarray,
        tree: cKDTree,
    ) -> np.ndarray:
        inverse = np.linalg.inv(matrix)
        normalized = _transform(points, inverse)
        distance, _indices = tree.query(normalized, k=1, workers=1)
        # Convert normalized California-height units back to the approximate
        # source-pixel scale so state and county terms remain interpretable.
        singular = np.linalg.svd(matrix[:2, :2], compute_uv=False)
        return np.asarray(distance) * float(np.mean(singular))

    def objective(parameters: Sequence[float]) -> float:
        matrix = _matrix_from_parameters(parameters, model, source_shape)
        component_losses = []
        for points in state_components:
            distances = reverse_distances(points, matrix, state_tree)
            component_losses.append(
                float(
                    np.median(distances)
                    + 0.35 * np.quantile(distances, 0.90)
                    + 8.0 * (1.0 - np.mean(distances <= 8.0))
                )
            )
        county_distances = reverse_distances(
            county_source, matrix, county_tree
        )
        county_loss = float(
            np.median(county_distances)
            + 0.20 * np.quantile(county_distances, 0.90)
        )
        # The worst observed state component cannot be sacrificed to a very
        # good coast-only or Nevada-only local optimum.
        return float(
            0.65 * np.mean(component_losses)
            + 0.25 * np.max(component_losses)
            + 0.10 * county_loss
        )

    result = differential_evolution(
        objective,
        bounds,
        seed=config.seed + 97,
        popsize=max(6, min(config.global_population, 14)),
        maxiter=max(36, min(config.global_iterations, 100)),
        polish=True,
        workers=1,
        updating="immediate",
    )
    matrix = _matrix_from_parameters(result.x, model, source_shape)
    component_metrics = []
    for report, points in zip(state_component_reports, state_components):
        distances = reverse_distances(points, matrix, state_tree)
        component_metrics.append(
            {
                **report,
                "median_px": float(np.median(distances)),
                "p90_px": float(np.quantile(distances, 0.90)),
                "within_8px_fraction": float(np.mean(distances <= 8.0)),
            }
        )
    county_distances = reverse_distances(county_source, matrix, county_tree)
    return matrix, {
        "method": "farms_observed_source_components_to_pinned_mapbox_seed",
        "projection": projection.id,
        "model": model,
        "optimizer_objective": float(result.fun),
        "optimizer_iterations": int(result.nit),
        "optimizer_success": bool(result.success),
        "state_components": component_metrics,
        "county_samples": int(len(county_source)),
        "county_median_px": float(np.median(county_distances)),
        "county_p90_px": float(np.quantile(county_distances, 0.90)),
        "source_visibility_model": (
            "observed_source_coast_and_nevada_components_only_for_seed"
        ),
        "mapbox_geometry_used_for_fit": True,
        "mapbox_geometry_used_for_source_hypothesis_construction": False,
        "strict_full_rendered_mapbox_gates_required_after_fit": True,
    }


def _evaluate_alignment_candidate(
    hypothesis: AlignmentSourceHypothesis,
    *,
    model: str,
    projection: ProjectionContext,
    source_shape: tuple[int, int],
    pixel_evidence: Mapping[str, np.ndarray],
    reference: PinnedMapboxReference,
    reference_center: np.ndarray,
    reference_state_height: float,
    config: AlignmentLoopConfig,
) -> dict[str, Any]:
    """Fit one variant and apply the unchanged strict Mapbox gate set."""

    semantic = hypothesis.semantic
    distance = {
        "primary": distance_transform_edt(~semantic.foreground_boundary).astype(
            np.float32
        ),
        "county": distance_transform_edt(~semantic.counties).astype(np.float32),
    }
    if (
        semantic.hydrography is not None
        and "hydrography_train" in pixel_evidence
    ):
        distance["hydrography"] = distance_transform_edt(
            ~semantic.hydrography
        ).astype(np.float32)
    hypothesis_evidence = dict(pixel_evidence)
    hypothesis_evidence["foreground_source_mask"] = semantic.foreground_interior
    evidence = _projection_evidence(
        hypothesis_evidence, projection, reference.grid
    )
    fit_config = (
        replace(config, county_training_weight=0.0)
        if semantic.county_observability_override == "absent"
        else config
    )
    farms_partial = (
        semantic.source_adapter_id == "farms-partial-california-topology-v2"
    )
    if farms_partial:
        # The visible source crop covers most, not all, of the state.  A full
        # California transform therefore legitimately spans more than one
        # source-canvas height.  Shape regularity (anisotropy/shear/rotation)
        # and every semantic threshold remain unchanged.
        fit_config = replace(
            fit_config,
            maximum_scale_fraction=max(
                fit_config.maximum_scale_fraction, 2.40
            ),
        )
    graticule_seed = _graticule_seed_matrix(
        hypothesis,
        projection,
        reference.grid,
        model=model,
    )
    farms_seed = _farms_partial_registration_seed(
        hypothesis,
        projection,
        reference,
        model=model,
        source_shape=source_shape,
        config=fit_config,
    )
    if graticule_seed is not None:
        matrix, graticule_report = graticule_seed
        objective, summaries, gates, regularity = _score_candidate(
            matrix, model, distance, evidence, fit_config
        )
        summaries["native_graticule_seed"] = graticule_report
    elif farms_seed is not None:
        matrix, farms_seed_report = farms_seed
        objective, summaries, gates, regularity = _score_candidate(
            matrix, model, distance, evidence, fit_config
        )
        summaries["partial_registration_seed"] = farms_seed_report
    else:
        matrix, objective, summaries, gates, regularity = _fit_candidate(
            model, source_shape, distance, evidence, fit_config
        )
    if farms_partial:
        transformed_land = _transform(evidence["state_land_train"], matrix)
        source_height, source_width = source_shape
        land_x = np.rint(transformed_land[:, 0]).astype(int)
        land_y = np.rint(transformed_land[:, 1]).astype(int)
        visible_land = (
            (land_x >= 0)
            & (land_x < source_width)
            & (land_y >= 0)
            & (land_y < source_height)
        )
        visible_count = int(np.count_nonzero(visible_land))
        contained = np.zeros(len(transformed_land), dtype=bool)
        contained[visible_land] = semantic.foreground_interior[
            land_y[visible_land], land_x[visible_land]
        ]
        visible_fraction = float(np.mean(visible_land))
        containment = (
            float(np.mean(contained[visible_land])) if visible_count else 0.0
        )
        summaries["silhouette"] = {
            "land_containment_fraction": containment,
            "visible_fraction": visible_fraction,
            "sample_count": int(len(transformed_land)),
            "visible_sample_count": visible_count,
            "visibility_mode": "partial_source_visible_land_only",
        }
        gates["silhouette_land_containment"] = {
            "passed": containment >= 0.90,
            "value": containment,
            "minimum": 0.90,
        }
        gates["partial_source_visible_land_support"] = {
            "passed": visible_fraction >= 0.20,
            "value": visible_fraction,
            "minimum": 0.20,
        }
    semantic_scores, semantic_gates, rendered = _semantic_full_line_validation(
        matrix,
        reference,
        semantic,
        reference_center,
        reference_state_height,
        config,
        projection,
    )
    county_capability = semantic_scores["source_capabilities"]["counties"]
    if not county_capability["required_for_acceptance"]:
        for name in (
            "county_holdout_median",
            "county_holdout_support",
            "geographically_balanced_counties",
        ):
            gates[name] = _not_applicable_gate(
                gates[name], county_capability["reason"]
            )
    round_trip = _projection_round_trip(projection, matrix, reference.grid)
    gates["projection_round_trip"] = {
        "passed": round_trip["finite"] and round_trip["maximum_error_px"] < 1e-5,
        "value": round_trip["maximum_error_px"],
        "maximum": 1e-5,
    }
    gates = {**gates, **semantic_gates}
    return {
        "matrix": matrix,
        "objective": objective,
        "summaries": summaries,
        "gates": gates,
        "regularity": regularity,
        "semantic_scores": semantic_scores,
        "rendered": rendered,
        "round_trip": round_trip,
        "evidence": evidence,
        "all_passed": all(
            value if isinstance(value, bool) else bool(value["passed"])
            for value in gates.values()
        ),
    }


def _coarse_source_hypothesis_shortlist(
    hypotheses: Sequence[AlignmentSourceHypothesis],
    *,
    projections: Sequence[ProjectionContext],
    source_shape: tuple[int, int],
    pixel_evidence: Mapping[str, np.ndarray],
    reference: PinnedMapboxReference,
    reference_center: np.ndarray,
    reference_state_height: float,
    config: AlignmentLoopConfig,
) -> tuple[tuple[AlignmentSourceHypothesis, ...], list[dict[str, Any]]]:
    """Use bounded low-budget Mapbox fits to choose full-budget variants.

    The source-only score is serialized but deliberately excluded from ranking.
    Every generated variant is compared with the pinned Mapbox reference using
    the same validation gates; the coarse pass can shortlist, never accept.
    """

    if not hypotheses:
        raise ValueError("At least one source alignment hypothesis is required")
    web_mercator = next(
        (projection for projection in projections if projection.id == "web_mercator"),
        None,
    )
    if web_mercator is None:
        raise ValueError("The source-hypothesis shortlist requires web_mercator")
    coarse_config = replace(
        config,
        global_iterations=config.source_hypothesis_coarse_iterations,
        global_population=config.source_hypothesis_coarse_population,
    )
    reports = []
    state_gate_names = {
        "semantic_full_state_median",
        "semantic_full_state_balanced_tail",
        "semantic_full_state_support",
        "semantic_full_state_symmetric_overlap",
        "silhouette_land_containment",
    }
    for hypothesis in hypotheses:
        evaluated = _evaluate_alignment_candidate(
            hypothesis,
            model="similarity",
            projection=web_mercator,
            source_shape=source_shape,
            pixel_evidence=pixel_evidence,
            reference=reference,
            reference_center=reference_center,
            reference_state_height=reference_state_height,
            config=coarse_config,
        )
        failed = sorted(
            name
            for name, value in evaluated["gates"].items()
            if not (value if isinstance(value, bool) else bool(value["passed"]))
        )
        state_failed = [name for name in failed if name in state_gate_names]
        reports.append(
            {
                "hypothesis_id": hypothesis.id,
                "variant_kind": hypothesis.variant_kind,
                "source_only_score": hypothesis.source_only_score,
                "source_only_score_used_for_ranking": False,
                "projection": web_mercator.id,
                "model": "similarity",
                "optimizer_iterations": coarse_config.global_iterations,
                "optimizer_population": coarse_config.global_population,
                "objective": float(evaluated["objective"]),
                "failed_gate_count": len(failed),
                "failed_state_gate_count": len(state_failed),
                "failed_gates": failed,
                "strict_gates_all_passed": bool(evaluated["all_passed"]),
                "normalized_reference_to_working_source_matrix": evaluated[
                    "matrix"
                ].tolist(),
                "semantic_full_line": evaluated["semantic_scores"],
            }
        )
    ranked = sorted(
        reports,
        key=lambda item: (
            item["failed_gate_count"],
            item["failed_state_gate_count"],
            item["objective"],
            item["hypothesis_id"],
        ),
    )
    by_id = {hypothesis.id: hypothesis for hypothesis in hypotheses}
    family_ids = {
        hypothesis.id
        for hypothesis in hypotheses
        if hypothesis.variant_kind.startswith("source_family_")
    }
    strict_family_first = next(
        (
            str(report["hypothesis_id"])
            for report in ranked
            if report["strict_gates_all_passed"]
            and str(report["hypothesis_id"]) in family_ids
        ),
        None,
    )
    # A coarse result cannot accept a transform, but a source-family candidate
    # that already passes the complete strict gate set is the highest-value
    # full-budget candidate. Keep it in the bounded full-budget ensemble; final
    # acceptance is based on a global ranking of every full-budget candidate,
    # never on evaluation order.
    selected_ids = [strict_family_first or "baseline-semantic"]
    for report in ranked:
        hypothesis_id = str(report["hypothesis_id"])
        if hypothesis_id not in selected_ids:
            selected_ids.append(hypothesis_id)
        if len(selected_ids) >= config.source_hypothesis_shortlist_size:
            break
    # A family adapter encodes a source-side semantic distinction that the
    # generic hypotheses cannot express (for example, a labeled Lambert state
    # line or a filled gradient-map panel).  The cheap shortlist fit uses only
    # Web Mercator + similarity, so it can under-rank exactly those hypotheses
    # when the source's native projection requires the later projection-aware
    # search.  Reserve one bounded slot for the adapter; the adapter still has
    # to pass every unchanged full-budget Mapbox gate.
    ranked_family_ids = [
        str(report["hypothesis_id"])
        for report in ranked
        if str(report["hypothesis_id"]) in family_ids
    ]
    reserved_family_id = ranked_family_ids[0] if ranked_family_ids else None
    if reserved_family_id is not None and not any(
        hypothesis_id in family_ids for hypothesis_id in selected_ids
    ):
        if len(selected_ids) >= config.source_hypothesis_shortlist_size:
            replace_at = next(
                (
                    index
                    for index in range(len(selected_ids) - 1, 0, -1)
                    if selected_ids[index] not in family_ids
                ),
                None,
            )
            if replace_at is not None:
                selected_ids[replace_at] = reserved_family_id
        else:
            selected_ids.append(reserved_family_id)
    selected = tuple(by_id[hypothesis_id] for hypothesis_id in selected_ids)
    return selected, reports


def _artifact_slug(value: str) -> str:
    normalized = "".join(
        character if character.isalnum() else "-" for character in value.lower()
    )
    return "-".join(part for part in normalized.split("-") if part)[:64]


def _allocate_source_hypothesis_root(
    output_root: Path, starting_automatic_ordinal: int
) -> Path:
    """Allocate an immutable diagnostic directory, including after a crash."""

    base_name = f"source-hypotheses-start-{starting_automatic_ordinal:02d}"
    candidate = output_root / base_name
    suffix = 1
    while candidate.exists():
        suffix += 1
        candidate = output_root / f"{base_name}-attempt-{suffix:02d}"
    return candidate


def _artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def _write_candidate_artifacts(
    output_root: Path,
    iteration: int,
    model: str,
    rgb: np.ndarray,
    edges: np.ndarray,
    semantic: SourceSemanticEvidence,
    rendered: Mapping[str, np.ndarray],
    matrix: np.ndarray,
    evidence: Mapping[str, np.ndarray],
    payload: Mapping[str, Any],
) -> tuple[Path, ...]:
    attempt = output_root / f"alignment-{iteration:02d}-{model}"
    attempt.mkdir(parents=True, exist_ok=False)
    overlay = rgb.copy()
    colors = {
        "primary_train": (101, 255, 155),
        "primary_holdout": (255, 220, 70),
        "county_train": (58, 209, 255),
        "county_holdout": (255, 95, 210),
    }
    for name, color in colors.items():
        points = np.rint(_transform(evidence[name], matrix)).astype(np.int32)
        height, width = edges.shape
        inside = (
            (points[:, 0] >= 0)
            & (points[:, 0] < width)
            & (points[:, 1] >= 0)
            & (points[:, 1] < height)
        )
        for x, y in points[inside]:
            cv2.circle(overlay, (int(x), int(y)), 1, color, -1, cv2.LINE_8)
    overlay_path = attempt / "source-mapbox-evidence.png"
    edges_path = attempt / "source-edges.png"
    state_semantic_path = attempt / "source-state-coast-evidence.png"
    county_semantic_path = attempt / "source-county-evidence.png"
    semantic_overlay_path = attempt / "semantic-full-line-validation.png"
    candidate_path = attempt / "candidate.json"
    Image.fromarray(overlay).save(overlay_path)
    Image.fromarray(edges.astype(np.uint8) * 255).save(edges_path)
    Image.fromarray(semantic.state_coast.astype(np.uint8) * 255).save(
        state_semantic_path
    )
    Image.fromarray(semantic.counties.astype(np.uint8) * 255).save(
        county_semantic_path
    )
    semantic_overlay = cv2.cvtColor(
        cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB
    )
    semantic_overlay[semantic.state_coast] = (85, 220, 120)
    semantic_overlay[semantic.counties] = (40, 180, 230)
    semantic_overlay[rendered["rendered_state"]] = (255, 220, 50)
    semantic_overlay[rendered["rendered_counties"]] = (255, 80, 210)
    Image.fromarray(semantic_overlay).save(semantic_overlay_path)
    candidate_path.write_text(json.dumps(payload, indent=2) + "\n")
    return (
        overlay_path,
        edges_path,
        state_semantic_path,
        county_semantic_path,
        semantic_overlay_path,
        candidate_path,
    )


def run_automatic_alignment_loop(
    source_path: Path,
    reference_manifest_path: Path,
    output_root: Path,
    experiment_log: NoHumanExperimentLog,
    *,
    config: AlignmentLoopConfig | None = None,
    source_family: str | None = None,
) -> AutomaticAlignmentResult:
    """Fit the bounded ensemble and accept its globally best passing transform."""

    config = config or AlignmentLoopConfig()
    source_path = source_path.resolve()
    if experiment_log.data["source"]["sha256"] != _sha256(source_path):
        raise ValueError("Experiment log source is not the original source supplied to alignment")
    alignment_record = experiment_log.data["alignment"]
    if alignment_record["accepted_automatic_iteration_count"] is not None:
        raise ValueError(
            "The automatic alignment is already accepted; invalidate it with a "
            "deterministic failed gate before resuming"
        )
    prior_iterations = alignment_record["iterations"]
    if any(
        not item.get("counts_toward_automatic_iteration_count", False)
        for item in prior_iterations
    ):
        raise ValueError(
            "A no-human alignment cannot resume a log containing manual or "
            "otherwise ineligible attempts"
        )
    reference = load_pinned_mapbox_reference(reference_manifest_path)
    logged_reference = experiment_log.data.get("mapbox_reference", {})
    logged_pin = str(logged_reference.get("manifest_sha256", ""))
    if not logged_pin:
        raise ValueError("Experiment log must pin the Mapbox reference manifest hash")
    if logged_pin != reference.pin["manifest_sha256"]:
        raise ValueError("Experiment log and alignment use different Mapbox references")
    rgb_original = _load_categorical_raster(source_path)
    rgb, working_scale = _resize_working(rgb_original, config.working_max_dimension)
    edges = _source_edges(rgb)
    semantic = _source_semantic_evidence(rgb)
    primary_train, primary_holdout, primary_train_cells, primary_holdout_cells = (
        _balanced_split(
            reference.state_coast,
            config.geographic_cells,
            config.primary_samples_per_cell,
        )
    )
    county_train, county_holdout, county_train_cells, county_holdout_cells = (
        _balanced_split(
            reference.counties,
            config.geographic_cells,
            config.county_samples_per_cell,
        )
    )
    land_y, land_x = np.nonzero(reference.state_land)
    land_sample_indices = np.linspace(
        0, len(land_x) - 1, min(len(land_x), 10000)
    ).round().astype(int)
    state_land_train = np.column_stack(
        (land_x[land_sample_indices], land_y[land_sample_indices])
    ).astype(np.float64)
    center, state_height, _, _ = _normalizer(reference.state_coast)
    pixel_evidence = {
        "primary_train": primary_train,
        "primary_holdout": primary_holdout,
        "county_train": county_train,
        "county_holdout": county_holdout,
        "state_land_train": state_land_train,
        "foreground_source_mask": semantic.foreground_interior,
        "primary_train_cells": primary_train_cells,
        "primary_holdout_cells": primary_holdout_cells,
        "county_train_cells": county_train_cells,
        "county_holdout_cells": county_holdout_cells,
    }
    if reference.waterways is not None:
        (
            hydrography_train,
            hydrography_holdout,
            hydrography_train_cells,
            hydrography_holdout_cells,
        ) = _balanced_split(
            reference.waterways,
            config.geographic_cells,
            config.county_samples_per_cell,
        )
        pixel_evidence.update(
            {
                "hydrography_train": hydrography_train,
                "hydrography_holdout": hydrography_holdout,
                "hydrography_train_cells": hydrography_train_cells,
                "hydrography_holdout_cells": hydrography_holdout_cells,
            }
        )
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    candidates: list[AlignmentCandidate] = []
    stop_reason = "projection_and_regular_model_sequence_exhausted"
    prior_automatic_count = sum(
        bool(item["counts_toward_automatic_iteration_count"])
        for item in prior_iterations
    )
    projection_contexts = _projection_contexts(reference)
    hypothesis_root = _allocate_source_hypothesis_root(
        output_root, prior_automatic_count + 1
    )
    source_hypotheses = _generate_alignment_source_hypotheses(
        source_path,
        hypothesis_root,
        rgb,
        semantic,
        working_scale=working_scale,
        config=config,
        source_family=source_family,
    )
    if any(hypothesis.graticule_lonlat for hypothesis in source_hypotheses):
        projection_contexts = tuple(
            sorted(
                projection_contexts,
                key=lambda item: (
                    item.id != "california_lambert_conformal_conic",
                    item.id,
                ),
            )
        )
    shortlisted_hypotheses, coarse_reports = _coarse_source_hypothesis_shortlist(
        source_hypotheses,
        projections=projection_contexts,
        source_shape=rgb.shape[:2],
        pixel_evidence=pixel_evidence,
        reference=reference,
        reference_center=center,
        reference_state_height=state_height,
        config=config,
    )
    hypothesis_plan_path = hypothesis_root / "mapbox-coarse-shortlist.json"
    hypothesis_plan = {
        "schema_version": 1,
        "kind": "automatic_alignment_source_hypothesis_shortlist",
        "source_sha256": _sha256(source_path),
        "source_family": source_family,
        "working_shape": [int(value) for value in rgb.shape[:2]],
        "working_scale_from_original": float(working_scale),
        "immutable_attempt_directory": str(hypothesis_root),
        "selection_authority": {
            "source_only_scores_used_for_ranking": False,
            "source_only_scores_used_for_acceptance": False,
            "pinned_mapbox_reference_used_for_coarse_ranking": True,
            "strict_full_budget_gates_required_for_acceptance": True,
        },
        "projection_priority": [
            projection.id for projection in projection_contexts
        ],
        "bounds": {
            "generated_hypothesis_limit": config.source_hypothesis_generation_limit,
            "shortlist_size": config.source_hypothesis_shortlist_size,
            "coarse_optimizer_iterations": config.source_hypothesis_coarse_iterations,
            "coarse_optimizer_population": config.source_hypothesis_coarse_population,
            "source_family_reserved_slots": (
                1
                if any(
                    hypothesis.variant_kind.startswith("source_family_")
                    for hypothesis in source_hypotheses
                )
                else 0
            ),
        },
        "variants": [
            _source_hypothesis_payload(hypothesis)
            for hypothesis in source_hypotheses
        ],
        "coarse_mapbox_evaluations": coarse_reports,
        "selected_hypothesis_ids": [
            hypothesis.id for hypothesis in shortlisted_hypotheses
        ],
    }
    hypothesis_plan_path.write_text(json.dumps(hypothesis_plan, indent=2) + "\n")
    candidate_specs = [
        (hypothesis, projection, model)
        for projection in projection_contexts
        for model in TRANSFORM_MODELS
        for hypothesis in shortlisted_hypotheses
    ]
    hypothesis_ordinals = {
        hypothesis.id: index
        for index, hypothesis in enumerate(shortlisted_hypotheses, start=1)
    }
    # Preflight the complete strict ensemble before recording any acceptance.
    # This deliberately pays for a second deterministic evaluation pass so the
    # large rendered validation masks do not have to remain resident for every
    # candidate at once.  It also makes "first passing candidate wins"
    # impossible: acceptance is decided only after all alternatives are known.
    preflight: list[dict[str, Any]] = []
    for ensemble_ordinal, (hypothesis, projection, model) in enumerate(
        candidate_specs, start=1
    ):
        evaluated = _evaluate_alignment_candidate(
            hypothesis,
            model=model,
            projection=projection,
            source_shape=rgb.shape[:2],
            pixel_evidence=pixel_evidence,
            reference=reference,
            reference_center=center,
            reference_state_height=state_height,
            config=config,
        )
        semantic_state = evaluated["semantic_scores"]["state_coast"]
        land_containment = float(
            evaluated["summaries"].get("silhouette", {}).get(
                "land_containment_fraction", 0.0
            )
        )
        state_median_px = float(
            semantic_state["reference_to_source"]["median_px"]
        )
        state_p90_px = float(
            semantic_state["reference_to_source"]["p90_px"]
        )
        state_f1 = float(semantic_state["f1"])
        preflight.append(
            {
                "ensemble_ordinal": ensemble_ordinal,
                "hypothesis": hypothesis,
                "projection": projection,
                "model": model,
                "objective": float(evaluated["objective"]),
                "matrix": np.asarray(evaluated["matrix"], dtype=np.float64),
                "all_passed": bool(evaluated["all_passed"]),
                "failed_gates": sorted(
                    name
                    for name, value in evaluated["gates"].items()
                    if not (
                        value
                        if isinstance(value, bool)
                        else bool(value["passed"])
                    )
                ),
                "state_median_px": state_median_px,
                "state_p90_px": state_p90_px,
                "state_f1": state_f1,
                "land_containment_fraction": land_containment,
                "semantic_alignment_score": (
                    state_p90_px
                    + state_median_px
                    + 10.0 * (1.0 - state_f1)
                    + 10.0 * (1.0 - land_containment)
                ),
            }
        )

    def ranking_key(item: Mapping[str, Any]) -> tuple[Any, ...]:
        return (
            float(item["semantic_alignment_score"]),
            float(item["state_p90_px"]),
            float(item["state_median_px"]),
            -float(item["state_f1"]),
            -float(item["land_containment_fraction"]),
            float(item["objective"]),
            str(item["projection"].id),
            str(item["model"]),
            str(item["hypothesis"].id),
            int(item["ensemble_ordinal"]),
        )

    passing_ranked = sorted(
        (item for item in preflight if item["all_passed"]),
        key=ranking_key,
    )
    selected = passing_ranked[0] if passing_ranked else None
    selected_ordinal = (
        int(selected["ensemble_ordinal"]) if selected is not None else None
    )
    selection_rank = {
        int(item["ensemble_ordinal"]): rank
        for rank, item in enumerate(sorted(preflight, key=ranking_key), start=1)
    }
    ranking_path = output_root / "candidate-ranking.json"
    ranking_payload = {
        "schema_version": 1,
        "kind": "automatic_alignment_global_candidate_ranking",
        "selection_policy": "global_best_passing_candidate",
        "selection_key": [
            "semantic_alignment_score_ascending",
            "state_p90_px_ascending",
            "state_median_px_ascending",
            "state_f1_descending",
            "land_containment_fraction_descending",
            "objective_ascending",
            "projection_id_ascending",
            "model_ascending",
            "hypothesis_id_ascending",
            "ensemble_ordinal_ascending",
        ],
        "source_sha256": _sha256(source_path),
        "mapbox_reference": dict(reference.pin),
        "candidate_count": len(preflight),
        "passing_candidate_count": len(passing_ranked),
        "selected_ensemble_ordinal": selected_ordinal,
        "candidates": [
            {
                "ensemble_ordinal": int(item["ensemble_ordinal"]),
                "selection_rank": selection_rank[int(item["ensemble_ordinal"])],
                "hypothesis_id": item["hypothesis"].id,
                "projection": item["projection"].id,
                "model": item["model"],
                "objective": item["objective"],
                "strict_gates_all_passed": item["all_passed"],
                "failed_gates": item["failed_gates"],
                "state_median_px": item["state_median_px"],
                "state_p90_px": item["state_p90_px"],
                "state_f1": item["state_f1"],
                "land_containment_fraction": item[
                    "land_containment_fraction"
                ],
                "semantic_alignment_score": item["semantic_alignment_score"],
                "globally_selected": (
                    int(item["ensemble_ordinal"]) == selected_ordinal
                ),
            }
            for item in sorted(preflight, key=ranking_key)
        ],
    }
    ranking_path.write_text(json.dumps(ranking_payload, indent=2) + "\n")

    # The accepted experiment-log iteration must be last because an accepted
    # phase is immutable.  Non-winning candidates retain their ensemble order;
    # the global winner, if any, is recorded after every comparison.
    record_order = [
        item for item in preflight if int(item["ensemble_ordinal"]) != selected_ordinal
    ]
    if selected is not None:
        record_order.append(selected)

    best_recorded_objective: float | None = None
    for record_offset, preflight_item in enumerate(record_order, start=1):
        hypothesis = preflight_item["hypothesis"]
        projection = preflight_item["projection"]
        model = str(preflight_item["model"])
        ensemble_ordinal = int(preflight_item["ensemble_ordinal"])
        iteration = prior_automatic_count + record_offset
        evaluated = _evaluate_alignment_candidate(
            hypothesis,
            model=model,
            projection=projection,
            source_shape=rgb.shape[:2],
            pixel_evidence=pixel_evidence,
            reference=reference,
            reference_center=center,
            reference_state_height=state_height,
            config=config,
        )
        matrix = evaluated["matrix"]
        objective = evaluated["objective"]
        summaries = evaluated["summaries"]
        gates = evaluated["gates"]
        regularity = evaluated["regularity"]
        semantic_scores = evaluated["semantic_scores"]
        rendered = evaluated["rendered"]
        round_trip = evaluated["round_trip"]
        evidence = evaluated["evidence"]
        all_passed = evaluated["all_passed"]
        if (
            not np.allclose(
                matrix,
                preflight_item["matrix"],
                rtol=1e-10,
                atol=1e-8,
            )
            or not np.isclose(
                objective,
                preflight_item["objective"],
                rtol=1e-10,
                atol=1e-8,
            )
            or bool(all_passed) != bool(preflight_item["all_passed"])
        ):
            raise RuntimeError(
                "Alignment candidate evaluation was not deterministic between "
                "global ranking and artifact materialization"
            )
        relative_improvement = None
        if best_recorded_objective is not None:
            relative_improvement = (best_recorded_objective - objective) / max(
                abs(best_recorded_objective), 1e-12
            )
        exhausted = record_offset == len(record_order)
        globally_selected = ensemble_ordinal == selected_ordinal
        decision = "accept" if globally_selected else (
            "blocked" if exhausted and selected is None else "retry"
        )
        scores = {
            "objective": objective,
            "working_scale_from_original": working_scale,
            "primary_train": summaries["primary_train"],
            "primary_holdout": summaries["primary_holdout"],
            "county_train": summaries["county_train"],
            "county_holdout": summaries["county_holdout"],
            "hydrography_train": summaries.get("hydrography_train"),
            "hydrography_holdout": summaries.get("hydrography_holdout"),
            "native_graticule_seed": summaries.get("native_graticule_seed"),
            "partial_registration_seed": summaries.get(
                "partial_registration_seed"
            ),
            "silhouette": summaries.get("silhouette"),
            "relative_improvement": relative_improvement,
            "normalized_reference_to_working_source_matrix": matrix.tolist(),
            "regularity": regularity,
            "semantic_full_line": semantic_scores,
            "projection_round_trip": round_trip,
            "projection": {
                "id": projection.id,
                "crs_wkt_sha256": projection.crs_wkt_sha256,
            },
            "source_alignment_hypothesis": _source_hypothesis_payload(
                hypothesis
            ),
            "global_candidate_selection": {
                "policy": "global_best_passing_candidate",
                "ensemble_ordinal": ensemble_ordinal,
                "selection_rank": selection_rank[ensemble_ordinal],
                "candidate_count": len(preflight),
                "passing_candidate_count": len(passing_ranked),
                "globally_selected": globally_selected,
                "ranking_artifact": _hypothesis_artifact_record(ranking_path),
            },
        }
        transform_contract = _projection_transform_contract(
            matrix,
            projection=projection,
            working_scale=working_scale,
            source_original_shape=rgb_original.shape[:2],
            source_working_shape=rgb.shape[:2],
            target_grid=reference.grid,
        )
        payload = {
            "schema_version": 1,
            "iteration": iteration,
            "model": model,
            "projection": projection.id,
            "scores": scores,
            "gates": gates,
            "decision": decision,
            "source_sha256": _sha256(source_path),
            "mapbox_reference": dict(reference.pin),
            "source_alignment_hypothesis": _source_hypothesis_payload(
                hypothesis
            ),
            "source_alignment_hypothesis_plan": _hypothesis_artifact_record(
                hypothesis_plan_path
            ),
            "global_candidate_selection": scores["global_candidate_selection"],
            "transform": transform_contract,
        }
        hypothesis_ordinal = hypothesis_ordinals[hypothesis.id]
        artifact_label = (
            f"hypothesis-{hypothesis_ordinal:02d}-"
            f"{_artifact_slug(hypothesis.id)}-{projection.id}-{model}"
        )
        paths = _write_candidate_artifacts(
            output_root,
            iteration,
            artifact_label,
            rgb,
            edges,
            hypothesis.semantic,
            rendered,
            matrix,
            evidence,
            payload,
        )
        all_artifact_paths = tuple(
            dict.fromkeys(
                (
                    *paths,
                    hypothesis_plan_path,
                    ranking_path,
                    *hypothesis.artifact_paths,
                )
            )
        )
        experiment_log.record_alignment_iteration(
            scores=scores,
            gates=gates,
            decision=decision,
            provenance=automatic_provenance(
                "mapscan.automatic_alignment_loop",
                [
                    "original_source_image_pixels",
                    "source_only_canvas_legend_support_hypotheses",
                    *(
                        ["source_family_semantic_adapter"]
                        if any(
                            item.variant_kind == "source_family_semantic_adapter"
                            for item in source_hypotheses
                        )
                        else []
                    ),
                    "pinned_mapbox_state_coast",
                    "pinned_mapbox_counties",
                ],
            ),
            method=(
                f"geographically balanced {projection.id} {model} Mapbox "
                f"registration using source hypothesis {hypothesis.id}"
            ),
            artifacts=[_artifact(path) for path in all_artifact_paths],
            note=(
                "Stopped after exhausting the pinned source-hypothesis, projection, "
                "and regular-transform ensemble."
                if exhausted and selected is None
                else (
                    "Passed strict gates but was not the globally best passing "
                    "candidate in the complete ensemble."
                    if all_passed and not globally_selected
                    else None
                )
            ),
        )
        candidate = AlignmentCandidate(
            iteration=iteration,
            model=f"{hypothesis.id}:{projection.id}:{model}",
            normalized_reference_to_working_source_matrix=matrix,
            reference_pixel_to_source_original_matrix=(
                np.asarray(
                    transform_contract["reference_pixel_to_source_original_matrix"],
                    dtype=np.float64,
                )
                if "reference_pixel_to_source_original_matrix" in transform_contract
                else None
            ),
            source_original_to_reference_pixel_matrix=(
                np.asarray(
                    transform_contract["source_original_to_reference_pixel_matrix"],
                    dtype=np.float64,
                )
                if "source_original_to_reference_pixel_matrix" in transform_contract
                else None
            ),
            target_grid=transform_contract["target_grid"],
            objective=objective,
            scores=scores,
            gates=gates,
            regularity=regularity,
            status=decision,
            artifact_paths=all_artifact_paths,
        )
        candidates.append(candidate)
        if globally_selected:
            accepted_path = output_root / "accepted-alignment.json"
            accepted_path.write_text(json.dumps(payload, indent=2) + "\n")
            return AutomaticAlignmentResult(
                status="pass",
                stop_reason="automatic_alignment_gates_passed",
                candidates=tuple(candidates),
                accepted=candidate,
            )
        best_recorded_objective = (
            objective
            if best_recorded_objective is None
            else min(best_recorded_objective, objective)
        )
    return AutomaticAlignmentResult(
        status="blocked",
        stop_reason=stop_reason,
        candidates=tuple(candidates),
        accepted=None,
    )


def invalidate_alignment_acceptance_for_failed_gate(
    experiment_log: NoHumanExperimentLog,
    *,
    gate_name: str,
    gate: Mapping[str, Any],
    score_name: str,
    score: Any,
    reason: str,
) -> dict[str, Any]:
    """Safely reopen a premature automatic acceptance after a new hard gate.

    The accepted attempt remains in place and keeps its automatic ordinal.  Its
    decision is changed to ``retry`` and the deterministic failed gate and score
    are appended.  This is only legal before extraction begins and never accepts
    a manual attempt.
    """

    if not gate_name.strip() or not score_name.strip() or not reason.strip():
        raise ValueError("Invalidation requires a gate, score, and reason")
    if gate.get("passed") is not False:
        raise ValueError("Invalidation requires a deterministically failed gate")
    alignment = experiment_log.data["alignment"]
    accepted_ordinal = alignment.get("accepted_automatic_iteration_count")
    if accepted_ordinal is None:
        raise ValueError("There is no accepted automatic alignment to invalidate")
    if experiment_log.data["extraction"]["iterations"]:
        raise ValueError("Cannot invalidate alignment after extraction has started")
    matches = [
        item
        for item in alignment["iterations"]
        if item.get("automatic_iteration") == accepted_ordinal
    ]
    if len(matches) != 1:
        raise ValueError("Accepted automatic alignment ordinal is inconsistent")
    item = matches[0]
    if (
        not item.get("counts_toward_automatic_iteration_count")
        or item.get("decision") != "accept"
    ):
        raise ValueError("Only an eligible accepted automatic attempt can be invalidated")
    item["gates"][gate_name] = dict(gate)
    item["scores"][score_name] = score
    item["all_gates_passed"] = False
    item["decision"] = "retry"
    item["note"] = (
        f"{item['note']} {reason}".strip() if item.get("note") else reason
    )
    alignment["accepted_automatic_iteration_count"] = None
    experiment_log.data["final"] = {"status": "in_progress", "blocker": None}
    experiment_log.data["updated_at"] = datetime.now(timezone.utc).replace(
        microsecond=0
    ).isoformat()
    return item
