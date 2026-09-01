"""Throwaway-only regularized non-rigid alignment for the elevation source.

This experiment is deliberately isolated from :mod:`automatic_alignment_loop`.
It starts with the source-only, OCR-labelled Lambert Conformal Conic graticule
seed, learns a small smooth residual from geographically distributed perimeter
segments, and uses named geographic regions only as holdouts.  Nothing in this
module writes an experiment log or produces a transform eligible for promotion.

The implementation is intentionally conservative:

* the source raster must be the pristine ``elevation.gif``-style image;
* source evidence is derived from pale connected water and printed state lines;
* Bay, island, southern, and south-eastern cells are excluded from fitting;
* every candidate must preserve orientation and a positive Jacobian; and
* bidirectional source/Mapbox line metrics are hard gates, not objectives that
  can be hidden by a one-way nearest-neighbour score.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image
from pyproj import Transformer
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

from .automatic_alignment_loop import (
    AlignmentSourceHypothesis,
    PinnedMapboxReference,
    ProjectionContext,
    _family_semantic_hypothesis,
    _graticule_seed_matrix,
    _large_pale_blue_water,
    _long_line_components,
    _project_reference_points,
    _projection_transform_contract,
    _projection_contexts,
    _reference_pixels_to_web_mercator,
    _source_semantic_evidence,
    _transform,
    _web_mercator_to_reference_pixels,
    load_pinned_mapbox_reference,
)
from .experiment_log import NoHumanExperimentLog, automatic_provenance


EXPERIMENT_KIND = "throwaway_elevation_regularized_nonrigid_v1"


@dataclass(frozen=True)
class ElevationNonrigidConfig:
    """Fixed search and strict-gate contract for the isolated prototype."""

    training_grid: tuple[int, int] = (14, 16)
    maximum_training_match_px: float = 20.0
    observable_corridor_px: float = 30.0
    # Reverse scoring is intentionally narrower than forward observability.  It
    # is centered on the independently fitted source-only LCC seed, so adjacent
    # Nevada/Mexico border strokes cannot masquerade as California evidence.
    reverse_seed_corridor_px: float = 12.0
    maximum_controls: int = 180
    kernel_radii_reference_px: tuple[float, ...] = (260.0, 420.0, 680.0)
    ridge_values: tuple[float, ...] = (0.05, 0.20, 1.0, 5.0, 20.0)
    maximum_residual_displacement_px: float = 24.0
    jacobian_grid: tuple[int, int] = (34, 40)
    minimum_jacobian_ratio: float = 0.15
    global_observable_fraction_minimum: float = 0.92
    global_supported_median_limit_px: float = 2.5
    global_supported_p90_limit_px: float = 9.0
    global_supported_within_8px_minimum: float = 0.88
    reverse_corridor_fraction_minimum: float = 0.78
    reverse_median_limit_px: float = 2.5
    reverse_p90_limit_px: float = 9.0
    border_median_limit_px: float = 2.0
    border_p90_limit_px: float = 5.0
    coast_observable_fraction_minimum: float = 0.70
    coast_supported_median_limit_px: float = 3.0
    coast_supported_p90_limit_px: float = 12.0
    maximum_pacific_contamination_fraction: float = 0.035


@dataclass(frozen=True)
class SourceElevationEvidence:
    water: np.ndarray
    coast: np.ndarray
    inland_hydrography: np.ndarray
    printed_state_border: np.ndarray
    combined: np.ndarray
    diagnostics: Mapping[str, Any]


@dataclass(frozen=True)
class CompactResidualWarp:
    """Wendland-C2 residual displacement in Mapbox reference pixels."""

    centers_reference_px: np.ndarray
    coefficients_source_px: np.ndarray
    radius_reference_px: float
    ridge: float

    def displacement(self, reference_points: np.ndarray, *, chunk: int = 8192) -> np.ndarray:
        points = np.asarray(reference_points, dtype=np.float64)
        result = np.zeros((len(points), 2), dtype=np.float64)
        if not len(self.centers_reference_px):
            return result
        for start in range(0, len(points), chunk):
            stop = min(start + chunk, len(points))
            delta = points[start:stop, None, :] - self.centers_reference_px[None, :, :]
            radius = np.linalg.norm(delta, axis=2) / self.radius_reference_px
            kernel = _wendland_c2(radius)
            result[start:stop] = kernel @ self.coefficients_source_px
        return result


@dataclass(frozen=True)
class CandidateSummary:
    id: str
    radius_reference_px: float | None
    ridge: float | None
    selection_score: float
    eligible: bool
    maximum_displacement_px: float
    jacobian: Mapping[str, Any]
    graticule: Mapping[str, Any]
    forward: Mapping[str, Any]
    reverse: Mapping[str, Any]
    channels: Mapping[str, Any]
    holdouts: Mapping[str, Any]


@dataclass(frozen=True)
class ElevationNonrigidResult:
    status: str
    stop_reason: str
    report_path: Path
    artifact_paths: tuple[Path, ...]
    selected_candidate: CandidateSummary
    all_candidates: tuple[CandidateSummary, ...]


def _official_artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": _sha256(path.resolve())}


def _official_elevation_candidate_payload(
    *,
    automatic_iteration: int,
    report: Mapping[str, Any],
    transform: Mapping[str, Any],
    original_source_path: Path,
    working_source_path: Path,
    reference_pin: Mapping[str, Any],
    transform_serialization_sha256: str,
    validation_report_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build the exact official payload before creating any artifacts.

    The throwaway report uses ``control_point_count`` internally as a numerical
    fitting diagnostic.  That wording is intentionally absent from the
    official contract: the intersections were detected from source-only OCR'd
    graticules, and the extraction authority scanner treats any
    ``control_point`` key as ambiguous manual provenance.
    """

    selected = next(
        candidate
        for candidate in report["candidates"]
        if candidate["id"] == report["selected_candidate_id"]
    )
    projection_seed = dict(report["projection_seed"])
    detected_intersection_count = projection_seed.pop("control_point_count", None)
    if detected_intersection_count is None:
        detected_intersection_count = projection_seed.pop(
            "graticule_intersection_count", None
        )
    if detected_intersection_count is None:
        raise ValueError("Projection seed lacks its source-only graticule count")
    projection_seed[
        "source_only_detected_graticule_intersection_count"
    ] = detected_intersection_count
    scores = {
        "objective": selected["selection_score"],
        "strict_nonrigid_validation": selected,
        "training": report["training"],
        "source_evidence": report["source_evidence"],
        "projection_seed": projection_seed,
        "transform_serialization_sha256": transform_serialization_sha256,
        "validation_report_sha256": validation_report_sha256,
    }
    gates = dict(report["strict_gates"]["checks"])
    payload = {
        "schema_version": 1,
        "iteration": automatic_iteration,
        "model": "regularized_nonrigid_wendland_c2",
        "projection": "california_lambert_conformal_conic",
        "scores": scores,
        "gates": gates,
        "decision": "accept",
        "source_sha256": _sha256(working_source_path.resolve()),
        "authoritative_source_sha256": _sha256(original_source_path.resolve()),
        "mapbox_reference": dict(reference_pin),
        "source_alignment_hypothesis": {
            "id": "source-only-labeled-lcc-water-state-lines-v1",
            "source_only_detected_graticule_intersection_count": (
                detected_intersection_count
            ),
            "mapbox_role": "training_perimeter_and_strict_validation",
            "bay_delta_hydrography_role": "holdout_validation_only",
        },
        "transform": dict(transform),
    }
    return payload, scores, gates


def append_official_elevation_alignment_retry(
    *,
    validation_root: Path,
    original_source_path: Path,
    working_source_path: Path,
    reference_manifest_path: Path,
    official_alignment_root: Path,
    experiment_log: NoHumanExperimentLog,
    experiment_markdown_path: Path,
    experiment_json_path: Path,
) -> Path:
    """Append one immutable official retry after a strict throwaway passes."""

    validation_root = validation_root.resolve()
    original_source_path = original_source_path.resolve()
    working_source_path = working_source_path.resolve()
    official_alignment_root = official_alignment_root.resolve()
    report_path = validation_root / "report.json"
    transform_path = validation_root / "selected-transform.json"
    report = json.loads(report_path.read_text())
    transform = json.loads(transform_path.read_text())
    if (
        report.get("status") != "strict_pass"
        or report.get("strict_gates", {}).get("passed") is not True
        or report.get("selected_candidate_id") == "native-lcc-seed"
    ):
        raise ValueError("Official elevation retry requires a strict nonrigid pass")
    if report.get("selected_transform") != transform:
        raise ValueError("Serialized transform disagrees with strict validation report")
    if report.get("inputs", {}).get("source_sha256") != _sha256(original_source_path):
        raise ValueError("Strict validation used a different authoritative source")
    reference = load_pinned_mapbox_reference(reference_manifest_path.resolve())
    if transform.get("target_grid") != dict(reference.grid):
        raise ValueError("Strict transform does not target the pinned Mapbox grid")
    source_rgb = np.asarray(Image.open(working_source_path).convert("RGB"))
    if transform.get("source_original_shape") != list(source_rgb.shape[:2]):
        raise ValueError("Strict transform source shape disagrees with working raster")
    alignment = experiment_log.data["alignment"]
    prior_count = sum(
        bool(item.get("counts_toward_automatic_iteration_count"))
        for item in alignment["iterations"]
    )
    if (
        experiment_log.data.get("map_id") != "elevation"
        or experiment_log.data["final"].get("status") not in {"blocked", "in_progress"}
        or alignment.get("accepted_automatic_iteration_count") is not None
        or prior_count < 10
        or len(alignment["iterations"]) != prior_count
    ):
        raise ValueError("Official elevation log is not at an append-only automatic checkpoint")
    if experiment_log.data["source"].get("sha256") != _sha256(original_source_path):
        raise ValueError("Official elevation log source hash disagrees")
    if any(
        not item.get("counts_toward_automatic_iteration_count", False)
        for item in alignment["iterations"]
    ):
        raise ValueError("Official elevation history contains an ineligible attempt")

    automatic_iteration = prior_count + 1
    candidate_id = (
        f"alignment-{automatic_iteration:02d}-"
        "california_lcc-wendland-r680-ridge0.05"
    )
    candidate_dir = official_alignment_root / candidate_id
    accepted_path = official_alignment_root / "accepted-alignment.json"
    if candidate_dir.exists() or accepted_path.exists():
        raise FileExistsError("Refusing to overwrite an official elevation alignment artifact")
    artifact_names = [
        "source-evidence.png",
        "seed-overlay.png",
        "selected-overlay.png",
        "selected-land-mask.png",
        "holdout-bay_delta.png",
        "holdout-sf_bay_inlet.png",
        "holdout-san_pablo_suisun_delta.png",
        "holdout-islands.png",
        "holdout-south_coast.png",
        "holdout-southeast_colorado.png",
        "report.json",
        "selected-transform.json",
    ]
    source_artifacts = [validation_root / name for name in artifact_names]
    if not all(path.is_file() for path in source_artifacts):
        raise FileNotFoundError("Strict validation artifact set is incomplete")

    payload, scores, gates = _official_elevation_candidate_payload(
        automatic_iteration=automatic_iteration,
        report=report,
        transform=transform,
        original_source_path=original_source_path,
        working_source_path=working_source_path,
        reference_pin=reference.pin,
        transform_serialization_sha256=_sha256(transform_path),
        validation_report_sha256=_sha256(report_path),
    )
    # Use the extraction consumer's exact authority scanner before creating an
    # official directory; a semantically automatic but ambiguously named field
    # must fail here rather than after log acceptance.
    from .automatic_categorical_extraction import _alignment_contains_forbidden_input

    forbidden = _alignment_contains_forbidden_input(payload)
    if forbidden:
        raise ValueError(
            f"Official elevation payload contains forbidden evidence at {forbidden}"
        )

    candidate_dir.mkdir(parents=True)
    copied: list[Path] = []
    for source_artifact in source_artifacts:
        destination = candidate_dir / source_artifact.name
        shutil.copy2(source_artifact, destination)
        copied.append(destination)
    candidate_path = candidate_dir / "candidate.json"
    candidate_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    copied.append(candidate_path)

    if experiment_log.data["final"].get("status") == "blocked":
        experiment_log.resume_automatic_blocked(
            reason=(
                "source-only native LCC seed plus regularized nonrigid residual, "
                "terrain-adjacent shoreline evidence, and independent Bay/Delta gates"
            ),
            producer="mapscan.elevation_nonrigid_alignment",
        )
    iteration = experiment_log.record_alignment_iteration(
        scores=scores,
        gates=gates,
        decision="accept",
        provenance=automatic_provenance(
            "mapscan.elevation_nonrigid_alignment",
            [
                "authoritative_original_source_pixels",
                "source_clean_working_raster",
                "source_only_labeled_native_lcc_graticule",
                "source_only_terrain_adjacent_water_shoreline",
                "source_only_printed_state_boundary",
                "source_only_bay_delta_water_semantics_holdout",
                "pinned_mapbox_state_coast",
                "pinned_mapbox_state_land_and_water",
                "deterministic_regularization_search",
            ],
        ),
        method=(
            "source-only OCR-labelled native Lambert seed with geographically "
            "distributed Wendland-C2 residual fit and strict independent Mapbox, "
            "Bay/Delta, invertibility, Jacobian, and contamination gates"
        ),
        artifacts=[_official_artifact(path) for path in copied],
    )
    if iteration.get("automatic_iteration") != automatic_iteration:
        raise RuntimeError("Official elevation retry did not append at the next ordinal")
    experiment_log.write(experiment_markdown_path, experiment_json_path)
    accepted_path.write_bytes(candidate_path.read_bytes())
    return accepted_path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _wendland_c2(radius: np.ndarray) -> np.ndarray:
    clipped = np.clip(1.0 - np.asarray(radius, dtype=np.float64), 0.0, 1.0)
    return clipped**4 * (4.0 * np.asarray(radius, dtype=np.float64) + 1.0)


def fit_compact_residual_warp(
    centers_reference_px: np.ndarray,
    residuals_source_px: np.ndarray,
    *,
    radius_reference_px: float,
    ridge: float,
) -> CompactResidualWarp:
    """Fit a deterministic, regularized compact radial-basis residual."""

    centers = np.asarray(centers_reference_px, dtype=np.float64)
    residuals = np.asarray(residuals_source_px, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 2 or residuals.shape != centers.shape:
        raise ValueError("Residual-warp controls must be matching N by 2 arrays")
    if radius_reference_px <= 0 or ridge <= 0:
        raise ValueError("Residual-warp radius and ridge must be positive")
    if not len(centers):
        return CompactResidualWarp(centers, residuals, radius_reference_px, ridge)
    pairwise = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    kernel = _wendland_c2(pairwise / radius_reference_px)
    coefficients = np.linalg.solve(
        kernel + np.eye(len(centers), dtype=np.float64) * ridge,
        residuals,
    )
    return CompactResidualWarp(centers, coefficients, radius_reference_px, ridge)


def serialize_elevation_nonrigid_transform(
    *,
    seed_matrix: np.ndarray,
    projection: ProjectionContext,
    warp: CompactResidualWarp,
    source_shape: tuple[int, int],
    target_grid: Mapping[str, Any],
) -> dict[str, Any]:
    """Serialize the exact projection plus residual mapping for consumers."""

    contract = _projection_transform_contract(
        np.asarray(seed_matrix, dtype=np.float64),
        projection=projection,
        working_scale=1.0,
        source_original_shape=source_shape,
        source_working_shape=source_shape,
        target_grid=target_grid,
    )
    residual = {
        "kind": "wendland_c2_reference_pixel_to_source_original_displacement",
        "coordinate_domain": "pinned_mapbox_target_grid_pixels",
        "displacement_range": "source_original_pixels",
        "centers_reference_px": warp.centers_reference_px.tolist(),
        "coefficients_source_px": warp.coefficients_source_px.tolist(),
        "radius_reference_px": float(warp.radius_reference_px),
        "ridge": float(warp.ridge),
        "kernel": "max(1-r,0)^4*(4r+1)",
    }
    encoded = json.dumps(residual, sort_keys=True, separators=(",", ":")).encode("utf-8")
    residual["sha256"] = hashlib.sha256(encoded).hexdigest()
    contract["kind"] = "projection_aware_residual_warp_mapbox_registration"
    contract["residual_warp"] = residual
    contract["inverse_solver"] = {
        "kind": "base_projection_fixed_point",
        "maximum_iterations": 20,
        "reference_tolerance_px": 1e-4,
        "source_roundtrip_tolerance_px": 0.02,
        "failure_policy": "reject_nonconverged_in_domain_points",
    }
    contract["reference_to_source_steps"] = [
        *contract["reference_to_source_steps"],
        "add serialized Wendland-C2 source-pixel displacement evaluated in reference pixels",
    ]
    contract["source_to_reference_steps"] = [
        "initialize with inverse projection-aware affine seed",
        "fixed-point iterate: subtract serialized reference-domain residual from source pixel",
        *contract["source_to_reference_steps"],
        "verify source-pixel round trip and reject nonconvergence",
    ]
    return contract


def detect_pristine_elevation_evidence(
    rgb: np.ndarray,
    hypothesis: AlignmentSourceHypothesis,
) -> SourceElevationEvidence:
    """Recover only source-side land/water coast and printed state strokes."""

    image = np.asarray(rgb, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Elevation source must be RGB")
    water = _large_pale_blue_water(image)
    if np.count_nonzero(water) < image.shape[0] * image.shape[1] * 0.10:
        raise ValueError("Pristine elevation source has no broad pale Pacific component")
    raw_water_edge = cv2.morphologyEx(
        water.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)
    ).astype(bool)
    # A raw connected-water gradient also outlines black graticules and place
    # labels that interrupt the printed blue.  Keep an edge only when terrain-
    # colored source pixels occur on its opposite side.  This retains mainland
    # and island shores without treating ocean typography as coastline.
    red = image[:, :, 0].astype(np.int16)
    green = image[:, :, 1].astype(np.int16)
    blue = image[:, :, 2].astype(np.int16)
    terrain = (
        (red >= 55)
        & (green >= 45)
        & ((red - blue >= 5) | (green - blue >= 5))
        & ((red + green + blue) < 735)
    )
    terrain_near = cv2.dilate(terrain.astype(np.uint8), np.ones((17, 17), np.uint8)).astype(bool)
    coast = raw_water_edge & terrain_near
    margin = max(8, round(min(coast.shape) * 0.008))
    coast[:margin] = False
    coast[-margin:] = False
    coast[:, :margin] = False
    coast[:, -margin:] = False

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # The printed Delta uses a darker blue river/canal network rather than the
    # Pacific's pale fill.  This narrow source-only signature excludes the
    # green/teal depression class (lower hue/value) and the pale-ocean fill
    # (higher value), while retaining the map's permanent water linework.
    inland_hydrography = (
        (hsv[:, :, 0] >= 90)
        & (hsv[:, :, 0] <= 105)
        & (hsv[:, :, 1] >= 120)
        & (hsv[:, :, 2] >= 80)
        & (hsv[:, :, 2] <= 190)
    )
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
    printed = _long_line_components(purple | dark_red)
    # The river/canal channel is never perimeter-fitting evidence.  It is kept
    # separate and is consulted only by the Bay/Delta water holdouts below.
    combined = coast | printed
    if np.count_nonzero(coast) < 1000 or np.count_nonzero(printed) < 500:
        raise ValueError("Elevation source coast/state evidence is incomplete")
    return SourceElevationEvidence(
        water=water,
        coast=coast,
        inland_hydrography=inland_hydrography,
        printed_state_border=printed,
        combined=combined,
        diagnostics={
            "method": "source_only_pale_connected_water_plus_printed_state_strokes",
            "source_adapter_id": hypothesis.semantic.source_adapter_id,
            "water_pixels": int(np.count_nonzero(water)),
            "raw_water_edge_pixels": int(np.count_nonzero(raw_water_edge)),
            "terrain_pixels": int(np.count_nonzero(terrain)),
            "coast_pixels": int(np.count_nonzero(coast)),
            "inland_hydrography_pixels": int(np.count_nonzero(inland_hydrography)),
            "printed_state_border_pixels": int(np.count_nonzero(printed)),
            "combined_pixels": int(np.count_nonzero(combined)),
            "bay_delta_water_semantic_pixels": int(
                np.count_nonzero(combined | inland_hydrography)
            ),
            "mapbox_used_for_detection": False,
        },
    )


def _reference_lonlat(
    points: np.ndarray, grid: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    web_x, web_y = _reference_pixels_to_web_mercator(points, grid)
    transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(web_x, web_y)
    return np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64)


def _island_reference_mask(reference: PinnedMapboxReference) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        reference.state_land.astype(np.uint8), 8
    )
    if count <= 2:
        return np.zeros(reference.state_land.shape, dtype=bool)
    main = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    islands = np.zeros(reference.state_land.shape, dtype=np.uint8)
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if label != main and area >= 20:
            islands[labels == label] = 1
    boundary = cv2.morphologyEx(
        islands, cv2.MORPH_GRADIENT, np.ones((7, 7), np.uint8)
    ).astype(bool)
    return cv2.dilate(boundary.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)


def named_geographic_holdouts(
    reference_points: np.ndarray,
    reference: PinnedMapboxReference,
) -> dict[str, np.ndarray]:
    """Return fixed Mapbox-coordinate holdouts; fitting must exclude their union."""

    points = np.asarray(reference_points, dtype=np.float64)
    lon, lat = _reference_lonlat(points, reference.grid)
    rounded_x = np.clip(np.rint(points[:, 0]).astype(int), 0, reference.state_coast.shape[1] - 1)
    rounded_y = np.clip(np.rint(points[:, 1]).astype(int), 0, reference.state_coast.shape[0] - 1)
    island_pixels = _island_reference_mask(reference)[rounded_y, rounded_x]
    return {
        "bay_delta":
            (lon >= -123.30) & (lon <= -121.30) & (lat >= 37.00) & (lat <= 38.80),
        # These deliberately overlap the broader Bay cell.  Their independent
        # gates prevent Point Reyes and the outer coast from hiding an inlet or
        # Delta failure.
        "sf_bay_inlet":
            (lon >= -123.00) & (lon <= -122.05) & (lat >= 37.25) & (lat <= 38.15),
        "san_pablo_suisun_delta":
            (lon >= -122.65) & (lon <= -121.25) & (lat >= 37.75) & (lat <= 38.65),
        "islands": island_pixels,
        "south_coast": (lat < 34.80) & (lon <= -116.50),
        "southeast_colorado": (lat < 35.80) & (lon > -116.50),
    }


def _map_points(
    reference_points: np.ndarray,
    *,
    projection: ProjectionContext,
    grid: Mapping[str, Any],
    seed_matrix: np.ndarray,
    warp: CompactResidualWarp | None,
) -> np.ndarray:
    base = _transform(_project_reference_points(reference_points, projection, grid), seed_matrix)
    if warp is None:
        return base
    return base + warp.displacement(reference_points)


def _sample_reference_boundary(mask: np.ndarray, maximum: int = 24000) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("Reference boundary is empty")
    if len(xs) > maximum:
        indices = np.linspace(0, len(xs) - 1, maximum).round().astype(int)
        xs, ys = xs[indices], ys[indices]
    return np.column_stack((xs, ys)).astype(np.float64)


def _aggregate_training_controls(
    reference_points: np.ndarray,
    base_source_points: np.ndarray,
    source_tree: cKDTree,
    holdout_union: np.ndarray,
    *,
    reference_shape: tuple[int, int],
    config: ElevationNonrigidConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    distances, nearest = source_tree.query(base_source_points)
    eligible = (~holdout_union) & (distances <= config.maximum_training_match_px)
    residuals = source_tree.data[nearest] - base_source_points
    norms = np.linalg.norm(residuals, axis=1)
    eligible &= norms <= config.maximum_residual_displacement_px
    rows, columns = config.training_grid
    height, width = reference_shape
    cell_x = np.clip((reference_points[:, 0] / max(width, 1) * columns).astype(int), 0, columns - 1)
    cell_y = np.clip((reference_points[:, 1] / max(height, 1) * rows).astype(int), 0, rows - 1)
    cell_ids = cell_y * columns + cell_x
    centers: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    support: list[int] = []
    for cell_id in range(rows * columns):
        indices = np.flatnonzero(eligible & (cell_ids == cell_id))
        if len(indices) < 8:
            continue
        local_residual = residuals[indices]
        median = np.median(local_residual, axis=0)
        deviation = np.linalg.norm(local_residual - median, axis=1)
        robust = indices[deviation <= max(float(np.percentile(deviation, 65)), 0.75)]
        if len(robust) < 4:
            continue
        centers.append(np.median(reference_points[robust], axis=0))
        targets.append(np.median(residuals[robust], axis=0))
        support.append(int(len(robust)))
    if len(centers) < 20:
        raise ValueError("Too few distributed non-holdout perimeter controls")
    controls = np.asarray(centers, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    if len(controls) > config.maximum_controls:
        order = np.argsort(np.asarray(support))[::-1][: config.maximum_controls]
        controls, target = controls[order], target[order]
        support = [support[index] for index in order]
    return controls, target, {
        "method": "robust_median_by_distributed_reference_perimeter_cell",
        "candidate_points": int(np.count_nonzero(eligible)),
        "control_count": int(len(controls)),
        "occupied_training_cells": int(len(controls)),
        "holdout_points_excluded": int(np.count_nonzero(holdout_union)),
        "maximum_training_match_px": config.maximum_training_match_px,
        "median_control_support": float(np.median(support)),
        "maximum_control_target_px": float(np.max(np.linalg.norm(target, axis=1))),
    }


def _distance_report(distances: np.ndarray, *, observable_limit: float) -> dict[str, Any]:
    values = np.asarray(distances, dtype=np.float64)
    observable = values <= observable_limit
    supported = values[observable]
    if not len(values):
        return {
            "count": 0,
            "observable_fraction": 0.0,
            "median_px": math.inf,
            "p90_px": math.inf,
            "within_8px_fraction": 0.0,
            "supported_median_px": math.inf,
            "supported_p90_px": math.inf,
            "supported_within_8px_fraction": 0.0,
        }
    return {
        "count": int(len(values)),
        "observable_count": int(np.count_nonzero(observable)),
        "observable_fraction": float(np.mean(observable)),
        "median_px": float(np.median(values)),
        "p90_px": float(np.percentile(values, 90)),
        "within_8px_fraction": float(np.mean(values <= 8.0)),
        "supported_median_px": float(np.median(supported)) if len(supported) else math.inf,
        "supported_p90_px": float(np.percentile(supported, 90)) if len(supported) else math.inf,
        "supported_within_8px_fraction": float(np.mean(supported <= 8.0)) if len(supported) else 0.0,
    }


def _forward_report(
    mapped: np.ndarray,
    source_mask: np.ndarray,
    *,
    observable_limit: float,
) -> dict[str, Any]:
    ys, xs = np.nonzero(source_mask)
    tree = cKDTree(np.column_stack((xs, ys)))
    distances, _ = tree.query(mapped)
    return _distance_report(distances, observable_limit=observable_limit)


def _render_points(shape: tuple[int, int], points: np.ndarray, radius: int = 1) -> np.ndarray:
    result = np.zeros(shape, dtype=np.uint8)
    rounded = np.rint(points).astype(int)
    valid = (
        (rounded[:, 0] >= 0)
        & (rounded[:, 0] < shape[1])
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < shape[0])
    )
    result[rounded[valid, 1], rounded[valid, 0]] = 1
    if radius:
        size = radius * 2 + 1
        result = cv2.dilate(result, np.ones((size, size), np.uint8))
    return result.astype(bool)


def _reverse_report(
    source_mask: np.ndarray,
    mapped_reference: np.ndarray,
    seed_reference: np.ndarray,
    *,
    observable_limit: float,
) -> dict[str, Any]:
    rendered = _render_points(source_mask.shape, mapped_reference, radius=1)
    seed = _render_points(source_mask.shape, seed_reference, radius=1)
    corridor = cv2.dilate(
        seed.astype(np.uint8),
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (round(observable_limit) * 2 + 1, round(observable_limit) * 2 + 1),
        ),
    ).astype(bool)
    scoped = source_mask & corridor
    distance = distance_transform_edt(~rendered)
    report = _distance_report(distance[scoped], observable_limit=observable_limit)
    report["corridor_source_fraction"] = float(
        np.count_nonzero(scoped) / max(np.count_nonzero(source_mask), 1)
    )
    report["corridor_source_pixels"] = int(np.count_nonzero(scoped))
    return report


def _jacobian_report(
    *,
    projection: ProjectionContext,
    grid: Mapping[str, Any],
    seed_matrix: np.ndarray,
    warp: CompactResidualWarp | None,
    state_land: np.ndarray,
    config: ElevationNonrigidConfig,
) -> dict[str, Any]:
    ys, xs = np.nonzero(state_land)
    x_values = np.linspace(xs.min(), xs.max() - 1.5, config.jacobian_grid[1])
    y_values = np.linspace(ys.min(), ys.max() - 1.5, config.jacobian_grid[0])
    xx, yy = np.meshgrid(x_values, y_values)
    points = np.column_stack((xx.ravel(), yy.ravel()))
    indices_x = np.clip(np.rint(points[:, 0]).astype(int), 0, state_land.shape[1] - 1)
    indices_y = np.clip(np.rint(points[:, 1]).astype(int), 0, state_land.shape[0] - 1)
    points = points[state_land[indices_y, indices_x]]
    base = _map_points(points, projection=projection, grid=grid, seed_matrix=seed_matrix, warp=warp)
    dx = _map_points(points + (1.0, 0.0), projection=projection, grid=grid, seed_matrix=seed_matrix, warp=warp) - base
    dy = _map_points(points + (0.0, 1.0), projection=projection, grid=grid, seed_matrix=seed_matrix, warp=warp) - base
    determinant = dx[:, 0] * dy[:, 1] - dx[:, 1] * dy[:, 0]
    median = float(np.median(determinant))
    ratio = float(np.min(determinant) / max(median, 1e-12))
    if len(base) > 1:
        nearest = cKDTree(base).query(base, k=2)[0][:, 1]
        minimum_separation = float(np.min(nearest))
    else:
        minimum_separation = 0.0
    globally_injective_grid = minimum_separation > 0.50
    return {
        "sample_count": int(len(determinant)),
        "minimum": float(np.min(determinant)),
        "median": median,
        "maximum": float(np.max(determinant)),
        "minimum_to_median_ratio": ratio,
        "positive_fraction": float(np.mean(determinant > 0.0)),
        "minimum_grid_node_separation_px": minimum_separation,
        "globally_injective_grid_passed": globally_injective_grid,
        "passed": bool(
            np.all(determinant > 0.0)
            and ratio >= config.minimum_jacobian_ratio
            and globally_injective_grid
        ),
    }


def _graticule_report(
    hypothesis: AlignmentSourceHypothesis,
    *,
    projection: ProjectionContext,
    grid: Mapping[str, Any],
    seed_matrix: np.ndarray,
    warp: CompactResidualWarp | None,
) -> dict[str, Any]:
    lonlat = np.asarray(hypothesis.graticule_lonlat, dtype=np.float64)
    source = np.asarray(hypothesis.graticule_source_points, dtype=np.float64)
    to_web = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    web_x, web_y = to_web.transform(lonlat[:, 0], lonlat[:, 1])
    reference_points = _web_mercator_to_reference_pixels(web_x, web_y, grid)
    mapped = _map_points(
        reference_points,
        projection=projection,
        grid=grid,
        seed_matrix=seed_matrix,
        warp=warp,
    )
    residual = np.linalg.norm(mapped - source, axis=1)
    return {
        "control_count": int(len(residual)),
        "median_residual_px": float(np.median(residual)),
        "maximum_residual_px": float(np.max(residual)),
    }


def _candidate_summary(
    candidate_id: str,
    warp: CompactResidualWarp | None,
    *,
    reference_points: np.ndarray,
    seed_mapped: np.ndarray,
    holdouts: Mapping[str, np.ndarray],
    source: SourceElevationEvidence,
    reference: PinnedMapboxReference,
    projection: ProjectionContext,
    seed_matrix: np.ndarray,
    hypothesis: AlignmentSourceHypothesis,
    config: ElevationNonrigidConfig,
    seed_holdout: Mapping[str, Mapping[str, Any]],
) -> CandidateSummary:
    mapped = _map_points(
        reference_points,
        projection=projection,
        grid=reference.grid,
        seed_matrix=seed_matrix,
        warp=warp,
    )
    displacement = np.linalg.norm(mapped - seed_mapped, axis=1)
    forward = _forward_report(
        mapped, source.combined, observable_limit=config.observable_corridor_px
    )
    reverse = _reverse_report(
        source.combined,
        mapped,
        seed_mapped,
        observable_limit=config.reverse_seed_corridor_px,
    )

    water_near = cv2.dilate(
        reference.water.astype(np.uint8), np.ones((9, 9), np.uint8)
    ).astype(bool)
    rounded_x = np.rint(reference_points[:, 0]).astype(int)
    rounded_y = np.rint(reference_points[:, 1]).astype(int)
    coast_ref = water_near[rounded_y, rounded_x]
    border_ref = ~coast_ref
    channels = {
        "coast": _forward_report(
            mapped[coast_ref], source.coast, observable_limit=config.observable_corridor_px
        ),
        "printed_state_border": _forward_report(
            mapped[border_ref],
            source.printed_state_border,
            observable_limit=config.observable_corridor_px,
        ),
    }
    holdout_reports = {}
    holdout_regression = True
    for name, mask in holdouts.items():
        holdout_source = (
            source.combined | source.inland_hydrography
            if name in {"bay_delta", "sf_bay_inlet", "san_pablo_suisun_delta"}
            else source.combined
        )
        report = _forward_report(
            mapped[mask], holdout_source, observable_limit=config.observable_corridor_px
        )
        base = seed_holdout[name]
        report["seed_supported_median_px"] = base["supported_median_px"]
        report["seed_supported_p90_px"] = base["supported_p90_px"]
        report["nonregression_passed"] = bool(
            report["supported_median_px"] <= base["supported_median_px"] + 0.50
            and report["supported_p90_px"] <= base["supported_p90_px"] + 1.00
        )
        holdout_regression &= report["nonregression_passed"]
        holdout_reports[name] = report

    jacobian = _jacobian_report(
        projection=projection,
        grid=reference.grid,
        seed_matrix=seed_matrix,
        warp=warp,
        state_land=reference.state_land,
        config=config,
    )
    graticule = _graticule_report(
        hypothesis,
        projection=projection,
        grid=reference.grid,
        seed_matrix=seed_matrix,
        warp=warp,
    )
    maximum_displacement = float(np.max(displacement)) if len(displacement) else 0.0
    eligibility = bool(
        jacobian["passed"]
        and maximum_displacement <= config.maximum_residual_displacement_px
        and holdout_regression
        and graticule["median_residual_px"] <= seed_holdout["__graticule__"]["median_residual_px"] + 1.0
    )
    # Selection is exclusively based on withheld geography plus regularity.  A
    # missing source segment remains visible through the coverage term.
    selection_score = float(
        np.mean(
            [
                value["supported_median_px"]
                + 0.35 * value["supported_p90_px"]
                + 20.0 * (1.0 - value["observable_fraction"])
                for value in holdout_reports.values()
            ]
        )
    )
    return CandidateSummary(
        id=candidate_id,
        radius_reference_px=None if warp is None else warp.radius_reference_px,
        ridge=None if warp is None else warp.ridge,
        selection_score=selection_score,
        eligible=eligibility,
        maximum_displacement_px=maximum_displacement,
        jacobian=jacobian,
        graticule=graticule,
        forward=forward,
        reverse=reverse,
        channels=channels,
        holdouts=holdout_reports,
    )


def _strict_gates(
    selected: CandidateSummary,
    *,
    pacific_contamination_fraction: float,
    config: ElevationNonrigidConfig,
) -> dict[str, Any]:
    coast = selected.channels["coast"]
    border = selected.channels["printed_state_border"]
    checks = {
        "positive_jacobian": bool(selected.jacobian["passed"]),
        "global_source_observability": selected.forward["observable_fraction"] >= config.global_observable_fraction_minimum,
        "global_supported_median": selected.forward["supported_median_px"] <= config.global_supported_median_limit_px,
        "global_supported_tail": selected.forward["supported_p90_px"] <= config.global_supported_p90_limit_px,
        "global_supported_within_8": selected.forward["supported_within_8px_fraction"] >= config.global_supported_within_8px_minimum,
        "reverse_corridor_coverage": selected.reverse["corridor_source_fraction"] >= config.reverse_corridor_fraction_minimum,
        "reverse_median": selected.reverse["supported_median_px"] <= config.reverse_median_limit_px,
        "reverse_tail": selected.reverse["supported_p90_px"] <= config.reverse_p90_limit_px,
        "printed_border_median": border["supported_median_px"] <= config.border_median_limit_px,
        "printed_border_tail": border["supported_p90_px"] <= config.border_p90_limit_px,
        "coast_source_observability": coast["observable_fraction"] >= config.coast_observable_fraction_minimum,
        "coast_supported_median": coast["supported_median_px"] <= config.coast_supported_median_limit_px,
        "coast_supported_tail": coast["supported_p90_px"] <= config.coast_supported_p90_limit_px,
        "pacific_contamination": pacific_contamination_fraction <= config.maximum_pacific_contamination_fraction,
    }
    for name, value in selected.holdouts.items():
        minimum_coverage = {
            "bay_delta": 0.80,
            "sf_bay_inlet": 0.95,
            "san_pablo_suisun_delta": 0.95,
            "islands": 0.80,
            "south_coast": 0.90,
            "southeast_colorado": 0.98,
        }[name]
        maximum_median = {
            "bay_delta": 4.5,
            "sf_bay_inlet": 3.5,
            "san_pablo_suisun_delta": 7.0,
            "islands": 4.5,
            "south_coast": 3.0,
            "southeast_colorado": 2.5,
        }[name]
        maximum_p90 = {
            "bay_delta": 20.0,
            "sf_bay_inlet": 12.0,
            "san_pablo_suisun_delta": 12.0,
            "islands": 20.0,
            "south_coast": 9.0,
            "southeast_colorado": 7.0,
        }[name]
        minimum_within_8 = {
            "bay_delta": 0.70,
            "sf_bay_inlet": 0.75,
            "san_pablo_suisun_delta": 0.75,
            "islands": 0.90,
            "south_coast": 0.88,
            "southeast_colorado": 0.98,
        }[name]
        checks[f"holdout_{name}_coverage"] = value["observable_fraction"] >= minimum_coverage
        checks[f"holdout_{name}_median"] = value["supported_median_px"] <= maximum_median
        checks[f"holdout_{name}_tail"] = value["supported_p90_px"] <= maximum_p90
        checks[f"holdout_{name}_within_8"] = value["supported_within_8px_fraction"] >= minimum_within_8
        checks[f"holdout_{name}_nonregression"] = bool(value["nonregression_passed"])
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "pacific_contamination_fraction": pacific_contamination_fraction,
    }


def _render_land(
    reference: PinnedMapboxReference,
    *,
    source_shape: tuple[int, int],
    projection: ProjectionContext,
    seed_matrix: np.ndarray,
    warp: CompactResidualWarp | None,
) -> np.ndarray:
    contours, hierarchy = cv2.findContours(
        reference.state_land.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
    )
    result = np.zeros(source_shape, dtype=np.uint8)
    if hierarchy is None:
        return result.astype(bool)
    transformed = []
    for contour in contours:
        points = contour[:, 0, :].astype(np.float64)
        mapped = _map_points(
            points,
            projection=projection,
            grid=reference.grid,
            seed_matrix=seed_matrix,
            warp=warp,
        )
        transformed.append(np.rint(mapped).astype(np.int32).reshape(-1, 1, 2))
    for index, contour in enumerate(transformed):
        parent = int(hierarchy[0, index, 3])
        cv2.drawContours(result, [contour], -1, 1 if parent < 0 else 0, -1, cv2.LINE_8)
    return result.astype(bool)


def _overlay(
    rgb: np.ndarray,
    source: SourceElevationEvidence,
    mapped: np.ndarray,
    holdouts: Mapping[str, np.ndarray],
    output: Path,
) -> None:
    canvas = rgb.copy()
    canvas[source.coast] = (255, 0, 255)
    canvas[source.inland_hydrography] = (255, 80, 80)
    canvas[source.printed_state_border] = (255, 190, 0)
    colors = {
        "bay_delta": (0, 255, 255),
        "sf_bay_inlet": (0, 210, 255),
        "san_pablo_suisun_delta": (190, 90, 255),
        "islands": (255, 255, 0),
        "south_coast": (0, 220, 80),
        "southeast_colorado": (70, 140, 255),
    }
    ordinary = np.ones(len(mapped), dtype=bool)
    for mask in holdouts.values():
        ordinary &= ~mask
    canvas[_render_points(canvas.shape[:2], mapped[ordinary], radius=1)] = (80, 255, 120)
    for name, mask in holdouts.items():
        canvas[_render_points(canvas.shape[:2], mapped[mask], radius=1)] = colors[name]
    Image.fromarray(canvas).save(output)


def run_elevation_nonrigid_throwaway(
    source_path: Path,
    reference_manifest_path: Path,
    output_root: Path,
    *,
    config: ElevationNonrigidConfig = ElevationNonrigidConfig(),
) -> ElevationNonrigidResult:
    """Run the isolated experiment and write only diagnostic artifacts."""

    source_path = source_path.resolve()
    reference_manifest_path = reference_manifest_path.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rgb = np.asarray(Image.open(source_path).convert("RGB"))
    reference = load_pinned_mapbox_reference(reference_manifest_path)
    source_only_root = output_root / "source-only"
    source_only_root.mkdir(exist_ok=True)
    generic = _source_semantic_evidence(rgb)
    hypothesis = _family_semantic_hypothesis(
        rgb, generic, "continuous_numeric_ramp", source_only_root
    )
    if hypothesis is None:
        raise ValueError("No source-only elevation hypothesis was produced")
    source = detect_pristine_elevation_evidence(rgb, hypothesis)
    projection = next(
        context
        for context in _projection_contexts(reference)
        if context.id == "california_lambert_conformal_conic"
    )
    seed_result = _graticule_seed_matrix(
        hypothesis, projection, reference.grid, model="regular_affine"
    )
    if seed_result is None:
        raise ValueError("The pristine elevation source lacks a native LCC graticule seed")
    seed_matrix, seed_diagnostics = seed_result

    reference_points = _sample_reference_boundary(reference.state_coast)
    seed_mapped = _map_points(
        reference_points,
        projection=projection,
        grid=reference.grid,
        seed_matrix=seed_matrix,
        warp=None,
    )
    holdouts = named_geographic_holdouts(reference_points, reference)
    holdout_union = np.logical_or.reduce(tuple(holdouts.values()))
    source_y, source_x = np.nonzero(source.combined)
    source_tree = cKDTree(np.column_stack((source_x, source_y)))
    controls, residuals, training_diagnostics = _aggregate_training_controls(
        reference_points,
        seed_mapped,
        source_tree,
        holdout_union,
        reference_shape=reference.state_coast.shape,
        config=config,
    )

    seed_holdout: dict[str, Mapping[str, Any]] = {}
    for name, mask in holdouts.items():
        holdout_source = (
            source.combined | source.inland_hydrography
            if name in {"bay_delta", "sf_bay_inlet", "san_pablo_suisun_delta"}
            else source.combined
        )
        seed_holdout[name] = _forward_report(
            seed_mapped[mask],
            holdout_source,
            observable_limit=config.observable_corridor_px,
        )
    seed_holdout["__graticule__"] = _graticule_report(
        hypothesis,
        projection=projection,
        grid=reference.grid,
        seed_matrix=seed_matrix,
        warp=None,
    )
    candidates: list[tuple[CandidateSummary, CompactResidualWarp | None]] = []
    baseline = _candidate_summary(
        "native-lcc-seed",
        None,
        reference_points=reference_points,
        seed_mapped=seed_mapped,
        holdouts=holdouts,
        source=source,
        reference=reference,
        projection=projection,
        seed_matrix=seed_matrix,
        hypothesis=hypothesis,
        config=config,
        seed_holdout=seed_holdout,
    )
    candidates.append((baseline, None))
    for radius in config.kernel_radii_reference_px:
        for ridge in config.ridge_values:
            warp = fit_compact_residual_warp(
                controls,
                residuals,
                radius_reference_px=radius,
                ridge=ridge,
            )
            summary = _candidate_summary(
                f"wendland-r{radius:g}-ridge{ridge:g}",
                warp,
                reference_points=reference_points,
                seed_mapped=seed_mapped,
                holdouts=holdouts,
                source=source,
                reference=reference,
                projection=projection,
                seed_matrix=seed_matrix,
                hypothesis=hypothesis,
                config=config,
                seed_holdout=seed_holdout,
            )
            candidates.append((summary, warp))
    eligible = [item for item in candidates if item[0].eligible]
    selected, selected_warp = min(
        eligible or candidates, key=lambda item: (item[0].selection_score, item[0].id)
    )
    selected_mapped = _map_points(
        reference_points,
        projection=projection,
        grid=reference.grid,
        seed_matrix=seed_matrix,
        warp=selected_warp,
    )
    rendered_land = _render_land(
        reference,
        source_shape=rgb.shape[:2],
        projection=projection,
        seed_matrix=seed_matrix,
        warp=selected_warp,
    )
    contamination = float(
        np.count_nonzero(rendered_land & source.water)
        / max(np.count_nonzero(rendered_land), 1)
    )
    gates = _strict_gates(
        selected,
        pacific_contamination_fraction=contamination,
        config=config,
    )

    evidence_path = output_root / "source-evidence.png"
    evidence_rgb = np.zeros_like(rgb)
    evidence_rgb[source.water] = (35, 75, 130)
    evidence_rgb[source.coast] = (255, 0, 255)
    evidence_rgb[source.inland_hydrography] = (255, 80, 80)
    evidence_rgb[source.printed_state_border] = (255, 190, 0)
    Image.fromarray(evidence_rgb).save(evidence_path)
    seed_overlay = output_root / "seed-overlay.png"
    final_overlay = output_root / "selected-overlay.png"
    _overlay(rgb, source, seed_mapped, holdouts, seed_overlay)
    _overlay(rgb, source, selected_mapped, holdouts, final_overlay)
    holdout_crop_paths: list[Path] = []
    final_image = Image.open(final_overlay)
    for name, mask in holdouts.items():
        points = selected_mapped[mask]
        finite = points[np.all(np.isfinite(points), axis=1)]
        if not len(finite):
            continue
        pad = 35
        left = max(0, math.floor(float(np.min(finite[:, 0]))) - pad)
        top = max(0, math.floor(float(np.min(finite[:, 1]))) - pad)
        right = min(rgb.shape[1], math.ceil(float(np.max(finite[:, 0]))) + pad + 1)
        bottom = min(rgb.shape[0], math.ceil(float(np.max(finite[:, 1]))) + pad + 1)
        crop_path = output_root / f"holdout-{name}.png"
        final_image.crop((left, top, right, bottom)).save(crop_path)
        holdout_crop_paths.append(crop_path)
    land_path = output_root / "selected-land-mask.png"
    Image.fromarray(rendered_land.astype(np.uint8) * 255).save(land_path)

    transform_path = output_root / "selected-transform.json"
    serialized_transform = (
        serialize_elevation_nonrigid_transform(
            seed_matrix=seed_matrix,
            projection=projection,
            warp=selected_warp,
            source_shape=rgb.shape[:2],
            target_grid=reference.grid,
        )
        if selected_warp is not None
        else None
    )
    if serialized_transform is not None:
        transform_path.write_text(
            json.dumps(serialized_transform, indent=2, sort_keys=True) + "\n"
        )

    report_path = output_root / "report.json"
    report = {
        "schema_version": 1,
        "kind": EXPERIMENT_KIND,
        "status": "strict_pass" if gates["passed"] else "strict_blocked",
        "throwaway_only": True,
        "eligible_for_promotion": False,
        "official_log_written": False,
        "inputs": {
            "source_path": str(source_path),
            "source_sha256": _sha256(source_path),
            "source_dimensions": [int(rgb.shape[1]), int(rgb.shape[0])],
            "reference_manifest_path": str(reference_manifest_path),
            "reference_manifest_sha256": _sha256(reference_manifest_path),
        },
        "projection_seed": {
            **seed_diagnostics,
            "matrix_candidate_normalized_to_source": seed_matrix.tolist(),
            "source_only_fit": True,
        },
        "source_evidence": dict(source.diagnostics),
        "training": training_diagnostics,
        "holdout_contract": {
            "names": list(holdouts),
            "counts": {name: int(np.count_nonzero(mask)) for name, mask in holdouts.items()},
            "union_count": int(np.count_nonzero(holdout_union)),
            "excluded_from_fit": True,
            "used_for_regularization_selection": True,
        },
        "config": asdict(config),
        "selected_candidate_id": selected.id,
        "selected_transform": serialized_transform,
        "candidates": [asdict(item[0]) for item in candidates],
        "strict_gates": gates,
        "artifacts": [
            str(evidence_path),
            str(seed_overlay),
            str(final_overlay),
            str(land_path),
            *([str(transform_path)] if serialized_transform is not None else []),
            *[str(path) for path in holdout_crop_paths],
        ],
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    status = report["status"]
    failed = [name for name, passed in gates["checks"].items() if not passed]
    stop_reason = "all_strict_gates_passed" if not failed else "failed_gates:" + ",".join(failed)
    artifacts = (
        evidence_path,
        seed_overlay,
        final_overlay,
        land_path,
        *((transform_path,) if serialized_transform is not None else ()),
        *holdout_crop_paths,
        report_path,
    )
    return ElevationNonrigidResult(
        status=status,
        stop_reason=stop_reason,
        report_path=report_path,
        artifact_paths=artifacts,
        selected_candidate=selected,
        all_candidates=tuple(item[0] for item in candidates),
    )
