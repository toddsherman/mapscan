"""Closed-loop perimeter registration for map images.

The first-stage aligner intentionally uses broad outline evidence.  This module
audits that result on an *unclipped* Web-Mercator warp, finds independent local
matches around the authoritative California perimeter, fits the simplest
globally supported correction, and repeats only when held-out anchors and the
rendered boundary both improve.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image
from pyproj import Transformer
from scipy.ndimage import distance_transform_edt
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.ops import transform as transform_geometry

from .extraction import (
    _projection_normalizer,
    _transform_normalized_to_source,
    warp_classified_to_web_mercator,
)
from .reference import load_california
from .refinement import _apply_matrix, fit_review_corrections


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _polygons(geometry) -> Iterable[Polygon]:
    if isinstance(geometry, Polygon):
        yield geometry
    elif isinstance(geometry, MultiPolygon):
        yield from geometry.geoms
    else:  # pragma: no cover - the Census state geometry is polygonal.
        raise TypeError(f"Expected polygonal geometry, received {geometry.geom_type}")


def _largest_polygon(geometry) -> Polygon:
    return max(_polygons(geometry), key=lambda item: item.area)


def _alignment_transform(
    alignment: Dict[str, object], state
) -> Dict[str, object]:
    mode = alignment.get("alignment_mode")
    if mode == "assisted":
        transform = {
            "projection": "assisted_reference_crs",
            "projection_crs": alignment["reference"]["crs"],
            "transform_model": alignment["transform_model"],
            "reference_to_source_matrix": alignment["reference_to_source_matrix"],
        }
    elif mode == "native_pdf_graticule":
        projection_crs = str(alignment["projection"]["crs"])
        _, center_x, center_y, state_height = _projection_normalizer(
            state, projection_crs
        )
        normalized_to_projected = np.asarray(
            [
                [state_height, 0.0, center_x],
                [0.0, -state_height, center_y],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        affine = np.asarray(
            alignment["projected_crs_to_page_affine"], dtype=np.float64
        )
        page_to_render = np.asarray(
            [
                [float(alignment["source"]["render_x_px_per_page_point"]), 0.0, 0.0],
                [0.0, float(alignment["source"]["render_y_px_per_page_point"]), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        projected_to_page = np.vstack((affine, [0.0, 0.0, 1.0]))
        matrix = page_to_render @ projected_to_page @ normalized_to_projected
        transform = {
            "projection": "native_pdf_graticule",
            "projection_crs": projection_crs,
            "transform_model": "native_pdf_graticule_affine",
            "reference_to_source_matrix": matrix.tolist(),
        }
    else:
        transform = dict(alignment["best"])
    if "web_mercator_correction" in alignment:
        transform["web_mercator_correction"] = alignment["web_mercator_correction"]
    return transform


def _native_grid(
    transform: Dict[str, object], state, source_shape: Tuple[int, int]
) -> Dict[str, object]:
    projected_web = transform_geometry(
        Transformer.from_crs("EPSG:4269", "EPSG:3857", always_xy=True).transform,
        state,
    )
    bounds = projected_web.bounds
    correction = transform.get("web_mercator_correction")
    if isinstance(correction, dict):
        grid = correction["grid"]
        return {
            "crs": "EPSG:3857",
            "bounds": [float(value) for value in grid["bounds"]],
            "width": int(grid["width"]),
            "height": int(grid["height"]),
        }
    if "reference_to_source_matrix" in transform:
        projected, center_x, center_y, state_height = _projection_normalizer(
            state, str(transform["projection_crs"])
        )
        outline = np.asarray(_largest_polygon(projected).exterior.coords, dtype=np.float64)
        normalized = np.column_stack(
            (
                (outline[:, 0] - center_x) / state_height,
                (center_y - outline[:, 1]) / state_height,
            )
        )
        source_outline = _transform_normalized_to_source(
            normalized, transform, source_shape
        )
        height = max(256, round(float(np.ptp(source_outline[:, 1]))))
    else:
        height = max(
            256,
            round(
                float(transform["parameters"]["state_height_fraction"])
                * source_shape[0]
            ),
        )
    width = max(
        256,
        round(height * (bounds[2] - bounds[0]) / (bounds[3] - bounds[1])),
    )
    return {
        "crs": "EPSG:3857",
        "bounds": [float(value) for value in bounds],
        "width": width,
        "height": height,
    }


def _edge_evidence(
    rgb: np.ndarray, valid: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    fine = cv2.GaussianBlur(gray, (3, 3), 0)
    coarse = cv2.GaussianBlur(gray, (7, 7), 0)
    median = float(np.median(fine[valid])) if np.any(valid) else 128.0
    lower = max(18, int(0.40 * median))
    upper = max(lower + 22, int(0.88 * median))
    edges = (cv2.Canny(fine, lower, upper, L2gradient=True) > 0) | (
        cv2.Canny(coarse, max(12, lower // 2), max(35, upper // 2), L2gradient=True)
        > 0
    )
    # Never let the edge of an image crop masquerade as a geographic border.
    safe_valid = cv2.erode(valid.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    edges &= safe_valid
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
    return edges, distance.astype(np.float32), nearest, gradient_x, gradient_y


def _perimeter_segments(
    state,
    bounds: Sequence[float],
    shape: Tuple[int, int],
    count: int,
    samples_per_segment: int = 71,
) -> list[Dict[str, object]]:
    projected = transform_geometry(
        Transformer.from_crs("EPSG:4269", "EPSG:3857", always_xy=True).transform,
        state,
    )
    perimeter = LineString(_largest_polygon(projected).exterior.coords)
    height, width = shape
    min_x, min_y, max_x, max_y = bounds

    def pixel(point) -> np.ndarray:
        return np.asarray(
            [
                (point.x - min_x) / (max_x - min_x) * (width - 1),
                (max_y - point.y) / (max_y - min_y) * (height - 1),
            ],
            dtype=np.float64,
        )

    # Each local template covers about one quarter of the space between anchors.
    half_length = perimeter.length / count * 0.24
    segments = []
    for index in range(count):
        center_distance = perimeter.length * (index + 0.5) / count
        distances = np.linspace(
            center_distance - half_length,
            center_distance + half_length,
            samples_per_segment,
        )
        points = np.asarray(
            [pixel(perimeter.interpolate(value % perimeter.length)) for value in distances]
        )
        center = points[len(points) // 2]
        tangent = points[min(len(points) // 2 + 3, len(points) - 1)] - points[
            max(len(points) // 2 - 3, 0)
        ]
        tangent /= max(float(np.linalg.norm(tangent)), 1e-9)
        normals = np.column_stack(
            (
                -np.gradient(points[:, 1]),
                np.gradient(points[:, 0]),
            )
        )
        normal_lengths = np.linalg.norm(normals, axis=1)
        normals = np.divide(
            normals,
            normal_lengths[:, None],
            out=np.zeros_like(normals),
            where=normal_lengths[:, None] > 1e-9,
        )
        segments.append(
            {
                "id": f"perimeter_{index:02d}",
                "index": index,
                "center": center,
                "points": points,
                "tangent": tangent,
                "normal": np.asarray([-tangent[1], tangent[0]]),
                "normals": normals,
            }
        )
    return segments


def _county_patch_segments(
    counties,
    bounds: Sequence[float],
    shape: Tuple[int, int],
    valid: np.ndarray,
    maximum_count: int = 28,
    patch_radius_px: int = 42,
) -> list[Dict[str, object]]:
    """Build distinctive county-line junction templates inside visible source coverage."""

    height, width = shape
    min_x, min_y, max_x, max_y = bounds
    transformer = Transformer.from_crs("EPSG:4269", "EPSG:3857", always_xy=True)
    line_mask = np.zeros(shape, dtype=np.uint8)

    def pixels(coords) -> np.ndarray:
        points = np.asarray(coords, dtype=np.float64)
        x = (points[:, 0] - min_x) / (max_x - min_x) * (width - 1)
        y = (max_y - points[:, 1]) / (max_y - min_y) * (height - 1)
        return np.rint(np.column_stack((x, y))).astype(np.int32).reshape((-1, 1, 2))

    for county in counties:
        projected = transform_geometry(transformer.transform, county)
        for polygon in _polygons(projected):
            cv2.polylines(
                line_mask,
                [pixels(polygon.exterior.coords)],
                True,
                1,
                1,
                cv2.LINE_8,
            )
    return _line_patch_segments(
        line_mask,
        valid,
        maximum_count=maximum_count,
        patch_radius_px=patch_radius_px,
        id_prefix="county_vector",
        family="county_junction",
    )


def _line_patch_segments(
    line_mask: np.ndarray,
    valid: np.ndarray,
    *,
    maximum_count: int = 28,
    patch_radius_px: int = 42,
    id_prefix: str = "county",
    family: str = "county_junction",
) -> list[Dict[str, object]]:
    """Build distinctive junction templates from a registered raster line mask."""

    line_mask = line_mask.astype(np.uint8, copy=False)
    shape = line_mask.shape
    height, width = shape
    safe_valid = cv2.erode(valid.astype(np.uint8), np.ones((13, 13), np.uint8)) > 0
    corner_image = cv2.dilate(line_mask, np.ones((3, 3), np.uint8)) * 255
    corners = cv2.goodFeaturesToTrack(
        corner_image,
        maxCorners=maximum_count * 4,
        qualityLevel=0.004,
        minDistance=max(30, round(min(shape) * 0.045)),
        blockSize=9,
        useHarrisDetector=True,
        k=0.04,
    )
    if corners is None:
        return []
    candidates = []
    for point in corners[:, 0, :]:
        center = np.rint(point).astype(int)
        if not (0 <= center[0] < width and 0 <= center[1] < height):
            continue
        if not safe_valid[center[1], center[0]]:
            continue
        x1, x2 = max(0, center[0] - patch_radius_px), min(
            width, center[0] + patch_radius_px + 1
        )
        y1, y2 = max(0, center[1] - patch_radius_px), min(
            height, center[1] + patch_radius_px + 1
        )
        local_y, local_x = np.nonzero(line_mask[y1:y2, x1:x2])
        points = np.column_stack((local_x + x1, local_y + y1)).astype(np.float64)
        if len(points) < 40:
            continue
        radial = np.linalg.norm(points - center, axis=1)
        points = points[radial <= patch_radius_px]
        if len(points) < 40:
            continue
        relative = points - center
        quadrants = {
            (int(delta[0] >= 0), int(delta[1] >= 0))
            for delta in relative
            if np.linalg.norm(delta) >= patch_radius_px * 0.30
        }
        # Three occupied quadrants favor junctions and hard bends over an
        # ambiguous straight county segment.
        if len(quadrants) < 3:
            continue
        if len(points) > 180:
            indices = np.linspace(0, len(points) - 1, 180).astype(int)
            points = points[indices]
        candidates.append(
            {
                "id": f"{id_prefix}_{len(candidates):02d}",
                "family": family,
                "index": len(candidates),
                "center": center.astype(np.float64),
                "points": points,
            }
        )
        if len(candidates) >= maximum_count:
            break
    return candidates


def _registered_reference_masks(
    manifest: Dict[str, object],
    manifest_path: Path,
    bounds: Sequence[float],
    shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Resize registered Web-Mercator line masks onto the current state grid."""

    if manifest.get("status") != "pass":
        raise ValueError("Registered county reference has not passed registration QA")
    source_bounds = np.asarray(manifest["web_grid"]["bounds"], dtype=np.float64)
    target_bounds = np.asarray(bounds, dtype=np.float64)
    tolerance = max(float(np.ptp(source_bounds)), 1.0) * 1e-9
    if np.max(np.abs(source_bounds - target_bounds)) > tolerance:
        raise ValueError("Registered county reference and alignment grids have different bounds")
    root = manifest_path.parent

    def load(artifact_name: str) -> np.ndarray:
        artifact = manifest["artifacts"][artifact_name]
        path = root / str(artifact["path"])
        if _sha256(path) != artifact["sha256"]:
            raise ValueError(f"Registered county-reference artifact hash mismatch: {path}")
        values = np.asarray(Image.open(path)) > 0
        return cv2.resize(
            values.astype(np.uint8),
            (shape[1], shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ) > 0

    return load("web_mercator_state_border"), load("web_mercator_county_border")


def _county_zero_shift_metric(
    segments: list[Dict[str, object]],
    accepted_matches: list[Dict[str, object]],
    valid: np.ndarray,
    distance: np.ndarray,
) -> Dict[str, object]:
    """Summarize zero-shift residuals for the same locally visible junctions."""

    accepted_ids = {
        item["id"] for item in accepted_matches if item["accepted_by_local_evidence"]
    }
    values = []
    for segment in segments:
        if segment["id"] not in accepted_ids:
            continue
        metrics = _county_patch_score(
            np.asarray(segment["points"]), np.zeros(2), valid, distance
        )
        if metrics["valid_fraction"] >= 0.70 and metrics["median_distance_px"] < 1e5:
            values.append(float(metrics["median_distance_px"]))
    return _residual_summary(np.asarray(values, dtype=np.float64))


def _county_patch_score(
    points: np.ndarray,
    shift: np.ndarray,
    valid: np.ndarray,
    distance: np.ndarray,
) -> Dict[str, float]:
    shifted = np.rint(points + shift).astype(np.int32)
    height, width = valid.shape
    inside = (
        (shifted[:, 0] >= 0)
        & (shifted[:, 0] < width)
        & (shifted[:, 1] >= 0)
        & (shifted[:, 1] < height)
    )
    if not np.any(inside):
        return {
            "score": 1e6,
            "valid_fraction": 0.0,
            "support_fraction": 0.0,
            "median_distance_px": 1e6,
            "p90_distance_px": 1e6,
        }
    sampled = shifted[inside]
    usable = valid[sampled[:, 1], sampled[:, 0]]
    valid_fraction = float(np.count_nonzero(usable) / len(points))
    if np.count_nonzero(usable) < max(30, len(points) // 2):
        return {
            "score": 1e6,
            "valid_fraction": valid_fraction,
            "support_fraction": 0.0,
            "median_distance_px": 1e6,
            "p90_distance_px": 1e6,
        }
    sampled = sampled[usable]
    values = distance[sampled[:, 1], sampled[:, 0]]
    median_distance = float(np.median(values))
    p90_distance = float(np.quantile(values, 0.9))
    support = float(np.mean(values <= 2.5))
    score = float(
        np.mean(np.minimum(values, 12.0))
        + 0.20 * min(p90_distance, 12.0)
        - 1.75 * support
        + 0.014 * np.linalg.norm(shift)
    )
    return {
        "score": score,
        "valid_fraction": valid_fraction,
        "support_fraction": support,
        "median_distance_px": median_distance,
        "p90_distance_px": p90_distance,
    }


def _match_county_anchor(
    segment: Dict[str, object],
    valid: np.ndarray,
    distance: np.ndarray,
    search_radius_px: int,
) -> Dict[str, object]:
    scored = []
    radius = min(search_radius_px, 28)
    for y_shift in range(-radius, radius + 1):
        for x_shift in range(-radius, radius + 1):
            if math.hypot(x_shift, y_shift) > radius:
                continue
            shift = np.asarray([x_shift, y_shift], dtype=np.float64)
            metrics = _county_patch_score(
                np.asarray(segment["points"]), shift, valid, distance
            )
            scored.append((float(metrics["score"]), shift, metrics))
    scored.sort(key=lambda item: item[0])
    best_score, best_shift, best = scored[0]
    distinct = [
        item for item in scored[1:] if np.linalg.norm(item[1] - best_shift) >= 5.0
    ]
    second_score = float(distinct[0][0]) if distinct else best_score
    baseline = next(
        metrics for _, shift, metrics in scored if np.array_equal(shift, np.zeros(2))
    )
    already_close = bool(
        baseline["median_distance_px"] <= 1.5
        and baseline["support_fraction"] >= 0.62
    )
    accepted = bool(
        best["valid_fraction"] >= 0.88
        and best["support_fraction"] >= 0.58
        and best["median_distance_px"] <= 2.0
        and best["p90_distance_px"] <= 6.0
        and (second_score - best_score >= 0.08 or already_close)
    )
    return {
        "id": segment["id"],
        "family": segment.get("family", "county_junction"),
        "index": segment["index"],
        "reference_pixel": np.asarray(segment["center"]).tolist(),
        "source_pixel": (np.asarray(segment["center"]) + best_shift).tolist(),
        "shift_px": best_shift.tolist(),
        "shift_magnitude_px": float(np.linalg.norm(best_shift)),
        "accepted_by_local_evidence": accepted,
        "best": best,
        "baseline": baseline,
        "uniqueness_margin": second_score - best_score,
    }


def _segment_score(
    points: np.ndarray,
    normals: np.ndarray,
    shift: np.ndarray,
    valid: np.ndarray,
    distance: np.ndarray,
    nearest: np.ndarray,
    gradient_x: np.ndarray,
    gradient_y: np.ndarray,
) -> Dict[str, float]:
    shifted = np.rint(points + shift).astype(np.int32)
    height, width = valid.shape
    inside = (
        (shifted[:, 0] >= 0)
        & (shifted[:, 0] < width)
        & (shifted[:, 1] >= 0)
        & (shifted[:, 1] < height)
    )
    if not np.any(inside):
        return {
            "score": 1e6,
            "valid_fraction": 0.0,
            "support_fraction": 0.0,
            "median_distance_px": 1e6,
            "p90_distance_px": 1e6,
            "median_direction_agreement": 0.0,
        }
    sampled = shifted[inside]
    valid_samples = valid[sampled[:, 1], sampled[:, 0]]
    valid_fraction = float(np.count_nonzero(valid_samples) / len(points))
    if np.count_nonzero(valid_samples) < max(12, len(points) // 2):
        return {
            "score": 1e6,
            "valid_fraction": valid_fraction,
            "support_fraction": 0.0,
            "median_distance_px": 1e6,
            "p90_distance_px": 1e6,
            "median_direction_agreement": 0.0,
        }
    sampled = sampled[valid_samples]
    sample_normals = normals[inside][valid_samples]
    values = distance[sampled[:, 1], sampled[:, 0]]
    nearest_y = nearest[0, sampled[:, 1], sampled[:, 0]]
    nearest_x = nearest[1, sampled[:, 1], sampled[:, 0]]
    agreement = np.abs(
        gradient_x[nearest_y, nearest_x] * sample_normals[:, 0]
        + gradient_y[nearest_y, nearest_x] * sample_normals[:, 1]
    )
    median_distance = float(np.median(values))
    p90_distance = float(np.quantile(values, 0.9))
    support = float(np.mean(values <= 2.5))
    direction = float(np.median(agreement))
    score = float(
        np.mean(np.minimum(values, 12.0))
        + 0.20 * min(p90_distance, 12.0)
        + 1.25 * (1.0 - direction)
        - 1.75 * support
        + 0.012 * np.linalg.norm(shift)
    )
    return {
        "score": score,
        "valid_fraction": valid_fraction,
        "support_fraction": support,
        "median_distance_px": median_distance,
        "p90_distance_px": p90_distance,
        "median_direction_agreement": direction,
    }


def _match_perimeter_anchor(
    segment: Dict[str, object],
    valid: np.ndarray,
    distance: np.ndarray,
    nearest: np.ndarray,
    gradient_x: np.ndarray,
    gradient_y: np.ndarray,
    search_radius_px: int,
    tangent_radius_px: int,
) -> Dict[str, object]:
    tangent = np.asarray(segment["tangent"])
    normal = np.asarray(segment["normal"])
    shifts = set()
    for normal_step in range(-search_radius_px, search_radius_px + 1):
        for tangent_step in range(-tangent_radius_px, tangent_radius_px + 1):
            if math.hypot(normal_step, tangent_step) > search_radius_px:
                continue
            shift = np.rint(normal * normal_step + tangent * tangent_step).astype(int)
            shifts.add((int(shift[0]), int(shift[1])))
    scored = []
    for shift_xy in shifts:
        shift = np.asarray(shift_xy, dtype=np.float64)
        metrics = _segment_score(
            np.asarray(segment["points"]),
            np.asarray(segment["normals"]),
            shift,
            valid,
            distance,
            nearest,
            gradient_x,
            gradient_y,
        )
        scored.append((float(metrics["score"]), shift, metrics))
    scored.sort(key=lambda item: item[0])
    best_score, best_shift, best = scored[0]
    distinct = [
        item for item in scored[1:] if np.linalg.norm(item[1] - best_shift) >= 4.0
    ]
    second_score = float(distinct[0][0]) if distinct else best_score
    baseline = next(
        metrics for _, shift, metrics in scored if np.array_equal(shift, np.zeros(2))
    )
    accepted = bool(
        best["valid_fraction"] >= 0.82
        and best["support_fraction"] >= 0.42
        and best["median_distance_px"] <= 2.5
        and best["p90_distance_px"] <= 7.0
        and best["median_direction_agreement"] >= 0.30
    )
    return {
        "id": segment["id"],
        "family": "state_perimeter",
        "index": segment["index"],
        "reference_pixel": np.asarray(segment["center"]).tolist(),
        "source_pixel": (np.asarray(segment["center"]) + best_shift).tolist(),
        "shift_px": best_shift.tolist(),
        "shift_magnitude_px": float(np.linalg.norm(best_shift)),
        "accepted_by_local_evidence": accepted,
        "best": best,
        "baseline": baseline,
        "uniqueness_margin": second_score - best_score,
    }


def _consistent_matches(matches: list[Dict[str, object]]) -> list[Dict[str, object]]:
    accepted = [item for item in matches if item["accepted_by_local_evidence"]]
    if len(accepted) < 4:
        return []
    current = np.asarray([item["source_pixel"] for item in accepted], dtype=np.float64)
    target = np.asarray([item["reference_pixel"] for item in accepted], dtype=np.float64)
    matrix, inliers = cv2.estimateAffine2D(
        current,
        target,
        method=cv2.RANSAC,
        ransacReprojThreshold=4.5,
        maxIters=5000,
        confidence=0.995,
        refineIters=20,
    )
    if matrix is None or inliers is None:
        return []
    result = []
    for item, is_inlier in zip(accepted, inliers.ravel() > 0):
        item["accepted_by_global_consistency"] = bool(is_inlier)
        if is_inlier:
            result.append(item)
    return result


def _select_distributed(
    matches: list[Dict[str, object]], count: int, shape: Tuple[int, int]
) -> list[Dict[str, object]]:
    if len(matches) <= count:
        return list(matches)
    height, width = shape
    coordinates = np.asarray(
        [item["reference_pixel"] for item in matches], dtype=np.float64
    ) / np.asarray([max(width - 1, 1), max(height - 1, 1)])
    quality = np.asarray(
        [
            float(item["best"]["support_fraction"])
            - 0.04 * float(item["best"]["median_distance_px"])
            + 0.01 * min(float(item["uniqueness_margin"]), 4.0)
            for item in matches
        ]
    )
    chosen = [int(np.argmax(quality))]
    while len(chosen) < count:
        remaining = [index for index in range(len(matches)) if index not in chosen]
        distances = np.asarray(
            [
                min(np.linalg.norm(coordinates[index] - coordinates[other]) for other in chosen)
                for index in remaining
            ]
        )
        scores = distances + 0.12 * quality[remaining]
        chosen.append(remaining[int(np.argmax(scores))])
    return [matches[index] for index in sorted(chosen)]


def _residual_summary(values: np.ndarray) -> Dict[str, object]:
    if not len(values):
        return {"count": 0, "median_px": None, "p90_px": None, "max_px": None}
    return {
        "count": int(len(values)),
        "median_px": float(np.median(values)),
        "p90_px": float(np.quantile(values, 0.9)),
        "max_px": float(np.max(values)),
    }


def _zero_shift_boundary_metric(
    segments: list[Dict[str, object]],
    valid: np.ndarray,
    distance: np.ndarray,
    nearest: np.ndarray,
    gradient_x: np.ndarray,
    gradient_y: np.ndarray,
) -> Dict[str, float]:
    values = []
    for segment in segments:
        score = _segment_score(
            np.asarray(segment["points"]),
            np.asarray(segment["normals"]),
            np.zeros(2),
            valid,
            distance,
            nearest,
            gradient_x,
            gradient_y,
        )
        if score["valid_fraction"] >= 0.82 and score["median_distance_px"] < 1e5:
            values.append(float(score["median_distance_px"]))
    return _residual_summary(np.asarray(values, dtype=np.float64))


def _warp_evidence(
    rgb: np.ndarray,
    state,
    transform: Dict[str, object],
    target_height: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object], tuple]:
    warped, grid = warp_classified_to_web_mercator(
        rgb,
        state,
        transform,
        rgb.shape[:2],
        target_height=target_height,
        clip_to_state=False,
    )
    valid_source = np.ones(rgb.shape[:2], dtype=np.uint8)
    valid, _ = warp_classified_to_web_mercator(
        valid_source,
        state,
        transform,
        rgb.shape[:2],
        target_height=target_height,
        clip_to_state=False,
    )
    evidence = _edge_evidence(warped, valid > 0)
    return warped, valid > 0, grid, evidence


def _write_diagnostic(
    path: Path,
    warped: np.ndarray,
    segments: list[Dict[str, object]],
    matches: list[Dict[str, object]],
) -> None:
    gray = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY)
    output = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    for segment in segments:
        points = np.rint(segment["points"]).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(output, [points], False, (0, 230, 230), 2, cv2.LINE_AA)
    for item in matches:
        start = tuple(np.rint(item["reference_pixel"]).astype(int))
        end = tuple(np.rint(item["source_pixel"]).astype(int))
        accepted = item.get(
            "accepted_by_global_consistency", item.get("accepted_by_local_evidence", False)
        )
        color = (45, 220, 90) if accepted else (230, 70, 70)
        cv2.arrowedLine(output, start, end, color, 2, cv2.LINE_AA, tipLength=0.22)
        cv2.circle(output, start, 4, (0, 230, 230), -1, cv2.LINE_AA)
    Image.fromarray(output, mode="RGB").save(path, quality=94, subsampling=0)


def _correction_record(
    matches: list[Dict[str, object]],
    working_shape: Tuple[int, int],
    native_grid: Dict[str, object],
) -> Dict[str, object]:
    working_height, working_width = working_shape
    scale = np.asarray(
        [
            max(int(native_grid["width"]) - 1, 1) / max(working_width - 1, 1),
            max(int(native_grid["height"]) - 1, 1) / max(working_height - 1, 1),
        ]
    )
    return {
        "schema_version": 1,
        "direction": "authoritative_reference_to_current_warped_source",
        "generated_by": "mapscan.auto_refinement",
        "grid": native_grid,
        "corrections": [
            {
                "id": item["id"],
                "reference": {
                    "pixel": {
                        "x": float(item["reference_pixel"][0] * scale[0]),
                        "y": float(item["reference_pixel"][1] * scale[1]),
                    }
                },
                "source": {
                    "pixel": {
                        "x": float(item["source_pixel"][0] * scale[0]),
                        "y": float(item["source_pixel"][1] * scale[1]),
                    }
                },
                "local_evidence": item["best"],
            }
            for item in matches
        ],
    }


def auto_refine_perimeter(
    image_path: Path,
    alignment_path: Path,
    reference_root: Path,
    output_dir: Path,
    *,
    max_iterations: int = 3,
    working_height: int = 1600,
    candidate_anchor_count: int = 16,
    fit_anchor_count: int = 8,
    search_radius_px: int = 36,
    tangent_radius_px: int = 12,
    preserve_geographic_registration: bool = False,
    county_reference_path: Path | None = None,
) -> Dict[str, object]:
    """Audit and conservatively refine a California perimeter registration."""

    if candidate_anchor_count < fit_anchor_count:
        raise ValueError("candidate_anchor_count must be at least fit_anchor_count")
    output_dir.mkdir(parents=True, exist_ok=True)
    rgb = np.asarray(Image.open(image_path).convert("RGB"))
    state, counties = load_california(reference_root)
    initial_alignment = json.loads(alignment_path.read_text())
    county_reference_manifest = (
        json.loads(county_reference_path.read_text())
        if county_reference_path is not None
        else None
    )
    current_alignment_path = alignment_path
    iteration_reports = []
    accepted_iterations = 0
    stop_reason = "maximum_iterations_reached"

    iteration_limit = max(1, max_iterations)
    for iteration_index in range(iteration_limit):
        iteration_dir = output_dir / f"iteration-{iteration_index + 1:02d}"
        iteration_dir.mkdir(parents=True, exist_ok=True)
        alignment = json.loads(current_alignment_path.read_text())
        transform = _alignment_transform(alignment, state)
        native_grid = _native_grid(transform, state, rgb.shape[:2])
        actual_working_height = min(working_height, int(native_grid["height"]))
        warped, valid, grid, evidence = _warp_evidence(
            rgb, state, transform, actual_working_height
        )
        edges, distance, nearest, gradient_x, gradient_y = evidence
        del edges
        segments = _perimeter_segments(
            state,
            grid["bounds"],
            warped.shape[:2],
            candidate_anchor_count,
        )
        perimeter_matches = [
            _match_perimeter_anchor(
                segment,
                valid,
                distance,
                nearest,
                gradient_x,
                gradient_y,
                search_radius_px,
                tangent_radius_px,
            )
            for segment in segments
        ]
        matches = list(perimeter_matches)
        consistent = _consistent_matches(perimeter_matches)
        county_segments = []
        county_matches = []
        registered_state_mask = None
        if county_reference_manifest is not None and county_reference_path is not None:
            registered_state_mask, registered_county_mask = _registered_reference_masks(
                county_reference_manifest,
                county_reference_path,
                grid["bounds"],
                warped.shape[:2],
            )
            county_segments = _line_patch_segments(
                registered_county_mask,
                valid,
                id_prefix="county_raster",
                family="registered_county_junction",
            )
            county_matches = [
                _match_county_anchor(segment, valid, distance, search_radius_px)
                for segment in county_segments
            ]
        if len(consistent) < fit_anchor_count and not county_segments:
            county_segments = _county_patch_segments(
                counties, grid["bounds"], warped.shape[:2], valid
            )
            county_matches = [
                _match_county_anchor(
                    segment, valid, distance, search_radius_px
                )
                for segment in county_segments
            ]
        if len(consistent) < fit_anchor_count:
            matches.extend(county_matches)
            consistent = _consistent_matches(matches)
        fit_matches = _select_distributed(
            consistent, min(fit_anchor_count, len(consistent)), warped.shape[:2]
        )
        fit_ids = {item["id"] for item in fit_matches}
        held_out = [item for item in consistent if item["id"] not in fit_ids]
        before_boundary = _zero_shift_boundary_metric(
            segments, valid, distance, nearest, gradient_x, gradient_y
        )
        before_county_reference = _county_zero_shift_metric(
            county_segments, county_matches, valid, distance
        )
        _write_diagnostic(
            iteration_dir / "perimeter-matches-before.jpg",
            warped,
            segments,
            list(perimeter_matches) + list(county_matches),
        )
        Image.fromarray(warped, mode="RGB").save(
            iteration_dir / "unclipped-source-warp.jpg", quality=94, subsampling=0
        )
        iteration_report: Dict[str, object] = {
            "iteration": iteration_index + 1,
            "alignment": str(current_alignment_path),
            "alignment_sha256": _sha256(current_alignment_path),
            "working_grid": grid,
            "native_grid": native_grid,
            "candidate_anchor_count": len(matches),
            "locally_accepted_anchor_count": int(
                sum(bool(item["accepted_by_local_evidence"]) for item in matches)
            ),
            "perimeter_locally_accepted_anchor_count": int(
                sum(
                    bool(item["accepted_by_local_evidence"])
                    for item in perimeter_matches
                )
            ),
            "county_candidate_anchor_count": len(county_matches),
            "county_locally_accepted_anchor_count": int(
                sum(bool(item["accepted_by_local_evidence"]) for item in county_matches)
            ),
            "registered_county_reference": (
                {
                    "path": str(county_reference_path),
                    "sha256": _sha256(county_reference_path),
                    "state_border_mask_pixel_count": int(
                        np.count_nonzero(registered_state_mask)
                    ),
                    "county_zero_shift_before": before_county_reference,
                }
                if county_reference_path is not None and registered_state_mask is not None
                else None
            ),
            "globally_consistent_anchor_count": len(consistent),
            "fit_anchor_ids": sorted(fit_ids),
            "held_out_anchor_ids": sorted(item["id"] for item in held_out),
            "boundary_before": before_boundary,
            "matches": matches,
            "accepted": False,
        }
        if preserve_geographic_registration:
            iteration_report["decision"] = "audit_only_preserve_stronger_registration"
            iteration_reports.append(iteration_report)
            stop_reason = "stronger_geographic_registration_preserved"
            break
        if max_iterations <= 0:
            iteration_report["decision"] = "audit_only"
            iteration_reports.append(iteration_report)
            stop_reason = "audit_only"
            break
        if len(fit_matches) < fit_anchor_count:
            perimeter_verified = bool(
                before_boundary["count"] >= 8
                and before_boundary["median_px"] is not None
                and before_boundary["median_px"] <= 2.0
                and before_boundary["p90_px"] <= 8.0
            )
            county_verified = bool(
                before_county_reference["count"] >= 8
                and before_county_reference["median_px"] is not None
                and before_county_reference["median_px"] <= 2.0
                and before_county_reference["p90_px"] <= 6.0
            )
            perimeter_verified = perimeter_verified or county_verified
            iteration_report["decision"] = (
                "alignment_verified_no_conservative_correction"
                if perimeter_verified
                else "insufficient_consistent_alignment_anchors"
            )
            iteration_reports.append(iteration_report)
            stop_reason = (
                "no_validated_improvement"
                if perimeter_verified
                else "insufficient_consistent_alignment_anchors"
            )
            break

        record = _correction_record(fit_matches, warped.shape[:2], native_grid)
        corrections_path = iteration_dir / "automatic-perimeter-corrections.json"
        corrections_path.write_text(json.dumps(record, indent=2) + "\n")
        native_scale = float(native_grid["height"]) / float(warped.shape[0])
        candidate_dir = iteration_dir / "candidate"
        try:
            fit_coordinates = np.asarray(
                [item["reference_pixel"] for item in fit_matches], dtype=np.float64
            )
            x_coverage = float(
                np.ptp(fit_coordinates[:, 0]) / max(warped.shape[1] - 1, 1)
            )
            y_coverage = float(
                np.ptp(fit_coordinates[:, 1]) / max(warped.shape[0] - 1, 1)
            )
            minimum_axis_coverage = min(0.35, max(0.12, min(x_coverage, y_coverage) * 0.75))
            fit_report = fit_review_corrections(
                current_alignment_path,
                corrections_path,
                candidate_dir,
                max_leave_one_out_p90_px=4.0 * native_scale,
                max_leave_one_out_max_px=8.0 * native_scale,
                minimum_axis_coverage=minimum_axis_coverage,
                minimum_hull_coverage=0.008 if county_matches else 0.05,
            )
        except ValueError as error:
            perimeter_verified = bool(
                before_boundary["count"] >= 8
                and before_boundary["median_px"] is not None
                and before_boundary["median_px"] <= 2.0
                and before_boundary["p90_px"] <= 8.0
            )
            county_verified = bool(
                before_county_reference["count"] >= 8
                and before_county_reference["median_px"] is not None
                and before_county_reference["median_px"] <= 2.0
                and before_county_reference["p90_px"] <= 6.0
            )
            perimeter_verified = perimeter_verified or county_verified
            iteration_report["decision"] = (
                "alignment_verified_no_conservative_correction"
                if perimeter_verified
                else "correction_fit_rejected"
            )
            iteration_report["fit_error"] = str(error)
            iteration_reports.append(iteration_report)
            stop_reason = (
                "no_validated_improvement"
                if perimeter_verified
                else "correction_fit_rejected"
            )
            break

        selected = next(
            item
            for item in fit_report["candidates"]
            if item["model"] == fit_report["selected_model"]
        )
        matrix_native = np.asarray(selected["matrix_current_to_target_pixels"])
        native_x = max(int(native_grid["width"]) - 1, 1)
        native_y = max(int(native_grid["height"]) - 1, 1)
        working_x = max(warped.shape[1] - 1, 1)
        working_y = max(warped.shape[0] - 1, 1)
        scale = np.asarray([native_x / working_x, native_y / working_y])
        held_current = np.asarray(
            [item["source_pixel"] for item in held_out], dtype=np.float64
        )
        held_target = np.asarray(
            [item["reference_pixel"] for item in held_out], dtype=np.float64
        )
        if len(held_out):
            predicted_native = _apply_matrix(matrix_native, held_current * scale)
            held_before = np.linalg.norm(held_current - held_target, axis=1)
            held_after = np.linalg.norm(
                predicted_native / scale - held_target, axis=1
            )
        else:
            held_before = np.asarray([], dtype=np.float64)
            held_after = np.asarray([], dtype=np.float64)
        candidate_alignment_path = candidate_dir / "alignment.json"
        candidate_alignment = json.loads(candidate_alignment_path.read_text())
        candidate_transform = _alignment_transform(candidate_alignment, state)
        after_warped, after_valid, after_grid, after_evidence = _warp_evidence(
            rgb, state, candidate_transform, actual_working_height
        )
        _, after_distance, after_nearest, after_gradient_x, after_gradient_y = after_evidence
        after_segments = _perimeter_segments(
            state,
            after_grid["bounds"],
            after_warped.shape[:2],
            candidate_anchor_count,
        )
        after_boundary = _zero_shift_boundary_metric(
            after_segments,
            after_valid,
            after_distance,
            after_nearest,
            after_gradient_x,
            after_gradient_y,
        )
        after_county_reference = _county_zero_shift_metric(
            county_segments,
            county_matches,
            after_valid,
            after_distance,
        )
        held_before_report = _residual_summary(held_before)
        held_after_report = _residual_summary(held_after)
        shift_median = float(
            np.median([item["shift_magnitude_px"] for item in fit_matches])
        )
        boundary_improvement = float(
            before_boundary["median_px"] - after_boundary["median_px"]
        )
        heldout_improvement = (
            float(held_before_report["p90_px"] - held_after_report["p90_px"])
            if len(held_out)
            else None
        )
        heldout_passes = bool(
            not len(held_out)
            or held_after_report["p90_px"] <= held_before_report["p90_px"] + 0.25
        )
        perimeter_passes = bool(
            before_boundary["count"] < 2
            or (
                after_boundary["median_px"]
                <= before_boundary["median_px"] + 0.75
                and after_boundary["p90_px"]
                <= before_boundary["p90_px"] + 1.5
            )
        )
        county_reference_passes = bool(
            before_county_reference["count"] < 4
            or (
                after_county_reference["count"] == before_county_reference["count"]
                and after_county_reference["median_px"]
                <= before_county_reference["median_px"] + 0.75
                and after_county_reference["p90_px"]
                <= before_county_reference["p90_px"] + 1.5
            )
        )
        improves = bool(
            shift_median >= 0.75
            and heldout_passes
            and perimeter_passes
            and county_reference_passes
            and (
                boundary_improvement >= 0.15
                or (len(held_out) and heldout_improvement is not None and heldout_improvement >= 0.35)
            )
        )
        iteration_report.update(
            {
                "fit": fit_report,
                "held_out_before": held_before_report,
                "held_out_after": held_after_report,
                "boundary_after": after_boundary,
                "median_proposed_shift_px": shift_median,
                "boundary_median_improvement_px": boundary_improvement,
                "held_out_p90_improvement_px": heldout_improvement,
                "perimeter_veto_passed": perimeter_passes,
                "county_reference_after": after_county_reference,
                "county_reference_veto_passed": county_reference_passes,
                "accepted": improves,
                "decision": "accepted" if improves else "no_validated_improvement",
                "candidate_alignment": str(candidate_alignment_path),
            }
        )
        _write_diagnostic(
            iteration_dir / "perimeter-after.jpg",
            after_warped,
            after_segments,
            [],
        )
        iteration_reports.append(iteration_report)
        if not improves:
            stop_reason = "no_validated_improvement"
            break
        current_alignment_path = candidate_alignment_path
        accepted_iterations += 1

    final_alignment = json.loads(current_alignment_path.read_text())
    final_path = output_dir / "alignment.json"
    final_path.write_text(json.dumps(final_alignment, indent=2) + "\n")
    graticule_metrics = initial_alignment.get("controls", {})
    result = {
        "schema_version": 1,
        "status": (
            "pass"
            if stop_reason
            in {
                "no_validated_improvement",
                "audit_only",
                "stronger_geographic_registration_preserved",
                "maximum_iterations_reached",
            }
            else "needs_attention"
        ),
        "audit_kind": "iterative_perimeter_alignment",
        "source": {
            "path": str(image_path),
            "sha256": _sha256(image_path),
            "width": int(rgb.shape[1]),
            "height": int(rgb.shape[0]),
        },
        "initial_alignment": {
            "path": str(alignment_path),
            "sha256": _sha256(alignment_path),
            "mode": initial_alignment.get("alignment_mode", "automatic"),
        },
        "final_alignment": {
            "path": str(final_path),
            "sha256": _sha256(final_path),
        },
        "accepted_iteration_count": accepted_iterations,
        "stop_reason": stop_reason,
        "preserve_geographic_registration": preserve_geographic_registration,
        "county_reference": (
            {
                "path": str(county_reference_path),
                "sha256": _sha256(county_reference_path),
            }
            if county_reference_path is not None
            else None
        ),
        "native_graticule_metrics": graticule_metrics,
        "iterations": iteration_reports,
    }
    (output_dir / "perimeter-refinement.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    return result


def auto_refine_perimeter_batch(
    config_path: Path, output_dir: Path
) -> Dict[str, object]:
    """Run the closed alignment loop across a configured source inventory."""

    config = json.loads(config_path.read_text())
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = []
    for item in config["cases"]:
        case_output = output_dir / str(item["id"])
        report = auto_refine_perimeter(
            Path(item["image"]),
            Path(item["alignment"]),
            Path(item.get("reference", config.get("reference", "reference/census-2025"))),
            case_output,
            max_iterations=int(item.get("max_iterations", config.get("max_iterations", 3))),
            working_height=int(item.get("working_height", config.get("working_height", 1600))),
            candidate_anchor_count=int(
                item.get("candidate_anchor_count", config.get("candidate_anchor_count", 16))
            ),
            fit_anchor_count=int(item.get("fit_anchor_count", config.get("fit_anchor_count", 8))),
            search_radius_px=int(item.get("search_radius_px", config.get("search_radius_px", 36))),
            tangent_radius_px=int(
                item.get("tangent_radius_px", config.get("tangent_radius_px", 12))
            ),
            preserve_geographic_registration=bool(
                item.get("preserve_geographic_registration", False)
            ),
            county_reference_path=(
                Path(item.get("county_reference", config.get("county_reference")))
                if item.get("county_reference", config.get("county_reference"))
                else None
            ),
        )
        cases.append(
            {
                "id": item["id"],
                "status": report["status"],
                "stop_reason": report["stop_reason"],
                "accepted_iteration_count": report["accepted_iteration_count"],
                "report": str(
                    (case_output / "perimeter-refinement.json").relative_to(output_dir)
                ),
                "final_alignment": str(
                    (case_output / "alignment.json").relative_to(output_dir)
                ),
            }
        )
    result = {
        "schema_version": 1,
        "audit_kind": "iterative_perimeter_alignment_batch",
        "status": "pass" if all(item["status"] == "pass" for item in cases) else "needs_attention",
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "cases": cases,
    }
    (output_dir / "perimeter-refinement-batch.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result
