"""Automatic California-outline registration experiment."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import matplotlib
import numpy as np
from pyproj import Transformer
from scipy.ndimage import distance_transform_edt
from scipy.optimize import differential_evolution, minimize
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import transform as transform_geometry

from .reference import iter_boundary_lines, load_california
from .vision import PreparedImage, prepare_image

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402  pylint: disable=wrong-import-position


PROJECTIONS: Dict[str, str] = {
    "california_albers": "EPSG:3310",
    "web_mercator": "EPSG:3857",
    "wgs84": "EPSG:4326",
    "conus_albers": "EPSG:5070",
}


@dataclass(frozen=True)
class FitParameters:
    center_x_fraction: float
    center_y_fraction: float
    state_height_fraction: float
    x_scale_ratio: float
    rotation_degrees: float
    x_shear: float


@dataclass(frozen=True)
class CandidateResult:
    projection: str
    projection_crs: str
    evidence_model: str
    coverage_model: str
    transform_model: str
    objective: float
    parameters: FitParameters
    metrics: Dict[str, float]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _largest_polygon(geometry):
    if isinstance(geometry, Polygon):
        return geometry
    if isinstance(geometry, MultiPolygon):
        return max(geometry.geoms, key=lambda item: item.area)
    raise TypeError(f"Expected polygonal state geometry, received {geometry.geom_type}")


def _resample_closed_line(coords: np.ndarray, count: int) -> np.ndarray:
    deltas = np.diff(coords, axis=0)
    lengths = np.linalg.norm(deltas, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    if cumulative[-1] == 0:
        return np.repeat(coords[:1], count, axis=0)
    targets = np.linspace(0, cumulative[-1], count, endpoint=False)
    x = np.interp(targets, cumulative, coords[:, 0])
    y = np.interp(targets, cumulative, coords[:, 1])
    return np.column_stack((x, y))


def _reference_points(state, crs: str, count: int = 1200) -> np.ndarray:
    transformer = Transformer.from_crs("EPSG:4269", crs, always_xy=True)
    projected = transform_geometry(transformer.transform, state)
    mainland = _largest_polygon(projected)
    coords = np.asarray(mainland.exterior.coords, dtype=np.float64)
    points = _resample_closed_line(coords, count)
    min_x, min_y, max_x, max_y = mainland.bounds
    state_height = max_y - min_y
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    normalized = np.empty_like(points)
    normalized[:, 0] = (points[:, 0] - center_x) / state_height
    normalized[:, 1] = (center_y - points[:, 1]) / state_height
    return normalized


def _eastern_border_points(state, crs: str, count: int = 420) -> np.ndarray:
    """Return the straight California–Nevada border from the northeast to the AZ tripoint."""

    mainland_source = _largest_polygon(state)
    coordinates = np.asarray(mainland_source.exterior.coords, dtype=np.float64)
    northeast = np.array([-120.0, 42.0])
    arizona_tripoint = np.array([-114.63, 35.0])
    northeast_index = int(np.argmin(np.linalg.norm(coordinates - northeast, axis=1)))
    tripoint_index = int(np.argmin(np.linalg.norm(coordinates - arizona_tripoint, axis=1)))

    first, second = sorted((northeast_index, tripoint_index))
    direct = coordinates[first : second + 1]
    wrapped = np.vstack((coordinates[second:], coordinates[: first + 1]))
    # The eastern path has the larger mean longitude (less negative).
    source_segment = direct if np.mean(direct[:, 0]) > np.mean(wrapped[:, 0]) else wrapped

    transformer = Transformer.from_crs("EPSG:4269", crs, always_xy=True)
    projected_state = transform_geometry(transformer.transform, state)
    projected_mainland = _largest_polygon(projected_state)
    projected_segment = np.column_stack(transformer.transform(
        source_segment[:, 0], source_segment[:, 1]
    ))
    sampled = _resample_closed_line(projected_segment, count)
    min_x, min_y, max_x, max_y = projected_mainland.bounds
    state_height = max_y - min_y
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    sampled[:, 0] = (sampled[:, 0] - center_x) / state_height
    sampled[:, 1] = (center_y - sampled[:, 1]) / state_height
    return sampled


def _eastern_border_hinge_point(state, crs: str) -> np.ndarray:
    """Return the normalized Lake Tahoe bend in the California-Nevada border."""

    mainland_source = _largest_polygon(state)
    coordinates = np.asarray(mainland_source.exterior.coords, dtype=np.float64)
    # The meridional boundary turns southeast at approximately 39 N, 120 W.
    source_hinge = coordinates[
        int(np.argmin(np.linalg.norm(coordinates - np.array([-120.0, 39.0]), axis=1)))
    ]
    transformer = Transformer.from_crs("EPSG:4269", crs, always_xy=True)
    projected_state = transform_geometry(transformer.transform, state)
    projected_mainland = _largest_polygon(projected_state)
    hinge_x, hinge_y = transformer.transform(source_hinge[0], source_hinge[1])
    min_x, min_y, max_x, max_y = projected_mainland.bounds
    state_height = max_y - min_y
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    return np.array(
        [[(hinge_x - center_x) / state_height, (center_y - hinge_y) / state_height]],
        dtype=np.float64,
    )


def _county_points(counties, crs: str, per_line: int = 35) -> np.ndarray:
    transformer = Transformer.from_crs("EPSG:4269", crs, always_xy=True)
    state_union = counties[0]
    for county in counties[1:]:
        state_union = state_union.union(county)
    projected_state = transform_geometry(transformer.transform, state_union)
    mainland = _largest_polygon(projected_state)
    min_x, min_y, max_x, max_y = mainland.bounds
    state_height = max_y - min_y
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    chunks: List[np.ndarray] = []
    for county in counties:
        projected = transform_geometry(transformer.transform, county)
        for line in iter_boundary_lines(projected):
            coords = np.asarray(line.coords, dtype=np.float64)
            if len(coords) < 2:
                continue
            sampled = _resample_closed_line(coords, per_line)
            sampled[:, 0] = (sampled[:, 0] - center_x) / state_height
            sampled[:, 1] = (center_y - sampled[:, 1]) / state_height
            chunks.append(sampled)
    return np.concatenate(chunks) if chunks else np.empty((0, 2))


def transform_points(
    normalized: np.ndarray, parameters: Sequence[float], image_shape: Tuple[int, int]
) -> np.ndarray:
    """Map normalized projected reference coordinates into source-image pixels."""

    height, width = image_shape
    cx, cy, state_height, x_ratio, rotation_degrees, shear = parameters
    theta = math.radians(rotation_degrees)
    cosine, sine = math.cos(theta), math.sin(theta)
    u, v = normalized[:, 0], normalized[:, 1]
    rotated_x = cosine * u - sine * v
    rotated_y = sine * u + cosine * v
    scale = state_height * height
    x = cx * width + scale * (x_ratio * rotated_x + shear * rotated_y)
    y = cy * height + scale * rotated_y
    return np.column_stack((x, y))


def _sample_distance(distance: np.ndarray, points: np.ndarray, miss_penalty: float) -> np.ndarray:
    height, width = distance.shape
    x = np.rint(points[:, 0]).astype(int)
    y = np.rint(points[:, 1]).astype(int)
    inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    values = np.full(len(points), miss_penalty, dtype=np.float64)
    values[inside] = distance[y[inside], x[inside]]
    return values


def _directional_distance(
    prepared: PreparedImage, points: np.ndarray, miss_penalty: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Measure edge distance and agreement between edge and outline direction."""

    height, width = prepared.distance.shape
    rounded_x = np.rint(points[:, 0]).astype(int)
    rounded_y = np.rint(points[:, 1]).astype(int)
    inside = (
        (rounded_x >= 0) & (rounded_x < width) & (rounded_y >= 0) & (rounded_y < height)
    )
    distance = np.full(len(points), miss_penalty, dtype=np.float64)
    direction = np.zeros(len(points), dtype=np.float64)
    if not np.any(inside):
        return distance, direction

    x = rounded_x[inside]
    y = rounded_y[inside]
    distance[inside] = prepared.distance[y, x]
    nearest_y = prepared.nearest_edge_y[y, x]
    nearest_x = prepared.nearest_edge_x[y, x]

    previous = np.roll(points, 1, axis=0)
    following = np.roll(points, -1, axis=0)
    tangent = following - previous
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    norm = np.linalg.norm(normal, axis=1)
    normal = np.divide(normal, norm[:, None], out=np.zeros_like(normal), where=norm[:, None] > 0)
    gradient_x = prepared.gradient_x[nearest_y, nearest_x]
    gradient_y = prepared.gradient_y[nearest_y, nearest_x]
    direction[inside] = np.abs(
        normal[inside, 0] * gradient_x + normal[inside, 1] * gradient_y
    )
    return distance, direction


def _color_evidence_grid(prepared: PreparedImage, size: int = 160) -> np.ndarray:
    return cv2.resize(
        prepared.color_evidence.astype(np.float32),
        (size, size),
        interpolation=cv2.INTER_AREA,
    )


def _color_containment(
    transformed_boundary: np.ndarray,
    image_shape: Tuple[int, int],
    color_grid: np.ndarray,
) -> float:
    """Return the fraction of data-like color evidence inside the candidate state."""

    total = float(color_grid.sum())
    if total <= 1e-6:
        return 0.5
    image_height, image_width = image_shape
    grid_height, grid_width = color_grid.shape
    polygon = transformed_boundary.copy()
    polygon[:, 0] *= grid_width / image_width
    polygon[:, 1] *= grid_height / image_height
    polygon = np.rint(polygon).astype(np.int32).reshape((-1, 1, 2))
    mask = np.zeros_like(color_grid, dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 1)
    return float((color_grid * mask).sum() / total)


def _fit_projection(
    prepared: PreparedImage,
    boundary: np.ndarray,
    eastern_boundary: np.ndarray,
    eastern_hinge: np.ndarray,
    counties: np.ndarray,
    projection_name: str,
    projection_crs: str,
    seed: int,
    county_weight: float = 0.0,
    partial_coverage: bool = False,
    transform_model: str = "affine_like",
) -> CandidateResult:
    height, width = prepared.distance.shape
    diagonal = math.hypot(width, height)
    train = boundary[np.arange(len(boundary)) % 5 != 0]
    holdout = boundary[np.arange(len(boundary)) % 5 == 0]
    county_train = counties[:: max(1, math.ceil(len(counties) / 1500))]
    miss_penalty = diagonal * 0.12
    cap = diagonal * 0.035
    color_grid = _color_evidence_grid(prepared)

    def expand_parameters(parameters: Sequence[float]) -> Sequence[float]:
        if transform_model == "similarity":
            center_x, center_y, state_height, rotation = parameters
            return (center_x, center_y, state_height, 1.0, rotation, 0.0)
        if transform_model == "affine_like":
            return parameters
        raise ValueError(f"Unknown transform model: {transform_model}")

    def visible_mask(points: np.ndarray) -> np.ndarray:
        return (
            (points[:, 0] >= 0)
            & (points[:, 0] < width)
            & (points[:, 1] >= 0)
            & (points[:, 1] < height)
        )

    def objective(parameters: Sequence[float]) -> float:
        model_parameters = expand_parameters(parameters)
        points = transform_points(train, model_parameters, (height, width))
        values, direction = _directional_distance(prepared, points, miss_penalty)
        visibility_penalty = 0.0
        if partial_coverage:
            visible = visible_mask(points)
            visible_fraction = float(np.mean(visible))
            if np.count_nonzero(visible) < 48:
                return 0.08 + (48 - np.count_nonzero(visible)) * 0.001
            visible_points = points[visible]
            x_span = float(np.ptp(visible_points[:, 0]) / width)
            y_span = float(np.ptp(visible_points[:, 1]) / height)
            # A partial map may show only one useful border arc, but that arc must
            # still carry meaningful spatial extent. Otherwise the optimizer can
            # hide nearly all of California off-canvas and fit a tiny accidental edge.
            visibility_penalty += max(0.0, 0.14 - visible_fraction) * 0.035
            visibility_penalty += max(0.0, 0.28 - max(x_span, y_span)) * 0.025
            values = values[visible]
            direction = direction[visible]
        directed_values = values + (1.0 - direction) * 4.0
        # A capped mean is robust to borders hidden by legends and labels. The
        # uncapped p75 term prevents a fit from explaining only a short segment.
        robust = np.mean(np.minimum(directed_values, cap))
        coverage = np.percentile(directed_values, 75)
        edge_score = (0.72 * robust + 0.28 * min(coverage, miss_penalty)) / diagonal
        if partial_coverage and np.any(prepared.eastern_straight_edges):
            transformed_east = transform_points(
                eastern_boundary, model_parameters, (height, width)
            )
            visible_east = visible_mask(transformed_east)
            if np.count_nonzero(visible_east) < 36:
                return 0.055 + (36 - np.count_nonzero(visible_east)) * 0.0005
            east_distance = _sample_distance(
                prepared.eastern_straight_distance,
                transformed_east[visible_east],
                miss_penalty,
            )
            east_robust = float(np.mean(np.minimum(east_distance, cap)))
            east_coverage = min(float(np.percentile(east_distance, 75)), miss_penalty)
            east_score = (0.72 * east_robust + 0.28 * east_coverage) / diagonal
            if prepared.eastern_border_hinge is not None:
                transformed_hinge = transform_points(
                    eastern_hinge, model_parameters, (height, width)
                )[0]
                hinge_distance = float(
                    np.linalg.norm(
                        transformed_hinge - np.asarray(prepared.eastern_border_hinge)
                    )
                )
                hinge_score = min(hinge_distance, cap) / diagonal
                east_score = 0.76 * east_score + 0.24 * hinge_score
            edge_score = 0.52 * edge_score + 0.48 * east_score
        if county_weight > 0 and len(county_train):
            transformed_counties = transform_points(
                county_train, model_parameters, (height, width)
            )
            county_distance = _sample_distance(
                prepared.distance, transformed_counties, miss_penalty
            )
            if partial_coverage:
                visible_counties = visible_mask(transformed_counties)
                if np.count_nonzero(visible_counties) >= 30:
                    county_distance = county_distance[visible_counties]
            county_robust = np.mean(np.minimum(county_distance, cap))
            county_coverage = min(float(np.percentile(county_distance, 75)), miss_penalty)
            county_score = (0.72 * county_robust + 0.28 * county_coverage) / diagonal
            edge_score = (1.0 - county_weight) * edge_score + county_weight * county_score
        containment = _color_containment(
            transform_points(boundary, model_parameters, (height, width)),
            (height, width),
            color_grid,
        )
        # A wrong fit can score well against dense hillshade or crop edges. Color
        # containment supplies the missing global check without assigning pixels
        # to legend categories yet.
        model_penalty = 0.0008 if partial_coverage else 0.0
        return float(
            edge_score
            + 0.010 * (1.0 - containment)
            + visibility_penalty
            + model_penalty
        )

    if partial_coverage:
        bounds = [
            (-0.8, 1.8),  # state center may lie well beyond a partial image
            (-0.8, 1.8),
            (0.65, 3.2),  # full state can be much larger than the image crop
            (0.65, 1.35),
            (-20.0, 20.0),
            (-0.35, 0.35),
        ]
    else:
        bounds = [
            (-0.18, 1.18),
            (-0.12, 1.12),
            (0.48, 1.75),
            (0.72, 1.28),
            (-13.0, 13.0),
            (-0.22, 0.22),
        ]
    if transform_model == "similarity":
        # Translation, uniform scale, and rotation are sufficient when the
        # source declares the same cartographic projection as the reference.
        bounds = [bounds[0], bounds[1], bounds[2], bounds[4]]
    global_fit = differential_evolution(
        objective,
        bounds,
        seed=seed,
        popsize=11,
        maxiter=55,
        polish=False,
        workers=1,
        updating="immediate",
    )
    local_fit = minimize(
        objective,
        global_fit.x,
        method="Nelder-Mead",
        options={"maxiter": 700, "xatol": 1e-5, "fatol": 1e-7},
    )
    optimized_parameters = local_fit.x if local_fit.fun <= global_fit.fun else global_fit.x
    parameters = expand_parameters(optimized_parameters)
    holdout_points = transform_points(holdout, parameters, (height, width))
    values, direction = _directional_distance(prepared, holdout_points, miss_penalty)
    visible_holdout = visible_mask(holdout_points)
    if partial_coverage:
        values = values[visible_holdout]
        direction = direction[visible_holdout]
    containment = _color_containment(
        transform_points(boundary, parameters, (height, width)),
        (height, width),
        color_grid,
    )
    metrics = {
        "holdout_median_px_at_working_resolution": float(np.median(values)),
        "holdout_p90_px_at_working_resolution": float(np.percentile(values, 90)),
        "holdout_mean_px_at_working_resolution": float(np.mean(values)),
        "holdout_within_3px_fraction": float(np.mean(values <= 3)),
        "holdout_within_5px_fraction": float(np.mean(values <= 5)),
        "holdout_within_10px_fraction": float(np.mean(values <= 10)),
        "holdout_median_direction_agreement": float(np.median(direction)),
        "data_color_containment_fraction": containment,
        "visible_reference_fraction": float(
            np.mean(visible_mask(transform_points(boundary, parameters, (height, width))))
        ),
        "visible_holdout_sample_count": float(len(values)),
        "working_width": float(width),
        "working_height": float(height),
    }
    factor = 1.0 / prepared.scale_from_original
    metrics["holdout_median_px_at_source_resolution"] = metrics[
        "holdout_median_px_at_working_resolution"
    ] * factor
    metrics["holdout_p90_px_at_source_resolution"] = metrics[
        "holdout_p90_px_at_working_resolution"
    ] * factor
    if len(county_train):
        county_distance = _sample_distance(
            prepared.distance,
            transform_points(county_train, parameters, (height, width)),
            miss_penalty,
        )
        if partial_coverage:
            transformed_counties = transform_points(
                county_train, parameters, (height, width)
            )
            visible_counties = visible_mask(transformed_counties)
            if np.count_nonzero(visible_counties):
                county_distance = county_distance[visible_counties]
        metrics["county_median_px_at_working_resolution"] = float(
            np.median(county_distance)
        )
        metrics["county_within_5px_fraction"] = float(np.mean(county_distance <= 5))
    transformed_east = transform_points(eastern_boundary, parameters, (height, width))
    visible_east = visible_mask(transformed_east)
    metrics["visible_eastern_border_fraction"] = float(np.mean(visible_east))
    if np.count_nonzero(visible_east) and np.any(prepared.eastern_straight_edges):
        east_distance = _sample_distance(
            prepared.eastern_straight_distance,
            transformed_east[visible_east],
            miss_penalty,
        )
        metrics["eastern_border_median_straight_line_residual_px"] = float(
            np.median(east_distance)
        )
    if prepared.eastern_border_hinge is not None:
        transformed_hinge = transform_points(
            eastern_hinge, parameters, (height, width)
        )[0]
        metrics["eastern_border_hinge_residual_px"] = float(
            np.linalg.norm(transformed_hinge - np.asarray(prepared.eastern_border_hinge))
        )
    return CandidateResult(
        projection=projection_name,
        projection_crs=projection_crs,
        evidence_model="outline_counties" if county_weight else "outline",
        coverage_model="partial_state" if partial_coverage else "full_or_most_state",
        transform_model=transform_model,
        objective=float(objective(optimized_parameters)),
        parameters=FitParameters(*map(float, parameters)),
        metrics=metrics,
    )


def _nearest_edge_points(edges: np.ndarray, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    _, indices = distance_transform_edt(~edges, return_indices=True)
    height, width = edges.shape
    x = np.clip(np.rint(points[:, 0]).astype(int), 0, width - 1)
    y = np.clip(np.rint(points[:, 1]).astype(int), 0, height - 1)
    nearest = np.column_stack((indices[1, y, x], indices[0, y, x]))
    residual = np.linalg.norm(nearest - points, axis=1)
    return nearest, residual


def _write_diagnostics(
    output_dir: Path,
    prepared: PreparedImage,
    boundary: np.ndarray,
    eastern_boundary: np.ndarray,
    county_points: np.ndarray,
    best: CandidateResult,
) -> None:
    params = list(asdict(best.parameters).values())
    image_shape = prepared.distance.shape
    projected = transform_points(boundary, params, image_shape)
    projected_east = transform_points(eastern_boundary, params, image_shape)
    projected_counties = transform_points(county_points, params, image_shape)
    sampled = projected[::12]
    nearest, residual = _nearest_edge_points(prepared.edges, sampled)

    fig, axes = plt.subplots(1, 2, figsize=(16, 9), constrained_layout=True)
    axes[0].imshow(prepared.rgb)
    axes[0].plot(projected[:, 0], projected[:, 1], color="#00ffff", linewidth=1.5)
    axes[0].plot(
        projected_east[:, 0], projected_east[:, 1], color="#ff7a00", linewidth=2.2
    )
    if prepared.eastern_border_hinge is not None:
        axes[0].scatter(
            [prepared.eastern_border_hinge[0]],
            [prepared.eastern_border_hinge[1]],
            s=44,
            facecolors="none",
            edgecolors="#ff3bf5",
            linewidths=1.8,
        )
    for source, target, error in zip(sampled, nearest, residual):
        if error <= 18:
            color = "#39ff14" if error <= 5 else "#ffb000"
            axes[0].plot(
                [source[0], target[0]], [source[1], target[1]], color=color, linewidth=0.55
            )
    axes[0].set_title(
        f"Reference outline on source ({best.projection}, {best.coverage_model})\n"
        "cyan=reference, orange=eastern border, magenta ring=detected hinge"
    )
    axes[0].set_xlim(0, prepared.rgb.shape[1])
    axes[0].set_ylim(prepared.rgb.shape[0], 0)
    axes[0].set_axis_off()

    axes[1].imshow(prepared.edges, cmap="gray")
    axes[1].plot(projected[:, 0], projected[:, 1], color="#00ffff", linewidth=1.2)
    axes[1].plot(
        projected_east[:, 0], projected_east[:, 1], color="#ff7a00", linewidth=2.0
    )
    if len(projected_counties):
        axes[1].scatter(
            projected_counties[::6, 0],
            projected_counties[::6, 1],
            s=0.35,
            color="#ff3bf5",
            alpha=0.55,
        )
    axes[1].set_title("Edge evidence (cyan=state, magenta=county validation)")
    axes[1].set_xlim(0, prepared.rgb.shape[1])
    axes[1].set_ylim(prepared.rgb.shape[0], 0)
    axes[1].set_axis_off()
    fig.savefig(output_dir / "alignment-diagnostic.png", dpi=170)
    plt.close(fig)

    overlay = prepared.rgb.copy()
    polyline = np.rint(projected).astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(overlay, [polyline], True, (0, 255, 255), 2, cv2.LINE_AA)
    eastern_polyline = np.rint(projected_east).astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(overlay, [eastern_polyline], False, (255, 122, 0), 3, cv2.LINE_AA)
    if prepared.eastern_border_hinge is not None:
        cv2.circle(
            overlay,
            tuple(np.rint(prepared.eastern_border_hinge).astype(int)),
            7,
            (255, 59, 245),
            2,
            cv2.LINE_AA,
        )
    Image.fromarray(overlay).save(output_dir / "source-with-reference.png")
    Image.fromarray((prepared.edges.astype(np.uint8) * 255)).save(output_dir / "detected-edges.png")
    Image.fromarray((prepared.color_evidence.astype(np.uint8) * 255)).save(
        output_dir / "color-evidence.png"
    )
    Image.fromarray((prepared.eastern_straight_edges.astype(np.uint8) * 255)).save(
        output_dir / "eastern-straight-line-evidence.png"
    )


def align_image(
    image_path: Path,
    reference_root: Path,
    output_dir: Path,
    max_dimension: int = 900,
    seed: int = 42,
    projection_names: Sequence[str] | None = None,
    coverage_models: Sequence[str] | None = None,
    transform_models: Sequence[str] | None = None,
) -> Dict[str, object]:
    """Fit California reference geometry to a source image and write diagnostics."""

    output_dir.mkdir(parents=True, exist_ok=True)
    state, counties = load_california(reference_root)
    prepared = prepare_image(image_path, max_dimension=max_dimension)
    candidates: List[CandidateResult] = []
    references: Dict[str, np.ndarray] = {}
    county_references: Dict[str, np.ndarray] = {}
    eastern_references: Dict[str, np.ndarray] = {}
    eastern_hinges: Dict[str, np.ndarray] = {}
    selected_projections = (
        list(PROJECTIONS) if projection_names is None else list(projection_names)
    )
    selected_coverage = (
        ["full_or_most_state", "partial_state"]
        if coverage_models is None
        else list(coverage_models)
    )
    selected_transforms = (
        ["affine_like"] if transform_models is None else list(transform_models)
    )
    for index, name in enumerate(selected_projections):
        crs = PROJECTIONS[name]
        boundary = _reference_points(state, crs)
        references[name] = boundary
        eastern_references[name] = _eastern_border_points(state, crs)
        eastern_hinges[name] = _eastern_border_hinge_point(state, crs)
        county_references[name] = _county_points(counties, crs)
        for coverage_index, coverage_model in enumerate(selected_coverage):
            partial_coverage = coverage_model == "partial_state"
            for transform_index, transform_model in enumerate(selected_transforms):
                for model_index, county_weight in enumerate((0.0, 0.32)):
                    candidates.append(
                        _fit_projection(
                            prepared,
                            boundary,
                            eastern_references[name],
                            eastern_hinges[name],
                            county_references[name],
                            name,
                            crs,
                            seed=(
                                seed
                                + index * 1009
                                + model_index * 7919
                                + coverage_index * 15401
                                + transform_index * 23447
                            ),
                            county_weight=county_weight,
                            partial_coverage=partial_coverage,
                            transform_model=transform_model,
                        )
                    )
    candidates.sort(key=lambda candidate: candidate.objective)
    best = candidates[0]
    _write_diagnostics(
        output_dir,
        prepared,
        references[best.projection],
        eastern_references[best.projection],
        county_references[best.projection],
        best,
    )
    best_by_coverage = {}
    for coverage_model in selected_coverage:
        candidate = next(
            item for item in candidates if item.coverage_model == coverage_model
        )
        best_by_coverage[coverage_model] = asdict(candidate)
        diagnostic_dir = output_dir / f"best-{coverage_model}"
        diagnostic_dir.mkdir(parents=True, exist_ok=True)
        _write_diagnostics(
            diagnostic_dir,
            prepared,
            references[candidate.projection],
            eastern_references[candidate.projection],
            county_references[candidate.projection],
            candidate,
        )
    report = {
        "schema_version": 1,
        "status": "diagnostic_only",
        "alignment_mode": "automatic",
        "source": {
            "path": str(image_path),
            "sha256": _sha256(image_path),
            "width": prepared.original_width,
            "height": prepared.original_height,
        },
        "reference": {
            "vintage": "2025",
            "source_crs": "EPSG:4269",
            "root": str(reference_root),
        },
        "best": asdict(best),
        "best_by_coverage": best_by_coverage,
        "candidates": [asdict(candidate) for candidate in candidates],
        "warning": (
            "This first-stage score measures proximity to generic image edges. "
            "It is not yet an acceptance decision and must be visually inspected."
        ),
    }
    (output_dir / "alignment.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


# Pillow is imported late so Matplotlib's backend is fixed before any renderer imports.
from PIL import Image  # noqa: E402  pylint: disable=wrong-import-position
