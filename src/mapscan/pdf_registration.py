"""Register a vector PDF map from its native geographic graticule.

This path is intentionally separate from image-edge alignment.  When a PDF
contains vector longitude/latitude curves and degree labels, those objects are
distributed geographic control points and are much stronger evidence than
nearby cartographic or geological edges.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import pdfplumber
from PIL import Image
from pyproj import Transformer
from scipy.optimize import linear_sum_assignment
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.ops import transform as transform_geometry

from .reference import iter_boundary_lines, load_california


# Illustrator PDFs in the corpus use both the degree sign and a visually
# identical ring-above glyph in geographic labels.
DEGREE_LABEL = re.compile(r"^(\d{2,3})[\N{DEGREE SIGN}\N{MASCULINE ORDINAL INDICATOR}\N{RING ABOVE}]$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fit_affine(source: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fit a 2-by-3 affine matrix and return per-control Euclidean residuals."""

    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 2:
        raise ValueError("source and target must both have shape (n, 2)")
    if len(source) < 3:
        raise ValueError("at least three control points are required")
    design = np.column_stack((source, np.ones(len(source))))
    matrix = np.vstack(
        [np.linalg.lstsq(design, target[:, axis], rcond=None)[0] for axis in range(2)]
    )
    prediction = design @ matrix.T
    residuals = np.linalg.norm(prediction - target, axis=1)
    return matrix, residuals


def apply_affine(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a 2-by-3 affine transform to a sequence of planar points."""

    points = np.asarray(points, dtype=np.float64)
    return np.column_stack((points, np.ones(len(points)))) @ matrix.T


def _style_key(curve: Dict[str, object]) -> Tuple[object, float]:
    color = curve.get("stroking_color")
    if isinstance(color, list):
        color = tuple(color)
    return color, round(float(curve.get("linewidth") or 0.0), 4)


def _long_curve_groups(page) -> Iterable[List[LineString]]:
    """Yield modest-sized groups of long curves sharing one drawing style."""

    groups: Dict[Tuple[object, float], List[LineString]] = defaultdict(list)
    minimum_span = min(float(page.width), float(page.height)) * 0.18
    for curve in page.curves:
        points = curve.get("pts") or []
        if len(points) < 2:
            continue
        horizontal_span = float(curve["x1"]) - float(curve["x0"])
        vertical_span = float(curve["bottom"]) - float(curve["top"])
        if max(horizontal_span, vertical_span) < minimum_span:
            continue
        line = LineString([(float(x), float(y)) for x, y in points])
        if not line.is_empty:
            groups[_style_key(curve)].append(line)
    for lines in groups.values():
        if 8 <= len(lines) <= 60:
            yield lines


def _bipartition(lines: Sequence[LineString]) -> Tuple[List[int], List[int], int] | None:
    """Split an intersecting curve grid into its two non-intersecting families."""

    adjacency: List[List[int]] = [[] for _ in lines]
    intersections = 0
    for first in range(len(lines)):
        for second in range(first + 1, len(lines)):
            if lines[first].intersects(lines[second]):
                adjacency[first].append(second)
                adjacency[second].append(first)
                intersections += 1
    if intersections < len(lines):
        return None

    colors: Dict[int, int] = {}
    for start in range(len(lines)):
        if start in colors or not adjacency[start]:
            continue
        colors[start] = 0
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for neighbor in adjacency[current]:
                desired = 1 - colors[current]
                if neighbor in colors and colors[neighbor] != desired:
                    return None
                if neighbor not in colors:
                    colors[neighbor] = desired
                    queue.append(neighbor)
    first_family = [index for index, color in colors.items() if color == 0]
    second_family = [index for index, color in colors.items() if color == 1]
    if min(len(first_family), len(second_family)) < 4:
        return None
    return first_family, second_family, intersections


def _find_graticule_families(page) -> Tuple[List[LineString], List[LineString]]:
    candidates = []
    for lines in _long_curve_groups(page):
        split = _bipartition(lines)
        if split is None:
            continue
        first, second, intersections = split
        balance = abs(len(first) - len(second))
        candidates.append(
            (
                intersections - balance * 2,
                [lines[index] for index in first],
                [lines[index] for index in second],
            )
        )
    if not candidates:
        raise ValueError("no bipartite vector graticule was found in the PDF")
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1], candidates[0][2]


def _degree_labels(
    page,
) -> Tuple[
    Dict[int, List[Tuple[float, float]]],
    Dict[int, List[Tuple[float, float]]],
]:
    longitude: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
    latitude: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
    for word in page.extract_words(x_tolerance=1, y_tolerance=1):
        match = DEGREE_LABEL.fullmatch(word["text"])
        if match is None:
            continue
        raw_value = int(match.group(1))
        point = (
            (float(word["x0"]) + float(word["x1"])) / 2,
            (float(word["top"]) + float(word["bottom"])) / 2,
        )
        if 100 <= raw_value <= 180:
            longitude[-raw_value].append(point)
        elif 25 <= raw_value <= 60:
            latitude[raw_value].append(point)
    if len(longitude) < 4 or len(latitude) < 4:
        raise ValueError("the PDF graticule does not have enough readable degree labels")
    return longitude, latitude


def _family_anchor_score(
    family: Sequence[LineString], labels: Dict[int, List[Tuple[float, float]]]
) -> float:
    distances = []
    for line in family:
        distances.append(
            min(
                line.distance(Point(point))
                for points in labels.values()
                for point in points
            )
        )
    return float(np.median(distances))


def _assign_values(
    family: Sequence[LineString],
    labels: Dict[int, List[Tuple[float, float]]],
    maximum_distance: float = 100.0,
) -> Dict[int, int]:
    values = list(labels)
    inferred_values = set()
    if len(values) == len(family) - 1:
        contiguous = set(range(min(values), max(values) + 1))
        if len(contiguous) == len(family):
            inferred_values = contiguous.difference(values)
    if len(values) < len(family) and len(inferred_values) != len(family) - len(values):
        raise ValueError("there are fewer unique degree labels than graticule curves")
    costs = np.asarray(
        [
            [min(line.distance(Point(point)) for point in labels[value]) for value in values]
            for line in family
        ]
    )
    row_indices, column_indices = linear_sum_assignment(costs)
    assignments = {
        int(row): values[int(column)] for row, column in zip(row_indices, column_indices)
    }
    unassigned_rows = set(range(len(family))).difference(assignments)
    if inferred_values and len(unassigned_rows) == len(inferred_values):
        for row, value in zip(sorted(unassigned_rows), sorted(inferred_values)):
            assignments[row] = value
    assigned_distances = costs[row_indices, column_indices]
    if len(assignments) != len(family) or float(np.max(assigned_distances)) > maximum_distance:
        raise ValueError(
            "the vector curves could not be tied unambiguously to nearby degree labels; "
            f"worst assignment distance was {float(np.max(assigned_distances)):.1f} points"
        )
    return assignments


def _intersection_point(first: LineString, second: LineString) -> Tuple[float, float] | None:
    hit = first.intersection(second)
    if hit.is_empty:
        return None
    if hit.geom_type == "Point":
        return float(hit.x), float(hit.y)
    if hit.geom_type == "MultiPoint":
        point = list(hit.geoms)[0]
        return float(point.x), float(point.y)
    point = hit.centroid
    return float(point.x), float(point.y)


def extract_graticule_controls(page) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Return longitude/latitude and PDF-page control arrays from vector objects."""

    first, second = _find_graticule_families(page)
    longitude_labels, latitude_labels = _degree_labels(page)
    direct_score = _family_anchor_score(first, longitude_labels) + _family_anchor_score(
        second, latitude_labels
    )
    swapped_score = _family_anchor_score(first, latitude_labels) + _family_anchor_score(
        second, longitude_labels
    )
    meridians, parallels = (first, second) if direct_score <= swapped_score else (second, first)
    meridian_values = _assign_values(meridians, longitude_labels)
    parallel_values = _assign_values(parallels, latitude_labels)

    geographic = []
    page_points = []
    for meridian_index, meridian in enumerate(meridians):
        for parallel_index, parallel in enumerate(parallels):
            point = _intersection_point(meridian, parallel)
            if point is None:
                continue
            geographic.append(
                (meridian_values[meridian_index], parallel_values[parallel_index])
            )
            page_points.append(point)
    if len(page_points) < 12:
        raise ValueError("too few labeled graticule intersections were available")
    metadata = {
        "meridians_degrees": sorted(meridian_values.values()),
        "parallels_degrees": sorted(parallel_values.values()),
        "intersection_count": len(page_points),
    }
    return np.asarray(geographic), np.asarray(page_points), metadata


def _draw_reference_overlay(
    render_path: Path,
    output_path: Path,
    reference_root: Path,
    source_crs: str,
    matrix: np.ndarray,
) -> Tuple[int, int]:
    image = np.asarray(Image.open(render_path).convert("RGB")).copy()
    height, width = image.shape[:2]
    state, counties = load_california(reference_root)
    transformer = Transformer.from_crs("EPSG:4269", source_crs, always_xy=True)

    def page_points(coordinates) -> np.ndarray:
        coordinates = np.asarray(coordinates, dtype=np.float64)
        x, y = transformer.transform(coordinates[:, 0], coordinates[:, 1])
        return apply_affine(np.column_stack((x, y)), matrix)

    for county in counties:
        for boundary in iter_boundary_lines(county):
            points = np.rint(page_points(boundary.coords)).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(image, [points], True, (255, 40, 225), 1, cv2.LINE_AA)
    for boundary in iter_boundary_lines(state):
        points = np.rint(page_points(boundary.coords)).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [points], True, (0, 255, 255), 3, cv2.LINE_AA)
    Image.fromarray(image).save(output_path)
    return width, height


def _polygon_parts(geometry) -> Iterable[Polygon]:
    if isinstance(geometry, Polygon):
        yield geometry
    elif isinstance(geometry, MultiPolygon):
        yield from geometry.geoms
    else:
        raise TypeError(f"expected polygon geometry, received {geometry.geom_type}")


def _target_pixels(
    coordinates, minimum_x: float, maximum_y: float, resolution: float
) -> np.ndarray:
    coordinates = np.asarray(coordinates, dtype=np.float64)
    return np.column_stack(
        (
            (coordinates[:, 0] - minimum_x) / resolution,
            (maximum_y - coordinates[:, 1]) / resolution,
        )
    )


def _write_warped_inspection(
    render_path: Path,
    output_dir: Path,
    reference_root: Path,
    source_crs: str,
    target_crs: str,
    matrix: np.ndarray,
    page_width: float,
    page_height: float,
    resolution: float,
) -> Dict[str, object]:
    """Nearest-neighbor warp into a north-up map grid and clip to California."""

    source = np.asarray(Image.open(render_path).convert("RGB"))
    source_height, source_width = source.shape[:2]
    page_scale_x = source_width / page_width
    page_scale_y = source_height / page_height
    state, counties = load_california(reference_root)
    to_target = Transformer.from_crs("EPSG:4269", target_crs, always_xy=True)
    target_state = transform_geometry(to_target.transform, state)
    raw_minimum_x, raw_minimum_y, raw_maximum_x, raw_maximum_y = target_state.bounds
    minimum_x = math.floor(raw_minimum_x / resolution) * resolution
    minimum_y = math.floor(raw_minimum_y / resolution) * resolution
    maximum_x = math.ceil(raw_maximum_x / resolution) * resolution
    maximum_y = math.ceil(raw_maximum_y / resolution) * resolution
    width = int(round((maximum_x - minimum_x) / resolution))
    height = int(round((maximum_y - minimum_y) / resolution))
    warped = np.zeros((height, width, 4), dtype=np.uint8)
    target_to_source = Transformer.from_crs(target_crs, source_crs, always_xy=True)
    target_x = minimum_x + (np.arange(width, dtype=np.float64) + 0.5) * resolution
    chunk_height = 256
    for top in range(0, height, chunk_height):
        bottom = min(top + chunk_height, height)
        target_y = maximum_y - (
            np.arange(top, bottom, dtype=np.float64) + 0.5
        ) * resolution
        x_grid, y_grid = np.meshgrid(target_x, target_y)
        source_x, source_y = target_to_source.transform(x_grid, y_grid)
        page_x = (
            matrix[0, 0] * source_x + matrix[0, 1] * source_y + matrix[0, 2]
        )
        page_y = (
            matrix[1, 0] * source_x + matrix[1, 1] * source_y + matrix[1, 2]
        )
        sampled = cv2.remap(
            source,
            (page_x * page_scale_x).astype(np.float32),
            (page_y * page_scale_y).astype(np.float32),
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        warped[top:bottom, :, :3] = sampled

    mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in _polygon_parts(target_state):
        exterior = np.rint(
            _target_pixels(polygon.exterior.coords, minimum_x, maximum_y, resolution)
        ).astype(np.int32)
        cv2.fillPoly(mask, [exterior], 255)
        for interior in polygon.interiors:
            hole = np.rint(
                _target_pixels(interior.coords, minimum_x, maximum_y, resolution)
            ).astype(np.int32)
            cv2.fillPoly(mask, [hole], 0)
    warped[:, :, 3] = mask
    Image.fromarray(warped).save(output_dir / "warped-clipped.png")

    inspection = warped[:, :, :3].copy()
    inspection[mask == 0] = (245, 245, 245)
    for county in counties:
        target_county = transform_geometry(to_target.transform, county)
        for boundary in iter_boundary_lines(target_county):
            points = np.rint(
                _target_pixels(boundary.coords, minimum_x, maximum_y, resolution)
            ).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(inspection, [points], True, (255, 40, 225), 1, cv2.LINE_AA)
    for boundary in iter_boundary_lines(target_state):
        points = np.rint(
            _target_pixels(boundary.coords, minimum_x, maximum_y, resolution)
        ).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(inspection, [points], True, (0, 255, 255), 2, cv2.LINE_AA)
    Image.fromarray(inspection).save(output_dir / "warped-inspection.png")
    return {
        "crs": target_crs,
        "resolution_crs_units_per_pixel": resolution,
        "bounds": [minimum_x, minimum_y, maximum_x, maximum_y],
        "width": width,
        "height": height,
        "resampling": "nearest",
        "clip": "2025 Census California state geometry",
        "clipped_path": str(output_dir / "warped-clipped.png"),
        "inspection_path": str(output_dir / "warped-inspection.png"),
    }


def align_pdf_graticule(
    pdf_path: Path,
    reference_root: Path,
    output_dir: Path,
    render_path: Path | None = None,
    projection_crs: str = "EPSG:3310",
    page_number: int = 1,
    warp_crs: str | None = None,
    warp_resolution: float = 250.0,
) -> Dict[str, object]:
    """Fit a declared projection to a PDF's native vector graticule."""

    output_dir.mkdir(parents=True, exist_ok=True)
    with pdfplumber.open(pdf_path) as document:
        if page_number < 1 or page_number > len(document.pages):
            raise ValueError(f"page {page_number} is outside this {len(document.pages)}-page PDF")
        page = document.pages[page_number - 1]
        geographic, page_controls, control_metadata = extract_graticule_controls(page)
        transformer = Transformer.from_crs("EPSG:4269", projection_crs, always_xy=True)
        projected_x, projected_y = transformer.transform(geographic[:, 0], geographic[:, 1])
        projected = np.column_stack((projected_x, projected_y))
        matrix, residuals = fit_affine(projected, page_controls)
        singular_values = np.linalg.svd(matrix[:, :2])[1]
        metrics = {
            "rms_page_point": float(math.sqrt(np.mean(residuals**2))),
            "median_page_point": float(np.median(residuals)),
            "p90_page_point": float(np.percentile(residuals, 90)),
            "max_page_point": float(np.max(residuals)),
            "affine_scale_anisotropy": float(singular_values[0] / singular_values[1]),
        }
        report: Dict[str, object] = {
            "schema_version": 1,
            "status": "diagnostic_only",
            "alignment_mode": "native_pdf_graticule",
            "source": {
                "pdf_path": str(pdf_path),
                "sha256": _sha256(pdf_path),
                "page_number": page_number,
                "page_width_points": float(page.width),
                "page_height_points": float(page.height),
            },
            "projection": {"crs": projection_crs},
            "controls": {**control_metadata, **metrics},
            "projected_crs_to_page_affine": matrix.tolist(),
            "warning": (
                "The transform is validated against native graticule controls. "
                "Boundary differences can still reflect map generalization, data vintage, "
                "and line thickness rather than registration drift."
            ),
        }
        if render_path is not None:
            width, height = _draw_reference_overlay(
                render_path,
                output_dir / "source-with-reference.png",
                reference_root,
                projection_crs,
                matrix,
            )
            report["source"]["render_path"] = str(render_path)  # type: ignore[index]
            report["source"]["render_width_px"] = width  # type: ignore[index]
            report["source"]["render_height_px"] = height  # type: ignore[index]
            report["source"]["render_x_px_per_page_point"] = width / float(page.width)  # type: ignore[index]
            report["source"]["render_y_px_per_page_point"] = height / float(page.height)  # type: ignore[index]
            if warp_crs is not None:
                report["warped_inspection"] = _write_warped_inspection(
                    render_path,
                    output_dir,
                    reference_root,
                    projection_crs,
                    warp_crs,
                    matrix,
                    float(page.width),
                    float(page.height),
                    warp_resolution,
                )
        (output_dir / "alignment.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
