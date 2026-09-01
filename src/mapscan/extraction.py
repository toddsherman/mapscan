"""Plan-driven classified-raster extraction and Web-Mercator warping.

The per-source plan is the semantic output of legend interpretation.  Pixel
classification and geographic warping remain deterministic and reproducible.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image
from pyproj import Transformer
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import transform as transform_geometry

from .canonical_clip import (
    active_boundary_publication_interior,
    canonical_publication_interior,
    close_west_coast_clipping_seam,
    snap_named_water_to_active_boundary,
)
from .reference import load_california
from .water_reference import rasterize_california_areawater
from .mapbox_water_reference import constrain_named_water_to_mapbox


DEFAULT_REGISTERED_COUNTY_REFERENCE = Path(
    "runs/county-reference-v2/county-reference.json"
)


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
    else:  # pragma: no cover - Census state geometry is polygonal.
        raise TypeError(f"Expected polygonal geometry, received {geometry.geom_type}")


def _largest_polygon(geometry) -> Polygon:
    return max(_polygons(geometry), key=lambda item: item.area)


def _projection_normalizer(state, crs: str):
    transformer = Transformer.from_crs("EPSG:4269", crs, always_xy=True)
    projected = transform_geometry(transformer.transform, state)
    mainland = _largest_polygon(projected)
    min_x, min_y, max_x, max_y = mainland.bounds
    state_height = max_y - min_y
    return projected, (min_x + max_x) / 2, (min_y + max_y) / 2, state_height


def _normalized_to_source(
    normalized: np.ndarray,
    parameters: Dict[str, float],
    shape: Tuple[int, int],
) -> np.ndarray:
    height, width = shape
    theta = math.radians(parameters["rotation_degrees"])
    cosine, sine = math.cos(theta), math.sin(theta)
    u, v = normalized[:, 0], normalized[:, 1]
    rotated_x = cosine * u - sine * v
    rotated_y = sine * u + cosine * v
    scale = parameters["state_height_fraction"] * height
    x = parameters["center_x_fraction"] * width + scale * (
        parameters["x_scale_ratio"] * rotated_x
        + parameters["x_shear"] * rotated_y
    )
    y = parameters["center_y_fraction"] * height + scale * rotated_y
    return np.column_stack((x, y))


def _transform_normalized_to_source(
    normalized: np.ndarray,
    transform: Dict[str, object],
    shape: Tuple[int, int],
) -> np.ndarray:
    if "reference_to_source_matrix" in transform:
        matrix = np.asarray(transform["reference_to_source_matrix"], dtype=np.float64)
        if matrix.shape != (3, 3):
            raise ValueError("Assisted reference-to-source matrix must be 3 by 3")
        return cv2.perspectiveTransform(
            normalized.reshape((-1, 1, 2)).astype(np.float64), matrix
        ).reshape((-1, 2))
    return _normalized_to_source(normalized, transform["parameters"], shape)


def _correct_target_web_coordinates(
    web_x: np.ndarray,
    web_y: np.ndarray,
    transform: Dict[str, object],
) -> Tuple[np.ndarray, np.ndarray]:
    """Map desired output locations back into the pre-correction Web-Mercator grid."""

    correction = transform.get("web_mercator_correction")
    if not isinstance(correction, dict):
        return web_x, web_y
    bounds = np.asarray(correction["grid"]["bounds"], dtype=np.float64)
    min_x, min_y, max_x, max_y = bounds
    normalized = np.column_stack(
        (
            (np.asarray(web_x).ravel() - min_x) / (max_x - min_x),
            (max_y - np.asarray(web_y).ravel()) / (max_y - min_y),
        )
    )
    operations = correction.get("operations")
    if not isinstance(operations, list):
        operations = [
            {
                "type": "matrix",
                "matrix": correction["target_to_current_normalized_matrix"],
            }
        ]
    corrected = normalized
    for operation in operations:
        if operation.get("type") == "matrix":
            matrix = np.asarray(operation["matrix"], dtype=np.float64)
            if matrix.shape != (3, 3):
                raise ValueError("Review correction matrix must be 3 by 3")
            corrected = cv2.perspectiveTransform(
                corrected.reshape((-1, 1, 2)), matrix
            ).reshape((-1, 2))
            continue
        if operation.get("type") == "lower_colorado_smoothstep_x":
            operation_grid = operation["grid"]
            width = max(int(operation_grid["width"]) - 1, 1)
            height = max(int(operation_grid["height"]) - 1, 1)
            pixel_x = corrected[:, 0] * width
            pixel_y = corrected[:, 1] * height

            def smoothstep(values: np.ndarray) -> np.ndarray:
                clipped = np.clip(values, 0.0, 1.0)
                return clipped * clipped * (3.0 - 2.0 * clipped)

            x_weight = smoothstep(
                (pixel_x - float(operation["start_x_px"]))
                / float(operation["ramp_width_px"])
            )
            y_weight = smoothstep(
                (pixel_y - float(operation["start_y_px"]))
                / float(operation["ramp_height_px"])
            )
            sampling_amplitude = float(
                operation["target_to_parent_sampling_amplitude_x_px"]
            )
            corrected = np.column_stack(
                (
                    (pixel_x + sampling_amplitude * x_weight * y_weight)
                    / width,
                    pixel_y / height,
                )
            )
            continue
        if operation.get("type") != "compact_wendland_c2_displacement":
            raise ValueError(f"Unsupported review correction operation: {operation.get('type')}")
        operation_grid = operation["grid"]
        width = max(int(operation_grid["width"]) - 1, 1)
        height = max(int(operation_grid["height"]) - 1, 1)
        pixel_x = corrected[:, 0] * width
        pixel_y = corrected[:, 1] * height
        displacement_x = np.zeros(len(corrected), dtype=np.float64)
        displacement_y = np.zeros(len(corrected), dtype=np.float64)
        radius = float(operation["radius_px"])
        for center, coefficient in zip(
            operation["centers_pixel"], operation["coefficients_pixel"]
        ):
            radial = np.hypot(pixel_x - center[0], pixel_y - center[1]) / radius
            supported = radial < 1.0
            one_minus = 1.0 - radial[supported]
            weight = np.power(one_minus, 4) * (4.0 * radial[supported] + 1.0)
            displacement_x[supported] += weight * coefficient[0]
            displacement_y[supported] += weight * coefficient[1]
        corrected = np.column_stack(
            (
                (pixel_x + displacement_x) / width,
                (pixel_y + displacement_y) / height,
            )
        )
    corrected_x = min_x + corrected[:, 0] * (max_x - min_x)
    corrected_y = max_y - corrected[:, 1] * (max_y - min_y)
    return corrected_x.reshape(np.shape(web_x)), corrected_y.reshape(np.shape(web_y))


def _state_mask_in_source(state, crs: str, transform, shape) -> np.ndarray:
    projected, center_x, center_y, state_height = _projection_normalizer(state, crs)
    mask = np.zeros(shape, dtype=np.uint8)
    has_correction = isinstance(transform.get("web_mercator_correction"), dict)
    to_web = Transformer.from_crs(crs, "EPSG:3857", always_xy=True)
    from_web = Transformer.from_crs("EPSG:3857", crs, always_xy=True)

    def ring_pixels(coords) -> np.ndarray:
        points = np.asarray(coords, dtype=np.float64)
        if has_correction:
            web_x, web_y = to_web.transform(points[:, 0], points[:, 1])
            corrected_web_x, corrected_web_y = _correct_target_web_coordinates(
                np.asarray(web_x), np.asarray(web_y), transform
            )
            corrected_x, corrected_y = from_web.transform(
                corrected_web_x, corrected_web_y
            )
            points = np.column_stack((corrected_x, corrected_y))
        normalized = np.column_stack(
            ((points[:, 0] - center_x) / state_height, (center_y - points[:, 1]) / state_height)
        )
        return np.rint(
            _transform_normalized_to_source(normalized, transform, shape)
        ).astype(np.int32)

    for polygon in _polygons(projected):
        cv2.fillPoly(mask, [ring_pixels(polygon.exterior.coords)], 1)
        for interior in polygon.interiors:
            cv2.fillPoly(mask, [ring_pixels(interior.coords)], 0)
    return mask.astype(bool)


def _legend_prototypes(rgb: np.ndarray, category: Dict[str, object]) -> np.ndarray:
    if "legend_rgb" in category:
        colors = np.asarray(category["legend_rgb"], dtype=np.uint8)
        if colors.ndim == 1:
            colors = colors[None, :]
        return colors

    x1, y1, x2, y2 = map(int, category["sample_rect"])
    sample = rgb[y1:y2, x1:x2]
    if not sample.size:
        raise ValueError(f"Empty legend sample rectangle for {category['id']}")
    count = int(category.get("prototype_count", 1))
    if count == 1:
        return np.median(sample.reshape(-1, 3), axis=0).round().astype(np.uint8)[None, :]

    # Gradient bars are sampled by x-position so their light-to-dark range is
    # retained without turning JPEG noise into hundreds of prototypes.
    prototypes = []
    axis = str(category.get("prototype_axis", "x"))
    axis_length = sample.shape[0] if axis == "y" else sample.shape[1]
    for indices in np.array_split(np.arange(axis_length), count):
        pixels = (
            sample[indices, :].reshape(-1, 3)
            if axis == "y"
            else sample[:, indices].reshape(-1, 3)
        )
        chroma = pixels.max(axis=1).astype(int) - pixels.min(axis=1).astype(int)
        colorful = pixels[chroma >= 12]
        if len(colorful):
            prototypes.append(np.median(colorful, axis=0))
    if not prototypes:
        raise ValueError(f"No usable gradient prototypes for {category['id']}")
    return np.asarray(prototypes).round().astype(np.uint8)


def _rgb_colors_to_lab(colors: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(colors[None, :, :], cv2.COLOR_RGB2LAB)[0].astype(np.float32)


def classify_categorical(
    rgb: np.ndarray,
    state_mask: np.ndarray,
    categories: Sequence[Dict[str, object]],
    max_distance: float,
    min_margin: float,
    exclusion_mask: np.ndarray | None = None,
    chunk_size: int = 400_000,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Classify pixels against per-class Lab prototype sets.

    Classification is deliberately conservative: a pixel needs both an
    absolute distance pass and separation from the runner-up class.
    """

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    valid = state_mask.reshape(-1).copy()
    if exclusion_mask is not None:
        valid &= ~exclusion_mask.reshape(-1)
    indices = np.flatnonzero(valid)
    output = np.zeros(rgb.shape[:2], dtype=np.uint8)
    ambiguous = 0

    trees = []
    prototype_report = []
    for category in categories:
        prototypes = _legend_prototypes(rgb, category)
        prototype_lab = _rgb_colors_to_lab(prototypes)
        trees.append(cKDTree(prototype_lab))
        prototype_report.append(prototypes.tolist())

    flat_output = output.reshape(-1)
    for start in range(0, len(indices), chunk_size):
        selected = indices[start : start + chunk_size]
        points = lab[selected]
        distances = np.column_stack([tree.query(points, workers=-1)[0] for tree in trees])
        best_index = np.argmin(distances, axis=1)
        best_distance = distances[np.arange(len(selected)), best_index]
        if len(categories) > 1:
            second_distance = np.partition(distances, 1, axis=1)[:, 1]
        else:
            second_distance = np.full(len(selected), np.inf)
        accepted = (best_distance <= max_distance) & (
            (second_distance - best_distance) >= min_margin
        )
        flat_output[selected[accepted]] = best_index[accepted] + 1
        ambiguous += int((~accepted).sum())

    report = {
        "eligible_pixel_count": int(len(indices)),
        "classified_pixel_count": int(np.count_nonzero(output)),
        "ambiguous_pixel_count": ambiguous,
        "legend_prototypes_rgb": prototype_report,
        "max_lab_distance": max_distance,
        "minimum_lab_margin": min_margin,
    }
    return output, report


def _pattern_feature_image(
    rgb: np.ndarray, window_size: int
) -> Tuple[np.ndarray, List[str]]:
    """Describe local color and ordered texture without depending on GIF indices.

    The descriptor is intentionally format-independent. Local Lab moments retain
    the rendered color of a swatch, while direction-specific neighbor differences
    distinguish patterns such as vertical, horizontal, and diagonal dithering that
    can have the same average RGB value.
    """

    if window_size < 3 or window_size % 2 == 0:
        raise ValueError("Pattern window size must be an odd integer of at least 3")
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    kernel = (window_size, window_size)
    local_mean = cv2.boxFilter(
        lab, -1, kernel, normalize=True, borderType=cv2.BORDER_REFLECT_101
    )
    local_square_mean = cv2.boxFilter(
        lab * lab, -1, kernel, normalize=True, borderType=cv2.BORDER_REFLECT_101
    )
    local_std = np.sqrt(np.maximum(local_square_mean - local_mean * local_mean, 0.0))

    padded = np.pad(lab, ((1, 1), (1, 1), (0, 0)), mode="reflect")
    height, width = lab.shape[:2]
    texture_energy = []
    texture_names = []
    for name, (dy, dx) in (
        ("horizontal_neighbor_delta", (0, 1)),
        ("vertical_neighbor_delta", (1, 0)),
        ("down_diagonal_neighbor_delta", (1, 1)),
        ("up_diagonal_neighbor_delta", (-1, 1)),
    ):
        shifted = padded[1 + dy : 1 + dy + height, 1 + dx : 1 + dx + width]
        delta = np.linalg.norm(lab - shifted, axis=2)
        texture_energy.append(
            cv2.boxFilter(
                delta, -1, kernel, normalize=True, borderType=cv2.BORDER_REFLECT_101
            )[:, :, None]
        )
        texture_names.append(name)

    features = np.concatenate(
        (lab, local_mean, local_std, *texture_energy), axis=2
    ).astype(np.float32)
    names = (
        [f"center_lab_{channel}" for channel in "Lab"]
        + [f"local_mean_lab_{channel}" for channel in "Lab"]
        + [f"local_std_lab_{channel}" for channel in "Lab"]
        + texture_names
    )
    return features, names


def _pattern_legend_samples(
    features: np.ndarray,
    category: Dict[str, object],
    window_size: int,
    prototype_stride: int,
) -> np.ndarray:
    if "sample_rect" not in category:
        raise ValueError(
            f"Patterned category {category['id']} requires a legend sample rectangle"
        )
    x1, y1, x2, y2 = map(int, category["sample_rect"])
    radius = window_size // 2
    x1 += radius
    y1 += radius
    x2 -= radius
    y2 -= radius
    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            f"Legend sample rectangle for {category['id']} is too small for "
            f"a {window_size} pixel pattern window"
        )
    sample = features[y1:y2:prototype_stride, x1:x2:prototype_stride]
    if not sample.size:
        raise ValueError(f"Empty patterned legend sample for {category['id']}")
    return sample.reshape(-1, features.shape[2])


def _pattern_legend_colors(
    rgb: np.ndarray, category: Dict[str, object]
) -> np.ndarray:
    x1, y1, x2, y2 = map(int, category["sample_rect"])
    sample = rgb[y1:y2, x1:x2]
    if not sample.size:
        raise ValueError(f"Empty patterned legend color sample for {category['id']}")
    return np.unique(sample.reshape(-1, 3), axis=0)


def _rgb_code_image(rgb: np.ndarray) -> np.ndarray:
    values = rgb.astype(np.int32)
    return (values[:, :, 0] << 16) | (values[:, :, 1] << 8) | values[:, :, 2]


def _rgb_codes(colors: np.ndarray) -> np.ndarray:
    values = colors.astype(np.int32)
    return (values[:, 0] << 16) | (values[:, 1] << 8) | values[:, 2]


def _complete_patterned_palette(
    rgb: np.ndarray,
    state_mask: np.ndarray,
    initial: np.ndarray,
    categories: Sequence[Dict[str, object]],
    histogram_window: int,
    histogram_max_distance: float,
    histogram_min_margin: float,
    preserve_dark_luminance: int,
    preserve_dark_ink: bool,
) -> Tuple[np.ndarray, Dict[str, object], np.ndarray, np.ndarray]:
    """Complete an indexed-looking map while retaining an auditable evidence split.

    Unique legend colors have already been classified in ``initial``. This stage
    first adds higher-confidence seeds for shared colors from local exact-color
    histograms, then assigns remaining shared-color pixels to the nearest seed
    among only their compatible legend classes. Non-legend cartographic pixels
    are reconstructed from the nearest classified region. Dark ink can instead
    remain NoData for source-faithful diagnostic previews when requested.
    """

    if histogram_window < 3 or histogram_window % 2 == 0:
        raise ValueError("Palette histogram window must be an odd integer of at least 3")
    source_codes = _rgb_code_image(rgb)
    category_code_counts = []
    code_members: Dict[int, List[int]] = {}
    for category_index, category in enumerate(categories):
        x1, y1, x2, y2 = map(int, category["sample_rect"])
        sample_codes = source_codes[y1:y2, x1:x2].reshape(-1)
        codes, counts = np.unique(sample_codes, return_counts=True)
        category_code_counts.append(
            {int(code): int(count) for code, count in zip(codes, counts)}
        )
        for code in codes:
            code_members.setdefault(int(code), []).append(category_index)

    relevant_codes = np.asarray(sorted(code_members), dtype=np.int32)
    code_to_column = {int(code): index for index, code in enumerate(relevant_codes)}
    reference_histograms = np.zeros(
        (len(categories), len(relevant_codes)), dtype=np.float32
    )
    for category_index, counts in enumerate(category_code_counts):
        total = float(sum(counts.values()))
        for code, count in counts.items():
            reference_histograms[category_index, code_to_column[code]] = count / total

    membership_count = np.zeros(source_codes.shape, dtype=np.uint8)
    candidate_groups: Dict[Tuple[int, ...], np.ndarray] = {}
    for code, members in code_members.items():
        selected = source_codes == code
        membership_count[selected] = len(members)
        key = tuple(members)
        if key in candidate_groups:
            candidate_groups[key] |= selected
        else:
            candidate_groups[key] = selected.copy()

    output = initial.copy()
    before_completion_count = int(np.count_nonzero(output))
    histogram_seed_count = 0
    histogram_assignments: Dict[str, int] = {}
    shared_unresolved = state_mask & (output == 0) & (membership_count > 1)
    selected_flat = np.flatnonzero(shared_unresolved)
    if len(selected_flat):
        selected_y, selected_x = np.unravel_index(selected_flat, source_codes.shape)
        local_histograms = np.empty(
            (len(selected_flat), len(relevant_codes)), dtype=np.float32
        )
        for column, code in enumerate(relevant_codes):
            frequency = cv2.boxFilter(
                (source_codes == code).astype(np.uint8),
                cv2.CV_32F,
                (histogram_window, histogram_window),
                normalize=True,
                borderType=cv2.BORDER_REFLECT_101,
            )
            local_histograms[:, column] = frequency[selected_y, selected_x]
        local_histograms /= np.maximum(
            local_histograms.sum(axis=1, keepdims=True), 1e-6
        )
        histogram_roots = np.sqrt(local_histograms)
        reference_roots = np.sqrt(reference_histograms)
        selected_codes = source_codes[selected_y, selected_x]
        accepted_classes = np.full(len(selected_flat), -1, dtype=np.int16)

        for candidates, _ in candidate_groups.items():
            if len(candidates) < 2:
                continue
            rows = np.flatnonzero(
                np.isin(selected_codes, [
                    code for code, members in code_members.items()
                    if tuple(members) == candidates
                ])
            )
            if not len(rows):
                continue
            candidate_array = np.asarray(candidates, dtype=np.int16)
            distances = np.sqrt(
                np.maximum(
                    np.sum(
                        (
                            histogram_roots[rows, None, :]
                            - reference_roots[candidate_array][None, :, :]
                        )
                        ** 2,
                        axis=2,
                    ),
                    0.0,
                )
            ) / math.sqrt(2.0)
            best_local = np.argmin(distances, axis=1)
            best_distance = distances[np.arange(len(rows)), best_local]
            second_distance = np.partition(distances, 1, axis=1)[:, 1]
            accepted = (best_distance <= histogram_max_distance) & (
                (second_distance - best_distance) >= histogram_min_margin
            )
            accepted_classes[rows[accepted]] = candidate_array[best_local[accepted]]

        accepted_rows = np.flatnonzero(accepted_classes >= 0)
        if len(accepted_rows):
            class_values = accepted_classes[accepted_rows] + 1
            output.reshape(-1)[selected_flat[accepted_rows]] = class_values
            histogram_seed_count = int(len(accepted_rows))
            for class_id, category in enumerate(categories, 1):
                count = int(np.count_nonzero(class_values == class_id))
                if count:
                    histogram_assignments[str(category["id"])] = count

    # Use a fixed seed snapshot so compatible group order cannot influence the
    # completion. A shared rendered color may only inherit one of the categories
    # that actually uses that color in the legend.
    seed_output = output.copy()
    compatible_fill_count = 0
    compatible_assignments: Dict[str, int] = {}
    for candidates, source_group_mask in candidate_groups.items():
        target = state_mask & (output == 0) & source_group_mask
        if not np.any(target):
            continue
        available = [
            category_index
            for category_index in candidates
            if np.any(seed_output == category_index + 1)
        ]
        if not available:
            continue
        target_y, target_x = np.nonzero(target)
        distances = np.column_stack(
            [
                distance_transform_edt(seed_output != category_index + 1)[
                    target_y, target_x
                ]
                for category_index in available
            ]
        )
        chosen = np.argmin(distances, axis=1)
        chosen_classes = np.asarray(available, dtype=np.uint8)[chosen] + 1
        output[target_y, target_x] = chosen_classes
        compatible_fill_count += int(len(target_y))
        for class_id, category in enumerate(categories, 1):
            count = int(np.count_nonzero(chosen_classes == class_id))
            if count:
                compatible_assignments[str(category["id"])] = (
                    compatible_assignments.get(str(category["id"]), 0) + count
                )

    # Pixels whose exact RGB never appears in the legend are labels, graticules,
    # anti-aliasing, boundaries, or scan noise. Reconstruct their underlying
    # category spatially, but preserve truly dark ink as transparent linework.
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    preserved_dark = np.zeros(state_mask.shape, dtype=bool)
    if preserve_dark_ink:
        preserved_dark = (
            state_mask
            & (output == 0)
            & (membership_count == 0)
            & (gray <= preserve_dark_luminance)
        )
    spatial_target = (
        state_mask
        & (output == 0)
        & (membership_count == 0)
        & ~preserved_dark
    )
    spatial_fill_count = int(np.count_nonzero(spatial_target))
    if spatial_fill_count:
        _, nearest_indices = distance_transform_edt(
            output == 0, return_indices=True
        )
        nearest_values = output[nearest_indices[0], nearest_indices[1]]
        output[spatial_target] = nearest_values[spatial_target]

    completion_mask = (output > 0) & (initial == 0)
    unresolved_legend_pixels = int(
        np.count_nonzero(state_mask & (output == 0) & (membership_count > 0))
    )
    report = {
        "observed_before_completion_pixel_count": before_completion_count,
        "histogram_window": histogram_window,
        "histogram_max_hellinger_distance": histogram_max_distance,
        "histogram_minimum_margin": histogram_min_margin,
        "histogram_seed_pixel_count": histogram_seed_count,
        "histogram_assignments_by_category": histogram_assignments,
        "compatible_shared_color_fill_pixel_count": compatible_fill_count,
        "compatible_assignments_by_category": compatible_assignments,
        "nonlegend_spatial_fill_pixel_count": spatial_fill_count,
        "preserved_dark_ink_pixel_count": int(np.count_nonzero(preserved_dark)),
        "preserve_dark_ink_enabled": preserve_dark_ink,
        "preserved_dark_luminance_maximum": preserve_dark_luminance,
        "unresolved_legend_color_pixel_count": unresolved_legend_pixels,
        "completed_pixel_count": int(np.count_nonzero(completion_mask)),
        "final_classified_pixel_count": int(np.count_nonzero(output)),
        "warning": (
            "Completed pixels are deterministic source-guided reconstruction, not direct "
            "single-pixel observations. Exact shared colors are restricted to compatible "
            "legend classes; non-legend pixels inherit the nearest classified region."
        ),
    }
    return output, report, completion_mask, preserved_dark


def classify_patterned_categorical(
    rgb: np.ndarray,
    state_mask: np.ndarray,
    categories: Sequence[Dict[str, object]],
    window_size: int,
    max_distance: float,
    min_margin: float,
    exclusion_mask: np.ndarray | None = None,
    prototype_stride: int = 1,
    chunk_size: int = 150_000,
    max_color_distance: float = 3.0,
    min_color_margin: float = 1.0,
    complete_palette: bool = False,
    histogram_window: int = 9,
    histogram_max_distance: float = 0.7,
    histogram_min_margin: float = 0.05,
    preserve_dark_luminance: int = 80,
    preserve_dark_ink: bool = True,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Classify categorical fills from their local rendered color and texture.

    Each legend rectangle contributes a cloud of local descriptors, retaining
    the phases of ordered dithering rather than reducing the swatch to one RGB
    value. A pixel is accepted only when its nearest category is sufficiently
    close and separated from the nearest competing category. Categories whose
    rendered legend evidence overlaps are reported and naturally become NoData
    instead of being assigned arbitrarily.
    """

    if len(categories) > 255:
        raise ValueError("Indexed categorical output supports at most 255 categories")
    if prototype_stride < 1:
        raise ValueError("Pattern prototype stride must be at least 1")

    features, feature_names = _pattern_feature_image(rgb, window_size)
    prototype_sets = [
        _pattern_legend_samples(features, category, window_size, prototype_stride)
        for category in categories
    ]
    legend_color_sets = [_pattern_legend_colors(rgb, category) for category in categories]
    color_trees = [
        cKDTree(_rgb_colors_to_lab(colors)) for colors in legend_color_sets
    ]

    # Normalize by typical within-swatch variation. This prevents the full color
    # ramp from suppressing subtle but repeatable texture evidence, while a floor
    # keeps perfectly flat feature dimensions numerically stable.
    within_swatch_deviations = []
    for prototypes in prototype_sets:
        center = np.median(prototypes, axis=0)
        within_swatch_deviations.append(np.abs(prototypes - center))
    deviations = np.concatenate(within_swatch_deviations, axis=0)
    feature_scale = np.maximum(np.percentile(deviations, 75, axis=0) * 2.0, 1.0)
    scaled_sets = [prototypes / feature_scale for prototypes in prototype_sets]
    trees = [cKDTree(prototypes) for prototypes in scaled_sets]

    confusable_pairs = []
    for first in range(len(categories)):
        for second in range(first + 1, len(categories)):
            first_to_second = trees[second].query(scaled_sets[first], workers=-1)[0]
            second_to_first = trees[first].query(scaled_sets[second], workers=-1)[0]
            cross_distances = np.concatenate((first_to_second, second_to_first))
            overlap_fraction = float(np.mean(cross_distances <= min_margin))
            if overlap_fraction > 0.0:
                confusable_pairs.append(
                    {
                        "first_category": categories[first]["id"],
                        "second_category": categories[second]["id"],
                        "minimum_cross_distance": float(np.min(cross_distances)),
                        "median_cross_distance": float(np.median(cross_distances)),
                        "prototype_overlap_fraction": overlap_fraction,
                    }
                )

    valid = state_mask.reshape(-1).copy()
    if exclusion_mask is not None:
        valid &= ~exclusion_mask.reshape(-1)
    indices = np.flatnonzero(valid)
    flat_features = features.reshape(-1, features.shape[2])
    flat_output = np.zeros(features.shape[0] * features.shape[1], dtype=np.uint8)
    ambiguous = 0
    direct_color_classified = 0
    local_pattern_classified = 0

    for start in range(0, len(indices), chunk_size):
        selected = indices[start : start + chunk_size]
        center_lab = flat_features[selected, :3]
        color_distances = np.column_stack(
            [tree.query(center_lab, workers=-1)[0] for tree in color_trees]
        )
        color_best_index = np.argmin(color_distances, axis=1)
        color_best_distance = color_distances[
            np.arange(len(selected)), color_best_index
        ]
        if len(categories) > 1:
            color_second_distance = np.partition(color_distances, 1, axis=1)[:, 1]
        else:
            color_second_distance = np.full(len(selected), np.inf)
        direct_accepted = (color_best_distance <= max_color_distance) & (
            (color_second_distance - color_best_distance) >= min_color_margin
        )
        flat_output[selected[direct_accepted]] = color_best_index[direct_accepted] + 1
        direct_color_classified += int(np.count_nonzero(direct_accepted))

        unresolved = ~direct_accepted
        unresolved_selected = selected[unresolved]
        if len(unresolved_selected):
            points = flat_features[unresolved_selected] / feature_scale
            distances = np.column_stack(
                [tree.query(points, workers=-1)[0] for tree in trees]
            )
            best_index = np.argmin(distances, axis=1)
            best_distance = distances[np.arange(len(unresolved_selected)), best_index]
            if len(categories) > 1:
                second_distance = np.partition(distances, 1, axis=1)[:, 1]
            else:
                second_distance = np.full(len(unresolved_selected), np.inf)
            pattern_accepted = (best_distance <= max_distance) & (
                (second_distance - best_distance) >= min_margin
            )
            flat_output[unresolved_selected[pattern_accepted]] = (
                best_index[pattern_accepted] + 1
            )
            local_pattern_classified += int(np.count_nonzero(pattern_accepted))
            ambiguous += int(np.count_nonzero(~pattern_accepted))

    output = flat_output.reshape(rgb.shape[:2])
    completion_report = None
    completion_mask = None
    preserved_dark_mask = None
    if complete_palette:
        output, completion_report, completion_mask, preserved_dark_mask = (
            _complete_patterned_palette(
                rgb,
                state_mask,
                output,
                categories,
                histogram_window=histogram_window,
                histogram_max_distance=histogram_max_distance,
                histogram_min_margin=histogram_min_margin,
                preserve_dark_luminance=preserve_dark_luminance,
                preserve_dark_ink=preserve_dark_ink,
            )
        )
    report = {
        "eligible_pixel_count": int(len(indices)),
        "classified_pixel_count": int(np.count_nonzero(output)),
        "ambiguous_pixel_count": int(len(indices) - np.count_nonzero(output)),
        "pattern_window_size": window_size,
        "prototype_stride": prototype_stride,
        "prototype_counts": [int(len(item)) for item in prototype_sets],
        "legend_color_counts": [int(len(item)) for item in legend_color_sets],
        "direct_color_classified_pixel_count": direct_color_classified,
        "local_pattern_classified_pixel_count": local_pattern_classified,
        "feature_names": feature_names,
        "feature_scale": feature_scale.tolist(),
        "max_color_lab_distance": max_color_distance,
        "minimum_color_lab_margin": min_color_margin,
        "max_pattern_distance": max_distance,
        "minimum_pattern_margin": min_margin,
        "confusable_legend_pairs": confusable_pairs,
        "warning": (
            "Local texture classification uses visible color and repeated pattern evidence. "
            "Pixels under labels, boundaries, or other occlusions and categories with "
            "indistinguishable rendered swatches remain NoData unless separately repaired."
        ),
    }
    if completion_report is not None:
        report["palette_completion"] = completion_report
        report["_source_completion_mask"] = completion_mask
        report["_source_preserved_dark_mask"] = preserved_dark_mask
    return output, report


def infer_sparse_chroma_overlays(
    rgb: np.ndarray,
    state_mask: np.ndarray,
    categories: Sequence[Dict[str, object]],
    min_chroma: float,
    coefficient_threshold: float,
    max_residual: float,
    complexity_penalty: float,
    chunk_size: int = 400_000,
) -> Tuple[List[np.ndarray], Dict[str, object]]:
    """Infer up to two transparent overlay colors from Lab chroma.

    Luminance is intentionally excluded, allowing the same overlay to be found
    over different gray precipitation classes.  The two chroma dimensions can
    identify at most two independent colors; three-way or opaque occlusions are
    left ambiguous rather than reconstructed.
    """

    refs = np.asarray([category["legend_rgb"] for category in categories], dtype=np.uint8)
    ref_ab = _rgb_colors_to_lab(refs)[:, 1:3] - 128.0
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    ab = lab[:, 1:3] - 128.0
    indices = np.flatnonzero(state_mask.reshape(-1))
    flat_masks = [np.zeros(rgb.shape[0] * rgb.shape[1], dtype=np.uint8) for _ in categories]
    ambiguous = 0
    inferred_overlap = 0
    subsets = [(i,) for i in range(len(categories))]
    subsets.extend(itertools.combinations(range(len(categories)), 2))

    for start in range(0, len(indices), chunk_size):
        selected = indices[start : start + chunk_size]
        points = ab[selected]
        chroma = np.linalg.norm(points, axis=1)
        candidate_scores = np.full((len(points), len(subsets)), np.inf, dtype=np.float32)
        candidate_coefficients: List[np.ndarray] = []
        candidate_residuals: List[np.ndarray] = []
        for subset_index, subset in enumerate(subsets):
            matrix = ref_ab[list(subset)].T
            coefficients = points @ np.linalg.pinv(matrix).T
            coefficients = np.maximum(coefficients, 0.0)
            reconstructed = coefficients @ matrix.T
            residual = np.linalg.norm(points - reconstructed, axis=1)
            score = residual + complexity_penalty * len(subset)
            candidate_scores[:, subset_index] = score
            candidate_coefficients.append(coefficients)
            candidate_residuals.append(residual)

        winners = np.argmin(candidate_scores, axis=1)
        for subset_index, subset in enumerate(subsets):
            chosen = winners == subset_index
            if not np.any(chosen):
                continue
            coefficients = candidate_coefficients[subset_index][chosen]
            residual = candidate_residuals[subset_index][chosen]
            accepted = (chroma[chosen] >= min_chroma) & (residual <= max_residual)
            chosen_indices = selected[chosen][accepted]
            accepted_coefficients = coefficients[accepted]
            memberships = accepted_coefficients >= coefficient_threshold
            membership_count = memberships.sum(axis=1)
            usable = membership_count > 0
            chosen_indices = chosen_indices[usable]
            memberships = memberships[usable]
            inferred_overlap += int(np.count_nonzero(memberships.sum(axis=1) > 1))
            for local_index, category_index in enumerate(subset):
                flat_masks[category_index][chosen_indices[memberships[:, local_index]]] = 1
        accepted_any = np.zeros(len(selected), dtype=bool)
        for mask in flat_masks:
            accepted_any |= mask[selected] > 0
        ambiguous += int(np.count_nonzero((chroma >= min_chroma) & ~accepted_any))

    masks = [mask.reshape(rgb.shape[:2]).astype(bool) for mask in flat_masks]
    return masks, {
        "eligible_pixel_count": int(len(indices)),
        "ambiguous_chromatic_pixel_count": ambiguous,
        "inferred_multi_overlay_pixel_count": inferred_overlap,
        "minimum_chroma": min_chroma,
        "coefficient_threshold": coefficient_threshold,
        "maximum_residual": max_residual,
        "warning": (
            "Sparse chroma unmixing can identify at most two simultaneous transparent "
            "overlays. Fully opaque or three-way occlusions remain unknowable from RGB."
        ),
    }


def classify_grayscale(
    rgb: np.ndarray,
    state_mask: np.ndarray,
    categories: Sequence[Dict[str, object]],
    smoothing_radius: int,
    max_distance: float,
    max_chroma: float,
    exclusion_mask: np.ndarray | None = None,
    adaptive_centers: bool = False,
    background_cutoff: int = 245,
) -> Tuple[np.ndarray, Dict[str, object]]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    kernel = max(3, smoothing_radius * 2 + 1)
    if kernel % 2 == 0:
        kernel += 1
    smooth = cv2.medianBlur(gray, kernel)
    chroma = rgb.max(axis=2).astype(np.int16) - rgb.min(axis=2).astype(np.int16)
    eligible = state_mask & (chroma <= max_chroma)
    if exclusion_mask is not None:
        eligible &= ~exclusion_mask
    legend_references = np.asarray(
        [category["legend_gray"] for category in categories], dtype=float
    )
    references = legend_references.copy()
    if adaptive_centers:
        calibration = smooth[eligible & (smooth < background_cutoff)].astype(np.float32)
        if len(calibration) < len(categories) * 100:
            raise ValueError("Insufficient map pixels to calibrate grayscale class centers")
        _, labels, centers = cv2.kmeans(
            calibration[:, None],
            len(categories),
            None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1),
            10,
            cv2.KMEANS_PP_CENTERS,
        )
        del labels
        # Both legend and map classes are ordered light to dark. The map
        # centers absorb transparency over a pale basemap without changing
        # semantic class order.
        references = np.sort(centers[:, 0])[::-1]
    distances = np.abs(smooth[:, :, None].astype(float) - references[None, None, :])
    best = np.argmin(distances, axis=2)
    best_distance = np.take_along_axis(distances, best[:, :, None], axis=2)[:, :, 0]
    accepted = eligible & (best_distance <= max_distance)
    output = np.zeros(rgb.shape[:2], dtype=np.uint8)
    output[accepted] = best[accepted] + 1
    return output, {
        "eligible_pixel_count": int(np.count_nonzero(eligible)),
        "classified_pixel_count": int(np.count_nonzero(output)),
        "ambiguous_pixel_count": int(np.count_nonzero(eligible & ~accepted)),
        "legend_gray": legend_references.tolist(),
        "map_rendered_gray_centers": references.tolist(),
        "adaptive_centers": adaptive_centers,
        "median_filter_radius": smoothing_radius,
        "max_gray_distance": max_distance,
        "warning": (
            "Broad grayscale classes are estimated after median suppression of hillshade. "
            "Fine boundaries and pixels beneath colored overlays require inspection."
        ),
    }


def classify_dot_pattern(
    rgb: np.ndarray,
    state_mask: np.ndarray,
    categories: Sequence[Dict[str, object]],
    max_distance: float,
    maximum_component_area: int,
    maximum_component_dimension: int,
    pattern_radius: int,
    minimum_quadrants: int,
    minimum_quadrant_pixels: int,
    density_window: int,
    minimum_window_pixels: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Keep small swatch-colored marks while rejecting lines and labels.

    This preserves the literal visible hatch pixels. It does not fill the
    spaces between dots or infer the hidden categorical value beneath them.
    """

    raw, report = classify_categorical(
        rgb,
        state_mask,
        categories,
        max_distance=max_distance,
        min_margin=0.0,
    )
    binary = raw > 0
    component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8), connectivity=8
    )
    geometry_keep = np.zeros(component_count, dtype=bool)
    for index in range(1, component_count):
        width = int(stats[index, cv2.CC_STAT_WIDTH])
        height = int(stats[index, cv2.CC_STAT_HEIGHT])
        area = int(stats[index, cv2.CC_STAT_AREA])
        geometry_keep[index] = (
            area <= maximum_component_area
            and width <= maximum_component_dimension
            and height <= maximum_component_dimension
        )
    candidate = geometry_keep[labels]
    integral = cv2.integral(candidate.astype(np.uint8))

    def rectangle_sum(x1: int, y1: int, x2: int, y2: int) -> int:
        x1 = max(0, min(candidate.shape[1], x1))
        x2 = max(0, min(candidate.shape[1], x2))
        y1 = max(0, min(candidate.shape[0], y1))
        y2 = max(0, min(candidate.shape[0], y2))
        return int(
            integral[y2, x2]
            - integral[y1, x2]
            - integral[y2, x1]
            + integral[y1, x1]
        )

    # A hatch dot has neighboring dots in two dimensions. A broken line may
    # also consist of tiny components, but its neighborhood is essentially
    # one-dimensional and therefore does not occupy three quadrants.
    keep = np.zeros(component_count, dtype=bool)
    gap = 2
    for index in np.flatnonzero(geometry_keep):
        center_x, center_y = np.rint(centroids[index]).astype(int)
        quadrants = (
            rectangle_sum(center_x - pattern_radius, center_y - pattern_radius, center_x - gap, center_y - gap),
            rectangle_sum(center_x + gap, center_y - pattern_radius, center_x + pattern_radius, center_y - gap),
            rectangle_sum(center_x - pattern_radius, center_y + gap, center_x - gap, center_y + pattern_radius),
            rectangle_sum(center_x + gap, center_y + gap, center_x + pattern_radius, center_y + pattern_radius),
        )
        keep[index] = sum(value >= minimum_quadrant_pixels for value in quadrants) >= minimum_quadrants
    filtered = keep[labels]
    density = cv2.boxFilter(
        candidate.astype(np.float32),
        -1,
        (density_window, density_window),
        normalize=False,
    )
    filtered &= density >= minimum_window_pixels
    output = filtered.astype(np.uint8)
    report.update(
        {
            "raw_color_matched_pixel_count": int(np.count_nonzero(binary)),
            "dot_pattern_pixel_count": int(np.count_nonzero(filtered)),
            "rejected_line_or_large_component_pixel_count": int(
                np.count_nonzero(binary & ~filtered)
            ),
            "maximum_component_area": maximum_component_area,
            "maximum_component_dimension": maximum_component_dimension,
            "pattern_radius": pattern_radius,
            "minimum_occupied_quadrants": minimum_quadrants,
            "minimum_pixels_per_quadrant": minimum_quadrant_pixels,
            "density_window": density_window,
            "minimum_window_pixels": minimum_window_pixels,
        }
    )
    return output, report


def _target_state_mask(state, bounds, shape) -> np.ndarray:
    projected = transform_geometry(
        Transformer.from_crs("EPSG:4269", "EPSG:3857", always_xy=True).transform,
        state,
    )
    min_x, min_y, max_x, max_y = bounds
    height, width = shape
    mask = np.zeros(shape, dtype=np.uint8)

    def ring_pixels(coords):
        points = np.asarray(coords, dtype=np.float64)
        x = (points[:, 0] - min_x) / (max_x - min_x) * width
        y = (max_y - points[:, 1]) / (max_y - min_y) * height
        return np.rint(np.column_stack((x, y))).astype(np.int32)

    for polygon in _polygons(projected):
        cv2.fillPoly(mask, [ring_pixels(polygon.exterior.coords)], 1)
        for interior in polygon.interiors:
            cv2.fillPoly(mask, [ring_pixels(interior.coords)], 0)
    return mask.astype(bool)


def warp_classified_to_web_mercator(
    source: np.ndarray,
    state,
    transform: Dict[str, object],
    source_shape: Tuple[int, int],
    target_height: int | None = None,
    clip_to_state: bool = True,
) -> Tuple[np.ndarray, Dict[str, object]]:
    projection_crs = str(transform["projection_crs"])
    projected_web = transform_geometry(
        Transformer.from_crs("EPSG:4269", "EPSG:3857", always_xy=True).transform,
        state,
    )
    bounds = projected_web.bounds
    min_x, min_y, max_x, max_y = bounds
    correction = transform.get("web_mercator_correction")
    correction_grid = correction.get("grid") if isinstance(correction, dict) else None
    if target_height is None:
        if isinstance(correction_grid, dict):
            target_height = int(correction_grid["height"])
        elif "reference_to_source_matrix" in transform:
            projected, center_x, center_y, state_height = _projection_normalizer(
                state, projection_crs
            )
            mainland = _largest_polygon(projected)
            outline = np.asarray(mainland.exterior.coords, dtype=np.float64)
            normalized_outline = np.column_stack(
                (
                    (outline[:, 0] - center_x) / state_height,
                    (center_y - outline[:, 1]) / state_height,
                )
            )
            source_outline = _transform_normalized_to_source(
                normalized_outline, transform, source_shape
            )
            target_height = max(256, round(float(np.ptp(source_outline[:, 1]))))
        else:
            parameters = transform["parameters"]
            target_height = max(
                256, round(float(parameters["state_height_fraction"]) * source_shape[0])
            )
    if isinstance(correction_grid, dict) and target_height == int(correction_grid["height"]):
        target_width = int(correction_grid["width"])
    else:
        target_width = max(256, round(target_height * (max_x - min_x) / (max_y - min_y)))

    cols = np.arange(target_width, dtype=np.float64) + 0.5
    rows = np.arange(target_height, dtype=np.float64) + 0.5
    web_x = min_x + cols / target_width * (max_x - min_x)
    web_y = max_y - rows / target_height * (max_y - min_y)
    grid_x, grid_y = np.meshgrid(web_x, web_y)
    sampling_web_x, sampling_web_y = _correct_target_web_coordinates(
        grid_x, grid_y, transform
    )
    if projection_crs == "EPSG:3857":
        projected_x, projected_y = sampling_web_x, sampling_web_y
    else:
        transformer = Transformer.from_crs("EPSG:3857", projection_crs, always_xy=True)
        projected_x, projected_y = transformer.transform(
            sampling_web_x, sampling_web_y
        )

    _, center_x, center_y, state_height = _projection_normalizer(state, projection_crs)
    normalized = np.column_stack(
        (
            (projected_x.ravel() - center_x) / state_height,
            (center_y - projected_y.ravel()) / state_height,
        )
    )
    source_points = _transform_normalized_to_source(normalized, transform, source_shape)
    map_x = source_points[:, 0].reshape(target_height, target_width).astype(np.float32)
    map_y = source_points[:, 1].reshape(target_height, target_width).astype(np.float32)
    remap_source = np.asarray(source)
    if remap_source.dtype == np.bool_:
        remap_source = remap_source.astype(np.uint8)
    elif remap_source.dtype not in (np.uint8, np.uint16, np.float32):
        remap_source = remap_source.astype(np.float32)
    warped = cv2.remap(
        remap_source,
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    if clip_to_state:
        target_mask = _target_state_mask(state, bounds, warped.shape[:2])
        warped[~target_mask] = 0
    return warped, {
        "crs": "EPSG:3857",
        "bounds": [float(value) for value in bounds],
        "width": target_width,
        "height": target_height,
        "resampling": "nearest",
        "clip_to_state": clip_to_state,
    }


def _draw_web_reference_lines(
    image: np.ndarray,
    bounds: Sequence[float],
    state,
    counties,
) -> np.ndarray:
    """Overlay authoritative state and county lines on a warped source image."""

    output = image.copy()
    height, width = output.shape[:2]
    min_x, min_y, max_x, max_y = bounds
    transformer = Transformer.from_crs("EPSG:4269", "EPSG:3857", always_xy=True)

    def pixels(coords) -> np.ndarray:
        points = np.asarray(coords, dtype=np.float64)
        x = (points[:, 0] - min_x) / (max_x - min_x) * width
        y = (max_y - points[:, 1]) / (max_y - min_y) * height
        return np.rint(np.column_stack((x, y))).astype(np.int32).reshape((-1, 1, 2))

    for county in counties:
        projected = transform_geometry(transformer.transform, county)
        for polygon in _polygons(projected):
            cv2.polylines(
                output,
                [pixels(polygon.exterior.coords)],
                True,
                (255, 0, 255),
                1,
                cv2.LINE_AA,
            )
    projected_state = transform_geometry(transformer.transform, state)
    for polygon in _polygons(projected_state):
        cv2.polylines(
            output,
            [pixels(polygon.exterior.coords)],
            True,
            (0, 255, 255),
            3,
            cv2.LINE_AA,
        )
    return output


def _web_reference_assets(
    bounds: Sequence[float],
    shape: Tuple[int, int],
    state,
    counties,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Render state/county overlays and single-pixel diagnostic masks."""

    height, width = shape
    min_x, min_y, max_x, max_y = bounds
    transformer = Transformer.from_crs("EPSG:4269", "EPSG:3857", always_xy=True)
    state_overlay = np.zeros((height, width, 4), dtype=np.uint8)
    county_overlay = np.zeros((height, width, 4), dtype=np.uint8)
    state_mask = np.zeros((height, width), dtype=np.uint8)
    county_mask = np.zeros((height, width), dtype=np.uint8)

    def pixels(coords) -> np.ndarray:
        points = np.asarray(coords, dtype=np.float64)
        x = (points[:, 0] - min_x) / (max_x - min_x) * width
        y = (max_y - points[:, 1]) / (max_y - min_y) * height
        return np.rint(np.column_stack((x, y))).astype(np.int32).reshape((-1, 1, 2))

    for county in counties:
        projected = transform_geometry(transformer.transform, county)
        for polygon in _polygons(projected):
            line = pixels(polygon.exterior.coords)
            cv2.polylines(county_mask, [line], True, 1, 1, cv2.LINE_8)
            cv2.polylines(
                county_overlay, [line], True, (255, 0, 255, 210), 2, cv2.LINE_AA
            )

    projected_state = transform_geometry(transformer.transform, state)
    for polygon in _polygons(projected_state):
        line = pixels(polygon.exterior.coords)
        cv2.polylines(state_mask, [line], True, 1, 1, cv2.LINE_8)
        cv2.polylines(
            state_overlay, [line], True, (0, 238, 238, 255), 4, cv2.LINE_AA
        )
    return state_overlay, county_overlay, state_mask.astype(bool), county_mask.astype(bool)


def _registered_county_reference_assets(
    manifest_path: Path,
    bounds: Sequence[float],
    shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Render the registered thin county strokes recovered from ``county.png``.

    The reference raster is intentionally resampled as subpixel coverage rather
    than redrawn with a fixed-width vector stroke. This preserves the source
    line weight when a high-resolution county reference is inspected on a
    smaller per-map grid.
    """

    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "pass":
        raise ValueError("Registered county reference has not passed validation")
    if manifest.get("reference_kind") != "styled_high_resolution_county_raster":
        raise ValueError("Registered county reference is not derived from county.png")

    source = manifest["source"]
    source_path = Path(str(source["path"])).resolve()
    if not source_path.is_file() or _sha256(source_path) != source["sha256"]:
        raise ValueError("Registered county-reference source hash mismatch")

    grid = manifest["web_grid"]
    if grid.get("crs") != "EPSG:3857" or not np.allclose(
        np.asarray(grid.get("bounds"), dtype=np.float64),
        np.asarray(bounds, dtype=np.float64),
        rtol=0,
        atol=1e-6,
    ):
        raise ValueError("Registered county-reference grid differs from review grid")

    artifact = manifest["artifacts"]["web_mercator_county_border"]
    mask_path = manifest_path.parent / str(artifact["path"])
    if not mask_path.is_file() or _sha256(mask_path) != artifact["sha256"]:
        raise ValueError("Registered county-reference artifact hash mismatch")
    source_mask = np.asarray(Image.open(mask_path).convert("L")) > 0
    expected_shape = (int(grid["height"]), int(grid["width"]))
    if source_mask.shape != expected_shape:
        raise ValueError("Registered county-reference artifact dimensions are stale")

    height, width = shape
    coverage = cv2.resize(
        source_mask.astype(np.float32),
        (width, height),
        interpolation=cv2.INTER_AREA,
    )
    county_mask = coverage > 0
    alpha = np.zeros((height, width), dtype=np.uint8)
    if np.any(county_mask):
        # A partially covered destination pixel represents a subpixel source
        # stroke. Keep it visible without inflating it to the former 2px line.
        alpha[county_mask] = np.clip(
            np.rint(coverage[county_mask] * 255.0 * 3.0), 96, 220
        ).astype(np.uint8)
    overlay = np.zeros((height, width, 4), dtype=np.uint8)
    overlay[county_mask, 0] = 255
    overlay[county_mask, 2] = 255
    overlay[:, :, 3] = alpha
    provenance = {
        "reference_kind": manifest["reference_kind"],
        "manifest": {
            "path": str(manifest_path),
            "sha256": _sha256(manifest_path),
        },
        "source": {
            "path": str(source_path),
            "sha256": source["sha256"],
        },
        "artifact": {
            "path": str(mask_path),
            "sha256": artifact["sha256"],
        },
        "rendering": "area-coverage resampling at recovered county.png stroke weight",
        "visible_pixel_count": int(np.count_nonzero(county_mask)),
    }
    return overlay, county_mask, provenance


def _county_residual_diagnostic(
    source_rgb: np.ndarray,
    county_mask: np.ndarray,
    state_line_mask: np.ndarray,
    county_reference_label: str = "authoritative county geometry",
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Color county lines by distance to nearby source edges.

    This is intentionally diagnostic. Dense relief, faults, labels, or other
    linework can create a nearby edge even when the geographic county line is
    displaced, so the metric must not become an automatic acceptance score.
    """

    gray = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 45, 125, L2gradient=True) > 0
    distance = distance_transform_edt(~edges)
    state_exclusion = cv2.dilate(
        state_line_mask.astype(np.uint8), np.ones((13, 13), dtype=np.uint8)
    ).astype(bool)
    samples = county_mask & ~state_exclusion
    values = distance[samples]
    overlay = np.zeros((*county_mask.shape, 4), dtype=np.uint8)
    if not len(values):
        return overlay, {
            "county_line_sample_count": 0,
            "warning": "No interior county-line samples were available for diagnostics.",
        }

    bands = (
        (samples & (distance <= 3), (52, 211, 153, 220)),
        (samples & (distance > 3) & (distance <= 8), (250, 176, 5, 225)),
        (samples & (distance > 8), (239, 68, 68, 235)),
    )
    kernel = np.ones((3, 3), dtype=np.uint8)
    for mask, color in bands:
        visible = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)
        overlay[visible] = color
    return overlay, {
        "county_line_sample_count": int(len(values)),
        "median_nearest_source_edge_px": float(np.median(values)),
        "p90_nearest_source_edge_px": float(np.percentile(values, 90)),
        "within_3px_fraction": float(np.mean(values <= 3)),
        "within_8px_fraction": float(np.mean(values <= 8)),
        "county_line_reference": county_reference_label,
        "source_edge_method": "Gaussian-3 Canny 45/125; state boundary excluded by 6px",
        "warning": (
            "Nearest-edge county residual is diagnostic only. Terrain, faults, labels, "
            "and other linework can produce false low distances; county vintage and "
            "generalization can produce false high distances."
        ),
    }


def _save_indexed(path: Path, values: np.ndarray) -> None:
    Image.fromarray(values.astype(np.uint8), mode="L").save(path, optimize=True)


def _preview(classified: np.ndarray, categories: Sequence[Dict[str, object]]) -> np.ndarray:
    rgba = np.zeros((*classified.shape, 4), dtype=np.uint8)
    for class_id, category in enumerate(categories, 1):
        color = category.get("display_rgb", category.get("legend_rgb", [255, 0, 255]))
        if isinstance(color[0], list):
            color = color[0]
        selected = classified == class_id
        rgba[selected, :3] = np.asarray(color, dtype=np.uint8)
        rgba[selected, 3] = 230
    return rgba


def _fill_indexed_nodata_in_mask(
    values: np.ndarray, valid_mask: np.ndarray
) -> Tuple[np.ndarray, int]:
    """Fill interior indexed NoData from the geographically nearest class."""

    output = values.copy()
    output[~valid_mask] = 0
    target = valid_mask & (output == 0)
    target_count = int(np.count_nonzero(target))
    if target_count == 0 or not np.any(output > 0):
        return output, 0
    _, nearest_indices = distance_transform_edt(
        output == 0, return_indices=True
    )
    nearest_values = output[nearest_indices[0], nearest_indices[1]]
    output[target] = nearest_values[target]
    return output, target_count


def _suppress_isolated_class_specks(
    values: np.ndarray,
    categories: Sequence[Dict[str, object]],
    maximum_area: int,
    minimum_surrounding_purity: float,
    preserve_near_white: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    """Replace tiny isolated antialias components with their surrounding class."""

    if maximum_area < 1:
        raise ValueError("Maximum speck area must be positive")
    if not 0.5 <= minimum_surrounding_purity <= 1.0:
        raise ValueError("Speck surrounding purity must be between 0.5 and 1.0")
    initial = values.copy()
    output = values.copy()
    changed = np.zeros(values.shape, dtype=bool)
    original_classes = np.zeros(values.shape, dtype=np.uint8)
    assignments: Dict[str, int] = {}
    reassigned_components = 0
    kernel = np.ones((3, 3), dtype=np.uint8)

    for class_id, category in enumerate(categories, 1):
        color = category.get("display_rgb", category.get("legend_rgb"))
        if (
            preserve_near_white
            and isinstance(color, list)
            and len(color) >= 3
            and not isinstance(color[0], list)
            and min(map(int, color[:3])) >= 245
        ):
            continue
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            (initial == class_id).astype(np.uint8), connectivity=8
        )
        for component_id in range(1, count):
            area = int(stats[component_id, cv2.CC_STAT_AREA])
            if area > maximum_area:
                continue
            component = labels == component_id
            ring = cv2.dilate(component.astype(np.uint8), kernel).astype(bool)
            ring &= ~component
            neighbors = initial[ring]
            neighbors = neighbors[(neighbors > 0) & (neighbors != class_id)]
            if not len(neighbors):
                continue
            counts = np.bincount(neighbors, minlength=len(categories) + 1)
            replacement = int(np.argmax(counts[1:]) + 1)
            purity = float(counts[replacement] / len(neighbors))
            if purity < minimum_surrounding_purity:
                continue
            output[component] = replacement
            changed[component] = True
            original_classes[component] = class_id
            reassigned_components += 1
            key = f"{category['id']}->{categories[replacement - 1]['id']}"
            assignments[key] = assignments.get(key, 0) + area

    report = {
        "maximum_component_area_inclusive": maximum_area,
        "minimum_surrounding_class_purity": minimum_surrounding_purity,
        "near_white_categories_preserved": preserve_near_white,
        "reassigned_component_count": reassigned_components,
        "reassigned_pixel_count": int(np.count_nonzero(changed)),
        "assignments_by_category": assignments,
        "warning": (
            "Only tiny isolated components with a majority surrounding class are "
            "reassigned. Original and replacement class IDs remain separately masked."
        ),
    }
    return output, changed, original_classes, report


def _publication_interior_with_water_exclusion(
    grid: Dict[str, object],
    water_exclusion: object,
    coastal_seam: object = None,
    canonical_interior_mode: str = "pinned_polygon",
) -> Tuple[np.ndarray, Dict[str, object], Optional[np.ndarray]]:
    if canonical_interior_mode == "pinned_polygon":
        base_interior, provenance = canonical_publication_interior(grid)
    elif canonical_interior_mode == "active_boundary_ring":
        base_interior, provenance = active_boundary_publication_interior(grid)
    else:
        raise ValueError(
            f"Unknown canonical publication interior mode: {canonical_interior_mode}"
        )
    if coastal_seam is not None:
        if not isinstance(coastal_seam, dict):
            raise ValueError("coastal_seam must be an object")
        base_interior, seam_report, _ = close_west_coast_clipping_seam(
            base_interior,
            grid,
            maximum_gap_px=int(coastal_seam.get("maximum_gap_px", 50)),
            maximum_x_fraction=float(
                coastal_seam.get("maximum_x_fraction", 0.72)
            ),
            start_y_fraction=float(coastal_seam.get("start_y_fraction", 0.04)),
            end_y_fraction=float(coastal_seam.get("end_y_fraction", 0.95)),
        )
        provenance = {**provenance, "coastal_seam": seam_report}
    if water_exclusion is None:
        return base_interior, provenance, None
    if not isinstance(water_exclusion, dict):
        raise ValueError("water_exclusion must be an object")
    reference = Path(str(water_exclusion.get("reference", "")))
    if not reference.is_dir():
        raise FileNotFoundError(f"Water exclusion reference does not exist: {reference}")
    water, report = rasterize_california_areawater(
        reference,
        grid,
        supersampling=int(water_exclusion.get("supersampling", 4)),
        required_feature_names=water_exclusion.get("required_feature_names", []),
        minimum_coverage_fraction=float(
            water_exclusion.get("minimum_coverage_fraction", 0.25)
        ),
        force_any_coverage_feature_names=water_exclusion.get(
            "force_any_coverage_feature_names", []
        ),
        include_only_feature_names=water_exclusion.get(
            "include_only_feature_names", []
        ),
        exclude_feature_names=water_exclusion.get("exclude_feature_names", []),
    )
    shoreline_snap = water_exclusion.get("canonical_shoreline_snap")
    if shoreline_snap is not None:
        if not isinstance(shoreline_snap, dict):
            raise ValueError("canonical_shoreline_snap must be an object")
        feature_names = [
            str(item) for item in shoreline_snap.get("feature_names", [])
        ]
        if not feature_names:
            raise ValueError("canonical_shoreline_snap needs named water features")
        snap_seed, seed_report = rasterize_california_areawater(
            reference,
            grid,
            supersampling=int(water_exclusion.get("supersampling", 4)),
            required_feature_names=feature_names,
            minimum_coverage_fraction=float(
                water_exclusion.get("minimum_coverage_fraction", 0.25)
            ),
            include_only_feature_names=feature_names,
        )
        mapbox_reference = shoreline_snap.get("mapbox_water_reference")
        if mapbox_reference:
            snapped_water, snap_report = constrain_named_water_to_mapbox(
                snap_seed,
                base_interior,
                grid,
                Path(str(mapbox_reference)),
                maximum_distance_px=float(
                    shoreline_snap.get("maximum_distance_px", 40.0)
                ),
                supersampling=int(
                    shoreline_snap.get("mapbox_supersampling", 4)
                ),
                minimum_coverage_fraction=float(
                    shoreline_snap.get(
                        "mapbox_minimum_coverage_fraction", 1.0
                    )
                ),
            )
        else:
            snapped_water, snap_report = snap_named_water_to_active_boundary(
                snap_seed,
                base_interior,
                grid,
                maximum_distance_px=float(
                    shoreline_snap.get("maximum_distance_px", 40.0)
                ),
            )
        unsnapped_water = water & ~snap_seed
        water = unsnapped_water | snapped_water
        report = {
            **report,
            "canonical_shoreline_snap": {
                "feature_names": feature_names,
                "seed_rasterization": seed_report,
                **snap_report,
                "unsnapped_named_water_pixel_count": int(
                    np.count_nonzero(unsnapped_water)
                ),
                "combined_water_pixel_count": int(np.count_nonzero(water)),
            },
        }
    if bool(water_exclusion.get("limit_to_pinned_polygon_interior", False)):
        pinned_water_eligibility, pinned_provenance = (
            canonical_publication_interior(grid)
        )
        water_before_eligibility = int(np.count_nonzero(water))
        water &= pinned_water_eligibility
        report = {
            **report,
            "pinned_polygon_water_eligibility": {
                "method": "water_removal_limited_to_pinned_mainland_interior",
                "pinned_valid_pixel_count": int(
                    np.count_nonzero(pinned_water_eligibility)
                ),
                "candidate_water_pixel_count": water_before_eligibility,
                "eligible_water_pixel_count": int(np.count_nonzero(water)),
                "outer_canonical_land_pixel_count_restored": int(
                    water_before_eligibility - np.count_nonzero(water)
                ),
                "pinned_reference": pinned_provenance,
                "policy": (
                    "The active lime ring owns the outer coast; pinned internal "
                    "water may remain transparent."
                ),
            },
        }
    removed = base_interior & water
    final = base_interior & ~water
    provenance = {
        **provenance,
        "base_valid_pixel_count": int(np.count_nonzero(base_interior)),
        "valid_pixel_count": int(np.count_nonzero(final)),
        "internal_water_exclusion": {
            **report,
            "excluded_interior_pixel_count": int(np.count_nonzero(removed)),
            "colored_water_policy": "always transparent",
        },
    }
    return final, provenance, removed


def _source_context_exclusion(
    rgb: np.ndarray,
    state_mask: np.ndarray,
    config: object,
) -> Tuple[np.ndarray, Dict[str, object] | None]:
    """Select source pixels whose color explicitly denotes non-data context."""

    if config is None:
        return np.zeros(state_mask.shape, dtype=bool), None
    if not isinstance(config, dict):
        raise ValueError("source_context_exclusion must be an object")
    blue = config.get("blue_dominant_water")
    if blue is None:
        return np.zeros(state_mask.shape, dtype=bool), None
    if not isinstance(blue, dict):
        raise ValueError("blue_dominant_water must be an object")
    channels = rgb.astype(np.int16)
    minimum_blue = int(blue.get("minimum_blue", 160))
    minimum_blue_over_red = int(blue.get("minimum_blue_over_red", 80))
    minimum_blue_over_green = int(blue.get("minimum_blue_over_green", 40))
    selected = state_mask & (channels[..., 2] >= minimum_blue) & (
        channels[..., 2] - channels[..., 0] >= minimum_blue_over_red
    ) & (channels[..., 2] - channels[..., 1] >= minimum_blue_over_green)
    return selected, {
        "method": "source_blue_dominance_marks_contextual_water_not_legend_data",
        "minimum_blue": minimum_blue,
        "minimum_blue_over_red": minimum_blue_over_red,
        "minimum_blue_over_green": minimum_blue_over_green,
        "source_pixel_count": int(np.count_nonzero(selected)),
        "policy": "source blue is transparent and may never seed or receive a class",
    }


def _restore_observed_class_evidence_from_external_water(
    publication_interior: np.ndarray,
    water_removed: np.ndarray | None,
    warped_classes: np.ndarray,
    enabled: bool,
) -> Tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Prevent external hydrography from deleting direct legend evidence."""

    restored = np.zeros(publication_interior.shape, dtype=bool)
    if not enabled or water_removed is None:
        return publication_interior, water_removed, restored
    restored = (warped_classes > 0) & water_removed
    if not np.any(restored):
        return publication_interior, water_removed, restored
    publication_interior = publication_interior | restored
    water_removed = water_removed & ~restored
    return publication_interior, water_removed, restored


def _save_layer_outputs(
    output_dir: Path,
    layer: Dict[str, object],
    source_values: np.ndarray,
    state,
    best: Dict[str, object],
    source_shape,
    water_exclusion: object = None,
    coastal_seam: object = None,
    source_context_exclusion: np.ndarray | None = None,
    source_context_report: Dict[str, object] | None = None,
) -> Tuple[Dict[str, object], np.ndarray]:
    layer_dir = output_dir / str(layer["id"])
    layer_dir.mkdir(parents=True, exist_ok=True)
    target_height = layer.get("target_height")
    warped, warp_report = warp_classified_to_web_mercator(
        source_values,
        state,
        best,
        source_shape,
        target_height=int(target_height) if target_height else None,
    )
    warped_before_context_clip = warped.copy()
    publication_interior, canonical_clip, water_removed = (
        _publication_interior_with_water_exclusion(
            warp_report,
            water_exclusion,
            coastal_seam,
        )
    )
    publication_interior, water_removed, restored_observed_water = (
        _restore_observed_class_evidence_from_external_water(
            publication_interior,
            water_removed,
            warped_before_context_clip,
            bool(layer.get("preserve_observed_class_pixels_against_external_water", False)),
        )
    )
    if np.any(restored_observed_water):
        internal = canonical_clip.setdefault("internal_water_exclusion", {})
        internal["observed_class_evidence_restoration"] = {
            "pixel_count": int(np.count_nonzero(restored_observed_water)),
            "policy": (
                "direct source pixels matching a legend class outrank external "
                "named or Mapbox hydrography"
            ),
        }
        internal["excluded_interior_pixel_count"] = int(
            np.count_nonzero(water_removed)
        )
        canonical_clip["valid_pixel_count"] = int(
            np.count_nonzero(publication_interior)
        )
    warped_source_context = np.zeros(warped.shape, dtype=bool)
    if source_context_exclusion is not None:
        warped_context, context_grid = warp_classified_to_web_mercator(
            source_context_exclusion.astype(np.uint8),
            state,
            best,
            source_shape,
            target_height=int(target_height) if target_height else None,
        )
        if context_grid != warp_report:
            raise ValueError("Source context and categorical warps use different grids")
        warped_source_context = (warped_context > 0) & publication_interior
        publication_interior &= ~warped_source_context
        water_removed = (
            warped_source_context
            if water_removed is None
            else (water_removed | warped_source_context)
        )
        internal = canonical_clip.setdefault("internal_water_exclusion", {})
        internal["source_context_exclusion"] = {
            **(source_context_report or {}),
            "web_mercator_pixel_count": int(np.count_nonzero(warped_source_context)),
        }
        internal["excluded_interior_pixel_count"] = int(
            np.count_nonzero(water_removed)
        )
        canonical_clip["valid_pixel_count"] = int(
            np.count_nonzero(publication_interior)
        )
    boundary_removed = (warped > 0) & ~publication_interior
    warped[~publication_interior] = 0
    completed_target_nodata = 0
    target_completion_mask = np.zeros(warped.shape, dtype=bool)
    if bool(layer.get("complete_target_state", False)):
        target_completion_mask = publication_interior & (warped == 0)
        warped, completed_target_nodata = _fill_indexed_nodata_in_mask(
            warped, publication_interior
        )
        target_completion_mask &= warped > 0
    speck_mask = np.zeros(warped.shape, dtype=bool)
    speck_original_classes = np.zeros(warped.shape, dtype=np.uint8)
    speck_report = None
    if bool(layer.get("suppress_isolated_specks", False)):
        warped, speck_mask, speck_original_classes, speck_report = (
            _suppress_isolated_class_specks(
                warped,
                layer["categories"],
                maximum_area=int(layer.get("maximum_speck_area", 16)),
                minimum_surrounding_purity=float(
                    layer.get("minimum_speck_surrounding_purity", 0.5)
                ),
                preserve_near_white=bool(
                    layer.get("preserve_near_white_specks", True)
                ),
            )
        )
    _save_indexed(layer_dir / "source-class-id.png", source_values)
    preclip_path = layer_dir / "web-mercator-class-id-before-context-clip.png"
    _save_indexed(preclip_path, warped_before_context_clip)
    _save_indexed(layer_dir / "web-mercator-class-id.png", warped)
    _save_indexed(
        layer_dir / "web-mercator-publication-interior-mask.png",
        publication_interior.astype(np.uint8) * 255,
    )
    _save_indexed(
        layer_dir / "web-mercator-boundary-removed-mask.png",
        boundary_removed.astype(np.uint8) * 255,
    )
    water_removed_path = layer_dir / "web-mercator-internal-water-mask.png"
    if water_removed is not None:
        _save_indexed(water_removed_path, water_removed.astype(np.uint8) * 255)
    restored_observed_water_path = (
        layer_dir / "web-mercator-observed-water-conflict-restored-mask.png"
    )
    if bool(layer.get("preserve_observed_class_pixels_against_external_water", False)):
        _save_indexed(
            restored_observed_water_path,
            restored_observed_water.astype(np.uint8) * 255,
        )
    target_completion_path = layer_dir / "web-mercator-target-completion-mask.png"
    if bool(layer.get("complete_target_state", False)):
        _save_indexed(
            target_completion_path,
            target_completion_mask.astype(np.uint8) * 255,
        )
    categories = layer["categories"]
    Image.fromarray(_preview(source_values, categories), mode="RGBA").save(
        layer_dir / "source-preview.png", optimize=True
    )
    Image.fromarray(_preview(warped, categories), mode="RGBA").save(
        layer_dir / "web-mercator-preview.png", optimize=True
    )
    category_counts = {
        category["id"]: int(np.count_nonzero(source_values == index))
        for index, category in enumerate(categories, 1)
    }
    web_category_counts = {
        category["id"]: int(np.count_nonzero(warped == index))
        for index, category in enumerate(categories, 1)
    }
    report = {
        "id": layer["id"],
        "kind": layer["kind"],
        "category_pixel_counts": category_counts,
        "web_mercator_category_pixel_counts": web_category_counts,
        "source_nodata_pixel_count": int(np.count_nonzero(source_values == 0)),
        "web_mercator_classified_pixel_count": int(np.count_nonzero(warped)),
        "web_mercator_completed_nodata_pixel_count": completed_target_nodata,
        "canonical_clip": {
            **canonical_clip,
            "colored_pixel_count_outside_boundary": int(
                np.count_nonzero(warped & ~publication_interior)
            ),
            "removed_pixel_count": int(np.count_nonzero(boundary_removed)),
            "artifacts": {
                "interior": {
                    "path": (
                        f"{layer['id']}/web-mercator-publication-interior-mask.png"
                    ),
                    "sha256": _sha256(
                        layer_dir / "web-mercator-publication-interior-mask.png"
                    ),
                },
                "removed": {
                    "path": f"{layer['id']}/web-mercator-boundary-removed-mask.png",
                    "sha256": _sha256(
                        layer_dir / "web-mercator-boundary-removed-mask.png"
                    ),
                },
            },
        },
        "warp": warp_report,
        "pre_context_clip_class_evidence": {
            "path": str(preclip_path.relative_to(output_dir)),
            "sha256": _sha256(preclip_path),
            "classified_pixel_count": int(
                np.count_nonzero(warped_before_context_clip)
            ),
        },
    }
    if water_removed is not None:
        report["canonical_clip"]["internal_water_exclusion"]["artifact"] = {
            "path": str(water_removed_path.relative_to(output_dir)),
            "sha256": _sha256(water_removed_path),
        }
        if source_context_report is not None:
            report["canonical_clip"]["internal_water_exclusion"][
                "source_context_exclusion"
            ]["artifact_role"] = "included_in_combined_internal_water_mask"
    if bool(layer.get("preserve_observed_class_pixels_against_external_water", False)):
        report["canonical_clip"].setdefault("internal_water_exclusion", {})[
            "observed_class_evidence_restoration"
        ] = {
            "pixel_count": int(np.count_nonzero(restored_observed_water)),
            "policy": (
                "direct source pixels matching a legend class outrank external "
                "named or Mapbox hydrography"
            ),
            "artifact": {
                "path": str(restored_observed_water_path.relative_to(output_dir)),
                "sha256": _sha256(restored_observed_water_path),
            },
        }
    if bool(layer.get("complete_target_state", False)):
        report["completion_artifacts"] = {
            "web_mercator_target_completion_mask": {
                "path": str(target_completion_path.relative_to(output_dir)),
                "sha256": _sha256(target_completion_path),
            }
        }
        report["web_mercator_completion_policy"] = (
            "nearest observed legend class inside the canonical publication interior; "
            "observed pixels remain unchanged"
        )
    if speck_report is not None:
        speck_mask_path = layer_dir / "web-mercator-speck-reassignment-mask.png"
        speck_original_path = layer_dir / "web-mercator-speck-original-class-id.png"
        _save_indexed(speck_mask_path, speck_mask.astype(np.uint8) * 255)
        _save_indexed(speck_original_path, speck_original_classes)
        report["speck_suppression"] = {
            **speck_report,
            "artifacts": {
                "mask": {
                    "path": str(speck_mask_path.relative_to(output_dir)),
                    "sha256": _sha256(speck_mask_path),
                },
                "original_class_id": {
                    "path": str(speck_original_path.relative_to(output_dir)),
                    "sha256": _sha256(speck_original_path),
                },
            },
        }
    return report, warped


def extract_from_plan(plan_path: Path, output_dir: Path) -> Dict[str, object]:
    plan = json.loads(plan_path.read_text())
    source_path = Path(plan["source"])
    alignment_path = Path(plan["alignment"])
    reference_root = Path(plan.get("reference", "reference/census-2025"))
    output_dir.mkdir(parents=True, exist_ok=True)
    rgb = np.asarray(Image.open(source_path).convert("RGB"))
    alignment = json.loads(alignment_path.read_text())
    if alignment.get("alignment_mode") == "assisted":
        best = {
            "projection": "assisted_reference_crs",
            "projection_crs": alignment["reference"]["crs"],
            "transform_model": alignment["transform_model"],
            "reference_to_source_matrix": alignment["reference_to_source_matrix"],
        }
        if "web_mercator_correction" in alignment:
            best["web_mercator_correction"] = alignment["web_mercator_correction"]
    else:
        best = dict(alignment["best"])
        if "web_mercator_correction" in alignment:
            best["web_mercator_correction"] = alignment["web_mercator_correction"]
    state, counties = load_california(reference_root)
    state_mask = _state_mask_in_source(
        state, best["projection_crs"], best, rgb.shape[:2]
    )
    Image.fromarray(state_mask.astype(np.uint8) * 255, mode="L").save(
        output_dir / "source-state-mask.png", optimize=True
    )
    source_context_mask, source_context_report = _source_context_exclusion(
        rgb, state_mask, plan.get("source_context_exclusion")
    )
    Image.fromarray(source_context_mask.astype(np.uint8) * 255, mode="L").save(
        output_dir / "source-context-exclusion-mask.png", optimize=True
    )
    warped_source, inspection_grid = warp_classified_to_web_mercator(
        rgb,
        state,
        best,
        rgb.shape[:2],
    )
    canonical_interior, _ = canonical_publication_interior(inspection_grid)
    if plan.get("coastal_seam") is not None:
        seam_config = plan["coastal_seam"]
        if not isinstance(seam_config, dict):
            raise ValueError("coastal_seam must be an object")
        canonical_interior, _, _ = close_west_coast_clipping_seam(
            canonical_interior,
            inspection_grid,
            maximum_gap_px=int(seam_config.get("maximum_gap_px", 50)),
            maximum_x_fraction=float(seam_config.get("maximum_x_fraction", 0.72)),
            start_y_fraction=float(seam_config.get("start_y_fraction", 0.04)),
            end_y_fraction=float(seam_config.get("end_y_fraction", 0.95)),
        )
    source_comparison = warped_source.copy()
    source_comparison[~canonical_interior] = 255
    Image.fromarray(source_comparison, mode="RGB").save(
        output_dir / "web-mercator-source-before-context-clip.png", optimize=True
    )
    publication_interior, canonical_clip_provenance, water_removed = (
        _publication_interior_with_water_exclusion(
            inspection_grid,
            plan.get("water_exclusion"),
            plan.get("coastal_seam"),
        )
    )
    warped_context, context_grid = warp_classified_to_web_mercator(
        source_context_mask.astype(np.uint8),
        state,
        best,
        rgb.shape[:2],
    )
    if context_grid != inspection_grid:
        raise ValueError("Review source-context and source warps use different grids")
    warped_context = (warped_context > 0) & publication_interior
    publication_interior &= ~warped_context
    water_removed = warped_context if water_removed is None else (water_removed | warped_context)
    if source_context_report is not None:
        internal = canonical_clip_provenance.setdefault("internal_water_exclusion", {})
        internal["source_context_exclusion"] = {
            **source_context_report,
            "web_mercator_pixel_count": int(np.count_nonzero(warped_context)),
        }
        internal["excluded_interior_pixel_count"] = int(np.count_nonzero(water_removed))
        canonical_clip_provenance["valid_pixel_count"] = int(
            np.count_nonzero(publication_interior)
        )
    source_boundary_removed = (
        np.any(warped_source != 255, axis=2) & ~publication_interior
    )
    # The alignment review must show the same declared data footprint as the
    # extracted classes. Otherwise named water can appear colored in the source
    # layer even though it is correctly transparent in the categorical layer.
    warped_source[~publication_interior] = 255
    Image.fromarray(publication_interior.astype(np.uint8) * 255).save(
        output_dir / "web-mercator-publication-interior-mask.png", optimize=True
    )
    Image.fromarray(source_boundary_removed.astype(np.uint8) * 255).save(
        output_dir / "web-mercator-source-boundary-removed-mask.png", optimize=True
    )
    water_removed_path = output_dir / "web-mercator-internal-water-mask.png"
    if water_removed is not None:
        Image.fromarray(water_removed.astype(np.uint8) * 255).save(
            water_removed_path, optimize=True
        )
    Image.fromarray(warped_source, mode="RGB").save(
        output_dir / "web-mercator-source.jpg", quality=94, subsampling=0
    )
    state_overlay, census_county_overlay, state_line_mask, census_county_line_mask = (
        _web_reference_assets(
            inspection_grid["bounds"], warped_source.shape[:2], state, counties
        )
    )
    configured_county_reference = plan.get("county_reference")
    county_diagnostic_enabled = configured_county_reference is not False
    county_reference_path = Path(
        str(configured_county_reference or DEFAULT_REGISTERED_COUNTY_REFERENCE)
    )
    county_reference_provenance = None
    if not county_diagnostic_enabled:
        county_overlay = np.zeros_like(census_county_overlay)
        county_line_mask = np.zeros_like(census_county_line_mask)
        county_reference_label = "not applicable for this source"
    elif county_reference_path.is_file():
        county_overlay, county_line_mask, county_reference_provenance = (
            _registered_county_reference_assets(
                county_reference_path,
                inspection_grid["bounds"],
                warped_source.shape[:2],
            )
        )
        county_reference_label = "registered thin county.png strokes"
    elif configured_county_reference:
        raise FileNotFoundError(
            f"Configured county reference does not exist: {county_reference_path}"
        )
    else:
        county_overlay = census_county_overlay
        county_line_mask = census_county_line_mask
        county_reference_label = "Census county geometry fallback"
    Image.fromarray(state_overlay, mode="RGBA").save(
        output_dir / "web-mercator-state-overlay.png", optimize=True
    )
    Image.fromarray(county_overlay, mode="RGBA").save(
        output_dir / "web-mercator-county-overlay.png", optimize=True
    )
    if county_diagnostic_enabled:
        county_residual_overlay, county_residual_report = _county_residual_diagnostic(
            warped_source,
            county_line_mask,
            state_line_mask,
            county_reference_label,
        )
    else:
        county_residual_overlay = np.zeros_like(census_county_overlay)
        county_residual_report = {
            "status": "not_applicable",
            "warning": (
                "County-line alignment is disabled because this source does not "
                "contain a usable county boundary network. State and coast evidence "
                "remain authoritative."
            ),
        }
    Image.fromarray(county_residual_overlay, mode="RGBA").save(
        output_dir / "web-mercator-county-residual.png", optimize=True
    )
    inspection = Image.fromarray(warped_source, mode="RGB").convert("RGBA")
    inspection = Image.alpha_composite(
        inspection, Image.fromarray(county_overlay, mode="RGBA")
    )
    inspection = Image.alpha_composite(
        inspection, Image.fromarray(state_overlay, mode="RGBA")
    ).convert("RGB")
    inspection.save(
        output_dir / "web-mercator-source-inspection.jpg", quality=94, subsampling=0
    )

    source_layers: Dict[str, np.ndarray] = {}
    layer_reports: List[Dict[str, object]] = []
    layer_definitions = {str(layer["id"]): layer for layer in plan["layers"]}

    # Overlay layers run first because their visible pixels are NoData for any
    # lower categorical layer they physically obscure.
    for layer in plan["layers"]:
        if layer["kind"] != "sparse_chroma_overlays":
            continue
        masks, extraction_report = infer_sparse_chroma_overlays(
            rgb,
            state_mask,
            layer["categories"],
            min_chroma=float(layer["min_chroma"]),
            coefficient_threshold=float(layer["coefficient_threshold"]),
            max_residual=float(layer["max_residual"]),
            complexity_penalty=float(layer["complexity_penalty"]),
        )
        values = np.zeros(rgb.shape[:2], dtype=np.uint8)
        # The indexed summary records one visible category per pixel, while
        # independent binary masks below preserve inferred overlaps.
        for index, mask in enumerate(masks, 1):
            values[(values == 0) & mask] = index
        report, _ = _save_layer_outputs(
            output_dir,
            layer,
            values,
            state,
            best,
            rgb.shape[:2],
            plan.get("water_exclusion"),
            plan.get("coastal_seam"),
        )
        report["extraction"] = extraction_report
        report["independent_masks"] = []
        layer_dir = output_dir / str(layer["id"])
        for index, (category, mask) in enumerate(zip(layer["categories"], masks), 1):
            binary_layer = {
                "id": f"{layer['id']}-{category['id']}",
                "kind": "binary_overlay",
                "categories": [category],
            }
            binary_values = mask.astype(np.uint8)
            binary_report, _ = _save_layer_outputs(
                layer_dir,
                binary_layer,
                binary_values,
                state,
                best,
                rgb.shape[:2],
                plan.get("water_exclusion"),
                plan.get("coastal_seam"),
            )
            report["independent_masks"].append(binary_report)
        source_layers[str(layer["id"])] = np.logical_or.reduce(masks)
        layer_reports.append(report)

    for layer in plan["layers"]:
        kind = layer["kind"]
        if kind == "sparse_chroma_overlays":
            continue
        exclusions = np.zeros(rgb.shape[:2], dtype=bool)
        exclusions |= source_context_mask
        for excluded_layer in layer.get("exclude_visible_layers", []):
            exclusions |= source_layers[str(excluded_layer)]
        if kind == "categorical":
            values, extraction_report = classify_categorical(
                rgb,
                state_mask,
                layer["categories"],
                max_distance=float(layer["max_distance"]),
                min_margin=float(layer["min_margin"]),
                exclusion_mask=exclusions,
            )
        elif kind == "patterned_categorical":
            values, extraction_report = classify_patterned_categorical(
                rgb,
                state_mask,
                layer["categories"],
                window_size=int(layer["window_size"]),
                max_distance=float(layer["max_distance"]),
                min_margin=float(layer["min_margin"]),
                exclusion_mask=exclusions,
                prototype_stride=int(layer.get("prototype_stride", 1)),
                max_color_distance=float(layer.get("max_color_distance", 3.0)),
                min_color_margin=float(layer.get("min_color_margin", 1.0)),
                complete_palette=bool(layer.get("complete_palette", False)),
                histogram_window=int(layer.get("histogram_window", 9)),
                histogram_max_distance=float(
                    layer.get("histogram_max_distance", 0.7)
                ),
                histogram_min_margin=float(layer.get("histogram_min_margin", 0.05)),
                preserve_dark_luminance=int(
                    layer.get("preserve_dark_luminance", 80)
                ),
                preserve_dark_ink=bool(layer.get("preserve_dark_ink", True)),
            )
        elif kind == "dot_pattern":
            values, extraction_report = classify_dot_pattern(
                rgb,
                state_mask,
                layer["categories"],
                max_distance=float(layer["max_distance"]),
                maximum_component_area=int(layer["maximum_component_area"]),
                maximum_component_dimension=int(layer["maximum_component_dimension"]),
                pattern_radius=int(layer["pattern_radius"]),
                minimum_quadrants=int(layer["minimum_quadrants"]),
                minimum_quadrant_pixels=int(layer["minimum_quadrant_pixels"]),
                density_window=int(layer["density_window"]),
                minimum_window_pixels=int(layer["minimum_window_pixels"]),
            )
        elif kind == "grayscale_categorical":
            values, extraction_report = classify_grayscale(
                rgb,
                state_mask,
                layer["categories"],
                smoothing_radius=int(layer["smoothing_radius"]),
                max_distance=float(layer["max_distance"]),
                max_chroma=float(layer["max_chroma"]),
                exclusion_mask=exclusions,
                adaptive_centers=bool(layer.get("adaptive_centers", False)),
                background_cutoff=int(layer.get("background_cutoff", 245)),
            )
        else:
            raise ValueError(f"Unsupported layer kind: {kind}")
        source_completion_mask = extraction_report.pop(
            "_source_completion_mask", None
        )
        source_preserved_dark_mask = extraction_report.pop(
            "_source_preserved_dark_mask", None
        )
        report, _ = _save_layer_outputs(
            output_dir,
            layer,
            values,
            state,
            best,
            rgb.shape[:2],
            plan.get("water_exclusion"),
            plan.get("coastal_seam"),
            source_context_mask,
            source_context_report,
        )
        report["extraction"] = extraction_report
        if source_completion_mask is not None:
            layer_dir = output_dir / str(layer["id"])
            completion_source_path = layer_dir / "source-completion-mask.png"
            completion_web_path = layer_dir / "web-mercator-completion-mask.png"
            dark_source_path = layer_dir / "source-preserved-dark-ink-mask.png"
            dark_web_path = layer_dir / "web-mercator-preserved-dark-ink-mask.png"
            _save_indexed(
                completion_source_path,
                np.asarray(source_completion_mask, dtype=np.uint8) * 255,
            )
            _save_indexed(
                dark_source_path,
                np.asarray(source_preserved_dark_mask, dtype=np.uint8) * 255,
            )
            warped_completion, _ = warp_classified_to_web_mercator(
                np.asarray(source_completion_mask, dtype=np.uint8),
                state,
                best,
                rgb.shape[:2],
                target_height=int(layer["target_height"])
                if layer.get("target_height")
                else None,
            )
            warped_dark, _ = warp_classified_to_web_mercator(
                np.asarray(source_preserved_dark_mask, dtype=np.uint8),
                state,
                best,
                rgb.shape[:2],
                target_height=int(layer["target_height"])
                if layer.get("target_height")
                else None,
            )
            _save_indexed(completion_web_path, warped_completion * 255)
            _save_indexed(dark_web_path, warped_dark * 255)
            report.setdefault("completion_artifacts", {}).update({
                "source_completion_mask": {
                    "path": str(completion_source_path.relative_to(output_dir)),
                    "sha256": _sha256(completion_source_path),
                },
                "web_mercator_completion_mask": {
                    "path": str(completion_web_path.relative_to(output_dir)),
                    "sha256": _sha256(completion_web_path),
                },
                "source_preserved_dark_ink_mask": {
                    "path": str(dark_source_path.relative_to(output_dir)),
                    "sha256": _sha256(dark_source_path),
                },
                "web_mercator_preserved_dark_ink_mask": {
                    "path": str(dark_web_path.relative_to(output_dir)),
                    "sha256": _sha256(dark_web_path),
                },
            })
        source_layers[str(layer["id"])] = values > 0
        layer_reports.append(report)

    # A layer that preserves observed legend evidence against external water
    # owns the reviewer footprint. Rebuild the review source from the lossless
    # canonical source so the Source/Extracted flip cannot show an obsolete,
    # more aggressive water cut.
    review_layer = next(
        (
            report
            for report in layer_reports
            if layer_definitions[str(report["id"])].get(
                "preserve_observed_class_pixels_against_external_water", False
            )
        ),
        None,
    )
    if review_layer is not None:
        review_layer_dir = output_dir / str(review_layer["id"])
        canonical_clip_provenance = dict(review_layer["canonical_clip"])
        review_interior = (
            np.asarray(
                Image.open(
                    review_layer_dir / "web-mercator-publication-interior-mask.png"
                ).convert("L")
            )
            > 0
        )
        review_source = source_comparison.copy()
        review_source[~review_interior] = 255
        Image.fromarray(review_source, mode="RGB").save(
            output_dir / "web-mercator-source.jpg", quality=94, subsampling=0
        )
        review_source_removed = (
            np.any(source_comparison != 255, axis=2) & ~review_interior
        )
        Image.fromarray(review_source_removed.astype(np.uint8) * 255).save(
            output_dir / "web-mercator-source-boundary-removed-mask.png",
            optimize=True,
        )
        shutil.copyfile(
            review_layer_dir / "web-mercator-publication-interior-mask.png",
            output_dir / "web-mercator-publication-interior-mask.png",
        )
        review_water_path = review_layer_dir / "web-mercator-internal-water-mask.png"
        if review_water_path.is_file():
            shutil.copyfile(
                review_water_path,
                output_dir / "web-mercator-internal-water-mask.png",
            )
        if county_diagnostic_enabled:
            county_residual_overlay, county_residual_report = (
                _county_residual_diagnostic(
                    review_source,
                    county_line_mask,
                    state_line_mask,
                    county_reference_label,
                )
            )
            Image.fromarray(county_residual_overlay, mode="RGBA").save(
                output_dir / "web-mercator-county-residual.png", optimize=True
            )
        inspection = Image.fromarray(review_source, mode="RGB").convert("RGBA")
        inspection = Image.alpha_composite(
            inspection, Image.fromarray(county_overlay, mode="RGBA")
        )
        inspection = Image.alpha_composite(
            inspection, Image.fromarray(state_overlay, mode="RGBA")
        ).convert("RGB")
        inspection.save(
            output_dir / "web-mercator-source-inspection.jpg",
            quality=94,
            subsampling=0,
        )

    warnings = list(plan.get("warnings", []))
    manifest = {
        "schema_version": 1,
        "status": "needs_visual_review",
        "dataset_id": plan["dataset_id"],
        "title": plan["title"],
        "source": {
            "path": str(source_path),
            "sha256": _sha256(source_path),
            "width": rgb.shape[1],
            "height": rgb.shape[0],
        },
        "plan": {"path": str(plan_path), "sha256": _sha256(plan_path)},
        "alignment": {
            "path": str(alignment_path),
            "source_sha256": alignment["source"].get("sha256"),
            "mode": alignment.get("alignment_mode", "automatic"),
            "projection": best["projection"],
            "projection_crs": best["projection_crs"],
            "diagnostic_status": alignment["status"],
            "metrics": alignment.get("metrics", best.get("metrics", {})),
            "inspection": {
                "path": "web-mercator-source-inspection.jpg",
                "state_line": "cyan",
                "county_lines": county_reference_label,
                "grid": inspection_grid,
            },
        },
        "review": {
            "assets": {
                "source": "web-mercator-source.jpg",
                "state_overlay": "web-mercator-state-overlay.png",
                "county_overlay": "web-mercator-county-overlay.png",
                "county_residual": "web-mercator-county-residual.png",
            },
            "county_reference": county_reference_provenance,
            "county_diagnostic_enabled": county_diagnostic_enabled,
            "canonical_clip": {
                **canonical_clip_provenance,
                "source_pixel_count_removed": int(
                    np.count_nonzero(source_boundary_removed)
                ),
                "source_pixels_outside_after_clip": int(
                    np.count_nonzero(
                        np.any(warped_source != 255, axis=2)
                        & ~canonical_interior
                    )
                ),
                "artifacts": {
                    "interior": {
                        "path": "web-mercator-publication-interior-mask.png",
                        "sha256": _sha256(
                            output_dir / "web-mercator-publication-interior-mask.png"
                        ),
                    },
                    "removed_source": {
                        "path": "web-mercator-source-boundary-removed-mask.png",
                        "sha256": _sha256(
                            output_dir / "web-mercator-source-boundary-removed-mask.png"
                        ),
                    },
                },
            },
            "county_residual": county_residual_report,
            "default_view": str(plan.get("review_default_view", "source")),
            "decision_path": "review-decision.json",
            "status": "not_reviewed",
        },
        "layers": layer_reports,
        "warnings": warnings,
    }
    if water_removed is not None:
        manifest["review"]["canonical_clip"]["internal_water_exclusion"][
            "artifact"
        ] = {
            "path": "web-mercator-internal-water-mask.png",
            "sha256": _sha256(water_removed_path),
        }
    (output_dir / "extraction.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output_dir / "plan.snapshot.json").write_text(json.dumps(plan, indent=2) + "\n")
    return manifest
