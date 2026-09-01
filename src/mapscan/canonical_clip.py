"""Rasterize the versioned publication interior for any MapScan target grid.

The active lime border is detailed line evidence and must never be rebuilt from
a fill. Clipping is a separate contract: a pinned California mainland polygon
supplies the mainland interior, while the active county-detail package supplies
the four author-approved island interiors.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image
from pyproj import Transformer


DEFAULT_MAINLAND_CLIP_MANIFEST = Path(
    "reference/canonical-california-clipping-v2/canonical-clipping.json"
)
DEFAULT_ACTIVE_CANONICAL_POINTER = Path(
    "reference/canonical-california-boundary.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text())


def _verified(path: Path, expected: object, label: str) -> Path:
    if not path.is_file() or _sha256(path) != str(expected):
        raise ValueError(f"Canonical {label} is missing or stale: {path}")
    return path


def _active_mainland_line(
    target_grid: Dict[str, object], active_pointer_path: Path
) -> Tuple[np.ndarray, Path, Dict[str, object]]:
    """Load the hash-verified active mainland line on ``target_grid``."""

    active_pointer_path = active_pointer_path.resolve()
    pointer = _load(active_pointer_path)
    active_record = pointer["manifest"]
    active_manifest_path = _verified(
        active_pointer_path.parent / str(active_record["path"]),
        active_record["sha256"],
        "active boundary manifest",
    )
    active = _load(active_manifest_path)
    active_grid = active["source_grid"]
    for field in ("crs", "bounds"):
        if active_grid.get(field) != target_grid.get(field):
            raise ValueError(f"Active canonical boundary differs at {field}")
    mainland_record = active["artifacts"]["mainland"]
    mainland_path = _verified(
        active_manifest_path.parent / str(mainland_record["path"]),
        mainland_record["sha256"],
        "mainland linework",
    )
    line_high = np.asarray(Image.open(mainland_path).convert("RGBA"))[..., 3] > 0
    height, width = int(target_grid["height"]), int(target_grid["width"])
    line = cv2.resize(
        line_high.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
    ) > 0
    return line, active_manifest_path, active


def _project_ring(
    coordinates: Sequence[Sequence[float]], grid: Dict[str, object]
) -> np.ndarray:
    transformer = Transformer.from_crs("EPSG:4326", str(grid["crs"]), always_xy=True)
    lon = np.asarray([item[0] for item in coordinates], dtype=np.float64)
    lat = np.asarray([item[1] for item in coordinates], dtype=np.float64)
    projected_x, projected_y = transformer.transform(lon, lat)
    min_x, min_y, max_x, max_y = (float(value) for value in grid["bounds"])
    width, height = int(grid["width"]), int(grid["height"])
    pixel_x = (np.asarray(projected_x) - min_x) / (max_x - min_x) * width - 0.5
    pixel_y = (max_y - np.asarray(projected_y)) / (max_y - min_y) * height - 0.5
    return np.rint(np.column_stack([pixel_x, pixel_y])).astype(np.int32)


def rasterize_polygon_geometry(
    geometry: Dict[str, object], grid: Dict[str, object]
) -> np.ndarray:
    """Rasterize one WGS84 Polygon, preserving any declared interior holes."""

    if geometry.get("type") != "Polygon":
        raise ValueError("Canonical mainland clipping geometry must be a Polygon")
    coordinates = geometry.get("coordinates")
    if not isinstance(coordinates, (list, tuple)) or not coordinates:
        raise ValueError("Canonical mainland clipping polygon has no rings")
    height, width = int(grid["height"]), int(grid["width"])
    raster = np.zeros((height, width), dtype=np.uint8)
    exterior = _project_ring(coordinates[0], grid)
    cv2.fillPoly(raster, [exterior], 1, lineType=cv2.LINE_8)
    for hole in coordinates[1:]:
        cv2.fillPoly(raster, [_project_ring(hole, grid)], 0, lineType=cv2.LINE_8)
    return raster > 0


def _closed_component_interiors(
    line_mask: np.ndarray, expected_components: int
) -> np.ndarray:
    """Fill closed line components without connecting distinct islands."""

    count, labels = cv2.connectedComponents(line_mask.astype(np.uint8), 8)
    if count - 1 != expected_components:
        raise ValueError(
            "Active canonical island linework has an unexpected component count"
        )
    result = np.zeros(line_mask.shape, dtype=bool)
    for component in range(1, count):
        selected = labels == component
        contours, hierarchy = cv2.findContours(
            selected.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if hierarchy is None or not contours:
            raise ValueError("Canonical island outline is not fillable")
        filled = np.zeros(line_mask.shape, dtype=np.uint8)
        cv2.drawContours(filled, contours, -1, 1, cv2.FILLED, cv2.LINE_8)
        result |= filled > 0
    return result


def active_boundary_publication_interior(
    target_grid: Dict[str, object],
    *,
    active_pointer_path: Path = DEFAULT_ACTIVE_CANONICAL_POINTER,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Fill the approved mainland ring and islands without rebuilding its display line."""

    active_pointer_path = active_pointer_path.resolve()
    pointer = _load(active_pointer_path)
    active_record = pointer["manifest"]
    active_manifest_path = _verified(
        active_pointer_path.parent / str(active_record["path"]),
        active_record["sha256"],
        "active boundary manifest",
    )
    active = _load(active_manifest_path)
    if active.get("canonical_boundary_id") != pointer.get("canonical_boundary_id"):
        raise ValueError("Active canonical pointer and manifest identifiers differ")
    active_grid = active["source_grid"]
    for field in ("crs", "bounds"):
        if active_grid.get(field) != target_grid.get(field):
            raise ValueError(f"Active canonical boundary differs at {field}")

    expected_shape = (
        int(active_grid["height"]),
        int(active_grid["width"]),
    )
    mainland_record = active["artifacts"]["mainland"]
    mainland_path = _verified(
        active_manifest_path.parent / str(mainland_record["path"]),
        mainland_record["sha256"],
        "mainland linework",
    )
    mainland_rgba = np.asarray(Image.open(mainland_path).convert("RGBA"))
    if mainland_rgba.shape[:2] != expected_shape:
        raise ValueError("Active canonical mainland dimensions are stale")
    mainland_high = _closed_component_interiors(mainland_rgba[..., 3] > 0, 1)

    island_count = int(active["topology"]["offshore_island_component_count"])
    island_record = active["artifacts"]["islands"]
    island_path = _verified(
        active_manifest_path.parent / str(island_record["path"]),
        island_record["sha256"],
        "island linework",
    )
    island_rgba = np.asarray(Image.open(island_path).convert("RGBA"))
    if island_rgba.shape[:2] != expected_shape:
        raise ValueError("Active canonical island dimensions are stale")
    island_high = _closed_component_interiors(
        island_rgba[..., 3] > 0, island_count
    )

    height, width = int(target_grid["height"]), int(target_grid["width"])
    mainland_mask = cv2.resize(
        mainland_high.astype(np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    ) > 0
    island_mask = cv2.resize(
        island_high.astype(np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    ) > 0
    if np.any(mainland_mask & island_mask):
        raise ValueError("Active canonical islands overlap the mainland interior")
    valid = mainland_mask | island_mask
    component_count = cv2.connectedComponents(valid.astype(np.uint8), 8)[0] - 1
    if component_count != 1 + island_count:
        raise ValueError("Active boundary publication components merged or vanished")

    provenance = {
        "method": "pinned_active_mainland_ring_fill_plus_active_islands",
        "active_pointer": {
            "path": str(active_pointer_path),
            "sha256": _sha256(active_pointer_path),
        },
        "active_manifest": {
            "path": str(active_manifest_path),
            "sha256": _sha256(active_manifest_path),
            "canonical_boundary_id": active["canonical_boundary_id"],
        },
        "component_count": component_count,
        "mainland_pixel_count": int(np.count_nonzero(mainland_mask)),
        "island_pixel_count": int(np.count_nonzero(island_mask)),
        "valid_pixel_count": int(np.count_nonzero(valid)),
        "display_border_reconstructed_from_fill": False,
        "clipping_policy": (
            "The approved high-resolution mainland ring is filled before target-grid "
            "resampling; the original hash-bound line remains the display authority."
        ),
    }
    return valid, provenance


def canonical_publication_interior(
    target_grid: Dict[str, object],
    *,
    mainland_manifest_path: Path = DEFAULT_MAINLAND_CLIP_MANIFEST,
    active_pointer_path: Path = DEFAULT_ACTIVE_CANONICAL_POINTER,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Return mainland plus active island interiors on ``target_grid``."""

    mainland_manifest_path = mainland_manifest_path.resolve()
    active_pointer_path = active_pointer_path.resolve()
    mainland_manifest = _load(mainland_manifest_path)
    if (
        mainland_manifest.get("status") != "pinned_pipeline_reference"
        or mainland_manifest.get("kind") != "clipping_interior"
        or mainland_manifest.get("canonical_clipping_id")
        != "california-mainland-clipping-v2"
    ):
        raise ValueError("Canonical mainland clipping reference is not pinned")
    geojson_record = mainland_manifest["artifacts"]["geojson"]
    geojson_path = _verified(
        mainland_manifest_path.parent / str(geojson_record["path"]),
        geojson_record["sha256"],
        "mainland GeoJSON",
    )
    feature_collection = _load(geojson_path)
    features = feature_collection.get("features", [])
    if feature_collection.get("type") != "FeatureCollection" or len(features) != 1:
        raise ValueError("Canonical mainland clipping GeoJSON has unexpected features")
    mainland_mask = rasterize_polygon_geometry(features[0]["geometry"], target_grid)
    height, width = int(target_grid["height"]), int(target_grid["width"])
    if cv2.connectedComponents(mainland_mask.astype(np.uint8), 8)[0] - 1 != 1:
        raise ValueError("Canonical mainland clip is not one target-grid component")

    pointer = _load(active_pointer_path)
    active_record = pointer["manifest"]
    active_manifest_path = _verified(
        active_pointer_path.parent / str(active_record["path"]),
        active_record["sha256"],
        "active boundary manifest",
    )
    active = _load(active_manifest_path)
    if active.get("canonical_boundary_id") != pointer.get("canonical_boundary_id"):
        raise ValueError("Active canonical pointer and manifest identifiers differ")
    active_grid = active["source_grid"]
    for field in ("crs", "bounds"):
        if active_grid.get(field) != target_grid.get(field):
            raise ValueError(f"Active canonical boundary differs at {field}")
    island_count = int(active["topology"]["offshore_island_component_count"])
    island_record = active["artifacts"]["islands"]
    island_path = _verified(
        active_manifest_path.parent / str(island_record["path"]),
        island_record["sha256"],
        "island linework",
    )
    island_rgba = np.asarray(Image.open(island_path).convert("RGBA"))
    if island_rgba.shape[:2] != (
        int(active_grid["height"]),
        int(active_grid["width"]),
    ):
        raise ValueError("Active canonical island dimensions are stale")
    island_high = _closed_component_interiors(
        island_rgba[..., 3] > 0, island_count
    )
    island_mask = cv2.resize(
        island_high.astype(np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    ) > 0
    valid = mainland_mask | island_mask
    component_count = cv2.connectedComponents(valid.astype(np.uint8), 8)[0] - 1
    if component_count != 1 + island_count:
        raise ValueError("Canonical publication interior components merged or vanished")

    provenance = {
        "method": "pinned_mainland_clip_plus_active_county_detail_islands",
        "mainland_manifest": {
            "path": str(mainland_manifest_path),
            "sha256": _sha256(mainland_manifest_path),
            "canonical_clipping_id": mainland_manifest["canonical_clipping_id"],
        },
        "active_pointer": {
            "path": str(active_pointer_path),
            "sha256": _sha256(active_pointer_path),
        },
        "active_manifest": {
            "path": str(active_manifest_path),
            "sha256": _sha256(active_manifest_path),
            "canonical_boundary_id": active["canonical_boundary_id"],
        },
        "component_count": component_count,
        "mainland_pixel_count": int(np.count_nonzero(mainland_mask)),
        "island_pixel_count": int(np.count_nonzero(island_mask)),
        "valid_pixel_count": int(np.count_nonzero(valid)),
        "display_border_reconstructed_from_fill": False,
    }
    return valid, provenance


def close_west_coast_clipping_seam(
    interior: np.ndarray,
    target_grid: Dict[str, object],
    *,
    maximum_gap_px: int = 50,
    maximum_x_fraction: float = 0.72,
    start_y_fraction: float = 0.04,
    end_y_fraction: float = 0.95,
    active_pointer_path: Path = DEFAULT_ACTIVE_CANONICAL_POINTER,
) -> Tuple[np.ndarray, Dict[str, object], np.ndarray]:
    """Close only narrow row-wise gaps between the clip and active west coast.

    The canonical display line remains authoritative and unchanged. Large gaps
    such as open bays are deliberately skipped; dataset-specific water masks
    remain responsible for internal named water.
    """

    if maximum_gap_px < 1:
        raise ValueError("West-coast seam maximum gap must be positive")
    if not 0.0 < maximum_x_fraction < 1.0:
        raise ValueError("West-coast seam x fraction must be between zero and one")
    if not 0.0 <= start_y_fraction < end_y_fraction <= 1.0:
        raise ValueError("West-coast seam y fractions are invalid")

    line, active_manifest_path, active = _active_mainland_line(
        target_grid, active_pointer_path
    )
    height, width = int(target_grid["height"]), int(target_grid["width"])
    if interior.shape != (height, width):
        raise ValueError("West-coast seam interior differs from the target grid")

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        interior.astype(np.uint8), 8
    )
    if component_count < 2:
        raise ValueError("West-coast seam interior has no mainland component")
    mainland_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mainland = labels == mainland_label
    seam = np.zeros_like(interior, dtype=bool)
    maximum_x = round(width * maximum_x_fraction)
    start_y = round(height * start_y_fraction)
    end_y = round(height * end_y_fraction)
    coast_by_row = np.full(height, np.nan, dtype=np.float64)
    rows_with_line = []
    for y in range(start_y, end_y):
        coast_x = np.flatnonzero(line[y, :maximum_x])
        if len(coast_x):
            coast_by_row[y] = float(coast_x.min())
            rows_with_line.append(y)
    if len(rows_with_line) < 2:
        raise ValueError("Active west coast has too few rows for seam interpolation")
    first_line_y, last_line_y = rows_with_line[0], rows_with_line[-1]
    interpolation_rows = np.arange(first_line_y, last_line_y + 1)
    interpolated = np.interp(
        interpolation_rows,
        np.asarray(rows_with_line, dtype=np.float64),
        coast_by_row[rows_with_line],
    )
    missing_before = np.isnan(coast_by_row[interpolation_rows])
    coast_by_row[interpolation_rows] = interpolated

    accepted_gaps = []
    for y in range(first_line_y, last_line_y + 1):
        mainland_x = np.flatnonzero(mainland[y])
        if not len(mainland_x):
            continue
        coast_west = int(np.rint(coast_by_row[y]))
        clip_west = int(mainland_x.min())
        gap = clip_west - coast_west
        if 0 < gap <= maximum_gap_px:
            seam[y, coast_west : clip_west + 1] = True
            accepted_gaps.append(gap)
    expanded = interior | seam
    report = {
        "method": "interpolated_rowwise_active_lime_west_coast_seam",
        "maximum_gap_px": int(maximum_gap_px),
        "maximum_x_fraction": float(maximum_x_fraction),
        "start_y_fraction": float(start_y_fraction),
        "end_y_fraction": float(end_y_fraction),
        "accepted_row_count": int(len(accepted_gaps)),
        "canonical_line_row_count": int(len(rows_with_line)),
        "interpolated_line_row_count": int(np.count_nonzero(missing_before)),
        "added_pixel_count": int(np.count_nonzero(seam & ~interior)),
        "accepted_gap_median_px": (
            float(np.median(accepted_gaps)) if accepted_gaps else 0.0
        ),
        "accepted_gap_max_px": int(max(accepted_gaps, default=0)),
        "large_opening_policy": "gaps above the limit remain unchanged",
        "active_manifest": {
            "path": str(active_manifest_path),
            "sha256": _sha256(active_manifest_path),
            "canonical_boundary_id": active["canonical_boundary_id"],
        },
    }
    return expanded, report, seam


def snap_named_water_to_active_boundary(
    water_seed: np.ndarray,
    eligible_interior: np.ndarray,
    target_grid: Dict[str, object],
    *,
    maximum_distance_px: float = 40.0,
    active_pointer_path: Path = DEFAULT_ACTIVE_CANONICAL_POINTER,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Snap a named Census water seed to the approved lime shoreline.

    Census identifies *which* water body is eligible. The image-edge-connected
    side of the active mainland line identifies *where* its shoreline ends.
    Limiting the result to a narrow distance from the named Census seed prevents
    an open bay entrance from turning the entire Pacific exterior into water.
    """

    if maximum_distance_px <= 0:
        raise ValueError("Canonical shoreline snap distance must be positive")
    height, width = int(target_grid["height"]), int(target_grid["width"])
    expected_shape = (height, width)
    if water_seed.shape != expected_shape or eligible_interior.shape != expected_shape:
        raise ValueError("Canonical shoreline snap masks differ from the target grid")
    water_seed = water_seed.astype(bool)
    eligible_interior = eligible_interior.astype(bool)
    if not np.any(water_seed):
        raise ValueError("Canonical shoreline snap has no named water seed")

    line, active_manifest_path, active = _active_mainland_line(
        target_grid, active_pointer_path
    )
    _, labels = cv2.connectedComponents((~line).astype(np.uint8), connectivity=4)
    edge_labels = np.unique(
        np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1]))
    )
    edge_labels = edge_labels[edge_labels != 0]
    exterior = np.isin(labels, edge_labels)
    distance = cv2.distanceTransform(
        (~water_seed).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_5
    )
    permitted = distance <= float(maximum_distance_px)
    snapped = exterior & permitted & eligible_interior
    overlap = snapped & water_seed
    if not np.any(overlap):
        raise ValueError("Named water does not meet the active lime shoreline exterior")

    report = {
        "method": "named_census_water_snapped_to_active_lime_exterior",
        "maximum_distance_px": float(maximum_distance_px),
        "seed_pixel_count": int(np.count_nonzero(water_seed)),
        "seed_pixel_count_inside_eligible_interior": int(
            np.count_nonzero(water_seed & eligible_interior)
        ),
        "snapped_pixel_count": int(np.count_nonzero(snapped)),
        "retained_seed_pixel_count": int(np.count_nonzero(overlap)),
        "added_to_reach_lime_pixel_count": int(
            np.count_nonzero(snapped & ~water_seed)
        ),
        "removed_landward_seed_pixel_count": int(
            np.count_nonzero(water_seed & eligible_interior & ~snapped)
        ),
        "active_lime_line_pixel_count": int(np.count_nonzero(line)),
        "exterior_policy": (
            "Only the image-edge-connected side of the exact active mainland line is "
            "eligible, and only within the configured distance of the named Census seed."
        ),
        "active_manifest": {
            "path": str(active_manifest_path),
            "sha256": _sha256(active_manifest_path),
            "canonical_boundary_id": active["canonical_boundary_id"],
        },
    }
    return snapped, report
