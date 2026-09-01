"""Promote and rasterize the single author-approved California mainland boundary."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Sequence

import cv2
import numpy as np
from PIL import Image
from pyproj import Transformer
from scipy.spatial import cKDTree

from .extraction import warp_classified_to_web_mercator
from .reference import load_california


CANONICAL_BOUNDARY_ID = "california-mainland-hybrid-v1"
COUNTY_DETAIL_BOUNDARY_CANDIDATE_ID = "california-county-detail-boundary-v2"
CANONICAL_BORDER_ID = "california-county-detail-border-v2"
ACTIVE_CANONICAL_POINTER = Path("reference/canonical-california-boundary.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text())


def _verified(path: Path, expected: object, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    if expected and _sha256(path) != str(expected):
        raise ValueError(f"Stale {label}: {path}")
    return path


def _mask_from_rgba(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGBA"))[..., 3] > 0


def _wgs84_bbox_slice(
    grid: Dict[str, object], bbox: Sequence[float]
) -> tuple[slice, slice]:
    """Convert a WGS84 audit box to a clipped raster window."""

    west, south, east, north = (float(value) for value in bbox)
    transformer = Transformer.from_crs("EPSG:4326", str(grid["crs"]), always_xy=True)
    projected_west, projected_south = transformer.transform(west, south)
    projected_east, projected_north = transformer.transform(east, north)
    min_x, min_y, max_x, max_y = (float(value) for value in grid["bounds"])
    width, height = int(grid["width"]), int(grid["height"])

    def pixel_x(value: float) -> int:
        return int(round((value - min_x) / (max_x - min_x) * width - 0.5))

    def pixel_y(value: float) -> int:
        return int(round((max_y - value) / (max_y - min_y) * height - 0.5))

    left, right = sorted((pixel_x(projected_west), pixel_x(projected_east)))
    top, bottom = sorted((pixel_y(projected_south), pixel_y(projected_north)))
    return (
        slice(max(0, top), min(height, bottom + 1)),
        slice(max(0, left), min(width, right + 1)),
    )


def _county_island_components(rgba: np.ndarray) -> tuple[np.ndarray, list[Dict[str, object]]]:
    """Select the four Channel Islands drawn with the county.png state stroke."""

    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError("county.png island extraction requires RGBA input")
    alpha = rgba[..., 3]
    luminance = np.mean(rgba[..., :3].astype(np.float32), axis=2)
    state_core = (alpha >= 180) & (luminance <= 30)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        state_core.astype(np.uint8), 8
    )
    if count <= 1:
        raise ValueError("county.png has no state-stroke components")
    mainland = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    main_x, main_y, main_width, main_height, _ = stats[mainland].tolist()
    selected = np.zeros(state_core.shape, dtype=bool)
    records: list[Dict[str, object]] = []
    for component in range(1, count):
        if component == mainland:
            continue
        x, y, width, height, area = stats[component].tolist()
        center_x, center_y = centroids[component].tolist()
        relative_x = (center_x - main_x) / max(main_width, 1)
        relative_y = (center_y - main_y) / max(main_height, 1)
        # This rejects the bottom-right source watermark and any stray county
        # fragments while retaining the four thick, closed offshore outlines.
        if not (
            area >= 400
            and width >= 40
            and height >= 30
            and 0.30 <= relative_x <= 0.70
            and relative_y >= 0.75
        ):
            continue
        selected |= labels == component
        records.append(
            {
                "source_component": int(component),
                "source_pixel_count": int(area),
                "source_bbox": [int(x), int(y), int(width), int(height)],
                "source_centroid": [float(center_x), float(center_y)],
            }
        )
    if len(records) != 4:
        raise ValueError(
            f"Expected four county.png offshore islands, found {len(records)}"
        )
    return selected, records


def _connect_nearby_components(
    mask: np.ndarray, *, maximum_gap_px: float
) -> tuple[np.ndarray, list[Dict[str, object]]]:
    """Close every short authority seam without joining a line to itself."""

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), 8
    )
    components = []
    for component in range(1, count):
        y, x = np.nonzero(labels == component)
        components.append((component, np.column_stack([x, y])))
    connected = mask.copy().astype(np.uint8)
    joins: list[Dict[str, object]] = []
    for index, (left_id, left) in enumerate(components):
        for right_id, right in components[index + 1 :]:
            tree = cKDTree(right)
            distances, right_indices = tree.query(left, k=1)
            left_index = int(np.argmin(distances))
            distance = float(distances[left_index])
            if distance > maximum_gap_px:
                continue
            start = tuple(int(value) for value in left[left_index])
            end = tuple(int(value) for value in right[int(right_indices[left_index])])
            cv2.line(connected, start, end, 1, 1, cv2.LINE_8)
            joins.append(
                {
                    "component_pair": [int(left_id), int(right_id)],
                    "start": list(start),
                    "end": list(end),
                    "gap_px": distance,
                }
            )
    final_count = cv2.connectedComponents(connected, 8)[0] - 1
    if final_count != 1:
        raise ValueError(
            f"County-detail mainland still has {final_count} disconnected spans"
        )
    return connected > 0, joins


def _compose_county_detail_mainland(
    continuous: np.ndarray, coast: np.ndarray
) -> tuple[np.ndarray, list[Dict[str, object]]]:
    """Replace fill-contour coast pixels with the complete registered stroke."""

    if continuous.shape != coast.shape:
        raise ValueError("Hybrid coast and continuous baseline use different grids")
    # Clear the fill-derived route around the authoritative coast, including
    # cross-mouth shortcuts. Restore the complete county line unchanged, then
    # join only distinct authority spans at their short measured endpoints.
    replacement_band = cv2.dilate(
        coast.astype(np.uint8), np.ones((21, 21), dtype=np.uint8)
    ) > 0
    mainland_unjoined = (continuous & ~replacement_band) | coast
    mainland, seam_joins = _connect_nearby_components(
        mainland_unjoined, maximum_gap_px=20.0
    )
    if np.any(coast & ~mainland):
        raise AssertionError("County coastline evidence was dropped")
    return mainland, seam_joins


def build_county_detail_boundary_candidate(
    canonical_manifest_path: Path,
    hybrid_perimeter_path: Path,
    county_reference_path: Path,
    reference_root: Path,
    output_dir: Path,
) -> Dict[str, object]:
    """Build review linework from county.png coast plus the accepted land arcs.

    This intentionally does not promote or mutate the approved v1 clipping
    reference. It creates the exact visual candidate the author must inspect.
    """

    canonical_manifest_path = canonical_manifest_path.resolve()
    hybrid_perimeter_path = hybrid_perimeter_path.resolve()
    county_reference_path = county_reference_path.resolve()
    reference_root = reference_root.resolve()
    output_dir = output_dir.resolve()
    canonical = _load(canonical_manifest_path)
    hybrid = _load(hybrid_perimeter_path)
    county = _load(county_reference_path)
    if canonical.get("status") != "approved_canonical_reference":
        raise ValueError("County-detail candidate requires the approved v1 baseline")
    if canonical.get("canonical_boundary_id") != CANONICAL_BOUNDARY_ID:
        raise ValueError("County-detail candidate has an unknown canonical baseline")
    if hybrid.get("status") != "pass_no_additional_warp":
        raise ValueError("Hybrid regional authority has not passed")
    authority = hybrid.get("regional_authority", {})
    if (
        authority.get("coast") != "registered_county_png_state_stroke"
        or authority.get("land_borders") != "Census_2025_state_geometry"
    ):
        raise ValueError("Hybrid perimeter does not use the accepted regional authority")
    if county.get("status") != "pass":
        raise ValueError("county.png reference has not passed registration")

    def hybrid_artifact(name: str) -> Path:
        record = hybrid["artifacts"][name]
        return _verified(
            hybrid_perimeter_path.parent / str(record["path"]),
            record["sha256"],
            name,
        )

    coast_path = hybrid_artifact("web-mercator-authoritative-coast-overlay.png")
    continuous_path = hybrid_artifact(
        "web-mercator-authoritative-unified-border-overlay.png"
    )
    coast = _mask_from_rgba(coast_path)
    continuous = _mask_from_rgba(continuous_path)
    mainland, seam_joins = _compose_county_detail_mainland(continuous, coast)

    source_path = _verified(
        Path(str(county["source"]["path"])).resolve(),
        county["source"]["sha256"],
        "county.png source",
    )
    source_rgba = np.asarray(Image.open(source_path).convert("RGBA"))
    source_islands, island_records = _county_island_components(source_rgba)
    state, _ = load_california(reference_root)
    warped_islands, island_grid = warp_classified_to_web_mercator(
        source_islands.astype(np.uint8),
        state,
        county["best"],
        source_islands.shape,
        target_height=coast.shape[0],
        clip_to_state=False,
    )
    expected_grid = canonical["source_grid"]
    for key in ("crs", "bounds", "width", "height"):
        if island_grid[key] != expected_grid[key]:
            raise ValueError(f"County islands changed canonical grid field {key}")
    islands = warped_islands > 0
    island_component_count = cv2.connectedComponents(islands.astype(np.uint8), 8)[0] - 1
    if island_component_count != 4:
        raise ValueError(
            f"Registered county islands changed topology: {island_component_count}"
        )
    if np.any(mainland & islands):
        raise ValueError("Registered county islands overlap the mainland line")
    combined = mainland | islands
    combined_component_count = cv2.connectedComponents(combined.astype(np.uint8), 8)[0] - 1
    if combined_component_count != 5:
        raise ValueError("County-detail candidate must contain mainland plus four islands")

    bay_bbox_wgs84 = [-123.3, 36.8, -121.0, 39.0]
    bay_window = _wgs84_bbox_slice(expected_grid, bay_bbox_wgs84)
    bay_dropped = int(np.count_nonzero(coast[bay_window] & ~mainland[bay_window]))
    bay_non_county = int(np.count_nonzero(mainland[bay_window] & ~coast[bay_window]))
    if bay_dropped or bay_non_county:
        raise ValueError(
            "San Francisco Bay must reproduce the registered county coast exactly"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    mainland_path = output_dir / "county-detail-mainland-border-overlay.png"
    islands_path = output_dir / "county-detail-island-border-overlay.png"
    combined_path = output_dir / "county-detail-boundary-overlay.png"
    for path, mask in (
        (mainland_path, mainland),
        (islands_path, islands),
        (combined_path, combined),
    ):
        rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
        rgba[mask] = [90, 255, 120, 255]
        Image.fromarray(rgba).save(path, optimize=True)

    result = {
        "schema_version": 1,
        "status": "needs_author_review",
        "candidate_boundary_id": COUNTY_DETAIL_BOUNDARY_CANDIDATE_ID,
        "base_canonical_boundary_id": CANONICAL_BOUNDARY_ID,
        "method": "registered_county_png_coast_and_islands_plus_accepted_Census_land_arcs",
        "authority": {
            "coast_and_bays": "registered_county_png_state_stroke_exact_pixels",
            "offshore_islands": "four_registered_county_png_state_stroke_components",
            "land_borders": "accepted_Census_2025_hybrid_spans",
        },
        "inputs": {
            "canonical_manifest": {
                "path": str(canonical_manifest_path),
                "sha256": _sha256(canonical_manifest_path),
            },
            "hybrid_perimeter": {
                "path": str(hybrid_perimeter_path),
                "sha256": _sha256(hybrid_perimeter_path),
            },
            "county_reference": {
                "path": str(county_reference_path),
                "sha256": _sha256(county_reference_path),
            },
            "county_source": {"path": str(source_path), "sha256": _sha256(source_path)},
        },
        "grid": expected_grid,
        "topology": {
            "mainland_component_count": 1,
            "offshore_island_component_count": island_component_count,
            "combined_component_count": combined_component_count,
            "county_coast_pixel_count": int(np.count_nonzero(coast)),
            "county_coast_dropped_pixel_count": int(np.count_nonzero(coast & ~mainland)),
            "san_francisco_bay": {
                "bbox_wgs84": bay_bbox_wgs84,
                "county_coast_dropped_pixel_count": bay_dropped,
                "non_county_pixel_count": bay_non_county,
                "exact": True,
            },
            "mainland_pixel_count": int(np.count_nonzero(mainland)),
            "island_pixel_count": int(np.count_nonzero(islands)),
            "authority_seam_joins": seam_joins,
            "source_islands": island_records,
        },
        "artifacts": {
            "mainland": {"path": mainland_path.name, "sha256": _sha256(mainland_path)},
            "islands": {"path": islands_path.name, "sha256": _sha256(islands_path)},
            "overlay": {"path": combined_path.name, "sha256": _sha256(combined_path)},
        },
        "publication_allowed": False,
        "author_review_required": True,
    }
    manifest_path = output_dir / "county-detail-boundary.json"
    manifest_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def promote_county_detail_border(
    candidate_manifest_path: Path,
    superseded_manifest_path: Path,
    output_dir: Path,
    *,
    author_statement: str,
) -> Dict[str, object]:
    """Promote the reviewed county-detail linework without inventing a fill.

    The exact reviewed border is a line network: its open bay entrances are
    intentional. Clipping interiors remain separately versioned evidence and
    must never be used to reconstruct this display/alignment border.
    """

    candidate_manifest_path = candidate_manifest_path.resolve()
    superseded_manifest_path = superseded_manifest_path.resolve()
    output_dir = output_dir.resolve()
    candidate = _load(candidate_manifest_path)
    superseded = _load(superseded_manifest_path)
    statement = author_statement.strip()
    if not statement:
        raise ValueError("Canonical border promotion requires an author statement")
    if (
        candidate.get("status") != "needs_author_review"
        or candidate.get("candidate_boundary_id")
        != COUNTY_DETAIL_BOUNDARY_CANDIDATE_ID
        or candidate.get("publication_allowed") is not False
    ):
        raise ValueError("County-detail candidate is not eligible for promotion")
    topology = candidate.get("topology", {})
    if (
        topology.get("mainland_component_count") != 1
        or topology.get("offshore_island_component_count") != 4
        or topology.get("combined_component_count") != 5
        or topology.get("county_coast_dropped_pixel_count") != 0
        or topology.get("san_francisco_bay", {}).get("exact") is not True
    ):
        raise ValueError("County-detail candidate topology has not passed")
    if (
        superseded.get("status") != "approved_canonical_reference"
        or superseded.get("canonical_boundary_id") != CANONICAL_BOUNDARY_ID
        or candidate.get("base_canonical_boundary_id") != CANONICAL_BOUNDARY_ID
    ):
        raise ValueError("Canonical predecessor is missing or mismatched")

    source_artifacts: Dict[str, Path] = {}
    for name in ("mainland", "islands", "overlay"):
        record = candidate["artifacts"][name]
        source_artifacts[name] = _verified(
            candidate_manifest_path.parent / str(record["path"]),
            record["sha256"],
            f"county-detail {name}",
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    destination_names = {
        "mainland": "canonical-mainland-border-overlay.png",
        "islands": "canonical-island-border-overlay.png",
        "overlay": "canonical-state-border-overlay.png",
    }
    promoted_artifacts: Dict[str, Dict[str, object]] = {}
    for name, source in source_artifacts.items():
        destination = output_dir / destination_names[name]
        shutil.copyfile(source, destination)
        if _sha256(destination) != _sha256(source):
            raise AssertionError(f"Canonical {name} copy changed bytes")
        promoted_artifacts[name] = {
            "path": destination.name,
            "sha256": _sha256(destination),
        }

    result = {
        "schema_version": 2,
        "status": "approved_canonical_reference",
        "kind": "display_and_alignment_border",
        "canonical_boundary_id": CANONICAL_BORDER_ID,
        "scope": "california_mainland_and_four_county_png_island_outlines",
        "source_grid": candidate["grid"],
        "authority": candidate["authority"],
        "approval": {
            "author_statement": statement,
            "inspection_confirmed": True,
            "candidate_manifest_sha256": _sha256(candidate_manifest_path),
            "candidate_boundary_id": candidate["candidate_boundary_id"],
        },
        "supersedes": {
            "canonical_boundary_id": superseded["canonical_boundary_id"],
            "manifest_sha256": _sha256(superseded_manifest_path),
        },
        "topology": topology,
        "artifacts": promoted_artifacts,
        "policy": {
            "future_alignment_and_review_border": "required",
            "publication_display_border": "required",
            "reconstruct_from_filled_mask": False,
            "clipping_interior_is_separate_evidence": True,
            "reason": (
                "The detailed county.png coast includes open bay entrances and linework "
                "that an exterior fill contour discards."
            ),
        },
    }
    manifest_path = output_dir / "canonical-boundary.json"
    serialized = json.dumps(result, indent=2) + "\n"
    if manifest_path.exists() and manifest_path.read_text() != serialized:
        raise ValueError("Existing canonical v2 package differs from this approval")
    manifest_path.write_text(serialized)
    return result


def activate_canonical_border(
    canonical_manifest_path: Path,
    pointer_path: Path = ACTIVE_CANONICAL_POINTER,
) -> Dict[str, object]:
    """Atomically identify the one canonical border future runs must consume."""

    canonical_manifest_path = canonical_manifest_path.resolve()
    pointer_path = pointer_path.resolve()
    canonical = _load(canonical_manifest_path)
    if (
        canonical.get("status") != "approved_canonical_reference"
        or canonical.get("kind") != "display_and_alignment_border"
        or canonical.get("canonical_boundary_id") != CANONICAL_BORDER_ID
        or canonical.get("approval", {}).get("inspection_confirmed") is not True
    ):
        raise ValueError("Only the approved county-detail border can become active")
    for name in ("mainland", "islands", "overlay"):
        record = canonical["artifacts"][name]
        _verified(
            canonical_manifest_path.parent / str(record["path"]),
            record["sha256"],
            f"canonical {name}",
        )
    pointer_path.parent.mkdir(parents=True, exist_ok=True)
    relative_manifest = os.path.relpath(canonical_manifest_path, pointer_path.parent)
    result = {
        "schema_version": 1,
        "status": "active_canonical_boundary",
        "canonical_boundary_id": CANONICAL_BORDER_ID,
        "manifest": {
            "path": relative_manifest,
            "sha256": _sha256(canonical_manifest_path),
        },
        "policy": "All future map alignment, review, and publication-border displays use this hash-bound canonical linework.",
    }
    pointer_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def load_active_canonical_border(
    pointer_path: Path = ACTIVE_CANONICAL_POINTER,
) -> tuple[Path, Dict[str, object], Dict[str, object]]:
    """Resolve and hash-check the active canonical border pointer."""

    pointer_path = pointer_path.resolve()
    pointer = _load(pointer_path)
    if (
        pointer.get("status") != "active_canonical_boundary"
        or pointer.get("canonical_boundary_id") != CANONICAL_BORDER_ID
    ):
        raise ValueError("Canonical border pointer is not active")
    record = pointer.get("manifest", {})
    manifest_path = _verified(
        pointer_path.parent / str(record.get("path", "")),
        record.get("sha256"),
        "active canonical border manifest",
    )
    canonical = _load(manifest_path)
    if (
        canonical.get("status") != "approved_canonical_reference"
        or canonical.get("canonical_boundary_id") != CANONICAL_BORDER_ID
    ):
        raise ValueError("Active canonical border manifest is invalid")
    return manifest_path, canonical, pointer


def _grid_from_materialization(materialization: Dict[str, object]) -> Dict[str, object]:
    source_run = Path(str(materialization["source_run"])).resolve()
    extraction = _load(source_run / "extraction.json")
    layers = extraction.get("layers", [])
    if not isinstance(layers, list) or len(layers) != 1:
        raise ValueError("Canonical boundary approval requires one extraction grid")
    return dict(layers[0]["warp"])


def promote_canonical_boundary(
    approved_materialization_dir: Path,
    approved_boundary_geojson: Path,
    output_dir: Path,
) -> Dict[str, object]:
    """Copy the approved mainland evidence into a reusable canonical package."""

    approved_materialization_dir = approved_materialization_dir.resolve()
    approved_boundary_geojson = approved_boundary_geojson.resolve()
    output_dir = output_dir.resolve()
    materialization_path = approved_materialization_dir / "materialization.json"
    decision_path = approved_materialization_dir / "materialization-review-decision.json"
    materialization = _load(materialization_path)
    decision = _load(decision_path)
    materialization_sha = _sha256(materialization_path)
    if decision.get("status") != "approved" or decision.get("inspection_confirmed") is not True:
        raise ValueError("Canonical boundary source lacks confirmed author approval")
    if decision.get("materialization_sha256") != materialization_sha:
        raise ValueError("Boundary approval does not bind the materialization")
    boundary_clip = materialization.get("boundary_clip", {})
    continuous_border_sha = str(boundary_clip.get("continuous_border_sha256", ""))
    mainland_interior_sha = str(boundary_clip.get("mainland_interior_sha256", ""))
    if decision.get("hybrid_border_sha256") != continuous_border_sha:
        raise ValueError("Boundary approval does not bind the displayed hybrid border")
    boundary_audit_record = boundary_clip.get("audit", {})
    boundary_audit_path = _verified(
        Path(str(boundary_audit_record["path"])).resolve(),
        boundary_audit_record.get("sha256"),
        "approved boundary audit",
    )
    if decision.get("boundary_clip_audit_sha256") != _sha256(boundary_audit_path):
        raise ValueError("Boundary approval does not bind the clipping audit")
    boundary_audit = _load(boundary_audit_path)
    boundary = boundary_audit.get("boundary", {})
    if boundary.get("connected_component_count") != 1:
        raise ValueError("Approved canonical mainland is not one component")
    interior_record = boundary["interior"]
    border_record = boundary["border"]
    interior_path = _verified(
        boundary_audit_path.parent / str(interior_record["path"]),
        interior_record.get("sha256"),
        "approved mainland interior",
    )
    border_path = _verified(
        boundary_audit_path.parent / str(border_record["path"]),
        border_record.get("sha256"),
        "approved mainland border",
    )
    if _sha256(interior_path) != mainland_interior_sha:
        raise ValueError("Materialization mainland hash does not match its boundary audit")
    if _sha256(border_path) != continuous_border_sha:
        raise ValueError("Materialization border hash does not match its boundary audit")

    geojson = _load(_verified(approved_boundary_geojson, None, "approved boundary GeoJSON"))
    features = geojson.get("features", [])
    if not isinstance(features, list) or len(features) != 1:
        raise ValueError("Approved mainland GeoJSON must contain exactly one feature")
    feature = features[0]
    if feature.get("properties", {}).get("source_sha256") != continuous_border_sha:
        raise ValueError("Boundary GeoJSON is not derived from the approved border raster")
    geometry = feature.get("geometry", {})
    coordinates = geometry.get("coordinates", [])
    if geometry.get("type") != "LineString" or len(coordinates) < 4:
        raise ValueError("Approved mainland GeoJSON is not a valid line ring")
    if coordinates[0] != coordinates[-1]:
        raise ValueError("Approved mainland GeoJSON is not closed")

    output_dir.mkdir(parents=True, exist_ok=True)
    destination_interior = output_dir / "canonical-mainland-interior-mask.png"
    destination_border = output_dir / "canonical-mainland-border-overlay.png"
    destination_geojson = output_dir / "canonical-mainland-boundary.geojson"
    shutil.copyfile(interior_path, destination_interior)
    shutil.copyfile(border_path, destination_border)
    shutil.copyfile(approved_boundary_geojson, destination_geojson)
    grid = _grid_from_materialization(materialization)
    roundtrip_interior, roundtrip_border = _rasterize_geojson_ring(coordinates, grid)
    approved_interior = np.asarray(Image.open(interior_path).convert("L")) > 0
    approved_border = (
        np.asarray(Image.open(border_path).convert("RGBA"))[..., 3] > 0
    )
    interior_roundtrip_mismatch = int(
        np.count_nonzero(roundtrip_interior != approved_interior)
    )
    border_roundtrip_mismatch = int(
        np.count_nonzero((roundtrip_border > 0) != approved_border)
    )
    if interior_roundtrip_mismatch or border_roundtrip_mismatch:
        raise ValueError(
            "Canonical GeoJSON does not exactly reproduce the author-approved raster grid"
        )
    result = {
        "schema_version": 1,
        "status": "approved_canonical_reference",
        "canonical_boundary_id": CANONICAL_BOUNDARY_ID,
        "scope": "california_mainland",
        "authority": {
            "coast_and_tahoe": "registered_county_png_approved_by_author",
            "land_borders_and_lower_colorado": "Census_2025_approved_by_author",
            "islands": "separate_source_supported_component_audit",
        },
        "source_grid": grid,
        "approval": {
            "materialization_sha256": materialization_sha,
            "decision_sha256": _sha256(decision_path),
            "boundary_clip_audit_sha256": _sha256(boundary_audit_path),
            "continuous_border_sha256": continuous_border_sha,
            "inspection_confirmed": True,
        },
        "geometry": {
            "crs": "EPSG:4326",
            "closed_vertex_count": len(coordinates),
            "connected_component_count": 1,
            "approved_grid_roundtrip": {
                "interior_mismatch_pixel_count": interior_roundtrip_mismatch,
                "border_mismatch_pixel_count": border_roundtrip_mismatch,
                "exact": True,
            },
        },
        "artifacts": {
            "interior": {
                "path": destination_interior.name,
                "sha256": _sha256(destination_interior),
            },
            "border": {
                "path": destination_border.name,
                "sha256": _sha256(destination_border),
            },
            "geojson": {
                "path": destination_geojson.name,
                "sha256": _sha256(destination_geojson),
            },
        },
        "policy": (
            "Every statewide dataset rasterizes this canonical mainland. Islands are "
            "added only by the separately hash-bound observed-source component audit."
        ),
    }
    manifest_path = output_dir / "canonical-boundary.json"
    serialized = json.dumps(result, indent=2) + "\n"
    if manifest_path.exists() and manifest_path.read_text() != serialized:
        existing = _load(manifest_path)
        if (
            existing.get("canonical_boundary_id") != CANONICAL_BOUNDARY_ID
            or existing.get("approval") != result["approval"]
            or existing.get("artifacts") != result["artifacts"]
        ):
            raise ValueError(
                "Existing canonical boundary package differs from approved evidence"
            )
    manifest_path.write_text(serialized)
    return result


def _project_geojson_ring_float(
    coordinates: Sequence[Sequence[float]], grid: Dict[str, object]
) -> np.ndarray:
    transformer = Transformer.from_crs("EPSG:4326", str(grid["crs"]), always_xy=True)
    lon = np.asarray([item[0] for item in coordinates], dtype=float)
    lat = np.asarray([item[1] for item in coordinates], dtype=float)
    projected_x, projected_y = transformer.transform(lon, lat)
    min_x, min_y, max_x, max_y = (float(value) for value in grid["bounds"])
    width, height = int(grid["width"]), int(grid["height"])
    pixel_x = (np.asarray(projected_x) - min_x) / (max_x - min_x) * width - 0.5
    pixel_y = (max_y - np.asarray(projected_y)) / (max_y - min_y) * height - 0.5
    return np.column_stack([pixel_x, pixel_y])


def _project_geojson_ring(
    coordinates: Sequence[Sequence[float]], grid: Dict[str, object]
) -> np.ndarray:
    return np.rint(_project_geojson_ring_float(coordinates, grid)).astype(np.int32)


def _write_vector_border(
    coordinates: Sequence[Sequence[float]], grid: Dict[str, object], path: Path
) -> None:
    """Write the ordered ring without coarse-grid integer quantization.

    The fill raster remains authoritative for clipping. This SVG exists only so
    diagnostic viewers do not close narrow coastal entrances such as the Golden
    Gate by rounding both shores onto the same one-pixel raster stroke.
    """

    points = _project_geojson_ring_float(coordinates, grid)
    width, height = int(grid["width"]), int(grid["height"])
    commands = [f"M {points[0, 0]:.4f} {points[0, 1]:.4f}"]
    commands.extend(f"L {x:.4f} {y:.4f}" for x, y in points[1:-1])
    commands.append("Z")
    path.write_text(
        "\n".join(
            [
                '<?xml version="1.0" encoding="UTF-8"?>',
                (
                    f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
                    f'viewBox="0 0 {width} {height}" '
                    'preserveAspectRatio="none" shape-rendering="geometricPrecision">'
                ),
                (
                    f'  <path d="{" ".join(commands)}" fill="none" stroke="#5aff78" '
                    'stroke-width="0.5" stroke-linecap="round" stroke-linejoin="round"/>'
                ),
                "</svg>",
                "",
            ]
        )
    )


def _rasterize_geojson_ring(
    coordinates: Sequence[Sequence[float]], grid: Dict[str, object]
) -> tuple[np.ndarray, np.ndarray]:
    points = _project_geojson_ring(coordinates, grid)
    width, height = int(grid["width"]), int(grid["height"])
    interior = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(interior, [points], 255, lineType=cv2.LINE_8)
    border = np.zeros((height, width), dtype=np.uint8)
    cv2.polylines(border, [points], True, 255, 1, lineType=cv2.LINE_8)
    if cv2.connectedComponents((border > 0).astype(np.uint8), 8)[0] - 1 != 1:
        raise ValueError("Canonical mainland border is not one continuous ordered line")
    return interior > 0, border


def rasterize_canonical_boundary(
    canonical_manifest_path: Path,
    target_grid: Dict[str, object],
    output_dir: Path,
) -> Dict[str, object]:
    """Rasterize the canonical mainland on a target grid without rebuilding it."""

    canonical_manifest_path = canonical_manifest_path.resolve()
    output_dir = output_dir.resolve()
    canonical = _load(canonical_manifest_path)
    if canonical.get("status") != "approved_canonical_reference":
        raise ValueError("Canonical boundary has not been author approved")
    if canonical.get("canonical_boundary_id") != CANONICAL_BOUNDARY_ID:
        raise ValueError("Unknown canonical boundary identifier")
    artifacts = canonical["artifacts"]
    _verified(
        canonical_manifest_path.parent / str(artifacts["interior"]["path"]),
        artifacts["interior"]["sha256"],
        "canonical mainland interior",
    )
    geojson_path = _verified(
        canonical_manifest_path.parent / str(artifacts["geojson"]["path"]),
        artifacts["geojson"]["sha256"],
        "canonical mainland GeoJSON",
    )
    geojson = _load(geojson_path)
    coordinates = geojson["features"][0]["geometry"]["coordinates"]
    interior, border = _rasterize_geojson_ring(coordinates, target_grid)
    method = "rasterize_author_approved_canonical_ordered_geojson_ring"
    interior_components = cv2.connectedComponents(interior.astype(np.uint8), 8)[0] - 1
    if interior_components != 1:
        raise ValueError("Canonical mainland interior is not one component on target grid")
    contours, hierarchy = cv2.findContours(
        interior.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
    )
    enclosed_hole_count = (
        sum(1 for item in hierarchy[0] if item[3] >= 0) if hierarchy is not None else 0
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    interior_path = output_dir / "canonical-mainland-interior-mask.png"
    border_path = output_dir / "canonical-mainland-border-overlay.png"
    border_vector_path = output_dir / "canonical-mainland-border-overlay.svg"
    Image.fromarray(interior.astype(np.uint8) * 255).save(interior_path, optimize=True)
    rgba = np.zeros((*interior.shape, 4), dtype=np.uint8)
    rgba[border > 0] = [90, 255, 120, 255]
    Image.fromarray(rgba).save(border_path, optimize=True)
    _write_vector_border(coordinates, target_grid, border_vector_path)
    report = {
        "schema_version": 1,
        "status": "pass",
        "method": method,
        "canonical_boundary_id": CANONICAL_BOUNDARY_ID,
        "canonical_manifest": {
            "path": str(canonical_manifest_path),
            "sha256": _sha256(canonical_manifest_path),
        },
        "target_grid": target_grid,
        "topology": {
            "interior_component_count": 1,
            "border_component_count": 1,
            "interior_pixel_count": int(np.count_nonzero(interior)),
            "border_pixel_count": int(np.count_nonzero(border)),
            "quantized_enclosed_water_component_count": enclosed_hole_count,
            "interior_is_exact_fill_of_displayed_border": True,
        },
        "artifacts": {
            "interior": {"path": interior_path.name, "sha256": _sha256(interior_path)},
            "border": {"path": border_path.name, "sha256": _sha256(border_path)},
            "border_vector": {
                "path": border_vector_path.name,
                "sha256": _sha256(border_vector_path),
                "stroke_width_target_pixels": 0.5,
                "purpose": "subpixel_fidelity_diagnostic_only",
            },
        },
        "publication_allowed": False,
        "author_review_required": True,
    }
    report_path = output_dir / "canonical-boundary-raster.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def rasterize_canonical_boundary_for_run(
    canonical_manifest_path: Path,
    extraction_run: Path,
    output_dir: Path,
) -> Dict[str, object]:
    extraction_run = extraction_run.resolve()
    extraction = _load(extraction_run / "extraction.json")
    layers = extraction.get("layers", [])
    if not isinstance(layers, list) or len(layers) != 1:
        raise ValueError("Canonical boundary rasterization requires one extraction grid")
    result = rasterize_canonical_boundary(
        canonical_manifest_path, dict(layers[0]["warp"]), output_dir
    )
    report_path = output_dir.resolve() / "canonical-boundary-raster.json"
    result["target_extraction"] = {
        "path": str((extraction_run / "extraction.json").resolve()),
        "sha256": _sha256(extraction_run / "extraction.json"),
    }
    report_path.write_text(json.dumps(result, indent=2) + "\n")
    return result
