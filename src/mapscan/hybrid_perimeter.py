"""Regional California perimeter authority and no-change audit.

The registered ``county.png`` state stroke is the detailed coastal reference.
Pinned Census geometry remains authoritative for land borders, including the
Colorado River.  This stage never changes an alignment; it proves whether the
current candidate already satisfies both regional references and emits a
single hybrid overlay for author review.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, Sequence

import cv2
import numpy as np
from PIL import Image
from pyproj import Transformer

from .auto_refinement import (
    _alignment_transform,
    _registered_reference_masks,
    _residual_summary,
    _warp_evidence,
)
from .extraction import (
    _largest_polygon,
    _target_state_mask,
    warp_classified_to_web_mercator,
)
from .lower_colorado_refinement import _image_evidence
from .lower_colorado_refinement import _best_shift as _best_border_shift
from .reference import load_california


SOUTHERN_COAST_SEAM_WGS84 = (-117.12694883725163, 32.53413690413119)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_border(mask: np.ndarray) -> np.ndarray:
    eroded = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    return mask & ~eroded


def _web_mercator_pixel(
    longitude: float,
    latitude: float,
    bounds: Sequence[float],
    shape: tuple[int, int],
) -> tuple[float, float]:
    x, y = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform(
        longitude, latitude
    )
    min_x, min_y, max_x, max_y = (float(value) for value in bounds)
    height, width = shape
    return (
        (x - min_x) / (max_x - min_x) * (width - 1),
        (max_y - y) / (max_y - min_y) * (height - 1),
    )


def _coastal_targets(
    mainland_mask: np.ndarray,
    county_state: np.ndarray,
    *,
    reference: str,
    start_y: int,
    end_y: int,
) -> list[tuple[int, int]]:
    targets = []
    for y in range(max(0, start_y), min(mainland_mask.shape[0], end_y)):
        mainland_x = np.flatnonzero(mainland_mask[y])
        if not len(mainland_x):
            continue
        census_coast_x = int(mainland_x[0])
        if reference == "census":
            targets.append((y, census_coast_x))
            continue
        county_x = np.flatnonzero(county_state[y])
        nearby = county_x[np.abs(county_x - census_coast_x) <= 60]
        if len(nearby):
            targets.append((y, int(round(float(np.median(nearby))))))
    return targets


def _coast_window(
    evidence: Dict[str, np.ndarray],
    targets: Sequence[tuple[int, int]],
    *,
    maximum_shift_px: int = 18,
) -> Dict[str, object]:
    height, width = evidence["blackhat"].shape
    candidates: Dict[str, list[tuple[int, float]]] = {
        "blackhat": [],
        "saturation": [],
        "chroma": [],
    }
    for shift in range(-maximum_shift_px, maximum_shift_px + 1):
        blackhat = []
        saturation = []
        chroma = []
        for y, target_x in targets:
            x = target_x + shift
            if not (0 <= y < height and 5 <= x < width - 5):
                continue
            blackhat.append(float(evidence["blackhat"][y, x]))
            # California land is east/right of its mainland coastline.
            saturation.append(
                float(np.mean(evidence["saturation"][y, x + 2 : x + 5]))
                - float(np.mean(evidence["saturation"][y, x - 4 : x - 1]))
            )
            chroma.append(
                float(np.mean(evidence["chroma"][y, x + 2 : x + 5]))
                - float(np.mean(evidence["chroma"][y, x - 4 : x - 1]))
            )
        if blackhat:
            candidates["blackhat"].append((shift, float(np.median(blackhat))))
            candidates["saturation"].append((shift, float(np.median(saturation))))
            candidates["chroma"].append((shift, float(np.median(chroma))))
    shifts = {}
    strengths = {}
    for key, values in candidates.items():
        shift, strength = max(values, key=lambda item: item[1])
        shifts[key] = int(shift)
        strengths[key] = float(strength)
    shift_values = np.asarray(list(shifts.values()), dtype=np.float64)
    return {
        "estimator_shifts_px": shifts,
        "estimator_strengths": strengths,
        "consensus_shift_px": float(np.median(shift_values)),
        "estimator_range_px": float(np.ptp(shift_values)),
        "accepted": bool(
            np.ptp(shift_values) <= 3
            and strengths["blackhat"] >= 8.0
            and strengths["saturation"] >= 15.0
            and strengths["chroma"] >= 4.0
            and np.max(np.abs(shift_values)) < maximum_shift_px
        ),
    }


def _coast_profiles(
    rgb: np.ndarray,
    mainland_mask: np.ndarray,
    county_state: np.ndarray,
    *,
    reference: str,
    spans: Iterable[tuple[int, int]],
) -> list[Dict[str, object]]:
    evidence = _image_evidence(rgb)
    profiles = []
    for index, (start, end) in enumerate(spans):
        targets = _coastal_targets(
            mainland_mask,
            county_state,
            reference=reference,
            start_y=start,
            end_y=end,
        )
        if len(targets) < max(25, round(0.60 * (end - start))):
            continue
        profile = _coast_window(evidence, targets)
        profile.update(
            {
                "id": f"coast_{index:02d}",
                "span": [int(start), int(end)],
                "center": [
                    float(np.median([x for _, x in targets])),
                    float(np.median([y for y, _ in targets])),
                ],
            }
        )
        profiles.append(profile)
    return profiles


def _right_border_profiles(
    rgb: np.ndarray,
    border_mask: np.ndarray,
    *,
    spans: Iterable[tuple[int, int]],
    x_range: tuple[int, int],
    maximum_shift_px: int = 12,
) -> list[Dict[str, object]]:
    """Measure a near-vertical/east-facing border against source evidence."""

    evidence = _image_evidence(rgb)
    profiles = []
    x_start, x_end = x_range
    for index, (start, end) in enumerate(spans):
        samples = []
        for y in range(max(0, start), min(border_mask.shape[0], end)):
            xs = np.flatnonzero(border_mask[y, x_start:x_end]) + x_start
            if len(xs):
                samples.append((y, int(xs[-1])))
        if len(samples) < max(25, round(0.60 * (end - start))):
            continue
        shifts, strengths = _best_border_shift(
            evidence,
            samples,
            axis="vertical",
            maximum_shift_px=maximum_shift_px,
        )
        values = np.asarray(list(shifts.values()), dtype=np.float64)
        profiles.append(
            {
                "id": f"tahoe_{index:02d}",
                "span": [int(start), int(end)],
                "center": [
                    float(np.median([item[1] for item in samples])),
                    float(np.median([item[0] for item in samples])),
                ],
                "estimator_shifts_px": shifts,
                "estimator_strengths": strengths,
                "consensus_shift_px": float(np.median(values)),
                "estimator_range_px": float(np.ptp(values)),
                "accepted": bool(
                    np.ptp(values) <= 3
                    and strengths["blackhat"] >= 25.0
                    and strengths["saturation"] >= 20.0
                    and strengths["chroma"] >= 8.0
                    and np.max(np.abs(values)) < maximum_shift_px
                ),
            }
        )
    return profiles


def _profile_summary(profiles: Sequence[Dict[str, object]]) -> Dict[str, object]:
    all_values = np.abs(
        np.asarray(
            [float(item["consensus_shift_px"]) for item in profiles],
            dtype=np.float64,
        )
    )
    accepted = [item for item in profiles if item["accepted"]]
    accepted_values = np.abs(
        np.asarray(
            [float(item["consensus_shift_px"]) for item in accepted],
            dtype=np.float64,
        )
    )
    return {
        "window_count": len(profiles),
        "accepted_count": len(accepted),
        "all_window_residual": _residual_summary(all_values),
        "accepted_window_residual": _residual_summary(accepted_values),
        "profiles": list(profiles),
    }


def _hybrid_masks(
    mainland_mask: np.ndarray,
    census_border: np.ndarray,
    county_state: np.ndarray,
    *,
    coast_start_y: int,
    coast_end_y: int,
    tahoe_region: tuple[int, int, int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    county_authority = np.zeros_like(county_state, dtype=bool)
    land = census_border.copy()
    for y in range(
        max(0, coast_start_y), min(coast_end_y, county_authority.shape[0])
    ):
        mainland_x = np.flatnonzero(mainland_mask[y])
        if not len(mainland_x):
            continue
        left = int(mainland_x[0])
        right = int(mainland_x[-1])
        # Cut well inside California so every concavity in the county.png coast
        # survives (Humboldt Bay, Point Reyes, San Francisco Bay) while the
        # northern/eastern/southern land-border arcs remain Census geometry.
        divider = round(left + 0.62 * (right - left))
        county_x = np.flatnonzero(county_state[y])
        coast_pixels = county_x[county_x <= divider]
        county_authority[y, coast_pixels] = True
        # Remove the simplified/legal Census coast but retain the land-border arc.
        land[y, : min(land.shape[1], divider + 1)] = False

    if tahoe_region is not None:
        x_start, y_start, x_end, y_end = tahoe_region
        for y in range(max(0, y_start), min(y_end, land.shape[0])):
            census_x = np.flatnonzero(census_border[y, x_start:x_end]) + x_start
            if not len(census_x):
                continue
            center = int(census_x[-1])
            county_x = np.flatnonzero(
                county_state[y, max(0, center - 80) : min(land.shape[1], center + 81)]
            ) + max(0, center - 80)
            if not len(county_x):
                continue
            # Replace only the Tahoe vertical-to-diagonal state-border corridor.
            land[y, max(x_start, center - 40) : x_end] = False
            county_authority[y, county_x] = True
    return county_authority, land


def _hybrid_mainland_interior(
    mainland_mask: np.ndarray,
    county_interior: np.ndarray,
    census_border: np.ndarray,
    *,
    coast_start_y: int,
    coast_end_y: int,
    tahoe_region: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """Compose the same regional authorities as a filled mainland surface.

    Building the publication geometry from an interior, rather than unioning
    independently rasterized line spans, guarantees that its exterior can be
    rendered as one closed contour and used as the exact clipping mask.
    """

    if mainland_mask.shape != county_interior.shape:
        raise ValueError("County and Census mainland masks use different grids")
    interior = mainland_mask.copy()
    for y in range(max(0, coast_start_y), min(coast_end_y, interior.shape[0])):
        mainland_x = np.flatnonzero(mainland_mask[y])
        if not len(mainland_x):
            continue
        left = int(mainland_x[0])
        right = int(mainland_x[-1])
        divider = round(left + 0.62 * (right - left))
        interior[y, : divider + 1] = county_interior[y, : divider + 1]

    if tahoe_region is not None:
        x_start, y_start, x_end, y_end = tahoe_region
        for y in range(max(0, y_start), min(y_end, interior.shape[0])):
            census_x = np.flatnonzero(census_border[y, x_start:x_end]) + x_start
            if not len(census_x):
                continue
            center = int(census_x[-1])
            replace_start = max(x_start, center - 40)
            interior[y, replace_start:x_end] = county_interior[
                y, replace_start:x_end
            ]

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        interior.astype(np.uint8), 8
    )
    if count <= 1:
        raise ValueError("Hybrid mainland interior is empty")
    component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == component


def _closed_mainland_geometry(interior: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return one filled mainland polygon and its single closed 8-connected ring."""

    contours, _ = cv2.findContours(
        interior.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        raise ValueError("Hybrid mainland interior has no exterior contour")
    contour = max(contours, key=cv2.contourArea)
    filled = np.zeros(interior.shape, dtype=np.uint8)
    border = np.zeros(interior.shape, dtype=np.uint8)
    cv2.drawContours(filled, [contour], -1, 1, cv2.FILLED, cv2.LINE_8)
    cv2.drawContours(border, [contour], -1, 1, 1, cv2.LINE_8)
    filled_mask = filled > 0
    border_mask = border > 0
    component_count = cv2.connectedComponents(border, 8)[0] - 1
    if component_count != 1:
        raise ValueError("Hybrid mainland border is not one continuous closed line")
    if np.any(border_mask & ~filled_mask):
        raise AssertionError("Hybrid border escaped its own filled interior")
    return filled_mask, border_mask


def _connect_southern_seam(
    county_authority: np.ndarray,
    census_authority: np.ndarray,
    seam: tuple[float, float],
) -> tuple[np.ndarray, Dict[str, object]]:
    """Close the tiny reference switch clipped by the southern grid edge."""

    seam_x, seam_y = seam
    local = np.zeros_like(county_authority, dtype=bool)
    local[
        max(0, round(seam_y) - 100) :,
        max(0, round(seam_x) - 120) : min(
            county_authority.shape[1], round(seam_x) + 121
        ),
    ] = True
    x_coordinates = np.arange(county_authority.shape[1])[None, :]
    county_y, county_x = np.nonzero(
        county_authority & local & (x_coordinates <= round(seam_x) + 8)
    )
    census_y, census_x = np.nonzero(
        census_authority & local & (x_coordinates >= round(seam_x) - 8)
    )
    if not len(county_x) or not len(census_x):
        raise ValueError("Southern hybrid seam is missing a reference span")
    distance_sq = (
        (county_x[:, None] - census_x[None, :]) ** 2
        + (county_y[:, None] - census_y[None, :]) ** 2
    )
    county_index, census_index = np.unravel_index(
        int(np.argmin(distance_sq)), distance_sq.shape
    )
    start = (int(county_x[county_index]), int(county_y[county_index]))
    end = (int(census_x[census_index]), int(census_y[census_index]))
    gap = float(np.sqrt(distance_sq[county_index, census_index]))
    connector = np.zeros_like(county_authority, dtype=np.uint8)
    cv2.line(connector, start, end, 1, 1, cv2.LINE_8)
    connector_mask = connector > 0
    report = {
        "county_png_endpoint_px": list(start),
        "Census_endpoint_px": list(end),
        "endpoint_gap_px": gap,
        "connector_pixel_count": int(np.count_nonzero(connector_mask)),
        "passed": bool(
            gap <= 16.0
            and abs(seam_y - (county_authority.shape[0] - 1)) <= 4.0
        ),
    }
    return connector_mask, report


def _rgba(mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    result = np.zeros((*mask.shape, 4), dtype=np.uint8)
    result[mask, :3] = color
    result[mask, 3] = 255
    return result


def _hybrid_overlay(coast: np.ndarray, land: np.ndarray) -> np.ndarray:
    result = np.zeros((*coast.shape, 4), dtype=np.uint8)
    result[coast, :3] = (0, 225, 255)
    result[coast, 3] = 255
    result[land, :3] = (255, 220, 0)
    result[land, 3] = 255
    return result


def _unified_overlay(coast: np.ndarray, land: np.ndarray) -> np.ndarray:
    """Render the two authoritative spans as one inspection line."""

    return _rgba(coast | land, (80, 255, 120))


def _diagnostic(
    rgb: np.ndarray,
    coast: np.ndarray,
    land: np.ndarray,
    county_profiles: Sequence[Dict[str, object]],
) -> np.ndarray:
    result = rgb.astype(np.float32)
    for mask, color in ((coast, (0, 225, 255)), (land, (255, 220, 0))):
        result[mask] = 0.10 * result[mask] + 0.90 * np.asarray(color)
    result = np.clip(result, 0, 255).astype(np.uint8)
    for profile in county_profiles:
        x, y = (round(float(value)) for value in profile["center"])
        color = (40, 225, 90) if profile["accepted"] else (40, 140, 255)
        cv2.circle(result, (x, y), 10, color, 2, cv2.LINE_AA)
    return result


def _run_once(
    run_dir: Path,
    county_reference_path: Path,
    reference_root: Path,
) -> Dict[str, object]:
    refinement_path = run_dir / "lower-colorado-refinement.json"
    refinement = json.loads(refinement_path.read_text())
    if refinement.get("status") != "needs_author_review":
        raise ValueError("Lower-Colorado parent has not passed automatic QA")
    alignment_path = run_dir / "alignment.json"
    if _sha256(alignment_path) != refinement["candidate_alignment"]["sha256"]:
        raise ValueError("Lower-Colorado alignment hash mismatch")
    county_reference = json.loads(county_reference_path.read_text())
    if county_reference.get("status") != "pass":
        raise ValueError("county.png reference has not passed registration QA")

    source_path = Path(str(refinement["source"]["path"]))
    if _sha256(source_path) != refinement["source"]["sha256"]:
        raise ValueError("Source hash mismatch")
    rgb = np.asarray(Image.open(source_path).convert("RGB"))
    state, _ = load_california(reference_root)
    mainland = _largest_polygon(state)
    alignment = json.loads(alignment_path.read_text())
    transform = _alignment_transform(alignment, state)
    grid = refinement["grid"]
    warped, _, rendered_grid, _ = _warp_evidence(
        rgb, state, transform, int(grid["height"])
    )
    if rendered_grid["bounds"] != grid["bounds"]:
        raise AssertionError("Hybrid audit grid changed")
    mainland_mask = _target_state_mask(mainland, grid["bounds"], warped.shape[:2])
    census_border = _state_border(mainland_mask)
    county_state, _ = _registered_reference_masks(
        county_reference, county_reference_path, grid["bounds"], warped.shape[:2]
    )
    county_interior_artifact = county_reference["artifacts"]["source_state_interior"]
    county_interior_path = (
        county_reference_path.parent / str(county_interior_artifact["path"])
    )
    if _sha256(county_interior_path) != county_interior_artifact["sha256"]:
        raise ValueError("Registered county-reference interior hash mismatch")
    county_source_interior = np.asarray(Image.open(county_interior_path)) > 0
    county_interior_values, county_interior_grid = warp_classified_to_web_mercator(
        county_source_interior.astype(np.uint8),
        state,
        county_reference["best"],
        county_source_interior.shape,
        target_height=int(grid["height"]),
        clip_to_state=False,
    )
    if (
        county_interior_grid["bounds"] != grid["bounds"]
        or county_interior_grid["width"] != grid["width"]
        or county_interior_grid["height"] != grid["height"]
    ):
        raise AssertionError("Registered county interior changed the review grid")
    county_interior = county_interior_values > 0

    y_scale = (int(grid["height"]) - 1) / (3920 - 1)
    x_scale = (int(grid["width"]) - 1) / (3398 - 1)
    coast_start_y = round(20 * y_scale)
    southern_seam = _web_mercator_pixel(
        *SOUTHERN_COAST_SEAM_WGS84,
        grid["bounds"],
        warped.shape[:2],
    )
    # The geographic junction falls within three pixels of this cropped grid's
    # bottom edge. Preserve county.png coast detail through the final row, then
    # close the clipped switch to the Census Mexico-border segment explicitly.
    coast_end_y = warped.shape[0]
    spans = [
        (start, min(start + round(150 * y_scale), coast_end_y))
        for start in range(round(150 * y_scale), coast_end_y, round(150 * y_scale))
        if min(start + round(150 * y_scale), coast_end_y) - start >= 25
    ]
    county_profiles = _coast_profiles(
        warped,
        mainland_mask,
        county_state,
        reference="county",
        spans=spans,
    )
    census_profiles = _coast_profiles(
        warped,
        mainland_mask,
        county_state,
        reference="census",
        spans=spans,
    )
    county_summary = _profile_summary(county_profiles)
    census_summary = _profile_summary(census_profiles)
    tahoe_region = (
        round(1100 * x_scale),
        round(800 * y_scale),
        round(2100 * x_scale),
        round(1700 * y_scale),
    )
    tahoe_spans = [
        (round(start * y_scale), round((start + 100) * y_scale))
        for start in range(800, 1700, 100)
    ]
    tahoe_county_profiles = _right_border_profiles(
        warped,
        county_state,
        spans=tahoe_spans,
        x_range=(tahoe_region[0], tahoe_region[2]),
    )
    tahoe_census_profiles = _right_border_profiles(
        warped,
        census_border,
        spans=tahoe_spans,
        x_range=(tahoe_region[0], tahoe_region[2]),
    )
    tahoe_county_summary = _profile_summary(tahoe_county_profiles)
    tahoe_census_summary = _profile_summary(tahoe_census_profiles)
    paired = []
    census_by_id = {item["id"]: item for item in census_profiles}
    for county in county_profiles:
        census = census_by_id.get(county["id"])
        if census is None:
            continue
        county_residual = abs(float(county["consensus_shift_px"]))
        census_residual = abs(float(census["consensus_shift_px"]))
        paired.append(
            {
                "id": county["id"],
                "county_png_residual_px": county_residual,
                "census_residual_px": census_residual,
                "county_png_better": county_residual < census_residual,
            }
        )
    better_fraction = float(np.mean([item["county_png_better"] for item in paired]))
    coast_pass = bool(
        county_summary["accepted_count"] >= 22
        and county_summary["all_window_residual"]["p90_px"] <= 2.0
        and county_summary["all_window_residual"]["max_px"] <= 3.0
        and better_fraction >= 0.55
    )
    tahoe_pass = bool(
        tahoe_county_summary["accepted_count"] >= 8
        and tahoe_county_summary["all_window_residual"]["p90_px"] <= 4.0
        and tahoe_census_summary["all_window_residual"]["p90_px"]
        - tahoe_county_summary["all_window_residual"]["p90_px"]
        >= 3.0
    )
    land_pass = bool(
        refinement["fixed_point_gate"]["passed"]
        and refinement["unchanged_region_veto"]["passed"]
        and refinement["independent_edge_holdouts"]["passed"]
        and tahoe_pass
    )
    if not coast_pass or not land_pass:
        raise ValueError("Current candidate does not pass the regional hybrid perimeter gate")

    coast_mask, land_mask = _hybrid_masks(
        mainland_mask,
        census_border,
        county_state,
        coast_start_y=coast_start_y,
        coast_end_y=coast_end_y,
        tahoe_region=tahoe_region,
    )
    seam_connector, seam_report = _connect_southern_seam(
        coast_mask, land_mask, southern_seam
    )
    if not seam_report["passed"]:
        raise ValueError("Southern hybrid seam is not continuous")
    coast_mask |= seam_connector
    hybrid_interior = _hybrid_mainland_interior(
        mainland_mask,
        county_interior,
        census_border,
        coast_start_y=coast_start_y,
        coast_end_y=coast_end_y,
        tahoe_region=tahoe_region,
    )
    hybrid_interior, continuous_border = _closed_mainland_geometry(hybrid_interior)
    Image.fromarray(_hybrid_overlay(coast_mask, land_mask), mode="RGBA").save(
        run_dir / "web-mercator-hybrid-state-overlay.png", optimize=True
    )
    Image.fromarray(_rgba(coast_mask, (0, 225, 255)), mode="RGBA").save(
        run_dir / "web-mercator-authoritative-coast-overlay.png", optimize=True
    )
    Image.fromarray(_rgba(land_mask, (255, 220, 0)), mode="RGBA").save(
        run_dir / "web-mercator-authoritative-land-border-overlay.png", optimize=True
    )
    Image.fromarray(_rgba(continuous_border, (80, 255, 120)), mode="RGBA").save(
        run_dir / "web-mercator-authoritative-unified-border-overlay.png",
        optimize=True,
    )
    Image.fromarray(hybrid_interior.astype(np.uint8) * 255, mode="L").save(
        run_dir / "web-mercator-authoritative-mainland-interior-mask.png",
        optimize=True,
    )
    Image.fromarray(
        _diagnostic(warped, coast_mask, land_mask, county_profiles), mode="RGB"
    ).save(run_dir / "hybrid-perimeter-diagnostic.jpg", quality=95, subsampling=0)

    artifacts = [
        "web-mercator-hybrid-state-overlay.png",
        "web-mercator-authoritative-coast-overlay.png",
        "web-mercator-authoritative-land-border-overlay.png",
        "web-mercator-authoritative-unified-border-overlay.png",
        "web-mercator-authoritative-mainland-interior-mask.png",
        "hybrid-perimeter-diagnostic.jpg",
    ]
    report = {
        "schema_version": 2,
        "status": "pass_no_additional_warp",
        "method": "regional_hybrid_perimeter_authority",
        "alignment": {"path": str(alignment_path), "sha256": _sha256(alignment_path)},
        "parent_refinement": {
            "path": str(refinement_path),
            "sha256": _sha256(refinement_path),
        },
        "regional_authority": {
            "coast": "registered_county_png_state_stroke",
            "Tahoe_hinge": "registered_county_png_state_stroke",
            "land_borders": "Census_2025_state_geometry",
            "lower_Colorado": "Census_2025_state_geometry",
            "rationale": (
                "county.png preserves source-matching coastal detail; Census is "
                "more accurate on the Colorado River and straight land borders; "
                "county.png is independently closer at the Tahoe hinge"
            ),
        },
        "coast": {
            "county_png": county_summary,
            "census": census_summary,
            "paired_windows": paired,
            "county_png_better_fraction": better_fraction,
            "passed": coast_pass,
        },
        "Tahoe_hinge": {
            "region": list(tahoe_region),
            "county_png": tahoe_county_summary,
            "census": tahoe_census_summary,
            "passed": tahoe_pass,
        },
        "southern_coast_seam": {
            "method": "versioned_WGS84_seed_to_registered_reference_switch",
            "seed_WGS84": list(SOUTHERN_COAST_SEAM_WGS84),
            "web_mercator_pixel": [float(value) for value in southern_seam],
            "county_png_coast_end_row_exclusive": coast_end_y,
            "clipped_by_southern_grid_edge": True,
            **seam_report,
        },
        "land_borders": {
            "upper_east": refinement["unchanged_region_veto"]["upper_east"]["after"],
            "mexico_border": refinement["unchanged_region_veto"]["mexico_border"]["after"],
            "lower_colorado_fit": refinement["fit_evidence"]["after"],
            "lower_colorado_holdout": refinement["independent_edge_holdouts"]["after"],
            "passed": land_pass,
        },
        "unified_border": {
            "definition": (
                "single_closed_exterior_of_a_filled_mainland_surface_using_"
                "registered_county_png_coast_and_Tahoe_hinge_plus_"
                "Census_2025_land_borders"
            ),
            "single_color_rgb": [80, 255, 120],
            "county_png_authority_pixel_count": int(np.count_nonzero(coast_mask)),
            "Census_authority_pixel_count": int(np.count_nonzero(land_mask)),
            "overlap_pixel_count": int(np.count_nonzero(coast_mask & land_mask)),
            "union_pixel_count": int(np.count_nonzero(coast_mask | land_mask)),
            "continuous_border_pixel_count": int(
                np.count_nonzero(continuous_border)
            ),
            "mainland_interior_pixel_count": int(
                np.count_nonzero(hybrid_interior)
            ),
            "connected_component_count": int(
                cv2.connectedComponents(continuous_border.astype(np.uint8), 8)[0]
                - 1
            ),
            "interior_is_exact_fill_of_displayed_border": True,
            "offshore_components_included": False,
            "passed": bool(
                np.array_equal(
                    np.asarray(
                        Image.open(
                            run_dir
                            / "web-mercator-authoritative-unified-border-overlay.png"
                        ).convert("RGBA")
                    )[..., 3]
                    > 0,
                    continuous_border,
                )
                and cv2.connectedComponents(
                    continuous_border.astype(np.uint8), 8
                )[0]
                - 1
                == 1
            ),
        },
        "decision": {
            "additional_warp": False,
            "reason": "current candidate passes both regional references",
            "publication_allowed": False,
            "author_review_required": True,
        },
        "artifacts": {
            name: {"path": name, "sha256": _sha256(run_dir / name)}
            for name in artifacts
        },
    }
    report_path = run_dir / "hybrid-perimeter-audit.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def audit_hybrid_perimeter(
    run_dir: Path,
    county_reference_path: Path,
    reference_root: Path,
) -> Dict[str, object]:
    first = _run_once(run_dir, county_reference_path, reference_root)
    names = [*first["artifacts"], "hybrid-perimeter-audit.json"]
    first_hashes = {name: _sha256(run_dir / name) for name in names}
    second = _run_once(run_dir, county_reference_path, reference_root)
    second_hashes = {name: _sha256(run_dir / name) for name in names}
    passed = first_hashes == second_hashes
    audit = {
        "schema_version": 1,
        "passed": passed,
        "method": "two_complete_same_path_rebuilds",
        "first": first_hashes,
        "second": second_hashes,
    }
    (run_dir / "hybrid-perimeter-determinism.json").write_text(
        json.dumps(audit, indent=2) + "\n"
    )
    if not passed:
        raise ValueError("Hybrid perimeter audit is not deterministic")
    return second
