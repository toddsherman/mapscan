"""Source-only topology adapter for the partial-coverage farms map.

The farms source is not a full California silhouette.  Its useful geographic
evidence consists of a neutral Pacific component, a flat neutral Nevada panel
whose west edge is the printed California/Nevada boundary, and neutral internal
county/region strokes.  This module isolates those channels without accepting
reference geometry, prior transforms, or manual control points.

Mapbox is deliberately absent from this API.  The alignment loop remains
responsible for ranking the resulting hypothesis and for enforcing the full
rendered state/coast and county gates.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Mapping

import cv2
import numpy as np


@dataclass(frozen=True)
class FarmsPartialTopology:
    """Source-native geographic channels on the complete working canvas."""

    state_coast: np.ndarray
    internal_topology: np.ndarray
    foreground_interior: np.ndarray
    foreground_boundary: np.ndarray
    layout_exclusion: np.ndarray
    neighboring_region: np.ndarray
    county_scope: np.ndarray
    ambiguous_topology_exclusion: np.ndarray
    southern_border_evidence: np.ndarray
    diagnostics: Mapping[str, Any]


def _mask_sha256(mask: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(mask.astype(np.uint8)).tobytes()).hexdigest()


def _large_white_layout_exclusion(rgb: np.ndarray) -> tuple[np.ndarray, list[int] | None]:
    """Exclude the upper-right legend/UI block using source layout only."""

    height, width = rgb.shape[:2]
    white = np.all(rgb >= 245, axis=2).astype(np.uint8)
    count, _labels, stats, _ = cv2.connectedComponentsWithStats(white, 8)
    candidates: list[tuple[int, int, int, int, int]] = []
    for label in range(1, count):
        x, y, component_width, component_height, area = map(int, stats[label])
        if (
            area >= height * width * 0.03
            and x >= width * 0.45
            and y <= height * 0.10
            and x + component_width < width
        ):
            candidates.append((area, x, y, component_width, component_height))

    excluded = np.zeros((height, width), dtype=bool)
    if not candidates:
        return excluded, None
    _area, x, y, component_width, component_height = max(candidates)
    # Exclude the measured UI rectangle, not an entire upper-right quadrant.
    # The flat Nevada panel and its vertical border run immediately beside and
    # below the legend; broad layout masking would destroy the best source-side
    # state-boundary evidence.
    padding = max(3, round(min(height, width) * 0.006))
    exclusion_x1 = max(0, x - padding)
    exclusion_y1 = max(0, y - padding)
    exclusion_x2 = min(width, x + component_width + padding)
    exclusion_y2 = min(height, y + component_height + padding)
    excluded[exclusion_y1:exclusion_y2, exclusion_x1:exclusion_x2] = True
    return excluded, [exclusion_x1, exclusion_y1, exclusion_x2, exclusion_y2]


def _right_flat_neighbor(
    rgb: np.ndarray,
    allowed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Recover Nevada fill and trace only its California-facing west edge.

    The dominant triangular panel touches the source's right side.  A second
    vertically elongated component of the same flat neutral color continues
    the border above the legend.  Selecting both by color and topology avoids
    treating arbitrary adjacent-state/county strokes as California's edge.
    """

    height, width = allowed.shape
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.int16)
    chroma = (
        rgb.max(axis=2).astype(np.int16) - rgb.min(axis=2).astype(np.int16)
    )
    best: tuple[float, int, np.ndarray, np.ndarray, int] | None = None
    reports: list[dict[str, Any]] = []
    for center in range(92, 193, 2):
        candidate = (
            (np.abs(gray - center) <= 5)
            & (chroma <= 8)
            & allowed
        ).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
        for label in range(1, count):
            x, y, component_width, component_height, area = map(
                int, stats[label]
            )
            right_fraction = (x + component_width) / width
            if (
                area < height * width * 0.012
                or x < width * 0.55
                or right_fraction < 0.95
                or y > height * 0.55
                or component_height < height * 0.12
            ):
                continue
            # Broad right contact and a tall component are characteristic of
            # the flat neighboring-state panel, unlike hillshade islands.
            score = (
                area / (height * width)
                + component_height / height * 0.02
                + right_fraction * 0.002
            )
            reports.append(
                {
                    "gray_center": center,
                    "box": [x, y, component_width, component_height],
                    "area_px": area,
                    "score": float(score),
                }
            )
            if best is None or score > best[0]:
                best = (score, center, labels, stats, label)
    if best is None:
        raise ValueError("No source-only flat right-side neighboring region found")

    _score, center, _selected_labels, _selected_stats, _selected_label = best
    candidate = (
        (np.abs(gray - center) <= 5)
        & (chroma <= 8)
        & allowed
    ).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
    main_candidates: list[tuple[int, int]] = []
    for label in range(1, count):
        x, y, component_width, component_height, area = map(int, stats[label])
        if (
            area >= height * width * 0.012
            and x >= width * 0.55
            and (x + component_width) / width >= 0.95
            and y <= height * 0.55
            and component_height >= height * 0.12
        ):
            main_candidates.append((area, label))
    if not main_candidates:
        raise ValueError("Flat neighboring-region candidate disappeared")
    _area, main_label = max(main_candidates)
    main_box = tuple(map(int, stats[main_label, :4]))
    main_x, main_y, _main_width, _main_height = main_box

    retained_labels = [main_label]
    companion_reports: list[dict[str, Any]] = []
    for label in range(1, count):
        if label == main_label:
            continue
        x, y, component_width, component_height, area = map(int, stats[label])
        retained = bool(
            area >= height * width * 0.003
            and component_height >= height * 0.12
            and y < main_y
            and x >= width * 0.50
            and x <= main_x + width * 0.03
            and x + component_width >= main_x - width * 0.12
        )
        if retained:
            retained_labels.append(label)
        if retained or area >= height * width * 0.002:
            companion_reports.append(
                {
                    "box": [x, y, component_width, component_height],
                    "area_px": area,
                    "retained": retained,
                }
            )

    neighboring = np.isin(labels, retained_labels)
    boundary = np.zeros((height, width), dtype=np.uint8)
    row_points: list[tuple[int, int]] = []
    for y in range(height):
        xs = np.flatnonzero(neighboring[y])
        if xs.size:
            row_points.append((int(xs.min()), y))
    if len(row_points) < max(20, round(height * 0.20)):
        raise ValueError("Neighboring region lacks a stable California-facing edge")

    runs: list[list[tuple[int, int]]] = []
    for point in row_points:
        if not runs or point[1] - runs[-1][-1][1] > 2:
            runs.append([point])
        else:
            runs[-1].append(point)
    for run in runs:
        if len(run) >= 2:
            cv2.polylines(
                boundary,
                [np.asarray(run, dtype=np.int32)],
                False,
                1,
                3,
                cv2.LINE_8,
            )
    connector_reports: list[dict[str, Any]] = []
    for first, second in zip(runs, runs[1:]):
        gap = second[0][1] - first[-1][1]
        retained = bool(gap <= height * 0.12)
        if retained:
            cv2.line(boundary, first[-1], second[0], 1, 3, cv2.LINE_8)
        connector_reports.append(
            {
                "from": [int(first[-1][0]), int(first[-1][1])],
                "to": [int(second[0][0]), int(second[0][1])],
                "row_gap": int(gap),
                "interpolated": retained,
            }
        )

    retained_stats = []
    for label in retained_labels:
        x, y, component_width, component_height, area = map(int, stats[label])
        retained_stats.append(
            {
                "box": [x, y, component_width, component_height],
                "area_px": area,
            }
        )
    return neighboring, boundary.astype(bool), {
        "method": "source_only_flat_right_neighbor_west_edge",
        "mapbox_used_for_neighbor_selection": False,
        "neutral_gray_center": int(center),
        "retained_components": retained_stats,
        "other_companions": companion_reports,
        "candidate_count": len(reports),
        "california_facing_edge_rows": len(row_points),
        "observed_edge_run_count": len(runs),
        "occlusion_connectors": connector_reports,
    }


def _exterior_boundary(mask: np.ndarray) -> np.ndarray:
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


def _internal_neutral_topology(
    rgb: np.ndarray,
    support: np.ndarray,
    state_coast: np.ndarray,
) -> np.ndarray:
    """Retain the printed neutral network inside the source-side CA support."""

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    chroma = (
        rgb.max(axis=2).astype(np.int16) - rgb.min(axis=2).astype(np.int16)
    )
    # At native resolution these strokes use a flat dark-neutral ink.  Area
    # downsampling raises their working-raster gray level, so retain the full
    # neutral range and use component extent (not a brittle dark cutoff) to
    # distinguish the connected cartographic network from relief flecks.
    ink = ((gray < 170) & (chroma < 30) & support).astype(np.uint8)
    # Bridge antialiasing gaps but reject glyph-sized or hillshade fragments.
    closed = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(closed, 8)
    height, width = support.shape
    topology = np.zeros_like(support, dtype=bool)
    for label in range(1, count):
        _x, _y, component_width, component_height, area = map(int, stats[label])
        if area < 8:
            continue
        if (
            component_width >= width * 0.025
            or component_height >= height * 0.025
        ):
            topology |= labels == label
    state_scope = cv2.dilate(
        state_coast.astype(np.uint8), np.ones((7, 7), dtype=np.uint8)
    ).astype(bool)
    topology &= ~state_scope
    # The thin inset frame is neutral too, but is page furniture rather than
    # geography.  Remove only a narrow edge band; state/county strokes may
    # still terminate immediately inside it on this partial-coverage source.
    edge_x = max(3, round(width * 0.028))
    edge_y = max(3, round(height * 0.022))
    topology[:edge_y] = False
    topology[-edge_y:] = False
    topology[:, :edge_x] = False
    topology[:, -edge_x:] = False
    return topology


def _source_only_southern_border(
    rgb: np.ndarray,
    pacific: np.ndarray,
    allowed: np.ndarray,
    east_boundary: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Detect the California-facing part of the printed southern border.

    The map continues into Arizona and Mexico below/right of California.  Its
    long inset frame and the Arizona/Mexico continuation are also horizontal,
    so a bare longest-line rule is unsafe.  A retained segment must instead:

    * occur below the observed Nevada trace but above the inset frame;
    * form a long, nearly horizontal neutral-ink run; and
    * begin beside the source-derived Pacific component.

    The result is source evidence only.  It is used to *omit* ambiguous county
    topology, never to invent an unprinted state boundary.
    """

    height, width = pacific.shape
    boundary_rows = np.flatnonzero(np.any(east_boundary, axis=1))
    if not boundary_rows.size:
        raise ValueError("Nevada edge is required before southern-border detection")
    last_east_row = int(boundary_rows.max())

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    chroma = (
        rgb.max(axis=2).astype(np.int16) - rgb.min(axis=2).astype(np.int16)
    )
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 35, 100)
    edges[(chroma > 42) | ~allowed] = 0
    raw_lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 720.0,
        threshold=max(12, round(width * 0.02)),
        minLineLength=max(24, round(width * 0.045)),
        maxLineGap=max(8, round(width * 0.025)),
    )
    if raw_lines is None:
        raise ValueError("No source-only southern-border line candidates found")

    minimum_y = max(last_east_row + max(12, round(height * 0.08)), round(height * 0.62))
    maximum_y = round(height * 0.91)
    intervals: list[dict[str, float]] = []
    for x1, y1, x2, y2 in raw_lines[:, 0]:
        length = math.hypot(float(x2 - x1), float(y2 - y1))
        angle = math.degrees(math.atan2(float(y2 - y1), float(x2 - x1)))
        if angle > 90.0:
            angle -= 180.0
        if angle < -90.0:
            angle += 180.0
        midpoint_y = (float(y1) + float(y2)) / 2.0
        west = float(min(x1, x2))
        east = float(max(x1, x2))
        if (
            abs(angle) <= 8.0
            and length >= width * 0.04
            and minimum_y <= midpoint_y <= maximum_y
        ):
            intervals.append(
                {
                    "west": west,
                    "east": east,
                    "y": midpoint_y,
                    "length": length,
                    "angle": angle,
                }
            )
    if not intervals:
        raise ValueError("No plausible source-only southern-border intervals found")

    # Merge collinear fragments across thematic occlusions.  The printed
    # United States/Mexico border continues from California through Arizona to
    # the right inset frame; dense crop pixels interrupt it for more than 100
    # working pixels in this source.  Requiring both Pacific and frame anchors
    # below prevents unrelated county lines from qualifying.
    row_tolerance = max(3.0, height * 0.006)
    gap_tolerance = max(24.0, width * 0.20)
    clusters: list[list[dict[str, float]]] = []
    for interval in sorted(intervals, key=lambda item: (item["y"], item["west"])):
        matches: list[tuple[float, int]] = []
        for index, cluster in enumerate(clusters):
            cluster_y = float(np.median([item["y"] for item in cluster]))
            cluster_west = min(item["west"] for item in cluster)
            cluster_east = max(item["east"] for item in cluster)
            horizontal_gap = max(
                0.0,
                interval["west"] - cluster_east,
                cluster_west - interval["east"],
            )
            if abs(interval["y"] - cluster_y) <= row_tolerance and horizontal_gap <= gap_tolerance:
                matches.append((horizontal_gap, index))
        if matches:
            clusters[min(matches)[1]].append(interval)
        else:
            clusters.append([interval])
    # A bridge fragment can connect two clusters that were created earlier in
    # sorted order.  Collapse those transitive connections deterministically.
    merged = True
    while merged:
        merged = False
        for first_index in range(len(clusters)):
            if merged:
                break
            first = clusters[first_index]
            first_y = float(np.median([item["y"] for item in first]))
            first_west = min(item["west"] for item in first)
            first_east = max(item["east"] for item in first)
            for second_index in range(first_index + 1, len(clusters)):
                second = clusters[second_index]
                second_y = float(np.median([item["y"] for item in second]))
                second_west = min(item["west"] for item in second)
                second_east = max(item["east"] for item in second)
                horizontal_gap = max(
                    0.0,
                    second_west - first_east,
                    first_west - second_east,
                )
                if abs(first_y - second_y) <= row_tolerance and horizontal_gap <= gap_tolerance:
                    clusters[first_index] = first + second
                    del clusters[second_index]
                    merged = True
                    break

    pacific_distance = cv2.distanceTransform(
        (~pacific).astype(np.uint8), cv2.DIST_L2, 5
    )
    reports: list[dict[str, Any]] = []
    selected: tuple[float, dict[str, Any]] | None = None
    for cluster in clusters:
        west = int(math.floor(min(item["west"] for item in cluster)))
        east = int(math.ceil(max(item["east"] for item in cluster)))
        y = int(round(float(np.median([item["y"] for item in cluster]))))
        span = east - west + 1
        x0 = max(0, west - max(3, round(width * 0.015)))
        x1 = min(width, west + max(4, round(width * 0.025)) + 1)
        y0 = max(0, y - max(3, round(height * 0.01)))
        y1 = min(height, y + max(3, round(height * 0.01)) + 1)
        pacific_gap = float(np.min(pacific_distance[y0:y1, x0:x1]))
        supported = bool(
            span >= width * 0.55
            and pacific_gap <= max(35.0, width * 0.075)
            and east >= width * 0.93
        )
        # Retain the uninterrupted Pacific-anchored portion separately.  It is
        # the only source-proven California-side southern span; later
        # fragments belong to the cross-state continuation after thematic
        # occlusions and cannot identify the Colorado intersection.
        core_fragments = sorted(cluster, key=lambda item: (item["west"], item["east"]))
        core_east = int(math.ceil(core_fragments[0]["east"]))
        core_gap_limit = max(12.0, width * 0.05)
        for fragment in core_fragments[1:]:
            if fragment["west"] - core_east > core_gap_limit:
                break
            core_east = max(core_east, int(math.ceil(fragment["east"])))
        report = {
            "west_x": west,
            "east_x": east,
            "pacific_anchored_core_east_x": core_east,
            "pacific_anchored_core_span_px": core_east - west + 1,
            "row_y": y,
            "span_px": span,
            "pacific_gap_px": pacific_gap,
            "hough_fragment_count": len(cluster),
            "supported": supported,
        }
        reports.append(report)
        if supported:
            score = float(span - 2.0 * pacific_gap)
            if selected is None or score > selected[0]:
                selected = (score, report)
    if selected is None:
        raise ValueError("Southern border lacks Pacific-adjacent source support")

    report = dict(selected[1])
    border = np.zeros((height, width), dtype=np.uint8)
    cv2.line(
        border,
        (int(report["west_x"]), int(report["row_y"])),
        (int(report["east_x"]), int(report["row_y"])),
        1,
        3,
        cv2.LINE_8,
    )
    border = border.astype(bool) & allowed
    report.update(
        {
            "method": "source_only_pacific_adjacent_neutral_southern_border",
            "candidate_clusters": reports,
            "mapbox_used": False,
            "evidence_sha256": _mask_sha256(border),
        }
    )
    return border, report


def _scope_internal_topology(
    topology: np.ndarray,
    allowed: np.ndarray,
    pacific: np.ndarray,
    neighboring: np.ndarray,
    east_boundary: np.ndarray,
    southern_border: np.ndarray,
    southern_border_diagnostics: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Omit source-proven adjacent networks without drawing Colorado River.

    The Colorado River boundary is largely obscured by dense thematic pixels.
    Rather than hallucinate it, the adapter applies two omission-only rules:

    * nothing south of the observed U.S./Mexico line is county evidence; and
    * below the observed Nevada exit, every neutral-line component that reaches
      the right inset-frame corridor is ambiguous adjacent-state evidence.

    The latter may omit a genuine line that connects through an occlusion, but
    it cannot invent a California match.  Non-right-touching southern-California
    components remain available.
    """

    height, width = allowed.shape
    east_rows = np.flatnonzero(np.any(east_boundary, axis=1))
    south_rows = np.flatnonzero(np.any(southern_border, axis=1))
    if not east_rows.size or not south_rows.size:
        raise ValueError("Observed east and southern borders are required")
    east_last_y = int(east_rows.max())
    south_y = int(round(float(np.median(south_rows))))
    south_xs = np.flatnonzero(southern_border[south_y])
    south_west_x = int(south_xs.min())
    south_east_x = int(south_xs.max())
    if south_y <= east_last_y + max(20, round(height * 0.12)):
        raise ValueError("Southern border is not separated from the Nevada exit")

    south_core_east_x = int(
        southern_border_diagnostics["pacific_anchored_core_east_x"]
    )
    if not (south_west_x < south_core_east_x <= south_east_x):
        raise ValueError("Southern-border California-side core is invalid")
    south_core_span = south_core_east_x - south_west_x + 1
    south_span = south_east_x - south_west_x + 1
    inward_guard = max(3, round(min(height, width) * 0.006))
    scope = allowed & ~pacific & ~neighboring
    scope[max(0, south_y - inward_guard + 1) :] = False

    # A conservative, omission-only envelope connects the observed Nevada
    # exit to the uninterrupted Pacific-side southern span.  It is *not* added
    # to state geometry and is intentionally pulled west quickly, so an
    # obscured Colorado River can never authorize Arizona county matches.
    east_last_xs = np.flatnonzero(east_boundary[east_last_y])
    east_last_x = int(round(float(np.median(east_last_xs))))
    transition_rows = max(24, round(height * 0.07))
    transition_end = min(south_y, east_last_y + transition_rows)
    conservative_cap = min(
        east_last_x,
        south_core_east_x + max(18, round(south_core_span * 0.25)),
    )
    limits = np.full(height, width - 1, dtype=np.float64)
    first_rows = np.arange(east_last_y + 1, transition_end + 1)
    if first_rows.size:
        limits[first_rows] = np.linspace(
            east_last_x,
            conservative_cap,
            len(first_rows) + 1,
        )[1:]
    later_rows = np.arange(transition_end + 1, south_y + 1)
    if later_rows.size:
        limits[later_rows] = np.linspace(
            conservative_cap,
            south_core_east_x,
            len(later_rows) + 1,
        )[1:]
    limits[east_last_y + 1 : south_y + 1] -= inward_guard
    x_grid = np.arange(width)[None, :]
    scope &= x_grid <= limits[:, None]

    # ``_internal_neutral_topology`` already removes the outer 2.8% furniture
    # band, including the two-pixel inset frame.  Test components against the
    # inner edge of that same source-only band so frame-terminating lines are
    # not accidentally disconnected before this classification.
    frame_corridor = max(20, round(width * 0.033))
    closed = cv2.morphologyEx(
        topology.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(closed, 8)
    # Southern clipping is pixel-local: a California component that merely
    # terminates at the international border keeps its northern portion.
    ambiguous = topology & ~scope
    rejected_components: list[dict[str, Any]] = []
    for label in range(1, count):
        component = labels == label
        lower = component.copy()
        lower[: east_last_y + 1] = False
        touches_right = bool(np.any(lower[:, width - frame_corridor :]))
        south_of_border = bool(np.any(component[max(0, south_y - inward_guard + 1) :]))
        if not (touches_right or south_of_border):
            continue
        original_pixels = topology & component
        if not np.any(original_pixels):
            continue
        if touches_right:
            ambiguous |= original_pixels
        elif south_of_border:
            local_south = np.zeros_like(topology, dtype=bool)
            local_south[max(0, south_y - inward_guard + 1) :] = True
            ambiguous |= original_pixels & local_south
        x, y, component_width, component_height, area = map(int, stats[label])
        rejected_components.append(
            {
                "bbox_working_px": [x, y, component_width, component_height],
                "closed_component_pixels": area,
                "topology_pixels": int(np.count_nonzero(original_pixels)),
                "touches_right_frame_below_nevada_exit": touches_right,
                "crosses_or_lies_south_of_southern_border": south_of_border,
            }
        )
    scoped_topology = topology & scope & ~ambiguous
    diagnostics = {
        "method": "source_only_southern_clip_and_right_frame_connected_adjacent_network_omission",
        "east_exit_last_row": east_last_y,
        "east_exit_working_px": [east_last_x, east_last_y],
        "southern_border_working_px": [
            south_west_x,
            south_y,
            south_east_x,
            south_y,
        ],
        "southern_border_span_px": south_span,
        "pacific_anchored_southern_core_east_x": south_core_east_x,
        "pacific_anchored_southern_core_span_px": south_core_span,
        "inward_guard_px": inward_guard,
        "conservative_omission_transition_rows": transition_rows,
        "conservative_omission_transition_end_row": transition_end,
        "conservative_omission_cap_x": conservative_cap,
        "right_frame_corridor_px": frame_corridor,
        "rejected_component_count": len(rejected_components),
        "rejected_components": rejected_components,
        "input_internal_topology_pixels": int(np.count_nonzero(topology)),
        "retained_internal_topology_pixels": int(np.count_nonzero(scoped_topology)),
        "omitted_internal_topology_pixels": int(np.count_nonzero(topology & ~scoped_topology)),
        "county_scope_pixels": int(np.count_nonzero(scope)),
        "ambiguous_adjacent_topology_pixels": int(np.count_nonzero(ambiguous)),
        "county_scope_sha256": _mask_sha256(scope),
        "ambiguous_exclusion_sha256": _mask_sha256(ambiguous),
        "retained_internal_topology_sha256": _mask_sha256(scoped_topology),
        "colorado_boundary_inferred_as_geometry": False,
        "omission_envelope_is_not_alignment_evidence": True,
        "ambiguous_lower_right_omitted_with_warning": True,
        "mapbox_used": False,
    }
    return scoped_topology, scope, ambiguous, diagnostics


def render_farms_county_scope_overlay(
    rgb: np.ndarray, topology: FarmsPartialTopology
) -> np.ndarray:
    """Render retained and omitted source county evidence for visual audit."""

    if rgb.shape[:2] != topology.county_scope.shape:
        raise ValueError("Overlay RGB and topology masks must share a canvas")
    canvas = rgb.copy()
    omitted_topology = topology.ambiguous_topology_exclusion & (
        cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) < 170
    )
    canvas[omitted_topology] = (255, 55, 45)
    canvas[topology.internal_topology] = (255, 0, 220)
    canvas[topology.state_coast] = (0, 235, 255)
    canvas[topology.southern_border_evidence] = (255, 225, 0)
    return canvas


def derive_farms_partial_topology(
    rgb: np.ndarray,
    pacific: np.ndarray,
) -> FarmsPartialTopology:
    """Derive isolated partial California evidence from the farms source.

    Args:
        rgb: RGB source raster at the alignment working resolution.
        pacific: Source-derived neutral Pacific component on the same canvas.

    Returns:
        Full-canvas source masks and deterministic, auditable diagnostics.

    Raises:
        ValueError: if source layout/topology does not support this adapter.
    """

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must have shape (height, width, 3)")
    if pacific.shape != rgb.shape[:2]:
        raise ValueError("pacific must match the RGB canvas")
    if pacific.dtype != bool:
        pacific = pacific.astype(bool)
    height, width = pacific.shape
    excluded, exclusion_box = _large_white_layout_exclusion(rgb)
    allowed = ~excluded
    margin = max(3, round(min(height, width) * 0.012))
    allowed[:margin] = False
    allowed[-margin:] = False
    allowed[:, :margin] = False
    allowed[:, -margin:] = False

    neighboring, east_boundary, neighbor_diagnostics = _right_flat_neighbor(
        rgb, allowed
    )
    coastline = _exterior_boundary(pacific)
    ys, xs = np.nonzero(pacific)
    if not len(xs):
        raise ValueError("Pacific component is empty")
    coastline[:, : int(xs.min()) + margin] = False
    coastline[int(ys.max()) - margin :] = False
    coastline &= allowed

    # ``east_boundary`` contains only the traced California-facing edge plus a
    # deterministic interpolation across the legend occlusion.  Preserve that
    # connector through the excluded UI rectangle; no UI pixels themselves are
    # admitted to the semantic channel.
    state_coast = (coastline & allowed) | east_boundary
    if np.count_nonzero(coastline) < 100 or np.count_nonzero(east_boundary) < 100:
        raise ValueError("Partial source lacks both coast and Nevada-border evidence")

    # Keep only the California-facing (west/southwest) side of the traced
    # Nevada edge while that edge is visible.  Below its exit, this mask alone
    # is intentionally noncommittal; the southern-border and right-connected
    # network rules below conservatively remove adjacent geography without
    # pretending to reconstruct the obscured Colorado River.
    california_side = np.ones((height, width), dtype=bool)
    boundary_rows = np.flatnonzero(np.any(east_boundary, axis=1))
    if boundary_rows.size:
        first_row = int(boundary_rows.min())
        last_row = int(boundary_rows.max())
        observed_rows: list[int] = []
        observed_limits: list[float] = []
        for y in boundary_rows:
            xs_on_edge = np.flatnonzero(east_boundary[y])
            if xs_on_edge.size:
                observed_rows.append(int(y))
                observed_limits.append(float(np.median(xs_on_edge)))
        rows = np.arange(first_row, last_row + 1)
        limits = np.interp(rows, observed_rows, observed_limits)
        extended_rows = np.arange(0, last_row + 1)
        extended_limits = np.interp(
            extended_rows,
            rows,
            limits,
            left=float(limits[0]),
            right=float(limits[-1]),
        )
        x_grid = np.arange(width)[None, :]
        california_side[: last_row + 1] = x_grid <= (
            extended_limits[:, None] + 4.0
        )
    else:
        raise ValueError("Nevada edge does not define a California-facing side")

    southern_border, southern_border_diagnostics = _source_only_southern_border(
        rgb, pacific, allowed, east_boundary
    )
    foreground = allowed & ~pacific & ~neighboring & california_side
    raw_internal_topology = _internal_neutral_topology(
        rgb, foreground, state_coast
    )
    (
        internal_topology,
        county_scope,
        ambiguous_topology_exclusion,
        county_scope_diagnostics,
    ) = _scope_internal_topology(
        raw_internal_topology,
        allowed,
        pacific,
        neighboring,
        east_boundary,
        southern_border,
        southern_border_diagnostics,
    )
    # Positive California support must be no broader than the county evidence
    # that the partial source can actually distinguish.  Pacific, Nevada,
    # Mexico, and ambiguous right-connected adjacent networks remain observed
    # negative evidence rather than silently becoming California.
    foreground &= county_scope
    foreground_boundary = state_coast.copy()

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    thematic = (hsv[:, :, 1] >= 55) & (hsv[:, :, 2] >= 45) & allowed
    thematic_count = int(np.count_nonzero(thematic))
    thematic_in_support = int(np.count_nonzero(thematic & foreground))
    diagnostics = {
        "method": "source_only_partial_california_coast_nevada_and_internal_topology",
        "mapbox_used_for_hypothesis_construction": False,
        "layout_exclusion_working": exclusion_box,
        "pacific_area_px": int(np.count_nonzero(pacific)),
        "coastline_px": int(np.count_nonzero(coastline)),
        "east_state_boundary_px": int(np.count_nonzero(east_boundary)),
        "state_coast_px": int(np.count_nonzero(state_coast)),
        "raw_internal_topology_px": int(
            np.count_nonzero(raw_internal_topology)
        ),
        "internal_topology_px": int(np.count_nonzero(internal_topology)),
        "foreground_area_px": int(np.count_nonzero(foreground)),
        "california_facing_side_area_px": int(
            np.count_nonzero(california_side)
        ),
        "east_boundary_first_row": int(boundary_rows.min()),
        "east_boundary_last_row": int(boundary_rows.max()),
        "southern_border": southern_border_diagnostics,
        "county_scope": county_scope_diagnostics,
        "thematic_support_fraction": (
            float(thematic_in_support / thematic_count) if thematic_count else None
        ),
        "neighboring_region": neighbor_diagnostics,
        "authority": {
            "manual_inputs_used": False,
            "prior_run_artifacts_used": False,
            "county_png_used": False,
            "mapbox_used_for_hypothesis_construction": False,
            "mapbox_required_for_acceptance": True,
        },
    }
    return FarmsPartialTopology(
        state_coast=state_coast,
        internal_topology=internal_topology,
        foreground_interior=foreground,
        foreground_boundary=foreground_boundary,
        layout_exclusion=excluded,
        neighboring_region=neighboring,
        county_scope=county_scope,
        ambiguous_topology_exclusion=ambiguous_topology_exclusion,
        southern_border_evidence=southern_border,
        diagnostics=diagnostics,
    )
