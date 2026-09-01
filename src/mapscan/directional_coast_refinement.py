"""Apply a small directional correction to the canonical western coast.

This stage is for review findings that identify a coherent coast-only drift but
do not justify changing an already-accepted global transform.  Controls are
sampled from the approved county-derived mainland line.  The eastern envelope
and all approved offshore components receive zero-motion pins, so the compact
Wendland correction decays before reaching evidence that is already aligned.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .continuous_extraction import _alignment_transform
from .extraction import warp_classified_to_web_mercator
from .refinement import fit_local_review_corrections
from .reference import load_california


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _line_mask(path: Path) -> np.ndarray:
    rgba = np.asarray(Image.open(path).convert("RGBA"))
    return rgba[..., 3] > 0


def _envelope_points(
    mask: np.ndarray,
    *,
    count: int,
    start_y_fraction: float,
    end_y_fraction: float,
    side: str,
    maximum_x_fraction: float | None = None,
) -> np.ndarray:
    if count < 2:
        raise ValueError("At least two envelope controls are required")
    if not 0 <= start_y_fraction < end_y_fraction <= 1:
        raise ValueError("Envelope fractions must be ordered within zero and one")
    if side not in {"west", "east"}:
        raise ValueError("Envelope side must be west or east")
    height, width = mask.shape
    points = []
    used_rows = set()
    for requested_y in np.linspace(
        start_y_fraction * height, end_y_fraction * height, count
    ):
        candidates = []
        for y in range(max(0, round(requested_y) - 10), min(height, round(requested_y) + 11)):
            x_values = np.flatnonzero(mask[y])
            if side == "west" and maximum_x_fraction is not None:
                x_values = x_values[x_values < width * maximum_x_fraction]
            if not len(x_values):
                continue
            x = int(x_values.min() if side == "west" else x_values.max())
            candidates.append((abs(y - requested_y), y, x))
        if not candidates:
            continue
        _, y, x = min(candidates)
        if y in used_rows:
            continue
        used_rows.add(y)
        points.append((float(x), float(y)))
    if len(points) < 2:
        raise ValueError(f"Could not sample the canonical {side} envelope")
    return np.asarray(points, dtype=np.float64)


def _component_pin_points(mask: np.ndarray) -> list[np.ndarray]:
    y, x = np.nonzero(mask)
    if not len(x):
        return []
    points = np.column_stack((x, y)).astype(np.float64)
    center = np.asarray([float(np.mean(x)), float(np.mean(y))])

    def centered_extreme(axis: int, value: float) -> np.ndarray:
        candidates = points[points[:, axis] == value]
        other = 1 - axis
        return candidates[int(np.argmin(np.abs(candidates[:, other] - center[other])))]

    candidates = [
        center,
        centered_extreme(0, float(np.min(x))),
        centered_extreme(0, float(np.max(x))),
        centered_extreme(1, float(np.min(y))),
        centered_extreme(1, float(np.max(y))),
    ]
    result = []
    seen = set()
    for point in candidates:
        key = tuple(np.rint(point).astype(int))
        if key in seen:
            continue
        seen.add(key)
        result.append(point)
    return result


def _horizontal_edge_pin_points(
    mask: np.ndarray, *, edge: str, count: int, band_fraction: float = 0.012
) -> np.ndarray:
    """Sample fixed pins along the already-aligned north or south edge."""

    if edge not in {"north", "south"}:
        raise ValueError("Horizontal edge must be north or south")
    if count < 2:
        raise ValueError("At least two horizontal-edge pins are required")
    y, x = np.nonzero(mask)
    if not len(x):
        raise ValueError("Canonical mainland line is empty")
    band_px = max(4, round(mask.shape[0] * band_fraction))
    edge_y = int(np.min(y) if edge == "north" else np.max(y))
    selected = y <= edge_y + band_px if edge == "north" else y >= edge_y - band_px
    points = np.column_stack((x[selected], y[selected])).astype(np.float64)
    if len(points) < count:
        raise ValueError(f"Could not sample the canonical {edge} edge")
    order = np.argsort(points[:, 0])
    points = points[order]
    indexes = np.rint(
        np.linspace(len(points) * 0.18, len(points) * 0.82 - 1, count)
    ).astype(int)
    return points[np.clip(indexes, 0, len(points) - 1)]


def _correction_item(
    identifier: str,
    target: np.ndarray,
    current: np.ndarray,
    evidence: Dict[str, object],
) -> Dict[str, object]:
    return {
        "id": identifier,
        "reference": {"pixel": {"x": float(target[0]), "y": float(target[1])}},
        "source": {"pixel": {"x": float(current[0]), "y": float(current[1])}},
        "evidence": evidence,
    }


def refine_directional_west_coast(
    alignment_path: Path,
    canonical_boundary_manifest_path: Path,
    output_dir: Path,
    *,
    northward_shift_px: float = 0.0,
    westward_shift_px: float = 0.0,
    radius_px: float = 380.0,
    coast_control_count: int = 13,
    east_pin_count: int = 9,
    start_y_fraction: float = 0.05,
    end_y_fraction: float = 0.94,
    horizontal_edge_pin_count: int = 0,
    image_path: Path | None = None,
    reference_root: Path | None = None,
    target_height: int = 1014,
) -> Dict[str, object]:
    """Create a compact directional coast correction and pinned child alignment."""

    if northward_shift_px < 0 or westward_shift_px < 0:
        raise ValueError("Directional coast shifts cannot be negative")
    if northward_shift_px == 0 and westward_shift_px == 0:
        raise ValueError("At least one directional coast shift must be positive")
    alignment_path = alignment_path.resolve()
    canonical_boundary_manifest_path = canonical_boundary_manifest_path.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    canonical = json.loads(canonical_boundary_manifest_path.read_text())
    grid = canonical["source_grid"]
    root = canonical_boundary_manifest_path.parent
    mainland_path = root / str(canonical["artifacts"]["mainland"]["path"])
    islands_path = root / str(canonical["artifacts"]["islands"]["path"])
    mainland = _line_mask(mainland_path)
    islands = _line_mask(islands_path)
    expected_shape = (int(grid["height"]), int(grid["width"]))
    if mainland.shape != expected_shape or islands.shape != expected_shape:
        raise ValueError("Canonical boundary artifacts do not match their declared grid")

    coast = _envelope_points(
        mainland,
        count=coast_control_count,
        start_y_fraction=start_y_fraction,
        end_y_fraction=end_y_fraction,
        side="west",
        maximum_x_fraction=0.72,
    )
    east = _envelope_points(
        mainland,
        count=east_pin_count,
        start_y_fraction=start_y_fraction,
        end_y_fraction=0.90,
        side="east",
    )
    north_edge = (
        _horizontal_edge_pin_points(
            mainland, edge="north", count=horizontal_edge_pin_count
        )
        if horizontal_edge_pin_count
        else np.empty((0, 2), dtype=np.float64)
    )
    south_edge = (
        _horizontal_edge_pin_points(
            mainland, edge="south", count=horizontal_edge_pin_count
        )
        if horizontal_edge_pin_count
        else np.empty((0, 2), dtype=np.float64)
    )
    corrections = []
    # A feature that should move west/north currently lies east/south of the
    # desired target. The inverse sampling correction therefore uses positive
    # x/y deltas at the authoritative target controls.
    shift = np.asarray(
        [float(westward_shift_px), float(northward_shift_px)]
    )
    for index, target in enumerate(coast):
        corrections.append(
            _correction_item(
                f"west_coast_north_{index:02d}",
                target,
                target + shift,
                {
                    "kind": "reviewed_directional_coast_drift",
                    "desired_output_motion": {
                        "west_px": float(westward_shift_px),
                        "north_px": float(northward_shift_px),
                    },
                },
            )
        )
    for index, point in enumerate(east):
        corrections.append(
            _correction_item(
                f"east_border_pin_{index:02d}",
                point,
                point,
                {"kind": "fixed_pin", "reason": "eastern border already aligned"},
            )
        )
    for edge_name, points in (("north", north_edge), ("south", south_edge)):
        for index, point in enumerate(points):
            corrections.append(
                _correction_item(
                    f"{edge_name}_edge_pin_{index:02d}",
                    point,
                    point,
                    {
                        "kind": "fixed_pin",
                        "reason": f"{edge_name} edge already aligned",
                    },
                )
            )

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        islands.astype(np.uint8), connectivity=8
    )
    island_pin_count = 0
    island_components = 0
    for label in range(1, component_count):
        if int(stats[label, cv2.CC_STAT_AREA]) < 20:
            continue
        island_components += 1
        for point_index, point in enumerate(_component_pin_points(labels == label)):
            corrections.append(
                _correction_item(
                    f"island_{island_components:02d}_pin_{point_index:02d}",
                    point,
                    point,
                    {"kind": "fixed_pin", "reason": "approved island already aligned"},
                )
            )
            island_pin_count += 1

    correction_record = {
        "schema_version": 1,
        "direction": "authoritative_reference_to_current_warped_source",
        "generated_by": "mapscan.directional_coast_refinement",
        "grid": grid,
        "corrections": corrections,
    }
    corrections_path = output_dir / "coast-corrections.json"
    corrections_path.write_text(json.dumps(correction_record, indent=2) + "\n")
    fit_dir = output_dir / "fit"
    fit_report = fit_local_review_corrections(
        alignment_path, corrections_path, fit_dir, radius_px=radius_px
    )
    candidate_alignment_path = fit_dir / "alignment.json"
    candidate_alignment = json.loads(candidate_alignment_path.read_text())
    local_operation = candidate_alignment["web_mercator_correction"]["operations"][0]
    deltas = np.asarray(local_operation["sampling_delta_at_controls_px"])
    coast_deltas = deltas[: len(coast)]
    pin_deltas = deltas[len(coast) :]
    if not np.allclose(
        coast_deltas, [westward_shift_px, northward_shift_px]
    ):
        raise ValueError("Fitted coast correction does not preserve requested motion")
    maximum_pin_delta = float(np.max(np.abs(pin_deltas))) if len(pin_deltas) else 0.0
    if maximum_pin_delta > 1e-9:
        raise ValueError("A fixed border or island pin moved")

    source_review = None
    if image_path is not None:
        image_path = image_path.resolve()
        if reference_root is None:
            raise ValueError("Source-only review rendering requires a reference root")
        source_rgb = np.asarray(Image.open(image_path).convert("RGB"))
        state, _ = load_california(reference_root.resolve())
        warped_source, review_grid = warp_classified_to_web_mercator(
            source_rgb,
            state,
            _alignment_transform(candidate_alignment),
            source_rgb.shape[:2],
            target_height=target_height,
            clip_to_state=False,
        )
        source_path = fit_dir / "web-mercator-source.png"
        Image.fromarray(warped_source).save(source_path, optimize=True)
        no_extraction_path = fit_dir / "no-extraction.png"
        Image.new(
            "RGBA", (warped_source.shape[1], warped_source.shape[0]), (0, 0, 0, 0)
        ).save(no_extraction_path, optimize=True)
        source_review = {
            "source": {"path": source_path.name, "sha256": _sha256(source_path)},
            "no_extraction_overlay": {
                "path": no_extraction_path.name,
                "sha256": _sha256(no_extraction_path),
            },
            "grid": review_grid,
            "classification_performed": False,
        }

    scale = min(1.0, 1400.0 / max(expected_shape))
    preview_size = (
        max(1, round(expected_shape[1] * scale)),
        max(1, round(expected_shape[0] * scale)),
    )
    diagnostic = Image.new("RGB", preview_size, (18, 18, 22))
    draw = ImageDraw.Draw(diagnostic)
    for item in corrections:
        start = item["reference"]["pixel"]
        end = item["source"]["pixel"]
        start_xy = (start["x"] * scale, start["y"] * scale)
        end_xy = (end["x"] * scale, end["y"] * scale)
        color = (95, 255, 92) if item["evidence"]["kind"] != "fixed_pin" else (80, 210, 255)
        draw.line((start_xy, end_xy), fill=color, width=max(1, round(2 * scale)))
        radius = max(2, round(5 * scale))
        draw.ellipse(
            (start_xy[0] - radius, start_xy[1] - radius, start_xy[0] + radius, start_xy[1] + radius),
            fill=color,
        )
    diagnostic_path = output_dir / "directional-coast-controls.png"
    diagnostic.save(diagnostic_path, optimize=True)

    report = {
        "schema_version": 1,
        "status": "needs_visual_review",
        "parent_alignment": {"path": str(alignment_path), "sha256": _sha256(alignment_path)},
        "candidate_alignment": {
            "path": str(candidate_alignment_path),
            "sha256": _sha256(candidate_alignment_path),
        },
        "canonical_boundary": {
            "path": str(canonical_boundary_manifest_path),
            "sha256": _sha256(canonical_boundary_manifest_path),
            "id": canonical.get("canonical_boundary_id"),
        },
        "policy": {
            "western_mainland": (
                f"move west {westward_shift_px:g} and north {northward_shift_px:g} "
                "canonical-grid pixels"
            ),
            "eastern_border": "fixed envelope pins",
            "north_and_south_edges": "fixed interior edge pins",
            "offshore_islands": "fixed center and four-extrema pins",
            "support": "compact Wendland C2 decay",
        },
        "controls": {
            "coast": int(len(coast)),
            "east_border_pins": int(len(east)),
            "north_edge_pins": int(len(north_edge)),
            "south_edge_pins": int(len(south_edge)),
            "island_components": island_components,
            "island_pins": island_pin_count,
            "radius_px": float(radius_px),
            "maximum_fixed_pin_delta_px": maximum_pin_delta,
        },
        "local_fit": fit_report,
        "source_only_review": source_review,
        "artifacts": {
            "corrections": {"path": corrections_path.name, "sha256": _sha256(corrections_path)},
            "diagnostic": {"path": diagnostic_path.name, "sha256": _sha256(diagnostic_path)},
        },
    }
    (output_dir / "directional-coast-refinement.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report
