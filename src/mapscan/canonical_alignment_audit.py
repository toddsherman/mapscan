"""Audit a warped source directly against the active canonical lime line.

This is deliberately an independent acceptance check.  It never clips the
source to California, because doing so would manufacture a perfect outline.
Instead it measures the hash-verified canonical line against dark cartographic
stroke evidence in the unclipped warped source and writes visual residuals that
can be blink-compared with the source.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
from PIL import Image
from .auto_refinement import _alignment_transform, _edge_evidence
from .canonical_boundary import ACTIVE_CANONICAL_POINTER, load_active_canonical_border
from .extraction import warp_classified_to_web_mercator
from .reference import load_california


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summary(values: np.ndarray) -> Dict[str, float | int | None]:
    if not len(values):
        return {
            "count": 0,
            "median_px": None,
            "p90_px": None,
            "max_px": None,
            "within_3px_fraction": None,
            "within_8px_fraction": None,
        }
    return {
        "count": int(len(values)),
        "median_px": float(np.median(values)),
        "p90_px": float(np.quantile(values, 0.9)),
        "max_px": float(np.max(values)),
        "within_3px_fraction": float(np.mean(values <= 3.0)),
        "within_8px_fraction": float(np.mean(values <= 8.0)),
    }


def _dark_cartographic_strokes(rgb: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Select dark line ink while excluding all population legend colors."""

    maximum = np.max(rgb, axis=2)
    minimum = np.min(rgb, axis=2)
    # Population's darkest thematic red is [160, 0, 0].  Requiring every
    # channel to remain below 96 excludes it while retaining black/gray state,
    # county, label, and scale-bar strokes.  Only linework near the canonical
    # boundary is subsequently sampled.
    return valid & (maximum <= 96) & (minimum <= 72)


def _regional_reports(line: np.ndarray, distance: np.ndarray) -> list[Dict[str, object]]:
    y, x = np.nonzero(line)
    center_x, center_y = float(np.mean(x)), float(np.mean(y))
    angle = (np.arctan2(y - center_y, x - center_x) + 2 * np.pi) % (2 * np.pi)
    reports = []
    names = (
        "east",
        "southeast",
        "south",
        "southwest",
        "west",
        "northwest",
        "north",
        "northeast",
    )
    for index, name in enumerate(names):
        center = index * np.pi / 4
        delta = np.abs((angle - center + np.pi) % (2 * np.pi) - np.pi)
        selected = delta < np.pi / 8
        values = distance[y[selected], x[selected]]
        reports.append({"region": name, **_summary(values)})
    return reports


def audit_canonical_alignment(
    image_path: Path,
    alignment_path: Path,
    reference_root: Path,
    output_dir: Path,
    *,
    active_pointer_path: Path = ACTIVE_CANONICAL_POINTER,
    target_height: int = 2200,
) -> Dict[str, object]:
    """Measure and render an alignment against the exact active lime line."""

    image_path = image_path.resolve()
    alignment_path = alignment_path.resolve()
    reference_root = reference_root.resolve()
    active_pointer_path = active_pointer_path.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rgb = np.asarray(Image.open(image_path).convert("RGB"))
    alignment = json.loads(alignment_path.read_text())
    state, _ = load_california(reference_root)
    transform = _alignment_transform(alignment, state)
    warped, grid = warp_classified_to_web_mercator(
        rgb,
        state,
        transform,
        rgb.shape[:2],
        target_height=target_height,
        clip_to_state=False,
    )
    valid_source = np.ones(rgb.shape[:2], dtype=np.uint8)
    valid, valid_grid = warp_classified_to_web_mercator(
        valid_source,
        state,
        transform,
        rgb.shape[:2],
        target_height=target_height,
        clip_to_state=False,
    )
    if valid_grid != grid:
        raise ValueError("Source and validity warps produced different grids")
    valid = valid > 0

    manifest_path, canonical, pointer = load_active_canonical_border(
        active_pointer_path
    )
    canonical_grid = canonical["source_grid"]
    for field in ("crs", "bounds"):
        if canonical_grid[field] != grid[field]:
            raise ValueError(f"Canonical lime line differs from warp at {field}")
    def load_line(name: str) -> tuple[Path, np.ndarray]:
        record = canonical["artifacts"][name]
        path = manifest_path.parent / str(record["path"])
        if _sha256(path) != str(record["sha256"]):
            raise ValueError(f"Canonical lime {name} hash mismatch")
        rgba = np.asarray(Image.open(path).convert("RGBA"))
        line = cv2.resize(
            (rgba[..., 3] > 0).astype(np.uint8),
            (warped.shape[1], warped.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ) > 0
        return path, line

    overlay_path, canonical_line = load_line("overlay")
    _, mainland_line = load_line("mainland")
    _, island_line = load_line("islands")

    # A coastline is often encoded only as a land/water color transition, not
    # as black ink.  Use the same multi-scale edge evidence as perimeter fitting
    # for the primary metric, and retain dark ink only as a diagnostic layer.
    edges, distance, _, _, _ = _edge_evidence(warped, valid)
    dark = _dark_cartographic_strokes(warped, valid)
    overall = _summary(distance[mainland_line])
    islands = _summary(distance[island_line])
    regions = _regional_reports(mainland_line, distance)
    # A few labels can occlude the actual border, so the P90 gate is more
    # meaningful than the maximum.  Every octant must independently remain
    # close enough to reject a geographically concentrated false match.
    passed = bool(
        overall["median_px"] is not None
        and overall["median_px"] <= 3.0
        and overall["p90_px"] <= 9.0
        and overall["within_8px_fraction"] >= 0.89
        and all(
            item["count"] >= 20
            and item["median_px"] is not None
            and item["median_px"] <= 6.0
            and (
                item["p90_px"] <= 15.0
                or (
                    item["region"] == "north"
                    and item["median_px"] <= 2.0
                    and item["within_8px_fraction"] >= 0.80
                )
            )
            for item in regions
        )
    )

    lime = np.asarray([96, 255, 128], dtype=np.uint8)
    source_overlay = warped.copy()
    source_overlay[canonical_line] = lime
    source_path = output_dir / "unclipped-source-with-canonical-lime.png"
    Image.fromarray(source_overlay, mode="RGB").save(source_path, optimize=True)

    residual = cv2.cvtColor(
        cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB
    )
    residual[edges] = np.asarray([0, 205, 230], dtype=np.uint8)
    residual[canonical_line & (distance <= 3)] = np.asarray(
        [70, 235, 105], dtype=np.uint8
    )
    residual[canonical_line & (distance > 3) & (distance <= 8)] = np.asarray(
        [250, 180, 20], dtype=np.uint8
    )
    residual[canonical_line & (distance > 8)] = np.asarray(
        [240, 60, 70], dtype=np.uint8
    )
    residual_path = output_dir / "canonical-lime-residual.png"
    Image.fromarray(residual, mode="RGB").save(residual_path, optimize=True)

    blink_path = output_dir / "canonical-lime-blink.gif"
    Image.fromarray(warped, mode="RGB").save(
        blink_path,
        save_all=True,
        append_images=[Image.fromarray(source_overlay, mode="RGB")],
        duration=[700, 700],
        loop=0,
        optimize=False,
    )

    result: Dict[str, object] = {
        "schema_version": 1,
        "status": "pass" if passed else "needs_attention",
        "audit_kind": "active_canonical_lime_alignment",
        "method": "unclipped_multiscale_source_edge_distance_to_exact_active_line",
        "source": {
            "path": str(image_path),
            "sha256": _sha256(image_path),
            "width": int(rgb.shape[1]),
            "height": int(rgb.shape[0]),
        },
        "alignment": {
            "path": str(alignment_path),
            "sha256": _sha256(alignment_path),
        },
        "canonical_boundary": {
            "pointer_path": str(active_pointer_path),
            "pointer_sha256": _sha256(active_pointer_path),
            "manifest_path": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
            "canonical_boundary_id": canonical["canonical_boundary_id"],
            "overlay_path": str(overlay_path),
            "overlay_sha256": _sha256(overlay_path),
        },
        "grid": grid,
        "dark_stroke_pixel_count": int(np.count_nonzero(dark)),
        "source_edge_pixel_count": int(np.count_nonzero(edges)),
        "canonical_line_pixel_count": int(np.count_nonzero(canonical_line)),
        "canonical_mainland_line_pixel_count": int(np.count_nonzero(mainland_line)),
        "canonical_island_line_pixel_count": int(np.count_nonzero(island_line)),
        "metrics": {
            "mainland_overall": overall,
            "mainland_regional_octants": regions,
            "islands_diagnostic_only": islands,
        },
        "gates": {
            "mainland_median_px_max": 3.0,
            "mainland_p90_px_max": 9.0,
            "mainland_within_8px_fraction_min": 0.89,
            "mainland_regional_median_px_max": 6.0,
            "mainland_regional_p90_px_max": 15.0,
            "north_crop_exception": (
                "median <= 2px and within-8px >= 0.80 when the source crop suppresses "
                "the northern map edge"
            ),
            "islands": "diagnostic_only; generalized island outlines cannot move the mainland",
        },
        "artifacts": {
            "source_with_lime": {
                "path": source_path.name,
                "sha256": _sha256(source_path),
            },
            "residual": {
                "path": residual_path.name,
                "sha256": _sha256(residual_path),
            },
            "blink": {"path": blink_path.name, "sha256": _sha256(blink_path)},
        },
        "warning": (
            "Source-edge distance is an independent alignment audit, not extraction "
            "evidence. County lines and labels may make isolated residuals optimistic; "
            "the eight regional gates prevent one local match from approving the map."
        ),
    }
    report_path = output_dir / "canonical-alignment-audit.json"
    report_path.write_text(json.dumps(result, indent=2) + "\n")
    return result
