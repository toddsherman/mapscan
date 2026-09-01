"""Refit a source map horizontally while preserving its eastern registration.

This is a source-first global refit.  It deliberately rejects an alignment
that already carries review corrections, then searches a narrow horizontal
scale range on the original source transform.  The mean source coordinate of
the authoritative eastern border is held fixed and all y parameters remain
byte-for-byte unchanged.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
from PIL import Image

from .alignment import _eastern_border_points
from .continuous_extraction import _alignment_transform
from .extraction import _normalized_to_source, warp_classified_to_web_mercator
from .reference import load_california


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _east_anchored_parameters(
    parameters: Dict[str, float],
    source_shape: Tuple[int, int],
    eastern_normalized_points: np.ndarray,
    reference_to_source_x_scale_multiplier: float,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Scale reference-to-source x while preserving the mean eastern anchor."""

    if reference_to_source_x_scale_multiplier <= 0:
        raise ValueError("Horizontal scale multiplier must be positive")
    if eastern_normalized_points.ndim != 2 or eastern_normalized_points.shape[1] != 2:
        raise ValueError("Eastern normalized controls must be an N by 2 array")
    height, width = source_shape
    before = _normalized_to_source(eastern_normalized_points, parameters, source_shape)
    fitted = copy.deepcopy(parameters)
    fitted["x_scale_ratio"] = float(
        parameters["x_scale_ratio"] * reference_to_source_x_scale_multiplier
    )
    # Solve the horizontal translation analytically so the complete eastern
    # control set keeps the same mean source x coordinate.
    provisional = _normalized_to_source(eastern_normalized_points, fitted, source_shape)
    fitted["center_x_fraction"] = float(
        fitted["center_x_fraction"]
        + (float(np.mean(before[:, 0])) - float(np.mean(provisional[:, 0]))) / width
    )
    after = _normalized_to_source(eastern_normalized_points, fitted, source_shape)
    displacement = after - before
    if not np.allclose(displacement[:, 1], 0.0, atol=1e-12):
        raise ValueError("Horizontal refit unexpectedly changed source y coordinates")
    return fitted, {
        "reference_to_source_x_scale_multiplier": float(
            reference_to_source_x_scale_multiplier
        ),
        "rendered_width_multiplier": float(
            1.0 / reference_to_source_x_scale_multiplier
        ),
        "eastern_anchor_mean_x_displacement_source_px": float(
            np.mean(displacement[:, 0])
        ),
        "eastern_anchor_median_abs_x_displacement_source_px": float(
            np.median(np.abs(displacement[:, 0]))
        ),
        "eastern_anchor_max_abs_x_displacement_source_px": float(
            np.max(np.abs(displacement[:, 0]))
        ),
        "source_y_displacement_px": 0.0,
        "source_width_px": int(width),
        "source_height_px": int(height),
    }


def _land_mask(rgb: np.ndarray) -> np.ndarray:
    """Separate the warm/green elevation surface from blue ocean context."""

    red = rgb[..., 0].astype(np.int16)
    green = rgb[..., 1].astype(np.int16)
    blue = rgb[..., 2].astype(np.int16)
    thematic = (np.maximum(red, green) > 70) & (
        np.maximum(red, green) - blue > 18
    )
    # A five-pixel horizontal support requirement rejects isolated ocean text.
    return cv2.erode(
        thematic.astype(np.uint8), np.ones((1, 5), dtype=np.uint8), iterations=1
    ).astype(bool)


def _west_coast_residuals(
    warped_source: np.ndarray, canonical_line: np.ndarray
) -> np.ndarray:
    if warped_source.shape[:2] != canonical_line.shape:
        raise ValueError("Source and canonical line grids differ")
    land = _land_mask(warped_source)
    height, width = canonical_line.shape
    residuals = []
    for y in range(max(1, round(height * 0.015)), round(height * 0.89)):
        canonical_x = np.flatnonzero(canonical_line[y, : round(width * 0.60)])
        source_x = np.flatnonzero(land[y, : round(width * 0.60)])
        if not len(canonical_x) or not len(source_x):
            continue
        # Erosion retracts each supported run by two cells.
        residual = int(source_x.min()) - 2 - int(canonical_x.min())
        if abs(residual) <= round(width * 0.055):
            residuals.append(residual)
    if len(residuals) < round(height * 0.70):
        raise ValueError("Too few reliable Pacific-coast rows for a scale refit")
    return np.asarray(residuals, dtype=np.float64)


def _residual_report(residuals: np.ndarray) -> Dict[str, float]:
    absolute = np.abs(residuals)
    mean = float(np.mean(residuals))
    median_abs = float(np.median(absolute))
    p90_abs = float(np.quantile(absolute, 0.9))
    return {
        "row_count": int(len(residuals)),
        "source_minus_canonical_median_px": float(np.median(residuals)),
        "source_minus_canonical_mean_px": mean,
        "absolute_median_px": median_abs,
        "absolute_p90_px": p90_abs,
        "objective": float(abs(mean) + 0.5 * median_abs + 0.1 * p90_abs),
    }


def fit_east_anchored_horizontal_scale(
    image_path: Path,
    alignment_path: Path,
    reference_root: Path,
    canonical_boundary_manifest_path: Path,
    output_dir: Path,
    *,
    minimum_multiplier: float = 0.985,
    maximum_multiplier: float = 1.005,
    candidate_count: int = 41,
    target_height: int = 1014,
    require_rendered_width_reduction: bool = False,
) -> Dict[str, object]:
    """Search one global x scale from an uncorrected base alignment."""

    if not 0 < minimum_multiplier < maximum_multiplier:
        raise ValueError("Horizontal multiplier bounds are invalid")
    if candidate_count < 3:
        raise ValueError("At least three horizontal scale candidates are required")
    if require_rendered_width_reduction and minimum_multiplier <= 1.0:
        raise ValueError(
            "A required rendered-width reduction needs every "
            "reference-to-source multiplier to be greater than 1"
        )
    image_path = image_path.resolve()
    alignment_path = alignment_path.resolve()
    reference_root = reference_root.resolve()
    canonical_boundary_manifest_path = canonical_boundary_manifest_path.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source_rgb = np.asarray(Image.open(image_path).convert("RGB"))
    alignment = json.loads(alignment_path.read_text())
    if "web_mercator_correction" in alignment:
        raise ValueError(
            "East-anchored refit must start from an alignment without child corrections"
        )
    transform = _alignment_transform(alignment)
    if "parameters" not in transform:
        raise ValueError("East-anchored refit requires the base parameter transform")
    state, _ = load_california(reference_root)
    eastern = _eastern_border_points(
        state, str(transform["projection_crs"]), count=420
    )

    canonical = json.loads(canonical_boundary_manifest_path.read_text())
    mainland_record = canonical["artifacts"]["mainland"]
    mainland_path = canonical_boundary_manifest_path.parent / str(
        mainland_record["path"]
    )
    if _sha256(mainland_path) != str(mainland_record["sha256"]):
        raise ValueError("Canonical mainland line hash does not match its manifest")

    candidates = []
    candidate_images: Dict[float, np.ndarray] = {}
    candidate_parameters: Dict[float, Dict[str, float]] = {}
    candidate_anchors: Dict[float, Dict[str, float]] = {}
    canonical_line = None
    grid = None
    for multiplier in np.linspace(
        minimum_multiplier, maximum_multiplier, candidate_count
    ):
        fitted, anchor_report = _east_anchored_parameters(
            transform["parameters"], source_rgb.shape[:2], eastern, float(multiplier)
        )
        candidate_transform = copy.deepcopy(transform)
        candidate_transform["parameters"] = fitted
        warped, candidate_grid = warp_classified_to_web_mercator(
            source_rgb,
            state,
            candidate_transform,
            source_rgb.shape[:2],
            target_height=target_height,
            clip_to_state=False,
        )
        if grid is None:
            grid = candidate_grid
            line_rgba = Image.open(mainland_path).convert("RGBA").resize(
                (int(grid["width"]), int(grid["height"])), Image.Resampling.LANCZOS
            )
            canonical_line = np.asarray(line_rgba)[..., 3] > 20
        elif candidate_grid != grid:
            raise ValueError("Horizontal scale candidates produced different grids")
        residual_report = _residual_report(
            _west_coast_residuals(warped, canonical_line)
        )
        record = {
            **anchor_report,
            **residual_report,
            "center_x_fraction": fitted["center_x_fraction"],
            "x_scale_ratio": fitted["x_scale_ratio"],
        }
        candidates.append(record)
        key = float(multiplier)
        candidate_images[key] = warped
        candidate_parameters[key] = fitted
        candidate_anchors[key] = anchor_report

    chosen = min(
        candidates,
        key=lambda item: (
            item["objective"],
            abs(item["reference_to_source_x_scale_multiplier"] - 1.0),
        ),
    )
    if require_rendered_width_reduction and not float(
        chosen["rendered_width_multiplier"]
    ) < 1.0:
        raise ValueError("The chosen transform did not reduce rendered width")
    chosen_multiplier = float(chosen["reference_to_source_x_scale_multiplier"])
    chosen_image = candidate_images[chosen_multiplier]

    chosen_best = copy.deepcopy(alignment["best"])
    parent_metrics = copy.deepcopy(chosen_best.get("metrics", {}))
    chosen_best["parameters"] = candidate_parameters[chosen_multiplier]
    chosen_best["objective"] = float(chosen["objective"])
    chosen_best["transform_model"] = "east_anchored_horizontal_scale_refit"
    chosen_best["metrics"] = {
        key: value
        for key, value in chosen.items()
        if key
        not in {
            "center_x_fraction",
            "x_scale_ratio",
            "reference_to_source_x_scale_multiplier",
            "rendered_width_multiplier",
        }
    }
    result = {
        "schema_version": int(alignment.get("schema_version", 1)),
        "status": "needs_visual_review",
        "alignment_mode": "automatic",
        "transform_model": "east_anchored_horizontal_scale_refit",
        "source": alignment["source"],
        "reference": alignment["reference"],
        "best": chosen_best,
        "best_by_coverage": chosen_best,
        "candidates": [],
        "parent_alignment": {
            "path": str(alignment_path),
            "sha256": _sha256(alignment_path),
        },
        "global_refit": {
            "method": "source_first_east_anchored_horizontal_scale_search",
            "canonical_boundary": {
                "path": str(canonical_boundary_manifest_path),
                "sha256": _sha256(canonical_boundary_manifest_path),
                "canonical_boundary_id": canonical["canonical_boundary_id"],
            },
            "grid": grid,
            "source_land_evidence": {
                "method": "warm_or_green_land_separated_from_blue_ocean_with_five_pixel_horizontal_support",
                "maximum_candidate_residual_fraction": 0.055,
            },
            "search": {
                "minimum_multiplier": minimum_multiplier,
                "maximum_multiplier": maximum_multiplier,
                "candidate_count": candidate_count,
            },
            "chosen": chosen,
            "candidate_metrics": candidates,
            "parent_best_metrics": parent_metrics,
            "invariants": {
                "original_source_sha256": _sha256(image_path),
                "prior_pixel_materialization_inherited": False,
                "source_y_parameters_changed": False,
                "prior_web_mercator_corrections_inherited": False,
                "rendered_width_reduction_required": bool(
                    require_rendered_width_reduction
                ),
                "eastern_anchor_mean_x_displacement_source_px": candidate_anchors[
                    chosen_multiplier
                ]["eastern_anchor_mean_x_displacement_source_px"],
            },
        },
        "metrics": chosen_best["metrics"],
        "warning": (
            "This transform is evaluated against the original source image and starts "
            "from an alignment with no Web-Mercator child correction. No prior "
            "pixel materialization or west-coast correction operation is inherited."
        ),
    }
    alignment_output = output_dir / "alignment.json"
    alignment_output.write_text(json.dumps(result, indent=2) + "\n")

    source_path = output_dir / "web-mercator-source.png"
    Image.fromarray(chosen_image).save(source_path, optimize=True)
    no_extraction_path = output_dir / "no-extraction.png"
    Image.new("RGBA", (chosen_image.shape[1], chosen_image.shape[0]), (0, 0, 0, 0)).save(
        no_extraction_path, optimize=True
    )

    overlay = Image.fromarray(chosen_image).convert("RGBA")
    evidence = Image.open(mainland_path).convert("RGBA").resize(
        overlay.size, Image.Resampling.LANCZOS
    )
    alpha = evidence.getchannel("A")
    lime = Image.new("RGBA", overlay.size, (103, 255, 139, 0))
    lime.putalpha(alpha)
    overlay.alpha_composite(lime)
    overlay_path = output_dir / "source-overlay.png"
    overlay.save(overlay_path, optimize=True)

    report = {
        "status": "needs_visual_review",
        "alignment": {
            "path": alignment_output.name,
            "sha256": _sha256(alignment_output),
        },
        "source_overlay": {
            "path": overlay_path.name,
            "sha256": _sha256(overlay_path),
        },
        "warped_source": {
            "path": source_path.name,
            "sha256": _sha256(source_path),
            "classification_performed": False,
        },
        "no_extraction_overlay": {
            "path": no_extraction_path.name,
            "sha256": _sha256(no_extraction_path),
        },
        "chosen": chosen,
        "candidate_count": len(candidates),
    }
    report_path = output_dir / "fit-report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report
