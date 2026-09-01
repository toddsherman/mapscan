"""Register a styled high-resolution California county reference raster."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
from scipy.optimize import minimize

from .alignment import PROJECTIONS, _reference_points, transform_points
from .extraction import _web_reference_assets, warp_classified_to_web_mercator
from .reference import load_california


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _largest_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), 8
    )
    if count <= 1:
        raise ValueError("No connected line component was found")
    component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == component


def _extract_styled_lines(rgba: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Separate the opaque-black state stroke from the gray county strokes."""

    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError("The county reference must preserve an RGBA alpha channel")
    alpha = rgba[:, :, 3]
    luminance = np.mean(rgba[:, :, :3].astype(np.float32), axis=2)

    # The supplied SVG-derived PNG uses an opaque black, thicker state stroke
    # and a #333-ish thinner county stroke.  Core pixels are deliberately used
    # here; antialiasing is restored by dilation only after class separation.
    state_core = (alpha >= 180) & (luminance <= 30)
    state_border = _largest_component(state_core)
    contours, _ = cv2.findContours(
        state_border.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        raise ValueError("The thick state outline could not be closed")
    mainland = max(contours, key=cv2.contourArea)
    state_interior = np.zeros(state_border.shape, dtype=np.uint8)
    cv2.fillPoly(state_interior, [mainland], 1)
    state_interior = state_interior > 0

    county_border = (
        (alpha >= 48)
        & (luminance >= 31)
        & (luminance <= 96)
        & state_interior
    )
    state_exclusion = cv2.dilate(
        state_border.astype(np.uint8), np.ones((7, 7), dtype=np.uint8)
    ) > 0
    county_border &= ~state_exclusion
    return state_border, county_border, state_interior


def _fit_reference(
    state,
    state_border: np.ndarray,
    maximum_dimension: int,
) -> Tuple[Dict[str, object], list[Dict[str, object]], np.ndarray]:
    height, width = state_border.shape
    scale = min(1.0, maximum_dimension / max(height, width))
    working = cv2.resize(
        state_border.astype(np.uint8),
        (round(width * scale), round(height * scale)),
        interpolation=cv2.INTER_NEAREST,
    ) > 0
    distance = distance_transform_edt(~working)
    ys, xs = np.nonzero(working)
    if not len(xs):
        raise ValueError("The state reference stroke is empty")
    source_bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    working_height, working_width = working.shape
    candidates = []

    for projection, crs in PROJECTIONS.items():
        reference = _reference_points(state, crs, count=2400)
        held_out_selector = np.arange(len(reference)) % 5 == 0
        training = reference[~held_out_selector]
        bbox_width = max(source_bbox[2] - source_bbox[0], 1)
        bbox_height = max(source_bbox[3] - source_bbox[1], 1)
        state_height_fraction = bbox_height / working_height
        center_x_fraction = (source_bbox[0] + source_bbox[2]) / 2 / working_width
        center_y_fraction = (source_bbox[1] + source_bbox[3]) / 2 / working_height
        reference_aspect = np.ptp(reference[:, 0]) / max(np.ptp(reference[:, 1]), 1e-9)
        x_scale_ratio = (bbox_width / bbox_height) / max(reference_aspect, 1e-9)
        initial = np.asarray(
            [
                center_x_fraction,
                center_y_fraction,
                state_height_fraction,
                x_scale_ratio,
                0.0,
                0.0,
            ],
            dtype=np.float64,
        )
        bounds = [
            (center_x_fraction - 0.08, center_x_fraction + 0.08),
            (center_y_fraction - 0.08, center_y_fraction + 0.08),
            (state_height_fraction * 0.90, state_height_fraction * 1.10),
            (x_scale_ratio * 0.70, x_scale_ratio * 1.30),
            (-8.0, 8.0),
            (-0.25, 0.25),
        ]

        def objective(parameters: np.ndarray) -> float:
            points = transform_points(training, parameters, working.shape)
            x = np.rint(points[:, 0]).astype(int)
            y = np.rint(points[:, 1]).astype(int)
            inside = (
                (x >= 0)
                & (x < working_width)
                & (y >= 0)
                & (y < working_height)
            )
            values = np.full(len(points), 80.0, dtype=np.float64)
            values[inside] = distance[y[inside], x[inside]]
            return float(
                np.mean(np.minimum(values, 30.0))
                + 0.35 * np.quantile(values, 0.90)
                + 20.0 * (1.0 - np.mean(inside))
            )

        fitted = minimize(
            objective,
            initial,
            method="Powell",
            bounds=bounds,
            options={"maxiter": 300, "xtol": 1e-6, "ftol": 1e-6},
        )
        all_points = transform_points(reference, fitted.x, working.shape)
        x = np.rint(all_points[:, 0]).astype(int)
        y = np.rint(all_points[:, 1]).astype(int)
        inside = (
            (x >= 0)
            & (x < working_width)
            & (y >= 0)
            & (y < working_height)
        )
        residuals = np.full(len(reference), 80.0, dtype=np.float64)
        residuals[inside] = distance[y[inside], x[inside]]
        holdout = residuals[held_out_selector]
        parameters = {
            "center_x_fraction": float(fitted.x[0]),
            "center_y_fraction": float(fitted.x[1]),
            "state_height_fraction": float(fitted.x[2]),
            "x_scale_ratio": float(fitted.x[3]),
            "rotation_degrees": float(fitted.x[4]),
            "x_shear": float(fitted.x[5]),
        }
        candidates.append(
            {
                "projection": projection,
                "projection_crs": crs,
                "evidence_model": "styled_thick_state_border",
                "coverage_model": "full_or_most_state",
                "transform_model": "affine_like",
                "objective": float(fitted.fun),
                "parameters": parameters,
                "metrics": {
                    "working_scale_from_source": scale,
                    "working_width": working_width,
                    "working_height": working_height,
                    "fit_sample_count": int(np.count_nonzero(~held_out_selector)),
                    "held_out_sample_count": int(np.count_nonzero(held_out_selector)),
                    "holdout_median_px_at_working_resolution": float(np.median(holdout)),
                    "holdout_p90_px_at_working_resolution": float(
                        np.quantile(holdout, 0.90)
                    ),
                    "holdout_max_px_at_working_resolution": float(np.max(holdout)),
                    "holdout_median_px_at_source_resolution": float(
                        np.median(holdout) / scale
                    ),
                    "holdout_p90_px_at_source_resolution": float(
                        np.quantile(holdout, 0.90) / scale
                    ),
                    "holdout_within_3px_working_fraction": float(
                        np.mean(holdout <= 3.0)
                    ),
                    "visible_reference_fraction": float(np.mean(inside)),
                },
            }
        )
    candidates.sort(
        key=lambda item: (
            item["metrics"]["holdout_p90_px_at_working_resolution"],
            item["objective"],
        )
    )
    return candidates[0], candidates, working


def _diagnostic(
    rgba: np.ndarray,
    state_border: np.ndarray,
    county_border: np.ndarray,
    state,
    best: Dict[str, object],
) -> np.ndarray:
    output = np.full((*state_border.shape, 3), 255, dtype=np.uint8)
    output[county_border] = (100, 100, 100)
    output[state_border] = (0, 0, 0)
    reference = _reference_points(state, str(best["projection_crs"]), count=2400)
    parameters = [
        best["parameters"][key]
        for key in (
            "center_x_fraction",
            "center_y_fraction",
            "state_height_fraction",
            "x_scale_ratio",
            "rotation_degrees",
            "x_shear",
        )
    ]
    points = np.rint(transform_points(reference, parameters, state_border.shape)).astype(
        np.int32
    )
    cv2.polylines(output, [points.reshape((-1, 1, 2))], True, (0, 190, 220), 2)
    del rgba
    return output


def register_county_reference(
    image_path: Path,
    reference_root: Path,
    output_dir: Path,
    *,
    maximum_dimension: int = 1800,
    web_height: int = 3600,
) -> Dict[str, object]:
    """Register the supplied styled county map and emit canonical line masks."""

    output_dir.mkdir(parents=True, exist_ok=True)
    rgba = np.asarray(Image.open(image_path).convert("RGBA"))
    state_border, county_border, state_interior = _extract_styled_lines(rgba)
    state, counties = load_california(reference_root)
    best, candidates, _ = _fit_reference(state, state_border, maximum_dimension)
    transform = dict(best)

    artifacts = {}
    source_masks = {
        "source_state_border": state_border,
        "source_county_border": county_border,
        "source_state_interior": state_interior,
    }
    for name, mask in source_masks.items():
        path = output_dir / f"{name.replace('_', '-')}-mask.png"
        Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(path, optimize=True)
        artifacts[name] = {"path": path.name, "sha256": _sha256(path)}

    warped_state, grid = warp_classified_to_web_mercator(
        state_border.astype(np.uint8),
        state,
        transform,
        state_border.shape,
        target_height=web_height,
        clip_to_state=False,
    )
    warped_county, county_grid = warp_classified_to_web_mercator(
        county_border.astype(np.uint8),
        state,
        transform,
        state_border.shape,
        target_height=web_height,
        clip_to_state=True,
    )
    if (
        grid["bounds"] != county_grid["bounds"]
        or grid["width"] != county_grid["width"]
        or grid["height"] != county_grid["height"]
    ):
        raise AssertionError("State and county reference grids differ")
    for name, mask in {
        "web_mercator_state_border": warped_state > 0,
        "web_mercator_county_border": warped_county > 0,
    }.items():
        path = output_dir / f"{name.replace('_', '-')}-mask.png"
        Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(path, optimize=True)
        artifacts[name] = {"path": path.name, "sha256": _sha256(path)}

    _, _, authoritative_state, authoritative_county = _web_reference_assets(
        grid["bounds"], warped_county.shape, state, counties
    )
    state_exclusion_radius = max(7, round(web_height * 0.003))
    if state_exclusion_radius % 2 == 0:
        state_exclusion_radius += 1
    interior_authoritative_county = authoritative_county & ~(
        cv2.dilate(
            authoritative_state.astype(np.uint8),
            np.ones((state_exclusion_radius, state_exclusion_radius), dtype=np.uint8),
        )
        > 0
    )
    county_distance = distance_transform_edt(~(warped_county > 0))
    county_residuals = county_distance[interior_authoritative_county]
    county_registration = {
        "authoritative_interior_county_sample_count": int(len(county_residuals)),
        "median_nearest_registered_county_px": float(np.median(county_residuals)),
        "p90_nearest_registered_county_px": float(
            np.quantile(county_residuals, 0.90)
        ),
        "within_3px_fraction": float(np.mean(county_residuals <= 3.0)),
        "within_8px_fraction": float(np.mean(county_residuals <= 8.0)),
        "state_boundary_exclusion_kernel_px": state_exclusion_radius,
        "warning": (
            "This measures agreement between two reference vintages/generalizations. "
            "The raster reference remains supplemental rather than authoritative."
        ),
    }

    diagnostic_path = output_dir / "registration-diagnostic.png"
    Image.fromarray(
        _diagnostic(rgba, state_border, county_border, state, best), mode="RGB"
    ).save(diagnostic_path, optimize=True)
    artifacts["diagnostic"] = {
        "path": diagnostic_path.name,
        "sha256": _sha256(diagnostic_path),
    }
    metrics = best["metrics"]
    passed = bool(
        metrics["visible_reference_fraction"] >= 0.99
        and metrics["holdout_median_px_at_source_resolution"] <= 4.0
        and metrics["holdout_p90_px_at_source_resolution"] <= 12.0
        and county_registration["median_nearest_registered_county_px"] <= 8.0
        and county_registration["p90_nearest_registered_county_px"] <= 24.0
    )
    result = {
        "schema_version": 1,
        "status": "pass" if passed else "needs_attention",
        "reference_kind": "styled_high_resolution_county_raster",
        "alignment_mode": "registered_raster_reference",
        "source": {
            "path": str(image_path),
            "sha256": _sha256(image_path),
            "width": int(rgba.shape[1]),
            "height": int(rgba.shape[0]),
            "mode": "RGBA",
        },
        "reference": {
            "root": str(reference_root),
            "crs": "EPSG:4269",
        },
        "style_interpretation": {
            "state_border": "largest connected opaque-black stroke component",
            "county_borders": "thin gray strokes inside the recovered state outline",
            "state_border_pixel_count": int(np.count_nonzero(state_border)),
            "county_border_pixel_count": int(np.count_nonzero(county_border)),
        },
        "best": best,
        "candidates": candidates,
        "web_grid": grid,
        "county_registration": county_registration,
        "artifacts": artifacts,
        "evidence_policy": (
            "The registered raster is supplemental high-resolution evidence. The Census "
            "state perimeter remains authoritative; county matches may validate or refine "
            "a fit only when they are locally visible and globally consistent."
        ),
    }
    manifest_path = output_dir / "county-reference.json"
    manifest_path.write_text(json.dumps(result, indent=2) + "\n")
    return result
