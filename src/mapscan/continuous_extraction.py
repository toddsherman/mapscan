"""Value-preserving extraction for maps whose legend is a continuous color ramp."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

from .canonical_clip import canonical_publication_interior
from .extraction import (
    _fill_indexed_nodata_in_mask,
    _publication_interior_with_water_exclusion,
    _state_mask_in_source,
    warp_classified_to_web_mercator,
)
from .label_detection import detect_apple_vision_label_regions
from .reference import load_california


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _alignment_transform(alignment: Dict[str, object]) -> Dict[str, object]:
    if alignment.get("alignment_mode") == "assisted":
        transform = {
            "projection": "assisted_reference_crs",
            "projection_crs": alignment["reference"]["crs"],
            "transform_model": alignment["transform_model"],
            "reference_to_source_matrix": alignment["reference_to_source_matrix"],
        }
    else:
        transform = dict(alignment["best"])
    if "web_mercator_correction" in alignment:
        transform["web_mercator_correction"] = alignment["web_mercator_correction"]
    return transform


def _restrict_active_boundary_expansion_to_warp(
    publication_interior: np.ndarray,
    pinned_polygon_interior: np.ndarray,
    warped_values: np.ndarray,
    complete_unsupported_from_nearest: bool = False,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Control completion in land exposed only by the active boundary ring."""

    shape = publication_interior.shape
    if pinned_polygon_interior.shape != shape or warped_values.shape != shape:
        raise ValueError("Active-boundary support masks differ in shape")
    expansion = publication_interior & ~pinned_polygon_interior
    supported = expansion & (warped_values > 0)
    unsupported = expansion & (warped_values == 0)
    final = (
        publication_interior.copy()
        if complete_unsupported_from_nearest
        else (publication_interior & pinned_polygon_interior) | supported
    )
    report = {
        "method": "active_boundary_expansion_requires_warped_source",
        "candidate_expansion_pixel_count": int(np.count_nonzero(expansion)),
        "warped_source_supported_pixel_count": int(np.count_nonzero(supported)),
        "unsupported_pixel_count": int(np.count_nonzero(unsupported)),
        "unsupported_pixel_count_removed": int(
            0
            if complete_unsupported_from_nearest
            else np.count_nonzero(unsupported)
        ),
        "unsupported_pixel_count_retained_for_completion": int(
            np.count_nonzero(unsupported)
            if complete_unsupported_from_nearest
            else 0
        ),
        "target_completion_policy": (
            "unsupported canonical land inherits the nearest encoded value; water remains excluded"
            if complete_unsupported_from_nearest
            else "forbidden outside the pinned polygon interior"
        ),
    }
    return final, report


def _source_analysis_mask(
    authoritative_state_mask: np.ndarray,
    canonical_interior_mode: str,
    evidence_margin_px: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Add a bounded source-space reconstruction margin for the active ring.

    The legacy vector state remains the authoritative source seed.  A small
    margin lets cartographic boundary ink be reconstructed before warping, but
    only active-boundary clipping may publish those extra source pixels.
    """

    if authoritative_state_mask.ndim != 2:
        raise ValueError("Continuous source state mask must be two-dimensional")
    if evidence_margin_px < 0:
        raise ValueError("source_evidence_margin_px cannot be negative")
    if evidence_margin_px and canonical_interior_mode != "active_boundary_ring":
        raise ValueError(
            "source_evidence_margin_px requires canonical_interior_mode "
            "active_boundary_ring"
        )
    if evidence_margin_px == 0:
        analysis = authoritative_state_mask.astype(bool, copy=True)
    else:
        size = evidence_margin_px * 2 + 1
        analysis = cv2.dilate(
            authoritative_state_mask.astype(np.uint8),
            np.ones((size, size), dtype=np.uint8),
            iterations=1,
        ).astype(bool)
    margin = analysis & ~authoritative_state_mask
    return analysis, {
        "method": "legacy_source_state_plus_bounded_cartographic_reconstruction_margin",
        "margin_px": evidence_margin_px,
        "authoritative_state_pixel_count": int(
            np.count_nonzero(authoritative_state_mask)
        ),
        "analysis_pixel_count": int(np.count_nonzero(analysis)),
        "margin_pixel_count": int(np.count_nonzero(margin)),
        "publication_authority": (
            "active_boundary_ring_after_web_mercator_warp"
            if canonical_interior_mode == "active_boundary_ring"
            else "pinned_polygon_plus_declared_coastal_seam_after_web_mercator_warp"
        ),
    }


def _rgb_to_lab(colors: np.ndarray) -> np.ndarray:
    shaped = np.asarray(colors, dtype=np.float32).reshape((-1, 1, 3)) / 255.0
    return cv2.cvtColor(shaped, cv2.COLOR_RGB2LAB).reshape((-1, 3))


def _blue_nearest_legend_fill_mask(
    rgb: np.ndarray,
    state_mask: np.ndarray,
    config: Dict[str, object] | None,
) -> Tuple[np.ndarray, Dict[str, object] | None]:
    """Select blue cartographic pixels for spatial nearest-ramp recovery.

    The printed elevation map uses blue for hydrography and other contextual
    ink, but the data publication decision is made later by the pinned water
    mask.  Inside the source state, blue therefore borrows the value of the
    nearest directly observed legend-ramp pixel instead of being projected
    onto the ramp or mistaken for the dark depression swatch.
    """

    selected = np.zeros(state_mask.shape, dtype=bool)
    if config is None:
        return selected, None
    if not isinstance(config, dict):
        raise ValueError("blue_nearest_legend_fill must be an object")
    minimum_blue_over_red = int(config.get("minimum_blue_over_red", 20))
    minimum_blue_over_green = int(config.get("minimum_blue_over_green", 5))
    if minimum_blue_over_red < 0 or minimum_blue_over_green < 0:
        raise ValueError("Blue nearest-legend thresholds cannot be negative")
    channels = rgb.astype(np.int16)
    selected = state_mask & (
        channels[..., 2] - channels[..., 0] >= minimum_blue_over_red
    ) & (
        channels[..., 2] - channels[..., 1] >= minimum_blue_over_green
    )
    return selected, {
        "method": "blue_dominant_pixels_borrow_nearest_direct_legend_ramp_value",
        "minimum_blue_over_red": minimum_blue_over_red,
        "minimum_blue_over_green": minimum_blue_over_green,
        "selected_source_pixel_count": int(np.count_nonzero(selected)),
        "water_policy": (
            "source blue is reconstructed first; pinned target hydrography "
            "remains transparent afterward"
        ),
    }


def _classify_continuous(
    rgb: np.ndarray,
    state_mask: np.ndarray,
    ramp_stops: list[Dict[str, object]],
    special_values: list[Dict[str, object]],
    maximum_residual: float,
    luminance_weight: float,
    neutral_ink_maximum: int | None = None,
    neutral_ink_chroma_maximum: int = 45,
    exclusion_colors: list[Dict[str, object]] | None = None,
    completion_inpaint_radius_px: float = 0.0,
    dark_ink_blackhat_threshold: int | None = None,
    dark_ink_blackhat_kernel_px: int = 9,
    label_occlusion_mask: np.ndarray | None = None,
    spatial_nearest_legend_fill_mask: np.ndarray | None = None,
    chunk_size: int = 300_000,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Project source colors onto a piecewise-linear legend path in Lab space."""

    stop_values = np.asarray([float(item["value"]) for item in ramp_stops])
    stop_lab = _rgb_to_lab(np.asarray([item["legend_rgb"] for item in ramp_stops]))
    order = np.argsort(stop_values)
    stop_values = stop_values[order]
    stop_lab = stop_lab[order]
    weights = np.asarray([luminance_weight, 1.0, 1.0], dtype=np.float32)
    weighted_stops = stop_lab * weights
    starts = weighted_stops[:-1]
    deltas = weighted_stops[1:] - weighted_stops[:-1]
    denominators = np.maximum(np.sum(deltas * deltas, axis=1), 1e-9)
    special_lab = (
        _rgb_to_lab(np.asarray([item["legend_rgb"] for item in special_values]))
        * weights
        if special_values
        else np.empty((0, 3), dtype=np.float32)
    )
    special_numeric = np.asarray(
        [float(item["value"]) for item in special_values], dtype=np.float32
    )

    excluded = np.zeros(state_mask.shape, dtype=bool)
    if neutral_ink_maximum is not None:
        chroma = rgb.max(axis=2).astype(np.int16) - rgb.min(axis=2).astype(np.int16)
        excluded |= (rgb.max(axis=2) <= neutral_ink_maximum) & (
            chroma <= neutral_ink_chroma_maximum
        )
    for item in exclusion_colors or []:
        color = np.asarray(item["rgb"], dtype=np.float32)
        radius = float(item.get("maximum_rgb_distance", 35.0))
        excluded |= np.linalg.norm(rgb.astype(np.float32) - color, axis=2) <= radius
    if dark_ink_blackhat_threshold is not None:
        kernel_size = max(3, int(dark_ink_blackhat_kernel_px))
        if kernel_size % 2 == 0:
            kernel_size += 1
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        blackhat = cv2.morphologyEx(
            gray,
            cv2.MORPH_BLACKHAT,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
            ),
        )
        dark_ink = blackhat >= dark_ink_blackhat_threshold
        dark_ink = cv2.dilate(
            dark_ink.astype(np.uint8), np.ones((3, 3), np.uint8)
        ).astype(bool)
        excluded |= dark_ink
    if label_occlusion_mask is not None:
        if label_occlusion_mask.shape != state_mask.shape:
            raise ValueError("Continuous label occlusion mask differs from the source")
        excluded |= label_occlusion_mask.astype(bool)
    if spatial_nearest_legend_fill_mask is not None:
        if spatial_nearest_legend_fill_mask.shape != state_mask.shape:
            raise ValueError("Spatial nearest-legend mask differs from the source")
        excluded |= spatial_nearest_legend_fill_mask.astype(bool)
    eligible_mask = state_mask & ~excluded
    flat_indices = np.flatnonzero(eligible_mask.ravel())
    output_values = np.full(rgb.shape[:2], np.nan, dtype=np.float32)
    output_residual = np.full(rgb.shape[:2], np.inf, dtype=np.float32)
    rgb_flat = rgb.reshape((-1, 3))
    values_flat = output_values.ravel()
    residual_flat = output_residual.ravel()
    for offset in range(0, len(flat_indices), chunk_size):
        indices = flat_indices[offset : offset + chunk_size]
        points = _rgb_to_lab(rgb_flat[indices]) * weights
        best_residual = np.full(len(points), np.inf, dtype=np.float32)
        best_value = np.zeros(len(points), dtype=np.float32)
        for segment_index, (start, delta, denominator) in enumerate(
            zip(starts, deltas, denominators)
        ):
            fraction = np.clip(
                np.sum((points - start) * delta, axis=1) / denominator, 0.0, 1.0
            )
            projected = start + fraction[:, None] * delta
            residual = np.linalg.norm(points - projected, axis=1)
            better = residual < best_residual
            best_residual[better] = residual[better]
            best_value[better] = (
                stop_values[segment_index]
                + fraction[better]
                * (stop_values[segment_index + 1] - stop_values[segment_index])
            )
        for special_index, special in enumerate(special_lab):
            residual = np.linalg.norm(points - special, axis=1)
            better = residual < best_residual
            best_residual[better] = residual[better]
            best_value[better] = special_numeric[special_index]
        accepted = best_residual <= maximum_residual
        values_flat[indices[accepted]] = best_value[accepted]
        residual_flat[indices] = best_residual
    accepted_mask = state_mask & np.isfinite(output_values)
    missing = state_mask & ~accepted_mask
    completed = output_values.copy()
    if np.any(accepted_mask) and np.any(missing):
        _, nearest = distance_transform_edt(~accepted_mask, return_indices=True)
        nearest_values = output_values[nearest[0], nearest[1]]
        spatial_fill = (
            missing & spatial_nearest_legend_fill_mask.astype(bool)
            if spatial_nearest_legend_fill_mask is not None
            else np.zeros(state_mask.shape, dtype=bool)
        )
        completed[spatial_fill] = nearest_values[spatial_fill]
        remaining_missing = missing & ~spatial_fill
        if completion_inpaint_radius_px > 0:
            seed = nearest_values.astype(np.float32)
            seed[accepted_mask] = output_values[accepted_mask]
            seed[spatial_fill] = completed[spatial_fill]
            inpainted = cv2.inpaint(
                seed,
                remaining_missing.astype(np.uint8),
                completion_inpaint_radius_px,
                cv2.INPAINT_TELEA,
            )
            completed[remaining_missing] = inpainted[remaining_missing]
        else:
            completed[remaining_missing] = nearest_values[remaining_missing]
    spatial_fill_count = int(
        np.count_nonzero(state_mask & spatial_nearest_legend_fill_mask)
        if spatial_nearest_legend_fill_mask is not None
        else 0
    )
    report = {
        "state_pixel_count": int(np.count_nonzero(state_mask)),
        "directly_observed_pixel_count": int(np.count_nonzero(accepted_mask)),
        "completed_occlusion_pixel_count": int(np.count_nonzero(missing)),
        "explicit_cartographic_ink_pixel_count": int(
            np.count_nonzero(state_mask & excluded)
        ),
        "ocr_label_occlusion_pixel_count": int(
            np.count_nonzero(state_mask & label_occlusion_mask)
            if label_occlusion_mask is not None
            else 0
        ),
        "spatial_nearest_legend_fill_pixel_count": spatial_fill_count,
        "direct_observation_fraction": float(
            np.count_nonzero(accepted_mask) / max(np.count_nonzero(state_mask), 1)
        ),
        "accepted_residual_median": float(np.median(output_residual[accepted_mask])),
        "accepted_residual_p90": float(np.quantile(output_residual[accepted_mask], 0.9)),
        "maximum_weighted_lab_residual": maximum_residual,
        "luminance_weight": luminance_weight,
        "completion_method": (
            f"Blue cartography uses exact nearest observed legend-ramp values; other occlusions use Telea inpaint radius {completion_inpaint_radius_px}px seeded from nearest observed values"
            if completion_inpaint_radius_px > 0
            else "nearest directly observed legend-ramp value within authoritative state mask"
        ),
    }
    return completed, missing, report


def _encode_values(
    values: np.ndarray, state_mask: np.ndarray, encoding: Dict[str, object]
) -> np.ndarray:
    offset = float(encoding["offset"])
    scale = float(encoding["scale"])
    encoded = np.zeros(values.shape, dtype=np.uint16)
    encoded[state_mask] = np.clip(
        np.rint((values[state_mask] - offset) / scale) + 1,
        1,
        np.iinfo(np.uint16).max,
    ).astype(np.uint16)
    return encoded


def _decode_values(encoded: np.ndarray, encoding: Dict[str, object]) -> np.ndarray:
    """Decode nonzero uint16 cells while leaving NoData as NaN."""

    decoded = np.full(encoded.shape, np.nan, dtype=np.float32)
    selected = encoded > 0
    decoded[selected] = float(encoding["offset"]) + (
        encoded[selected].astype(np.float32) - 1.0
    ) * float(encoding["scale"])
    return decoded


def _replace_small_special_components_from_nearest(
    encoded: np.ndarray,
    publication_interior: np.ndarray,
    special_values: list[Dict[str, object]],
    encoding: Dict[str, object],
    config: Dict[str, object] | None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object] | None]:
    """Replace small isolated special-color marks with nearby ramp evidence."""

    changed = np.zeros(encoded.shape, dtype=bool)
    if config is None or not special_values:
        return encoded.copy(), changed, None
    if not isinstance(config, dict):
        raise ValueError("small_special_value_nearest_fill must be an object")
    maximum_area = int(config.get("maximum_component_pixel_count", 49))
    if maximum_area < 1:
        raise ValueError("Small special-value component limit must be positive")
    require_surrounded = bool(config.get("require_surrounded_by_data", True))
    offset = float(encoding["offset"])
    scale = float(encoding["scale"])
    special_codes = []
    components_by_value: Dict[str, object] = {}
    kernel = np.ones((3, 3), dtype=np.uint8)
    for item in special_values:
        code = int(round((float(item["value"]) - offset) / scale) + 1)
        special_codes.append(code)
        selected = publication_interior & (encoded == code)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            selected.astype(np.uint8), 8
        )
        replaced_components = 0
        replaced_pixels = 0
        for component_id in range(1, count):
            area = int(stats[component_id, cv2.CC_STAT_AREA])
            if area > maximum_area:
                continue
            component = labels == component_id
            if require_surrounded:
                ring = cv2.dilate(component.astype(np.uint8), kernel).astype(bool)
                ring &= ~component
                if np.any(ring & ~publication_interior) or np.any(
                    ring & (encoded == 0)
                ):
                    continue
            changed |= component
            replaced_components += 1
            replaced_pixels += area
        components_by_value[str(item.get("id", item["value"]))] = {
            "encoded_value": code,
            "source_component_count": int(count - 1),
            "replaced_component_count": replaced_components,
            "replaced_pixel_count": replaced_pixels,
        }
    output = encoded.copy()
    if np.any(changed):
        seed = publication_interior & (encoded > 0) & ~np.isin(
            encoded, np.asarray(special_codes, dtype=encoded.dtype)
        )
        if not np.any(seed):
            raise ValueError("No ordinary legend values can replace special-value dots")
        _, nearest = distance_transform_edt(~seed, return_indices=True)
        output[changed] = encoded[nearest[0][changed], nearest[1][changed]]
    return output, changed, {
        "method": "small_surrounded_special_components_borrow_nearest_ordinary_legend_value",
        "maximum_component_pixel_count_inclusive": maximum_area,
        "require_surrounded_by_data": require_surrounded,
        "replaced_component_count": int(
            sum(item["replaced_component_count"] for item in components_by_value.values())
        ),
        "replaced_pixel_count": int(np.count_nonzero(changed)),
        "components_by_value": components_by_value,
        "preservation_policy": "larger special-value regions remain authoritative",
    }


def _value_preview(
    values: np.ndarray,
    state_mask: np.ndarray,
    ramp_stops: list[Dict[str, object]],
    special_values: list[Dict[str, object]],
) -> np.ndarray:
    stop_values = np.asarray([float(item["value"]) for item in ramp_stops])
    stop_rgb = np.asarray([item["display_rgb"] for item in ramp_stops], dtype=np.float32)
    order = np.argsort(stop_values)
    stop_values, stop_rgb = stop_values[order], stop_rgb[order]
    rgba = np.zeros((*values.shape, 4), dtype=np.uint8)
    selected_values = values[state_mask]
    for channel in range(3):
        rgba[:, :, channel][state_mask] = np.clip(
            np.interp(selected_values, stop_values, stop_rgb[:, channel]), 0, 255
        ).astype(np.uint8)
    for item in special_values:
        selected = state_mask & np.isclose(values, float(item["value"]), atol=0.51)
        rgba[selected, :3] = np.asarray(item["display_rgb"], dtype=np.uint8)
    rgba[state_mask, 3] = 235
    return rgba


def _evidence_overlay(mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    selected = mask.astype(bool)
    rgba[selected, :3] = np.asarray(color, dtype=np.uint8)
    rgba[selected, 3] = 180
    return rgba


def _extract_arrays(plan: Dict[str, object]) -> Dict[str, object]:
    source_path = Path(plan["source"])
    alignment_path = Path(plan["alignment"])
    reference_root = Path(plan.get("reference", "reference/census-2025"))
    rgb = np.asarray(Image.open(source_path).convert("RGB"))
    alignment = json.loads(alignment_path.read_text())
    transform = _alignment_transform(alignment)
    state, _ = load_california(reference_root)
    authoritative_state_mask = _state_mask_in_source(
        state, str(transform["projection_crs"]), transform, rgb.shape[:2]
    )
    canonical_interior_mode = str(
        plan.get("canonical_interior_mode", "pinned_polygon")
    )
    state_mask, source_analysis = _source_analysis_mask(
        authoritative_state_mask,
        canonical_interior_mode,
        int(plan.get("source_evidence_margin_px", 0)),
    )
    label_occlusion_mask = np.zeros(state_mask.shape, dtype=bool)
    label_detection = None
    label_config = plan.get("label_occlusion_detection")
    if label_config is not None:
        if not isinstance(label_config, dict):
            raise ValueError("label_occlusion_detection must be an object")
        if str(label_config.get("engine")) != "apple_vision":
            raise ValueError("Continuous extraction has an unsupported label engine")
        label_occlusion_mask, label_detection = detect_apple_vision_label_regions(
            source_path,
            state_mask,
            minimum_confidence=float(label_config.get("minimum_confidence", 0.5)),
            minimum_letter_count=int(label_config.get("minimum_letter_count", 3)),
            padding_px=int(label_config.get("padding_px", 3)),
            maximum_word_dimension_px=int(
                label_config.get("maximum_word_dimension_px", 280)
            ),
            minimum_valid_overlap_pixels=int(
                label_config.get("minimum_valid_overlap_pixels", 8)
            ),
            blackhat_kernel_px=int(label_config.get("blackhat_kernel_px", 15)),
            blackhat_threshold=int(label_config.get("blackhat_threshold", 8)),
            glyph_dilation_px=int(label_config.get("glyph_dilation_px", 2)),
            glyph_component_minimum_area_px=int(
                label_config.get("glyph_component_minimum_area_px", 2)
            ),
            glyph_component_maximum_area_px=int(
                label_config.get("glyph_component_maximum_area_px", 240)
            ),
            glyph_component_maximum_dimension_px=int(
                label_config.get("glyph_component_maximum_dimension_px", 32)
            ),
            glyph_component_maximum_aspect=float(
                label_config.get("glyph_component_maximum_aspect", 8.0)
            ),
            glyph_component_minimum_extent=float(
                label_config.get("glyph_component_minimum_extent", 0.05)
            ),
        )
    blue_nearest_fill_mask, blue_nearest_fill = _blue_nearest_legend_fill_mask(
        rgb,
        state_mask,
        plan.get("blue_nearest_legend_fill"),
    )
    values, completion_mask, classification = _classify_continuous(
        rgb,
        state_mask,
        plan["ramp_stops"],
        plan.get("special_values", []),
        maximum_residual=float(plan["maximum_weighted_lab_residual"]),
        luminance_weight=float(plan.get("luminance_weight", 0.4)),
        neutral_ink_maximum=(
            int(plan["neutral_ink_maximum"])
            if plan.get("neutral_ink_maximum") is not None
            else None
        ),
        neutral_ink_chroma_maximum=int(
            plan.get("neutral_ink_chroma_maximum", 45)
        ),
        exclusion_colors=plan.get("cartographic_ink_colors", []),
        completion_inpaint_radius_px=float(
            plan.get("completion_inpaint_radius_px", 0.0)
        ),
        dark_ink_blackhat_threshold=(
            int(plan["dark_ink_blackhat_threshold"])
            if plan.get("dark_ink_blackhat_threshold") is not None
            else None
        ),
        dark_ink_blackhat_kernel_px=int(
            plan.get("dark_ink_blackhat_kernel_px", 9)
        ),
        label_occlusion_mask=label_occlusion_mask,
        spatial_nearest_legend_fill_mask=blue_nearest_fill_mask,
    )
    classification["blue_nearest_legend_fill"] = blue_nearest_fill
    source_margin = state_mask & ~authoritative_state_mask
    classification["source_evidence_margin"] = {
        **source_analysis,
        "directly_observed_pixel_count": int(
            np.count_nonzero(source_margin & ~completion_mask)
        ),
        "reconstructed_pixel_count": int(
            np.count_nonzero(source_margin & completion_mask)
        ),
    }
    encoded = _encode_values(values, state_mask, plan["encoding"])
    target_height = plan.get("target_height")
    clip_to_legacy_target_state = canonical_interior_mode != "active_boundary_ring"
    warped_raw, grid = warp_classified_to_web_mercator(
        encoded,
        state,
        transform,
        rgb.shape[:2],
        target_height=int(target_height) if target_height else None,
        clip_to_state=clip_to_legacy_target_state,
    )
    warped_source_completion, completion_grid = warp_classified_to_web_mercator(
        completion_mask.astype(np.uint8),
        state,
        transform,
        rgb.shape[:2],
        target_height=int(target_height) if target_height else None,
        clip_to_state=clip_to_legacy_target_state,
    )
    if completion_grid != grid:
        raise ValueError("Continuous value and evidence warps use different grids")
    warped_label_occlusion, label_grid = warp_classified_to_web_mercator(
        label_occlusion_mask.astype(np.uint8),
        state,
        transform,
        rgb.shape[:2],
        target_height=int(target_height) if target_height else None,
        clip_to_state=clip_to_legacy_target_state,
    )
    if label_grid != grid:
        raise ValueError("Continuous value and OCR evidence warps use different grids")
    warped_blue_nearest_fill, blue_fill_grid = warp_classified_to_web_mercator(
        blue_nearest_fill_mask.astype(np.uint8),
        state,
        transform,
        rgb.shape[:2],
        target_height=int(target_height) if target_height else None,
        clip_to_state=clip_to_legacy_target_state,
    )
    if blue_fill_grid != grid:
        raise ValueError("Continuous value and blue-fill evidence warps use different grids")
    warped_source, source_grid = warp_classified_to_web_mercator(
        rgb,
        state,
        transform,
        rgb.shape[:2],
        target_height=int(target_height) if target_height else None,
        clip_to_state=False,
    )
    if source_grid != {**grid, "clip_to_state": False}:
        raise ValueError("Continuous source review warp differs from the value grid")

    publication_interior, canonical_clip, water_removed = (
        _publication_interior_with_water_exclusion(
            grid,
            plan.get("water_exclusion"),
            plan.get("coastal_seam"),
            canonical_interior_mode,
        )
    )
    if canonical_interior_mode == "active_boundary_ring":
        pinned_polygon_interior, _ = canonical_publication_interior(grid)
        publication_interior, support_report = (
            _restrict_active_boundary_expansion_to_warp(
                publication_interior,
                pinned_polygon_interior,
                warped_raw,
                complete_unsupported_from_nearest=bool(
                    plan.get("complete_unsupported_canonical_land", False)
                ),
            )
        )
        canonical_clip = {
            **canonical_clip,
            "valid_pixel_count": int(np.count_nonzero(publication_interior)),
            "active_boundary_warp_support": support_report,
        }
    boundary_removed = (warped_raw > 0) & ~publication_interior
    warped = warped_raw.copy()
    warped[~publication_interior] = 0
    warped_source_completion = (warped_source_completion > 0) & publication_interior
    warped_label_occlusion = (warped_label_occlusion > 0) & publication_interior
    warped_blue_nearest_fill = (
        (warped_blue_nearest_fill > 0) & publication_interior
    )

    target_completion_mask = np.zeros(warped.shape, dtype=bool)
    if bool(plan.get("complete_target_state", True)):
        target_completion_mask = publication_interior & (warped == 0)
        warped, completed_count = _fill_indexed_nodata_in_mask(
            warped, publication_interior
        )
        target_completion_mask &= warped > 0
        if completed_count != int(np.count_nonzero(target_completion_mask)):
            raise ValueError("Continuous target completion count is inconsistent")

    warped, small_special_fill_mask, small_special_fill = (
        _replace_small_special_components_from_nearest(
            warped,
            publication_interior,
            plan.get("special_values", []),
            plan["encoding"],
            plan.get("small_special_value_nearest_fill"),
        )
    )

    warped_source_completion &= warped > 0
    warped_label_occlusion &= warped_source_completion
    warped_blue_nearest_fill &= warped_source_completion
    target_direct = (
        publication_interior
        & (warped > 0)
        & ~warped_source_completion
        & ~target_completion_mask
        & ~small_special_fill_mask
    )
    unknown = publication_interior & (warped == 0)
    if bool(plan.get("complete_target_state", True)) and np.any(unknown):
        raise ValueError("Continuous extraction retains NoData inside the publication interior")
    if np.any(warped[~publication_interior] > 0):
        raise ValueError("Continuous extraction retains values outside the canonical boundary")

    target = {
        "publication_interior_pixel_count": int(np.count_nonzero(publication_interior)),
        "directly_observed_pixel_count": int(np.count_nonzero(target_direct)),
        "source_completion_pixel_count": int(
            np.count_nonzero(warped_source_completion)
        ),
        "ocr_label_completion_pixel_count": int(
            np.count_nonzero(warped_label_occlusion)
        ),
        "blue_nearest_legend_fill_pixel_count": int(
            np.count_nonzero(warped_blue_nearest_fill)
        ),
        "target_completion_pixel_count": int(
            np.count_nonzero(target_completion_mask)
        ),
        "small_special_value_nearest_fill_pixel_count": int(
            np.count_nonzero(small_special_fill_mask)
        ),
        "final_value_pixel_count": int(np.count_nonzero(warped)),
        "unknown_inside_pixel_count": int(np.count_nonzero(unknown)),
        "colored_outside_pixel_count": int(
            np.count_nonzero(warped[~publication_interior])
        ),
        "boundary_removed_pixel_count": int(np.count_nonzero(boundary_removed)),
        "internal_water_removed_pixel_count": int(
            np.count_nonzero(water_removed) if water_removed is not None else 0
        ),
        "completion_policy": (
            "nearest encoded value inside the canonical publication interior; "
            "source-derived values remain unchanged"
        ),
        "small_special_value_nearest_fill": small_special_fill,
    }
    return {
        "rgb": rgb,
        "state_mask": state_mask,
        "authoritative_state_mask": authoritative_state_mask,
        "values": values,
        "encoded": encoded,
        "warped": warped,
        "warped_source": warped_source,
        "source_completion_mask": completion_mask,
        "source_label_occlusion_mask": label_occlusion_mask,
        "source_blue_nearest_legend_fill_mask": blue_nearest_fill_mask,
        "web_source_completion_mask": warped_source_completion,
        "web_label_occlusion_mask": warped_label_occlusion,
        "web_blue_nearest_legend_fill_mask": warped_blue_nearest_fill,
        "target_completion_mask": target_completion_mask,
        "small_special_value_nearest_fill_mask": small_special_fill_mask,
        "publication_interior": publication_interior,
        "boundary_removed": boundary_removed,
        "water_removed": water_removed,
        "grid": grid,
        "classification": classification,
        "label_detection": label_detection,
        "canonical_clip": canonical_clip,
        "target": target,
    }


def extract_continuous_plan(plan_path: Path, output_dir: Path) -> Dict[str, object]:
    plan = json.loads(plan_path.read_text())
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = _extract_arrays(plan)
    rgb = arrays["rgb"]
    state_mask = arrays["state_mask"]
    values = arrays["values"]
    encoded = arrays["encoded"]
    warped = arrays["warped"]
    completion_mask = arrays["source_completion_mask"]
    source_value_path = output_dir / "source-value-encoded.png"
    web_value_path = output_dir / "web-mercator-value-encoded.png"
    completion_path = output_dir / "source-completion-mask.png"
    source_label_path = output_dir / "source-label-occlusion-mask.png"
    source_blue_fill_path = output_dir / "source-blue-nearest-legend-fill-mask.png"
    web_source_completion_path = output_dir / "web-mercator-source-completion-mask.png"
    web_label_path = output_dir / "web-mercator-label-occlusion-mask.png"
    web_blue_fill_path = output_dir / "web-mercator-blue-nearest-legend-fill-mask.png"
    target_completion_path = output_dir / "web-mercator-target-completion-mask.png"
    small_special_fill_path = output_dir / "web-mercator-small-special-nearest-fill-mask.png"
    publication_interior_path = output_dir / "web-mercator-publication-interior-mask.png"
    boundary_removed_path = output_dir / "web-mercator-boundary-removed-mask.png"
    water_removed_path = output_dir / "web-mercator-internal-water-mask.png"
    source_preview_path = output_dir / "source-preview.png"
    web_preview_path = output_dir / "web-mercator-preview.png"
    web_source_path = output_dir / "web-mercator-source.jpg"
    transparent_overlay_path = output_dir / "web-mercator-transparent-overlay.png"
    label_overlay_path = output_dir / "web-mercator-label-occlusion-overlay.png"
    blue_fill_overlay_path = output_dir / "web-mercator-blue-nearest-legend-fill-overlay.png"
    source_completion_overlay_path = output_dir / "web-mercator-source-completion-overlay.png"
    target_completion_overlay_path = output_dir / "web-mercator-target-completion-overlay.png"
    small_special_fill_overlay_path = output_dir / "web-mercator-small-special-nearest-fill-overlay.png"
    water_overlay_path = output_dir / "web-mercator-internal-water-overlay.png"
    Image.fromarray(encoded, mode="I;16").save(source_value_path, optimize=True)
    Image.fromarray(warped, mode="I;16").save(web_value_path, optimize=True)
    Image.fromarray(completion_mask.astype(np.uint8) * 255, mode="L").save(
        completion_path, optimize=True
    )
    Image.fromarray(
        arrays["source_label_occlusion_mask"].astype(np.uint8) * 255, mode="L"
    ).save(source_label_path, optimize=True)
    Image.fromarray(
        arrays["source_blue_nearest_legend_fill_mask"].astype(np.uint8) * 255,
        mode="L",
    ).save(source_blue_fill_path, optimize=True)
    Image.fromarray(
        arrays["web_source_completion_mask"].astype(np.uint8) * 255, mode="L"
    ).save(web_source_completion_path, optimize=True)
    Image.fromarray(
        arrays["web_label_occlusion_mask"].astype(np.uint8) * 255, mode="L"
    ).save(web_label_path, optimize=True)
    Image.fromarray(
        arrays["web_blue_nearest_legend_fill_mask"].astype(np.uint8) * 255,
        mode="L",
    ).save(web_blue_fill_path, optimize=True)
    Image.fromarray(
        arrays["target_completion_mask"].astype(np.uint8) * 255, mode="L"
    ).save(target_completion_path, optimize=True)
    Image.fromarray(
        arrays["small_special_value_nearest_fill_mask"].astype(np.uint8) * 255,
        mode="L",
    ).save(small_special_fill_path, optimize=True)
    Image.fromarray(
        arrays["publication_interior"].astype(np.uint8) * 255, mode="L"
    ).save(publication_interior_path, optimize=True)
    Image.fromarray(
        arrays["boundary_removed"].astype(np.uint8) * 255, mode="L"
    ).save(boundary_removed_path, optimize=True)
    if arrays["water_removed"] is not None:
        Image.fromarray(
            arrays["water_removed"].astype(np.uint8) * 255, mode="L"
        ).save(water_removed_path, optimize=True)
    Image.fromarray(
        _value_preview(
            values,
            state_mask,
            plan["ramp_stops"],
            plan.get("special_values", []),
        ),
        mode="RGBA",
    ).save(source_preview_path, optimize=True)
    Image.fromarray(
        _value_preview(
            _decode_values(warped, plan["encoding"]),
            warped > 0,
            plan["ramp_stops"],
            plan.get("special_values", []),
        ),
        mode="RGBA",
    ).save(web_preview_path, optimize=True)
    Image.fromarray(arrays["warped_source"]).save(
        web_source_path, quality=94, subsampling=0
    )
    Image.fromarray(
        np.zeros((*warped.shape, 4), dtype=np.uint8)
    ).save(transparent_overlay_path, optimize=True)
    Image.fromarray(
        _evidence_overlay(arrays["web_label_occlusion_mask"], (255, 0, 220))
    ).save(label_overlay_path, optimize=True)
    Image.fromarray(
        _evidence_overlay(arrays["web_blue_nearest_legend_fill_mask"], (80, 160, 255))
    ).save(blue_fill_overlay_path, optimize=True)
    Image.fromarray(
        _evidence_overlay(arrays["web_source_completion_mask"], (0, 220, 255))
    ).save(source_completion_overlay_path, optimize=True)
    Image.fromarray(
        _evidence_overlay(arrays["target_completion_mask"], (255, 220, 0))
    ).save(target_completion_overlay_path, optimize=True)
    Image.fromarray(
        _evidence_overlay(
            arrays["small_special_value_nearest_fill_mask"], (255, 120, 40)
        )
    ).save(small_special_fill_overlay_path, optimize=True)
    Image.fromarray(
        _evidence_overlay(
            arrays["water_removed"]
            if arrays["water_removed"] is not None
            else np.zeros(warped.shape, dtype=bool),
            (45, 125, 255),
        )
    ).save(water_overlay_path, optimize=True)
    artifacts = {
        "source_value": {"path": source_value_path.name, "sha256": _sha256(source_value_path)},
        "web_mercator_value": {"path": web_value_path.name, "sha256": _sha256(web_value_path)},
        "source_completion_mask": {"path": completion_path.name, "sha256": _sha256(completion_path)},
        "source_label_occlusion_mask": {
            "path": source_label_path.name,
            "sha256": _sha256(source_label_path),
        },
        "source_blue_nearest_legend_fill_mask": {
            "path": source_blue_fill_path.name,
            "sha256": _sha256(source_blue_fill_path),
        },
        "web_mercator_source_completion_mask": {
            "path": web_source_completion_path.name,
            "sha256": _sha256(web_source_completion_path),
        },
        "web_mercator_label_occlusion_mask": {
            "path": web_label_path.name,
            "sha256": _sha256(web_label_path),
        },
        "web_mercator_blue_nearest_legend_fill_mask": {
            "path": web_blue_fill_path.name,
            "sha256": _sha256(web_blue_fill_path),
        },
        "web_mercator_target_completion_mask": {
            "path": target_completion_path.name,
            "sha256": _sha256(target_completion_path),
        },
        "web_mercator_small_special_nearest_fill_mask": {
            "path": small_special_fill_path.name,
            "sha256": _sha256(small_special_fill_path),
        },
        "web_mercator_publication_interior_mask": {
            "path": publication_interior_path.name,
            "sha256": _sha256(publication_interior_path),
        },
        "web_mercator_boundary_removed_mask": {
            "path": boundary_removed_path.name,
            "sha256": _sha256(boundary_removed_path),
        },
        "source_preview": {"path": source_preview_path.name, "sha256": _sha256(source_preview_path)},
        "web_mercator_preview": {"path": web_preview_path.name, "sha256": _sha256(web_preview_path)},
        "web_mercator_source": {
            "path": web_source_path.name,
            "sha256": _sha256(web_source_path),
        },
        "web_mercator_transparent_overlay": {
            "path": transparent_overlay_path.name,
            "sha256": _sha256(transparent_overlay_path),
        },
        "web_mercator_label_occlusion_overlay": {
            "path": label_overlay_path.name,
            "sha256": _sha256(label_overlay_path),
        },
        "web_mercator_blue_nearest_legend_fill_overlay": {
            "path": blue_fill_overlay_path.name,
            "sha256": _sha256(blue_fill_overlay_path),
        },
        "web_mercator_source_completion_overlay": {
            "path": source_completion_overlay_path.name,
            "sha256": _sha256(source_completion_overlay_path),
        },
        "web_mercator_target_completion_overlay": {
            "path": target_completion_overlay_path.name,
            "sha256": _sha256(target_completion_overlay_path),
        },
        "web_mercator_small_special_nearest_fill_overlay": {
            "path": small_special_fill_overlay_path.name,
            "sha256": _sha256(small_special_fill_overlay_path),
        },
        "web_mercator_internal_water_overlay": {
            "path": water_overlay_path.name,
            "sha256": _sha256(water_overlay_path),
        },
    }
    if arrays["water_removed"] is not None:
        artifacts["web_mercator_internal_water_mask"] = {
            "path": water_removed_path.name,
            "sha256": _sha256(water_removed_path),
        }
    result = {
        "schema_version": 2,
        "status": "needs_visual_review",
        "extraction_kind": "continuous_color_ramp",
        "dataset_id": plan["dataset_id"],
        "title": plan["title"],
        "units": plan.get("units"),
        "source": {
            "path": plan["source"],
            "sha256": _sha256(Path(plan["source"])),
            "width": int(rgb.shape[1]),
            "height": int(rgb.shape[0]),
        },
        "plan": {"path": str(plan_path), "sha256": _sha256(plan_path)},
        "alignment": plan["alignment"],
        "encoding": plan["encoding"],
        "ramp_stops": plan["ramp_stops"],
        "special_values": plan.get("special_values", []),
        "classification": arrays["classification"],
        "label_detection": arrays["label_detection"],
        "target": arrays["target"],
        "warp": arrays["grid"],
        "canonical_clip": arrays["canonical_clip"],
        "artifacts": artifacts,
        "review": {
            "default_view": "source",
            "decision_path": "review-decision.json",
            "corrections_path": "alignment-corrections.json",
            "county_diagnostic_enabled": False,
            "assets": {
                "source": web_source_path.name,
                "county_overlay": transparent_overlay_path.name,
                "county_residual": transparent_overlay_path.name,
            },
            "layers": [
                {
                    "id": "continuous-elevation-ramp",
                    "label": "Extracted elevation ramp",
                    "path": web_preview_path.name,
                },
                {
                    "id": "ocr-label-occlusions",
                    "label": "OCR label reconstruction",
                    "path": label_overlay_path.name,
                },
                {
                    "id": "blue-nearest-legend-fill",
                    "label": "Blue pixels replaced from nearest legend data",
                    "path": blue_fill_overlay_path.name,
                },
                {
                    "id": "all-source-completion",
                    "label": "All source reconstruction",
                    "path": source_completion_overlay_path.name,
                },
                {
                    "id": "target-edge-completion",
                    "label": "Target edge completion",
                    "path": target_completion_overlay_path.name,
                },
                {
                    "id": "small-special-nearest-fill",
                    "label": "Small dark dots replaced from nearest data",
                    "path": small_special_fill_overlay_path.name,
                },
                {
                    "id": "internal-water-exclusion",
                    "label": "Transparent water exclusion",
                    "path": water_overlay_path.name,
                },
            ],
        },
        "warnings": plan.get("warnings", []),
    }
    (output_dir / "continuous-extraction.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


def audit_continuous_source_diff(run_dir: Path, output_dir: Path) -> Dict[str, object]:
    """Recompute a continuous extraction and require bit-exact stored values."""

    manifest_path = run_dir / "continuous-extraction.json"
    manifest = json.loads(manifest_path.read_text())
    plan_path = Path(manifest["plan"]["path"])
    plan = json.loads(plan_path.read_text())
    arrays = _extract_arrays(plan)
    expected_source = arrays["encoded"]
    expected_web = arrays["warped"]
    stored_source_path = run_dir / manifest["artifacts"]["source_value"]["path"]
    stored_web_path = run_dir / manifest["artifacts"]["web_mercator_value"]["path"]
    stored_source = np.asarray(Image.open(stored_source_path), dtype=np.uint16)
    stored_web = np.asarray(Image.open(stored_web_path), dtype=np.uint16)
    source_diff = expected_source != stored_source
    web_diff = expected_web != stored_web
    evidence_arrays = {
        "source_label_occlusion_mask": arrays["source_label_occlusion_mask"],
        "source_blue_nearest_legend_fill_mask": arrays[
            "source_blue_nearest_legend_fill_mask"
        ],
        "web_mercator_source_completion_mask": arrays["web_source_completion_mask"],
        "web_mercator_label_occlusion_mask": arrays["web_label_occlusion_mask"],
        "web_mercator_blue_nearest_legend_fill_mask": arrays[
            "web_blue_nearest_legend_fill_mask"
        ],
        "web_mercator_target_completion_mask": arrays["target_completion_mask"],
        "web_mercator_small_special_nearest_fill_mask": arrays[
            "small_special_value_nearest_fill_mask"
        ],
        "web_mercator_publication_interior_mask": arrays["publication_interior"],
        "web_mercator_boundary_removed_mask": arrays["boundary_removed"],
    }
    if arrays["water_removed"] is not None:
        evidence_arrays["web_mercator_internal_water_mask"] = arrays["water_removed"]
    evidence_different_pixel_counts = {}
    for artifact_name, expected in evidence_arrays.items():
        record = manifest["artifacts"].get(artifact_name)
        if not isinstance(record, dict):
            raise ValueError(f"Continuous extraction is missing {artifact_name}")
        stored_path = run_dir / str(record["path"])
        if _sha256(stored_path) != str(record["sha256"]):
            raise ValueError(f"Continuous extraction artifact is stale: {stored_path}")
        stored = np.asarray(Image.open(stored_path)) > 0
        evidence_different_pixel_counts[artifact_name] = int(
            np.count_nonzero(expected.astype(bool) != stored)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    source_diff_path = output_dir / "source-value-diff-mask.png"
    web_diff_path = output_dir / "web-mercator-value-diff-mask.png"
    Image.fromarray(source_diff.astype(np.uint8) * 255, mode="L").save(
        source_diff_path, optimize=True
    )
    Image.fromarray(web_diff.astype(np.uint8) * 255, mode="L").save(
        web_diff_path, optimize=True
    )
    passed = (
        not np.any(source_diff)
        and not np.any(web_diff)
        and not any(evidence_different_pixel_counts.values())
    )
    result = {
        "schema_version": 1,
        "audit_kind": "continuous_source_diff",
        "status": "pass" if passed else "fail",
        "dataset_id": manifest["dataset_id"],
        "run": str(run_dir),
        "source_different_pixel_count": int(np.count_nonzero(source_diff)),
        "web_different_pixel_count": int(np.count_nonzero(web_diff)),
        "classification": arrays["classification"],
        "target": arrays["target"],
        "evidence_different_pixel_counts": evidence_different_pixel_counts,
        "artifacts": {
            "source_diff": {"path": source_diff_path.name, "sha256": _sha256(source_diff_path)},
            "web_diff": {"path": web_diff_path.name, "sha256": _sha256(web_diff_path)},
        },
    }
    (output_dir / "source-diff-audit.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result
