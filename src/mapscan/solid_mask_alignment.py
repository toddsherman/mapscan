"""Refine a map alignment from the outer edge of solid thematic classes.

This is deliberately narrower than the generic edge aligner.  It is intended
for maps whose mutually-exclusive classes use exact, solid RGB fills.  Their
union provides geographic edge evidence without letting labels, roads, relief,
or browser chrome participate in the fit.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Iterable, Sequence

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, map_coordinates
from scipy.optimize import minimize

from .auto_refinement import _alignment_transform
from .extraction import warp_classified_to_web_mercator
from .reference import load_california
from .refinement import _normalized_matrix


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exact_palette_mask(rgb: np.ndarray, colors: Sequence[Sequence[int]]) -> np.ndarray:
    palette = np.asarray(colors, dtype=np.uint8)
    if palette.ndim != 2 or palette.shape[1] != 3 or len(palette) < 2:
        raise ValueError("At least two RGB triplets are required")
    packed = (
        rgb[..., 0].astype(np.uint32) << 16
        | rgb[..., 1].astype(np.uint32) << 8
        | rgb[..., 2].astype(np.uint32)
    )
    packed_palette = (
        palette[:, 0].astype(np.uint32) << 16
        | palette[:, 1].astype(np.uint32) << 8
        | palette[:, 2].astype(np.uint32)
    )
    return np.isin(packed, packed_palette)


def _retain_components(
    mask: np.ndarray, *, minimum_area: int, maximum_components: int
) -> tuple[np.ndarray, list[Dict[str, object]]]:
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    components = []
    for label in range(1, count):
        x, y, width, height, area = stats[label]
        if int(area) < minimum_area:
            continue
        components.append(
            {
                "label": label,
                "area": int(area),
                "bbox": [int(x), int(y), int(width), int(height)],
                "centroid": [float(value) for value in centroids[label]],
            }
        )
    components.sort(key=lambda item: int(item["area"]), reverse=True)
    components = components[:maximum_components]
    retained = np.isin(labels, [int(item["label"]) for item in components])
    return retained, components


def _external_boundary(mask: np.ndarray, maximum_components: int) -> np.ndarray:
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:maximum_components]
    boundary = np.zeros(mask.shape, dtype=np.uint8)
    if contours:
        cv2.drawContours(boundary, contours, -1, 1, 1, cv2.LINE_8)
    return boundary.astype(bool)


def _overlay_line_mask(path: Path) -> np.ndarray:
    image = np.asarray(Image.open(path).convert("RGBA"))
    alpha = image[..., 3]
    if np.any(alpha):
        return alpha > 0
    return np.any(image[..., :3] > 0, axis=2)


def _balanced_points(mask: np.ndarray, maximum_points: int = 24000) -> np.ndarray:
    y, x = np.nonzero(mask)
    points = np.column_stack((x, y)).astype(np.float64)
    if len(points) <= maximum_points:
        return points
    # A regular stride preserves geographic distribution and is deterministic.
    indices = np.linspace(0, len(points) - 1, maximum_points).astype(np.int64)
    return points[indices]


def _similarity_matrix(
    parameters: Sequence[float], width: int, height: int
) -> np.ndarray:
    log_scale, angle, translate_x, translate_y = [float(value) for value in parameters]
    scale = math.exp(log_scale)
    cosine, sine = math.cos(angle), math.sin(angle)
    linear = scale * np.asarray([[cosine, -sine], [sine, cosine]])
    center = np.asarray([(width - 1) / 2.0, (height - 1) / 2.0])
    offset = center + np.asarray([translate_x, translate_y]) - linear @ center
    return np.asarray(
        [
            [linear[0, 0], linear[0, 1], offset[0]],
            [linear[1, 0], linear[1, 1], offset[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _apply(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack((points, np.ones(len(points)))) @ matrix.T
    return homogeneous[:, :2] / homogeneous[:, 2, None]


def _sample_distance(distance: np.ndarray, points: np.ndarray) -> np.ndarray:
    height, width = distance.shape
    inside = (
        (points[:, 0] >= 0)
        & (points[:, 0] <= width - 1)
        & (points[:, 1] >= 0)
        & (points[:, 1] <= height - 1)
    )
    values = np.full(len(points), 50.0, dtype=np.float64)
    if np.any(inside):
        values[inside] = map_coordinates(
            distance,
            [points[inside, 1], points[inside, 0]],
            order=1,
            mode="nearest",
        )
    return values


def _robust_cost(values: np.ndarray, *, delta: float = 3.0, cap: float = 24.0) -> float:
    clipped = np.minimum(values, cap)
    loss = np.where(
        clipped <= delta,
        0.5 * np.square(clipped),
        delta * (clipped - 0.5 * delta),
    )
    return float(np.mean(loss))


def _distance_summary(values: np.ndarray) -> Dict[str, float]:
    return {
        "median_px": float(np.median(values)),
        "p75_px": float(np.quantile(values, 0.75)),
        "p90_px": float(np.quantile(values, 0.9)),
        "mean_px": float(np.mean(values)),
        "within_3px_fraction": float(np.mean(values <= 3.0)),
        "within_8px_fraction": float(np.mean(values <= 8.0)),
    }


def _directed_metrics(
    matrix: np.ndarray,
    source_points: np.ndarray,
    target_points: np.ndarray,
    source_distance: np.ndarray,
    target_distance: np.ndarray,
) -> Dict[str, object]:
    source_to_target = _sample_distance(target_distance, _apply(matrix, source_points))
    inverse = np.linalg.inv(matrix)
    target_to_source = _sample_distance(source_distance, _apply(inverse, target_points))
    return {
        "source_to_target": _distance_summary(source_to_target),
        "target_to_source": _distance_summary(target_to_source),
        "symmetric_robust_cost": float(
            0.5 * _robust_cost(source_to_target) + 0.5 * _robust_cost(target_to_source)
        ),
    }


def _render_diagnostic(
    target: np.ndarray,
    current: np.ndarray,
    corrected: np.ndarray,
) -> np.ndarray:
    output = np.full((*target.shape, 3), 18, dtype=np.uint8)
    output[target] = (255, 50, 210)
    output[current] = (0, 225, 240)
    output[corrected] = (90, 255, 80)
    overlap = target & corrected
    output[overlap] = (255, 255, 255)
    return output


def _warp_boundary(boundary: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    height, width = boundary.shape
    return cv2.warpPerspective(
        boundary.astype(np.uint8),
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)


def refine_solid_mask_alignment(
    image_path: Path,
    alignment_path: Path,
    reference_root: Path,
    canonical_boundary_manifest_path: Path,
    output_dir: Path,
    colors: Sequence[Sequence[int]],
    *,
    minimum_component_area: int = 500,
    maximum_components: int = 5,
    correspondence_gate_px: float = 30.0,
) -> Dict[str, object]:
    """Fit and persist a conservative similarity correction from class edges."""

    image_path = image_path.resolve()
    alignment_path = alignment_path.resolve()
    reference_root = reference_root.resolve()
    canonical_boundary_manifest_path = canonical_boundary_manifest_path.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rgb = np.asarray(Image.open(image_path).convert("RGB"))
    alignment = json.loads(alignment_path.read_text())
    boundary_manifest = json.loads(canonical_boundary_manifest_path.read_text())
    boundary_artifact = canonical_boundary_manifest_path.parent / str(
        boundary_manifest["artifacts"]["overlay"]["path"]
    )
    if _sha256(boundary_artifact) != boundary_manifest["artifacts"]["overlay"]["sha256"]:
        raise ValueError("Canonical boundary artifact hash mismatch")

    grid = boundary_manifest["source_grid"]
    target = _overlay_line_mask(boundary_artifact)
    if target.shape != (int(grid["height"]), int(grid["width"])):
        raise ValueError("Canonical boundary grid and overlay dimensions do not match")

    exact = _exact_palette_mask(rgb, colors)
    retained, components = _retain_components(
        exact,
        minimum_area=minimum_component_area,
        maximum_components=maximum_components,
    )
    state, _ = load_california(reference_root)
    transform = _alignment_transform(alignment, state)
    warped, warp_grid = warp_classified_to_web_mercator(
        retained.astype(np.uint8),
        state,
        transform,
        rgb.shape[:2],
        target_height=int(grid["height"]),
        clip_to_state=False,
    )
    if warped.shape != target.shape or not np.allclose(warp_grid["bounds"], grid["bounds"]):
        raise ValueError("Parent alignment and canonical boundary do not share a grid")
    current_boundary = _external_boundary(warped > 0, maximum_components)

    target_distance = distance_transform_edt(~target)
    source_distance = distance_transform_edt(~current_boundary)
    all_source_points = _balanced_points(current_boundary)
    all_target_points = _balanced_points(target)
    source_gate = _sample_distance(target_distance, all_source_points) <= correspondence_gate_px
    target_gate = _sample_distance(source_distance, all_target_points) <= correspondence_gate_px
    source_points = all_source_points[source_gate]
    target_points = all_target_points[target_gate]
    if len(source_points) < 1000 or len(target_points) < 1000:
        raise ValueError("Solid class boundary has insufficient canonical correspondence")

    height, width = target.shape

    def objective(parameters: np.ndarray) -> float:
        matrix = _similarity_matrix(parameters, width, height)
        moved_source = _apply(matrix, source_points)
        inverse = np.linalg.inv(matrix)
        moved_target = _apply(inverse, target_points)
        return float(
            0.5 * _robust_cost(_sample_distance(target_distance, moved_source))
            + 0.5 * _robust_cost(_sample_distance(source_distance, moved_target))
        )

    bounds = [
        (math.log(0.985), math.log(1.015)),
        (math.radians(-0.5), math.radians(0.5)),
        (-36.0, 36.0),
        (-36.0, 36.0),
    ]
    optimization = minimize(
        objective,
        np.zeros(4, dtype=np.float64),
        method="Powell",
        bounds=bounds,
        options={"xtol": 1e-6, "ftol": 1e-7, "maxiter": 350},
    )
    matrix = _similarity_matrix(optimization.x, width, height)
    identity = np.eye(3, dtype=np.float64)
    before = _directed_metrics(
        identity, source_points, target_points, source_distance, target_distance
    )
    after = _directed_metrics(
        matrix, source_points, target_points, source_distance, target_distance
    )
    if after["symmetric_robust_cost"] >= before["symmetric_robust_cost"]:
        raise ValueError("Solid-mask correction did not improve held geographic evidence")

    normalized_current_to_target = _normalized_matrix(matrix, width, height)
    incremental_target_to_parent = np.linalg.inv(normalized_current_to_target)
    incremental_target_to_parent /= incremental_target_to_parent[2, 2]
    parent_correction = alignment.get("web_mercator_correction")
    if isinstance(parent_correction, dict):
        parent_grid = parent_correction["grid"]
        if (
            int(parent_grid["width"]) != width
            or int(parent_grid["height"]) != height
            or not np.allclose(parent_grid["bounds"], grid["bounds"])
        ):
            raise ValueError("Parent and canonical correction grids do not match")
        parent_target_to_base = np.asarray(
            parent_correction["target_to_current_normalized_matrix"], dtype=np.float64
        )
        target_to_base = parent_target_to_base @ incremental_target_to_parent
        composition_depth = int(parent_correction.get("composition_depth", 1)) + 1
    else:
        target_to_base = incremental_target_to_parent
        composition_depth = 1
    target_to_base /= target_to_base[2, 2]

    corrected_boundary = _warp_boundary(current_boundary, matrix)
    Image.fromarray(exact.astype(np.uint8) * 255, mode="L").save(
        output_dir / "source-exact-class-mask.png", optimize=True
    )
    Image.fromarray(current_boundary.astype(np.uint8) * 255, mode="L").save(
        output_dir / "current-web-mercator-boundary.png", optimize=True
    )
    Image.fromarray(corrected_boundary.astype(np.uint8) * 255, mode="L").save(
        output_dir / "corrected-web-mercator-boundary.png", optimize=True
    )
    Image.fromarray(
        _render_diagnostic(target, current_boundary, corrected_boundary), mode="RGB"
    ).save(output_dir / "alignment-diagnostic.png", optimize=True)

    parent_model = alignment.get("transform_model")
    if not isinstance(parent_model, str):
        parent_model = str(alignment.get("best", {}).get("transform_model", "unknown"))
    refined = json.loads(json.dumps(alignment))
    refined["schema_version"] = max(2, int(refined.get("schema_version", 1)))
    refined["transform_model"] = f"{parent_model}+web_mercator_solid_mask_similarity"
    refined["parent_alignment"] = {
        "path": str(alignment_path),
        "sha256": _sha256(alignment_path),
    }
    refined["web_mercator_correction"] = {
        "model": "similarity",
        "direction": "desired_target_to_pre_correction_sampling_grid",
        "grid": grid,
        "target_to_current_normalized_matrix": target_to_base.tolist(),
        "incremental_target_to_parent_normalized_matrix": incremental_target_to_parent.tolist(),
        "current_to_target_pixel_matrix": matrix.tolist(),
        "composition_depth": composition_depth,
        "generated_by": "mapscan.solid_mask_alignment",
        "canonical_boundary": {
            "manifest_path": str(canonical_boundary_manifest_path),
            "manifest_sha256": _sha256(canonical_boundary_manifest_path),
            "artifact_path": str(boundary_artifact),
            "artifact_sha256": _sha256(boundary_artifact),
        },
    }
    refined["metrics"] = {
        **alignment.get("metrics", {}),
        "solid_mask_alignment_before": before,
        "solid_mask_alignment_after": after,
    }
    refined["warning"] = (
        "Automated solid-class boundary refinement; inspect the generated diagnostic "
        "before promoting extracted data."
    )
    alignment_output = output_dir / "alignment.json"
    alignment_output.write_text(json.dumps(refined, indent=2) + "\n")

    report = {
        "schema_version": 1,
        "status": "pass",
        "source": {"path": str(image_path), "sha256": _sha256(image_path)},
        "parent_alignment": {
            "path": str(alignment_path),
            "sha256": _sha256(alignment_path),
        },
        "palette": [[int(value) for value in color] for color in colors],
        "exact_class_pixel_count": int(np.count_nonzero(exact)),
        "retained_component_pixel_count": int(np.count_nonzero(retained)),
        "retained_components": components,
        "correspondence_gate_px": correspondence_gate_px,
        "source_correspondence_count": int(len(source_points)),
        "target_correspondence_count": int(len(target_points)),
        "optimization": {
            "method": "Powell",
            "success": bool(optimization.success),
            "message": str(optimization.message),
            "function_evaluations": int(optimization.nfev),
            "log_scale": float(optimization.x[0]),
            "scale": float(math.exp(optimization.x[0])),
            "rotation_degrees": float(math.degrees(optimization.x[1])),
            "translation_x_px": float(optimization.x[2]),
            "translation_y_px": float(optimization.x[3]),
        },
        "before": before,
        "after": after,
        "artifacts": {
            name: {"path": name, "sha256": _sha256(output_dir / name)}
            for name in (
                "source-exact-class-mask.png",
                "current-web-mercator-boundary.png",
                "corrected-web-mercator-boundary.png",
                "alignment-diagnostic.png",
                "alignment.json",
            )
        },
    }
    (output_dir / "solid-mask-alignment-report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report
