"""Conservative, auditable completion prototype for plant hardiness.

This experiment deliberately does not seek full coverage.  It preserves every
approved nonzero class, then accepts only one-pass additions supported by both
local topology and the independently generated nearest-class completion.  All
other in-state NoData remains explicit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, distance_transform_edt, label

from mapscan.extraction import (
    _preview,
    _target_state_mask,
    warp_classified_to_web_mercator,
)
from mapscan.reference import load_california


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text())


def _alignment_transform(alignment: Dict[str, object]) -> Dict[str, object]:
    if alignment.get("alignment_mode") == "assisted":
        transform: Dict[str, object] = {
            "projection": "assisted_reference_crs",
            "projection_crs": alignment["reference"]["crs"],
            "transform_model": alignment["transform_model"],
            "reference_to_source_matrix": alignment["reference_to_source_matrix"],
        }
    else:
        transform = dict(alignment["best"])
    if "web_mercator_correction" in alignment:
        transform["web_mercator_correction"] = alignment[
            "web_mercator_correction"
        ]
    return transform


def _save_mask(path: Path, mask: np.ndarray) -> None:
    Image.fromarray(mask.astype(np.uint8) * 255).save(path, optimize=True)


def _save_values(path: Path, values: np.ndarray) -> None:
    Image.fromarray(values.astype(np.uint8)).save(path, optimize=True)


def _class_counts(values: np.ndarray, mask: np.ndarray, categories) -> Dict[str, int]:
    return {
        str(category["id"]): int(np.count_nonzero(mask & (values == index)))
        for index, category in enumerate(categories, 1)
    }


def _source_color_extension(
    rgb: np.ndarray,
    source_class: np.ndarray,
    source_state: np.ndarray,
    categories,
) -> tuple[np.ndarray, Dict[str, object]]:
    """Recover only slight compression misses with strong spatial agreement."""

    prototypes = np.asarray(
        [category["legend_rgb"] for category in categories], dtype=np.uint8
    )
    prototype_lab = cv2.cvtColor(
        prototypes[None, :, :], cv2.COLOR_RGB2LAB
    )[0].astype(np.float32)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    distances = np.linalg.norm(
        lab[:, :, None, :] - prototype_lab[None, None, :, :], axis=3
    )
    order = np.argsort(distances, axis=2)
    best = np.take_along_axis(distances, order[:, :, :1], axis=2)[:, :, 0]
    second = np.take_along_axis(distances, order[:, :, 1:2], axis=2)[:, :, 0]
    margin = second - best
    nearest_class = order[:, :, 0].astype(np.uint8) + 1

    # The original accepted radius was 18 Lab units.  This stage considers only
    # a narrow 18-22 compression fringe, still requiring the original 2-unit
    # runner-up margin plus strong, distributed observed neighbors of one class.
    candidates = (
        source_state
        & (source_class == 0)
        & (best > 18.0)
        & (best <= 22.0)
        & (margin >= 2.0)
    )
    output = np.zeros_like(source_class, dtype=np.uint8)
    height, width = source_class.shape
    for y, x in zip(*np.nonzero(candidates)):
        class_id = int(nearest_class[y, x])
        y1, y2 = max(0, y - 3), min(height, y + 4)
        x1, x2 = max(0, x - 3), min(width, x + 4)
        patch = source_class[y1:y2, x1:x2]
        support = patch[patch > 0]
        if len(support) < 4:
            continue
        if float(np.mean(support == class_id)) < 0.90:
            continue
        quadrants = (
            source_class[y1:y, x1:x],
            source_class[y1:y, x + 1 : min(width, x + 4)],
            source_class[y + 1 : min(height, y + 4), x1:x],
            source_class[y + 1 : min(height, y + 4), x + 1 : min(width, x + 4)],
        )
        if sum(bool(np.any(item == class_id)) for item in quadrants) < 3:
            continue
        output[y, x] = class_id
    report = {
        "candidate_pixel_count": int(np.count_nonzero(candidates)),
        "accepted_source_pixel_count": int(np.count_nonzero(output)),
        "lab_distance_range": [18.0, 22.0],
        "minimum_lab_margin": 2.0,
        "neighbor_radius_px": 3,
        "minimum_same_class_support_fraction": 0.90,
        "minimum_supported_quadrants": 3,
        "accepted_by_class": _class_counts(output, output > 0, categories),
    }
    return output, report


def _small_same_class_enclosures(
    base: np.ndarray,
    target_state: np.ndarray,
    aggressive: np.ndarray,
    *,
    maximum_area_exclusive: int = 50,
    maximum_radius_px: float = 5.0,
) -> tuple[np.ndarray, Dict[str, object]]:
    """Assign small components only when their complete rim has one class."""

    missing = target_state & (base == 0)
    components, count = label(missing, structure=np.ones((3, 3), dtype=np.uint8))
    distance = distance_transform_edt(base == 0)
    values = np.zeros_like(base, dtype=np.uint8)
    accepted_components = 0
    rejected_disagreement = 0
    for component_id in range(1, count + 1):
        component = components == component_id
        area = int(np.count_nonzero(component))
        if area >= maximum_area_exclusive:
            continue
        if float(np.max(distance[component])) > maximum_radius_px:
            continue
        rim = binary_dilation(component, structure=np.ones((3, 3), dtype=bool)) & ~component
        classes = np.unique(base[rim & (base > 0)])
        if len(classes) != 1:
            continue
        class_id = int(classes[0])
        agreed = component & (aggressive == class_id)
        rejected_disagreement += int(np.count_nonzero(component & ~agreed))
        if np.array_equal(agreed, component):
            values[component] = class_id
            accepted_components += 1
    return values, {
        "missing_component_count": int(count),
        "accepted_component_count": accepted_components,
        "accepted_pixel_count": int(np.count_nonzero(values)),
        "maximum_area_exclusive": maximum_area_exclusive,
        "maximum_radius_px": maximum_radius_px,
        "topology_disagreement_pixel_count": rejected_disagreement,
    }


def _locally_surrounded_pixels(
    base: np.ndarray,
    target_state: np.ndarray,
    aggressive: np.ndarray,
    *,
    radius_px: int = 3,
) -> tuple[np.ndarray, Dict[str, object]]:
    """One-shot fill for pixels surrounded in all quadrants by one class."""

    distance = distance_transform_edt(base == 0)
    candidates = target_state & (base == 0) & (distance <= float(radius_px))
    values = np.zeros_like(base, dtype=np.uint8)
    height, width = base.shape
    for y, x in zip(*np.nonzero(candidates)):
        y1, y2 = max(0, y - radius_px), min(height, y + radius_px + 1)
        x1, x2 = max(0, x - radius_px), min(width, x + radius_px + 1)
        patch = base[y1:y2, x1:x2]
        classes = np.unique(patch[patch > 0])
        if len(classes) != 1:
            continue
        class_id = int(classes[0])
        quadrants = (
            base[y1:y, x1:x],
            base[y1:y, x + 1 : min(width, x + radius_px + 1)],
            base[y + 1 : min(height, y + radius_px + 1), x1:x],
            base[
                y + 1 : min(height, y + radius_px + 1),
                x + 1 : min(width, x + radius_px + 1),
            ],
        )
        if not all(bool(np.any(item == class_id)) for item in quadrants):
            continue
        if int(aggressive[y, x]) != class_id:
            continue
        values[y, x] = class_id
    return values, {
        "candidate_pixel_count": int(np.count_nonzero(candidates)),
        "accepted_pixel_count": int(np.count_nonzero(values)),
        "radius_px": radius_px,
        "required_supported_quadrants": 4,
    }


def _montage(
    source_rgb: np.ndarray,
    approved: np.ndarray,
    aggressive: np.ndarray,
    conservative: np.ndarray,
    accepted: np.ndarray,
    unresolved: np.ndarray,
    categories,
) -> Image.Image:
    previews = {
        "Warped source": Image.fromarray(source_rgb.astype(np.uint8)),
        "Approved 118 stamps": Image.fromarray(
            _preview(approved, categories)
        ).convert("RGB"),
        "Aggressive fixed point": Image.fromarray(
            _preview(aggressive, categories)
        ).convert("RGB"),
        "Conservative candidate": Image.fromarray(
            _preview(conservative, categories)
        ).convert("RGB"),
    }
    base_preview = np.asarray(
        Image.fromarray(_preview(conservative, categories)).convert("RGB")
    ).copy()
    accepted_overlay = base_preview.copy()
    accepted_overlay[accepted] = [0, 255, 255]
    unresolved_overlay = base_preview.copy()
    unresolved_overlay[unresolved] = [255, 0, 255]
    previews["Accepted additions (cyan)"] = Image.fromarray(accepted_overlay)
    previews["Unresolved (magenta)"] = Image.fromarray(unresolved_overlay)

    width, height = approved.shape[1], approved.shape[0]
    header = 24
    canvas = Image.new("RGB", (width * 3, (height + header) * 2), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    for index, (title, image) in enumerate(previews.items()):
        column = index % 3
        row = index // 3
        x = column * width
        y = row * (height + header)
        draw.text((x + 6, y + 5), title, fill=(245, 245, 245))
        canvas.paste(image, (x, y + header))
    return canvas


def run(
    run_dir: Path,
    approved_path: Path,
    aggressive_path: Path,
    output_dir: Path,
) -> Dict[str, object]:
    run_dir = run_dir.resolve()
    approved_path = approved_path.resolve()
    aggressive_path = aggressive_path.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    extraction_path = run_dir / "extraction.json"
    manifest = _load(extraction_path)
    plan_path = Path(str(manifest["plan"]["path"])).resolve()
    plan = _load(plan_path)
    layer = plan["layers"][0]
    categories = layer["categories"]
    layer_id = str(layer["id"])
    source_path = Path(str(plan["source"])).resolve()
    alignment_path = Path(str(manifest["alignment"]["path"])).resolve()
    alignment = _load(alignment_path)
    reference_root = Path(str(plan.get("reference", "reference/census-2025"))).resolve()
    state, _ = load_california(reference_root)

    rgb = np.asarray(Image.open(source_path).convert("RGB"))
    source_class_path = run_dir / layer_id / "source-class-id.png"
    source_state_path = run_dir / "source-state-mask.png"
    source_class = np.asarray(Image.open(source_class_path), dtype=np.uint8)
    source_state = np.asarray(Image.open(source_state_path)) > 0
    approved = np.asarray(Image.open(approved_path), dtype=np.uint8)
    aggressive = np.asarray(Image.open(aggressive_path), dtype=np.uint8)
    target_state = _target_state_mask(
        state, manifest["layers"][0]["warp"]["bounds"], approved.shape
    )
    if approved.shape != aggressive.shape or approved.shape != target_state.shape:
        raise ValueError("Approved, aggressive, and target-state rasters must match")

    transform = _alignment_transform(alignment)
    source_rgb_web, source_grid = warp_classified_to_web_mercator(
        rgb, state, transform, rgb.shape[:2]
    )
    source_extension, color_report = _source_color_extension(
        rgb, source_class, source_state, categories
    )
    color_web, color_grid = warp_classified_to_web_mercator(
        source_extension, state, transform, rgb.shape[:2]
    )
    if source_rgb_web.shape[:2] != approved.shape or color_web.shape != approved.shape:
        raise ValueError("Recomputed source evidence does not match approved grid")

    approved_clipped = approved.copy()
    boundary_clipped = (approved_clipped > 0) & ~target_state
    approved_clipped[~target_state] = 0
    aggressive_clipped = aggressive.copy()
    aggressive_clipped[~target_state] = 0

    color_values = np.where(
        target_state
        & (approved_clipped == 0)
        & (color_web > 0)
        & (color_web == aggressive_clipped),
        color_web,
        0,
    ).astype(np.uint8)
    color_report.update(
        {
            "accepted_web_pixel_count": int(np.count_nonzero(color_values)),
            "rejected_topology_disagreement_web_pixel_count": int(
                np.count_nonzero(
                    target_state
                    & (approved_clipped == 0)
                    & (color_web > 0)
                    & (color_web != aggressive_clipped)
                )
            ),
        }
    )
    enclosed_values, enclosed_report = _small_same_class_enclosures(
        approved_clipped, target_state, aggressive_clipped
    )
    local_values, local_report = _locally_surrounded_pixels(
        approved_clipped, target_state, aggressive_clipped
    )

    stage_values = (color_values, enclosed_values, local_values)
    conflict = np.zeros_like(target_state)
    nonzero_stack = np.stack([values > 0 for values in stage_values])
    for first in range(len(stage_values)):
        for second in range(first + 1, len(stage_values)):
            overlap = nonzero_stack[first] & nonzero_stack[second]
            conflict |= overlap & (stage_values[first] != stage_values[second])

    accepted_values = np.zeros_like(approved_clipped)
    for values in stage_values:
        selected = (accepted_values == 0) & (values > 0) & ~conflict
        accepted_values[selected] = values[selected]
    accepted_mask = accepted_values > 0
    candidate = approved_clipped.copy()
    candidate[(candidate == 0) & accepted_mask] = accepted_values[
        (candidate == 0) & accepted_mask
    ]
    unresolved = target_state & (candidate == 0)
    aggressive_only = target_state & (aggressive_clipped > 0) & (candidate == 0)

    original_missing = target_state & (approved_clipped == 0)
    base_distance = distance_transform_edt(approved_clipped == 0)
    accepted_distances = base_distance[accepted_mask]
    outputs = {
        "conservative_class_id": output_dir / "web-mercator-class-id-conservative.png",
        "color_values": output_dir / "web-mercator-color-supported-values.png",
        "color_mask": output_dir / "web-mercator-color-supported-mask.png",
        "enclosed_values": output_dir / "web-mercator-enclosed-values.png",
        "enclosed_mask": output_dir / "web-mercator-enclosed-mask.png",
        "local_values": output_dir / "web-mercator-local-surrounded-values.png",
        "local_mask": output_dir / "web-mercator-local-surrounded-mask.png",
        "accepted_mask": output_dir / "web-mercator-accepted-completion-mask.png",
        "unresolved_mask": output_dir / "web-mercator-unresolved-mask.png",
        "aggressive_only_mask": output_dir / "web-mercator-aggressive-only-mask.png",
        "boundary_clipped_mask": output_dir / "web-mercator-boundary-clipped-mask.png",
        "diagnostic": output_dir / "diagnostic-montage.png",
    }
    _save_values(outputs["conservative_class_id"], candidate)
    for key, values in (
        ("color_values", color_values),
        ("enclosed_values", enclosed_values),
        ("local_values", local_values),
    ):
        _save_values(outputs[key], values)
    for key, mask in (
        ("color_mask", color_values > 0),
        ("enclosed_mask", enclosed_values > 0),
        ("local_mask", local_values > 0),
        ("accepted_mask", accepted_mask),
        ("unresolved_mask", unresolved),
        ("aggressive_only_mask", aggressive_only),
        ("boundary_clipped_mask", boundary_clipped),
    ):
        _save_mask(outputs[key], mask)
    _montage(
        source_rgb_web,
        approved_clipped,
        aggressive_clipped,
        candidate,
        accepted_mask,
        unresolved,
        categories,
    ).save(outputs["diagnostic"], optimize=True)

    report: Dict[str, object] = {
        "schema_version": 1,
        "status": "needs_visual_review",
        "audit_kind": "conservative_plant_completion_prototype",
        "approval_carried_forward": False,
        "inputs": {
            "extraction": {"path": str(extraction_path), "sha256": _sha256(extraction_path)},
            "plan": {"path": str(plan_path), "sha256": _sha256(plan_path)},
            "source": {"path": str(source_path), "sha256": _sha256(source_path)},
            "alignment": {"path": str(alignment_path), "sha256": _sha256(alignment_path)},
            "source_class": {"path": str(source_class_path), "sha256": _sha256(source_class_path)},
            "source_state": {"path": str(source_state_path), "sha256": _sha256(source_state_path)},
            "approved": {"path": str(approved_path), "sha256": _sha256(approved_path)},
            "aggressive_fixed_point": {
                "path": str(aggressive_path),
                "sha256": _sha256(aggressive_path),
            },
        },
        "grid": source_grid,
        "color_grid": color_grid,
        "policy": {
            "approved_nonzero_pixels": "preserved exactly inside target state",
            "automatic_additions": (
                "one-pass only; each assignment requires local evidence plus exact "
                "agreement with the independently generated nearest-class fixed point"
            ),
            "uncertain": "retained as explicit NoData",
            "outside_state": "clipped and separately masked",
            "rare_classes": "no long-range or recursive propagation",
        },
        "metrics": {
            "target_state_pixel_count": int(np.count_nonzero(target_state)),
            "approved_nonzero_inside_state": int(
                np.count_nonzero((approved > 0) & target_state)
            ),
            "approved_nonzero_outside_state_clipped": int(
                np.count_nonzero(boundary_clipped)
            ),
            "initial_unresolved_pixel_count": int(np.count_nonzero(original_missing)),
            "accepted_completion_pixel_count": int(np.count_nonzero(accepted_mask)),
            "accepted_completion_fraction_of_initial_unresolved": float(
                np.count_nonzero(accepted_mask)
                / max(np.count_nonzero(original_missing), 1)
            ),
            "unresolved_pixel_count": int(np.count_nonzero(unresolved)),
            "unresolved_fraction_of_state": float(
                np.count_nonzero(unresolved) / max(np.count_nonzero(target_state), 1)
            ),
            "aggressive_only_rejected_pixel_count": int(
                np.count_nonzero(aggressive_only)
            ),
            "stage_conflict_pixel_count": int(np.count_nonzero(conflict)),
            "approved_nonzero_changed_class_count": int(
                np.count_nonzero(
                    target_state & (approved > 0) & (candidate != approved)
                )
            ),
            "accepted_distance_from_approved_px": {
                "median": float(np.median(accepted_distances))
                if len(accepted_distances)
                else None,
                "p90": float(np.percentile(accepted_distances, 90))
                if len(accepted_distances)
                else None,
                "maximum": float(np.max(accepted_distances))
                if len(accepted_distances)
                else None,
            },
            "accepted_by_class": _class_counts(
                accepted_values, accepted_mask, categories
            ),
            "unresolved_aggressive_class_context": _class_counts(
                aggressive_clipped, unresolved, categories
            ),
        },
        "evidence_stages": {
            "source_color_extension": color_report,
            "small_same_class_enclosures": enclosed_report,
            "locally_surrounded_pixels": local_report,
        },
        "artifacts": {
            key: {"path": path.name, "sha256": _sha256(path)}
            for key, path in outputs.items()
        },
        "information_ceiling": (
            "The unresolved pixels lack enough independent color and topology evidence. "
            "Full-state nearest-class completion is deterministic but is not semantic "
            "proof, especially for rare 10a and 11a regions."
        ),
    }
    report_path = output_dir / "conservative-completion.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("approved", type=Path)
    parser.add_argument("aggressive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.run, args.approved, args.aggressive, args.output), indent=2))


if __name__ == "__main__":
    main()
