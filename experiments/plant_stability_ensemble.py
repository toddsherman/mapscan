"""Conservative perturbation-stability audit for plant hardiness gaps.

This experiment does not complete the accepted classification. It evaluates only
pixels that remain NoData after a separate conservative completion experiment,
and writes an isolated, hash-bound confidence surface. Approved pixels are the
only propagation seeds; earlier inferred pixels never become evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import (
    binary_erosion,
    distance_transform_edt,
    label,
    maximum_filter,
    minimum_filter,
)

from mapscan.extraction import _preview, _target_state_mask
from mapscan.reference import load_california


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text())


def _save_values(path: Path, values: np.ndarray) -> None:
    Image.fromarray(values.astype(np.uint8)).save(path, optimize=True)


def _save_mask(path: Path, mask: np.ndarray) -> None:
    Image.fromarray(mask.astype(np.uint8) * 255).save(path, optimize=True)


def _nearest_labels(values: np.ndarray) -> np.ndarray:
    if not np.any(values > 0):
        raise ValueError("Nearest-label variant requires at least one seed")
    _, nearest = distance_transform_edt(values == 0, return_indices=True)
    return values[nearest[0], nearest[1]].astype(np.uint8)


def _eroded_class_cores(values: np.ndarray, iterations: int) -> np.ndarray:
    output = np.zeros_like(values, dtype=np.uint8)
    structure = np.ones((3, 3), dtype=bool)
    for class_id in np.unique(values[values > 0]):
        core = binary_erosion(
            values == class_id,
            structure=structure,
            iterations=iterations,
            border_value=0,
        )
        output[core] = class_id
    if not np.any(output > 0):
        raise ValueError(f"All class seeds vanished after erosion {iterations}")
    return output


def _geodesic_labels(
    seeds: np.ndarray, valid_mask: np.ndarray, connectivity: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """Synchronously propagate labels while leaving equal-time conflicts open."""

    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    output = seeds.copy().astype(np.uint8)
    active = valid_mask & (output == 0)
    conflicts = np.zeros_like(active)
    if connectivity == 4:
        offsets = ((-1, 0), (1, 0), (0, -1), (0, 1))
    else:
        offsets = tuple(
            (dy, dx)
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
            if dy != 0 or dx != 0
        )
    height, width = output.shape
    step_count = 0
    for step_count in range(1, height + width + 1):
        proposed = np.zeros_like(output)
        multi_class = np.zeros_like(active)
        for dy, dx in offsets:
            destination = (
                slice(max(0, dy), min(height, height + dy)),
                slice(max(0, dx), min(width, width + dx)),
            )
            source = (
                slice(max(0, -dy), min(height, height - dy)),
                slice(max(0, -dx), min(width, width - dx)),
            )
            source_values = output[source]
            destination_active = active[destination]
            proposal_window = proposed[destination]
            valid = destination_active & (source_values > 0)
            multi_class[destination] |= (
                valid
                & (proposal_window > 0)
                & (proposal_window != source_values)
            )
            first = valid & (proposal_window == 0)
            proposal_window[first] = source_values[first]
        fill = active & (proposed > 0) & ~multi_class
        conflicts |= active & multi_class
        if not np.any(fill):
            break
        output[fill] = proposed[fill]
        active[fill | conflicts] = False
    return output, conflicts, step_count


def _local_compatibility(
    approved: np.ndarray,
    proposed: np.ndarray,
    candidates: np.ndarray,
    *,
    radius_px: int,
    minimum_support_pixels: int,
    required_supported_quadrants: int,
) -> tuple[np.ndarray, Dict[str, object]]:
    """Require a pure, spatially distributed neighborhood of approved evidence."""

    distance = distance_transform_edt(approved == 0)
    compatible = np.zeros_like(candidates)
    height, width = approved.shape
    considered = candidates & (distance <= float(radius_px))
    for y, x in zip(*np.nonzero(considered)):
        y1, y2 = max(0, y - radius_px), min(height, y + radius_px + 1)
        x1, x2 = max(0, x - radius_px), min(width, x + radius_px + 1)
        patch = approved[y1:y2, x1:x2]
        support = patch[patch > 0]
        class_id = int(proposed[y, x])
        if len(support) < minimum_support_pixels:
            continue
        if np.any(support != class_id):
            continue
        quadrants = (
            approved[y1:y, x1:x],
            approved[y1:y, x + 1 : x2],
            approved[y + 1 : y2, x1:x],
            approved[y + 1 : y2, x + 1 : x2],
        )
        supported_quadrants = sum(
            bool(np.any(quadrant == class_id)) for quadrant in quadrants
        )
        if supported_quadrants < required_supported_quadrants:
            continue
        compatible[y, x] = True
    selected_distances = distance[compatible]
    return compatible, {
        "candidate_pixel_count": int(np.count_nonzero(candidates)),
        "within_radius_pixel_count": int(np.count_nonzero(considered)),
        "accepted_pixel_count": int(np.count_nonzero(compatible)),
        "radius_px": radius_px,
        "minimum_support_pixels": minimum_support_pixels,
        "required_supported_quadrants": required_supported_quadrants,
        "required_support_purity": 1.0,
        "accepted_distance_from_approved_px": {
            "median": (
                float(np.median(selected_distances))
                if len(selected_distances)
                else None
            ),
            "p90": (
                float(np.percentile(selected_distances, 90))
                if len(selected_distances)
                else None
            ),
            "maximum": (
                float(np.max(selected_distances))
                if len(selected_distances)
                else None
            ),
        },
    }


def _class_counts(values: np.ndarray, mask: np.ndarray, categories) -> Dict[str, int]:
    return {
        str(category["id"]): int(np.count_nonzero(mask & (values == index)))
        for index, category in enumerate(categories, 1)
    }


def _component_report(
    unresolved: np.ndarray, accepted: np.ndarray, large_area: int
) -> Dict[str, object]:
    components, count = label(
        unresolved, structure=np.ones((3, 3), dtype=np.uint8)
    )
    sizes = np.bincount(components.ravel())
    rows = []
    for component_id in range(1, count + 1):
        area = int(sizes[component_id])
        component = components == component_id
        accepted_count = int(np.count_nonzero(component & accepted))
        if area >= large_area:
            y, x = np.nonzero(component)
            rows.append(
                {
                    "component_id": component_id,
                    "area_px": area,
                    "accepted_edge_pixel_count": accepted_count,
                    "remaining_unresolved_pixel_count": area - accepted_count,
                    "bbox_xyxy": [
                        int(x.min()),
                        int(y.min()),
                        int(x.max()),
                        int(y.max()),
                    ],
                }
            )
    rows.sort(key=lambda row: (-int(row["area_px"]), int(row["component_id"])))
    large_mask = (components > 0) & (sizes[components] >= large_area)
    return {
        "component_count": int(count),
        "large_component_minimum_area_px": large_area,
        "large_component_count": len(rows),
        "large_component_unresolved_pixel_count_before": int(
            np.count_nonzero(large_mask)
        ),
        "large_component_accepted_edge_pixel_count": int(
            np.count_nonzero(large_mask & accepted)
        ),
        "large_component_remaining_unresolved_pixel_count": int(
            np.count_nonzero(large_mask & ~accepted)
        ),
        "large_components": rows,
    }


def _montage(
    approved: np.ndarray,
    conservative: np.ndarray,
    aggressive: np.ndarray,
    consensus_values: np.ndarray,
    accepted: np.ndarray,
    remaining: np.ndarray,
    categories,
) -> Image.Image:
    base_preview = np.asarray(
        Image.fromarray(_preview(conservative, categories)).convert("RGB")
    )
    accepted_preview = base_preview.copy()
    accepted_preview[accepted] = [0, 255, 255]
    remaining_preview = base_preview.copy()
    remaining_preview[remaining] = [255, 0, 255]
    consensus_preview = np.asarray(
        Image.fromarray(_preview(consensus_values, categories)).convert("RGB")
    )
    panels = (
        (
            "Approved evidence only",
            Image.fromarray(_preview(approved, categories)).convert("RGB"),
        ),
        ("Prior conservative surface", Image.fromarray(base_preview)),
        (
            "Aggressive fixed point",
            Image.fromarray(_preview(aggressive, categories)).convert("RGB"),
        ),
        ("Raw all-method consensus", Image.fromarray(consensus_preview)),
        ("Strict accepted (cyan)", Image.fromarray(accepted_preview)),
        ("Still unresolved (magenta)", Image.fromarray(remaining_preview)),
    )
    height, width = approved.shape
    header = 24
    canvas = Image.new("RGB", (width * 3, (height + header) * 2), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    for index, (title, image) in enumerate(panels):
        column, row = index % 3, index // 3
        x, y = column * width, row * (height + header)
        draw.text((x + 6, y + 5), title, fill=(245, 245, 245))
        canvas.paste(image, (x, y + header))
    return canvas


def run(
    run_dir: Path,
    approved_path: Path,
    conservative_path: Path,
    aggressive_path: Path,
    output_dir: Path,
) -> Dict[str, object]:
    run_dir = run_dir.resolve()
    approved_path = approved_path.resolve()
    conservative_path = conservative_path.resolve()
    aggressive_path = aggressive_path.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    extraction_path = run_dir / "extraction.json"
    extraction = _load(extraction_path)
    plan_path = Path(str(extraction["plan"]["path"])).resolve()
    plan = _load(plan_path)
    categories = plan["layers"][0]["categories"]
    reference_root = Path(
        str(plan.get("reference", "reference/census-2025"))
    ).resolve()
    state, _ = load_california(reference_root)

    approved = np.asarray(Image.open(approved_path), dtype=np.uint8)
    conservative = np.asarray(Image.open(conservative_path), dtype=np.uint8)
    aggressive = np.asarray(Image.open(aggressive_path), dtype=np.uint8)
    if not (approved.shape == conservative.shape == aggressive.shape):
        raise ValueError("Approved, conservative, and aggressive rasters must match")
    target_state = _target_state_mask(
        state, extraction["layers"][0]["warp"]["bounds"], approved.shape
    )
    approved_clipped = approved.copy()
    approved_clipped[~target_state] = 0
    conservative_clipped = conservative.copy()
    conservative_clipped[~target_state] = 0
    aggressive_clipped = aggressive.copy()
    aggressive_clipped[~target_state] = 0

    if np.any(
        target_state
        & (approved_clipped > 0)
        & (conservative_clipped != approved_clipped)
    ):
        raise ValueError("Conservative input changed approved nonzero evidence")
    unresolved_before = target_state & (conservative_clipped == 0)

    euclidean = _nearest_labels(approved_clipped)
    eroded_1 = _nearest_labels(_eroded_class_cores(approved_clipped, 1))
    eroded_2 = _nearest_labels(_eroded_class_cores(approved_clipped, 2))
    geodesic_4, geodesic_4_conflict, geodesic_4_steps = _geodesic_labels(
        approved_clipped, target_state, 4
    )
    geodesic_8, geodesic_8_conflict, geodesic_8_steps = _geodesic_labels(
        approved_clipped, target_state, 8
    )
    variants = {
        "aggressive_fixed_point": aggressive_clipped,
        "euclidean_approved": euclidean,
        "eroded_core_1": eroded_1,
        "eroded_core_2": eroded_2,
        "geodesic_4": geodesic_4,
        "geodesic_8": geodesic_8,
    }
    stack = np.stack(list(variants.values()))
    invariant = (
        unresolved_before
        & (np.min(stack, axis=0) == np.max(stack, axis=0))
        & (aggressive_clipped > 0)
    )
    coordinate_stable = (
        minimum_filter(euclidean, size=3, mode="nearest")
        == maximum_filter(euclidean, size=3, mode="nearest")
    )
    perturbation_stable = invariant & coordinate_stable
    locally_compatible, local_report = _local_compatibility(
        approved_clipped,
        euclidean,
        perturbation_stable,
        radius_px=3,
        minimum_support_pixels=4,
        required_supported_quadrants=2,
    )
    # A class can be numerically stable at the coastline/state outline while
    # still being an alignment or outline-occlusion question. Keep that evidence
    # separate from internal holes so it cannot silently become ocean fill.
    state_interior_distance = distance_transform_edt(target_state)
    # Plant v4's measured perimeter maximum is 4 px. Add the 3 px local
    # evidence radius and one guard pixel so residual warp error cannot turn a
    # coastline-outline gap into an automatically accepted class assignment.
    boundary_review_radius_px = 8.0
    boundary_proximal = locally_compatible & (
        state_interior_distance <= boundary_review_radius_px
    )
    accepted = locally_compatible & ~boundary_proximal
    accepted_values = np.where(accepted, euclidean, 0).astype(np.uint8)
    boundary_proximal_values = np.where(
        boundary_proximal, euclidean, 0
    ).astype(np.uint8)
    ensemble_surface = conservative_clipped.copy()
    ensemble_surface[accepted] = accepted_values[accepted]
    remaining = target_state & (ensemble_surface == 0)
    raw_consensus_values = np.where(perturbation_stable, euclidean, 0).astype(
        np.uint8
    )

    if np.any(accepted & ~unresolved_before):
        raise AssertionError("Ensemble accepted a pixel outside the unresolved set")
    if np.any(accepted_values[accepted] != aggressive_clipped[accepted]):
        raise AssertionError("Accepted ensemble values disagree with fixed point")
    if np.any(
        target_state
        & (conservative_clipped > 0)
        & (ensemble_surface != conservative_clipped)
    ):
        raise AssertionError("Ensemble changed a prior nonzero value")

    outputs = {
        "ensemble_surface": output_dir / "web-mercator-class-id-ensemble-audit.png",
        "consensus_values": output_dir / "web-mercator-consensus-values.png",
        "invariant_mask": output_dir / "web-mercator-all-method-invariant-mask.png",
        "coordinate_stable_mask": output_dir
        / "web-mercator-coordinate-stable-mask.png",
        "accepted_values": output_dir / "web-mercator-accepted-values.png",
        "accepted_mask": output_dir / "web-mercator-accepted-mask.png",
        "boundary_proximal_values": output_dir
        / "web-mercator-boundary-proximal-values.png",
        "boundary_proximal_mask": output_dir
        / "web-mercator-boundary-proximal-mask.png",
        "remaining_mask": output_dir / "web-mercator-remaining-unresolved-mask.png",
        "geodesic_4_conflicts": output_dir
        / "web-mercator-geodesic-4-conflict-mask.png",
        "geodesic_8_conflicts": output_dir
        / "web-mercator-geodesic-8-conflict-mask.png",
        "diagnostic": output_dir / "diagnostic-montage.png",
    }
    for key, values in (
        ("ensemble_surface", ensemble_surface),
        ("consensus_values", raw_consensus_values),
        ("accepted_values", accepted_values),
        ("boundary_proximal_values", boundary_proximal_values),
    ):
        _save_values(outputs[key], values)
    for key, mask in (
        ("invariant_mask", invariant),
        ("coordinate_stable_mask", coordinate_stable & unresolved_before),
        ("accepted_mask", accepted),
        ("boundary_proximal_mask", boundary_proximal),
        ("remaining_mask", remaining),
        ("geodesic_4_conflicts", geodesic_4_conflict),
        ("geodesic_8_conflicts", geodesic_8_conflict),
    ):
        _save_mask(outputs[key], mask)
    variant_artifacts = {}
    for name, values in variants.items():
        path = output_dir / f"variant-{name}.png"
        _save_values(path, values)
        variant_artifacts[name] = {"path": path.name, "sha256": _sha256(path)}
    _montage(
        approved_clipped,
        conservative_clipped,
        aggressive_clipped,
        raw_consensus_values,
        accepted,
        remaining,
        categories,
    ).save(outputs["diagnostic"], optimize=True)

    components = _component_report(unresolved_before, accepted, large_area=50)
    accepted_count = int(np.count_nonzero(accepted))
    unresolved_count = int(np.count_nonzero(unresolved_before))
    report: Dict[str, object] = {
        "schema_version": 1,
        "status": "experimental_confidence_surface_only",
        "audit_kind": "plant_unresolved_perturbation_stability_ensemble",
        "approval_carried_forward": False,
        "inputs": {
            "extraction": {
                "path": str(extraction_path),
                "sha256": _sha256(extraction_path),
            },
            "plan": {"path": str(plan_path), "sha256": _sha256(plan_path)},
            "approved_seed_evidence": {
                "path": str(approved_path),
                "sha256": _sha256(approved_path),
            },
            "prior_conservative_surface": {
                "path": str(conservative_path),
                "sha256": _sha256(conservative_path),
                "use": "defines unresolved pixels only; its inferred pixels are not seeds",
            },
            "aggressive_fixed_point": {
                "path": str(aggressive_path),
                "sha256": _sha256(aggressive_path),
                "use": "agreement variant only; never sufficient evidence by itself",
            },
        },
        "policy": {
            "scope": "only prior-conservative NoData inside authoritative state",
            "seed_evidence": "approved 118-stamp raster only",
            "method_agreement": "all six variants must return the same nonzero class",
            "coordinate_perturbation": "3x3 neighborhood of Euclidean labels must be constant",
            "local_compatibility": (
                "within 3 px of approved evidence, at least 4 approved support pixels, "
                "100% same class, distributed across at least 2 strict quadrants"
            ),
            "state_boundary": (
                "locally compatible pixels within 8 px of the authoritative state "
                "boundary are separated for review and not auto-accepted; the band is "
                "plant v4's 4 px maximum perimeter residual plus the 3 px local radius "
                "and one guard pixel"
            ),
            "growth": "single audit pass; accepted pixels never become seeds",
            "rare_classes": "no exception or long-range propagation",
        },
        "metrics": {
            "target_state_pixel_count": int(np.count_nonzero(target_state)),
            "unresolved_pixel_count_before": unresolved_count,
            "all_method_invariant_pixel_count": int(np.count_nonzero(invariant)),
            "perturbation_stable_pixel_count": int(
                np.count_nonzero(perturbation_stable)
            ),
            "locally_compatible_pixel_count": int(
                np.count_nonzero(locally_compatible)
            ),
            "boundary_proximal_review_pixel_count": int(
                np.count_nonzero(boundary_proximal)
            ),
            "boundary_review_radius_px": boundary_review_radius_px,
            "accepted_pixel_count": accepted_count,
            "accepted_fraction_of_unresolved": float(
                accepted_count / max(unresolved_count, 1)
            ),
            "remaining_unresolved_pixel_count": int(np.count_nonzero(remaining)),
            "remaining_unresolved_fraction_of_state": float(
                np.count_nonzero(remaining) / max(np.count_nonzero(target_state), 1)
            ),
            "changed_prior_nonzero_pixel_count": int(
                np.count_nonzero(
                    target_state
                    & (conservative_clipped > 0)
                    & (ensemble_surface != conservative_clipped)
                )
            ),
            "accepted_by_class": _class_counts(
                accepted_values, accepted, categories
            ),
            "boundary_proximal_by_class": _class_counts(
                boundary_proximal_values, boundary_proximal, categories
            ),
            "geodesic": {
                "connectivity_4": {
                    "step_count": geodesic_4_steps,
                    "conflict_pixel_count": int(
                        np.count_nonzero(geodesic_4_conflict)
                    ),
                },
                "connectivity_8": {
                    "step_count": geodesic_8_steps,
                    "conflict_pixel_count": int(
                        np.count_nonzero(geodesic_8_conflict)
                    ),
                },
            },
        },
        "local_compatibility": local_report,
        "component_audit": components,
        "variants": variant_artifacts,
        "artifacts": {
            key: {"path": path.name, "sha256": _sha256(path)}
            for key, path in outputs.items()
        },
        "decision": {
            "all_aggressive_completion_pixels": "no_go",
            "ensemble_surface": "audit_only_not_approved",
            "information_ceiling": (
                "Method invariance proves numerical stability, not the class hidden under "
                "large city-label, road, coastline, or island occlusions. Only locally "
                "supported internal edge pixels are separated; boundary-proximal pixels "
                "require review and occlusion interiors remain explicit NoData."
            ),
        },
    }
    report_path = output_dir / "stability-ensemble.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("approved", type=Path)
    parser.add_argument("conservative", type=Path)
    parser.add_argument("aggressive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = run(
        args.run,
        args.approved,
        args.conservative,
        args.aggressive,
        args.output,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
