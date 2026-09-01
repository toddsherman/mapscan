"""Deterministic neighbor completion for explicitly unresolved categorical pixels.

This stage is intentionally separate from observed classification and authored
stamp evidence.  It clips to a hash-bound mainland interior plus the four
closed components of the active canonical island linework, then fills only the
remaining unresolved pixels inside that mask.  Every assumed pixel remains
separately masked and reviewable.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Sequence

import cv2
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from .canonical_boundary import load_active_canonical_border


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text())


def _resolve(value: object) -> Path:
    return Path(str(value)).expanduser().resolve()


def _verified(path: Path, expected: object, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    if expected and _sha256(path) != str(expected):
        raise ValueError(f"Stale {label}: {path}")
    return path


def _artifact(root: Path, record: Dict[str, object], label: str) -> Path:
    return _verified(root / str(record["path"]), record.get("sha256"), label)


def _materialized_layer(root: Path) -> tuple[Dict[str, object], Dict[str, object]]:
    manifest = _load(root / "materialization.json")
    layers = manifest.get("layers", [])
    if not isinstance(layers, list) or len(layers) != 1:
        raise ValueError("Neighbor completion requires one categorical layer")
    return manifest, layers[0]


def _preview(values: np.ndarray, categories: Sequence[Dict[str, object]]) -> np.ndarray:
    rgba = np.zeros((*values.shape, 4), dtype=np.uint8)
    for class_id, category in enumerate(categories, 1):
        color = category.get("display_rgb", category.get("legend_rgb", [255, 0, 255]))
        if isinstance(color, list) and color and isinstance(color[0], list):
            color = color[0]
        selected = values == class_id
        rgba[selected, :3] = np.asarray(color[:3], dtype=np.uint8)
        rgba[selected, 3] = 235
    return rgba


def closed_line_component_interiors(
    line_mask: np.ndarray, *, expected_components: int
) -> np.ndarray:
    """Fill each closed line component without joining independent islands."""

    line = line_mask.astype(bool)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        line.astype(np.uint8), 8
    )
    if count - 1 != expected_components:
        raise ValueError(
            f"Expected {expected_components} closed line components, found {count - 1}"
        )
    filled = np.zeros(line.shape, dtype=bool)
    for component in range(1, count):
        x, y, width, height, _ = stats[component].tolist()
        x0, y0 = max(0, x - 2), max(0, y - 2)
        x1, y1 = min(line.shape[1], x + width + 2), min(line.shape[0], y + height + 2)
        barrier = labels[y0:y1, x0:x1] == component
        free = ~barrier
        free_count, free_labels = cv2.connectedComponents(
            free.astype(np.uint8), 4
        )
        exterior_labels = set()
        for edge in (
            free_labels[0, :],
            free_labels[-1, :],
            free_labels[:, 0],
            free_labels[:, -1],
        ):
            exterior_labels.update(int(value) for value in np.unique(edge) if value)
        if not exterior_labels or free_count <= 1:
            raise ValueError("Canonical island line has no measurable exterior")
        exterior = np.isin(free_labels, list(exterior_labels))
        interior = ~exterior
        if int(np.count_nonzero(interior & ~barrier)) == 0:
            raise ValueError("Canonical island line is open and cannot define an interior")
        filled[y0:y1, x0:x1] |= interior
    return filled


def fill_unknown_from_neighbors(
    seed_values: np.ndarray,
    unknown_inside: np.ndarray,
    manual_seed_mask: np.ndarray,
    *,
    neighbors: int = 16,
    manual_weight: float = 2.0,
) -> tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, object]]:
    """Fill connected unknown regions from their nearest known boundary pixels.

    A component never borrows from another unknown component.  For every pixel,
    the nearest ``neighbors`` known pixels touching that component vote by
    inverse-square distance.  Authored nonzero stamp pixels receive the declared
    multiplier.  Ties resolve to the lower class id for byte determinism.
    """

    if not (
        seed_values.shape == unknown_inside.shape == manual_seed_mask.shape
    ):
        raise ValueError("Neighbor-completion rasters use different grids")
    if neighbors < 1:
        raise ValueError("neighbors must be positive")
    if manual_weight < 1.0:
        raise ValueError("manual_weight must be at least 1")
    seed = seed_values.astype(np.uint8)
    unknown = unknown_inside.astype(bool)
    manual = manual_seed_mask.astype(bool) & (seed > 0)
    output = seed.copy()
    confidence = np.zeros(seed.shape, dtype=np.uint8)
    nearest_distance = np.zeros(seed.shape, dtype=np.float32)
    manual_neighbor = np.zeros(seed.shape, dtype=bool)
    manual_changed_choice = np.zeros(seed.shape, dtype=bool)
    fallback = np.zeros(seed.shape, dtype=bool)
    known_global = np.column_stack(np.nonzero(seed > 0))
    if unknown.any() and known_global.size == 0:
        raise ValueError("Cannot fill unknown pixels without class seed evidence")
    global_tree = cKDTree(known_global) if known_global.size else None
    component_count, components = cv2.connectedComponents(
        unknown.astype(np.uint8), 8
    )
    component_records = []
    maximum_class = int(seed.max())
    kernel = np.ones((3, 3), dtype=np.uint8)

    for component in range(1, component_count):
        region = components == component
        query = np.column_stack(np.nonzero(region))
        ring = cv2.dilate(region.astype(np.uint8), kernel, iterations=1).astype(bool)
        boundary_seed_mask = ring & ~region & (seed > 0)
        boundary = np.column_stack(np.nonzero(boundary_seed_mask))
        used_global_fallback = boundary.size == 0
        if used_global_fallback:
            boundary = known_global
            tree = global_tree
            fallback[region] = True
        else:
            tree = cKDTree(boundary)
        assert tree is not None
        k = min(int(neighbors), len(boundary))
        distances, indices = tree.query(query, k=k, workers=1)
        if k == 1:
            distances = distances[:, None]
            indices = indices[:, None]
        neighbor_coordinates = boundary[np.asarray(indices, dtype=np.int64)]
        neighbor_classes = seed[
            neighbor_coordinates[..., 0], neighbor_coordinates[..., 1]
        ]
        neighbor_manual = manual[
            neighbor_coordinates[..., 0], neighbor_coordinates[..., 1]
        ]
        base_weights = 1.0 / np.square(np.asarray(distances, dtype=np.float64) + 0.5)
        weighted = base_weights * np.where(neighbor_manual, manual_weight, 1.0)
        scores = np.zeros((len(query), maximum_class + 1), dtype=np.float64)
        base_scores = np.zeros_like(scores)
        for class_id in range(1, maximum_class + 1):
            selected = neighbor_classes == class_id
            scores[:, class_id] = np.sum(weighted * selected, axis=1)
            base_scores[:, class_id] = np.sum(base_weights * selected, axis=1)
        predicted = np.argmax(scores[:, 1:], axis=1).astype(np.uint8) + 1
        predicted_without_manual_weight = (
            np.argmax(base_scores[:, 1:], axis=1).astype(np.uint8) + 1
        )
        total = np.sum(scores[:, 1:], axis=1)
        winner = scores[np.arange(len(query)), predicted]
        confidence_values = np.divide(
            winner,
            total,
            out=np.ones_like(winner),
            where=total > 0,
        )
        rows, columns = query[:, 0], query[:, 1]
        output[rows, columns] = predicted
        confidence[rows, columns] = np.clip(
            np.rint(confidence_values * 255.0), 0, 255
        ).astype(np.uint8)
        nearest_distance[rows, columns] = np.asarray(distances)[:, 0]
        manual_neighbor[rows, columns] = np.any(neighbor_manual, axis=1)
        manual_changed_choice[rows, columns] = predicted != predicted_without_manual_weight
        component_records.append(
            {
                "component": int(component),
                "pixel_count": int(len(query)),
                "boundary_seed_count": int(np.count_nonzero(boundary_seed_mask)),
                "used_global_fallback": bool(used_global_fallback),
                "maximum_nearest_seed_distance_px": float(np.max(np.asarray(distances)[:, 0])),
            }
        )

    metrics: Dict[str, object] = {
        "component_count": int(component_count - 1),
        "global_fallback_component_count": int(
            sum(record["used_global_fallback"] for record in component_records)
        ),
        "manual_neighbor_pixel_count": int(np.count_nonzero(manual_neighbor)),
        "manual_weight_changed_choice_pixel_count": int(
            np.count_nonzero(manual_changed_choice)
        ),
        "nearest_seed_distance_px": {
            "median": float(np.median(nearest_distance[unknown])) if unknown.any() else 0.0,
            "p90": float(np.percentile(nearest_distance[unknown], 90)) if unknown.any() else 0.0,
            "maximum": float(np.max(nearest_distance[unknown])) if unknown.any() else 0.0,
        },
        "largest_components": sorted(
            component_records, key=lambda record: int(record["pixel_count"]), reverse=True
        )[:25],
    }
    diagnostics = {
        "confidence": confidence,
        "nearest_distance": nearest_distance,
        "manual_neighbor": manual_neighbor,
        "manual_changed_choice": manual_changed_choice,
        "fallback": fallback,
    }
    return output, diagnostics, metrics


def partition_completion_pixels(
    seed_values: np.ndarray,
    audited_unknown: np.ndarray,
    valid_interior: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Partition the reviewed unknowns and any other interior NoData pixels.

    The earlier fidelity audit intentionally tracked only source-visible holes.
    Publication integrity is stricter: every zero-valued pixel inside the
    approved interior must be completed, including grid-edge pixels that were
    outside that audit's source-support mask.
    """

    if not (
        seed_values.shape == audited_unknown.shape == valid_interior.shape
    ):
        raise ValueError("Completion partition rasters use different grids")
    audited = audited_unknown.astype(bool)
    valid = valid_interior.astype(bool)
    unclassified = seed_values == 0
    inside = unclassified & valid
    audited_inside = audited & inside
    additional_inside = inside & ~audited
    audited_outside = audited & ~valid
    return inside, audited_outside, additional_inside


def complete_neighbor_unknowns(config_path: Path) -> Dict[str, object]:
    """Create a hash-bound, non-published completion candidate from a config."""

    config_path = config_path.resolve()
    config = _load(config_path)
    if int(config.get("schema_version", 0)) != 1:
        raise ValueError("Unsupported neighbor-completion config")
    extraction_run = _resolve(config["extraction_run"])
    approved_root = _resolve(config["approved_materialization"])
    conservative_path = _resolve(config["conservative_audit"])
    stability_path = _resolve(config["stability_audit"])
    clipping_path = _resolve(config["canonical_boundary_audit"])
    active_pointer_path = _resolve(config["active_canonical_boundary"])
    output_dir = _resolve(config["output_dir"])
    if (output_dir / "materialization-review-decision.json").exists():
        raise ValueError("Refusing to overwrite a reviewed completion directory")

    extraction_path = extraction_run / "extraction.json"
    extraction = _load(extraction_path)
    plan_path = extraction_run / "plan.snapshot.json"
    plan = _load(plan_path)
    categories = plan["layers"][0]["categories"]
    grid = extraction["alignment"]["inspection"]["grid"]
    shape = (int(grid["height"]), int(grid["width"]))
    approved_manifest, approved_layer = _materialized_layer(approved_root)
    approved_path = _artifact(
        approved_root, approved_layer["artifacts"]["class_id"], "approved class raster"
    )
    manual_path = _artifact(
        approved_root, approved_layer["artifacts"]["manual_mask"], "authored stamp mask"
    )
    conservative = _load(conservative_path)
    stability = _load(stability_path)
    clipping = _load(clipping_path)
    if conservative.get("status") != "needs_visual_review":
        raise ValueError("Conservative completion audit is not reviewable")
    if stability.get("status") != "experimental_confidence_surface_only":
        raise ValueError("Stability completion is not an audit surface")
    if clipping.get("status") != "pass":
        raise ValueError("Canonical clipping interior has not passed")
    if clipping.get("target_extraction", {}).get("sha256") != _sha256(extraction_path):
        raise ValueError("Canonical clipping interior does not bind this extraction")

    active_manifest_path, active, pointer = load_active_canonical_border(
        active_pointer_path
    )
    active_grid = active["source_grid"]
    for field in ("crs", "bounds"):
        if active_grid.get(field) != grid.get(field):
            raise ValueError(f"Active canonical boundary differs at {field}")
    if int(active["topology"]["offshore_island_component_count"]) != 4:
        raise ValueError("Active canonical boundary must contain four island outlines")

    stability_root = stability_path.parent
    seed_path = _artifact(
        stability_root, stability["artifacts"]["ensemble_surface"], "tiered seed surface"
    )
    unresolved_path = _artifact(
        stability_root, stability["artifacts"]["remaining_mask"], "remaining unknown mask"
    )
    approved_clear_path = _artifact(
        conservative_path.parent,
        conservative["artifacts"]["boundary_clipped_mask"],
        "authoritative approved-clear mask",
    )
    clipping_interior_path = _artifact(
        clipping_path.parent, clipping["artifacts"]["interior"], "canonical clipping interior"
    )
    island_record = active["artifacts"]["islands"]
    island_line_path = _verified(
        active_manifest_path.parent / str(island_record["path"]),
        island_record["sha256"],
        "active canonical island linework",
    )

    approved = np.asarray(Image.open(approved_path), dtype=np.uint8)
    seed = np.asarray(Image.open(seed_path), dtype=np.uint8)
    unresolved = np.asarray(Image.open(unresolved_path)) > 0
    manual = np.asarray(Image.open(manual_path)) > 0
    mainland = np.asarray(Image.open(clipping_interior_path)) > 0
    approved_clear = np.asarray(Image.open(approved_clear_path)) > 0
    if not all(
        values.shape == shape
        for values in (approved, seed, unresolved, manual, mainland, approved_clear)
    ):
        raise ValueError("Neighbor-completion inputs use different grids")
    invalid_approved_change = (
        (approved > 0)
        & (seed != approved)
        & ~(approved_clear & (seed == 0))
    )
    if np.any(invalid_approved_change):
        raise ValueError("Tiered seed surface changes approved nonzero evidence")
    expected_unresolved = int(stability["metrics"]["remaining_unresolved_pixel_count"])
    if int(np.count_nonzero(unresolved)) != expected_unresolved:
        raise ValueError("Remaining unknown count differs from its audit")

    island_rgba = np.asarray(Image.open(island_line_path).convert("RGBA"))
    island_high = closed_line_component_interiors(
        island_rgba[..., 3] > 0,
        expected_components=int(active["topology"]["offshore_island_component_count"]),
    )
    island_interior = cv2.resize(
        island_high.astype(np.uint8),
        (shape[1], shape[0]),
        interpolation=cv2.INTER_NEAREST,
    ) > 0
    valid_interior = mainland | island_interior
    unknown_inside, unknown_outside, additional_inside = partition_completion_pixels(
        seed, unresolved, valid_interior
    )
    audited_unknown_inside = unresolved & valid_interior

    output, diagnostics, fill_metrics = fill_unknown_from_neighbors(
        seed,
        unknown_inside,
        manual,
        neighbors=int(config.get("neighbors", 16)),
        manual_weight=float(config.get("manual_weight", 2.0)),
    )
    existing_outside = (seed > 0) & ~valid_interior
    output[~valid_interior] = 0
    if np.any(output[unknown_inside] == 0):
        raise AssertionError("Neighbor completion left unknown pixels inside the border")
    if np.any(output[~valid_interior] != 0):
        raise AssertionError("Neighbor completion retained pixels outside the border")
    preserved = valid_interior & (seed > 0)
    if np.any(output[preserved] != seed[preserved]):
        raise AssertionError("Neighbor completion changed prior classified evidence")

    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: Dict[str, Dict[str, object]] = {}

    def save(name: str, values: np.ndarray) -> Path:
        path = output_dir / name
        Image.fromarray(values).save(path, optimize=True)
        artifacts[name] = {"path": name, "sha256": _sha256(path)}
        return path

    save("web-mercator-class-id-neighbor-completed.png", output.astype(np.uint8))
    save("web-mercator-preview-neighbor-completed.png", _preview(output, categories))
    save("web-mercator-valid-interior-mask.png", (valid_interior * 255).astype(np.uint8))
    save("web-mercator-neighbor-assumption-mask.png", (unknown_inside * 255).astype(np.uint8))
    save("web-mercator-unknown-outside-removed-mask.png", (unknown_outside * 255).astype(np.uint8))
    save("web-mercator-existing-outside-removed-mask.png", (existing_outside * 255).astype(np.uint8))
    save("web-mercator-manual-neighbor-mask.png", (diagnostics["manual_neighbor"] * 255).astype(np.uint8))
    save("web-mercator-manual-weight-changed-choice-mask.png", (diagnostics["manual_changed_choice"] * 255).astype(np.uint8))
    save("web-mercator-neighbor-confidence.png", diagnostics["confidence"].astype(np.uint8))
    distance_scaled = np.clip(np.rint(diagnostics["nearest_distance"] * 256.0), 0, 65535).astype(np.uint16)
    save("web-mercator-nearest-seed-distance-x256.png", distance_scaled)

    by_class = {
        str(class_id): int(np.count_nonzero(unknown_inside & (output == class_id)))
        for class_id in range(1, len(categories) + 1)
    }
    result: Dict[str, object] = {
        "schema_version": 1,
        "status": "needs_visual_review",
        "publication_allowed": False,
        "policy": {
            "outside_active_border": "transparent",
            "inside_remaining_unknown": "inverse_distance_neighbor_vote",
            "authored_nonzero_stamp_seed_weight": float(config.get("manual_weight", 2.0)),
            "neighbor_count": int(config.get("neighbors", 16)),
            "tie_break": "lowest_class_id",
            "prior_inside_observed_and_authored_evidence_changed": False,
            "prior_outside_evidence": "clipped_and_separately_masked",
            "publication_integrity": "fill_every_zero_pixel_inside_valid_interior",
        },
        "grid": grid,
        "metrics": {
            "remaining_unknown_before": expected_unresolved,
            "unknown_inside_filled": int(np.count_nonzero(unknown_inside)),
            "audited_unknown_inside_filled": int(
                np.count_nonzero(audited_unknown_inside)
            ),
            "additional_unclassified_inside_filled": int(
                np.count_nonzero(additional_inside)
            ),
            "unknown_outside_removed": int(np.count_nonzero(unknown_outside)),
            "existing_classified_outside_removed": int(np.count_nonzero(existing_outside)),
            "remaining_unknown_inside": int(np.count_nonzero(unknown_inside & (output == 0))),
            "classified_outside_valid_interior": int(np.count_nonzero(output[~valid_interior])),
            "final_classified": int(np.count_nonzero(output)),
            "filled_by_class_id": by_class,
            "neighbor_fill": fill_metrics,
        },
        "inputs": {
            "config": {"path": str(config_path), "sha256": _sha256(config_path)},
            "extraction": {"path": str(extraction_path), "sha256": _sha256(extraction_path)},
            "approved_materialization": {
                "path": str(approved_root / "materialization.json"),
                "sha256": _sha256(approved_root / "materialization.json"),
                "class_id_sha256": _sha256(approved_path),
            },
            "tiered_seed_surface": {"path": str(seed_path), "sha256": _sha256(seed_path)},
            "remaining_unknown": {"path": str(unresolved_path), "sha256": _sha256(unresolved_path)},
            "clipping_interior": {
                "path": str(clipping_interior_path),
                "sha256": _sha256(clipping_interior_path),
                "canonical_boundary_id": clipping.get("canonical_boundary_id"),
                "role": "separately_versioned_mainland_clipping_evidence",
            },
            "active_canonical_border": {
                "pointer": {"path": str(active_pointer_path), "sha256": _sha256(active_pointer_path)},
                "manifest": {"path": str(active_manifest_path), "sha256": _sha256(active_manifest_path)},
                "canonical_boundary_id": active.get("canonical_boundary_id"),
                "island_line_sha256": _sha256(island_line_path),
                "role": "approved_display_border_and_four_closed_island_interiors",
            },
        },
        "artifacts": artifacts,
    }
    report_path = output_dir / "neighbor-completion.json"
    report_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def promote_neighbor_completion(
    neighbor_report_path: Path,
    review_session_path: Path,
    output_dir: Path,
    *,
    author_statement: str,
) -> Dict[str, object]:
    """Promote the exact reviewed neighbor surface into an approved candidate."""

    neighbor_report_path = neighbor_report_path.resolve()
    review_session_path = review_session_path.resolve()
    output_dir = output_dir.resolve()
    statement = author_statement.strip()
    if not statement:
        raise ValueError("Neighbor completion approval requires an author statement")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Neighbor completion promotion requires a fresh output directory")
    report = _load(neighbor_report_path)
    review = _load(review_session_path)
    if (
        report.get("status") != "needs_visual_review"
        or report.get("publication_allowed") is not False
    ):
        raise ValueError("Neighbor completion is not eligible for review promotion")
    review_neighbor = review.get("evidence", {}).get("neighbor_completion")
    if (
        not isinstance(review_neighbor, dict)
        or Path(str(review_neighbor.get("path", ""))).resolve() != neighbor_report_path
        or review_neighbor.get("sha256") != _sha256(neighbor_report_path)
    ):
        raise ValueError("Review session does not bind the neighbor completion")

    approved_manifest_path = Path(
        str(report["inputs"]["approved_materialization"]["path"])
    ).resolve()
    approved_root = approved_manifest_path.parent
    approved_manifest = _verified(
        approved_manifest_path,
        report["inputs"]["approved_materialization"]["sha256"],
        "approved predecessor materialization",
    )
    predecessor, predecessor_layer = _materialized_layer(approved_root)
    extraction_path = _verified(
        Path(str(report["inputs"]["extraction"]["path"])).resolve(),
        report["inputs"]["extraction"]["sha256"],
        "neighbor-completion extraction",
    )
    extraction = _load(extraction_path)
    source_run = extraction_path.parent
    plan_path = source_run / "plan.snapshot.json"
    plan = _load(plan_path)
    layer_id = str(predecessor_layer["layer_id"])
    categories = plan["layers"][0]["categories"]

    def report_artifact(name: str, label: str) -> Path:
        return _artifact(
            neighbor_report_path.parent,
            report["artifacts"][name],
            label,
        )

    class_path = report_artifact(
        "web-mercator-class-id-neighbor-completed.png", "neighbor-completed class raster"
    )
    preview_path = report_artifact(
        "web-mercator-preview-neighbor-completed.png", "neighbor-completed preview"
    )
    valid_path = report_artifact(
        "web-mercator-valid-interior-mask.png", "publication interior"
    )
    neighbor_mask_path = report_artifact(
        "web-mercator-neighbor-assumption-mask.png", "neighbor-assumption mask"
    )
    outside_unknown_path = report_artifact(
        "web-mercator-unknown-outside-removed-mask.png", "outside unknown mask"
    )
    outside_existing_path = report_artifact(
        "web-mercator-existing-outside-removed-mask.png", "outside evidence mask"
    )
    confidence_path = report_artifact(
        "web-mercator-neighbor-confidence.png", "neighbor confidence"
    )
    distance_path = report_artifact(
        "web-mercator-nearest-seed-distance-x256.png", "neighbor seed distance"
    )
    seed_path = _verified(
        Path(str(report["inputs"]["tiered_seed_surface"]["path"])).resolve(),
        report["inputs"]["tiered_seed_surface"]["sha256"],
        "tiered seed surface",
    )

    approved_class_path = _artifact(
        approved_root, predecessor_layer["artifacts"]["class_id"], "predecessor class raster"
    )
    observed_path = _artifact(
        approved_root, predecessor_layer["artifacts"]["observed_mask"], "observed evidence mask"
    )
    manual_path = _artifact(
        approved_root, predecessor_layer["artifacts"]["manual_mask"], "authored stamp mask"
    )
    manual_values_path = _artifact(
        approved_root, predecessor_layer["artifacts"]["manual_values"], "authored stamp values"
    )
    active_manifest_path = _verified(
        Path(str(report["inputs"]["active_canonical_border"]["manifest"]["path"])).resolve(),
        report["inputs"]["active_canonical_border"]["manifest"]["sha256"],
        "active canonical manifest",
    )
    active = _load(active_manifest_path)
    active_overlay_record = active["artifacts"]["overlay"]
    active_overlay_path = _verified(
        active_manifest_path.parent / str(active_overlay_record["path"]),
        active_overlay_record["sha256"],
        "active canonical display border",
    )

    values = np.asarray(Image.open(class_path), dtype=np.uint8)
    valid = np.asarray(Image.open(valid_path)) > 0
    approved_values = np.asarray(Image.open(approved_class_path), dtype=np.uint8)
    seed_values = np.asarray(Image.open(seed_path), dtype=np.uint8)
    observed = np.asarray(Image.open(observed_path)) > 0
    manual = np.asarray(Image.open(manual_path)) > 0
    manual_values = np.asarray(Image.open(manual_values_path), dtype=np.uint8)
    neighbor_mask = np.asarray(Image.open(neighbor_mask_path)) > 0
    outside_existing = np.asarray(Image.open(outside_existing_path)) > 0
    shape = values.shape
    if not all(
        raster.shape == shape
        for raster in (
            valid,
            approved_values,
            seed_values,
            observed,
            manual,
            manual_values,
            neighbor_mask,
            outside_existing,
        )
    ):
        raise ValueError("Promotion evidence uses different grids")
    if np.any((values > 0) != valid):
        raise ValueError("Reviewed neighbor surface does not exactly fill its interior")
    if np.any(values[valid & (seed_values > 0)] != seed_values[valid & (seed_values > 0)]):
        raise ValueError("Reviewed neighbor surface changes retained seed evidence")

    conservative_mask = valid & (seed_values > 0) & (approved_values == 0)
    inference_mask = conservative_mask | neighbor_mask
    observed_output = observed & valid
    manual_output = manual & valid
    manual_values_output = manual_values.copy()
    manual_values_output[~valid] = 0
    output_dir.mkdir(parents=True, exist_ok=False)
    layer_dir = output_dir / layer_id
    layer_dir.mkdir(parents=True)

    artifacts: Dict[str, Dict[str, object]] = {}

    def save_layer(name: str, array: np.ndarray) -> None:
        path = layer_dir / name
        Image.fromarray(array).save(path, optimize=True)
        artifacts[name] = {
            "path": f"{layer_id}/{name}",
            "sha256": _sha256(path),
        }

    save_layer("web-mercator-class-id-final.png", values)
    shutil.copyfile(preview_path, layer_dir / "web-mercator-preview-final.png")
    artifacts["web-mercator-preview-final.png"] = {
        "path": f"{layer_id}/web-mercator-preview-final.png",
        "sha256": _sha256(layer_dir / "web-mercator-preview-final.png"),
    }
    save_layer("web-mercator-observed-retained-mask.png", (observed_output * 255).astype(np.uint8))
    save_layer("web-mercator-manual-override-mask.png", (manual_output * 255).astype(np.uint8))
    save_layer("web-mercator-manual-values.png", manual_values_output)
    save_layer("web-mercator-inference-retained-mask.png", (inference_mask * 255).astype(np.uint8))
    save_layer("web-mercator-conservative-completion-mask.png", (conservative_mask * 255).astype(np.uint8))
    save_layer("web-mercator-neighbor-assumption-mask.png", (neighbor_mask * 255).astype(np.uint8))
    shutil.copyfile(confidence_path, layer_dir / "web-mercator-neighbor-confidence.png")
    artifacts["web-mercator-neighbor-confidence.png"] = {
        "path": f"{layer_id}/web-mercator-neighbor-confidence.png",
        "sha256": _sha256(layer_dir / "web-mercator-neighbor-confidence.png"),
    }
    shutil.copyfile(distance_path, layer_dir / "web-mercator-nearest-seed-distance-x256.png")
    artifacts["web-mercator-nearest-seed-distance-x256.png"] = {
        "path": f"{layer_id}/web-mercator-nearest-seed-distance-x256.png",
        "sha256": _sha256(layer_dir / "web-mercator-nearest-seed-distance-x256.png"),
    }
    shutil.copyfile(outside_unknown_path, layer_dir / "web-mercator-unknown-outside-removed-mask.png")
    artifacts["web-mercator-unknown-outside-removed-mask.png"] = {
        "path": f"{layer_id}/web-mercator-unknown-outside-removed-mask.png",
        "sha256": _sha256(layer_dir / "web-mercator-unknown-outside-removed-mask.png"),
    }
    shutil.copyfile(outside_existing_path, layer_dir / "web-mercator-boundary-removed-mask.png")
    artifacts["web-mercator-boundary-removed-mask.png"] = {
        "path": f"{layer_id}/web-mercator-boundary-removed-mask.png",
        "sha256": _sha256(layer_dir / "web-mercator-boundary-removed-mask.png"),
    }

    publication_interior_path = output_dir / "publication-interior-mask.png"
    shutil.copyfile(valid_path, publication_interior_path)
    component_count, component_labels, component_stats, _ = cv2.connectedComponentsWithStats(
        valid.astype(np.uint8), 8
    )
    component_ids = sorted(
        range(1, component_count),
        key=lambda component: int(component_stats[component, cv2.CC_STAT_AREA]),
        reverse=True,
    )
    if len(component_ids) != 5:
        raise ValueError("Approved canonical plant boundary must contain mainland plus four islands")
    boundary_rgba = np.zeros((*shape, 4), dtype=np.uint8)
    contours, _ = cv2.findContours(
        valid.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    cv2.drawContours(boundary_rgba, contours, -1, (40, 255, 110, 255), 1)
    publication_border_path = output_dir / "publication-boundary-overlay.png"
    Image.fromarray(boundary_rgba).save(publication_border_path, optimize=True)
    components = []
    for index, component in enumerate(component_ids):
        selected = component_labels == component
        components.append(
            {
                "id": "mainland" if index == 0 else f"canonical-island-{index:02d}",
                "role": (
                    "canonical_mainland_clipping"
                    if index == 0
                    else "author_approved_canonical_island"
                ),
                "authority": (
                    "california-mainland-hybrid-v1 clipping interior"
                    if index == 0
                    else active["canonical_boundary_id"]
                ),
                "interior_pixel_count": int(np.count_nonzero(selected)),
                "observed_source_pixel_count": int(np.count_nonzero(selected & observed_output)),
                "neighbor_assumption_pixel_count": int(np.count_nonzero(selected & neighbor_mask)),
            }
        )
    boundary_audit = {
        "schema_version": 1,
        "status": "pass",
        "method": "separate_canonical_clipping_interior_plus_active_county_detail_islands",
        "boundary": {
            "connected_component_count": len(component_ids),
            "expected_component_count": 5,
            "mainland_interior_pixel_count": int(components[0]["interior_pixel_count"]),
            "publication_interior_pixel_count": int(np.count_nonzero(valid)),
            "selection_policy": {
                "mainland": "separately_versioned canonical clipping interior",
                "islands": "four author-approved active canonical outlines",
                "display_border_reconstructed_from_fill": False,
            },
            "components": components,
            "interior": {
                "path": publication_interior_path.name,
                "sha256": _sha256(publication_interior_path),
            },
            "integrity_border": {
                "path": publication_border_path.name,
                "sha256": _sha256(publication_border_path),
            },
            "canonical_display_border": {
                "path": str(active_overlay_path),
                "sha256": _sha256(active_overlay_path),
                "canonical_boundary_id": active["canonical_boundary_id"],
                "grid": active["source_grid"],
            },
        },
        "layers": [
            {
                "layer_id": layer_id,
                "passed": True,
                "colored_pixel_count_outside_boundary": 0,
                "unclassified_pixel_count_inside_boundary": 0,
            }
        ],
    }
    boundary_audit_path = output_dir / "boundary-clip-audit.json"
    boundary_audit_path.write_text(json.dumps(boundary_audit, indent=2) + "\n")

    class_counts = {
        str(class_id): int(np.count_nonzero(values == class_id))
        for class_id in range(1, len(categories) + 1)
    }
    mapped_artifacts = {
        "class_id": artifacts["web-mercator-class-id-final.png"],
        "preview": artifacts["web-mercator-preview-final.png"],
        "observed_mask": artifacts["web-mercator-observed-retained-mask.png"],
        "inference_mask": artifacts["web-mercator-inference-retained-mask.png"],
        "manual_mask": artifacts["web-mercator-manual-override-mask.png"],
        "manual_values": artifacts["web-mercator-manual-values.png"],
        "conservative_completion_mask": artifacts[
            "web-mercator-conservative-completion-mask.png"
        ],
        "neighbor_assumption_mask": artifacts[
            "web-mercator-neighbor-assumption-mask.png"
        ],
        "neighbor_confidence": artifacts["web-mercator-neighbor-confidence.png"],
        "nearest_seed_distance": artifacts[
            "web-mercator-nearest-seed-distance-x256.png"
        ],
        "outside_unknown_removed_mask": artifacts[
            "web-mercator-unknown-outside-removed-mask.png"
        ],
        "boundary_removed_mask": artifacts["web-mercator-boundary-removed-mask.png"],
    }
    materialization = {
        "schema_version": 1,
        "status": "needs_visual_review",
        "dataset_id": "california-plant-hardiness-zones-neighbor-v1",
        "source_run": str(source_run),
        "extraction_manifest_sha256": _sha256(extraction_path),
        "neighbor_completion": {
            "path": str(neighbor_report_path),
            "sha256": _sha256(neighbor_report_path),
            "method": report["policy"]["inside_remaining_unknown"],
        },
        "boundary_clip": {
            "audit": {
                "path": str(boundary_audit_path),
                "sha256": _sha256(boundary_audit_path),
            },
            "continuous_border_sha256": _sha256(active_overlay_path),
            "mainland_interior_sha256": report["inputs"]["clipping_interior"]["sha256"],
            "publication_interior_sha256": _sha256(publication_interior_path),
            "colored_pixel_count_outside_boundary": 0,
            "unclassified_pixel_count_inside_boundary": 0,
            "canonical_border": {
                "canonical_boundary_id": active["canonical_boundary_id"],
                "manifest_sha256": _sha256(active_manifest_path),
                "display_overlay_sha256": _sha256(active_overlay_path),
            },
        },
        "precedence": [
            "observed_classification",
            "authored_stamp_values",
            "conservative_local_completion",
            "neighbor_assumption",
            "canonical_boundary_clip",
        ],
        "warning": (
            "This unpublished candidate fills all retained interior unknowns from "
            "neighbor evidence. Every assumption and every exterior removal remains "
            "separately masked."
        ),
        "layers": [
            {
                "layer_id": layer_id,
                "width": int(shape[1]),
                "height": int(shape[0]),
                "stamp_operation_count": int(predecessor_layer.get("stamp_operation_count", 0)),
                "observed_retained_pixel_count": int(np.count_nonzero(observed_output)),
                "manual_override_pixel_count": int(np.count_nonzero(manual_output)),
                "manual_nonzero_pixel_count": int(np.count_nonzero(manual_values_output)),
                "conservative_completion_pixel_count": int(np.count_nonzero(conservative_mask)),
                "neighbor_assumption_pixel_count": int(np.count_nonzero(neighbor_mask)),
                "unknown_outside_removed_pixel_count": int(
                    report["metrics"]["unknown_outside_removed"]
                ),
                "boundary_removed_pixel_count": int(np.count_nonzero(outside_existing)),
                "final_classified_pixel_count": int(np.count_nonzero(values)),
                "final_pixels_by_class_id": class_counts,
                "unclassified_pixel_count_after": 0,
                "colored_pixel_count_outside_boundary": 0,
                "artifacts": mapped_artifacts,
            }
        ],
    }
    materialization_path = output_dir / "materialization.json"
    materialization_path.write_text(json.dumps(materialization, indent=2) + "\n")

    alignment_path = Path(str(plan["alignment"]))
    if not alignment_path.is_absolute():
        alignment_path = Path.cwd() / alignment_path
    _verified(alignment_path, None, "accepted alignment")
    decision = {
        "schema_version": 1,
        "scope": "neighbor_completed_categorical_candidate",
        "status": "approved",
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "materialization_sha256": _sha256(materialization_path),
        "materialization_path": str(materialization_path),
        "alignment_sha256": _sha256(alignment_path),
        "canonical_boundary_manifest_sha256": _sha256(active_manifest_path),
        "canonical_display_border_sha256": _sha256(active_overlay_path),
        "boundary_clip_audit_sha256": _sha256(boundary_audit_path),
        "hybrid_border_sha256": _sha256(active_overlay_path),
        "neighbor_completion_sha256": _sha256(neighbor_report_path),
        "review_session_sha256": _sha256(review_session_path),
        "predecessor_materialization_sha256": _sha256(approved_manifest),
        "author_statement": statement,
        "inspection_confirmed": True,
        "approval_carried_forward": False,
        "evidence": {
            "interior_assumption_pixel_count": int(
                report["metrics"]["unknown_inside_filled"]
            ),
            "exterior_unknown_pixel_count": int(
                report["metrics"]["unknown_outside_removed"]
            ),
            "exterior_prior_evidence_pixel_count": int(np.count_nonzero(outside_existing)),
            "authored_stamp_neighbor_pixel_count": int(
                report["metrics"]["neighbor_fill"]["manual_neighbor_pixel_count"]
            ),
            "authored_stamp_weight_changed_choice_pixel_count": int(
                report["metrics"]["neighbor_fill"][
                    "manual_weight_changed_choice_pixel_count"
                ]
            ),
            "final_classified_pixel_count": int(np.count_nonzero(values)),
            "unclassified_pixel_count_inside_boundary": 0,
            "classified_pixel_count_outside_boundary": 0,
            "boundary_component_count": len(component_ids),
        },
    }
    decision_path = output_dir / "materialization-review-decision.json"
    decision_path.write_text(json.dumps(decision, indent=2) + "\n")
    return {
        "status": "approved",
        "materialization": {
            "path": str(materialization_path),
            "sha256": _sha256(materialization_path),
        },
        "decision": {
            "path": str(decision_path),
            "sha256": _sha256(decision_path),
        },
        "boundary_audit": {
            "path": str(boundary_audit_path),
            "sha256": _sha256(boundary_audit_path),
        },
    }
