"""Recover legend classes hidden by a source map's state-perimeter stroke.

This is deliberately an extraction repair, not an alignment adjustment.  A
categorical map can be geographically registered while still showing an
artificial transparent setback from the perimeter: the dark cartographic outline
is not a legend class, so ordinary color classification drops it.  At target
resolution that one- or two-pixel source stroke can become a conspicuous gap.

The repair is narrowly gated.  It only copies the nearest already-extracted
class into neutral/dark source pixels that are supported by both the
source-derived perimeter and the independently projected pinned Mapbox
perimeter, lie inside the Mapbox land mask, and remain within a small source
pixel distance of existing thematic evidence.  It never grows data from the
target perimeter alone and never uses a prior public raster as source evidence.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

from .automatic_alignment_loop import (
    _source_semantic_evidence,
    load_pinned_mapbox_reference,
)
from .automatic_categorical_extraction import (
    _reference_to_source_remap,
    _save_ids,
    _save_mask,
    _source_data_mask,
    _source_to_reference_remap,
    _source_to_reference,
    _supersample_alignment_transform,
)


SCHEMA_VERSION = "mapscan.automatic-categorical-extraction.v1"
REPAIR_SCHEMA_VERSION = "mapscan.state-perimeter-cartographic-occlusion-repair.v1"


@dataclass(frozen=True)
class PerimeterOcclusionConfig:
    """Conservative source-pixel limits for state-perimeter completion."""

    reference_perimeter_tolerance_px: int = 3
    source_perimeter_neighborhood_px: int = 2
    maximum_completion_distance_px: float = 3.5
    maximum_neutral_channel_spread: int = 45
    maximum_neutral_mean: float = 190.0
    maximum_repaired_fraction_of_existing_data: float = 0.02
    target_perimeter_audit_distance_px: float = 8.0
    maximum_added_target_distance_from_perimeter_px: float = 32.0
    minimum_target_perimeter_coverage_gain: float = 0.01
    minimum_target_coast_coverage_gain: float = 0.01
    minimum_target_inland_border_coverage_gain: float = 0.01

    def __post_init__(self) -> None:
        if self.reference_perimeter_tolerance_px < 0:
            raise ValueError("reference_perimeter_tolerance_px cannot be negative")
        if self.source_perimeter_neighborhood_px < 0:
            raise ValueError("source_perimeter_neighborhood_px cannot be negative")
        if self.maximum_completion_distance_px <= 0:
            raise ValueError("maximum_completion_distance_px must be positive")
        if not 0 <= self.maximum_neutral_channel_spread <= 255:
            raise ValueError("maximum_neutral_channel_spread must be 0..255")
        if not 0 <= self.maximum_neutral_mean <= 255:
            raise ValueError("maximum_neutral_mean must be 0..255")
        if not 0 < self.maximum_repaired_fraction_of_existing_data < 1:
            raise ValueError("maximum repaired fraction must be between zero and one")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "sha256": _sha256(path),
        "byte_count": path.stat().st_size,
    }


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _bound_path(record: object, base: Path, label: str) -> Path:
    if not isinstance(record, Mapping):
        raise ValueError(f"missing {label} record")
    raw = str(record.get("path", ""))
    digest = str(record.get("sha256", ""))
    if not raw or len(digest) != 64:
        raise ValueError(f"invalid {label} record")
    path = Path(raw)
    candidates = [path.resolve()] if path.is_absolute() else [
        (base / path).resolve(),
        (Path.cwd() / path).resolve(),
    ]
    for candidate in candidates:
        if candidate.is_file() and _sha256(candidate) == digest:
            return candidate
    raise ValueError(f"{label} does not resolve to its declared hash")


def _reported_artifact(report: Mapping[str, Any], root: Path, suffix: str) -> Path:
    matches = [
        item
        for item in report.get("artifacts", [])
        if isinstance(item, Mapping) and str(item.get("path", "")).endswith(suffix)
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one declared {suffix} artifact")
    return _bound_path(matches[0], root, suffix)


def _processing_transform(
    pointer: Mapping[str, Any],
    transform: Mapping[str, Any],
    reference_grid: Mapping[str, Any],
) -> dict[str, Any]:
    """Lift an accepted alignment onto the extraction's declared target grid.

    High-resolution categorical extractions retain the original accepted
    alignment but evaluate it on a corner-preserving supersampled Mapbox grid.
    A perimeter audit must perform the same conjugation before remapping; using
    the base transform would silently compare arrays at incompatible scales.
    """

    factor = int(pointer.get("target_supersampling", 1))
    lifted = _supersample_alignment_transform(transform, factor)
    lifted_grid = lifted.get("target_grid")
    declared_grid = pointer.get("processing_target_grid", lifted_grid)
    for label, grid in (
        ("lifted alignment", lifted_grid),
        ("accepted extraction", declared_grid),
        ("processing Mapbox reference", reference_grid),
    ):
        if not isinstance(grid, Mapping):
            raise ValueError(f"{label} has no target grid")
    for grid in (declared_grid, reference_grid):
        if int(grid["width"]) != int(lifted_grid["width"]) or int(
            grid["height"]
        ) != int(lifted_grid["height"]):
            raise ValueError(
                "Perimeter repair reference does not match the extraction's "
                "supersampled target grid"
            )
        if not np.allclose(
            np.asarray(grid["bounds"], dtype=np.float64),
            np.asarray(lifted_grid["bounds"], dtype=np.float64),
            rtol=0.0,
            atol=1e-6,
        ):
            raise ValueError(
                "Perimeter repair reference bounds do not match the extraction grid"
            )
    return lifted


def recover_perimeter_cartographic_occlusions(
    source_rgb: np.ndarray,
    source_class_ids: np.ndarray,
    source_land: np.ndarray,
    source_state_perimeter: np.ndarray,
    source_reference_perimeter: np.ndarray,
    *,
    config: PerimeterOcclusionConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fill only mutually supported perimeter ink beside a legend class."""

    config = config or PerimeterOcclusionConfig()
    shape = source_class_ids.shape
    if source_rgb.shape != (*shape, 3):
        raise ValueError("source RGB and class-id shapes differ")
    for name, values in (
        ("source land", source_land),
        ("source state perimeter", source_state_perimeter),
        ("source reference perimeter", source_reference_perimeter),
    ):
        if values.shape != shape:
            raise ValueError(f"{name} shape differs from source class ids")

    classified = source_class_ids > 0
    if not np.any(classified):
        raise ValueError("source class raster has no thematic evidence")
    reference_size = config.reference_perimeter_tolerance_px * 2 + 1
    near_reference_perimeter = cv2.dilate(
        source_reference_perimeter.astype(np.uint8),
        np.ones((reference_size, reference_size), np.uint8),
    ) > 0
    supported_source_perimeter = source_state_perimeter & near_reference_perimeter
    neighborhood_size = config.source_perimeter_neighborhood_px * 2 + 1
    perimeter_neighborhood = cv2.dilate(
        supported_source_perimeter.astype(np.uint8),
        np.ones((neighborhood_size, neighborhood_size), np.uint8),
    ) > 0

    channels = source_rgb.astype(np.int16)
    channel_spread = channels.max(axis=2) - channels.min(axis=2)
    channel_mean = channels.mean(axis=2)
    neutral_dark = (
        (channel_spread <= config.maximum_neutral_channel_spread)
        & (channel_mean <= config.maximum_neutral_mean)
    )
    distance, nearest = distance_transform_edt(
        ~classified, return_distances=True, return_indices=True
    )
    repair_mask = (
        source_land
        & ~classified
        & perimeter_neighborhood
        & neutral_dark
        & (distance <= config.maximum_completion_distance_px)
    )
    repaired = source_class_ids.copy()
    repaired[repair_mask] = source_class_ids[
        nearest[0][repair_mask], nearest[1][repair_mask]
    ]
    if np.any(repaired[repair_mask] == 0):
        raise AssertionError("coastal repair produced an unclassified repair pixel")
    if np.any(repaired[~repair_mask] != source_class_ids[~repair_mask]):
        raise AssertionError("coastal repair changed pixels outside its mask")

    class_counts = {
        str(class_id): int(np.count_nonzero(repair_mask & (repaired == class_id)))
        for class_id in sorted(int(value) for value in np.unique(repaired[repair_mask]))
    }
    diagnostics = {
        "kind": "source_state_perimeter_cartographic_occlusion_completion",
        "source_state_perimeter_pixel_count": int(
            np.count_nonzero(source_state_perimeter)
        ),
        "source_reference_perimeter_pixel_count": int(
            np.count_nonzero(source_reference_perimeter)
        ),
        "mutually_supported_source_perimeter_pixel_count": int(
            np.count_nonzero(supported_source_perimeter)
        ),
        "perimeter_neighborhood_pixel_count": int(
            np.count_nonzero(perimeter_neighborhood)
        ),
        "source_repaired_pixel_count": int(np.count_nonzero(repair_mask)),
        "source_existing_classified_pixel_count": int(np.count_nonzero(classified)),
        "source_repaired_fraction_of_existing_data": float(
            np.count_nonzero(repair_mask) / max(np.count_nonzero(classified), 1)
        ),
        "repaired_class_counts": class_counts,
        "constraints": {
            "inside_independently_projected_mapbox_land": True,
            "source_perimeter_derived_without_class_raster": True,
            "pinned_mapbox_perimeter_projected_independently": True,
            "source_and_mapbox_perimeters_both_required": True,
            "neutral_dark_source_pixels_only": True,
            "nearest_existing_source_class_only": True,
            "maximum_completion_distance_px": config.maximum_completion_distance_px,
            "maximum_neutral_channel_spread": config.maximum_neutral_channel_spread,
            "maximum_neutral_mean": config.maximum_neutral_mean,
        },
    }
    return repaired, repair_mask, diagnostics


def _perimeter_coverage(
    class_ids: np.ndarray,
    target_land: np.ndarray,
    target_perimeter: np.ndarray,
    distance_px: float,
) -> tuple[float, int]:
    distance = cv2.distanceTransform(
        (~target_perimeter).astype(np.uint8), cv2.DIST_L2, 5
    )
    ring = target_land & (distance <= distance_px)
    count = int(np.count_nonzero(ring))
    return float(np.count_nonzero((class_ids > 0) & ring) / max(count, 1)), count


def _render(class_ids: np.ndarray, legend: Mapping[str, Any]) -> np.ndarray:
    palette = np.zeros((256, 3), np.uint8)
    for entry in legend.get("entries", []):
        palette[int(entry["class_id"])] = np.asarray(entry["rgb"], np.uint8)
    return palette[class_ids]


def run_perimeter_occlusion_repair(
    base_extraction: Path,
    reference_manifest: Path,
    output_dir: Path,
    *,
    config: PerimeterOcclusionConfig | None = None,
) -> Path:
    """Create an immutable, hash-bound extraction superseding a base result."""

    config = config or PerimeterOcclusionConfig()
    base_extraction = base_extraction.resolve()
    if base_extraction.is_dir():
        pointer_path = base_extraction / "accepted-extraction.json"
        base_root = base_extraction
    else:
        pointer_path = base_extraction
        base_root = base_extraction.parent
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"output exists: {output_dir}")

    pointer = _json(pointer_path)
    if pointer.get("schema_version") != SCHEMA_VERSION or pointer.get("status") != "accepted":
        raise ValueError("base extraction is not an accepted categorical extraction")
    source_path = _bound_path(pointer.get("source"), base_root, "source")
    alignment_path = _bound_path(pointer.get("alignment"), base_root, "alignment")
    legend_path = _bound_path(pointer.get("legend"), base_root, "legend")
    alignment = _json(alignment_path)
    legend = _json(legend_path)
    base_iteration_name = str(pointer.get("accepted_iteration", ""))
    base_iteration_path = base_root / base_iteration_name / "iteration.json"
    base_iteration = _json(base_iteration_path)
    source_ids_path = _reported_artifact(
        base_iteration, base_root, f"{base_iteration_name}/source-class-id.png"
    )
    source_observed_path = _reported_artifact(
        base_iteration, base_root, f"{base_iteration_name}/source-observed-mask.png"
    )
    source_inferred_path = _reported_artifact(
        base_iteration, base_root, f"{base_iteration_name}/source-inferred-mask.png"
    )
    target_ids_path = _reported_artifact(
        base_iteration, base_root, f"{base_iteration_name}/web-mercator-class-id.png"
    )

    reference = load_pinned_mapbox_reference(reference_manifest.resolve())
    for key in ("style_sha256", "tilejson_sha256", "tile_aggregate_sha256"):
        if alignment.get("mapbox_reference", {}).get(key) != reference.pin[key]:
            raise ValueError("alignment and perimeter repair use different Mapbox bytes")
    accepted_transform = alignment.get("transform")
    if not isinstance(accepted_transform, Mapping):
        raise ValueError("alignment has no transform")
    transform = _processing_transform(pointer, accepted_transform, reference.grid)
    with Image.open(source_path) as source_image:
        source_rgb = np.asarray(source_image.convert("RGB"))
    source_ids = np.asarray(Image.open(source_ids_path))
    source_observed = np.asarray(Image.open(source_observed_path).convert("L")) > 0
    source_inferred = np.asarray(Image.open(source_inferred_path).convert("L")) > 0
    base_target_ids = np.asarray(Image.open(target_ids_path))
    if source_ids.shape != source_rgb.shape[:2]:
        raise ValueError("base source class raster and source dimensions differ")

    source_land = _source_data_mask(
        reference.state_land,
        reference.water,
        transform,
        source_ids.shape,
    )
    semantic = _source_semantic_evidence(source_rgb)
    source_map_x, source_map_y = _source_to_reference_remap(
        transform, source_ids.shape
    )
    source_reference_perimeter = cv2.remap(
        reference.state_coast.astype(np.uint8),
        source_map_x,
        source_map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ) > 0
    first = recover_perimeter_cartographic_occlusions(
        source_rgb,
        source_ids,
        source_land,
        semantic.state_coast,
        source_reference_perimeter,
        config=config,
    )
    second = recover_perimeter_cartographic_occlusions(
        source_rgb,
        source_ids,
        source_land,
        semantic.state_coast,
        source_reference_perimeter,
        config=config,
    )
    repaired_ids, repair_mask, repair_diagnostics = second
    deterministic = all(np.array_equal(left, right) for left, right in zip(first[:2], second[:2]))

    remap = _reference_to_source_remap(transform)
    target_ids = _source_to_reference(
        repaired_ids, transform, cv2.INTER_NEAREST, 0, remap
    )
    target_observed = _source_to_reference(
        source_observed.astype(np.uint8), transform, cv2.INTER_NEAREST, 0, remap
    ) > 0
    target_repair = _source_to_reference(
        repair_mask.astype(np.uint8), transform, cv2.INTER_NEAREST, 0, remap
    ) > 0
    target_land = reference.state_land & ~reference.water
    raw_outside_count = int(np.count_nonzero((target_ids > 0) & ~target_land))
    target_ids[~target_land] = 0
    target_observed &= target_land & (target_ids > 0)
    target_repair &= target_land & (target_ids > 0)
    target_inferred = (target_ids > 0) & ~target_observed

    removed_target = (base_target_ids > 0) & (target_ids == 0)
    added_target = (base_target_ids == 0) & (target_ids > 0)
    changed_existing_class = (
        (base_target_ids > 0)
        & (target_ids > 0)
        & (base_target_ids != target_ids)
    )
    target_perimeter = reference.state_coast
    target_coast = target_perimeter & (
        cv2.dilate(reference.water.astype(np.uint8), np.ones((17, 17), np.uint8))
        > 0
    )
    target_inland_border = target_perimeter & ~target_coast
    perimeter_distance = cv2.distanceTransform(
        (~target_perimeter).astype(np.uint8), cv2.DIST_L2, 5
    )
    base_coverage, perimeter_ring_count = _perimeter_coverage(
        base_target_ids,
        target_land,
        target_perimeter,
        config.target_perimeter_audit_distance_px,
    )
    repaired_coverage, _ = _perimeter_coverage(
        target_ids,
        target_land,
        target_perimeter,
        config.target_perimeter_audit_distance_px,
    )
    base_coast_coverage, coast_ring_count = _perimeter_coverage(
        base_target_ids,
        target_land,
        target_coast,
        config.target_perimeter_audit_distance_px,
    )
    repaired_coast_coverage, _ = _perimeter_coverage(
        target_ids,
        target_land,
        target_coast,
        config.target_perimeter_audit_distance_px,
    )
    base_inland_coverage, inland_ring_count = _perimeter_coverage(
        base_target_ids,
        target_land,
        target_inland_border,
        config.target_perimeter_audit_distance_px,
    )
    repaired_inland_coverage, _ = _perimeter_coverage(
        target_ids,
        target_land,
        target_inland_border,
        config.target_perimeter_audit_distance_px,
    )
    maximum_added_distance = float(
        np.max(perimeter_distance[added_target]) if np.any(added_target) else 0.0
    )
    repaired_fraction = float(
        repair_diagnostics["source_repaired_fraction_of_existing_data"]
    )
    gates: dict[str, Any] = {
        "base_source_classes_preserved": {
            "passed": bool(np.all(repaired_ids[source_ids > 0] == source_ids[source_ids > 0])),
            "removed_source_pixel_count": int(
                np.count_nonzero((source_ids > 0) & (repaired_ids == 0))
            ),
        },
        "repair_is_small_relative_to_source_evidence": {
            "passed": repaired_fraction
            <= config.maximum_repaired_fraction_of_existing_data,
            "value": repaired_fraction,
            "maximum": config.maximum_repaired_fraction_of_existing_data,
        },
        "target_base_classes_not_removed": {
            "passed": not bool(np.any(removed_target)),
            "removed_target_pixel_count": int(np.count_nonzero(removed_target)),
        },
        "target_existing_classes_not_reclassified": {
            "passed": not bool(np.any(changed_existing_class)),
            "changed_target_pixel_count": int(np.count_nonzero(changed_existing_class)),
        },
        "target_water_and_exterior_empty": {
            "passed": not bool(np.any(target_ids[~target_land] > 0)),
            "raw_preclip_pixel_count": raw_outside_count,
            "postclip_pixel_count": int(np.count_nonzero(target_ids[~target_land] > 0)),
        },
        "added_target_pixels_remain_on_state_perimeter": {
            "passed": maximum_added_distance
            <= config.maximum_added_target_distance_from_perimeter_px,
            "value": maximum_added_distance,
            "maximum": config.maximum_added_target_distance_from_perimeter_px,
        },
        "near_perimeter_coverage_improves": {
            "passed": repaired_coverage - base_coverage
            >= config.minimum_target_perimeter_coverage_gain,
            "before": base_coverage,
            "after": repaired_coverage,
            "gain": repaired_coverage - base_coverage,
            "minimum_gain": config.minimum_target_perimeter_coverage_gain,
            "distance_px": config.target_perimeter_audit_distance_px,
            "land_ring_pixel_count": perimeter_ring_count,
        },
        "near_coast_coverage_improves": {
            "passed": repaired_coast_coverage - base_coast_coverage
            >= config.minimum_target_coast_coverage_gain,
            "before": base_coast_coverage,
            "after": repaired_coast_coverage,
            "gain": repaired_coast_coverage - base_coast_coverage,
            "minimum_gain": config.minimum_target_coast_coverage_gain,
            "distance_px": config.target_perimeter_audit_distance_px,
            "land_ring_pixel_count": coast_ring_count,
        },
        "near_inland_border_coverage_improves": {
            "passed": repaired_inland_coverage - base_inland_coverage
            >= config.minimum_target_inland_border_coverage_gain,
            "before": base_inland_coverage,
            "after": repaired_inland_coverage,
            "gain": repaired_inland_coverage - base_inland_coverage,
            "minimum_gain": config.minimum_target_inland_border_coverage_gain,
            "distance_px": config.target_perimeter_audit_distance_px,
            "land_ring_pixel_count": inland_ring_count,
        },
        "all_legend_classes_preserved": {
            "passed": set(int(value) for value in np.unique(source_ids) if value)
            == set(int(value) for value in np.unique(repaired_ids) if value),
        },
        "deterministic_source_repair": {"passed": deterministic},
        "manual_input_used": {"passed": True, "value": False},
    }
    passed = all(bool(value["passed"]) for value in gates.values())
    if not passed:
        failures = [name for name, value in gates.items() if not value["passed"]]
        raise ValueError(f"perimeter occlusion repair failed gates: {failures}")

    output_dir.mkdir(parents=True)
    shutil.copytree(legend_path.parent, output_dir / "legend")
    iteration_number = int(pointer["automatic_iteration_count"]) + 1
    iteration_name = f"extraction-{iteration_number:02d}"
    iteration_dir = output_dir / iteration_name
    iteration_dir.mkdir()
    paths = {
        "source_ids": iteration_dir / "source-class-id.png",
        "source_observed": iteration_dir / "source-observed-mask.png",
        "source_inferred": iteration_dir / "source-inferred-mask.png",
        "source_repair": iteration_dir / "source-perimeter-occlusion-mask.png",
        "source_reference_perimeter": iteration_dir
        / "source-projected-mapbox-perimeter-mask.png",
        "source_reconstruction": iteration_dir / "source-reconstruction.png",
        "target_ids": iteration_dir / "web-mercator-class-id.png",
        "target_observed": iteration_dir / "web-mercator-observed-mask.png",
        "target_inferred": iteration_dir / "web-mercator-inferred-mask.png",
        "target_repair": iteration_dir / "web-mercator-perimeter-occlusion-mask.png",
        "target_reconstruction": iteration_dir / "web-mercator-reconstruction.png",
    }
    _save_ids(paths["source_ids"], repaired_ids)
    _save_mask(paths["source_observed"], source_observed)
    _save_mask(paths["source_inferred"], source_inferred | repair_mask)
    _save_mask(paths["source_repair"], repair_mask)
    _save_mask(paths["source_reference_perimeter"], source_reference_perimeter)
    Image.fromarray(_render(repaired_ids, legend)).save(
        paths["source_reconstruction"], optimize=True
    )
    _save_ids(paths["target_ids"], target_ids)
    _save_mask(paths["target_observed"], target_observed)
    _save_mask(paths["target_inferred"], target_inferred)
    _save_mask(paths["target_repair"], target_repair)
    Image.fromarray(_render(target_ids, legend)).save(
        paths["target_reconstruction"], optimize=True
    )

    repair_preview = source_rgb.copy()
    repair_preview[repair_mask] = (255, 0, 210)
    comparison_path = iteration_dir / "source-perimeter-occlusion-comparison.png"
    Image.fromarray(
        np.concatenate(
            (source_rgb, _render(source_ids, legend), _render(repaired_ids, legend), repair_preview),
            axis=1,
        )
    ).save(comparison_path, optimize=True)
    added_preview = np.zeros((*target_ids.shape, 3), np.uint8)
    added_preview[target_ids > 0] = _render(target_ids, legend)[target_ids > 0]
    added_preview[added_target] = (255, 0, 210)
    target_comparison_path = iteration_dir / "target-perimeter-additions.png"
    Image.fromarray(added_preview).save(target_comparison_path, optimize=True)

    repair_report_path = output_dir / "perimeter-occlusion-repair.json"
    repair_report = {
        "schema_version": REPAIR_SCHEMA_VERSION,
        "status": "pass",
        "source": {"path": str(source_path), "sha256": _sha256(source_path)},
        "base_extraction": {"path": str(pointer_path), "sha256": _sha256(pointer_path)},
        "alignment": {"path": str(alignment_path), "sha256": _sha256(alignment_path)},
        "mapbox_reference": {
            "path": str(reference.manifest_path),
            "manifest_sha256": reference.pin["manifest_sha256"],
        },
        "method": repair_diagnostics,
        "target": {
            "added_pixel_count": int(np.count_nonzero(added_target)),
            "removed_pixel_count": int(np.count_nonzero(removed_target)),
            "changed_existing_class_pixel_count": int(
                np.count_nonzero(changed_existing_class)
            ),
            "maximum_added_distance_from_mapbox_perimeter_px": maximum_added_distance,
            "perimeter_ring_coverage_before": base_coverage,
            "perimeter_ring_coverage_after": repaired_coverage,
            "coast_ring_coverage_before": base_coast_coverage,
            "coast_ring_coverage_after": repaired_coast_coverage,
            "inland_border_ring_coverage_before": base_inland_coverage,
            "inland_border_ring_coverage_after": repaired_inland_coverage,
        },
        "gates": gates,
        "authority": {
            "source_image_used": True,
            "source_legend_classes_used": True,
            "source_derived_state_perimeter_used": True,
            "independently_projected_mapbox_perimeter_used": True,
            "mapbox_land_used_for_scope_and_clip": True,
            "prior_public_raster_used_as_source_evidence": False,
            "manual_input_used": False,
            "base_artifacts_mutated": False,
        },
    }
    repair_report_path.write_text(json.dumps(repair_report, indent=2) + "\n")

    artifacts = [_artifact(path, output_dir) for path in paths.values()]
    artifacts.extend(
        _artifact(path, output_dir)
        for path in (comparison_path, target_comparison_path, repair_report_path)
    )
    iteration_report_path = iteration_dir / "iteration.json"
    iteration_report = {
        "schema_version": SCHEMA_VERSION,
        "iteration": iteration_number,
        "decision": "accept",
        "legend_sha256": _sha256(output_dir / "legend" / legend_path.name),
        "scores": {
            "processing_target_grid": dict(transform["target_grid"]),
            "source_domain_pixel_count": int(np.count_nonzero(repaired_ids)),
            "source_observed_pixel_count": int(np.count_nonzero(source_observed)),
            "source_inferred_pixel_count": int(
                np.count_nonzero((source_inferred | repair_mask) & (repaired_ids > 0))
            ),
            "perimeter_occlusion_repair": repair_report["target"] | repair_diagnostics,
        },
        "gates": gates,
        "artifacts": artifacts,
    }
    iteration_report_path.write_text(json.dumps(iteration_report, indent=2) + "\n")

    accepted_path = output_dir / "accepted-extraction.json"
    accepted = {
        "schema_version": SCHEMA_VERSION,
        "status": "accepted",
        "automatic_iteration_count": iteration_number,
        "source": {"path": str(source_path), "sha256": _sha256(source_path)},
        "alignment": {"path": str(alignment_path), "sha256": _sha256(alignment_path)},
        "legend": {
            "path": f"legend/{legend_path.name}",
            "sha256": _sha256(output_dir / "legend" / legend_path.name),
        },
        "accepted_iteration": iteration_name,
        "aligned_source_extent": pointer.get("aligned_source_extent"),
        "processing_target_grid": dict(transform["target_grid"]),
        "target_supersampling": int(pointer.get("target_supersampling", 1)),
        "perimeter_occlusion_repair": {
            "path": repair_report_path.name,
            "sha256": _sha256(repair_report_path),
        },
    }
    accepted_path.write_text(json.dumps(accepted, indent=2) + "\n")
    return accepted_path


# Backward-compatible programmatic name for the v4 coastline-only entry point.
# New runs use full state-perimeter support through the implementation above.
CoastalOcclusionConfig = PerimeterOcclusionConfig
run_coastal_occlusion_repair = run_perimeter_occlusion_repair
