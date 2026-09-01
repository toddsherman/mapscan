"""Storage-bounded 3x replay of the source-native geologic extraction.

The accepted geologic registration is defined over the original 7088 by 9375
Poppler rendering of the authoritative PDF.  This module preserves that source
pixel space and only densifies the pinned Web-Mercator target grid.  It never
reads an earlier class raster.

The first pass is retained as compact hashes and metrics.  A second, independent
pass must reproduce those hashes before the accepted class id and evidence masks
are written.  Source comparisons are performed in 6 by 6 geographic cells after
sampling the 3x result back into original source pixels; only the six worst
native-resolution chips are retained.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt

from .automatic_alignment_loop import load_pinned_mapbox_reference
from .experiment_log import NoHumanExperimentLog, automatic_provenance
from .geologic_pdf_extraction import (
    GeologicLegendClass,
    GeologicPdfExtractionConfig,
    _legend_payload,
    _read_native_vector_evidence,
    _rasterize_native_fills,
    _source_pixels_to_reference_chunk,
    _warp_target_to_source_chunked,
)
from .resolution_fidelity_audit import (
    plan_native_regional_diffs,
    reference_to_source_points,
)
from .source_working_raster import load_working_raster_artifact


SCHEMA_VERSION = "mapscan.geologic-pdf-highres-extraction.v1"
PRODUCER = "mapscan.geologic_pdf_highres"
TARGET_FACTOR = 3
TARGET_WIDTH = 10192
TARGET_HEIGHT = 11758
REGIONAL_ROWS = 6
REGIONAL_COLUMNS = 6
RETAINED_WORST_CHIPS = 6


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    digest = hashlib.sha256()
    view = np.ascontiguousarray(values).view(np.uint8).reshape(-1)
    for offset in range(0, view.size, 8 * 1024 * 1024):
        digest.update(view[offset : offset + 8 * 1024 * 1024])
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _gate_passed(value: Any) -> bool:
    return bool(value if isinstance(value, bool) else value["passed"])


def _artifact(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "sha256": _sha256(path),
        "byte_count": path.stat().st_size,
    }


def _save_ids(path: Path, values: np.ndarray) -> None:
    Image.fromarray(values.astype(np.uint8), mode="L").save(path, optimize=True)


def _save_mask(path: Path, values: np.ndarray) -> None:
    Image.fromarray(values.astype(np.uint8) * 255, mode="L").save(
        path, optimize=True
    )


def _corner_preserving_dimension(base: int, factor: int) -> int:
    return (int(base) - 1) * int(factor) + 1


def _highres_transform(
    alignment: Mapping[str, Any], processing_grid: Mapping[str, Any]
) -> dict[str, Any]:
    """Return the accepted transform on a corner-preserving denser grid."""

    transform = copy.deepcopy(dict(alignment["transform"]))
    base_grid = transform["target_grid"]
    if str(base_grid["crs"]) != str(processing_grid["crs"]):
        raise ValueError("3x reference CRS differs from accepted alignment")
    if list(map(float, base_grid["bounds"])) != list(
        map(float, processing_grid["bounds"])
    ):
        raise ValueError("3x reference bounds differ from accepted alignment")
    if int(processing_grid["width"]) != _corner_preserving_dimension(
        int(base_grid["width"]), TARGET_FACTOR
    ) or int(processing_grid["height"]) != _corner_preserving_dimension(
        int(base_grid["height"]), TARGET_FACTOR
    ):
        raise ValueError("processing reference is not the required corner-preserving 3x grid")
    if (int(processing_grid["width"]), int(processing_grid["height"])) != (
        TARGET_WIDTH,
        TARGET_HEIGHT,
    ):
        raise ValueError("processing reference does not have the required 10192x11758 grid")
    transform["target_grid"] = copy.deepcopy(dict(processing_grid))
    transform["resampling_contract"] = {
        "source_diagnostic": "linear_from_original_7088x9375_render",
        "categorical_class_ids": "nearest_from_source_native_vector_raster",
        "prior_class_raster_used": False,
        "corner_preserving_target_supersampling": TARGET_FACTOR,
    }
    return transform


def _reference_rows_to_source(
    transform: Mapping[str, Any], top: int, bottom: int
) -> tuple[np.ndarray, np.ndarray]:
    width = int(transform["target_grid"]["width"])
    pixel_x, pixel_y = np.meshgrid(
        np.arange(width, dtype=np.float64),
        np.arange(top, bottom, dtype=np.float64),
    )
    source_x, source_y = reference_to_source_points(transform, pixel_x, pixel_y)
    return np.asarray(source_x, np.float32), np.asarray(source_y, np.float32)


def _warp_source_ids_to_target(
    source_ids: np.ndarray,
    transform: Mapping[str, Any],
    *,
    rows_per_chunk: int = 64,
) -> np.ndarray:
    """Sample source-native ids directly onto the 3x target, one row slab at a time."""

    height = int(transform["target_grid"]["height"])
    width = int(transform["target_grid"]["width"])
    target = np.empty((height, width), dtype=np.uint8)
    for top in range(0, height, rows_per_chunk):
        bottom = min(top + rows_per_chunk, height)
        map_x, map_y = _reference_rows_to_source(transform, top, bottom)
        target[top:bottom] = cv2.remap(
            source_ids,
            map_x,
            map_y,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    return target


def _complete_small_gaps(
    observed_ids: np.ndarray, domain: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Fill domain gaps with exact local nearest neighbors without a global EDT.

    Native PDF fills leave only thin rasterization seams.  Each missing component
    is completed in a padded local window, giving the same nearest-neighbor rule
    as a global distance transform while avoiding two full-grid index arrays.
    """

    observed = domain & (observed_ids > 0)
    if not np.any(observed):
        raise ValueError("source-native geologic fills do not intersect the 3x domain")
    missing = domain & ~observed
    complete = observed_ids.copy()
    if not np.any(missing):
        complete[~domain] = 0
        return complete, missing
    contours, _ = cv2.findContours(
        missing.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        padding = 2
        while True:
            left, top = max(0, x - padding), max(0, y - padding)
            right = min(observed.shape[1], x + width + padding)
            bottom = min(observed.shape[0], y + height + padding)
            local_observed = observed[top:bottom, left:right]
            if np.any(local_observed):
                break
            if (left, top, right, bottom) == (0, 0, observed.shape[1], observed.shape[0]):
                raise ValueError("a 3x domain gap has no observed class neighbor")
            padding *= 2
        local_missing = missing[top:bottom, left:right]
        contour_mask = np.zeros(local_missing.shape, dtype=np.uint8)
        shifted = contour.copy()
        shifted[:, 0, 0] -= left
        shifted[:, 0, 1] -= top
        cv2.drawContours(contour_mask, [shifted], -1, 1, thickness=cv2.FILLED)
        component = local_missing & (contour_mask > 0)
        if not np.any(component):
            continue
        indices = distance_transform_edt(
            ~local_observed, return_distances=False, return_indices=True
        )
        local_complete = complete[top:bottom, left:right]
        local_complete[component] = local_complete[
            indices[0][component], indices[1][component]
        ]
    unresolved = missing & (complete == 0)
    if np.any(unresolved):
        raise ValueError(
            f"local nearest completion left {int(np.count_nonzero(unresolved))} pixels"
        )
    complete[~domain] = 0
    return complete, missing


def _palette(classes: Sequence[GeologicLegendClass]) -> np.ndarray:
    result = np.zeros((256, 3), dtype=np.uint8)
    for category in classes:
        result[category.class_id] = category.display_rgb
    return result


def _nearest_visible_palette(
    source_rgb: np.ndarray,
    classes: Sequence[GeologicLegendClass],
    domain: np.ndarray,
    maximum_distance: float,
) -> tuple[np.ndarray, np.ndarray]:
    nearest_ids = np.zeros(domain.shape, dtype=np.uint8)
    nearest_distance = np.full(domain.shape, np.inf, dtype=np.float32)
    pixels = source_rgb.astype(np.int32)
    best = np.full(domain.shape, np.iinfo(np.int32).max, dtype=np.int32)
    for category in classes:
        delta = pixels - np.asarray(category.display_rgb, dtype=np.int32)
        squared = np.sum(delta * delta, axis=2, dtype=np.int32)
        better = squared < best
        best[better] = squared[better]
        nearest_ids[better] = category.class_id
    nearest_distance[domain] = np.sqrt(best[domain].astype(np.float32))
    visible = domain & (nearest_distance <= maximum_distance)
    nearest_ids[~visible] = 0
    return nearest_ids, visible


def _source_cell_comparisons(
    source_rgb_path: Path,
    source_expected_ids: np.ndarray,
    target_ids: np.ndarray,
    transform: Mapping[str, Any],
    classes: Sequence[GeologicLegendClass],
    output_dir: Path,
    *,
    maximum_visible_rgb_distance: float,
) -> tuple[list[dict[str, Any]], list[Path]]:
    """Measure and retain worst 6x6 diffs in original source pixels."""

    cells = plan_native_regional_diffs(
        transform, rows=REGIONAL_ROWS, columns=REGIONAL_COLUMNS
    )
    palette = _palette(classes)
    reports: list[dict[str, Any]] = []
    chip_material: list[tuple[float, str, np.ndarray]] = []
    with Image.open(source_rgb_path) as source_image:
        source_image = source_image.convert("RGB")
        for cell in cells:
            left, top, right, bottom = map(
                int, cell["source_original_pixel_bounds"]
            )
            if right <= left or bottom <= top:
                continue
            expected = source_expected_ids[top:bottom, left:right]
            domain = expected > 0
            expected_count = int(np.count_nonzero(domain))
            if expected_count == 0:
                continue
            map_x, map_y = _source_pixels_to_reference_chunk(
                transform, top, bottom, source_expected_ids.shape[1]
            )
            map_x = map_x[:, left:right]
            map_y = map_y[:, left:right]
            sampled = cv2.remap(
                target_ids,
                map_x,
                map_y,
                interpolation=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            semantic_match = domain & (sampled == expected)
            semantic_fraction = float(
                np.count_nonzero(semantic_match) / expected_count
            )
            source_rgb = np.asarray(
                source_image.crop((left, top, right, bottom)), dtype=np.uint8
            )
            visible_ids, visible = _nearest_visible_palette(
                source_rgb,
                classes,
                domain,
                maximum_visible_rgb_distance,
            )
            visible_count = int(np.count_nonzero(visible))
            visible_match = visible & (sampled == visible_ids)
            visible_fraction = float(
                np.count_nonzero(visible_match) / max(visible_count, 1)
            )
            mismatch = domain & ~semantic_match
            report = {
                "id": cell["id"],
                "source_original_pixel_bounds": [left, top, right, bottom],
                "target_pixel_bounds": cell["target_pixel_bounds"],
                "comparison_space": "original_7088x9375_source_pixels",
                "source_semantic_expected_pixel_count": expected_count,
                "source_semantic_match_pixel_count": int(
                    np.count_nonzero(semantic_match)
                ),
                "source_semantic_match_fraction": semantic_fraction,
                "source_visible_pixel_count": visible_count,
                "source_visible_match_fraction": visible_fraction,
                "source_semantic_mismatch_pixel_count": int(np.count_nonzero(mismatch)),
                "passed": semantic_fraction >= 0.94,
            }
            reports.append(report)

            reconstruction = palette[sampled]
            diff_panel = reconstruction.copy()
            diff_panel[mismatch] = (255, 0, 255)
            diff_panel[~domain] = (0, 0, 0)
            montage = np.concatenate((source_rgb, reconstruction, diff_panel), axis=1)
            chip_material.append((1.0 - semantic_fraction, str(cell["id"]), montage))

    chip_dir = output_dir / "worst-native-regional-chips"
    chip_dir.mkdir()
    retained: list[Path] = []
    for rank, (_error, cell_id, montage) in enumerate(
        sorted(chip_material, key=lambda item: (-item[0], item[1]))[
            :RETAINED_WORST_CHIPS
        ],
        1,
    ):
        canvas = Image.fromarray(montage, mode="RGB")
        draw = ImageDraw.Draw(canvas)
        panel_width = montage.shape[1] // 3
        for index, label in enumerate(
            ("original PDF render", "3x extraction", "semantic diff (magenta)")
        ):
            x = index * panel_width + 12
            draw.rectangle((x - 4, 8, x + 245, 38), fill=(0, 0, 0))
            draw.text((x, 13), label, fill=(255, 255, 255))
        path = chip_dir / f"{rank:02d}-{cell_id}.png"
        canvas.save(path, optimize=True)
        retained.append(path)
    return reports, retained


@dataclass(frozen=True)
class GeologicHighresResult:
    status: str
    output_root: Path
    accepted_path: Path | None
    iterations: int
    gates: Mapping[str, Any]


def _fresh_log(
    base_log_path: Path,
    processing_manifest_path: Path,
    processing_pin: Mapping[str, Any],
) -> NoHumanExperimentLog:
    log = NoHumanExperimentLog.load(base_log_path)
    data = copy.deepcopy(log.data)
    data["map_id"] = "geologic-highres-3x"
    data["extraction"] = {
        "iterations": [],
        "accepted_automatic_iteration_count": None,
    }
    data["final"] = {"status": "in_progress", "blocker": None}
    data["processing_reference"] = {
        "path": str(processing_manifest_path.resolve()),
        **dict(processing_pin),
        "role": "unchanged-pinned-vector-bytes-rerasterized-at-3x",
    }
    data["resolution_fidelity"] = {
        "target_grid": [TARGET_WIDTH, TARGET_HEIGHT],
        "target_supersampling": TARGET_FACTOR,
        "source_render": [7088, 9375],
        "prior_class_raster_used": False,
        "native_regional_grid": [REGIONAL_ROWS, REGIONAL_COLUMNS],
    }
    result = NoHumanExperimentLog.__new__(NoHumanExperimentLog)
    result.data = data
    return result


def run_geologic_pdf_highres_extraction(
    source_adapter_manifest_path: Path,
    accepted_alignment_path: Path,
    base_reference_manifest_path: Path,
    processing_reference_manifest_path: Path,
    base_experiment_path: Path,
    output_root: Path,
    *,
    config: GeologicPdfExtractionConfig = GeologicPdfExtractionConfig(),
) -> GeologicHighresResult:
    """Run the immutable two-pass 3x geologic extraction."""

    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite high-resolution run: {output_root}")
    working = load_working_raster_artifact(source_adapter_manifest_path.resolve())
    alignment = _read_json(accepted_alignment_path.resolve())
    base_reference = load_pinned_mapbox_reference(base_reference_manifest_path.resolve())
    processing_reference = load_pinned_mapbox_reference(
        processing_reference_manifest_path.resolve()
    )
    processing_manifest = _read_json(processing_reference_manifest_path.resolve())
    if (working.width, working.height) != (7088, 9375):
        raise ValueError("geologic source adapter is not the authoritative 7088x9375 render")
    if str(working.source_path.resolve()) != str(
        (Path(__file__).resolve().parents[2] / "examples/geologic.pdf").resolve()
    ):
        raise ValueError("source adapter does not point to pristine examples/geologic.pdf")
    if alignment.get("decision") != "accept" or int(alignment.get("iteration", 0)) != 1:
        raise ValueError("high-resolution replay requires accepted geologic alignment 1")
    if alignment.get("exact_transform_provenance", {}).get("original_pdf", {}).get(
        "sha256"
    ) != working.source_sha256:
        raise ValueError("accepted alignment and pristine PDF hashes differ")
    if alignment.get("source_sha256") != working.working_raster_sha256:
        raise ValueError("accepted alignment and original rendered source hashes differ")
    if alignment.get("mapbox_reference") != base_reference.pin:
        raise ValueError("accepted alignment is not pinned to the supplied base reference")
    if {
        key: processing_manifest["derivation"]["source_raw_hashes"][key]
        for key in (
            "style_sha256",
            "tilejson_sha256",
            "tile_aggregate_sha256",
        )
    } != {
        key: alignment["mapbox_reference"][key]
        for key in (
            "style_sha256",
            "tilejson_sha256",
            "tile_aggregate_sha256",
        )
    }:
        raise ValueError("3x processing reference does not preserve accepted raw Mapbox bytes")
    if processing_manifest.get("derivation", {}).get("raw_bytes_preserved_exactly") is not True:
        raise ValueError("3x processing reference did not attest exact raw byte preservation")
    transform = _highres_transform(alignment, processing_reference.grid)

    output_root.mkdir(parents=True)
    extraction_root = output_root / "automatic-extraction"
    extraction_root.mkdir()
    legend_root = extraction_root / "legend"
    legend_root.mkdir()
    report_root = extraction_root / "iterations"
    report_root.mkdir()
    accepted_root = extraction_root / "accepted"
    log = _fresh_log(
        base_experiment_path.resolve(),
        processing_reference_manifest_path.resolve(),
        processing_reference.pin,
    )
    experiment_markdown = output_root / "EXPERIMENT.md"
    experiment_json = output_root / "EXPERIMENT.json"
    log.write(experiment_markdown, experiment_json)

    evidence = _read_native_vector_evidence(working, config)
    legend_path = legend_root / "legend.json"
    legend_path.write_text(json.dumps(_legend_payload(evidence, working), indent=2) + "\n")
    target_domain = processing_reference.state_land & ~processing_reference.water
    del processing_reference

    source_domain = _warp_target_to_source_chunked(
        target_domain.astype(np.uint8),
        transform,
        (working.height, working.width),
    ) > 0
    previous_hashes: dict[str, str] | None = None
    accepted_ids: np.ndarray | None = None
    accepted_observed: np.ndarray | None = None
    accepted_inferred: np.ndarray | None = None
    accepted_source_ids: np.ndarray | None = None
    accepted_gates: dict[str, Any] = {}
    accepted_scores: dict[str, Any] = {}
    accepted_regional_reports: list[dict[str, Any]] = []
    accepted_chip_paths: list[Path] = []
    accepted_regional_path: Path | None = None

    for iteration in (1, 2):
        source_ids = _rasterize_native_fills(
            evidence.records, (working.height, working.width)
        )
        source_ids[~source_domain] = 0
        target_observed_ids = _warp_source_ids_to_target(source_ids, transform)
        target_observed_ids[~target_domain] = 0
        target_ids, inferred = _complete_small_gaps(target_observed_ids, target_domain)
        observed = target_domain & (target_observed_ids > 0)
        hashes = {
            "source_class_id_sha256": _array_sha256(source_ids),
            "target_class_id_sha256": _array_sha256(target_ids),
            "target_observed_mask_sha256": _array_sha256(observed),
            "target_inferred_mask_sha256": _array_sha256(inferred),
        }
        class_count = len({int(value) for value in np.unique(target_ids) if value > 0})
        observed_fraction = float(np.count_nonzero(observed) / np.count_nonzero(target_domain))
        inferred_fraction = float(np.count_nonzero(inferred) / np.count_nonzero(target_domain))
        replay_stable = previous_hashes is not None and hashes == previous_hashes
        gates: dict[str, Any] = {
            "pristine_pdf_and_original_render_authority": True,
            "accepted_alignment_reused_without_refit": True,
            "unchanged_raw_mapbox_vector_bytes": True,
            "corner_preserving_10192x11758_processing_grid": True,
            "prior_class_raster_not_used": True,
            "all_legend_classes_preserved": {
                "passed": class_count == len(evidence.classes),
                "value": class_count,
                "required": len(evidence.classes),
            },
            "target_observed_coverage": {
                "passed": observed_fraction >= config.minimum_target_observed_fraction,
                "value": observed_fraction,
                "minimum": config.minimum_target_observed_fraction,
            },
            "target_inferred_fraction": {
                "passed": inferred_fraction <= config.maximum_target_inferred_fraction,
                "value": inferred_fraction,
                "maximum": config.maximum_target_inferred_fraction,
            },
            "water_and_exterior_empty": not bool(np.any(target_ids[~target_domain] > 0)),
            "successive_source_native_fixed_point": replay_stable,
        }
        scores = {
            "legend_class_count": len(evidence.classes),
            "matched_native_fill_object_count": evidence.matched_filled_object_count,
            "matched_native_contour_count": evidence.matched_contour_count,
            "target_grid": [TARGET_WIDTH, TARGET_HEIGHT],
            "source_render": [working.width, working.height],
            "target_land_pixel_count": int(np.count_nonzero(target_domain)),
            "target_observed_pixel_count": int(np.count_nonzero(observed)),
            "target_inferred_pixel_count": int(np.count_nonzero(inferred)),
            "target_observed_fraction": observed_fraction,
            "target_inferred_fraction": inferred_fraction,
            "target_observed_class_count": class_count,
            "array_hashes": hashes,
            "successive_arrays_equal": replay_stable,
        }
        regional_reports: list[dict[str, Any]] = []
        chip_paths: list[Path] = []
        regional_path: Path | None = None
        if replay_stable and all(_gate_passed(value) for value in gates.values()):
            accepted_root.mkdir()
            regional_reports, chip_paths = _source_cell_comparisons(
                working.working_raster_path,
                source_ids,
                target_ids,
                transform,
                evidence.classes,
                accepted_root,
                maximum_visible_rgb_distance=config.maximum_visible_rgb_distance,
            )
            supported_cells = len(regional_reports)
            passing_cells = sum(bool(report["passed"]) for report in regional_reports)
            minimum_semantic_fraction = min(
                (
                    float(report["source_semantic_match_fraction"])
                    for report in regional_reports
                ),
                default=0.0,
            )
            regional_gate = {
                "passed": supported_cells > 0 and passing_cells == supported_cells,
                "supported_cells": supported_cells,
                "passing_cells": passing_cells,
                "minimum_source_semantic_match_fraction": minimum_semantic_fraction,
                "minimum_required_per_cell": 0.94,
                "grid": [REGIONAL_ROWS, REGIONAL_COLUMNS],
            }
            gates["source_native_6x6_regional_diff"] = regional_gate
            scores.update(
                {
                    "source_native_regional_supported_cells": supported_cells,
                    "source_native_regional_passing_cells": passing_cells,
                    "source_native_regional_minimum_semantic_match_fraction": (
                        minimum_semantic_fraction
                    ),
                }
            )
            regional_path = accepted_root / "source-native-6x6-regional-diff.json"
            regional_path.write_text(
                json.dumps(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "grid": [REGIONAL_ROWS, REGIONAL_COLUMNS],
                        "comparison_space": "original_7088x9375_source_pixels",
                        "gate": regional_gate,
                        "cells": regional_reports,
                        "retained_worst_chip_count": len(chip_paths),
                    },
                    indent=2,
                )
                + "\n"
            )
        decision = "accept" if all(_gate_passed(value) for value in gates.values()) else "retry"
        report = {
            "schema_version": SCHEMA_VERSION,
            "iteration": iteration,
            "decision": decision,
            "scores": scores,
            "gates": gates,
            "storage_policy": (
                "compact hashes only for rejected pass; accepted ids/masks and six worst "
                "source-native regional chips only"
            ),
        }
        report_path = report_root / f"iteration-{iteration:02d}.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        log.record_extraction_iteration(
            scores=scores,
            gates=gates,
            decision=decision,
            provenance=automatic_provenance(
                PRODUCER,
                [
                    "pristine_geologic_pdf",
                    "original_7088x9375_pdf_render",
                    "accepted_native_graticule_transform",
                    "unchanged_pinned_mapbox_vector_bytes_rasterized_at_3x",
                    "source_native_vector_fill_rasterization",
                    "deterministic_fixed_point_replay",
                ],
            ),
            method=(
                "source-native PDF vector fills sampled directly onto a corner-preserving "
                "3x Web-Mercator grid; no prior class raster"
            ),
            artifacts=[{"path": str(report_path), "sha256": _sha256(report_path)}],
        )
        log.write(experiment_markdown, experiment_json)
        if decision == "accept":
            accepted_ids = target_ids
            accepted_observed = observed
            accepted_inferred = inferred
            accepted_source_ids = source_ids
            accepted_gates = gates
            accepted_scores = scores
            accepted_regional_reports = regional_reports
            accepted_chip_paths = chip_paths
            accepted_regional_path = regional_path
            break
        previous_hashes = hashes
        del source_ids, target_observed_ids, target_ids, observed, inferred

    if accepted_ids is None or accepted_source_ids is None:
        blocker = "3x geologic fixed-point replay did not pass all automatic gates"
        log.finalize("blocked", blocker)
        log.write(experiment_markdown, experiment_json)
        return GeologicHighresResult("blocked", output_root, None, 2, accepted_gates)

    if accepted_regional_path is None or not accepted_regional_reports:
        raise ValueError("accepted 3x extraction is missing its source-native regional gate")
    ids_path = accepted_root / "mapbox-class-id.png"
    observed_path = accepted_root / "mapbox-observed-mask.png"
    inferred_path = accepted_root / "mapbox-inferred-mask.png"
    _save_ids(ids_path, accepted_ids)
    _save_mask(observed_path, accepted_observed)
    _save_mask(inferred_path, accepted_inferred)
    regional_gate = accepted_gates["source_native_6x6_regional_diff"]
    supported_cells = int(regional_gate["supported_cells"])
    passing_cells = int(regional_gate["passing_cells"])
    minimum_semantic_fraction = float(
        regional_gate["minimum_source_semantic_match_fraction"]
    )
    artifacts = [
        ids_path,
        observed_path,
        inferred_path,
        accepted_regional_path,
        *accepted_chip_paths,
    ]
    accepted_path = extraction_root / "accepted-extraction.json"
    accepted_path.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "status": "accepted",
                "automatic_alignment_iteration_count": 1,
                "automatic_extraction_iteration_count": 2,
                "source": {
                    "path": str(working.source_path),
                    "sha256": working.source_sha256,
                    "render_path": str(working.working_raster_path),
                    "render_sha256": working.working_raster_sha256,
                    "render_dimensions": [working.width, working.height],
                },
                "accepted_alignment": {
                    "path": str(accepted_alignment_path.resolve()),
                    "sha256": _sha256(accepted_alignment_path.resolve()),
                    "reused_without_refit": True,
                },
                "processing_reference": {
                    "path": str(processing_reference_manifest_path.resolve()),
                    "sha256": _sha256(processing_reference_manifest_path.resolve()),
                    "target_grid": transform["target_grid"],
                    "raw_vector_bytes_preserved_exactly": True,
                },
                "legend": _artifact(legend_path, extraction_root),
                "scores": {
                    **accepted_scores,
                    "source_native_regional_supported_cells": supported_cells,
                    "source_native_regional_passing_cells": passing_cells,
                    "source_native_regional_minimum_semantic_match_fraction": (
                        minimum_semantic_fraction
                    ),
                },
                "gates": accepted_gates,
                "artifacts": [_artifact(path, extraction_root) for path in artifacts],
            },
            indent=2,
        )
        + "\n"
    )
    log.finalize("complete")
    log.write(experiment_markdown, experiment_json)
    return GeologicHighresResult("accepted", output_root, accepted_path, 2, accepted_gates)
