"""Pristine-source ordered-band extraction under the accepted nonlinear candidate."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from PIL import Image
from scipy.interpolate import RBFInterpolator

from .automatic_alignment_loop import (
    _project_reference_points,
    _projection_contexts,
    load_pinned_mapbox_reference,
)
from .automatic_categorical_extraction import _nearest_completion, _save_ids, _save_mask
from .automatic_ordered_band_extraction import (
    ORDERED_SEMANTICS,
    OrderedBandConfig,
    OrderedBandEntry,
    _classify_ordered_hue,
    _render_classes,
    detect_ordered_band_legend,
)
from .native_alignment_validation import _scaled_semantic
from .quake_nonlinear_refinement import ThinPlateWarp


SCHEMA_VERSION = "mapscan.quake-native-nonlinear-extraction.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _load_warp(report: dict[str, Any]) -> tuple[ThinPlateWarp, np.ndarray]:
    candidate = report["candidate"]
    center = np.asarray(candidate["center"], dtype=float)
    span = np.asarray(candidate["span"], dtype=float)
    controls = np.asarray(candidate["controls_source_original"], dtype=float)
    values = np.asarray(candidate["displacements_source_original_px"], dtype=float)
    interpolator = RBFInterpolator(
        (controls - center) / span,
        values,
        kernel="thin_plate_spline",
        smoothing=float(candidate["smoothing"]),
        degree=1,
    )
    return ThinPlateWarp(center, span, controls, values, interpolator), np.asarray(
        candidate["base_matrix"], dtype=float
    )


def _target_from_source(
    source: np.ndarray,
    reference: Any,
    projection: Any,
    base_matrix: np.ndarray,
    warp: ThinPlateWarp,
    *,
    interpolation: int,
    rows_per_chunk: int = 128,
) -> np.ndarray:
    height, width = reference.state_land.shape
    result = np.zeros((height, width, *source.shape[2:]), dtype=source.dtype)
    for top in range(0, height, rows_per_chunk):
        bottom = min(height, top + rows_per_chunk)
        yy, xx = np.mgrid[top:bottom, 0:width]
        reference_points = np.column_stack((xx.ravel(), yy.ravel())).astype(float)
        projected = _project_reference_points(reference_points, projection, reference.grid)
        projected_h = np.column_stack((projected, np.ones(len(projected))))
        base = projected_h @ base_matrix.T
        base = base[:, :2] / base[:, 2:3]
        mapped = warp.map_original(base)
        map_x = mapped[:, 0].reshape(bottom - top, width).astype(np.float32)
        map_y = mapped[:, 1].reshape(bottom - top, width).astype(np.float32)
        result[top:bottom] = cv2.remap(
            source,
            map_x,
            map_y,
            interpolation=interpolation,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    return result


def _regional_observed(domain: np.ndarray, observed: np.ndarray) -> list[dict[str, Any]]:
    reports = []
    for row in range(6):
        for column in range(6):
            top, bottom = round(row * domain.shape[0] / 6), round((row + 1) * domain.shape[0] / 6)
            left, right = round(column * domain.shape[1] / 6), round((column + 1) * domain.shape[1] / 6)
            local = domain[top:bottom, left:right]
            count = int(np.count_nonzero(local))
            if count < 100:
                continue
            fraction = float(np.mean(observed[top:bottom, left:right][local]))
            reports.append({"row": row, "column": column, "domain_pixel_count": count, "observed_fraction": fraction})
    return reports


def run_quake_nonlinear_extraction(
    map_dir: Path,
    reference_path: Path,
    refinement_path: Path,
    output_dir: Path,
) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"output exists: {output_dir}")
    refinement = _read(refinement_path)
    if refinement["status"] != "pass":
        raise ValueError("nonlinear alignment must pass before extraction")
    adapter = _read(map_dir / "source-clean/source-adapter.json")
    alignment = _read(map_dir / "automatic-alignment/accepted-alignment.json")
    source_path = Path(adapter["source"]["path"]).resolve()
    if _sha256(source_path) != adapter["source"]["sha256"]:
        raise ValueError("pristine source hash mismatch")
    if refinement["authority"]["prior_class_raster_used"]:
        raise ValueError("refinement unexpectedly used a prior class raster")
    reference = load_pinned_mapbox_reference(reference_path.resolve())
    projection = next(
        item for item in _projection_contexts(reference)
        if item.id == alignment["transform"]["projection"]["id"]
    )
    warp, base_matrix = _load_warp(refinement)
    with Image.open(source_path) as image:
        source_rgb = np.asarray(image.convert("RGB"))
    output_dir.mkdir(parents=True)
    hypothesis = alignment["scores"]["source_alignment_hypothesis"]
    with tempfile.TemporaryDirectory(prefix="mapscan-quake-extraction-") as temporary:
        scratch = Path(temporary)
        semantic, detector = _scaled_semantic(
            source_rgb, "ordered_gradient_bands", scratch,
            validation_scale=1.0, original_shape=source_rgb.shape[:2],
            generator_hypothesis_id=hypothesis["generator_hypothesis_id"],
            hypothesis_variant_kind=hypothesis["variant_kind"],
        )
        source_domain = semantic.foreground_interior.copy()
        legend = detect_ordered_band_legend(
            source_path, source_rgb, source_domain, output_dir,
            config=OrderedBandConfig(),
        )
    preview = output_dir / "legend/ordered-band-detection.png"
    if preview.exists():
        preview.unlink()
    config = OrderedBandConfig()

    def extract_once():
        observed_ids, complete_ids, inferred, meaningful = _classify_ordered_hue(
            source_rgb, source_domain, legend.entries, config
        )
        packed = np.dstack((complete_ids, (observed_ids > 0).astype(np.uint8)))
        target_packed = _target_from_source(
            packed, reference, projection, base_matrix, warp,
            interpolation=cv2.INTER_NEAREST,
        )
        target_ids = target_packed[..., 0]
        target_observed = target_packed[..., 1] > 0
        target_domain = reference.state_land & ~reference.water
        target_ids[~target_domain] = 0
        target_observed &= target_domain
        if np.any(target_domain & (target_ids == 0)):
            target_ids, _ = _nearest_completion(target_ids, target_domain)
        target_inferred = target_domain & ~target_observed
        return observed_ids, complete_ids, inferred, meaningful, target_ids, target_observed, target_inferred

    first = extract_once()
    second = extract_once()
    fixed = all(np.array_equal(left, right) for left, right in zip(first, second))
    observed_ids, source_ids, source_inferred, meaningful, target_ids, target_observed, target_inferred = second
    reconstruction = _render_classes(source_ids, legend.entries)
    replay_ids, _, _, _ = _classify_ordered_hue(reconstruction, source_domain, legend.entries, config)
    semantic_mismatch = meaningful & (replay_ids != observed_ids)
    observed_fraction = float(np.mean((observed_ids > 0)[source_domain]))
    inferred_fraction = float(np.mean(source_inferred[source_domain]))
    mismatch_fraction = float(np.mean(semantic_mismatch[meaningful]))
    target_domain = reference.state_land & ~reference.water
    regional = _regional_observed(source_domain, observed_ids > 0)
    regional_pass = sum(item["observed_fraction"] >= 0.70 for item in regional)
    classes = {int(value) for value in np.unique(target_ids) if value}
    gates = {
        "fresh_pristine_source_reclassified": True,
        "nonlinear_alignment_all_gates_passed": refinement["status"] == "pass",
        "exact_six_ordered_legend_semantics": [
            (entry.roman, entry.label.split(" ", 1)[1].upper()) for entry in legend.entries
        ] == list(ORDERED_SEMANTICS),
        "all_six_classes_preserved": classes == set(range(1, 7)),
        "observed_source_coverage": {"passed": observed_fraction >= 0.72, "value": observed_fraction, "minimum": 0.72},
        "inferred_source_fraction": {"passed": inferred_fraction <= 0.28, "value": inferred_fraction, "maximum": 0.28},
        "meaningful_source_reconstruction_mismatch": {"passed": mismatch_fraction <= 0.01, "value": mismatch_fraction, "maximum": 0.01},
        "source_domain_complete": bool(np.all(source_ids[source_domain] > 0)),
        "source_layout_empty": not bool(np.any(source_ids[~source_domain] > 0)),
        "target_domain_complete": bool(np.all(target_ids[target_domain] > 0)),
        "mapbox_water_and_exterior_empty": not bool(np.any(target_ids[~target_domain] > 0)),
        "observed_and_inferred_disjoint": not bool(np.any(target_observed & target_inferred)),
        "native_6x6_observation_support": {"passed": regional_pass >= 8, "value": regional_pass, "minimum": 8, "supported_cells": len(regional)},
        "successive_pristine_source_fixed_point": fixed,
    }
    passed = all(bool(value if isinstance(value, bool) else value["passed"]) for value in gates.values())
    iteration = output_dir / "extraction-02"
    iteration.mkdir()
    _save_ids(iteration / "source-class-id.png", source_ids)
    _save_ids(iteration / "mapbox-class-id.png", target_ids)
    _save_mask(iteration / "mapbox-observed-mask.png", target_observed)
    _save_mask(iteration / "mapbox-inferred-mask.png", target_inferred)
    masks = iteration / "class-masks"
    masks.mkdir()
    for entry in legend.entries:
        _save_mask(masks / f"{entry.class_id:02d}-{entry.roman.lower()}.png", target_ids == entry.class_id)
    report = output_dir / "accepted-extraction.json"
    report.write_text(json.dumps({
        "schema_version": SCHEMA_VERSION, "status": "accepted" if passed else "blocked",
        "automatic_iteration_count": 2,
        "source": {"path": str(source_path), "sha256": _sha256(source_path), "shape": list(source_rgb.shape[:2])},
        "alignment": {"path": str(refinement_path.resolve()), "sha256": _sha256(refinement_path)},
        "detector": detector,
        "legend": {"path": "legend/ordered-band-legend.json", "sha256": _sha256(output_dir / "legend/ordered-band-legend.json")},
        "scores": {
            "source_observed_fraction": observed_fraction,
            "source_inferred_fraction": inferred_fraction,
            "meaningful_source_mismatch_fraction": mismatch_fraction,
            "source_class_counts": {entry.label: int(np.count_nonzero(source_ids == entry.class_id)) for entry in legend.entries},
            "mapbox_class_counts": {entry.label: int(np.count_nonzero(target_ids == entry.class_id)) for entry in legend.entries},
            "native_6x6": regional,
        },
        "gates": gates,
        "layers": {entry.roman.lower(): {"class_id": entry.class_id, "label": entry.label, "mask": f"extraction-02/class-masks/{entry.class_id:02d}-{entry.roman.lower()}.png"} for entry in legend.entries},
        "observed_mask": "extraction-02/mapbox-observed-mask.png",
        "inferred_mask": "extraction-02/mapbox-inferred-mask.png",
        "authority": {
            "pristine_source_used": True, "prior_class_raster_used": False,
            "manual_input_used": False, "public_artifacts_mutated": False,
            "fidelity_supersession_pointer_written": False,
        },
    }, indent=2) + "\n")
    return report
