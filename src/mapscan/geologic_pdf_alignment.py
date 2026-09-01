"""No-human registration of the geologic PDF from its native graticule.

This adapter deliberately has a narrower contract than the general image
aligner.  The affine fit is determined only by vector longitude/latitude
intersections preserved by :mod:`mapscan.source_working_raster`.  The pinned
Mapbox state/coast raster is then used as independent validation evidence; it
is never allowed to alter the fitted transform.

There are intentionally no parameters for Census geometry, ``county.png``, a
prior MapScan transform, or human controls.  The source-clean and Mapbox
authority manifests are hash validated before an iteration can be recorded.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
from PIL import Image
from pyproj import CRS, Transformer
from scipy.ndimage import distance_transform_edt

from .automatic_alignment_loop import (
    _project_reference_points,
    _projected_to_candidate_normalized,
    _projection_contexts,
    _projection_round_trip,
    _projection_transform_contract,
    _render_projected_reference_line,
    _transform,
    load_pinned_mapbox_reference,
)
from .experiment_log import NoHumanExperimentLog, automatic_provenance
from .pdf_registration import fit_affine
from .source_working_raster import WorkingRasterArtifact, load_working_raster_artifact


SCHEMA_VERSION = "mapscan.geologic-pdf-alignment.v1"
PRODUCER = "mapscan.geologic_pdf_alignment"
NATIVE_PROJECTION_ID = "california_albers"
NATIVE_PROJECTION_CRS = "EPSG:3310"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_new(path: Path, content: bytes) -> None:
    """Atomically create an immutable result artifact."""

    if path.exists():
        raise FileExistsError(f"refusing to overwrite geologic alignment artifact: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_bytes(content)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def _artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


@dataclass(frozen=True)
class GeologicPdfAlignmentConfig:
    """Pinned gates for the dedicated geologic registration."""

    projection_id: str = NATIVE_PROJECTION_ID
    projection_crs: str = NATIVE_PROJECTION_CRS
    validation_max_dimension: int = 1200
    canny_low: int = 60
    canny_high: int = 160
    minimum_graticule_controls: int = 24
    graticule_median_limit_px: float = 0.75
    graticule_p90_limit_px: float = 1.5
    graticule_maximum_limit_px: float = 3.0
    state_median_limit_px: float = 2.5
    state_p90_limit_px: float = 4.0
    state_within_8px_minimum: float = 0.90
    state_symmetric_f1_minimum: float = 0.70
    state_overlap_tolerance_px: float = 5.0
    state_reverse_corridor_px: float = 8.0
    geographic_cells: tuple[int, int] = (4, 5)
    minimum_supported_cells: int = 8
    maximum_cell_p90_px: float = 8.0
    minimum_cell_pass_fraction: float = 0.80

    def __post_init__(self) -> None:
        if self.projection_id != NATIVE_PROJECTION_ID:
            raise ValueError(
                f"geologic adapter pins projection_id to {NATIVE_PROJECTION_ID}"
            )
        if CRS.from_user_input(self.projection_crs) != CRS.from_epsg(3310):
            raise ValueError("geologic adapter pins the native fit to EPSG:3310")
        if self.validation_max_dimension < 256:
            raise ValueError("validation_max_dimension must be at least 256")
        if not 0 < self.canny_low < self.canny_high <= 255:
            raise ValueError("Canny thresholds are invalid")
        if self.minimum_graticule_controls < 12:
            raise ValueError("at least 12 native graticule controls are required")
        if self.geographic_cells[0] < 2 or self.geographic_cells[1] < 2:
            raise ValueError("geographic cell grid must be at least 2 by 2")


@dataclass(frozen=True)
class GeologicPdfAlignmentResult:
    status: str
    stop_reason: str
    iteration: int
    candidate_path: Path
    accepted_alignment_path: Path | None
    artifact_paths: tuple[Path, ...]

    @property
    def accepted(self) -> Path | None:
        """Compatibility with the batch runner's alignment-result contract."""

        return self.accepted_alignment_path


def _load_native_controls(
    working: WorkingRasterArtifact,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], Path]:
    manifest = working.manifest
    if working.source_path.suffix.lower() != ".pdf":
        raise ValueError("geologic native-graticule alignment requires an original PDF")
    source = manifest.get("source", {})
    if source.get("authoritative") is not True:
        raise ValueError("the original PDF is not marked authoritative")
    pdf = manifest.get("pdf", {})
    if int(pdf.get("selected_page_number", 0)) < 1:
        raise ValueError("source adapter does not pin a PDF page")
    graticule = pdf.get("graticule_evidence", {})
    if graticule.get("status") != "detected":
        raise ValueError("source adapter did not preserve a native PDF graticule")
    controls_name = str(graticule.get("controls_path", ""))
    if Path(controls_name).name != controls_name:
        raise ValueError("graticule controls must be local to the source adapter")
    controls_path = working.manifest_path.parent / controls_name
    expected_hash = str(graticule.get("controls_sha256", ""))
    if not expected_hash or _sha256(controls_path) != expected_hash:
        raise ValueError("native PDF graticule controls hash mismatch")
    payload = json.loads(controls_path.read_text())
    if (
        payload.get("kind") != "pdf_vector_graticule_controls"
        or payload.get("status") != "detected"
        or payload.get("geographic_crs") != "EPSG:4269"
        or payload.get("page_coordinate_space") != "pdfplumber_page_points"
        or int(payload.get("page_number", 0)) != int(pdf["selected_page_number"])
    ):
        raise ValueError("native PDF graticule controls contract is invalid")
    controls = payload.get("controls")
    if not isinstance(controls, list):
        raise ValueError("native PDF graticule controls are missing")
    try:
        geographic = np.asarray(
            [[float(item["longitude"]), float(item["latitude"])] for item in controls],
            dtype=np.float64,
        )
        page_points = np.asarray(
            [[float(item["pdf_x_points"]), float(item["pdf_y_points"])] for item in controls],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("native PDF graticule controls are malformed") from error
    if (
        geographic.ndim != 2
        or geographic.shape[1:] != (2,)
        or len(geographic) < 12
        or page_points.shape != geographic.shape
        or not np.all(np.isfinite(geographic))
        or not np.all(np.isfinite(page_points))
    ):
        raise ValueError("native PDF graticule controls are insufficient or non-finite")
    conversion = manifest.get("conversion", {})
    if conversion.get("adapter") != "poppler-pdftoppm-page-rgb-v1":
        raise ValueError("working raster was not rendered by the source-clean PDF adapter")
    scale_x = float(conversion.get("actual_x_pixels_per_page_point", 0.0))
    scale_y = float(conversion.get("actual_y_pixels_per_page_point", 0.0))
    if not np.isfinite([scale_x, scale_y]).all() or min(scale_x, scale_y) <= 0:
        raise ValueError("PDF page-point to raster-pixel conversion is invalid")
    raster_points = page_points * np.asarray([scale_x, scale_y])
    return geographic, raster_points, payload, controls_path


def _fit_native_transform(
    geographic: np.ndarray,
    raster_points: np.ndarray,
    projection_context: Any,
) -> tuple[np.ndarray, np.ndarray]:
    transformer = Transformer.from_crs(
        "EPSG:4269", projection_context.crs, always_xy=True
    )
    projected_x, projected_y = transformer.transform(
        geographic[:, 0], geographic[:, 1]
    )
    projected = np.column_stack((projected_x, projected_y)).astype(np.float64)
    normalized = _projected_to_candidate_normalized(projected, projection_context)
    matrix, residuals = fit_affine(normalized, raster_points)
    return np.vstack((matrix, [0.0, 0.0, 1.0])), residuals


def _line_summary(values: np.ndarray) -> dict[str, Any]:
    if not len(values):
        return {
            "pixel_count": 0,
            "median_px": 1e6,
            "p90_px": 1e6,
            "within_5px_fraction": 0.0,
            "within_8px_fraction": 0.0,
        }
    return {
        "pixel_count": int(len(values)),
        "median_px": float(np.median(values)),
        "p90_px": float(np.quantile(values, 0.90)),
        "within_5px_fraction": float(np.mean(values <= 5.0)),
        "within_8px_fraction": float(np.mean(values <= 8.0)),
    }


def _balanced_state_report(
    reference_mask: np.ndarray,
    source_edge_distance: np.ndarray,
    matrix: np.ndarray,
    projection_context: Any,
    grid: Mapping[str, Any],
    config: GeologicPdfAlignmentConfig,
) -> dict[str, Any]:
    ys, xs = np.nonzero(reference_mask)
    reference_points = np.column_stack((xs, ys)).astype(np.float64)
    source_points = np.rint(
        _transform(
            _project_reference_points(reference_points, projection_context, grid),
            matrix,
        )
    ).astype(np.int32)
    source_height, source_width = source_edge_distance.shape
    inside = (
        (source_points[:, 0] >= 0)
        & (source_points[:, 0] < source_width)
        & (source_points[:, 1] >= 0)
        & (source_points[:, 1] < source_height)
    )
    distances = np.full(len(source_points), math.hypot(source_width, source_height))
    distances[inside] = source_edge_distance[
        source_points[inside, 1], source_points[inside, 0]
    ]
    rows, columns = config.geographic_cells
    reference_height, reference_width = reference_mask.shape
    cells: list[dict[str, Any]] = []
    for row in range(rows):
        for column in range(columns):
            selected = (
                (ys >= row * reference_height / rows)
                & (ys < (row + 1) * reference_height / rows)
                & (xs >= column * reference_width / columns)
                & (xs < (column + 1) * reference_width / columns)
            )
            if int(np.count_nonzero(selected)) < 20:
                continue
            values = distances[selected]
            p90 = float(np.quantile(values, 0.90))
            cells.append(
                {
                    "row": row,
                    "column": column,
                    "pixel_count": int(len(values)),
                    "median_px": float(np.median(values)),
                    "p90_px": p90,
                    "passed": p90 <= config.maximum_cell_p90_px,
                }
            )
    pass_fraction = float(np.mean([item["passed"] for item in cells])) if cells else 0.0
    passed = bool(
        len(cells) >= config.minimum_supported_cells
        and pass_fraction >= config.minimum_cell_pass_fraction
    )
    return {
        "passed": passed,
        "cell_count": len(cells),
        "minimum_cell_count": config.minimum_supported_cells,
        "cell_pass_fraction": pass_fraction,
        "minimum_cell_pass_fraction": config.minimum_cell_pass_fraction,
        "maximum_cell_p90_px": config.maximum_cell_p90_px,
        "cells": cells,
    }


def _state_coast_validation(
    rgb: np.ndarray,
    reference: Any,
    projection_context: Any,
    original_matrix: np.ndarray,
    config: GeologicPdfAlignmentConfig,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, np.ndarray], np.ndarray, float]:
    original_height, original_width = rgb.shape[:2]
    scale = min(1.0, config.validation_max_dimension / max(rgb.shape[:2]))
    width = max(1, round(original_width * scale))
    height = max(1, round(original_height * scale))
    validation_rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
    scale_x, scale_y = width / original_width, height / original_height
    validation_matrix = np.diag([scale_x, scale_y, 1.0]) @ original_matrix
    rendered_state = _render_projected_reference_line(
        reference.state_coast,
        projection_context,
        validation_matrix,
        reference.grid,
        validation_rgb.shape[:2],
    )
    gray = cv2.cvtColor(validation_rgb, cv2.COLOR_RGB2GRAY)
    source_edges = cv2.Canny(gray, config.canny_low, config.canny_high) > 0
    source_distance = distance_transform_edt(~source_edges)
    reference_distance = distance_transform_edt(~rendered_state)
    forward_values = source_distance[rendered_state]
    forward = _line_summary(forward_values)
    reverse_scope = source_edges & (
        reference_distance <= config.state_reverse_corridor_px
    )
    reverse_values = reference_distance[reverse_scope]
    reverse = _line_summary(reverse_values)
    precision = float(
        np.mean(forward_values <= config.state_overlap_tolerance_px)
    ) if len(forward_values) else 0.0
    recall = float(
        np.mean(reverse_values <= config.state_overlap_tolerance_px)
    ) if len(reverse_values) else 0.0
    f1 = float(2.0 * precision * recall / max(precision + recall, 1e-12))
    balanced = _balanced_state_report(
        reference.state_coast,
        source_distance,
        validation_matrix,
        projection_context,
        reference.grid,
        config,
    )
    scores = {
        "state_coast": {
            "reference_to_source": forward,
            "source_to_reference_within_corridor": reverse,
            "reverse_corridor_px": config.state_reverse_corridor_px,
            "overlap_tolerance_px": config.state_overlap_tolerance_px,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        "state_geographically_balanced": balanced,
    }
    gates = {
        "semantic_full_state_median": {
            "passed": forward["median_px"] <= config.state_median_limit_px,
            "value": forward["median_px"],
            "maximum": config.state_median_limit_px,
        },
        "semantic_full_state_p90": {
            "passed": forward["p90_px"] <= config.state_p90_limit_px,
            "value": forward["p90_px"],
            "maximum": config.state_p90_limit_px,
        },
        "semantic_full_state_support": {
            "passed": forward["within_8px_fraction"]
            >= config.state_within_8px_minimum,
            "value": forward["within_8px_fraction"],
            "minimum": config.state_within_8px_minimum,
        },
        "semantic_full_state_symmetric_overlap": {
            "passed": f1 >= config.state_symmetric_f1_minimum,
            "value": f1,
            "minimum": config.state_symmetric_f1_minimum,
        },
        "semantic_full_state_geographic_balance": {
            "passed": balanced["passed"],
            "value": balanced["cell_pass_fraction"],
            "minimum": config.minimum_cell_pass_fraction,
            "supported_cell_count": balanced["cell_count"],
            "minimum_supported_cell_count": config.minimum_supported_cells,
        },
    }
    rendered = {
        "source_edges": source_edges,
        "rendered_state": rendered_state,
        "validation_rgb": validation_rgb,
    }
    return scores, gates, rendered, validation_matrix, scale


def _write_diagnostics(
    attempt: Path,
    rendered: Mapping[str, np.ndarray],
    raster_controls: np.ndarray,
    original_shape: tuple[int, int],
) -> tuple[Path, Path, Path]:
    edges_path = attempt / "source-edge-evidence.png"
    validation_path = attempt / "mapbox-state-coast-validation.png"
    controls_path = attempt / "native-graticule-controls.png"
    Image.fromarray(rendered["source_edges"].astype(np.uint8) * 255).save(edges_path)

    overlay = np.asarray(rendered["validation_rgb"]).copy()
    source_edges = np.asarray(rendered["source_edges"], dtype=bool)
    state = np.asarray(rendered["rendered_state"], dtype=bool)
    overlay[source_edges] = (40, 185, 235)
    overlay[state] = (115, 255, 90)
    overlay[source_edges & state] = (255, 255, 255)
    Image.fromarray(overlay).save(validation_path)

    controls_overlay = np.asarray(rendered["validation_rgb"]).copy()
    original_height, original_width = original_shape
    scale_x = controls_overlay.shape[1] / original_width
    scale_y = controls_overlay.shape[0] / original_height
    points = np.rint(raster_controls * [scale_x, scale_y]).astype(np.int32)
    for x, y in points:
        if 0 <= x < controls_overlay.shape[1] and 0 <= y < controls_overlay.shape[0]:
            cv2.circle(controls_overlay, (int(x), int(y)), 3, (255, 65, 190), -1)
    Image.fromarray(controls_overlay).save(controls_path)
    return edges_path, validation_path, controls_path


def run_geologic_pdf_alignment(
    source_adapter_manifest_path: Path,
    reference_manifest_path: Path,
    output_root: Path,
    experiment_log: NoHumanExperimentLog,
    *,
    config: GeologicPdfAlignmentConfig | None = None,
) -> GeologicPdfAlignmentResult:
    """Fit the PDF graticule and independently gate it against Mapbox.

    The supplied experiment log must be fresh.  This prevents an older
    transform, an excluded attempt, or another run's candidate from becoming an
    undeclared input to this dedicated path.
    """

    config = config or GeologicPdfAlignmentConfig()
    if experiment_log.data["alignment"]["iterations"]:
        raise ValueError("geologic PDF adapter requires a fresh no-human alignment log")
    if experiment_log.data["alignment"]["accepted_automatic_iteration_count"] is not None:
        raise ValueError("geologic PDF alignment is already accepted")

    working = load_working_raster_artifact(source_adapter_manifest_path)
    logged_source_hash = str(experiment_log.data.get("source", {}).get("sha256", ""))
    if logged_source_hash not in {working.source_sha256, working.working_raster_sha256}:
        raise ValueError("experiment log does not target the source-clean PDF or its raster")

    reference = load_pinned_mapbox_reference(reference_manifest_path)
    logged_reference_hash = str(
        experiment_log.data.get("mapbox_reference", {}).get("manifest_sha256", "")
    )
    if logged_reference_hash != reference.pin["manifest_sha256"]:
        raise ValueError("experiment log and adapter use different pinned Mapbox references")

    geographic, raster_controls, controls_payload, controls_path = _load_native_controls(
        working
    )
    if len(geographic) < config.minimum_graticule_controls:
        raise ValueError(
            f"native graticule has {len(geographic)} controls; "
            f"{config.minimum_graticule_controls} required"
        )
    projection_context = next(
        context
        for context in _projection_contexts(reference)
        if context.id == config.projection_id
    )
    original_matrix, residuals = _fit_native_transform(
        geographic, raster_controls, projection_context
    )
    graticule_scores = {
        "control_count": int(len(residuals)),
        "rms_source_pixel": float(math.sqrt(np.mean(residuals**2))),
        "median_source_pixel": float(np.median(residuals)),
        "p90_source_pixel": float(np.quantile(residuals, 0.90)),
        "maximum_source_pixel": float(np.max(residuals)),
    }
    graticule_gates = {
        "native_graticule_control_count": {
            "passed": len(residuals) >= config.minimum_graticule_controls,
            "value": int(len(residuals)),
            "minimum": config.minimum_graticule_controls,
        },
        "native_graticule_median": {
            "passed": graticule_scores["median_source_pixel"]
            <= config.graticule_median_limit_px,
            "value": graticule_scores["median_source_pixel"],
            "maximum": config.graticule_median_limit_px,
        },
        "native_graticule_p90": {
            "passed": graticule_scores["p90_source_pixel"]
            <= config.graticule_p90_limit_px,
            "value": graticule_scores["p90_source_pixel"],
            "maximum": config.graticule_p90_limit_px,
        },
        "native_graticule_maximum": {
            "passed": graticule_scores["maximum_source_pixel"]
            <= config.graticule_maximum_limit_px,
            "value": graticule_scores["maximum_source_pixel"],
            "maximum": config.graticule_maximum_limit_px,
        },
    }

    with Image.open(working.working_raster_path) as image:
        rgb = np.asarray(image.convert("RGB"))
    semantic_scores, semantic_gates, rendered, validation_matrix, validation_scale = (
        _state_coast_validation(
            rgb, reference, projection_context, original_matrix, config
        )
    )
    round_trip = _projection_round_trip(
        projection_context, original_matrix, reference.grid
    )
    gates: dict[str, Any] = {
        "source_clean_pdf_authority": True,
        **graticule_gates,
        "projection_round_trip": {
            "passed": round_trip["finite"]
            and round_trip["maximum_error_px"] < 1e-5,
            "value": round_trip["maximum_error_px"],
            "maximum": 1e-5,
        },
        **semantic_gates,
    }
    all_passed = all(
        value if isinstance(value, bool) else bool(value["passed"])
        for value in gates.values()
    )
    decision = "accept" if all_passed else "blocked"
    iteration = 1

    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    attempt = output_root / "alignment-01-native-pdf-graticule-california-albers"
    attempt.mkdir(parents=False, exist_ok=False)
    diagnostic_paths = _write_diagnostics(
        attempt, rendered, raster_controls, rgb.shape[:2]
    )
    transform = _projection_transform_contract(
        original_matrix,
        projection=projection_context,
        working_scale=1.0,
        source_original_shape=rgb.shape[:2],
        source_working_shape=rgb.shape[:2],
        target_grid=reference.grid,
    )
    scores = {
        "objective": graticule_scores["rms_source_pixel"],
        "native_graticule": graticule_scores,
        "semantic_full_line": semantic_scores,
        "projection_round_trip": round_trip,
        "projection": {
            "id": projection_context.id,
            "crs_wkt_sha256": projection_context.crs_wkt_sha256,
        },
        "validation_scale_from_original": validation_scale,
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "adapter_schema": SCHEMA_VERSION,
        "iteration": iteration,
        "model": "native_pdf_graticule_affine",
        "projection": projection_context.id,
        "scores": scores,
        "gates": gates,
        "decision": decision,
        # The reusable transform is over the source-clean rendered raster.  The
        # original PDF remains authoritative in exact_transform_provenance.
        "source_sha256": working.working_raster_sha256,
        "mapbox_reference": dict(reference.pin),
        "transform": transform,
        "exact_transform_provenance": {
            "producer": PRODUCER,
            "fit_evidence": "original_pdf_native_vector_graticule_only",
            "mapbox_role": "independent_validation_only",
            "original_pdf": {
                "path": str(working.source_path),
                "sha256": working.source_sha256,
            },
            "source_adapter_manifest": {
                "path": str(working.manifest_path),
                "sha256": _sha256(working.manifest_path),
            },
            "working_raster": {
                "path": str(working.working_raster_path),
                "sha256": working.working_raster_sha256,
                "decoded_rgb_sha256": working.decoded_rgb_sha256,
            },
            "native_graticule_controls": {
                "path": str(controls_path),
                "sha256": _sha256(controls_path),
                "page_number": controls_payload["page_number"],
                "geographic_crs": controls_payload["geographic_crs"],
                "control_count": len(geographic),
            },
            "fit": {
                "estimator": "deterministic_unweighted_affine_least_squares",
                "projection_crs": config.projection_crs,
                "candidate_normalized_to_source_original_pixel_matrix": original_matrix.tolist(),
            },
            "pinned_mapbox_manifest": {
                "path": str(reference.manifest_path),
                "sha256": reference.pin["manifest_sha256"],
                "target_grid_crs": reference.grid["crs"],
            },
        },
    }
    candidate_path = attempt / "candidate.json"
    _write_new(candidate_path, _json_bytes(payload))
    artifact_paths = (*diagnostic_paths, candidate_path)
    experiment_log.record_alignment_iteration(
        scores=scores,
        gates=gates,
        decision=decision,
        provenance=automatic_provenance(
            PRODUCER,
            [
                "original_pdf_native_vector_graticule",
                "source_clean_pdf_working_raster",
                "pinned_mapbox_state_coast",
            ],
        ),
        method="native PDF graticule affine with independent Mapbox state/coast gate",
        artifacts=[_artifact(path) for path in artifact_paths],
        note=(
            None
            if all_passed
            else "Native graticule or independent Mapbox state/coast validation failed."
        ),
    )
    accepted_path: Path | None = None
    if all_passed:
        accepted_path = output_root / "accepted-alignment.json"
        _write_new(accepted_path, _json_bytes(payload))
    return GeologicPdfAlignmentResult(
        status="pass" if all_passed else "blocked",
        stop_reason=(
            "native_graticule_and_mapbox_state_coast_gates_passed"
            if all_passed
            else "native_graticule_or_mapbox_state_coast_gate_failed"
        ),
        iteration=iteration,
        candidate_path=candidate_path,
        accepted_alignment_path=accepted_path,
        artifact_paths=artifact_paths,
    )
