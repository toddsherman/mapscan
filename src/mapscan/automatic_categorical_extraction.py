"""Deterministic, no-human categorical extraction from an accepted alignment.

This module is deliberately separate from MapScan's historical extraction
pipeline.  Its only semantic evidence is the original image's automatically
detected legend.  It does not accept a plan, stamps, paint, review decisions,
or a previous extraction.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import io
import itertools
import json
import math
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw
from pyproj import CRS, Transformer

from .automatic_alignment_loop import load_pinned_mapbox_reference
from .dither_texture_classifier import (
    DitherTextureModel,
    build_dither_texture_model,
    classify_dither_texture,
)
from .experiment_log import NoHumanExperimentLog, automatic_provenance
from .mapbox_reference_migration import (
    verify_non_counting_reference_migration_receipt,
)
from .source_alignment_hypotheses import (
    LegendGroup,
    LegendSwatch,
    SourceHypothesisConfig,
    detect_repeated_legend_swatches,
)


SCHEMA_VERSION = "mapscan.automatic-categorical-extraction.v1"
PRODUCER = "mapscan.automatic_categorical_extraction"
FORBIDDEN_ALIGNMENT_TOKENS = {
    "approval",
    "arrow",
    "brush",
    "control_point",
    "county.png",
    "human",
    "manual",
    "paint",
    "stamp",
}
MINIMUM_CHROMATIC_PALETTE_CHROMA = 16.0
MINIMUM_CHROMATIC_EVIDENCE_FRACTION = 0.60
MINIMUM_CHROMATIC_DIRECTION_COSINE = 0.50


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


def _save_rgb(path: Path, values: np.ndarray) -> None:
    Image.fromarray(values.astype(np.uint8), mode="RGB").save(path, optimize=True)


def _save_mask(path: Path, values: np.ndarray) -> None:
    Image.fromarray(values.astype(np.uint8) * 255, mode="L").save(path, optimize=True)


def _save_ids(path: Path, values: np.ndarray) -> None:
    if int(values.max(initial=0)) > 255:
        raise ValueError("Categorical extraction supports at most 255 legend classes")
    Image.fromarray(values.astype(np.uint8), mode="L").save(path, optimize=True)


@dataclass(frozen=True)
class OCRWord:
    text: str
    confidence: float
    left: int
    top: int
    width: int
    height: int
    line_key: tuple[int, int, int]

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def center_y(self) -> float:
        return self.top + self.height / 2.0


@dataclass(frozen=True)
class LegendEntry:
    class_id: int
    label: str
    rgb: tuple[int, int, int]
    swatch_bbox: tuple[int, int, int, int]
    label_bbox: tuple[int, int, int, int]
    ocr_confidence: float
    swatch_status: str


@dataclass(frozen=True)
class LegendDetection:
    entries: tuple[LegendEntry, ...]
    tesseract_version: str
    swatch_axis_x: float
    row_step_px: float
    artifacts: tuple[Path, ...] = field(default_factory=tuple)
    texture_model: DitherTextureModel | None = None


@dataclass(frozen=True)
class ExtractionLoopConfig:
    """Fixed iteration schedule and conservative automatic quality gates."""

    # The final repeated policy is required for an evidence fixed point.
    policies: tuple[tuple[float, float], ...] = (
        (2.0, 1.0),
        (4.0, 0.75),
        (6.0, 0.5),
        (6.0, 0.5),
    )
    minimum_legend_entries: int = 2
    minimum_label_coverage: float = 1.0
    minimum_observed_fraction: float = 0.72
    maximum_inferred_fraction: float = 0.28
    maximum_meaningful_source_mismatch_fraction: float = 0.01
    meaningful_source_lab_distance: float = 12.0
    geographic_rows: int = 4
    geographic_columns: int = 3
    minimum_geographic_observed_fraction: float = 0.50
    minimum_geographic_cells: int = 5
    minimum_class_direct_support_pixels: int = 12
    minimum_class_plausible_support_pixels: int = 32
    minimum_class_direct_fraction_of_plausible: float = 0.05
    minimum_rare_class_direct_support_pixels: int = 4
    minimum_rare_class_direct_fraction_of_plausible: float = 0.25
    minimum_rare_class_independent_clusters: int = 2
    # A value greater than one consumes a separately rasterized, hash-pinned
    # Mapbox reference over the same bounds.  The accepted alignment remains
    # immutable; its reference-pixel coordinate system is scaled exactly.
    target_supersampling: int = 1
    compact_rejected_artifacts: bool = False
    compact_target_artifacts: bool = False

    def __post_init__(self) -> None:
        if len(self.policies) < 2 or self.policies[-1] != self.policies[-2]:
            raise ValueError("The extraction schedule must end with a repeated policy")
        if self.minimum_legend_entries < 2:
            raise ValueError("minimum_legend_entries must be at least two")
        for value in (
            self.minimum_label_coverage,
            self.minimum_observed_fraction,
            self.maximum_inferred_fraction,
            self.maximum_meaningful_source_mismatch_fraction,
            self.minimum_geographic_observed_fraction,
            self.minimum_class_direct_fraction_of_plausible,
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError("Fraction gates must be between zero and one")
        if self.minimum_class_direct_support_pixels < 1:
            raise ValueError("Per-class direct support minimum must be positive")
        if self.minimum_class_plausible_support_pixels < 1:
            raise ValueError("Per-class plausible support minimum must be positive")
        if self.minimum_rare_class_direct_support_pixels < 1:
            raise ValueError("Rare-class direct support minimum must be positive")
        if self.minimum_rare_class_independent_clusters < 2:
            raise ValueError("Rare-class support must require independent clusters")
        if not isinstance(self.target_supersampling, int) or not 1 <= self.target_supersampling <= 4:
            raise ValueError("Target supersampling must be an integer from one to four")
        if not isinstance(self.compact_rejected_artifacts, bool):
            raise ValueError("Compact rejected-artifact policy must be boolean")
        if not isinstance(self.compact_target_artifacts, bool):
            raise ValueError("Compact target-artifact policy must be boolean")


@dataclass(frozen=True)
class ExtractionIteration:
    iteration: int
    status: str
    scores: Mapping[str, Any]
    gates: Mapping[str, Any]
    artifact_paths: tuple[Path, ...]


def _resumable_prior_extraction_count(experiment_log: NoHumanExperimentLog) -> int:
    """Return retained automatic retries while rejecting ambiguous history."""

    extraction = experiment_log.data["extraction"]
    if extraction.get("accepted_automatic_iteration_count") is not None:
        raise ValueError("Extraction is already accepted")
    iterations = extraction.get("iterations", [])
    for expected, item in enumerate(iterations, start=1):
        provenance = item.get("provenance", {})
        if (
            item.get("decision") not in {"retry", "blocked"}
            or item.get("counts_toward_automatic_iteration_count") is not True
            or item.get("automatic_iteration") != expected
            or provenance.get("actor_kind") != "automated"
            or provenance.get("manual_arrows") is not False
            or provenance.get("manual_stamps") is not False
            or provenance.get("human_approval") is not False
        ):
            raise ValueError(
                "Extraction can resume only prior rejected automatic attempts"
            )
    return len(iterations)


def _canonical_extraction_iteration_number(
    prior_automatic_iteration_count: int, replay_number: int
) -> int:
    if prior_automatic_iteration_count < 0 or replay_number < 1:
        raise ValueError("Extraction iteration ordinals must be positive")
    return prior_automatic_iteration_count + replay_number


def _accepted_extraction_payload(
    accepted: ExtractionIteration,
    source_path: Path,
    alignment_path: Path,
    legend_path: Path,
    processing_reference_manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Build a pointer whose ordinal is canonical across every resumed run."""

    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "accepted",
        "automatic_iteration_count": accepted.iteration,
        "source": {"path": str(source_path), "sha256": _sha256(source_path)},
        "alignment": {
            "path": str(alignment_path),
            "sha256": _sha256(alignment_path),
        },
        "legend": {
            "path": "legend/legend.json",
            "sha256": _sha256(legend_path),
        },
        "accepted_iteration": f"extraction-{accepted.iteration:02d}",
    }
    extent = accepted.scores.get("classification_domain", {}).get(
        "aligned_source_extent"
    )
    if isinstance(extent, Mapping):
        payload["aligned_source_extent"] = dict(extent)
    processing_grid = accepted.scores.get("processing_target_grid")
    if isinstance(processing_grid, Mapping):
        payload["processing_target_grid"] = dict(processing_grid)
    target_supersampling = accepted.scores.get("target_supersampling")
    if isinstance(target_supersampling, int):
        payload["target_supersampling"] = target_supersampling
    if processing_reference_manifest_path is not None:
        payload["processing_reference"] = {
            "path": str(processing_reference_manifest_path.resolve()),
            "sha256": _sha256(processing_reference_manifest_path),
        }
    return payload


@dataclass(frozen=True)
class AutomaticCategoricalExtractionResult:
    status: str
    stop_reason: str
    legend: LegendDetection
    iterations: tuple[ExtractionIteration, ...]
    accepted: ExtractionIteration | None


@dataclass(frozen=True)
class _Swatch:
    rgb: tuple[int, int, int]
    bbox: tuple[int, int, int, int]
    area: int
    detection_kind: str = "uniform_component"

    @property
    def center_y(self) -> float:
        return self.bbox[1] + self.bbox[3] / 2.0


@dataclass(frozen=True)
class _LabelCandidate:
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]
    method: str


def _tesseract_version(executable: str) -> str:
    result = subprocess.run(
        [executable, "--version"], check=True, text=True, capture_output=True
    )
    return result.stdout.splitlines()[0].strip()


def _png_bytes(values: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(values.astype(np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


def _run_tesseract_tsv(
    values: np.ndarray,
    *,
    psm: int,
    extra_config: Sequence[str] = (),
) -> str:
    """OCR an in-memory raster so AVIF/GIF inputs never depend on Leptonica."""

    executable = shutil.which("tesseract")
    if executable is None:
        raise RuntimeError("Automatic legend labels require the tesseract executable")
    result = subprocess.run(
        [
            executable,
            "stdin",
            "stdout",
            "--oem",
            "1",
            "--psm",
            str(psm),
            *extra_config,
            "tsv",
        ],
        input=_png_bytes(values),
        check=True,
        capture_output=True,
    )
    return result.stdout.decode("utf-8", errors="strict")


def _run_tesseract_ocr(source_path: Path) -> tuple[list[OCRWord], str, str]:
    executable = shutil.which("tesseract")
    if executable is None:
        raise RuntimeError("Automatic legend labels require the tesseract executable")
    version = _tesseract_version(executable)
    with Image.open(source_path) as image:
        frames = int(getattr(image, "n_frames", 1))
        if frames != 1:
            raise ValueError("Automatic categorical extraction requires a single-frame source")
        source_rgb = np.asarray(image.convert("RGB"))
    passes: list[str] = []
    for psm in (6, 11):
        passes.append(_run_tesseract_tsv(source_rgb, psm=psm))
    combined = _combine_tesseract_tsv(passes)
    words = _parse_tesseract_tsv(combined)
    return words, combined, version


def _combine_tesseract_tsv(passes: Sequence[str]) -> str:
    """Combine alternate automatic layouts while preserving unique line ids."""

    fieldnames: list[str] | None = None
    output = io.StringIO()
    writer = None
    for pass_index, text in enumerate(passes):
        reader = csv.DictReader(io.StringIO(text), delimiter="\t")
        if fieldnames is None:
            fieldnames = list(reader.fieldnames or [])
            writer = csv.DictWriter(output, delimiter="\t", fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
        if list(reader.fieldnames or []) != fieldnames or writer is None:
            raise ValueError("Tesseract TSV layouts have incompatible columns")
        for row in reader:
            row["block_num"] = str(int(row["block_num"]) + pass_index * 10_000)
            writer.writerow(row)
    return output.getvalue()


def _parse_tesseract_tsv(text: str) -> list[OCRWord]:
    words: list[OCRWord] = []
    reader = csv.DictReader(io.StringIO(text), delimiter="\t")
    for row in reader:
        value = str(row.get("text", "")).strip()
        if not value or row.get("level") != "5":
            continue
        try:
            confidence = float(row["conf"])
            word = OCRWord(
                text=value,
                confidence=confidence,
                left=int(row["left"]),
                top=int(row["top"]),
                width=int(row["width"]),
                height=int(row["height"]),
                line_key=(int(row["block_num"]), int(row["par_num"]), int(row["line_num"])),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Malformed Tesseract TSV") from error
        words.append(word)
    return words


def _alignment_contains_forbidden_input(value: Any, path: str = "") -> str | None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            current = f"{path}.{key}" if path else str(key)
            normalized = str(key).lower().replace("-", "_")
            if any(token.replace(".", "_") in normalized for token in FORBIDDEN_ALIGNMENT_TOKENS):
                return current
            found = _alignment_contains_forbidden_input(child, current)
            if found:
                return found
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            found = _alignment_contains_forbidden_input(child, f"{path}[{index}]")
            if found:
                return found
    elif isinstance(value, str):
        normalized = value.lower().replace("-", "_")
        if any(token.replace(".", "_") in normalized for token in FORBIDDEN_ALIGNMENT_TOKENS):
            return path or "value"
    return None


def _accepted_alignment_sha256_from_iterations(
    iterations: Sequence[Mapping[str, Any]],
    accepted_iteration_count: int,
) -> str:
    """Return the immutable candidate hash recorded by the accepted attempt.

    ``accepted-alignment.json`` is a convenience pointer copied from the
    accepted attempt's immutable ``candidate.json`` artifact.  Hashes stored
    inside that pointer cannot protect against a coherent rewrite of both the
    transform and its internal hashes, so extraction anchors the complete
    pointer bytes to the independently recorded experiment iteration.
    """

    accepted = [
        item
        for item in iterations
        if item.get("automatic_iteration") == accepted_iteration_count
        and item.get("decision") == "accept"
        and item.get("counts_toward_automatic_iteration_count") is True
    ]
    if len(accepted) != 1:
        raise ValueError(
            "Experiment authority does not identify one accepted automatic alignment"
        )
    candidates = [
        artifact
        for artifact in accepted[0].get("artifacts", [])
        if Path(str(artifact.get("path", ""))).name == "candidate.json"
    ]
    if len(candidates) != 1:
        raise ValueError(
            "Accepted automatic alignment does not have one immutable candidate artifact"
        )
    expected = candidates[0].get("sha256")
    if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
        raise ValueError("Accepted automatic alignment candidate hash is invalid")
    return expected


def _load_accepted_alignment(
    alignment_path: Path,
    source_path: Path,
    reference_grid: Mapping[str, Any],
    reference_pin: Mapping[str, Any] | None = None,
    accepted_iteration_count: int | None = None,
    map_id: str | None = None,
    reference_revisions: Sequence[Mapping[str, Any]] = (),
    alignment_iterations: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if alignment_iterations is not None:
        if accepted_iteration_count is None:
            raise ValueError(
                "Accepted-alignment hash authority requires an accepted iteration count"
            )
        expected_alignment_sha256 = _accepted_alignment_sha256_from_iterations(
            alignment_iterations, accepted_iteration_count
        )
        if _sha256(alignment_path) != expected_alignment_sha256:
            raise ValueError(
                "Accepted-alignment pointer hash disagrees with the immutable automatic iteration"
            )
    elif map_id is not None:
        # All production extraction callers provide a map id.  Keeping this
        # branch strict prevents them from silently falling back to trusting
        # self-authenticated fields inside the pointer, while direct legacy
        # readers that do not claim experiment authority remain compatible.
        raise ValueError("Production extraction requires accepted-alignment hash authority")
    alignment = json.loads(alignment_path.read_text())
    if alignment.get("decision") != "accept":
        raise ValueError("Extraction requires an accepted automatic alignment")
    forbidden = _alignment_contains_forbidden_input(alignment)
    if forbidden:
        raise ValueError(f"Alignment contains forbidden legacy/manual input at {forbidden}")
    if alignment.get("source_sha256") != _sha256(source_path):
        raise ValueError("Accepted alignment does not target the original source hash")
    if reference_pin is not None and alignment.get("mapbox_reference") != dict(reference_pin):
        if map_id is None or accepted_iteration_count is None:
            raise ValueError("Accepted alignment does not target the pinned Mapbox artifacts")
        try:
            verify_non_counting_reference_migration_receipt(
                map_id=map_id,
                alignment_path=alignment_path,
                accepted_alignment_reference=alignment.get("mapbox_reference", {}),
                current_reference=reference_pin,
                accepted_automatic_iteration_count=accepted_iteration_count,
                reference_revisions=reference_revisions,
            )
        except ValueError as error:
            raise ValueError(
                "Accepted alignment does not target the pinned Mapbox artifacts"
            ) from error
    if accepted_iteration_count is not None and alignment.get("iteration") != accepted_iteration_count:
        raise ValueError("Accepted alignment iteration disagrees with the experiment log")
    transform = alignment.get("transform", {})
    transform_kind = transform.get("kind")
    if transform_kind not in {
        "regular_global_mapbox_registration",
        "projection_aware_mapbox_registration",
        "projection_aware_residual_warp_mapbox_registration",
    }:
        raise ValueError("Extraction requires an automatic global Mapbox registration")
    if transform.get("source_pixel_space") != "original_raster":
        raise ValueError("Alignment transform is not defined over original source pixels")
    if transform.get("reference_pixel_space") != "pinned_mapbox_target_grid":
        raise ValueError("Alignment transform is not defined over the pinned Mapbox grid")
    target_grid = transform.get("target_grid", {})
    for key in ("crs", "bounds", "width", "height"):
        if target_grid.get(key) != reference_grid.get(key):
            raise ValueError(f"Alignment and pinned Mapbox grids disagree on {key}")
    if transform_kind == "regular_global_mapbox_registration":
        source_to_reference = np.asarray(
            transform.get("source_original_to_reference_pixel_matrix"), dtype=np.float64
        )
        reference_to_source = np.asarray(
            transform.get("reference_pixel_to_source_original_matrix"), dtype=np.float64
        )
    else:
        projection = transform.get("projection", {})
        wkt = projection.get("crs_wkt")
        expected_wkt_hash = projection.get("crs_wkt_sha256")
        if not isinstance(wkt, str) or not wkt or not isinstance(expected_wkt_hash, str):
            raise ValueError("Projection-aware alignment is missing its pinned CRS")
        actual_wkt_hash = hashlib.sha256(wkt.encode("utf-8")).hexdigest()
        if actual_wkt_hash != expected_wkt_hash:
            raise ValueError("Projection-aware alignment CRS hash does not match")
        normalization_center = np.asarray(
            projection.get("normalization_center"), dtype=np.float64
        )
        normalization_scale = float(projection.get("normalization_scale", 0.0))
        if (
            normalization_center.shape != (2,)
            or not np.all(np.isfinite(normalization_center))
            or not np.isfinite(normalization_scale)
            or normalization_scale <= 0
        ):
            raise ValueError("Projection-aware alignment normalization is invalid")
        if projection.get("always_xy") is not True:
            raise ValueError("Projection-aware alignment must pin always_xy behavior")
        try:
            CRS.from_wkt(wkt)
        except Exception as error:
            raise ValueError("Projection-aware alignment CRS cannot be parsed") from error
        source_to_reference = np.asarray(
            transform.get("source_original_to_candidate_normalized_matrix"),
            dtype=np.float64,
        )
        reference_to_source = np.asarray(
            transform.get("candidate_normalized_to_source_original_matrix"),
            dtype=np.float64,
        )
        if transform_kind == "projection_aware_residual_warp_mapbox_registration":
            residual = transform.get("residual_warp", {})
            centers = np.asarray(residual.get("centers_reference_px"), dtype=np.float64)
            coefficients = np.asarray(
                residual.get("coefficients_source_px"), dtype=np.float64
            )
            radius = float(residual.get("radius_reference_px", 0.0))
            ridge = float(residual.get("ridge", 0.0))
            if (
                residual.get("kind")
                != "wendland_c2_reference_pixel_to_source_original_displacement"
                or residual.get("coordinate_domain")
                != "pinned_mapbox_target_grid_pixels"
                or residual.get("displacement_range") != "source_original_pixels"
                or residual.get("kernel") != "max(1-r,0)^4*(4r+1)"
                or centers.ndim != 2
                or centers.shape[1:] != (2,)
                or not 1 <= len(centers) <= 512
                or coefficients.shape != centers.shape
                or not np.all(np.isfinite(centers))
                or not np.all(np.isfinite(coefficients))
                or not np.isfinite(radius)
                or radius <= 0
                or not np.isfinite(ridge)
                or ridge <= 0
            ):
                raise ValueError("Projection residual-warp alignment is invalid")
            encoded_residual = dict(residual)
            expected_residual_sha = encoded_residual.pop("sha256", None)
            actual_residual_sha = hashlib.sha256(
                json.dumps(
                    encoded_residual, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            ).hexdigest()
            if expected_residual_sha != actual_residual_sha:
                raise ValueError("Projection residual-warp hash does not match")
            inverse = transform.get("inverse_solver", {})
            if (
                inverse.get("kind") != "base_projection_fixed_point"
                or not 1 <= int(inverse.get("maximum_iterations", 0)) <= 50
                or not 0 < float(inverse.get("reference_tolerance_px", 0.0)) <= 0.1
                or not 0 < float(inverse.get("source_roundtrip_tolerance_px", 0.0)) <= 0.1
                or inverse.get("failure_policy")
                != "reject_nonconverged_in_domain_points"
            ):
                raise ValueError("Projection residual-warp inverse contract is invalid")
    if source_to_reference.shape != (3, 3) or reference_to_source.shape != (3, 3):
        raise ValueError("Accepted alignment matrices must be 3 by 3")
    if not np.all(np.isfinite(source_to_reference)) or not np.all(
        np.isfinite(reference_to_source)
    ):
        raise ValueError("Accepted alignment matrices contain non-finite values")
    if transform_kind in {
        "projection_aware_mapbox_registration",
        "projection_aware_residual_warp_mapbox_registration",
    } and (
        not np.allclose(source_to_reference[2], [0.0, 0.0, 1.0], atol=1e-10)
        or not np.allclose(reference_to_source[2], [0.0, 0.0, 1.0], atol=1e-10)
    ):
        raise ValueError("Projection-aware extraction currently requires affine matrices")
    if not np.allclose(source_to_reference @ reference_to_source, np.eye(3), atol=1e-5):
        raise ValueError("Accepted alignment matrices are not inverses")
    return alignment


def _reference_pixels_to_web_mercator(
    pixel_x: np.ndarray,
    pixel_y: np.ndarray,
    grid: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    minimum_x, minimum_y, maximum_x, maximum_y = map(float, grid["bounds"])
    width, height = int(grid["width"]), int(grid["height"])
    x = minimum_x + pixel_x / max(width - 1, 1) * (maximum_x - minimum_x)
    y = maximum_y - pixel_y / max(height - 1, 1) * (maximum_y - minimum_y)
    return x, y


def _web_mercator_to_reference_pixels(
    x: np.ndarray,
    y: np.ndarray,
    grid: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    minimum_x, minimum_y, maximum_x, maximum_y = map(float, grid["bounds"])
    width, height = int(grid["width"]), int(grid["height"])
    pixel_x = (x - minimum_x) / (maximum_x - minimum_x) * max(width - 1, 1)
    pixel_y = (maximum_y - y) / (maximum_y - minimum_y) * max(height - 1, 1)
    return pixel_x, pixel_y


def _supersampled_target_grid(
    grid: Mapping[str, Any], factor: int
) -> dict[str, Any]:
    """Return a corner-preserving higher-resolution copy of ``grid``.

    Pixel coordinates in alignment transforms denote samples spanning the
    closed interval from zero through ``width - 1`` (and height respectively).
    Scaling that interval, rather than blindly multiplying the dimensions,
    makes every original reference pixel land exactly on ``factor * pixel`` in
    the processing grid.  This is important for residual-warp centers and for
    reproducible source round trips.
    """

    if not isinstance(factor, int) or factor < 1:
        raise ValueError("Target supersampling factor must be a positive integer")
    width, height = int(grid["width"]), int(grid["height"])
    if width < 2 or height < 2:
        raise ValueError("Target grid must have at least two pixels per axis")
    return {
        **dict(grid),
        "width": (width - 1) * factor + 1,
        "height": (height - 1) * factor + 1,
    }


def _supersample_alignment_transform(
    transform: Mapping[str, Any], factor: int
) -> dict[str, Any]:
    """Evaluate an accepted transform on a denser reference-pixel grid.

    The accepted alignment bytes are never changed.  Projection-aware base
    matrices live outside reference-pixel space and therefore remain exact;
    regular homographies and compact residual coordinates are explicitly
    conjugated into the denser pixel space.
    """

    result = copy.deepcopy(dict(transform))
    original_grid = dict(transform["target_grid"])
    result["target_grid"] = _supersampled_target_grid(original_grid, factor)
    if factor == 1:
        return result

    scale = np.diag([float(factor), float(factor), 1.0])
    inverse_scale = np.diag([1.0 / factor, 1.0 / factor, 1.0])
    if transform["kind"] == "regular_global_mapbox_registration":
        source_to_reference = np.asarray(
            transform["source_original_to_reference_pixel_matrix"],
            dtype=np.float64,
        )
        reference_to_source = np.asarray(
            transform["reference_pixel_to_source_original_matrix"],
            dtype=np.float64,
        )
        result["source_original_to_reference_pixel_matrix"] = (
            scale @ source_to_reference
        ).tolist()
        result["reference_pixel_to_source_original_matrix"] = (
            reference_to_source @ inverse_scale
        ).tolist()
    elif transform["kind"] == "projection_aware_residual_warp_mapbox_registration":
        residual = result["residual_warp"]
        residual["centers_reference_px"] = (
            np.asarray(residual["centers_reference_px"], dtype=np.float64)
            * float(factor)
        ).tolist()
        residual["radius_reference_px"] = float(
            residual["radius_reference_px"]
        ) * float(factor)
        encoded = dict(residual)
        encoded.pop("sha256", None)
        residual["sha256"] = hashlib.sha256(
            json.dumps(encoded, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
    return result


def _validate_supersampled_processing_reference(
    base_reference: Any,
    processing_reference: Any,
    factor: int,
) -> None:
    """Fail closed unless a derived reference preserves pinned Mapbox bytes."""

    expected_grid = _supersampled_target_grid(base_reference.grid, factor)
    if dict(processing_reference.grid) != expected_grid:
        raise ValueError(
            "Supersampled processing reference does not use the exact derived grid"
        )
    for key in ("style_sha256", "tilejson_sha256", "tile_aggregate_sha256"):
        if processing_reference.pin.get(key) != base_reference.pin.get(key):
            raise ValueError(
                f"Supersampled processing reference changed pinned Mapbox {key}"
            )
    manifest = json.loads(processing_reference.manifest_path.read_text())
    derivation = manifest.get("derivation", {})
    if (
        derivation.get("source_manifest_sha256")
        != base_reference.pin.get("manifest_sha256")
        or derivation.get("raw_bytes_preserved_exactly") is not True
        or derivation.get("only_derived_masks_and_overlays_recomputed") is not True
    ):
        raise ValueError(
            "Supersampled processing reference is not derived from the accepted Mapbox pin"
        )


def _residual_displacement(
    transform: Mapping[str, Any],
    pixel_x: np.ndarray,
    pixel_y: np.ndarray,
    *,
    chunk_pixels: int = 8192,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the serialized compact residual without a dense full-grid tensor."""

    residual = transform["residual_warp"]
    centers = np.asarray(residual["centers_reference_px"], dtype=np.float64)
    coefficients = np.asarray(residual["coefficients_source_px"], dtype=np.float64)
    radius_scale = float(residual["radius_reference_px"])
    flat_x = np.asarray(pixel_x, dtype=np.float64).ravel()
    flat_y = np.asarray(pixel_y, dtype=np.float64).ravel()
    output = np.zeros((len(flat_x), 2), dtype=np.float64)
    for start in range(0, len(flat_x), chunk_pixels):
        stop = min(start + chunk_pixels, len(flat_x))
        points = np.column_stack((flat_x[start:stop], flat_y[start:stop]))
        scaled = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2) / radius_scale
        remaining = np.clip(1.0 - scaled, 0.0, 1.0)
        kernel = remaining**4 * (4.0 * scaled + 1.0)
        output[start:stop] = kernel @ coefficients
    return output[:, 0].reshape(pixel_x.shape), output[:, 1].reshape(pixel_y.shape)


def _projection_reference_to_source_base(
    transform: Mapping[str, Any],
    pixel_x: np.ndarray,
    pixel_y: np.ndarray,
    transformer: Transformer,
) -> tuple[np.ndarray, np.ndarray]:
    projection = transform["projection"]
    center = np.asarray(projection["normalization_center"], dtype=np.float64)
    scale = float(projection["normalization_scale"])
    matrix = np.asarray(
        transform["candidate_normalized_to_source_original_matrix"], dtype=np.float64
    )
    web_x, web_y = _reference_pixels_to_web_mercator(
        pixel_x, pixel_y, transform["target_grid"]
    )
    projected_x, projected_y = transformer.transform(web_x, web_y, errcheck=True)
    normalized_x = (projected_x - center[0]) / scale
    normalized_y = -(projected_y - center[1]) / scale
    mapped_x = normalized_x * matrix[0, 0] + normalized_y * matrix[0, 1] + matrix[0, 2]
    mapped_y = normalized_x * matrix[1, 0] + normalized_y * matrix[1, 1] + matrix[1, 2]
    return mapped_x, mapped_y


def _projection_source_to_reference_base(
    transform: Mapping[str, Any],
    source_x: np.ndarray,
    source_y: np.ndarray,
    transformer: Transformer,
) -> tuple[np.ndarray, np.ndarray]:
    projection = transform["projection"]
    matrix = np.asarray(
        transform["source_original_to_candidate_normalized_matrix"], dtype=np.float64
    )
    center = np.asarray(projection["normalization_center"], dtype=np.float64)
    scale = float(projection["normalization_scale"])
    normalized_x = source_x * matrix[0, 0] + source_y * matrix[0, 1] + matrix[0, 2]
    normalized_y = source_x * matrix[1, 0] + source_y * matrix[1, 1] + matrix[1, 2]
    projected_x = normalized_x * scale + center[0]
    projected_y = -normalized_y * scale + center[1]
    web_x, web_y = transformer.transform(projected_x, projected_y, errcheck=True)
    return _web_mercator_to_reference_pixels(web_x, web_y, transform["target_grid"])


def _reference_to_source_remap(
    transform: Mapping[str, Any],
    *,
    rows_per_chunk: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Return source coordinates for every pinned Mapbox target pixel."""

    grid = transform["target_grid"]
    width, height = int(grid["width"]), int(grid["height"])
    if rows_per_chunk < 1:
        raise ValueError("rows_per_chunk must be positive")
    map_x = np.empty((height, width), dtype=np.float32)
    map_y = np.empty((height, width), dtype=np.float32)
    x_values = np.arange(width, dtype=np.float64)
    transform_kind = transform["kind"]
    regular = transform_kind == "regular_global_mapbox_registration"
    if regular:
        matrix = np.asarray(
            transform["reference_pixel_to_source_original_matrix"], dtype=np.float64
        )
        transformer = None
        projection = None
    else:
        projection = transform["projection"]
        transformer = Transformer.from_crs(
            "EPSG:3857", CRS.from_wkt(projection["crs_wkt"]), always_xy=True
        )
        center = np.asarray(projection["normalization_center"], dtype=np.float64)
        scale = float(projection["normalization_scale"])
        matrix = np.asarray(
            transform["candidate_normalized_to_source_original_matrix"],
            dtype=np.float64,
        )
    for top in range(0, height, rows_per_chunk):
        bottom = min(height, top + rows_per_chunk)
        pixel_x, pixel_y = np.meshgrid(
            x_values, np.arange(top, bottom, dtype=np.float64)
        )
        if regular:
            points = np.stack((pixel_x, pixel_y), axis=-1).reshape(-1, 1, 2)
            mapped = cv2.perspectiveTransform(points, matrix).reshape(
                bottom - top, width, 2
            )
        else:
            assert transformer is not None and projection is not None
            mapped_x, mapped_y = _projection_reference_to_source_base(
                transform, pixel_x, pixel_y, transformer
            )
            if transform_kind == "projection_aware_residual_warp_mapbox_registration":
                residual_x, residual_y = _residual_displacement(
                    transform, pixel_x, pixel_y
                )
                mapped_x = mapped_x + residual_x
                mapped_y = mapped_y + residual_y
            mapped = np.stack((mapped_x, mapped_y), axis=-1)
        if not np.all(np.isfinite(mapped)):
            raise ValueError("Reference-to-source remap produced non-finite coordinates")
        map_x[top:bottom] = mapped[..., 0].astype(np.float32)
        map_y[top:bottom] = mapped[..., 1].astype(np.float32)
    return map_x, map_y


def _source_to_reference_remap(
    transform: Mapping[str, Any],
    source_shape: tuple[int, int],
    *,
    rows_per_chunk: int = 256,
    diagnostics: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return target-grid coordinates for every original source pixel."""

    source_height, source_width = source_shape
    if rows_per_chunk < 1:
        raise ValueError("rows_per_chunk must be positive")
    map_x = np.empty((source_height, source_width), dtype=np.float32)
    map_y = np.empty((source_height, source_width), dtype=np.float32)
    x_values = np.arange(source_width, dtype=np.float64)
    transform_kind = transform["kind"]
    regular = transform_kind == "regular_global_mapbox_registration"
    inverse_iteration_maximum = 0
    inverse_final_update_maximum = 0.0
    source_roundtrip_error_maximum = 0.0
    residual_chunk_count = 0
    if regular:
        matrix = np.asarray(
            transform["source_original_to_reference_pixel_matrix"], dtype=np.float64
        )
        transformer = None
        projection = None
    else:
        projection = transform["projection"]
        matrix = np.asarray(
            transform["source_original_to_candidate_normalized_matrix"],
            dtype=np.float64,
        )
        center = np.asarray(projection["normalization_center"], dtype=np.float64)
        scale = float(projection["normalization_scale"])
        transformer = Transformer.from_crs(
            CRS.from_wkt(projection["crs_wkt"]), "EPSG:3857", always_xy=True
        )
    for top in range(0, source_height, rows_per_chunk):
        bottom = min(source_height, top + rows_per_chunk)
        pixel_x, pixel_y = np.meshgrid(
            x_values, np.arange(top, bottom, dtype=np.float64)
        )
        points = np.stack((pixel_x, pixel_y), axis=-1)
        if regular:
            mapped = cv2.perspectiveTransform(
                points.reshape(-1, 1, 2), matrix
            ).reshape(bottom - top, source_width, 2)
            mapped_x, mapped_y = mapped[..., 0], mapped[..., 1]
        else:
            assert transformer is not None and projection is not None
            mapped_x, mapped_y = _projection_source_to_reference_base(
                transform, pixel_x, pixel_y, transformer
            )
            if transform_kind == "projection_aware_residual_warp_mapbox_registration":
                residual_chunk_count += 1
                inverse = transform["inverse_solver"]
                maximum_iterations = int(inverse["maximum_iterations"])
                reference_tolerance = float(inverse["reference_tolerance_px"])
                converged = False
                for _iteration in range(maximum_iterations):
                    residual_x, residual_y = _residual_displacement(
                        transform, mapped_x, mapped_y
                    )
                    next_x, next_y = _projection_source_to_reference_base(
                        transform,
                        pixel_x - residual_x,
                        pixel_y - residual_y,
                        transformer,
                    )
                    delta = np.maximum(np.abs(next_x - mapped_x), np.abs(next_y - mapped_y))
                    if (
                        not np.all(np.isfinite(next_x))
                        or not np.all(np.isfinite(next_y))
                        or not np.all(np.isfinite(delta))
                    ):
                        raise ValueError(
                            "Projection residual-warp inverse produced non-finite iteration state"
                        )
                    mapped_x, mapped_y = next_x, next_y
                    if float(np.max(delta)) <= reference_tolerance:
                        converged = True
                        break
                if not converged:
                    raise ValueError(
                        "Projection residual-warp inverse did not reach reference tolerance"
                    )
                inverse_iteration_maximum = max(
                    inverse_iteration_maximum, _iteration + 1
                )
                inverse_final_update_maximum = max(
                    inverse_final_update_maximum, float(np.max(delta))
                )
                forward_transformer = Transformer.from_crs(
                    "EPSG:3857",
                    CRS.from_wkt(projection["crs_wkt"]),
                    always_xy=True,
                )
                recovered_x, recovered_y = _projection_reference_to_source_base(
                    transform, mapped_x, mapped_y, forward_transformer
                )
                residual_x, residual_y = _residual_displacement(
                    transform, mapped_x, mapped_y
                )
                roundtrip = np.maximum(
                    np.abs(recovered_x + residual_x - pixel_x),
                    np.abs(recovered_y + residual_y - pixel_y),
                )
                if float(np.max(roundtrip)) > float(
                    inverse["source_roundtrip_tolerance_px"]
                ):
                    raise ValueError(
                        "Projection residual-warp inverse did not converge"
                    )
                source_roundtrip_error_maximum = max(
                    source_roundtrip_error_maximum, float(np.max(roundtrip))
                )
        if not np.all(np.isfinite(mapped_x)) or not np.all(np.isfinite(mapped_y)):
            raise ValueError("Source-to-reference remap produced non-finite coordinates")
        map_x[top:bottom] = np.asarray(mapped_x, dtype=np.float32)
        map_y[top:bottom] = np.asarray(mapped_y, dtype=np.float32)
    if diagnostics is not None:
        diagnostics.clear()
        diagnostics.update(
            {
                "transform_kind": transform_kind,
                "source_shape": [source_height, source_width],
                "residual_chunk_count": residual_chunk_count,
                "converged": bool(
                    transform_kind
                    != "projection_aware_residual_warp_mapbox_registration"
                    or residual_chunk_count > 0
                ),
                "maximum_iterations_used": inverse_iteration_maximum,
                "maximum_final_reference_update_px": inverse_final_update_maximum,
                "maximum_source_roundtrip_error_px": source_roundtrip_error_maximum,
            }
        )
    return map_x, map_y


def _source_to_reference(
    values: np.ndarray,
    transform: Mapping[str, Any],
    interpolation: int,
    border_value: int | tuple[int, int, int] = 0,
    remap: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    map_x, map_y = remap or _reference_to_source_remap(transform)
    return cv2.remap(
        values,
        map_x,
        map_y,
        interpolation=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


def _source_data_mask(
    state_land: np.ndarray,
    water: np.ndarray,
    transform: Mapping[str, Any],
    source_shape: tuple[int, int],
) -> np.ndarray:
    reference_data = state_land & ~water
    map_x, map_y = _source_to_reference_remap(transform, source_shape)
    return cv2.remap(
        reference_data.astype(np.uint8),
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ) > 0


def _target_source_coverage_mask(
    transform: Mapping[str, Any],
    source_shape: tuple[int, int],
    remap: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Return target pixels whose centers exist in the aligned source raster.

    A partial-state source cannot authorize completion over California areas
    that were never present in the original raster.  This geometric footprint
    is independent of legend colors and therefore remains valid for sparse and
    near-white categorical datasets alike.
    """

    source_height, source_width = source_shape
    map_x, map_y = remap or _reference_to_source_remap(transform)
    return (
        np.isfinite(map_x)
        & np.isfinite(map_y)
        & (map_x >= 0.0)
        & (map_x <= max(source_width - 1, 0))
        & (map_y >= 0.0)
        & (map_y <= max(source_height - 1, 0))
    )


def _candidate_swatches(rgb: np.ndarray, source_data_mask: np.ndarray) -> list[_Swatch]:
    height, width = rgb.shape[:2]
    minimum_area = max(80, round(height * width * 0.00004))
    maximum_area = round(height * width * 0.004)
    flat = rgb.reshape(-1, 3)
    colors, counts = np.unique(flat, axis=0, return_counts=True)
    candidates: list[_Swatch] = []
    for color, count in zip(colors[counts >= minimum_area], counts[counts >= minimum_area]):
        if count > height * width * 0.45:
            continue
        mask = np.all(rgb == color, axis=2).astype(np.uint8)
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        for component_id in range(1, component_count):
            left, top, box_width, box_height, area = (
                int(value) for value in stats[component_id]
            )
            if not minimum_area <= area <= maximum_area:
                continue
            if box_width < max(8, width * 0.008) or box_height < max(8, height * 0.006):
                continue
            aspect = box_width / box_height
            extent = area / (box_width * box_height)
            if not 0.65 <= aspect <= 6.0 or extent < 0.90:
                continue
            component = labels == component_id
            outside_fraction = 1.0 - float(np.count_nonzero(component & source_data_mask)) / area
            if outside_fraction < 0.98:
                continue
            # A swatch color should also occur as actual data. This rejects page
            # furniture, scale bars, titles, and most neighboring-state fills.
            inside_count = int(np.count_nonzero(np.all(rgb == color, axis=2) & source_data_mask))
            if inside_count < 24:
                continue
            candidates.append(
                _Swatch(
                    rgb=tuple(int(value) for value in color),
                    bbox=(left, top, box_width, box_height),
                    area=area,
                )
            )
    return candidates


def _select_vertical_swatch_series(candidates: Sequence[_Swatch]) -> list[_Swatch]:
    if len(candidates) < 2:
        raise ValueError("No automatically detectable legend swatch series")
    groups: list[list[_Swatch]] = []
    for anchor in candidates:
        left, _, width, height = anchor.bbox
        group = [
            item
            for item in candidates
            if abs(item.bbox[0] - left) <= max(5, width * 0.08)
            and abs(item.bbox[2] - width) <= max(5, width * 0.18)
            and abs(item.bbox[3] - height) <= max(5, height * 0.18)
        ]
        # A legend cannot map two rows to the same vertical position.
        dedup: list[_Swatch] = []
        for item in sorted(group, key=lambda value: value.center_y):
            if not dedup or item.center_y - dedup[-1].center_y > height * 0.65:
                dedup.append(item)
            elif item.area > dedup[-1].area:
                dedup[-1] = item
        # A primary-color categorical legend is a sequence of distinct swatches.
        # Repeated same-color rectangles are commonly tiled ocean/background
        # artifacts and must not win merely because there are many of them.
        if len(dedup) >= 2 and len({item.rgb for item in dedup}) >= 2:
            groups.append(dedup)
    if not groups:
        raise ValueError("No aligned legend swatch column was found")

    def score(group: list[_Swatch]) -> tuple[float, float, float]:
        centers = np.asarray([item.center_y for item in group])
        diffs = np.diff(centers)
        median = float(np.median(diffs)) if len(diffs) else math.inf
        regular = float(np.median(np.abs(diffs - median)) / max(median, 1.0))
        unique_count = len({item.rgb for item in group})
        distinct = unique_count / len(group)
        return (3.0 * unique_count + len(group) - 2.5 * regular, distinct, -regular)

    result = max(groups, key=score)
    centers = np.asarray([item.center_y for item in result])
    step = float(np.median(np.diff(centers)))
    tolerance = max(4.0, step * 0.24)
    # Retain the longest regularly spaced run. This excludes isolated rectangular
    # data polygons that happen to share the legend column's x coordinate.
    best: list[_Swatch] = []
    for start in range(len(result)):
        current = [result[start]]
        for item in result[start + 1 :]:
            gap = item.center_y - current[-1].center_y
            multiple = max(1, round(gap / max(step, 1.0)))
            if abs(gap - multiple * step) <= tolerance:
                current.append(item)
            elif gap > step * 1.8:
                break
        if len(current) > len(best):
            best = current
    if len(best) < 2:
        raise ValueError("Legend swatches do not form a regular vertical series")
    return best


def _group_step(group: LegendGroup) -> float:
    centers = np.asarray(
        [(item.box_original[1] + item.box_original[3]) / 2.0 for item in group.swatches],
        dtype=np.float64,
    )
    return float(np.median(np.diff(centers))) if len(centers) >= 2 else math.inf


def _swatch_from_detected(item: LegendSwatch) -> _Swatch:
    x1, y1, x2, y2 = item.box_original
    return _Swatch(
        rgb=item.rgb,
        bbox=(x1, y1, x2 - x1, y2 - y1),
        area=max(1, (x2 - x1) * (y2 - y1)),
        detection_kind=item.detection_kind,
    )


def _sample_grid_swatch(
    rgb: np.ndarray,
    *,
    center_x: float,
    center_y: float,
    width: int,
    height: int,
) -> _Swatch:
    x1 = max(0, min(rgb.shape[1] - 1, round(center_x - width / 2.0)))
    y1 = max(0, min(rgb.shape[0] - 1, round(center_y - height / 2.0)))
    x2 = min(rgb.shape[1], x1 + width)
    y2 = min(rgb.shape[0], y1 + height)
    patch = rgb[y1:y2, x1:x2]
    if not patch.size:
        raise ValueError("Recovered legend grid swatch lies outside the source")
    color = tuple(int(value) for value in np.median(patch.reshape(-1, 3), axis=0))
    return _Swatch(
        rgb=color,
        bbox=(x1, y1, x2 - x1, y2 - y1),
        area=max(1, (x2 - x1) * (y2 - y1)),
        detection_kind="recovered_regular_grid_slot",
    )


def _compatible_legend_columns(groups: Sequence[LegendGroup]) -> list[LegendGroup]:
    """Return a conservative multi-column legend grid, if one is evident."""

    if len(groups) < 3:
        return []
    anchors = sorted(groups, key=lambda group: (-len(group.swatches), -group.score))
    anchor = anchors[0]
    anchor_step = _group_step(anchor)
    if not np.isfinite(anchor_step) or anchor_step <= 0:
        return []
    anchor_heights = [item.box_original[3] - item.box_original[1] for item in anchor.swatches]
    anchor_height = float(np.median(anchor_heights))
    anchor_centers = np.asarray(
        [(item.box_original[1] + item.box_original[3]) / 2.0 for item in anchor.swatches]
    )
    compatible: list[LegendGroup] = []
    for group in groups:
        if len(group.swatches) < 3:
            continue
        step = _group_step(group)
        heights = [item.box_original[3] - item.box_original[1] for item in group.swatches]
        height = float(np.median(heights))
        if not (0.72 <= step / anchor_step <= 1.28):
            continue
        if not (0.55 <= height / max(anchor_height, 1.0) <= 1.80):
            continue
        centers = np.asarray(
            [(item.box_original[1] + item.box_original[3]) / 2.0 for item in group.swatches]
        )
        row_distance = np.min(np.abs(centers[:, None] - anchor_centers[None, :]), axis=1)
        if float(np.mean(row_distance <= anchor_step * 0.32)) < 0.60:
            continue
        compatible.append(group)
    compatible.sort(
        key=lambda group: float(
            np.median(
                [(item.box_original[0] + item.box_original[2]) / 2.0 for item in group.swatches]
            )
        )
    )
    if len(compatible) < 3:
        return []
    centers_x = np.asarray(
        [
            np.median(
                [(item.box_original[0] + item.box_original[2]) / 2.0 for item in group.swatches]
            )
            for group in compatible
        ]
    )
    widths = np.asarray(
        [
            np.median([item.box_original[2] - item.box_original[0] for item in group.swatches])
            for group in compatible
        ]
    )
    if np.any(np.diff(centers_x) <= np.maximum(widths[:-1], widths[1:]) * 1.25):
        return []
    return compatible


def _multicolumn_lattice_contract(
    groups: Sequence[LegendGroup],
) -> tuple[bool, list[float], dict[str, Any]]:
    """Decide whether compatible columns form a rectangular legend grid.

    Equal swatch size and spacing do not imply that every column has the same
    semantic rows.  Sectioned categorical legends commonly use several
    independently regular but ragged columns.  A true grid may recover an
    isolated missing/dithered cell.  A short column may also be recovered when
    at least three other columns establish the complete lattice and every
    shorter column is a contiguous edge truncation.  It must not contain
    multi-row gaps inside its observed span.  The decision depends only on
    detected swatch geometry.
    """

    if not groups:
        return False, [], {"reason": "no_compatible_groups"}
    anchor = max(groups, key=lambda group: (len(group.swatches), group.score))
    step = _group_step(anchor)
    all_centers = [
        (item.box_original[1] + item.box_original[3]) / 2.0
        for group in groups
        for item in group.swatches
    ]
    first_center = min(all_centers)
    last_center = max(all_centers)
    row_count = int(round((last_center - first_center) / step)) + 1
    if not 3 <= row_count <= 20:
        return False, [], {
            "reason": "implausible_row_count",
            "row_count": row_count,
        }
    row_centers = [first_center + index * step for index in range(row_count)]
    column_reports: list[dict[str, Any]] = []
    strictly_rectangular = True
    for group in groups:
        detected_centers = np.asarray(
            [
                (item.box_original[1] + item.box_original[3]) / 2.0
                for item in group.swatches
            ],
            dtype=np.float64,
        )
        occupied = [
            bool(np.min(np.abs(detected_centers - center)) <= step * 0.32)
            for center in row_centers
        ]
        longest_missing_run = 0
        current_missing_run = 0
        for present in occupied:
            if present:
                current_missing_run = 0
            else:
                current_missing_run += 1
                longest_missing_run = max(longest_missing_run, current_missing_run)
        occupied_fraction = float(np.mean(occupied))
        column_rectangular = bool(
            occupied_fraction >= 0.65 and longest_missing_run <= 1
        )
        strictly_rectangular &= column_rectangular
        occupied_rows = [
            index for index, present in enumerate(occupied) if present
        ]
        if occupied_rows:
            internal_missing_rows = [
                index
                for index in range(occupied_rows[0], occupied_rows[-1] + 1)
                if not occupied[index]
            ]
        else:
            internal_missing_rows = list(range(row_count))
        edge_truncated = bool(
            len(group.swatches) >= 3 and len(internal_missing_rows) <= 1
        )
        column_reports.append(
            {
                "group_id": group.id,
                "detected_count": len(group.swatches),
                "occupied_rows": occupied_rows,
                "missing_rows": [
                    index for index, present in enumerate(occupied) if not present
                ],
                "internal_missing_rows": internal_missing_rows,
                "edge_truncated": edge_truncated,
                "occupied_fraction": occupied_fraction,
                "longest_missing_run": longest_missing_run,
                "rectangular": column_rectangular,
            }
        )
    anchor_count = sum(report["rectangular"] for report in column_reports)
    recovered_edge_truncations = bool(
        anchor_count >= 3
        and all(report["edge_truncated"] for report in column_reports)
    )
    rectangular = strictly_rectangular or recovered_edge_truncations
    return rectangular, row_centers, {
        "reason": (
            "rectangular"
            if strictly_rectangular
            else "rectangular_with_edge_truncations"
            if recovered_edge_truncations
            else "structurally_ragged"
        ),
        "row_count": row_count,
        "row_step_px": step,
        "rectangular_anchor_column_count": anchor_count,
        "columns": column_reports,
    }


def _automatic_swatch_sequence(
    rgb: np.ndarray,
) -> tuple[list[_Swatch], dict[str, Any]]:
    """Select the strongest regular sequence, preserving recovered gap rows.

    Multi-column legends are accepted only when at least three independently
    detected columns share the same row lattice.  Missing cells are then
    sampled at lattice positions; this is needed for dithered palettes whose
    solid connected component can disappear in one row.
    """

    height, width = rgb.shape[:2]
    scale = min(1.0, 1200.0 / max(height, width))
    working = (
        rgb
        if scale == 1.0
        else cv2.resize(
            rgb,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    )
    groups = detect_repeated_legend_swatches(
        working,
        scale_from_original=scale,
        original_shape=(height, width),
        config=SourceHypothesisConfig(working_max_dimension=1200),
    )
    if not groups:
        raise ValueError("No automatically detectable regular legend swatch sequence")
    selected_groups = _compatible_legend_columns(groups)
    if not selected_groups:
        selected = sorted(groups[0].swatches, key=lambda item: item.box_original[1])
        swatches = [_swatch_from_detected(item) for item in selected]
        selection_kind = "strongest_regular_group"
    else:
        rectangular, row_centers, lattice_diagnostics = (
            _multicolumn_lattice_contract(selected_groups)
        )
        swatches = []
        if rectangular:
            step = float(lattice_diagnostics["row_step_px"])
            for group in selected_groups:
                detected = [_swatch_from_detected(item) for item in group.swatches]
                center_x = float(
                    np.median([item.bbox[0] + item.bbox[2] / 2.0 for item in detected])
                )
                box_width = max(
                    4, round(float(np.median([item.bbox[2] for item in detected])))
                )
                box_height = max(
                    4, round(float(np.median([item.bbox[3] for item in detected])))
                )
                for row_center in row_centers:
                    nearest = min(
                        detected, key=lambda item: abs(item.center_y - row_center)
                    )
                    if abs(nearest.center_y - row_center) <= step * 0.32:
                        swatches.append(nearest)
                    else:
                        swatches.append(
                            _sample_grid_swatch(
                                rgb,
                                center_x=center_x,
                                center_y=row_center,
                                width=box_width,
                                height=box_height,
                            )
                        )
            selection_kind = "compatible_regular_multicolumn_grid"
        else:
            for group in selected_groups:
                swatches.extend(
                    _swatch_from_detected(item)
                    for item in sorted(
                        group.swatches, key=lambda item: item.box_original[1]
                    )
                )
            selection_kind = "compatible_ragged_multicolumn"
    diagnostics = {
        "selection_kind": selection_kind,
        "working_scale_from_original": scale,
        "selected_swatch_count": len(swatches),
        "selected_group_ids": [group.id for group in (selected_groups or [groups[0]])],
        "multicolumn_lattice": (
            lattice_diagnostics if selected_groups else None
        ),
        "candidate_groups": [
            {
                "id": group.id,
                "score": group.score,
                "box_original": list(group.box_original),
                "swatch_count": len(group.swatches),
                "swatches": [
                    {
                        "box_original": list(item.box_original),
                        "rgb": list(item.rgb),
                        "detection_kind": item.detection_kind,
                    }
                    for item in group.swatches
                ],
            }
            for group in groups
        ],
    }
    return swatches, diagnostics


def _line_records(words: Sequence[OCRWord]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int, int], list[OCRWord]] = {}
    for word in words:
        grouped.setdefault(word.line_key, []).append(word)
    records = []
    for key, line_words in grouped.items():
        line_words = sorted(line_words, key=lambda value: value.left)
        records.append({"key": key, "words": line_words})
    return records


def _normalized_label(words: Sequence[OCRWord]) -> str:
    text = " ".join(word.text for word in words).strip()
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\s*-\s*", " - ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _deduplicate_ocr_words(words: Sequence[OCRWord]) -> list[OCRWord]:
    """Collapse the same glyph reported by multiple page-segmentation passes."""

    result: list[OCRWord] = []
    for word in sorted(words, key=lambda item: (-item.confidence, item.top, item.left)):
        duplicate = any(
            word.text.casefold() == prior.text.casefold()
            and abs(word.left - prior.left) <= max(2, round(min(word.width, prior.width) * 0.12))
            and abs(word.center_y - prior.center_y)
            <= max(3, round(max(word.height, prior.height) * 0.30))
            for prior in result
        )
        if not duplicate:
            result.append(word)
    return sorted(result, key=lambda item: (item.top, item.left))


def _candidate_from_words(words: Sequence[OCRWord], method: str) -> _LabelCandidate | None:
    words = _deduplicate_ocr_words(words)
    if not words:
        return None
    # Reconstruct visual lines without relying on Tesseract's pass-specific ids.
    lines: list[list[OCRWord]] = []
    for word in sorted(words, key=lambda item: (item.center_y, item.left)):
        matching = next(
            (
                line
                for line in lines
                if abs(
                    word.center_y
                    - float(np.median([candidate.center_y for candidate in line]))
                )
                <= max(3.0, 0.55 * max(word.height, np.median([item.height for item in line])))
            ),
            None,
        )
        if matching is None:
            lines.append([word])
        else:
            matching.append(word)
    lines.sort(key=lambda line: float(np.median([word.center_y for word in line])))
    line_labels = [_normalized_label(sorted(line, key=lambda word: word.left)) for line in lines]
    label = re.sub(r"\s+", " ", " ".join(value for value in line_labels if value)).strip()
    if not label or not any(character.isalnum() for character in label):
        return None
    left = min(word.left for word in words)
    top = min(word.top for word in words)
    right = max(word.right for word in words)
    bottom = max(word.top + word.height for word in words)
    return _LabelCandidate(
        label=label,
        confidence=float(np.mean([word.confidence for word in words])),
        bbox=(left, top, right - left, bottom - top),
        method=method,
    )


def _translated_region_words(
    tsv: str,
    *,
    origin_x: int,
    origin_y: int,
    scale: float,
    pass_offset: int,
) -> list[OCRWord]:
    translated: list[OCRWord] = []
    for word in _parse_tesseract_tsv(tsv):
        translated.append(
            OCRWord(
                text=word.text,
                confidence=word.confidence,
                left=origin_x + round(word.left / scale),
                top=origin_y + round(word.top / scale),
                width=max(1, round(word.width / scale)),
                height=max(1, round(word.height / scale)),
                line_key=(word.line_key[0] + pass_offset, *word.line_key[1:]),
            )
        )
    return translated


def _ocr_region_candidates(
    rgb: np.ndarray,
    box: tuple[int, int, int, int],
    *,
    embedded: bool,
) -> tuple[list[_LabelCandidate], list[dict[str, Any]], np.ndarray]:
    x, y, width, height = box
    crop = rgb[y : y + height, x : x + width]
    if not crop.size:
        return [], [], crop
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    variants: list[tuple[str, np.ndarray, float, int, tuple[str, ...]]] = []
    if embedded:
        scale = 5.0
        enlarged = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        adaptive = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            15,
            5,
        )
        adaptive = cv2.resize(
            adaptive, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
        )
        whitelist = ("-c", "tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
        compact_whitelist = ("-c", "tessedit_char_whitelist=0123456789ab")
        variants.extend(
            [
                ("embedded-gray-psm7", enlarged, scale, 7, whitelist),
                ("embedded-gray-psm8", enlarged, scale, 8, whitelist),
                ("embedded-adaptive-psm8", adaptive, scale, 8, whitelist),
                (
                    "embedded-gray-8x-psm8",
                    cv2.resize(gray, None, fx=8.0, fy=8.0, interpolation=cv2.INTER_CUBIC),
                    8.0,
                    8,
                    whitelist,
                ),
                (
                    "embedded-adaptive-8x-psm8",
                    cv2.resize(
                        cv2.adaptiveThreshold(
                            gray,
                            255,
                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                            cv2.THRESH_BINARY,
                            15,
                            5,
                        ),
                        None,
                        fx=8.0,
                        fy=8.0,
                        interpolation=cv2.INTER_CUBIC,
                    ),
                    8.0,
                    8,
                    whitelist,
                ),
                (
                    "embedded-compact-8x-psm8",
                    cv2.resize(gray, None, fx=8.0, fy=8.0, interpolation=cv2.INTER_CUBIC),
                    8.0,
                    8,
                    compact_whitelist,
                ),
            ]
        )
    else:
        scale = 3.0
        enlarged = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        variants.extend(
            [
                ("right-gray-psm6", enlarged, scale, 6, ()),
                ("right-gray-psm7", enlarged, scale, 7, ()),
            ]
        )
    candidates: list[_LabelCandidate] = []
    diagnostic: list[dict[str, Any]] = []
    for index, (method, values, resize_scale, psm, config) in enumerate(variants, 1):
        try:
            tsv = _run_tesseract_tsv(values, psm=psm, extra_config=config)
            words = _translated_region_words(
                tsv,
                origin_x=x,
                origin_y=y,
                scale=resize_scale,
                pass_offset=index * 10_000,
            )
            candidate = _candidate_from_words(words, method)
        except subprocess.CalledProcessError as error:
            candidate = None
            diagnostic.append(
                {
                    "method": method,
                    "status": "ocr_error",
                    "returncode": error.returncode,
                }
            )
            continue
        if candidate is not None:
            candidates.append(candidate)
        diagnostic.append(
            {
                "method": method,
                "status": "readable" if candidate is not None else "no_text",
                "label": candidate.label if candidate is not None else None,
                "confidence": candidate.confidence if candidate is not None else None,
            }
        )
    return candidates, diagnostic, crop


def _clean_integrated_label(label: str) -> str:
    label = re.sub(r"^[|\[\]()_]+\s*", "", label).strip()
    # A right-hand legend frame is commonly OCR'd as a trailing pipe.  Strip
    # frame-shaped furniture while preserving meaningful terminal parentheses
    # such as ``Beans (Dry)``.
    label = re.sub(r"\s*[|\[\]_]+\s*$", "", label).strip()
    # Detached terminal punctuation is another common OCR rendering of a
    # nearby frame/leader. Preserve punctuation attached to the final token.
    label = re.sub(r"\s+[:;]\s*$", "", label).strip()
    label = re.sub(r"(?<=[A-Za-z])\s*-\s*(?=[A-Za-z])", "-", label)
    return re.sub(r"\s+", " ", label).strip()


def _label_is_clean(label: str) -> bool:
    """Reject residual OCR furniture and obvious merged heading rows."""

    compact = label.strip()
    if not compact or not any(character.isalnum() for character in compact):
        return False
    if re.search(r"[|\[\]_]\s*$", compact):
        return False
    # Decimal labels such as ``5.5`` intentionally repeat the same digit on
    # both sides of the decimal point.  They are not duplicated OCR words.
    # Recognize the numeric legend grammar before applying the prose-only
    # repeated-token safeguard below.
    if re.fullmatch(
        r"\s*[<>]?\s*\d+(?:\.\d+)?(?:\s*-\s*\d+(?:\.\d+)?)?\s*",
        compact,
    ):
        return True
    words = re.findall(r"[A-Za-z0-9]+", compact.casefold())
    if any(first == second for first, second in zip(words, words[1:])):
        return False
    return True


def _exclude_vertical_label_border(
    rgb: np.ndarray,
    *,
    left: int,
    right: int,
    top: int,
    bottom: int,
    minimum_clearance: int,
) -> int:
    """Trim a continuous printed frame from an automatically cropped label row.

    This uses only pixels in the current swatch row.  Text strokes do not span
    most of the row height, whereas a legend-panel border does; requiring a
    two-column run avoids treating an isolated tall glyph as the crop edge.
    """

    if right - left < max(8, minimum_clearance) or bottom <= top:
        return right
    patch = rgb[top:bottom, left:right]
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    dark_fraction = np.mean(gray <= 110, axis=0)
    candidates = np.flatnonzero(dark_fraction >= 0.80)
    if candidates.size < 2:
        return right
    for _, run in itertools.groupby(
        enumerate(candidates.tolist()), key=lambda pair: pair[1] - pair[0]
    ):
        indices = [item[1] for item in run]
        if len(indices) < 2:
            continue
        candidate = left + indices[0]
        maximum_edge_distance = max(
            minimum_clearance, round((right - left) * 0.25)
        )
        if (
            candidate - left >= minimum_clearance
            and right - candidate <= maximum_edge_distance
        ):
            return max(left, candidate - 2)
    return right


def _label_candidate_quality(
    candidate: _LabelCandidate, *, numeric_mode: bool = False
) -> tuple[int, int, int, int, int, float]:
    label = candidate.label.strip()
    clean = int(_label_is_clean(_clean_integrated_label(label)))
    compact_numeric = int(bool(re.fullmatch(r"\d{1,3}[A-Za-z]", label)))
    numeric_pattern = int(
        bool(
            re.fullmatch(
                r"\s*[<>]?\s*\d+(?:\.\d+)?(?:\s*-\s*\d+(?:\.\d+)?)?\s*",
                label,
            )
        )
    )
    semantic_length = sum(character.isalnum() for character in label)
    contextual_page_ocr = int(candidate.method == "full-page-row-association")
    return (
        clean,
        compact_numeric,
        numeric_pattern if numeric_mode else 0,
        contextual_page_ocr,
        semantic_length,
        candidate.confidence,
    )


def _best_label_candidate(
    candidates: Sequence[_LabelCandidate], *, numeric_mode: bool = False
) -> _LabelCandidate | None:
    usable = [
        candidate
        for candidate in candidates
        if candidate.label.strip()
        and not candidate.label.lower().startswith("source")
        and any(character.isalnum() for character in candidate.label)
    ]
    if usable:
        maximum_semantic_length = max(
            sum(character.isalnum() for character in candidate.label)
            for candidate in usable
        )
        # Prefer page-context OCR only when it captured substantially the same
        # row as the local crop; an incomplete page candidate must not replace
        # a complete wrapped label.
        usable = [
            candidate
            for candidate in usable
            if candidate.method != "full-page-row-association"
            or sum(character.isalnum() for character in candidate.label)
            >= math.ceil(maximum_semantic_length * 0.80)
        ]
        maximum_confidence = max(candidate.confidence for candidate in usable)
        usable = [
            candidate
            for candidate in usable
            if candidate.confidence >= maximum_confidence - 10.0
        ]
    return (
        max(
            usable,
            key=lambda candidate: _label_candidate_quality(
                candidate, numeric_mode=numeric_mode
            ),
        )
        if usable
        else None
    )


def _restore_ocr_decimal_points(entries: Sequence[LegendEntry]) -> list[LegendEntry]:
    """Restore decimal points dropped from an otherwise decimal numeric series.

    Tesseract commonly reads ``7.5`` as ``75`` at small legend scales.  This
    correction is allowed only when every label is numeric, several neighboring
    rows visibly retain a decimal point, and the repaired sequence is strictly
    increasing.  It therefore cannot rewrite arbitrary semantic legend text.
    """

    labels = [entry.label.strip() for entry in entries]
    if not labels or not all(re.fullmatch(r"\d+(?:\.\d+)?", label) for label in labels):
        return list(entries)
    if sum("." in label for label in labels) < max(3, math.ceil(len(labels) * 0.40)):
        return list(entries)
    repaired = [
        (f"{label[:-1]}.{label[-1]}" if "." not in label and len(label) >= 2 else label)
        for label in labels
    ]
    values = [float(label) for label in repaired]
    if any(second <= first for first, second in zip(values, values[1:])):
        return list(entries)
    return [
        LegendEntry(
            class_id=entry.class_id,
            label=label,
            rgb=entry.rgb,
            swatch_bbox=entry.swatch_bbox,
            label_bbox=entry.label_bbox,
            ocr_confidence=entry.ocr_confidence,
            swatch_status=entry.swatch_status,
        )
        for entry, label in zip(entries, repaired)
    ]


def _detect_legend(
    rgb: np.ndarray,
    source_data_mask: np.ndarray,
    words: Sequence[OCRWord],
) -> tuple[list[LegendEntry], float, float]:
    swatches = _select_vertical_swatch_series(_candidate_swatches(rgb, source_data_mask))
    swatches = sorted(swatches, key=lambda value: value.center_y)
    steps = np.diff([item.center_y for item in swatches])
    step = float(np.median(steps))
    if step <= 0:
        raise ValueError("Legend swatch row spacing is invalid")
    median_width = int(round(np.median([item.bbox[2] for item in swatches])))
    median_height = int(round(np.median([item.bbox[3] for item in swatches])))
    median_left = int(round(np.median([item.bbox[0] for item in swatches])))
    right = median_left + median_width
    line_records = _line_records(words)

    # Include OCR-labelled rows one step beyond a visible series. This recovers
    # white swatches that merge perfectly into a white legend panel.
    candidate_rows: list[tuple[float, _Swatch | None]] = [
        (item.center_y, item) for item in swatches
    ]
    for center in (swatches[0].center_y - step, swatches[-1].center_y + step):
        matching_words = [
            word
            for record in line_records
            for word in record["words"]
            if word.left >= right + max(4, median_width * 0.08)
            and abs(word.center_y - center) <= step * 0.32
            and word.confidence >= 25
        ]
        if matching_words:
            candidate_rows.append((center, None))
    candidate_rows.sort(key=lambda value: value[0])

    entries: list[LegendEntry] = []
    for center, swatch in candidate_rows:
        possible_lines = []
        for record in line_records:
            label_words = [
                word
                for word in record["words"]
                if word.left >= right + max(4, median_width * 0.08)
            ]
            if not label_words:
                continue
            line_center = float(np.median([word.center_y for word in label_words]))
            if abs(line_center - center) <= step * 0.38:
                possible_lines.append((abs(line_center - center), label_words))
        if not possible_lines:
            raise ValueError("A legend swatch has no automatically detected OCR label")
        def line_quality(candidate: tuple[float, list[OCRWord]]) -> tuple[int, int, float, float]:
            distance, candidate_words = candidate
            candidate_label = _normalized_label(candidate_words)
            semantic_chars = sum(character.isdigit() for character in candidate_label)
            semantic_chars += 2 * sum(character in "<>" for character in candidate_label)
            return (
                semantic_chars,
                len(candidate_label),
                float(np.mean([word.confidence for word in candidate_words])),
                -distance,
            )

        _, label_words = max(possible_lines, key=line_quality)
        label = _normalized_label(label_words)
        if not label or label.lower().startswith("source"):
            raise ValueError("Automatic OCR did not produce a usable legend label")
        left = min(word.left for word in label_words)
        top = min(word.top for word in label_words)
        label_right = max(word.right for word in label_words)
        bottom = max(word.top + word.height for word in label_words)
        if swatch is None:
            swatch_left = median_left
            swatch_top = int(round(center - median_height / 2.0))
            swatch_bbox = (swatch_left, swatch_top, median_width, median_height)
            patch = rgb[
                max(0, swatch_top) : min(rgb.shape[0], swatch_top + median_height),
                max(0, swatch_left) : min(rgb.shape[1], swatch_left + median_width),
            ]
            if not patch.size:
                raise ValueError("Inferred invisible swatch lies outside the source")
            color = tuple(int(value) for value in np.median(patch.reshape(-1, 3), axis=0))
            status = "invisible_row_inferred_from_regular_legend_layout"
        else:
            swatch_bbox = swatch.bbox
            color = swatch.rgb
            status = "visible_exact_color_rectangle"
        entries.append(
            LegendEntry(
                class_id=len(entries) + 1,
                label=label,
                rgb=color,
                swatch_bbox=swatch_bbox,
                label_bbox=(left, top, label_right - left, bottom - top),
                ocr_confidence=float(np.mean([word.confidence for word in label_words])),
                swatch_status=status,
            )
        )
    if len({entry.rgb for entry in entries}) != len(entries):
        raise ValueError("Automatically detected legend contains duplicate colors")
    if len({entry.label for entry in entries}) != len(entries):
        raise ValueError("Automatically detected legend contains duplicate labels")
    return entries, float(median_left + median_width / 2.0), step


def _swatch_columns(swatches: Sequence[_Swatch]) -> list[list[_Swatch]]:
    median_width = float(np.median([item.bbox[2] for item in swatches]))
    tolerance = max(5.0, median_width * 0.45)
    columns: list[list[_Swatch]] = []
    for item in sorted(swatches, key=lambda value: (value.bbox[0], value.center_y)):
        center_x = item.bbox[0] + item.bbox[2] / 2.0
        match = next(
            (
                column
                for column in columns
                if abs(
                    center_x
                    - float(
                        np.median(
                            [candidate.bbox[0] + candidate.bbox[2] / 2.0 for candidate in column]
                        )
                    )
                )
                <= tolerance
            ),
            None,
        )
        if match is None:
            columns.append([item])
        else:
            match.append(item)
    for column in columns:
        column.sort(key=lambda item: item.center_y)
    columns.sort(
        key=lambda column: float(
            np.median([item.bbox[0] + item.bbox[2] / 2.0 for item in column])
        )
    )
    return columns


def _write_swatch_sequence_artifacts(
    output_dir: Path,
    source_rgb: np.ndarray,
    swatches: Sequence[_Swatch],
    diagnostics: Mapping[str, Any],
    tsv_text: str,
    tesseract_version: str,
) -> tuple[Path, Path, Path]:
    legend_dir = output_dir / "legend"
    legend_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = legend_dir / "ocr.tsv"
    tsv_path.write_text(tsv_text)
    sequence_path = legend_dir / "swatch-sequence.json"
    sequence_payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "swatches_detected_labels_pending",
        "tesseract_version": tesseract_version,
        **dict(diagnostics),
        "selected_swatches": [
            {
                "index": index,
                "rgb": list(item.rgb),
                "bbox": list(item.bbox),
                "detection_kind": item.detection_kind,
            }
            for index, item in enumerate(swatches, 1)
        ],
    }
    sequence_path.write_text(json.dumps(sequence_payload, indent=2) + "\n")
    preview = Image.fromarray(source_rgb.copy(), mode="RGB")
    draw = ImageDraw.Draw(preview)
    for index, item in enumerate(swatches, 1):
        x, y, width, height = item.bbox
        draw.rectangle(
            (x - 2, y - 2, x + width + 2, y + height + 2),
            outline=(0, 255, 120),
            width=3,
        )
        draw.text((x, max(0, y - 14)), str(index), fill=(255, 0, 255))
    preview_path = legend_dir / "swatch-sequence.png"
    preview.save(preview_path, optimize=True)
    return tsv_path, sequence_path, preview_path


def _detect_complete_legend(
    rgb: np.ndarray,
    swatches: Sequence[_Swatch],
    words: Sequence[OCRWord],
    output_dir: Path,
) -> tuple[list[LegendEntry], float, float, list[dict[str, Any]]]:
    columns = _swatch_columns(swatches)
    label_root = output_dir / "legend" / "label-regions"
    label_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    flat_index = 0
    for column_index, column in enumerate(columns):
        next_left = (
            min(item.bbox[0] for item in columns[column_index + 1])
            if column_index + 1 < len(columns)
            else rgb.shape[1]
        )
        centers = [item.center_y for item in column]
        column_step = float(np.median(np.diff(centers))) if len(centers) > 1 else max(12.0, column[0].bbox[3] * 1.5)
        for row_index, swatch in enumerate(column):
            flat_index += 1
            x, y, width, height = swatch.bbox
            # Section headings occupy the intentionally large gaps between
            # ragged legend groups.  Associate OCR with the actual swatch row,
            # not the full midpoint-to-midpoint gap, so headings cannot become
            # label prefixes.  A 35% pad still captures wrapped two-line labels.
            vertical_padding = max(2, round(height * 0.35))
            top = max(0, y - vertical_padding)
            bottom = min(rgb.shape[0], y + height + vertical_padding)
            label_left = min(rgb.shape[1], x + width + max(2, round(width * 0.08)))
            label_right = max(label_left, min(rgb.shape[1], next_left - max(2, round(width * 0.10))))
            label_right = _exclude_vertical_label_border(
                rgb,
                left=label_left,
                right=label_right,
                top=top,
                bottom=bottom,
                minimum_clearance=max(width, round(width * 1.5)),
            )
            full_words = [
                word
                for word in words
                if word.confidence >= 20
                and label_left <= word.left + word.width / 2.0
                and word.right <= label_right + 2
                and top <= word.center_y < bottom
            ]
            full_candidate = _candidate_from_words(full_words, "full-page-row-association")
            right_candidates, right_diagnostics, right_crop = _ocr_region_candidates(
                rgb,
                (label_left, top, max(0, label_right - label_left), max(0, bottom - top)),
                embedded=False,
            )
            pad = 2
            embedded_box = (
                max(0, x - pad),
                max(0, y - pad),
                min(rgb.shape[1], x + width + pad) - max(0, x - pad),
                min(rgb.shape[0], y + height + pad) - max(0, y - pad),
            )
            embedded_candidates, embedded_diagnostics, embedded_crop = _ocr_region_candidates(
                rgb, embedded_box, embedded=True
            )
            expanded_pad = 4
            expanded_box = (
                max(0, x - expanded_pad),
                max(0, y - expanded_pad),
                min(rgb.shape[1], x + width + expanded_pad) - max(0, x - expanded_pad),
                min(rgb.shape[0], y + height + expanded_pad) - max(0, y - expanded_pad),
            )
            expanded_candidates, expanded_diagnostics, expanded_crop = _ocr_region_candidates(
                rgb, expanded_box, embedded=True
            )
            embedded_candidates += expanded_candidates
            embedded_diagnostics += [
                {**item, "method": "expanded-" + str(item["method"])}
                for item in expanded_diagnostics
            ]
            _save_rgb(label_root / f"{flat_index:02d}-right.png", right_crop)
            _save_rgb(label_root / f"{flat_index:02d}-embedded.png", embedded_crop)
            _save_rgb(label_root / f"{flat_index:02d}-embedded-expanded.png", expanded_crop)
            rows.append(
                {
                    "index": flat_index,
                    "column": column_index + 1,
                    "row": row_index + 1,
                    "swatch": swatch,
                    "right_box": (label_left, top, label_right - label_left, bottom - top),
                    "right_candidates": [
                        *right_candidates,
                        *([full_candidate] if full_candidate is not None else []),
                    ],
                    "embedded_candidates": embedded_candidates,
                    "right_ocr": right_diagnostics,
                    "embedded_ocr": embedded_diagnostics,
                }
            )

    embedded_best = [_best_label_candidate(row["embedded_candidates"]) for row in rows]
    compact_embedded = [
        candidate
        for candidate in embedded_best
        if candidate is not None and re.fullmatch(r"\d{1,3}[A-Za-z]", candidate.label)
    ]
    embedded_mode = (
        len(compact_embedded) >= max(3, math.ceil(len(rows) * 0.70))
        and len({candidate.label.casefold() for candidate in compact_embedded})
        == len(compact_embedded)
    )
    numeric_label_rows = sum(
        any(
            re.fullmatch(
                r"\s*[<>]?\s*\d+(?:\.\d+)?(?:\s*-\s*\d+(?:\.\d+)?)?\s*",
                candidate.label,
            )
            for candidate in row["right_candidates"]
        )
        for row in rows
    )
    numeric_mode = numeric_label_rows >= max(3, math.ceil(len(rows) * 0.70))

    entries: list[LegendEntry] = []
    label_diagnostics: list[dict[str, Any]] = []
    for row, embedded_candidate in zip(rows, embedded_best):
        candidates = row["embedded_candidates"] if embedded_mode else row["right_candidates"]
        selected = _best_label_candidate(candidates, numeric_mode=numeric_mode)
        label_diagnostics.append(
            {
                "index": row["index"],
                "column": row["column"],
                "row": row["row"],
                "mode": "embedded_in_swatch" if embedded_mode else "right_of_swatch",
                "selected": (
                    {
                        "label": selected.label,
                        "confidence": selected.confidence,
                        "bbox": list(selected.bbox),
                        "method": selected.method,
                    }
                    if selected is not None
                    else None
                ),
                "right_ocr": row["right_ocr"],
                "embedded_ocr": row["embedded_ocr"],
            }
        )
        if selected is None:
            continue
        swatch = row["swatch"]
        if swatch.detection_kind == "recovered_regular_gap":
            status = "recovered_regular_gap"
        elif swatch.detection_kind == "recovered_regular_grid_slot":
            status = "recovered_regular_grid_slot"
        else:
            status = "visible_regular_legend_component"
        cleaned_label = _clean_integrated_label(selected.label)
        if not _label_is_clean(cleaned_label):
            label_diagnostics[-1]["contamination_rejection"] = {
                "raw_label": selected.label,
                "cleaned_label": cleaned_label,
                "reason": "heading_or_border_furniture",
            }
            continue
        entries.append(
            LegendEntry(
                class_id=len(entries) + 1,
                label=cleaned_label,
                rgb=swatch.rgb,
                swatch_bbox=swatch.bbox,
                label_bbox=selected.bbox,
                ocr_confidence=selected.confidence,
                swatch_status=status,
            )
        )
    diagnostic_path = output_dir / "legend" / "label-region-ocr.json"
    diagnostic_path.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "label_mode": "embedded_in_swatch" if embedded_mode else "right_of_swatch",
                "rows": label_diagnostics,
            },
            indent=2,
        )
        + "\n"
    )
    missing = [item["index"] for item in label_diagnostics if item["selected"] is None]
    if missing:
        raise ValueError(
            "Legend swatches have no automatically readable OCR labels at rows "
            + ", ".join(map(str, missing))
        )
    if len(entries) != len(swatches):
        raise ValueError("Every detected legend swatch must have one readable OCR label")
    if numeric_mode:
        entries = _restore_ocr_decimal_points(entries)
        for diagnostic, entry in zip(label_diagnostics, entries):
            diagnostic["normalized_label"] = entry.label
    if len({entry.label.casefold() for entry in entries}) != len(entries):
        raise ValueError("Automatically detected legend contains duplicate labels")
    axis = float(np.median([item.bbox[0] + item.bbox[2] / 2.0 for item in swatches]))
    steps = [
        np.diff([item.center_y for item in column])
        for column in columns
        if len(column) > 1
    ]
    step = float(np.median(np.concatenate(steps)))
    return entries, axis, step, label_diagnostics


def _write_legend_artifacts(
    output_dir: Path,
    source_rgb: np.ndarray,
    entries: Sequence[LegendEntry],
    tsv_text: str,
    tesseract_version: str,
    swatch_axis_x: float,
    row_step_px: float,
    label_diagnostics: Sequence[Mapping[str, Any]] = (),
    texture_model: DitherTextureModel | None = None,
) -> tuple[Path, ...]:
    legend_dir = output_dir / "legend"
    legend_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = legend_dir / "ocr.tsv"
    tsv_path.write_text(tsv_text)
    json_path = legend_dir / "legend.json"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "method": "regular_swatch_sequence_plus_per_row_tesseract_labels",
        "tesseract_version": tesseract_version,
        "swatch_axis_x": swatch_axis_x,
        "row_step_px": row_step_px,
        "classification_method": (
            "local_texture_signature"
            if texture_model is not None
            else "legend_color_lab_prototypes"
        ),
        "texture_signature_artifact": (
            "dither-texture-signatures.json" if texture_model is not None else None
        ),
        "classification_domain": (
            "accepted_mapbox_state_land_minus_water_remapped_to_original_source"
        ),
        "near_white_class_ids": [
            entry.class_id
            for entry in entries
            if min(entry.rgb) >= 238 and max(entry.rgb) - min(entry.rgb) <= 10
        ],
        "label_diagnostics": list(label_diagnostics),
        "entries": [
            {
                "class_id": entry.class_id,
                "label": entry.label,
                "rgb": list(entry.rgb),
                "swatch_bbox": list(entry.swatch_bbox),
                "label_bbox": list(entry.label_bbox),
                "ocr_confidence": entry.ocr_confidence,
                "swatch_status": entry.swatch_status,
            }
            for entry in entries
        ],
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    preview = Image.fromarray(source_rgb.copy(), mode="RGB")
    draw = ImageDraw.Draw(preview)
    for entry in entries:
        x, y, width, height = entry.swatch_bbox
        draw.rectangle((x - 2, y - 2, x + width + 2, y + height + 2), outline=(0, 255, 120), width=3)
        x, y, width, height = entry.label_bbox
        draw.rectangle((x - 2, y - 2, x + width + 2, y + height + 2), outline=(0, 220, 255), width=3)
    preview_path = legend_dir / "legend-detection.png"
    preview.save(preview_path, optimize=True)
    return tsv_path, json_path, preview_path


def detect_legend(
    source_path: Path,
    source_rgb: np.ndarray,
    source_data_mask: np.ndarray,
    output_dir: Path,
) -> LegendDetection:
    words, tsv_text, version = _run_tesseract_ocr(source_path)
    legend_dir = output_dir / "legend"
    legend_dir.mkdir(parents=True, exist_ok=True)
    try:
        swatches, swatch_diagnostics = _automatic_swatch_sequence(source_rgb)
    except Exception as error:
        (legend_dir / "ocr.tsv").write_text(tsv_text)
        (legend_dir / "legend-rejection.json").write_text(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "stage": "swatch_sequence",
                    "error": f"{type(error).__name__}: {error}",
                },
                indent=2,
            )
            + "\n"
        )
        raise
    _write_swatch_sequence_artifacts(
        output_dir,
        source_rgb,
        swatches,
        swatch_diagnostics,
        tsv_text,
        version,
    )
    try:
        entries, axis, step, label_diagnostics = _detect_complete_legend(
            source_rgb, swatches, words, output_dir
        )
    except Exception as error:
        (legend_dir / "legend-rejection.json").write_text(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "stage": "readable_labels",
                    "selected_swatch_count": len(swatches),
                    "error": f"{type(error).__name__}: {error}",
                },
                indent=2,
            )
            + "\n"
        )
        raise
    texture_model: DitherTextureModel | None = None
    texture_path: Path | None = None
    if len({entry.rgb for entry in entries}) != len(entries):
        texture_model = build_dither_texture_model(
            source_rgb, [entry.swatch_bbox for entry in entries]
        )
        texture_path = legend_dir / "dither-texture-signatures.json"
        texture_path.write_text(json.dumps(texture_model.diagnostics(), indent=2) + "\n")
        if not texture_model.is_distinguishable:
            artifacts = _write_legend_artifacts(
                output_dir,
                source_rgb,
                entries,
                tsv_text,
                version,
                axis,
                step,
                label_diagnostics,
                texture_model,
            )
            (legend_dir / "legend-rejection.json").write_text(
                json.dumps(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "stage": "dither_texture_signatures",
                        "selected_swatch_count": len(swatches),
                        "ambiguous_pairs": [
                            {
                                "first_class_id": first,
                                "second_class_id": second,
                                "center_distance": distance,
                            }
                            for first, second, distance in texture_model.ambiguous_pairs
                        ],
                        "error": (
                            "Legend rows are not distinguishable in local texture/color "
                            "distribution space"
                        ),
                        "artifacts": [path.name for path in (*artifacts, texture_path)],
                    },
                    indent=2,
                )
                + "\n"
            )
            raise ValueError(
                "Automatically detected dither legend contains source-indistinguishable rows"
            )
    artifacts = _write_legend_artifacts(
        output_dir,
        source_rgb,
        entries,
        tsv_text,
        version,
        axis,
        step,
        label_diagnostics,
        texture_model,
    )
    if texture_path is not None:
        artifacts = (*artifacts, texture_path)
    return LegendDetection(tuple(entries), version, axis, step, artifacts, texture_model)


def _palette_lab(entries: Sequence[LegendEntry]) -> np.ndarray:
    rgb = np.asarray([entry.rgb for entry in entries], dtype=np.uint8)
    return cv2.cvtColor(rgb[None, :, :], cv2.COLOR_RGB2LAB)[0].astype(np.float32)


def _palette_chromaticity_compatible(
    lab: np.ndarray,
    class_ids: np.ndarray,
    palette_lab: np.ndarray,
    *,
    minimum_palette_chroma: float = MINIMUM_CHROMATIC_PALETTE_CHROMA,
    minimum_chroma_fraction: float = MINIMUM_CHROMATIC_EVIDENCE_FRACTION,
    minimum_direction_cosine: float = MINIMUM_CHROMATIC_DIRECTION_COSINE,
) -> np.ndarray:
    """Reject achromatic page/terrain ink mistaken for a chromatic class.

    A pale legend swatch can be close to gray in three-dimensional Lab distance.
    Learning that gray as a class prototype turns hillshade, water fills, and
    anti-aliased boundary ink into false thematic support.  Lab chroma and hue
    direction provide a generic fail-closed constraint: neutral legend classes
    remain unrestricted, while evidence for a chromatic class must retain a
    meaningful fraction of the swatch's chroma in roughly the same direction.

    ``class_ids == 0`` is always incompatible so callers can apply the result
    directly as a support mask.
    """

    if lab.shape[:2] != class_ids.shape:
        raise ValueError("Lab pixels and class ids must have the same image shape")
    compatible = class_ids > 0
    if not np.any(compatible):
        return compatible
    centered_palette = palette_lab[:, 1:3] - 128.0
    palette_chroma = np.linalg.norm(centered_palette, axis=1)
    centered_pixels = lab[..., 1:3] - 128.0
    pixel_chroma = np.linalg.norm(centered_pixels, axis=2)
    for class_index, (base_vector, base_chroma) in enumerate(
        zip(centered_palette, palette_chroma), 1
    ):
        selected = class_ids == class_index
        if not np.any(selected) or base_chroma < minimum_palette_chroma:
            continue
        selected_chroma = pixel_chroma[selected]
        chroma_ok = selected_chroma >= base_chroma * minimum_chroma_fraction
        direction = np.sum(centered_pixels[selected] * base_vector, axis=1) / np.maximum(
            selected_chroma * base_chroma, 1e-6
        )
        compatible[selected] = chroma_ok & (direction >= minimum_direction_cosine)
    return compatible


def _classify(
    rgb: np.ndarray,
    domain: np.ndarray,
    palette_lab: np.ndarray,
    maximum_distance: float,
    minimum_margin: float,
    *,
    chunk_size: int = 300_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    indices = np.flatnonzero(domain.ravel())
    class_ids = np.zeros(domain.shape, dtype=np.uint8)
    nearest_ids = np.zeros(domain.shape, dtype=np.uint8)
    nearest_distances = np.full(domain.shape, np.inf, dtype=np.float32)
    flat_class = class_ids.ravel()
    flat_nearest = nearest_ids.ravel()
    flat_distance = nearest_distances.ravel()
    for start in range(0, len(indices), chunk_size):
        selected = indices[start : start + chunk_size]
        distances = np.linalg.norm(
            lab[selected, None, :] - palette_lab[None, :, :], axis=2
        )
        best = np.argmin(distances, axis=1)
        best_distance = distances[np.arange(len(selected)), best]
        if palette_lab.shape[0] > 1:
            second = np.partition(distances, 1, axis=1)[:, 1]
        else:
            second = np.full(len(selected), np.inf)
        accepted = (best_distance <= maximum_distance) & (
            second - best_distance >= minimum_margin
        )
        flat_nearest[selected] = best + 1
        flat_distance[selected] = best_distance
        flat_class[selected[accepted]] = best[accepted] + 1
    return class_ids, nearest_ids, nearest_distances


def _learn_class_palette_prototypes(
    rgb: np.ndarray,
    domain: np.ndarray,
    entries: Sequence[LegendEntry],
    base_palette_lab: np.ndarray,
    *,
    seed_maximum_distance: float,
    maximum_prototypes_per_class: int = 64,
) -> tuple[tuple[np.ndarray, ...], dict[str, Any]]:
    """Model compression/dither variants seeded conservatively by legend color."""

    seed_ids, _, _ = _classify(
        rgb,
        domain,
        base_palette_lab,
        seed_maximum_distance,
        0.5,
    )
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    chromatic_seed = _palette_chromaticity_compatible(
        lab, seed_ids, base_palette_lab
    )
    prototypes: list[np.ndarray] = []
    diagnostics: list[dict[str, Any]] = []
    quantization = 4.0
    for entry, base in zip(entries, base_palette_lab):
        samples = lab[(seed_ids == entry.class_id) & chromatic_seed]
        if len(samples):
            bins = np.rint(samples / quantization).astype(np.int16)
            unique, counts = np.unique(bins, axis=0, return_counts=True)
            order = sorted(
                range(len(unique)),
                key=lambda index: (-int(counts[index]), tuple(int(value) for value in unique[index])),
            )[:maximum_prototypes_per_class]
            learned = unique[order].astype(np.float32) * quantization
            learned = np.clip(learned, 0.0, 255.0)
            values = np.concatenate((base[None, :], learned), axis=0)
        else:
            values = base[None, :]
        values = np.unique(values.astype(np.float32), axis=0)
        prototypes.append(values)
        diagnostics.append(
            {
                "class_id": entry.class_id,
                "label": entry.label,
                "seed_pixel_count": int(len(samples)),
                "prototype_count": int(len(values)),
                "prototype_lab": values.tolist(),
            }
        )
    return tuple(prototypes), {
        "method": "legend-seeded-source-lab-quantized-prototypes",
        "seed_maximum_lab_distance": seed_maximum_distance,
        "seed_minimum_lab_margin": 0.5,
        "lab_quantization_step": quantization,
        "maximum_prototypes_per_class": maximum_prototypes_per_class,
        "classes": diagnostics,
    }


def _classify_with_prototypes(
    rgb: np.ndarray,
    domain: np.ndarray,
    prototypes: Sequence[np.ndarray],
    maximum_distance: float,
    minimum_margin: float,
    *,
    chunk_size: int = 50_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.flatnonzero(domain.ravel())
    class_ids = np.zeros(domain.shape, dtype=np.uint8)
    nearest_ids = np.zeros(domain.shape, dtype=np.uint8)
    nearest_distances = np.full(domain.shape, np.inf, dtype=np.float32)
    if not len(indices):
        return class_ids, nearest_ids, nearest_distances

    # Map images frequently repeat a relatively small exact RGB vocabulary.
    # Classifying that vocabulary and expanding through the inverse map is
    # pixel-exact, unlike sampling, and bounds prototype-distance work for
    # pristine multi-megapixel sources.
    domain_rgb = rgb.reshape(-1, 3)[indices]
    unique_rgb, inverse = np.unique(domain_rgb, axis=0, return_inverse=True)
    lab = cv2.cvtColor(unique_rgb[:, None, :], cv2.COLOR_RGB2LAB)[
        :, 0, :
    ].astype(np.float32)
    unique_class = np.zeros(len(unique_rgb), dtype=np.uint8)
    unique_nearest = np.zeros(len(unique_rgb), dtype=np.uint8)
    unique_distance = np.full(len(unique_rgb), np.inf, dtype=np.float32)
    for start in range(0, len(unique_rgb), chunk_size):
        selected = np.arange(start, min(start + chunk_size, len(unique_rgb)))
        class_distances = np.empty((len(selected), len(prototypes)), dtype=np.float32)
        for class_index, class_prototypes in enumerate(prototypes):
            best = np.full(len(selected), np.inf, dtype=np.float32)
            # Bound the temporary tensor for palettes with many learned variants.
            for prototype_start in range(0, len(class_prototypes), 16):
                block = class_prototypes[prototype_start : prototype_start + 16]
                distances = np.linalg.norm(
                    lab[selected, None, :] - block[None, :, :], axis=2
                )
                best = np.minimum(best, np.min(distances, axis=1))
            class_distances[:, class_index] = best
        best = np.argmin(class_distances, axis=1)
        best_distance = class_distances[np.arange(len(selected)), best]
        if len(prototypes) > 1:
            second = np.partition(class_distances, 1, axis=1)[:, 1]
        else:
            second = np.full(len(selected), np.inf)
        accepted = (best_distance <= maximum_distance) & (
            second - best_distance >= minimum_margin
        )
        unique_nearest[selected] = best + 1
        unique_distance[selected] = best_distance
        unique_class[selected[accepted]] = best[accepted] + 1
    class_ids.ravel()[indices] = unique_class[inverse]
    nearest_ids.ravel()[indices] = unique_nearest[inverse]
    nearest_distances.ravel()[indices] = unique_distance[inverse]
    return class_ids, nearest_ids, nearest_distances


def _classify_semantic_evidence(
    rgb: np.ndarray,
    domain: np.ndarray,
    prototypes: Sequence[np.ndarray],
    texture_model: DitherTextureModel | None,
    maximum_distance: float,
    minimum_margin: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if texture_model is not None:
        return classify_dither_texture(
            rgb,
            domain,
            texture_model,
            maximum_distance,
            minimum_margin,
        )
    return _classify_with_prototypes(
        rgb, domain, prototypes, maximum_distance, minimum_margin
    )


def _nearest_completion(observed_ids: np.ndarray, domain: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    observed = observed_ids > 0
    if not np.any(observed):
        raise ValueError("No source pixels match the automatically detected legend")
    _, nearest = cv2.distanceTransformWithLabels(
        (~observed).astype(np.uint8),
        cv2.DIST_L2,
        5,
        labelType=cv2.DIST_LABEL_PIXEL,
    )
    # OpenCV labels zero pixels in row-major order from one. Build the same LUT.
    observed_values = observed_ids[observed]
    completed = observed_ids.copy()
    missing = domain & ~observed
    labels = nearest[missing]
    if np.any(labels <= 0) or int(labels.max(initial=0)) > len(observed_values):
        # scipy's exact nearest indices are a safe deterministic fallback for
        # unusual OpenCV builds with component rather than pixel labels.
        from scipy.ndimage import distance_transform_edt

        _, exact = distance_transform_edt(~observed, return_indices=True)
        completed[missing] = observed_ids[exact[0][missing], exact[1][missing]]
    else:
        completed[missing] = observed_values[labels - 1]
    inferred = missing & (completed > 0)
    completed[~domain] = 0
    return completed, inferred


def _small_enclosed_support_holes(
    support: np.ndarray,
    land_domain: np.ndarray,
    *,
    maximum_area: int = 50,
) -> np.ndarray:
    """Fill only tiny no-data components fully enclosed by thematic support."""

    missing = land_domain & ~support
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        missing.astype(np.uint8), 8
    )
    result = support.copy()
    height, width = support.shape
    for component in range(1, component_count):
        left, top, box_width, box_height, area = map(int, stats[component])
        if area > maximum_area:
            continue
        touches_image_edge = (
            left == 0 or top == 0 or left + box_width == width or top + box_height == height
        )
        if touches_image_edge:
            continue
        roi_left = left - 1
        roi_top = top - 1
        roi_right = left + box_width + 1
        roi_bottom = top + box_height + 1
        component_mask = (
            labels[roi_top:roi_bottom, roi_left:roi_right] == component
        )
        dilated = cv2.dilate(
            component_mask.astype(np.uint8), np.ones((3, 3), np.uint8)
        ) > 0
        boundary = dilated & ~component_mask
        land_roi = land_domain[roi_top:roi_bottom, roi_left:roi_right]
        support_roi = support[roi_top:roi_bottom, roi_left:roi_right]
        if not np.any(boundary & ~land_roi) and np.all(support_roi[boundary]):
            result_roi = result[roi_top:roi_bottom, roi_left:roi_right]
            result_roi[component_mask] = True
    return result


def _automatic_classification_domain(
    rgb: np.ndarray,
    mapbox_land_domain: np.ndarray,
    entries: Sequence[LegendEntry],
    prototypes: Sequence[np.ndarray],
    config: ExtractionLoopConfig,
    texture_model: DitherTextureModel | None = None,
) -> tuple[np.ndarray, dict[str, Any], np.ndarray, np.ndarray]:
    """Choose full-state or sparse thematic support without a source-type flag."""

    final_distance, final_margin = config.policies[-1]
    support_ids, _, _ = _classify_semantic_evidence(
        rgb,
        mapbox_land_domain,
        prototypes,
        texture_model,
        final_distance,
        final_margin,
    )
    chromaticity_diagnostics: dict[str, Any] = {
        "applied": texture_model is None,
        "minimum_palette_chroma": MINIMUM_CHROMATIC_PALETTE_CHROMA,
        "minimum_evidence_chroma_fraction": MINIMUM_CHROMATIC_EVIDENCE_FRACTION,
        "minimum_direction_cosine": MINIMUM_CHROMATIC_DIRECTION_COSINE,
        "direct_rejected_pixel_count": 0,
        "plausible_rejected_pixel_count": 0,
        "classes": [],
    }
    direct_chromatic_rejected_by_class: dict[int, int] = {}
    if texture_model is None:
        source_lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        palette_lab = _palette_lab(entries)
        support_chromatic = _palette_chromaticity_compatible(
            source_lab, support_ids, palette_lab
        )
        for entry in entries:
            direct_chromatic_rejected_by_class[entry.class_id] = int(
                np.count_nonzero((support_ids == entry.class_id) & ~support_chromatic)
            )
        chromaticity_diagnostics["direct_rejected_pixel_count"] = int(
            np.count_nonzero((support_ids > 0) & ~support_chromatic)
        )
        support_ids[~support_chromatic] = 0
    support = support_ids > 0
    plausible_ids, _, _ = _classify_semantic_evidence(
        rgb,
        mapbox_land_domain,
        prototypes,
        texture_model,
        config.meaningful_source_lab_distance,
        0.5,
    )
    plausible_chromatic_rejected_by_class: dict[int, int] = {}
    if texture_model is None:
        plausible_chromatic = _palette_chromaticity_compatible(
            source_lab, plausible_ids, palette_lab
        )
        for entry in entries:
            plausible_chromatic_rejected_by_class[entry.class_id] = int(
                np.count_nonzero(
                    (plausible_ids == entry.class_id) & ~plausible_chromatic
                )
            )
        chromaticity_diagnostics["plausible_rejected_pixel_count"] = int(
            np.count_nonzero((plausible_ids > 0) & ~plausible_chromatic)
        )
        plausible_ids[~plausible_chromatic] = 0
    plausible = plausible_ids > 0
    per_class_support: list[dict[str, Any]] = []
    for entry in entries:
        direct_count = int(np.count_nonzero(support_ids == entry.class_id))
        plausible_count = int(np.count_nonzero(plausible_ids == entry.class_id))
        direct_fraction = direct_count / max(plausible_count, 1)
        robust_support = bool(
            direct_count >= config.minimum_class_direct_support_pixels
            and plausible_count >= config.minimum_class_plausible_support_pixels
            and direct_fraction
            >= config.minimum_class_direct_fraction_of_plausible
        )
        direct_components, _, _, _ = cv2.connectedComponentsWithStats(
            (support_ids == entry.class_id).astype(np.uint8),
            connectivity=8,
        )
        independent_clusters = max(int(direct_components) - 1, 0)
        rare_spatial_support = bool(
            direct_count >= config.minimum_rare_class_direct_support_pixels
            and direct_fraction
            >= config.minimum_rare_class_direct_fraction_of_plausible
            and independent_clusters
            >= config.minimum_rare_class_independent_clusters
        )
        passed = robust_support or rare_spatial_support
        per_class_support.append(
            {
                "class_id": entry.class_id,
                "label": entry.label,
                "direct_pixel_count": direct_count,
                "plausible_pixel_count": plausible_count,
                "direct_fraction_of_plausible": direct_fraction,
                "independent_direct_cluster_count": independent_clusters,
                "support_tier": (
                    "robust"
                    if robust_support
                    else "rare_spatially_corroborated"
                    if rare_spatial_support
                    else "insufficient"
                ),
                "passed": passed,
            }
        )
        chromaticity_diagnostics["classes"].append(
            {
                "class_id": entry.class_id,
                "label": entry.label,
                "direct_rejected_pixel_count": direct_chromatic_rejected_by_class.get(
                    entry.class_id, 0
                ),
                "plausible_rejected_pixel_count": (
                    plausible_chromatic_rejected_by_class.get(entry.class_id, 0)
                ),
            }
        )
    land_count = int(np.count_nonzero(mapbox_land_domain))
    support_fraction = int(np.count_nonzero(support)) / max(land_count, 1)
    has_near_white_class = any(
        min(entry.rgb) >= 238 and max(entry.rgb) - min(entry.rgb) <= 10
        for entry in entries
    )
    sparse = not has_near_white_class and support_fraction < 0.50
    if sparse:
        domain = _small_enclosed_support_holes(plausible, mapbox_land_domain)
        kind = "automatic_sparse_thematic_support"
    else:
        domain = mapbox_land_domain.copy()
        kind = "accepted_mapbox_state_land_minus_water"
    return (
        domain,
        {
            "kind": kind,
            "classification_method": (
                "local_texture_signature"
                if texture_model is not None
                else "legend_seeded_lab_prototypes"
            ),
            "sparse_thematic_support": sparse,
            "has_near_white_legend_class": has_near_white_class,
            "mapbox_land_pixel_count": land_count,
            "final_policy_direct_support_pixel_count": int(np.count_nonzero(support)),
            "final_policy_direct_support_fraction_of_land": support_fraction,
            "broader_plausible_thematic_pixel_count": int(np.count_nonzero(plausible)),
            "broader_plausible_thematic_fraction_of_land": int(
                np.count_nonzero(plausible)
            )
            / max(land_count, 1),
            "classification_domain_pixel_count": int(np.count_nonzero(domain)),
            "maximum_enclosed_hole_area_px": 50 if sparse else 0,
            "chromaticity_gate": chromaticity_diagnostics,
            "per_class_support": {
                "passed": all(item["passed"] for item in per_class_support),
                "minimum_direct_pixels": config.minimum_class_direct_support_pixels,
                "minimum_plausible_pixels": (
                    config.minimum_class_plausible_support_pixels
                ),
                "minimum_direct_fraction_of_plausible": (
                    config.minimum_class_direct_fraction_of_plausible
                ),
                "rare_class_policy": {
                    "minimum_direct_pixels": (
                        config.minimum_rare_class_direct_support_pixels
                    ),
                    "minimum_direct_fraction_of_plausible": (
                        config.minimum_rare_class_direct_fraction_of_plausible
                    ),
                    "minimum_independent_clusters": (
                        config.minimum_rare_class_independent_clusters
                    ),
                },
                "classes": per_class_support,
                "failed_class_ids": [
                    item["class_id"]
                    for item in per_class_support
                    if not item["passed"]
                ],
            },
        },
        support,
        plausible,
    )


def _render(class_ids: np.ndarray, entries: Sequence[LegendEntry]) -> np.ndarray:
    rgb = np.zeros((*class_ids.shape, 3), dtype=np.uint8)
    for entry in entries:
        rgb[class_ids == entry.class_id] = entry.rgb
    return rgb


def _render_texture_classes(
    class_ids: np.ndarray,
    entries: Sequence[LegendEntry],
    source_rgb: np.ndarray,
) -> np.ndarray:
    """Render semantic ids with source-derived legend textures, not invented colors."""

    output = np.zeros((*class_ids.shape, 3), dtype=np.uint8)
    yy, xx = np.indices(class_ids.shape)
    for entry in entries:
        x, y, width, height = entry.swatch_bbox
        tile = source_rgb[y : y + height, x : x + width]
        if not tile.size:
            raise ValueError("dither legend tile is empty")
        mask = class_ids == entry.class_id
        output[mask] = tile[yy[mask] % height, xx[mask] % width]
    return output


def _texture_roundtrip_domain(class_ids: np.ndarray, window_size: int) -> np.ndarray:
    """Keep only pixels whose texture window belongs to one semantic class."""

    radius = window_size // 2
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
    interior = np.zeros(class_ids.shape, dtype=bool)
    for class_id in np.unique(class_ids):
        if not class_id:
            continue
        interior |= cv2.erode((class_ids == class_id).astype(np.uint8), kernel) > 0
    return interior


def _geographic_cell_metrics(
    domain: np.ndarray,
    observed: np.ndarray,
    rows: int,
    columns: int,
) -> list[dict[str, Any]]:
    height, width = domain.shape
    reports = []
    for row in range(rows):
        top, bottom = round(row * height / rows), round((row + 1) * height / rows)
        for column in range(columns):
            left, right = round(column * width / columns), round((column + 1) * width / columns)
            cell_domain = domain[top:bottom, left:right]
            count = int(np.count_nonzero(cell_domain))
            if not count:
                continue
            observed_count = int(np.count_nonzero(observed[top:bottom, left:right] & cell_domain))
            reports.append(
                {
                    "id": f"r{row + 1}-c{column + 1}",
                    "pixel_bounds": [left, top, right, bottom],
                    "domain_pixel_count": count,
                    "observed_pixel_count": observed_count,
                    "observed_fraction": observed_count / count,
                }
            )
    return reports


def _write_geographic_crops(
    iteration_dir: Path,
    reports: Sequence[Mapping[str, Any]],
    aligned_source: np.ndarray,
    reconstruction: np.ndarray,
    diff: np.ndarray,
) -> list[Path]:
    root = iteration_dir / "geographic-crops"
    root.mkdir()
    paths = []
    for report in reports:
        left, top, right, bottom = (int(value) for value in report["pixel_bounds"])
        source = aligned_source[top:bottom, left:right]
        extracted = reconstruction[top:bottom, left:right]
        overlay = np.where(diff[top:bottom, left:right, None], [255, 0, 255], extracted)
        montage = np.concatenate((source, extracted, overlay.astype(np.uint8)), axis=1)
        path = root / f"{report['id']}-source-extraction-diff.png"
        _save_rgb(path, montage)
        paths.append(path)
    return paths


def _write_native_target_crops(
    iteration_dir: Path,
    reports: Sequence[Mapping[str, Any]],
    source_rgb: np.ndarray,
    reference_to_source_remap: tuple[np.ndarray, np.ndarray],
    class_ids: np.ndarray,
    entries: Sequence[LegendEntry],
    *,
    maximum_dimension: int = 1024,
) -> tuple[Path, ...]:
    """Write bounded, native-grid comparison chips without a full RGB render."""

    if maximum_dimension < 256:
        raise ValueError("Native target crops must retain at least 256 pixels")
    map_x, map_y = reference_to_source_remap
    paths: list[Path] = []
    for report in reports:
        left, top, right, bottom = map(int, report["pixel_bounds"])
        cell_ids = class_ids[top:bottom, left:right]
        evidence_y, evidence_x = np.nonzero(cell_ids > 0)
        if not len(evidence_x):
            continue
        evidence_center_x = left + int(np.median(evidence_x))
        evidence_center_y = top + int(np.median(evidence_y))
        width, height = right - left, bottom - top
        if width > maximum_dimension:
            left = max(left, evidence_center_x - maximum_dimension // 2)
            left = min(left, right - maximum_dimension)
            right = left + maximum_dimension
        if height > maximum_dimension:
            top = max(top, evidence_center_y - maximum_dimension // 2)
            top = min(top, bottom - maximum_dimension)
            bottom = top + maximum_dimension
        aligned = cv2.remap(
            source_rgb,
            map_x[top:bottom, left:right],
            map_y[top:bottom, left:right],
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        extracted = _render(class_ids[top:bottom, left:right], entries)
        overlay = cv2.addWeighted(aligned, 0.5, extracted, 0.5, 0.0)
        montage = np.concatenate((aligned, extracted, overlay), axis=1)
        path = iteration_dir / f"{report['id']}-native-source-extraction-overlay.png"
        _save_rgb(path, montage)
        paths.append(path)
    return tuple(paths)


def run_automatic_categorical_extraction(
    source_path: Path,
    alignment_path: Path,
    mapbox_manifest_path: Path,
    output_dir: Path,
    experiment_log: NoHumanExperimentLog,
    experiment_markdown_path: Path,
    experiment_json_path: Path,
    *,
    config: ExtractionLoopConfig = ExtractionLoopConfig(),
    processing_reference_manifest_path: Path | None = None,
) -> AutomaticCategoricalExtractionResult:
    """Run immutable automatic attempts and record every attempt immediately."""

    source_path = source_path.resolve()
    alignment_path = alignment_path.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise ValueError("Automatic extraction requires a fresh output directory")
    if experiment_log.data["alignment"]["accepted_automatic_iteration_count"] is None:
        raise ValueError("Experiment log has no accepted automatic alignment")
    prior_iteration_count = _resumable_prior_extraction_count(experiment_log)
    base_reference = load_pinned_mapbox_reference(mapbox_manifest_path)
    accepted_iteration_count = experiment_log.data["alignment"][
        "accepted_automatic_iteration_count"
    ]
    alignment = _load_accepted_alignment(
        alignment_path,
        source_path,
        base_reference.grid,
        base_reference.pin,
        accepted_iteration_count,
        map_id=str(experiment_log.data["map_id"]),
        reference_revisions=experiment_log.data.get("mapbox_reference_revisions", []),
        alignment_iterations=experiment_log.data["alignment"]["iterations"],
    )
    reference = base_reference
    transform = alignment["transform"]
    if config.target_supersampling > 1:
        if processing_reference_manifest_path is None:
            raise ValueError(
                "A separately rasterized processing reference is required for "
                "target supersampling"
            )
        reference = load_pinned_mapbox_reference(processing_reference_manifest_path)
        _validate_supersampled_processing_reference(
            base_reference, reference, config.target_supersampling
        )
        transform = _supersample_alignment_transform(
            transform, config.target_supersampling
        )
    elif processing_reference_manifest_path is not None:
        raise ValueError(
            "A processing reference may be supplied only when target supersampling is enabled"
        )
    output_dir.mkdir(parents=True)

    source_rgb = np.asarray(Image.open(source_path).convert("RGB"))
    source_height, source_width = source_rgb.shape[:2]
    declared_shape = transform.get("source_original_shape")
    if declared_shape is not None and list(map(int, declared_shape)) != [
        source_height,
        source_width,
    ]:
        raise ValueError("Accepted alignment source dimensions disagree with the source")
    mapbox_source_land_domain = _source_data_mask(
        reference.state_land,
        reference.water,
        transform,
        (source_height, source_width),
    )
    mapbox_land_mask_path = output_dir / "source-mapbox-land-mask.png"
    _save_mask(mapbox_land_mask_path, mapbox_source_land_domain)

    legend = detect_legend(
        source_path, source_rgb, mapbox_source_land_domain, output_dir
    )
    if len(legend.entries) < config.minimum_legend_entries:
        raise ValueError("Automatically detected legend has too few entries")
    if legend.texture_model is None:
        palette_lab = _palette_lab(legend.entries)
        class_prototypes, prototype_diagnostics = _learn_class_palette_prototypes(
            source_rgb,
            mapbox_source_land_domain,
            legend.entries,
            palette_lab,
            seed_maximum_distance=config.meaningful_source_lab_distance,
        )
    else:
        class_prototypes = tuple(
            np.asarray([signature.center], dtype=np.float32)
            for signature in legend.texture_model.signatures
        )
        prototype_diagnostics = legend.texture_model.diagnostics()
    prototype_path = output_dir / "legend" / "classification-prototypes.json"
    prototype_path.write_text(json.dumps(prototype_diagnostics, indent=2) + "\n")
    (
        source_domain,
        domain_diagnostics,
        direct_support,
        plausible_support,
    ) = _automatic_classification_domain(
        source_rgb,
        mapbox_source_land_domain,
        legend.entries,
        class_prototypes,
        config,
        legend.texture_model,
    )
    layout = ~source_domain
    source_mask_path = output_dir / "source-data-mask.png"
    layout_path = output_dir / "source-layout-mask.png"
    direct_support_path = output_dir / "source-direct-thematic-support-mask.png"
    plausible_support_path = output_dir / "source-plausible-thematic-support-mask.png"
    domain_diagnostics_path = output_dir / "source-domain.json"
    _save_mask(source_mask_path, source_domain)
    _save_mask(layout_path, layout)
    _save_mask(direct_support_path, direct_support)
    _save_mask(plausible_support_path, plausible_support)
    excluded_plausible = plausible_support & ~source_domain
    source_support_audit_path = output_dir / "source-thematic-support-audit.png"
    support_preview = np.zeros_like(source_rgb)
    support_preview[plausible_support] = source_rgb[plausible_support]
    excluded_preview = support_preview.copy()
    excluded_preview[excluded_plausible] = (255, 0, 255)
    _save_rgb(
        source_support_audit_path,
        np.concatenate((source_rgb, support_preview, excluded_preview), axis=1),
    )
    excluded_plausible_count = int(np.count_nonzero(excluded_plausible))
    plausible_support_count = int(np.count_nonzero(plausible_support))
    excluded_plausible_fraction = excluded_plausible_count / max(
        plausible_support_count, 1
    )
    domain_diagnostics["excluded_plausible_thematic_pixel_count"] = excluded_plausible_count
    domain_diagnostics["excluded_plausible_thematic_fraction"] = excluded_plausible_fraction
    domain_diagnostics_path.write_text(json.dumps(domain_diagnostics, indent=2) + "\n")
    target_width, target_height = int(reference.grid["width"]), int(reference.grid["height"])
    source_to_reference_remap = _reference_to_source_remap(transform)
    target_source_coverage = _target_source_coverage_mask(
        transform,
        (source_height, source_width),
        source_to_reference_remap,
    )
    mapbox_target_data_domain = reference.state_land & ~reference.water
    missing_source_extent = mapbox_target_data_domain & ~target_source_coverage
    covered_target_data_domain = mapbox_target_data_domain & target_source_coverage
    mapbox_target_data_count = int(np.count_nonzero(mapbox_target_data_domain))
    covered_target_data_count = int(np.count_nonzero(covered_target_data_domain))
    missing_source_extent_count = int(np.count_nonzero(missing_source_extent))
    domain_diagnostics["aligned_source_extent"] = {
        "kind": "transform_derived_original_source_raster_footprint",
        "source_original_shape": [source_height, source_width],
        "mapbox_target_data_pixel_count": mapbox_target_data_count,
        "covered_mapbox_target_data_pixel_count": covered_target_data_count,
        "missing_source_extent_pixel_count": missing_source_extent_count,
        "covered_fraction": covered_target_data_count
        / max(mapbox_target_data_count, 1),
        "partial_extent": missing_source_extent_count > 0,
        "missing_extent_policy": "preserve_as_nodata_never_infer",
    }
    domain_diagnostics_path.write_text(json.dumps(domain_diagnostics, indent=2) + "\n")
    target_source_coverage_path = output_dir / "target-source-coverage-mask.png"
    missing_source_extent_path = output_dir / "target-missing-source-extent-mask.png"
    _save_mask(target_source_coverage_path, target_source_coverage)
    _save_mask(missing_source_extent_path, missing_source_extent)
    if domain_diagnostics["sparse_thematic_support"]:
        target_domain = _source_to_reference(
            source_domain.astype(np.uint8),
            transform,
            cv2.INTER_NEAREST,
            0,
            source_to_reference_remap,
        ) > 0
        target_domain &= covered_target_data_domain
    else:
        target_domain = covered_target_data_domain
    extent_cell_count = len(
        _geographic_cell_metrics(
            covered_target_data_domain,
            covered_target_data_domain,
            config.geographic_rows,
            config.geographic_columns,
        )
    )
    domain_diagnostics["aligned_source_extent"][
        "available_geographic_cell_count"
    ] = extent_cell_count
    domain_diagnostics_path.write_text(json.dumps(domain_diagnostics, indent=2) + "\n")
    # A supersampled aligned RGB raster is large.  Defer it until an accepted
    # attempt (or until the caller explicitly requests verbose rejected
    # artifacts) so automatic retries remain storage bounded.
    aligned_source: np.ndarray | None = None
    previous_ids: np.ndarray | None = None
    previous_observed: np.ndarray | None = None
    iterations: list[ExtractionIteration] = []
    accepted: ExtractionIteration | None = None

    for replay_number, (maximum_distance, minimum_margin) in enumerate(
        config.policies, 1
    ):
        iteration_number = _canonical_extraction_iteration_number(
            prior_iteration_count, replay_number
        )
        iteration_dir = output_dir / f"extraction-{iteration_number:02d}"
        iteration_dir.mkdir()
        observed_ids, nearest_color_ids, nearest_distances = _classify_semantic_evidence(
            source_rgb,
            source_domain,
            class_prototypes,
            legend.texture_model,
            maximum_distance,
            minimum_margin,
        )
        completed_ids, inferred = _nearest_completion(observed_ids, source_domain)
        observed = observed_ids > 0
        meaningful = source_domain & (
            nearest_distances <= config.meaningful_source_lab_distance
        )
        # A pixel can be plausible source evidence without meeting the stricter
        # observed threshold for this iteration.  Preserve its nearest legend
        # class directly; only pixels without meaningful color evidence use
        # spatial nearest completion.
        meaningful_inferred = meaningful & ~observed & (nearest_color_ids > 0)
        completed_ids[meaningful_inferred] = nearest_color_ids[meaningful_inferred]
        inferred = source_domain & ~observed & (completed_ids > 0)
        reconstruction = (
            _render_texture_classes(completed_ids, legend.entries, source_rgb)
            if legend.texture_model is not None
            else _render(completed_ids, legend.entries)
        )
        meaningful_mismatch = meaningful & (completed_ids != nearest_color_ids)
        # Reclassifying exact legend colors must reproduce the same semantic map.
        if legend.texture_model is None:
            roundtrip_ids, _, _ = _classify_with_prototypes(
                reconstruction, source_domain, class_prototypes, 0.01, 0.0
            )
            roundtrip_fixed = np.array_equal(roundtrip_ids, completed_ids)
        else:
            roundtrip_domain = _texture_roundtrip_domain(
                completed_ids, legend.texture_model.window_size
            )
            roundtrip_ids, _, _ = classify_dither_texture(
                reconstruction,
                roundtrip_domain,
                legend.texture_model,
                maximum_distance,
                minimum_margin,
            )
            roundtrip_fixed = bool(np.any(roundtrip_domain)) and np.array_equal(
                roundtrip_ids[roundtrip_domain], completed_ids[roundtrip_domain]
            )
        stable = (
            previous_ids is not None
            and previous_observed is not None
            and np.array_equal(previous_ids, completed_ids)
            and np.array_equal(previous_observed, observed)
        )

        web_ids = _source_to_reference(
            completed_ids,
            transform,
            cv2.INTER_NEAREST,
            0,
            source_to_reference_remap,
        )
        web_observed = _source_to_reference(
            observed.astype(np.uint8),
            transform,
            cv2.INTER_NEAREST,
            0,
            source_to_reference_remap,
        ) > 0
        web_ids[~target_domain] = 0
        if np.any(target_domain & (web_ids == 0)):
            web_ids, _ = _nearest_completion(web_ids, target_domain)
        web_observed &= target_domain
        web_inferred = target_domain & (web_ids > 0) & ~web_observed
        observed_count = int(np.count_nonzero(observed))
        domain_count = int(np.count_nonzero(source_domain))
        inferred_count = int(np.count_nonzero(inferred))
        meaningful_count = int(np.count_nonzero(meaningful))
        mismatch_count = int(np.count_nonzero(meaningful_mismatch))
        observed_fraction = observed_count / max(domain_count, 1)
        inferred_fraction = inferred_count / max(domain_count, 1)
        mismatch_fraction = mismatch_count / max(meaningful_count, 1)
        cell_reports = _geographic_cell_metrics(
            target_domain,
            web_observed,
            config.geographic_rows,
            config.geographic_columns,
        )
        passing_cells = sum(
            report["observed_fraction"] >= config.minimum_geographic_observed_fraction
            for report in cell_reports
        )
        required_geographic_cells = min(
            config.minimum_geographic_cells, extent_cell_count
        )
        class_counts = {
            entry.label: int(np.count_nonzero(observed_ids == entry.class_id))
            for entry in legend.entries
        }
        label_coverage = sum(bool(entry.label.strip()) for entry in legend.entries) / len(legend.entries)
        gates: dict[str, Any] = {
            "legend_entry_count": {
                "passed": len(legend.entries) >= config.minimum_legend_entries,
                "value": len(legend.entries),
                "minimum": config.minimum_legend_entries,
            },
            "legend_label_coverage": {
                "passed": label_coverage >= config.minimum_label_coverage,
                "value": label_coverage,
                "minimum": config.minimum_label_coverage,
            },
            "observed_source_coverage": {
                "passed": observed_fraction >= config.minimum_observed_fraction,
                "value": observed_fraction,
                "minimum": config.minimum_observed_fraction,
            },
            "inferred_source_fraction": {
                "passed": inferred_fraction <= config.maximum_inferred_fraction,
                "value": inferred_fraction,
                "maximum": config.maximum_inferred_fraction,
            },
            "meaningful_source_reconstruction_mismatch": {
                "passed": mismatch_fraction <= config.maximum_meaningful_source_mismatch_fraction,
                "value": mismatch_fraction,
                "maximum": config.maximum_meaningful_source_mismatch_fraction,
            },
            "source_domain_complete": bool(np.all(completed_ids[source_domain] > 0)),
            "source_layout_empty": not bool(np.any(completed_ids[layout] > 0)),
            "mapbox_water_and_exterior_empty": not bool(np.any(web_ids[~target_domain] > 0)),
            "missing_source_extent_remains_nodata": not bool(
                np.any(web_ids[missing_source_extent] > 0)
            ),
            "target_domain_complete": bool(np.all(web_ids[target_domain] > 0)),
            "sparse_plausible_thematic_completeness": {
                "passed": excluded_plausible_count == 0,
                "excluded_pixel_count": excluded_plausible_count,
                "excluded_fraction": excluded_plausible_fraction,
                "maximum_excluded_fraction": 0.0,
            },
            "per_class_prototype_support": dict(
                domain_diagnostics["per_class_support"]
            ),
            "roundtrip_reclassification_fixed_point": roundtrip_fixed,
            "successive_iteration_fixed_point": stable,
            "geographically_balanced_observation": {
                "passed": passing_cells >= required_geographic_cells,
                "value": passing_cells,
                "minimum": required_geographic_cells,
                "configured_full_extent_minimum": config.minimum_geographic_cells,
                "available_extent_cell_count": extent_cell_count,
            },
        }
        all_gates_pass = all(
            value if isinstance(value, bool) else bool(value["passed"])
            for value in gates.values()
        )
        decision = "accept" if all_gates_pass else "retry"

        source_ids_path = iteration_dir / "source-class-id.png"
        observed_path = iteration_dir / "source-observed-mask.png"
        inferred_path = iteration_dir / "source-inferred-mask.png"
        source_reconstruction_path = iteration_dir / "source-reconstruction.png"
        source_diff_path = iteration_dir / "source-semantic-diff-mask.png"
        web_ids_path = iteration_dir / "web-mercator-class-id.png"
        web_observed_path = iteration_dir / "web-mercator-observed-mask.png"
        web_inferred_path = iteration_dir / "web-mercator-inferred-mask.png"
        web_reconstruction_path = iteration_dir / "web-mercator-reconstruction.png"
        aligned_source_path = iteration_dir / "aligned-source.png"
        comparison_path = iteration_dir / "source-extraction-comparison.png"
        _save_ids(source_ids_path, completed_ids)
        _save_mask(source_diff_path, meaningful_mismatch)
        _save_ids(web_ids_path, web_ids)
        full_artifacts = decision == "accept" or not config.compact_rejected_artifacts
        if full_artifacts:
            _save_mask(observed_path, observed)
            _save_mask(inferred_path, inferred)
            _save_rgb(source_reconstruction_path, reconstruction)
            _save_mask(web_observed_path, web_observed)
            _save_mask(web_inferred_path, web_inferred)
            if config.compact_target_artifacts:
                crop_paths = _write_native_target_crops(
                    iteration_dir,
                    cell_reports,
                    source_rgb,
                    source_to_reference_remap,
                    web_ids,
                    legend.entries,
                )
                artifact_paths = (
                    source_ids_path,
                    observed_path,
                    inferred_path,
                    source_reconstruction_path,
                    source_diff_path,
                    web_ids_path,
                    web_observed_path,
                    web_inferred_path,
                    target_source_coverage_path,
                    missing_source_extent_path,
                    *crop_paths,
                )
            else:
                web_reconstruction = (
                    _render_texture_classes(web_ids, legend.entries, source_rgb)
                    if legend.texture_model is not None
                    else _render(web_ids, legend.entries)
                )
                _save_rgb(web_reconstruction_path, web_reconstruction)
                if aligned_source is None:
                    aligned_source = _source_to_reference(
                        source_rgb,
                        transform,
                        cv2.INTER_LINEAR,
                        (0, 0, 0),
                        source_to_reference_remap,
                    )
                _save_rgb(aligned_source_path, aligned_source)
                diff_rgb = web_reconstruction.copy()
                target_semantic_diff = target_domain & (web_ids == 0)
                diff_rgb[target_semantic_diff] = (255, 0, 255)
                _save_rgb(
                    comparison_path,
                    np.concatenate((aligned_source, web_reconstruction, diff_rgb), axis=1),
                )
                crop_paths = _write_geographic_crops(
                    iteration_dir,
                    cell_reports,
                    aligned_source,
                    web_reconstruction,
                    target_semantic_diff,
                )
                artifact_paths = (
                    source_ids_path,
                    observed_path,
                    inferred_path,
                    source_reconstruction_path,
                    source_diff_path,
                    web_ids_path,
                    web_observed_path,
                    web_inferred_path,
                    web_reconstruction_path,
                    aligned_source_path,
                    comparison_path,
                    target_source_coverage_path,
                    missing_source_extent_path,
                    *crop_paths,
                )
        else:
            artifact_paths = (
                source_ids_path,
                source_diff_path,
                web_ids_path,
                target_source_coverage_path,
                missing_source_extent_path,
            )
        report_path = iteration_dir / "iteration.json"
        scores = {
            "maximum_lab_distance": maximum_distance,
            "minimum_lab_margin": minimum_margin,
            "processing_target_grid": dict(reference.grid),
            "target_supersampling": config.target_supersampling,
            "source_domain_pixel_count": domain_count,
            "source_observed_pixel_count": observed_count,
            "source_inferred_pixel_count": inferred_count,
            "source_observed_fraction": observed_fraction,
            "source_inferred_fraction": inferred_fraction,
            "meaningful_source_pixel_count": meaningful_count,
            "meaningful_source_mismatch_pixel_count": mismatch_count,
            "meaningful_source_mismatch_fraction": mismatch_fraction,
            "legend_class_observed_pixel_counts": class_counts,
            "classification_domain": domain_diagnostics,
            "geographic_cells": cell_reports,
        }
        report = {
            "schema_version": SCHEMA_VERSION,
            "iteration": iteration_number,
            "decision": decision,
            "scores": scores,
            "gates": gates,
            "legend_sha256": _sha256(output_dir / "legend" / "legend.json"),
            "artifacts": [_artifact(path, output_dir) for path in artifact_paths],
        }
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        complete_artifacts = (*artifact_paths, report_path)
        experiment_log.record_extraction_iteration(
            scores=scores,
            gates=gates,
            decision=decision,
            provenance=automatic_provenance(
                PRODUCER,
                [
                    "original_source_image_pixels",
                    "accepted_automatic_alignment_transform",
                    "pinned_mapbox_land_and_water",
                    "automatically_detected_legend_swatch_pixels",
                    "automatically_detected_legend_ocr_labels",
                    "source_reconstruction_diff",
                ],
            ),
            method=(
                "legend-color Lab classification, nearest observed completion, "
                "source reconstruction diff, geographic holdouts, and fixed-point replay"
            ),
            artifacts=[
                {"path": str(path), "sha256": _sha256(path)}
                for path in complete_artifacts
            ],
        )
        experiment_log.write(experiment_markdown_path, experiment_json_path)
        iteration = ExtractionIteration(
            iteration_number,
            "accepted" if decision == "accept" else "retry",
            scores,
            gates,
            tuple(complete_artifacts),
        )
        iterations.append(iteration)
        previous_ids = completed_ids.copy()
        previous_observed = observed.copy()
        if decision == "accept":
            accepted = iteration
            break

    if accepted is not None:
        accepted_manifest_path = output_dir / "accepted-extraction.json"
        accepted_manifest_path.write_text(
            json.dumps(
                _accepted_extraction_payload(
                    accepted,
                    source_path,
                    alignment_path,
                    output_dir / "legend" / "legend.json",
                    processing_reference_manifest_path,
                ),
                indent=2,
            )
            + "\n"
        )
        experiment_log.finalize("complete")
        experiment_log.write(experiment_markdown_path, experiment_json_path)
        return AutomaticCategoricalExtractionResult(
            "accepted",
            "automatic source-diff, geographic, and fixed-point gates passed",
            legend,
            tuple(iterations),
            accepted,
        )

    blocker = (
        f"No extraction candidate passed all automatic gates after "
        f"{len(config.policies)} deterministic iterations"
    )
    experiment_log.finalize("blocked", blocker)
    experiment_log.write(experiment_markdown_path, experiment_json_path)
    return AutomaticCategoricalExtractionResult(
        "blocked", blocker, legend, tuple(iterations), None
    )
