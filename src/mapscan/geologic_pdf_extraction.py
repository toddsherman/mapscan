"""Source-clean categorical extraction for the native geologic PDF.

The general raster extractor is intentionally not used here.  This adapter
reads the legend and the thematic paths from the authoritative PDF content
stream, projects those native fills through the already accepted graticule
registration, and uses the pinned Mapbox masks only for land/water clipping.

There are no parameters for historical MapScan outputs, ``county.png``, Census
geometry, or human edits.  The first deterministic pass is retained as an
iteration; a byte-identical replay is required before the second pass can be
accepted.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import pdfplumber
from PIL import Image, ImageDraw
from pyproj import CRS, Transformer
from scipy.ndimage import distance_transform_edt

from .automatic_alignment_loop import load_pinned_mapbox_reference
from .automatic_categorical_extraction import (
    _load_accepted_alignment,
    _reference_to_source_remap,
    _source_to_reference,
    _web_mercator_to_reference_pixels,
)
from .experiment_log import NoHumanExperimentLog, automatic_provenance
from .pdf_legend import _legend_region, _swatches
from .pdf_vector_extraction import _color_key, _path_contours
from .source_working_raster import (
    WorkingRasterArtifact,
    load_working_raster_artifact,
)


SCHEMA_VERSION = "mapscan.geologic-pdf-vector-extraction.v1"
PRODUCER = "mapscan.geologic_pdf_extraction"
FORBIDDEN_PATH_TOKENS = (
    "automatic-alignment-orphaned-race",
    "county.png",
    "census",
    "manual",
    "legacy",
)

# Adobe Illustrator embedded this legacy geologic-age font without a Unicode
# mapping.  pdfplumber therefore returns the WinAnsi source characters below,
# even though the rendered glyphs are the standard readable unit codes.  The
# mapping is applied only after the exact embedded font family is verified.
GEOAGE_FULL_ALPHA_SYMBOL_MAP = {
    "@": "To",
    "@c": "Toc",
    "gr{": "grCz",
    "}gr": "grMz",
    "}v": "Mzv",
    "^": "Tr",
    "|gr": "grPz",
    "|": "Pz",
    "|v": "Pzv",
    "_": "Є",
    "=c": "pЄc",
    "=gr": "grpЄ",
    "=": "pЄ",
}


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


def _save_ids(path: Path, values: np.ndarray) -> None:
    if int(values.max(initial=0)) > 255:
        raise ValueError("geologic extraction supports at most 255 classes")
    Image.fromarray(values.astype(np.uint8), mode="L").save(path, optimize=True)


def _save_mask(path: Path, values: np.ndarray) -> None:
    Image.fromarray(values.astype(np.uint8) * 255, mode="L").save(path, optimize=True)


def _save_rgb(path: Path, values: np.ndarray) -> None:
    Image.fromarray(values.astype(np.uint8), mode="RGB").save(path, optimize=True)


def _slug(text: str, fallback: str) -> str:
    cleaned = "".join(character.lower() if character.isalnum() else "-" for character in text)
    cleaned = "-".join(part for part in cleaned.split("-") if part)
    return cleaned or fallback


@dataclass(frozen=True)
class GeologicLegendClass:
    class_id: int
    id: str
    native_text_code: str
    raw_pdf_text_code: str
    native_fontname: str
    label_decoding: str
    fill_cmyk: tuple[float, float, float, float]
    display_rgb: tuple[int, int, int]
    swatch_bbox_page_points: tuple[float, float, float, float]


@dataclass(frozen=True)
class NativeVectorEvidence:
    classes: tuple[GeologicLegendClass, ...]
    records: tuple[tuple[int, tuple[np.ndarray, ...]], ...]
    matched_fill_objects_by_class: Mapping[int, int]
    matched_stroke_objects_by_class: Mapping[int, int]
    matched_filled_object_count: int
    matched_contour_count: int
    unsupported_path_operators: tuple[str, ...]
    page_number: int
    total_curve_object_count: int
    total_line_object_count: int
    total_rect_object_count: int
    excluded_non_thematic_curve_object_count: int
    excluded_non_thematic_line_object_count: int
    excluded_non_thematic_rect_object_count: int


@dataclass(frozen=True)
class GeologicPdfExtractionConfig:
    minimum_legend_classes: int = 10
    required_replay_count: int = 2
    minimum_target_observed_fraction: float = 0.88
    maximum_target_inferred_fraction: float = 0.12
    maximum_visible_rgb_distance: float = 28.0
    minimum_source_roundtrip_agreement: float = 0.94
    geographic_rows: int = 4
    geographic_columns: int = 3
    minimum_cell_observed_fraction: float = 0.75
    minimum_cell_semantic_match_fraction: float = 0.92
    minimum_passing_geographic_cells: int = 8

    def __post_init__(self) -> None:
        if self.minimum_legend_classes < 2:
            raise ValueError("minimum_legend_classes must be at least two")
        if self.required_replay_count != 2:
            raise ValueError("the native vector fixed-point contract requires two passes")
        for value in (
            self.minimum_target_observed_fraction,
            self.maximum_target_inferred_fraction,
            self.minimum_source_roundtrip_agreement,
            self.minimum_cell_observed_fraction,
            self.minimum_cell_semantic_match_fraction,
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError("fraction gates must be between zero and one")


@dataclass(frozen=True)
class GeologicPdfExtractionIteration:
    iteration: int
    decision: str
    scores: Mapping[str, Any]
    gates: Mapping[str, Any]
    report_path: Path
    artifact_paths: tuple[Path, ...]


@dataclass(frozen=True)
class GeologicPdfExtractionResult:
    status: str
    stop_reason: str
    iterations: tuple[GeologicPdfExtractionIteration, ...]
    accepted_extraction_path: Path | None
    artifact_paths: tuple[Path, ...]

    @property
    def accepted(self) -> Path | None:
        return self.accepted_extraction_path


def _assert_clean_path(path: Path, kind: str) -> None:
    normalized = str(path.resolve()).lower()
    matched = next((token for token in FORBIDDEN_PATH_TOKENS if token in normalized), None)
    if matched:
        raise ValueError(f"{kind} path contains forbidden no-human evidence: {matched}")


def _validate_inputs(
    source_adapter_manifest_path: Path,
    accepted_alignment_path: Path,
    mapbox_manifest_path: Path,
    experiment_log: NoHumanExperimentLog,
) -> tuple[WorkingRasterArtifact, Any, dict[str, Any]]:
    for path, kind in (
        (source_adapter_manifest_path, "source adapter"),
        (accepted_alignment_path, "accepted alignment"),
        (mapbox_manifest_path, "Mapbox manifest"),
    ):
        _assert_clean_path(path, kind)
    working = load_working_raster_artifact(source_adapter_manifest_path)
    if working.source_path.suffix.lower() != ".pdf":
        raise ValueError("native geologic extraction requires the authoritative PDF")
    pdf = working.manifest.get("pdf", {})
    if pdf.get("vector_evidence", {}).get("status") != "available":
        raise ValueError("source-clean adapter did not preserve native vector evidence")
    if int(pdf.get("selected_page_number", 0)) < 1:
        raise ValueError("source-clean adapter does not pin a PDF page")

    reference = load_pinned_mapbox_reference(mapbox_manifest_path)
    accepted_count = experiment_log.data["alignment"][
        "accepted_automatic_iteration_count"
    ]
    if accepted_count is None:
        raise ValueError("extraction requires an accepted automatic alignment")
    if experiment_log.data["extraction"]["iterations"]:
        raise ValueError("throwaway extraction log already contains attempts")
    alignment = _load_accepted_alignment(
        accepted_alignment_path.resolve(),
        working.working_raster_path,
        reference.grid,
        reference.pin,
        accepted_iteration_count=int(accepted_count),
        map_id=str(experiment_log.data["map_id"]),
        reference_revisions=experiment_log.data.get("mapbox_reference_revisions", []),
        alignment_iterations=experiment_log.data["alignment"]["iterations"],
    )
    provenance = alignment.get("exact_transform_provenance", {})
    if provenance.get("producer") != "mapscan.geologic_pdf_alignment":
        raise ValueError("accepted transform was not produced by the native PDF aligner")
    if provenance.get("fit_evidence") != "original_pdf_native_vector_graticule_only":
        raise ValueError("accepted transform did not use only the native PDF graticule")
    if provenance.get("mapbox_role") != "independent_validation_only":
        raise ValueError("accepted transform has an invalid Mapbox evidence role")
    original = provenance.get("original_pdf", {})
    if original.get("sha256") != working.source_sha256:
        raise ValueError("accepted transform original-PDF hash mismatch")
    adapter_record = provenance.get("source_adapter_manifest", {})
    if adapter_record.get("sha256") != _sha256(working.manifest_path):
        raise ValueError("accepted transform source-adapter hash mismatch")
    return working, reference, alignment


def _establish_legend_classes(
    swatches: Sequence[Mapping[str, Any]],
    *,
    minimum_classes: int,
) -> tuple[GeologicLegendClass, ...]:
    if len(swatches) < minimum_classes:
        raise ValueError("native PDF legend has too few categorical swatches")
    labels: list[str] = []
    raw_labels: list[str] = []
    fontnames: list[str] = []
    decoding_methods: list[str] = []
    colors: list[tuple[float, ...]] = []
    for swatch in swatches:
        raw_label = str(swatch.get("raw_pdf_text") or "").strip()
        fontname = str(swatch.get("native_fontname") or "")
        label, decoding = _decode_native_legend_label(raw_label, fontname)
        color = _color_key(swatch.get("fill_cmyk"))
        if not label:
            raise ValueError("native PDF legend contains an unreadable class code")
        if color is None or len(color) != 4:
            raise ValueError("native PDF legend contains a non-CMYK class swatch")
        labels.append(label)
        raw_labels.append(raw_label)
        fontnames.append(fontname)
        decoding_methods.append(decoding)
        colors.append(color)
    if len(set(labels)) != len(labels):
        raise ValueError("native PDF legend class codes are not unique")
    if len(set(colors)) != len(colors):
        raise ValueError("native PDF legend colors are not unique")

    used: set[str] = set()
    classes = []
    for index, (swatch, label, raw_label, fontname, decoding, color) in enumerate(
        zip(swatches, labels, raw_labels, fontnames, decoding_methods, colors), 1
    ):
        base = _slug(label, f"unit-{index:03d}")
        identifier = base
        suffix = 2
        while identifier in used:
            identifier = f"{base}-{suffix}"
            suffix += 1
        used.add(identifier)
        bbox = tuple(float(value) for value in swatch["bbox_page_points"])
        rgb = tuple(int(value) for value in swatch["fill_rgb"])
        classes.append(
            GeologicLegendClass(
                class_id=index,
                id=identifier,
                native_text_code=label,
                raw_pdf_text_code=raw_label,
                native_fontname=fontname,
                label_decoding=decoding,
                fill_cmyk=tuple(float(value) for value in color),
                display_rgb=rgb,
                swatch_bbox_page_points=bbox,
            )
        )
    return tuple(classes)


def _decode_native_legend_label(raw_label: str, fontname: str) -> tuple[str, str]:
    if not raw_label:
        return "", "unreadable"
    if raw_label in GEOAGE_FULL_ALPHA_SYMBOL_MAP:
        if "GeoageFullAlpha" not in fontname:
            raise ValueError(
                "legacy geologic symbol code occurred without the pinned embedded font"
            )
        return (
            GEOAGE_FULL_ALPHA_SYMBOL_MAP[raw_label],
            "embedded_geoage_full_alpha_symbol_map_v1",
        )
    if any(character in raw_label for character in "@^|_={}"):
        raise ValueError(
            f"unmapped embedded geologic legend glyph code: {raw_label!r}"
        )
    return raw_label, "native_pdf_unicode_text"


def _swatch_fontname(page: Any, bbox: Sequence[float]) -> str:
    x0, top, x1, bottom = (float(value) for value in bbox)
    fonts: dict[str, int] = {}
    for character in page.chars:
        center_x = (float(character["x0"]) + float(character["x1"])) / 2.0
        center_y = (float(character["top"]) + float(character["bottom"])) / 2.0
        if x0 <= center_x <= x1 and top <= center_y <= bottom:
            name = str(character.get("fontname") or "")
            fonts[name] = fonts.get(name, 0) + 1
    if not fonts:
        raise ValueError("native PDF legend swatch has no embedded text font evidence")
    return max(fonts, key=lambda name: (fonts[name], name))


def _rendered_swatch_rgb(
    source_rgb: np.ndarray,
    bbox_page_points: Sequence[float],
    scale_x: float,
    scale_y: float,
    expected_rgb: Sequence[int] | None = None,
) -> tuple[int, int, int]:
    """Return the dominant rendered swatch color, excluding its thin border.

    The PDF declares CMYK colors, while Poppler applies the document's color
    conversion when producing the authoritative working RGB raster.  Sampling
    the uniform swatch interior preserves that actual conversion and makes the
    source-diff compare like with like.  Text glyphs cannot win because their
    dark pixels occupy much less area than the background fill.
    """

    x0, y0, x1, y1 = (float(value) for value in bbox_page_points)
    left, right = round(x0 * scale_x), round(x1 * scale_x)
    top, bottom = round(y0 * scale_y), round(y1 * scale_y)
    inset_x = max(1, round((right - left) * 0.08))
    inset_y = max(1, round((bottom - top) * 0.12))
    crop = source_rgb[
        top + inset_y : bottom - inset_y,
        left + inset_x : right - inset_x,
    ]
    if not crop.size:
        raise ValueError("native PDF legend swatch has an empty rendered interior")
    colors, counts = np.unique(crop.reshape(-1, 3), axis=0, return_counts=True)
    if expected_rgb is None:
        index = int(np.argmax(counts))
    else:
        # Retain only recurring flat/background colors, then use the declared
        # CMYK approximation to distinguish that background from dark glyphs.
        recurring = counts >= max(2, int(np.max(counts) * 0.20))
        candidates = colors[recurring].astype(np.int32)
        expected = np.asarray(expected_rgb, dtype=np.int32)
        distances = np.sum((candidates - expected) ** 2, axis=1)
        selected = candidates[int(np.argmin(distances))]
        matches = np.all(colors == selected, axis=1)
        index = int(np.flatnonzero(matches)[0])
    return tuple(int(value) for value in colors[index])


def _read_native_vector_evidence(
    working: WorkingRasterArtifact,
    config: GeologicPdfExtractionConfig,
) -> NativeVectorEvidence:
    pdf_record = working.manifest["pdf"]
    page_number = int(pdf_record["selected_page_number"])
    conversion = working.manifest["conversion"]
    scale_x = float(conversion["actual_x_pixels_per_page_point"])
    scale_y = float(conversion["actual_y_pixels_per_page_point"])
    if min(scale_x, scale_y) <= 0 or not np.isfinite([scale_x, scale_y]).all():
        raise ValueError("source-clean PDF raster scale is invalid")
    source_rgb = np.asarray(Image.open(working.working_raster_path).convert("RGB"))

    with pdfplumber.open(working.source_path) as pdf:
        if page_number > len(pdf.pages):
            raise ValueError("source-clean PDF page is no longer available")
        page = pdf.pages[page_number - 1]
        swatches = [dict(value) for value in _swatches(page, _legend_region(page))]
        for swatch in swatches:
            swatch["native_fontname"] = _swatch_fontname(
                page, swatch["bbox_page_points"]
            )
            swatch["fill_rgb"] = list(
                _rendered_swatch_rgb(
                    source_rgb,
                    swatch["bbox_page_points"],
                    scale_x,
                    scale_y,
                    swatch["fill_rgb"],
                )
            )
        classes = _establish_legend_classes(
            swatches, minimum_classes=config.minimum_legend_classes
        )
        color_to_class = {
            _color_key(category.fill_cmyk): category.class_id for category in classes
        }
        fill_counts = {category.class_id: 0 for category in classes}
        stroke_counts = {category.class_id: 0 for category in classes}
        records: list[tuple[int, tuple[np.ndarray, ...]]] = []
        unsupported: set[str] = set()
        contour_count = 0
        curves = page.objects.get("curve", [])
        lines = page.objects.get("line", [])
        rects = page.objects.get("rect", [])
        excluded_curve_count = 0
        for curve in curves:
            fill_class = color_to_class.get(_color_key(curve.get("non_stroking_color")))
            stroke_class = color_to_class.get(_color_key(curve.get("stroking_color")))
            if stroke_class is not None:
                stroke_counts[stroke_class] += 1
            if not curve.get("fill") or fill_class is None:
                excluded_curve_count += 1
                continue
            contours, operators = _path_contours(
                curve.get("path", []), scale_x, scale_y
            )
            unsupported.update(operators)
            if not contours:
                excluded_curve_count += 1
                continue
            immutable = tuple(contour.copy() for contour in contours)
            records.append((fill_class, immutable))
            fill_counts[fill_class] += 1
            contour_count += len(immutable)

    missing = [
        category.native_text_code
        for category in classes
        if fill_counts[category.class_id] == 0
    ]
    if missing:
        raise ValueError(
            "legend semantics cannot be established because native fills are missing: "
            + ", ".join(missing)
        )
    return NativeVectorEvidence(
        classes=classes,
        records=tuple(records),
        matched_fill_objects_by_class=fill_counts,
        matched_stroke_objects_by_class=stroke_counts,
        matched_filled_object_count=len(records),
        matched_contour_count=contour_count,
        unsupported_path_operators=tuple(sorted(unsupported)),
        page_number=page_number,
        total_curve_object_count=len(curves),
        total_line_object_count=len(lines),
        total_rect_object_count=len(rects),
        excluded_non_thematic_curve_object_count=excluded_curve_count,
        excluded_non_thematic_line_object_count=len(lines),
        excluded_non_thematic_rect_object_count=len(rects),
    )


def _rasterize_native_fills(
    records: Sequence[tuple[int, Sequence[np.ndarray]]],
    shape: tuple[int, int],
) -> np.ndarray:
    class_ids = np.zeros(shape, dtype=np.uint8)
    for class_id, contours in records:
        cv2.fillPoly(class_ids, list(contours), int(class_id), lineType=cv2.LINE_8)
    return class_ids


def _nearest_completion(
    observed_ids: np.ndarray, domain: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    observed = domain & (observed_ids > 0)
    if not np.any(observed):
        raise ValueError("native vector fills do not intersect Mapbox land")
    missing = domain & ~observed
    complete = observed_ids.copy()
    _, indices = distance_transform_edt(~observed, return_indices=True)
    complete[missing] = observed_ids[indices[0][missing], indices[1][missing]]
    complete[~domain] = 0
    inferred = missing & (complete > 0)
    return complete, inferred


def _render(ids: np.ndarray, classes: Sequence[GeologicLegendClass]) -> np.ndarray:
    rgb = np.zeros((*ids.shape, 3), dtype=np.uint8)
    for category in classes:
        rgb[ids == category.class_id] = category.display_rgb
    return rgb


def _classify_visible_palette(
    source_rgb: np.ndarray,
    classes: Sequence[GeologicLegendClass],
    domain: np.ndarray,
    *,
    rows_per_chunk: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Return nearest legend class and RGB distance for each domain pixel."""

    if rows_per_chunk < 1:
        raise ValueError("rows_per_chunk must be positive")
    height, width = domain.shape
    nearest_ids = np.zeros((height, width), dtype=np.uint8)
    nearest_distance = np.full((height, width), np.inf, dtype=np.float32)
    palette = [
        (category.class_id, np.asarray(category.display_rgb, dtype=np.int32))
        for category in classes
    ]
    for top in range(0, height, rows_per_chunk):
        bottom = min(height, top + rows_per_chunk)
        pixels = source_rgb[top:bottom].astype(np.int32)
        best_squared = np.full((bottom - top, width), np.iinfo(np.int32).max, np.int32)
        best_ids = np.zeros((bottom - top, width), dtype=np.uint8)
        for class_id, color in palette:
            delta = pixels - color
            squared = np.sum(delta * delta, axis=2, dtype=np.int32)
            better = squared < best_squared
            best_squared[better] = squared[better]
            best_ids[better] = class_id
        cell_domain = domain[top:bottom]
        nearest_ids[top:bottom][cell_domain] = best_ids[cell_domain]
        nearest_distance[top:bottom][cell_domain] = np.sqrt(
            best_squared[cell_domain].astype(np.float32)
        )
    return nearest_ids, nearest_distance


def _source_pixels_to_reference_chunk(
    transform: Mapping[str, Any],
    top: int,
    bottom: int,
    source_width: int,
) -> tuple[np.ndarray, np.ndarray]:
    x_values = np.arange(source_width, dtype=np.float64)
    pixel_x, pixel_y = np.meshgrid(
        x_values, np.arange(top, bottom, dtype=np.float64)
    )
    points = np.stack((pixel_x, pixel_y), axis=-1)
    if transform["kind"] == "regular_global_mapbox_registration":
        matrix = np.asarray(
            transform["source_original_to_reference_pixel_matrix"], dtype=np.float64
        )
        mapped = cv2.perspectiveTransform(points.reshape(-1, 1, 2), matrix).reshape(
            bottom - top, source_width, 2
        )
        return mapped[..., 0].astype(np.float32), mapped[..., 1].astype(np.float32)

    projection = transform["projection"]
    matrix = np.asarray(
        transform["source_original_to_candidate_normalized_matrix"], dtype=np.float64
    )
    normalized = points @ matrix[:2, :2].T + matrix[:2, 2]
    center = np.asarray(projection["normalization_center"], dtype=np.float64)
    scale = float(projection["normalization_scale"])
    projected_x = normalized[..., 0] * scale + center[0]
    projected_y = -normalized[..., 1] * scale + center[1]
    transformer = Transformer.from_crs(
        CRS.from_wkt(projection["crs_wkt"]), "EPSG:3857", always_xy=True
    )
    web_x, web_y = transformer.transform(projected_x, projected_y, errcheck=True)
    map_x, map_y = _web_mercator_to_reference_pixels(
        web_x, web_y, transform["target_grid"]
    )
    if not np.isfinite(map_x).all() or not np.isfinite(map_y).all():
        raise ValueError("source reconstruction produced non-finite target coordinates")
    return np.asarray(map_x, np.float32), np.asarray(map_y, np.float32)


def _warp_target_to_source_chunked(
    target: np.ndarray,
    transform: Mapping[str, Any],
    source_shape: tuple[int, int],
    *,
    rows_per_chunk: int = 128,
) -> np.ndarray:
    source_height, source_width = source_shape
    if rows_per_chunk < 1:
        raise ValueError("rows_per_chunk must be positive")
    output_shape = (source_height, source_width) + target.shape[2:]
    output = np.empty(output_shape, dtype=target.dtype)
    for top in range(0, source_height, rows_per_chunk):
        bottom = min(source_height, top + rows_per_chunk)
        map_x, map_y = _source_pixels_to_reference_chunk(
            transform, top, bottom, source_width
        )
        output[top:bottom] = cv2.remap(
            target,
            map_x,
            map_y,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    return output


def _geographic_metrics(
    domain: np.ndarray,
    observed: np.ndarray,
    visible_match: np.ndarray,
    visible: np.ndarray,
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
            domain_count = int(np.count_nonzero(cell_domain))
            if not domain_count:
                continue
            cell_observed = observed[top:bottom, left:right] & cell_domain
            observed_count = int(np.count_nonzero(cell_observed))
            cell_visible = visible[top:bottom, left:right] & cell_observed
            visible_count = int(np.count_nonzero(cell_visible))
            match_count = int(
                np.count_nonzero(
                    visible_match[top:bottom, left:right] & cell_visible
                )
            )
            reports.append(
                {
                    "id": f"r{row + 1}-c{column + 1}",
                    "pixel_bounds": [left, top, right, bottom],
                    "domain_pixel_count": domain_count,
                    "observed_pixel_count": observed_count,
                    "observed_fraction": observed_count / domain_count,
                    "source_visible_pixel_count": visible_count,
                    "source_visible_match_fraction": match_count / max(visible_count, 1),
                }
            )
    return reports


def _source_geographic_roundtrip_metrics(
    expected: np.ndarray,
    matched: np.ndarray,
    transform: Mapping[str, Any],
    target_shape: tuple[int, int],
    rows: int,
    columns: int,
    *,
    rows_per_chunk: int = 128,
) -> list[dict[str, Any]]:
    """Score source-native roundtrip pixels in fixed Mapbox geographic cells."""

    source_height, source_width = expected.shape
    target_height, target_width = target_shape
    counts = np.zeros(rows * columns, dtype=np.int64)
    matches = np.zeros(rows * columns, dtype=np.int64)
    for top in range(0, source_height, rows_per_chunk):
        bottom = min(source_height, top + rows_per_chunk)
        map_x, map_y = _source_pixels_to_reference_chunk(
            transform, top, bottom, source_width
        )
        cell_expected = expected[top:bottom]
        inside = (
            cell_expected
            & (map_x >= 0)
            & (map_x < target_width)
            & (map_y >= 0)
            & (map_y < target_height)
        )
        if not np.any(inside):
            continue
        column = np.minimum(
            (map_x[inside] / target_width * columns).astype(np.int32), columns - 1
        )
        row = np.minimum(
            (map_y[inside] / target_height * rows).astype(np.int32), rows - 1
        )
        cell_ids = row * columns + column
        counts += np.bincount(cell_ids, minlength=rows * columns)
        matched_values = matched[top:bottom][inside]
        matches += np.bincount(
            cell_ids[matched_values], minlength=rows * columns
        )
    reports = []
    for row in range(rows):
        for column in range(columns):
            index = row * columns + column
            if not counts[index]:
                continue
            reports.append(
                {
                    "id": f"r{row + 1}-c{column + 1}",
                    "source_semantic_expected_pixel_count": int(counts[index]),
                    "source_semantic_match_pixel_count": int(matches[index]),
                    "source_semantic_match_fraction": float(
                        matches[index] / counts[index]
                    ),
                }
            )
    return reports


def _diagnostic(
    aligned_source: np.ndarray,
    reconstruction: np.ndarray,
    inferred: np.ndarray,
    *,
    maximum_height: int = 1600,
) -> np.ndarray:
    # The full PDF contains intentional linework and relief that are explicitly
    # excluded from the categorical product.  A 50/50 visual blend makes spatial
    # drift and gross class disagreement visible without falsely painting that
    # legitimate overprint as categorical failure.
    review = cv2.addWeighted(aligned_source, 0.5, reconstruction, 0.5, 0.0)
    review[inferred] = (0, 255, 255)
    scale = min(1.0, maximum_height / aligned_source.shape[0])
    if scale < 1.0:
        size = (
            max(1, round(aligned_source.shape[1] * scale)),
            max(1, round(aligned_source.shape[0] * scale)),
        )
        panels = [
            cv2.resize(value, size, interpolation=cv2.INTER_AREA)
            for value in (aligned_source, reconstruction, review)
        ]
    else:
        panels = [aligned_source, reconstruction, review]
    result = np.concatenate(panels, axis=1)
    canvas = Image.fromarray(result)
    draw = ImageDraw.Draw(canvas)
    labels = (
        "aligned source",
        "native-vector extraction",
        "50/50 source-extraction; cyan inferred",
    )
    panel_width = panels[0].shape[1]
    for index, label in enumerate(labels):
        x = index * panel_width + 12
        draw.rectangle((x - 4, 8, x + 250, 36), fill=(0, 0, 0))
        draw.text((x, 12), label, fill=(255, 255, 255))
    return np.asarray(canvas)


def _legend_payload(
    evidence: NativeVectorEvidence,
    working: WorkingRasterArtifact,
) -> dict[str, Any]:
    categories = []
    for category in evidence.classes:
        categories.append(
            {
                "class_id": category.class_id,
                "id": category.id,
                "label": category.native_text_code,
                "raw_pdf_text_code": category.raw_pdf_text_code,
                "native_fontname": category.native_fontname,
                "label_decoding": category.label_decoding,
                "semantic_evidence": (
                    "native_text_object_inside_legend_swatch_with_verified_embedded_font"
                ),
                "fill_cmyk": list(category.fill_cmyk),
                "display_rgb": list(category.display_rgb),
                "swatch_bbox_page_points": list(category.swatch_bbox_page_points),
                "matched_native_fill_object_count": evidence.matched_fill_objects_by_class[
                    category.class_id
                ],
                "matched_native_stroke_object_count": evidence.matched_stroke_objects_by_class[
                    category.class_id
                ],
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "semantics_established",
        "page_number": evidence.page_number,
        "class_count": len(categories),
        "semantic_contract": (
            "Each class is keyed by the unique native PDF text object printed "
            "inside its unique CMYK legend swatch. Legacy GeoageFullAlpha glyph "
            "codes are decoded only by the pinned embedded-font symbol map; "
            "unmapped symbols reject the run."
        ),
        "source": {"path": str(working.source_path), "sha256": working.source_sha256},
        "categories": categories,
    }


def run_geologic_pdf_vector_extraction(
    source_adapter_manifest_path: Path,
    accepted_alignment_path: Path,
    mapbox_manifest_path: Path,
    output_dir: Path,
    experiment_log: NoHumanExperimentLog,
    experiment_markdown_path: Path,
    experiment_json_path: Path,
    *,
    config: GeologicPdfExtractionConfig = GeologicPdfExtractionConfig(),
) -> GeologicPdfExtractionResult:
    """Run the immutable native-vector extraction and fixed-point replay."""

    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise ValueError("geologic PDF extraction requires a fresh output directory")
    working, reference, alignment = _validate_inputs(
        source_adapter_manifest_path.resolve(),
        accepted_alignment_path.resolve(),
        mapbox_manifest_path.resolve(),
        experiment_log,
    )
    output_dir.mkdir(parents=True)
    evidence = _read_native_vector_evidence(working, config)
    legend_root = output_dir / "legend"
    legend_root.mkdir()
    legend_path = legend_root / "legend.json"
    legend_path.write_text(json.dumps(_legend_payload(evidence, working), indent=2) + "\n")

    source_rgb = np.asarray(Image.open(working.working_raster_path).convert("RGB"))
    source_shape = (working.height, working.width)
    transform = alignment["transform"]
    target_domain = reference.state_land & ~reference.water
    reference_to_source = _reference_to_source_remap(transform)
    aligned_source = _source_to_reference(
        source_rgb,
        transform,
        cv2.INTER_LINEAR,
        (0, 0, 0),
        reference_to_source,
    )
    source_domain = _warp_target_to_source_chunked(
        target_domain.astype(np.uint8), transform, source_shape
    ) > 0

    previous_source_ids: np.ndarray | None = None
    previous_target_ids: np.ndarray | None = None
    iterations: list[GeologicPdfExtractionIteration] = []
    all_artifacts: list[Path] = [legend_path]
    accepted_path: Path | None = None

    for iteration_number in range(1, config.required_replay_count + 1):
        iteration_dir = output_dir / f"extraction-{iteration_number:02d}"
        iteration_dir.mkdir()
        native_ids = _rasterize_native_fills(evidence.records, source_shape)
        native_ids[~source_domain] = 0
        target_observed_ids = _source_to_reference(
            native_ids,
            transform,
            cv2.INTER_NEAREST,
            0,
            reference_to_source,
        )
        target_observed_ids[~target_domain] = 0
        target_complete_ids, target_inferred = _nearest_completion(
            target_observed_ids, target_domain
        )
        target_observed = target_domain & (target_observed_ids > 0)
        target_reconstruction = _render(target_complete_ids, evidence.classes)

        nearest_source_ids, nearest_source_distance = _classify_visible_palette(
            aligned_source, evidence.classes, target_observed
        )
        source_visible = target_observed & (
            nearest_source_distance <= config.maximum_visible_rgb_distance
        )
        # Text, faults, roads, relief and other overprint are not categorical
        # evidence.  Among pixels visibly matching any legend swatch, however,
        # the nearest swatch must agree with the native vector class.
        visible_match = source_visible & (
            nearest_source_ids == target_observed_ids
        )
        visual_mismatch = source_visible & ~visible_match

        reconstructed_source_ids = _warp_target_to_source_chunked(
            target_complete_ids, transform, source_shape
        )
        source_expected = source_domain & (native_ids > 0)
        source_roundtrip_match = source_expected & (
            reconstructed_source_ids == native_ids
        )
        source_roundtrip_agreement = float(
            np.count_nonzero(source_roundtrip_match)
            / max(np.count_nonzero(source_expected), 1)
        )
        target_observed_fraction = float(
            np.count_nonzero(target_observed) / max(np.count_nonzero(target_domain), 1)
        )
        target_inferred_fraction = float(
            np.count_nonzero(target_inferred) / max(np.count_nonzero(target_domain), 1)
        )
        visible_coverage_fraction = float(
            np.count_nonzero(source_visible)
            / max(np.count_nonzero(target_observed), 1)
        )
        visible_match_fraction = float(
            np.count_nonzero(visible_match) / max(np.count_nonzero(source_visible), 1)
        )
        cell_reports = _geographic_metrics(
            target_domain,
            target_observed,
            visible_match,
            source_visible,
            config.geographic_rows,
            config.geographic_columns,
        )
        source_cell_reports = _source_geographic_roundtrip_metrics(
            source_expected,
            source_roundtrip_match,
            transform,
            target_domain.shape,
            config.geographic_rows,
            config.geographic_columns,
        )
        source_cells = {report["id"]: report for report in source_cell_reports}
        cell_reports = [
            {**report, **source_cells.get(report["id"], {})}
            for report in cell_reports
            if report["id"] in source_cells
        ]
        passing_cells = sum(
            report["observed_fraction"] >= config.minimum_cell_observed_fraction
            and report["source_semantic_match_fraction"]
            >= config.minimum_cell_semantic_match_fraction
            for report in cell_reports
        )
        stable = (
            previous_source_ids is not None
            and previous_target_ids is not None
            and np.array_equal(previous_source_ids, native_ids)
            and np.array_equal(previous_target_ids, target_complete_ids)
        )
        all_classes_have_native_fills = all(
            evidence.matched_fill_objects_by_class[category.class_id] > 0
            for category in evidence.classes
        )
        observed_class_count = len(
            set(int(value) for value in np.unique(target_observed_ids) if value > 0)
        )
        non_thematic_exclusion_complete = (
            evidence.matched_filled_object_count
            + evidence.excluded_non_thematic_curve_object_count
            == evidence.total_curve_object_count
            and evidence.excluded_non_thematic_line_object_count
            == evidence.total_line_object_count
            and evidence.excluded_non_thematic_rect_object_count
            == evidence.total_rect_object_count
        )
        gates: dict[str, Any] = {
            "legend_semantics_established": len(evidence.classes)
            >= config.minimum_legend_classes,
            "all_legend_classes_have_native_fills": all_classes_have_native_fills,
            "all_legend_classes_preserved_in_mapbox_output": {
                "passed": observed_class_count == len(evidence.classes),
                "value": observed_class_count,
                "required": len(evidence.classes),
            },
            "all_native_path_operators_supported": not evidence.unsupported_path_operators,
            "all_non_thematic_vectors_classified_and_excluded": (
                non_thematic_exclusion_complete
            ),
            "target_observed_coverage": {
                "passed": target_observed_fraction
                >= config.minimum_target_observed_fraction,
                "value": target_observed_fraction,
                "minimum": config.minimum_target_observed_fraction,
            },
            "target_inferred_fraction": {
                "passed": target_inferred_fraction
                <= config.maximum_target_inferred_fraction,
                "value": target_inferred_fraction,
                "maximum": config.maximum_target_inferred_fraction,
            },
            "global_source_fill_only_semantic_reconstruction": {
                "passed": source_roundtrip_agreement
                >= config.minimum_source_roundtrip_agreement,
                "value": source_roundtrip_agreement,
                "minimum": config.minimum_source_roundtrip_agreement,
            },
            "mapbox_water_and_exterior_empty": not bool(
                np.any(target_complete_ids[~target_domain] > 0)
            ),
            "observed_and_inferred_are_disjoint": not bool(
                np.any(target_observed & target_inferred)
            ),
            "geographic_source_diff": {
                "passed": passing_cells >= config.minimum_passing_geographic_cells,
                "value": passing_cells,
                "minimum": config.minimum_passing_geographic_cells,
                "supported_cells": len(cell_reports),
            },
            "successive_native_vector_fixed_point": stable,
        }
        all_gates_pass = all(
            value if isinstance(value, bool) else bool(value["passed"])
            for value in gates.values()
        )
        decision = (
            "accept"
            if all_gates_pass
            else ("retry" if iteration_number < config.required_replay_count else "blocked")
        )

        source_ids_path = iteration_dir / "source-observed-class-id.png"
        source_domain_path = iteration_dir / "source-mapbox-land-mask.png"
        source_roundtrip_path = iteration_dir / "source-roundtrip-class-id.png"
        source_diff_path = iteration_dir / "source-roundtrip-diff-mask.png"
        target_ids_path = iteration_dir / "mapbox-class-id.png"
        observed_path = iteration_dir / "mapbox-observed-mask.png"
        inferred_path = iteration_dir / "mapbox-inferred-mask.png"
        reconstruction_path = iteration_dir / "mapbox-reconstruction.png"
        aligned_source_path = iteration_dir / "mapbox-aligned-source.png"
        visual_diff_path = iteration_dir / "mapbox-source-diff-mask.png"
        diagnostic_path = iteration_dir / "source-extraction-diagnostic.png"
        _save_ids(source_ids_path, native_ids)
        _save_mask(source_domain_path, source_domain)
        _save_ids(source_roundtrip_path, reconstructed_source_ids)
        _save_mask(source_diff_path, source_expected & ~source_roundtrip_match)
        _save_ids(target_ids_path, target_complete_ids)
        _save_mask(observed_path, target_observed)
        _save_mask(inferred_path, target_inferred)
        _save_rgb(reconstruction_path, target_reconstruction)
        _save_rgb(aligned_source_path, aligned_source)
        _save_mask(visual_diff_path, visual_mismatch)
        _save_rgb(
            diagnostic_path,
            _diagnostic(
                aligned_source,
                target_reconstruction,
                target_inferred,
            ),
        )
        artifact_paths = (
            source_ids_path,
            source_domain_path,
            source_roundtrip_path,
            source_diff_path,
            target_ids_path,
            observed_path,
            inferred_path,
            reconstruction_path,
            aligned_source_path,
            visual_diff_path,
            diagnostic_path,
        )
        class_counts = {
            category.id: int(np.count_nonzero(target_observed_ids == category.class_id))
            for category in evidence.classes
        }
        scores = {
            "legend_class_count": len(evidence.classes),
            "matched_native_fill_object_count": evidence.matched_filled_object_count,
            "matched_native_contour_count": evidence.matched_contour_count,
            "matched_native_stroke_object_count": int(
                sum(evidence.matched_stroke_objects_by_class.values())
            ),
            "total_native_curve_object_count": evidence.total_curve_object_count,
            "total_native_line_object_count": evidence.total_line_object_count,
            "total_native_rect_object_count": evidence.total_rect_object_count,
            "excluded_non_thematic_curve_object_count": (
                evidence.excluded_non_thematic_curve_object_count
            ),
            "excluded_non_thematic_line_object_count": (
                evidence.excluded_non_thematic_line_object_count
            ),
            "excluded_non_thematic_rect_object_count": (
                evidence.excluded_non_thematic_rect_object_count
            ),
            "unsupported_path_operators": list(evidence.unsupported_path_operators),
            "mapbox_land_pixel_count": int(np.count_nonzero(target_domain)),
            "target_observed_pixel_count": int(np.count_nonzero(target_observed)),
            "target_inferred_pixel_count": int(np.count_nonzero(target_inferred)),
            "target_observed_fraction": target_observed_fraction,
            "target_inferred_fraction": target_inferred_fraction,
            "mapbox_observed_class_count": observed_class_count,
            "global_source_render_match_fraction": visible_match_fraction,
            "global_source_render_visible_coverage_fraction": visible_coverage_fraction,
            "source_roundtrip_agreement_fraction": source_roundtrip_agreement,
            "class_observed_pixel_counts": class_counts,
            "geographic_cells": cell_reports,
            "successive_native_source_equal": stable,
            "successive_mapbox_output_equal": stable,
        }
        report_path = iteration_dir / "iteration.json"
        report = {
            "schema_version": SCHEMA_VERSION,
            "iteration": iteration_number,
            "decision": decision,
            "scores": scores,
            "gates": gates,
            "provenance": {
                "original_pdf": {
                    "path": str(working.source_path),
                    "sha256": working.source_sha256,
                },
                "source_adapter": {
                    "path": str(working.manifest_path),
                    "sha256": _sha256(working.manifest_path),
                },
                "accepted_exact_graticule_alignment": {
                    "path": str(accepted_alignment_path.resolve()),
                    "sha256": _sha256(accepted_alignment_path.resolve()),
                },
                "pinned_mapbox_manifest": {
                    "path": str(mapbox_manifest_path.resolve()),
                    "sha256": _sha256(mapbox_manifest_path.resolve()),
                },
                "prior_run_artifacts_used": False,
                "manual_inputs_used": False,
            },
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
                    "authoritative_original_pdf_native_legend",
                    "authoritative_original_pdf_native_vector_fills_and_strokes",
                    "accepted_exact_native_graticule_transform",
                    "pinned_mapbox_land_and_water",
                    "source_reconstruction_diff",
                    "deterministic_native_vector_fixed_point_replay",
                ],
            ),
            method=(
                "native PDF legend-code and CMYK association, exact vector fill "
                "rasterization, Mapbox land/water clipping, source reconstruction, "
                "geographic source diff, and deterministic fixed-point replay"
            ),
            artifacts=[
                {"path": str(path), "sha256": _sha256(path)}
                for path in (*complete_artifacts, legend_path)
            ],
        )
        experiment_log.write(experiment_markdown_path, experiment_json_path)
        iteration = GeologicPdfExtractionIteration(
            iteration_number,
            decision,
            scores,
            gates,
            report_path,
            tuple(complete_artifacts),
        )
        iterations.append(iteration)
        all_artifacts.extend(complete_artifacts)
        previous_source_ids = native_ids.copy()
        previous_target_ids = target_complete_ids.copy()
        if decision == "accept":
            accepted_path = output_dir / "accepted-extraction.json"
            accepted_path.write_text(
                json.dumps(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "status": "accepted",
                        "automatic_iteration_count": iteration_number,
                        "source": {
                            "path": str(working.source_path),
                            "sha256": working.source_sha256,
                        },
                        "source_adapter": {
                            "path": str(working.manifest_path),
                            "sha256": _sha256(working.manifest_path),
                        },
                        "alignment": {
                            "path": str(accepted_alignment_path.resolve()),
                            "sha256": _sha256(accepted_alignment_path.resolve()),
                        },
                        "legend": {
                            "path": "legend/legend.json",
                            "sha256": _sha256(legend_path),
                        },
                        "accepted_iteration": f"extraction-{iteration_number:02d}",
                        "observed_class_ids": (
                            f"extraction-{iteration_number:02d}/mapbox-class-id.png"
                        ),
                        "observed_mask": (
                            f"extraction-{iteration_number:02d}/mapbox-observed-mask.png"
                        ),
                        "inferred_mask": (
                            f"extraction-{iteration_number:02d}/mapbox-inferred-mask.png"
                        ),
                    },
                    indent=2,
                )
                + "\n"
            )
            all_artifacts.append(accepted_path)
            experiment_log.finalize("complete")
            experiment_log.write(experiment_markdown_path, experiment_json_path)
            break

    if accepted_path is not None:
        return GeologicPdfExtractionResult(
            "accepted",
            "native legend, source-diff, geographic, and fixed-point gates passed",
            tuple(iterations),
            accepted_path,
            tuple(all_artifacts),
        )
    blocker = (
        "Native PDF categorical extraction did not pass every source-diff and "
        "fixed-point gate after two deterministic passes"
    )
    experiment_log.finalize("blocked", blocker)
    experiment_log.write(experiment_markdown_path, experiment_json_path)
    return GeologicPdfExtractionResult(
        "blocked", blocker, tuple(iterations), None, tuple(all_artifacts)
    )
