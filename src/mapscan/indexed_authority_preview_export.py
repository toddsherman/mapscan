"""Package accepted nonstandard extractions as indexed staging datasets.

The normal categorical exporter consumes ``automatic-categorical-extraction``
results.  Quake and elevation have stronger, specialized acceptance contracts,
so this module converts only their already-accepted pixels into the same
lossless indexed publication format.  It never reclassifies source pixels.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from .automatic_extraction_preview_export import (
    _base_mapbox_reference,
    _json_object,
    _minimum_nonundersampling_zoom,
    _screen_pixel_span,
    _target_grid,
    _validate_identifier,
)
from .tile_export import _sha256, _write_indexed_class_id_pyramid, _wgs84_bounds


PRODUCER = "mapscan.indexed_authority_preview_export"


def _hash_bound_file(path: Path, expected_hash: str, label: str) -> Path:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
        raise ValueError(f"{label.capitalize()} has no valid SHA-256 declaration")
    if _sha256(path) != expected_hash:
        raise ValueError(f"{label.capitalize()} differs from its declared hash")
    return path


def _relative_artifact(root: Path, raw_path: object, label: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"Accepted extraction has no {label} path")
    candidate = (root / raw_path).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"Accepted {label} escapes its extraction directory") from error
    if not candidate.is_file():
        raise FileNotFoundError(f"Missing accepted {label}: {candidate}")
    return candidate


def _artifact_hash(report: Mapping[str, Any], relative_path: str) -> str:
    matches = [
        artifact
        for artifact in report.get("artifacts", [])
        if isinstance(artifact, Mapping) and artifact.get("path") == relative_path
    ]
    if len(matches) != 1:
        raise ValueError(f"Iteration report does not uniquely bind {relative_path}")
    digest = str(matches[0].get("sha256", ""))
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError(f"Iteration report has no valid hash for {relative_path}")
    return digest


def _category(
    class_id: int, identifier: str, label: str, color: Sequence[int]
) -> dict[str, Any]:
    _validate_identifier(identifier, "category id")
    rgb = [int(value) for value in color]
    if len(rgb) != 3 or any(value < 0 or value > 255 for value in rgb):
        raise ValueError(f"Category {identifier} has an invalid display color")
    return {
        "id": identifier,
        "class_id": class_id,
        "label": label,
        "display_rgb": rgb,
    }


def _export_indexed_preview(
    output_dir: Path,
    *,
    dataset_id: str,
    title: str,
    layer_id: str,
    layer_label: str,
    values: np.ndarray,
    categories: Sequence[Mapping[str, Any]],
    source_path: Path,
    source_sha256: str,
    accepted_extraction_path: Path,
    accepted_extraction: Mapping[str, Any],
    accepted_alignment_path: Path,
    accepted_alignment: Mapping[str, Any],
    accepted_inputs: Sequence[Mapping[str, Any]],
    legend_path: Path,
    legend_sha256: str,
    mapbox_reference_dir: Path,
    minimum_zoom: int = 4,
    maximum_zoom: int = 9,
    overview_supersampling: int = 4,
) -> dict[str, Any]:
    """Write a review-only indexed package from an accepted class raster."""

    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Preview output directory is not empty: {output_dir}")
    dataset_id = _validate_identifier(dataset_id, "dataset id")
    layer_id = _validate_identifier(layer_id, "layer id")
    if not title.strip() or not layer_label.strip():
        raise ValueError("Dataset and layer titles are required")
    if not 0 <= minimum_zoom <= maximum_zoom <= 22:
        raise ValueError("Tile zooms must satisfy 0 <= minimum <= maximum <= 22")

    alignment_grid = _target_grid(accepted_alignment)
    if values.ndim != 2 or values.shape != (
        alignment_grid["height"],
        alignment_grid["width"],
    ):
        raise ValueError("Accepted class raster dimensions differ from alignment grid")
    values = values.astype(np.uint8, copy=False)
    category_records = [dict(category) for category in categories]
    expected_ids = list(range(1, len(category_records) + 1))
    if [int(record.get("class_id", 0)) for record in category_records] != expected_ids:
        raise ValueError("Category class ids must be contiguous from one")
    category_ids = [str(record.get("id", "")) for record in category_records]
    if len(set(category_ids)) != len(category_ids):
        raise ValueError("Category identifiers must be unique")
    for category_id in category_ids:
        _validate_identifier(category_id, "category id")
    if int(values.max(initial=0)) > len(category_records):
        raise ValueError("Accepted raster contains a class absent from the legend")

    minimum_native_zoom = _minimum_nonundersampling_zoom(alignment_grid)
    if maximum_zoom < minimum_native_zoom:
        raise ValueError(
            f"Maximum zoom undersamples the accepted raster; use z{minimum_native_zoom}"
        )
    reference = _base_mapbox_reference(
        accepted_alignment, mapbox_reference_dir, alignment_grid
    )
    source_path = _hash_bound_file(source_path, source_sha256, "source image")
    accepted_extraction_path = _hash_bound_file(
        accepted_extraction_path,
        _sha256(accepted_extraction_path),
        "accepted extraction pointer",
    )
    accepted_alignment_path = _hash_bound_file(
        accepted_alignment_path,
        _sha256(accepted_alignment_path),
        "accepted alignment",
    )
    legend_path = _hash_bound_file(legend_path, legend_sha256, "accepted legend")

    output_dir.mkdir(parents=True, exist_ok=True)
    source_target = output_dir / f"source{source_path.suffix.lower() or '.img'}"
    shutil.copy2(source_path, source_target)
    overlay = reference["state_coast_overlay"]
    overlay_target = output_dir / "mapbox-state-coast-overlay.png"
    shutil.copy2(Path(str(overlay["path"])), overlay_target)
    if _sha256(overlay_target) != overlay["sha256"]:
        raise ValueError("Copied Mapbox state/coast diagnostic changed")

    class_raster_path = output_dir / "accepted-class-id.png"
    Image.fromarray(values, mode="L").save(class_raster_path, optimize=True)
    class_raster_hash = _sha256(class_raster_path)
    bounds_3857 = tuple(float(value) for value in alignment_grid["bounds"])
    bounds_wgs84 = _wgs84_bounds(bounds_3857)
    indexed_raster = _write_indexed_class_id_pyramid(
        values,
        bounds_3857,
        output_dir,
        layer_id,
        len(category_records),
        minimum_zoom,
        maximum_zoom,
        overview_supersampling,
    )
    categories_with_counts = [
        {
            **record,
            "pixel_count": int(np.count_nonzero(values == int(record["class_id"]))),
        }
        for record in category_records
    ]
    boundary = {
        "kind": "pinned_mapbox_state_coast_diagnostic",
        "authority": "accepted_alignment_mapbox_reference",
        "diagnostic_only": True,
        "raster": overlay_target.name,
        "raster_sha256": _sha256(overlay_target),
        "raster_width": int(overlay["width"]),
        "raster_height": int(overlay["height"]),
        "raster_bounds": list(bounds_wgs84),
    }
    provenance = {
        "schema_version": 1,
        "kind": "autonomous_specialized_extraction_staging_preview_provenance",
        "producer": PRODUCER,
        "status": "needs_visual_review",
        "publication_approved": False,
        "approval": {"status": "not_approved"},
        "accepted_extraction": {
            "path": str(accepted_extraction_path),
            "sha256": _sha256(accepted_extraction_path),
            "schema_version": accepted_extraction.get("schema_version"),
            "automatic_iteration_count": int(
                accepted_extraction.get("automatic_iteration_count", 0)
            ),
            "accepted_iteration": accepted_extraction.get("accepted_iteration"),
        },
        "accepted_alignment": {
            "path": str(accepted_alignment_path),
            "sha256": _sha256(accepted_alignment_path),
            "automatic_iteration_count": int(accepted_alignment.get("iteration", 0)),
        },
        "source": {
            "path": str(source_path),
            "sha256": source_sha256,
            "copied_path": source_target.name,
            "copied_sha256": _sha256(source_target),
        },
        "legend": {"path": str(legend_path), "sha256": legend_sha256},
        "accepted_inputs": [dict(record) for record in accepted_inputs],
        "class_raster": {
            "path": class_raster_path.name,
            "sha256": class_raster_hash,
            "derivation": "deterministic_class_id_composition_from_accepted_inputs",
            "width": alignment_grid["width"],
            "height": alignment_grid["height"],
        },
        "target_grid": alignment_grid,
        "categorical_tile_encoding": "indexed_class_id",
        "processing_reference": reference,
        "boundary": boundary,
        "nodata": {
            "class_id": 0,
            "transparent": True,
            "pixel_count": int(np.count_nonzero(values == 0)),
            "policy": "preserve_accepted_transparent_nodata",
        },
        "native_resolution": {
            "minimum_nonundersampling_zoom": minimum_native_zoom,
            "screen_pixel_span_at_maximum_zoom": list(
                _screen_pixel_span(alignment_grid, maximum_zoom)
            ),
        },
        "manual_inputs": {
            "manual_arrows": False,
            "manual_stamps": False,
            "human_approval": False,
        },
    }
    provenance_path = output_dir / "autonomous-preview-provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    dataset = {
        "schema_version": 1,
        "status": "needs_visual_review",
        "id": dataset_id,
        "title": title,
        "bounds": list(bounds_wgs84),
        "center": [
            (bounds_wgs84[0] + bounds_wgs84[2]) / 2.0,
            (bounds_wgs84[1] + bounds_wgs84[3]) / 2.0,
        ],
        "minimum_zoom": minimum_zoom,
        "maximum_native_zoom": maximum_zoom,
        "overscaling": "nearest",
        "categorical_tile_encoding": "indexed_class_id",
        "overview": {
            "mode": "dominant_class_with_fractional_coverage",
            "supersampling": overview_supersampling,
            "overview_zooms": list(range(minimum_zoom, maximum_zoom)),
            "exact_binary_zoom": maximum_zoom,
        },
        "source_image": source_target.name,
        "approval": {"status": "not_approved"},
        "boundary": boundary,
        "provenance": {
            "manifest": provenance_path.name,
            "sha256": _sha256(provenance_path),
        },
        "layers": [
            {
                "id": layer_id,
                "label": layer_label,
                "kind": "categorical",
                "bounds": list(bounds_wgs84),
                "nodata_class_id": 0,
                "categories": categories_with_counts,
                "indexed_raster": indexed_raster,
            }
        ],
    }
    dataset_path = output_dir / "dataset.json"
    dataset_path.write_text(json.dumps(dataset, indent=2) + "\n")
    return {**dataset, "dataset_manifest_sha256": _sha256(dataset_path)}


def export_quake_nonlinear_staging_preview(
    accepted_extraction_path: Path,
    accepted_base_alignment_path: Path,
    mapbox_reference_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Package the six accepted nonlinear quake masks without reclassification."""

    pointer_path = accepted_extraction_path.resolve()
    pointer = _json_object(pointer_path, "accepted quake extraction")
    if (
        pointer.get("schema_version")
        != "mapscan.quake-native-nonlinear-extraction.v1"
        or pointer.get("status") != "accepted"
    ):
        raise ValueError("Quake extraction is not the accepted nonlinear result")
    if not all(pointer.get("gates", {}).get(key) is True for key in (
        "fresh_pristine_source_reclassified",
        "nonlinear_alignment_all_gates_passed",
        "exact_six_ordered_legend_semantics",
        "all_six_classes_preserved",
        "source_domain_complete",
        "target_domain_complete",
        "mapbox_water_and_exterior_empty",
    )):
        raise ValueError("Accepted quake extraction no longer passes its semantic gates")

    source_record = pointer.get("source", {})
    source_path = _hash_bound_file(
        Path(str(source_record.get("path", ""))),
        str(source_record.get("sha256", "")),
        "quake source image",
    )
    legend_record = pointer.get("legend", {})
    legend_path = _relative_artifact(
        pointer_path.parent, legend_record.get("path"), "quake legend"
    )
    legend_hash = str(legend_record.get("sha256", ""))
    _hash_bound_file(legend_path, legend_hash, "quake legend")
    legend = _json_object(legend_path, "quake legend")
    legend_classes = legend.get("classes", [])
    layers = pointer.get("layers", {})
    if not isinstance(layers, Mapping) or len(layers) != 6 or len(legend_classes) != 6:
        raise ValueError("Accepted quake extraction must contain exactly six classes")

    masks: list[np.ndarray] = []
    inputs: list[dict[str, Any]] = []
    categories: list[dict[str, Any]] = []
    expected_counts = pointer.get("scores", {}).get("mapbox_class_counts", {})
    for expected_class_id, legend_class in enumerate(legend_classes, 1):
        matching = [
            record
            for record in layers.values()
            if isinstance(record, Mapping)
            and int(record.get("class_id", 0)) == expected_class_id
        ]
        if len(matching) != 1:
            raise ValueError("Quake class ids are not uniquely declared")
        record = matching[0]
        label = str(record.get("label", ""))
        if (
            int(legend_class.get("class_id", 0)) != expected_class_id
            or legend_class.get("label") != label
        ):
            raise ValueError("Quake masks and ordered legend disagree")
        mask_path = _relative_artifact(
            pointer_path.parent, record.get("mask"), f"quake class {expected_class_id}"
        )
        with Image.open(mask_path) as image:
            mask = np.asarray(image.convert("L")) > 0
        expected_count = int(expected_counts.get(label, -1))
        if int(np.count_nonzero(mask)) != expected_count:
            raise ValueError(f"Quake class {expected_class_id} count differs from acceptance")
        masks.append(mask)
        inputs.append(
            {
                "kind": "accepted_class_mask",
                "class_id": expected_class_id,
                "label": label,
                "path": str(mask_path),
                "sha256": _sha256(mask_path),
                "pixel_count": expected_count,
            }
        )
        categories.append(
            _category(
                expected_class_id,
                f"{str(legend_class.get('roman', '')).casefold()}-{re.sub(r'[^a-z0-9]+', '-', label.casefold()).strip('-').split('-', 1)[-1]}",
                label,
                legend_class.get("representative_rgb", []),
            )
        )
    overlap = np.sum(np.stack(masks, axis=0), axis=0)
    if int(overlap.max(initial=0)) > 1:
        raise ValueError("Accepted quake class masks overlap")
    values = np.zeros(masks[0].shape, dtype=np.uint8)
    for class_id, mask in enumerate(masks, 1):
        values[mask] = class_id

    base_alignment_path = accepted_base_alignment_path.resolve()
    base_alignment = _json_object(base_alignment_path, "accepted quake base alignment")
    nonlinear_record = pointer.get("alignment", {})
    nonlinear_path = _hash_bound_file(
        Path(str(nonlinear_record.get("path", ""))),
        str(nonlinear_record.get("sha256", "")),
        "accepted nonlinear quake alignment",
    )
    nonlinear = _json_object(nonlinear_path, "accepted nonlinear quake alignment")
    base_record = nonlinear.get("accepted_alignment", {})
    if (
        str(base_record.get("sha256", "")) != _sha256(base_alignment_path)
        or nonlinear.get("status") != "pass"
    ):
        raise ValueError("Nonlinear quake refinement does not bind the base alignment")
    inputs.append(
        {
            "kind": "accepted_nonlinear_alignment",
            "path": str(nonlinear_path),
            "sha256": _sha256(nonlinear_path),
        }
    )
    return _export_indexed_preview(
        output_dir,
        dataset_id="earthquake-shaking-potential-california-2025-final-hybrid-v1",
        title="Earthquake Shaking Potential for California",
        layer_id="modified-mercalli-intensity",
        layer_label="Modified Mercalli Intensity",
        values=values,
        categories=categories,
        source_path=source_path,
        source_sha256=str(source_record["sha256"]),
        accepted_extraction_path=pointer_path,
        accepted_extraction=pointer,
        accepted_alignment_path=base_alignment_path,
        accepted_alignment=base_alignment,
        accepted_inputs=inputs,
        legend_path=legend_path,
        legend_sha256=legend_hash,
        mapbox_reference_dir=mapbox_reference_dir,
    )


def export_elevation_bands_staging_preview(
    accepted_extraction_path: Path,
    mapbox_reference_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Package the accepted elevation bands and depression class for selection."""

    pointer_path = accepted_extraction_path.resolve()
    pointer = _json_object(pointer_path, "accepted elevation extraction")
    if (
        pointer.get("schema_version")
        != "mapscan.automatic-continuous-numeric-extraction.v1"
        or pointer.get("status") != "accepted"
    ):
        raise ValueError("Elevation extraction is not the accepted numeric result")
    source_record = pointer.get("source", {})
    source_path = _hash_bound_file(
        Path(str(source_record.get("path", ""))),
        str(source_record.get("sha256", "")),
        "elevation source image",
    )
    alignment_record = pointer.get("alignment", {})
    alignment_path = _hash_bound_file(
        Path(str(alignment_record.get("path", ""))),
        str(alignment_record.get("sha256", "")),
        "accepted elevation alignment",
    )
    alignment = _json_object(alignment_path, "accepted elevation alignment")
    legend_record = pointer.get("legend", {})
    legend_path = _relative_artifact(
        pointer_path.parent, legend_record.get("path"), "elevation legend"
    )
    legend_hash = str(legend_record.get("sha256", ""))
    _hash_bound_file(legend_path, legend_hash, "elevation legend")
    legend = _json_object(legend_path, "elevation legend")

    accepted_iteration = str(pointer.get("accepted_iteration", ""))
    report_path = pointer_path.parent / accepted_iteration / "iteration.json"
    report = _json_object(report_path, "accepted elevation iteration")
    if report.get("decision") != "accept":
        raise ValueError("Elevation iteration is not accepted")
    band_relative = str(
        pointer.get("layers", {})
        .get("quantized_elevation_bands", {})
        .get("class_id_raster", "")
    )
    depression_relative = str(
        pointer.get("layers", {}).get("depression", {}).get("mask", "")
    )
    band_path = _relative_artifact(pointer_path.parent, band_relative, "elevation bands")
    depression_path = _relative_artifact(
        pointer_path.parent, depression_relative, "elevation depression mask"
    )
    band_hash = _artifact_hash(report, band_relative)
    depression_hash = _artifact_hash(report, depression_relative)
    _hash_bound_file(band_path, band_hash, "accepted elevation bands")
    _hash_bound_file(depression_path, depression_hash, "accepted depression mask")
    with Image.open(band_path) as image:
        bands = np.asarray(image)
    with Image.open(depression_path) as image:
        depression = np.asarray(image.convert("L")) > 0
    if bands.ndim != 2 or depression.shape != bands.shape:
        raise ValueError("Elevation accepted rasters have inconsistent dimensions")
    if int(bands.max(initial=0)) != 7 or np.any((bands > 0) & depression):
        raise ValueError("Elevation bands and depression semantics are inconsistent")
    values = np.where(depression, 1, np.where(bands > 0, bands + 1, 0)).astype(
        np.uint8
    )
    continuous_relative = str(
        pointer.get("layers", {}).get("elevation_meters", {}).get("raster", "")
    )
    continuous_path = _relative_artifact(
        pointer_path.parent,
        continuous_relative,
        "continuous elevation raster",
    )
    continuous_hash = _artifact_hash(report, continuous_relative)
    _hash_bound_file(
        continuous_path, continuous_hash, "accepted continuous elevation raster"
    )

    intervals = (
        pointer.get("layers", {})
        .get("quantized_elevation_bands", {})
        .get("intervals_m", [])
    )
    stops = {float(stop["value_m"]): stop for stop in legend.get("numeric_stops", [])}
    special = legend.get("special_semantics", [])
    if len(intervals) != 7 or len(special) != 1:
        raise ValueError("Elevation legend does not contain the accepted eight classes")
    categories = [
        _category(
            1,
            "depression",
            "Depression (depth not specified)",
            special[0].get("representative_rgb", []),
        )
    ]
    for output_class_id, interval in enumerate(intervals, 2):
        minimum = float(interval["minimum"])
        maximum = float(interval["maximum"])
        maximum_label = f"{int(maximum):,}"
        minimum_label = f"{int(minimum):,}"
        categories.append(
            _category(
                output_class_id,
                f"elevation-{int(minimum)}-to-{int(maximum)}-m",
                f"{minimum_label}–<{maximum_label} m",
                stops[minimum].get("representative_rgb", []),
            )
        )
    inputs = [
        {
            "kind": "accepted_quantized_elevation_band_raster",
            "path": str(band_path),
            "sha256": band_hash,
        },
        {
            "kind": "accepted_depression_mask",
            "path": str(depression_path),
            "sha256": depression_hash,
        },
        {
            "kind": "accepted_continuous_elevation_raster",
            "path": str(continuous_path),
            "sha256": continuous_hash,
            "publication_role": "accepted_numeric_authority_retained_in_provenance",
        },
    ]
    return _export_indexed_preview(
        output_dir,
        dataset_id="california-topography-elevation",
        title="California Topography and Elevation",
        layer_id="elevation-bands",
        layer_label="Elevation bands",
        values=values,
        categories=categories,
        source_path=source_path,
        source_sha256=str(source_record["sha256"]),
        accepted_extraction_path=pointer_path,
        accepted_extraction=pointer,
        accepted_alignment_path=alignment_path,
        accepted_alignment=alignment,
        accepted_inputs=inputs,
        legend_path=legend_path,
        legend_sha256=legend_hash,
        mapbox_reference_dir=mapbox_reference_dir,
    )
