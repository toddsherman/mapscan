"""Hash-verified staging previews for accepted automatic categorical results.

This exporter is intentionally review-only.  It consumes the immutable
relationships declared by ``accepted-extraction.json`` and its accepted
iteration, but it never imports a historical boundary or an approval decision.
Class zero remains transparent NoData throughout the XYZ pyramid.
"""

from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from .tile_export import (
    WEB_MERCATOR_HALF_WORLD,
    _aggregate_hash,
    _sample_class_overview,
    _sample_class_tile,
    _sha256,
    _tile_range,
    _write_indexed_class_id_pyramid,
    _wgs84_bounds,
)


EXTRACTION_SCHEMA = "mapscan.automatic-categorical-extraction.v1"
PRODUCER = "mapscan.automatic_extraction_preview_export"


def _json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{label.capitalize()} must be a JSON object")
    return value


def _declared_hash(record: Mapping[str, Any], label: str) -> str:
    digest = str(record.get("sha256", ""))
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError(f"{label.capitalize()} has no valid SHA-256 declaration")
    return digest


def _hash_bound_path(
    record: object,
    *,
    base: Path,
    label: str,
    require_within_base: bool = False,
) -> tuple[Path, str]:
    if not isinstance(record, Mapping):
        raise ValueError(f"Accepted extraction has no {label} record")
    raw_path = str(record.get("path", ""))
    if not raw_path:
        raise ValueError(f"{label.capitalize()} has no declared path")
    expected_hash = _declared_hash(record, label)
    source = Path(raw_path)
    if source.is_absolute():
        candidates = [source.resolve()]
    else:
        candidates = [(base / source).resolve(), (Path.cwd() / source).resolve()]
    if require_within_base:
        root = base.resolve()
        candidates = [
            candidate
            for candidate in candidates
            if candidate == root or root in candidate.parents
        ]
        if not candidates:
            raise ValueError(f"{label.capitalize()} escapes the extraction output")
    existing = [candidate for candidate in candidates if candidate.is_file()]
    if not existing:
        raise FileNotFoundError(f"Missing hash-bound {label}: {raw_path}")
    matching = [candidate for candidate in existing if _sha256(candidate) == expected_hash]
    if not matching:
        raise ValueError(f"{label.capitalize()} does not match its declared hash")
    return matching[0], expected_hash


def _report_artifact(
    report: Mapping[str, Any], extraction_dir: Path, relative_path: str
) -> tuple[Path, str]:
    matches = [
        item
        for item in report.get("artifacts", [])
        if isinstance(item, Mapping) and item.get("path") == relative_path
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Accepted iteration must declare exactly one {relative_path} artifact"
        )
    path, digest = _hash_bound_path(
        matches[0],
        base=extraction_dir,
        label=relative_path,
        require_within_base=True,
    )
    byte_count = matches[0].get("byte_count")
    if byte_count is not None and int(byte_count) != path.stat().st_size:
        raise ValueError(f"{relative_path} byte count differs from its report")
    return path, digest


def _validated_grid(grid: object, label: str) -> dict[str, Any]:
    if not isinstance(grid, Mapping) or grid.get("crs") != "EPSG:3857":
        raise ValueError(f"{label.capitalize()} must use EPSG:3857")
    bounds = [float(value) for value in grid.get("bounds", [])]
    if (
        len(bounds) != 4
        or not all(math.isfinite(value) for value in bounds)
        or bounds[0] >= bounds[2]
        or bounds[1] >= bounds[3]
    ):
        raise ValueError(f"{label.capitalize()} has invalid bounds")
    width = int(grid.get("width", 0))
    height = int(grid.get("height", 0))
    if width < 1 or height < 1:
        raise ValueError(f"{label.capitalize()} has invalid dimensions")
    return {
        "crs": "EPSG:3857",
        "bounds": bounds,
        "width": width,
        "height": height,
    }


def _target_grid(alignment: Mapping[str, Any]) -> dict[str, Any]:
    if alignment.get("decision") != "accept":
        raise ValueError("Accepted alignment artifact does not contain an acceptance")
    transform = alignment.get("transform")
    if not isinstance(transform, Mapping):
        raise ValueError("Accepted alignment has no transform")
    return _validated_grid(transform.get("target_grid"), "accepted alignment target grid")


def _processing_grid(
    pointer: Mapping[str, Any],
    alignment: Mapping[str, Any],
    extraction_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Resolve a hash-pinned supersampled grid without trusting pointer claims."""

    base_grid = _target_grid(alignment)
    factor = int(pointer.get("target_supersampling", 1))
    declared = pointer.get("processing_target_grid")
    reference_record = pointer.get("processing_reference")
    if factor == 1:
        if declared is not None and _validated_grid(
            declared, "processing target grid"
        ) != base_grid:
            raise ValueError("Unit-sampled processing grid differs from alignment grid")
        if reference_record is not None:
            raise ValueError("Unit-sampled extraction must not declare a derived reference")
        return base_grid, None
    if not 2 <= factor <= 4:
        raise ValueError("Target supersampling must be between two and four")
    processing_grid = _validated_grid(declared, "processing target grid")
    expected = {
        **base_grid,
        "width": (base_grid["width"] - 1) * factor + 1,
        "height": (base_grid["height"] - 1) * factor + 1,
    }
    if processing_grid != expected:
        raise ValueError("Processing target grid is not a corner-preserving supersample")
    reference_path, reference_hash = _hash_bound_path(
        reference_record,
        base=extraction_dir,
        label="supersampled processing reference",
    )
    manifest = _json_object(reference_path, "supersampled processing reference")
    if (
        manifest.get("status") != "pinned_reference"
        or manifest.get("kind") != "mapbox_california_state_coast_water_counties"
        or _validated_grid(manifest.get("target_grid"), "processing reference grid")
        != processing_grid
    ):
        raise ValueError("Supersampled processing reference contract is invalid")
    accepted_pin = alignment.get("mapbox_reference")
    if not isinstance(accepted_pin, Mapping):
        raise ValueError("Accepted alignment has no pinned Mapbox reference")
    derivation = manifest.get("derivation", {})
    raw_hashes = {
        "style_sha256": manifest.get("style", {}).get("sha256"),
        "tilejson_sha256": manifest.get("tileset", {}).get("tilejson_sha256"),
        "tile_aggregate_sha256": manifest.get("tile_aggregate_sha256"),
    }
    if (
        derivation.get("source_manifest_sha256")
        != accepted_pin.get("manifest_sha256")
        or derivation.get("raw_bytes_preserved_exactly") is not True
        or derivation.get("only_derived_masks_and_overlays_recomputed") is not True
        or int(derivation.get("target_grid_supersampling", 0)) != factor
        or any(raw_hashes[key] != accepted_pin.get(key) for key in raw_hashes)
    ):
        raise ValueError(
            "Supersampled processing reference changed accepted Mapbox authority"
        )
    processing_reference = {
        "path": str(reference_path),
        "sha256": reference_hash,
        "target_supersampling": factor,
        "raw_hashes": raw_hashes,
    }
    artifacts = manifest.get("artifacts")
    overlay_record = (
        artifacts.get("state_coast_overlay")
        if isinstance(artifacts, Mapping)
        else None
    )
    if overlay_record is not None:
        overlay_path, overlay_hash = _hash_bound_path(
            overlay_record,
            base=reference_path.parent,
            label="processing Mapbox state/coast overlay",
            require_within_base=True,
        )
        with Image.open(overlay_path) as overlay_image:
            overlay_width, overlay_height = overlay_image.size
        if (
            overlay_width != processing_grid["width"]
            or overlay_height != processing_grid["height"]
        ):
            raise ValueError(
                "Processing Mapbox state/coast overlay dimensions differ from grid"
            )
        processing_reference["state_coast_overlay"] = {
            "path": str(overlay_path),
            "sha256": overlay_hash,
            "width": overlay_width,
            "height": overlay_height,
        }
    return processing_grid, processing_reference


def _base_mapbox_reference(
    alignment: Mapping[str, Any],
    reference_dir: Path,
    grid: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and expose the derived overlay from an accepted raw Mapbox pin."""

    reference_dir = reference_dir.resolve()
    manifest_path = reference_dir / "manifest.json"
    manifest = _json_object(manifest_path, "Mapbox reference manifest")
    if (
        manifest.get("status") != "pinned_reference"
        or manifest.get("kind")
        != "mapbox_california_state_coast_water_counties"
        or _validated_grid(manifest.get("target_grid"), "Mapbox reference grid")
        != dict(grid)
    ):
        raise ValueError("Mapbox reference does not match the accepted target grid")
    accepted_pin = alignment.get("mapbox_reference")
    if not isinstance(accepted_pin, Mapping):
        raise ValueError("Accepted alignment has no pinned Mapbox reference")
    raw_hashes = {
        "style_sha256": manifest.get("style", {}).get("sha256"),
        "tilejson_sha256": manifest.get("tileset", {}).get("tilejson_sha256"),
        "tile_aggregate_sha256": manifest.get("tile_aggregate_sha256"),
    }
    if any(raw_hashes[key] != accepted_pin.get(key) for key in raw_hashes):
        raise ValueError("Mapbox reference changed accepted raw authority")
    overlay_record = manifest.get("artifacts", {}).get("state_coast_overlay")
    overlay_path, overlay_hash = _hash_bound_path(
        overlay_record,
        base=reference_dir,
        label="Mapbox state/coast overlay",
        require_within_base=True,
    )
    with Image.open(overlay_path) as overlay_image:
        overlay_width, overlay_height = overlay_image.size
    if (
        overlay_width != int(grid["width"])
        or overlay_height != int(grid["height"])
    ):
        raise ValueError("Mapbox state/coast overlay dimensions differ from grid")
    return {
        "path": str(manifest_path),
        "sha256": _sha256(manifest_path),
        "raw_hashes": raw_hashes,
        "state_coast_overlay": {
            "path": str(overlay_path),
            "sha256": overlay_hash,
            "width": overlay_width,
            "height": overlay_height,
        },
    }


def _screen_pixel_span(
    grid: Mapping[str, Any], zoom: int
) -> tuple[float, float]:
    """Return the exact rendered Web Mercator span at one XYZ zoom."""

    minimum_x, minimum_y, maximum_x, maximum_y = map(float, grid["bounds"])
    world_span = WEB_MERCATOR_HALF_WORLD * 2.0
    pixels_per_world = 256.0 * float(1 << zoom)
    return (
        (maximum_x - minimum_x) / world_span * pixels_per_world,
        (maximum_y - minimum_y) / world_span * pixels_per_world,
    )


def _minimum_nonundersampling_zoom(grid: Mapping[str, Any]) -> int:
    """Smallest zoom whose screen span meets both raster dimensions."""

    span_x0, span_y0 = _screen_pixel_span(grid, 0)
    required_scale = max(
        float(grid["width"]) / span_x0,
        float(grid["height"]) / span_y0,
    )
    return max(0, int(math.ceil(math.log2(required_scale))))


def _legend_entries(legend: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_entries = legend.get("entries")
    if not isinstance(raw_entries, Sequence) or isinstance(raw_entries, (str, bytes)):
        raise ValueError("Legend has no entries")
    entries: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    for expected_class_id, raw in enumerate(raw_entries, 1):
        if not isinstance(raw, Mapping) or int(raw.get("class_id", 0)) != expected_class_id:
            raise ValueError("Legend class ids must be contiguous from one")
        label = str(raw.get("label", "")).strip()
        rgb = raw.get("rgb")
        if (
            not label
            or not isinstance(rgb, Sequence)
            or isinstance(rgb, (str, bytes))
            or len(rgb) != 3
        ):
            raise ValueError(f"Legend class {expected_class_id} is incomplete")
        color = [int(value) for value in rgb]
        if any(value < 0 or value > 255 for value in color):
            raise ValueError(f"Legend class {expected_class_id} has an invalid RGB color")
        identifier = re.sub(r"[^a-z0-9]+", "-", label.casefold()).strip("-")
        identifier = identifier or f"class-{expected_class_id}"
        if identifier in used_ids:
            identifier = f"{identifier}-{expected_class_id}"
        used_ids.add(identifier)
        entries.append(
            {
                "id": identifier,
                "class_id": expected_class_id,
                "label": label,
                "display_rgb": color,
            }
        )
    if not entries:
        raise ValueError("Legend has no entries")
    return entries


def _default_dataset_id(extraction_dir: Path) -> str:
    name = extraction_dir.name
    if name == "automatic-extraction" or name.startswith("extraction-run-"):
        name = extraction_dir.parent.name
    identifier = re.sub(r"[^a-z0-9]+", "-", name.casefold()).strip("-")
    return identifier or "automatic-categorical-preview"


def _validate_identifier(value: str, label: str) -> str:
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", value):
        raise ValueError(f"{label.capitalize()} must be a lowercase kebab-case id")
    return value


def export_automatic_categorical_staging_preview(
    extraction_dir: Path,
    output_dir: Path,
    *,
    dataset_id: str | None = None,
    title: str | None = None,
    layer_id: str = "classes",
    minimum_zoom: int = 4,
    maximum_zoom: int = 9,
    overview_supersampling: int = 4,
    tile_encoding: str = "per_class_rgba",
    retained_source_path: Path | None = None,
    retained_source_sha256: str | None = None,
    mapbox_reference_dir: Path | None = None,
    publication_state_mask_path: Path | None = None,
    publication_state_mask_sha256: str | None = None,
) -> dict[str, Any]:
    """Export an accepted automatic extraction as an unapproved staging preview."""

    extraction_input = extraction_dir.resolve()
    if extraction_input.is_file():
        if extraction_input.name != "accepted-extraction.json":
            raise ValueError("Extraction input file must be accepted-extraction.json")
        pointer_path = extraction_input
        extraction_dir = extraction_input.parent
    else:
        extraction_dir = extraction_input
        pointer_path = extraction_dir / "accepted-extraction.json"
    output_dir = output_dir.resolve()
    if not 0 <= minimum_zoom <= maximum_zoom <= 22:
        raise ValueError("Tile zooms must satisfy 0 <= minimum <= maximum <= 22")
    if overview_supersampling < 2 or overview_supersampling > 8:
        raise ValueError("Overview supersampling must be between two and eight")
    if tile_encoding not in {"per_class_rgba", "indexed_class_id"}:
        raise ValueError("Unsupported categorical tile encoding")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Preview output directory is not empty: {output_dir}")

    pointer = _json_object(pointer_path, "accepted extraction pointer")
    if pointer.get("schema_version") != EXTRACTION_SCHEMA:
        raise ValueError("Preview requires an automatic categorical extraction v1 result")
    if pointer.get("status") != "accepted":
        raise ValueError("Automatic extraction is not accepted")
    extraction_iteration = int(pointer.get("automatic_iteration_count", 0))
    accepted_iteration = str(pointer.get("accepted_iteration", ""))
    accepted_iteration_match = re.fullmatch(r"extraction-([0-9]+)", accepted_iteration)
    if extraction_iteration < 1 or accepted_iteration_match is None:
        raise ValueError("Accepted extraction iteration pointer is inconsistent")
    local_iteration = int(accepted_iteration_match.group(1))

    source_path, source_hash = _hash_bound_path(
        pointer.get("source"), base=extraction_dir, label="source image"
    )
    alignment_path, alignment_hash = _hash_bound_path(
        pointer.get("alignment"), base=extraction_dir, label="accepted alignment"
    )
    legend_path, legend_hash = _hash_bound_path(
        pointer.get("legend"),
        base=extraction_dir,
        label="legend",
        require_within_base=True,
    )
    alignment = _json_object(alignment_path, "accepted alignment")
    alignment_iteration = int(alignment.get("iteration", 0))
    if alignment_iteration < 1:
        raise ValueError("Accepted alignment has no automatic iteration count")
    grid, processing_reference = _processing_grid(
        pointer, alignment, extraction_dir
    )
    if mapbox_reference_dir is not None:
        if processing_reference is not None:
            raise ValueError(
                "Explicit base Mapbox reference is invalid for a supersampled extraction"
            )
        processing_reference = _base_mapbox_reference(
            alignment, mapbox_reference_dir, grid
        )
    minimum_native_zoom = _minimum_nonundersampling_zoom(grid)
    if maximum_zoom < minimum_native_zoom:
        span_x, span_y = _screen_pixel_span(grid, maximum_zoom)
        raise ValueError(
            "Maximum zoom undersamples the accepted class raster: "
            f"z{maximum_zoom} provides {span_x:.3f} by {span_y:.3f} pixels; "
            f"z{minimum_native_zoom} is the minimum non-undersampling zoom"
        )
    legend = _json_object(legend_path, "legend")
    if legend.get("schema_version") != EXTRACTION_SCHEMA:
        raise ValueError("Legend schema differs from the accepted extraction")
    categories = _legend_entries(legend)

    iteration_report_path = extraction_dir / accepted_iteration / "iteration.json"
    iteration_report = _json_object(iteration_report_path, "accepted iteration report")
    if (
        iteration_report.get("schema_version") != EXTRACTION_SCHEMA
        or int(iteration_report.get("iteration", 0)) != local_iteration
        or iteration_report.get("decision") != "accept"
        or iteration_report.get("legend_sha256") != legend_hash
    ):
        raise ValueError("Accepted iteration report does not match the extraction pointer")
    reported_grid = iteration_report.get("scores", {}).get("processing_target_grid")
    if reported_grid is not None and _validated_grid(
        reported_grid, "accepted iteration processing grid"
    ) != grid:
        raise ValueError("Accepted iteration processing grid differs from pointer")
    class_relative = f"{accepted_iteration}/web-mercator-class-id.png"
    class_path, class_hash = _report_artifact(
        iteration_report, extraction_dir, class_relative
    )
    with Image.open(class_path) as image:
        values = np.asarray(image)
    if values.ndim != 2 or values.shape != (grid["height"], grid["width"]):
        raise ValueError("Accepted class raster dimensions differ from target_grid")
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError("Accepted class raster must contain integer class ids")
    values = values.astype(np.uint16, copy=False)
    if int(values.max(initial=0)) > len(categories):
        raise ValueError("Accepted class raster contains ids absent from the legend")

    extent = pointer.get("aligned_source_extent")
    partial_extent = isinstance(extent, Mapping) and extent.get("partial_extent") is True
    missing_source_count = 0
    missing_mask_hash = None
    if partial_extent:
        missing_relative = "target-missing-source-extent-mask.png"
        missing_path, missing_mask_hash = _report_artifact(
            iteration_report, extraction_dir, missing_relative
        )
        with Image.open(missing_path) as image:
            missing = np.asarray(image.convert("L")) > 0
        if missing.shape != values.shape:
            raise ValueError("Missing-source-extent mask dimensions differ from target_grid")
        missing_source_count = int(np.count_nonzero(missing))
        if missing_source_count != int(extent.get("missing_source_extent_pixel_count", -1)):
            raise ValueError("Missing-source-extent count differs from accepted pointer")
        if np.any(values[missing] != 0):
            raise ValueError("Partial missing source extent contains classified pixels")

    resolved_dataset_id = _validate_identifier(
        dataset_id or _default_dataset_id(extraction_dir), "dataset id"
    )
    resolved_layer_id = _validate_identifier(layer_id, "layer id")
    resolved_title = title or resolved_dataset_id.replace("-", " ").title()
    bounds_3857 = tuple(float(value) for value in grid["bounds"])
    bounds_wgs84 = _wgs84_bounds(bounds_3857)

    output_dir.mkdir(parents=True, exist_ok=True)
    publication_values = values
    publication_clip = None
    publication_class_path = class_path
    publication_class_hash = class_hash
    if publication_state_mask_path is not None:
        mask_path = publication_state_mask_path.resolve()
        if not mask_path.is_file():
            raise FileNotFoundError(f"Missing publication state mask: {mask_path}")
        mask_hash = _sha256(mask_path)
        if (
            publication_state_mask_sha256 is not None
            and mask_hash != publication_state_mask_sha256
        ):
            raise ValueError("Publication state mask differs from its declared hash")
        with Image.open(mask_path) as image:
            state_interior = np.asarray(image.convert("L")) > 0
        if state_interior.shape != values.shape:
            raise ValueError("Publication state mask dimensions differ from target_grid")
        exterior_count = int(np.count_nonzero((values > 0) & ~state_interior))
        publication_values = values.copy()
        publication_values[~state_interior] = 0
        publication_class_path = output_dir / "publication-class-id.png"
        Image.fromarray(publication_values.astype(np.uint8)).save(
            publication_class_path, optimize=True
        )
        publication_class_hash = _sha256(publication_class_path)
        publication_clip = {
            "method": "direct_mapbox_state_interior_mask_v1",
            "state_interior_mask_path": str(mask_path),
            "state_interior_mask_sha256": mask_hash,
            "accepted_colored_pixel_count_outside_state": exterior_count,
            "publication_colored_pixel_count_outside_state": 0,
            "mutated_accepted_extraction": False,
        }
    retained_source = (
        retained_source_path.resolve()
        if retained_source_path is not None
        else source_path
    )
    if not retained_source.is_file():
        raise FileNotFoundError(f"Missing retained source image: {retained_source}")
    retained_source_hash = _sha256(retained_source)
    if (
        retained_source_sha256 is not None
        and retained_source_hash != retained_source_sha256
    ):
        raise ValueError("Retained source image differs from its declared hash")
    source_suffix = retained_source.suffix.lower() or ".img"
    copied_source = output_dir / f"source{source_suffix}"
    shutil.copy2(retained_source, copied_source)
    if _sha256(copied_source) != retained_source_hash:
        raise ValueError("Copied retained source image differs from its source")

    boundary = None
    if processing_reference is not None:
        overlay = processing_reference.get("state_coast_overlay")
        if isinstance(overlay, Mapping):
            overlay_source = Path(str(overlay["path"]))
            copied_overlay = output_dir / "mapbox-state-coast-overlay.png"
            shutil.copy2(overlay_source, copied_overlay)
            copied_overlay_hash = _sha256(copied_overlay)
            if copied_overlay_hash != overlay["sha256"]:
                raise ValueError(
                    "Copied Mapbox state/coast overlay differs from its pinned hash"
                )
            boundary = {
                "kind": "pinned_mapbox_state_coast_diagnostic",
                "authority": "accepted_alignment_mapbox_reference",
                "diagnostic_only": True,
                "raster": copied_overlay.name,
                "raster_sha256": copied_overlay_hash,
                "raster_width": int(overlay["width"]),
                "raster_height": int(overlay["height"]),
                "raster_bounds": list(bounds_wgs84),
            }

    category_paths: dict[int, list[Path]] = {
        category["class_id"]: [] for category in categories
    }
    indexed_raster = None
    if tile_encoding == "indexed_class_id":
        indexed_raster = _write_indexed_class_id_pyramid(
            publication_values.astype(np.uint8, copy=False),
            bounds_3857,
            output_dir,
            resolved_layer_id,
            len(categories),
            minimum_zoom,
            maximum_zoom,
            overview_supersampling,
        )
    else:
        for zoom in range(minimum_zoom, maximum_zoom + 1):
            x_range, y_range = _tile_range(bounds_3857, zoom)
            for tile_x in x_range:
                for tile_y in y_range:
                    coverage_alpha = None
                    if zoom < maximum_zoom:
                        class_tile, coverage_alpha = _sample_class_overview(
                            publication_values,
                            bounds_3857,
                            zoom,
                            tile_x,
                            tile_y,
                            len(categories),
                            overview_supersampling,
                        )
                    else:
                        class_tile = _sample_class_tile(
                            publication_values, bounds_3857, zoom, tile_x, tile_y
                        )
                    for category in categories:
                        class_id = int(category["class_id"])
                        tile_path = (
                            output_dir
                            / "tiles"
                            / resolved_layer_id
                            / str(category["id"])
                            / str(zoom)
                            / str(tile_x)
                            / f"{tile_y}.png"
                        )
                        tile_path.parent.mkdir(parents=True, exist_ok=True)
                        if coverage_alpha is None:
                            alpha = np.where(class_tile == class_id, 255, 0).astype(
                                np.uint8
                            )
                        else:
                            alpha = np.where(
                                class_tile == class_id, coverage_alpha, 0
                            ).astype(np.uint8)
                        white = np.full(alpha.shape, 255, dtype=np.uint8)
                        Image.fromarray(np.dstack((white, white, white, alpha))).save(
                            tile_path, optimize=True
                        )
                        category_paths[class_id].append(tile_path)

    category_manifests: list[dict[str, Any]] = []
    for category in categories:
        class_id = int(category["class_id"])
        category_id = str(category["id"])
        category_manifest = {
            **category,
            "pixel_count": int(np.count_nonzero(publication_values == class_id)),
        }
        if tile_encoding == "per_class_rgba":
            paths = category_paths[class_id]
            tilejson_dir = output_dir / "tiles" / resolved_layer_id / category_id
            tilejson_path = tilejson_dir / "tilejson.json"
            tilejson_path.write_text(
                json.dumps(
                    {
                        "tilejson": "3.0.0",
                        "name": str(category["label"]),
                        "scheme": "xyz",
                        "tiles": ["{z}/{x}/{y}.png"],
                        "minzoom": minimum_zoom,
                        "maxzoom": maximum_zoom,
                        "bounds": list(bounds_wgs84),
                    },
                    indent=2,
                )
                + "\n"
            )
            paths.append(tilejson_path)
            tile_set_hash = _aggregate_hash(paths, output_dir)
            category_manifest.update(
                {
                    "tile_template": (
                        f"tiles/{resolved_layer_id}/{category_id}/{{z}}/{{x}}/{{y}}.png"
                        f"?v={tile_set_hash[:16]}"
                    ),
                    "tilejson": (
                        f"tiles/{resolved_layer_id}/{category_id}/tilejson.json"
                    ),
                    "tile_file_count": len(paths) - 1,
                    "tile_set_sha256": tile_set_hash,
                    "encoding": "rgba_white_mask_with_coverage_overviews",
                }
            )
        category_manifests.append(category_manifest)

    provenance = {
        "schema_version": 1,
        "kind": "autonomous_automatic_categorical_staging_preview_provenance",
        "producer": PRODUCER,
        "status": "needs_visual_review",
        "publication_approved": False,
        "approval": {"status": "not_approved"},
        "boundary": boundary,
        "accepted_extraction": {
            "path": str(pointer_path),
            "sha256": _sha256(pointer_path),
            "schema_version": EXTRACTION_SCHEMA,
            "automatic_iteration_count": extraction_iteration,
            "accepted_iteration": accepted_iteration,
            "accepted_iteration_local_count": local_iteration,
            "iteration_report_sha256": _sha256(iteration_report_path),
        },
        "accepted_alignment": {
            "path": str(alignment_path),
            "sha256": alignment_hash,
            "automatic_iteration_count": alignment_iteration,
        },
        "source": {
            "path": str(retained_source),
            "sha256": retained_source_hash,
            "copied_path": copied_source.name,
            "copied_sha256": _sha256(copied_source),
        },
        "accepted_working_source": {
            "path": str(source_path),
            "sha256": source_hash,
        },
        "legend": {"path": str(legend_path), "sha256": legend_hash},
        "accepted_class_raster": {
            "path": str(class_path),
            "sha256": class_hash,
            "width": grid["width"],
            "height": grid["height"],
        },
        "class_raster": {
            "path": (
                publication_class_path.name
                if publication_class_path.parent == output_dir
                else str(publication_class_path)
            ),
            "sha256": publication_class_hash,
            "width": grid["width"],
            "height": grid["height"],
            "derivation": (
                "accepted_class_raster_clipped_to_pinned_mapbox_state_interior"
                if publication_clip is not None
                else "accepted_class_raster_unchanged"
            ),
        },
        "publication_clip": publication_clip,
        "native_resolution": {
            "minimum_nonundersampling_zoom": minimum_native_zoom,
            "screen_pixel_span_at_maximum_zoom": list(
                _screen_pixel_span(grid, maximum_zoom)
            ),
        },
        "target_grid": grid,
        "categorical_tile_encoding": tile_encoding,
        "processing_reference": processing_reference,
        "nodata": {
            "class_id": 0,
            "transparent": True,
            "pixel_count": int(np.count_nonzero(publication_values == 0)),
            "partial_source_extent": partial_extent,
            "missing_source_extent_pixel_count": missing_source_count,
            "missing_source_extent_mask_sha256": missing_mask_hash,
            "policy": "preserve_as_transparent_nodata_never_infer",
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
        "id": resolved_dataset_id,
        "title": resolved_title,
        "bounds": list(bounds_wgs84),
        "center": [
            (bounds_wgs84[0] + bounds_wgs84[2]) / 2.0,
            (bounds_wgs84[1] + bounds_wgs84[3]) / 2.0,
        ],
        "minimum_zoom": minimum_zoom,
        "maximum_native_zoom": maximum_zoom,
        "overscaling": "nearest",
        "categorical_tile_encoding": tile_encoding,
        "overview": {
            "mode": "dominant_class_with_fractional_coverage",
            "supersampling": overview_supersampling,
            "overview_zooms": list(range(minimum_zoom, maximum_zoom)),
            "exact_binary_zoom": maximum_zoom,
        },
        "source_image": copied_source.name,
        "approval": {"status": "not_approved"},
        "boundary": boundary,
        "provenance": {
            "manifest": provenance_path.name,
            "sha256": _sha256(provenance_path),
        },
        "layers": [
            {
                "id": resolved_layer_id,
                "label": resolved_title,
                "kind": "categorical",
                "bounds": list(bounds_wgs84),
                "nodata_class_id": 0,
                "categories": category_manifests,
                **(
                    {"indexed_raster": indexed_raster}
                    if indexed_raster is not None
                    else {}
                ),
            }
        ],
    }
    dataset_path = output_dir / "dataset.json"
    dataset_path.write_text(json.dumps(dataset, indent=2) + "\n")
    return {**dataset, "dataset_manifest_sha256": _sha256(dataset_path)}
