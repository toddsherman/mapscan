"""Deterministic fills for tiny single-class holes in categorical rasters."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image
from scipy import ndimage

from .manual_stamp import apply_clone_stamp_operations


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fill_small_enclosed_holes(
    values: np.ndarray,
    maximum_area_exclusive: int = 50,
    protected_zero_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Fill zero components smaller than the limit when one class surrounds them.

    Eight-neighbor connectivity is used for both the zero component and its
    boundary. Components touching the raster edge, meeting multiple classes, or
    intersecting an author-protected zero pixel remain zero.
    """

    if values.ndim != 2:
        raise ValueError("Enclosed-hole fill requires a two-dimensional class raster")
    if maximum_area_exclusive <= 1:
        raise ValueError("Maximum enclosed-hole area must be greater than 1")
    if protected_zero_mask is None:
        protected_zero_mask = np.zeros(values.shape, dtype=bool)
    elif protected_zero_mask.shape != values.shape:
        raise ValueError("Protected-zero mask must match the class raster")
    else:
        protected_zero_mask = np.asarray(protected_zero_mask, dtype=bool)

    structure = np.ones((3, 3), dtype=bool)
    labels, component_count = ndimage.label(values == 0, structure=structure)
    sizes = np.bincount(labels.ravel())
    objects = ndimage.find_objects(labels)
    fill_values = np.zeros(values.shape, dtype=np.uint8)
    fill_mask = np.zeros(values.shape, dtype=bool)
    filled_components = 0
    pixels_by_class: Dict[str, int] = {}

    candidate_ids = np.flatnonzero(
        (sizes > 0) & (sizes < maximum_area_exclusive)
    )
    for component_id in candidate_ids:
        if component_id == 0:
            continue
        component_slice = objects[int(component_id) - 1]
        if component_slice is None:
            continue
        y_slice, x_slice = component_slice
        if (
            y_slice.start == 0
            or x_slice.start == 0
            or y_slice.stop == values.shape[0]
            or x_slice.stop == values.shape[1]
        ):
            continue

        component = labels[component_slice] == component_id
        if np.any(protected_zero_mask[component_slice][component]):
            continue
        expanded = (
            slice(y_slice.start - 1, y_slice.stop + 1),
            slice(x_slice.start - 1, x_slice.stop + 1),
        )
        local_component = labels[expanded] == component_id
        boundary = ndimage.binary_dilation(
            local_component, structure=structure
        ) & ~local_component
        boundary_classes = np.unique(values[expanded][boundary])
        if boundary_classes.size != 1 or boundary_classes[0] == 0:
            continue

        class_id = int(boundary_classes[0])
        local_mask = fill_mask[component_slice]
        local_values = fill_values[component_slice]
        local_mask[component] = True
        local_values[component] = class_id
        filled_components += 1
        key = str(class_id)
        pixels_by_class[key] = pixels_by_class.get(key, 0) + int(
            sizes[component_id]
        )

    report: Dict[str, object] = {
        "connectivity": 8,
        "maximum_area_exclusive": int(maximum_area_exclusive),
        "zero_component_count": int(component_count),
        "filled_component_count": int(filled_components),
        "filled_pixel_count": int(np.count_nonzero(fill_mask)),
        "filled_pixels_by_class_id": pixels_by_class,
    }
    return fill_values, fill_mask, report


def generate_enclosed_fill_artifact(
    run_dir: Path,
    output_dir: Path,
    maximum_area_exclusive: int = 50,
) -> Dict[str, object]:
    """Create hashed, separately masked enclosed-hole fills for a review run."""

    run_dir = run_dir.resolve()
    output_dir = output_dir.resolve()
    manifest_path = run_dir / "extraction.json"
    stamp_path = run_dir / "stamp-corrections.json"
    for required in (manifest_path, stamp_path):
        if not required.exists():
            raise FileNotFoundError(f"Missing enclosed-fill input: {required}")
    manifest = json.loads(manifest_path.read_text())
    stamps = json.loads(stamp_path.read_text())
    if int(stamps.get("schema_version", 0)) not in {2, 3}:
        raise ValueError("Enclosed fill requires solid stamp correction schema 2 or 3")
    extraction_hash = _sha256(manifest_path)
    if stamps.get("extraction_manifest_sha256") != extraction_hash:
        raise ValueError("Stamp corrections do not match the extraction manifest")

    output_dir.mkdir(parents=True, exist_ok=True)
    operations = stamps.get("operations", [])
    reports = []
    for layer in manifest.get("layers", []):
        if layer.get("kind") != "categorical":
            continue
        layer_id = str(layer["id"])
        observed_path = run_dir / layer_id / "web-mercator-class-id.png"
        if not observed_path.exists():
            continue
        observed = np.asarray(Image.open(observed_path), dtype=np.uint8)
        layer_operations = [
            item for item in operations if item.get("layer_id") == layer_id
        ]
        manual_values, manual_mask = apply_clone_stamp_operations(
            observed, layer_operations
        )
        composed = observed.copy()
        composed[manual_mask] = manual_values[manual_mask]
        protected_zero = manual_mask & (manual_values == 0)
        fill_values, fill_mask, report = fill_small_enclosed_holes(
            composed,
            maximum_area_exclusive=maximum_area_exclusive,
            protected_zero_mask=protected_zero,
        )

        layer_dir = output_dir / layer_id
        layer_dir.mkdir(parents=True, exist_ok=True)
        values_path = layer_dir / "web-mercator-enclosed-fill-values.png"
        mask_path = layer_dir / "web-mercator-enclosed-fill-mask.png"
        Image.fromarray(fill_values).save(values_path, optimize=True)
        Image.fromarray(fill_mask.astype(np.uint8) * 255).save(
            mask_path, optimize=True
        )
        reports.append(
            {
                "layer_id": layer_id,
                **report,
                "values": {
                    "path": str(values_path.relative_to(output_dir)),
                    "sha256": _sha256(values_path),
                },
                "mask": {
                    "path": str(mask_path.relative_to(output_dir)),
                    "sha256": _sha256(mask_path),
                },
            }
        )

    if not reports:
        raise ValueError("No categorical layer was available for enclosed-hole fill")
    result: Dict[str, object] = {
        "schema_version": 1,
        "status": "author_rule",
        "dataset_id": manifest["dataset_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_run": str(run_dir),
        "extraction_manifest_sha256": extraction_hash,
        "stamp_corrections_sha256": _sha256(stamp_path),
        "policy": {
            "target": "zero_class_id_components_only",
            "connectivity": 8,
            "maximum_area_exclusive": int(maximum_area_exclusive),
            "boundary": "exactly_one_nonzero_class",
            "edge_components": "never_fill",
            "manual_zero_pixels": "never_fill",
        },
        "layers": reports,
    }
    (output_dir / "enclosed-fill.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result
