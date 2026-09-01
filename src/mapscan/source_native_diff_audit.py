"""Source-native regional acceptance audit for automatic categorical results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
from PIL import Image
from pyproj import CRS, Transformer

from .automatic_categorical_extraction import (
    _projection_reference_to_source_base,
    _residual_displacement,
    _sha256,
    _supersample_alignment_transform,
)


SCHEMA_VERSION = "mapscan.source-native-categorical-diff-audit.v1"
PRODUCER = "mapscan.source_native_diff_audit"


def _load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L")) > 0


def _load_ids(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        values = np.asarray(image)
    if values.ndim != 2 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError(f"Class raster is not an integer plane: {path}")
    return values


def _artifact(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "sha256": _sha256(path),
        "byte_count": path.stat().st_size,
    }


def _reference_point_to_source(
    transform: Mapping[str, Any], longitude: float, latitude: float
) -> tuple[float, float]:
    world_x, world_y = Transformer.from_crs(
        "EPSG:4326", "EPSG:3857", always_xy=True
    ).transform(longitude, latitude)
    grid = transform["target_grid"]
    minimum_x, minimum_y, maximum_x, maximum_y = map(float, grid["bounds"])
    reference_x = (world_x - minimum_x) / (maximum_x - minimum_x) * (
        int(grid["width"]) - 1
    )
    reference_y = (maximum_y - world_y) / (maximum_y - minimum_y) * (
        int(grid["height"]) - 1
    )
    pixel_x = np.asarray([[reference_x]], dtype=np.float64)
    pixel_y = np.asarray([[reference_y]], dtype=np.float64)
    if transform["kind"] == "regular_global_mapbox_registration":
        points = np.stack((pixel_x, pixel_y), axis=-1).reshape(-1, 1, 2)
        matrix = np.asarray(
            transform["reference_pixel_to_source_original_matrix"], dtype=np.float64
        )
        mapped = cv2.perspectiveTransform(points, matrix).reshape(1, 1, 2)
        return float(mapped[0, 0, 0]), float(mapped[0, 0, 1])
    candidate = CRS.from_wkt(transform["projection"]["crs_wkt"])
    candidate_transformer = Transformer.from_crs(
        "EPSG:3857", candidate, always_xy=True
    )
    source_x, source_y = _projection_reference_to_source_base(
        transform, pixel_x, pixel_y, candidate_transformer
    )
    if transform["kind"] == "projection_aware_residual_warp_mapbox_registration":
        residual_x, residual_y = _residual_displacement(
            transform, pixel_x, pixel_y
        )
        source_x += residual_x
        source_y += residual_y
    return float(source_x[0, 0]), float(source_y[0, 0])


def run_source_native_diff_audit(
    extraction_dir: Path,
    output_dir: Path,
    *,
    rows: int = 6,
    columns: int = 6,
    monterey_wgs84: tuple[float, float] = (-121.8947, 36.6002),
    monterey_radius_source_px: int = 300,
    maximum_region_mismatch_fraction: float = 0.01,
) -> dict[str, Any]:
    """Accept only a complete, regionally faithful original-source reconstruction."""

    extraction_dir = extraction_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Source-native audit output already exists: {output_dir}")
    if rows < 2 or columns < 2:
        raise ValueError("Regional audit requires at least two rows and columns")
    pointer_path = extraction_dir / "accepted-extraction.json"
    pointer = json.loads(pointer_path.read_text())
    accepted = str(pointer["accepted_iteration"])
    report_path = extraction_dir / accepted / "iteration.json"
    iteration = json.loads(report_path.read_text())
    if iteration.get("decision") != "accept":
        raise ValueError("Source-native audit requires an accepted extraction")
    source_path = Path(pointer["source"]["path"]).resolve()
    alignment_path = Path(pointer["alignment"]["path"]).resolve()
    legend_path = extraction_dir / str(pointer["legend"]["path"])
    for path, record, label in (
        (source_path, pointer["source"], "source"),
        (alignment_path, pointer["alignment"], "alignment"),
        (legend_path, pointer["legend"], "legend"),
    ):
        if _sha256(path) != record["sha256"]:
            raise ValueError(f"Accepted {label} hash changed before source-native audit")
    iteration_dir = extraction_dir / accepted
    source_ids_path = iteration_dir / "source-class-id.png"
    observed_path = iteration_dir / "source-observed-mask.png"
    inferred_path = iteration_dir / "source-inferred-mask.png"
    reconstruction_path = iteration_dir / "source-reconstruction.png"
    mismatch_path = iteration_dir / "source-semantic-diff-mask.png"
    domain_path = extraction_dir / "source-data-mask.png"
    with Image.open(source_path) as image:
        source_rgb = np.asarray(image.convert("RGB"))
    with Image.open(reconstruction_path) as image:
        reconstruction = np.asarray(image.convert("RGB"))
    class_ids = _load_ids(source_ids_path)
    domain = _load_mask(domain_path)
    observed = _load_mask(observed_path)
    inferred = _load_mask(inferred_path)
    mismatch = _load_mask(mismatch_path)
    expected_shape = source_rgb.shape[:2]
    if reconstruction.shape[:2] != expected_shape or any(
        values.shape != expected_shape
        for values in (class_ids, domain, observed, inferred, mismatch)
    ):
        raise ValueError("Source-native audit inputs have inconsistent dimensions")

    output_dir.mkdir(parents=True)
    region_reports: list[dict[str, Any]] = []
    artifacts: list[Path] = []
    height, width = expected_shape
    for row in range(rows):
        top, bottom = round(row * height / rows), round((row + 1) * height / rows)
        for column in range(columns):
            left, right = round(column * width / columns), round(
                (column + 1) * width / columns
            )
            cell_domain = domain[top:bottom, left:right]
            domain_count = int(np.count_nonzero(cell_domain))
            if domain_count == 0:
                continue
            cell_ids = class_ids[top:bottom, left:right]
            cell_observed = observed[top:bottom, left:right] & cell_domain
            cell_inferred = inferred[top:bottom, left:right] & cell_domain
            cell_mismatch = mismatch[top:bottom, left:right] & cell_domain
            mismatch_count = int(np.count_nonzero(cell_mismatch))
            complete = bool(np.all(cell_ids[cell_domain] > 0))
            original_crop = source_rgb[top:bottom, left:right]
            reconstruction_crop = reconstruction[top:bottom, left:right]
            overlay = original_crop.copy()
            blended = cv2.addWeighted(original_crop, 0.5, reconstruction_crop, 0.5, 0)
            overlay[cell_domain] = blended[cell_domain]
            overlay[cell_mismatch] = (255, 0, 255)
            montage = np.concatenate(
                (original_crop, reconstruction_crop, overlay), axis=1
            )
            crop_path = output_dir / f"r{row + 1}-c{column + 1}-native-diff.png"
            Image.fromarray(montage).save(crop_path, optimize=True)
            artifacts.append(crop_path)
            region_reports.append(
                {
                    "id": f"r{row + 1}-c{column + 1}",
                    "source_pixel_bounds": [left, top, right, bottom],
                    "domain_pixel_count": domain_count,
                    "classified_pixel_count": int(
                        np.count_nonzero((cell_ids > 0) & cell_domain)
                    ),
                    "observed_pixel_count": int(np.count_nonzero(cell_observed)),
                    "inferred_pixel_count": int(np.count_nonzero(cell_inferred)),
                    "meaningful_mismatch_pixel_count": mismatch_count,
                    "meaningful_mismatch_fraction": mismatch_count / domain_count,
                    "complete": complete,
                    "passed": complete
                    and mismatch_count / domain_count
                    <= maximum_region_mismatch_fraction,
                    "artifact": crop_path.name,
                }
            )

    alignment = json.loads(alignment_path.read_text())
    factor = int(pointer.get("target_supersampling", 1))
    transform = _supersample_alignment_transform(alignment["transform"], factor)
    monterey_x, monterey_y = _reference_point_to_source(
        transform, *monterey_wgs84
    )
    center_x, center_y = round(monterey_x), round(monterey_y)
    left = max(0, center_x - monterey_radius_source_px)
    right = min(width, center_x + monterey_radius_source_px + 1)
    top = max(0, center_y - monterey_radius_source_px)
    bottom = min(height, center_y + monterey_radius_source_px + 1)
    monterey_domain = domain[top:bottom, left:right]
    monterey_ids = class_ids[top:bottom, left:right]
    monterey_mismatch = mismatch[top:bottom, left:right] & monterey_domain
    legend = json.loads(legend_path.read_text())
    labels = {
        int(entry["class_id"]): str(entry["label"])
        for entry in legend["entries"]
    }
    monterey_classes = [
        {
            "class_id": int(class_id),
            "label": labels[int(class_id)],
            "pixel_count": int(np.count_nonzero(monterey_ids == class_id)),
        }
        for class_id in np.unique(monterey_ids[monterey_ids > 0])
    ]
    monterey_original = source_rgb[top:bottom, left:right]
    monterey_reconstruction = reconstruction[top:bottom, left:right]
    monterey_overlay = cv2.addWeighted(
        monterey_original, 0.5, monterey_reconstruction, 0.5, 0
    )
    monterey_overlay[monterey_mismatch] = (255, 0, 255)
    monterey_path = output_dir / "monterey-native-diff.png"
    Image.fromarray(
        np.concatenate(
            (monterey_original, monterey_reconstruction, monterey_overlay), axis=1
        )
    ).save(monterey_path, optimize=True)
    artifacts.append(monterey_path)
    monterey_report = {
        "wgs84": list(monterey_wgs84),
        "source_center_pixel": [monterey_x, monterey_y],
        "source_pixel_bounds": [left, top, right, bottom],
        "radius_source_px": monterey_radius_source_px,
        "domain_pixel_count": int(np.count_nonzero(monterey_domain)),
        "classified_pixel_count": int(
            np.count_nonzero((monterey_ids > 0) & monterey_domain)
        ),
        "observed_pixel_count": int(
            np.count_nonzero(observed[top:bottom, left:right] & monterey_domain)
        ),
        "inferred_pixel_count": int(
            np.count_nonzero(inferred[top:bottom, left:right] & monterey_domain)
        ),
        "meaningful_mismatch_pixel_count": int(np.count_nonzero(monterey_mismatch)),
        "classes": monterey_classes,
        "artifact": monterey_path.name,
    }
    gates = {
        "six_by_six_regions_available": len(region_reports) >= 12,
        "all_available_regions_complete": all(
            report["complete"] for report in region_reports
        ),
        "all_available_regions_below_mismatch_limit": all(
            report["passed"] for report in region_reports
        ),
        "global_source_domain_complete": bool(np.all(class_ids[domain] > 0)),
        "global_source_layout_empty": not bool(np.any(class_ids[~domain] > 0)),
        "monterey_contains_classified_evidence": (
            monterey_report["classified_pixel_count"] > 0
        ),
        "monterey_meaningful_mismatch_empty": (
            monterey_report["meaningful_mismatch_pixel_count"] == 0
        ),
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "producer": PRODUCER,
        "decision": "accept" if all(gates.values()) else "reject",
        "source": {"path": str(source_path), "sha256": _sha256(source_path)},
        "accepted_extraction": {
            "path": str(pointer_path),
            "sha256": _sha256(pointer_path),
            "iteration": accepted,
            "iteration_report_sha256": _sha256(report_path),
        },
        "grid": {
            "coordinate_space": "original_source_pixels",
            "rows": rows,
            "columns": columns,
            "available_region_count": len(region_reports),
            "maximum_region_mismatch_fraction": maximum_region_mismatch_fraction,
        },
        "regions": region_reports,
        "monterey": monterey_report,
        "gates": gates,
        "artifacts": [_artifact(path, output_dir) for path in artifacts],
    }
    report_path_out = output_dir / "source-native-diff-audit.json"
    report_path_out.write_text(json.dumps(report, indent=2) + "\n")
    if report["decision"] != "accept":
        raise ValueError("Source-native regional diff audit rejected the extraction")
    return report
