"""Build an auditable source/extraction flip comparison for categorical maps."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Sequence

import cv2
import numpy as np
from PIL import Image
from pyproj import Transformer
from scipy.ndimage import distance_transform_edt


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> Dict[str, object]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _palette(categories: Sequence[Dict[str, object]]) -> np.ndarray:
    colors = []
    for category in categories:
        raw = category.get("display_rgb", category.get("legend_rgb"))
        color = raw[0] if isinstance(raw[0], list) else raw
        colors.append(color[:3])
    return np.asarray(colors, dtype=np.uint8)


def _render(values: np.ndarray, palette: np.ndarray) -> np.ndarray:
    result = np.full((*values.shape, 3), 255, dtype=np.uint8)
    for class_id, color in enumerate(palette, 1):
        result[values == class_id] = color
    return result


def _resize(rgb: np.ndarray, height: int) -> np.ndarray:
    if rgb.shape[0] <= height:
        return rgb
    width = round(rgb.shape[1] * height / rgb.shape[0])
    return cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)


def _distance_summary(values: np.ndarray) -> Dict[str, float | int | None]:
    if not len(values):
        return {"count": 0, "median": None, "p90": None, "max": None}
    return {
        "count": int(len(values)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.9)),
        "max": float(np.max(values)),
    }


def _observed_source_masks(
    preclip: np.ndarray,
    interior: np.ndarray,
    water: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep direct evidence visible even when the candidate mask removed it."""

    candidate_domain = interior | water
    observed_source = candidate_domain & (preclip > 0)
    removed_observed = observed_source & ~interior
    retained_observed = observed_source & interior
    return observed_source, retained_observed, removed_observed


def region_pixel_bounds(
    region: Dict[str, object], grid: Dict[str, object]
) -> tuple[int, int, int, int]:
    """Convert one WGS84 review box to a clipped raster pixel window."""

    raw = region.get("bounds_wgs84")
    if not isinstance(raw, list) or len(raw) != 4:
        raise ValueError("comparison region requires four bounds_wgs84 values")
    west, south, east, north = (float(value) for value in raw)
    if not west < east or not south < north:
        raise ValueError("comparison region bounds_wgs84 are not ordered")
    transformer = Transformer.from_crs(
        "EPSG:4326", str(grid["crs"]), always_xy=True
    )
    x0, y0 = transformer.transform(west, south)
    x1, y1 = transformer.transform(east, north)
    min_x, min_y, max_x, max_y = (float(value) for value in grid["bounds"])
    width, height = int(grid["width"]), int(grid["height"])
    left = max(0, min(width, int(np.floor((x0 - min_x) / (max_x - min_x) * width))))
    right = max(0, min(width, int(np.ceil((x1 - min_x) / (max_x - min_x) * width))))
    top = max(0, min(height, int(np.floor((max_y - y1) / (max_y - min_y) * height))))
    bottom = max(0, min(height, int(np.ceil((max_y - y0) / (max_y - min_y) * height))))
    if right <= left or bottom <= top:
        raise ValueError(f"comparison region lies outside the review grid: {region.get('id')}")
    return left, top, right, bottom


def _save_regional_comparisons(
    layer_dir: Path,
    regions: Sequence[Dict[str, object]],
    grid: Dict[str, object],
    source_rgb: np.ndarray,
    reconstructed: np.ndarray,
    comparison: np.ndarray,
    mismatch: np.ndarray,
    observed_source: np.ndarray,
    removed_observed: np.ndarray,
    changed_observed: np.ndarray,
    completed: np.ndarray,
    nearest_disagreement: np.ndarray,
    no_data: np.ndarray,
    outside: np.ndarray,
    colored_water: np.ndarray,
) -> list[Dict[str, object]]:
    """Write native-resolution local flips and evidence counts."""

    results: list[Dict[str, object]] = []
    if not regions:
        return results
    root = layer_dir / "regions"
    root.mkdir()
    for raw_region in regions:
        region_id = str(raw_region["id"])
        label = str(raw_region.get("label", region_id.replace("-", " ").title()))
        left, top, right, bottom = region_pixel_bounds(raw_region, grid)
        rows, cols = slice(top, bottom), slice(left, right)
        region_dir = root / region_id
        region_dir.mkdir()
        source_path = region_dir / "source.png"
        extracted_path = region_dir / "extracted.png"
        comparison_path = region_dir / "comparison.png"
        mismatch_path = region_dir / "mismatch-mask.png"
        montage_path = region_dir / "source-extraction-comparison.png"
        blink_path = region_dir / "source-extraction-blink.gif"
        source_crop = source_rgb[rows, cols]
        extracted_crop = reconstructed[rows, cols]
        comparison_crop = comparison[rows, cols]
        mismatch_crop = mismatch[rows, cols]
        Image.fromarray(source_crop, mode="RGB").save(source_path, optimize=True)
        Image.fromarray(extracted_crop, mode="RGB").save(
            extracted_path, optimize=True
        )
        Image.fromarray(comparison_crop, mode="RGB").save(
            comparison_path, optimize=True
        )
        Image.fromarray(mismatch_crop.astype(np.uint8) * 255, mode="L").save(
            mismatch_path, optimize=True
        )
        Image.fromarray(
            np.concatenate((source_crop, extracted_crop, comparison_crop), axis=1),
            mode="RGB",
        ).save(montage_path, optimize=True)
        Image.fromarray(source_crop, mode="RGB").save(
            blink_path,
            save_all=True,
            append_images=[Image.fromarray(extracted_crop, mode="RGB")],
            duration=[700, 700],
            loop=0,
            optimize=False,
        )
        metrics = {
            "observed_source_pixel_count": int(
                np.count_nonzero(observed_source[rows, cols])
            ),
            "observed_source_pixel_removed_count": int(
                np.count_nonzero(removed_observed[rows, cols])
            ),
            "observed_source_pixel_changed_count": int(
                np.count_nonzero(changed_observed[rows, cols])
            ),
            "derived_completion_pixel_count": int(
                np.count_nonzero(completed[rows, cols])
            ),
            "completion_nearest_class_disagreement_count": int(
                np.count_nonzero(nearest_disagreement[rows, cols])
            ),
            "interior_nodata_pixel_count": int(
                np.count_nonzero(no_data[rows, cols])
            ),
            "colored_outside_pixel_count": int(np.count_nonzero(outside[rows, cols])),
            "colored_water_pixel_count": int(
                np.count_nonzero(colored_water[rows, cols])
            ),
        }
        status = "pass" if all(
            metrics[key] == 0
            for key in (
                "observed_source_pixel_removed_count",
                "observed_source_pixel_changed_count",
                "completion_nearest_class_disagreement_count",
                "interior_nodata_pixel_count",
                "colored_outside_pixel_count",
                "colored_water_pixel_count",
            )
        ) else "fail"
        results.append(
            {
                "id": region_id,
                "label": label,
                "status": status,
                "bounds_wgs84": raw_region["bounds_wgs84"],
                "pixel_bounds": [left, top, right, bottom],
                "native_width": right - left,
                "native_height": bottom - top,
                "metrics": metrics,
                "artifacts": {
                    name: {
                        "path": str(path.relative_to(layer_dir)),
                        "sha256": _sha256(path),
                    }
                    for name, path in (
                        ("source", source_path),
                        ("extracted", extracted_path),
                        ("comparison", comparison_path),
                        ("mismatch_mask", mismatch_path),
                        ("montage", montage_path),
                        ("blink", blink_path),
                    )
                },
            }
        )
    return results


def build_categorical_comparison(
    run_dir: Path,
    fidelity_audit_path: Path,
    source_diff_batch_path: Path,
    case_id: str,
    output_dir: Path,
    *,
    review_height: int = 1400,
) -> Dict[str, object]:
    """Verify and render the fixed-point extraction against its aligned source."""

    run_dir = run_dir.resolve()
    fidelity_audit_path = fidelity_audit_path.resolve()
    source_diff_batch_path = source_diff_batch_path.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise ValueError("Categorical comparison requires a fresh output directory")
    output_dir.mkdir(parents=True)

    extraction_path = run_dir / "extraction.json"
    plan_path = run_dir / "plan.snapshot.json"
    extraction = _load(extraction_path)
    plan = _load(plan_path)
    fidelity = _load(fidelity_audit_path)
    batch = _load(source_diff_batch_path)
    if fidelity.get("status") != "pass":
        raise ValueError("Categorical fidelity audit has not passed")
    if fidelity.get("extraction_manifest_sha256") != _sha256(extraction_path):
        raise ValueError("Categorical fidelity audit targets another extraction")
    case = next(
        (item for item in batch.get("cases", []) if str(item.get("id")) == case_id),
        None,
    )
    if not isinstance(case, dict) or case.get("status") != "pass":
        raise ValueError("Source-diff case has not passed")
    if case.get("fixed_point_reached") is not True:
        raise ValueError("Source-diff case has not reached a fixed point")
    case_dir = source_diff_batch_path.parent / case_id
    case_report_path = source_diff_batch_path.parent / str(case["report"])
    case_report = _load(case_report_path)

    definitions = {str(item["id"]): item for item in plan["layers"]}
    manifests = {str(item["id"]): item for item in extraction["layers"]}
    fidelity_layers = {str(item["id"]): item for item in fidelity["layers"]}
    report_layers = {str(item["id"]): item for item in case_report["layers"]}
    comparison_source_path = run_dir / "web-mercator-source-before-context-clip.png"
    if not comparison_source_path.is_file():
        comparison_source_path = run_dir / "web-mercator-source.jpg"
    source_rgb = np.asarray(Image.open(comparison_source_path).convert("RGB"))
    comparison_regions = plan.get("comparison_regions", [])
    if not isinstance(comparison_regions, list):
        raise ValueError("comparison_regions must be a list")
    comparison_grid = extraction["alignment"]["inspection"]["grid"]
    layer_results = []

    for layer_id, definition in definitions.items():
        if definition.get("kind") != "categorical":
            continue
        manifest = manifests[layer_id]
        fidelity_layer = fidelity_layers[layer_id]
        diff_layer = report_layers[layer_id]
        baseline_path = run_dir / layer_id / "web-mercator-class-id.png"
        preclip_path = (
            run_dir / layer_id / "web-mercator-class-id-before-context-clip.png"
        )
        completion_path = run_dir / layer_id / "web-mercator-target-completion-mask.png"
        interior_path = run_dir / layer_id / "web-mercator-publication-interior-mask.png"
        water_path = run_dir / layer_id / "web-mercator-internal-water-mask.png"
        fixed_record = diff_layer["artifacts"]["audited_class_id"]
        fixed_path = case_dir / str(fixed_record["path"])
        if _sha256(fixed_path) != str(fixed_record["sha256"]):
            raise ValueError("Fixed-point categorical artifact is stale")

        baseline = np.asarray(Image.open(baseline_path), dtype=np.uint8)
        preclip = (
            np.asarray(Image.open(preclip_path), dtype=np.uint8)
            if preclip_path.is_file()
            else baseline.copy()
        )
        fixed = np.asarray(Image.open(fixed_path), dtype=np.uint8)
        completion = np.asarray(Image.open(completion_path).convert("L")) > 0
        interior = np.asarray(Image.open(interior_path).convert("L")) > 0
        water = (
            np.asarray(Image.open(water_path).convert("L")) > 0
            if water_path.is_file()
            else np.zeros(interior.shape, dtype=bool)
        )
        if not (
            baseline.shape
            == preclip.shape
            == fixed.shape
            == completion.shape
            == interior.shape
            and source_rgb.shape[:2] == fixed.shape
        ):
            raise ValueError("Categorical comparison artifacts use different grids")

        palette = _palette(definition["categories"])
        reconstructed = _render(fixed, palette)
        # Review rendering only: distinguish transparent water and exterior
        # from the legitimate pure-white lowest population class.
        reconstructed[~interior] = [22, 28, 36]
        reconstructed[water] = [66, 166, 225]
        observed_source, observed, removed_observed = _observed_source_masks(
            preclip, interior, water
        )
        changed_observed = observed & (fixed != preclip)
        no_data = interior & (fixed == 0)
        outside = ~interior & (fixed > 0)
        colored_water = water & (fixed > 0)
        completed = interior & completion
        if not np.any(observed):
            raise ValueError("Categorical comparison has no observed source evidence")
        _, nearest = distance_transform_edt(~observed, return_indices=True)
        nearest_observed_class = preclip[nearest[0], nearest[1]]
        nearest_disagreement = completed & (fixed != nearest_observed_class)

        observed_color = palette[preclip[observed] - 1].astype(np.float32)
        source_color = source_rgb[observed].astype(np.float32)
        jpeg_color_distance = np.linalg.norm(source_color - observed_color, axis=1)

        comparison = reconstructed.copy()
        if np.any(completed):
            comparison[completed] = np.rint(
                comparison[completed].astype(np.float32) * 0.45
                + np.asarray([40, 215, 245], dtype=np.float32) * 0.55
            ).astype(np.uint8)
        comparison[~interior] = 255
        comparison[removed_observed | changed_observed | nearest_disagreement] = [
            245,
            50,
            70,
        ]

        source_small = _resize(source_rgb, review_height)
        reconstructed_small = _resize(reconstructed, review_height)
        comparison_small = _resize(comparison, review_height)
        montage = np.concatenate(
            (source_small, reconstructed_small, comparison_small), axis=1
        )
        layer_dir = output_dir / layer_id
        layer_dir.mkdir()
        montage_path = layer_dir / "source-extraction-comparison.png"
        Image.fromarray(montage, mode="RGB").save(montage_path, optimize=True)
        blink_path = layer_dir / "source-extraction-blink.gif"
        Image.fromarray(source_small, mode="RGB").save(
            blink_path,
            save_all=True,
            append_images=[Image.fromarray(reconstructed_small, mode="RGB")],
            duration=[700, 700],
            loop=0,
            optimize=False,
        )
        completion_mask_path = layer_dir / "derived-completion-mask.png"
        Image.fromarray(completion.astype(np.uint8) * 255, mode="L").save(
            completion_mask_path, optimize=True
        )
        mismatch_path = layer_dir / "comparison-mismatch-mask.png"
        mismatch = (
            removed_observed
            | changed_observed
            | nearest_disagreement
            | no_data
            | outside
        )
        Image.fromarray(mismatch.astype(np.uint8) * 255, mode="L").save(
            mismatch_path, optimize=True
        )

        regional_comparisons = _save_regional_comparisons(
            layer_dir,
            comparison_regions,
            comparison_grid,
            source_rgb,
            reconstructed,
            comparison,
            mismatch,
            observed_source,
            removed_observed,
            changed_observed,
            completed,
            nearest_disagreement,
            no_data,
            outside,
            colored_water,
        )

        semantic_changes = int(fidelity_layer["semantic_class_change_pixel_count"])
        passed = bool(
            np.array_equal(fixed, baseline)
            and not np.any(removed_observed)
            and not np.any(changed_observed)
            and not np.any(nearest_disagreement)
            and not np.any(no_data)
            and not np.any(outside)
            and not np.any(colored_water)
            and semantic_changes == 0
            and all(item["status"] == "pass" for item in regional_comparisons)
        )
        layer_results.append(
            {
                "id": layer_id,
                "status": "pass" if passed else "fail",
                "fixed_point_matches_extraction_bytes": bool(
                    np.array_equal(fixed, baseline)
                ),
                "observed_source_pixel_count": int(
                    np.count_nonzero(observed_source)
                ),
                "observed_source_pixel_removed_by_context_or_water_count": int(
                    np.count_nonzero(removed_observed)
                ),
                "observed_source_pixel_changed_count": int(
                    np.count_nonzero(changed_observed)
                ),
                "derived_completion_pixel_count": int(np.count_nonzero(completed)),
                "completion_nearest_observed_class_disagreement_count": int(
                    np.count_nonzero(nearest_disagreement)
                ),
                "interior_nodata_pixel_count": int(np.count_nonzero(no_data)),
                "colored_outside_canonical_interior_pixel_count": int(
                    np.count_nonzero(outside)
                ),
                "colored_internal_water_pixel_count": int(
                    np.count_nonzero(colored_water)
                ),
                "semantic_class_change_across_threshold_ensemble_pixel_count": semantic_changes,
                "aligned_source_jpeg_distance_to_assigned_legend_rgb": _distance_summary(
                    jpeg_color_distance
                ),
                "threshold_iterations": fidelity_layer["variants"],
                "threshold_decision": fidelity_layer["threshold_decision"],
                "source_diff_iterations": case["comparison_iterations"],
                "regional_comparisons": regional_comparisons,
                "artifacts": {
                    "montage": {
                        "path": str(montage_path.relative_to(output_dir)),
                        "sha256": _sha256(montage_path),
                        "panels": [
                            "aligned clipped source",
                            "fixed-point reconstructed classes",
                            "reconstruction with derived completion cyan and mismatches red",
                        ],
                    },
                    "blink": {
                        "path": str(blink_path.relative_to(output_dir)),
                        "sha256": _sha256(blink_path),
                    },
                    "derived_completion_mask": {
                        "path": str(completion_mask_path.relative_to(output_dir)),
                        "sha256": _sha256(completion_mask_path),
                    },
                    "mismatch_mask": {
                        "path": str(mismatch_path.relative_to(output_dir)),
                        "sha256": _sha256(mismatch_path),
                    },
                },
            }
        )

    if not layer_results:
        raise ValueError("Run contains no categorical layers")
    result: Dict[str, object] = {
        "schema_version": 1,
        "audit_kind": "iterative_categorical_source_comparison",
        "status": (
            "pass" if all(item["status"] == "pass" for item in layer_results) else "fail"
        ),
        "run": str(run_dir),
        "comparison_source": {
            "path": str(comparison_source_path),
            "sha256": _sha256(comparison_source_path),
            "role": "lossless aligned source before contextual water and line removal",
        },
        "extraction_manifest_sha256": _sha256(extraction_path),
        "fidelity_audit": {
            "path": str(fidelity_audit_path),
            "sha256": _sha256(fidelity_audit_path),
        },
        "source_diff_batch": {
            "path": str(source_diff_batch_path),
            "sha256": _sha256(source_diff_batch_path),
            "case_id": case_id,
            "fixed_point_reached": True,
        },
        "layers": layer_results,
        "confidence_contract": [
            "all observed class IDs preserved",
            "no directly observed source class pixel removed by contextual water",
            "every derived land pixel inherits its nearest observed class",
            "zero interior NoData",
            "zero exterior color",
            "zero semantic class changes across the threshold ensemble",
            "two consecutive source-diff iterations produce identical bytes",
        ],
    }
    report_path = output_dir / "categorical-comparison.json"
    report_path.write_text(json.dumps(result, indent=2) + "\n")
    return result
