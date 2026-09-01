"""Extract categorical geology fills directly from a vector PDF."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Iterable

import cv2
import numpy as np
import pdfplumber
from PIL import Image

from .auto_refinement import _alignment_transform
from .extraction import (
    _fill_indexed_nodata_in_mask,
    _preview,
    _state_mask_in_source,
    warp_classified_to_web_mercator,
)
from .reference import load_california


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _color_key(color) -> tuple[float, ...] | None:
    if not isinstance(color, (tuple, list)):
        return None
    return tuple(round(float(value), 3) for value in color)


def _cmyk_to_rgb(color) -> list[int]:
    cyan, magenta, yellow, black = (float(value) for value in color)
    return [
        round(255 * (1.0 - min(1.0, cyan + black))),
        round(255 * (1.0 - min(1.0, magenta + black))),
        round(255 * (1.0 - min(1.0, yellow + black))),
    ]


def _slug(text: str, fallback: str) -> str:
    cleaned = "".join(character.lower() if character.isalnum() else "-" for character in text)
    cleaned = "-".join(part for part in cleaned.split("-") if part)
    return cleaned or fallback


def _categories(legend: Dict[str, object]) -> list[Dict[str, object]]:
    categories = []
    used = set()
    for index, swatch in enumerate(legend["swatches"], 1):
        raw_label = str(swatch.get("raw_pdf_text") or swatch.get("ocr_text") or "").strip()
        base_id = _slug(raw_label, f"unit-{index:03d}")
        category_id = base_id
        suffix = 2
        while category_id in used:
            category_id = f"{base_id}-{suffix}"
            suffix += 1
        used.add(category_id)
        categories.append(
            {
                "id": category_id,
                "label": raw_label or f"Geologic unit {index}",
                "class_id": index,
                "fill_cmyk": swatch["fill_cmyk"],
                "display_rgb": swatch.get(
                    "fill_rgb", _cmyk_to_rgb(swatch["fill_cmyk"])
                ),
                "label_status": swatch.get("label_status", "provisional"),
                "ocr_confidence": swatch.get("ocr_confidence"),
            }
        )
    return categories


def _cubic(
    start: np.ndarray,
    control_1: np.ndarray,
    control_2: np.ndarray,
    end: np.ndarray,
) -> Iterable[np.ndarray]:
    span = max(
        np.linalg.norm(control_1 - start),
        np.linalg.norm(control_2 - control_1),
        np.linalg.norm(end - control_2),
    )
    steps = max(4, min(32, int(math.ceil(span / 3.0))))
    for fraction in np.linspace(0.0, 1.0, steps + 1)[1:]:
        one_minus = 1.0 - fraction
        yield (
            one_minus**3 * start
            + 3 * one_minus**2 * fraction * control_1
            + 3 * one_minus * fraction**2 * control_2
            + fraction**3 * end
        )


def _path_contours(path, scale_x: float, scale_y: float) -> tuple[list[np.ndarray], set[str]]:
    contours = []
    current = []
    current_point = None
    unsupported = set()

    def point(value) -> np.ndarray:
        return np.asarray([float(value[0]) * scale_x, float(value[1]) * scale_y])

    def finish() -> None:
        nonlocal current
        if len(current) >= 3:
            contour = np.rint(np.asarray(current)).astype(np.int32).reshape((-1, 1, 2))
            contours.append(contour)
        current = []

    for command in path:
        operator = str(command[0])
        if operator == "m":
            finish()
            current_point = point(command[1])
            current = [current_point]
        elif operator == "l" and current_point is not None:
            current_point = point(command[1])
            current.append(current_point)
        elif operator == "c" and current_point is not None:
            control_1 = point(command[1])
            control_2 = point(command[2])
            end = point(command[3])
            current.extend(_cubic(current_point, control_1, control_2, end))
            current_point = end
        elif operator == "v" and current_point is not None:
            # PDF shorthand cubic: the first control point is the current point.
            control_2 = point(command[1])
            end = point(command[2])
            current.extend(_cubic(current_point, current_point, control_2, end))
            current_point = end
        elif operator == "y" and current_point is not None:
            # PDF shorthand cubic: the second control point is the end point.
            control_1 = point(command[1])
            end = point(command[2])
            current.extend(_cubic(current_point, control_1, end, end))
            current_point = end
        elif operator == "h":
            finish()
            current_point = None
        else:
            unsupported.add(operator)
    finish()
    return contours, unsupported


def extract_pdf_vector_fills(
    pdf_path: Path,
    alignment_path: Path,
    legend_path: Path,
    reference_root: Path,
    output_dir: Path,
) -> Dict[str, object]:
    """Rasterize only native PDF fills whose CMYK exactly matches a legend swatch."""

    output_dir.mkdir(parents=True, exist_ok=True)
    alignment = json.loads(alignment_path.read_text())
    legend = json.loads(legend_path.read_text())
    categories = _categories(legend)
    color_to_class = {
        _color_key(category["fill_cmyk"]): int(category["class_id"])
        for category in categories
    }
    source = alignment["source"]
    width = int(source["render_width_px"])
    height = int(source["render_height_px"])
    scale_x = float(source["render_x_px_per_page_point"])
    scale_y = float(source["render_y_px_per_page_point"])
    classified = np.zeros((height, width), dtype=np.uint8)
    matched_object_count = 0
    matched_path_count = 0
    unsupported_operators = set()
    per_class_objects = {int(category["class_id"]): 0 for category in categories}

    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[int(source.get("page_number", 1)) - 1]
        for curve in page.objects.get("curve", []):
            if not curve.get("fill"):
                continue
            class_id = color_to_class.get(_color_key(curve.get("non_stroking_color")))
            if class_id is None:
                continue
            contours, unsupported = _path_contours(curve.get("path", []), scale_x, scale_y)
            unsupported_operators.update(unsupported)
            if not contours:
                continue
            cv2.fillPoly(classified, contours, class_id, lineType=cv2.LINE_8)
            matched_object_count += 1
            matched_path_count += len(contours)
            per_class_objects[class_id] += 1

    state, _ = load_california(reference_root)
    transform = _alignment_transform(alignment, state)
    state_mask = _state_mask_in_source(
        state, str(transform["projection_crs"]), transform, classified.shape
    )
    classified[~state_mask] = 0
    direct = classified > 0
    completed, completed_count = _fill_indexed_nodata_in_mask(classified, state_mask)
    completion_mask = state_mask & ~direct & (completed > 0)
    warped, grid = warp_classified_to_web_mercator(
        completed, state, transform, classified.shape
    )
    source_path = output_dir / "source-class-id.png"
    web_path = output_dir / "web-mercator-class-id.png"
    completion_path = output_dir / "source-completion-mask.png"
    preview_path = output_dir / "source-preview.png"
    Image.fromarray(completed, mode="L").save(source_path, optimize=True)
    Image.fromarray(warped, mode="L").save(web_path, optimize=True)
    Image.fromarray(completion_mask.astype(np.uint8) * 255, mode="L").save(
        completion_path, optimize=True
    )
    Image.fromarray(_preview(completed, categories), mode="RGBA").save(
        preview_path, optimize=True
    )
    state_count = int(np.count_nonzero(state_mask))
    direct_count = int(np.count_nonzero(direct))
    category_pixels = {
        category["id"]: int(np.count_nonzero(completed == category["class_id"]))
        for category in categories
    }
    result = {
        "schema_version": 1,
        "status": "needs_legend_semantic_review",
        "extraction_kind": "native_pdf_vector_fills",
        "dataset_id": "california-geologic-map-2010",
        "title": "Geologic Map of California 2010",
        "source": {
            "path": str(pdf_path),
            "sha256": _sha256(pdf_path),
            "page_number": int(source.get("page_number", 1)),
            "width": width,
            "height": height,
        },
        "alignment": {"path": str(alignment_path), "sha256": _sha256(alignment_path)},
        "reference": {"path": str(reference_root)},
        "legend": {"path": str(legend_path), "sha256": _sha256(legend_path)},
        "categories": categories,
        "vector_extraction": {
            "matched_filled_curve_object_count": matched_object_count,
            "matched_contour_count": matched_path_count,
            "unsupported_path_operators": sorted(unsupported_operators),
            "matched_objects_by_class_id": per_class_objects,
            "state_pixel_count": state_count,
            "direct_vector_fill_pixel_count": direct_count,
            "direct_vector_fill_fraction": direct_count / max(state_count, 1),
            "completed_pixel_count": int(completed_count),
            "completion_method": "nearest native vector-fill class within authoritative state mask",
        },
        "category_pixel_counts": category_pixels,
        "warp": grid,
        "artifacts": {
            "source_class_id": {"path": source_path.name, "sha256": _sha256(source_path)},
            "web_mercator_class_id": {"path": web_path.name, "sha256": _sha256(web_path)},
            "source_completion_mask": {"path": completion_path.name, "sha256": _sha256(completion_path)},
            "source_preview": {"path": preview_path.name, "sha256": _sha256(preview_path)},
        },
        "semantic_limit": (
            "Fill colors and unit codes are native PDF evidence. Several geologic-age glyphs "
            "have nonstandard font encoding, so publication remains blocked until the unit "
            "codes and generalized rock descriptions are paired and validated."
        ),
    }
    (output_dir / "pdf-vector-extraction.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


def audit_pdf_vector_diff(run_dir: Path, output_dir: Path) -> Dict[str, object]:
    """Compare native vector coverage with the rendered source's legend-colored evidence."""

    manifest_path = run_dir / "pdf-vector-extraction.json"
    manifest = json.loads(manifest_path.read_text())
    alignment = json.loads(Path(manifest["alignment"]["path"]).read_text())
    render_path = Path(alignment["source"]["render_path"])
    rgb = np.asarray(Image.open(render_path).convert("RGB"))
    classified = np.asarray(
        Image.open(run_dir / manifest["artifacts"]["source_class_id"]["path"])
    )
    categories = manifest["categories"]
    prototypes = np.asarray([item["display_rgb"] for item in categories], dtype=np.int16)
    state, _ = load_california(Path(manifest["reference"]["path"]))
    transform = _alignment_transform(alignment, state)
    state_mask = _state_mask_in_source(
        state, str(transform["projection_crs"]), transform, classified.shape
    )
    # Rendered antialiasing can move edge pixels away from exact colors. A
    # conservative RGB radius identifies only clearly visible fill evidence.
    visible = np.zeros(classified.shape, dtype=bool)
    for prototype in prototypes:
        distance = np.linalg.norm(rgb.astype(np.float32) - prototype, axis=2)
        visible |= distance <= 8.0
    # The same colors also occur in the legend, marginalia, and neighboring
    # map decorations.  They are not map evidence and must not be counted as
    # extraction omissions.
    visible &= state_mask
    dropped = visible & (classified == 0)
    output_dir.mkdir(parents=True, exist_ok=True)
    dropped_path = output_dir / "rendered-visible-fill-dropped-mask.png"
    Image.fromarray(dropped.astype(np.uint8) * 255, mode="L").save(
        dropped_path, optimize=True
    )
    result = {
        "schema_version": 1,
        "audit_kind": "pdf_vector_render_diff",
        "status": "pass" if not np.any(dropped) else "fail",
        "dataset_id": manifest["dataset_id"],
        "run": str(run_dir),
        "rendered_visible_fill_pixel_count": int(np.count_nonzero(visible)),
        "dropped_rendered_visible_fill_pixel_count": int(np.count_nonzero(dropped)),
        "artifact": {"path": dropped_path.name, "sha256": _sha256(dropped_path)},
        "semantic_status": manifest["status"],
    }
    (output_dir / "source-diff-audit.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result
