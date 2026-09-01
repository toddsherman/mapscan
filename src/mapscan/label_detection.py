"""Detect neutral dark map labels without treating colored data as text."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
from PIL import Image


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def isolate_neutral_dark_labels(
    rgb: np.ndarray,
    valid_mask: np.ndarray,
    maximum_channel_value: int = 120,
    maximum_channel_spread: int = 38,
    closing_size_px: int = 2,
) -> np.ndarray:
    """Return a white OCR page containing only dark, nearly neutral pixels."""

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Label isolation requires an RGB image")
    if valid_mask.shape != rgb.shape[:2]:
        raise ValueError("Label isolation mask must match the source image")
    maximum = rgb.max(axis=2)
    minimum = rgb.min(axis=2)
    dark_neutral = (
        (maximum < maximum_channel_value)
        & ((maximum - minimum) < maximum_channel_spread)
        & valid_mask.astype(bool)
    )
    if closing_size_px > 1:
        kernel = np.ones((closing_size_px, closing_size_px), dtype=np.uint8)
        dark_neutral = cv2.morphologyEx(
            dark_neutral.astype(np.uint8), cv2.MORPH_CLOSE, kernel
        ).astype(bool)
    page = np.full(valid_mask.shape, 255, dtype=np.uint8)
    page[dark_neutral] = 0
    return page


def detect_apple_vision_label_regions(
    source_path: Path,
    valid_mask: np.ndarray,
    *,
    minimum_confidence: float = 0.5,
    minimum_letter_count: int = 3,
    padding_px: int = 3,
    maximum_word_dimension_px: int = 280,
    minimum_valid_overlap_pixels: int = 8,
    blackhat_kernel_px: int = 15,
    blackhat_threshold: int = 8,
    glyph_dilation_px: int = 2,
    glyph_component_minimum_area_px: int = 2,
    glyph_component_maximum_area_px: int = 240,
    glyph_component_maximum_dimension_px: int = 32,
    glyph_component_maximum_aspect: float = 8.0,
    glyph_component_minimum_extent: float = 0.05,
) -> tuple[np.ndarray, Dict[str, object]]:
    """Detect map-label boxes with macOS Vision and pin every accepted proposal.

    Vision is used because large, widely spaced, and rotated cartographic labels
    are frequently missed by axis-aligned Tesseract passes. The returned mask is
    evidence only: callers must retain it separately from reconstructed values.
    """

    source_path = source_path.resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Missing OCR source image: {source_path}")
    if valid_mask.ndim != 2:
        raise ValueError("Apple Vision label detection needs a two-dimensional mask")
    if not 0.0 <= minimum_confidence <= 1.0:
        raise ValueError("Apple Vision minimum confidence must be between zero and one")
    if minimum_letter_count < 1 or padding_px < 0:
        raise ValueError("Apple Vision label constraints are invalid")
    if blackhat_kernel_px < 3 or blackhat_threshold < 1 or glyph_dilation_px < 0:
        raise ValueError("Apple Vision glyph extraction constraints are invalid")
    if (
        glyph_component_minimum_area_px < 1
        or glyph_component_maximum_area_px < glyph_component_minimum_area_px
        or glyph_component_maximum_dimension_px < 1
        or glyph_component_maximum_aspect < 1.0
        or not 0.0 < glyph_component_minimum_extent <= 1.0
    ):
        raise ValueError("Apple Vision glyph component constraints are invalid")
    swiftc = shutil.which("swiftc")
    script = Path(__file__).resolve().parents[2] / "scripts" / "vision_text.swift"
    if swiftc is None or not script.is_file():
        raise RuntimeError("Apple Vision label detection requires swiftc and vision_text.swift")

    rgb = np.asarray(Image.open(source_path).convert("RGB"))
    if rgb.shape[:2] != valid_mask.shape:
        raise ValueError("Apple Vision source and valid mask dimensions differ")
    height, width = valid_mask.shape
    kernel_size = int(blackhat_kernel_px)
    if kernel_size % 2 == 0:
        kernel_size += 1
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blackhat = cv2.morphologyEx(
        gray,
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)),
    )
    dark_glyph_candidate = blackhat >= int(blackhat_threshold)
    accepted = []
    rejected = {
        "low_confidence_or_nonword": 0,
        "invalid_or_large_box": 0,
        "outside_valid_map": 0,
        "insufficient_glyph_support": 0,
    }
    label_mask = np.zeros(valid_mask.shape, dtype=np.uint8)
    with tempfile.TemporaryDirectory(prefix="mapscan-vision-labels-") as temporary:
        binary = Path(temporary) / "vision-text"
        compilation = subprocess.run(
            [swiftc, str(script), "-o", str(binary)],
            check=True,
            capture_output=True,
            text=True,
        )
        recognition = subprocess.run(
            [str(binary), str(source_path)],
            check=True,
            capture_output=True,
            text=True,
        )
    raw = json.loads(recognition.stdout)
    for item in raw:
        text = str(item.get("text", "")).strip()
        confidence = float(item.get("confidence", 0.0))
        letters = re.sub(r"[^A-Za-z]", "", text)
        if confidence < minimum_confidence or len(letters) < minimum_letter_count:
            rejected["low_confidence_or_nonword"] += 1
            continue
        normalized = item.get("normalized_bbox_bottom_left", [])
        if not isinstance(normalized, list) or len(normalized) != 4:
            rejected["invalid_or_large_box"] += 1
            continue
        x, bottom, box_width, box_height = (float(value) for value in normalized)
        x0 = max(0, int(np.floor(x * width)) - padding_px)
        y0 = max(0, int(np.floor((1.0 - bottom - box_height) * height)) - padding_px)
        x1 = min(width, int(np.ceil((x + box_width) * width)) + padding_px)
        y1 = min(height, int(np.ceil((1.0 - bottom) * height)) + padding_px)
        if (
            x1 <= x0
            or y1 <= y0
            or max(x1 - x0, y1 - y0) > maximum_word_dimension_px
        ):
            rejected["invalid_or_large_box"] += 1
            continue
        region = valid_mask[y0:y1, x0:x1].astype(bool)
        overlap = int(np.count_nonzero(region))
        if overlap < minimum_valid_overlap_pixels:
            rejected["outside_valid_map"] += 1
            continue
        candidate = dark_glyph_candidate[y0:y1, x0:x1] & region
        component_count, components, stats, _ = cv2.connectedComponentsWithStats(
            candidate.astype(np.uint8), connectivity=8
        )
        glyph = np.zeros(candidate.shape, dtype=bool)
        accepted_component_count = 0
        for component in range(1, component_count):
            area = int(stats[component, cv2.CC_STAT_AREA])
            component_width = int(stats[component, cv2.CC_STAT_WIDTH])
            component_height = int(stats[component, cv2.CC_STAT_HEIGHT])
            aspect = max(component_width, component_height) / max(
                min(component_width, component_height), 1
            )
            extent = area / max(component_width * component_height, 1)
            if (
                glyph_component_minimum_area_px <= area <= glyph_component_maximum_area_px
                and max(component_width, component_height)
                <= glyph_component_maximum_dimension_px
                and aspect <= glyph_component_maximum_aspect
                and extent >= glyph_component_minimum_extent
            ):
                glyph |= components == component
                accepted_component_count += 1
        if glyph_dilation_px > 0:
            size = glyph_dilation_px * 2 + 1
            glyph = cv2.dilate(
                glyph.astype(np.uint8), np.ones((size, size), dtype=np.uint8)
            ).astype(bool)
            glyph &= region
        if np.count_nonzero(glyph) < 3:
            rejected["insufficient_glyph_support"] += 1
            continue
        window = label_mask[y0:y1, x0:x1]
        newly_added = glyph & (window == 0)
        window[glyph] = 1
        accepted.append(
            {
                "text": text,
                "confidence": confidence,
                "bbox": [x0, y0, x1 - x0, y1 - y0],
                "valid_overlap_pixel_count": overlap,
                "glyph_pixel_count": int(np.count_nonzero(glyph)),
                "accepted_glyph_component_count": accepted_component_count,
                "new_mask_pixel_count": int(np.count_nonzero(newly_added)),
            }
        )

    mask = label_mask.astype(bool)
    report = {
        "method": "apple_vision_accurate_text_boxes_clipped_to_source_state",
        "engine": "Apple Vision VNRecognizeTextRequest",
        "source": str(source_path),
        "source_sha256": _sha256(source_path),
        "script": {"path": str(script), "sha256": _sha256(script)},
        "compiler": compilation.args[0],
        "raw_detection_count": len(raw),
        "accepted_detection_count": len(accepted),
        "label_occlusion_pixel_count": int(np.count_nonzero(mask)),
        "parameters": {
            "minimum_confidence": minimum_confidence,
            "minimum_letter_count": minimum_letter_count,
            "padding_px": padding_px,
            "maximum_word_dimension_px": maximum_word_dimension_px,
            "minimum_valid_overlap_pixels": minimum_valid_overlap_pixels,
            "blackhat_kernel_px": kernel_size,
            "blackhat_threshold": blackhat_threshold,
            "glyph_dilation_px": glyph_dilation_px,
            "glyph_component_minimum_area_px": glyph_component_minimum_area_px,
            "glyph_component_maximum_area_px": glyph_component_maximum_area_px,
            "glyph_component_maximum_dimension_px": (
                glyph_component_maximum_dimension_px
            ),
            "glyph_component_maximum_aspect": glyph_component_maximum_aspect,
            "glyph_component_minimum_extent": glyph_component_minimum_extent,
        },
        "accepted_detections": accepted,
        "rejected_detection_counts": rejected,
        "warning": (
            "OCR boxes are cartographic-occlusion evidence, never numeric evidence. "
            "Any values reconstructed beneath them must remain separately masked."
        ),
    }
    return mask, report


def detect_run_labels(
    run_dir: Path,
    output_dir: Optional[Path] = None,
    page_segmentation_mode: int = 3,
    maximum_channel_value: int = 120,
    maximum_channel_spread: int = 38,
    closing_size_px: int = 2,
) -> Dict[str, object]:
    """Create a neutral-text OCR page and Tesseract TSV for a processed run."""

    run_dir = run_dir.resolve()
    plan_path = run_dir / "plan.snapshot.json"
    state_mask_path = run_dir / "source-state-mask.png"
    if not plan_path.exists() or not state_mask_path.exists():
        raise FileNotFoundError(
            "Label detection needs plan.snapshot.json and source-state-mask.png"
        )
    plan = json.loads(plan_path.read_text())
    source_path = Path(plan["source"]).resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Missing source image: {source_path}")
    tesseract = shutil.which("tesseract")
    if tesseract is None:
        raise RuntimeError("Tesseract is required for neutral label detection")

    output_dir = (output_dir or (run_dir / "label-detection")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rgb = np.asarray(Image.open(source_path).convert("RGB"))
    valid_mask = np.asarray(Image.open(state_mask_path)) > 0
    page = isolate_neutral_dark_labels(
        rgb,
        valid_mask,
        maximum_channel_value=maximum_channel_value,
        maximum_channel_spread=maximum_channel_spread,
        closing_size_px=closing_size_px,
    )
    page_path = output_dir / "neutral-dark-labels.png"
    Image.fromarray(page, mode="L").save(page_path, optimize=True)
    output_base = output_dir / "neutral-dark-labels"
    subprocess.run(
        [
            tesseract,
            str(page_path),
            str(output_base),
            "-l",
            "eng",
            "--psm",
            str(page_segmentation_mode),
            "tsv",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    tsv_path = output_base.with_suffix(".tsv")
    result = {
        "schema_version": 1,
        "status": "needs_inference_review",
        "source": str(source_path),
        "source_sha256": _sha256(source_path),
        "state_mask": str(state_mask_path),
        "state_mask_sha256": _sha256(state_mask_path),
        "ocr_page": str(page_path),
        "ocr_page_sha256": _sha256(page_path),
        "ocr_tsv": str(tsv_path),
        "ocr_tsv_sha256": _sha256(tsv_path),
        "method": "neutral_dark_pixel_isolation_then_tesseract",
        "parameters": {
            "page_segmentation_mode": page_segmentation_mode,
            "maximum_channel_value": maximum_channel_value,
            "maximum_channel_spread": maximum_channel_spread,
            "closing_size_px": closing_size_px,
        },
        "warning": (
            "OCR boxes identify likely cartographic labels but are not data evidence. "
            "Any reconstruction beneath them must remain separately masked and reviewed."
        ),
    }
    (output_dir / "label-detection.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result
