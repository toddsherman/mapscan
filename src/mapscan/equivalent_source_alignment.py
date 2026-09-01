"""Lift a normalized alignment onto a demonstrably equivalent source image."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
from PIL import Image


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rgb(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGBA")
    rgba = np.asarray(image, dtype=np.uint8)
    alpha = rgba[..., 3:4].astype(np.float32) / 255.0
    return np.rint(rgba[..., :3] * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)


def lift_alignment_to_equivalent_source(
    alignment_path: Path,
    new_source_path: Path,
    output_dir: Path,
    *,
    minimum_luma_correlation: float = 0.99,
    maximum_rgb_mae: float = 5.0,
    maximum_axis_scale_mismatch: float = 0.001,
) -> Dict[str, object]:
    """Reuse normalized transform parameters only after source-equivalence QA.

    This is intended for a higher-resolution copy of the same page/crop. It is
    not a general registration shortcut: content, crop, or aspect changes fail
    closed and must return to the normal alignment pipeline.
    """

    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Equivalent-source alignment output must be a fresh directory")
    output_dir.mkdir(parents=True, exist_ok=True)

    parent = json.loads(alignment_path.read_text())
    old_source_path = Path(str(parent["source"]["path"]))
    if not old_source_path.is_absolute():
        old_source_path = Path.cwd() / old_source_path
    old_source_path = old_source_path.resolve()
    new_source_path = new_source_path.resolve()
    if not old_source_path.is_file() or not new_source_path.is_file():
        raise ValueError("Both the parent and replacement source images must exist")
    if _sha256(old_source_path) != parent["source"]["sha256"]:
        raise ValueError("Parent source bytes do not match the alignment manifest")

    old_rgb = _rgb(old_source_path)
    new_rgb = _rgb(new_source_path)
    old_height, old_width = old_rgb.shape[:2]
    new_height, new_width = new_rgb.shape[:2]
    scale_x = new_width / old_width
    scale_y = new_height / old_height
    axis_scale_mismatch = abs(scale_x / scale_y - 1.0)
    resized = cv2.resize(
        new_rgb, (old_width, old_height), interpolation=cv2.INTER_AREA
    )
    absolute_difference = np.abs(
        old_rgb.astype(np.int16) - resized.astype(np.int16)
    )
    old_luma = cv2.cvtColor(old_rgb, cv2.COLOR_RGB2GRAY).astype(np.float64)
    new_luma = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY).astype(np.float64)
    luma_correlation = float(np.corrcoef(old_luma.ravel(), new_luma.ravel())[0, 1])
    rgb_mae = float(np.mean(absolute_difference))
    rgb_p95_absolute_error = float(np.percentile(absolute_difference, 95))
    passed = bool(
        np.isfinite(luma_correlation)
        and luma_correlation >= minimum_luma_correlation
        and rgb_mae <= maximum_rgb_mae
        and axis_scale_mismatch <= maximum_axis_scale_mismatch
    )
    report: Dict[str, object] = {
        "schema_version": 1,
        "status": "pass" if passed else "reject",
        "method": "same_crop_downsampled_rgb_equivalence",
        "parent_alignment": {
            "path": str(alignment_path),
            "sha256": _sha256(alignment_path),
        },
        "old_source": {
            "path": str(old_source_path),
            "sha256": _sha256(old_source_path),
            "width": old_width,
            "height": old_height,
        },
        "new_source": {
            "path": str(new_source_path),
            "sha256": _sha256(new_source_path),
            "width": new_width,
            "height": new_height,
        },
        "metrics": {
            "axis_scale_x": scale_x,
            "axis_scale_y": scale_y,
            "axis_scale_mismatch": axis_scale_mismatch,
            "luma_correlation_after_area_downsample": luma_correlation,
            "rgb_mae_after_area_downsample": rgb_mae,
            "rgb_p95_absolute_error_after_area_downsample": rgb_p95_absolute_error,
        },
        "thresholds": {
            "minimum_luma_correlation": minimum_luma_correlation,
            "maximum_rgb_mae": maximum_rgb_mae,
            "maximum_axis_scale_mismatch": maximum_axis_scale_mismatch,
        },
        "publication_allowed": False,
    }
    report_path = output_dir / "equivalent-source-lift.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    if not passed:
        raise ValueError("Replacement image is not equivalent to the aligned source")

    lifted = copy.deepcopy(parent)
    lifted["status"] = "diagnostic_only"
    lifted["alignment_mode"] = "equivalent_source_lift"
    lifted["source"] = {
        "path": str(new_source_path),
        "sha256": _sha256(new_source_path),
        "width": new_width,
        "height": new_height,
    }
    lifted["equivalent_source_lift"] = {
        "report": {
            "path": str(report_path),
            "sha256": _sha256(report_path),
        },
        "parent_alignment": report["parent_alignment"],
        "metrics": report["metrics"],
    }
    lifted["review"] = None
    lifted["warning"] = (
        "Normalized alignment parameters were lifted from a hash-bound, visually "
        "equivalent lower-resolution source. Fine alignment and author review are "
        "still required before extraction approval."
    )
    alignment_output_path = output_dir / "alignment.json"
    alignment_output_path.write_text(json.dumps(lifted, indent=2) + "\n")
    return lifted
