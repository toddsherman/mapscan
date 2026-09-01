"""Observed-ink diagnostics for named line and polygon feature maps."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
from PIL import Image

from .extraction import _state_mask_in_source, warp_classified_to_web_mercator
from .reference import load_california


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _alignment_transform(alignment: Dict[str, object]) -> Dict[str, object]:
    """Return the extraction transform, including any reviewed grid correction."""

    if alignment.get("alignment_mode") == "assisted":
        transform = {
            "projection": "assisted_reference_crs",
            "projection_crs": alignment["reference"]["crs"],
            "transform_model": alignment["transform_model"],
            "reference_to_source_matrix": alignment["reference_to_source_matrix"],
        }
    else:
        transform = dict(alignment["best"])
    if "web_mercator_correction" in alignment:
        transform["web_mercator_correction"] = alignment["web_mercator_correction"]
    return transform


def extract_observed_feature_ink(plan_path: Path, output_dir: Path) -> Dict[str, object]:
    """Extract the source-observed ink mask without interpreting text as geometry."""

    plan = json.loads(plan_path.read_text())
    source_path = Path(plan["source"])
    alignment_path = Path(plan["alignment"])
    reference_root = Path(plan.get("reference", "reference/census-2025"))
    rgb = np.asarray(Image.open(source_path).convert("RGB"))
    alignment = json.loads(alignment_path.read_text())
    transform = _alignment_transform(alignment)
    state, _ = load_california(reference_root)
    state_mask = _state_mask_in_source(
        state, str(transform["projection_crs"]), transform, rgb.shape[:2]
    )
    gate = plan["observed_ink"]["initial_hsv_gate_opencv"]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    observed = (
        (hsv[:, :, 0] >= int(gate["hue_min"]))
        & (hsv[:, :, 0] <= int(gate["hue_max"]))
        & (hsv[:, :, 1] >= int(gate["saturation_min"]))
        & (hsv[:, :, 2] >= int(gate["value_min"]))
        & state_mask
    )
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(
        observed.astype(np.uint8), connectivity=8
    )
    areas = stats[1:, cv2.CC_STAT_AREA] if component_count > 1 else np.asarray([])
    preview = rgb.copy()
    preview[observed] = np.rint(
        preview[observed].astype(np.float32) * 0.25
        + np.asarray([0, 238, 238], dtype=np.float32) * 0.75
    ).astype(np.uint8)
    output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(observed.astype(np.uint8) * 255, mode="L").save(
        output_dir / "source-observed-ink.png", optimize=True
    )
    Image.fromarray(preview, mode="RGB").save(
        output_dir / "source-observed-ink-preview.png", quality=95, subsampling=0
    )
    warped_mask, warp_report = warp_classified_to_web_mercator(
        observed.astype(np.uint8), state, transform, rgb.shape[:2]
    )
    warped_source, _ = warp_classified_to_web_mercator(
        rgb, state, transform, rgb.shape[:2], target_height=int(warp_report["height"])
    )
    web_preview = warped_source.copy()
    web_selected = warped_mask > 0
    web_preview[web_selected] = np.rint(
        web_preview[web_selected].astype(np.float32) * 0.25
        + np.asarray([0, 238, 238], dtype=np.float32) * 0.75
    ).astype(np.uint8)
    Image.fromarray((warped_mask > 0).astype(np.uint8) * 255, mode="L").save(
        output_dir / "web-mercator-observed-ink.png", optimize=True
    )
    Image.fromarray(warped_source, mode="RGB").save(
        output_dir / "web-mercator-source.jpg", quality=94, subsampling=0
    )
    Image.fromarray(web_preview, mode="RGB").save(
        output_dir / "web-mercator-observed-ink-preview.png", quality=95, subsampling=0
    )
    result = {
        "schema_version": 1,
        "status": "needs_visual_review",
        "dataset_id": plan["dataset_id"],
        "plan": str(plan_path),
        "plan_sha256": _sha256(plan_path),
        "source": str(source_path),
        "source_sha256": _sha256(source_path),
        "alignment": str(alignment_path),
        "alignment_sha256": _sha256(alignment_path),
        "observed_ink": {
            "pixel_count_before_state_clip": int(
                np.count_nonzero(
                    (hsv[:, :, 0] >= int(gate["hue_min"]))
                    & (hsv[:, :, 0] <= int(gate["hue_max"]))
                    & (hsv[:, :, 1] >= int(gate["saturation_min"]))
                    & (hsv[:, :, 2] >= int(gate["value_min"]))
                )
            ),
            "pixel_count_after_state_clip": int(np.count_nonzero(observed)),
            "connected_component_count": int(component_count - 1),
            "components_at_least_10px": int(np.count_nonzero(areas >= 10)),
            "components_at_least_100px": int(np.count_nonzero(areas >= 100)),
            "hsv_gate_opencv": gate,
            "semantic_status": "uninterpreted_shared_blue_ink",
        },
        "warp": warp_report,
        "warnings": [
            "Observed blue ink includes both feature geometry and labels.",
            "No text pixels have been removed and no geometry has been reconnected.",
            "This evidence mask is not yet a publishable river or lake layer.",
        ],
    }
    (output_dir / "feature-extraction.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
