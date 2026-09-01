"""Audit authoritative offshore components against observed source evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
from PIL import Image

from .extraction import _target_state_mask
from .reference import load_california


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_declared_path(value: object, root: Path) -> Path:
    path = Path(str(value))
    candidates = [path] if path.is_absolute() else [Path.cwd() / path, root / path]
    resolved = next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)
    if resolved is None:
        raise FileNotFoundError(path)
    return resolved


def _closed_component(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        raise ValueError("Authoritative boundary component has no contour")
    contour = max(contours, key=cv2.contourArea)
    filled = np.zeros(mask.shape, dtype=np.uint8)
    border = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(filled, [contour], -1, 1, cv2.FILLED, cv2.LINE_8)
    cv2.drawContours(border, [contour], -1, 1, 1, cv2.LINE_8)
    if cv2.connectedComponents(border, 8)[0] - 1 != 1:
        raise ValueError("Authoritative island border is not one closed line")
    return filled > 0, border > 0


def audit_source_supported_boundary_components(
    extraction_run: Path,
    output_dir: Path,
    minimum_observed_pixels: int = 1,
    *,
    allow_legacy_snapshot: bool = False,
) -> Dict[str, object]:
    """Select only authoritative islands containing observed categorical ink."""

    extraction_run = extraction_run.resolve()
    output_dir = output_dir.resolve()
    if minimum_observed_pixels < 1:
        raise ValueError("Island support requires at least one observed pixel")
    extraction_path = extraction_run / "extraction.json"
    extraction = json.loads(extraction_path.read_text())
    layers = extraction.get("layers", [])
    if len(layers) != 1 or layers[0].get("kind") != "categorical":
        raise ValueError("Boundary-component audit currently requires one categorical layer")
    layer = layers[0]
    layer_id = str(layer["id"])
    observed_path = extraction_run / layer_id / "web-mercator-class-id.png"
    observed = np.asarray(Image.open(observed_path), dtype=np.uint8)
    grid = layer["warp"]
    if observed.shape != (int(grid["height"]), int(grid["width"])):
        raise ValueError("Observed raster and declared Web-Mercator grid differ")

    plan_record = extraction["plan"]
    plan_path = _resolve_declared_path(plan_record["path"], extraction_run)
    plan_hash_matches_extraction = _sha256(plan_path) == plan_record["sha256"]
    if not plan_hash_matches_extraction:
        snapshot_path = extraction_run / "plan.snapshot.json"
        if not allow_legacy_snapshot or not snapshot_path.is_file():
            raise ValueError("Extraction plan is stale")
        plan_path = snapshot_path
    plan = json.loads(plan_path.read_text())
    reference_root = Path(str(plan.get("reference", "reference/census-2025")))
    if not reference_root.is_absolute():
        reference_root = (Path.cwd() / reference_root).resolve()
    state, _ = load_california(reference_root)
    authoritative = _target_state_mask(state, grid["bounds"], observed.shape)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        authoritative.astype(np.uint8), 8
    )
    if count <= 1:
        raise ValueError("Authoritative California mask is empty")
    mainland_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    island_labels = sorted(
        (index for index in range(1, count) if index != mainland_label),
        key=lambda index: (
            -int(stats[index, cv2.CC_STAT_AREA]),
            int(stats[index, cv2.CC_STAT_TOP]),
            int(stats[index, cv2.CC_STAT_LEFT]),
        ),
    )
    selected_interior = np.zeros(observed.shape, dtype=bool)
    selected_border = np.zeros(observed.shape, dtype=bool)
    components = []
    selected_index = 0
    for authoritative_label in island_labels:
        raw_component = labels == authoritative_label
        support_count = int(np.count_nonzero(raw_component & (observed > 0)))
        selected = support_count >= minimum_observed_pixels
        filled, border = _closed_component(raw_component)
        if selected:
            selected_index += 1
            selected_interior |= filled
            selected_border |= border
        x, y, width, height, _ = (
            int(value) for value in stats[authoritative_label]
        )
        components.append(
            {
                "id": f"island-{selected_index:02d}" if selected else None,
                "role": "source_supported_island" if selected else "unsupported_island",
                "authoritative_component_label": authoritative_label,
                "selected": selected,
                "authority": "Census_2025_state_geometry",
                "interior_pixel_count": int(np.count_nonzero(filled)),
                "border_pixel_count": int(np.count_nonzero(border)),
                "observed_source_pixel_count": support_count,
                "observed_source_fraction": support_count
                / max(int(np.count_nonzero(filled)), 1),
                "bbox": [x, y, width, height],
                "centroid_pixel": [
                    float(centroids[authoritative_label][0]),
                    float(centroids[authoritative_label][1]),
                ],
            }
        )

    selected_count = sum(item["selected"] is True for item in components)
    actual_interior_count = cv2.connectedComponents(
        selected_interior.astype(np.uint8), 8
    )[0] - 1
    actual_border_count = cv2.connectedComponents(
        selected_border.astype(np.uint8), 8
    )[0] - 1
    if actual_interior_count != selected_count or actual_border_count != selected_count:
        raise AssertionError("Selected island component topology changed")
    output_dir.mkdir(parents=True, exist_ok=True)
    interior_path = output_dir / "web-mercator-source-supported-island-interior-mask.png"
    border_path = output_dir / "web-mercator-source-supported-island-border-overlay.png"
    Image.fromarray(selected_interior.astype(np.uint8) * 255).save(
        interior_path, optimize=True
    )
    border_rgba = np.zeros((*selected_border.shape, 4), dtype=np.uint8)
    border_rgba[selected_border] = [80, 255, 120, 255]
    Image.fromarray(border_rgba, mode="RGBA").save(border_path, optimize=True)
    report = {
        "schema_version": 1,
        "status": "pass" if plan_hash_matches_extraction else "diagnostic_pass",
        "method": "authoritative_islands_selected_by_observed_source_class_pixels",
        "selection_policy": {
            "qualifying_evidence": "raw_observed_web_mercator_class_id_only",
            "minimum_observed_pixel_count": minimum_observed_pixels,
            "manual_or_inferred_pixels_can_select_component": False,
        },
        "extraction": {"path": str(extraction_path), "sha256": _sha256(extraction_path)},
        "plan": {
            "path": str(plan_path),
            "sha256": _sha256(plan_path),
            "matches_extraction_declared_sha256": plan_hash_matches_extraction,
            "legacy_snapshot_override": not plan_hash_matches_extraction,
        },
        "alignment": extraction.get("alignment"),
        "observed_source": {
            "path": str(observed_path),
            "sha256": _sha256(observed_path),
        },
        "grid": grid,
        "mainland": {
            "authoritative_component_label": mainland_label,
            "interior_pixel_count": int(stats[mainland_label, cv2.CC_STAT_AREA]),
            "observed_source_pixel_count": int(
                np.count_nonzero((labels == mainland_label) & (observed > 0))
            ),
        },
        "islands": components,
        "selected_island_component_count": selected_count,
        "rejected_island_component_count": len(components) - selected_count,
        "artifacts": {
            "island_interior": {
                "path": interior_path.name,
                "sha256": _sha256(interior_path),
            },
            "island_border": {
                "path": border_path.name,
                "sha256": _sha256(border_path),
            },
        },
    }
    report_path = output_dir / "source-supported-boundary-components.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report
