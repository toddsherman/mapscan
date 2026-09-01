import hashlib
import json

import numpy as np
import pytest
from PIL import Image

from mapscan.boundary_clip import (
    _load_mainland_boundary,
    clip_materialization_to_boundary,
)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _image(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(values, dtype=np.uint8)).save(path)
    return {"path": str(path), "sha256": _sha256(path)}


def test_boundary_clip_uses_displayed_ring_and_removes_all_outside_color(tmp_path):
    source_run = tmp_path / "run"
    materialized = tmp_path / "candidate-v1"
    perimeter_dir = tmp_path / "perimeter"
    output = tmp_path / "candidate-v2"
    layer_id = "hazard"

    plan_path = tmp_path / "plan.json"
    plan = {
        "layers": [
            {
                "id": layer_id,
                "categories": [
                    {"id": "one", "display_rgb": [10, 20, 30]},
                    {"id": "two", "display_rgb": [40, 50, 60]},
                ],
            }
        ]
    }
    plan_path.write_text(json.dumps(plan))
    source_run.mkdir()
    extraction = {
        "plan": {"path": str(plan_path), "sha256": _sha256(plan_path)},
    }
    extraction_path = source_run / "extraction.json"
    extraction_path.write_text(json.dumps(extraction))

    values = np.ones((7, 7), dtype=np.uint8)
    values[3, 3] = 0
    class_path = materialized / layer_id / "class.png"
    preview_path = materialized / layer_id / "preview.png"
    manual_path = materialized / layer_id / "manual.png"
    class_record = _image(class_path, values)
    preview_record = _image(preview_path, np.zeros((7, 7, 4), dtype=np.uint8))
    manual_record = _image(manual_path, np.ones((7, 7), dtype=np.uint8) * 255)
    manifest = {
        "status": "needs_visual_review",
        "source_run": str(source_run),
        "extraction_manifest_sha256": _sha256(extraction_path),
        "layers": [
            {
                "layer_id": layer_id,
                "artifacts": {
                    "class_id": {"path": f"{layer_id}/class.png", "sha256": class_record["sha256"]},
                    "preview": {"path": f"{layer_id}/preview.png", "sha256": preview_record["sha256"]},
                    "manual_mask": {"path": f"{layer_id}/manual.png", "sha256": manual_record["sha256"]},
                },
            }
        ],
    }
    (materialized / "materialization.json").write_text(json.dumps(manifest))

    interior = np.zeros((7, 7), dtype=np.uint8)
    interior[1:6, 1:6] = 255
    border = np.zeros((7, 7, 4), dtype=np.uint8)
    border[1, 1:6] = (80, 255, 120, 255)
    border[5, 1:6] = (80, 255, 120, 255)
    border[1:6, 1] = (80, 255, 120, 255)
    border[1:6, 5] = (80, 255, 120, 255)
    mask_path = perimeter_dir / "web-mercator-authoritative-mainland-interior-mask.png"
    border_path = perimeter_dir / "web-mercator-authoritative-unified-border-overlay.png"
    _image(mask_path, interior)
    _image(border_path, border)
    perimeter = {
        "status": "pass_no_additional_warp",
        "unified_border": {
            "passed": True,
            "connected_component_count": 1,
            "interior_is_exact_fill_of_displayed_border": True,
        },
        "artifacts": {
            mask_path.name: {"path": mask_path.name, "sha256": _sha256(mask_path)},
            border_path.name: {"path": border_path.name, "sha256": _sha256(border_path)},
        },
    }
    perimeter_path = perimeter_dir / "hybrid-perimeter-audit.json"
    perimeter_path.write_text(json.dumps(perimeter))

    result = clip_materialization_to_boundary(materialized, perimeter_path, output)

    layer = result["layers"][0]
    final = np.asarray(Image.open(output / layer["artifacts"]["class_id"]["path"]))
    assert np.count_nonzero(final) == 25
    assert np.count_nonzero(final[interior == 0]) == 0
    assert final[3, 3] == 1
    assert layer["boundary_removed_pixel_count"] == 24
    assert layer["boundary_completion_pixel_count"] == 1
    assert result["boundary_clip"]["colored_pixel_count_outside_boundary"] == 0
    clipped_manual = np.asarray(Image.open(output / layer["artifacts"]["manual_mask"]["path"]))
    assert np.count_nonzero(clipped_manual[interior == 0]) == 0
    assert not (output / "materialization-review-decision.json").exists()


def test_boundary_clip_refuses_an_output_directory_with_a_prior_approval(tmp_path):
    output = tmp_path / "candidate-v3"
    output.mkdir()
    (output / "materialization-review-decision.json").write_text(
        json.dumps({"status": "approved"})
    )

    with pytest.raises(ValueError, match="fresh versioned output directory"):
        clip_materialization_to_boundary(
            tmp_path / "candidate-v2",
            tmp_path / "hybrid-perimeter-audit.json",
            output,
        )


def test_boundary_clip_accepts_a_hash_bound_canonical_mainland_audit(tmp_path):
    interior_path = tmp_path / "canonical-interior.png"
    border_path = tmp_path / "canonical-border.png"
    _image(interior_path, np.ones((3, 3), dtype=np.uint8) * 255)
    _image(border_path, np.ones((3, 3, 4), dtype=np.uint8) * 255)
    audit = {
        "status": "pass",
        "canonical_boundary_id": "california-mainland-hybrid-v1",
        "topology": {
            "interior_component_count": 1,
            "border_component_count": 1,
            "interior_is_exact_fill_of_displayed_border": True,
        },
        "artifacts": {
            "interior": {"path": interior_path.name, "sha256": _sha256(interior_path)},
            "border": {"path": border_path.name, "sha256": _sha256(border_path)},
        },
    }
    audit_path = tmp_path / "canonical-boundary-raster.json"
    audit_path.write_text(json.dumps(audit))

    _, loaded_interior, loaded_border, boundary_id = _load_mainland_boundary(
        audit_path
    )

    assert loaded_interior == interior_path
    assert loaded_border == border_path
    assert boundary_id == "california-mainland-hybrid-v1"


def test_boundary_clip_preserves_only_hash_bound_source_supported_islands(tmp_path):
    source_run = tmp_path / "run"
    materialized = tmp_path / "candidate-v1"
    perimeter_dir = tmp_path / "perimeter"
    component_dir = tmp_path / "components"
    output = tmp_path / "candidate-v2"
    layer_id = "hazard"
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "layers": [
                    {
                        "id": layer_id,
                        "categories": [{"id": "one", "display_rgb": [10, 20, 30]}],
                    }
                ]
            }
        )
    )
    source_run.mkdir()
    extraction_path = source_run / "extraction.json"
    extraction_path.write_text(
        json.dumps({"plan": {"path": str(plan_path), "sha256": _sha256(plan_path)}})
    )
    values = np.ones((9, 9), dtype=np.uint8)
    class_path = materialized / layer_id / "class.png"
    preview_path = materialized / layer_id / "preview.png"
    class_record = _image(class_path, values)
    preview_record = _image(preview_path, np.zeros((9, 9, 4), dtype=np.uint8))
    manifest = {
        "status": "needs_visual_review",
        "source_run": str(source_run),
        "extraction_manifest_sha256": _sha256(extraction_path),
        "layers": [
            {
                "layer_id": layer_id,
                "artifacts": {
                    "class_id": {"path": f"{layer_id}/class.png", "sha256": class_record["sha256"]},
                    "preview": {"path": f"{layer_id}/preview.png", "sha256": preview_record["sha256"]},
                },
            }
        ],
    }
    (materialized / "materialization.json").write_text(json.dumps(manifest))

    mainland = np.zeros((9, 9), dtype=np.uint8)
    mainland[2:7, 2:7] = 255
    mainland_border = np.zeros((9, 9, 4), dtype=np.uint8)
    mainland_border[2, 2:7] = [80, 255, 120, 255]
    mainland_border[6, 2:7] = [80, 255, 120, 255]
    mainland_border[2:7, 2] = [80, 255, 120, 255]
    mainland_border[2:7, 6] = [80, 255, 120, 255]
    mainland_path = perimeter_dir / "web-mercator-authoritative-mainland-interior-mask.png"
    mainland_border_path = perimeter_dir / "web-mercator-authoritative-unified-border-overlay.png"
    _image(mainland_path, mainland)
    _image(mainland_border_path, mainland_border)
    perimeter = {
        "status": "pass_no_additional_warp",
        "unified_border": {
            "passed": True,
            "connected_component_count": 1,
            "interior_is_exact_fill_of_displayed_border": True,
        },
        "artifacts": {
            mainland_path.name: {"path": mainland_path.name, "sha256": _sha256(mainland_path)},
            mainland_border_path.name: {"path": mainland_border_path.name, "sha256": _sha256(mainland_border_path)},
        },
    }
    perimeter_path = perimeter_dir / "hybrid-perimeter-audit.json"
    perimeter_path.write_text(json.dumps(perimeter))

    islands = np.zeros((9, 9), dtype=np.uint8)
    islands[0, 0] = 255
    islands[8, 8] = 255
    island_border = np.zeros((9, 9, 4), dtype=np.uint8)
    island_border[0, 0] = [80, 255, 120, 255]
    island_border[8, 8] = [80, 255, 120, 255]
    island_path = component_dir / "islands.png"
    island_border_path = component_dir / "island-border.png"
    _image(island_path, islands)
    _image(island_border_path, island_border)
    component_audit = {
        "status": "pass",
        "extraction": {"sha256": _sha256(extraction_path)},
        "selection_policy": {
            "qualifying_evidence": "raw_observed_web_mercator_class_id_only",
            "manual_or_inferred_pixels_can_select_component": False,
        },
        "islands": [
            {"id": "island-01", "role": "source_supported_island", "selected": True, "interior_pixel_count": 1, "border_pixel_count": 1, "observed_source_pixel_count": 2},
            {"id": "island-02", "role": "source_supported_island", "selected": True, "interior_pixel_count": 1, "border_pixel_count": 1, "observed_source_pixel_count": 1},
            {"id": None, "role": "unsupported_island", "selected": False, "observed_source_pixel_count": 0},
        ],
        "artifacts": {
            "island_interior": {"path": island_path.name, "sha256": _sha256(island_path)},
            "island_border": {"path": island_border_path.name, "sha256": _sha256(island_border_path)},
        },
    }
    component_audit_path = component_dir / "source-supported-boundary-components.json"
    component_audit_path.write_text(json.dumps(component_audit))

    result = clip_materialization_to_boundary(
        materialized,
        perimeter_path,
        output,
        component_audit_path=component_audit_path,
    )

    final = np.asarray(Image.open(output / layer_id / "class.png"))
    assert np.count_nonzero(final) == 27
    assert final[0, 0] == 1 and final[8, 8] == 1
    assert final[0, 8] == 0
    assert result["boundary_clip"]["boundary_component_count"] == 3
    assert result["boundary_clip"]["expected_boundary_component_count"] == 3
    assert len(result["boundary_clip"]["components"]) == 3
