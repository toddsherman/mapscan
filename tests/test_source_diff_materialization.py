import hashlib
import json

import numpy as np
import pytest
from PIL import Image

from mapscan.source_diff_materialization import promote_source_diff_materialization


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_image(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(values, dtype=np.uint8)).save(path)
    return {"path": str(path), "sha256": _sha256(path)}


def test_promotes_fixed_point_surface_and_preserves_completion_provenance(tmp_path):
    materialized = tmp_path / "materialized"
    batch_dir = tmp_path / "diff"
    output = tmp_path / "final"
    layer_id = "hazard"
    base_class = materialized / layer_id / "class.png"
    base_preview = materialized / layer_id / "preview.png"
    manual = materialized / layer_id / "manual.png"
    _write_image(base_class, [[1, 0], [0, 2]])
    _write_image(base_preview, np.zeros((2, 2, 4), dtype=np.uint8))
    _write_image(manual, [[255, 0], [0, 0]])
    manifest = {
        "status": "needs_visual_review",
        "dataset_id": "test",
        "extraction_manifest_sha256": "extract-hash",
        "layers": [
            {
                "layer_id": layer_id,
                "artifacts": {
                    "class_id": {"path": f"{layer_id}/class.png", "sha256": _sha256(base_class)},
                    "preview": {"path": f"{layer_id}/preview.png", "sha256": _sha256(base_preview)},
                    "manual_mask": {"path": f"{layer_id}/manual.png", "sha256": _sha256(manual)},
                },
            }
        ],
    }
    materialized.mkdir(exist_ok=True)
    (materialized / "materialization.json").write_text(json.dumps(manifest))

    case_dir = batch_dir / "case"
    final_class = case_dir / layer_id / "audited.png"
    final_preview = case_dir / layer_id / "audited-preview.png"
    _write_image(final_class, [[1, 2], [2, 2]])
    _write_image(final_preview, np.ones((2, 2, 4), dtype=np.uint8))
    first_dir = case_dir / "iteration-01"
    completion = first_dir / layer_id / "completion.png"
    _write_image(completion, [[0, 255], [255, 0]])
    first_report = {
        "layers": [
            {
                "id": layer_id,
                "completed_pixel_count": 2,
                "artifacts": {
                    "completion": {
                        "path": f"{layer_id}/completion.png",
                        "sha256": _sha256(completion),
                    }
                },
            }
        ]
    }
    first_report_path = first_dir / "source-diff-audit.json"
    first_report_path.write_text(json.dumps(first_report))
    final_report = {
        "status": "pass",
        "manifest_sha256": "extract-hash",
        "layers": [
            {
                "id": layer_id,
                "dropped_visible_source_evidence_pixel_count": 0,
                "unclassified_pixel_count_after": 0,
                "artifacts": {
                    "audited_class_id": {
                        "path": f"{layer_id}/audited.png",
                        "sha256": _sha256(final_class),
                    },
                    "audited_preview": {
                        "path": f"{layer_id}/audited-preview.png",
                        "sha256": _sha256(final_preview),
                    },
                },
            }
        ],
    }
    final_report_path = case_dir / "source-diff-audit.json"
    final_report_path.write_text(json.dumps(final_report))
    batch = {
        "status": "pass",
        "cases": [
            {
                "id": "case",
                "status": "pass",
                "fixed_point_reached": True,
                "report": "case/source-diff-audit.json",
                "comparison_iterations": [
                    {"iteration": 1, "report": "case/iteration-01/source-diff-audit.json"},
                    {"iteration": 2, "report": "case/iteration-02/source-diff-audit.json"},
                ],
            }
        ],
    }
    batch_dir.mkdir(exist_ok=True)
    (batch_dir / "source-diff-batch.json").write_text(json.dumps(batch))

    result = promote_source_diff_materialization(materialized, batch_dir, "case", output)

    layer = result["layers"][0]
    assert layer["final_classified_pixel_count"] == 4
    assert layer["source_diff_completion_pixel_count"] == 2
    assert layer["dropped_visible_source_evidence_pixel_count"] == 0
    assert layer["unclassified_pixel_count_after"] == 0
    assert _sha256(output / layer_id / "class.png") == _sha256(final_class)
    assert _sha256(output / layer_id / "manual.png") == _sha256(manual)
    assert not (output / "materialization-review-decision.json").exists()


def test_promotion_refuses_an_output_directory_with_a_prior_approval(tmp_path):
    output = tmp_path / "candidate-v2"
    output.mkdir()
    (output / "materialization-review-decision.json").write_text(
        json.dumps({"status": "approved"})
    )

    with pytest.raises(ValueError, match="fresh versioned output directory"):
        promote_source_diff_materialization(
            tmp_path / "candidate-v1",
            tmp_path / "source-diff",
            "plant-hardiness",
            output,
        )


def test_promotes_legacy_iteration_only_fixed_point_artifacts(tmp_path):
    materialized = tmp_path / "materialized"
    batch_dir = tmp_path / "diff"
    output = tmp_path / "final"
    layer_id = "zones"
    base_class = materialized / layer_id / "class.png"
    base_preview = materialized / layer_id / "preview.png"
    _write_image(base_class, [[1, 0], [0, 2]])
    _write_image(base_preview, np.zeros((2, 2, 4), dtype=np.uint8))
    manifest = {
        "status": "needs_visual_review",
        "dataset_id": "legacy-layout",
        "extraction_manifest_sha256": "extract-hash",
        "layers": [
            {
                "layer_id": layer_id,
                "artifacts": {
                    "class_id": {
                        "path": f"{layer_id}/class.png",
                        "sha256": _sha256(base_class),
                    },
                    "preview": {
                        "path": f"{layer_id}/preview.png",
                        "sha256": _sha256(base_preview),
                    },
                },
            }
        ],
    }
    materialized.mkdir(exist_ok=True)
    (materialized / "materialization.json").write_text(json.dumps(manifest))

    case_dir = batch_dir / "case"
    first_dir = case_dir / "iteration-01"
    stable_dir = case_dir / "iteration-02"
    final_class = stable_dir / layer_id / "audited.png"
    final_preview = stable_dir / layer_id / "audited-preview.png"
    completion = first_dir / layer_id / "completion.png"
    _write_image(final_class, [[1, 2], [2, 2]])
    _write_image(final_preview, np.ones((2, 2, 4), dtype=np.uint8))
    _write_image(completion, [[0, 255], [255, 0]])
    first_report = {
        "layers": [
            {
                "id": layer_id,
                "completed_pixel_count": 2,
                "artifacts": {
                    "completion": {
                        "path": f"{layer_id}/completion.png",
                        "sha256": _sha256(completion),
                    }
                },
            }
        ]
    }
    first_report_path = first_dir / "source-diff-audit.json"
    first_report_path.parent.mkdir(parents=True, exist_ok=True)
    first_report_path.write_text(json.dumps(first_report))
    final_report = {
        "status": "pass",
        "manifest_sha256": "extract-hash",
        "layers": [
            {
                "id": layer_id,
                "candidate": {
                    "path": str(final_class),
                    "sha256": _sha256(final_class),
                },
                "dropped_visible_source_evidence_pixel_count": 0,
                "unclassified_pixel_count_after": 0,
                "artifacts": {
                    "audited_class_id": {
                        "path": f"{layer_id}/audited.png",
                        "sha256": _sha256(final_class),
                    },
                    "audited_preview": {
                        "path": f"{layer_id}/audited-preview.png",
                        "sha256": _sha256(final_preview),
                    },
                },
            }
        ],
    }
    final_report_path = case_dir / "source-diff-audit.json"
    final_report_path.parent.mkdir(parents=True, exist_ok=True)
    final_report_path.write_text(json.dumps(final_report))
    stable_report_path = stable_dir / "source-diff-audit.json"
    stable_report_path.write_text(json.dumps(final_report))
    batch = {
        "status": "pass",
        "cases": [
            {
                "id": "case",
                "status": "pass",
                "fixed_point_reached": True,
                "report": "case/source-diff-audit.json",
                "comparison_iterations": [
                    {"iteration": 1, "report": "case/iteration-01/source-diff-audit.json"},
                    {"iteration": 2, "report": "case/iteration-02/source-diff-audit.json"},
                ],
            }
        ],
    }
    batch_dir.mkdir(exist_ok=True)
    (batch_dir / "source-diff-batch.json").write_text(json.dumps(batch))

    result = promote_source_diff_materialization(materialized, batch_dir, "case", output)

    assert result["layers"][0]["final_classified_pixel_count"] == 4
    assert _sha256(output / layer_id / "class.png") == _sha256(final_class)
    assert _sha256(output / layer_id / "preview.png") == _sha256(final_preview)
