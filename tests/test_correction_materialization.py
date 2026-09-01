import hashlib
import json

import numpy as np
import pytest
from PIL import Image

from mapscan.correction_materialization import materialize_review_corrections


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _materialization_run(tmp_path):
    layer_id = "hazard"
    layer_dir = tmp_path / layer_id
    inference_dir = tmp_path / "inference-v2" / layer_id
    layer_dir.mkdir()
    inference_dir.mkdir(parents=True)
    observed = np.zeros((6, 6), dtype=np.uint8)
    observed[1, 1] = 2
    observed[4, 4] = 1
    observed[4, 3] = 3
    inferred = observed.copy()
    inferred[0, 5] = 1
    inferred[5, 0] = 3
    inference_mask = np.zeros_like(observed)
    inference_mask[0, 5] = 255
    inference_mask[5, 0] = 255
    Image.fromarray(observed).save(
        layer_dir / "web-mercator-class-id.png"
    )
    Image.fromarray(inferred).save(
        inference_dir / "web-mercator-class-id-inferred.png"
    )
    Image.fromarray(inference_mask).save(
        inference_dir / "web-mercator-inference-mask.png"
    )
    manifest = {
        "dataset_id": "sample",
        "layers": [{"id": layer_id}],
    }
    plan = {
        "layers": [
            {
                "id": layer_id,
                "categories": [
                    {"id": "one", "display_rgb": [10, 20, 30]},
                    {"id": "two", "display_rgb": [40, 50, 60]},
                    {"id": "three", "display_rgb": [70, 80, 90]},
                ],
            }
        ]
    }
    (tmp_path / "extraction.json").write_text(json.dumps(manifest))
    (tmp_path / "plan.snapshot.json").write_text(json.dumps(plan))
    inference_manifest_path = tmp_path / "inference-v2" / "inference.json"
    inference_manifest_path.write_text(
        json.dumps({"dataset_id": "sample", "layers": [{"layer_id": layer_id}]})
    )
    stamps = {
        "schema_version": 2,
        "extraction_manifest_sha256": _sha256(tmp_path / "extraction.json"),
        "inference_manifest_sha256": _sha256(inference_manifest_path),
        "operations": [
            {
                "layer_id": layer_id,
                "source": [1, 1],
                "target": [4, 4],
                "radius_px": 1,
            }
        ],
    }
    (tmp_path / "stamp-corrections.json").write_text(json.dumps(stamps))
    exclusions = {
        "schema_version": 1,
        "extraction_manifest_sha256": stamps["extraction_manifest_sha256"],
        "inference_manifest_sha256": stamps["inference_manifest_sha256"],
        "operations": [
            {"layer_id": layer_id, "center": [0, 5], "radius_px": 1}
        ],
    }
    (tmp_path / "inference-exclusions.json").write_text(json.dumps(exclusions))
    return observed


def test_materialization_applies_exclusions_then_solid_manual_override(tmp_path):
    observed = _materialization_run(tmp_path)
    output = tmp_path / "candidate"
    result = materialize_review_corrections(tmp_path, output)
    final = np.asarray(
        Image.open(output / "hazard" / "web-mercator-class-id-final.png")
    )
    manual_mask = np.asarray(
        Image.open(output / "hazard" / "web-mercator-manual-override-mask.png")
    )
    assert final[0, 5] == 1
    assert final[5, 0] == 0
    assert final[4, 4] == 2
    assert final[4, 3] == 0
    assert manual_mask[4, 4] == 255
    assert observed[4, 4] == 1
    assert result["status"] == "needs_visual_review"
    assert result["precedence"][-1] == "manual_override_patch"
    assert result["layers"][0]["manual_override_pixel_count"] == 5


def test_materialization_rejects_stale_stamp_hash(tmp_path):
    _materialization_run(tmp_path)
    stamps_path = tmp_path / "stamp-corrections.json"
    stamps = json.loads(stamps_path.read_text())
    stamps["extraction_manifest_sha256"] = "stale"
    stamps_path.write_text(json.dumps(stamps))
    with pytest.raises(ValueError, match="extraction manifest"):
        materialize_review_corrections(tmp_path, tmp_path / "candidate")


def test_materialization_can_exclude_all_automatic_inference(tmp_path):
    observed = _materialization_run(tmp_path)
    output = tmp_path / "candidate"
    result = materialize_review_corrections(
        tmp_path, output, include_inference=False
    )
    final = np.asarray(
        Image.open(output / "hazard" / "web-mercator-class-id-final.png")
    )
    inference_mask = np.asarray(
        Image.open(
            output
            / "hazard"
            / "web-mercator-inference-retained-mask.png"
        )
    )
    assert final[0, 5] == observed[0, 5] == 0
    assert final[5, 0] == observed[5, 0] == 0
    assert not np.any(inference_mask)
    assert result["inference"] is None
    assert result["precedence"] == [
        "observed_classification",
        "manual_override_patch",
    ]


def test_materialization_applies_configured_small_enclosed_fill(tmp_path):
    _materialization_run(tmp_path)
    observed = np.ones((6, 6), dtype=np.uint8)
    observed[3, 3] = 0
    Image.fromarray(observed).save(
        tmp_path / "hazard" / "web-mercator-class-id.png"
    )
    stamps_path = tmp_path / "stamp-corrections.json"
    stamps = json.loads(stamps_path.read_text())
    stamps["operations"] = []
    stamps_path.write_text(json.dumps(stamps))
    (tmp_path / "enclosed-hole-fill-selection.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "enabled": True,
                "maximum_area_exclusive": 50,
            }
        )
    )
    output = tmp_path / "candidate"
    result = materialize_review_corrections(
        tmp_path, output, include_inference=False
    )
    final = np.asarray(
        Image.open(output / "hazard" / "web-mercator-class-id-final.png")
    )
    fill_mask = np.asarray(
        Image.open(
            output / "hazard" / "web-mercator-enclosed-fill-mask.png"
        )
    )
    assert final[3, 3] == 1
    assert fill_mask[3, 3] == 255
    assert result["layers"][0]["enclosed_fill_pixel_count"] == 1
    assert result["precedence"][-2:] == [
        "small_enclosed_zero_fill",
        "manual_override_patch",
    ]
