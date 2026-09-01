from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mapscan.indexed_coverage_attestation import attest_indexed_staging_coverage


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path, *, exterior: bool = False) -> tuple[Path, Path]:
    staging = tmp_path / "staging"
    staging.mkdir()
    raster = np.asarray([[0, 1], [1 if exterior else 0, 2]], dtype=np.uint8)
    mask = np.asarray([[0, 255], [0, 255]], dtype=np.uint8)
    Image.fromarray(raster).save(tmp_path / "classes.png")
    Image.fromarray(mask).save(tmp_path / "state.png")
    provenance = {
        "status": "needs_visual_review",
        "class_raster": {"sha256": "a" * 64},
    }
    provenance_path = staging / "autonomous-preview-provenance.json"
    provenance_path.write_text(json.dumps(provenance) + "\n")
    dataset = {
        "status": "needs_visual_review",
        "id": "test",
        "provenance": {"sha256": _sha256(provenance_path)},
        "layers": [{"id": "classes"}],
    }
    (staging / "dataset.json").write_text(json.dumps(dataset) + "\n")
    plan = {
        "schema_version": 1,
        "state_interior_mask": "state.png",
        "layers": [
            {
                "layer_id": "classes",
                "accepted_raster_sha256": "a" * 64,
                "publication_raster": "classes.png",
                "semantic_kind": "mutually_exclusive_categorical",
                "coverage_contract": "sparse_visible_evidence",
            }
        ],
    }
    plan_path = tmp_path / "coverage.json"
    plan_path.write_text(json.dumps(plan) + "\n")
    return staging, plan_path


def test_attests_exact_inside_and_outside_counts(tmp_path: Path):
    staging, plan = _fixture(tmp_path)

    result = attest_indexed_staging_coverage(staging, plan)

    record = result["publication_coverage"]["layers"][0]
    assert record["colored_pixel_count_outside_state"] == 0
    assert record["classified_pixel_count_inside_state"] == 2
    assert record["nodata_pixel_count_inside_state"] == 0
    dataset = json.loads((staging / "dataset.json").read_text())
    assert dataset["provenance"]["sha256"] == _sha256(
        staging / "autonomous-preview-provenance.json"
    )


def test_rejects_exterior_data_without_updating_staging(tmp_path: Path):
    staging, plan = _fixture(tmp_path, exterior=True)
    original_dataset = (staging / "dataset.json").read_bytes()
    original_provenance = (staging / "autonomous-preview-provenance.json").read_bytes()

    with pytest.raises(ValueError, match="contains 1 exterior pixels"):
        attest_indexed_staging_coverage(staging, plan)

    assert (staging / "dataset.json").read_bytes() == original_dataset
    assert (staging / "autonomous-preview-provenance.json").read_bytes() == original_provenance
