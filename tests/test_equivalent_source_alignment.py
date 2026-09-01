import hashlib
import json

import cv2
import numpy as np
from PIL import Image
import pytest

from mapscan.equivalent_source_alignment import lift_alignment_to_equivalent_source


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parent(tmp_path, source):
    path = tmp_path / "alignment.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "diagnostic_only",
                "alignment_mode": "automatic",
                "source": {
                    "path": str(source),
                    "sha256": _sha256(source),
                    "width": 80,
                    "height": 100,
                },
                "best": {
                    "coverage_model": "partial_state",
                    "parameters": {"center_x_fraction": 0.5},
                },
            }
        )
    )
    return path


def test_lifts_normalized_alignment_for_verified_same_crop(tmp_path):
    yy, xx = np.mgrid[:100, :80]
    old = np.stack(((xx * 3) % 255, (yy * 2) % 255, (xx + yy) % 255), axis=2).astype(
        np.uint8
    )
    old_path = tmp_path / "old.png"
    new_path = tmp_path / "new.png"
    Image.fromarray(old).save(old_path)
    high = cv2.resize(old, (240, 300), interpolation=cv2.INTER_NEAREST)
    Image.fromarray(high).save(new_path)

    lifted = lift_alignment_to_equivalent_source(
        _parent(tmp_path, old_path), new_path, tmp_path / "lifted"
    )

    assert lifted["alignment_mode"] == "equivalent_source_lift"
    assert lifted["source"]["width"] == 240
    assert lifted["best"]["parameters"]["center_x_fraction"] == 0.5
    report = json.loads((tmp_path / "lifted/equivalent-source-lift.json").read_text())
    assert report["status"] == "pass"
    assert report["metrics"]["luma_correlation_after_area_downsample"] > 0.999


def test_rejects_different_content(tmp_path):
    old = np.zeros((100, 80, 3), dtype=np.uint8)
    old[20:80, 20:60] = 255
    new = np.zeros((300, 240, 3), dtype=np.uint8)
    new[:, :120] = 255
    old_path = tmp_path / "old.png"
    new_path = tmp_path / "new.png"
    Image.fromarray(old).save(old_path)
    Image.fromarray(new).save(new_path)

    with pytest.raises(ValueError, match="not equivalent"):
        lift_alignment_to_equivalent_source(
            _parent(tmp_path, old_path), new_path, tmp_path / "rejected"
        )

    report = json.loads((tmp_path / "rejected/equivalent-source-lift.json").read_text())
    assert report["status"] == "reject"
