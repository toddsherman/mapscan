from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import numpy as np
import pytest

from mapscan.historical_rainfall_source_audit import (
    _patch_support,
    inspect_gif_container,
    run_historical_rainfall_source_ambiguity_audit,
)


ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_RAINFALL_SOURCE = ROOT / "examples/rainfall.gif"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_patch_support_is_translation_invariant() -> None:
    tile = np.asarray(
        [
            [1, 2, 1, 2],
            [2, 1, 2, 1],
            [1, 2, 1, 2],
            [2, 1, 2, 1],
        ],
        dtype=np.uint8,
    )
    shifted = np.roll(tile, 2, axis=1)
    assert _patch_support(tile, 2) == _patch_support(shifted, 2)


@pytest.mark.skipif(
    not HISTORICAL_RAINFALL_SOURCE.is_file(),
    reason="historical rainfall source was retired from the example corpus",
)
def test_real_historical_rainfall_gif_has_no_hidden_metadata_channel() -> None:
    source = HISTORICAL_RAINFALL_SOURCE
    report = inspect_gif_container(source.read_bytes())
    assert report["version"] == "GIF89a"
    assert report["global_palette_size"] == 256
    assert report["image_descriptor_count"] == 1
    assert report["extension_blocks"] == []
    assert report["trailer_offset"] == source.stat().st_size - 1


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract is required")
@pytest.mark.skipif(
    not HISTORICAL_RAINFALL_SOURCE.is_file(),
    reason="historical rainfall source was retired from the example corpus",
)
def test_real_historical_rainfall_source_only_audit_proves_semantic_ambiguity(
    tmp_path: Path,
) -> None:
    source = HISTORICAL_RAINFALL_SOURCE
    official_log = ROOT / "runs/mapbox-autonomous-restart-v1/rainfall-historical/EXPERIMENT.json"
    official_before = _sha256(official_log)

    report = run_historical_rainfall_source_ambiguity_audit(
        source, tmp_path / "source-audit"
    )

    assert report["status"] == "blocked_source_indistinguishable"
    assert report["legend"]["class_count"] == 35
    assert report["official_extraction_attempt_allowed"] is False
    assert report["gates"]["every_semantic_class_source_distinguishable"] is False
    groups = {
        tuple(item["class_ids"]): item
        for item in report["source_indistinguishable_groups"]
    }
    assert set(groups) == {(1, 2), (4, 5, 6)}
    assert groups[(1, 2)]["multinomial_equal_distribution_p_value"] > 0.95
    assert groups[(1, 2)]["maximum_lab_texture_center_distance"] < 0.13
    triple = groups[(4, 5, 6)]
    assert triple["shared_raw_palette_alphabet"] == [219, 220]
    assert triple["multinomial_equal_distribution_p_value"] > 0.99
    assert triple["maximum_lab_texture_center_distance"] == 0.0
    assert all(
        item["jaccard"] == 1.0
        for item in triple["two_by_two_patch_support_overlap"]
    )
    assert report["source_only_order_topology"]["status"] == "nonunique"
    assert report["source_only_order_topology"]["bridge_component_count"] >= 1
    assert (
        report["numeric_contour_ocr"][
            "high_confidence_nonlegend_contour_value_count"
        ]
        == 0
    )
    assert all(Path(item["path"]).is_file() for item in report["artifacts"])
    assert _sha256(official_log) == official_before
