import json
from pathlib import Path

import numpy as np
import pytest

from mapscan.continuous_tile_export import (
    WEB_MERCATOR_HALF_WORLD,
    _continuous_band_definitions,
    _continuous_band_masks,
    _sample_rgba_tile,
    export_continuous_tiles,
)


def test_native_rgba_sampling_preserves_color_and_nodata() -> None:
    rgb = np.array(
        [
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [255, 255, 255]],
        ],
        dtype=np.uint8,
    )
    valid = np.array([[True, False], [True, True]])
    world = WEB_MERCATOR_HALF_WORLD
    tile = _sample_rgba_tile(rgb, valid, (-world, -world, world, world), 0, 0, 0)

    assert tile[32, 32].tolist() == [255, 0, 0, 255]
    assert tile[32, 224].tolist() == [0, 255, 0, 0]
    assert tile[224, 32].tolist() == [0, 0, 255, 255]
    assert tile[224, 224].tolist() == [255, 255, 255, 255]


def test_continuous_selection_bands_are_exhaustive_and_mutually_exclusive() -> None:
    ramp_stops = [
        {"value": 0, "display_rgb": [10, 20, 30]},
        {"value": 100, "display_rgb": [40, 50, 60]},
        {"value": 250, "display_rgb": [70, 80, 90]},
    ]
    special_values = [
        {"id": "depression", "value": -500, "display_rgb": [0, 70, 60]}
    ]
    definitions = _continuous_band_definitions(ramp_stops, special_values)
    assert [item["label"] for item in definitions] == [
        "Below 0 m / depression",
        "0–<100 m",
        "100–<250 m",
        "250 m+",
    ]
    # offset=-500, scale=1: code 1=-500, 500=-1, 501=0, 600=99,
    # 601=100, 750=249, and 751=250. Zero remains NoData.
    encoded = np.asarray([[1, 500, 501, 600, 601, 750, 751, 0]], dtype=np.uint16)
    interior = encoded > 0
    masks = _continuous_band_masks(
        encoded,
        interior,
        {"offset": -500.0, "scale": 1.0},
        definitions,
    )
    assert [int(np.count_nonzero(mask)) for mask in masks] == [2, 2, 2, 1]
    assert np.all(np.sum(np.stack(masks), axis=0)[interior] == 1)
    assert not np.any(np.sum(np.stack(masks), axis=0)[~interior])


def test_continuous_export_rejects_a_stale_approval(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    manifest = {
        "extraction_kind": "continuous_color_ramp",
        "dataset_id": "test-continuous",
    }
    (run / "continuous-extraction.json").write_text(json.dumps(manifest) + "\n")
    (run / "review-decision.json").write_text(
        json.dumps(
            {
                "status": "approved",
                "review_manifest_sha256": "stale",
            }
        )
        + "\n"
    )
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"status": "pass"}) + "\n")

    with pytest.raises(ValueError, match="approval is stale"):
        export_continuous_tiles(run, audit, tmp_path / "output")


def test_continuous_review_preview_does_not_require_approval(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "continuous-extraction.json").write_text(
        json.dumps(
            {
                "extraction_kind": "continuous_color_ramp",
                "dataset_id": "test-continuous",
            }
        )
        + "\n"
    )
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps(
            {
                "status": "pass",
                "run": str(run.resolve()),
                "source_different_pixel_count": 0,
                "web_different_pixel_count": 0,
                "evidence_different_pixel_counts": {},
            }
        )
        + "\n"
    )

    with pytest.raises(ValueError, match="no artifacts"):
        export_continuous_tiles(
            run,
            audit,
            tmp_path / "output",
            review_preview=True,
        )
