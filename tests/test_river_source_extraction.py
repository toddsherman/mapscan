from __future__ import annotations

from pathlib import Path
import hashlib
import json

import cv2
import numpy as np
import pytest
from PIL import Image

from mapscan.experiment_log import NoHumanExperimentLog
from mapscan.river_source_extraction import (
    RiverSourceConfig,
    _assert_clean_path,
    _assign_label_semantics,
    _blue_ink,
    _closed_water_polygons,
    _compact_ink_in_review_regions,
    _deduplicate_labels,
    _detect_dotted_chains,
    _group_named_features,
    _iou,
    _load_state_interior,
    _regional_metrics,
    _resumable_prior_extraction_count,
    _roundtrip_channel_gates,
    _signature,
    _warp_target_to_source,
    run_named_hydrography_extraction,
)


REFERENCE = {
    "id": "mapbox-california-reference-v1",
    "state_sha256": "a" * 64,
    "county_sha256": "b" * 64,
    "water_sha256": "c" * 64,
}


def _detection(text: str, confidence: float, x: float, angle: int = 0) -> dict:
    return {
        "text": text,
        "confidence": confidence,
        "ocr_angle_degrees": angle,
        "source_orientation_degrees": float(-angle),
        "source_center": [x, 20.0],
        "source_word_length_px": 30.0,
        "source_word_height_px": 10.0,
        "source_polygon": [
            [x - 15, 15],
            [x + 15, 15],
            [x + 15, 25],
            [x - 15, 25],
        ],
    }


def test_blue_ink_keeps_saturated_line_but_excludes_pale_ocean() -> None:
    rgb = np.full((20, 30, 3), 255, dtype=np.uint8)
    rgb[:, :10] = (180, 205, 220)
    rgb[8:12, 15:25] = (20, 80, 180)
    detected = _blue_ink(rgb, np.ones(rgb.shape[:2], bool), RiverSourceConfig())
    assert not np.any(detected[:, :10])
    assert np.all(detected[8:12, 15:25])


def test_label_deduplication_preserves_verbatim_ocr_alternatives() -> None:
    labels = _deduplicate_labels(
        [
            _detection("Sacramento", 91.0, 40.0, 0),
            _detection("Sacrarnento", 80.0, 42.0, 15),
            _detection("Kern", 85.0, 100.0, -15),
        ]
    )
    assert len(labels) == 2
    first = labels[0]
    assert first["text_verbatim_ocr"] == "Sacramento"
    assert first["rotation_support_count"] == 2
    assert first["spelling_ambiguous"] is True
    assert set(first["ocr_spelling_alternatives"]) == {"sacramento", "sacrarnento"}


def test_neighbouring_tokens_form_verbatim_named_feature_and_share_semantics() -> None:
    tokens = list(
        _deduplicate_labels(
            [
                _detection("Owens", 95.0, 30.0),
                _detection("Lake", 94.0, 68.0),
                _detection("(dry)", 93.0, 100.0),
            ]
        )
    )
    for token in tokens:
        token["associated_feature_type"] = "river-or-stream"
    features = _group_named_features(tokens)
    assert len(features) == 1
    assert features[0]["name_verbatim_ocr"] == "Owens Lake (dry)"
    assert features[0]["associated_feature_type"] == "dry-lake"
    assert len(features[0]["source_token_ids"]) == 3


def test_stacked_lake_name_tokens_group_despite_ocr_orientation_disagreement() -> None:
    tokens = list(
        _deduplicate_labels(
            [
                _detection("Lake", 95.0, 40.0, 0),
                _detection("Tahoe", 94.0, 80.0, 75),
            ]
        )
    )
    tokens[0]["source_center"] = [40.0, 20.0]
    tokens[1]["source_center"] = [40.0, 37.0]
    for token in tokens:
        token["associated_feature_type"] = "river-or-stream"
    features = _group_named_features(tokens)
    assert len(features) == 1
    assert features[0]["name_verbatim_ocr"] == "Lake Tahoe"
    assert features[0]["associated_feature_type"] == "lake-or-reservoir"


def test_dotted_chain_detector_keeps_collinear_dashes_not_isolated_mark() -> None:
    mask = np.zeros((80, 120), dtype=bool)
    for x in range(10, 90, 12):
        mask[30:32, x : x + 4] = True
    mask[65:68, 104:107] = True
    detected, report = _detect_dotted_chains(mask)
    assert np.any(detected[30:32, 10:86])
    assert not np.any(detected[65:68, 104:107])
    assert report["selected_component_count"] >= 6


def test_text_review_corridor_does_not_consume_crossing_linework() -> None:
    observed = np.zeros((50, 100), dtype=bool)
    observed[25, 2:98] = True
    for x in (34, 43, 52, 61):
        observed[18:23, x : x + 3] = True
    review = np.zeros_like(observed)
    review[12:30, 28:70] = True
    glyphs, report = _compact_ink_in_review_regions(observed, review)
    assert np.any(glyphs[18:23, 34:64])
    assert not np.any(glyphs[25, 2:98])
    assert report["accepted_compact_component_count"] == 4
    assert report["rejected_long_or_large_component_count"] == 1


def test_closed_polygons_require_literal_dry_label_for_dry_semantics() -> None:
    lines = np.zeros((120, 180), dtype=np.uint8)
    cv2.rectangle(lines, (20, 20), (40, 34), 1, thickness=2)
    cv2.ellipse(lines, (130, 75), (12, 9), 0, 0, 360, 1, thickness=2)
    rgb = np.full((120, 180, 3), 245, dtype=np.uint8)
    wet_outline, wet_interior, dry_outline, dry_interior, records = (
        _closed_water_polygons(lines.astype(bool), rgb, [[130.0, 75.0]])
    )
    assert np.any(wet_outline[18:37, 18:43])
    assert np.any(wet_interior[23:32, 23:38])
    assert np.any(dry_outline[64:87, 116:145])
    assert np.any(dry_interior[69:82, 121:140])
    assert {record["semantic_type"] for record in records} == {
        "lake-or-reservoir",
        "dry-lake",
    }


def test_empty_feature_channel_cannot_win_label_proximity() -> None:
    river = np.zeros((30, 30), dtype=bool)
    river[15, 2:28] = True
    empty = np.zeros_like(river)
    labels = _deduplicate_labels([_detection("Mokelumne", 92.0, 15.0)])
    associated = _assign_label_semantics(labels, river, empty, empty, empty)
    assert associated[0]["associated_feature_type"] == "river-or-stream"


def test_signature_and_geographic_roundtrip_cover_independent_channels() -> None:
    channel_a = np.zeros((12, 15), dtype=bool)
    channel_b = np.zeros_like(channel_a)
    channel_a[1:5, 1:5] = True
    channel_b[7:11, 9:14] = True
    signature = _signature([channel_a, channel_b])
    yy, xx = np.indices(signature.shape, dtype=np.float32)
    assert np.array_equal(_warp_target_to_source(signature, (xx, yy)), signature)
    reports = _regional_metrics(signature, signature, (xx, yy), signature.shape, 2, 2)
    assert reports
    assert all(report["match_fraction"] == 1.0 for report in reports)


def test_empty_optional_channel_is_an_exact_roundtrip() -> None:
    empty = np.zeros((12, 15), dtype=bool)
    nonempty = empty.copy()
    nonempty[4:7, 5:9] = True
    assert _iou(empty, empty) == 1.0
    assert _iou(empty, nonempty) == 0.0
    assert _iou(nonempty, empty) == 0.0


def test_roundtrip_gate_excludes_zero_support_optional_channel_but_checks_it_stays_empty() -> None:
    supported = np.zeros((12, 15), dtype=bool)
    supported[4:7, 5:9] = True
    empty = np.zeros_like(supported)
    ious, supported_gate, optional_gate = _roundtrip_channel_gates(
        ["observed", "optional-inferred"],
        [supported, empty],
        [supported.copy(), empty.copy()],
        0.78,
    )
    assert ious == [1.0, 1.0]
    assert supported_gate["passed"] is True
    assert supported_gate["evaluated_source_supported_channels"] == ["observed"]
    assert supported_gate["excluded_zero_support_optional_channels"] == [
        "optional-inferred"
    ]
    assert optional_gate == {"passed": True, "channels": ["optional-inferred"]}

    leaked = empty.copy()
    leaked[1, 1] = True
    _, supported_gate, optional_gate = _roundtrip_channel_gates(
        ["observed", "optional-inferred"],
        [supported, empty],
        [supported.copy(), leaked],
        0.78,
    )
    assert supported_gate["passed"] is True
    assert optional_gate["passed"] is False


def test_river_extraction_resume_keeps_prior_rejected_automatic_ordinals() -> None:
    class FakeLog:
        data = {
            "extraction": {
                "accepted_automatic_iteration_count": None,
                "iterations": [
                    {
                        "automatic_iteration": ordinal,
                        "counts_toward_automatic_iteration_count": True,
                        "decision": decision,
                        "provenance": {
                            "actor_kind": "automated",
                            "manual_arrows": False,
                            "manual_stamps": False,
                            "human_approval": False,
                        },
                    }
                    for ordinal, decision in ((1, "retry"), (2, "blocked"))
                ],
            }
        }

    assert _resumable_prior_extraction_count(FakeLog()) == 2  # type: ignore[arg-type]
    FakeLog.data["extraction"]["iterations"][1]["provenance"]["human_approval"] = True
    with pytest.raises(ValueError, match="prior rejected automatic attempts"):
        _resumable_prior_extraction_count(FakeLog())  # type: ignore[arg-type]


def test_pinned_state_interior_keeps_internal_water_in_hydrography_domain(
    tmp_path: Path,
) -> None:
    values = np.zeros((8, 10), dtype=np.uint8)
    values[1:7, 2:9] = 255
    values[3:5, 4:7] = 255  # An internal-water cell remains in the footprint.
    mask_path = tmp_path / "state-interior.png"
    Image.fromarray(values, mode="L").save(mask_path)
    digest = hashlib.sha256(mask_path.read_bytes()).hexdigest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "artifacts": {
                    "state_interior_mask": {
                        "path": mask_path.name,
                        "sha256": digest,
                    }
                }
            }
        )
    )
    loaded = _load_state_interior(manifest_path, values.shape)
    assert np.array_equal(loaded, values > 0)
    mask_path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        _load_state_interior(manifest_path, values.shape)


def test_config_requires_exactly_two_replays() -> None:
    with pytest.raises(ValueError, match="exactly two"):
        RiverSourceConfig(required_replay_count=1)


@pytest.mark.parametrize(
    "path",
    [
        "/tmp/automatic-alignment-orphaned-race/result.json",
        "/tmp/county.png/result.json",
        "/tmp/census/result.json",
        "/tmp/manual/result.json",
        "/tmp/legacy/result.json",
        "/tmp/river-semantics/result.json",
        "/tmp/rivers-extract-v4/result.json",
    ],
)
def test_extractor_rejects_forbidden_evidence_paths(path: str) -> None:
    with pytest.raises(ValueError, match="forbidden no-human evidence"):
        _assert_clean_path(Path(path), "test input")


def test_publishable_extraction_is_inert_without_accepted_alignment(tmp_path: Path) -> None:
    source = tmp_path / "rivers.jpg"
    source.write_bytes(b"pristine-rivers-source")
    log = NoHumanExperimentLog(
        "rivers",
        source,
        source_type="named_linear_and_polygon_features_without_legend",
        mapbox_reference=REFERENCE,
    )
    output = tmp_path / "official-extraction"
    with pytest.raises(ValueError, match="requires accepted automatic alignment"):
        run_named_hydrography_extraction(
            tmp_path / "not-read-source-adapter.json",
            tmp_path / "not-read-alignment.json",
            tmp_path / "not-read-mapbox.json",
            output,
            log,
            tmp_path / "EXPERIMENT.md",
            tmp_path / "EXPERIMENT.json",
        )
    assert not output.exists()
    assert log.data["extraction"]["iterations"] == []
