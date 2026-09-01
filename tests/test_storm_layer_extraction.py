from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mapscan.automatic_categorical_extraction import OCRWord
from mapscan.extraction import classify_grayscale, infer_sparse_chroma_overlays
from mapscan.storm_layer_extraction import (
    OVERLAY_IDS,
    PRECIPITATION_IDS,
    StormExtractionConfig,
    _assert_clean_path,
    _detect_semantic_legend,
    _regional_metrics,
    _semantic_signature,
    _warp_target_to_source,
)


def _word(text: str, left: int, top: int, width: int, line: int) -> OCRWord:
    return OCRWord(text, 96.0, left, top, width, 12, (1, 1, line))


def _synthetic_legend() -> tuple[np.ndarray, list[OCRWord]]:
    image = np.full((260, 430, 3), 255, dtype=np.uint8)
    specifications = [
        ((171, 171, 171), ("4-8", "in", "(102-204", "mm)")),
        ((136, 136, 136), ("8-16", "in", "(204-408", "mm)")),
        ((88, 88, 88), ("16-25.7", "in", "(408-652", "mm)")),
        ((200, 135, 136), ("Landslide", "susceptibility", ">", "9")),
        ((220, 141, 187), ("Maximum", "wind", "speed", ">", "60", "mile/h")),
        ((142, 215, 246), ("Predicted", "flooding")),
    ]
    words: list[OCRWord] = []
    for line, (color, tokens) in enumerate(specifications, 1):
        top = 20 + (line - 1) * 36
        image[top - 2 : top + 15, 88:106] = color
        left = 115
        for token in tokens:
            width = max(8, len(token) * 7)
            words.append(_word(token, left, top, width, line))
            left += width + 5
    return image, words


def test_semantic_legend_reads_six_unique_swatches_and_labels() -> None:
    image, words = _synthetic_legend()
    entries = _detect_semantic_legend(image, words, minimum_confidence=90.0)
    assert tuple(entry.layer_id for entry in entries[:3]) == PRECIPITATION_IDS
    assert tuple(entry.layer_id for entry in entries[3:]) == OVERLAY_IDS
    assert [entry.rgb for entry in entries] == [
        (171, 171, 171),
        (136, 136, 136),
        (88, 88, 88),
        (200, 135, 136),
        (220, 141, 187),
        (142, 215, 246),
    ]


def test_legend_rejects_a_second_physical_copy_of_a_semantic_line() -> None:
    image, words = _synthetic_legend()
    words.extend(
        [
            OCRWord("Predicted", 97.0, 115, 240, 60, 12, (2, 1, 1)),
            OCRWord("flooding", 97.0, 180, 240, 55, 12, (2, 1, 1)),
        ]
    )
    with pytest.raises(ValueError, match="exactly one physical OCR legend line"):
        _detect_semantic_legend(image, words, minimum_confidence=90.0)


def test_overlays_are_independent_and_precipitation_beneath_them_is_unknown() -> None:
    source = np.full((80, 100, 3), 250, dtype=np.uint8)
    source[10:70, 10:40] = (171, 171, 171)
    source[10:70, 40:70] = (136, 136, 136)
    source[10:70, 70:90] = (88, 88, 88)
    source[20:35, 20:50] = (200, 135, 136)
    source[35:50, 45:75] = (220, 141, 187)
    source[50:65, 60:85] = (142, 215, 246)
    domain = np.ones(source.shape[:2], dtype=bool)
    overlays, _ = infer_sparse_chroma_overlays(
        source,
        domain,
        [
            {"legend_rgb": [200, 135, 136]},
            {"legend_rgb": [220, 141, 187]},
            {"legend_rgb": [142, 215, 246]},
        ],
        10.0,
        0.32,
        6.0,
        1.4,
    )
    occluded = np.logical_or.reduce(overlays)
    ids, _ = classify_grayscale(
        source,
        domain,
        [{"legend_gray": 171}, {"legend_gray": 136}, {"legend_gray": 88}],
        1,
        20.0,
        13.0,
        exclusion_mask=occluded,
        adaptive_centers=False,
        background_cutoff=245,
    )
    assert all(np.any(mask) for mask in overlays)
    assert not np.any(ids[occluded])
    assert {int(value) for value in np.unique(ids) if value} == {1, 2, 3}


def test_semantic_signature_and_geographic_roundtrip_cover_all_channels() -> None:
    precipitation = np.zeros((10, 10), dtype=np.uint8)
    precipitation[:4, :4] = 2
    overlays = [
        np.zeros((10, 10), dtype=bool),
        np.zeros((10, 10), dtype=bool),
        np.zeros((10, 10), dtype=bool),
    ]
    overlays[0][5:7, 5:7] = True
    overlays[1][6:8, 6:8] = True
    overlays[2][8:10, 8:10] = True
    occluded = np.logical_or.reduce(overlays)
    signature = _semantic_signature(precipitation, occluded, overlays)
    assert signature[6, 6] != signature[5, 5]
    yy, xx = np.indices(signature.shape, dtype=np.float32)
    assert np.array_equal(_warp_target_to_source(signature, (xx, yy)), signature)
    reports = _regional_metrics(signature, signature, (xx, yy), signature.shape, 2, 2)
    assert reports
    assert all(item["match_fraction"] == 1.0 for item in reports)


def test_storm_config_requires_exact_two_pass_replay() -> None:
    with pytest.raises(ValueError, match="exactly two"):
        StormExtractionConfig(required_replay_count=1)


@pytest.mark.parametrize(
    "path",
    [
        "/tmp/automatic-alignment-orphaned-race/result.json",
        "/tmp/county.png/result.json",
        "/tmp/census/result.json",
        "/tmp/manual/result.json",
        "/tmp/legacy/result.json",
        "/tmp/runs/landslide-extract-v5/result.json",
    ],
)
def test_storm_extractor_rejects_forbidden_evidence_paths(path: str) -> None:
    with pytest.raises(ValueError, match="forbidden no-human evidence"):
        _assert_clean_path(Path(path), "test input")
