from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mapscan.automatic_categorical_extraction import OCRWord
from mapscan.fire_layer_extraction import (
    _assert_clean_path,
    _detect_lra_legend,
    _detect_lra_source_dots,
    _find_label_line,
    _infer_overprinted_hazard,
    _regional_roundtrip_metrics,
    _warp_target_to_source,
)


def _lra_words() -> list[OCRWord]:
    return [
        OCRWord("Local", 97.0, 170, 80, 35, 14, (100, 1, 1)),
        OCRWord("Responsibility", 96.0, 210, 80, 85, 14, (100, 1, 1)),
        OCRWord("Area", 98.0, 300, 80, 30, 14, (100, 1, 1)),
    ]


def _legend_image() -> np.ndarray:
    image = np.full((180, 360, 3), 255, dtype=np.uint8)
    color = (7, 40, 61)
    for y in range(55, 115, 10):
        for x in range(105, 165, 10):
            image[y : y + 6, x : x + 6] = color
    return image


def test_lra_label_requires_one_high_confidence_exact_ocr_line() -> None:
    bbox, confidence = _find_label_line(
        _lra_words(), "Local Responsibility Area", 90.0
    )
    assert bbox == (170, 80, 160, 14)
    assert confidence == 96.0

    duplicated = _lra_words() + [
        OCRWord("Local", 97.0, 170, 120, 35, 14, (101, 1, 1)),
        OCRWord("Responsibility", 97.0, 210, 120, 85, 14, (101, 1, 1)),
        OCRWord("Area", 97.0, 300, 120, 30, 14, (101, 1, 1)),
    ]
    with pytest.raises(ValueError, match="exactly one readable OCR line"):
        _find_label_line(duplicated, "Local Responsibility Area", 90.0)


def test_lra_legend_and_map_dots_are_derived_separately_from_dark_lines() -> None:
    legend_image = _legend_image()
    legend = _detect_lra_legend(
        legend_image,
        _lra_words(),
        minimum_confidence=90.0,
        minimum_dot_count=30,
    )
    assert legend.label == "Local Responsibility Area"
    assert legend.dot_component_count == 36
    assert legend.dot_rgb == (7, 40, 61)
    assert legend.median_column_spacing == 10.0
    assert legend.median_row_spacing == 10.0

    source = np.full((180, 360, 3), 255, dtype=np.uint8)
    for y in range(30, 130, 8):
        for x in range(30, 130, 8):
            source[y : y + 2, x : x + 2] = legend.dot_rgb
    # County-like line and label-like block use the same ink but are connected
    # objects, not repeated dot components.
    source[150:152, 20:330] = legend.dot_rgb
    source[40:60, 220:270] = legend.dot_rgb
    domain = np.ones(source.shape[:2], dtype=bool)
    evidence = _detect_lra_source_dots(source, domain, legend)
    assert evidence.repeated_component_count >= 140
    assert np.any(evidence.mask[30:132, 30:132])
    assert not np.any(evidence.mask[150:152, 170:210])
    assert not np.any(evidence.mask[45:55, 235:255])


def test_hazard_completion_only_fills_dark_overprint_with_local_consensus() -> None:
    observed = np.zeros((41, 61), dtype=np.uint8)
    observed[8:33, 5:28] = 1
    observed[8:33, 33:56] = 2
    rgb = np.full((41, 61, 3), 255, dtype=np.uint8)
    rgb[observed == 1] = (235, 212, 102)
    rgb[observed == 2] = (246, 163, 94)
    dark = (7, 40, 61)
    rgb[18:22, 10:23] = dark
    rgb[18:22, 38:51] = dark
    observed[18:22, 10:23] = 0
    observed[18:22, 38:51] = 0
    # White gap is true no-hazard background and must not be invented.
    domain = np.ones(observed.shape, dtype=bool)
    complete, inferred = _infer_overprinted_hazard(observed, rgb, domain, dark)
    assert np.all(complete[19:21, 13:20] == 1)
    assert np.all(complete[19:21, 41:48] == 2)
    assert np.any(inferred)
    assert not np.any(complete[:, 28:33])


def test_fire_roundtrip_and_regional_metrics_keep_layers_independent() -> None:
    target = np.arange(100, dtype=np.uint8).reshape(10, 10)
    yy, xx = np.indices(target.shape, dtype=np.float32)
    assert np.array_equal(_warp_target_to_source(target, (xx, yy)), target)

    hazard = np.zeros((10, 10), dtype=bool)
    hazard[:5] = True
    lra = np.zeros((10, 10), dtype=bool)
    lra[7:9, 7:9] = True
    reports = _regional_roundtrip_metrics(
        hazard,
        hazard,
        lra,
        lra,
        (xx, yy),
        target.shape,
        2,
        2,
    )
    assert len(reports) == 3
    assert all(report["combined_match_fraction"] == 1.0 for report in reports)


@pytest.mark.parametrize(
    "path",
    [
        "/tmp/automatic-alignment-orphaned-race/result.json",
        "/tmp/county.png/result.json",
        "/tmp/census/result.json",
        "/tmp/manual/result.json",
        "/tmp/legacy/result.json",
    ],
)
def test_fire_extractor_rejects_forbidden_evidence_paths(path: str) -> None:
    with pytest.raises(ValueError, match="forbidden no-human evidence"):
        _assert_clean_path(Path(path), "test input")
