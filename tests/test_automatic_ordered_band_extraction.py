from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from mapscan.automatic_alignment_loop import load_pinned_mapbox_reference
from mapscan.automatic_categorical_extraction import (
    _load_accepted_alignment,
    _source_data_mask,
)
from mapscan.automatic_ordered_band_extraction import (
    ORDERED_SEMANTICS,
    OrderedBandConfig,
    OrderedBandEntry,
    _candidate_row_segments,
    _classify_ordered_hue,
    _select_regular_six,
    _warm_support,
    detect_ordered_band_legend,
)


ROOT = Path(__file__).resolve().parents[1]


def test_regular_six_row_detector_ignores_unrelated_warm_layout() -> None:
    rgb = np.full((700, 900, 3), 244, dtype=np.uint8)
    colors = (
        (150, 25, 42),
        (198, 42, 39),
        (234, 82, 47),
        (247, 132, 63),
        (251, 187, 100),
        (254, 231, 163),
    )
    for index, color in enumerate(colors):
        top = 100 + index * 36
        rgb[top : top + 30, 420:700] = color
    # A thematic-looking decoration is not part of the six-row sequence.
    rgb[620:645, 40:200] = (230, 70, 40)
    warm = _warm_support(rgb, 40, 40)
    selected = _select_regular_six(_candidate_row_segments(warm))
    assert len(selected) == 6
    assert [row[0] for row in selected] == [99, 135, 171, 207, 243, 279]
    assert all(row[2] == (420, 700) for row in selected)


def _synthetic_entries() -> tuple[OrderedBandEntry, ...]:
    entries = []
    ranges = ((21, 27), (15, 20), (9, 14), (3, 8), (-1, 2), (-6, -2))
    colors = (
        (254, 231, 163),
        (251, 187, 100),
        (247, 132, 63),
        (234, 82, 47),
        (198, 42, 39),
        (150, 25, 42),
    )
    for class_id, ((roman, descriptor), interval, color) in enumerate(
        zip(ORDERED_SEMANTICS, ranges, colors), 1
    ):
        entries.append(
            OrderedBandEntry(
                class_id,
                roman,
                f"{roman} {descriptor.title()}",
                (0, 0, 10, 10),
                color,
                float(interval[0]),
                float(interval[1]),
                float(sum(interval) / 2),
                99.0,
                descriptor,
            )
        )
    return tuple(entries)


def test_ordered_hue_classifier_preserves_bands_and_infers_dark_ink() -> None:
    entries = _synthetic_entries()
    rgb = np.zeros((60, 120, 3), dtype=np.uint8)
    for class_id, entry in enumerate(entries, 1):
        rgb[:, (class_id - 1) * 20 : class_id * 20] = entry.representative_rgb
    # City-label ink is deliberately not accepted as observed thematic color.
    rgb[20:28, 5:115] = (8, 8, 8)
    domain = np.ones(rgb.shape[:2], dtype=bool)
    observed, complete, inferred, meaningful = _classify_ordered_hue(
        rgb, domain, entries, OrderedBandConfig()
    )
    assert set(np.unique(complete)) == set(range(1, 7))
    assert np.all(observed[20:28, 5:115] == 0)
    assert np.all(inferred[20:28, 5:115])
    assert np.all(complete[:, 0:20][~inferred[:, 0:20]] == 1)
    assert not np.any(meaningful[20:28, 5:115])


def test_real_quake_legend_recovers_six_semantics_and_manifolds(tmp_path: Path) -> None:
    run_root = ROOT / "runs" / "mapbox-autonomous-restart-v1"
    source = (run_root / "quake/source-clean/working-raster.png").resolve()
    alignment_path = run_root / "quake/automatic-alignment/accepted-alignment.json"
    mapbox_manifest = (
        ROOT / "reference/mapbox-light-v11-california-z9-v1/manifest.json"
    ).resolve()
    reference = load_pinned_mapbox_reference(mapbox_manifest)
    alignment = _load_accepted_alignment(
        alignment_path,
        source,
        reference.grid,
        reference.pin,
        accepted_iteration_count=12,
    )
    rgb = np.asarray(Image.open(source).convert("RGB"))
    domain = _source_data_mask(
        reference.state_land, reference.water, alignment["transform"], rgb.shape[:2]
    )
    legend = detect_ordered_band_legend(source, rgb, domain, tmp_path)
    assert [(entry.roman, entry.label.split(" ", 1)[1].upper()) for entry in legend.entries] == list(
        ORDERED_SEMANTICS
    )
    assert len(legend.entries) == 6
    assert all(entry.descriptor_ocr_confidence >= 90.0 for entry in legend.entries)
    assert legend.common_swatch_x_range == (1910, 2423)
    assert [entry.hue_center for entry in legend.entries] == sorted(
        (entry.hue_center for entry in legend.entries), reverse=True
    )
