from __future__ import annotations

import numpy as np
import pytest

from mapscan.geologic_pdf_extraction import (
    GeologicLegendClass,
    _assert_clean_path,
    _classify_visible_palette,
    _establish_legend_classes,
    _nearest_completion,
    _rasterize_native_fills,
    _rendered_swatch_rgb,
    _source_geographic_roundtrip_metrics,
    _warp_target_to_source_chunked,
)


def _swatch(label, color, rgb, left):
    return {
        "raw_pdf_text": label,
        "fill_cmyk": color,
        "fill_rgb": rgb,
        "bbox_page_points": [left, 10.0, left + 10.0, 20.0],
    }


def test_native_legend_semantics_require_unique_text_and_cmyk() -> None:
    classes = _establish_legend_classes(
        [
            _swatch("Qa", [0.0, 0.0, 0.3, 0.0], [255, 255, 178], 10.0),
            _swatch("gr-m", [0.08, 0.5, 0.2, 0.0], [235, 128, 204], 30.0),
        ],
        minimum_classes=2,
    )
    assert [item.native_text_code for item in classes] == ["Qa", "gr-m"]
    assert [item.id for item in classes] == ["qa", "gr-m"]
    assert classes[0].fill_cmyk == (0.0, 0.0, 0.3, 0.0)

    duplicate_text = [
        _swatch("Qa", [0.0, 0.0, 0.3, 0.0], [255, 255, 178], 10.0),
        _swatch("Qa", [0.1, 0.0, 0.3, 0.0], [230, 255, 178], 30.0),
    ]
    with pytest.raises(ValueError, match="codes are not unique"):
        _establish_legend_classes(duplicate_text, minimum_classes=2)

    duplicate_color = [
        _swatch("Qa", [0.0, 0.0, 0.3, 0.0], [255, 255, 178], 10.0),
        _swatch("Qb", [0.0, 0.0, 0.3, 0.0], [255, 255, 178], 30.0),
    ]
    with pytest.raises(ValueError, match="colors are not unique"):
        _establish_legend_classes(duplicate_color, minimum_classes=2)


def test_unreadable_native_legend_code_rejects_instead_of_guessing() -> None:
    with pytest.raises(ValueError, match="unreadable class code"):
        _establish_legend_classes(
            [
                _swatch("", [0.0, 0.0, 0.3, 0.0], [255, 255, 178], 10.0),
                _swatch("Qb", [0.0, 0.1, 0.3, 0.0], [255, 230, 178], 30.0),
            ],
            minimum_classes=2,
        )


def test_embedded_geoage_font_codes_are_decoded_to_readable_unit_labels() -> None:
    raw = ["@", "^", "|", "_", "="]
    swatches = []
    for index, label in enumerate(raw):
        swatch = _swatch(
            label,
            [index / 10.0, 0.1, 0.2, 0.0],
            [220 - index, 180, 130],
            10.0 + 20.0 * index,
        )
        swatch["native_fontname"] = "ITPFOY+GeoageFullAlpha"
        swatches.append(swatch)
    classes = _establish_legend_classes(swatches, minimum_classes=5)
    assert [item.native_text_code for item in classes] == ["To", "Tr", "Pz", "Є", "pЄ"]
    assert [item.raw_pdf_text_code for item in classes] == raw
    assert all(
        item.label_decoding == "embedded_geoage_full_alpha_symbol_map_v1"
        for item in classes
    )

    swatches[0]["native_fontname"] = "UnknownFont"
    with pytest.raises(ValueError, match="without the pinned embedded font"):
        _establish_legend_classes(swatches, minimum_classes=5)


def test_native_vector_rasterization_and_completion_are_deterministic_and_separate() -> None:
    first = np.asarray([[[1, 1]], [[4, 1]], [[4, 4]], [[1, 4]]], dtype=np.int32)
    second = np.asarray([[[6, 1]], [[8, 1]], [[8, 4]], [[6, 4]]], dtype=np.int32)
    records = ((1, (first,)), (2, (second,)))
    a = _rasterize_native_fills(records, (6, 10))
    b = _rasterize_native_fills(records, (6, 10))
    assert np.array_equal(a, b)
    domain = np.zeros_like(a, dtype=bool)
    domain[1:5, 1:9] = True
    completed, inferred = _nearest_completion(a, domain)
    observed = domain & (a > 0)
    assert not np.any(observed & inferred)
    assert np.all(completed[domain] > 0)
    assert not np.any(completed[~domain])
    assert np.all(completed[observed] == a[observed])


def test_rendered_swatch_color_uses_uniform_pdf_render_not_cmyk_approximation() -> None:
    source = np.full((20, 20, 3), (174, 166, 111), dtype=np.uint8)
    source[8:12, 8:12] = 15  # simulated dark unit-code glyph
    source[[0, -1], :] = 0
    source[:, [0, -1]] = 0
    assert _rendered_swatch_rgb(source, (0, 0, 20, 20), 1.0, 1.0) == (
        174,
        166,
        111,
    )


def test_visible_source_diff_compares_to_nearest_native_legend_class() -> None:
    classes = (
        GeologicLegendClass(
            1, "a", "A", "A", "Font", "native_pdf_unicode_text",
            (0, 0, 0, 0), (100, 150, 200), (0, 0, 1, 1)
        ),
        GeologicLegendClass(
            2, "b", "B", "B", "Font", "native_pdf_unicode_text",
            (0, 0, 0, 0), (220, 80, 40), (1, 0, 2, 1)
        ),
    )
    source = np.asarray([[[102, 149, 197], [210, 85, 44], [0, 0, 0]]], dtype=np.uint8)
    domain = np.asarray([[True, True, False]])
    ids, distances = _classify_visible_palette(source, classes, domain)
    assert np.array_equal(ids, [[1, 2, 0]])
    assert distances[0, 0] < 5
    assert distances[0, 1] < 12
    assert np.isinf(distances[0, 2])


def test_chunked_source_reconstruction_uses_inverse_target_sampling() -> None:
    target = np.arange(42, dtype=np.uint8).reshape(6, 7)
    transform = {
        "kind": "regular_global_mapbox_registration",
        "source_original_to_reference_pixel_matrix": np.eye(3).tolist(),
    }
    reconstructed = _warp_target_to_source_chunked(
        target, transform, target.shape, rows_per_chunk=2
    )
    assert np.array_equal(reconstructed, target)


def test_source_roundtrip_is_scored_in_mapbox_geographic_cells() -> None:
    expected = np.ones((6, 6), dtype=bool)
    matched = expected.copy()
    matched[0, 0] = False
    transform = {
        "kind": "regular_global_mapbox_registration",
        "source_original_to_reference_pixel_matrix": np.eye(3).tolist(),
    }
    reports = _source_geographic_roundtrip_metrics(
        expected, matched, transform, (6, 6), 2, 2, rows_per_chunk=2
    )
    by_id = {report["id"]: report for report in reports}
    assert by_id["r1-c1"]["source_semantic_expected_pixel_count"] == 9
    assert by_id["r1-c1"]["source_semantic_match_fraction"] == pytest.approx(8 / 9)
    assert by_id["r2-c2"]["source_semantic_match_fraction"] == 1.0


@pytest.mark.parametrize(
    "path",
    [
        "/tmp/automatic-alignment-orphaned-race-20260830/result.json",
        "/tmp/county.png/result.json",
        "/tmp/manual/result.json",
        "/tmp/census/result.json",
        "/tmp/legacy/result.json",
    ],
)
def test_forbidden_legacy_or_human_paths_are_rejected(path) -> None:
    with pytest.raises(ValueError, match="forbidden no-human evidence"):
        _assert_clean_path(__import__("pathlib").Path(path), "test input")
