import json
import subprocess

import numpy as np
from PIL import Image

from mapscan.continuous_extraction import (
    _blue_nearest_legend_fill_mask,
    _classify_continuous,
    _replace_small_special_components_from_nearest,
    _restrict_active_boundary_expansion_to_warp,
    _source_analysis_mask,
)
from mapscan.label_detection import detect_apple_vision_label_regions


def test_continuous_classification_reconstructs_ocr_box_as_separate_occlusion():
    rgb = np.full((5, 6, 3), [30, 135, 0], dtype=np.uint8)
    state = np.ones((5, 6), dtype=bool)
    labels = np.zeros((5, 6), dtype=bool)
    labels[2, 3] = True
    values, completed, report = _classify_continuous(
        rgb,
        state,
        [
            {"value": 0, "legend_rgb": [29, 107, 0]},
            {"value": 100, "legend_rgb": [30, 135, 0]},
        ],
        [],
        maximum_residual=30,
        luminance_weight=0.35,
        label_occlusion_mask=labels,
    )
    assert completed[2, 3]
    assert np.isfinite(values[2, 3])
    assert report["ocr_label_occlusion_pixel_count"] == 1
    assert report["directly_observed_pixel_count"] == state.sum() - 1


def test_blue_pixels_borrow_the_nearest_spatial_legend_value():
    rgb = np.array(
        [[[29, 107, 0], [36, 105, 150], [36, 105, 150], [230, 179, 0]]],
        dtype=np.uint8,
    )
    state = np.ones((1, 4), dtype=bool)
    blue, selection = _blue_nearest_legend_fill_mask(
        rgb,
        state,
        {"minimum_blue_over_red": 20, "minimum_blue_over_green": 5},
    )

    values, completed, report = _classify_continuous(
        rgb,
        state,
        [
            {"value": 0, "legend_rgb": [29, 107, 0]},
            {"value": 500, "legend_rgb": [230, 179, 0]},
        ],
        [],
        maximum_residual=3,
        luminance_weight=0.35,
        spatial_nearest_legend_fill_mask=blue,
        completion_inpaint_radius_px=5,
    )

    assert np.array_equal(blue, [[False, True, True, False]])
    assert selection["selected_source_pixel_count"] == 2
    assert np.all(completed[blue])
    assert np.all(np.isfinite(values[blue]))
    assert values[0, 1] == 0
    assert values[0, 2] == 500
    assert report["directly_observed_pixel_count"] == 2
    assert report["spatial_nearest_legend_fill_pixel_count"] == 2


def test_small_special_value_dots_borrow_nearest_ordinary_value():
    encoded = np.full((7, 9), 501, dtype=np.uint16)
    encoded[2, 2] = 1
    encoded[3:5, 5:7] = 1
    interior = np.ones(encoded.shape, dtype=bool)

    output, changed, report = _replace_small_special_components_from_nearest(
        encoded,
        interior,
        [{"id": "depression", "value": -500}],
        {"offset": -500, "scale": 1},
        {
            "maximum_component_pixel_count": 2,
            "require_surrounded_by_data": True,
        },
    )

    assert output[2, 2] == 501
    assert np.all(output[3:5, 5:7] == 1)
    assert changed[2, 2]
    assert not np.any(changed[3:5, 5:7])
    assert report["replaced_component_count"] == 1
    assert report["replaced_pixel_count"] == 1
    assert report["preservation_policy"] == (
        "larger special-value regions remain authoritative"
    )


def test_active_boundary_expansion_keeps_only_warp_supported_pixels():
    publication = np.zeros((3, 6), dtype=bool)
    publication[:, 1:] = True
    pinned = np.zeros_like(publication)
    pinned[:, 2:] = True
    values = np.zeros_like(publication, dtype=np.uint16)
    values[0, 1] = 100
    values[:, 2:] = 200

    final, report = _restrict_active_boundary_expansion_to_warp(
        publication, pinned, values
    )

    assert final[0, 1]
    assert not final[1, 1]
    assert not final[2, 1]
    assert np.array_equal(final[:, 2:], publication[:, 2:])
    assert report["warped_source_supported_pixel_count"] == 1
    assert report["unsupported_pixel_count_removed"] == 2
    assert report["target_completion_policy"] == (
        "forbidden outside the pinned polygon interior"
    )


def test_active_boundary_expansion_can_complete_unsupported_canonical_land():
    publication = np.zeros((3, 6), dtype=bool)
    publication[:, 1:] = True
    pinned = np.zeros_like(publication)
    pinned[:, 2:] = True
    values = np.zeros_like(publication, dtype=np.uint16)
    values[0, 1] = 100
    values[:, 2:] = 200

    final, report = _restrict_active_boundary_expansion_to_warp(
        publication,
        pinned,
        values,
        complete_unsupported_from_nearest=True,
    )

    assert np.array_equal(final, publication)
    assert report["unsupported_pixel_count"] == 2
    assert report["unsupported_pixel_count_removed"] == 0
    assert report["unsupported_pixel_count_retained_for_completion"] == 2
    assert report["target_completion_policy"] == (
        "unsupported canonical land inherits the nearest encoded value; "
        "water remains excluded"
    )


def test_active_boundary_source_analysis_uses_a_bounded_margin():
    authoritative = np.zeros((7, 8), dtype=bool)
    authoritative[2:5, 3:6] = True

    analysis, report = _source_analysis_mask(
        authoritative, "active_boundary_ring", 1
    )

    expected = np.zeros_like(authoritative)
    expected[1:6, 2:7] = True
    assert np.array_equal(analysis, expected)
    assert report["margin_px"] == 1
    assert report["authoritative_state_pixel_count"] == 9
    assert report["analysis_pixel_count"] == 25
    assert report["margin_pixel_count"] == 16
    assert report["publication_authority"] == (
        "active_boundary_ring_after_web_mercator_warp"
    )


def test_pinned_polygon_source_analysis_names_the_actual_publication_authority():
    authoritative = np.ones((3, 4), dtype=bool)

    analysis, report = _source_analysis_mask(authoritative, "pinned_polygon", 0)

    assert np.array_equal(analysis, authoritative)
    assert report["publication_authority"] == (
        "pinned_polygon_plus_declared_coastal_seam_after_web_mercator_warp"
    )


def test_apple_vision_boxes_are_clipped_to_the_valid_source_map(
    tmp_path, monkeypatch
):
    source = tmp_path / "source.png"
    source_rgb = np.full((10, 20, 3), 220, dtype=np.uint8)
    source_rgb[6:8, 6:10] = 10
    Image.fromarray(source_rgb).save(source)
    valid = np.zeros((10, 20), dtype=bool)
    valid[3:9, 2:18] = True
    detections = [
        {
            "text": "MOJAVE",
            "confidence": 0.9,
            "normalized_bbox_bottom_left": [0.25, 0.2, 0.3, 0.2],
        },
        {
            "text": "Outside",
            "confidence": 1.0,
            "normalized_bbox_bottom_left": [0.8, 0.9, 0.15, 0.08],
        },
    ]

    monkeypatch.setattr("mapscan.label_detection.shutil.which", lambda _: "/usr/bin/swiftc")

    def fake_run(args, **kwargs):
        stdout = "" if args[0] == "/usr/bin/swiftc" else json.dumps(detections)
        return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")

    monkeypatch.setattr("mapscan.label_detection.subprocess.run", fake_run)
    mask, report = detect_apple_vision_label_regions(
        source,
        valid,
        padding_px=1,
        minimum_valid_overlap_pixels=2,
        blackhat_threshold=5,
        glyph_dilation_px=0,
    )
    assert np.all(mask <= valid)
    assert report["accepted_detection_count"] == 1
    assert report["rejected_detection_counts"]["outside_valid_map"] == 1
    assert report["accepted_detections"][0]["text"] == "MOJAVE"
    assert report["label_occlusion_pixel_count"] == int(mask.sum())
