import numpy as np

from mapscan.river_semantics import (
    detect_rotation_aware_text_like_regions,
    expand_confirmed_text_regions,
    extract_confirmed_glyph_components,
    infer_reconnections,
    select_text_detections,
)


def _detection(confidence, angle, x, orientation=0.0):
    return {
        "text": "River",
        "confidence": confidence,
        "ocr_angle_degrees": angle,
        "source_orientation_degrees": orientation,
        "source_center": [x, 20.0],
        "source_word_length_px": 40.0,
        "source_word_height_px": 12.0,
        "source_polygon": [[x - 20, 14], [x + 20, 14], [x + 20, 26], [x - 20, 26]],
        "observed_ink_pixel_count": 30,
        "region_pixel_count": 480,
        "observed_ink_density": 0.0625,
    }


def test_text_selection_accepts_high_confidence_or_multi_rotation_consensus():
    candidates = [
        _detection(88, 0, 20),
        _detection(48, 10, 80, 170),
        _detection(47, 20, 82, 160),
        _detection(38, 60, 140, 120),
    ]

    accepted, ambiguous = select_text_detections(candidates)

    assert len(accepted) == 3
    assert accepted[0]["acceptance_reason"] == "high_confidence_rotation_aware_ocr"
    assert {item["ocr_angle_degrees"] for item in accepted[1:]} == {10, 20}
    assert ambiguous[0]["ocr_angle_degrees"] == 60


def test_reconnection_is_separate_from_observed_evidence():
    observed = np.zeros((31, 61), dtype=bool)
    observed[15, 5:24] = True
    observed[15, 37:56] = True
    text_regions = np.zeros(observed.shape, dtype=bool)
    text_regions[10:21, 20:41] = True
    geometry = observed.copy()

    inferred, records, rejected = infer_reconnections(
        geometry,
        observed,
        text_regions,
        maximum_gap_px=16,
        maximum_component_area_px=100,
        maximum_component_dimension_px=40,
    )

    assert np.count_nonzero(inferred) > 0
    assert not np.any(inferred & observed)
    assert np.all(inferred <= text_regions)
    assert records
    assert sum(rejected.values()) >= 0


def test_text_expansion_adds_small_neighboring_glyphs_but_not_long_linework():
    observed = np.zeros((50, 110), dtype=bool)
    observed[18:24, 40:44] = True
    observed[18:24, 49:53] = True
    observed[18:24, 56:60] = True
    observed[27, 5:105] = True
    core = np.zeros(observed.shape, dtype=bool)
    core[15:27, 37:47] = True
    detection = _detection(90, 0, 42)
    detection["source_center"] = [42.0, 21.0]
    detection["source_word_length_px"] = 8.0
    detection["source_word_height_px"] = 8.0

    expanded, report = expand_confirmed_text_regions(observed, core, [detection])

    text = observed & expanded
    assert np.any(text[18:24, 49:53])
    assert np.any(text[18:24, 56:60])
    assert not np.any(text[27, 5:105])
    assert report["rejected_long_or_large_component_count"] >= 1


def test_confirmed_glyph_components_exclude_linework_inside_ocr_box():
    observed = np.zeros((50, 110), dtype=bool)
    observed[18:24, 40:44] = True
    observed[18:24, 49:53] = True
    observed[27, 5:105] = True
    detection = _detection(90, 0, 50)
    detection["source_center"] = [50.0, 22.0]
    detection["source_word_length_px"] = 70.0
    detection["source_word_height_px"] = 14.0
    detection["source_polygon"] = [[10, 14], [90, 14], [90, 31], [10, 31]]

    glyphs, report = extract_confirmed_glyph_components(observed, [detection])

    assert np.any(glyphs[18:24, 40:44])
    assert np.any(glyphs[18:24, 49:53])
    assert not np.any(glyphs[27, 5:105])
    assert report["accepted_glyph_component_count"] == 2
    assert report["rejected_line_or_large_component_count"] == 1


def test_rotation_aware_text_like_regions_are_review_only():
    observed = np.zeros((80, 180), dtype=bool)
    for x in (50, 64, 78, 92):
        observed[33:41, x : x + 5] = True
    observed[60, 5:175] = True

    text_like, review_regions, report = detect_rotation_aware_text_like_regions(
        observed,
        angles=(0,),
        thick_core_distance_px=1.8,
        horizontal_closing_width_px=17,
        longitudinal_padding_px=8,
    )

    assert np.any(text_like[33:41, 50:97])
    assert np.any(review_regions[30:45, 45:105])
    assert not np.any(text_like[60, 5:175])
    assert report["accepted_rotated_region_count"] >= 1
