import hashlib
import json

import cv2
import numpy as np
from PIL import Image

from mapscan.extraction import (
    _fill_indexed_nodata_in_mask,
    _suppress_isolated_class_specks,
    _county_residual_diagnostic,
    _registered_county_reference_assets,
    _transform_normalized_to_source,
    classify_categorical,
    classify_grayscale,
    classify_patterned_categorical,
    infer_sparse_chroma_overlays,
)


def _file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_assisted_projective_transform_maps_normalized_reference_points():
    points = np.array([[0.0, 0.0], [1.0, 2.0]], dtype=float)
    transform = {
        "reference_to_source_matrix": [
            [2.0, 0.0, 10.0],
            [0.0, 3.0, 20.0],
            [0.0, 0.0, 1.0],
        ]
    }
    mapped = _transform_normalized_to_source(points, transform, (100, 100))
    assert np.allclose(mapped, [[10.0, 20.0], [12.0, 26.0]])


def test_county_residual_diagnostic_finds_nearby_source_line():
    rgb = np.full((40, 40, 3), 255, dtype=np.uint8)
    rgb[:, 12:14] = 0
    county = np.zeros((40, 40), dtype=bool)
    county[5:35, 13] = True
    state = np.zeros((40, 40), dtype=bool)
    overlay, report = _county_residual_diagnostic(rgb, county, state)
    assert report["median_nearest_source_edge_px"] <= 1
    assert report["within_3px_fraction"] == 1
    assert np.count_nonzero(overlay[:, :, 3]) > 0


def test_registered_county_reference_preserves_thin_raster_strokes(tmp_path):
    source_path = tmp_path / "county.png"
    Image.new("RGBA", (20, 20), "white").save(source_path)
    mask = np.zeros((120, 100), dtype=np.uint8)
    mask[:, 52] = 255
    mask_path = tmp_path / "county-mask.png"
    Image.fromarray(mask).save(mask_path)
    manifest = {
        "status": "pass",
        "reference_kind": "styled_high_resolution_county_raster",
        "source": {"path": str(source_path), "sha256": _file_sha256(source_path)},
        "web_grid": {
            "crs": "EPSG:3857",
            "bounds": [0.0, 0.0, 10.0, 10.0],
            "width": 100,
            "height": 120,
        },
        "artifacts": {
            "web_mercator_county_border": {
                "path": mask_path.name,
                "sha256": _file_sha256(mask_path),
            }
        },
    }
    manifest_path = tmp_path / "county-reference.json"
    manifest_path.write_text(json.dumps(manifest))

    overlay, rendered_mask, provenance = _registered_county_reference_assets(
        manifest_path, [0.0, 0.0, 10.0, 10.0], (30, 25)
    )

    assert overlay.shape == (30, 25, 4)
    assert np.all(np.count_nonzero(rendered_mask, axis=1) == 1)
    assert np.all(overlay[rendered_mask, 3] >= 96)
    assert provenance["visible_pixel_count"] == 30
    assert provenance["source"]["sha256"] == _file_sha256(source_path)


def test_fill_indexed_nodata_in_mask_does_not_cross_clip_boundary():
    values = np.zeros((5, 7), dtype=np.uint8)
    values[2, 1] = 1
    values[2, 5] = 2
    valid = np.zeros(values.shape, dtype=bool)
    valid[1:4, 1:6] = True

    completed, count = _fill_indexed_nodata_in_mask(values, valid)

    assert count == 13
    assert np.all(completed[valid] > 0)
    assert not np.any(completed[~valid])
    assert completed[2, 2] == 1
    assert completed[2, 4] == 2


def test_fill_indexed_nodata_in_mask_ignores_outside_class_seeds():
    values = np.zeros((5, 7), dtype=np.uint8)
    values[2, 0] = 2
    values[2, 3] = 1
    valid = np.zeros(values.shape, dtype=bool)
    valid[1:4, 2:5] = True

    completed, count = _fill_indexed_nodata_in_mask(values, valid)

    assert count == 8
    assert np.all(completed[valid] == 1)
    assert not np.any(completed[~valid])


def test_suppress_isolated_specks_reassigns_tiny_component_but_keeps_white():
    values = np.ones((9, 12), dtype=np.uint8)
    values[2:7, 7:11] = 3
    values[4, 3] = 2
    values[4, 8] = 4
    categories = [
        {"id": "red", "legend_rgb": [200, 20, 20]},
        {"id": "brown", "legend_rgb": [120, 70, 30]},
        {"id": "blue", "legend_rgb": [30, 50, 200]},
        {"id": "white", "legend_rgb": [255, 255, 255]},
    ]

    cleaned, changed, original, report = _suppress_isolated_class_specks(
        values,
        categories,
        maximum_area=4,
        minimum_surrounding_purity=0.5,
        preserve_near_white=True,
    )

    assert cleaned[4, 3] == 1
    assert cleaned[4, 8] == 4
    assert changed[4, 3]
    assert not changed[4, 8]
    assert original[4, 3] == 2
    assert report["reassigned_pixel_count"] == 1


def test_categorical_classification_preserves_ambiguity_as_nodata():
    rgb = np.array([[[220, 30, 30], [30, 50, 220], [120, 120, 120]]], dtype=np.uint8)
    categories = [
        {"id": "red", "legend_rgb": [220, 30, 30]},
        {"id": "blue", "legend_rgb": [30, 50, 220]},
    ]
    values, report = classify_categorical(
        rgb,
        np.ones((1, 3), dtype=bool),
        categories,
        max_distance=5,
        min_margin=1,
    )
    assert values.tolist() == [[1, 2, 0]]
    assert report["ambiguous_pixel_count"] == 1


def test_patterned_classification_separates_equal_mean_oriented_textures():
    rgb = np.full((48, 72, 3), 240, dtype=np.uint8)
    dark = np.array([50, 60, 70], dtype=np.uint8)
    light = np.array([230, 220, 210], dtype=np.uint8)

    vertical = np.empty((16, 24, 3), dtype=np.uint8)
    horizontal = np.empty((16, 24, 3), dtype=np.uint8)
    for y in range(16):
        for x in range(24):
            vertical[y, x] = dark if x % 2 == 0 else light
            horizontal[y, x] = dark if y % 2 == 0 else light
    rgb[0:16, 0:24] = vertical
    rgb[0:16, 32:56] = horizontal
    rgb[24:40, 0:24] = vertical
    rgb[24:40, 32:56] = horizontal

    categories = [
        {"id": "vertical", "sample_rect": [0, 0, 24, 16]},
        {"id": "horizontal", "sample_rect": [32, 0, 56, 16]},
    ]
    state_mask = np.zeros(rgb.shape[:2], dtype=bool)
    state_mask[24:40, 0:56] = True
    values, report = classify_patterned_categorical(
        rgb,
        state_mask,
        categories,
        window_size=5,
        max_distance=2.0,
        min_margin=0.25,
    )
    assert values[28:36, 4:20].tolist() == [[1] * 16] * 8
    assert values[28:36, 36:52].tolist() == [[2] * 16] * 8
    assert report["classified_pixel_count"] > 0
    assert report["confusable_legend_pairs"] == []


def test_patterned_classification_leaves_identical_swatches_ambiguous():
    rgb = np.zeros((24, 40, 3), dtype=np.uint8)
    texture = np.indices((12, 12)).sum(axis=0) % 2
    swatch = np.where(texture[:, :, None] == 0, 40, 220).astype(np.uint8)
    swatch = np.repeat(swatch, 3, axis=2)
    rgb[0:12, 0:12] = swatch
    rgb[0:12, 16:28] = swatch
    rgb[12:24, 0:12] = swatch
    categories = [
        {"id": "first", "sample_rect": [0, 0, 12, 12]},
        {"id": "second", "sample_rect": [16, 0, 28, 12]},
    ]
    state_mask = np.zeros(rgb.shape[:2], dtype=bool)
    state_mask[12:24, 0:12] = True
    values, report = classify_patterned_categorical(
        rgb,
        state_mask,
        categories,
        window_size=3,
        max_distance=2.0,
        min_margin=0.1,
    )
    assert not np.any(values)
    assert report["ambiguous_pixel_count"] == int(np.count_nonzero(state_mask))
    assert report["confusable_legend_pairs"][0]["prototype_overlap_fraction"] > 0


def test_patterned_palette_completion_fills_regions_but_preserves_dark_ink():
    rgb = np.full((40, 64, 3), 255, dtype=np.uint8)
    red = np.array([220, 40, 40], dtype=np.uint8)
    blue = np.array([40, 70, 220], dtype=np.uint8)
    shared = np.array([180, 180, 180], dtype=np.uint8)
    for y in range(8):
        for x in range(16):
            rgb[y, x] = red if (x + y) % 2 == 0 else shared
            rgb[y, 24 + x] = blue if (x + y) % 2 == 0 else shared

    rgb[16:32, 0:4] = red
    rgb[16:32, 4:28] = shared
    rgb[16:32, 28] = 0
    rgb[16:32, 29:52] = shared
    rgb[16:32, 52:56] = blue
    categories = [
        {"id": "red", "sample_rect": [0, 0, 16, 8]},
        {"id": "blue", "sample_rect": [24, 0, 40, 8]},
    ]
    state_mask = np.zeros(rgb.shape[:2], dtype=bool)
    state_mask[16:32, 0:56] = True
    values, report = classify_patterned_categorical(
        rgb,
        state_mask,
        categories,
        window_size=3,
        max_distance=1.0,
        min_margin=0.1,
        max_color_distance=1.0,
        min_color_margin=1.0,
        complete_palette=True,
        histogram_window=5,
        histogram_max_distance=0.7,
        histogram_min_margin=0.05,
        preserve_dark_luminance=40,
    )
    assert np.all(values[20:28, 4:24] == 1)
    assert np.all(values[20:28, 33:52] == 2)
    assert not np.any(values[16:32, 28])
    completion = report["palette_completion"]
    assert completion["completed_pixel_count"] > 0
    assert completion["preserved_dark_ink_pixel_count"] == 16
    assert completion["preserve_dark_ink_enabled"] is True
    assert np.count_nonzero(report["_source_completion_mask"]) > 0


def test_patterned_palette_completion_can_reconstruct_dark_occlusions():
    rgb = np.full((32, 40, 3), 255, dtype=np.uint8)
    red = np.array([220, 40, 40], dtype=np.uint8)
    rgb[0:8, 0:12] = red
    rgb[12:28, 0:36] = red
    rgb[12:28, 18:21] = 0
    categories = [{"id": "red", "sample_rect": [0, 0, 12, 8]}]
    state_mask = np.zeros(rgb.shape[:2], dtype=bool)
    state_mask[12:28, 0:36] = True

    values, report = classify_patterned_categorical(
        rgb,
        state_mask,
        categories,
        window_size=3,
        max_distance=1.0,
        min_margin=0.1,
        max_color_distance=1.0,
        min_color_margin=1.0,
        complete_palette=True,
        preserve_dark_luminance=40,
        preserve_dark_ink=False,
    )

    assert np.all(values[state_mask] == 1)
    completion = report["palette_completion"]
    assert completion["preserve_dark_ink_enabled"] is False
    assert completion["preserved_dark_ink_pixel_count"] == 0
    assert np.count_nonzero(report["_source_preserved_dark_mask"]) == 0


def test_sparse_chroma_unmixing_can_retain_two_independent_overlays():
    categories = [
        {"id": "red", "legend_rgb": [200, 135, 136]},
        {"id": "magenta", "legend_rgb": [220, 141, 187]},
        {"id": "cyan", "legend_rgb": [142, 215, 246]},
    ]
    refs = np.asarray([[category["legend_rgb"] for category in categories]], dtype=np.uint8)
    reference_lab = cv2.cvtColor(refs, cv2.COLOR_RGB2LAB)[0].astype(float)
    mixed_ab = 0.65 * (reference_lab[0, 1:] - 128) + 0.55 * (
        reference_lab[1, 1:] - 128
    )
    mixed_lab = np.array([[[175, *(mixed_ab + 128)]]], dtype=np.uint8)
    mixed_rgb = cv2.cvtColor(mixed_lab, cv2.COLOR_LAB2RGB)
    masks, report = infer_sparse_chroma_overlays(
        mixed_rgb,
        np.ones((1, 1), dtype=bool),
        categories,
        min_chroma=5,
        coefficient_threshold=0.25,
        max_residual=3,
        complexity_penalty=0,
    )
    assert masks[0][0, 0]
    assert masks[1][0, 0]
    assert not masks[2][0, 0]
    assert report["inferred_multi_overlay_pixel_count"] == 1


def test_grayscale_legend_classes_do_not_absorb_pale_basemap():
    rgb = np.zeros((9, 28, 3), dtype=np.uint8)
    for start, value in zip((0, 7, 14, 21), (171, 139, 91, 230)):
        rgb[:, start : start + 7] = value
    categories = [
        {"id": "light", "legend_gray": 171},
        {"id": "medium", "legend_gray": 139},
        {"id": "dark", "legend_gray": 91},
    ]
    values, report = classify_grayscale(
        rgb,
        np.ones(rgb.shape[:2], dtype=bool),
        categories,
        smoothing_radius=1,
        max_distance=20,
        max_chroma=5,
        adaptive_centers=False,
    )
    assert values[4, [3, 10, 17, 24]].tolist() == [1, 2, 3, 0]
    assert report["ambiguous_pixel_count"] > 0
