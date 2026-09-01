from __future__ import annotations

import numpy as np
import pytest

from mapscan.coastal_occlusion_repair import (
    PerimeterOcclusionConfig,
    _processing_transform,
    recover_perimeter_cartographic_occlusions,
)


def test_repairs_only_dark_coastal_ink_from_nearest_existing_class() -> None:
    rgb = np.full((15, 15, 3), 255, dtype=np.uint8)
    class_ids = np.zeros((15, 15), dtype=np.uint8)
    class_ids[4:11, 8:12] = 2
    rgb[class_ids == 2] = (84, 150, 70)

    # A two-pixel neutral coastline stroke hides thematic data just inland.
    rgb[4:11, 6:8] = (18, 18, 18)
    source_land = np.zeros((15, 15), dtype=bool)
    source_land[:, 6:] = True
    source_coast = np.zeros((15, 15), dtype=bool)
    source_coast[:, 6] = True
    repaired, repair_mask, diagnostics = recover_perimeter_cartographic_occlusions(
        rgb,
        class_ids,
        source_land,
        source_coast,
        source_coast,
    )

    assert np.all(repaired[4:11, 6:8] == 2)
    assert np.all(repair_mask[4:11, 6:8])
    assert np.array_equal(repaired[class_ids > 0], class_ids[class_ids > 0])
    assert not np.any(repair_mask[:, :6])
    assert not np.any(repair_mask[rgb[:, :, 0] == 255])
    assert diagnostics["source_repaired_pixel_count"] == 14
    assert diagnostics["repaired_class_counts"] == {"2": 14}


def test_does_not_outpaint_non_neutral_or_noncoastal_gaps() -> None:
    rgb = np.full((13, 13, 3), 255, dtype=np.uint8)
    class_ids = np.zeros((13, 13), dtype=np.uint8)
    class_ids[3:10, 7:11] = 1
    rgb[class_ids == 1] = (40, 160, 70)
    rgb[4, 6] = (10, 10, 10)  # eligible coastal ink
    rgb[5, 6] = (30, 80, 150)  # chromatic cartography, not neutral ink
    rgb[6, 9] = (10, 10, 10)  # dark, but not a missing class or coast pixel
    class_ids[6, 9] = 1
    rgb[9, 3] = (10, 10, 10)  # dark and nearby, but not coastal

    land = np.zeros((13, 13), dtype=bool)
    land[:, 6:] = True
    coast = np.zeros((13, 13), dtype=bool)
    coast[:, 6] = True
    repaired, mask, _ = recover_perimeter_cartographic_occlusions(
        rgb, class_ids, land, coast, coast
    )

    assert repaired[4, 6] == 1
    assert mask[4, 6]
    assert repaired[5, 6] == 0
    assert not mask[5, 6]
    assert repaired[9, 3] == 0
    assert not mask[9, 3]
    assert repaired[6, 9] == 1
    assert not mask[6, 9]


def test_requires_source_and_projected_mapbox_perimeter_support() -> None:
    rgb = np.full((15, 15, 3), 255, dtype=np.uint8)
    class_ids = np.zeros((15, 15), dtype=np.uint8)
    class_ids[3:7, 2:5] = 1
    class_ids[8:12, 8:11] = 2
    rgb[class_ids == 1] = (49, 151, 49)
    rgb[class_ids == 2] = (206, 154, 156)
    rgb[3:7, 1] = (15, 15, 15)
    rgb[8:12, 7] = (15, 15, 15)

    source_perimeter = np.zeros((15, 15), dtype=bool)
    source_perimeter[:, 1] = True  # false source-only line
    source_perimeter[:, 7] = True  # true state perimeter
    mapbox_perimeter = np.zeros((15, 15), dtype=bool)
    mapbox_perimeter[:, 7] = True
    land = np.ones((15, 15), dtype=bool)

    repaired, mask, _ = recover_perimeter_cartographic_occlusions(
        rgb,
        class_ids,
        land,
        source_perimeter,
        mapbox_perimeter,
        config=PerimeterOcclusionConfig(reference_perimeter_tolerance_px=2),
    )

    assert not np.any(mask[3:7, 1])
    assert np.all(repaired[3:7, 1] == 0)
    assert np.all(mask[8:12, 7])
    assert np.all(repaired[8:12, 7] == 2)


def test_rejects_negative_reference_perimeter_tolerance() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        PerimeterOcclusionConfig(reference_perimeter_tolerance_px=-1)


def test_processing_transform_matches_declared_supersampled_grid() -> None:
    transform = {
        "kind": "regular_global_mapbox_registration",
        "target_grid": {
            "crs": "EPSG:3857",
            "bounds": [0.0, 0.0, 9.0, 9.0],
            "width": 4,
            "height": 5,
        },
        "source_original_to_reference_pixel_matrix": [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        "reference_pixel_to_source_original_matrix": [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
    }
    processing_grid = {
        "crs": "EPSG:3857",
        "bounds": [0.0, 0.0, 9.0, 9.0],
        "width": 10,
        "height": 13,
    }
    pointer = {
        "target_supersampling": 3,
        "processing_target_grid": processing_grid,
    }

    lifted = _processing_transform(pointer, transform, processing_grid)

    assert lifted["target_grid"] == processing_grid
    assert lifted["source_original_to_reference_pixel_matrix"][0][0] == 3.0
    assert lifted["reference_pixel_to_source_original_matrix"][0][0] == 1.0 / 3.0


def test_processing_transform_rejects_wrong_supersampled_reference() -> None:
    transform = {
        "kind": "regular_global_mapbox_registration",
        "target_grid": {
            "bounds": [0.0, 0.0, 9.0, 9.0],
            "width": 4,
            "height": 5,
        },
        "source_original_to_reference_pixel_matrix": np.eye(3).tolist(),
        "reference_pixel_to_source_original_matrix": np.eye(3).tolist(),
    }
    pointer = {
        "target_supersampling": 3,
        "processing_target_grid": {
            "bounds": [0.0, 0.0, 9.0, 9.0],
            "width": 10,
            "height": 13,
        },
    }

    with pytest.raises(ValueError, match="supersampled target grid"):
        _processing_transform(
            pointer,
            transform,
            {
                "bounds": [0.0, 0.0, 9.0, 9.0],
                "width": 4,
                "height": 5,
            },
        )
