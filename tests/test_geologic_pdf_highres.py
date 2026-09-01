import numpy as np

from mapscan.geologic_pdf_highres import (
    TARGET_HEIGHT,
    TARGET_WIDTH,
    _complete_small_gaps,
    _corner_preserving_dimension,
    _highres_transform,
)


def test_required_grid_is_corner_preserving_three_x() -> None:
    assert _corner_preserving_dimension(3398, 3) == TARGET_WIDTH
    assert _corner_preserving_dimension(3920, 3) == TARGET_HEIGHT


def test_highres_transform_changes_only_target_sampling_contract() -> None:
    alignment = {
        "transform": {
            "kind": "projection_aware_mapbox_registration",
            "candidate_normalized_to_source_original_matrix": [[1, 0, 2], [0, 1, 3], [0, 0, 1]],
            "target_grid": {
                "crs": "EPSG:3857",
                "bounds": [1, 2, 3, 4],
                "width": 3398,
                "height": 3920,
            },
        }
    }
    processing = {
        "crs": "EPSG:3857",
        "bounds": [1, 2, 3, 4],
        "width": TARGET_WIDTH,
        "height": TARGET_HEIGHT,
    }
    result = _highres_transform(alignment, processing)
    assert result["candidate_normalized_to_source_original_matrix"] == [
        [1, 0, 2],
        [0, 1, 3],
        [0, 0, 1],
    ]
    assert result["target_grid"] == processing
    assert result["resampling_contract"]["prior_class_raster_used"] is False


def test_local_completion_uses_neighbor_classes_and_preserves_exterior() -> None:
    observed = np.array(
        [
            [0, 1, 1, 1, 0],
            [0, 1, 0, 2, 0],
            [0, 1, 2, 2, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    domain = np.array(
        [
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=bool,
    )
    complete, inferred = _complete_small_gaps(observed, domain)
    assert inferred[1, 2]
    assert complete[1, 2] in {1, 2}
    assert np.all(complete[~domain] == 0)
