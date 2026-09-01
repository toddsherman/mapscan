import numpy as np
import pytest

from mapscan.manual_stamp import (
    apply_clone_stamp_operations,
    apply_inference_exclusions,
    validate_inference_exclusions,
    validate_stamp_operations,
)


def test_clone_stamp_copies_a_solid_multiclass_override_patch():
    observed = np.zeros((30, 50), dtype=np.uint8)
    observed[5:15, 5:10] = 1
    observed[5:15, 10:15] = 2
    observed[20:25, 32:35] = 3
    operations = [
        {
            "layer_id": "hazard",
            "source": [10, 10],
            "target": [35, 20],
            "radius_px": 7,
        }
    ]
    manual, mask = apply_clone_stamp_operations(observed, operations)
    assert np.count_nonzero(manual == 1) > 0
    assert np.count_nonzero(manual == 2) > 0
    assert np.all(manual[20:25, 32:35] == 1)
    assert np.all(mask[20:25, 32:35])
    assert mask[20, 38]
    assert manual[20, 38] == 2
    assert mask[26, 35]
    assert manual[26, 35] == 0
    assert np.all(observed[20:25, 32:35] == 3)


def test_stamp_validation_rejects_bad_radius_and_normalizes_points():
    operations = validate_stamp_operations(
        [
            {
                "layer_id": "hazard",
                "source": [10.12345, 12.5],
                "target": [30.5, 40.5],
                "radius_px": 8,
            }
        ],
        width=100,
        height=100,
        layer_ids=["hazard"],
    )
    assert operations[0]["source"] == [10.123, 12.5]
    assert operations[0]["source_mode"] == "observed"
    with pytest.raises(ValueError, match="radius"):
        validate_stamp_operations(
            [
                {
                    "layer_id": "hazard",
                    "source": [10, 10],
                    "target": [20, 20],
                    "radius_px": 0,
                }
            ],
            width=100,
            height=100,
            layer_ids=["hazard"],
        )


def test_composite_source_can_reuse_an_earlier_manual_stamp():
    observed = np.zeros((15, 25), dtype=np.uint8)
    observed[5, 3] = 2
    operations = [
        {
            "layer_id": "hazard",
            "source": [3, 5],
            "target": [10, 5],
            "radius_px": 1,
        },
        {
            "layer_id": "hazard",
            "source": [10, 5],
            "target": [18, 5],
            "radius_px": 1,
            "source_mode": "composite_at_operation_time",
        },
    ]
    manual, mask = apply_clone_stamp_operations(observed, operations)
    assert manual[5, 10] == 2
    assert manual[5, 18] == 2
    assert mask[5, 18]


def test_inference_exclusion_only_marks_existing_inference_pixels():
    inference = np.zeros((30, 40), dtype=bool)
    inference[10:20, 12:28] = True
    operations = validate_inference_exclusions(
        [
            {
                "layer_id": "hazard",
                "center": [20, 15],
                "radius_px": 8,
            }
        ],
        width=40,
        height=30,
        layer_ids=["hazard"],
    )
    excluded = apply_inference_exclusions(inference, operations)
    assert np.count_nonzero(excluded) > 0
    assert np.all(excluded <= inference)
    assert not excluded[0, 0]
