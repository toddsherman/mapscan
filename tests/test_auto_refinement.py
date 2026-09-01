import numpy as np
from scipy.ndimage import distance_transform_edt

from mapscan.auto_refinement import (
    _consistent_matches,
    _match_perimeter_anchor,
    _select_distributed,
)


def test_local_perimeter_match_recovers_normal_displacement():
    shape = (120, 120)
    edge = np.zeros(shape, dtype=bool)
    edge[15:106, 57] = True
    distance, nearest = distance_transform_edt(~edge, return_indices=True)
    gradient_x = np.zeros(shape, dtype=np.float32)
    gradient_y = np.zeros(shape, dtype=np.float32)
    gradient_x[edge] = 1.0
    points = np.column_stack((np.full(71, 50.0), np.linspace(20, 100, 71)))
    segment = {
        "id": "perimeter_00",
        "index": 0,
        "center": np.asarray([50.0, 60.0]),
        "points": points,
        "tangent": np.asarray([0.0, 1.0]),
        "normal": np.asarray([-1.0, 0.0]),
        "normals": np.tile(np.asarray([1.0, 0.0]), (len(points), 1)),
    }
    result = _match_perimeter_anchor(
        segment,
        np.ones(shape, dtype=bool),
        distance,
        nearest,
        gradient_x,
        gradient_y,
        search_radius_px=15,
        tangent_radius_px=4,
    )
    assert result["accepted_by_local_evidence"] is True
    assert abs(result["shift_px"][0] - 7) <= 1
    assert abs(result["shift_px"][1]) <= 1


def test_global_consistency_rejects_one_bad_local_match():
    matches = []
    for index, point in enumerate(
        ([10, 10], [90, 10], [10, 90], [90, 90], [50, 15], [15, 50], [85, 50], [50, 85])
    ):
        shift = np.asarray([4.0, -3.0])
        if index == 7:
            shift = np.asarray([22.0, 18.0])
        reference = np.asarray(point, dtype=float)
        matches.append(
            {
                "id": f"perimeter_{index:02d}",
                "index": index,
                "reference_pixel": reference.tolist(),
                "source_pixel": (reference + shift).tolist(),
                "accepted_by_local_evidence": True,
            }
        )
    consistent = _consistent_matches(matches)
    assert len(consistent) == 7
    assert "perimeter_07" not in {item["id"] for item in consistent}


def test_distributed_selection_spans_perimeter():
    matches = []
    for index in range(12):
        angle = 2 * np.pi * index / 12
        matches.append(
            {
                "id": f"perimeter_{index:02d}",
                "reference_pixel": [50 + 45 * np.cos(angle), 50 + 45 * np.sin(angle)],
                "best": {"support_fraction": 0.9, "median_distance_px": 0.5},
                "uniqueness_margin": 1.0,
            }
        )
    chosen = _select_distributed(matches, 8, (101, 101))
    coordinates = np.asarray([item["reference_pixel"] for item in chosen])
    assert len(chosen) == 8
    assert np.ptp(coordinates[:, 0]) > 80
    assert np.ptp(coordinates[:, 1]) > 80
