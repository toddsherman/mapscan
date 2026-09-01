from mapscan.feature_extraction import _alignment_transform


def test_automatic_alignment_applies_top_level_web_mercator_correction():
    correction = {
        "model": "affine",
        "target_to_current_normalized_matrix": [
            [1.0, 0.0, 0.01],
            [0.0, 1.0, -0.02],
            [0.0, 0.0, 1.0],
        ],
    }
    alignment = {
        "alignment_mode": "automatic",
        "best": {
            "projection": "california_albers",
            "projection_crs": "EPSG:3310",
            "transform_model": "affine_like",
        },
        "web_mercator_correction": correction,
    }

    transform = _alignment_transform(alignment)

    assert transform["web_mercator_correction"] is correction
    assert transform["projection_crs"] == "EPSG:3310"
    assert "web_mercator_correction" not in alignment["best"]
