import json

import numpy as np
import pytest

from mapscan.refinement import fit_local_review_corrections, fit_review_corrections


def _inputs(tmp_path, concentrated=False):
    alignment = {
        "schema_version": 1,
        "status": "diagnostic_only",
        "alignment_mode": "assisted",
        "transform_model": "projective_homography",
        "reference_to_source_matrix": np.eye(3).tolist(),
        "metrics": {"control_point_median_px": 2.0},
    }
    alignment_path = tmp_path / "base.json"
    alignment_path.write_text(json.dumps(alignment))
    if concentrated:
        current = np.asarray(
            [[100, 100], [110, 100], [100, 110], [110, 110], [105, 102], [102, 105]],
            dtype=float,
        )
    else:
        current = np.asarray(
            [
                [50, 50],
                [950, 50],
                [50, 950],
                [950, 950],
                [500, 100],
                [100, 500],
                [900, 500],
                [500, 900],
            ],
            dtype=float,
        )
    affine = np.asarray([[1.02, -0.01, -18], [0.005, 1.01, 7], [0, 0, 1]])
    homogeneous = np.column_stack((current, np.ones(len(current)))) @ affine.T
    target = homogeneous[:, :2]
    corrections = {
        "direction": "current_warped_source_to_authoritative_target",
        "grid": {
            "crs": "EPSG:3857",
            "bounds": [0, 0, 1000, 1000],
            "width": 1001,
            "height": 1001,
        },
        "corrections": [
            {
                "current": {"pixel": {"x": float(source[0]), "y": float(source[1])}},
                "target": {"pixel": {"x": float(goal[0]), "y": float(goal[1])}},
            }
            for source, goal in zip(current, target)
        ],
    }
    corrections_path = tmp_path / "corrections.json"
    corrections_path.write_text(json.dumps(corrections))
    return alignment_path, corrections_path


def test_refinement_selects_affine_before_projective(tmp_path):
    alignment, corrections = _inputs(tmp_path)
    report = fit_review_corrections(
        alignment, corrections, tmp_path / "result", max_leave_one_out_p90_px=0.01
    )
    assert report["selected_model"] == "affine"
    refined = json.loads((tmp_path / "result" / "alignment.json").read_text())
    assert refined["web_mercator_correction"]["model"] == "affine"
    assert refined["web_mercator_correction"]["coverage"]["x_fraction"] == 0.9


def test_refinement_reads_transform_model_from_automatic_best_candidate(tmp_path):
    alignment_path, corrections = _inputs(tmp_path)
    automatic = {
        "schema_version": 1,
        "status": "diagnostic_only",
        "alignment_mode": "automatic",
        "best": {
            "projection": "conus_albers",
            "projection_crs": "EPSG:5070",
            "transform_model": "affine_like",
            "parameters": {},
        },
    }
    alignment_path.write_text(json.dumps(automatic))
    report = fit_review_corrections(
        alignment_path,
        corrections,
        tmp_path / "automatic-result",
        max_leave_one_out_p90_px=0.01,
    )
    assert report["selected_model"] == "affine"
    refined = json.loads(
        (tmp_path / "automatic-result" / "alignment.json").read_text()
    )
    assert refined["transform_model"].startswith("affine_like+")


def test_refinement_infers_legacy_automatic_affine_like_model(tmp_path):
    alignment_path, corrections = _inputs(tmp_path)
    automatic = {
        "schema_version": 1,
        "status": "diagnostic_only",
        "alignment_mode": "automatic",
        "best": {
            "projection": "conus_albers",
            "projection_crs": "EPSG:5070",
            "parameters": {
                "center_x_fraction": 0.5,
                "center_y_fraction": 0.5,
                "state_height_fraction": 0.9,
                "x_scale_ratio": 1.0,
                "rotation_degrees": 0.0,
                "x_shear": 0.0,
            },
        },
    }
    alignment_path.write_text(json.dumps(automatic))
    fit_review_corrections(
        alignment_path,
        corrections,
        tmp_path / "legacy-result",
        max_leave_one_out_p90_px=0.01,
    )
    refined = json.loads((tmp_path / "legacy-result" / "alignment.json").read_text())
    assert refined["transform_model"].startswith("affine_like+")


def test_refinement_rejects_concentrated_controls(tmp_path):
    alignment, corrections = _inputs(tmp_path, concentrated=True)
    with pytest.raises(ValueError, match="too concentrated"):
        fit_review_corrections(alignment, corrections, tmp_path / "result")


def test_refinement_composes_with_parent_correction(tmp_path):
    alignment, corrections = _inputs(tmp_path)
    fit_review_corrections(
        alignment, corrections, tmp_path / "first", max_leave_one_out_p90_px=0.01
    )
    first_alignment = tmp_path / "first" / "alignment.json"
    fit_review_corrections(
        first_alignment,
        corrections,
        tmp_path / "second",
        max_leave_one_out_p90_px=0.01,
    )
    first = json.loads(first_alignment.read_text())["web_mercator_correction"]
    second = json.loads((tmp_path / "second" / "alignment.json").read_text())[
        "web_mercator_correction"
    ]
    parent = np.asarray(first["target_to_current_normalized_matrix"])
    incremental = np.asarray(second["incremental_target_to_parent_normalized_matrix"])
    combined = np.asarray(second["target_to_current_normalized_matrix"])
    assert second["composition_depth"] == 2
    assert np.allclose(combined, parent @ incremental)


def test_three_distributed_points_can_select_only_validated_simple_model(tmp_path):
    alignment = {
        "schema_version": 1,
        "status": "diagnostic_only",
        "alignment_mode": "assisted",
        "transform_model": "projective_homography",
        "reference_to_source_matrix": np.eye(3).tolist(),
    }
    alignment_path = tmp_path / "base.json"
    alignment_path.write_text(json.dumps(alignment))
    source = np.asarray([[100, 100], [900, 200], [500, 900]], dtype=float)
    reference = source + np.asarray([[5, 4], [4, 5], [5, 5]], dtype=float)
    corrections = {
        "direction": "authoritative_reference_to_current_warped_source",
        "grid": {
            "crs": "EPSG:3857",
            "bounds": [0, 0, 1000, 1000],
            "width": 1001,
            "height": 1001,
        },
        "corrections": [
            {
                "reference": {"pixel": {"x": float(goal[0]), "y": float(goal[1])}},
                "source": {"pixel": {"x": float(point[0]), "y": float(point[1])}},
            }
            for point, goal in zip(source, reference)
        ],
    }
    corrections_path = tmp_path / "small-corrections.json"
    corrections_path.write_text(json.dumps(corrections))
    report = fit_review_corrections(
        alignment_path, corrections_path, tmp_path / "result"
    )
    assert report["selected_model"] == "translation"
    affine = next(item for item in report["candidates"] if item["model"] == "affine")
    assert affine["evaluated"] is False


def test_axis_spanning_but_nearly_collinear_points_are_not_global_controls(tmp_path):
    alignment = {
        "schema_version": 1,
        "status": "diagnostic_only",
        "alignment_mode": "assisted",
        "transform_model": "projective_homography",
        "reference_to_source_matrix": np.eye(3).tolist(),
    }
    alignment_path = tmp_path / "base.json"
    alignment_path.write_text(json.dumps(alignment))
    source = np.asarray([[100, 100], [500, 500], [900, 905]], dtype=float)
    reference = source + np.asarray([5, 4])
    record = {
        "direction": "authoritative_reference_to_current_warped_source",
        "grid": {
            "crs": "EPSG:3857",
            "bounds": [0, 0, 1000, 1000],
            "width": 1001,
            "height": 1001,
        },
        "corrections": [
            {
                "reference": {"pixel": {"x": float(goal[0]), "y": float(goal[1])}},
                "source": {"pixel": {"x": float(point[0]), "y": float(point[1])}},
            }
            for point, goal in zip(source, reference)
        ],
    }
    corrections_path = tmp_path / "linear.json"
    corrections_path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="substantial map area"):
        fit_review_corrections(alignment_path, corrections_path, tmp_path / "result")


def test_compact_local_refinement_is_exact_and_preserves_parent_operation(tmp_path):
    alignment = {
        "schema_version": 2,
        "status": "diagnostic_only",
        "alignment_mode": "assisted",
        "transform_model": "projective_homography+review",
        "reference_to_source_matrix": np.eye(3).tolist(),
        "web_mercator_correction": {
            "model": "affine",
            "grid": {
                "crs": "EPSG:3857",
                "bounds": [0, 0, 1000, 1000],
                "width": 1001,
                "height": 1001,
            },
            "target_to_current_normalized_matrix": np.eye(3).tolist(),
            "composition_depth": 1,
        },
    }
    alignment_path = tmp_path / "base.json"
    alignment_path.write_text(json.dumps(alignment))
    reference = np.asarray([[100, 100], [800, 200], [700, 850]], dtype=float)
    source = reference + np.asarray([[5, 4], [4, 3], [6, 5]], dtype=float)
    record = {
        "direction": "authoritative_reference_to_current_warped_source",
        "grid": alignment["web_mercator_correction"]["grid"],
        "corrections": [
            {
                "reference": {"pixel": {"x": float(goal[0]), "y": float(goal[1])}},
                "source": {"pixel": {"x": float(point[0]), "y": float(point[1])}},
            }
            for goal, point in zip(reference, source)
        ],
    }
    corrections_path = tmp_path / "local.json"
    corrections_path.write_text(json.dumps(record))
    report = fit_local_review_corrections(
        alignment_path, corrections_path, tmp_path / "result", radius_px=250
    )
    assert report["control_residuals"]["max_px"] < 1e-8
    assert report["sampled_jacobian_min"] > 0
    refined = json.loads((tmp_path / "result" / "alignment.json").read_text())
    operations = refined["web_mercator_correction"]["operations"]
    assert operations[0]["type"] == "compact_wendland_c2_displacement"
    assert operations[1]["type"] == "matrix"
