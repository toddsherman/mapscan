import json

import numpy as np
import pytest

from mapscan.resolution_fidelity_audit import (
    analyze_transform_resolution,
    audit_restart_run,
    plan_native_regional_diffs,
)


def _affine_transform(scale, *, source_shape=(1000, 1000), offset=(0, 0)):
    forward = np.asarray(
        [[scale, 0.0, offset[0]], [0.0, scale, offset[1]], [0.0, 0.0, 1.0]]
    )
    return {
        "kind": "regular_global_mapbox_registration",
        "source_original_shape": list(source_shape),
        "working_scale_from_original": 0.25,
        "target_grid": {
            "crs": "EPSG:3857",
            "bounds": [0.0, 0.0, 100.0, 100.0],
            "width": 100,
            "height": 100,
        },
        "reference_pixel_to_source_original_matrix": forward.tolist(),
        "source_original_to_reference_pixel_matrix": np.linalg.inv(forward).tolist(),
    }


def test_resolution_policy_detects_real_source_downsampling():
    report = analyze_transform_resolution(_affine_transform(2.2))

    assert report["source_pixels_per_target_pixel"][
        "p95_maximum_axis"
    ] == pytest.approx(2.2)
    assert report["target_grid_supersampling_factor"] == 3
    assert report["target_grid_resolution_loss"] is True
    assert report["registration_detail_risk"] is True


def test_resolution_policy_does_not_supersample_an_already_finer_target():
    report = analyze_transform_resolution(_affine_transform(0.5))

    assert report["target_grid_supersampling_factor"] == 1
    assert report["target_grid_resolution_loss"] is False


def test_native_crop_plan_omits_cells_outside_a_partial_source():
    transform = _affine_transform(1.0, source_shape=(60, 60), offset=(-50, 0))
    cells = plan_native_regional_diffs(transform, rows=2, columns=2)

    assert [cell["id"] for cell in cells] == ["r1-c2", "r2-c2"]
    assert all(cell["comparison_space"] == "original_source_pixels" for cell in cells)


def test_batch_audit_summarizes_iterations_and_supersampled_grid(tmp_path):
    run_root = tmp_path / "restart"
    map_dir = run_root / "sample"
    alignment_dir = map_dir / "automatic-alignment"
    extraction_dir = map_dir / "automatic-extraction"
    iteration_dir = extraction_dir / "extraction-02"
    iteration_dir.mkdir(parents=True)
    alignment_dir.mkdir()
    experiment = {
        "source": {"path": "/source.png", "sha256": "a" * 64},
        "mapbox_reference": {"zoom": 9},
        "alignment": {"accepted_automatic_iteration_count": 4},
        "extraction": {"accepted_automatic_iteration_count": 2},
        "final": {"status": "complete", "blocker": None},
    }
    (map_dir / "EXPERIMENT.json").write_text(json.dumps(experiment))
    (alignment_dir / "accepted-alignment.json").write_text(
        json.dumps({"transform": _affine_transform(2.0)})
    )
    (extraction_dir / "accepted-extraction.json").write_text(
        json.dumps({"accepted_iteration": "extraction-02"})
    )
    (iteration_dir / "iteration.json").write_text(
        json.dumps(
            {
                "scores": {
                    "meaningful_source_mismatch_fraction": 0.01,
                    "unrelated_count": 2,
                },
                "gates": {},
            }
        )
    )

    report = audit_restart_run(run_root)

    sample = report["maps"][0]
    assert sample["accepted_alignment_iteration"] == 4
    assert sample["accepted_extraction_iteration"] == 2
    assert sample["recommended_target_grid"]["width"] == 298
    assert sample["recommended_target_grid"]["height"] == 298
    assert sample["recommended_target_grid"]["corner_preserving"] is True
    assert sample["extraction_fidelity_metrics"] == {
        "meaningful_source_mismatch_fraction": 0.01
    }
    assert sample["extraction_fidelity_risk"]["measured_mismatch"] is True
    assert report["execution_queue"][0]["map_id"] == "sample"
