from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

import mapscan.farms_v9_rollforward as farms_v9_module
from mapscan.farms_v9_rollforward import (
    CANONICAL_V9_FINAL_OUTPUT_RELATIVE_PATH,
    EXPECTED_V8_COUNTY_TEST_FILE_SHA256,
    EXPECTED_V8_COUNTY_TEST_LOGICAL_SHA256,
    EXPECTED_V8_COUNTY_VALIDATION_FILE_SHA256,
    EXPECTED_V8_COUNTY_VALIDATION_LOGICAL_SHA256,
    EXPECTED_V8_FROZEN_CANDIDATE_SHA256,
    EXPECTED_V8_VALIDATION_REPORT_SHA256,
    EXPECTED_V9_FROZEN_CANDIDATE_SHA256,
    EXPECTED_V9_PARTITION_RECEIPT_SHA256,
    EXPECTED_V9_VALIDATION_REPORT_SHA256,
    _final_acceptance_gate_decisions,
    _load_exact_v8_role,
    _require_canonical_v9_final_output,
    _require_exact_artifact_sha256,
    _verify_county_assignment_exclusion,
    build_farms_v9_county_split,
    farms_v9_final_gate_contract,
    projection_residual_consumer_roundtrip_report,
    run_farms_v9_one_shot_final_acceptance,
    write_farms_v9_partition_receipt,
)


OLD_TEST = Path(
    "runs/farms-v2-nonrigid-mapbox-v2-validation-preflight-scoped-topology-v8/"
    "mapbox-county-test.png"
)
OLD_VALIDATION = Path(
    "runs/farms-v2-nonrigid-mapbox-v2-validation-preflight-scoped-topology-v8/"
    "mapbox-county-validation.png"
)
V8_REPORT = OLD_TEST.parent / "validation-report.json"
V8_FROZEN = OLD_TEST.parent / "frozen-validation-candidate.json"
V9_FROZEN = Path(
    "runs/farms-v2-nonrigid-mapbox-v2-validation-preflight-rollforward-v9/"
    "frozen-validation-candidate.json"
)


def _old_test() -> np.ndarray:
    return np.asarray(Image.open(OLD_TEST).convert("L")) > 0


def test_v9_split_is_deterministic_buffered_and_target_only():
    old = _old_test()
    first = build_farms_v9_county_split(old)
    second = build_farms_v9_county_split(old.copy())

    assert first.diagnostics == second.diagnostics
    assert np.array_equal(first.validation, second.validation)
    assert np.array_equal(first.final_acceptance, second.final_acceptance)
    assert not np.any(first.validation & first.final_acceptance)
    assert not np.any((first.validation | first.final_acceptance) & first.bay_guard)
    assert np.min(distance_transform_edt(~first.final_acceptance)[first.validation]) == 25
    assert first.diagnostics["authority"]["source_pixels_used"] is False
    assert first.diagnostics["authority"]["candidate_transform_used"] is False
    assert first.diagnostics["authority"]["golden_gate_or_east_bay_masks_read"] is False


def test_v9_real_split_has_frozen_counts_and_geographic_coverage():
    split = build_farms_v9_county_split(_old_test())
    counts = split.diagnostics["pixel_counts"]

    assert counts == {
        "old_county_test": 82_831,
        "bay_guard_intersection": 37_365,
        "eligible": 45_466,
        "validation": 20_222,
        "final_acceptance": 18_281,
        "unused_buffer": 6_963,
    }
    validation = split.diagnostics[
        "validation_geographic_coverage_common_eligible_bounds"
    ]
    final = split.diagnostics[
        "final_acceptance_geographic_coverage_common_eligible_bounds"
    ]
    assert validation["supported_cell_count"] == 15
    assert validation["supported_rows"] == [0, 1, 2, 3, 4, 5]
    assert validation["supported_columns"] == [0, 1, 2, 3, 4, 5]
    assert final["supported_cell_count"] == 16
    assert final["supported_rows"] == [0, 1, 2, 3, 4, 5]
    assert final["supported_columns"] == [0, 1, 2, 3, 4, 5]
    assert validation["common_bounds_xyxy"] == final["common_bounds_xyxy"]
    validation_full = split.diagnostics["validation_geographic_coverage_full_grid"]
    final_full = split.diagnostics["final_acceptance_geographic_coverage_full_grid"]
    assert validation_full["supported_cell_count"] == 13
    assert final_full["supported_cell_count"] == 14
    assert validation_full["common_bounds_xyxy"] == [0, 0, 3398, 3920]
    assert final_full["common_bounds_xyxy"] == [0, 0, 3398, 3920]


def test_v9_split_rejects_noncanonical_grid_or_parameters():
    with np.testing.assert_raises(ValueError):
        build_farms_v9_county_split(np.zeros((100, 100), dtype=bool))
    with np.testing.assert_raises(ValueError):
        build_farms_v9_county_split(_old_test(), role_cells=(12, 12))


def test_v9_receipt_pins_both_authoritative_v8_roles(tmp_path: Path):
    receipt_path = write_farms_v9_partition_receipt(
        OLD_TEST,
        OLD_VALIDATION,
        tmp_path / "partition",
    )
    receipt = json.loads(receipt_path.read_text())
    assert receipt["kind"] == "farms_v9_target_only_county_partition_receipt_v2"
    ancestry = receipt["v8_role_ancestry"]
    assert ancestry["old_county_test"]["file_sha256"] == (
        EXPECTED_V8_COUNTY_TEST_FILE_SHA256
    )
    assert ancestry["old_county_test"]["logical_sha256"] == (
        EXPECTED_V8_COUNTY_TEST_LOGICAL_SHA256
    )
    assert ancestry["old_county_validation"]["file_sha256"] == (
        EXPECTED_V8_COUNTY_VALIDATION_FILE_SHA256
    )
    assert ancestry["old_county_validation"]["logical_sha256"] == (
        EXPECTED_V8_COUNTY_VALIDATION_LOGICAL_SHA256
    )
    persisted = np.asarray(
        Image.open(
            receipt["artifacts"]["consumed_v8_validation_now_training"]["path"]
        ).convert("L")
    ) > 0
    expected = np.asarray(Image.open(OLD_VALIDATION).convert("L")) > 0
    assert np.array_equal(persisted, expected)


def test_v9_exact_role_loader_rejects_same_shape_substitute(tmp_path: Path):
    substituted = _old_test().copy()
    substituted[0, 0] = ~substituted[0, 0]
    path = tmp_path / "same-shape-substitute.png"
    Image.fromarray(substituted.astype(np.uint8) * 255).save(path)
    with np.testing.assert_raises_regex(ValueError, "file hash mismatch"):
        _load_exact_v8_role(
            path,
            role="county_test",
            expected_file_sha256=EXPECTED_V8_COUNTY_TEST_FILE_SHA256,
            expected_logical_sha256=EXPECTED_V8_COUNTY_TEST_LOGICAL_SHA256,
        )


def test_v9_history_rejects_whole_file_tamper(tmp_path: Path):
    _require_exact_artifact_sha256(
        V8_REPORT,
        label="v8 validation report",
        expected_sha256=EXPECTED_V8_VALIDATION_REPORT_SHA256,
    )
    _require_exact_artifact_sha256(
        V8_FROZEN,
        label="v8 frozen candidate",
        expected_sha256=EXPECTED_V8_FROZEN_CANDIDATE_SHA256,
    )
    tampered = tmp_path / "validation-report.json"
    tampered.write_bytes(V8_REPORT.read_bytes() + b" ")
    with np.testing.assert_raises_regex(ValueError, "validation report hash mismatch"):
        _require_exact_artifact_sha256(
            tampered,
            label="v8 validation report",
            expected_sha256=EXPECTED_V8_VALIDATION_REPORT_SHA256,
        )


def test_v9_assignment_must_use_exact_24px_heldout_exclusion():
    rendered = np.zeros((100, 100), dtype=bool)
    rendered[50, 20] = True
    exclusion = cv2.dilate(
        rendered.astype(np.uint8), np.ones((49, 49), np.uint8)
    ).astype(bool)
    assignment = np.zeros_like(rendered)
    assignment[50, 50] = True
    report = _verify_county_assignment_exclusion(
        source_assignment=assignment,
        rendered_heldout=rendered,
        expected_exclusion=exclusion,
        actual_exclusion=exclusion.copy(),
    )
    assert report["passed"] is True
    assert report["minimum_assignment_to_rendered_heldout_working_px"] == 30

    changed_exclusion = exclusion.copy()
    changed_exclusion[0, 0] = ~changed_exclusion[0, 0]
    with np.testing.assert_raises_regex(ValueError, "frozen v9 exclusion"):
        _verify_county_assignment_exclusion(
            source_assignment=assignment,
            rendered_heldout=rendered,
            expected_exclusion=exclusion,
            actual_exclusion=changed_exclusion,
        )

    unsafe_assignment = np.zeros_like(rendered)
    unsafe_assignment[50, 40] = True
    with np.testing.assert_raises_regex(ValueError, "intersects"):
        _verify_county_assignment_exclusion(
            source_assignment=unsafe_assignment,
            rendered_heldout=rendered,
            expected_exclusion=exclusion,
            actual_exclusion=exclusion,
        )


def test_v9_final_contract_is_frozen_before_sealed_scores():
    contract = farms_v9_final_gate_contract()
    assert contract["expected_sha256"]["partition_receipt"] == (
        EXPECTED_V9_PARTITION_RECEIPT_SHA256
    )
    assert contract["expected_sha256"]["validation_report"] == (
        EXPECTED_V9_VALIDATION_REPORT_SHA256
    )
    assert contract["expected_sha256"]["frozen_validation_candidate"] == (
        EXPECTED_V9_FROZEN_CANDIDATE_SHA256
    )
    assert contract["candidate_policy"] == (
        "exact_frozen_v9_candidate_only_no_tuning_or_fallback"
    )
    assert contract["final_county"] == {
        "minimum_visible_count": 20,
        "minimum_visible_fraction": 0.25,
        "maximum_median_working_px": 7.0,
        "minimum_within_8px_fraction": 0.58,
        "balanced_grid": [6, 6],
        "balanced_minimum_visible_count_per_cell": 20,
        "balanced_minimum_visible_cells": 6,
        "balanced_minimum_visible_rows": 3,
        "balanced_minimum_visible_columns": 3,
        "balanced_maximum_p90_working_px": 12.0,
        "balanced_minimum_cell_pass_fraction": 0.7,
        "balanced_minimum_axis_pass_fraction": 0.5,
    }
    assert contract["authority"]["parameter_tuning_after_final_scores"] is False
    assert contract["authority"]["fallback_candidate_after_final_scores"] is False


def test_v9_final_gates_are_conjunctive_and_do_not_select_a_candidate():
    county = {
        "visible_count": 100,
        "visible_fraction": 0.5,
        "median_px": 1.0,
        "within_8px_fraction": 0.9,
    }
    bay = {
        "visible_count": 100,
        "visible_fraction": 0.9,
        "median_px": 1.0,
        "p90_px": 4.0,
        "within_8px_fraction": 0.9,
    }
    passed = _final_acceptance_gate_decisions(
        county=county,
        balanced_county={"passed": True},
        golden_gate=bay,
        east_bay=bay,
        regularity={"passed": True},
        serialized_roundtrip={"passed": True},
        full_consumer_preflight={"passed": True},
    )
    assert all(passed.values())

    failed_bay = dict(bay)
    failed_bay["p90_px"] = 12.01
    failed = _final_acceptance_gate_decisions(
        county=county,
        balanced_county={"passed": True},
        golden_gate=failed_bay,
        east_bay=bay,
        regularity={"passed": True},
        serialized_roundtrip={"passed": True},
        full_consumer_preflight={"passed": True},
    )
    assert failed["golden_gate_named_bay"] is False
    assert sum(not value for value in failed.values()) == 1


def test_v9_final_rejects_alternate_fresh_output_root(tmp_path: Path):
    with np.testing.assert_raises_regex(ValueError, "must use canonical path"):
        _require_canonical_v9_final_output(tmp_path / "runs" / "alternate-final")


def test_v9_existing_canonical_output_blocks_retry(
    tmp_path: Path, monkeypatch
):
    canonical = tmp_path / CANONICAL_V9_FINAL_OUTPUT_RELATIVE_PATH
    canonical.mkdir(parents=True)
    monkeypatch.setattr(
        farms_v9_module, "CANONICAL_V9_FINAL_OUTPUT_ROOT", canonical.resolve()
    )
    with np.testing.assert_raises_regex(FileExistsError, "retry is forbidden"):
        _require_canonical_v9_final_output(canonical)


def test_v9_copied_pristine_source_cannot_relocate_one_shot_root(tmp_path: Path):
    copied_source = tmp_path / "relocated" / "examples" / "farmsv2.png"
    copied_source.parent.mkdir(parents=True)
    shutil.copyfile("examples/farmsv2.png", copied_source)
    relocated_output = (
        copied_source.parents[1] / CANONICAL_V9_FINAL_OUTPUT_RELATIVE_PATH
    )
    with np.testing.assert_raises_regex(ValueError, "must use canonical path"):
        run_farms_v9_one_shot_final_acceptance(
            source_path=copied_source,
            reference_manifest_path=Path(
                "reference/mapbox-light-v11-california-z9-v2/manifest.json"
            ),
            v8_preflight_root=OLD_TEST.parent,
            partition_receipt_path=Path(
                "runs/farms-v2-nonrigid-mapbox-v2-v9-partition-receipt/"
                "partition-receipt.json"
            ),
            v9_validation_report_path=Path(
                "runs/farms-v2-nonrigid-mapbox-v2-validation-preflight-rollforward-v9/"
                "validation-report.json"
            ),
            v9_frozen_candidate_path=Path(
                "runs/farms-v2-nonrigid-mapbox-v2-validation-preflight-rollforward-v9/"
                "frozen-validation-candidate.json"
            ),
            output_root=relocated_output,
        )


def test_v9_frozen_projection_residual_contract_roundtrips_like_consumer():
    transform = json.loads(V9_FROZEN.read_text())["serialized_transform"]
    grid = transform["target_grid"]
    x = np.linspace(200, int(grid["width"]) - 201, 9)
    y = np.linspace(200, int(grid["height"]) - 201, 11)
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack((xx.ravel(), yy.ravel()))
    report = projection_residual_consumer_roundtrip_report(transform, points)
    assert report["passed"] is True
    assert report["converged"] is True
    assert report["point_count"] == 99
    assert report["maximum_source_roundtrip_error_px"] <= 0.02
    assert report["maximum_reference_roundtrip_error_px"] <= 0.02
    assert report["iterations_used"] <= 20
    assert report["production_helpers_used"] == [
        "_projection_reference_to_source_base",
        "_projection_source_to_reference_base",
        "_residual_displacement",
    ]

    nonconvergent = json.loads(json.dumps(transform))
    nonconvergent["inverse_solver"]["maximum_iterations"] = 1
    nonconvergent["inverse_solver"]["reference_tolerance_px"] = 1e-8
    failed = projection_residual_consumer_roundtrip_report(
        nonconvergent, points
    )
    assert failed["converged"] is False
    assert failed["passed"] is False
