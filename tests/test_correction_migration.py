import hashlib
import json

import numpy as np
from PIL import Image

import mapscan.correction_migration as correction_migration
from mapscan.correction_migration import (
    audit_alignment_application,
    migrate_stamp_corrections,
)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fixture(tmp_path):
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()
    grid = {
        "crs": "EPSG:3857",
        "bounds": [0.0, 0.0, 10.0, 10.0],
        "width": 10,
        "height": 10,
    }
    source_alignment = tmp_path / "source-alignment.json"
    source_alignment.write_text(
        json.dumps(
            {
                "alignment_mode": "assisted",
                "reference": {"crs": "EPSG:3310"},
                "transform_model": "projective_homography",
                "reference_to_source_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "web_mercator_correction": {"grid": grid},
            }
        )
    )
    target_alignment = tmp_path / "target-alignment.json"
    target_alignment.write_text(
        json.dumps(
            {
                "alignment_mode": "assisted",
                "reference": {"crs": "EPSG:3310"},
                "transform_model": "projective_homography+child",
                "reference_to_source_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "parent_alignment": {"sha256": _sha256(source_alignment)},
                "web_mercator_correction": {
                    "grid": grid,
                    "current_to_target_pixel_matrix": [
                        [1, 0, 1],
                        [0, 1, 2],
                        [0, 0, 1],
                    ],
                },
            }
        )
    )
    source_manifest = {
        "dataset_id": "old",
        "source": {"sha256": "same"},
        "alignment": {"path": str(source_alignment)},
        "layers": [{"id": "hazard", "warp": grid}],
    }
    target_manifest = {
        "dataset_id": "new",
        "source": {"sha256": "same"},
        "alignment": {"path": str(target_alignment)},
        "layers": [{"id": "hazard", "warp": grid}],
    }
    (source / "extraction.json").write_text(json.dumps(source_manifest))
    (target / "extraction.json").write_text(json.dumps(target_manifest))
    stamps = {
        "schema_version": 3,
        "dataset_id": "old",
        "extraction_manifest_sha256": _sha256(source / "extraction.json"),
        "inference_manifest_sha256": None,
        "coordinate_space": {"width": 10, "height": 10},
        "operations": [
            {
                "layer_id": "hazard",
                "source": [1, 1],
                "target": [4, 4],
                "radius_px": 1,
                "source_mode": "observed",
            },
            {
                "layer_id": "hazard",
                "source": [4, 4],
                "target": [7, 7],
                "radius_px": 1,
                "source_mode": "composite_at_operation_time",
            },
        ],
    }
    (source / "stamp-corrections.json").write_text(json.dumps(stamps))
    (source / "inference-selection.json").write_text(
        json.dumps({"schema_version": 1, "enabled": False})
    )
    (source / "enclosed-hole-fill-selection.json").write_text(
        json.dumps({"schema_version": 1, "enabled": True, "maximum_area_exclusive": 50})
    )
    return source, target, target_alignment


def test_migration_moves_observed_sources_but_not_targets_or_composite_sources(tmp_path):
    source, target, _ = _write_fixture(tmp_path)
    result = migrate_stamp_corrections(source, target)
    migrated = json.loads((target / "stamp-corrections.json").read_text())
    assert migrated["dataset_id"] == "new"
    assert migrated["extraction_manifest_sha256"] == _sha256(
        target / "extraction.json"
    )
    assert migrated["operations"][0]["source"] == [2.0, 3.0]
    assert migrated["operations"][0]["target"] == [4, 4]
    assert migrated["operations"][1]["source"] == [4, 4]
    assert migrated["operations"][1]["target"] == [7, 7]
    assert result["approval_carried_forward"] is False
    assert result["observed_source_operation_count"] == 1
    assert result["composite_source_operation_count"] == 1
    assert json.loads((target / "inference-selection.json").read_text())["enabled"] is False


def test_migration_rejects_alignment_without_reviewed_parent_hash(tmp_path):
    source, target, target_alignment = _write_fixture(tmp_path)
    alignment = json.loads(target_alignment.read_text())
    alignment["parent_alignment"]["sha256"] = "wrong"
    target_alignment.write_text(json.dumps(alignment))
    try:
        migrate_stamp_corrections(source, target)
    except ValueError as error:
        assert "hash-linked child" in str(error)
    else:
        raise AssertionError("Expected stale alignment parent to be rejected")


def test_migration_allows_legacy_run_without_optional_selection_files(tmp_path):
    source, target, _ = _write_fixture(tmp_path)
    (source / "inference-selection.json").unlink()
    (source / "enclosed-hole-fill-selection.json").unlink()

    result = migrate_stamp_corrections(source, target)

    assert (target / "stamp-corrections.json").exists()
    assert not (target / "inference-selection.json").exists()
    assert not (target / "enclosed-hole-fill-selection.json").exists()
    assert set(result["outputs"]) == {"stamp_corrections"}


def test_migration_allows_a_hash_identical_validated_noop_alignment(tmp_path):
    source, target, _ = _write_fixture(tmp_path)
    source_manifest = json.loads((source / "extraction.json").read_text())
    target_manifest_path = target / "extraction.json"
    target_manifest = json.loads(target_manifest_path.read_text())
    target_manifest["alignment"]["path"] = source_manifest["alignment"]["path"]
    target_manifest_path.write_text(json.dumps(target_manifest))

    result = migrate_stamp_corrections(source, target)
    migrated = json.loads((target / "stamp-corrections.json").read_text())

    assert result["target_alignment"]["alignment_chain_length"] == 0
    assert result["strategy"]["incremental_mappings"] == []
    assert migrated["operations"][0]["source"] == [1.0, 1.0]
    assert migrated["operations"][1]["source"] == [4, 4]


def test_migration_traverses_projective_and_local_alignment_children(tmp_path):
    source, target, projective_alignment = _write_fixture(tmp_path)
    grid = {
        "crs": "EPSG:3857",
        "bounds": [0.0, 0.0, 10.0, 10.0],
        "width": 10,
        "height": 10,
    }
    local_alignment = tmp_path / "local-alignment.json"
    local_operation = {
        "type": "lower_colorado_smoothstep_x",
        "grid": grid,
        "start_x_px": 0.0,
        "ramp_width_px": 1.0,
        "start_y_px": 0.0,
        "ramp_height_px": 1.0,
        "source_to_target_amplitude_x_px": 2.0,
        "target_to_parent_sampling_amplitude_x_px": -2.0,
    }
    local_alignment.write_text(
        json.dumps(
            {
                "alignment_mode": "assisted",
                "reference": {"crs": "EPSG:3310"},
                "transform_model": "projective_homography+child+local",
                "reference_to_source_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "parent_alignment": {
                    "path": str(projective_alignment),
                    "sha256": _sha256(projective_alignment),
                },
                "web_mercator_correction": {
                    "grid": grid,
                    "current_to_target_pixel_matrix": [
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                    ],
                },
                "automatic_lower_colorado_refinement": {
                    "operation": local_operation
                },
            }
        )
    )
    manifest_path = target / "extraction.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["alignment"]["path"] = str(local_alignment)
    manifest_path.write_text(json.dumps(manifest))

    result = migrate_stamp_corrections(source, target)
    migrated = json.loads((target / "stamp-corrections.json").read_text())

    assert migrated["operations"][0]["source"] == [4.0, 3.0]
    assert migrated["operations"][0]["target"] == [4, 4]
    assert migrated["operations"][1]["source"] == [4, 4]
    assert result["target_alignment"]["alignment_chain_length"] == 2
    assert [item["type"] for item in result["strategy"]["incremental_mappings"]] == [
        "projective_parent_to_child",
        "inverse_lower_colorado_smoothstep_x",
    ]


def test_alignment_application_audit_accepts_hash_identical_validated_noop(
    tmp_path, monkeypatch
):
    source = tmp_path / "source"
    target = tmp_path / "target"
    (source / "hazard").mkdir(parents=True)
    (target / "hazard").mkdir(parents=True)
    alignment = tmp_path / "alignment.json"
    alignment.write_text(
        json.dumps(
            {
                "alignment_mode": "assisted",
                "reference": {"crs": "EPSG:3310"},
                "transform_model": "projective_homography",
                "reference_to_source_matrix": [
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                ],
            }
        )
    )
    grid = {
        "crs": "EPSG:3857",
        "bounds": [0.0, 0.0, 2.0, 2.0],
        "width": 2,
        "height": 2,
    }
    manifest = {
        "source": {"width": 2, "height": 2},
        "alignment": {"path": str(alignment)},
        "layers": [{"id": "hazard", "warp": grid}],
    }
    (source / "extraction.json").write_text(json.dumps(manifest))
    (target / "extraction.json").write_text(json.dumps(manifest))
    (target / "plan.snapshot.json").write_text(
        json.dumps({"reference": str(tmp_path / "reference")})
    )
    values = np.asarray([[1, 2], [3, 4]], dtype=np.uint8)
    for run in (source, target):
        Image.fromarray(values).save(run / "hazard" / "web-mercator-class-id.png")
    Image.fromarray(values).save(target / "hazard" / "source-class-id.png")

    monkeypatch.setattr(correction_migration, "load_california", lambda _: (object(), None))
    monkeypatch.setattr(
        correction_migration,
        "warp_classified_to_web_mercator",
        lambda classified, *_: (classified.copy(), grid),
    )

    result = audit_alignment_application(source, target, tmp_path / "audit")

    assert result["status"] == "pass"
    assert result["application_mode"] == "validated_noop"
    assert result["target_recompute_mismatch_pixel_count"] == 0
    assert result["target_vs_parent_recompute_difference_pixel_count"] == 0
    assert result["target_alignment"]["current_to_target_pixel_matrix"] == [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
