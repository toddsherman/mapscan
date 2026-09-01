import json

from mapscan.fine_alignment_determinism import (
    CORE_ARTIFACTS,
    audit_fine_alignment_determinism,
)


def _run(path, suffix=b""):
    path.mkdir()
    fit = {
        "selected_model": "affine",
        "candidates": [
            {
                "model": "affine",
                "matrix_current_to_target_pixels": [[1, 0, 2], [0, 1, 3], [0, 0, 1]],
            }
        ],
    }
    (path / "correction-fit.json").write_text(json.dumps(fit))
    report = {
        "source": {"sha256": "source"},
        "parent_alignment": {"sha256": "parent"},
        "county_reference": {"source_sha256": "county"},
        "independent_spatial_holdouts": {"passed": True},
        "after": {"shift_residual": {"p90_px": 1}},
        "state_boundary_veto": {"passed": True},
        "transform_regularity": {"passed": True},
    }
    (path / "county-fine-alignment.json").write_text(json.dumps(report))
    for name in CORE_ARTIFACTS:
        (path / name).write_bytes(name.encode() + suffix)


def test_repeat_audit_passes_identical_core_evidence(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    _run(first)
    _run(second)

    report = audit_fine_alignment_determinism(
        first, second, first / "determinism-audit.json"
    )

    assert report["passed"] is True
    assert report["matrix_identical"] is True


def test_repeat_audit_fails_changed_pixels(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    _run(first)
    _run(second)
    (second / CORE_ARTIFACTS[-1]).write_bytes(b"changed")

    report = audit_fine_alignment_determinism(
        first, second, first / "determinism-audit.json"
    )

    assert report["passed"] is False
    assert report["artifact_hashes"][CORE_ARTIFACTS[-1]]["identical"] is False
