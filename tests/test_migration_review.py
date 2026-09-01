import pytest

from mapscan.migration_review import _decision_document


def _payload():
    return {
        "candidate_materialization": {
            "path": "/candidate",
            "sha256": "new-hash",
            "source_diff": {"fixed_point_reached": True},
        },
        "approved_materialization": {"path": "/approved", "sha256": "old-hash"},
        "v4_alignment": {"sha256": "alignment-hash"},
        "hybrid_border": {"sha256": "hybrid-hash"},
        "boundary_clip": {"audit_sha256": "boundary-audit-hash"},
        "reference_overlay": {
            "manifest": {"sha256": "county-overlay-hash"},
            "source": {"sha256": "county-source-hash"},
        },
        "metrics": {"changed_pixel_count": 12},
    }


def test_migration_review_decision_binds_only_new_materialization():
    decision = _decision_document(
        _payload(),
        {
            "status": "approved",
            "statement": "alignment looks good",
            "inspection_confirmed": True,
        },
    )
    assert decision["materialization_sha256"] == "new-hash"
    assert decision["comparison_materialization_sha256"] == "old-hash"
    assert decision["alignment_sha256"] == "alignment-hash"
    assert decision["hybrid_border_sha256"] == "hybrid-hash"
    assert decision["boundary_clip_audit_sha256"] == "boundary-audit-hash"
    assert decision["source_diff"] == {"fixed_point_reached": True}
    assert decision["reference_overlay_manifest_sha256"] == "county-overlay-hash"
    assert decision["reference_source_sha256"] == "county-source-hash"
    assert decision["approval_carried_forward"] is False


def test_migration_review_requires_explicit_inspection_confirmation():
    with pytest.raises(ValueError, match="Confirm"):
        _decision_document(
            _payload(),
            {
                "status": "approved",
                "statement": "alignment looks good",
                "inspection_confirmed": False,
            },
        )
