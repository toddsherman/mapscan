import json
from pathlib import Path

import pytest

from mapscan.fine_alignment_review import build_fine_alignment_review_payload


def test_fine_alignment_review_payload_requires_passing_determinism(tmp_path: Path):
    report = {
        "status": "needs_author_review",
        "fixed_point_gate": {"passed": True},
        "independent_spatial_holdouts": {"passed": True},
        "transform_regularity": {"passed": True},
        "state_boundary_veto": {"passed": True},
    }
    (tmp_path / "county-fine-alignment.json").write_text(json.dumps(report))
    (tmp_path / "determinism-audit.json").write_text(json.dumps({"passed": False}))
    with pytest.raises(ValueError, match="determinism"):
        build_fine_alignment_review_payload(tmp_path)
