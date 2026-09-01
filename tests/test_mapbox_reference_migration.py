from types import SimpleNamespace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from mapscan.automatic_alignment_loop import _normalizer
from mapscan.mapbox_reference_migration import (
    _accepted_transform_reference_normalizer,
    verify_non_counting_reference_migration_receipt,
)


def test_reference_migration_preserves_accepted_v1_normalizer():
    old_state = np.zeros((100, 120), dtype=bool)
    old_state[10:91, 20] = True
    old_state[10:91, 80] = True
    new_state = np.zeros_like(old_state)
    new_state[20:81, 30] = True
    new_state[20:81, 70] = True
    grid = {
        "crs": "EPSG:3857",
        "bounds": [0.0, 0.0, 120.0, 100.0],
        "width": 120,
        "height": 100,
    }
    old_reference = SimpleNamespace(grid=grid, state_coast=old_state)
    new_reference = SimpleNamespace(grid=grid, state_coast=new_state)

    center, height = _accepted_transform_reference_normalizer(
        old_reference, new_reference
    )
    old_center, old_height, _, _ = _normalizer(old_state)
    new_center, new_height, _, _ = _normalizer(new_state)

    assert np.array_equal(center, old_center)
    assert height == old_height
    assert not np.array_equal(center, new_center) or height != new_height


def test_historical_alignment_requires_exact_hashed_passing_migration_receipt(
    tmp_path: Path,
):
    alignment = tmp_path / "accepted-alignment.json"
    alignment.write_text('{"decision":"accept"}\n')
    alignment_hash = hashlib.sha256(alignment.read_bytes()).hexdigest()
    old_pin = {
        "id": "mapbox/light-v11@z9",
        "style_sha256": "a" * 64,
        "tilejson_sha256": "b" * 64,
        "tile_aggregate_sha256": "c" * 64,
        "manifest_sha256": "d" * 64,
    }
    new_pin = {**old_pin, "manifest_sha256": "e" * 64}
    gates = {"state": {"passed": True}, "counties": {"passed": True}}
    audit = {
        "kind": "non_counting_mapbox_reference_migration_audit",
        "status": "pass",
        "resume_count": 0,
        "authority": {
            "counts_as_alignment_iteration": False,
            "changes_existing_iteration_counts": False,
            "manual_input_used": False,
            "old_transform_reoptimized": False,
            "strict_current_semantic_gates_used": True,
        },
        "old_reference": old_pin,
        "new_reference": new_pin,
        "records": [
            {
                "map_id": "sample",
                "original_accepted_automatic_iteration_count": 7,
                "counts_as_alignment_iteration": False,
                "accepted_alignment_path": str(alignment),
                "accepted_alignment_sha256": alignment_hash,
                "old_reference": old_pin,
                "new_reference": new_pin,
                "new_semantic_gates": gates,
                "failed_new_gates": [],
                "status": "retain_original_acceptance",
            }
        ],
    }
    audit_path = tmp_path / "reference-migration-audit.json"
    audit_path.write_text(json.dumps(audit))
    audit_hash = hashlib.sha256(audit_path.read_bytes()).hexdigest()
    revision = {
        "producer": "mapscan.mapbox_reference_migration",
        "previous_reference": {"manifest_sha256": old_pin["manifest_sha256"]},
        "current_reference": {"manifest_sha256": new_pin["manifest_sha256"]},
        "raw_mapbox_bytes_preserved_exactly": True,
        "raw_hashes": {
            "style_sha256": old_pin["style_sha256"],
            "tilejson_sha256": old_pin["tilejson_sha256"],
            "tiles_sha256": old_pin["tile_aggregate_sha256"],
        },
        "source_manifest_sha256": old_pin["manifest_sha256"],
        "automatic_iteration_count_changed": False,
        "non_counting_audit": {"path": str(audit_path), "sha256": audit_hash},
    }

    receipt = verify_non_counting_reference_migration_receipt(
        map_id="sample",
        alignment_path=alignment,
        accepted_alignment_reference=old_pin,
        current_reference=new_pin,
        accepted_automatic_iteration_count=7,
        reference_revisions=[revision],
    )
    assert receipt["record"]["status"] == "retain_original_acceptance"

    alignment.write_text('{"decision":"accept","tampered":true}\n')
    with pytest.raises(ValueError, match="one exact passing"):
        verify_non_counting_reference_migration_receipt(
            map_id="sample",
            alignment_path=alignment,
            accepted_alignment_reference=old_pin,
            current_reference=new_pin,
            accepted_automatic_iteration_count=7,
            reference_revisions=[revision],
        )
