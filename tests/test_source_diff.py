import numpy as np
from PIL import Image

from mapscan.source_diff import (
    FULL_STATE,
    SPARSE_EVIDENCE,
    _audit_layer_arrays,
    _promote_final_extraction_artifacts,
    _sha256,
)


def test_full_state_diff_repairs_every_internal_gap_without_crossing_clip():
    candidate = np.zeros((5, 7), dtype=np.uint8)
    candidate[2, 1] = 1
    candidate[2, 5] = 2
    state = np.zeros(candidate.shape, dtype=bool)
    state[1:4, 1:6] = True
    evidence = candidate > 0

    repaired, report, masks = _audit_layer_arrays(
        candidate, state, evidence, FULL_STATE, True
    )

    assert report["status"] == "pass"
    assert report["unclassified_pixel_count_before"] == 13
    assert report["completed_pixel_count"] == 13
    assert report["unclassified_pixel_count_after"] == 0
    assert np.all(repaired[state] > 0)
    assert not np.any(repaired[~state])
    assert np.count_nonzero(masks["completion"]) == 13


def test_sparse_diff_preserves_background_but_fails_dropped_source_evidence():
    candidate = np.zeros((5, 7), dtype=np.uint8)
    candidate[2, 1] = 1
    state = np.ones(candidate.shape, dtype=bool)
    evidence = np.zeros(candidate.shape, dtype=bool)
    evidence[2, 1] = True
    evidence[2, 5] = True

    repaired, report, masks = _audit_layer_arrays(
        candidate, state, evidence, SPARSE_EVIDENCE, True
    )

    assert report["status"] == "fail"
    assert report["completed_pixel_count"] == 0
    assert report["dropped_visible_source_evidence_pixel_count"] == 1
    assert np.array_equal(repaired, candidate)
    assert masks["dropped_visible_source_evidence"][2, 5]


def test_fixed_point_report_promotes_its_declared_artifacts(tmp_path):
    iteration = tmp_path / "iteration-03"
    case = tmp_path / "case"
    artifact = iteration / "hazard" / "audited.png"
    artifact.parent.mkdir(parents=True)
    Image.fromarray(np.asarray([[1, 2]], dtype=np.uint8)).save(artifact)
    report = {
        "layers": [
            {
                "artifacts": {
                    "audited_class_id": {
                        "path": "hazard/audited.png",
                        "sha256": _sha256(artifact),
                    }
                }
            }
        ]
    }

    _promote_final_extraction_artifacts(report, iteration, case)

    promoted = case / "hazard" / "audited.png"
    assert promoted.is_file()
    assert _sha256(promoted) == _sha256(artifact)
