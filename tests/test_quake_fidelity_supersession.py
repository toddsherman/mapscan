import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_hash_bound(pointer_path: Path, record: dict) -> Path:
    artifact = (pointer_path.parent / record["path"]).resolve()
    assert artifact.is_file()
    assert _sha256(artifact) == record["sha256"]
    return artifact


def test_quake_fidelity_supersession_binds_alignment_and_extraction():
    pointer_path = (
        PROJECT_ROOT
        / "runs/mapbox-autonomous-restart-v1/quake/fidelity-supersession.json"
    )
    pointer = json.loads(pointer_path.read_text())

    assert pointer["schema_version"] == "mapscan.fidelity-supersession.v1"
    assert pointer["status"] == "active"
    assert pointer["native_fidelity"]["decision"] == "pass"
    assert pointer["authority"] == {
        "manual_inputs": False,
        "human_approval": False,
        "historical_artifacts_mutated": False,
        "public_artifacts_mutated": False,
    }

    old_extraction_path = _assert_hash_bound(pointer_path, pointer["supersedes"])
    new_extraction_path = _assert_hash_bound(pointer_path, pointer["replacement"])
    old_alignment_path = _assert_hash_bound(
        pointer_path, pointer["alignment_supersession"]["supersedes"]
    )
    new_alignment_path = _assert_hash_bound(
        pointer_path, pointer["alignment_supersession"]["replacement"]
    )
    base_report_path = _assert_hash_bound(
        pointer_path, pointer["native_fidelity"]["base_report"]
    )
    _assert_hash_bound(pointer_path, pointer["native_fidelity"]["report"])
    _assert_hash_bound(pointer_path, pointer["native_fidelity"]["extraction_report"])

    old_extraction = json.loads(old_extraction_path.read_text())
    new_extraction = json.loads(new_extraction_path.read_text())
    old_alignment = json.loads(old_alignment_path.read_text())
    new_alignment = json.loads(new_alignment_path.read_text())
    base_report = json.loads(base_report_path.read_text())

    assert old_extraction["status"] == "accepted"
    assert old_alignment["decision"] == "accept"
    assert base_report["status"] == "retry_candidate_isolated"
    assert new_alignment["status"] == "pass"
    assert new_extraction["status"] == "accepted"
    assert new_extraction["automatic_iteration_count"] == 2
    assert new_extraction["authority"]["pristine_source_used"] is True
    assert new_extraction["authority"]["prior_class_raster_used"] is False
    assert all(
        gate if isinstance(gate, bool) else gate["passed"]
        for gate in new_extraction["gates"].values()
    )
