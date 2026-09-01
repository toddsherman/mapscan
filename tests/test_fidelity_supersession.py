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


def test_population_fidelity_supersession_is_hash_bound_and_passing():
    pointer_path = (
        PROJECT_ROOT
        / "runs/mapbox-autonomous-restart-v1/population/fidelity-supersession.json"
    )
    pointer = json.loads(pointer_path.read_text())

    assert pointer["status"] == "active"
    assert pointer["authority"] == {
        "manual_inputs": False,
        "human_approval": False,
        "historical_artifacts_mutated": False,
    }
    assert pointer["native_fidelity"]["decision"] == "pass"

    for record in (
        pointer["supersedes"],
        pointer["replacement"],
        pointer["native_fidelity"]["report"],
        pointer["native_fidelity"]["repair_report"],
    ):
        artifact = (pointer_path.parent / record["path"]).resolve()
        assert artifact.is_file()
        assert _sha256(artifact) == record["sha256"]

    replacement_path = (
        pointer_path.parent / pointer["replacement"]["path"]
    ).resolve()
    replacement = json.loads(replacement_path.read_text())
    assert replacement["status"] == "accepted"
    assert replacement["automatic_iteration_count"] == 4

    report_path = (
        pointer_path.parent / pointer["native_fidelity"]["report"]["path"]
    ).resolve()
    assert json.loads(report_path.read_text())["decision"] == "pass"
