from pathlib import Path

from mapscan.live_alignment_nonregression import audit_live_alignment_nonregression


def test_checked_in_live_registry_passes_alignment_nonregression(tmp_path: Path):
    repository_root = Path(__file__).parents[1]
    report = audit_live_alignment_nonregression(
        repository_root / "config" / "live-dataset-authority-v2.json",
        tmp_path / "report.json",
    )

    assert report["status"] == "pass"
    assert report["failure_count"] == 0
    assert report["dataset_count"] == 11
    assert report["failed_public_directories"] == []
