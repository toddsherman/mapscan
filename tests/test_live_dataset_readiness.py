from pathlib import Path

from mapscan.live_dataset_readiness import audit_live_dataset_readiness


def test_checked_in_live_corpus_passes_readiness_audit(tmp_path: Path) -> None:
    root = Path(__file__).parents[1]
    report = audit_live_dataset_readiness(
        root / "config/live-dataset-authority-v2.json",
        root / "viewer/public/data/catalog.json",
        root / "viewer/public/data/datasets",
        root / "runs/corpus-readiness-audit-v1/live-alignment-nonregression.json",
        tmp_path / "readiness.json",
    )

    assert report["status"] == "pass"
    assert report["dataset_count"] == 11
    assert report["passing_dataset_count"] == 11
    assert report["failure_count"] == 0
    assert report["failed_public_directories"] == []
    assert all(dataset["passed_determinism_gates"] for dataset in report["datasets"])
    assert all(
        dataset["colored_pixel_count_outside_state"] == 0
        for dataset in report["datasets"]
    )
