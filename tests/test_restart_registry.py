import json

import pytest

from mapscan.experiment_log import NoHumanExperimentLog, automatic_provenance
from mapscan.restart_registry import NoHumanRestartRegistry, SCHEMA_VERSION


REFERENCE = {
    "id": "mapbox-california-reference-v1",
    "provider": "mapbox",
    "state_sha256": "a" * 64,
    "county_sha256": "b" * 64,
    "water_sha256": "c" * 64,
}


def _write_manifest(tmp_path, maps, **extra):
    path = tmp_path / "restart.json"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "mapbox_reference": REFERENCE,
        "maps": maps,
        **extra,
    }
    path.write_text(json.dumps(payload))
    return path


def _source(tmp_path, name, content=b"image"):
    path = tmp_path / name
    path.write_bytes(content)
    return path


def _automatic(*inputs):
    return automatic_provenance("restart-test", inputs)


def test_registry_initializes_logs_snapshot_and_index(tmp_path):
    _source(tmp_path, "forest.png", b"forest")
    _source(tmp_path, "farmsv2.png", b"farms-v2")
    manifest = _write_manifest(
        tmp_path,
        [
            {"id": "forest", "title": "Forest cover", "source": "forest.png"},
            {"id": "farms-v2", "source": "farmsv2.png"},
        ],
    )
    registry = NoHumanRestartRegistry(manifest, tmp_path / "runs" / "restart-v1")

    result = registry.initialize()

    assert result["map_count"] == 2
    assert (tmp_path / "runs/restart-v1/restart-manifest.snapshot.json").is_file()
    assert (tmp_path / "runs/restart-v1/forest/EXPERIMENT.md").is_file()
    forest = json.loads(
        (tmp_path / "runs/restart-v1/forest/EXPERIMENT.json").read_text()
    )
    assert forest["source"]["sha256"]
    assert forest["source"]["source_type"] == "image/png"
    assert forest["mapbox_reference"] == REFERENCE
    index = (tmp_path / "runs/restart-v1/INDEX.md").read_text()
    assert "Forest cover" in index
    assert "farms-v2" in index
    assert "| 0 | — | 0 | — | in_progress |" in index


def test_registry_resumes_logs_and_refreshes_automatic_counts(tmp_path):
    _source(tmp_path, "forest.png")
    manifest = _write_manifest(
        tmp_path, [{"id": "forest", "source": "forest.png"}]
    )
    run_root = tmp_path / "restart"
    registry = NoHumanRestartRegistry(manifest, run_root)
    registry.initialize()
    log = NoHumanExperimentLog.load(run_root / "forest/EXPERIMENT.json")
    log.record_alignment_iteration(
        scores={"p90_px": 4.0},
        gates={"holdouts": False},
        decision="retry",
        provenance=_automatic("mapbox_state", "source_pixels"),
        method="first warp",
    )
    log.record_alignment_iteration(
        scores={"p90_px": 1.0},
        gates={"holdouts": True},
        decision="accept",
        provenance=_automatic("mapbox_state", "mapbox_counties", "source_pixels"),
        method="second warp",
    )
    log.record_extraction_iteration(
        scores={"removed": 0},
        gates={"source_diff": True},
        decision="accept",
        provenance=_automatic("aligned_source", "legend", "source_diff"),
        method="classification",
    )
    log.finalize("complete")
    log.write(run_root / "forest/EXPERIMENT.md")

    resumed = NoHumanRestartRegistry(manifest, run_root)
    resumed.initialize()

    index = (run_root / "INDEX.md").read_text()
    assert "| 2 | 2 | 1 | 1 | complete |" in index
    assert resumed.logs["forest"].data["final"]["status"] == "complete"


@pytest.mark.parametrize("ids", [["forest", "forest"], ["forest", "FOREST"]])
def test_registry_rejects_duplicate_ids_even_on_case_insensitive_filesystems(
    tmp_path, ids
):
    _source(tmp_path, "one.png")
    _source(tmp_path, "two.png")
    manifest = _write_manifest(
        tmp_path,
        [
            {"id": ids[0], "source": "one.png"},
            {"id": ids[1], "source": "two.png"},
        ],
    )

    with pytest.raises(ValueError, match="duplicate map id"):
        NoHumanRestartRegistry(manifest, tmp_path / "run")


def test_registry_rejects_missing_source(tmp_path):
    manifest = _write_manifest(
        tmp_path, [{"id": "missing", "source": "does-not-exist.png"}]
    )

    with pytest.raises(FileNotFoundError, match="missing source"):
        NoHumanRestartRegistry(manifest, tmp_path / "run")


@pytest.mark.parametrize("name", ["county.png", "farms.png", "COUNTY.PNG"])
def test_registry_rejects_deleted_deprecated_sources(tmp_path, name):
    _source(tmp_path, name)
    manifest = _write_manifest(tmp_path, [{"id": "legacy", "source": name}])

    with pytest.raises(ValueError, match="deprecated restart input"):
        NoHumanRestartRegistry(manifest, tmp_path / "run")


@pytest.mark.parametrize(
    "legacy_field",
    [
        "manual_arrow_path",
        "stamp_audit",
        "alignment_path",
        "materialization_path",
        "human_approval",
    ],
)
def test_registry_rejects_legacy_manual_and_existing_result_fields(
    tmp_path, legacy_field
):
    _source(tmp_path, "forest.png")
    manifest = _write_manifest(
        tmp_path,
        [
            {
                "id": "forest",
                "source": "forest.png",
                legacy_field: "legacy/result.json",
            }
        ],
    )

    with pytest.raises(ValueError, match="legacy/manual field"):
        NoHumanRestartRegistry(manifest, tmp_path / "run")


def test_registry_requires_a_content_pinned_mapbox_reference(tmp_path):
    _source(tmp_path, "forest.png")
    manifest = tmp_path / "restart.json"
    manifest.write_text(
        json.dumps(
            {
                "mapbox_reference": {"id": "mapbox-current"},
                "maps": [{"id": "forest", "source": "forest.png"}],
            }
        )
    )

    with pytest.raises(ValueError, match="requires at least one SHA-256"):
        NoHumanRestartRegistry(manifest, tmp_path / "run")


def test_registry_refuses_resume_after_source_changes(tmp_path):
    source = _source(tmp_path, "forest.png", b"before")
    manifest = _write_manifest(
        tmp_path, [{"id": "forest", "source": "forest.png"}]
    )
    run_root = tmp_path / "run"
    NoHumanRestartRegistry(manifest, run_root).initialize()
    source.write_bytes(b"after")

    with pytest.raises(ValueError, match="source hashes changed"):
        NoHumanRestartRegistry(manifest, run_root).initialize()
