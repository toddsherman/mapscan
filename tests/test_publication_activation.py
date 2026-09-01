import hashlib
import json
from pathlib import Path

from PIL import Image

from mapscan.publication_activation import activate_staging_dataset


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _aggregate(paths: list[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(_sha256(path).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def _fixture(tmp_path: Path) -> Path:
    staging = tmp_path / "staging"
    tile = staging / "tiles/zones/one/4/2/3.png"
    tile.parent.mkdir(parents=True)
    Image.new("RGBA", (2, 2), (255, 255, 255, 255)).save(tile)
    tilejson = staging / "tiles/zones/one/tilejson.json"
    _write_json(tilejson, {"tilejson": "3.0.0", "tiles": ["{z}/{x}/{y}.png"]})
    tile_hash = _aggregate([tile, tilejson], staging)

    boundary = staging / "canonical-boundary.png"
    Image.new("RGBA", (3, 4), (0, 255, 0, 255)).save(boundary)
    geojson = staging / "boundary.geojson"
    _write_json(
        geojson,
        {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {},
                    "geometry": {"type": "LineString", "coordinates": [[0, 0], [1, 1]]},
                }
            ],
        },
    )
    (staging / "source.png").write_bytes(b"source")
    boundary_contract = {
        "canonical_boundary_id": "canonical-v1",
        "canonical_display_border_sha256": _sha256(boundary),
        "raster": boundary.name,
        "raster_sha256": _sha256(boundary),
        "raster_width": 3,
        "raster_height": 4,
        "geojson": geojson.name,
        "geojson_sha256": _sha256(geojson),
        "geojson_feature_count": 1,
        "continuous_border_component_count": 1,
        "expected_boundary_component_count": 1,
        "components": [{"id": "mainland"}],
        "colored_pixel_count_outside_boundary": 0,
        "unclassified_pixel_count_inside_boundary": 0,
        "coverage_contract": "full_state",
    }
    provenance = {
        "kind": "approved_publication_provenance",
        "dataset_id": "fixture",
        "materialization_sha256": "materialized",
        "approval": {"status": "approved", "decision_sha256": "approved"},
        "boundary": boundary_contract,
    }
    _write_json(staging / "provenance.json", provenance)
    dataset = {
        "status": "approved_publication",
        "id": "fixture",
        "source_image": "source.png",
        "materialization": {"sha256": "materialized"},
        "approval": {"status": "approved", "sha256": "approved"},
        "provenance": {"sha256": _sha256(staging / "provenance.json")},
        "boundary": boundary_contract,
        "layers": [
            {
                "id": "zones",
                "categories": [
                    {
                        "id": "one",
                        "tilejson": "tiles/zones/one/tilejson.json",
                        "tile_template": f"tiles/zones/one/{{z}}/{{x}}/{{y}}.png?v={tile_hash[:16]}",
                        "tile_file_count": 1,
                        "tile_set_sha256": tile_hash,
                    }
                ],
            }
        ],
    }
    _write_json(staging / "dataset.json", dataset)
    return staging


def test_activates_byte_identical_staging_assets_with_new_decision(tmp_path):
    staging = _fixture(tmp_path)
    output = tmp_path / "public/fixture"
    result = activate_staging_dataset(
        staging, output, author_statement="lgtm"
    )
    decision = json.loads((output / "public-activation-decision.json").read_text())
    assert result["status"] == "approved_public_activation"
    assert decision["author_statement"] == "lgtm"
    assert decision["staging_assets_copied_byte_identically"] is True
    assert decision["boundary"]["colored_pixel_count_outside_boundary"] == 0
    for source in staging.rglob("*"):
        if source.is_file():
            target = output / source.relative_to(staging)
            assert _sha256(source) == _sha256(target)


def test_refuses_tampered_staging_tile(tmp_path):
    staging = _fixture(tmp_path)
    tile = next((staging / "tiles").rglob("*.png"))
    tile.write_bytes(b"tampered")
    try:
        activate_staging_dataset(
            staging, tmp_path / "public/fixture", author_statement="lgtm"
        )
    except ValueError as error:
        assert "Tile-set hash differs" in str(error)
    else:
        raise AssertionError("Tampered tile should block public activation")


def test_activates_exact_sparse_visible_evidence_contract(tmp_path):
    staging = _fixture(tmp_path)
    dataset_path = staging / "dataset.json"
    provenance_path = staging / "provenance.json"
    dataset = json.loads(dataset_path.read_text())
    provenance = json.loads(provenance_path.read_text())
    for manifest in (dataset, provenance):
        manifest["boundary"]["coverage_contract"] = "sparse_visible_evidence"
        manifest["boundary"]["unclassified_pixel_count_inside_boundary"] = 7
    _write_json(provenance_path, provenance)
    dataset["provenance"]["sha256"] = _sha256(provenance_path)
    _write_json(dataset_path, dataset)

    output = tmp_path / "public/fixture"
    result = activate_staging_dataset(staging, output, author_statement="lgtm")

    assert result["boundary"]["coverage_contract"] == "sparse_visible_evidence"
    assert result["boundary"]["unclassified_pixel_count_inside_boundary"] == 7


def test_refuses_full_state_package_with_interior_nodata(tmp_path):
    staging = _fixture(tmp_path)
    dataset_path = staging / "dataset.json"
    provenance_path = staging / "provenance.json"
    dataset = json.loads(dataset_path.read_text())
    provenance = json.loads(provenance_path.read_text())
    for manifest in (dataset, provenance):
        manifest["boundary"]["unclassified_pixel_count_inside_boundary"] = 1
    _write_json(provenance_path, provenance)
    dataset["provenance"]["sha256"] = _sha256(provenance_path)
    _write_json(dataset_path, dataset)

    try:
        activate_staging_dataset(
            staging, tmp_path / "public/fixture", author_statement="lgtm"
        )
    except ValueError as error:
        assert "Full-state staging data contains NoData" in str(error)
    else:
        raise AssertionError("Interior NoData should block full-state activation")
