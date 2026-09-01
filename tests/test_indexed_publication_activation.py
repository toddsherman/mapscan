from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from mapscan.indexed_publication_activation import activate_indexed_staging_dataset


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


def _write_package(root: Path) -> None:
    root.mkdir()
    source = root / "source.png"
    boundary = root / "mapbox-state-coast-overlay.png"
    Image.new("RGBA", (2, 2), (10, 20, 30, 255)).save(source)
    Image.new("RGBA", (2, 2), (0, 255, 0, 255)).save(boundary)

    tile_root = root / "tiles" / "classes" / "class-id"
    tile = tile_root / "4" / "2" / "3.png"
    tile.parent.mkdir(parents=True)
    Image.new("LA", (2, 2), (1, 255)).save(tile)
    tilejson = tile_root / "tilejson.json"
    tilejson.write_text(json.dumps({"tilejson": "3.0.0"}) + "\n")
    tile_set_sha256 = _aggregate([tile, tilejson], root)

    provenance = {
        "kind": "autonomous_test_staging_preview_provenance",
        "status": "needs_visual_review",
        "publication_approved": False,
        "approval": {"status": "not_approved"},
        "source": {
            "path": "/machine/private/source.png",
            "sha256": _sha256(source),
            "copied_path": "source.png",
        },
        "accepted_alignment": {
            "path": "/machine/private/accepted-alignment.json",
            "sha256": "a" * 64,
        },
        "class_raster": {
            "path": "extraction-04/web-mercator-class-id.png",
            "sha256": "c" * 64,
        },
        "publication_coverage": {
            "layers": [
                {
                    "layer_id": "classes",
                    "accepted_raster_sha256": "c" * 64,
                    "semantic_kind": "mutually_exclusive_categorical",
                    "coverage_contract": "sparse_visible_evidence",
                    "colored_pixel_count_outside_state": 0,
                    "nodata_pixel_count_inside_state": 7,
                }
            ]
        },
    }
    provenance_path = root / "autonomous-preview-provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")

    dataset = {
        "schema_version": 1,
        "status": "needs_visual_review",
        "id": "test-indexed-dataset",
        "title": "Test indexed dataset",
        "bounds": [-124.5, 32.5, -114.1, 42.0],
        "minimum_zoom": 4,
        "maximum_native_zoom": 4,
        "overscaling": "nearest",
        "categorical_tile_encoding": "indexed_class_id",
        "source_image": "source.png",
        "approval": {"status": "not_approved"},
        "boundary": {
            "kind": "pinned_mapbox_state_coast_diagnostic",
            "authority": "accepted_alignment_mapbox_reference",
            "diagnostic_only": True,
            "raster": boundary.name,
            "raster_sha256": _sha256(boundary),
            "raster_width": 2,
            "raster_height": 2,
            "raster_bounds": [-124.5, 32.5, -114.1, 42.0],
        },
        "provenance": {
            "manifest": provenance_path.name,
            "sha256": _sha256(provenance_path),
        },
        "layers": [
            {
                "id": "classes",
                "label": "Classes",
                "kind": "categorical",
                "bounds": [-124.5, 32.5, -114.1, 42.0],
                "nodata_class_id": 0,
                "categories": [
                    {
                        "id": "one",
                        "class_id": 1,
                        "label": "One",
                        "display_rgb": [10, 20, 30],
                        "pixel_count": 4,
                    }
                ],
                "indexed_raster": {
                    "encoding": "png_luma_alpha_uint8_class_id_v1",
                    "class_id_channel": "red_after_browser_png_decode",
                    "coverage_channel": "alpha",
                    "nodata_class_id": 0,
                    "class_id_range": [1, 1],
                    "raster_color_mix": [1, 0, 0, 0],
                    "raster_color_range": [0, 1],
                    "tile_template": (
                        "tiles/classes/class-id/{z}/{x}/{y}.png"
                        f"?v={tile_set_sha256[:16]}"
                    ),
                    "tilejson": "tiles/classes/class-id/tilejson.json",
                    "tile_file_count": 1,
                    "tile_image_byte_count": tile.stat().st_size,
                    "tile_set_sha256": tile_set_sha256,
                },
            }
        ],
    }
    (root / "dataset.json").write_text(json.dumps(dataset, indent=2) + "\n")


def test_activates_indexed_assets_and_sanitizes_public_provenance(tmp_path: Path):
    staging = tmp_path / "staging"
    public = tmp_path / "public"
    _write_package(staging)
    staging_dataset_sha256 = _sha256(staging / "dataset.json")
    staging_provenance_sha256 = _sha256(
        staging / "autonomous-preview-provenance.json"
    )

    decision = activate_indexed_staging_dataset(
        staging,
        public,
        author_statement="Proceed",
    )

    dataset = json.loads((public / "dataset.json").read_text())
    provenance = json.loads((public / "provenance.json").read_text())
    assert dataset["status"] == "approved_publication"
    assert dataset["approval"]["author_statement"] == "Proceed"
    assert provenance["kind"] == "autonomous_indexed_publication_provenance"
    assert "/machine/private" not in json.dumps(provenance)
    assert "extraction-04/web-mercator-class-id.png" not in json.dumps(provenance)
    assert provenance["class_raster"]["sha256"] == "c" * 64
    assert not (public / "autonomous-preview-provenance.json").exists()
    assert decision["staging_assets_copied_byte_identically"] is True
    assert decision["metadata_transition_only"] is True
    assert (public / "source.png").stat().st_ino == (staging / "source.png").stat().st_ino
    assert _sha256(staging / "dataset.json") == staging_dataset_sha256
    assert (
        _sha256(staging / "autonomous-preview-provenance.json")
        == staging_provenance_sha256
    )
    assert decision["coverage"][0]["nodata_pixel_count_inside_state"] == 7


def test_activates_with_hash_bound_autonomous_nonregression_evidence(
    tmp_path: Path,
):
    staging = tmp_path / "staging"
    public = tmp_path / "public"
    evidence_path = tmp_path / "nonregression.json"
    _write_package(staging)
    evidence_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "mapscan_autonomous_publication_nonregression_evidence",
                "status": "pass",
                "policy": "global best alignment and fixed-point extraction",
                "staging_dataset_sha256": _sha256(staging / "dataset.json"),
                "staging_provenance_sha256": _sha256(
                    staging / "autonomous-preview-provenance.json"
                ),
                "gates": {
                    "alignment_global_best": True,
                    "extraction_fixed_point": {"passed": True},
                },
            },
            indent=2,
        )
        + "\n"
    )

    decision = activate_indexed_staging_dataset(
        staging,
        public,
        author_statement="No-human publication policy requested for this workflow",
        autonomous_evidence_path=evidence_path,
    )

    dataset = json.loads((public / "dataset.json").read_text())
    assert dataset["approval"]["mode"] == "autonomous_nonregression_activation"
    assert dataset["approval"]["autonomous_evidence"]["sha256"] == _sha256(
        evidence_path
    )
    assert decision["approval_mode"] == "autonomous_nonregression_activation"


def test_rejects_a_changed_indexed_tile_before_creating_output(tmp_path: Path):
    staging = tmp_path / "staging"
    public = tmp_path / "public"
    _write_package(staging)
    tile = staging / "tiles" / "classes" / "class-id" / "4" / "2" / "3.png"
    tile.write_bytes(tile.read_bytes() + b"changed")

    with pytest.raises(ValueError, match="tile bytes differ"):
        activate_indexed_staging_dataset(
            staging,
            public,
            author_statement="Proceed",
        )
    assert not public.exists()


def _rewrite_provenance(root: Path, update) -> None:
    provenance_path = root / "autonomous-preview-provenance.json"
    provenance = json.loads(provenance_path.read_text())
    update(provenance)
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    dataset_path = root / "dataset.json"
    dataset = json.loads(dataset_path.read_text())
    dataset["provenance"]["sha256"] = _sha256(provenance_path)
    dataset_path.write_text(json.dumps(dataset, indent=2) + "\n")


def test_rejects_a_symlink_anywhere_in_the_staging_package(tmp_path: Path):
    staging = tmp_path / "staging"
    _write_package(staging)
    (staging / "linked-source.png").symlink_to(staging / "source.png")

    with pytest.raises(ValueError, match="must not contain symlinks"):
        activate_indexed_staging_dataset(
            staging, tmp_path / "public", author_statement="Proceed"
        )
    assert not (tmp_path / "public").exists()


def test_rejects_a_staging_directory_symlink(tmp_path: Path):
    staging = tmp_path / "staging"
    linked_staging = tmp_path / "linked-staging"
    _write_package(staging)
    linked_staging.symlink_to(staging, target_is_directory=True)

    with pytest.raises(ValueError, match="must not be a symlink"):
        activate_indexed_staging_dataset(
            linked_staging, tmp_path / "public", author_statement="Proceed"
        )
    assert not (tmp_path / "public").exists()


def test_rejects_dataset_path_traversal(tmp_path: Path):
    staging = tmp_path / "staging"
    _write_package(staging)
    dataset_path = staging / "dataset.json"
    dataset = json.loads(dataset_path.read_text())
    dataset["source_image"] = "../source.png"
    dataset_path.write_text(json.dumps(dataset, indent=2) + "\n")

    with pytest.raises(ValueError, match="path traversal"):
        activate_indexed_staging_dataset(
            staging, tmp_path / "public", author_statement="Proceed"
        )
    assert not (tmp_path / "public").exists()


def test_rejects_provenance_path_traversal_instead_of_hiding_it(tmp_path: Path):
    staging = tmp_path / "staging"
    _write_package(staging)
    _rewrite_provenance(
        staging,
        lambda provenance: provenance["source"].update(
            {"copied_path": "../private/source.png"}
        ),
    )

    with pytest.raises(ValueError, match="contains path traversal"):
        activate_indexed_staging_dataset(
            staging, tmp_path / "public", author_statement="Proceed"
        )
    assert not (tmp_path / "public").exists()


def test_rejects_missing_or_unsafe_per_layer_coverage(tmp_path: Path):
    staging = tmp_path / "staging"
    _write_package(staging)
    _rewrite_provenance(
        staging,
        lambda provenance: provenance["publication_coverage"]["layers"][0].update(
            {"colored_pixel_count_outside_state": 1}
        ),
    )

    with pytest.raises(ValueError, match="contains data outside the state"):
        activate_indexed_staging_dataset(
            staging, tmp_path / "public", author_statement="Proceed"
        )
    assert not (tmp_path / "public").exists()


def test_rejects_a_missing_per_layer_coverage_contract(tmp_path: Path):
    staging = tmp_path / "staging"
    _write_package(staging)
    _rewrite_provenance(
        staging,
        lambda provenance: provenance.pop("publication_coverage"),
    )

    with pytest.raises(ValueError, match="no publication coverage contract"):
        activate_indexed_staging_dataset(
            staging, tmp_path / "public", author_statement="Proceed"
        )
    assert not (tmp_path / "public").exists()


def test_rejects_coverage_not_bound_to_the_accepted_raster(tmp_path: Path):
    staging = tmp_path / "staging"
    _write_package(staging)
    _rewrite_provenance(
        staging,
        lambda provenance: provenance["publication_coverage"]["layers"][0].update(
            {"accepted_raster_sha256": "d" * 64}
        ),
    )

    with pytest.raises(ValueError, match="differs from accepted raster"):
        activate_indexed_staging_dataset(
            staging, tmp_path / "public", author_statement="Proceed"
        )
    assert not (tmp_path / "public").exists()


def test_rejects_a_source_not_bound_to_its_provenance_hash(tmp_path: Path):
    staging = tmp_path / "staging"
    _write_package(staging)
    _rewrite_provenance(
        staging,
        lambda provenance: provenance["source"].update({"sha256": "e" * 64}),
    )

    with pytest.raises(ValueError, match="retained source image differs"):
        activate_indexed_staging_dataset(
            staging, tmp_path / "public", author_statement="Proceed"
        )
    assert not (tmp_path / "public").exists()


def test_cleans_a_partially_created_output_after_any_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    staging = tmp_path / "staging"
    public = tmp_path / "public"
    _write_package(staging)

    def reject_public(*_args, **_kwargs):
        raise ValueError("post-copy verification failed")

    monkeypatch.setattr(
        "mapscan.indexed_publication_activation._verify_public", reject_public
    )
    with pytest.raises(ValueError, match="post-copy verification failed"):
        activate_indexed_staging_dataset(
            staging, public, author_statement="Proceed"
        )
    assert not public.exists()


def test_manifest_only_activation_reuses_verified_staging_assets(tmp_path: Path):
    staging = tmp_path / "staging" / "candidate-v1"
    public = tmp_path / "datasets" / "published"
    staging.parent.mkdir()
    _write_package(staging)
    original_tile = staging / "tiles/classes/class-id/4/2/3.png"

    decision = activate_indexed_staging_dataset(
        staging,
        public,
        author_statement="Proceed",
        public_id="published-indexed-dataset",
        public_title="Published indexed dataset",
        asset_base="/mapscan/data/staging/candidate-v1/",
    )

    assert sorted(path.name for path in public.iterdir()) == [
        "dataset.json",
        "provenance.json",
        "public-activation-decision.json",
    ]
    dataset = json.loads((public / "dataset.json").read_text())
    provenance = json.loads((public / "provenance.json").read_text())
    assert dataset["id"] == "published-indexed-dataset"
    assert dataset["title"] == "Published indexed dataset"
    assert dataset["asset_base"] == "/mapscan/data/staging/candidate-v1/"
    assert dataset["source_image"] == "source.png"
    assert provenance["staging"]["artifact_dataset_id"] == "test-indexed-dataset"
    assert provenance["staging"]["artifact_title"] == "Test indexed dataset"
    assert provenance["shared_assets"]["immutable_assets_sha256"] == decision[
        "immutable_assets_sha256"
    ]
    assert decision["staging_assets_copied_byte_identically"] is False
    assert decision["staging_assets_shared_byte_identically"] is True
    assert decision["asset_base"] == "/mapscan/data/staging/candidate-v1/"
    assert original_tile.is_file()


@pytest.mark.parametrize(
    "asset_base",
    [
        "",
        "mapscan/data/staging/candidate/",
        "//example.com/assets/",
        "/mapscan/data/../private/",
        "/mapscan/data/staging/candidate/?token=secret",
        "/mapscan/data/staging/candidate/#fragment",
        "https://example.com/assets/",
    ],
)
def test_rejects_unsafe_shared_asset_bases(tmp_path: Path, asset_base: str):
    staging = tmp_path / "staging"
    _write_package(staging)

    with pytest.raises(ValueError, match="Shared asset base"):
        activate_indexed_staging_dataset(
            staging,
            tmp_path / "public",
            author_statement="Proceed",
            asset_base=asset_base,
        )
    assert not (tmp_path / "public").exists()


def test_manifest_only_activation_still_rejects_tampered_assets(tmp_path: Path):
    staging = tmp_path / "staging"
    _write_package(staging)
    tile = staging / "tiles/classes/class-id/4/2/3.png"
    tile.write_bytes(tile.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="tile bytes differ"):
        activate_indexed_staging_dataset(
            staging,
            tmp_path / "public",
            author_statement="Proceed",
            asset_base="/mapscan/data/staging/candidate-v1/",
        )
    assert not (tmp_path / "public").exists()


def test_accepts_a_hash_bound_tile_directory_distinct_from_the_layer_id(tmp_path: Path):
    staging = tmp_path / "staging"
    _write_package(staging)
    original = staging / "tiles/classes/class-id"
    renamed = staging / "tiles/classes-v2-clipped/class-id"
    renamed.parent.mkdir(parents=True)
    original.rename(renamed)
    dataset_path = staging / "dataset.json"
    dataset = json.loads(dataset_path.read_text())
    indexed = dataset["layers"][0]["indexed_raster"]
    indexed["tilejson"] = "tiles/classes-v2-clipped/class-id/tilejson.json"
    tile = renamed / "4/2/3.png"
    tilejson = renamed / "tilejson.json"
    tile_hash = _aggregate([tile, tilejson], staging)
    indexed["tile_set_sha256"] = tile_hash
    indexed["tile_template"] = (
        "tiles/classes-v2-clipped/class-id/{z}/{x}/{y}.png"
        f"?v={tile_hash[:16]}"
    )
    dataset_path.write_text(json.dumps(dataset, indent=2) + "\n")

    result = activate_indexed_staging_dataset(
        staging,
        tmp_path / "public",
        author_statement="Proceed",
        asset_base="/mapscan/data/staging/test/",
    )

    assert result["staging_assets_shared_byte_identically"] is True
