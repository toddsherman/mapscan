import hashlib
import json

import numpy as np
from PIL import Image

from mapscan.tile_export import (
    WEB_MERCATOR_HALF_WORLD,
    _sample_class_overview,
    estimate_xyz_tile_file_count,
    export_categorical_tiles,
)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_indexed_export_fixture(tmp_path, category_count=2):
    run = tmp_path / "runs" / "indexed-sample"
    materialized = run / "materialized-v1"
    layer_dir = materialized / "hazard"
    layer_dir.mkdir(parents=True)
    values = np.array(
        [[1, 1, category_count, category_count], [1, 1, category_count, category_count]],
        dtype=np.uint8,
    )
    class_path = layer_dir / "web-mercator-class-id-final.png"
    Image.fromarray(values).save(class_path)
    categories = [
        {
            "id": f"class-{class_id}",
            "label": f"Class {class_id}",
            "legend_rgb": [class_id, class_id, class_id],
        }
        for class_id in range(1, category_count + 1)
    ]
    (run / "plan.snapshot.json").write_text(
        json.dumps(
            {
                "dataset_id": "indexed-sample",
                "title": "Indexed sample",
                "layers": [{"id": "hazard", "categories": categories}],
            }
        )
    )
    bounds = [-WEB_MERCATOR_HALF_WORLD, 0, 0, WEB_MERCATOR_HALF_WORLD]
    (run / "extraction.json").write_text(
        json.dumps(
            {
                "layers": [
                    {"id": "hazard", "warp": {"crs": "EPSG:3857", "bounds": bounds}}
                ]
            }
        )
    )
    manifest = {
        "dataset_id": "indexed-sample",
        "source_run": str(run),
        "layers": [
            {
                "layer_id": "hazard",
                "final_pixels_by_class_id": {
                    "1": 4,
                    str(category_count): 4,
                },
                "artifacts": {
                    "class_id": {
                        "path": "hazard/web-mercator-class-id-final.png",
                        "sha256": _sha256(class_path),
                    }
                },
            }
        ],
    }
    manifest_path = materialized / "materialization.json"
    manifest_path.write_text(json.dumps(manifest))
    (materialized / "materialization-review-decision.json").write_text(
        json.dumps(
            {
                "status": "approved",
                "materialization_sha256": _sha256(manifest_path),
            }
        )
    )
    return materialized


def test_exports_one_indexed_class_id_tile_for_many_categories(tmp_path):
    materialized = _write_indexed_export_fixture(tmp_path, category_count=53)
    output = tmp_path / "indexed-published"

    result = export_categorical_tiles(
        materialized,
        output,
        minimum_zoom=1,
        maximum_zoom=1,
        tile_encoding="indexed_class_id",
    )

    tile_path = output / "tiles/hazard/class-id/1/0/0.png"
    image = Image.open(tile_path)
    tile = np.asarray(image)
    layer = result["layers"][0]
    indexed = layer["indexed_raster"]
    assert image.mode == "LA"
    assert tile.shape == (256, 256, 2)
    assert tuple(tile[32, 32]) == (1, 255)
    assert tuple(tile[32, 224]) == (53, 255)
    assert indexed["tile_file_count"] == 1
    assert indexed["class_id_range"] == [1, 53]
    assert indexed["raster_color_mix"] == [1, 0, 0, 0]
    assert indexed["raster_color_range"] == [0, 1]
    assert result["categorical_tile_encoding"] == "indexed_class_id"
    assert len(list((output / "tiles").rglob("*.png"))) == 1
    assert all("tile_template" not in category for category in layer["categories"])


def test_estimates_indexed_xyz_file_count_without_category_multiplier():
    quadrant = (-WEB_MERCATOR_HALF_WORLD, 0, 0, WEB_MERCATOR_HALF_WORLD)
    assert estimate_xyz_tile_file_count(quadrant, 1, 1) == 1
    assert estimate_xyz_tile_file_count(quadrant, 0, 2) == 6
    with np.testing.assert_raises_regex(ValueError, "Tile zooms"):
        estimate_xyz_tile_file_count(quadrant, 3, 2)


def test_exports_recolorable_category_masks_from_approved_raster(tmp_path):
    run = tmp_path / "runs" / "sample"
    materialized = run / "materialized-v1"
    layer_dir = materialized / "hazard"
    layer_dir.mkdir(parents=True)
    values = np.array(
        [[1, 1, 2, 2], [1, 1, 2, 2], [2, 2, 1, 1], [2, 2, 1, 1]],
        dtype=np.uint8,
    )
    Image.fromarray(values).save(layer_dir / "web-mercator-class-id-final.png")
    (run / "plan.snapshot.json").write_text(
        json.dumps(
            {
                "dataset_id": "sample",
                "title": "Sample",
                "layers": [
                    {
                        "id": "hazard",
                        "categories": [
                            {"id": "one", "label": "One", "legend_rgb": [1, 2, 3]},
                            {"id": "two", "label": "Two", "display_rgb": [4, 5, 6]},
                        ],
                    }
                ],
            }
        )
    )
    bounds = [-WEB_MERCATOR_HALF_WORLD, 0, 0, WEB_MERCATOR_HALF_WORLD]
    (run / "extraction.json").write_text(
        json.dumps(
            {
                "layers": [
                    {"id": "hazard", "warp": {"crs": "EPSG:3857", "bounds": bounds}}
                ]
            }
        )
    )
    manifest = {
        "dataset_id": "sample",
        "source_run": str(run),
        "layers": [
            {
                "layer_id": "hazard",
                "final_pixels_by_class_id": {"1": 8, "2": 8},
                "artifacts": {
                    "class_id": {
                        "path": "hazard/web-mercator-class-id-final.png",
                        "sha256": _sha256(layer_dir / "web-mercator-class-id-final.png"),
                    }
                },
            }
        ],
    }
    manifest_path = materialized / "materialization.json"
    manifest_path.write_text(json.dumps(manifest))
    (materialized / "materialization-review-decision.json").write_text(
        json.dumps(
            {
                "status": "approved",
                "materialization_sha256": _sha256(manifest_path),
            }
        )
    )
    output = tmp_path / "published"
    result = export_categorical_tiles(
        materialized, output, minimum_zoom=1, maximum_zoom=1
    )
    one = np.asarray(Image.open(output / "tiles/hazard/one/1/0/0.png"))
    two = np.asarray(Image.open(output / "tiles/hazard/two/1/0/0.png"))
    assert one[32, 32, 3] == 255
    assert one[32, 224, 3] == 0
    assert two[32, 32, 3] == 0
    assert two[32, 224, 3] == 255
    assert result["layers"][0]["categories"][0]["tile_file_count"] == 1
    assert result["layers"][0]["categories"][0]["display_rgb"] == [1, 2, 3]
    assert result["overscaling"] == "nearest"
    assert result["overview"]["mode"] == "dominant_class_with_fractional_coverage"
    assert "?v=" in result["layers"][0]["categories"][0]["tile_template"]
    assert result["status"] == "approved_publication"
    assert result["approval"]["status"] == "approved"
    assert result["provenance"]["manifest"] == "provenance.json"
    assert (output / "provenance.json").exists()
    assert "path" not in result["materialization"]
    assert "path" not in result["approval"]


def test_boundary_clipped_export_requires_exact_boundary_approval(tmp_path):
    run = tmp_path / "runs" / "sample"
    materialized = run / "materialized-v1"
    layer_dir = materialized / "hazard"
    layer_dir.mkdir(parents=True)
    Image.fromarray(np.ones((4, 4), dtype=np.uint8)).save(
        layer_dir / "web-mercator-class-id-final.png"
    )
    (run / "plan.snapshot.json").write_text(
        json.dumps(
            {
                "dataset_id": "sample",
                "layers": [
                    {
                        "id": "hazard",
                        "categories": [
                            {"id": "one", "label": "One", "legend_rgb": [1, 2, 3]}
                        ],
                    }
                ],
            }
        )
    )
    (run / "extraction.json").write_text(
        json.dumps(
            {
                "layers": [
                    {
                        "id": "hazard",
                        "warp": {
                            "crs": "EPSG:3857",
                            "bounds": [-WEB_MERCATOR_HALF_WORLD, 0, 0, WEB_MERCATOR_HALF_WORLD],
                        },
                    }
                ]
            }
        )
    )
    audit_path = materialized / "boundary-clip-audit.json"
    interior_path = materialized / "boundary-interior.png"
    Image.fromarray(np.ones((4, 4), dtype=np.uint8) * 255).save(interior_path)
    audit_path.write_text(
        json.dumps(
            {
                "status": "pass",
                "method": "test",
                "boundary": {
                    "connected_component_count": 1,
                    "mainland_interior_pixel_count": 16,
                    "interior": {
                        "path": interior_path.name,
                        "sha256": _sha256(interior_path),
                    },
                },
                "layers": [{"passed": True}],
            }
        )
    )
    audit_hash = _sha256(audit_path)
    manifest = {
        "dataset_id": "sample",
        "source_run": str(run),
        "boundary_clip": {
            "audit": {"path": str(audit_path), "sha256": audit_hash},
            "continuous_border_sha256": "border-hash",
            "mainland_interior_sha256": "interior-hash",
            "colored_pixel_count_outside_boundary": 0,
            "unclassified_pixel_count_inside_boundary": 0,
        },
        "layers": [
            {
                "layer_id": "hazard",
                "width": 4,
                "height": 4,
                "final_classified_pixel_count": 16,
                "final_pixels_by_class_id": {"1": 16},
                "artifacts": {
                    "class_id": {
                        "path": "hazard/web-mercator-class-id-final.png",
                        "sha256": _sha256(layer_dir / "web-mercator-class-id-final.png"),
                    }
                },
            }
        ],
    }
    manifest_path = materialized / "materialization.json"
    manifest_path.write_text(json.dumps(manifest))
    decision_path = materialized / "materialization-review-decision.json"
    decision_path.write_text(
        json.dumps(
            {
                "status": "approved",
                "materialization_sha256": _sha256(manifest_path),
                "boundary_clip_audit_sha256": audit_hash,
                "hybrid_border_sha256": "wrong-border-hash",
            }
        )
    )
    with np.testing.assert_raises_regex(ValueError, "continuous boundary"):
        export_categorical_tiles(materialized, tmp_path / "published", 1, 1)

    decision = json.loads(decision_path.read_text())
    decision["hybrid_border_sha256"] = "border-hash"
    decision_path.write_text(json.dumps(decision))
    result = export_categorical_tiles(materialized, tmp_path / "published", 1, 1)
    assert result["boundary"]["continuous_border_component_count"] == 1
    assert result["boundary"]["colored_pixel_count_outside_boundary"] == 0
    assert result["boundary"]["geojson"] == "boundary.geojson"
    assert (tmp_path / "published" / "boundary.geojson").exists()
    public_text = (tmp_path / "published" / "provenance.json").read_text()
    assert str(tmp_path) not in public_text

    multi_interior = np.zeros((4, 4), dtype=np.uint8)
    multi_interior[0:2, 0:2] = 255
    multi_interior[2:4, 3] = 255
    Image.fromarray(multi_interior).save(interior_path)
    Image.fromarray((multi_interior > 0).astype(np.uint8)).save(
        layer_dir / "web-mercator-class-id-final.png"
    )
    audit = json.loads(audit_path.read_text())
    canonical_overlay = materialized / "canonical-overlay.png"
    overlay = np.zeros((8, 8, 4), dtype=np.uint8)
    overlay[1:7, 1, :] = [40, 255, 110, 255]
    Image.fromarray(overlay).save(canonical_overlay)
    canonical_hash = _sha256(canonical_overlay)
    audit["boundary"] = {
        "connected_component_count": 2,
        "expected_component_count": 2,
        "mainland_interior_pixel_count": 4,
        "publication_interior_pixel_count": 6,
        "interior": {"path": interior_path.name, "sha256": _sha256(interior_path)},
        "components": [
            {"id": "mainland", "role": "required_hybrid_mainland", "interior_pixel_count": 4},
            {"id": "island-01", "role": "source_supported_island", "interior_pixel_count": 2, "observed_source_pixel_count": 1},
        ],
        "canonical_display_border": {
            "path": str(canonical_overlay),
            "sha256": canonical_hash,
            "canonical_boundary_id": "canonical-test-v1",
            "grid": {
                "crs": "EPSG:3857",
                "bounds": [-WEB_MERCATOR_HALF_WORLD, 0, 0, WEB_MERCATOR_HALF_WORLD],
                "width": 8,
                "height": 8,
            },
        },
    }
    audit_path.write_text(json.dumps(audit))
    audit_hash = _sha256(audit_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["boundary_clip"].update(
        {
            "audit": {"path": str(audit_path), "sha256": audit_hash},
            "continuous_border_sha256": canonical_hash,
            "boundary_component_count": 2,
            "expected_boundary_component_count": 2,
            "publication_interior_sha256": _sha256(interior_path),
            "canonical_border": {
                "canonical_boundary_id": "canonical-test-v1",
                "manifest_sha256": "canonical-manifest-hash",
                "display_overlay_sha256": canonical_hash,
            },
        }
    )
    manifest["layers"][0]["final_classified_pixel_count"] = 6
    manifest["layers"][0]["final_pixels_by_class_id"] = {"1": 6}
    manifest["layers"][0]["artifacts"]["class_id"]["sha256"] = _sha256(
        layer_dir / "web-mercator-class-id-final.png"
    )
    manifest_path.write_text(json.dumps(manifest))
    decision = json.loads(decision_path.read_text())
    decision["materialization_sha256"] = _sha256(manifest_path)
    decision["boundary_clip_audit_sha256"] = audit_hash
    decision["hybrid_border_sha256"] = canonical_hash
    decision["canonical_display_border_sha256"] = canonical_hash
    decision["canonical_boundary_manifest_sha256"] = "canonical-manifest-hash"
    decision_path.write_text(json.dumps(decision))

    multi = export_categorical_tiles(materialized, tmp_path / "published-multi", 1, 1)
    assert multi["boundary"]["continuous_border_component_count"] == 2
    assert multi["boundary"]["expected_boundary_component_count"] == 2
    assert multi["boundary"]["geojson_feature_count"] == 2
    assert multi["boundary"]["raster"] == "canonical-boundary.png"
    assert multi["boundary"]["raster_sha256"] == canonical_hash
    assert multi["boundary"]["raster_width"] == 8
    assert (tmp_path / "published-multi" / "canonical-boundary.png").exists()
    assert str(tmp_path) not in json.dumps(multi["boundary"])
    geojson = json.loads((tmp_path / "published-multi" / "boundary.geojson").read_text())
    assert [feature["properties"]["role"] for feature in geojson["features"]] == [
        "required_hybrid_mainland",
        "source_supported_island",
    ]
    assert all(
        feature["geometry"]["coordinates"][0]
        == feature["geometry"]["coordinates"][-1]
        for feature in geojson["features"]
    )


def test_low_zoom_overview_preserves_fractional_boundary_coverage():
    values = np.zeros((7, 7), dtype=np.uint8)
    values[:, :3] = 1
    dominant, alpha = _sample_class_overview(
        values,
        (-WEB_MERCATOR_HALF_WORLD, 0, 0, WEB_MERCATOR_HALF_WORLD),
        0,
        0,
        0,
        class_count=1,
        supersampling=4,
    )
    partially_covered = alpha[(alpha > 0) & (alpha < 255)]
    assert len(partially_covered) > 0
    assert np.all(dominant[alpha > 0] == 1)
    assert np.all(dominant[alpha == 0] == 0)
