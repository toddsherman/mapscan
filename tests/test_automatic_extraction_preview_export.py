import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mapscan.automatic_extraction_preview_export import (
    _minimum_nonundersampling_zoom,
    _screen_pixel_span,
    export_automatic_categorical_staging_preview,
)
from mapscan.tile_export import WEB_MERCATOR_HALF_WORLD


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def _fixture(tmp_path: Path) -> dict[str, Path]:
    root = tmp_path / "dataset" / "automatic-extraction"
    iteration = root / "extraction-02"
    iteration.mkdir(parents=True)
    source = tmp_path / "source.png"
    Image.fromarray(np.full((3, 4, 3), 127, dtype=np.uint8)).save(source)
    grid = {
        "crs": "EPSG:3857",
        "bounds": [
            -WEB_MERCATOR_HALF_WORLD,
            -WEB_MERCATOR_HALF_WORLD,
            WEB_MERCATOR_HALF_WORLD,
            WEB_MERCATOR_HALF_WORLD,
        ],
        "width": 2,
        "height": 2,
    }
    alignment = tmp_path / "accepted-alignment.json"
    _write_json(
        alignment,
        {
            "schema_version": 1,
            "decision": "accept",
            "iteration": 7,
            "transform": {
                "reference_pixel_space": "pinned_mapbox_target_grid",
                "target_grid": grid,
            },
        },
    )
    legend = root / "legend" / "legend.json"
    _write_json(
        legend,
        {
            "schema_version": "mapscan.automatic-categorical-extraction.v1",
            "entries": [
                {"class_id": 1, "label": "One Class", "rgb": [1, 2, 3]},
                {"class_id": 2, "label": "Two Class", "rgb": [4, 5, 6]},
            ],
        },
    )
    classes = iteration / "web-mercator-class-id.png"
    Image.fromarray(np.asarray([[1, 0], [2, 1]], dtype=np.uint8)).save(classes)
    missing = root / "target-missing-source-extent-mask.png"
    Image.fromarray(np.asarray([[0, 255], [0, 0]], dtype=np.uint8)).save(missing)
    report = iteration / "iteration.json"
    _write_json(
        report,
        {
            "schema_version": "mapscan.automatic-categorical-extraction.v1",
            "iteration": 2,
            "decision": "accept",
            "legend_sha256": _sha256(legend),
            "artifacts": [
                {
                    "path": "extraction-02/web-mercator-class-id.png",
                    "sha256": _sha256(classes),
                    "byte_count": classes.stat().st_size,
                },
                {
                    "path": "target-missing-source-extent-mask.png",
                    "sha256": _sha256(missing),
                    "byte_count": missing.stat().st_size,
                },
            ],
        },
    )
    pointer = root / "accepted-extraction.json"
    _write_json(
        pointer,
        {
            "schema_version": "mapscan.automatic-categorical-extraction.v1",
            "status": "accepted",
            "automatic_iteration_count": 2,
            "source": {"path": str(source), "sha256": _sha256(source)},
            "alignment": {
                "path": str(alignment),
                "sha256": _sha256(alignment),
            },
            "legend": {"path": "legend/legend.json", "sha256": _sha256(legend)},
            "accepted_iteration": "extraction-02",
            "aligned_source_extent": {
                "partial_extent": True,
                "missing_source_extent_pixel_count": 1,
                "missing_extent_policy": "preserve_as_nodata_never_infer",
            },
        },
    )
    return {
        "root": root,
        "source": source,
        "alignment": alignment,
        "legend": legend,
        "classes": classes,
        "missing": missing,
        "report": report,
        "pointer": pointer,
    }


def test_exports_hash_verified_unapproved_preview_with_transparent_nodata(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "staging"

    result = export_automatic_categorical_staging_preview(
        fixture["root"],
        output,
        dataset_id="fixture",
        title="Fixture",
        minimum_zoom=0,
        maximum_zoom=1,
        overview_supersampling=2,
    )

    dataset = json.loads((output / "dataset.json").read_text())
    provenance = json.loads(
        (output / "autonomous-preview-provenance.json").read_text()
    )
    assert result["status"] == "needs_visual_review"
    assert dataset["approval"] == {"status": "not_approved"}
    assert dataset["boundary"] is None
    assert provenance["status"] == "needs_visual_review"
    assert provenance["publication_approved"] is False
    assert provenance["boundary"] is None
    assert provenance["accepted_alignment"]["automatic_iteration_count"] == 7
    assert provenance["accepted_alignment"]["sha256"] == _sha256(
        fixture["alignment"]
    )
    assert provenance["accepted_extraction"]["automatic_iteration_count"] == 2
    assert provenance["accepted_extraction"]["sha256"] == _sha256(
        fixture["pointer"]
    )
    assert provenance["class_raster"]["sha256"] == _sha256(fixture["classes"])
    assert provenance["nodata"] == {
        "class_id": 0,
        "transparent": True,
        "pixel_count": 1,
        "partial_source_extent": True,
        "missing_source_extent_pixel_count": 1,
        "missing_source_extent_mask_sha256": _sha256(fixture["missing"]),
        "policy": "preserve_as_transparent_nodata_never_infer",
    }
    assert _sha256(output / "source.png") == _sha256(fixture["source"])
    categories = dataset["layers"][0]["categories"]
    assert [(item["label"], item["display_rgb"]) for item in categories] == [
        ("One Class", [1, 2, 3]),
        ("Two Class", [4, 5, 6]),
    ]
    assert [item["pixel_count"] for item in categories] == [2, 1]
    assert all("?v=" in item["tile_template"] for item in categories)

    # The upper-right native tile samples class zero. Every recolorable class
    # mask must therefore remain fully transparent there.
    for category in ("one-class", "two-class"):
        tile = np.asarray(
            Image.open(output / f"tiles/classes/{category}/1/1/0.png")
        )
        assert tile.shape == (256, 256, 4)
        assert not np.any(tile[..., 3])
    provenance_text = (output / "autonomous-preview-provenance.json").read_text()
    assert "lime" not in provenance_text.casefold()
    assert "county" not in provenance_text.casefold()


def test_exports_one_shared_indexed_pyramid_and_viewer_manifest(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "indexed-staging"

    result = export_automatic_categorical_staging_preview(
        fixture["root"],
        output,
        dataset_id="indexed-fixture",
        title="Indexed Fixture",
        minimum_zoom=0,
        maximum_zoom=1,
        overview_supersampling=2,
        tile_encoding="indexed_class_id",
    )

    layer = result["layers"][0]
    indexed = layer["indexed_raster"]
    assert result["categorical_tile_encoding"] == "indexed_class_id"
    assert result["maximum_native_zoom"] == 1
    assert indexed["encoding"] == "png_luma_alpha_uint8_class_id_v1"
    assert indexed["tile_file_count"] == 5
    assert indexed["class_id_range"] == [1, 2]
    assert indexed["raster_color_mix"] == [1, 0, 0, 0]
    assert indexed["raster_color_range"] == [0, 1]
    assert all("tile_template" not in category for category in layer["categories"])
    assert len(list((output / "tiles").rglob("*.png"))) == 5
    assert not any((output / "tiles").glob("classes/one-class/**"))

    native = Image.open(output / "tiles/classes/class-id/1/1/0.png")
    native_values = np.asarray(native)
    assert native.mode == "LA"
    assert native_values.shape == (256, 256, 2)
    assert not np.any(native_values[..., 1])

    provenance = json.loads(
        (output / "autonomous-preview-provenance.json").read_text()
    )
    assert provenance["categorical_tile_encoding"] == "indexed_class_id"


def test_accepts_the_accepted_extraction_pointer_path_directly(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)

    result = export_automatic_categorical_staging_preview(
        fixture["pointer"],
        tmp_path / "staging",
        minimum_zoom=0,
        maximum_zoom=0,
        overview_supersampling=2,
    )

    assert result["status"] == "needs_visual_review"
    assert result["minimum_zoom"] == 0
    assert result["maximum_native_zoom"] == 0


def test_accepts_cumulative_iteration_count_across_extraction_runs(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    pointer = json.loads(fixture["pointer"].read_text())
    pointer["automatic_iteration_count"] = 6
    _write_json(fixture["pointer"], pointer)

    result = export_automatic_categorical_staging_preview(
        fixture["root"],
        tmp_path / "staging",
        minimum_zoom=0,
        maximum_zoom=0,
        overview_supersampling=2,
    )

    provenance = json.loads(
        (tmp_path / "staging/autonomous-preview-provenance.json").read_text()
    )
    assert result["status"] == "needs_visual_review"
    assert provenance["accepted_extraction"]["automatic_iteration_count"] == 6
    assert provenance["accepted_extraction"]["accepted_iteration"] == "extraction-02"
    assert provenance["accepted_extraction"]["accepted_iteration_local_count"] == 2


def test_can_retain_pristine_source_separately_from_working_source(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    pristine = tmp_path / "pristine.jpg"
    Image.fromarray(np.full((5, 6, 3), 223, dtype=np.uint8)).save(pristine)

    export_automatic_categorical_staging_preview(
        fixture["root"],
        tmp_path / "staging",
        minimum_zoom=0,
        maximum_zoom=0,
        overview_supersampling=2,
        retained_source_path=pristine,
        retained_source_sha256=_sha256(pristine),
    )

    dataset = json.loads((tmp_path / "staging/dataset.json").read_text())
    provenance = json.loads(
        (tmp_path / "staging/autonomous-preview-provenance.json").read_text()
    )
    assert dataset["source_image"] == "source.jpg"
    assert _sha256(tmp_path / "staging/source.jpg") == _sha256(pristine)
    assert provenance["source"]["sha256"] == _sha256(pristine)
    assert provenance["accepted_working_source"]["sha256"] == _sha256(
        fixture["source"]
    )


def test_can_clip_publication_pixels_without_mutating_accepted_raster(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    state_mask = tmp_path / "state-mask.png"
    Image.fromarray(np.asarray([[255, 255], [0, 255]], dtype=np.uint8)).save(
        state_mask
    )

    result = export_automatic_categorical_staging_preview(
        fixture["root"],
        tmp_path / "staging",
        minimum_zoom=0,
        maximum_zoom=0,
        overview_supersampling=2,
        tile_encoding="indexed_class_id",
        publication_state_mask_path=state_mask,
        publication_state_mask_sha256=_sha256(state_mask),
    )

    provenance = json.loads(
        (tmp_path / "staging/autonomous-preview-provenance.json").read_text()
    )
    publication_raster = tmp_path / "staging/publication-class-id.png"
    assert np.array_equal(
        np.asarray(Image.open(publication_raster)),
        np.asarray([[1, 0], [0, 1]], dtype=np.uint8),
    )
    assert np.array_equal(
        np.asarray(Image.open(fixture["classes"])),
        np.asarray([[1, 0], [2, 1]], dtype=np.uint8),
    )
    assert provenance["accepted_class_raster"]["sha256"] == _sha256(
        fixture["classes"]
    )
    assert provenance["class_raster"]["sha256"] == _sha256(publication_raster)
    assert provenance["publication_clip"] == {
        "method": "direct_mapbox_state_interior_mask_v1",
        "state_interior_mask_path": str(state_mask.resolve()),
        "state_interior_mask_sha256": _sha256(state_mask),
        "accepted_colored_pixel_count_outside_state": 1,
        "publication_colored_pixel_count_outside_state": 0,
        "mutated_accepted_extraction": False,
    }
    assert [category["pixel_count"] for category in result["layers"][0]["categories"]] == [
        2,
        0,
    ]


def _make_supersampled_fixture(fixture: dict[str, Path]) -> Path:
    base_pin = {
        "manifest_sha256": "a" * 64,
        "style_sha256": "b" * 64,
        "tilejson_sha256": "c" * 64,
        "tile_aggregate_sha256": "d" * 64,
    }
    alignment = json.loads(fixture["alignment"].read_text())
    alignment["mapbox_reference"] = base_pin
    _write_json(fixture["alignment"], alignment)
    high_grid = {
        **alignment["transform"]["target_grid"],
        "width": 3,
        "height": 3,
    }
    Image.fromarray(
        np.asarray([[1, 1, 0], [2, 1, 0], [2, 2, 0]], dtype=np.uint8)
    ).save(fixture["classes"])
    Image.fromarray(
        np.asarray(
            [[0, 0, 255], [0, 0, 255], [0, 0, 255]], dtype=np.uint8
        )
    ).save(fixture["missing"])
    report = json.loads(fixture["report"].read_text())
    report["scores"] = {"processing_target_grid": high_grid}
    for artifact, path in zip(
        report["artifacts"], (fixture["classes"], fixture["missing"])
    ):
        artifact["sha256"] = _sha256(path)
        artifact["byte_count"] = path.stat().st_size
    _write_json(fixture["report"], report)
    processing = fixture["root"].parent / "processing-reference.json"
    _write_json(
        processing,
        {
            "status": "pinned_reference",
            "kind": "mapbox_california_state_coast_water_counties",
            "target_grid": high_grid,
            "style": {"sha256": base_pin["style_sha256"]},
            "tileset": {"tilejson_sha256": base_pin["tilejson_sha256"]},
            "tile_aggregate_sha256": base_pin["tile_aggregate_sha256"],
            "derivation": {
                "source_manifest_sha256": base_pin["manifest_sha256"],
                "raw_bytes_preserved_exactly": True,
                "only_derived_masks_and_overlays_recomputed": True,
                "target_grid_supersampling": 2,
            },
        },
    )
    pointer = json.loads(fixture["pointer"].read_text())
    pointer["alignment"]["sha256"] = _sha256(fixture["alignment"])
    pointer["processing_target_grid"] = high_grid
    pointer["target_supersampling"] = 2
    pointer["processing_reference"] = {
        "path": str(processing),
        "sha256": _sha256(processing),
    }
    pointer["aligned_source_extent"]["missing_source_extent_pixel_count"] = 3
    _write_json(fixture["pointer"], pointer)
    return processing


def test_exports_hash_verified_corner_preserving_supersampled_result(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    processing = _make_supersampled_fixture(fixture)

    result = export_automatic_categorical_staging_preview(
        fixture["root"],
        tmp_path / "staging-high",
        minimum_zoom=0,
        maximum_zoom=0,
        overview_supersampling=2,
    )

    provenance = json.loads(
        (tmp_path / "staging-high/autonomous-preview-provenance.json").read_text()
    )
    assert result["status"] == "needs_visual_review"
    assert provenance["class_raster"]["width"] == 3
    assert provenance["processing_reference"] == {
        "path": str(processing.resolve()),
        "sha256": _sha256(processing),
        "target_supersampling": 2,
        "raw_hashes": {
            "style_sha256": "b" * 64,
            "tilejson_sha256": "c" * 64,
            "tile_aggregate_sha256": "d" * 64,
        },
    }


def test_exports_hash_pinned_mapbox_state_coast_diagnostic(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    processing = _make_supersampled_fixture(fixture)
    overlay = processing.parent / "state-coast-overlay.png"
    Image.fromarray(
        np.asarray(
            [
                [[0, 0, 0, 0], [0, 255, 0, 255], [0, 0, 0, 0]],
                [[0, 255, 0, 255], [0, 0, 0, 0], [0, 255, 0, 255]],
                [[0, 0, 0, 0], [0, 255, 0, 255], [0, 0, 0, 0]],
            ],
            dtype=np.uint8,
        )
    ).save(overlay)
    manifest = json.loads(processing.read_text())
    manifest["artifacts"] = {
        "state_coast_overlay": {
            "path": overlay.name,
            "sha256": _sha256(overlay),
        }
    }
    _write_json(processing, manifest)
    pointer = json.loads(fixture["pointer"].read_text())
    pointer["processing_reference"]["sha256"] = _sha256(processing)
    _write_json(fixture["pointer"], pointer)

    output = tmp_path / "staging-mapbox-diagnostic"
    result = export_automatic_categorical_staging_preview(
        fixture["root"],
        output,
        minimum_zoom=0,
        maximum_zoom=0,
        overview_supersampling=2,
    )

    boundary = result["boundary"]
    assert boundary == {
        "kind": "pinned_mapbox_state_coast_diagnostic",
        "authority": "accepted_alignment_mapbox_reference",
        "diagnostic_only": True,
        "raster": "mapbox-state-coast-overlay.png",
        "raster_sha256": _sha256(overlay),
        "raster_width": 3,
        "raster_height": 3,
        "raster_bounds": [-180.0, -85.0511287798066, 180.0, 85.0511287798066],
    }
    assert _sha256(output / boundary["raster"]) == _sha256(overlay)
    provenance = json.loads(
        (output / "autonomous-preview-provenance.json").read_text()
    )
    assert provenance["boundary"] == boundary


def test_rejects_supersampled_reference_with_changed_mapbox_authority(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    processing = _make_supersampled_fixture(fixture)
    manifest = json.loads(processing.read_text())
    manifest["tile_aggregate_sha256"] = "e" * 64
    _write_json(processing, manifest)
    pointer = json.loads(fixture["pointer"].read_text())
    pointer["processing_reference"]["sha256"] = _sha256(processing)
    _write_json(fixture["pointer"], pointer)

    with pytest.raises(ValueError, match="changed accepted Mapbox authority"):
        export_automatic_categorical_staging_preview(
            fixture["root"],
            tmp_path / "staging-high",
            minimum_zoom=0,
            maximum_zoom=0,
            overview_supersampling=2,
        )


def test_high_resolution_farms_grid_requires_zoom_eleven():
    grid = {
        "crs": "EPSG:3857",
        "bounds": [
            -13857273.186886752,
            3833019.2460767706,
            -12705028.292139662,
            5162403.053672604,
        ],
        "width": 10192,
        "height": 11758,
    }

    assert _screen_pixel_span(grid, 10) == pytest.approx(
        (7537.2167168, 8695.94120428179)
    )
    assert _screen_pixel_span(grid, 11) == pytest.approx(
        (15074.4334336, 17391.88240856358)
    )
    assert _minimum_nonundersampling_zoom(grid) == 11


@pytest.mark.parametrize("target", ["source", "alignment", "legend", "classes"])
def test_rejects_changed_hash_bound_inputs(tmp_path: Path, target: str) -> None:
    fixture = _fixture(tmp_path)
    path = fixture[target]
    if target in {"source", "classes"}:
        Image.fromarray(np.zeros((2, 2), dtype=np.uint8)).save(path)
    else:
        value = json.loads(path.read_text())
        value["tampered"] = True
        _write_json(path, value)

    with pytest.raises(ValueError, match="declared hash"):
        export_automatic_categorical_staging_preview(
            fixture["root"],
            tmp_path / "staging",
            minimum_zoom=0,
            maximum_zoom=1,
        )


def test_rejects_classification_inside_partial_missing_source_extent(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    Image.fromarray(np.asarray([[1, 1], [2, 1]], dtype=np.uint8)).save(
        fixture["classes"]
    )
    report = json.loads(fixture["report"].read_text())
    artifact = report["artifacts"][0]
    artifact["sha256"] = _sha256(fixture["classes"])
    artifact["byte_count"] = fixture["classes"].stat().st_size
    _write_json(fixture["report"], report)

    with pytest.raises(ValueError, match="missing source extent contains classified"):
        export_automatic_categorical_staging_preview(
            fixture["root"],
            tmp_path / "staging",
            minimum_zoom=0,
            maximum_zoom=1,
        )
