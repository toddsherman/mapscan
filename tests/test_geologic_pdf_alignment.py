import hashlib
import json

import cv2
import numpy as np
import pytest
from PIL import Image

import mapscan.automatic_alignment_loop as automatic_alignment
from mapscan.automatic_categorical_extraction import _load_accepted_alignment
from mapscan.experiment_log import NoHumanExperimentLog, automatic_provenance
from mapscan.geologic_pdf_alignment import (
    GeologicPdfAlignmentConfig,
    run_geologic_pdf_alignment,
)
from mapscan.source_working_raster import _canonical_png, _decoded_rgb_sha256


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _save_mask(root, name, mask, *, rgba=False):
    path = root / name
    if rgba:
        image = np.zeros((*mask.shape, 4), dtype=np.uint8)
        image[mask] = (100, 255, 150, 255)
        Image.fromarray(image).save(path)
    else:
        Image.fromarray(mask.astype(np.uint8) * 255).save(path)
    return {
        "path": name,
        "sha256": _sha256(path),
        "pixel_count": int(np.count_nonzero(mask)),
    }


def _reference(tmp_path):
    root = tmp_path / "reference"
    root.mkdir()
    shape = (140, 120)
    state = np.zeros(shape, dtype=bool)
    cv2.rectangle(state.view(np.uint8), (14, 8), (105, 131), 1, 1)
    # Make the perimeter geographically distinctive instead of a pure box.
    cv2.line(state.view(np.uint8), (14, 65), (27, 72), 1, 1)
    cv2.line(state.view(np.uint8), (27, 72), (14, 79), 1, 1)
    land = np.zeros(shape, dtype=bool)
    cv2.rectangle(land.view(np.uint8), (15, 9), (104, 130), 1, -1)
    counties = np.zeros(shape, dtype=bool)
    cv2.line(counties.view(np.uint8), (40, 12), (40, 128), 1, 1)
    water = ~land
    artifacts = {
        "state_coast_overlay": _save_mask(root, "state.png", state, rgba=True),
        "county_overlay": _save_mask(root, "counties.png", counties, rgba=True),
        "state_land_mask": _save_mask(root, "land.png", land),
        "water_mask": _save_mask(root, "water.png", water),
    }
    manifest = {
        "schema_version": 1,
        "status": "pinned_reference",
        "kind": "mapbox_california_state_coast_water_counties",
        "authority": {
            "previous_mapscan_canonical_used": False,
            "county_png_used": False,
            "census_used": False,
        },
        "style": {"id": "mapbox/light-v11", "sha256": "a" * 64},
        "tileset": {"tilejson_sha256": "b" * 64},
        "tile_aggregate_sha256": "c" * 64,
        "zoom": 9,
        "target_grid": {
            "crs": "EPSG:3857",
            "bounds": [
                -13857273.186886752,
                3833019.2460767706,
                -12705028.292139662,
                5162403.053672604,
            ],
            "width": shape[1],
            "height": shape[0],
        },
        "artifacts": artifacts,
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path, state


def _adapter(tmp_path, reference_manifest, state, *, draw_state=True):
    source_pdf = tmp_path / "geologic.pdf"
    source_pdf.write_bytes(b"%PDF-1.4\n% synthetic source authority\n%%EOF\n")
    reference = automatic_alignment.load_pinned_mapbox_reference(reference_manifest)
    context = next(
        item
        for item in automatic_alignment._projection_contexts(reference)
        if item.id == "california_albers"
    )
    native_matrix = np.asarray(
        [[185.0, 3.0, 110.0], [-2.0, 185.0, 125.0], [0.0, 0.0, 1.0]]
    )
    source_shape = (260, 240)
    rendered = automatic_alignment._render_projected_reference_line(
        state,
        context,
        native_matrix,
        reference.grid,
        source_shape,
    )
    rgb = np.full((*source_shape, 3), 245, dtype=np.uint8)
    if draw_state:
        rgb[cv2.dilate(rendered.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0] = 20

    longitudes = np.linspace(-124.0, -116.0, 6)
    latitudes = np.linspace(33.0, 41.0, 5)
    geographic = np.asarray(
        [(longitude, latitude) for latitude in latitudes for longitude in longitudes],
        dtype=np.float64,
    )
    transformer = automatic_alignment.Transformer.from_crs(
        "EPSG:4269", context.crs, always_xy=True
    )
    projected_x, projected_y = transformer.transform(
        geographic[:, 0], geographic[:, 1]
    )
    normalized = automatic_alignment._projected_to_candidate_normalized(
        np.column_stack((projected_x, projected_y)), context
    )
    raster_controls = automatic_alignment._transform(normalized, native_matrix)
    controls = {
        "schema_version": 1,
        "kind": "pdf_vector_graticule_controls",
        "status": "detected",
        "detector": "mapscan.pdf_registration.extract_graticule_controls",
        "page_number": 1,
        "geographic_crs": "EPSG:4269",
        "page_coordinate_space": "pdfplumber_page_points",
        "metadata": {"intersection_count": len(geographic)},
        "controls": [
            {
                "longitude": float(lon_lat[0]),
                "latitude": float(lon_lat[1]),
                "pdf_x_points": float(point[0]),
                "pdf_y_points": float(point[1]),
            }
            for lon_lat, point in zip(geographic, raster_controls)
        ],
    }
    adapter_root = tmp_path / "source-adapter"
    adapter_root.mkdir()
    controls_path = adapter_root / "pdf-graticule-controls.json"
    controls_path.write_text(json.dumps(controls, sort_keys=True))
    working_path = adapter_root / "working-raster.png"
    image = Image.fromarray(rgb)
    working_path.write_bytes(_canonical_png(image))
    manifest = {
        "schema_version": "mapscan.source-working-raster.v1",
        "kind": "source_clean_working_raster",
        "authority": {
            "original_source_authoritative": True,
            "prior_alignment_used": False,
            "prior_extraction_used": False,
            "manual_input_used": False,
        },
        "source": {
            "path": str(source_pdf),
            "filename": source_pdf.name,
            "suffix": ".pdf",
            "media_type": "application/pdf",
            "byte_count": source_pdf.stat().st_size,
            "sha256": _sha256(source_pdf),
            "authoritative": True,
        },
        "conversion": {
            "adapter": "poppler-pdftoppm-page-rgb-v1",
            "actual_x_pixels_per_page_point": 1.0,
            "actual_y_pixels_per_page_point": 1.0,
            "page_number": 1,
            "dpi": 72,
        },
        "working_raster": {
            "path": working_path.name,
            "format": "PNG",
            "mode": "RGB",
            "width": image.width,
            "height": image.height,
            "sha256": _sha256(working_path),
            "decoded_rgb_sha256": _decoded_rgb_sha256(image),
        },
        "pdf": {
            "selected_page_number": 1,
            "page_geometry": {
                "width_points": image.width,
                "height_points": image.height,
            },
            "vector_evidence": {"status": "available"},
            "graticule_evidence": {
                "status": "detected",
                "detector": "mapscan.pdf_registration.extract_graticule_controls",
                "control_count": len(geographic),
                "controls_path": controls_path.name,
                "controls_sha256": _sha256(controls_path),
            },
        },
    }
    manifest_path = adapter_root / "source-adapter.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))
    return manifest_path, source_pdf, working_path


def _case(tmp_path, *, draw_state=True):
    reference_manifest, state = _reference(tmp_path)
    adapter_manifest, source_pdf, working = _adapter(
        tmp_path, reference_manifest, state, draw_state=draw_state
    )
    reference = automatic_alignment.load_pinned_mapbox_reference(reference_manifest)
    log = NoHumanExperimentLog(
        "geologic",
        source_pdf,
        mapbox_reference={"manifest_sha256": reference.pin["manifest_sha256"]},
        created_at="2026-08-30T12:00:00+00:00",
    )
    return reference_manifest, adapter_manifest, source_pdf, working, reference, log


def test_native_graticule_alignment_is_accepted_and_extraction_compatible(tmp_path):
    reference_manifest, adapter_manifest, source_pdf, working, reference, log = _case(
        tmp_path
    )
    result = run_geologic_pdf_alignment(
        adapter_manifest,
        reference_manifest,
        tmp_path / "result",
        log,
        config=GeologicPdfAlignmentConfig(validation_max_dimension=300),
    )

    assert result.status == "pass"
    assert result.accepted_alignment_path is not None
    assert log.data["alignment"]["accepted_automatic_iteration_count"] == 1
    iteration = log.data["alignment"]["iterations"][0]
    assert iteration["counts_toward_automatic_iteration_count"] is True
    assert iteration["provenance"]["input_kinds"] == [
        "original_pdf_native_vector_graticule",
        "source_clean_pdf_working_raster",
        "pinned_mapbox_state_coast",
    ]
    payload = json.loads(result.accepted_alignment_path.read_text())
    assert payload["source_sha256"] == _sha256(working)
    assert payload["exact_transform_provenance"]["original_pdf"]["sha256"] == _sha256(
        source_pdf
    )
    assert payload["exact_transform_provenance"]["mapbox_role"] == (
        "independent_validation_only"
    )
    assert payload["scores"]["native_graticule"]["maximum_source_pixel"] < 1e-8
    assert payload["gates"]["semantic_full_state_geographic_balance"]["passed"] is True
    transform = payload["transform"]
    assert transform["kind"] == "projection_aware_mapbox_registration"
    assert transform["target_grid"]["crs"] == "EPSG:3857"
    assert transform["projection"]["id"] == "california_albers"
    assert "reference_pixel_to_source_original_matrix" not in transform
    assert np.allclose(
        transform["candidate_normalized_to_source_original_matrix"],
        payload["exact_transform_provenance"]["fit"][
            "candidate_normalized_to_source_original_pixel_matrix"
        ],
    )
    loaded = _load_accepted_alignment(
        result.accepted_alignment_path,
        working,
        reference.grid,
        reference.pin,
        accepted_iteration_count=1,
    )
    assert loaded["decision"] == "accept"


def test_independent_mapbox_gate_blocks_an_exact_graticule_fit_without_source_boundary(
    tmp_path,
):
    reference_manifest, adapter_manifest, _, _, _, log = _case(
        tmp_path, draw_state=False
    )
    result = run_geologic_pdf_alignment(
        adapter_manifest,
        reference_manifest,
        tmp_path / "result",
        log,
        config=GeologicPdfAlignmentConfig(validation_max_dimension=300),
    )

    assert result.status == "blocked"
    assert result.accepted_alignment_path is None
    iteration = log.data["alignment"]["iterations"][0]
    assert iteration["scores"]["native_graticule"]["maximum_source_pixel"] < 1e-8
    assert iteration["gates"]["semantic_full_state_support"]["passed"] is False
    assert iteration["decision"] == "blocked"


@pytest.mark.parametrize(
    "authority_key",
    ["previous_mapscan_canonical_used", "county_png_used", "census_used"],
)
def test_mapbox_reference_rejects_legacy_county_or_census_authority(
    tmp_path, authority_key
):
    reference_manifest, adapter_manifest, source_pdf, _, _, _ = _case(tmp_path)
    reference_data = json.loads(reference_manifest.read_text())
    reference_data["authority"][authority_key] = True
    reference_manifest.write_text(json.dumps(reference_data))
    log = NoHumanExperimentLog(
        "geologic",
        source_pdf,
        mapbox_reference={"manifest_sha256": _sha256(reference_manifest)},
    )

    with pytest.raises(ValueError, match="old canonical|county.png|Census"):
        run_geologic_pdf_alignment(
            adapter_manifest, reference_manifest, tmp_path / "result", log
        )


@pytest.mark.parametrize(
    "authority_key",
    ["prior_alignment_used", "prior_extraction_used", "manual_input_used"],
)
def test_source_adapter_rejects_prior_or_manual_inputs(tmp_path, authority_key):
    reference_manifest, adapter_manifest, source_pdf, _, reference, _ = _case(tmp_path)
    adapter_data = json.loads(adapter_manifest.read_text())
    adapter_data["authority"][authority_key] = True
    adapter_manifest.write_text(json.dumps(adapter_data))
    log = NoHumanExperimentLog(
        "geologic",
        source_pdf,
        mapbox_reference={"manifest_sha256": reference.pin["manifest_sha256"]},
    )

    with pytest.raises(ValueError, match="source-clean authority"):
        run_geologic_pdf_alignment(
            adapter_manifest, reference_manifest, tmp_path / "result", log
        )


def test_prior_iteration_cannot_be_consumed_as_an_alignment_input(tmp_path):
    reference_manifest, adapter_manifest, _, _, _, log = _case(tmp_path)
    log.record_alignment_iteration(
        scores={"prior": 1.0},
        gates={"prior": True},
        decision="retry",
        provenance=automatic_provenance("another.automatic.run", ["source_pixels"]),
        method="prior automatic attempt",
    )

    with pytest.raises(ValueError, match="fresh no-human"):
        run_geologic_pdf_alignment(
            adapter_manifest, reference_manifest, tmp_path / "result", log
        )
