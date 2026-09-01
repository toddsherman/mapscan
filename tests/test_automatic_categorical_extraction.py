import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pyproj import CRS, Transformer
from PIL import Image

import mapscan.automatic_categorical_extraction as categorical_extraction
from mapscan.automatic_categorical_extraction import (
    ExtractionIteration,
    ExtractionLoopConfig,
    LegendDetection,
    LegendEntry,
    OCRWord,
    _automatic_classification_domain,
    _classify,
    _automatic_swatch_sequence,
    _accepted_extraction_payload,
    _canonical_extraction_iteration_number,
    _detect_legend,
    _label_is_clean,
    _load_accepted_alignment,
    _nearest_completion,
    _palette_lab,
    _palette_chromaticity_compatible,
    _projection_reference_to_source_base,
    _reference_to_source_remap,
    _residual_displacement,
    _render,
    _resumable_prior_extraction_count,
    _source_to_reference_remap,
    _supersample_alignment_transform,
    _supersampled_target_grid,
    _target_source_coverage_mask,
    _write_legend_artifacts,
    detect_legend,
    run_automatic_categorical_extraction,
)
from mapscan.experiment_log import NoHumanExperimentLog, automatic_provenance
from mapscan.dither_texture_classifier import build_dither_texture_model


PALETTE = (
    (255, 255, 255),
    (255, 190, 190),
    (235, 50, 50),
    (150, 0, 0),
)
HISTORICAL_RAINFALL_SOURCE = Path(__file__).parents[1] / "examples" / "rainfall.gif"


def _synthetic_source() -> tuple[np.ndarray, np.ndarray, list[OCRWord]]:
    rgb = np.full((360, 500, 3), 255, dtype=np.uint8)
    domain = np.zeros((360, 500), dtype=bool)
    domain[20:340, 20:280] = True
    for index, color in enumerate(PALETTE):
        rgb[20 + index * 80 : 20 + (index + 1) * 80, 20:280] = color
    # Contextual line ink inside the data region must become inferred evidence.
    rgb[20:340, 145:150] = (0, 0, 0)

    swatch_left, width, height = 390, 30, 20
    centers = (110, 150, 190, 230)
    # The first (white) swatch is invisible against the white page.
    for center, color in zip(centers[1:], PALETTE[1:]):
        rgb[center - height // 2 : center + height // 2, swatch_left : swatch_left + width] = color
    labels = ("< 1", "1 - 10", "10 - 100", "> 100")
    words = [
        OCRWord(
            text=label,
            confidence=96.0,
            left=435,
            top=center - 9,
            width=55,
            height=18,
            line_key=(1, 1, index + 1),
        )
        for index, (center, label) in enumerate(zip(centers, labels))
    ]
    return rgb, domain, words


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _alignment(
    path: Path,
    source: Path,
    grid: dict,
    *,
    mapbox_reference=None,
    **extra,
) -> Path:
    payload = {
        "schema_version": 1,
        "iteration": 1,
        "decision": "accept",
        "source_sha256": _sha(source),
        "mapbox_reference": mapbox_reference,
        "transform": {
            "kind": "regular_global_mapbox_registration",
            "source_pixel_space": "original_raster",
            "reference_pixel_space": "pinned_mapbox_target_grid",
            "source_original_to_reference_pixel_matrix": np.eye(3).tolist(),
            "reference_pixel_to_source_original_matrix": np.eye(3).tolist(),
            "target_grid": grid,
        },
        **extra,
    }
    path.write_text(json.dumps(payload))
    return path


def _accepted_log(source: Path, alignment=None) -> NoHumanExperimentLog:
    log = NoHumanExperimentLog(
        "synthetic",
        source,
        mapbox_reference={"id": "synthetic-mapbox-reference", "sha256": "abc"},
    )
    log.record_alignment_iteration(
        scores={"perimeter_p90_px": 1.0},
        gates={"automatic_alignment": True},
        decision="accept",
        provenance=automatic_provenance(
            "test.automatic_alignment",
            ["original_source_image_pixels", "pinned_mapbox_state_coast"],
        ),
        method="synthetic automatic alignment",
        artifacts=(
            [
                {
                    "path": str(
                        alignment.with_name("candidate.json")
                    ),
                    "sha256": _sha(alignment),
                }
            ]
            if alignment is not None
            else []
        ),
    )
    return log


@pytest.mark.parametrize(
    ("resume_style", "prior_count", "replay_number", "expected_ordinal"),
    [
        ("deer", 4, 2, 6),
        ("rainfall-1981-2010", 4, 4, 8),
    ],
)
def test_resumed_manifest_uses_canonical_global_extraction_ordinal(
    tmp_path: Path,
    resume_style: str,
    prior_count: int,
    replay_number: int,
    expected_ordinal: int,
) -> None:
    source = tmp_path / f"{resume_style}-source.png"
    Image.new("RGB", (10, 10), "white").save(source)
    alignment = tmp_path / f"{resume_style}-alignment.json"
    alignment.write_text('{"decision":"accept"}\n')
    legend = tmp_path / "legend.json"
    legend.write_text('{"entries":[]}\n')
    ordinal = _canonical_extraction_iteration_number(prior_count, replay_number)
    accepted = ExtractionIteration(ordinal, "accepted", {}, {}, ())
    payload = _accepted_extraction_payload(accepted, source, alignment, legend)

    assert ordinal == expected_ordinal
    assert payload["automatic_iteration_count"] == expected_ordinal
    assert payload["accepted_iteration"] == f"extraction-{expected_ordinal:02d}"


def test_resume_guard_counts_only_contiguous_rejected_automatic_history(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.png"
    Image.new("RGB", (10, 10), "white").save(source)
    log = _accepted_log(source)
    for ordinal in range(1, 5):
        log.record_extraction_iteration(
            scores={"retry": ordinal},
            gates={"fixed_point": False},
            decision="retry",
            provenance=automatic_provenance(
                "test.automatic_extraction", ["original_source_pixels"]
            ),
            method="automatic rejected extraction retry",
        )
    assert _resumable_prior_extraction_count(log) == 4

    log.data["extraction"]["iterations"][-1]["provenance"]["human_approval"] = True
    with pytest.raises(ValueError, match="prior rejected automatic attempts"):
        _resumable_prior_extraction_count(log)


def test_detects_swatch_series_and_invisible_white_row():
    rgb, domain, words = _synthetic_source()

    entries, axis, step = _detect_legend(rgb, domain, words)

    assert [entry.label for entry in entries] == ["< 1", "1 - 10", "10 - 100", "> 100"]
    assert [entry.rgb for entry in entries] == list(PALETTE)
    assert entries[0].swatch_status == "invisible_row_inferred_from_regular_legend_layout"
    assert all(
        entry.swatch_status == "visible_exact_color_rectangle" for entry in entries[1:]
    )
    assert axis == pytest.approx(405)
    assert step == pytest.approx(40)


def test_alignment_loader_rejects_any_legacy_or_manual_input(tmp_path):
    source = tmp_path / "source.png"
    Image.new("RGB", (10, 10), "white").save(source)
    grid = {"crs": "EPSG:3857", "bounds": [0, 0, 1, 1], "width": 10, "height": 10}
    alignment = _alignment(
        tmp_path / "alignment.json",
        source,
        grid,
        manual_control_points=[],
    )

    with pytest.raises(ValueError, match="forbidden legacy/manual input"):
        _load_accepted_alignment(alignment, source, grid)


def test_production_alignment_loader_anchors_pointer_to_accepted_iteration(tmp_path):
    source = tmp_path / "source.png"
    Image.new("RGB", (10, 10), "white").save(source)
    grid = {"crs": "EPSG:3857", "bounds": [0, 0, 1, 1], "width": 10, "height": 10}
    alignment = _alignment(tmp_path / "accepted-alignment.json", source, grid)
    iterations = [
        {
            "automatic_iteration": 1,
            "counts_toward_automatic_iteration_count": True,
            "decision": "accept",
            "artifacts": [
                {
                    "path": str(tmp_path / "alignment-01" / "candidate.json"),
                    "sha256": _sha(alignment),
                }
            ],
        }
    ]

    _load_accepted_alignment(
        alignment,
        source,
        grid,
        accepted_iteration_count=1,
        map_id="synthetic",
        alignment_iterations=iterations,
    )

    payload = json.loads(alignment.read_text())
    payload["transform"]["source_original_to_reference_pixel_matrix"][0][2] = 1.0
    payload["transform"]["reference_pixel_to_source_original_matrix"][0][2] = -1.0
    alignment.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="pointer hash disagrees"):
        _load_accepted_alignment(
            alignment,
            source,
            grid,
            accepted_iteration_count=1,
            map_id="synthetic",
            alignment_iterations=iterations,
        )


def test_production_alignment_loader_requires_external_hash_authority(tmp_path):
    source = tmp_path / "source.png"
    Image.new("RGB", (10, 10), "white").save(source)
    grid = {"crs": "EPSG:3857", "bounds": [0, 0, 1, 1], "width": 10, "height": 10}
    alignment = _alignment(tmp_path / "accepted-alignment.json", source, grid)

    with pytest.raises(ValueError, match="requires accepted-alignment hash authority"):
        _load_accepted_alignment(
            alignment,
            source,
            grid,
            accepted_iteration_count=1,
            map_id="synthetic",
        )


def test_supersampled_transform_preserves_original_reference_samples():
    grid = {
        "crs": "EPSG:3857",
        "bounds": [0.0, 0.0, 40.0, 30.0],
        "width": 5,
        "height": 4,
    }
    source_to_reference = np.asarray(
        [[1.1, 0.1, 0.5], [0.05, 0.9, 0.25], [0.0, 0.0, 1.0]]
    )
    transform = {
        "kind": "regular_global_mapbox_registration",
        "target_grid": grid,
        "source_original_to_reference_pixel_matrix": source_to_reference.tolist(),
        "reference_pixel_to_source_original_matrix": np.linalg.inv(
            source_to_reference
        ).tolist(),
    }

    high = _supersample_alignment_transform(transform, 2)
    assert high["target_grid"] == {
        **grid,
        "width": 9,
        "height": 7,
    }
    low_x, low_y = _reference_to_source_remap(transform)
    high_x, high_y = _reference_to_source_remap(high)
    assert np.allclose(high_x[::2, ::2], low_x, atol=1e-6)
    assert np.allclose(high_y[::2, ::2], low_y, atol=1e-6)


def test_supersampled_residual_preserves_displacement_and_rehashes():
    residual = {
        "kind": "wendland_c2_reference_pixel_to_source_original_displacement",
        "coordinate_domain": "pinned_mapbox_target_grid_pixels",
        "displacement_range": "source_original_pixels",
        "centers_reference_px": [[2.0, 3.0]],
        "coefficients_source_px": [[0.5, -0.25]],
        "radius_reference_px": 8.0,
        "ridge": 0.1,
        "kernel": "max(1-r,0)^4*(4r+1)",
    }
    residual["sha256"] = hashlib.sha256(
        json.dumps(residual, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    transform = {
        "kind": "projection_aware_residual_warp_mapbox_registration",
        "target_grid": {
            "crs": "EPSG:3857",
            "bounds": [0.0, 0.0, 4.0, 4.0],
            "width": 5,
            "height": 5,
        },
        "residual_warp": residual,
    }

    high = _supersample_alignment_transform(transform, 2)
    x = np.asarray([[1.0, 2.0, 3.0]])
    y = np.asarray([[2.0, 3.0, 4.0]])
    low_dx, low_dy = _residual_displacement(transform, x, y)
    high_dx, high_dy = _residual_displacement(high, x * 2.0, y * 2.0)
    assert np.allclose(high_dx, low_dx)
    assert np.allclose(high_dy, low_dy)
    encoded = dict(high["residual_warp"])
    digest = encoded.pop("sha256")
    assert digest == hashlib.sha256(
        json.dumps(encoded, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@pytest.mark.parametrize("factor", [0, 1.5])
def test_supersampled_target_grid_rejects_invalid_factors(factor):
    grid = {
        "crs": "EPSG:3857",
        "bounds": [0.0, 0.0, 1.0, 1.0],
        "width": 5,
        "height": 5,
    }
    with pytest.raises(ValueError, match="positive integer"):
        _supersampled_target_grid(grid, factor)


def test_projection_aware_alignment_remaps_without_flattening_projection(tmp_path):
    source = tmp_path / "source.png"
    Image.new("RGB", (10, 10), "white").save(source)
    geographic = CRS.from_epsg(4326)
    wkt = geographic.to_wkt(version="WKT2_2019", pretty=False)
    to_web = Transformer.from_crs(geographic, "EPSG:3857", always_xy=True)
    minimum_x, minimum_y = to_web.transform(-124.0, 32.0)
    maximum_x, maximum_y = to_web.transform(-114.0, 42.0)
    grid = {
        "crs": "EPSG:3857",
        "bounds": [minimum_x, minimum_y, maximum_x, maximum_y],
        "width": 10,
        "height": 10,
    }
    candidate_to_source = np.asarray(
        [[9.0, 0.0, 4.5], [0.0, 9.0, 4.5], [0.0, 0.0, 1.0]]
    )
    transform = {
        "kind": "projection_aware_mapbox_registration",
        "source_pixel_space": "original_raster",
        "reference_pixel_space": "pinned_mapbox_target_grid",
        "source_original_shape": [10, 10],
        "projection": {
            "id": "geographic",
            "always_xy": True,
            "crs_wkt": wkt,
            "crs_wkt_sha256": hashlib.sha256(wkt.encode("utf-8")).hexdigest(),
            "normalization_center": [-119.0, 37.0],
            "normalization_scale": 10.0,
        },
        "candidate_normalized_to_source_original_matrix": candidate_to_source.tolist(),
        "source_original_to_candidate_normalized_matrix": np.linalg.inv(
            candidate_to_source
        ).tolist(),
        "target_grid": grid,
    }
    alignment = tmp_path / "alignment.json"
    alignment.write_text(
        json.dumps(
            {
                "decision": "accept",
                "source_sha256": _sha(source),
                "transform": transform,
            }
        )
    )

    loaded = _load_accepted_alignment(alignment, source, grid)
    assert loaded["transform"]["projection"]["id"] == "geographic"
    reference_x, reference_y = _reference_to_source_remap(transform)
    consumer_diagnostics = {}
    source_x, source_y = _source_to_reference_remap(
        transform, (10, 10), diagnostics=consumer_diagnostics
    )
    assert consumer_diagnostics["converged"] is True
    assert consumer_diagnostics["maximum_final_reference_update_px"] <= 1e-4
    assert consumer_diagnostics["maximum_source_roundtrip_error_px"] <= 0.02

    assert reference_x[0, 0] == pytest.approx(0.0, abs=1e-4)
    assert reference_y[0, 0] == pytest.approx(0.0, abs=1e-4)
    assert reference_x[-1, -1] == pytest.approx(9.0, abs=1e-4)
    assert reference_y[-1, -1] == pytest.approx(9.0, abs=1e-4)
    # Web Mercator y is nonlinear in geographic latitude; an interior row must
    # not collapse to the same fraction as a flat pixel homography.
    assert reference_y[4, 5] != pytest.approx(4.0, abs=1e-3)
    assert source_x[0, 0] == pytest.approx(0.0, abs=1e-4)
    assert source_y[0, 0] == pytest.approx(0.0, abs=1e-4)
    assert source_x[-1, -1] == pytest.approx(9.0, abs=1e-4)
    assert source_y[-1, -1] == pytest.approx(9.0, abs=1e-4)


def test_projection_residual_warp_roundtrips_and_hash_is_enforced(tmp_path):
    source = tmp_path / "source.png"
    Image.new("RGB", (10, 10), "white").save(source)
    geographic = CRS.from_epsg(4326)
    wkt = geographic.to_wkt(version="WKT2_2019", pretty=False)
    to_web = Transformer.from_crs(geographic, "EPSG:3857", always_xy=True)
    minimum_x, minimum_y = to_web.transform(-124.0, 32.0)
    maximum_x, maximum_y = to_web.transform(-114.0, 42.0)
    grid = {
        "crs": "EPSG:3857",
        "bounds": [minimum_x, minimum_y, maximum_x, maximum_y],
        "width": 10,
        "height": 10,
    }
    candidate_to_source = np.asarray(
        [[9.0, 0.0, 4.5], [0.0, 9.0, 4.5], [0.0, 0.0, 1.0]]
    )
    residual = {
        "kind": "wendland_c2_reference_pixel_to_source_original_displacement",
        "coordinate_domain": "pinned_mapbox_target_grid_pixels",
        "displacement_range": "source_original_pixels",
        "centers_reference_px": [[4.5, 4.5]],
        "coefficients_source_px": [[0.5, -0.25]],
        "radius_reference_px": 20.0,
        "ridge": 0.1,
        "kernel": "max(1-r,0)^4*(4r+1)",
    }
    residual["sha256"] = hashlib.sha256(
        json.dumps(residual, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    transform = {
        "kind": "projection_aware_residual_warp_mapbox_registration",
        "source_pixel_space": "original_raster",
        "reference_pixel_space": "pinned_mapbox_target_grid",
        "source_original_shape": [10, 10],
        "projection": {
            "id": "geographic",
            "always_xy": True,
            "crs_wkt": wkt,
            "crs_wkt_sha256": hashlib.sha256(wkt.encode()).hexdigest(),
            "normalization_center": [-119.0, 37.0],
            "normalization_scale": 10.0,
        },
        "candidate_normalized_to_source_original_matrix": candidate_to_source.tolist(),
        "source_original_to_candidate_normalized_matrix": np.linalg.inv(
            candidate_to_source
        ).tolist(),
        "residual_warp": residual,
        "inverse_solver": {
            "kind": "base_projection_fixed_point",
            "maximum_iterations": 20,
            "reference_tolerance_px": 1e-4,
            "source_roundtrip_tolerance_px": 0.02,
            "failure_policy": "reject_nonconverged_in_domain_points",
        },
        "target_grid": grid,
    }
    alignment = tmp_path / "alignment.json"
    payload = {
        "decision": "accept",
        "source_sha256": _sha(source),
        "transform": transform,
    }
    alignment.write_text(json.dumps(payload))

    _load_accepted_alignment(alignment, source, grid)
    reference_x, _reference_y = _reference_to_source_remap(transform)
    assert reference_x[4, 4] > 4.0
    residual_consumer_diagnostics = {}
    source_x, source_y = _source_to_reference_remap(
        transform, (10, 10), diagnostics=residual_consumer_diagnostics
    )
    assert residual_consumer_diagnostics["converged"] is True
    assert (
        residual_consumer_diagnostics["maximum_final_reference_update_px"]
        <= 1e-4
    )
    assert (
        residual_consumer_diagnostics["maximum_source_roundtrip_error_px"]
        <= 0.02
    )
    forward = Transformer.from_crs("EPSG:3857", geographic, always_xy=True)
    recovered_x, recovered_y = _projection_reference_to_source_base(
        transform, source_x, source_y, forward
    )
    residual_x, residual_y = _residual_displacement(transform, source_x, source_y)
    yy, xx = np.indices((10, 10), dtype=np.float64)
    assert np.max(np.abs(recovered_x + residual_x - xx)) < 0.02
    assert np.max(np.abs(recovered_y + residual_y - yy)) < 0.02

    nonconvergent = json.loads(json.dumps(transform))
    nonconvergent["inverse_solver"]["maximum_iterations"] = 1
    nonconvergent["inverse_solver"]["reference_tolerance_px"] = 1e-8
    with pytest.raises(
        ValueError, match="did not reach reference tolerance"
    ):
        _source_to_reference_remap(nonconvergent, (10, 10))

    projective_payload = json.loads(json.dumps(payload))
    projective = np.asarray(
        [[9.0, 0.0, 4.5], [0.0, 9.0, 4.5], [0.01, 0.0, 1.0]]
    )
    projective_payload["transform"][
        "candidate_normalized_to_source_original_matrix"
    ] = projective.tolist()
    projective_payload["transform"][
        "source_original_to_candidate_normalized_matrix"
    ] = np.linalg.inv(projective).tolist()
    alignment.write_text(json.dumps(projective_payload))
    with pytest.raises(ValueError, match="requires affine matrices"):
        _load_accepted_alignment(alignment, source, grid)

    alignment.write_text(json.dumps(payload))
    authority = [
        {
            "automatic_iteration": 1,
            "counts_toward_automatic_iteration_count": True,
            "decision": "accept",
            "artifacts": [
                {
                    "path": str(tmp_path / "alignment-01" / "candidate.json"),
                    "sha256": _sha(alignment),
                }
            ],
        }
    ]
    coherent_mutation = json.loads(json.dumps(payload))
    coherent_residual = coherent_mutation["transform"]["residual_warp"]
    coherent_residual["coefficients_source_px"] = [[0.6, -0.25]]
    hash_payload = dict(coherent_residual)
    hash_payload.pop("sha256")
    coherent_residual["sha256"] = hashlib.sha256(
        json.dumps(hash_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    alignment.write_text(json.dumps(coherent_mutation))
    with pytest.raises(ValueError, match="pointer hash disagrees"):
        _load_accepted_alignment(
            alignment,
            source,
            grid,
            accepted_iteration_count=1,
            map_id="synthetic",
            alignment_iterations=authority,
        )

    payload["transform"]["residual_warp"]["coefficients_source_px"] = [[0.6, -0.25]]
    alignment.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="residual-warp hash"):
        _load_accepted_alignment(alignment, source, grid)


def test_observed_and_inferred_pixels_roundtrip_without_leaking_into_layout():
    rgb, domain, words = _synthetic_source()
    entries, _, _ = _detect_legend(rgb, domain, words)
    observed_ids, _, _ = _classify(rgb, domain, _palette_lab(entries), 2.0, 1.0)

    completed, inferred = _nearest_completion(observed_ids, domain)
    reconstruction = _render(completed, entries)
    roundtrip, _, _ = _classify(
        reconstruction, domain, _palette_lab(entries), 0.01, 0.0
    )

    assert np.count_nonzero(inferred) > 0
    assert not np.any(inferred & (observed_ids > 0))
    assert np.all(completed[domain] > 0)
    assert not np.any(completed[~domain])
    assert np.array_equal(roundtrip, completed)


def test_near_white_legend_class_is_never_assigned_outside_mapbox_land_domain():
    rgb, domain, words = _synthetic_source()
    entries, _, _ = _detect_legend(rgb, domain, words)

    observed, nearest, _ = _classify(
        rgb, domain, _palette_lab(entries), 6.0, 0.5
    )

    assert entries[0].rgb == (255, 255, 255)
    assert not np.any(observed[~domain])
    assert not np.any(nearest[~domain])


def test_target_source_coverage_marks_partial_raster_extent_as_nodata():
    transform = {
        "kind": "regular_global_mapbox_registration",
        "reference_pixel_to_source_original_matrix": np.eye(3).tolist(),
        "target_grid": {
            "crs": "EPSG:3857",
            "bounds": [0.0, 0.0, 9.0, 7.0],
            "width": 10,
            "height": 8,
        },
    }

    coverage = _target_source_coverage_mask(transform, (8, 6))

    assert np.all(coverage[:, :6])
    assert not np.any(coverage[:, 6:])


def test_sparse_domain_contains_every_broader_plausible_thematic_pixel():
    rgb = np.full((40, 50, 3), (205, 205, 205), dtype=np.uint8)
    rgb[5:12, 7:16] = (220, 40, 40)
    rgb[25:34, 31:44] = (40, 190, 60)
    land = np.ones(rgb.shape[:2], dtype=bool)
    entries = (
        LegendEntry(1, "red", (220, 40, 40), (0, 0, 1, 1), (0, 0, 1, 1), 100, "test"),
        LegendEntry(2, "green", (40, 190, 60), (0, 0, 1, 1), (0, 0, 1, 1), 100, "test"),
    )
    prototypes = tuple(value[None, :] for value in _palette_lab(entries))

    domain, diagnostics, direct, plausible = _automatic_classification_domain(
        rgb, land, entries, prototypes, ExtractionLoopConfig()
    )

    assert diagnostics["kind"] == "automatic_sparse_thematic_support"
    assert np.any(direct)
    assert np.any(plausible)
    assert not np.any(plausible & ~domain)
    assert np.array_equal(domain, plausible)


def test_chromaticity_gate_rejects_gray_ink_for_pale_blue_class():
    entries = (
        LegendEntry(1, "pale blue", (191, 209, 255), (0, 0, 1, 1), (0, 0, 1, 1), 100, "test"),
    )
    rgb = np.full((12, 18, 3), (209, 208, 213), dtype=np.uint8)
    rgb[3:9, 6:12] = entries[0].rgb
    lab = categorical_extraction.cv2.cvtColor(
        rgb, categorical_extraction.cv2.COLOR_RGB2LAB
    ).astype(np.float32)
    class_ids = np.ones(rgb.shape[:2], dtype=np.uint8)

    compatible = _palette_chromaticity_compatible(
        lab, class_ids, _palette_lab(entries)
    )

    assert np.all(compatible[3:9, 6:12])
    assert not np.any(compatible[:3])
    assert not np.any(compatible[9:])
    assert not np.any(compatible[:, :6])
    assert not np.any(compatible[:, 12:])


def test_sparse_domain_excludes_gray_prototype_contamination():
    gray = (209, 208, 213)
    pale_blue = (191, 209, 255)
    red = (220, 40, 40)
    rgb = np.full((40, 50, 3), gray, dtype=np.uint8)
    rgb[5:15, 7:19] = pale_blue
    rgb[24:34, 31:43] = red
    land = np.ones(rgb.shape[:2], dtype=bool)
    entries = (
        LegendEntry(1, "pale blue", pale_blue, (0, 0, 1, 1), (0, 0, 1, 1), 100, "test"),
        LegendEntry(2, "red", red, (0, 0, 1, 1), (0, 0, 1, 1), 100, "test"),
    )
    palette = _palette_lab(entries)
    gray_lab = categorical_extraction.cv2.cvtColor(
        np.asarray([[gray]], dtype=np.uint8),
        categorical_extraction.cv2.COLOR_RGB2LAB,
    )[0, 0].astype(np.float32)
    # Simulate a learned pale-blue prototype contaminated by grayscale
    # hillshade. The domain gate must still fail closed.
    prototypes = (
        np.stack((palette[0], gray_lab)),
        palette[1][None, :],
    )

    domain, diagnostics, direct, plausible = _automatic_classification_domain(
        rgb, land, entries, prototypes, ExtractionLoopConfig()
    )

    assert diagnostics["kind"] == "automatic_sparse_thematic_support"
    assert np.all(domain[5:15, 7:19])
    assert np.all(domain[24:34, 31:43])
    assert not np.any(domain[:5])
    assert not np.any(domain[15:24])
    assert not np.any(domain[34:])
    assert np.array_equal(domain, direct)
    assert np.array_equal(domain, plausible)
    chromaticity = diagnostics["chromaticity_gate"]
    assert chromaticity["applied"] is True
    assert chromaticity["direct_rejected_pixel_count"] == 1760
    assert chromaticity["plausible_rejected_pixel_count"] == 1760
    assert chromaticity["classes"][0]["direct_rejected_pixel_count"] == 1760


def test_per_class_support_gate_reports_a_rare_unsupported_legend_class():
    rgb = np.full((30, 40, 3), (220, 40, 40), dtype=np.uint8)
    rgb[4:6, 5:7] = (40, 190, 60)
    land = np.ones(rgb.shape[:2], dtype=bool)
    entries = (
        LegendEntry(1, "common", (220, 40, 40), (0, 0, 1, 1), (0, 0, 1, 1), 100, "test"),
        LegendEntry(2, "rare", (40, 190, 60), (0, 0, 1, 1), (0, 0, 1, 1), 100, "test"),
    )
    prototypes = tuple(value[None, :] for value in _palette_lab(entries))

    _, diagnostics, _, _ = _automatic_classification_domain(
        rgb, land, entries, prototypes, ExtractionLoopConfig()
    )

    support = diagnostics["per_class_support"]
    assert support["passed"] is False
    assert support["failed_class_ids"] == [2]
    assert support["classes"][0]["passed"] is True
    assert support["classes"][1]["direct_pixel_count"] == 4
    assert support["classes"][1]["plausible_pixel_count"] == 4
    assert support["classes"][1]["independent_direct_cluster_count"] == 1
    assert support["classes"][1]["support_tier"] == "insufficient"


def test_per_class_support_preserves_rare_spatially_corroborated_class():
    rgb = np.full((30, 40, 3), (220, 40, 40), dtype=np.uint8)
    rgb[4:6, 5:7] = (40, 190, 60)
    rgb[22:24, 31:33] = (40, 190, 60)
    land = np.ones(rgb.shape[:2], dtype=bool)
    entries = (
        LegendEntry(1, "common", (220, 40, 40), (0, 0, 1, 1), (0, 0, 1, 1), 100, "test"),
        LegendEntry(2, "rare", (40, 190, 60), (0, 0, 1, 1), (0, 0, 1, 1), 100, "test"),
    )
    prototypes = tuple(value[None, :] for value in _palette_lab(entries))

    _, diagnostics, _, _ = _automatic_classification_domain(
        rgb, land, entries, prototypes, ExtractionLoopConfig()
    )

    support = diagnostics["per_class_support"]
    assert support["passed"] is True
    assert support["failed_class_ids"] == []
    assert support["classes"][1]["direct_pixel_count"] == 8
    assert support["classes"][1]["independent_direct_cluster_count"] == 2
    assert support["classes"][1]["support_tier"] == "rare_spatially_corroborated"


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract is required")
def test_real_deer_complete_legend_has_ten_readable_classes(tmp_path):
    source = Path(__file__).parents[1] / "examples" / "deer.png"
    rgb = np.asarray(Image.open(source).convert("RGB"))

    legend = detect_legend(
        source, rgb, np.ones(rgb.shape[:2], dtype=bool), tmp_path / "deer"
    )

    assert [entry.label for entry in legend.entries] == [
        "Rocky Mountain Mule Deer",
        "Columbian Black-tailed Deer",
        "California Mule Deer",
        "Inyo Mule Deer",
        "Burro Deer",
        "Southern Mule Deer",
        "Rocky Mountain & Columbian Black-tailed Deer",
        "Columbian Black-tailed & California Mule Deer",
        "Southern Mule & Burro Deer",
        "Deer Rare or Absent",
    ]
    assert legend.entries[1].swatch_status == "recovered_regular_gap"
    assert legend.entries[-1].rgb == (255, 255, 255)
    assert (tmp_path / "deer" / "legend" / "swatch-sequence.json").is_file()


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract is required")
def test_real_forest_complete_legend_has_eight_classes_and_no_false_black(tmp_path):
    source = Path(__file__).parents[1] / "examples" / "forest.jpg"
    rgb = np.asarray(Image.open(source).convert("RGB"))

    legend = detect_legend(
        source, rgb, np.ones(rgb.shape[:2], dtype=bool), tmp_path / "forest"
    )

    assert [entry.label for entry in legend.entries] == [
        "Redwood",
        "Western hardwoods",
        "Douglas fir",
        "Ponderosa pine",
        "Fir-Spruce",
        "Pinyon-Juniper",
        "Lodgepole pine",
        "Chaparral",
    ]
    assert len(legend.entries) == 8
    assert all(max(entry.rgb) >= 90 for entry in legend.entries)
    assert not any(max(entry.rgb) <= 50 for entry in legend.entries)


def test_real_farms_preserves_detected_ragged_multicolumn_legend_geometry():
    source = Path(__file__).parents[1] / "examples" / "farmsv2.png"
    rgb = np.asarray(Image.open(source).convert("RGB"))

    swatches, diagnostics = _automatic_swatch_sequence(rgb)

    groups = {
        item["id"]: item for item in diagnostics["candidate_groups"]
    }
    selected_counts = sorted(
        len(groups[group_id]["swatches"])
        for group_id in diagnostics["selected_group_ids"]
    )
    assert diagnostics["selection_kind"] == "compatible_ragged_multicolumn"
    assert diagnostics["multicolumn_lattice"]["reason"] == "structurally_ragged"
    assert selected_counts == [13, 14, 18]
    assert len(swatches) == 45
    assert all(
        item.detection_kind != "recovered_regular_grid_slot" for item in swatches
    )


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract is required")
def test_real_farms_legend_has_45_unique_clean_actual_row_labels(tmp_path):
    source = Path(__file__).parents[1] / "examples" / "farmsv2.png"
    rgb = np.asarray(Image.open(source).convert("RGB"))

    legend = detect_legend(
        source, rgb, np.ones(rgb.shape[:2], dtype=bool), tmp_path / "farms"
    )

    labels = [entry.label for entry in legend.entries]
    assert labels == [
        "Wheat",
        "Miscellaneous Grain and Hay",
        "Rice",
        "Wild Rice",
        "Mixed Pasture",
        "Miscellaneous Grasses",
        "Alfalfa and Alfalfa Mixtures",
        "Young Perennials",
        "Cotton",
        "Beans (Dry)",
        "Sunflowers",
        "Miscellaneous Field Crops",
        "Safflower",
        "Corn, Sorghum and Sudan",
        "Tomatoes",
        "Cole Crops",
        "Lettuce/Leafy Greens",
        "Carrots",
        "Potatoes and Sweet Potatoes",
        "Melons, Squash and Cucumbers",
        "Flowers, Nursery and Christmas Tree Farms",
        "Bush Berries",
        "Onions and Garlic",
        "Strawberries",
        "Peppers",
        "Miscellaneous Truck Crops",
        "Citrus",
        "Dates",
        "Olives",
        "Kiwis",
        "Miscellaneous Subtropical Fruits",
        "Avocados",
        "Almonds",
        "Apples",
        "Plums, Prunes and Apricots",
        "Cherries",
        "Pistachios",
        "Pears",
        "Peaches/Nectarines",
        "Pomegranates",
        "Miscellaneous Deciduous",
        "Walnuts",
        "Grapes",
        "Greenhouse",
        "Idle",
    ]
    assert len({label.casefold() for label in labels}) == 45
    assert all(label and not label.endswith("|") for label in labels)
    assert "Beans (Dry)" in labels
    assert all(
        max(entry.swatch_bbox[1], entry.label_bbox[1])
        < min(
            entry.swatch_bbox[1] + entry.swatch_bbox[3],
            entry.label_bbox[1] + entry.label_bbox[3],
        )
        for entry in legend.entries
    )
    assert all(
        entry.swatch_status == "visible_regular_legend_component"
        for entry in legend.entries
    )


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract is required")
def test_real_plantzone_embedded_legend_has_thirteen_readable_classes(tmp_path):
    source = Path(__file__).parents[1] / "examples" / "plantzone.avif"
    rgb = np.asarray(Image.open(source).convert("RGB"))

    legend = detect_legend(
        source, rgb, np.ones(rgb.shape[:2], dtype=bool), tmp_path / "plantzone"
    )

    assert [entry.label for entry in legend.entries] == [
        "5a",
        "5b",
        "6a",
        "6b",
        "7a",
        "7b",
        "8a",
        "8b",
        "9a",
        "9b",
        "10a",
        "10b",
        "11a",
    ]


@pytest.mark.skipif(
    not HISTORICAL_RAINFALL_SOURCE.is_file(),
    reason="historical rainfall source was retired from the example corpus",
)
def test_real_historical_rainfall_recovers_complete_35_cell_legend_grid():
    source = HISTORICAL_RAINFALL_SOURCE
    rgb = np.asarray(Image.open(source).convert("RGB"))

    swatches, diagnostics = _automatic_swatch_sequence(rgb)

    assert diagnostics["selection_kind"] == "compatible_regular_multicolumn_grid"
    assert len(swatches) == 35
    assert len({item.bbox[0] for item in swatches}) == 5
    assert any(item.detection_kind == "recovered_regular_grid_slot" for item in swatches)


def test_decimal_legend_label_is_not_rejected_as_duplicate_prose():
    assert _label_is_clean("5.5")
    assert _label_is_clean("4.0 - 4.5")
    assert not _label_is_clean("Rain Rain")


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract is required")
@pytest.mark.skipif(
    not HISTORICAL_RAINFALL_SOURCE.is_file(),
    reason="historical rainfall source was retired from the example corpus",
)
def test_real_historical_rainfall_persists_35_rows_then_rejects_collapsed_textures(
    tmp_path,
):
    source = HISTORICAL_RAINFALL_SOURCE
    rgb = np.asarray(Image.open(source).convert("RGB"))
    output = tmp_path / "rainfall-historical"

    with pytest.raises(ValueError, match="source-indistinguishable rows"):
        detect_legend(source, rgb, np.ones(rgb.shape[:2], dtype=bool), output)

    legend = json.loads((output / "legend" / "legend.json").read_text())
    assert len(legend["entries"]) == 35
    assert [item["label"] for item in legend["entries"][:8]] == [
        "2.5",
        "3.5",
        "4.5",
        "5.0",
        "5.5",
        "6.5",
        "7.0",
        "7.5",
    ]
    texture = json.loads(
        (output / "legend" / "dither-texture-signatures.json").read_text()
    )
    assert len(texture["classes"]) == 35
    assert texture["ambiguous_pairs"]
    rejection = json.loads((output / "legend" / "legend-rejection.json").read_text())
    assert rejection["stage"] == "dither_texture_signatures"


def test_end_to_end_loop_logs_every_attempt_and_accepts_only_fixed_point(
    tmp_path, monkeypatch
):
    rgb, domain, words = _synthetic_source()
    source = tmp_path / "source.png"
    Image.fromarray(rgb, mode="RGB").save(source)
    grid = {
        "crs": "EPSG:3857",
        "bounds": [0.0, 0.0, 500.0, 360.0],
        "width": 500,
        "height": 360,
    }
    reference_pin = {"id": "synthetic-mapbox-reference", "manifest_sha256": "abc"}
    alignment = _alignment(
        tmp_path / "alignment.json",
        source,
        grid,
        mapbox_reference=reference_pin,
    )
    reference = SimpleNamespace(
        grid=grid,
        state_land=domain.copy(),
        water=np.zeros_like(domain),
        pin=reference_pin,
    )
    monkeypatch.setattr(
        "mapscan.automatic_categorical_extraction.load_pinned_mapbox_reference",
        lambda _: reference,
    )
    monkeypatch.setattr(
        "mapscan.automatic_categorical_extraction._run_tesseract_ocr",
        lambda _: (words, "synthetic tsv\n", "tesseract synthetic"),
    )
    def synthetic_legend(source_path, source_rgb, source_data_mask, output_dir):
        entries, axis, step = _detect_legend(source_rgb, source_data_mask, words)
        artifacts = _write_legend_artifacts(
            output_dir,
            source_rgb,
            entries,
            "synthetic tsv\n",
            "tesseract synthetic",
            axis,
            step,
        )
        return LegendDetection(tuple(entries), "tesseract synthetic", axis, step, artifacts)

    monkeypatch.setattr(
        "mapscan.automatic_categorical_extraction.detect_legend", synthetic_legend
    )
    log = _accepted_log(source, alignment)
    output = tmp_path / "automatic-extraction"
    log_md = tmp_path / "EXPERIMENT.md"
    log_json = tmp_path / "EXPERIMENT.json"

    result = run_automatic_categorical_extraction(
        source,
        alignment,
        tmp_path / "unused-manifest.json",
        output,
        log,
        log_md,
        log_json,
        config=ExtractionLoopConfig(
            policies=((2.0, 1.0), (2.0, 1.0)),
            minimum_observed_fraction=0.90,
            maximum_inferred_fraction=0.10,
            geographic_rows=3,
            geographic_columns=3,
            minimum_geographic_cells=3,
        ),
    )

    assert result.status == "accepted"
    assert result.accepted is not None and result.accepted.iteration == 2
    saved = json.loads(log_json.read_text())
    assert [item["decision"] for item in saved["extraction"]["iterations"]] == [
        "retry",
        "accept",
    ]
    assert saved["extraction"]["accepted_automatic_iteration_count"] == 2
    assert saved["final"]["status"] == "complete"
    assert (output / "accepted-extraction.json").is_file()
    assert (output / "extraction-02" / "source-observed-mask.png").is_file()
    assert (output / "extraction-02" / "source-inferred-mask.png").is_file()
    for attempt in saved["extraction"]["iterations"]:
        provenance = attempt["provenance"]
        assert provenance["actor_kind"] == "automated"
        assert provenance["manual_arrows"] is False
        assert provenance["manual_stamps"] is False
        assert provenance["human_approval"] is False


def test_end_to_end_partial_extent_never_completes_missing_state_area(
    tmp_path, monkeypatch
):
    source_height, source_width = 40, 60
    target_width = 100
    rgb = np.empty((source_height, source_width, 3), dtype=np.uint8)
    rgb[:, : source_width // 2] = (255, 255, 255)
    rgb[:, source_width // 2 :] = (200, 20, 20)
    source = tmp_path / "partial-source.png"
    Image.fromarray(rgb).save(source)
    grid = {
        "crs": "EPSG:3857",
        "bounds": [0.0, 0.0, float(target_width - 1), float(source_height - 1)],
        "width": target_width,
        "height": source_height,
    }
    reference_pin = {"id": "synthetic-mapbox-reference", "manifest_sha256": "abc"}
    alignment = _alignment(
        tmp_path / "accepted-alignment.json",
        source,
        grid,
        mapbox_reference=reference_pin,
    )
    reference = SimpleNamespace(
        grid=grid,
        state_land=np.ones((source_height, target_width), dtype=bool),
        water=np.zeros((source_height, target_width), dtype=bool),
        pin=reference_pin,
    )
    monkeypatch.setattr(
        "mapscan.automatic_categorical_extraction.load_pinned_mapbox_reference",
        lambda _: reference,
    )
    entries = (
        LegendEntry(1, "near-white class", (255, 255, 255), (0, 0, 5, 5), (0, 0, 1, 1), 100, "test"),
        LegendEntry(2, "red class", (200, 20, 20), (5, 0, 5, 5), (0, 0, 1, 1), 100, "test"),
    )

    def synthetic_partial_legend(source_path, source_rgb, source_data_mask, output_dir):
        legend_dir = output_dir / "legend"
        legend_dir.mkdir()
        legend_path = legend_dir / "legend.json"
        legend_path.write_text(
            json.dumps(
                {
                    "entries": [
                        {"class_id": entry.class_id, "label": entry.label, "rgb": entry.rgb}
                        for entry in entries
                    ]
                }
            )
            + "\n"
        )
        return LegendDetection(
            entries,
            "synthetic",
            0.0,
            1.0,
            (legend_path,),
        )

    monkeypatch.setattr(
        "mapscan.automatic_categorical_extraction.detect_legend",
        synthetic_partial_legend,
    )
    source_to_reference = categorical_extraction._source_to_reference
    remap_calls = []

    def checked_source_to_reference(
        values, transform, interpolation, border_value=0, remap=None
    ):
        assert isinstance(remap, tuple) and len(remap) == 2
        remap_calls.append(remap)
        return source_to_reference(
            values, transform, interpolation, border_value, remap
        )

    monkeypatch.setattr(
        categorical_extraction,
        "_source_to_reference",
        checked_source_to_reference,
    )
    log = _accepted_log(source, alignment)
    output = tmp_path / "partial-extraction"
    result = run_automatic_categorical_extraction(
        source,
        alignment,
        tmp_path / "unused-manifest.json",
        output,
        log,
        tmp_path / "EXPERIMENT.md",
        tmp_path / "EXPERIMENT.json",
        config=ExtractionLoopConfig(
            policies=((2.0, 1.0), (2.0, 1.0)),
            minimum_observed_fraction=0.99,
            maximum_inferred_fraction=0.01,
            geographic_rows=2,
            geographic_columns=2,
            minimum_geographic_cells=5,
        ),
    )

    assert result.status == "accepted"
    assert result.accepted is not None
    assert len(remap_calls) >= 5
    extent = result.accepted.scores["classification_domain"]["aligned_source_extent"]
    assert extent["partial_extent"] is True
    assert extent["missing_source_extent_pixel_count"] == source_height * (
        target_width - source_width
    )
    assert result.accepted.gates["missing_source_extent_remains_nodata"] is True
    geographic_gate = result.accepted.gates["geographically_balanced_observation"]
    assert geographic_gate["configured_full_extent_minimum"] == 5
    assert geographic_gate["available_extent_cell_count"] == 4
    assert geographic_gate["minimum"] == 4
    accepted_pointer = json.loads((output / "accepted-extraction.json").read_text())
    assert accepted_pointer["aligned_source_extent"] == extent
    web_ids = np.asarray(
        Image.open(output / "extraction-02" / "web-mercator-class-id.png")
    )
    assert np.all(web_ids[:, :source_width] > 0)
    assert not np.any(web_ids[:, source_width:])


def test_end_to_end_texture_loop_preserves_duplicate_median_legend_rows(
    tmp_path, monkeypatch
):
    height, width = 90, 120
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    yy, xx = np.indices((height, 75))
    first = np.empty((height, 75, 3), dtype=np.uint8)
    first[:] = (180, 120, 90)
    first[(xx + 2 * yy) % 5 >= 3] = (220, 150, 100)
    second = np.empty((height, 75, 3), dtype=np.uint8)
    second[:] = (180, 120, 90)
    second[(xx + 2 * yy + 1) % 5 >= 3] = (130, 90, 70)
    rgb[:45, :75] = first[:45]
    rgb[45:, :75] = second[45:]
    swatch_boxes = ((88, 12, 24, 18), (88, 52, 24, 18))
    for box, patch in zip(swatch_boxes, (first[:18, :24], second[:18, :24])):
        x, y, box_width, box_height = box
        rgb[y : y + box_height, x : x + box_width] = patch
    source = tmp_path / "dither.png"
    Image.fromarray(rgb).save(source)
    domain = np.zeros((height, width), dtype=bool)
    domain[:, :75] = True
    entries = (
        LegendEntry(1, "class one", (180, 120, 90), swatch_boxes[0], (0, 0, 1, 1), 100, "test"),
        LegendEntry(2, "class two", (180, 120, 90), swatch_boxes[1], (0, 0, 1, 1), 100, "test"),
    )
    texture_model = build_dither_texture_model(rgb, swatch_boxes)
    assert texture_model.is_distinguishable
    grid = {
        "crs": "EPSG:3857",
        "bounds": [0.0, 0.0, float(width), float(height)],
        "width": width,
        "height": height,
    }
    reference_pin = {"id": "synthetic-mapbox-reference", "manifest_sha256": "abc"}
    alignment = _alignment(
        tmp_path / "alignment.json", source, grid, mapbox_reference=reference_pin
    )
    reference = SimpleNamespace(
        grid=grid,
        state_land=domain.copy(),
        water=np.zeros_like(domain),
        pin=reference_pin,
    )
    monkeypatch.setattr(
        "mapscan.automatic_categorical_extraction.load_pinned_mapbox_reference",
        lambda _: reference,
    )

    def synthetic_texture_legend(source_path, source_rgb, source_data_mask, output_dir):
        artifacts = _write_legend_artifacts(
            output_dir,
            source_rgb,
            entries,
            "synthetic tsv\n",
            "tesseract synthetic",
            100,
            40,
            texture_model=texture_model,
        )
        return LegendDetection(
            entries,
            "tesseract synthetic",
            100,
            40,
            artifacts,
            texture_model,
        )

    monkeypatch.setattr(
        "mapscan.automatic_categorical_extraction.detect_legend",
        synthetic_texture_legend,
    )
    log = _accepted_log(source, alignment)
    result = run_automatic_categorical_extraction(
        source,
        alignment,
        tmp_path / "unused-manifest.json",
        tmp_path / "texture-extraction",
        log,
        tmp_path / "EXPERIMENT.md",
        tmp_path / "EXPERIMENT.json",
        config=ExtractionLoopConfig(
            policies=((12.0, 1.0), (12.0, 1.0)),
            minimum_observed_fraction=0.70,
            maximum_inferred_fraction=0.30,
            geographic_rows=2,
            geographic_columns=2,
            minimum_geographic_cells=2,
        ),
    )

    assert result.status == "accepted"
    assert result.accepted is not None
    assert result.accepted.scores["legend_class_observed_pixel_counts"]["class one"] > 0
    assert result.accepted.scores["legend_class_observed_pixel_counts"]["class two"] > 0
