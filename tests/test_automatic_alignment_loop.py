import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

import mapscan.automatic_alignment_loop as automatic_alignment
from mapscan.automatic_alignment_loop import (
    AlignmentLoopConfig,
    _balanced_split,
    _score_candidate,
    _semantic_full_line_validation,
    _source_semantic_evidence,
    _transform_contract,
    invalidate_alignment_acceptance_for_failed_gate,
    load_pinned_mapbox_reference,
    run_automatic_alignment_loop,
)
from mapscan.experiment_log import NoHumanExperimentLog
from mapscan.source_working_raster import prepare_source_working_raster


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reference(tmp_path, *, legacy=False):
    root = tmp_path / "reference"
    root.mkdir()
    shape = (120, 100)
    state = np.zeros(shape, dtype=np.uint8)
    cv2.rectangle(state, (12, 8), (87, 111), 1, 1)
    counties = np.zeros(shape, dtype=np.uint8)
    for x in (28, 46, 64):
        cv2.line(counties, (x, 12), (x, 107), 1, 1)
    for y in (32, 57, 82):
        cv2.line(counties, (16, y), (83, y), 1, 1)
    land = np.zeros(shape, dtype=np.uint8)
    cv2.rectangle(land, (13, 9), (86, 110), 1, -1)
    water = ~land.astype(bool)

    def save(name, mask, rgba=False):
        path = root / name
        if rgba:
            values = np.zeros((*shape, 4), dtype=np.uint8)
            values[mask.astype(bool)] = (100, 255, 150, 255)
            Image.fromarray(values).save(path)
        else:
            Image.fromarray(mask.astype(np.uint8) * 255).save(path)
        return {"path": name, "sha256": _sha256(path), "pixel_count": int(mask.sum())}

    artifacts = {
        "state_coast_overlay": save("state.png", state, True),
        "county_overlay": save("counties.png", counties, True),
        "state_land_mask": save("land.png", land),
        "water_mask": save("water.png", water),
    }
    manifest = {
        "schema_version": 1,
        "status": "pinned_reference",
        "kind": "mapbox_california_state_coast_water_counties",
        "authority": {
            "previous_mapscan_canonical_used": legacy,
            "county_png_used": False,
            "census_used": False,
        },
        "style": {"id": "mapbox/light-v11", "sha256": "a" * 64},
        "tileset": {"tilejson_sha256": "b" * 64},
        "tile_aggregate_sha256": "c" * 64,
        "zoom": 9,
        "target_grid": {
            "crs": "EPSG:3857",
            "bounds": [-1.0, -2.0, 3.0, 4.0],
            "width": shape[1],
            "height": shape[0],
        },
        "artifacts": artifacts,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return manifest_path, state.astype(bool), counties.astype(bool)


def _source(tmp_path, state, counties):
    rgb = np.full((*state.shape, 3), 255, dtype=np.uint8)
    rgb[state | counties] = 0
    path = tmp_path / "source.png"
    Image.fromarray(rgb).save(path)
    return path


def _config(**overrides):
    values = {
        "working_max_dimension": 128,
        "geographic_cells": (4, 4),
        "primary_samples_per_cell": 50,
        "county_samples_per_cell": 40,
        "global_iterations": 1,
        "global_population": 4,
        "local_iterations": 2,
        "minimum_primary_holdout_cells": 6,
        "minimum_county_holdout_cells": 4,
        "source_hypothesis_generation_limit": 1,
        "source_hypothesis_shortlist_size": 1,
        "source_hypothesis_coarse_iterations": 1,
        "source_hypothesis_coarse_population": 4,
    }
    values.update(overrides)
    return AlignmentLoopConfig(**values)


def test_reference_loader_rejects_any_legacy_authority(tmp_path):
    manifest_path, _, _ = _reference(tmp_path, legacy=True)

    with pytest.raises(ValueError, match="old canonical"):
        load_pinned_mapbox_reference(manifest_path)


def test_reference_loader_verifies_every_artifact_hash(tmp_path):
    manifest_path, _, _ = _reference(tmp_path)
    (manifest_path.parent / "state.png").write_bytes(b"tampered")

    with pytest.raises(ValueError, match="hash mismatch"):
        load_pinned_mapbox_reference(manifest_path)


def test_geographic_split_represents_each_usable_cell_in_train_and_holdout():
    mask = np.zeros((80, 100), dtype=bool)
    mask[5:75:2, 5:95:2] = True
    train, holdout, train_cells, holdout_cells = _balanced_split(mask, (4, 5), 30)

    assert len(train) and len(holdout)
    assert set(train_cells) == set(holdout_cells) == set(range(20))


def test_transform_contract_round_trips_original_and_reference_pixels():
    normalized_to_working = np.asarray(
        [[72.0, 0.0, 45.0], [0.0, 72.0, 58.0], [0.0, 0.0, 1.0]]
    )
    contract = _transform_contract(
        normalized_to_working,
        reference_center=np.asarray([50.0, 60.0]),
        reference_state_height=100.0,
        working_scale=0.5,
        source_original_shape=(240, 200),
        source_working_shape=(120, 100),
        target_grid={
            "crs": "EPSG:3857",
            "bounds": [0, 1, 2, 3],
            "width": 100,
            "height": 120,
        },
    )
    forward = np.asarray(contract["source_original_to_reference_pixel_matrix"])
    reverse = np.asarray(contract["reference_pixel_to_source_original_matrix"])

    assert np.allclose(forward @ reverse, np.eye(3))
    assert contract["resampling_contract"]["categorical_class_ids"] == "nearest"


def test_projection_normalization_unflips_image_y_before_inverse_projection(tmp_path):
    manifest_path, _, _ = _reference(tmp_path)
    reference = load_pinned_mapbox_reference(manifest_path)
    context = next(
        item
        for item in automatic_alignment._projection_contexts(reference)
        if item.id == "geographic"
    )
    points = np.asarray([[4.0, 7.0], [49.5, 59.5], [93.0, 112.0]])
    web_x, web_y = automatic_alignment._reference_pixels_to_web_mercator(
        points, reference.grid
    )
    projected_x, projected_y = context.reference_to_candidate.transform(web_x, web_y)
    projected = np.column_stack((projected_x, projected_y))
    normalized = automatic_alignment._projected_to_candidate_normalized(
        projected, context
    )
    recovered_projected = automatic_alignment._candidate_normalized_to_projected(
        normalized, context
    )

    assert np.allclose(recovered_projected, projected, atol=1e-10)
    assert np.all(
        np.sign(normalized[:, 1])
        == -np.sign(projected[:, 1] - context.normalization_center[1])
    )
    round_trip = automatic_alignment._projection_round_trip(
        context, np.asarray([[100.0, 0.0, 50.0], [0.0, 100.0, 60.0], [0.0, 0.0, 1.0]]), reference.grid
    )
    assert round_trip["finite"] is True
    assert round_trip["maximum_error_px"] < 1e-5
    contract = automatic_alignment._projection_transform_contract(
        np.asarray(
            [[100.0, 0.0, 50.0], [0.0, 100.0, 60.0], [0.0, 0.0, 1.0]]
        ),
        projection=context,
        working_scale=1.0,
        source_original_shape=(120, 100),
        source_working_shape=(120, 100),
        target_grid=reference.grid,
    )
    assert contract["projection"]["projected_axis_order"] == [
        "easting_right",
        "northing_up",
    ]
    assert contract["projection"]["candidate_normalized_axis_order"] == [
        "image_x_right",
        "image_y_down",
    ]


def test_regularity_rejects_collapsed_scale_even_with_positive_determinant():
    regularity = automatic_alignment._regularity(
        np.asarray([[0.05, 0.0, 50.0], [0.0, 0.05, 60.0], [0.0, 0.0, 1.0]]),
        model="similarity",
        config=_config(),
        source_shape=(120, 100),
    )

    assert regularity["determinant"] > 0
    assert regularity["minimum_scale_fraction"] < _config().minimum_scale_fraction
    assert regularity["passed"] is False


def test_loop_logs_each_regular_attempt_and_writes_reusable_accepted_transform(
    tmp_path, monkeypatch
):
    manifest_path, state, counties = _reference(tmp_path)
    source_path = _source(tmp_path, state, counties)
    reference = load_pinned_mapbox_reference(manifest_path)
    log = NoHumanExperimentLog(
        "synthetic-map",
        source_path,
        mapbox_reference=reference.pin,
        created_at="2026-08-30T12:00:00+00:00",
    )
    config = _config()
    monkeypatch.setattr(
        automatic_alignment,
        "_source_edges",
        lambda _rgb: state | counties,
    )
    monkeypatch.setattr(
        automatic_alignment, "PROJECTION_CANDIDATES", (("web_mercator", "EPSG:3857"),)
    )
    calls = []

    def fit_contract(model, source_shape, distance, evidence, supplied_config):
        calls.append(model)
        center = np.asarray([49.5, 59.5])
        state_height = 103.0
        x_shift = 10.0 if model == "similarity" else 0.0
        matrix = np.asarray(
            [[state_height, 0, center[0] + x_shift], [0, state_height, center[1]], [0, 0, 1]],
            dtype=float,
        )
        objective, summaries, gates, regularity = _score_candidate(
            matrix, model, distance, evidence, supplied_config
        )
        return matrix, objective, summaries, gates, regularity

    monkeypatch.setattr(automatic_alignment, "_fit_candidate", fit_contract)
    real_validation = automatic_alignment._semantic_full_line_validation

    def validate_contract(matrix, *args, **kwargs):
        scores, gates, rendered = real_validation(matrix, *args, **kwargs)
        if float(matrix[0, 2]) > 55.0:
            gates["synthetic_refinement_required"] = {
                "passed": False,
                "value": float(matrix[0, 2]),
                "maximum": 55.0,
            }
        return scores, gates, rendered

    monkeypatch.setattr(
        automatic_alignment, "_semantic_full_line_validation", validate_contract
    )
    output = tmp_path / "run"
    result = run_automatic_alignment_loop(
        source_path, manifest_path, output, log, config=config
    )

    assert calls[-2:] == ["similarity", "regular_affine"]
    assert result.status == "pass"
    assert result.accepted.iteration == 2
    assert log.data["alignment"]["accepted_automatic_iteration_count"] == 2
    assert all(
        item["provenance"]["manual_arrows"] is False
        and item["provenance"]["manual_stamps"] is False
        for item in log.data["alignment"]["iterations"]
    )
    accepted = json.loads((output / "accepted-alignment.json").read_text())
    assert accepted["transform"]["source_pixel_space"] == "original_raster"
    assert accepted["transform"]["target_grid"]["crs"] == "EPSG:3857"
    forward = np.asarray(
        accepted["transform"]["source_original_to_reference_pixel_matrix"]
    )
    reverse = np.asarray(
        accepted["transform"]["reference_pixel_to_source_original_matrix"]
    )
    assert np.allclose(forward @ reverse, np.eye(3))
    ranking = json.loads((output / "candidate-ranking.json").read_text())
    assert ranking["selection_policy"] == "global_best_passing_candidate"
    assert ranking["selected_ensemble_ordinal"] == 2


def test_loop_ranks_all_passing_candidates_instead_of_accepting_first(
    tmp_path, monkeypatch
):
    manifest_path, state, counties = _reference(tmp_path)
    source_path = _source(tmp_path, state, counties)
    reference = load_pinned_mapbox_reference(manifest_path)
    log = NoHumanExperimentLog(
        "synthetic-global-rank",
        source_path,
        mapbox_reference=reference.pin,
    )
    monkeypatch.setattr(
        automatic_alignment,
        "_source_edges",
        lambda _rgb: state | counties,
    )
    monkeypatch.setattr(
        automatic_alignment,
        "PROJECTION_CANDIDATES",
        (("web_mercator", "EPSG:3857"),),
    )

    real_evaluate = automatic_alignment._evaluate_alignment_candidate

    def force_both_to_pass(*args, **kwargs):
        evaluated = real_evaluate(*args, **kwargs)
        model = kwargs["model"]
        evaluated["objective"] = 10.0 if model == "similarity" else 1.0
        state_scores = evaluated["semantic_scores"]["state_coast"]
        state_scores["reference_to_source"]["median_px"] = (
            2.0 if model == "similarity" else 0.0
        )
        state_scores["reference_to_source"]["p90_px"] = (
            8.0 if model == "similarity" else 2.0
        )
        state_scores["f1"] = 0.70 if model == "similarity" else 0.95
        evaluated["summaries"]["silhouette"][
            "land_containment_fraction"
        ] = 0.95
        evaluated["gates"] = {
            name: (
                True
                if isinstance(value, bool)
                else {**value, "passed": True}
            )
            for name, value in evaluated["gates"].items()
        }
        evaluated["all_passed"] = True
        return evaluated

    monkeypatch.setattr(
        automatic_alignment,
        "_evaluate_alignment_candidate",
        force_both_to_pass,
    )
    output = tmp_path / "globally-ranked-run"
    result = run_automatic_alignment_loop(
        source_path,
        manifest_path,
        output,
        log,
        config=_config(),
    )

    assert result.status == "pass"
    assert result.accepted is not None
    assert result.accepted.model.endswith(":regular_affine")
    assert len(log.data["alignment"]["iterations"]) == 2
    assert log.data["alignment"]["iterations"][0]["all_gates_passed"] is True
    assert log.data["alignment"]["iterations"][0]["decision"] == "retry"
    assert log.data["alignment"]["iterations"][1]["decision"] == "accept"
    ranking = json.loads((output / "candidate-ranking.json").read_text())
    assert ranking["passing_candidate_count"] == 2
    assert ranking["selected_ensemble_ordinal"] == 2
    assert ranking["candidates"][0]["model"] == "regular_affine"


def test_loop_stops_deterministically_when_regular_affine_does_not_improve(
    tmp_path, monkeypatch
):
    manifest_path, state, counties = _reference(tmp_path)
    source_path = _source(tmp_path, state, counties)
    reference = load_pinned_mapbox_reference(manifest_path)
    log = NoHumanExperimentLog(
        "synthetic-blocked",
        source_path,
        mapbox_reference=reference.pin,
    )
    config = _config(minimum_relative_improvement=0.01)
    monkeypatch.setattr(automatic_alignment, "_source_edges", lambda _rgb: state | counties)
    monkeypatch.setattr(
        automatic_alignment, "PROJECTION_CANDIDATES", (("web_mercator", "EPSG:3857"),)
    )

    def unchanged_fit(model, source_shape, distance, evidence, supplied_config):
        matrix = np.asarray(
            [[103.0, 0, 100.0], [0, 103.0, 95.0], [0, 0, 1]], dtype=float
        )
        objective, summaries, gates, regularity = _score_candidate(
            matrix, model, distance, evidence, supplied_config
        )
        return matrix, objective, summaries, gates, regularity

    monkeypatch.setattr(automatic_alignment, "_fit_candidate", unchanged_fit)
    result = run_automatic_alignment_loop(
        source_path, manifest_path, tmp_path / "blocked-run", log, config=config
    )

    assert result.status == "blocked"
    assert result.stop_reason == "projection_and_regular_model_sequence_exhausted"
    assert len(log.data["alignment"]["iterations"]) == 2
    assert log.data["alignment"]["iterations"][-1]["decision"] == "blocked"
    assert log.data["alignment"]["accepted_automatic_iteration_count"] is None


def test_unsupported_partial_coverage_fails_instead_of_using_legacy_evidence():
    with pytest.raises(ValueError, match="full_or_most_state"):
        AlignmentLoopConfig(source_coverage="partial_state")


def test_source_hypothesis_attempt_directory_is_crash_safe(tmp_path):
    output = tmp_path / "automatic-alignment"
    output.mkdir()
    first = automatic_alignment._allocate_source_hypothesis_root(output, 11)
    assert first.name == "source-hypotheses-start-11"
    first.mkdir()
    second = automatic_alignment._allocate_source_hypothesis_root(output, 11)
    assert second.name == "source-hypotheses-start-11-attempt-02"
    second.mkdir()
    third = automatic_alignment._allocate_source_hypothesis_root(output, 11)
    assert third.name == "source-hypotheses-start-11-attempt-03"


@pytest.mark.parametrize(
    ("filename", "expected_canvas_kind"),
    (
        ("rainfall.png", "exclude_right_ui"),
        ("farmsv2.png", "exclude_right_ui_and_header"),
    ),
)
def test_real_ui_map_hypotheses_preserve_full_working_canvas(
    tmp_path, filename, expected_canvas_kind
):
    source = Path(__file__).parents[1] / "examples" / filename
    config = AlignmentLoopConfig(
        working_max_dimension=900,
        source_hypothesis_generation_limit=8,
    )
    original = automatic_alignment._load_categorical_raster(source)
    rgb, scale = automatic_alignment._resize_working(
        original, config.working_max_dimension
    )
    semantic = _source_semantic_evidence(rgb)

    variants = automatic_alignment._generate_alignment_source_hypotheses(
        source,
        tmp_path / filename.replace(".", "-"),
        rgb,
        semantic,
        working_scale=scale,
        config=config,
    )

    assert variants[0].id == "baseline-semantic"
    assert any(
        item.diagnostics.get("canvas_kind") == expected_canvas_kind
        for item in variants
        if item.variant_kind == "roi_clipped_cartographic_evidence"
    )
    assert any(item.variant_kind == "source_support_boundary" for item in variants)
    for item in variants:
        assert item.semantic.state_coast.shape == rgb.shape[:2]
        assert item.semantic.foreground_interior.shape == rgb.shape[:2]


def test_ordered_gradient_family_combines_filled_state_with_source_boundary(
    tmp_path,
):
    source = Path(__file__).parents[1] / "examples" / "quake.jpg"
    config = AlignmentLoopConfig(
        working_max_dimension=900,
        source_hypothesis_generation_limit=8,
    )
    original = automatic_alignment._load_categorical_raster(source)
    rgb, scale = automatic_alignment._resize_working(
        original, config.working_max_dimension
    )
    semantic = _source_semantic_evidence(rgb)

    variants = automatic_alignment._generate_alignment_source_hypotheses(
        source,
        tmp_path / "quake",
        rgb,
        semantic,
        working_scale=scale,
        config=config,
        source_family="ordered_gradient_bands",
    )

    family = next(
        item
        for item in variants
        if item.id == "source-family--warm-ordered-gradient-map-panel-v1"
    )
    hybrids = [
        item
        for item in variants
        if item.variant_kind == "source_family_boundary_hybrid"
    ]
    assert 1 <= len(hybrids) <= 3
    assert all(
        np.array_equal(
            item.semantic.foreground_interior,
            family.semantic.foreground_interior,
        )
        for item in hybrids
    )
    assert any(
        not np.array_equal(item.semantic.state_coast, family.semantic.state_coast)
        for item in hybrids
    )
    assert all(
        item.diagnostics["mapbox_used_for_hybrid_construction"] is False
        for item in hybrids
    )


def test_labeled_lambert_family_explicitly_logs_absent_county_channel(tmp_path):
    source = Path(__file__).parents[1] / "examples" / "elevation.gif"
    adapter_root = tmp_path / "source"
    working = prepare_source_working_raster(source, adapter_root)
    config = AlignmentLoopConfig(
        working_max_dimension=900,
        source_hypothesis_generation_limit=8,
    )
    original = automatic_alignment._load_categorical_raster(
        working.alignment_input_path
    )
    rgb, scale = automatic_alignment._resize_working(
        original, config.working_max_dimension
    )
    semantic = _source_semantic_evidence(rgb)

    variants = automatic_alignment._generate_alignment_source_hypotheses(
        working.alignment_input_path,
        tmp_path / "elevation",
        rgb,
        semantic,
        working_scale=scale,
        config=config,
        source_family="continuous_numeric_ramp",
    )

    family = next(
        item
        for item in variants
        if item.id == "source-family--labeled-lambert-boundary-and-pale-pacific-v1"
    )
    assert family.semantic.county_observability_override == "absent"
    assert family.semantic.source_adapter_id == (
        "labeled-lambert-boundary-and-pale-pacific-v1"
    )
    assert np.count_nonzero(family.semantic.state_coast) > 100
    assert family.semantic.state_coast.shape == rgb.shape[:2]


def test_rivers_family_separates_source_hydrography_from_state_mask(tmp_path):
    source = Path(__file__).parents[1] / "examples" / "rivers.jpg"
    original = automatic_alignment._load_categorical_raster(source)
    rgb, _scale = automatic_alignment._resize_working(original, 900)
    semantic = _source_semantic_evidence(rgb)

    family = automatic_alignment._family_semantic_hypothesis(
        rgb,
        semantic,
        "named_linear_and_polygon_features_without_legend",
        tmp_path,
    )

    assert family is not None
    assert family.semantic.source_adapter_id == (
        "closed-california-region-and-hydrography-v3"
    )
    assert family.diagnostics["closed_region"][
        "mapbox_used_for_region_construction"
    ] is False
    assert 0.15 < np.mean(family.semantic.foreground_interior) < 0.45
    assert family.semantic.hydrography is not None
    assert np.count_nonzero(family.semantic.hydrography) > 100
    assert np.mean(family.semantic.hydrography) < 0.10
    assert not np.array_equal(
        family.semantic.hydrography,
        family.semantic.state_coast,
    )
    assert "saturated_blue_hydrography_auxiliary" in family.diagnostics[
        "evidence_channels"
    ]
    assert (
        "saturated_blue_colorado_scoped_to_southeast_perimeter"
        in family.diagnostics["evidence_channels"]
    )

    # The pale Pacific is not allowed into the saturated line channel.
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    pale_pacific = (
        (hsv[:, :, 0] >= 85)
        & (hsv[:, :, 0] <= 125)
        & (hsv[:, :, 1] < 60)
    )
    pale_fraction = np.count_nonzero(
        family.semantic.hydrography & pale_pacific
    ) / np.count_nonzero(family.semantic.hydrography)
    # Three-pixel line bridging may cross a few low-saturation antialias
    # pixels, but the broad Pacific component must not dominate the channel.
    assert pale_fraction < 0.20


def test_farms_family_reports_partial_neutral_pacific_without_mapbox(tmp_path):
    source = Path(__file__).parents[1] / "examples" / "farmsv2.png"
    original = automatic_alignment._load_categorical_raster(source)
    rgb, _scale = automatic_alignment._resize_working(original, 900)
    semantic = _source_semantic_evidence(rgb)

    family = automatic_alignment._family_semantic_hypothesis(
        rgb,
        semantic,
        "categorical_sparse",
        tmp_path,
    )

    assert family is not None
    assert family.semantic.source_adapter_id == (
        "farms-partial-california-topology-v2"
    )
    assert family.diagnostics["source_perimeter_capability"] == "partial"
    assert family.diagnostics["pacific"][
        "mapbox_used_for_pacific_selection"
    ] is False
    assert family.diagnostics["layout_exclusion_working"] is not None
    assert np.count_nonzero(family.semantic.state_coast) > 100
    assert np.count_nonzero(family.semantic.counties) > 100
    assert not np.any(
        family.semantic.counties & ~family.semantic.foreground_interior
    )
    assert not np.any(
        family.semantic.counties & family.semantic.state_coast
    )
    assert family.diagnostics["partial_topology"][
        "mapbox_used_for_hypothesis_construction"
    ] is False
    assert family.diagnostics["partial_topology"]["authority"][
        "county_png_used"
    ] is False


def test_sparse_family_does_not_invent_neutral_pacific_for_forest(tmp_path):
    source = Path(__file__).parents[1] / "examples" / "forest.jpg"
    original = automatic_alignment._load_categorical_raster(source)
    rgb, _scale = automatic_alignment._resize_working(original, 900)
    semantic = _source_semantic_evidence(rgb)

    family = automatic_alignment._family_semantic_hypothesis(
        rgb,
        semantic,
        "categorical_sparse",
        tmp_path,
    )

    assert family is None


def test_official_pinned_reference_exposes_label_free_waterway_channel():
    manifest = (
        Path(__file__).parents[1]
        / "reference"
        / "mapbox-light-v11-california-z9-v1"
        / "manifest.json"
    )
    reference = load_pinned_mapbox_reference(manifest)

    assert reference.waterways is not None
    assert reference.waterways.shape == reference.state_coast.shape
    assert np.count_nonzero(reference.waterways) > 100


def test_tesseract_tsv_decoder_replaces_non_utf8_diagnostics(monkeypatch, tmp_path):
    calls = []

    class Completed:
        returncode = 0
        stdout = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"

    monkeypatch.setattr(automatic_alignment.shutil, "which", lambda _name: "/ocr")

    def run(*args, **kwargs):
        calls.append((args, kwargs))
        return Completed()

    monkeypatch.setattr(automatic_alignment.subprocess, "run", run)
    image_path = tmp_path / "crop.png"
    Image.new("L", (4, 4)).save(image_path)

    assert automatic_alignment._tesseract_words(
        image_path, page_segmentation_mode=6
    ) == []
    assert calls[0][1]["encoding"] == "utf-8"
    assert calls[0][1]["errors"] == "replace"


def test_shortlist_reserves_projection_aware_source_family_slot(monkeypatch):
    shape = (24, 20)
    empty = np.zeros(shape, dtype=bool)
    semantic = automatic_alignment.SourceSemanticEvidence(
        state_coast=empty,
        counties=empty,
        dark_cartographic_ink=empty,
        border_connected_water=empty,
        foreground_interior=empty,
        foreground_boundary=empty,
    )

    def hypothesis(identifier, kind):
        return automatic_alignment.AlignmentSourceHypothesis(
            id=identifier,
            variant_kind=kind,
            semantic=semantic,
            source_only_score=None,
            roi_working=(0, 0, shape[1], shape[0]),
            generator_hypothesis_id=None,
            diagnostics={},
        )

    hypotheses = (
        hypothesis("baseline-semantic", "baseline_semantic"),
        hypothesis("generic-best", "source_support_boundary"),
        hypothesis("generic-second", "source_support_boundary"),
        hypothesis("family-native-projection", "source_family_semantic_adapter"),
    )

    def evaluate(item, **_kwargs):
        objectives = {
            "generic-best": 1.0,
            "generic-second": 2.0,
            "family-native-projection": 50.0,
            "baseline-semantic": 100.0,
        }
        return {
            "objective": objectives[item.id],
            "gates": {"semantic_full_state_median": {"passed": False}},
            "all_passed": False,
            "matrix": np.eye(3),
            "semantic_scores": {},
        }

    monkeypatch.setattr(
        automatic_alignment, "_evaluate_alignment_candidate", evaluate
    )
    projection = type("Projection", (), {"id": "web_mercator"})()
    selected, _ = automatic_alignment._coarse_source_hypothesis_shortlist(
        hypotheses,
        projections=(projection,),
        source_shape=shape,
        pixel_evidence={},
        reference=object(),
        reference_center=np.zeros(2),
        reference_state_height=1.0,
        config=_config(source_hypothesis_shortlist_size=3),
    )

    assert [item.id for item in selected] == [
        "baseline-semantic",
        "generic-best",
        "family-native-projection",
    ]


def test_shortlist_runs_strict_passing_source_family_before_baseline(monkeypatch):
    shape = (24, 20)
    empty = np.zeros(shape, dtype=bool)
    semantic = automatic_alignment.SourceSemanticEvidence(
        state_coast=empty,
        counties=empty,
        dark_cartographic_ink=empty,
        border_connected_water=empty,
        foreground_interior=empty,
        foreground_boundary=empty,
    )

    def hypothesis(identifier, kind):
        return automatic_alignment.AlignmentSourceHypothesis(
            id=identifier,
            variant_kind=kind,
            semantic=semantic,
            source_only_score=None,
            roi_working=(0, 0, shape[1], shape[0]),
            generator_hypothesis_id=None,
            diagnostics={},
        )

    hypotheses = (
        hypothesis("baseline-semantic", "baseline_semantic"),
        hypothesis("family-strict-pass", "source_family_semantic_adapter"),
    )

    def evaluate(item, **_kwargs):
        passed = item.id == "family-strict-pass"
        return {
            "objective": 1.0 if passed else 100.0,
            "gates": {"semantic_full_state_median": {"passed": passed}},
            "all_passed": passed,
            "matrix": np.eye(3),
            "semantic_scores": {},
        }

    monkeypatch.setattr(
        automatic_alignment, "_evaluate_alignment_candidate", evaluate
    )
    projection = type("Projection", (), {"id": "web_mercator"})()
    selected, _ = automatic_alignment._coarse_source_hypothesis_shortlist(
        hypotheses,
        projections=(projection,),
        source_shape=shape,
        pixel_evidence={},
        reference=object(),
        reference_center=np.zeros(2),
        reference_state_height=1.0,
        config=_config(source_hypothesis_shortlist_size=2),
    )

    assert [item.id for item in selected] == [
        "family-strict-pass",
        "baseline-semantic",
    ]


def test_semantic_full_line_gate_rejects_bright_thematic_distractor_edges(tmp_path):
    manifest_path, state, counties = _reference(tmp_path)
    reference = load_pinned_mapbox_reference(manifest_path)
    rgb = np.full((*state.shape, 3), 255, dtype=np.uint8)
    rgb[state | counties] = 0
    shifted = np.zeros(state.shape, dtype=np.uint8)
    shifted[:, 10:] = (state | counties)[:, :-10]
    rgb[shifted > 0] = (245, 45, 45)
    semantic = _source_semantic_evidence(rgb)
    center = np.asarray([49.5, 59.5])
    state_height = 103.0
    wrong = np.asarray(
        [[state_height, 0, center[0] + 10], [0, state_height, center[1]], [0, 0, 1]],
        dtype=float,
    )

    scores, gates, _ = _semantic_full_line_validation(
        wrong,
        reference,
        semantic,
        center,
        state_height,
        _config(),
    )

    assert scores["source_capabilities"]["counties"]["status"] == "observable"
    assert gates["semantic_full_county_median"]["passed"] is False
    assert gates["semantic_full_county_symmetric_overlap"]["passed"] is False


def test_absent_county_channel_is_logged_and_county_gates_are_not_applicable(
    tmp_path,
):
    manifest_path, state, _ = _reference(tmp_path)
    reference = load_pinned_mapbox_reference(manifest_path)
    empty = np.zeros_like(state)
    semantic = automatic_alignment.SourceSemanticEvidence(
        state_coast=state,
        counties=empty,
        dark_cartographic_ink=empty,
        border_connected_water=empty,
        foreground_interior=reference.state_land,
        foreground_boundary=state,
    )
    matrix = np.asarray(
        [[103.0, 0.0, 49.5], [0.0, 103.0, 59.5], [0.0, 0.0, 1.0]]
    )

    scores, gates, _ = _semantic_full_line_validation(
        matrix,
        reference,
        semantic,
        np.asarray([49.5, 59.5]),
        103.0,
        _config(),
    )

    capability = scores["source_capabilities"]["counties"]
    assert capability["status"] == "absent"
    assert capability["required_for_acceptance"] is False
    assert gates["source_county_channel_observability"]["value"] == "absent"
    for name in (
        "semantic_full_county_median",
        "semantic_full_county_support",
        "semantic_full_county_symmetric_overlap",
    ):
        assert gates[name]["passed"] is True
        assert gates[name]["status"] == "not_applicable"
        assert gates[name]["observed_gate"]["passed"] is False


def test_fragmented_plantzone_like_labels_are_not_a_county_network(tmp_path):
    manifest_path, state, _ = _reference(tmp_path)
    reference = load_pinned_mapbox_reference(manifest_path)
    fragments = np.zeros(state.shape, dtype=np.uint8)
    for x, y, label in (
        (18, 24, "A"),
        (42, 24, "B"),
        (67, 24, "C"),
        (20, 48, "D"),
        (64, 48, "E"),
        (18, 74, "F"),
        (43, 74, "G"),
        (68, 74, "H"),
        (28, 98, "I"),
        (59, 98, "J"),
    ):
        cv2.putText(
            fragments,
            label,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.30,
            1,
            1,
            cv2.LINE_8,
        )
        cv2.circle(fragments, (x + 7, y - 3), 1, 1, -1)
    empty = np.zeros_like(state)
    semantic = automatic_alignment.SourceSemanticEvidence(
        state_coast=state,
        counties=fragments,
        dark_cartographic_ink=fragments,
        border_connected_water=empty,
        foreground_interior=reference.state_land,
        foreground_boundary=state,
    )
    matrix = np.asarray(
        [[103.0, 0.0, 49.5], [0.0, 103.0, 59.5], [0.0, 0.0, 1.0]]
    )

    scores, gates, _ = _semantic_full_line_validation(
        matrix,
        reference,
        semantic,
        np.asarray([49.5, 59.5]),
        103.0,
        _config(),
    )

    capability = scores["source_capabilities"]["counties"]
    assert capability["source_to_reference_line_ratio"] >= 0.10
    assert capability["occupied_cell_count"] >= 5
    assert capability["connected_internal_network_observed"] is False
    assert capability["topology"]["qualifying_component_count"] == 0
    assert capability["status"] == "absent"
    for name in (
        "semantic_full_county_median",
        "semantic_full_county_support",
        "semantic_full_county_symmetric_overlap",
    ):
        assert gates[name]["passed"] is True
        assert gates[name]["status"] == "not_applicable"


def test_balanced_state_tail_tolerates_one_stylized_region_at_same_pixel_limit(
    tmp_path,
):
    manifest_path, state, counties = _reference(tmp_path)
    reference = load_pinned_mapbox_reference(manifest_path)
    stylized = state.copy()
    stylized[111, 12:88] = False
    stylized[96, 12:88] = True
    empty = np.zeros_like(state)
    semantic = automatic_alignment.SourceSemanticEvidence(
        state_coast=stylized,
        counties=counties,
        dark_cartographic_ink=counties,
        border_connected_water=empty,
        foreground_interior=reference.state_land,
        foreground_boundary=stylized,
    )
    matrix = np.asarray(
        [[103.0, 0.0, 49.5], [0.0, 103.0, 59.5], [0.0, 0.0, 1.0]]
    )

    scores, gates, _ = _semantic_full_line_validation(
        matrix,
        reference,
        semantic,
        np.asarray([49.5, 59.5]),
        103.0,
        _config(),
    )

    assert scores["state_coast"]["reference_to_source"]["p90_px"] > 12.0
    tail = scores["state_geographically_balanced_tail"]
    assert tail["pixel_limit_px"] == 12.0
    assert tail["cell_pass_fraction"] >= 0.70
    assert gates["semantic_full_state_balanced_tail"]["passed"] is True


def test_balanced_state_tail_rejects_region_wide_drift(tmp_path):
    manifest_path, state, counties = _reference(tmp_path)
    reference = load_pinned_mapbox_reference(manifest_path)
    shifted = np.zeros_like(state)
    shifted[:, 14:] = state[:, :-14]
    empty = np.zeros_like(state)
    semantic = automatic_alignment.SourceSemanticEvidence(
        state_coast=shifted,
        counties=counties,
        dark_cartographic_ink=counties,
        border_connected_water=empty,
        foreground_interior=reference.state_land,
        foreground_boundary=shifted,
    )
    matrix = np.asarray(
        [[103.0, 0.0, 49.5], [0.0, 103.0, 59.5], [0.0, 0.0, 1.0]]
    )

    scores, gates, _ = _semantic_full_line_validation(
        matrix,
        reference,
        semantic,
        np.asarray([49.5, 59.5]),
        103.0,
        _config(),
    )

    tail = scores["state_geographically_balanced_tail"]
    assert tail["cell_pass_fraction"] < 0.70
    assert min(tail["row_pass_fractions"].values()) < 0.50
    assert gates["semantic_full_state_balanced_tail"]["passed"] is False


def test_premature_acceptance_can_be_invalidated_and_resumed(tmp_path, monkeypatch):
    manifest_path, state, counties = _reference(tmp_path)
    source_path = _source(tmp_path, state, counties)
    reference = load_pinned_mapbox_reference(manifest_path)
    log = NoHumanExperimentLog("resume", source_path, mapbox_reference=reference.pin)
    log.record_alignment_iteration(
        scores={"legacy_one_way_distance": 1.0},
        gates={"legacy_one_way_gate": True},
        decision="accept",
        provenance=automatic_alignment.automatic_provenance(
            "old-automatic-aligner", ["original_source", "mapbox_reference"]
        ),
        method="old one-way candidate",
    )

    invalidated = invalidate_alignment_acceptance_for_failed_gate(
        log,
        gate_name="semantic_full_county_support",
        gate={"passed": False, "value": 0.2, "minimum": 0.58},
        score_name="semantic_revalidation",
        score={"county_support": 0.2},
        reason="Full-line semantic revalidation rejected the one-way match.",
    )

    assert invalidated["decision"] == "retry"
    assert invalidated["automatic_iteration"] == 1
    assert log.data["alignment"]["accepted_automatic_iteration_count"] is None

    monkeypatch.setattr(
        automatic_alignment,
        "PROJECTION_CANDIDATES",
        (("web_mercator", "EPSG:3857"),),
    )

    def fit_correct(model, source_shape, distance, evidence, supplied_config):
        matrix = np.asarray(
            [[103.0, 0.0, 49.5], [0.0, 103.0, 59.5], [0.0, 0.0, 1.0]]
        )
        objective, summaries, gates, regularity = _score_candidate(
            matrix, model, distance, evidence, supplied_config
        )
        return matrix, objective, summaries, gates, regularity

    monkeypatch.setattr(automatic_alignment, "_fit_candidate", fit_correct)
    result = run_automatic_alignment_loop(
        source_path,
        manifest_path,
        tmp_path / "resumed-run",
        log,
        config=_config(),
    )

    assert result.status == "pass"
    assert result.accepted.iteration == 3
    assert [
        item["automatic_iteration"] for item in log.data["alignment"]["iterations"]
    ] == [1, 2, 3]
