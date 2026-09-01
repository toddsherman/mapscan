from __future__ import annotations

from dataclasses import replace
import inspect
from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import distance_transform_edt

import mapscan.farms_nonrigid_alignment as farms_alignment

from mapscan.automatic_alignment_loop import (
    _matrix_from_parameters,
    _projection_contexts,
    load_pinned_mapbox_reference,
)
from mapscan.elevation_nonrigid_alignment import CompactResidualWarp
from mapscan.farms_nonrigid_alignment import (
    EXPECTED_MAPBOX_V2_GRID,
    EXPECTED_MAPBOX_V2_MANIFEST_SHA256,
    FarmsNonrigidConfig,
    _load_source,
    _named_pacific_holdouts,
    _prepare_water_seed_training_context,
    _require_seed_inside_scale_interval,
    _require_pinned_v2_reference,
    _source_observable_extents,
    _source_extent_membership,
    _supported_coast_gate,
    _topology_preserving_binary_reduce,
    _bounded_validation_state,
    _balanced_county_validation_report,
    _county_validation_geographic_cells,
    _water_overlay,
    _water_seed_training_score,
    build_mapbox_farms_semantics,
    derive_water_residual_controls,
    serialize_farms_nonrigid_transform,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "examples" / "farmsv2.png"
REFERENCE = ROOT / "reference" / "mapbox-light-v11-california-z9-v2" / "manifest.json"


def _reference():
    result = load_pinned_mapbox_reference(REFERENCE)
    _require_pinned_v2_reference(result)
    return result


def test_thin_native_water_survives_binary_area_reduction():
    native = np.zeros((12, 12), dtype=bool)
    native[5, 5] = True

    reduced = _topology_preserving_binary_reduce(native, (3, 3))

    assert reduced.shape == (3, 3)
    assert np.count_nonzero(reduced) == 1


def test_bounded_shortlist_is_exactly_three_by_three_by_twelve():
    config = FarmsNonrigidConfig()
    assert config.seed_scale_modes == (
        ("partial-low", 0.55, 1.0, 3513, "source_partial_extent"),
        ("partial-high", 1.2, 2.4, 3512, "source_partial_extent"),
        ("canvas-high", 1.2, 2.4, 3512, "canvas_bounds_hypothesis"),
    )
    assert (
        len(config.projection_ids)
        * len(config.seed_scale_modes)
        * len(config.county_residual_kernel_radii_reference_px)
        * len(config.county_residual_ridge_values)
        == 108
    )
    assert _require_seed_inside_scale_interval(
        {"parameters": [0.0, 0.0, 0.75, 0.0]}, (0.55, 1.0)
    ) == 0.75
    with pytest.raises(ValueError, match="escaped"):
        _require_seed_inside_scale_interval(
            {"parameters": [0.0, 0.0, 1.1, 0.0]}, (0.55, 1.0)
        )


def test_exact_corrected_mapbox_v2_pin_is_required(tmp_path):
    reference = _reference()
    assert reference.pin["manifest_sha256"] == EXPECTED_MAPBOX_V2_MANIFEST_SHA256
    assert reference.state_land.shape == EXPECTED_MAPBOX_V2_GRID

    changed_manifest = tmp_path / "manifest.json"
    changed_manifest.write_bytes(REFERENCE.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="exact corrected Mapbox-v2 manifest"):
        _require_pinned_v2_reference(
            replace(reference, manifest_path=changed_manifest)
        )


def test_native_resolution_water_component_and_reduction_are_auditable():
    source = _load_source(SOURCE, FarmsNonrigidConfig())
    diagnostics = source.water_diagnostics
    coastal = [
        item
        for item in diagnostics["internal_components"]
        if item["coastal_internal_water"]
    ]

    assert diagnostics["native_source_shape"] == [5500, 4250]
    assert diagnostics["suisun_delta_observability"] == "supported_native_resolution_strict"
    assert diagnostics["binary_reduction"].startswith("float32_cv2_inter_area")
    assert len(coastal) == 1
    assert coastal[0]["bbox_original"] == [894, 2219, 318, 112]
    assert coastal[0]["pixel_count"] == 6782
    assert np.count_nonzero(source.native_internal_water_original) == 6782
    assert np.count_nonzero(source.native_internal_water) == 292


def test_partial_source_extent_omits_only_unobserved_layout_not_negative_geography():
    source = _load_source(SOURCE, FarmsNonrigidConfig())
    masks, diagnostics = _source_observable_extents(source, FarmsNonrigidConfig())
    panel = masks["map_panel"]
    positive = masks["california_positive_support"]
    county = masks["county_lines"]
    margin = diagnostics["page_margin_working_px"]

    assert not np.any(panel & source.topology.layout_exclusion)
    assert not np.any(positive & source.topology.layout_exclusion)
    assert not np.any(positive & source.pacific)
    assert not np.any(positive & source.topology.neighboring_region)
    assert not np.any(positive & source.native_internal_water)
    assert np.count_nonzero(panel & source.pacific) > 90_000
    assert np.count_nonzero(panel & source.topology.neighboring_region) > 20_000
    assert np.array_equal(county, source.topology.county_scope)
    assert not np.any(county & source.topology.layout_exclusion)
    assert not np.any(county & source.pacific)
    assert not np.any(county & source.topology.neighboring_region)
    assert not np.any(
        source.topology.internal_topology
        & source.topology.ambiguous_topology_exclusion
    )
    assert diagnostics["county_line_visibility_method"].startswith("source_only")
    assert (
        diagnostics["ambiguous_lower_right_county_evidence_policy"]
        == "omitted_with_warning"
    )
    assert diagnostics["ambiguous_lower_right_county_evidence_pixels"] > 2_500
    assert not np.any(panel[:margin])
    assert not np.any(panel[-margin:])
    assert not np.any(panel[:, :margin])
    assert not np.any(panel[:, -margin:])
    retained_foreground_fraction = np.count_nonzero(
        positive & source.topology.foreground_interior
    ) / np.count_nonzero(source.topology.foreground_interior)
    assert retained_foreground_fraction > 0.95


def test_partial_source_extent_reports_off_canvas_and_occluded_separately():
    extent = np.ones((5, 6), dtype=bool)
    extent[1:3, 2:4] = False
    points = np.asarray(
        [(-1, 0), (0, 0), (2, 1), (3, 2), (5, 4), (6, 4)],
        dtype=np.int32,
    )

    inside_canvas, visible = _source_extent_membership(points, extent)

    assert inside_canvas.tolist() == [False, True, True, True, True, False]
    assert visible.tolist() == [False, True, False, False, True, False]


def test_balanced_county_validation_rejects_clustered_aggregate_pass(monkeypatch):
    config = FarmsNonrigidConfig(
        county_validation_balanced_cells=(3, 3),
        county_validation_balanced_minimum_visible_cells=3,
        county_validation_balanced_minimum_visible_rows=2,
        county_validation_balanced_minimum_visible_columns=2,
    )
    validation = np.zeros((90, 90), dtype=bool)
    # Three source-visible geographic cells: two dense perfect clusters and a
    # coherent wrong region.  The aggregate support exceeds 0.58, but the
    # geographically balanced contract must still reject it.
    validation[5:25, 5:25] = True
    validation[35:55, 35:55] = True
    validation[65:85, 65:85] = True
    cells = _county_validation_geographic_cells(validation, config)
    source = np.zeros_like(validation)
    source[5:25, 5:25] = True
    source[35:55, 35:55] = True
    source[65:85, 55:65] = True
    monkeypatch.setattr(
        farms_alignment,
        "_map_points",
        lambda points, projection, grid, seed_matrix, warp: points.copy(),
    )
    report = _balanced_county_validation_report(
        cells,
        source,
        # Identity projection/matrix/warp make reference and source pixels
        # coincide in this focused synthetic contract.
        type("Projection", (), {"id": "synthetic"})(),
        {
            "width": 90,
            "height": 90,
            "bounds_web_mercator": [0.0, 0.0, 90.0, 90.0],
        },
        np.eye(3),
        CompactResidualWarp(
            centers_reference_px=np.empty((0, 2)),
            coefficients_source_px=np.empty((0, 2)),
            radius_reference_px=1.0,
            ridge=0.0,
        ),
        observable_extent=np.ones_like(source),
        config=config,
    )

    assert report["supported_cell_count"] == 3
    assert report["cell_pass_fraction"] < 1.0
    assert report["passed"] is False
    assert any(
        cell["p90_px"] > config.county_validation_balanced_p90_limit_px
        for cell in report["failed_supported_cells"]
    )


def test_native_pacific_edge_removes_inset_neatline_but_keeps_island_holes():
    source = _load_source(SOURCE, FarmsNonrigidConfig())
    edge = source.native_pacific_coast_edge_original
    diagnostics = source.water_diagnostics
    x, y, width, height = diagnostics["native_pacific_component_bbox_original"]
    margin = diagnostics["native_pacific_component_neatline_margin_original_px"]

    assert not np.any(edge[:, : x + margin])
    assert not np.any(edge[y + height - margin :, :])
    # A detached-island hole identified entirely from the native Pacific
    # component remains after suppressing the long left/bottom neatlines.
    assert np.count_nonzero(edge[2460:2510, 675:725]) > 20


def test_water_training_validation_and_acceptance_geographies_are_buffered():
    reference = _reference()
    semantics = build_mapbox_farms_semantics(reference)
    masks = _named_pacific_holdouts(semantics, reference)
    validation = np.zeros(EXPECTED_MAPBOX_V2_GRID, dtype=bool)
    test = np.zeros(EXPECTED_MAPBOX_V2_GRID, dtype=bool)
    training = np.zeros(EXPECTED_MAPBOX_V2_GRID, dtype=bool)
    assert set(masks) == {
        "outer_pacific_validation_microsegments",
        "outer_pacific_test_microsegments",
        "north_bay_san_pablo_validation_microsegments",
        "south_bay_validation_microsegments",
        "golden_gate_test_microsegments",
        "east_bay_test_microsegments",
        "outer_pacific_training_microsegments",
        "observed_suisun_delta_training_microsegments",
    }
    for name, mask in masks.items():
        assert np.count_nonzero(mask) >= 20
        if name.endswith("training_microsegments"):
            training |= mask
        elif name.endswith("validation_microsegments"):
            validation |= mask
        elif name.endswith("test_microsegments"):
            test |= mask
    assert not np.any(validation & test)
    assert not np.any(training & validation)
    assert not np.any(training & test)
    assert np.count_nonzero(masks["outer_pacific_validation_microsegments"]) == 2151
    assert np.count_nonzero(masks["outer_pacific_training_microsegments"]) == 1802
    assert np.count_nonzero(masks["outer_pacific_test_microsegments"]) == 1389
    outer_heldout = (
        masks["outer_pacific_validation_microsegments"]
        | masks["outer_pacific_test_microsegments"]
    )
    outer_distance = distance_transform_edt(~outer_heldout)
    assert (
        outer_distance[masks["outer_pacific_training_microsegments"]].min()
        > FarmsNonrigidConfig().water_training_guard_reference_px
    )


def test_water_seed_training_prefers_supported_partial_map_basin():
    config = FarmsNonrigidConfig()
    source = _load_source(SOURCE, config)
    reference = _reference()
    semantics = build_mapbox_farms_semantics(reference)
    masks = _named_pacific_holdouts(semantics, reference, config)
    projection = next(
        item
        for item in _projection_contexts(reference)
        if item.id == "california_albers"
    )
    context = _prepare_water_seed_training_context(
        reference, source, projection, masks, config
    )
    supported = _matrix_from_parameters(
        [
            0.7852692208161476,
            0.5520482796012919,
            1.7064476445553127,
            0.018552252421239146,
        ],
        "similarity",
        source.rgb_working.shape[:2],
    )
    alias = _matrix_from_parameters(
        [
            0.7274372072817611,
            0.48266463984656555,
            1.411557506006917,
            2.901899894073868,
        ],
        "similarity",
        source.rgb_working.shape[:2],
    )

    supported_score, supported_report = _water_seed_training_score(
        supported, context, config
    )
    alias_score, alias_report = _water_seed_training_score(alias, context, config)

    assert supported_score < alias_score * 0.10
    assert supported_report["heldout_scores_evaluated"] is False
    assert {
        item["source_line_id"] for item in supported_report["channels"]
    } == {"native_pacific_coast_edge", "native_internal_water_edge"}
    internal = next(
        item
        for item in supported_report["channels"]
        if item["channel"].startswith("observed_suisun_delta")
    )
    assert internal["source_line_id"] == "native_internal_water_edge"
    assert not next(
        item
        for item in alias_report["channels"]
        if item["channel"].startswith("observed_suisun_delta")
    )["supported"]


def test_water_seed_training_excludes_remote_and_heldout_only_source_edges():
    training = np.column_stack(
        (np.arange(25, dtype=np.float64) + 10.0, np.full(25, 10.0))
    )
    heldout = training.copy()
    remote = np.column_stack(
        (np.arange(25, dtype=np.float64) + 100.0, np.full(25, 120.0))
    )
    context = {
        "method": "synthetic_train_only",
        "map_panel": np.ones((150, 150), dtype=bool),
        "source_shape": (150, 150),
        "channels": [
            {
                "name": "outer_pacific_training_microsegments",
                "source_line_id": "native_pacific_coast_edge",
                "source_line_sha256": "source",
                "source_points": np.concatenate((training, remote), axis=0),
                "training_reference_pixel_count": len(training),
                "training_reference_mask_sha256": "training",
                "training_normalized": training,
                "heldout_reference_pixel_count": len(heldout),
                "heldout_reference_mask_sha256": "heldout",
                "heldout_normalized": heldout,
            }
        ],
    }

    _score, report = _water_seed_training_score(
        np.eye(3), context, FarmsNonrigidConfig()
    )

    channel = report["channels"][0]
    assert channel["source_assignment_pixel_count"] == 0
    assert channel["supported"] is False


def test_water_seed_training_requires_sixty_percent_source_visibility():
    training = np.column_stack(
        (np.arange(100, dtype=np.float64) + 10.0, np.full(100, 10.0))
    )
    panel = np.zeros((130, 130), dtype=bool)
    panel[10, 10:60] = True
    context = {
        "method": "synthetic_partial_visibility",
        "map_panel": panel,
        "source_shape": panel.shape,
        "channels": [
            {
                "name": "outer_pacific_training_microsegments",
                "source_line_id": "native_pacific_coast_edge",
                "source_line_sha256": "source",
                "source_points": training,
                "training_reference_pixel_count": len(training),
                "training_reference_mask_sha256": "training",
                "training_normalized": training,
                "heldout_reference_pixel_count": len(training),
                "heldout_reference_mask_sha256": "heldout",
                "heldout_normalized": training + np.asarray([0.0, 100.0]),
            }
        ],
    }

    _score, report = _water_seed_training_score(
        np.eye(3), context, FarmsNonrigidConfig()
    )

    channel = report["channels"][0]
    assert channel["visible_training_count"] == 50
    assert channel["visible_training_fraction"] == 0.5
    assert channel["source_assignment_pixel_count"] == 100
    assert channel["supported"] is False


def test_water_residual_controls_use_channel_specific_source_edges():
    config = FarmsNonrigidConfig()
    source = _load_source(SOURCE, config)
    reference = _reference()
    semantics = build_mapbox_farms_semantics(reference)
    masks = _named_pacific_holdouts(semantics, reference, config)
    projection = next(
        item
        for item in _projection_contexts(reference)
        if item.id == "california_albers"
    )
    matrix = _matrix_from_parameters(
        [
            0.7852692208161476,
            0.5520482796012919,
            1.7064476445553127,
            0.018552252421239146,
        ],
        "similarity",
        source.rgb_working.shape[:2],
    )

    controls, _targets, report, _artifacts = derive_water_residual_controls(
        reference, source, projection, matrix, masks, config
    )

    assert len(controls) >= 4
    assert report["outer_pacific_control_count"] >= 2
    assert report["observed_suisun_delta_control_count"] >= 1
    assert report["suisun_delta_source_line_is_internal_water_only"] is True
    internal = next(
        item
        for item in report["channels"]
        if item["channel"].startswith("observed_suisun_delta")
    )
    assert internal["source_line_id"] == "native_internal_water_edge"
    assert internal["minimum_source_assignment_to_heldout_working_px"] > 24.0


def test_supported_coast_gate_keeps_strict_p90_tail():
    config = FarmsNonrigidConfig()
    report = {
        "visible_count": 100,
        "visible_fraction": 1.0,
        "median_px": 1.0,
        "p90_px": config.bay_holdout_p90_limit_px + 0.01,
        "within_8px_fraction": 0.99,
    }

    assert not _supported_coast_gate(report, config, bay=True)


def test_validation_preflight_code_path_cannot_score_acceptance_tests():
    source = inspect.getsource(_bounded_validation_state)

    assert 'split_points["test"]' not in source
    assert 'endswith("test_microsegments")' not in source
    assert "validation only" in _bounded_validation_state.__doc__


def test_serialized_residual_base_is_affine_for_generic_consumer():
    reference = _reference()
    projection = next(
        item for item in _projection_contexts(reference) if item.id == "web_mercator"
    )
    warp = CompactResidualWarp(
        centers_reference_px=np.empty((0, 2), dtype=np.float64),
        coefficients_source_px=np.empty((0, 2), dtype=np.float64),
        radius_reference_px=400.0,
        ridge=1.0,
    )
    transform = serialize_farms_nonrigid_transform(
        seed_matrix_working=np.eye(3),
        projection=projection,
        warp_working=warp,
        working_scale=0.2,
        source_original_shape=(5500, 4250),
        source_working_shape=(900, 695),
        target_grid=reference.grid,
    )

    assert transform["reference_pixel_to_source_original_matrix"][2] == [0.0, 0.0, 1.0]
    assert transform["source_original_to_reference_pixel_matrix"][2] == [0.0, 0.0, 1.0]


def test_water_overlay_persists_full_and_internal_component_crop(tmp_path):
    source = _load_source(SOURCE, FarmsNonrigidConfig())
    full = tmp_path / "water.png"
    crop = tmp_path / "water-crop.png"

    diagnostics = _water_overlay(
        source.rgb_working,
        source,
        source.native_internal_water_edge,
        full,
        crop,
    )

    assert full.is_file()
    assert crop.is_file()
    x1, y1, x2, y2 = diagnostics["internal_water_crop_bounds_working_px"]
    assert x1 < 146 < x2
    assert y1 < 363 < y2
