import json
import math

from mapscan.review import (
    APP_HTML,
    build_review_payload,
    write_alignment_corrections,
    write_review_decision,
)


def test_review_page_has_direct_source_extraction_flip_controls():
    assert 'id="show-source"' in APP_HTML
    assert 'id="show-classification"' in APP_HTML
    assert "event.code!=='Space'" in APP_HTML
    assert "setComparisonView(comparisonView==='source'?'classification':'source')" in APP_HTML
    assert 'id="review-region"' in APP_HTML
    assert "fitBounds(region.pixel_bounds)" in APP_HTML


def _review_run(tmp_path):
    assets = {
        "source": "web-mercator-source.jpg",
        "state_overlay": "web-mercator-state-overlay.png",
        "county_overlay": "web-mercator-county-overlay.png",
        "county_residual": "web-mercator-county-residual.png",
    }
    for relative in assets.values():
        (tmp_path / relative).write_bytes(b"asset")
    layer = tmp_path / "sample-layer"
    layer.mkdir()
    (layer / "web-mercator-preview.png").write_bytes(b"preview")
    manifest = {
        "dataset_id": "sample",
        "title": "Sample map",
        "status": "needs_visual_review",
        "alignment": {
            "mode": "assisted",
            "projection_crs": "EPSG:3310",
            "inspection": {"grid": {"width": 800, "height": 1000}},
        },
        "review": {
            "assets": assets,
            "default_view": "classification",
            "county_residual": {"median_nearest_source_edge_px": 2.0},
            "decision_path": "review-decision.json",
        },
        "layers": [{"id": "sample-layer", "category_pixel_counts": {"a": 12}}],
    }
    (tmp_path / "extraction.json").write_text(json.dumps(manifest))
    return tmp_path


def test_review_payload_discovers_assets_and_classification(tmp_path):
    run = _review_run(tmp_path)
    payload = build_review_payload(run)
    assert payload["width"] == 800
    assert payload["assets"]["source"] == "/asset/web-mercator-source.jpg"
    assert payload["layers"][0]["id"] == "sample-layer"
    assert payload["default_view"] == "classification"


def test_review_payload_exposes_plan_defined_geographic_regions(tmp_path, monkeypatch):
    run = _review_run(tmp_path)
    manifest_path = run / "extraction.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["alignment"]["inspection"]["grid"] = {
        "crs": "EPSG:4326",
        "bounds": [0.0, 0.0, 10.0, 10.0],
        "width": 100,
        "height": 100,
    }
    manifest_path.write_text(json.dumps(manifest))
    (run / "plan.snapshot.json").write_text(
        json.dumps(
            {
                "comparison_regions": [
                    {
                        "id": "sample",
                        "label": "Sample region",
                        "bounds_wgs84": [2.0, 3.0, 5.0, 7.0],
                    }
                ]
            }
        )
    )
    monkeypatch.setattr("mapscan.review.ACTIVE_CANONICAL_POINTER", tmp_path / "missing")
    payload = build_review_payload(run)
    assert payload["review_regions"] == [
        {
            "id": "sample",
            "label": "Sample region",
            "bounds_wgs84": [2.0, 3.0, 5.0, 7.0],
            "pixel_bounds": [20, 30, 50, 70],
        }
    ]


def test_review_payload_can_disable_inapplicable_county_diagnostic(tmp_path, monkeypatch):
    run = _review_run(tmp_path)
    manifest_path = run / "extraction.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["review"]["county_diagnostic_enabled"] = False
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr("mapscan.review.ACTIVE_CANONICAL_POINTER", tmp_path / "missing")
    payload = build_review_payload(run)
    assert payload["county_diagnostic_enabled"] is False
    assert payload["decision"] is None


def test_review_decision_records_evidence_policy(tmp_path):
    run = _review_run(tmp_path)
    payload = build_review_payload(run)
    decision = write_review_decision(
        run, payload, {"status": "needs_revision", "notes": "Local drift"}
    )
    assert decision["status"] == "needs_revision"
    assert decision["evidence_policy"]["primary"] == "state_border_and_coastline"
    saved = json.loads((run / "review-decision.json").read_text())
    assert saved["extraction_manifest_sha256"] == payload["manifest_sha256"]


def test_alignment_corrections_record_direction_and_map_coordinates(tmp_path):
    run = _review_run(tmp_path)
    payload = build_review_payload(run)
    payload["alignment"]["inspection"]["grid"].update(
        {"crs": "EPSG:3857", "bounds": [100.0, 200.0, 900.0, 1200.0]}
    )
    record = write_alignment_corrections(
        run,
        payload,
        {
            "corrections": [
                {"reference": [100.0, 250.0], "source": [112.0, 242.0]},
                {"reference": [400.0, 500.0], "source": [400.0, 500.0]},
            ]
        },
    )
    assert record["direction"] == "authoritative_reference_to_current_warped_source"
    assert record["summary"]["count"] == 2
    displacement = record["corrections"][0]["required_source_displacement_px"]
    assert displacement["dx"] == -12.0
    assert displacement["dy"] == 8.0
    assert math.isclose(displacement["magnitude"], math.hypot(12.0, -8.0))
    assert record["corrections"][1]["required_source_displacement_px"]["magnitude"] == 0.0
    assert (run / "alignment-corrections.json").exists()


def test_alignment_corrections_reject_points_outside_raster(tmp_path):
    run = _review_run(tmp_path)
    payload = build_review_payload(run)
    payload["alignment"]["inspection"]["grid"].update(
        {"crs": "EPSG:3857", "bounds": [0.0, 0.0, 1.0, 1.0]}
    )
    try:
        write_alignment_corrections(
            run,
            payload,
            {"corrections": [{"reference": [-1, 5], "source": [0, 5]}]},
        )
    except ValueError as error:
        assert "outside" in str(error)
    else:
        raise AssertionError("Expected an out-of-raster correction to fail")


def test_review_payload_adapts_continuous_ramp_runs(tmp_path, monkeypatch):
    for name in (
        "web-mercator-source.jpg",
        "transparent.png",
        "web-mercator-preview.png",
        "label-overlay.png",
    ):
        (tmp_path / name).write_bytes(b"asset")
    alignment_path = tmp_path / "alignment.json"
    alignment_path.write_text(
        json.dumps(
            {
                "alignment_mode": "automatic",
                "best": {
                    "projection_crs": "EPSG:3310",
                    "metrics": {
                        "holdout_median_px_at_source_resolution": 1.0,
                        "holdout_p90_px_at_source_resolution": 2.0,
                    },
                },
            }
        )
    )
    grid = {
        "crs": "EPSG:3857",
        "bounds": [0.0, 0.0, 10.0, 20.0],
        "width": 80,
        "height": 100,
    }
    manifest = {
        "extraction_kind": "continuous_color_ramp",
        "dataset_id": "elevation",
        "title": "Elevation",
        "status": "needs_visual_review",
        "alignment": str(alignment_path),
        "warp": grid,
        "target": {"colored_outside_pixel_count": 0},
        "canonical_clip": {"component_count": 5},
        "review": {
            "assets": {
                "source": "web-mercator-source.jpg",
                "county_overlay": "transparent.png",
                "county_residual": "transparent.png",
            },
            "layers": [
                {
                    "id": "ramp",
                    "label": "Elevation ramp",
                    "path": "web-mercator-preview.png",
                },
                {
                    "id": "labels",
                    "label": "OCR labels",
                    "path": "label-overlay.png",
                },
            ],
        },
    }
    (tmp_path / "continuous-extraction.json").write_text(json.dumps(manifest))
    pointer = tmp_path / "pointer.json"
    pointer.write_text("{}")
    canonical_manifest = tmp_path / "canonical.json"
    canonical_manifest.write_text("{}")
    monkeypatch.setattr("mapscan.review.ACTIVE_CANONICAL_POINTER", pointer)
    monkeypatch.setattr(
        "mapscan.review.load_active_canonical_border",
        lambda _: (
            canonical_manifest,
            {
                "canonical_boundary_id": "canonical-v2",
                "source_grid": grid,
                "artifacts": {"overlay": {"sha256": "overlay-hash"}},
                "topology": {
                    "mainland_component_count": 1,
                    "offshore_island_component_count": 4,
                },
            },
            {},
        ),
    )
    payload = build_review_payload(tmp_path)
    assert payload["width"] == 80
    assert payload["continuous"]["target"]["colored_outside_pixel_count"] == 0
    assert payload["layers"][1]["id"] == "labels"
    assert payload["assets"]["state_overlay"] == "/canonical/overlay"
    assert payload["alignment"]["metrics"]["control_point_median_px"] == 1.0


def test_review_payload_can_gate_alignment_before_extraction(tmp_path, monkeypatch):
    (tmp_path / "web-mercator-source.png").write_bytes(b"source")
    (tmp_path / "no-extraction.png").write_bytes(b"transparent")
    grid = {
        "crs": "EPSG:3857",
        "bounds": [0.0, 0.0, 10.0, 20.0],
        "width": 80,
        "height": 100,
    }
    (tmp_path / "alignment.json").write_text(
        json.dumps(
            {
                "status": "needs_visual_review",
                "alignment_mode": "automatic",
                "best": {"projection_crs": "EPSG:3310", "metrics": {}},
                "global_refit": {"grid": grid},
            }
        )
    )
    pointer = tmp_path / "pointer.json"
    pointer.write_text("{}")
    canonical_manifest = tmp_path / "canonical.json"
    canonical_manifest.write_text("{}")
    monkeypatch.setattr("mapscan.review.ACTIVE_CANONICAL_POINTER", pointer)
    monkeypatch.setattr(
        "mapscan.review.load_active_canonical_border",
        lambda _: (
            canonical_manifest,
            {
                "canonical_boundary_id": "canonical-v2",
                "source_grid": grid,
                "artifacts": {"overlay": {"sha256": "overlay-hash"}},
                "topology": {
                    "mainland_component_count": 1,
                    "offshore_island_component_count": 4,
                },
            },
            {},
        ),
    )

    payload = build_review_payload(tmp_path)
    assert payload["alignment_only"] is True
    assert payload["assets"]["source"] == "/asset/web-mercator-source.png"
    assert payload["layers"][0]["label"] == "Data extraction not run"
    decision = write_review_decision(
        tmp_path, payload, {"status": "needs_revision", "notes": "Coast drifts"}
    )
    assert decision["alignment_manifest_sha256"] == payload["manifest_sha256"]
    assert "extraction_manifest_sha256" not in decision
