import hashlib
import json

import pytest

from mapscan.classification_review import (
    APP_HTML,
    build_classification_payload,
    write_classification_decision,
    write_inference_decision,
    write_inference_exclusions,
    write_stamp_corrections,
)


def test_manual_stamp_uses_observed_class_display_color_and_opacity():
    assert 'id="manual-opacity"' not in APP_HTML
    assert "const classAlpha=Number(document.querySelector('#class-opacity').value)/100" in APP_HTML
    assert "ctx.globalAlpha=classAlpha;if(classPixels)ctx.drawImage(overlay,0,0)" in APP_HTML
    assert "ctx.globalAlpha=classAlpha;if(manualValues)ctx.drawImage(manualOverlay,0,0)" in APP_HTML


def _classification_run(tmp_path, alignment_status="approved"):
    (tmp_path / "web-mercator-source.jpg").write_bytes(b"source")
    layer_dir = tmp_path / "hazard"
    layer_dir.mkdir()
    (layer_dir / "web-mercator-class-id.png").write_bytes(b"classes")
    manifest = {
        "dataset_id": "sample",
        "title": "Sample",
        "alignment": {
            "inspection": {"grid": {"width": 80, "height": 100}}
        },
        "layers": [
            {
                "id": "hazard",
                "kind": "categorical",
                "category_pixel_counts": {"low": 12, "high": 7},
                "source_nodata_pixel_count": 21,
                "extraction": {
                    "eligible_pixel_count": 40,
                    "classified_pixel_count": 19,
                    "ambiguous_pixel_count": 21,
                },
            }
        ],
    }
    plan = {
        "layers": [
            {
                "id": "hazard",
                "categories": [
                    {"id": "low", "label": "Low", "display_rgb": [1, 2, 3]},
                    {"id": "high", "label": "High", "display_rgb": [4, 5, 6]},
                ],
            }
        ]
    }
    (tmp_path / "extraction.json").write_text(json.dumps(manifest))
    (tmp_path / "plan.snapshot.json").write_text(json.dumps(plan))
    (tmp_path / "review-decision.json").write_text(
        json.dumps({"scope": "alignment", "status": alignment_status, "reviewed_at": "now"})
    )
    return tmp_path


def test_classification_payload_exposes_categories_and_class_ids(tmp_path):
    payload = build_classification_payload(_classification_run(tmp_path))
    layer = payload["layers"][0]
    assert layer["class_id_url"] == "/asset/hazard/web-mercator-class-id.png"
    assert layer["categories"][0] == {
        "class_id": 1,
        "id": "low",
        "label": "Low",
        "display_rgb": [1, 2, 3],
        "pixel_count": 12,
    }
    assert payload["alignment_decision"]["status"] == "approved"
    assert layer["metrics"]["source_classified_pixel_count"] == 19


def test_classification_payload_and_decision_disclose_zero_coverage_classes(tmp_path):
    run = _classification_run(tmp_path)
    manifest_path = run / "extraction.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["warnings"] = ["Two rendered legend swatches are indistinguishable."]
    manifest["layers"][0]["category_pixel_counts"]["high"] = 0
    manifest_path.write_text(json.dumps(manifest))

    payload = build_classification_payload(run)
    assert payload["warnings"] == [
        "Two rendered legend swatches are indistinguishable."
    ]
    assert payload["layers"][0]["zero_coverage_categories"] == [
        {"id": "high", "label": "High"}
    ]

    decision = write_classification_decision(
        run, payload, {"status": "approved", "notes": "Reviewed the limitation"}
    )
    assert decision["known_limitations"]["zero_coverage_categories"] == [
        {
            "layer_id": "hazard",
            "categories": [{"id": "high", "label": "High"}],
        }
    ]
    assert decision["known_limitations"]["manifest_warnings"] == payload["warnings"]


def test_classification_approval_records_separate_scope(tmp_path):
    run = _classification_run(tmp_path)
    payload = build_classification_payload(run)
    decision = write_classification_decision(
        run, payload, {"status": "approved", "notes": "Checked each class"}
    )
    assert decision["scope"] == "classification"
    assert decision["alignment_review"]["status"] == "approved"
    assert json.loads((run / "classification-review-decision.json").read_text())[
        "status"
    ] == "approved"


def test_classification_cannot_be_approved_before_alignment(tmp_path):
    run = _classification_run(tmp_path, alignment_status="needs_revision")
    payload = build_classification_payload(run)
    with pytest.raises(ValueError, match="alignment is approved"):
        write_classification_decision(run, payload, {"status": "approved"})


def test_inference_approval_is_separate_and_requires_classification(tmp_path):
    run = _classification_run(tmp_path)
    inference_layer = run / "inference" / "hazard"
    inference_layer.mkdir(parents=True)
    (inference_layer / "web-mercator-class-id-inferred.png").write_bytes(b"classes")
    (inference_layer / "web-mercator-inference-mask.png").write_bytes(b"mask")
    (run / "inference" / "inference.json").write_text(
        json.dumps(
            {
                "layers": [
                    {
                        "layer_id": "hazard",
                        "inferred_pixel_count": 9,
                        "method": "test",
                    }
                ]
            }
        )
    )
    payload = build_classification_payload(run)
    assert payload["layers"][0]["inference"]["inferred_pixel_count"] == 9
    with pytest.raises(ValueError, match="classification is approved"):
        write_inference_decision(run, payload, {"status": "approved"})
    write_classification_decision(run, payload, {"status": "approved"})
    payload = build_classification_payload(run)
    decision = write_inference_decision(
        run, payload, {"status": "approved", "notes": "Only small gaps"}
    )
    assert decision["scope"] == "inference"
    assert decision["classification_review"]["status"] == "approved"


def test_classification_payload_prefers_highest_versioned_inference(tmp_path):
    run = _classification_run(tmp_path)
    for name, pixel_count in (("inference", 9), ("inference-v4", 42)):
        inference_layer = run / name / "hazard"
        inference_layer.mkdir(parents=True)
        (inference_layer / "web-mercator-class-id-inferred.png").write_bytes(b"classes")
        (inference_layer / "web-mercator-inference-mask.png").write_bytes(b"mask")
        (run / name / "inference.json").write_text(
            json.dumps(
                {
                    "layers": [
                        {
                            "layer_id": "hazard",
                            "inferred_pixel_count": pixel_count,
                            "method": name,
                        }
                    ]
                }
            )
        )
    payload = build_classification_payload(run)
    inference = payload["layers"][0]["inference"]
    assert inference["artifact_root"] == "inference-v4"
    assert inference["inferred_pixel_count"] == 42
    assert inference["class_id_url"].startswith("/asset/inference-v4/")


def test_clone_stamp_corrections_are_hashed_and_exposed(tmp_path):
    run = _classification_run(tmp_path)
    payload = build_classification_payload(run)
    corrections = write_stamp_corrections(
        run,
        payload,
        {
            "operations": [
                {
                    "layer_id": "hazard",
                    "source": [10, 20],
                    "target": [30, 40],
                    "radius_px": 8,
                }
            ]
        },
    )
    assert corrections["scope"] == "manual_clone_stamp"
    assert corrections["schema_version"] == 3
    assert corrections["policy"]["source"] == (
        "observed_for_legacy_operations_or_composite_at_operation_time"
    )
    assert corrections["policy"]["target"] == (
        "manual_override_patch_at_any_raster_pixel"
    )
    assert build_classification_payload(run)["stamp_corrections"]["operations"] == [
        {
            "layer_id": "hazard",
            "source": [10.0, 20.0],
            "target": [30.0, 40.0],
            "radius_px": 8.0,
            "source_mode": "observed",
        }
    ]


def test_disabled_inference_keeps_observed_source_stamp_corrections(tmp_path):
    run = _classification_run(tmp_path)
    payload = build_classification_payload(run)
    write_stamp_corrections(
        run,
        payload,
        {
            "operations": [
                {
                    "layer_id": "hazard",
                    "source": [10, 20],
                    "target": [30, 40],
                    "radius_px": 8,
                }
            ]
        },
    )
    (run / "inference-selection.json").write_text(
        json.dumps({"schema_version": 1, "enabled": False})
    )
    disabled = build_classification_payload(run)
    assert disabled["layers"][0]["inference"] is None
    assert len(disabled["stamp_corrections"]["operations"]) == 1


def test_payload_exposes_current_enclosed_hole_fill(tmp_path):
    run = _classification_run(tmp_path)
    payload = build_classification_payload(run)
    write_stamp_corrections(run, payload, {"operations": []})
    fill_layer = run / "enclosed-fill-v1" / "hazard"
    fill_layer.mkdir(parents=True)
    (fill_layer / "web-mercator-enclosed-fill-values.png").write_bytes(b"values")
    (fill_layer / "web-mercator-enclosed-fill-mask.png").write_bytes(b"mask")
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    (run / "enclosed-fill-v1" / "enclosed-fill.json").write_text(
        json.dumps(
            {
                "extraction_manifest_sha256": sha(run / "extraction.json"),
                "stamp_corrections_sha256": sha(run / "stamp-corrections.json"),
                "layers": [
                    {
                        "layer_id": "hazard",
                        "filled_component_count": 3,
                        "filled_pixel_count": 12,
                        "maximum_area_exclusive": 50,
                    }
                ],
            }
        )
    )
    (run / "enclosed-hole-fill-selection.json").write_text(
        json.dumps({"schema_version": 1, "enabled": True})
    )
    enclosed = build_classification_payload(run)["layers"][0]["enclosed_fill"]
    assert enclosed["filled_component_count"] == 3
    assert enclosed["filled_pixel_count"] == 12
    assert enclosed["values_url"].endswith(
        "/hazard/web-mercator-enclosed-fill-values.png"
    )


def test_inference_exclusions_are_hashed_and_exposed(tmp_path):
    run = _classification_run(tmp_path)
    inference_layer = run / "inference" / "hazard"
    inference_layer.mkdir(parents=True)
    (inference_layer / "web-mercator-class-id-inferred.png").write_bytes(b"classes")
    (inference_layer / "web-mercator-inference-mask.png").write_bytes(b"mask")
    (run / "inference" / "inference.json").write_text(
        json.dumps({"layers": [{"layer_id": "hazard", "inferred_pixel_count": 9}]})
    )
    payload = build_classification_payload(run)
    exclusions = write_inference_exclusions(
        run,
        payload,
        {
            "operations": [
                {"layer_id": "hazard", "center": [30, 40], "radius_px": 8}
            ]
        },
    )
    assert exclusions["scope"] == "manual_inference_exclusion"
    assert exclusions["policy"]["observed"] == "never_modified"
    assert build_classification_payload(run)["inference_exclusions"]["operations"] == [
        {"layer_id": "hazard", "center": [30.0, 40.0], "radius_px": 8.0}
    ]

    (run / "inference" / "inference.json").write_text(
        json.dumps({"layers": [{"layer_id": "hazard", "inferred_pixel_count": 10}]})
    )
    assert build_classification_payload(run)["inference_exclusions"] is None
