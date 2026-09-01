import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

import mapscan.reviewed_extraction_promotion as promotion
from mapscan.reviewed_extraction_promotion import promote_reviewed_extraction
from mapscan.tile_export import export_categorical_tiles


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def _fixture(tmp_path: Path, *, kind: str = "categorical"):
    run = tmp_path / "runs" / "reviewed"
    layer_dir = run / "zones"
    layer_dir.mkdir(parents=True)
    valid = np.zeros((8, 9), dtype=np.uint8)
    valid[1:6, 1:5] = 255
    valid[0, 7] = 255
    valid[2, 7] = 255
    valid[4, 7] = 255
    valid[6, 7] = 255
    values = (valid > 0).astype(np.uint8)
    values[1:3, 1:3] = 2
    preview = np.zeros((8, 9, 4), dtype=np.uint8)
    preview[values == 1] = [10, 20, 30, 255]
    preview[values == 2] = [40, 50, 60, 255]
    completion = np.zeros_like(valid)
    completion[5, 4] = 255
    speck = np.zeros_like(valid)
    speck[1, 1] = 255
    originals = np.zeros_like(valid)
    originals[1, 1] = 1
    removed = np.zeros_like(valid)
    Image.fromarray(values).save(layer_dir / "web-mercator-class-id.png")
    Image.fromarray(preview).save(layer_dir / "web-mercator-preview.png")
    Image.fromarray(valid).save(layer_dir / "web-mercator-publication-interior-mask.png")
    Image.fromarray(completion).save(layer_dir / "web-mercator-target-completion-mask.png")
    Image.fromarray(speck).save(layer_dir / "web-mercator-speck-reassignment-mask.png")
    Image.fromarray(originals).save(layer_dir / "web-mercator-speck-original-class-id.png")
    Image.fromarray(removed).save(layer_dir / "web-mercator-boundary-removed-mask.png")

    alignment = tmp_path / "alignment.json"
    _write_json(alignment, {"status": "accepted"})
    plan = {
        "dataset_id": "reviewed",
        "title": "Reviewed",
        "source": "source.png",
        "alignment": str(alignment),
        "layers": [
            {
                "id": "zones",
                "categories": [
                    {"id": "one", "label": "One", "display_rgb": [10, 20, 30]},
                    {"id": "two", "label": "Two", "display_rgb": [40, 50, 60]},
                ],
            }
        ],
    }
    plan_path = tmp_path / "plan.json"
    _write_json(plan_path, plan)
    (run / "plan.snapshot.json").write_bytes(plan_path.read_bytes())

    canonical_dir = tmp_path / "canonical" / "v2"
    canonical_dir.mkdir(parents=True)
    overlay = np.zeros((12, 14, 4), dtype=np.uint8)
    overlay[1:8, 1] = [40, 255, 110, 255]
    overlay[1, 1:5] = [40, 255, 110, 255]
    overlay[3, 10] = [40, 255, 110, 255]
    overlay[5, 10] = [40, 255, 110, 255]
    overlay[7, 10] = [40, 255, 110, 255]
    overlay[9, 10] = [40, 255, 110, 255]
    overlay_path = canonical_dir / "overlay.png"
    Image.fromarray(overlay).save(overlay_path)
    canonical = {
        "status": "approved_canonical_reference",
        "canonical_boundary_id": "california-county-detail-border-v2",
        "source_grid": {
            "crs": "EPSG:3857",
            "bounds": [0, 0, 14, 12],
            "width": 14,
            "height": 12,
        },
        "topology": {"combined_component_count": 5},
        "artifacts": {
            "overlay": {"path": overlay_path.name, "sha256": _sha256(overlay_path)}
        },
    }
    canonical_path = canonical_dir / "canonical-boundary.json"
    _write_json(canonical_path, canonical)
    pointer_path = tmp_path / "canonical" / "active.json"
    _write_json(
        pointer_path,
        {
            "status": "active_canonical_boundary",
            "canonical_boundary_id": "california-county-detail-border-v2",
            "manifest": {
                "path": "v2/canonical-boundary.json",
                "sha256": _sha256(canonical_path),
            },
        },
    )

    extraction = {
        "dataset_id": "reviewed",
        "plan": {"path": str(plan_path), "sha256": _sha256(plan_path)},
        "layers": [
            {
                "id": "zones",
                "kind": kind,
                "warp": {
                    "crs": "EPSG:3857",
                    "bounds": [0, 0, 9, 8],
                    "width": 9,
                    "height": 8,
                },
                "canonical_clip": {
                    "component_count": 5,
                    "active_manifest": {
                        "canonical_boundary_id": canonical["canonical_boundary_id"],
                        "sha256": _sha256(canonical_path),
                    },
                    "artifacts": {
                        "interior": {
                            "path": "zones/web-mercator-publication-interior-mask.png",
                            "sha256": _sha256(layer_dir / "web-mercator-publication-interior-mask.png"),
                        },
                        "removed": {
                            "path": "zones/web-mercator-boundary-removed-mask.png",
                            "sha256": _sha256(layer_dir / "web-mercator-boundary-removed-mask.png"),
                        },
                    },
                },
                "completion_artifacts": {
                    "web_mercator_target_completion_mask": {
                        "path": "zones/web-mercator-target-completion-mask.png",
                        "sha256": _sha256(layer_dir / "web-mercator-target-completion-mask.png"),
                    }
                },
                "speck_suppression": {
                    "reassigned_pixel_count": 1,
                    "artifacts": {
                        "mask": {
                            "path": "zones/web-mercator-speck-reassignment-mask.png",
                            "sha256": _sha256(layer_dir / "web-mercator-speck-reassignment-mask.png"),
                        },
                        "original_class_id": {
                            "path": "zones/web-mercator-speck-original-class-id.png",
                            "sha256": _sha256(layer_dir / "web-mercator-speck-original-class-id.png"),
                        },
                    },
                },
            }
        ],
    }
    extraction_path = run / "extraction.json"
    _write_json(extraction_path, extraction)
    extraction_hash = _sha256(extraction_path)
    _write_json(
        run / "review-decision.json",
        {"status": "approved", "extraction_manifest_sha256": extraction_hash},
    )
    _write_json(
        run / "classification-review-decision.json",
        {"status": "approved", "extraction_manifest_sha256": extraction_hash},
    )

    audit_dir = tmp_path / "source-diff" / "case"
    audit_dir.mkdir(parents=True)
    audit_path = audit_dir / "source-diff-audit.json"
    class_hash = _sha256(layer_dir / "web-mercator-class-id.png")
    _write_json(
        audit_path,
        {
            "status": "pass",
            "manifest_sha256": extraction_hash,
            "layers": [
                {"artifacts": {"audited_class_id": {"sha256": class_hash}}}
            ],
        },
    )
    batch_path = tmp_path / "source-diff" / "source-diff-batch.json"
    _write_json(
        batch_path,
        {
            "status": "pass",
            "cases": [
                {
                    "id": "case",
                    "status": "pass",
                    "fixed_point_reached": True,
                    "report": "case/source-diff-audit.json",
                    "comparison_iterations": [
                        {"iteration": 1, "signature_sha256": "same"},
                        {"iteration": 2, "signature_sha256": "same"},
                    ],
                }
            ],
        },
    )
    return run, pointer_path, batch_path, values


def test_promotes_only_byte_identical_dual_reviewed_extraction(tmp_path):
    run, pointer, batch, values = _fixture(
        tmp_path, kind="patterned_categorical"
    )
    output = tmp_path / "materialized"
    result = promote_reviewed_extraction(
        run,
        output,
        author_statement="lgtm",
        source_diff_batch_path=batch,
        source_diff_case_id="case",
        canonical_pointer_path=pointer,
    )
    promoted = np.asarray(
        Image.open(output / "zones/web-mercator-class-id-final.png"), dtype=np.uint8
    )
    assert np.array_equal(promoted, values)
    manifest = json.loads((output / "materialization.json").read_text())
    decision = json.loads((output / "materialization-review-decision.json").read_text())
    assert result["boundary_component_count"] == 5
    assert manifest["layers"][0]["target_completion_pixel_count"] == 1
    assert manifest["layers"][0]["speck_reassignment_pixel_count"] == 1
    assert manifest["source_diff"]["fixed_point_reached"] is True
    assert decision["status"] == "approved"
    assert decision["author_statement"] == "lgtm"
    assert decision["approval_carried_forward"] is False
    assert decision["materialization_sha256"] == _sha256(
        output / "materialization.json"
    )


def test_water_exclusions_can_split_data_without_redefining_outer_boundary(
    tmp_path, monkeypatch
):
    run, pointer, batch, _ = _fixture(tmp_path)
    layer_dir = run / "zones"
    canonical = np.asarray(
        Image.open(layer_dir / "web-mercator-publication-interior-mask.png")
    ) > 0
    publication = canonical.copy()
    publication[1:6, 3] = False
    values = np.asarray(
        Image.open(layer_dir / "web-mercator-class-id.png"), dtype=np.uint8
    ).copy()
    values[~publication] = 0
    preview = np.asarray(
        Image.open(layer_dir / "web-mercator-preview.png").convert("RGBA"),
        dtype=np.uint8,
    ).copy()
    preview[~publication] = 0
    Image.fromarray(publication.astype(np.uint8) * 255).save(
        layer_dir / "web-mercator-publication-interior-mask.png"
    )
    water = canonical & ~publication
    Image.fromarray(water.astype(np.uint8) * 255).save(
        layer_dir / "web-mercator-internal-water-mask.png"
    )
    Image.fromarray(values).save(layer_dir / "web-mercator-class-id.png")
    Image.fromarray(preview).save(layer_dir / "web-mercator-preview.png")

    extraction_path = run / "extraction.json"
    extraction = json.loads(extraction_path.read_text())
    clip = extraction["layers"][0]["canonical_clip"]
    clip["base_valid_pixel_count"] = int(np.count_nonzero(canonical))
    clip["valid_pixel_count"] = int(np.count_nonzero(publication))
    clip["mainland_manifest"] = {"path": extraction["plan"]["path"]}
    clip["internal_water_exclusion"] = {
        "method": "test_water",
        "excluded_interior_pixel_count": int(np.count_nonzero(water)),
        "artifact": {
            "path": "zones/web-mercator-internal-water-mask.png",
            "sha256": _sha256(layer_dir / "web-mercator-internal-water-mask.png"),
        },
    }
    clip["artifacts"]["interior"]["sha256"] = _sha256(
        layer_dir / "web-mercator-publication-interior-mask.png"
    )
    _write_json(extraction_path, extraction)
    extraction_hash = _sha256(extraction_path)
    for filename in ("review-decision.json", "classification-review-decision.json"):
        decision = json.loads((run / filename).read_text())
        decision["extraction_manifest_sha256"] = extraction_hash
        _write_json(run / filename, decision)
    batch_manifest = json.loads(batch.read_text())
    report_path = batch.parent / batch_manifest["cases"][0]["report"]
    report = json.loads(report_path.read_text())
    report["manifest_sha256"] = extraction_hash
    report["layers"][0]["artifacts"]["audited_class_id"]["sha256"] = _sha256(
        layer_dir / "web-mercator-class-id.png"
    )
    _write_json(report_path, report)
    monkeypatch.setattr(
        promotion,
        "canonical_publication_interior",
        lambda *args, **kwargs: (canonical, {}),
    )

    output = tmp_path / "materialized-water-aware"
    result = promote_reviewed_extraction(
        run,
        output,
        author_statement="lgtm",
        source_diff_batch_path=batch,
        source_diff_case_id="case",
        canonical_pointer_path=pointer,
    )
    audit = json.loads((output / "boundary-clip-audit.json").read_text())
    manifest = json.loads((output / "materialization.json").read_text())
    assert result["boundary_component_count"] == 5
    assert audit["boundary"]["connected_component_count"] == 5
    assert audit["boundary"]["publication_interior_component_count"] == 6
    assert audit["boundary"]["publication_interior_exclusion_pixel_count"] == 5
    assert manifest["boundary_clip"]["publication_interior_component_count"] == 6
    assert manifest["boundary_clip"]["internal_water_exclusion"]["method"] == "test_water"
    assert manifest["boundary_clip"]["internal_water_exclusion"]["artifact"][
        "sha256"
    ] == _sha256(output / "internal-water-exclusion-mask.png")
    exported = export_categorical_tiles(
        output,
        tmp_path / "tiles-water-aware",
        minimum_zoom=0,
        maximum_zoom=1,
        overview_supersampling=2,
    )
    assert exported["boundary"]["continuous_border_component_count"] == 5
    assert exported["boundary"]["geojson_feature_count"] == 5
    assert exported["boundary"]["publication_interior_component_count"] == 6
    assert exported["boundary"]["publication_interior_exclusion_pixel_count"] == 5
    assert exported["boundary"]["internal_water_exclusion"]["method"] == "test_water"
    assert exported["boundary"]["internal_water_exclusion"]["artifact"][
        "path"
    ] == "internal-water-exclusion-mask.png"
    assert (tmp_path / "tiles-water-aware/internal-water-exclusion-mask.png").is_file()


def test_reconstructs_declared_lime_coastal_seam_before_water_audit(
    tmp_path, monkeypatch
):
    run, pointer, batch, _ = _fixture(tmp_path)
    layer_dir = run / "zones"
    canonical = np.asarray(
        Image.open(layer_dir / "web-mercator-publication-interior-mask.png")
    ) > 0
    expanded = canonical.copy()
    expanded[1:6, 5] = True
    publication = expanded.copy()
    publication[3, 5] = False
    values = np.asarray(
        Image.open(layer_dir / "web-mercator-class-id.png"), dtype=np.uint8
    ).copy()
    values[expanded & ~canonical] = 1
    values[~publication] = 0
    preview = np.asarray(
        Image.open(layer_dir / "web-mercator-preview.png").convert("RGBA"),
        dtype=np.uint8,
    ).copy()
    preview[expanded & ~canonical] = [10, 20, 30, 255]
    preview[~publication] = 0
    Image.fromarray(publication.astype(np.uint8) * 255).save(
        layer_dir / "web-mercator-publication-interior-mask.png"
    )
    Image.fromarray(values).save(layer_dir / "web-mercator-class-id.png")
    Image.fromarray(preview).save(layer_dir / "web-mercator-preview.png")

    seam_report = {
        "method": "narrow_rowwise_active_lime_west_coast_seam",
        "maximum_gap_px": 4,
        "maximum_x_fraction": 0.5,
        "start_y_fraction": 0.0,
        "end_y_fraction": 1.0,
        "accepted_row_count": 5,
        "added_pixel_count": 5,
        "accepted_gap_median_px": 1.0,
        "accepted_gap_max_px": 1,
        "large_opening_policy": "gaps above the limit remain unchanged",
        "active_manifest": {
            "canonical_boundary_id": "california-county-detail-border-v2",
            "sha256": "active",
        },
    }
    extraction_path = run / "extraction.json"
    extraction = json.loads(extraction_path.read_text())
    clip = extraction["layers"][0]["canonical_clip"]
    clip["base_valid_pixel_count"] = int(np.count_nonzero(expanded))
    clip["valid_pixel_count"] = int(np.count_nonzero(publication))
    clip["mainland_manifest"] = {"path": extraction["plan"]["path"]}
    clip["coastal_seam"] = seam_report
    clip["internal_water_exclusion"] = {"excluded_interior_pixel_count": 1}
    clip["artifacts"]["interior"]["sha256"] = _sha256(
        layer_dir / "web-mercator-publication-interior-mask.png"
    )
    _write_json(extraction_path, extraction)
    extraction_hash = _sha256(extraction_path)
    for filename in ("review-decision.json", "classification-review-decision.json"):
        decision = json.loads((run / filename).read_text())
        decision["extraction_manifest_sha256"] = extraction_hash
        _write_json(run / filename, decision)
    batch_manifest = json.loads(batch.read_text())
    report_path = batch.parent / batch_manifest["cases"][0]["report"]
    report = json.loads(report_path.read_text())
    report["manifest_sha256"] = extraction_hash
    report["layers"][0]["artifacts"]["audited_class_id"]["sha256"] = _sha256(
        layer_dir / "web-mercator-class-id.png"
    )
    _write_json(report_path, report)
    monkeypatch.setattr(
        promotion,
        "canonical_publication_interior",
        lambda *args, **kwargs: (canonical, {}),
    )
    monkeypatch.setattr(
        promotion,
        "close_west_coast_clipping_seam",
        lambda *args, **kwargs: (expanded, seam_report, expanded & ~canonical),
    )

    output = tmp_path / "materialized-seam-water"
    result = promote_reviewed_extraction(
        run,
        output,
        author_statement="approved lime seam",
        source_diff_batch_path=batch,
        source_diff_case_id="case",
        canonical_pointer_path=pointer,
    )
    audit = json.loads((output / "boundary-clip-audit.json").read_text())
    manifest = json.loads((output / "materialization.json").read_text())
    assert result["boundary_component_count"] == 5
    assert audit["boundary"]["canonical_interior_pixel_count"] == int(
        np.count_nonzero(expanded)
    )
    assert audit["boundary"]["publication_interior_exclusion_pixel_count"] == 1
    assert audit["boundary"]["coastal_seam"]["added_pixel_count"] == 5
    assert manifest["boundary_clip"]["coastal_seam"]["method"].startswith(
        "narrow_rowwise_active_lime"
    )


def test_refuses_reviewed_extraction_with_interior_nodata(tmp_path):
    run, pointer, batch, _ = _fixture(tmp_path)
    path = run / "zones/web-mercator-class-id.png"
    values = np.asarray(Image.open(path), dtype=np.uint8).copy()
    values[1, 1] = 0
    Image.fromarray(values).save(path)
    try:
        promote_reviewed_extraction(
            run,
            tmp_path / "materialized",
            author_statement="lgtm",
            source_diff_batch_path=batch,
            source_diff_case_id="case",
            canonical_pointer_path=pointer,
        )
    except ValueError as error:
        assert "coverage contract" in str(error)
    else:
        raise AssertionError("Interior NoData should block promotion")


def test_promotes_explicit_sparse_visible_evidence_without_filling_nodata(tmp_path):
    run, pointer, batch, _ = _fixture(tmp_path)
    extraction_path = run / "extraction.json"
    extraction = json.loads(extraction_path.read_text())
    plan_path = Path(extraction["plan"]["path"])
    plan = json.loads(plan_path.read_text())
    plan["layers"][0]["coverage_expectation"] = "sparse_visible_evidence"
    _write_json(plan_path, plan)
    (run / "plan.snapshot.json").write_bytes(plan_path.read_bytes())
    extraction["plan"]["sha256"] = _sha256(plan_path)
    _write_json(extraction_path, extraction)
    extraction_hash = _sha256(extraction_path)
    for filename in ("review-decision.json", "classification-review-decision.json"):
        decision = json.loads((run / filename).read_text())
        decision["extraction_manifest_sha256"] = extraction_hash
        _write_json(run / filename, decision)

    class_path = run / "zones/web-mercator-class-id.png"
    values = np.asarray(Image.open(class_path), dtype=np.uint8).copy()
    values[1, 1] = 0
    Image.fromarray(values).save(class_path)
    batch_manifest = json.loads(batch.read_text())
    report_path = batch.parent / batch_manifest["cases"][0]["report"]
    report = json.loads(report_path.read_text())
    report["manifest_sha256"] = extraction_hash
    report["layers"][0]["artifacts"]["audited_class_id"]["sha256"] = _sha256(
        class_path
    )
    _write_json(report_path, report)

    output = tmp_path / "materialized-sparse"
    result = promote_reviewed_extraction(
        run,
        output,
        author_statement="approved sparse evidence",
        source_diff_batch_path=batch,
        source_diff_case_id="case",
        canonical_pointer_path=pointer,
    )
    promoted = np.asarray(
        Image.open(output / "zones/web-mercator-class-id-final.png"), dtype=np.uint8
    )
    manifest = json.loads((output / "materialization.json").read_text())
    audit = json.loads((output / "boundary-clip-audit.json").read_text())
    assert np.array_equal(promoted, values)
    assert result["unclassified_pixel_count_inside_boundary"] == 1
    assert manifest["layers"][0]["coverage_expectation"] == "sparse_visible_evidence"
    assert manifest["boundary_clip"]["coverage_contract"] == "sparse_visible_evidence"
    assert audit["layers"][0]["unclassified_pixel_count_inside_boundary"] == 1
