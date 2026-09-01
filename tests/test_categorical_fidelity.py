import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from mapscan.categorical_fidelity import audit_categorical_fidelity
from mapscan.extraction import classify_categorical


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def _fixture(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    layer_dir = run / "zones"
    layer_dir.mkdir(parents=True)
    rgb = np.full((8, 9, 3), 255, dtype=np.uint8)
    rgb[1:5, 1:4] = [200, 30, 30]
    rgb[2:7, 5:8] = [30, 180, 45]
    rgb[4, 3] = [214, 45, 45]
    source = tmp_path / "source.png"
    Image.fromarray(rgb).save(source)
    state = np.ones(rgb.shape[:2], dtype=np.uint8) * 255
    Image.fromarray(state).save(run / "source-state-mask.png")
    categories = [
        {"id": "red", "label": "Red", "legend_rgb": [200, 30, 30]},
        {"id": "green", "label": "Green", "legend_rgb": [30, 180, 45]},
    ]
    values, _ = classify_categorical(
        rgb, state > 0, categories, max_distance=24.0, min_margin=3.0
    )
    Image.fromarray(values).save(layer_dir / "source-class-id.png")
    Image.fromarray(values).save(layer_dir / "web-mercator-class-id.png")
    Image.fromarray(state).save(layer_dir / "web-mercator-publication-interior-mask.png")
    plan = {
        "schema_version": 1,
        "dataset_id": "fixture",
        "title": "Fixture",
        "source": str(source),
        "layers": [
            {
                "id": "zones",
                "kind": "categorical",
                "max_distance": 24.0,
                "min_margin": 3.0,
                "categories": categories,
            }
        ],
    }
    plan_path = tmp_path / "plan.json"
    _write_json(plan_path, plan)
    (run / "plan.snapshot.json").write_bytes(plan_path.read_bytes())
    _write_json(
        run / "extraction.json",
        {
            "source": {"path": str(source), "sha256": _sha256(source)},
            "plan": {"path": str(plan_path), "sha256": _sha256(plan_path)},
            "layers": [{"id": "zones", "kind": "categorical"}],
        },
    )
    return run


def test_audits_exact_recomputation_and_writes_category_evidence(tmp_path):
    run = _fixture(tmp_path)
    output = tmp_path / "audit"
    result = audit_categorical_fidelity(run, output)
    layer = result["layers"][0]
    assert result["status"] == "pass"
    assert layer["baseline_recomputed_exactly"] is True
    assert layer["semantic_class_change_pixel_count"] == 0
    assert layer["web_colored_pixel_count_outside_canonical_interior"] == 0
    assert len(layer["categories"]) == 2
    assert (output / "zones/categories/red-source-mask.png").is_file()
    assert (output / "zones/source-fidelity-diagnostic.jpg").is_file()


def test_refuses_a_changed_stored_classification(tmp_path):
    run = _fixture(tmp_path)
    path = run / "zones/source-class-id.png"
    values = np.asarray(Image.open(path), dtype=np.uint8).copy()
    values[0, 0] = 1
    Image.fromarray(values).save(path)
    try:
        audit_categorical_fidelity(run, tmp_path / "audit")
    except ValueError as error:
        assert "do not recompute" in str(error)
    else:
        raise AssertionError("Changed source classes should block the audit")


def test_preserves_an_unobserved_legend_category_as_zero_coverage(tmp_path):
    run = _fixture(tmp_path)
    plan_path = Path(json.loads((run / "extraction.json").read_text())["plan"]["path"])
    plan = json.loads(plan_path.read_text())
    plan["layers"][0]["categories"].append(
        {"id": "blue", "label": "Blue", "legend_rgb": [20, 40, 220]}
    )
    _write_json(plan_path, plan)
    (run / "plan.snapshot.json").write_bytes(plan_path.read_bytes())
    extraction = json.loads((run / "extraction.json").read_text())
    extraction["plan"]["sha256"] = _sha256(plan_path)
    _write_json(run / "extraction.json", extraction)

    result = audit_categorical_fidelity(run, tmp_path / "audit")
    layer = result["layers"][0]

    assert result["status"] == "pass"
    assert layer["empty_categories"] == ["blue"]
    assert layer["empty_category_policy"] == "preserve_legend_entry_as_zero_coverage"
    assert layer["empty_categories_require_visual_review"] is True
    assert layer["categories"][2]["visible_source_status"] == "absent"
