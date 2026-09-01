import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from mapscan.extraction_preview_export import export_extraction_preview_tiles


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_exports_unapproved_extraction_preview_without_carrying_approval(tmp_path: Path) -> None:
    run = tmp_path / "run"
    output = tmp_path / "preview"
    reference = tmp_path / "reference"
    layer_dir = run / "zones"
    layer_dir.mkdir(parents=True)
    reference.mkdir()
    values = np.array([[1, 0], [2, 1]], dtype=np.uint8)
    Image.fromarray(values).save(layer_dir / "web-mercator-class-id.png")
    overlay = np.zeros((2, 2, 4), dtype=np.uint8)
    overlay[..., 1] = 255
    overlay[..., 3] = 255
    Image.fromarray(overlay).save(reference / "boundary.png")
    boundary = {
        "canonical_boundary_id": "test-boundary",
        "source_grid": {
            "crs": "EPSG:3857",
            "bounds": [-1000, -1000, 1000, 1000],
            "width": 2,
            "height": 2,
        },
        "topology": {"combined_component_count": 1},
        "artifacts": {
            "overlay": {
                "path": "boundary.png",
                "sha256": _sha256(reference / "boundary.png"),
            }
        },
    }
    boundary_path = reference / "canonical-boundary.json"
    boundary_path.write_text(json.dumps(boundary))
    plan = {
        "dataset_id": "test",
        "title": "Test",
        "layers": [
            {
                "id": "zones",
                "categories": [
                    {"id": "one", "label": "One", "display_rgb": [1, 2, 3]},
                    {"id": "two", "label": "Two", "display_rgb": [4, 5, 6]},
                ],
            }
        ],
    }
    plan_path = run / "plan.snapshot.json"
    plan_path.write_text(json.dumps(plan))
    extraction = {
        "status": "needs_visual_review",
        "dataset_id": "test",
        "title": "Test",
        "plan": {"sha256": _sha256(plan_path)},
        "layers": [
            {
                "id": "zones",
                "kind": "categorical",
                "warp": {
                    "bounds": [-1000, -1000, 1000, 1000],
                    "width": 2,
                    "height": 2,
                },
                "web_mercator_category_pixel_counts": {"one": 2, "two": 1},
                "web_mercator_classified_pixel_count": 3,
                "canonical_clip": {
                    "valid_pixel_count": 3,
                    "colored_pixel_count_outside_boundary": 0,
                    "active_manifest": {
                        "path": str(boundary_path),
                        "sha256": _sha256(boundary_path),
                    },
                    "artifacts": {"interior": {"sha256": "abc"}},
                },
            }
        ],
    }
    (run / "extraction.json").write_text(json.dumps(extraction))

    result = export_extraction_preview_tiles(
        run,
        output,
        minimum_zoom=0,
        maximum_zoom=1,
        overview_supersampling=2,
    )

    dataset = json.loads((output / "dataset.json").read_text())
    provenance = json.loads((output / "preview-provenance.json").read_text())
    assert result["status"] == "needs_visual_review"
    assert dataset["approval"] == {"status": "not_approved"}
    assert provenance["publication_approved"] is False
    assert dataset["boundary"]["unclassified_pixel_count_inside_boundary"] == 0
    assert dataset["layers"][0]["categories"][0]["pixel_count"] == 2
    assert "?v=" in dataset["layers"][0]["categories"][0]["tile_template"]
    assert (output / "canonical-boundary.png").is_file()
    assert (output / "tiles/zones/one/1/0/0.png").is_file()
