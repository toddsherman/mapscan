import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from mapscan.directional_coast_refinement import refine_directional_west_coast


def _write_line(path: Path, size: tuple[int, int], segments: list[tuple]) -> None:
    image = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    for segment in segments:
        draw.line(segment, fill=(90, 255, 80, 255), width=1)
    image.save(path)


def test_directional_coast_refinement_pins_east_and_islands(tmp_path: Path) -> None:
    width, height = 120, 140
    canonical_root = tmp_path / "canonical"
    canonical_root.mkdir()
    _write_line(
        canonical_root / "mainland.png",
        (width, height),
        [((12, 5), (12, 120)), ((12, 5), (92, 5)), ((92, 5), (104, 120)), ((12, 120), (104, 120))],
    )
    island = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    ImageDraw.Draw(island).ellipse((36, 122, 46, 132), outline=(90, 255, 80, 255), width=1)
    island.save(canonical_root / "islands.png")
    grid = {
        "crs": "EPSG:3857",
        "bounds": [0.0, 0.0, 120.0, 140.0],
        "width": width,
        "height": height,
        "resampling": "nearest",
        "clip_to_state": True,
    }
    manifest = canonical_root / "canonical.json"
    manifest.write_text(
        json.dumps(
            {
                "canonical_boundary_id": "test-canonical",
                "source_grid": grid,
                "artifacts": {
                    "mainland": {"path": "mainland.png"},
                    "islands": {"path": "islands.png"},
                },
            }
        )
    )
    alignment = tmp_path / "alignment.json"
    alignment.write_text(
        json.dumps({"schema_version": 1, "best": {"transform_model": "affine_like"}})
    )

    report = refine_directional_west_coast(
        alignment,
        manifest,
        tmp_path / "out",
        northward_shift_px=3.0,
        radius_px=24.0,
        coast_control_count=5,
        east_pin_count=5,
        horizontal_edge_pin_count=3,
        start_y_fraction=0.08,
        end_y_fraction=0.80,
    )

    assert report["status"] == "needs_visual_review"
    assert report["controls"]["coast"] == 5
    assert report["controls"]["north_edge_pins"] == 3
    assert report["controls"]["south_edge_pins"] == 3
    assert report["controls"]["island_components"] == 1
    assert report["controls"]["maximum_fixed_pin_delta_px"] == 0.0
    candidate = json.loads((tmp_path / "out" / "fit" / "alignment.json").read_text())
    operation = candidate["web_mercator_correction"]["operations"][0]
    deltas = np.asarray(operation["sampling_delta_at_controls_px"])
    assert np.allclose(deltas[:5], [0.0, 3.0])
    assert np.allclose(deltas[5:], 0.0)
    assert candidate["web_mercator_correction"]["local_fit"]["sampled_jacobian_min"] > 0


def test_directional_coast_refinement_supports_west_only_motion(tmp_path: Path) -> None:
    width, height = 120, 140
    canonical_root = tmp_path / "canonical"
    canonical_root.mkdir()
    _write_line(
        canonical_root / "mainland.png",
        (width, height),
        [((12, 5), (12, 120)), ((92, 5), (104, 120))],
    )
    _write_line(
        canonical_root / "islands.png",
        (width, height),
        [((36, 125), (46, 125)), ((41, 120), (41, 130))],
    )
    grid = {
        "crs": "EPSG:3857",
        "bounds": [0.0, 0.0, 120.0, 140.0],
        "width": width,
        "height": height,
        "resampling": "nearest",
        "clip_to_state": True,
    }
    manifest = canonical_root / "canonical.json"
    manifest.write_text(
        json.dumps(
            {
                "canonical_boundary_id": "test-canonical",
                "source_grid": grid,
                "artifacts": {
                    "mainland": {"path": "mainland.png"},
                    "islands": {"path": "islands.png"},
                },
            }
        )
    )
    alignment = tmp_path / "alignment.json"
    alignment.write_text(
        json.dumps({"schema_version": 1, "best": {"transform_model": "affine_like"}})
    )

    refine_directional_west_coast(
        alignment,
        manifest,
        tmp_path / "out",
        westward_shift_px=4.0,
        radius_px=24.0,
        coast_control_count=5,
        east_pin_count=5,
        start_y_fraction=0.08,
        end_y_fraction=0.80,
    )

    candidate = json.loads((tmp_path / "out" / "fit" / "alignment.json").read_text())
    deltas = np.asarray(
        candidate["web_mercator_correction"]["operations"][0][
            "sampling_delta_at_controls_px"
        ]
    )
    assert np.allclose(deltas[:5], [4.0, 0.0])
    assert np.allclose(deltas[5:], 0.0)
