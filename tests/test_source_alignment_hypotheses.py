import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from mapscan.source_alignment_hypotheses import (
    SourceHypothesisConfig,
    generate_source_alignment_hypotheses,
)


def _synthetic_web_map(path: Path):
    height, width = 320, 480
    pixels = np.full((height, width, 3), (224, 235, 238), dtype=np.uint8)
    pixels[:, 350:] = (249, 249, 249)
    pixels[:30, 350:] = (48, 48, 48)
    cv2.line(pixels, (349, 0), (349, height - 1), (120, 120, 120), 2)

    state = np.zeros((height, width), dtype=np.uint8)
    polygon = np.asarray(
        [[42, 24], [245, 30], [285, 95], [325, 150], [296, 260], [170, 302], [92, 250], [52, 150]],
        dtype=np.int32,
    )
    cv2.fillPoly(state, [polygon], 1)
    colors = [
        (92, 125, 166),
        (129, 156, 190),
        (211, 164, 171),
        (190, 119, 132),
        (174, 91, 105),
    ]
    for index, color in enumerate(colors):
        band = state.astype(bool).copy()
        band[: 24 + index * 56] = False
        band[24 + (index + 1) * 56 :] = False
        pixels[band] = color
    # The last few polygon rows may fall outside the equal bands.
    pixels[(state > 0) & np.all(pixels == (224, 235, 238), axis=2)] = colors[-1]
    cv2.polylines(pixels, [polygon], True, (30, 30, 30), 2)

    for index, color in enumerate(colors):
        y = 68 + index * 39
        cv2.rectangle(pixels, (384, y), (401, y + 15), color, -1)
        cv2.putText(
            pixels,
            f"class {index + 1}",
            (410, y + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (40, 40, 40),
            1,
            cv2.LINE_AA,
        )
    Image.fromarray(pixels).save(path)
    return state.astype(bool), colors


def test_sidebar_swatch_palette_produces_multiple_auditable_hypotheses(tmp_path):
    source = tmp_path / "source.png"
    expected_state, colors = _synthetic_web_map(source)

    result = generate_source_alignment_hypotheses(source, tmp_path / "output")

    assert any(len(group.swatches) == len(colors) for group in result.legend_groups)
    assert any(canvas.id == "full-canvas" for canvas in result.canvases)
    cropped = [canvas for canvas in result.canvases if canvas.kind == "exclude_right_ui"]
    assert cropped
    assert 340 <= cropped[0].box_working[2] <= 355

    palette = next(
        hypothesis
        for hypothesis in result.hypotheses
        if hypothesis.canvas_id == cropped[0].id
        and hypothesis.support_kind == "legend_palette"
        and len(hypothesis.palette_rgb) == len(colors)
    )
    support_path = result.manifest_path.parent / palette.artifacts["support_mask"]
    support = np.asarray(Image.open(support_path)) > 0
    intersection = np.count_nonzero(support & expected_state)
    union = np.count_nonzero(support | expected_state)
    assert intersection / union >= 0.90
    assert palette.diagnostics["palette_usage_fraction"] == 1.0
    assert palette.diagnostics["component_coherence"] >= 0.90
    assert (result.manifest_path.parent / palette.artifacts["state_boundary_mask"]).is_file()
    assert result.layout_diagnostic_path.is_file()
    assert result.legend_diagnostic_path.is_file()

    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["authority"] == {
        "original_source_pixels_only": True,
        "legacy_alignment_used": False,
        "legacy_extraction_used": False,
        "manual_input_used": False,
    }
    assert len(manifest["canvas_hypotheses"]) >= 2
    assert len(manifest["hypotheses"]) >= 2


def test_real_rainfall_screenshot_separates_sidebar_and_recovers_palette_support(
    tmp_path,
):
    source = Path(__file__).parents[1] / "examples" / "rainfall.png"
    result = generate_source_alignment_hypotheses(
        source,
        tmp_path / "rainfall",
        config=SourceHypothesisConfig(working_max_dimension=900),
    )

    legend = max(result.legend_groups, key=lambda group: len(group.swatches))
    assert len(legend.swatches) == 10
    cropped = max(
        (canvas for canvas in result.canvases if canvas.kind == "exclude_right_ui"),
        key=lambda canvas: canvas.score,
    )
    source_width = Image.open(source).width
    right_fraction = cropped.box_original[2] / source_width
    assert 0.70 <= right_fraction <= 0.78
    full = next(canvas for canvas in result.canvases if canvas.id == "full-canvas")
    assert cropped.score > full.score

    support = max(
        (
            hypothesis
            for hypothesis in result.hypotheses
            if hypothesis.canvas_id == cropped.id
            and hypothesis.legend_group_id == legend.id
        ),
        key=lambda hypothesis: hypothesis.score,
    )
    assert support.diagnostics["observed_palette_color_count"] == 10
    assert support.diagnostics["palette_usage_fraction"] == 1.0
    assert support.diagnostics["component_coherence"] >= 0.90
    assert 0.30 <= support.diagnostics["support_fraction_of_roi"] <= 0.40
    boundary = np.asarray(
        Image.open(result.manifest_path.parent / support.artifacts["state_boundary_mask"])
    )
    assert np.count_nonzero(boundary) >= 3_000


@pytest.mark.parametrize("filename", ["population.png", "forest.jpg"])
def test_ordinary_full_map_sources_always_retain_full_canvas(filename, tmp_path):
    source = Path(__file__).parents[1] / "examples" / filename
    with Image.open(source) as image:
        expected_box = (0, 0, image.width, image.height)
    result = generate_source_alignment_hypotheses(
        source,
        tmp_path / filename,
        config=SourceHypothesisConfig(working_max_dimension=700),
    )

    full = next(canvas for canvas in result.canvases if canvas.id == "full-canvas")
    assert full.box_original == expected_box
    assert any(
        hypothesis.canvas_id == full.id for hypothesis in result.hypotheses
    )


def test_real_deer_legend_recovers_all_ten_aligned_swatches(tmp_path):
    source = Path(__file__).parents[1] / "examples" / "deer.png"
    result = generate_source_alignment_hypotheses(source, tmp_path / "deer")

    legend = max(result.legend_groups, key=lambda group: group.score)
    assert len(legend.swatches) == 10
    assert sum(
        swatch.detection_kind == "recovered_regular_gap"
        for swatch in legend.swatches
    ) == 1
    # The non-uniform red-orange swatch omitted by the old detector is second.
    assert legend.swatches[1].rgb[0] > legend.swatches[1].rgb[1]
    assert legend.swatches[1].rgb[1] > legend.swatches[1].rgb[2]


def test_real_forest_legend_has_eight_colors_and_excludes_false_black_box(tmp_path):
    source = Path(__file__).parents[1] / "examples" / "forest.jpg"
    result = generate_source_alignment_hypotheses(source, tmp_path / "forest")

    legend = max(result.legend_groups, key=lambda group: group.score)
    assert len(legend.swatches) == 8
    assert all(max(swatch.rgb) >= 90 for swatch in legend.swatches)
    assert legend.box_original[1] >= 380
    assert legend.box_original[3] <= 550


def test_no_legend_still_emits_a_foreground_fallback(tmp_path):
    pixels = np.full((160, 180, 3), (225, 235, 245), dtype=np.uint8)
    cv2.ellipse(pixels, (85, 80), (48, 66), 0, 0, 360, (30, 165, 80), -1)
    source = tmp_path / "plain.png"
    Image.fromarray(pixels).save(source)

    result = generate_source_alignment_hypotheses(source, tmp_path / "output")

    assert not result.legend_groups
    assert any(
        hypothesis.support_kind == "border_connected_foreground"
        for hypothesis in result.hypotheses
    )


def test_output_directory_must_be_new_to_preserve_attempt_artifacts(tmp_path):
    source = tmp_path / "plain.png"
    Image.new("RGB", (40, 50), (120, 160, 200)).save(source)
    output = tmp_path / "existing"
    output.mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        generate_source_alignment_hypotheses(source, output)
