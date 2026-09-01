#!/usr/bin/env python3
"""Compare a categorical extraction with its internal-water exclusion mask."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from pyproj import Transformer


def _mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8) > 0


def _pixel_bounds(grid: dict, bbox: list[float]) -> tuple[int, int, int, int]:
    west, south, east, north = bbox
    min_x, min_y, max_x, max_y = (float(value) for value in grid["bounds"])
    width, height = int(grid["width"]), int(grid["height"])
    transformer = Transformer.from_crs("EPSG:4326", grid["crs"], always_xy=True)
    left_x, top_y = transformer.transform(west, north)
    right_x, bottom_y = transformer.transform(east, south)
    left = round((left_x - min_x) / (max_x - min_x) * width)
    right = round((right_x - min_x) / (max_x - min_x) * width)
    top = round((max_y - top_y) / (max_y - min_y) * height)
    bottom = round((max_y - bottom_y) / (max_y - min_y) * height)
    return (
        max(0, left),
        max(0, top),
        min(width, right),
        min(height, bottom),
    )


def _overlay(base: Image.Image, mask: np.ndarray, color: tuple[int, int, int, int]) -> Image.Image:
    surface = base.convert("RGBA")
    tint_values = np.zeros((surface.height, surface.width, 4), dtype=np.uint8)
    tint_values[mask] = color
    surface.alpha_composite(Image.fromarray(tint_values))
    return surface


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("extraction", type=Path)
    parser.add_argument("--layer", required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    extraction = args.extraction.resolve()
    manifest = json.loads((extraction / "extraction.json").read_text())
    layer = next(item for item in manifest["layers"] if item["id"] == args.layer)
    grid = layer["warp"]
    box = _pixel_bounds(grid, list(args.bbox))
    layer_dir = extraction / args.layer

    water = _mask(layer_dir / "web-mercator-internal-water-mask.png")
    removed = _mask(layer_dir / "web-mercator-boundary-removed-mask.png")
    completion = _mask(layer_dir / "web-mercator-completion-mask.png")
    publication = _mask(layer_dir / "web-mercator-publication-interior-mask.png")
    direct_removed = water & removed & ~completion
    completed_removed = water & removed & completion
    left, top, right, bottom = box
    crop = np.s_[top:bottom, left:right]

    source = Image.open(extraction / "web-mercator-source.jpg").convert("RGB")
    preview = Image.open(layer_dir / "web-mercator-preview.png").convert("RGBA")
    baseline_report = None
    restored = np.zeros_like(water)
    removed_classes = np.zeros_like(water)
    reassigned = np.zeros_like(water)
    if args.baseline:
        baseline_layer = args.baseline.resolve() / args.layer
        baseline_classes = np.asarray(
            Image.open(baseline_layer / "web-mercator-class-id.png"), dtype=np.uint8
        )
        candidate_classes = np.asarray(
            Image.open(layer_dir / "web-mercator-class-id.png"), dtype=np.uint8
        )
        restored = (baseline_classes == 0) & (candidate_classes > 0)
        removed_classes = (baseline_classes > 0) & (candidate_classes == 0)
        reassigned = (
            (baseline_classes > 0)
            & (candidate_classes > 0)
            & (baseline_classes != candidate_classes)
        )
        baseline_report = {
            "path": str(args.baseline.resolve()),
            "restored_pixel_count": int(np.count_nonzero(restored)),
            "removed_pixel_count": int(np.count_nonzero(removed_classes)),
            "reassigned_pixel_count": int(np.count_nonzero(reassigned)),
            "restored_pixel_count_in_bbox": int(np.count_nonzero(restored[crop])),
            "removed_pixel_count_in_bbox": int(
                np.count_nonzero(removed_classes[crop])
            ),
            "reassigned_pixel_count_in_bbox": int(np.count_nonzero(reassigned[crop])),
        }
    scale = 5
    panels = [
        source.convert("RGBA"),
        _overlay(source, water, (255, 0, 255, 145)),
        _overlay(source, direct_removed, (0, 255, 0, 210)),
        _overlay(preview.convert("RGB"), completed_removed, (255, 145, 0, 210)),
    ]
    if args.baseline:
        panels.extend(
            [
                _overlay(preview.convert("RGB"), restored, (0, 255, 255, 220)),
                _overlay(preview.convert("RGB"), removed_classes, (255, 45, 45, 220)),
                _overlay(preview.convert("RGB"), reassigned, (255, 215, 0, 220)),
            ]
        )
    cropped = [
        panel.crop(box).resize(
            ((right - left) * scale, (bottom - top) * scale),
            Image.Resampling.NEAREST,
        )
        for panel in panels
    ]
    margin = 36
    canvas = Image.new(
        "RGB",
        (sum(panel.width for panel in cropped), cropped[0].height + margin),
        "white",
    )
    labels = [
        "warped source",
        "water mask (magenta)",
        "direct source class removed (green)",
        "completed source class removed (orange)",
    ]
    if args.baseline:
        labels.extend(
            [
                "restored versus baseline (cyan)",
                "removed versus baseline (red)",
                "reassigned versus baseline (yellow)",
            ]
        )
    draw = ImageDraw.Draw(canvas)
    x = 0
    for panel, label in zip(cropped, labels):
        canvas.paste(panel.convert("RGB"), (x, margin))
        draw.text((x + 8, 10), label, fill="black")
        x += panel.width

    args.output.mkdir(parents=True, exist_ok=True)
    image_path = args.output / "water-exclusion-diagnostic.png"
    report_path = args.output / "water-exclusion-diagnostic.json"
    canvas.save(image_path, optimize=True)
    report = {
        "schema_version": 1,
        "extraction": str(extraction),
        "layer": args.layer,
        "bbox_lon_lat": list(args.bbox),
        "bbox_pixels": list(box),
        "water_pixel_count": int(np.count_nonzero(water[crop])),
        "direct_source_class_removed_pixel_count": int(
            np.count_nonzero(direct_removed[crop])
        ),
        "completed_source_class_removed_pixel_count": int(
            np.count_nonzero(completed_removed[crop])
        ),
        "publication_interior_pixel_count": int(np.count_nonzero(publication[crop])),
        "baseline_comparison": baseline_report,
        "artifacts": {"diagnostic": image_path.name},
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
