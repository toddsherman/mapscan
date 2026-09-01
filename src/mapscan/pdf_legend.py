"""Extract native color swatches from a vector PDF legend."""

from __future__ import annotations

import csv
import io
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pdfplumber
from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageOps


def cmyk_to_rgb(color: Sequence[float]) -> Tuple[int, int, int]:
    """Convert normalized device CMYK to an sRGB-style diagnostic color."""

    cyan, magenta, yellow, black = map(float, color)
    return tuple(
        int(round(255 * (1.0 - min(1.0, component + black))))
        for component in (cyan, magenta, yellow)
    )


def _legend_region(page) -> Tuple[float, float, float, float]:
    words = page.extract_words(x_tolerance=1, y_tolerance=1)
    geologic = [word for word in words if "GEOLOGIC" in word["text"].upper()]
    legend = [word for word in words if "LEGEND" in word["text"].upper()]
    symbols = [word for word in words if word["text"].upper() == "SYMBOLS"]
    title_pair = None
    for first in geologic:
        for second in legend:
            if abs(float(first["top"]) - float(second["top"])) < 20:
                title_pair = first, second
                break
        if title_pair is not None:
            break
    if title_pair is None:
        # Illustrator may expose a visibly normal title through a custom font
        # encoding that does not reconstruct into words. The main legend still
        # forms the page's largest upper-right cluster of small filled rectangles.
        candidates = []
        for rectangle in page.rects:
            width = float(rectangle["x1"]) - float(rectangle["x0"])
            height = float(rectangle["bottom"]) - float(rectangle["top"])
            fill = rectangle.get("non_stroking_color")
            if (
                float(rectangle["x0"]) > float(page.width) * 0.45
                and float(page.height) * 0.12 < float(rectangle["top"]) < float(page.height) * 0.5
                and 10 <= width <= 80
                and 8 <= height <= 45
                and isinstance(fill, tuple)
                and len(fill) == 4
            ):
                candidates.append(rectangle)
        if len(candidates) < 8:
            raise ValueError("a main PDF legend region could not be located")
        return (
            max(0.0, min(float(item["x0"]) for item in candidates) - 180.0),
            max(0.0, min(float(item["top"]) for item in candidates) - 120.0),
            min(float(page.width), max(float(item["x1"]) for item in candidates) + 120.0),
            min(float(page.height), max(float(item["bottom"]) for item in candidates) + 180.0),
        )
    title_top = min(float(title_pair[0]["top"]), float(title_pair[1]["top"]))
    title_left = min(float(title_pair[0]["x0"]), float(title_pair[1]["x0"]))
    following_symbols = [word for word in symbols if float(word["top"]) > title_top]
    bottom = (
        min(float(word["top"]) for word in following_symbols)
        if following_symbols
        else float(page.height) * 0.7
    )
    return max(0.0, title_left - 450.0), title_top, float(page.width) - 100.0, bottom


def _text_inside(page, bbox: Sequence[float]) -> str:
    x0, top, x1, bottom = bbox
    characters = []
    for character in page.chars:
        center_x = (float(character["x0"]) + float(character["x1"])) / 2
        center_y = (float(character["top"]) + float(character["bottom"])) / 2
        if x0 - 1 <= center_x <= x1 + 1 and top - 1 <= center_y <= bottom + 1:
            characters.append(character)
    characters.sort(key=lambda item: (round(float(item["top"]), 1), float(item["x0"])))
    return "".join(str(character["text"]) for character in characters).strip()


def _swatches(page, region: Sequence[float]) -> List[Dict[str, object]]:
    region_x0, region_top, region_x1, region_bottom = region
    records: List[Dict[str, object]] = []
    seen = set()
    for rectangle in page.rects:
        x0 = float(rectangle["x0"])
        top = float(rectangle["top"])
        x1 = float(rectangle["x1"])
        bottom = float(rectangle["bottom"])
        width = x1 - x0
        height = bottom - top
        fill = rectangle.get("non_stroking_color")
        if not (
            region_x0 <= x0 <= x1 <= region_x1
            and region_top <= top <= bottom <= region_bottom
            and 10 <= width <= 80
            and 8 <= height <= 45
            and isinstance(fill, tuple)
            and len(fill) == 4
        ):
            continue
        key = (
            round(x0, 2),
            round(top, 2),
            round(x1, 2),
            round(bottom, 2),
            tuple(round(float(value), 4) for value in fill),
        )
        if key in seen:
            continue
        seen.add(key)
        bbox = [x0, top, x1, bottom]
        records.append(
            {
                "bbox_page_points": bbox,
                "fill_cmyk": list(map(float, fill)),
                "fill_rgb": list(cmyk_to_rgb(fill)),
                "raw_pdf_text": _text_inside(page, bbox),
            }
        )
    records.sort(key=lambda item: (item["bbox_page_points"][1], item["bbox_page_points"][0]))  # type: ignore[index]
    return records


def _ocr_swatch(
    render: Image.Image,
    bbox: Sequence[float],
    page_size: Sequence[float],
    executable: str,
    temporary_dir: Path,
    index: int,
) -> Tuple[str, float | None]:
    page_width, page_height = page_size
    scale_x = render.width / page_width
    scale_y = render.height / page_height
    x0, top, x1, bottom = bbox
    inset_x = min(2.0, (x1 - x0) * 0.08)
    inset_y = min(2.0, (bottom - top) * 0.08)
    crop = render.crop(
        (
            round((x0 + inset_x) * scale_x),
            round((top + inset_y) * scale_y),
            round((x1 - inset_x) * scale_x),
            round((bottom - inset_y) * scale_y),
        )
    )
    crop = ImageOps.autocontrast(ImageOps.grayscale(crop))
    crop = crop.resize((crop.width * 18, crop.height * 18), Image.Resampling.LANCZOS)
    crop = ImageEnhance.Contrast(crop).enhance(1.7)
    crop_path = temporary_dir / f"swatch-{index:03d}.png"
    crop.save(crop_path)
    process = subprocess.run(
        [executable, str(crop_path), "stdout", "--psm", "7", "tsv"],
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode != 0:
        return "", None
    words = []
    confidences = []
    for row in csv.DictReader(io.StringIO(process.stdout), delimiter="\t"):
        text = (row.get("text") or "").strip()
        try:
            confidence = float(row.get("conf") or -1)
        except ValueError:
            confidence = -1
        if text and confidence >= 0:
            words.append(text)
            confidences.append(confidence)
    combined = " ".join(words).strip()
    combined = re.sub(r"^[|\[\](){} ]+|[|\[\](){} ]+$", "", combined).strip()
    return combined, (sum(confidences) / len(confidences) if confidences else None)


def _write_contact_sheet(records: Sequence[Dict[str, object]], output_path: Path) -> None:
    row_height = 54
    width = 900
    image = Image.new("RGB", (width, max(1, len(records)) * row_height + 36), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((16, 12), "Extracted PDF legend swatches (OCR labels are provisional)", fill="black", font=font)
    for index, record in enumerate(records):
        y = 36 + index * row_height
        rgb = tuple(record["fill_rgb"])  # type: ignore[arg-type]
        draw.rectangle((16, y + 8, 96, y + 44), fill=rgb, outline="black")
        draw.text(
            (112, y + 10),
            f"{record['id']}  OCR={record.get('ocr_text')!r}  native={record.get('raw_pdf_text')!r}",
            fill="black",
            font=font,
        )
        draw.text(
            (112, y + 28),
            f"CMYK={record['fill_cmyk']}  confidence={record.get('ocr_confidence')}",
            fill=(60, 60, 60),
            font=font,
        )
    image.save(output_path)


def extract_pdf_legend(
    pdf_path: Path,
    output_dir: Path,
    render_path: Path | None = None,
    page_number: int = 1,
) -> Dict[str, object]:
    """Extract main-legend swatch colors and provisional labels from one PDF page."""

    output_dir.mkdir(parents=True, exist_ok=True)
    with pdfplumber.open(pdf_path) as document:
        page = document.pages[page_number - 1]
        region = _legend_region(page)
        records = _swatches(page, region)
        executable = shutil.which("tesseract") if render_path is not None else None
        if render_path is not None and executable is not None:
            render = Image.open(render_path).convert("RGB")
            with tempfile.TemporaryDirectory(prefix="mapscan-legend-") as temporary:
                temporary_dir = Path(temporary)
                for index, record in enumerate(records):
                    text, confidence = _ocr_swatch(
                        render,
                        record["bbox_page_points"],  # type: ignore[arg-type]
                        (float(page.width), float(page.height)),
                        executable,
                        temporary_dir,
                        index,
                    )
                    record["ocr_text"] = text
                    record["ocr_confidence"] = confidence
        for index, record in enumerate(records, start=1):
            record["id"] = f"pdf-swatch-{index:03d}"
            record["label_status"] = "provisional_ocr"
        report = {
            "schema_version": 1,
            "status": "diagnostic_only",
            "source": {"pdf_path": str(pdf_path), "page_number": page_number},
            "legend_region_page_points": list(region),
            "swatch_count": len(records),
            "swatches": records,
            "warning": (
                "Colors come from native PDF fill objects. Unit codes are provisional because "
                "the PDF uses custom font encodings; descriptions and hierarchy are not yet attached."
            ),
        }
        (output_dir / "legend-swatches.json").write_text(json.dumps(report, indent=2) + "\n")
        _write_contact_sheet(records, output_dir / "legend-swatches.png")
        return report
