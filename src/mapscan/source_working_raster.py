"""Source-clean working rasters for the no-human MapScan restart.

The original file is always the authority.  This module only decodes or renders
that file into a canonical RGB PNG that downstream automatic alignment can
consume.  It deliberately accepts no prior alignment, extracted pixels, manual
corrections, or artifacts from an older run.

The PNG file hash is useful for artifact integrity.  The separate decoded-RGB
hash is the stable content identity: it covers the exact width, height, mode,
and uncompressed RGB bytes and therefore does not depend on PNG compression.
"""

from __future__ import annotations

import hashlib
import io
import json
import mimetypes
import os
import shutil
import struct
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pdfplumber
from PIL import Image, ImageOps, __version__ as PILLOW_VERSION


SCHEMA_VERSION = "mapscan.source-working-raster.v1"
RASTER_SUFFIXES = {
    ".avif",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
PDF_SUFFIX = ".pdf"
WORKING_RASTER_NAME = "working-raster.png"
MANIFEST_NAME = "source-adapter.json"
GRATICULE_CONTROLS_NAME = "pdf-graticule-controls.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _decoded_rgb_sha256(image: Image.Image) -> str:
    if image.mode != "RGB":
        raise ValueError("decoded RGB hash requires an RGB image")
    width, height = image.size
    header = b"mapscan-decoded-rgb-v1\0" + struct.pack(">II", width, height)
    return _sha256_bytes(header + image.tobytes())


def _canonical_png(image: Image.Image) -> bytes:
    if image.mode != "RGB":
        raise ValueError("working raster must be RGB")
    output = io.BytesIO()
    # No inherited PNG metadata and no adaptive optimizer: the same decoded
    # pixels and pinned Pillow version produce the same working artifact.
    image.save(output, format="PNG", optimize=False, compress_level=9)
    return output.getvalue()


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _write_without_overwrite(path: Path, content: bytes) -> None:
    """Atomically write new content, or verify an identical existing artifact."""

    if path.exists():
        if path.is_file() and path.read_bytes() == content:
            return
        raise FileExistsError(
            f"refusing to overwrite a different source-adapter artifact: {path}"
        )
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_bytes(content)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _metadata_value(value: Any) -> Any:
    """Return a deterministic, JSON-safe representation of source metadata."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return {"byte_count": len(value), "sha256": _sha256_bytes(value)}
    if isinstance(value, (tuple, list)):
        return [_metadata_value(item) for item in value]
    if isinstance(value, Mapping):
        return {
            str(key): _metadata_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    return str(value)


def _source_record(path: Path) -> dict[str, Any]:
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return {
        "path": str(path),
        "filename": path.name,
        "suffix": path.suffix.lower(),
        "media_type": media_type,
        "byte_count": path.stat().st_size,
        "sha256": _sha256(path),
        "authoritative": True,
    }


def _alpha_to_white(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    return Image.alpha_composite(background, rgba).convert("RGB")


def _decode_single_frame_raster(path: Path) -> tuple[Image.Image, dict[str, Any]]:
    with Image.open(path) as source:
        frame_count = int(getattr(source, "n_frames", 1))
        if frame_count != 1:
            raise ValueError(
                f"source-clean adapter requires a single-frame raster; found {frame_count}"
            )
        source.seek(0)
        detected_format = str(source.format or "unknown")
        source_mode = source.mode
        source_size = [int(source.width), int(source.height)]
        info = dict(source.info)
        exif = source.getexif()
        orientation = int(exif.get(274, 1)) if exif else 1
        visual = ImageOps.exif_transpose(source)
        has_alpha = "A" in visual.getbands() or "transparency" in info
        rgb = _alpha_to_white(visual) if has_alpha else visual.convert("RGB")
        rgb.load()

    preserved_metadata = {
        key: _metadata_value(info[key])
        for key in (
            "background",
            "duration",
            "loop",
            "transparency",
            "dpi",
            "icc_profile",
            "exif",
        )
        if key in info
    }
    provenance = {
        "adapter": "pillow-single-frame-rgb-v1",
        "pillow_version": PILLOW_VERSION,
        "detected_format": detected_format,
        "source_mode": source_mode,
        "source_size": source_size,
        "frame_count": frame_count,
        "selected_frame": 0,
        "exif_orientation": orientation,
        "exif_orientation_applied": orientation not in {0, 1},
        "alpha_handling": "composite_over_opaque_white" if has_alpha else "none",
        "color_management": (
            "literal decoder RGB values; embedded ICC metadata is hashed but not applied"
        ),
        "preserved_source_metadata": preserved_metadata,
    }
    return rgb, provenance


def _poppler_version(executable: str) -> str:
    completed = subprocess.run(
        [executable, "-v"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("pdftoppm did not report a version")
    return lines[0]


def _render_pdf_page(
    path: Path,
    output_dir: Path,
    *,
    page_number: int,
    dpi: int,
    pdftoppm_executable: str | Path | None,
) -> tuple[Image.Image, dict[str, Any]]:
    executable = (
        str(pdftoppm_executable)
        if pdftoppm_executable is not None
        else shutil.which("pdftoppm")
    )
    if not executable:
        raise RuntimeError("pdftoppm is required to render a PDF working raster")
    if not Path(executable).is_file():
        raise FileNotFoundError(executable)
    version = _poppler_version(executable)
    arguments = [
        "-f",
        str(page_number),
        "-l",
        str(page_number),
        "-singlefile",
        "-r",
        str(dpi),
        "-cropbox",
        "-freetype",
        "yes",
        "-aa",
        "yes",
        "-aaVector",
        "yes",
        "-thinlinemode",
        "none",
        "-png",
    ]
    with tempfile.TemporaryDirectory(prefix=".pdf-render-", dir=output_dir) as raw:
        temporary = Path(raw)
        prefix = temporary / "selected-page"
        command = [executable, *arguments, str(path), str(prefix)]
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"pdftoppm failed for page {page_number}: {detail}")
        rendered = prefix.with_suffix(".png")
        if not rendered.is_file():
            raise RuntimeError("pdftoppm did not produce the selected PDF page")
        with Image.open(rendered) as source:
            rgb = source.convert("RGB")
            rgb.load()
    return rgb, {
        "adapter": "poppler-pdftoppm-page-rgb-v1",
        "pdftoppm_version": version,
        "pdftoppm_executable": str(Path(executable).resolve()),
        "arguments_before_source_and_output": arguments,
        "page_number": page_number,
        "dpi": dpi,
        "page_box": "cropbox",
        "output_format": "png",
        "post_render_conversion": "Pillow convert RGB",
        "pillow_version": PILLOW_VERSION,
    }


def _graticule_controls_payload(page: Any, page_number: int) -> dict[str, Any]:
    """Discover and preserve native vector graticule controls when available."""

    try:
        from .pdf_registration import extract_graticule_controls

        geographic, page_points, metadata = extract_graticule_controls(page)
    except Exception as error:  # A usable PDF raster need not contain a graticule.
        return {
            "status": "not_detected",
            "detector": "mapscan.pdf_registration.extract_graticule_controls",
            "reason": f"{type(error).__name__}: {error}",
        }

    controls = [
        {
            "longitude": float(lon_lat[0]),
            "latitude": float(lon_lat[1]),
            "pdf_x_points": float(pdf_point[0]),
            "pdf_y_points": float(pdf_point[1]),
        }
        for lon_lat, pdf_point in zip(geographic.tolist(), page_points.tolist())
    ]
    return {
        "schema_version": 1,
        "kind": "pdf_vector_graticule_controls",
        "status": "detected",
        "detector": "mapscan.pdf_registration.extract_graticule_controls",
        "page_number": page_number,
        "geographic_crs": "EPSG:4269",
        "page_coordinate_space": "pdfplumber_page_points",
        "metadata": _metadata_value(metadata),
        "controls": controls,
    }


def _inspect_pdf(
    path: Path,
    *,
    page_number: int,
    inspect_vectors: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    with pdfplumber.open(path) as document:
        page_count = len(document.pages)
        if page_number < 1 or page_number > page_count:
            raise ValueError(
                f"page {page_number} is outside this {page_count}-page PDF"
            )
        page = document.pages[page_number - 1]
        page_record: dict[str, Any] = {
            "page_count": page_count,
            "selected_page_number": page_number,
            "document_metadata": _metadata_value(document.metadata or {}),
            "page_geometry": {
                "width_points": float(page.width),
                "height_points": float(page.height),
                "rotation_degrees": int(page.rotation or 0),
                "media_box_points": [float(value) for value in page.mediabox],
                "crop_box_points": [float(value) for value in page.cropbox],
            },
            "vector_inspection_requested": bool(inspect_vectors),
        }
        if not inspect_vectors:
            page_record["vector_evidence"] = {"status": "not_inspected"}
            return page_record, None

        try:
            objects = page.objects
            object_counts = {
                str(kind): len(values)
                for kind, values in sorted(objects.items())
            }
            vector_count = sum(
                object_counts.get(kind, 0)
                for kind in ("curve", "line", "rect")
            )
            page_record["vector_evidence"] = {
                "status": "available" if vector_count else "none",
                "object_counts": object_counts,
                "vector_path_object_count": vector_count,
            }
        except Exception as error:
            page_record["vector_evidence"] = {
                "status": "inspection_error",
                "reason": f"{type(error).__name__}: {error}",
            }
            return page_record, None

        graticule = _graticule_controls_payload(page, page_number)
        if graticule.get("status") == "detected":
            page_record["graticule_evidence"] = {
                "status": "detected",
                "detector": graticule["detector"],
                "control_count": len(graticule["controls"]),
                "metadata": graticule["metadata"],
            }
            return page_record, graticule
        page_record["graticule_evidence"] = graticule
        return page_record, None


@dataclass(frozen=True)
class WorkingRasterArtifact:
    """A hash-validated working PNG and its source-clean provenance."""

    source_path: Path
    working_raster_path: Path
    manifest_path: Path
    source_sha256: str
    working_raster_sha256: str
    decoded_rgb_sha256: str
    width: int
    height: int
    manifest: Mapping[str, Any]

    @property
    def alignment_input_path(self) -> Path:
        """Path to pass to the automatic alignment loop."""

        return self.working_raster_path


def load_working_raster_artifact(manifest_path: Path) -> WorkingRasterArtifact:
    """Load an adapter manifest only after validating source and output hashes."""

    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported source-working-raster schema")
    authority = manifest.get("authority", {})
    if authority != {
        "original_source_authoritative": True,
        "prior_alignment_used": False,
        "prior_extraction_used": False,
        "manual_input_used": False,
    }:
        raise ValueError("working raster does not have source-clean authority")
    source_record = manifest.get("source", {})
    source_path = Path(str(source_record.get("path", ""))).resolve()
    working_record = manifest.get("working_raster", {})
    working_path = manifest_path.parent / str(working_record.get("path", ""))
    if not source_path.is_file() or _sha256(source_path) != source_record.get("sha256"):
        raise ValueError("authoritative source hash mismatch")
    if not working_path.is_file() or _sha256(working_path) != working_record.get("sha256"):
        raise ValueError("working raster hash mismatch")
    with Image.open(working_path) as image:
        encoded_mode = image.mode
        rgb = image.convert("RGB")
        rgb.load()
    decoded_hash = _decoded_rgb_sha256(rgb)
    if decoded_hash != working_record.get("decoded_rgb_sha256"):
        raise ValueError("working raster decoded-RGB hash mismatch")
    if encoded_mode != "RGB":
        raise ValueError("working raster file must be encoded as RGB")
    if [rgb.width, rgb.height] != [
        int(working_record.get("width", 0)),
        int(working_record.get("height", 0)),
    ]:
        raise ValueError("working raster dimensions disagree with its manifest")
    graticule = manifest.get("pdf", {}).get("graticule_evidence", {})
    controls_name = graticule.get("controls_path")
    controls_hash = graticule.get("controls_sha256")
    if controls_name is not None or controls_hash is not None:
        controls_path = manifest_path.parent / str(controls_name or "")
        if (
            not controls_path.is_file()
            or not isinstance(controls_hash, str)
            or _sha256(controls_path) != controls_hash
        ):
            raise ValueError("PDF graticule-controls hash mismatch")
    return WorkingRasterArtifact(
        source_path=source_path,
        working_raster_path=working_path.resolve(),
        manifest_path=manifest_path,
        source_sha256=str(source_record["sha256"]),
        working_raster_sha256=str(working_record["sha256"]),
        decoded_rgb_sha256=decoded_hash,
        width=int(working_record["width"]),
        height=int(working_record["height"]),
        manifest=manifest,
    )


def prepare_source_working_raster(
    source_path: Path,
    output_dir: Path,
    *,
    pdf_page_number: int | None = None,
    pdf_dpi: int = 300,
    inspect_pdf_vectors: bool = True,
    pdftoppm_executable: str | Path | None = None,
) -> WorkingRasterArtifact:
    """Create a canonical RGB alignment input directly from one original source.

    Raster sources must have exactly one frame.  PDF sources require an explicit
    ``pdf_page_number`` so the selected page is part of the reproducible input
    contract.  Existing different artifacts are never overwritten.
    """

    source_path = source_path.resolve()
    output_dir = output_dir.resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    suffix = source_path.suffix.lower()
    if suffix not in RASTER_SUFFIXES | {PDF_SUFFIX}:
        raise ValueError(f"unsupported source type: {suffix or '(no suffix)'}")
    if suffix == PDF_SUFFIX:
        if pdf_page_number is None:
            raise ValueError("PDF source requires an explicit pdf_page_number")
        if not isinstance(pdf_page_number, int) or pdf_page_number < 1:
            raise ValueError("pdf_page_number must be a positive integer")
        if not isinstance(pdf_dpi, int) or pdf_dpi < 72 or pdf_dpi > 1200:
            raise ValueError("pdf_dpi must be an integer between 72 and 1200")
    elif pdf_page_number is not None:
        raise ValueError("pdf_page_number is only valid for PDF sources")

    output_dir.mkdir(parents=True, exist_ok=True)
    source = _source_record(source_path)
    graticule_payload: dict[str, Any] | None = None
    pdf_record: dict[str, Any] | None = None
    if suffix == PDF_SUFFIX:
        assert pdf_page_number is not None
        pdf_record, graticule_payload = _inspect_pdf(
            source_path,
            page_number=pdf_page_number,
            inspect_vectors=inspect_pdf_vectors,
        )
        rgb, conversion = _render_pdf_page(
            source_path,
            output_dir,
            page_number=pdf_page_number,
            dpi=pdf_dpi,
            pdftoppm_executable=pdftoppm_executable,
        )
        geometry = pdf_record["page_geometry"]
        crop_box = geometry["crop_box_points"]
        crop_width = abs(float(crop_box[2]) - float(crop_box[0]))
        crop_height = abs(float(crop_box[3]) - float(crop_box[1]))
        conversion["actual_x_pixels_per_page_point"] = (
            rgb.width / crop_width
        )
        conversion["actual_y_pixels_per_page_point"] = (
            rgb.height / crop_height
        )
    else:
        rgb, conversion = _decode_single_frame_raster(source_path)

    png = _canonical_png(rgb)
    working_path = output_dir / WORKING_RASTER_NAME
    _write_without_overwrite(working_path, png)
    working = {
        "path": WORKING_RASTER_NAME,
        "format": "PNG",
        "mode": "RGB",
        "width": rgb.width,
        "height": rgb.height,
        "sha256": _sha256_bytes(png),
        "decoded_rgb_sha256": _decoded_rgb_sha256(rgb),
    }
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "source_clean_working_raster",
        "authority": {
            "original_source_authoritative": True,
            "prior_alignment_used": False,
            "prior_extraction_used": False,
            "manual_input_used": False,
        },
        "source": source,
        "conversion": conversion,
        "working_raster": working,
    }
    if pdf_record is not None:
        manifest["pdf"] = pdf_record
    if graticule_payload is not None:
        controls_content = _json_bytes(graticule_payload)
        controls_path = output_dir / GRATICULE_CONTROLS_NAME
        _write_without_overwrite(controls_path, controls_content)
        manifest["pdf"]["graticule_evidence"].update(
            {
                "controls_path": GRATICULE_CONTROLS_NAME,
                "controls_sha256": _sha256_bytes(controls_content),
            }
        )

    manifest_path = output_dir / MANIFEST_NAME
    _write_without_overwrite(manifest_path, _json_bytes(manifest))
    return load_working_raster_artifact(manifest_path)
