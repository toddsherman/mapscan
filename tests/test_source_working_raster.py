import hashlib
import shutil

import numpy as np
import pytest
from PIL import Image, features

from mapscan.source_working_raster import (
    GRATICULE_CONTROLS_NAME,
    MANIFEST_NAME,
    SCHEMA_VERSION,
    _graticule_controls_payload,
    load_working_raster_artifact,
    prepare_source_working_raster,
)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _minimal_vector_pdf(width=144, height=72):
    """Return a tiny one-page PDF without relying on a PDF authoring package."""

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width} {height}] "
            "/CropBox [0 0 144 72] /Resources << >> /Contents 4 0 R >>"
        ).encode(),
    ]
    stream = b"q 0.1 0.7 0.2 rg 8 8 60 48 re f Q 1 0 0 RG 2 w 0 0 m 144 72 l S"
    objects.append(
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream"
    )
    output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for index, body in enumerate(objects, 1):
        offsets.append(len(output))
        output.extend(f"{index} 0 obj\n".encode())
        output.extend(body)
        output.extend(b"\nendobj\n")
    xref = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode())
    output.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref}\n%%EOF\n"
        ).encode()
    )
    return bytes(output)


@pytest.mark.parametrize(
    "suffix,image_format,save_options",
    [
        (".gif", "GIF", {}),
        (".webp", "WEBP", {"lossless": True, "quality": 100, "method": 6}),
        pytest.param(
            ".avif",
            "AVIF",
            {"lossless": True},
            marks=pytest.mark.skipif(
                not features.check("avif"), reason="Pillow AVIF codec unavailable"
            ),
        ),
    ],
)
def test_single_frame_formats_become_deterministic_faithful_rgb(
    tmp_path, suffix, image_format, save_options
):
    pixels = np.zeros((12, 16, 3), dtype=np.uint8)
    pixels[:, :5] = (15, 80, 220)
    pixels[:, 5:11] = (230, 180, 20)
    pixels[:, 11:] = (40, 190, 70)
    source = tmp_path / f"source{suffix}"
    Image.fromarray(pixels).save(source, image_format, **save_options)
    original_hash = _sha256(source)
    with Image.open(source) as decoded:
        expected = np.asarray(decoded.convert("RGB"))

    first = prepare_source_working_raster(source, tmp_path / "first")
    second = prepare_source_working_raster(source, tmp_path / "second")

    assert _sha256(source) == original_hash
    with Image.open(first.working_raster_path) as working:
        assert working.mode == "RGB"
        assert np.array_equal(np.asarray(working), expected)
    assert first.source_sha256 == original_hash
    assert first.working_raster_sha256 == second.working_raster_sha256
    assert first.decoded_rgb_sha256 == second.decoded_rgb_sha256
    assert first.manifest["schema_version"] == SCHEMA_VERSION
    assert first.manifest["authority"] == {
        "original_source_authoritative": True,
        "prior_alignment_used": False,
        "prior_extraction_used": False,
        "manual_input_used": False,
    }
    assert first.alignment_input_path.name == "working-raster.png"


def test_multi_frame_gif_is_rejected_instead_of_silently_selecting_a_frame(tmp_path):
    source = tmp_path / "animated.gif"
    first = Image.new("RGB", (8, 8), (255, 0, 0))
    second = Image.new("RGB", (8, 8), (0, 0, 255))
    first.save(source, save_all=True, append_images=[second], duration=50, loop=0)

    with pytest.raises(ValueError, match="single-frame"):
        prepare_source_working_raster(source, tmp_path / "output")


def test_loader_rejects_source_or_working_raster_tampering(tmp_path):
    source = tmp_path / "source.png"
    Image.new("RGB", (10, 9), (30, 60, 90)).save(source)
    artifact = prepare_source_working_raster(source, tmp_path / "output")
    source.write_bytes(b"changed")

    with pytest.raises(ValueError, match="source hash mismatch"):
        load_working_raster_artifact(artifact.manifest_path)


@pytest.mark.skipif(shutil.which("pdftoppm") is None, reason="Poppler unavailable")
def test_pdf_page_is_rendered_at_pinned_dpi_with_vector_provenance(tmp_path):
    source = tmp_path / "source.pdf"
    source.write_bytes(_minimal_vector_pdf())
    original_hash = _sha256(source)

    first = prepare_source_working_raster(
        source,
        tmp_path / "first",
        pdf_page_number=1,
        pdf_dpi=144,
    )
    second = prepare_source_working_raster(
        source,
        tmp_path / "second",
        pdf_page_number=1,
        pdf_dpi=144,
    )

    assert _sha256(source) == original_hash
    assert (first.width, first.height) == (288, 144)
    assert first.working_raster_sha256 == second.working_raster_sha256
    assert first.decoded_rgb_sha256 == second.decoded_rgb_sha256
    assert first.manifest["conversion"]["dpi"] == 144
    assert first.manifest["conversion"]["page_number"] == 1
    assert first.manifest["conversion"]["page_box"] == "cropbox"
    assert first.manifest["conversion"]["pdftoppm_version"].startswith("pdftoppm version")
    assert first.manifest["pdf"]["page_geometry"] == {
        "width_points": 144.0,
        "height_points": 72.0,
        "rotation_degrees": 0,
        "media_box_points": [0.0, 0.0, 144.0, 72.0],
        "crop_box_points": [0.0, 0.0, 144.0, 72.0],
    }
    evidence = first.manifest["pdf"]["vector_evidence"]
    assert evidence["status"] == "available"
    assert evidence["vector_path_object_count"] >= 2
    assert first.manifest["pdf"]["graticule_evidence"]["status"] == "not_detected"
    assert not (first.manifest_path.parent / GRATICULE_CONTROLS_NAME).exists()


def test_pdf_requires_an_explicit_valid_page(tmp_path):
    source = tmp_path / "source.pdf"
    source.write_bytes(_minimal_vector_pdf())

    with pytest.raises(ValueError, match="explicit pdf_page_number"):
        prepare_source_working_raster(source, tmp_path / "no-page")
    with pytest.raises(ValueError, match="outside this 1-page PDF"):
        prepare_source_working_raster(
            source, tmp_path / "wrong-page", pdf_page_number=2
        )


def test_detected_graticule_controls_are_preserved_in_full(monkeypatch):
    geographic = np.asarray([[-124.0, 42.0], [-120.0, 38.0]])
    page_points = np.asarray([[10.5, 20.25], [30.75, 40.125]])

    def fake_controls(_page):
        return geographic, page_points, {
            "meridians_degrees": [-124, -120],
            "parallels_degrees": [38, 42],
            "intersection_count": 2,
        }

    import mapscan.pdf_registration as registration

    monkeypatch.setattr(registration, "extract_graticule_controls", fake_controls)
    payload = _graticule_controls_payload(object(), 3)

    assert payload["status"] == "detected"
    assert payload["page_number"] == 3
    assert payload["geographic_crs"] == "EPSG:4269"
    assert payload["controls"] == [
        {
            "longitude": -124.0,
            "latitude": 42.0,
            "pdf_x_points": 10.5,
            "pdf_y_points": 20.25,
        },
        {
            "longitude": -120.0,
            "latitude": 38.0,
            "pdf_x_points": 30.75,
            "pdf_y_points": 40.125,
        },
    ]
    assert MANIFEST_NAME == "source-adapter.json"
