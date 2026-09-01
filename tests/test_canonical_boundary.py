import hashlib
import json

import cv2
import numpy as np
from PIL import Image

from mapscan.canonical_boundary import (
    CANONICAL_BORDER_ID,
    CANONICAL_BOUNDARY_ID,
    _compose_county_detail_mainland,
    _connect_nearby_components,
    _county_island_components,
    activate_canonical_border,
    load_active_canonical_border,
    promote_county_detail_border,
    rasterize_canonical_boundary,
)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_rasterizes_the_approved_canonical_raster_on_a_matching_grid(tmp_path):
    interior_path = tmp_path / "interior.png"
    border_path = tmp_path / "border.png"
    geojson_path = tmp_path / "boundary.geojson"
    Image.fromarray(np.ones((4, 4), dtype=np.uint8) * 255).save(interior_path)
    Image.fromarray(np.ones((4, 4), dtype=np.uint8) * 255).save(border_path)
    geojson_path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {},
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [
                                [0, 0],
                                [1, 0],
                                [1, 1],
                                [0, 1],
                                [0, 0],
                            ],
                        },
                    }
                ],
            }
        )
    )
    source_grid = {
        "crs": "EPSG:4326",
        "bounds": [0.0, 0.0, 1.0, 1.0],
        "width": 4,
        "height": 4,
    }
    manifest = {
        "schema_version": 1,
        "status": "approved_canonical_reference",
        "canonical_boundary_id": CANONICAL_BOUNDARY_ID,
        "source_grid": source_grid,
        "artifacts": {
            "interior": {"path": interior_path.name, "sha256": _sha256(interior_path)},
            "border": {"path": border_path.name, "sha256": _sha256(border_path)},
            "geojson": {"path": geojson_path.name, "sha256": _sha256(geojson_path)},
        },
    }
    manifest_path = tmp_path / "canonical-boundary.json"
    manifest_path.write_text(json.dumps(manifest))
    target_grid = {**source_grid, "width": 2, "height": 2}

    result = rasterize_canonical_boundary(
        manifest_path, target_grid, tmp_path / "target"
    )

    assert result["status"] == "pass"
    assert result["method"] == "rasterize_author_approved_canonical_ordered_geojson_ring"
    assert result["topology"]["interior_component_count"] == 1
    assert result["topology"]["border_component_count"] == 1
    assert result["topology"]["interior_pixel_count"] == 4
    vector = result["artifacts"]["border_vector"]
    vector_path = tmp_path / "target" / vector["path"]
    assert _sha256(vector_path) == vector["sha256"]
    svg = vector_path.read_text()
    assert 'width="2" height="2"' in svg
    assert 'viewBox="0 0 2 2"' in svg
    assert 'stroke="#5aff78"' in svg
    assert 'stroke-width="0.5"' in svg
    assert svg.count("<path ") == 1


def test_extracts_four_county_state_stroke_islands_and_rejects_watermark():
    rgba = np.zeros((600, 600, 4), dtype=np.uint8)
    cv2.rectangle(rgba, (40, 20), (560, 580), (0, 0, 0, 255), 6)
    for left, top in ((210, 450), (290, 450), (360, 480), (280, 520)):
        cv2.rectangle(rgba, (left, top), (left + 50, top + 32), (0, 0, 0, 255), 6)
    cv2.putText(
        rgba,
        "brand",
        (490, 570),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0, 255),
        2,
        cv2.LINE_8,
    )

    islands, records = _county_island_components(rgba)

    assert len(records) == 4
    assert cv2.connectedComponents(islands.astype(np.uint8), 8)[0] - 1 == 4


def test_connects_only_distinct_authority_spans_with_short_seams():
    mask = np.zeros((60, 60), dtype=bool)
    mask[10, 10:41] = True
    mask[13:41, 43] = True
    mask[43, 10:41] = True
    mask[13:41, 7] = True

    connected, joins = _connect_nearby_components(mask, maximum_gap_px=5.0)

    assert cv2.connectedComponents(connected.astype(np.uint8), 8)[0] - 1 == 1
    assert len(joins) == 4
    assert max(item["gap_px"] for item in joins) < 5.0


def test_county_coast_replacement_removes_a_fill_derived_bay_shortcut():
    coast = np.zeros((100, 100), dtype=bool)
    coast[10:81, 20] = True
    coast[10:81, 40] = True
    coast[80, 20:41] = True
    continuous = coast.copy()
    continuous[10, 20:41] = True

    candidate, joins = _compose_county_detail_mainland(continuous, coast)

    assert np.array_equal(candidate, coast)
    assert not candidate[10, 30]
    assert joins == []


def test_promotes_and_activates_the_exact_county_detail_linework(tmp_path):
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    mainland = np.zeros((80, 80, 4), dtype=np.uint8)
    cv2.rectangle(mainland, (5, 5), (74, 74), (90, 255, 120, 255), 1)
    islands = np.zeros_like(mainland)
    for left, top in ((12, 55), (25, 58), (38, 61), (51, 64)):
        cv2.rectangle(islands, (left, top), (left + 5, top + 4), (90, 255, 120, 255), 1)
    combined = np.maximum(mainland, islands)
    paths = {}
    for name, values in (
        ("mainland", mainland),
        ("islands", islands),
        ("overlay", combined),
    ):
        path = candidate_root / f"{name}.png"
        Image.fromarray(values).save(path)
        paths[name] = {"path": path.name, "sha256": _sha256(path)}
    candidate = {
        "schema_version": 1,
        "status": "needs_author_review",
        "candidate_boundary_id": "california-county-detail-boundary-v2",
        "base_canonical_boundary_id": CANONICAL_BOUNDARY_ID,
        "authority": {
            "coast_and_bays": "registered_county_png_state_stroke_exact_pixels",
            "offshore_islands": "four_registered_county_png_state_stroke_components",
            "land_borders": "accepted_Census_2025_hybrid_spans",
        },
        "grid": {
            "crs": "EPSG:3857",
            "bounds": [0, 0, 80, 80],
            "width": 80,
            "height": 80,
        },
        "topology": {
            "mainland_component_count": 1,
            "offshore_island_component_count": 4,
            "combined_component_count": 5,
            "county_coast_dropped_pixel_count": 0,
            "san_francisco_bay": {"exact": True},
        },
        "artifacts": paths,
        "publication_allowed": False,
    }
    candidate_path = candidate_root / "county-detail-boundary.json"
    candidate_path.write_text(json.dumps(candidate))
    predecessor_path = tmp_path / "canonical-v1.json"
    predecessor_path.write_text(
        json.dumps(
            {
                "status": "approved_canonical_reference",
                "canonical_boundary_id": CANONICAL_BOUNDARY_ID,
            }
        )
    )

    output_root = tmp_path / "canonical-v2"
    promoted = promote_county_detail_border(
        candidate_path,
        predecessor_path,
        output_root,
        author_statement="Use this as the canonical border for all maps going forward.",
    )
    pointer_path = tmp_path / "canonical-california-boundary.json"
    pointer = activate_canonical_border(
        output_root / "canonical-boundary.json", pointer_path
    )
    resolved_path, resolved, resolved_pointer = load_active_canonical_border(
        pointer_path
    )

    assert promoted["canonical_boundary_id"] == CANONICAL_BORDER_ID
    assert promoted["policy"]["reconstruct_from_filled_mask"] is False
    assert promoted["artifacts"]["overlay"]["sha256"] == paths["overlay"]["sha256"]
    assert pointer["canonical_boundary_id"] == CANONICAL_BORDER_ID
    assert resolved_path == (output_root / "canonical-boundary.json").resolve()
    assert resolved == promoted
    assert resolved_pointer == pointer
