import hashlib
import json

import cv2
import numpy as np
from PIL import Image

from mapscan.canonical_clip import (
    active_boundary_publication_interior,
    canonical_publication_interior,
    close_west_coast_clipping_seam,
    snap_named_water_to_active_boundary,
)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_canonical_publication_interior_combines_mainland_and_active_island(tmp_path):
    grid = {
        "crs": "EPSG:4326",
        "bounds": [0.0, 0.0, 10.0, 10.0],
        "width": 100,
        "height": 100,
    }
    mainland_dir = tmp_path / "mainland"
    mainland_dir.mkdir()
    geojson_path = mainland_dir / "boundary.geojson"
    geojson_path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[
                                    [1.0, 9.0],
                                    [6.0, 9.0],
                                    [6.0, 4.0],
                                    [1.0, 4.0],
                                    [1.0, 9.0],
                                ], [
                                    [2.0, 8.0],
                                    [3.0, 8.0],
                                    [3.0, 7.0],
                                    [2.0, 7.0],
                                    [2.0, 8.0],
                                ],
                            ],
                        },
                    }
                ],
            }
        )
    )
    mainland_manifest_path = mainland_dir / "canonical-boundary.json"
    mainland_manifest_path.write_text(
        json.dumps(
            {
                "status": "pinned_pipeline_reference",
                "kind": "clipping_interior",
                "canonical_clipping_id": "california-mainland-clipping-v2",
                "artifacts": {
                    "geojson": {
                        "path": geojson_path.name,
                        "sha256": _sha256(geojson_path),
                    }
                },
            }
        )
    )

    active_dir = tmp_path / "active"
    active_dir.mkdir()
    island = np.zeros((100, 100, 4), dtype=np.uint8)
    cv2.rectangle(island, (74, 70), (86, 82), (90, 255, 120, 255), 1)
    island_path = active_dir / "islands.png"
    Image.fromarray(island).save(island_path)
    active_manifest_path = active_dir / "canonical-boundary.json"
    active_manifest_path.write_text(
        json.dumps(
            {
                "canonical_boundary_id": "california-county-detail-border-v2",
                "source_grid": grid,
                "topology": {"offshore_island_component_count": 1},
                "artifacts": {
                    "islands": {
                        "path": island_path.name,
                        "sha256": _sha256(island_path),
                    }
                },
            }
        )
    )
    pointer_path = tmp_path / "active.json"
    pointer_path.write_text(
        json.dumps(
            {
                "canonical_boundary_id": "california-county-detail-border-v2",
                "manifest": {
                    "path": "active/canonical-boundary.json",
                    "sha256": _sha256(active_manifest_path),
                },
            }
        )
    )

    valid, provenance = canonical_publication_interior(
        grid,
        mainland_manifest_path=mainland_manifest_path,
        active_pointer_path=pointer_path,
    )

    assert valid[15, 15]
    assert not valid[25, 25]
    assert valid[76, 80]
    assert not valid[95, 95]
    assert provenance["component_count"] == 2
    assert provenance["display_border_reconstructed_from_fill"] is False
    assert (
        provenance["mainland_manifest"]["canonical_clipping_id"]
        == "california-mainland-clipping-v2"
    )


def test_active_boundary_publication_interior_preserves_concave_bay_and_island(
    tmp_path,
):
    grid = {
        "crs": "EPSG:3857",
        "bounds": [0.0, 0.0, 100.0, 100.0],
        "width": 100,
        "height": 100,
    }
    active_dir = tmp_path / "active"
    active_dir.mkdir()

    mainland = np.zeros((100, 100, 4), dtype=np.uint8)
    points = np.asarray(
        [
            [10, 10],
            [70, 10],
            [70, 80],
            [45, 80],
            [45, 45],
            [25, 45],
            [25, 80],
            [10, 80],
        ],
        dtype=np.int32,
    )
    cv2.polylines(mainland, [points], True, (90, 255, 120, 255), 1)
    mainland_path = active_dir / "mainland.png"
    Image.fromarray(mainland).save(mainland_path)

    islands = np.zeros((100, 100, 4), dtype=np.uint8)
    cv2.rectangle(islands, (80, 75), (90, 85), (90, 255, 120, 255), 1)
    island_path = active_dir / "islands.png"
    Image.fromarray(islands).save(island_path)

    active_manifest_path = active_dir / "canonical-boundary.json"
    active_manifest_path.write_text(
        json.dumps(
            {
                "canonical_boundary_id": "california-county-detail-border-v2",
                "source_grid": grid,
                "topology": {"offshore_island_component_count": 1},
                "artifacts": {
                    "mainland": {
                        "path": mainland_path.name,
                        "sha256": _sha256(mainland_path),
                    },
                    "islands": {
                        "path": island_path.name,
                        "sha256": _sha256(island_path),
                    },
                },
            }
        )
    )
    pointer_path = tmp_path / "active.json"
    pointer_path.write_text(
        json.dumps(
            {
                "canonical_boundary_id": "california-county-detail-border-v2",
                "manifest": {
                    "path": "active/canonical-boundary.json",
                    "sha256": _sha256(active_manifest_path),
                },
            }
        )
    )

    valid, provenance = active_boundary_publication_interior(
        grid, active_pointer_path=pointer_path
    )

    assert valid[20, 20]
    assert not valid[60, 35]
    assert valid[80, 85]
    assert not valid[95, 95]
    assert provenance["component_count"] == 2
    assert provenance["display_border_reconstructed_from_fill"] is False
    assert provenance["method"] == (
        "pinned_active_mainland_ring_fill_plus_active_islands"
    )


def test_close_west_coast_seam_fills_only_narrow_gap_and_preserves_island(tmp_path):
    grid = {
        "crs": "EPSG:3857",
        "bounds": [0.0, 0.0, 30.0, 20.0],
        "width": 30,
        "height": 20,
    }
    active_dir = tmp_path / "active"
    active_dir.mkdir()
    line = np.zeros((20, 30, 4), dtype=np.uint8)
    line[2:18, 5, 3] = 255
    line_path = active_dir / "mainland.png"
    Image.fromarray(line).save(line_path)
    manifest_path = active_dir / "canonical-boundary.json"
    manifest_path.write_text(
        json.dumps(
            {
                "canonical_boundary_id": "california-county-detail-border-v2",
                "source_grid": grid,
                "artifacts": {
                    "mainland": {
                        "path": line_path.name,
                        "sha256": _sha256(line_path),
                    }
                },
            }
        )
    )
    pointer_path = tmp_path / "active.json"
    pointer_path.write_text(
        json.dumps(
            {
                "canonical_boundary_id": "california-county-detail-border-v2",
                "manifest": {
                    "path": "active/canonical-boundary.json",
                    "sha256": _sha256(manifest_path),
                },
            }
        )
    )
    interior = np.zeros((20, 30), dtype=bool)
    interior[2:18, 8:24] = True
    interior[14:17, 1:3] = True

    expanded, report, seam = close_west_coast_clipping_seam(
        interior,
        grid,
        maximum_gap_px=4,
        maximum_x_fraction=0.5,
        start_y_fraction=0.0,
        end_y_fraction=1.0,
        active_pointer_path=pointer_path,
    )

    assert np.all(expanded[2:18, 5:8])
    assert np.array_equal(expanded[14:17, 1:3], interior[14:17, 1:3])
    assert report["accepted_row_count"] == 16
    assert report["interpolated_line_row_count"] == 0
    assert report["accepted_gap_max_px"] == 3
    assert report["added_pixel_count"] == int(np.count_nonzero(seam & ~interior))


def test_west_coast_seam_interpolates_missing_canonical_rows(tmp_path):
    grid = {
        "crs": "EPSG:4326",
        "bounds": [0.0, 0.0, 30.0, 20.0],
        "width": 30,
        "height": 20,
    }
    active_dir = tmp_path / "active"
    active_dir.mkdir()
    line = np.zeros((20, 30, 4), dtype=np.uint8)
    for y in range(2, 18, 2):
        line[y, 5] = (0, 0, 0, 255)
    line_path = active_dir / "mainland.png"
    Image.fromarray(line).save(line_path)
    manifest_path = active_dir / "canonical-boundary.json"
    manifest_path.write_text(
        json.dumps(
            {
                "canonical_boundary_id": "california-county-detail-border-v2",
                "source_grid": grid,
                "artifacts": {
                    "mainland": {
                        "path": line_path.name,
                        "sha256": _sha256(line_path),
                    }
                },
            }
        )
    )
    pointer_path = tmp_path / "active.json"
    pointer_path.write_text(
        json.dumps(
            {
                "canonical_boundary_id": "california-county-detail-border-v2",
                "manifest": {
                    "path": "active/canonical-boundary.json",
                    "sha256": _sha256(manifest_path),
                },
            }
        )
    )
    interior = np.zeros((20, 30), dtype=bool)
    interior[2:17, 8:24] = True

    expanded, report, _ = close_west_coast_clipping_seam(
        interior,
        grid,
        maximum_gap_px=4,
        maximum_x_fraction=0.5,
        start_y_fraction=0.0,
        end_y_fraction=1.0,
        active_pointer_path=pointer_path,
    )

    assert np.all(expanded[2:17, 5:8])
    assert report["canonical_line_row_count"] == 8
    assert report["interpolated_line_row_count"] == 7


def test_named_water_uses_lime_line_as_cut_edge(tmp_path):
    grid = {
        "crs": "EPSG:3857",
        "bounds": [0.0, 0.0, 30.0, 20.0],
        "width": 30,
        "height": 20,
    }
    active_dir = tmp_path / "active"
    active_dir.mkdir()
    line = np.zeros((20, 30, 4), dtype=np.uint8)
    cv2.rectangle(line, (10, 2), (25, 17), (0, 0, 0, 255), 1)
    line_path = active_dir / "mainland.png"
    Image.fromarray(line).save(line_path)
    manifest_path = active_dir / "canonical-boundary.json"
    manifest_path.write_text(
        json.dumps(
            {
                "canonical_boundary_id": "california-county-detail-border-v2",
                "source_grid": grid,
                "artifacts": {
                    "mainland": {
                        "path": line_path.name,
                        "sha256": _sha256(line_path),
                    }
                },
            }
        )
    )
    pointer_path = tmp_path / "active.json"
    pointer_path.write_text(
        json.dumps(
            {
                "canonical_boundary_id": "california-county-detail-border-v2",
                "manifest": {
                    "path": "active/canonical-boundary.json",
                    "sha256": _sha256(manifest_path),
                },
            }
        )
    )
    interior = np.ones((20, 30), dtype=bool)
    census_water = np.zeros_like(interior)
    census_water[5:15, 7:14] = True

    snapped, report = snap_named_water_to_active_boundary(
        census_water,
        interior,
        grid,
        maximum_distance_px=5,
        active_pointer_path=pointer_path,
    )

    assert np.any(snapped[:, 5:7])
    assert np.any(snapped[:, 7:10])
    assert not np.any(snapped[3:17, 11:25])
    assert report["added_to_reach_lime_pixel_count"] > 0
    assert report["removed_landward_seed_pixel_count"] > 0
    assert (
        report["active_manifest"]["canonical_boundary_id"]
        == "california-county-detail-border-v2"
    )
