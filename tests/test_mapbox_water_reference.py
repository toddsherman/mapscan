import gzip
import hashlib
import json
from pathlib import Path

import mapbox_vector_tile
import numpy as np
import pytest
from shapely.geometry import box, mapping

from mapscan.mapbox_water_reference import (
    WEB_MERCATOR_HALF_WORLD,
    constrain_named_water_to_mapbox,
    rasterize_pinned_mapbox_water,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reference(tmp_path: Path) -> Path:
    root = tmp_path / "mapbox-water"
    tile_path = root / "tiles" / "0" / "0" / "0.mvt"
    tile_path.parent.mkdir(parents=True)
    encoded = mapbox_vector_tile.encode(
        {
            "name": "water",
            "features": [
                {
                    "geometry": mapping(box(0, 0, 2048, 4096)),
                    "properties": {},
                }
            ],
        },
        default_options={"y_coord_down": True},
    )
    tile_path.write_bytes(gzip.compress(encoded))
    style_path = root / "style.json"
    style_path.write_text(
        json.dumps(
            {
                "sources": {
                    "composite": {
                        "type": "vector",
                        "url": "mapbox://mapbox.mapbox-streets-v8",
                    }
                },
                "layers": [
                    {
                        "id": "water",
                        "type": "fill",
                        "source": "composite",
                        "source-layer": "water",
                    }
                ],
            }
        )
    )
    record = {
        "z": 0,
        "x": 0,
        "y": 0,
        "path": "tiles/0/0/0.mvt",
        "sha256": _sha256(tile_path),
        "byte_count": tile_path.stat().st_size,
        "url_without_token": "https://api.mapbox.com/v4/mapbox.mapbox-streets-v8/0/0/0.mvt",
        "water_feature_count": 1,
    }
    aggregate = hashlib.sha256()
    aggregate.update(record["path"].encode())
    aggregate.update(b"\0")
    aggregate.update(record["sha256"].encode())
    aggregate.update(b"\n")
    manifest = {
        "schema_version": 1,
        "status": "pinned_reference",
        "kind": "mapbox_vector_water",
        "provider": "Mapbox",
        "style": {
            "id": "mapbox/light-v11",
            "path": style_path.name,
            "sha256": _sha256(style_path),
        },
        "source": {
            "map_id": "mapbox.mapbox-streets-v8",
            "source_layer": "water",
        },
        "zoom": 0,
        "coverage_bounds_web_mercator": [
            -WEB_MERCATOR_HALF_WORLD,
            -WEB_MERCATOR_HALF_WORLD,
            WEB_MERCATOR_HALF_WORLD,
            WEB_MERCATOR_HALF_WORLD,
        ],
        "tile_count": 1,
        "tile_aggregate_sha256": aggregate.hexdigest(),
        "tiles": [record],
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


def _grid() -> dict:
    return {
        "crs": "EPSG:3857",
        "bounds": [
            -WEB_MERCATOR_HALF_WORLD,
            -WEB_MERCATOR_HALF_WORLD,
            WEB_MERCATOR_HALF_WORLD,
            WEB_MERCATOR_HALF_WORLD,
        ],
        "width": 8,
        "height": 8,
    }


def test_rasterizes_only_pinned_mapbox_water_inside_eligibility(tmp_path):
    root = _reference(tmp_path)
    eligible = np.ones((8, 8), dtype=bool)
    eligible[:2] = False

    water, report = rasterize_pinned_mapbox_water(
        root,
        _grid(),
        eligible_mask=eligible,
        supersampling=2,
        minimum_coverage_fraction=1.0,
    )

    assert not np.any(water[:2])
    assert np.all(water[2:, :3])
    assert not np.any(water[:, 5:])
    assert report["style_id"] == "mapbox/light-v11"
    assert report["tile_count"] == 1
    assert report["water_pixel_count"] == int(np.count_nonzero(water))


def test_census_seed_selects_region_but_mapbox_decides_land_and_water(tmp_path):
    root = _reference(tmp_path)
    interior = np.ones((8, 8), dtype=bool)
    seed = np.zeros((8, 8), dtype=bool)
    seed[:, 2:6] = True

    water, report = constrain_named_water_to_mapbox(
        seed,
        interior,
        _grid(),
        root,
        maximum_distance_px=10,
        supersampling=2,
        minimum_coverage_fraction=1.0,
    )

    assert np.all(water[:, :3])
    assert not np.any(water[:, 5:])
    assert report["retained_seed_pixel_count"] > 0
    assert report["restored_landward_seed_pixel_count"] > 0
    assert report["added_beyond_seed_pixel_count"] > 0


def test_rejects_a_changed_pinned_mapbox_tile(tmp_path):
    root = _reference(tmp_path)
    (root / "tiles" / "0" / "0" / "0.mvt").write_bytes(b"changed")

    with pytest.raises(ValueError, match="missing or stale"):
        rasterize_pinned_mapbox_water(
            root,
            _grid(),
            eligible_mask=np.ones((8, 8), dtype=bool),
        )
