import gzip
import hashlib
import json
from pathlib import Path

import mapbox_vector_tile
import numpy as np
from PIL import Image
from shapely.geometry import LineString, Polygon, mapping

from mapscan.mapbox_california_reference import (
    WEB_MERCATOR_HALF_WORLD,
    _enclosed_islands,
    build_mapbox_california_reference,
    derive_mapbox_california_reference_from_pinned,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tile() -> bytes:
    water = Polygon([(0, 0), (900, 0), (900, 4096), (0, 4096)])
    state = LineString(
        [(900, 600), (3200, 600), (3200, 3500), (900, 3500)]
    )
    county = LineString([(900, 2000), (3200, 2000)])
    payload = mapbox_vector_tile.encode(
        [
            {
                "name": "water",
                "features": [{"geometry": mapping(water), "properties": {}}],
            },
            {
                "name": "admin",
                "features": [
                    {
                        "geometry": mapping(state),
                        "properties": {
                            "admin_level": 1,
                            "iso_3166_1": "US",
                            "worldview": "all",
                            "maritime": "false",
                            "disputed": "false",
                        },
                    },
                    {
                        "geometry": mapping(county),
                        "properties": {
                            "admin_level": 2,
                            "iso_3166_1": "US",
                            "worldview": "all",
                            "maritime": "false",
                            "disputed": "false",
                        },
                    },
                ],
            },
        ],
        default_options={"y_coord_down": True},
    )
    return gzip.compress(payload)


def test_builds_reference_only_from_pinned_mapbox_inputs(tmp_path, monkeypatch):
    style = {
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
            },
            {
                "id": "admin",
                "type": "line",
                "source": "composite",
                "source-layer": "admin",
            },
        ],
    }
    tilejson = {"vector_layers": [{"id": "water"}, {"id": "admin"}]}
    tile = _tile()

    def fake_download(url, access_token):
        assert access_token == "pk.test"
        if "/styles/" in url:
            return json.dumps(style).encode()
        if url.endswith(".json"):
            return json.dumps(tilejson).encode()
        return tile

    monkeypatch.setattr("mapscan.mapbox_california_reference._download", fake_download)
    grid = {
        "crs": "EPSG:3857",
        "bounds": [
            -WEB_MERCATOR_HALF_WORLD,
            -WEB_MERCATOR_HALF_WORLD,
            WEB_MERCATOR_HALF_WORLD,
            WEB_MERCATOR_HALF_WORLD,
        ],
        "width": 64,
        "height": 64,
    }
    output = tmp_path / "reference"
    manifest = build_mapbox_california_reference(
        output,
        "pk.test",
        grid,
        bounds_wgs84=(-180.0, -85.0, 179.0, 85.0),
        zoom=0,
        california_seed_wgs84=(0.0, 0.0),
        maximum_island_distance_m=1,
        validate_controls=False,
    )

    assert manifest["status"] == "pinned_reference"
    assert manifest["authority"]["previous_mapscan_canonical_used"] is False
    assert manifest["authority"]["county_png_used"] is False
    assert manifest["authority"]["census_used"] is False
    assert manifest["tileset"]["source_layers"] == {
        "water": "water",
        "admin": "admin",
    }
    assert manifest["feature_counts"] == {
        "water": 1,
        "state_or_country": 1,
        "county": 1,
    }
    assert manifest["topology"]["state_land_pixel_count"] > 0
    assert manifest["topology"]["state_interior_pixel_count"] >= manifest["topology"][
        "state_land_pixel_count"
    ]
    assert manifest["topology"]["county_pixel_count"] > 0
    assert Image.open(output / "california-state-coast-overlay.png").mode == "RGBA"
    assert _sha256(output / "style.json") == manifest["style"]["sha256"]
    assert (output / "manifest.json").is_file()


def test_rejects_non_mapbox_grid(tmp_path):
    try:
        build_mapbox_california_reference(
            tmp_path / "reference",
            "pk.test",
            {"crs": "EPSG:4326", "bounds": [0, 0, 1, 1], "width": 2, "height": 2},
        )
    except ValueError as error:
        assert "EPSG:3857" in str(error)
    else:
        raise AssertionError("Expected a projection failure")


def test_detached_islands_must_be_on_pacific_side_of_local_mainland():
    passable = np.zeros((80, 100), dtype=bool)
    passable[10:70, 45:70] = True
    passable[30:36, 30:36] = True  # Pacific-side island.
    passable[30:36, 78:84] = True  # Equally near, but east of California.
    mainland = np.zeros_like(passable)
    mainland[10:70, 45:70] = True

    islands = _enclosed_islands(passable, mainland, maximum_distance_px=20)

    assert islands[32, 32]
    assert not islands[32, 80]


def test_derives_new_reference_without_changing_pinned_bytes(tmp_path, monkeypatch):
    style = {
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
            },
            {
                "id": "admin",
                "type": "line",
                "source": "composite",
                "source-layer": "admin",
            },
        ],
    }
    tilejson = {"vector_layers": [{"id": "water"}, {"id": "admin"}]}
    tile = _tile()

    def fake_download(url, access_token):
        if "/styles/" in url:
            return json.dumps(style).encode()
        if url.endswith(".json"):
            return json.dumps(tilejson).encode()
        return tile

    monkeypatch.setattr("mapscan.mapbox_california_reference._download", fake_download)
    grid = {
        "crs": "EPSG:3857",
        "bounds": [
            -WEB_MERCATOR_HALF_WORLD,
            -WEB_MERCATOR_HALF_WORLD,
            WEB_MERCATOR_HALF_WORLD,
            WEB_MERCATOR_HALF_WORLD,
        ],
        "width": 64,
        "height": 64,
    }
    v1 = tmp_path / "v1"
    build_mapbox_california_reference(
        v1,
        "pk.test",
        grid,
        bounds_wgs84=(-180.0, -85.0, 179.0, 85.0),
        zoom=0,
        california_seed_wgs84=(0.0, 0.0),
        maximum_island_distance_m=1,
        validate_controls=False,
    )

    v2 = tmp_path / "v2"
    derived = derive_mapbox_california_reference_from_pinned(
        v1 / "manifest.json", v2
    )

    assert derived["derivation"]["raw_bytes_preserved_exactly"] is True
    assert _sha256(v1 / "style.json") == _sha256(v2 / "style.json")
    assert _sha256(v1 / "tilejson.json") == _sha256(v2 / "tilejson.json")
    assert _sha256(v1 / "tiles/0/0/0.mvt") == _sha256(v2 / "tiles/0/0/0.mvt")
    assert derived["tile_aggregate_sha256"] == json.loads(
        (v1 / "manifest.json").read_text()
    )["tile_aggregate_sha256"]


def test_derives_corner_preserving_supersampled_reference_from_same_raw_bytes(
    tmp_path, monkeypatch
):
    style = {
        "sources": {
            "composite": {
                "type": "vector",
                "url": "mapbox://mapbox.mapbox-streets-v8",
            }
        },
        "layers": [
            {"source": "composite", "source-layer": "water"},
            {"source": "composite", "source-layer": "admin"},
        ],
    }
    tilejson = {"vector_layers": [{"id": "water"}, {"id": "admin"}]}
    tile = _tile()

    def fake_download(url, _access_token):
        if "/styles/" in url:
            return json.dumps(style).encode()
        if url.endswith(".json"):
            return json.dumps(tilejson).encode()
        return tile

    monkeypatch.setattr("mapscan.mapbox_california_reference._download", fake_download)
    grid = {
        "crs": "EPSG:3857",
        "bounds": [
            -WEB_MERCATOR_HALF_WORLD,
            -WEB_MERCATOR_HALF_WORLD,
            WEB_MERCATOR_HALF_WORLD,
            WEB_MERCATOR_HALF_WORLD,
        ],
        "width": 32,
        "height": 24,
    }
    base = tmp_path / "base"
    build_mapbox_california_reference(
        base,
        "pk.test",
        grid,
        bounds_wgs84=(-180.0, -85.0, 179.0, 85.0),
        zoom=0,
        california_seed_wgs84=(0.0, 0.0),
        maximum_island_distance_m=1,
        validate_controls=False,
    )

    high = tmp_path / "high"
    manifest = derive_mapbox_california_reference_from_pinned(
        base / "manifest.json", high, target_supersampling=2
    )

    assert manifest["target_grid"] == {**grid, "width": 63, "height": 47}
    assert manifest["derivation"]["target_grid_supersampling"] == 2
    assert manifest["derivation"]["raw_bytes_preserved_exactly"] is True
    assert _sha256(base / "style.json") == _sha256(high / "style.json")
    assert _sha256(base / "tiles/0/0/0.mvt") == _sha256(
        high / "tiles/0/0/0.mvt"
    )
    assert Image.open(high / "california-land-mask.png").size == (63, 47)
