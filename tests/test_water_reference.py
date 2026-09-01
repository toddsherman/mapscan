import hashlib
import json

import numpy as np
import pytest
import shapefile

import mapscan.extraction as extraction
from mapscan.water_reference import rasterize_california_areawater


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _water_reference(tmp_path):
    root = tmp_path / "water"
    county = root / "counties" / "06000"
    county.mkdir(parents=True)
    target = county / "tl_2025_06000_areawater"
    writer = shapefile.Writer(str(target), shapeType=shapefile.POLYGON)
    writer.field("FULLNAME", "C", size=100)
    writer.field("MTFCC", "C", size=5)
    writer.poly([[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]])
    writer.record("Suisun Bay", "H2051")
    writer.close()
    components = {
        path.name: _sha256(path)
        for path in sorted(county.iterdir())
        if path.is_file()
    }
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "pinned_reference",
                "vintage": "2025",
                "state_fips": "06",
                "package_count": 1,
                "packages": {
                    "06000": {
                        "directory": "counties/06000",
                        "components": components,
                    }
                },
            }
        )
    )
    return root


def test_rasterizes_any_coverage_and_verifies_named_features(tmp_path):
    root = _water_reference(tmp_path)
    grid = {
        "crs": "EPSG:4326",
        "bounds": [-1.0, -1.0, 2.0, 2.0],
        "width": 30,
        "height": 30,
    }
    water, report = rasterize_california_areawater(
        root,
        grid,
        supersampling=4,
        required_feature_names=["Suisun Bay"],
        force_any_coverage_feature_names=["Suisun Bay"],
        include_only_feature_names=["Suisun Bay"],
    )

    assert water.shape == (30, 30)
    assert water[15, 15]
    assert not water[2, 2]
    assert report["required_feature_match_counts"] == {"Suisun Bay": 1}
    assert report["force_any_coverage_feature_match_counts"] == {"Suisun Bay": 1}
    assert report["selected_feature_count"] == 1
    assert report["water_pixel_count"] == int(np.count_nonzero(water))
    with pytest.raises(ValueError, match="Honker Bay"):
        rasterize_california_areawater(
            root,
            grid,
            required_feature_names=["Honker Bay"],
        )


def test_rasterizer_can_exclude_a_named_surface_after_selection(tmp_path):
    root = _water_reference(tmp_path)
    grid = {
        "crs": "EPSG:4326",
        "bounds": [-1.0, -1.0, 2.0, 2.0],
        "width": 30,
        "height": 30,
    }

    water, report = rasterize_california_areawater(
        root,
        grid,
        supersampling=4,
        required_feature_names=["Suisun Bay"],
        exclude_feature_names=["Suisun Bay"],
    )

    assert not np.any(water)
    assert report["selected_feature_count"] == 0
    assert report["exclude_feature_names"] == ["Suisun Bay"]


def test_publication_interior_subtracts_water_without_changing_border_contract(
    tmp_path, monkeypatch
):
    reference = tmp_path / "water"
    reference.mkdir()
    base = np.ones((5, 7), dtype=bool)
    water = np.zeros_like(base)
    water[2, 3] = True
    monkeypatch.setattr(
        extraction,
        "canonical_publication_interior",
        lambda grid: (base.copy(), {"valid_pixel_count": int(base.size)}),
    )
    monkeypatch.setattr(
        extraction,
        "rasterize_california_areawater",
        lambda *args, **kwargs: (
            water.copy(),
            {"method": "test", "water_pixel_count": 1},
        ),
    )

    final, provenance, removed = extraction._publication_interior_with_water_exclusion(
        {"width": 7, "height": 5},
        {"reference": str(reference), "supersampling": 4},
    )

    assert not final[2, 3]
    assert np.count_nonzero(final) == base.size - 1
    assert np.array_equal(removed, water)
    assert provenance["base_valid_pixel_count"] == base.size
    assert provenance["valid_pixel_count"] == base.size - 1
    assert provenance["internal_water_exclusion"]["excluded_interior_pixel_count"] == 1


def test_active_lime_coast_is_not_erased_by_pinned_internal_water(
    tmp_path, monkeypatch
):
    reference = tmp_path / "water"
    reference.mkdir()
    active = np.ones((5, 7), dtype=bool)
    pinned = np.zeros_like(active)
    pinned[1:4, 1:6] = True
    water = np.zeros_like(active)
    water[2, 0] = True
    water[2, 3] = True
    monkeypatch.setattr(
        extraction,
        "active_boundary_publication_interior",
        lambda grid: (active.copy(), {"valid_pixel_count": int(active.size)}),
    )
    monkeypatch.setattr(
        extraction,
        "canonical_publication_interior",
        lambda grid: (pinned.copy(), {"valid_pixel_count": int(pinned.sum())}),
    )
    monkeypatch.setattr(
        extraction,
        "rasterize_california_areawater",
        lambda *args, **kwargs: (
            water.copy(),
            {"method": "test", "water_pixel_count": 2},
        ),
    )

    final, provenance, removed = extraction._publication_interior_with_water_exclusion(
        {"width": 7, "height": 5},
        {
            "reference": str(reference),
            "limit_to_pinned_polygon_interior": True,
        },
        canonical_interior_mode="active_boundary_ring",
    )

    assert final[2, 0]
    assert not final[2, 3]
    assert not removed[2, 0]
    assert removed[2, 3]
    eligibility = provenance["internal_water_exclusion"][
        "pinned_polygon_water_eligibility"
    ]
    assert eligibility["candidate_water_pixel_count"] == 2
    assert eligibility["eligible_water_pixel_count"] == 1
    assert eligibility["outer_canonical_land_pixel_count_restored"] == 1


def test_publication_interior_replaces_selected_census_edge_with_lime_snap(
    tmp_path, monkeypatch
):
    reference = tmp_path / "water"
    reference.mkdir()
    base = np.ones((5, 7), dtype=bool)
    all_water = np.zeros_like(base)
    all_water[2, 2:5] = True
    all_water[4, 6] = True
    snap_seed = np.zeros_like(base)
    snap_seed[2, 2:5] = True
    snapped = np.zeros_like(base)
    snapped[1:4, 1:3] = True
    calls = []

    monkeypatch.setattr(
        extraction,
        "canonical_publication_interior",
        lambda grid: (base.copy(), {"valid_pixel_count": int(base.size)}),
    )

    def rasterize(*args, **kwargs):
        calls.append(kwargs.get("include_only_feature_names"))
        if kwargs.get("include_only_feature_names") == ["San Francisco Bay"]:
            return snap_seed.copy(), {"method": "seed"}
        return all_water.copy(), {"method": "all"}

    monkeypatch.setattr(extraction, "rasterize_california_areawater", rasterize)
    monkeypatch.setattr(
        extraction,
        "snap_named_water_to_active_boundary",
        lambda *args, **kwargs: (snapped.copy(), {"method": "lime"}),
    )

    final, provenance, removed = extraction._publication_interior_with_water_exclusion(
        {"width": 7, "height": 5},
        {
            "reference": str(reference),
            "include_only_feature_names": ["San Francisco Bay", "Sacramento Riv"],
            "canonical_shoreline_snap": {
                "feature_names": ["San Francisco Bay"],
                "maximum_distance_px": 4,
            },
        },
    )

    expected = snapped.copy()
    expected[4, 6] = True
    assert np.array_equal(removed, expected)
    assert np.array_equal(final, base & ~expected)
    assert calls[-1] == ["San Francisco Bay"]
    snap_report = provenance["internal_water_exclusion"]["canonical_shoreline_snap"]
    assert snap_report["method"] == "lime"
    assert snap_report["combined_water_pixel_count"] == int(np.count_nonzero(expected))
