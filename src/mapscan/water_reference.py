"""Pinned California area-hydrography reference and rasterization helpers."""

from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import shapefile
from PIL import Image, ImageDraw
from pyproj import Transformer
from shapely.geometry import MultiPolygon, Polygon, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as transform_geometry

from .reference import CALIFORNIA_FIPS, TIGER_VINTAGE


AREAWATER_BASE_URL = (
    f"https://www2.census.gov/geo/tiger/TIGER{TIGER_VINTAGE}/AREAWATER"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _county_records(boundary_reference: Path) -> list[dict]:
    matches = sorted((boundary_reference / "county").glob("*.shp"))
    if len(matches) != 1:
        raise FileNotFoundError(
            "California water acquisition needs the pinned Census county shapefile"
        )
    reader = shapefile.Reader(str(matches[0]))
    fields = [field[0] for field in reader.fields[1:]]
    return sorted(
        (
            dict(zip(fields, record))
            for record in reader.records()
            if str(dict(zip(fields, record)).get("STATEFP")) == CALIFORNIA_FIPS
        ),
        key=lambda item: str(item["GEOID"]),
    )


def fetch_california_areawater(
    boundary_reference: Path,
    output_root: Path,
    *,
    force: bool = False,
) -> Dict[str, object]:
    """Download every county-level 2025 TIGER/Line Area Hydrography package."""

    boundary_reference = boundary_reference.resolve()
    output_root = output_root.resolve()
    archive_root = output_root / "archives"
    county_root = output_root / "counties"
    archive_root.mkdir(parents=True, exist_ok=True)
    county_root.mkdir(parents=True, exist_ok=True)
    packages: Dict[str, object] = {}
    for county in _county_records(boundary_reference):
        geoid = str(county["GEOID"])
        filename = f"tl_{TIGER_VINTAGE}_{geoid}_areawater.zip"
        url = f"{AREAWATER_BASE_URL}/{filename}"
        archive_path = archive_root / filename
        extract_dir = county_root / geoid
        if force and archive_path.exists():
            archive_path.unlink()
        if force and extract_dir.exists():
            shutil.rmtree(extract_dir)
        if not archive_path.exists():
            request = urllib.request.Request(url, headers={"User-Agent": "MapScan/0.1"})
            with urllib.request.urlopen(request) as response, archive_path.open(
                "wb"
            ) as output:
                shutil.copyfileobj(response, output)
        if not extract_dir.exists():
            extract_dir.mkdir(parents=True)
            with zipfile.ZipFile(archive_path) as bundle:
                bundle.extractall(extract_dir)
        components = {
            path.name: _sha256(path)
            for path in sorted(extract_dir.iterdir())
            if path.is_file()
        }
        if len(list(extract_dir.glob("*.shp"))) != 1:
            raise ValueError(f"Area Hydrography package {geoid} is incomplete")
        packages[geoid] = {
            "county_name": str(county["NAME"]),
            "url": url,
            "archive": str(archive_path.relative_to(output_root)),
            "archive_sha256": _sha256(archive_path),
            "directory": str(extract_dir.relative_to(output_root)),
            "components": components,
        }
    manifest = {
        "schema_version": 1,
        "status": "pinned_reference",
        "source": "U.S. Census Bureau TIGER/Line Area Hydrography Shapefiles",
        "vintage": TIGER_VINTAGE,
        "state_fips": CALIFORNIA_FIPS,
        "boundary_reference": {
            "path": str(boundary_reference),
            "manifest_sha256": _sha256(boundary_reference / "manifest.json"),
        },
        "package_count": len(packages),
        "packages": packages,
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _verified_features(reference_root: Path) -> Iterable[Tuple[dict, BaseGeometry]]:
    manifest_path = reference_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("status") != "pinned_reference"
        or manifest.get("vintage") != TIGER_VINTAGE
        or manifest.get("state_fips") != CALIFORNIA_FIPS
    ):
        raise ValueError("Area Hydrography reference is not the pinned California vintage")
    packages = manifest.get("packages", {})
    if int(manifest.get("package_count", 0)) != len(packages) or not packages:
        raise ValueError("Area Hydrography package inventory is incomplete")
    for geoid, package in sorted(packages.items()):
        directory = reference_root / str(package["directory"])
        components = package.get("components", {})
        for filename, expected in components.items():
            path = directory / filename
            if not path.is_file() or _sha256(path) != str(expected):
                raise ValueError(f"Area Hydrography component is missing or stale: {path}")
        matches = sorted(directory.glob("*.shp"))
        if len(matches) != 1:
            raise ValueError(f"Area Hydrography county {geoid} has no unique shapefile")
        reader = shapefile.Reader(str(matches[0]))
        fields = [field[0] for field in reader.fields[1:]]
        for item in reader.iterShapeRecords():
            yield dict(zip(fields, item.record)), shape(item.shape.__geo_interface__)


def _draw_polygon(
    draw: ImageDraw.ImageDraw,
    polygon: Polygon,
    grid: Dict[str, object],
    supersampling: int,
) -> None:
    min_x, min_y, max_x, max_y = (float(value) for value in grid["bounds"])
    width = int(grid["width"]) * supersampling
    height = int(grid["height"]) * supersampling

    def pixels(coordinates):
        return [
            (
                (float(x) - min_x) / (max_x - min_x) * width - 0.5,
                (max_y - float(y)) / (max_y - min_y) * height - 0.5,
            )
            for x, y in coordinates
        ]

    draw.polygon(pixels(polygon.exterior.coords), fill=255)
    for interior in polygon.interiors:
        draw.polygon(pixels(interior.coords), fill=0)


def rasterize_california_areawater(
    reference_root: Path,
    grid: Dict[str, object],
    *,
    supersampling: int = 4,
    required_feature_names: Iterable[str] = (),
    minimum_coverage_fraction: float = 0.25,
    force_any_coverage_feature_names: Iterable[str] = (),
    include_only_feature_names: Iterable[str] = (),
    exclude_feature_names: Iterable[str] = (),
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Rasterize Area Hydrography without inflating every tiny feature to a cell."""

    reference_root = reference_root.resolve()
    if supersampling < 1 or supersampling > 16:
        raise ValueError("Area Hydrography supersampling must be between 1 and 16")
    if not 0 < minimum_coverage_fraction <= 1:
        raise ValueError("Area Hydrography minimum coverage must be in (0, 1]")
    width, height = int(grid["width"]), int(grid["height"])
    surface = Image.new("L", (width * supersampling, height * supersampling), 0)
    draw = ImageDraw.Draw(surface)
    forced_surface = Image.new(
        "L", (width * supersampling, height * supersampling), 0
    )
    forced_draw = ImageDraw.Draw(forced_surface)
    transformer = Transformer.from_crs(
        "EPSG:4269", str(grid["crs"]), always_xy=True
    )
    min_x, min_y, max_x, max_y = (float(value) for value in grid["bounds"])
    requested = [str(item) for item in required_feature_names]
    forced_names = [str(item) for item in force_any_coverage_feature_names]
    forced_keys = {item.casefold() for item in forced_names}
    included_names = [str(item) for item in include_only_feature_names]
    included_keys = {item.casefold() for item in included_names}
    excluded_names = [str(item) for item in exclude_feature_names]
    excluded_keys = {item.casefold() for item in excluded_names}
    matched = {name: 0 for name in requested}
    forced_matched = {name: 0 for name in forced_names}
    feature_count = 0
    selected_feature_count = 0
    rendered_feature_count = 0
    for record, geometry in _verified_features(reference_root):
        feature_count += 1
        name = str(record.get("FULLNAME") or "")
        for required in requested:
            if name.casefold() == required.casefold():
                matched[required] += 1
        for required in forced_names:
            if name.casefold() == required.casefold():
                forced_matched[required] += 1
        if included_keys and name.casefold() not in included_keys:
            continue
        if name.casefold() in excluded_keys:
            continue
        selected_feature_count += 1
        projected = transform_geometry(transformer.transform, geometry)
        bounds = projected.bounds
        if bounds[2] < min_x or bounds[0] > max_x or bounds[3] < min_y or bounds[1] > max_y:
            continue
        polygons = (
            [projected]
            if isinstance(projected, Polygon)
            else list(projected.geoms)
            if isinstance(projected, MultiPolygon)
            else []
        )
        if not polygons:
            continue
        rendered_feature_count += 1
        for polygon in polygons:
            _draw_polygon(draw, polygon, grid, supersampling)
            if name.casefold() in forced_keys:
                _draw_polygon(forced_draw, polygon, grid, supersampling)
    missing = [name for name, count in matched.items() if count == 0]
    if missing:
        raise ValueError(
            "Required Area Hydrography features were not found: " + ", ".join(missing)
        )
    missing_forced = [name for name, count in forced_matched.items() if count == 0]
    if missing_forced:
        raise ValueError(
            "Forced Area Hydrography features were not found: "
            + ", ".join(missing_forced)
        )
    high = np.asarray(surface, dtype=np.uint8) > 0
    coverage = high.reshape(height, supersampling, width, supersampling).mean(
        axis=(1, 3)
    )
    forced_high = np.asarray(forced_surface, dtype=np.uint8) > 0
    forced = forced_high.reshape(
        height, supersampling, width, supersampling
    ).any(axis=(1, 3))
    water = (coverage >= minimum_coverage_fraction) | forced
    manifest_path = reference_root / "manifest.json"
    report = {
        "method": "census_tiger_areawater_fractional_coverage_with_named_narrow_water",
        "reference_manifest": {
            "path": str(manifest_path),
            "sha256": _sha256(manifest_path),
        },
        "vintage": TIGER_VINTAGE,
        "supersampling": supersampling,
        "minimum_coverage_fraction": minimum_coverage_fraction,
        "feature_count": feature_count,
        "selected_feature_count": selected_feature_count,
        "include_only_feature_names": included_names,
        "exclude_feature_names": excluded_names,
        "rendered_feature_count": rendered_feature_count,
        "required_feature_match_counts": matched,
        "force_any_coverage_feature_match_counts": forced_matched,
        "force_any_coverage_pixel_count": int(np.count_nonzero(forced)),
        "water_pixel_count": int(np.count_nonzero(water)),
        "coverage_policy": (
            "When an include-only list is present, no unlisted hydrography can remove "
            "data. Explicit exclusions are applied after inclusion selection. Ordinary "
            "selected water must cover the declared fraction of a categorical cell; "
            "explicitly named narrow water may use any supersampled contact."
        ),
    }
    return water, report
