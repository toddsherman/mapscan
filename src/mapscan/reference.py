"""Acquire and load authoritative California boundary geometry."""

from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import shapefile
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon, shape
from shapely.geometry.base import BaseGeometry


TIGER_VINTAGE = "2025"
BASE_URL = f"https://www2.census.gov/geo/tiger/TIGER{TIGER_VINTAGE}"
REFERENCE_PACKAGES = {
    "state": f"{BASE_URL}/STATE/tl_{TIGER_VINTAGE}_us_state.zip",
    "county": f"{BASE_URL}/COUNTY/tl_{TIGER_VINTAGE}_us_county.zip",
}
CALIFORNIA_FIPS = "06"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_reference_data(root: Path, force: bool = False) -> Dict[str, object]:
    """Download and extract the pinned Census TIGER/Line packages."""

    root.mkdir(parents=True, exist_ok=True)
    packages: Dict[str, object] = {}
    for name, url in REFERENCE_PACKAGES.items():
        archive = root / Path(url).name
        extract_dir = root / name
        if force and archive.exists():
            archive.unlink()
        if force and extract_dir.exists():
            shutil.rmtree(extract_dir)
        if not archive.exists():
            request = urllib.request.Request(url, headers={"User-Agent": "MapScan/0.1"})
            with urllib.request.urlopen(request) as response, archive.open("wb") as output:
                shutil.copyfileobj(response, output)
        if not extract_dir.exists():
            extract_dir.mkdir(parents=True)
            with zipfile.ZipFile(archive) as bundle:
                bundle.extractall(extract_dir)
        packages[name] = {
            "url": url,
            "archive": str(archive),
            "sha256": _sha256(archive),
        }

    manifest = {
        "source": "U.S. Census Bureau TIGER/Line Shapefiles",
        "vintage": TIGER_VINTAGE,
        "packages": packages,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _reader(root: Path, layer: str) -> shapefile.Reader:
    matches = sorted((root / layer).glob("*.shp"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one shapefile in {root / layer}; found {len(matches)}"
        )
    return shapefile.Reader(str(matches[0]))


def _records(reader: shapefile.Reader) -> Iterable[Tuple[dict, BaseGeometry]]:
    fields = [field[0] for field in reader.fields[1:]]
    for item in reader.iterShapeRecords():
        record = dict(zip(fields, item.record))
        yield record, shape(item.shape.__geo_interface__)


def load_california(root: Path) -> Tuple[BaseGeometry, List[BaseGeometry]]:
    """Load California state and county polygons in TIGER's source CRS (EPSG:4269)."""

    state_geometry = None
    for record, geometry in _records(_reader(root, "state")):
        if str(record.get("STATEFP")) == CALIFORNIA_FIPS:
            state_geometry = geometry
            break
    if state_geometry is None:
        raise ValueError("California was not present in the state reference package")

    counties = [
        geometry
        for record, geometry in _records(_reader(root, "county"))
        if str(record.get("STATEFP")) == CALIFORNIA_FIPS
    ]
    if not counties:
        raise ValueError("California counties were not present in the county package")
    return state_geometry, counties


def iter_boundary_lines(geometry: BaseGeometry) -> Iterable[LineString]:
    """Yield individual line strings from polygonal or linear geometry."""

    boundary = geometry.boundary if isinstance(geometry, (Polygon, MultiPolygon)) else geometry
    if isinstance(boundary, LineString):
        yield boundary
    elif isinstance(boundary, MultiLineString):
        yield from boundary.geoms
    else:
        for part in boundary.geoms:
            if isinstance(part, LineString):
                yield part
            elif isinstance(part, MultiLineString):
                yield from part.geoms
