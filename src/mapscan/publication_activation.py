"""Activate an approved staging tile package without changing its data bytes."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable

from PIL import Image


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Dict[str, object]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _aggregate_hash(paths: Iterable[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(_sha256(path).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _package_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name != "public-activation-decision.json"
    )


def _verified_package(root: Path) -> Dict[str, object]:
    dataset_path = root / "dataset.json"
    provenance_path = root / "provenance.json"
    if not dataset_path.is_file() or not provenance_path.is_file():
        raise FileNotFoundError("Staging package needs dataset.json and provenance.json")
    dataset = _load_json(dataset_path)
    provenance = _load_json(provenance_path)
    if dataset.get("status") != "approved_publication":
        raise ValueError("Staging dataset is not an approved publication")
    if dataset.get("approval", {}).get("status") != "approved":
        raise ValueError("Staging dataset has no approved materialization")
    if provenance.get("kind") != "approved_publication_provenance":
        raise ValueError("Staging provenance has the wrong kind")
    if provenance.get("dataset_id") != dataset.get("id"):
        raise ValueError("Dataset and provenance identifiers differ")
    if provenance.get("approval", {}).get("status") != "approved":
        raise ValueError("Staging provenance has no approved decision")
    if dataset.get("provenance", {}).get("sha256") != _sha256(provenance_path):
        raise ValueError("Dataset provenance hash is stale")
    if dataset.get("materialization", {}).get("sha256") != provenance.get(
        "materialization_sha256"
    ):
        raise ValueError("Dataset and provenance materialization hashes differ")
    if dataset.get("approval", {}).get("sha256") != provenance.get(
        "approval", {}
    ).get("decision_sha256"):
        raise ValueError("Dataset and provenance approval hashes differ")

    source_name = dataset.get("source_image")
    if not isinstance(source_name, str) or not (root / source_name).is_file():
        raise ValueError("Staging package has no source image")

    continuous = dataset.get("continuous")
    if isinstance(continuous, dict):
        value_record = continuous.get("value_raster")
        provenance_continuous = provenance.get("continuous")
        if not isinstance(value_record, dict) or not isinstance(
            provenance_continuous, dict
        ):
            raise ValueError("Continuous staging package has no numeric provenance")
        value_path = root / str(value_record.get("path", ""))
        if (
            not value_path.is_file()
            or _sha256(value_path) != value_record.get("sha256")
            or value_record.get("sha256")
            != provenance_continuous.get("value_raster_sha256")
        ):
            raise ValueError("Continuous value raster is missing or hash-mismatched")
        with Image.open(value_path) as image:
            if image.size != (
                int(value_record.get("width", 0)),
                int(value_record.get("height", 0)),
            ):
                raise ValueError("Continuous value raster dimensions are stale")

    boundary = dataset.get("boundary")
    if not isinstance(boundary, dict):
        raise ValueError("Public activation requires an exact boundary contract")
    component_count = int(boundary.get("continuous_border_component_count", 0))
    expected_count = int(boundary.get("expected_boundary_component_count", 0))
    components = boundary.get("components")
    if (
        component_count < 1
        or component_count != expected_count
        or not isinstance(components, list)
        or len(components) != component_count
    ):
        raise ValueError("Boundary component contract is incomplete")
    if int(boundary.get("colored_pixel_count_outside_boundary", -1)) != 0:
        raise ValueError("Staging data contains colored pixels outside the boundary")
    coverage_contract = str(boundary.get("coverage_contract", "full_state"))
    if coverage_contract not in {"full_state", "sparse_visible_evidence"}:
        raise ValueError(f"Unsupported boundary coverage contract: {coverage_contract}")
    unclassified_inside = int(
        boundary.get("unclassified_pixel_count_inside_boundary", -1)
    )
    if unclassified_inside < 0:
        raise ValueError("Boundary has no valid interior NoData count")
    if coverage_contract == "full_state" and unclassified_inside != 0:
        raise ValueError("Full-state staging data contains NoData inside the boundary")

    provenance_boundary = provenance.get("boundary")
    if not isinstance(provenance_boundary, dict):
        raise ValueError("Staging provenance has no exact boundary contract")
    if (
        str(provenance_boundary.get("coverage_contract", "full_state"))
        != coverage_contract
        or int(
            provenance_boundary.get(
                "unclassified_pixel_count_inside_boundary", -1
            )
        )
        != unclassified_inside
        or int(provenance_boundary.get("colored_pixel_count_outside_boundary", -1))
        != 0
    ):
        raise ValueError("Dataset and provenance boundary coverage differ")
    canonical_id = str(boundary.get("canonical_boundary_id", ""))
    if not canonical_id:
        raise ValueError("Boundary has no canonical identifier")
    raster_path = root / str(boundary.get("raster", ""))
    geojson_path = root / str(boundary.get("geojson", ""))
    if (
        not raster_path.is_file()
        or _sha256(raster_path) != boundary.get("raster_sha256")
        or boundary.get("raster_sha256")
        != boundary.get("canonical_display_border_sha256")
    ):
        raise ValueError("Canonical boundary raster is missing or hash-mismatched")
    if (
        not geojson_path.is_file()
        or _sha256(geojson_path) != boundary.get("geojson_sha256")
        or int(boundary.get("geojson_feature_count", 0)) != component_count
    ):
        raise ValueError("Boundary GeoJSON is missing or hash-mismatched")
    with Image.open(raster_path) as image:
        expected_size = (
            int(boundary.get("raster_width", 0)),
            int(boundary.get("raster_height", 0)),
        )
        if image.size != expected_size:
            raise ValueError("Canonical boundary dimensions are stale")

    category_records = []
    seen_categories: set[tuple[str, str]] = set()
    for layer in dataset.get("layers", []):
        if not isinstance(layer, dict):
            raise ValueError("Dataset layer record is invalid")
        layer_id = str(layer.get("id", ""))
        for category in layer.get("categories", []):
            if not isinstance(category, dict):
                raise ValueError("Dataset category record is invalid")
            category_id = str(category.get("id", ""))
            key = (layer_id, category_id)
            if not layer_id or not category_id or key in seen_categories:
                raise ValueError("Dataset category identifiers must be unique")
            seen_categories.add(key)
            tile_root = root / "tiles" / layer_id / category_id
            tilejson_path = root / str(category.get("tilejson", ""))
            tiles = sorted(tile_root.glob("*/*/*.png"))
            if len(tiles) != int(category.get("tile_file_count", -1)):
                raise ValueError(f"Tile count differs for {layer_id}/{category_id}")
            if not tilejson_path.is_file():
                raise ValueError(f"TileJSON is missing for {layer_id}/{category_id}")
            tile_set_hash = _aggregate_hash([*tiles, tilejson_path], root)
            if tile_set_hash != category.get("tile_set_sha256"):
                raise ValueError(f"Tile-set hash differs for {layer_id}/{category_id}")
            template = str(category.get("tile_template", ""))
            if f"?v={tile_set_hash[:16]}" not in template:
                raise ValueError(f"Tile cache key is stale for {layer_id}/{category_id}")
            category_records.append(
                {
                    "layer_id": layer_id,
                    "category_id": category_id,
                    "tile_file_count": len(tiles),
                    "tile_set_sha256": tile_set_hash,
                }
            )
    if not category_records:
        raise ValueError("Staging package has no raster tiles")

    files = _package_files(root)
    return {
        "dataset": dataset,
        "dataset_manifest_sha256": _sha256(dataset_path),
        "provenance_sha256": _sha256(provenance_path),
        "package_file_count": len(files),
        "package_sha256": _aggregate_hash(files, root),
        "canonical_boundary_id": canonical_id,
        "canonical_boundary_sha256": _sha256(raster_path),
        "boundary_component_count": component_count,
        "coverage_contract": coverage_contract,
        "unclassified_pixel_count_inside_boundary": unclassified_inside,
        "categories": category_records,
    }


def activate_staging_dataset(
    staging_dir: Path,
    output_dir: Path,
    *,
    author_statement: str,
) -> Dict[str, object]:
    """Copy an approved staging package into a fresh public dataset directory."""

    statement = author_statement.strip()
    if not statement:
        raise ValueError("A public activation statement is required")
    staging_dir = staging_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise ValueError("Public activation requires a fresh output directory")
    staging = _verified_package(staging_dir)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(staging_dir, output_dir)
    public = _verified_package(output_dir)
    for key in (
        "dataset_manifest_sha256",
        "provenance_sha256",
        "package_file_count",
        "package_sha256",
        "canonical_boundary_id",
        "canonical_boundary_sha256",
        "boundary_component_count",
        "coverage_contract",
        "unclassified_pixel_count_inside_boundary",
        "categories",
    ):
        if public[key] != staging[key]:
            shutil.rmtree(output_dir)
            raise ValueError(f"Public package differs from staging: {key}")

    dataset = public["dataset"]
    decision = {
        "schema_version": 1,
        "kind": "public_activation_decision",
        "status": "approved_public_activation",
        "dataset_id": dataset["id"],
        "author_statement": statement,
        "activated_at": datetime.now(timezone.utc).isoformat(),
        "staging_assets_copied_byte_identically": True,
        "approval_carried_forward": False,
        "dataset_manifest_sha256": public["dataset_manifest_sha256"],
        "provenance_sha256": public["provenance_sha256"],
        "materialization_sha256": dataset["materialization"]["sha256"],
        "materialization_approval_sha256": dataset["approval"]["sha256"],
        "package_file_count": public["package_file_count"],
        "package_sha256": public["package_sha256"],
        "boundary": {
            "canonical_boundary_id": public["canonical_boundary_id"],
            "canonical_boundary_sha256": public["canonical_boundary_sha256"],
            "component_count": public["boundary_component_count"],
            "colored_pixel_count_outside_boundary": 0,
            "coverage_contract": public["coverage_contract"],
            "unclassified_pixel_count_inside_boundary": public[
                "unclassified_pixel_count_inside_boundary"
            ],
        },
        "categories": public["categories"],
    }
    decision_path = output_dir / "public-activation-decision.json"
    decision_path.write_text(json.dumps(decision, indent=2) + "\n")
    return {
        **decision,
        "public_directory": str(output_dir),
        "public_activation_decision_sha256": _sha256(decision_path),
    }
