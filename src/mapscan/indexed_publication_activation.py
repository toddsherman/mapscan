"""Activate reviewed autonomous indexed-raster packages without changing data bytes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable

from PIL import Image


STAGING_PROVENANCE = "autonomous-preview-provenance.json"
PUBLIC_PROVENANCE = "provenance.json"
DECISION = "public-activation-decision.json"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
IDENTIFIER_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
PATH_FIELD_NAMES = {
    "copied_path",
    "copied_pdf",
    "copied_render",
    "geojson",
    "manifest",
    "path",
    "raster",
}
SUPPORTED_COVERAGE_CONTRACTS = {"full_state", "sparse_visible_evidence"}
AUTONOMOUS_EVIDENCE_KIND = "mapscan_autonomous_publication_nonregression_evidence"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Dict[str, Any]:
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


def _files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())


def _immutable_files(root: Path) -> list[Path]:
    metadata = {"dataset.json", STAGING_PROVENANCE, PUBLIC_PROVENANCE, DECISION}
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and str(path.relative_to(root)) not in metadata
    )


def _validate_package_tree(root: Path) -> None:
    if not root.is_dir():
        raise FileNotFoundError(f"Indexed package directory does not exist: {root}")
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Indexed package must not contain symlinks: {path}")


def _looks_absolute(value: str) -> bool:
    return (
        value.startswith("/")
        or value.startswith("file://")
        or re.match(r"^[A-Za-z]:[\\/]", value) is not None
    )


def _is_path_field(key: str) -> bool:
    normalized = key.casefold()
    return normalized in PATH_FIELD_NAMES or normalized.endswith("_path")


def _package_path(root: Path, raw_value: object, label: str) -> Path:
    if not isinstance(raw_value, str) or not raw_value:
        raise ValueError(f"{label.capitalize()} has no package-relative path")
    if _looks_absolute(raw_value) or "\\" in raw_value:
        raise ValueError(f"{label.capitalize()} must be package-relative")
    relative = Path(raw_value)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"{label.capitalize()} contains path traversal")
    candidate = root.joinpath(*relative.parts)
    try:
        candidate.resolve(strict=False).relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{label.capitalize()} escapes the package") from error
    if not candidate.is_file():
        raise ValueError(f"{label.capitalize()} is missing from the package")
    return candidate


def _sanitize_provenance(value: Any, root: Path, *, parent_key: str = "") -> Any:
    """Remove nonportable paths while retaining adjacent hashes and metrics."""

    if isinstance(value, dict):
        sanitized: Dict[str, Any] = {}
        for key, item in value.items():
            if isinstance(item, str):
                if _looks_absolute(item):
                    continue
                if _is_path_field(key):
                    if "\\" in item or any(
                        part in {"", ".", ".."} for part in Path(item).parts
                    ):
                        raise ValueError(
                            f"Provenance path field {key} contains path traversal"
                        )
                    candidate = root.joinpath(*Path(item).parts)
                    try:
                        candidate.resolve(strict=False).relative_to(root.resolve())
                    except ValueError as error:
                        raise ValueError(
                            f"Provenance path field {key} escapes the package"
                        ) from error
                    if not candidate.is_file():
                        continue
                    sanitized[key] = candidate.relative_to(root).as_posix()
                    continue
            sanitized[key] = _sanitize_provenance(item, root, parent_key=key)
        return sanitized
    if isinstance(value, list):
        return [
            _sanitize_provenance(item, root, parent_key=parent_key) for item in value
        ]
    return value


def _reject_provenance_traversal(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if (
                isinstance(item, str)
                and _is_path_field(key)
                and not _looks_absolute(item)
                and (
                    "\\" in item
                    or any(part in {"", ".", ".."} for part in Path(item).parts)
                )
            ):
                raise ValueError(f"Provenance path field {key} contains path traversal")
            _reject_provenance_traversal(item)
    elif isinstance(value, list):
        for item in value:
            _reject_provenance_traversal(item)


def _validate_public_provenance_paths(value: Any, root: Path) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, str):
                if key == "asset_base":
                    _validated_asset_base(item)
                    continue
                if _looks_absolute(item):
                    raise ValueError(
                        "Indexed public provenance contains a machine-local path"
                    )
                if _is_path_field(key):
                    _package_path(root, item, f"public provenance {key}")
            _validate_public_provenance_paths(item, root)
    elif isinstance(value, list):
        for item in value:
            _validate_public_provenance_paths(item, root)


def _valid_sha256(value: object) -> bool:
    return isinstance(value, str) and SHA256_PATTERN.fullmatch(value) is not None


def _validated_asset_base(value: str) -> str:
    if (
        not value.startswith("/")
        or value.startswith("//")
        or not value.endswith("/")
        or "?" in value
        or "#" in value
        or "\\" in value
        or ":" in value
    ):
        raise ValueError("Shared asset base must be a root-relative URL ending in /")
    segments = value.split("/")[1:-1]
    if not segments or any(
        segment in {"", ".", ".."}
        or re.fullmatch(r"[A-Za-z0-9._~-]+", segment) is None
        for segment in segments
    ):
        raise ValueError("Shared asset base contains an unsafe URL segment")
    return value


def _upstream_raster_hashes(
    dataset: Dict[str, Any], provenance: Dict[str, Any]
) -> Dict[str, str]:
    layer_ids = [str(layer.get("id", "")) for layer in dataset.get("layers", [])]
    hashes: Dict[str, str] = {}
    class_raster = provenance.get("class_raster")
    if len(layer_ids) == 1 and isinstance(class_raster, dict):
        digest = class_raster.get("sha256")
        if _valid_sha256(digest):
            hashes[layer_ids[0]] = str(digest)
    for collection_name in ("layers", "channels"):
        collection = provenance.get(collection_name)
        if not isinstance(collection, list):
            continue
        for record in collection:
            if not isinstance(record, dict):
                continue
            layer_id = str(record.get("id", ""))
            accepted = record.get("accepted_raster")
            digest = (
                accepted.get("sha256")
                if isinstance(accepted, dict)
                else record.get("sha256")
            )
            if layer_id and _valid_sha256(digest):
                hashes[layer_id] = str(digest)
    return hashes


def _declared_source_hash(
    dataset: Dict[str, Any], provenance: Dict[str, Any]
) -> str:
    source_name = str(dataset.get("source_image", ""))
    source = provenance.get("source")
    if not isinstance(source, dict):
        raise ValueError("Indexed provenance has no source hash contract")
    candidates: list[object] = []
    if source.get("copied_path") == source_name:
        candidates.extend((source.get("copied_sha256"), source.get("sha256")))
    if source.get("copied_render") == source_name:
        candidates.append(source.get("render_sha256"))
    if source.get("copied_pdf") == source_name:
        candidates.append(source.get("pdf_sha256"))
    digests = [str(value) for value in candidates if _valid_sha256(value)]
    if not digests:
        raise ValueError("Indexed provenance does not bind its retained source image")
    if len(set(digests)) != 1:
        raise ValueError("Indexed provenance source hashes disagree")
    return digests[0]


def _coverage_contract(
    dataset: Dict[str, Any], provenance: Dict[str, Any]
) -> list[Dict[str, Any]]:
    publication_coverage = provenance.get("publication_coverage")
    if not isinstance(publication_coverage, dict):
        raise ValueError("Indexed package has no publication coverage contract")
    raw_layers = publication_coverage.get("layers")
    if not isinstance(raw_layers, list) or not raw_layers:
        raise ValueError("Indexed package has no per-layer coverage contracts")

    dataset_layer_ids = [
        str(layer.get("id", "")) for layer in dataset.get("layers", [])
    ]
    accepted_hashes = _upstream_raster_hashes(dataset, provenance)
    records: list[Dict[str, Any]] = []
    seen: set[str] = set()
    for raw in raw_layers:
        if not isinstance(raw, dict):
            raise ValueError("Indexed layer coverage record is invalid")
        layer_id = str(raw.get("layer_id", ""))
        if not layer_id or layer_id in seen:
            raise ValueError("Indexed layer coverage identifiers must be unique")
        seen.add(layer_id)
        digest = raw.get("accepted_raster_sha256")
        if not _valid_sha256(digest):
            raise ValueError(f"Indexed layer {layer_id} has no accepted-raster hash")
        if accepted_hashes.get(layer_id) != digest:
            raise ValueError(
                f"Indexed layer {layer_id} coverage hash differs from accepted raster"
            )
        semantic_kind = str(raw.get("semantic_kind", "")).strip()
        if not semantic_kind:
            raise ValueError(f"Indexed layer {layer_id} has no semantic kind")
        coverage_kind = str(raw.get("coverage_contract", ""))
        if coverage_kind not in SUPPORTED_COVERAGE_CONTRACTS:
            raise ValueError(f"Indexed layer {layer_id} has an invalid coverage contract")
        outside = raw.get("colored_pixel_count_outside_state")
        nodata_inside = raw.get("nodata_pixel_count_inside_state")
        if isinstance(outside, bool) or not isinstance(outside, int) or outside != 0:
            raise ValueError(f"Indexed layer {layer_id} contains data outside the state")
        if (
            isinstance(nodata_inside, bool)
            or not isinstance(nodata_inside, int)
            or nodata_inside < 0
        ):
            raise ValueError(f"Indexed layer {layer_id} has no valid interior NoData count")
        records.append(
            {
                "layer_id": layer_id,
                "accepted_raster_sha256": digest,
                "semantic_kind": semantic_kind,
                "coverage_contract": coverage_kind,
                "colored_pixel_count_outside_state": 0,
                "nodata_pixel_count_inside_state": nodata_inside,
            }
        )
    if seen != set(dataset_layer_ids) or len(dataset_layer_ids) != len(seen):
        raise ValueError("Coverage contracts do not match the indexed dataset layers")
    return records


def _verify_indexed_assets(root: Path, dataset: Dict[str, Any]) -> Dict[str, Any]:
    if dataset.get("categorical_tile_encoding") != "indexed_class_id":
        raise ValueError("Dataset is not an indexed class-id package")

    source_name = dataset.get("source_image")
    source_path = _package_path(root, source_name, "retained source image")

    boundary = dataset.get("boundary")
    if not isinstance(boundary, dict):
        raise ValueError("Indexed package has no pinned boundary diagnostic")
    if (
        boundary.get("kind") != "pinned_mapbox_state_coast_diagnostic"
        or boundary.get("authority") != "accepted_alignment_mapbox_reference"
        or boundary.get("diagnostic_only") is not True
    ):
        raise ValueError("Indexed package has the wrong boundary authority")
    boundary_path = _package_path(root, boundary.get("raster"), "boundary raster")
    if _sha256(boundary_path) != boundary.get("raster_sha256"):
        raise ValueError("Indexed package boundary is missing or hash-mismatched")
    with Image.open(boundary_path) as image:
        expected_size = (
            int(boundary.get("raster_width", 0)),
            int(boundary.get("raster_height", 0)),
        )
        if image.size != expected_size:
            raise ValueError("Indexed package boundary dimensions are stale")

    records: list[Dict[str, Any]] = []
    seen_layers: set[str] = set()
    seen_categories: set[str] = set()
    for layer in dataset.get("layers", []):
        if not isinstance(layer, dict):
            raise ValueError("Indexed package layer record is invalid")
        layer_id = str(layer.get("id", ""))
        if (
            IDENTIFIER_PATTERN.fullmatch(layer_id) is None
            or layer_id in seen_layers
        ):
            raise ValueError("Indexed package layer identifiers must be unique")
        seen_layers.add(layer_id)
        if int(layer.get("nodata_class_id", -1)) != 0:
            raise ValueError(f"Indexed layer {layer_id} does not reserve class zero")

        categories = layer.get("categories")
        if not isinstance(categories, list) or not categories:
            raise ValueError(f"Indexed layer {layer_id} has no categories")
        class_ids: list[int] = []
        category_ids: list[str] = []
        for category in categories:
            if not isinstance(category, dict):
                raise ValueError(f"Indexed layer {layer_id} has an invalid category")
            category_id = str(category.get("id", ""))
            class_id = int(category.get("class_id", 0))
            if (
                IDENTIFIER_PATTERN.fullmatch(category_id) is None
                or category_id in seen_categories
            ):
                raise ValueError("Dataset category identifiers must be globally unique")
            if class_id < 1 or class_id > 255 or class_id in class_ids:
                raise ValueError(f"Indexed layer {layer_id} has invalid class identifiers")
            seen_categories.add(category_id)
            category_ids.append(category_id)
            class_ids.append(class_id)

        indexed = layer.get("indexed_raster")
        if not isinstance(indexed, dict):
            raise ValueError(f"Indexed layer {layer_id} has no indexed raster contract")
        if indexed.get("encoding") != "png_luma_alpha_uint8_class_id_v1":
            raise ValueError(f"Indexed layer {layer_id} has an unsupported encoding")
        if int(indexed.get("nodata_class_id", -1)) != 0:
            raise ValueError(f"Indexed layer {layer_id} has stale NoData metadata")
        if list(indexed.get("class_id_range", [])) != [min(class_ids), max(class_ids)]:
            raise ValueError(f"Indexed layer {layer_id} has a stale class range")
        if sorted(class_ids) != list(range(1, max(class_ids) + 1)):
            raise ValueError(f"Indexed layer {layer_id} class identifiers are not contiguous")

        tilejson_path = _package_path(
            root, indexed.get("tilejson"), f"indexed layer {layer_id} TileJSON"
        )
        tilejson_relative = tilejson_path.relative_to(root)
        if (
            len(tilejson_relative.parts) != 4
            or tilejson_relative.parts[0] != "tiles"
            or tilejson_relative.parts[2:] != ("class-id", "tilejson.json")
        ):
            raise ValueError(f"Indexed layer {layer_id} TileJSON path is not canonical")
        tile_root = tilejson_path.parent
        tiles = sorted(tile_root.glob("*/*/*.png"))
        if len(tiles) != int(indexed.get("tile_file_count", -1)):
            raise ValueError(f"Indexed tile count differs for {layer_id}")
        tile_bytes = sum(path.stat().st_size for path in tiles)
        if tile_bytes != int(indexed.get("tile_image_byte_count", -1)):
            raise ValueError(f"Indexed tile bytes differ for {layer_id}")
        tile_set_hash = _aggregate_hash([*tiles, tilejson_path], root)
        if tile_set_hash != indexed.get("tile_set_sha256"):
            raise ValueError(f"Indexed tile-set hash differs for {layer_id}")
        expected_template = (
            f"{tile_root.relative_to(root).as_posix()}/{{z}}/{{x}}/{{y}}.png"
            f"?v={tile_set_hash[:16]}"
        )
        if indexed.get("tile_template") != expected_template:
            raise ValueError(f"Indexed tile cache key is stale for {layer_id}")
        records.append(
            {
                "layer_id": layer_id,
                "category_ids": sorted(category_ids),
                "class_ids": sorted(class_ids),
                "tile_file_count": len(tiles),
                "tile_image_byte_count": tile_bytes,
                "tile_set_sha256": tile_set_hash,
            }
        )

    if not records:
        raise ValueError("Indexed package has no raster layers")
    return {
        "source_image_sha256": _sha256(source_path),
        "boundary_sha256": _sha256(boundary_path),
        "layers": records,
    }


def _verify_staging(root: Path) -> Dict[str, Any]:
    _validate_package_tree(root)
    dataset_path = root / "dataset.json"
    provenance_path = root / STAGING_PROVENANCE
    if not dataset_path.is_file() or not provenance_path.is_file():
        raise FileNotFoundError("Indexed staging package needs dataset and provenance")
    dataset = _load_json(dataset_path)
    provenance = _load_json(provenance_path)
    if dataset.get("status") != "needs_visual_review":
        raise ValueError("Indexed staging dataset is not awaiting visual review")
    if dataset.get("approval", {}).get("status") != "not_approved":
        raise ValueError("Indexed staging dataset already carries an approval")
    if provenance.get("status") != "needs_visual_review":
        raise ValueError("Indexed staging provenance has the wrong status")
    if provenance.get("publication_approved") is not False:
        raise ValueError("Indexed staging provenance already claims publication")
    _reject_provenance_traversal(provenance)
    if dataset.get("provenance", {}).get("manifest") != STAGING_PROVENANCE:
        raise ValueError("Indexed staging provenance filename is stale")
    if dataset.get("provenance", {}).get("sha256") != _sha256(provenance_path):
        raise ValueError("Indexed staging provenance hash is stale")
    assets = _verify_indexed_assets(root, dataset)
    if assets["source_image_sha256"] != _declared_source_hash(dataset, provenance):
        raise ValueError("Indexed retained source image differs from provenance")
    coverage = _coverage_contract(dataset, provenance)
    return {
        "dataset": dataset,
        "provenance": provenance,
        "dataset_sha256": _sha256(dataset_path),
        "provenance_sha256": _sha256(provenance_path),
        "package_sha256": _aggregate_hash(_files(root), root),
        "immutable_asset_count": len(_immutable_files(root)),
        "immutable_assets_sha256": _aggregate_hash(_immutable_files(root), root),
        "coverage": coverage,
        **assets,
    }


def _verify_autonomous_evidence(
    path: Path, staging: Dict[str, Any]
) -> Dict[str, Any]:
    evidence_path = path.resolve()
    evidence = _load_json(evidence_path)
    if (
        evidence.get("schema_version") != 1
        or evidence.get("kind") != AUTONOMOUS_EVIDENCE_KIND
        or evidence.get("status") != "pass"
    ):
        raise ValueError("Autonomous activation evidence does not contain a pass")
    if (
        evidence.get("staging_dataset_sha256") != staging["dataset_sha256"]
        or evidence.get("staging_provenance_sha256")
        != staging["provenance_sha256"]
    ):
        raise ValueError("Autonomous activation evidence targets different staged bytes")
    gates = evidence.get("gates")
    if not isinstance(gates, dict) or not gates or not all(
        _gate
        if isinstance(_gate, bool)
        else bool(_gate.get("passed"))
        if isinstance(_gate, dict)
        else False
        for _gate in gates.values()
    ):
        raise ValueError("Autonomous activation evidence has a failed or invalid gate")
    return {
        "sha256": _sha256(evidence_path),
        "policy": str(evidence.get("policy", "")),
        "gates": gates,
    }


def _verify_public(
    root: Path,
    *,
    expected_asset_root: Path | None = None,
    expected_asset_base: str | None = None,
) -> Dict[str, Any]:
    _validate_package_tree(root)
    dataset_path = root / "dataset.json"
    provenance_path = root / PUBLIC_PROVENANCE
    if not dataset_path.is_file() or not provenance_path.is_file():
        raise FileNotFoundError("Indexed public package needs dataset and provenance")
    if (root / STAGING_PROVENANCE).exists():
        raise ValueError("Public indexed package leaks staging provenance")
    dataset = _load_json(dataset_path)
    provenance = _load_json(provenance_path)
    if dataset.get("status") != "approved_publication":
        raise ValueError("Indexed public dataset is not approved")
    if dataset.get("approval", {}).get("status") != "approved":
        raise ValueError("Indexed public dataset has no author approval")
    if dataset.get("publication_approved") is not True:
        raise ValueError("Indexed public dataset does not declare publication")
    if (
        provenance.get("kind") != "autonomous_indexed_publication_provenance"
        or provenance.get("status") != "approved_publication"
        or provenance.get("publication_approved") is not True
    ):
        raise ValueError("Indexed public provenance has the wrong publication state")
    if dataset.get("provenance", {}).get("manifest") != PUBLIC_PROVENANCE:
        raise ValueError("Indexed public provenance filename is stale")
    if dataset.get("provenance", {}).get("sha256") != _sha256(provenance_path):
        raise ValueError("Indexed public provenance hash is stale")
    declared_asset_base = dataset.get("asset_base")
    if expected_asset_base is None:
        if declared_asset_base is not None:
            raise ValueError("Copied public package unexpectedly declares shared assets")
        asset_root = root
    else:
        if declared_asset_base != expected_asset_base:
            raise ValueError("Indexed public asset base differs from activation request")
        _validated_asset_base(str(declared_asset_base))
        if expected_asset_root is None:
            raise ValueError("Shared public package has no verified local asset root")
        asset_root = expected_asset_root
        _validate_package_tree(asset_root)
    _validate_public_provenance_paths(provenance, asset_root)
    assets = _verify_indexed_assets(asset_root, dataset)
    if assets["source_image_sha256"] != _declared_source_hash(dataset, provenance):
        raise ValueError("Indexed retained source image differs from provenance")
    coverage = _coverage_contract(dataset, provenance)
    return {
        "dataset": dataset,
        "dataset_sha256": _sha256(dataset_path),
        "provenance_sha256": _sha256(provenance_path),
        "immutable_asset_count": len(_immutable_files(asset_root)),
        "immutable_assets_sha256": _aggregate_hash(
            _immutable_files(asset_root), asset_root
        ),
        "coverage": coverage,
        **assets,
    }


def activate_indexed_staging_dataset(
    staging_dir: Path,
    output_dir: Path,
    *,
    author_statement: str,
    public_id: str | None = None,
    public_title: str | None = None,
    asset_base: str | None = None,
    autonomous_evidence_path: Path | None = None,
) -> Dict[str, Any]:
    """Promote a reviewed indexed package into a fresh public directory."""

    statement = author_statement.strip()
    if not statement:
        raise ValueError("A public activation statement is required")
    staging_input = Path(staging_dir)
    if staging_input.is_symlink():
        raise ValueError("Indexed staging package must not be a symlink")
    staging_dir = staging_input.resolve()
    output_dir = Path(output_dir).resolve()
    shared_asset_base = (
        _validated_asset_base(asset_base) if asset_base is not None else None
    )
    if output_dir.exists():
        raise ValueError("Indexed public activation requires a fresh output directory")
    staging = _verify_staging(staging_dir)
    autonomous_evidence = (
        _verify_autonomous_evidence(autonomous_evidence_path, staging)
        if autonomous_evidence_path is not None
        else None
    )
    approval_mode = (
        "autonomous_nonregression_activation"
        if autonomous_evidence is not None
        else "author_public_activation"
    )
    resolved_public_id = public_id or str(staging["dataset"].get("id", ""))
    if IDENTIFIER_PATTERN.fullmatch(resolved_public_id) is None:
        raise ValueError("Indexed public dataset id must be lowercase kebab-case")
    resolved_public_title = public_title or str(staging["dataset"].get("title", ""))
    if not resolved_public_title.strip():
        raise ValueError("Indexed public dataset title is required")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        if shared_asset_base is None:
            shutil.copytree(staging_dir, output_dir, copy_function=os.link)
            (output_dir / STAGING_PROVENANCE).unlink()
        else:
            output_dir.mkdir()

        activated_at = datetime.now(timezone.utc).isoformat()
        public_provenance = _sanitize_provenance(
            staging["provenance"], staging_dir
        )
        public_provenance.update(
            {
                "kind": "autonomous_indexed_publication_provenance",
                "status": "approved_publication",
                "publication_approved": True,
                "approval": {
                    "status": "approved",
                    "mode": approval_mode,
                    "authorization_statement": statement,
                    "activated_at": activated_at,
                    **(
                        {"autonomous_evidence": autonomous_evidence}
                        if autonomous_evidence is not None
                        else {"author_statement": statement}
                    ),
                },
                "staging": {
                    "artifact_dataset_id": staging["dataset"]["id"],
                    "artifact_title": staging["dataset"]["title"],
                    "dataset_sha256": staging["dataset_sha256"],
                    "provenance_sha256": staging["provenance_sha256"],
                    "package_sha256": staging["package_sha256"],
                },
            }
        )
        if shared_asset_base is not None:
            public_provenance["shared_assets"] = {
                "asset_base": shared_asset_base,
                "immutable_asset_count": staging["immutable_asset_count"],
                "immutable_assets_sha256": staging["immutable_assets_sha256"],
            }
        provenance_path = output_dir / PUBLIC_PROVENANCE
        provenance_path.write_text(json.dumps(public_provenance, indent=2) + "\n")

        public_dataset = dict(staging["dataset"])
        public_dataset["id"] = resolved_public_id
        public_dataset["title"] = resolved_public_title
        public_dataset["status"] = "approved_publication"
        public_dataset["publication_approved"] = True
        public_dataset["approval"] = {
            "status": "approved",
            "mode": approval_mode,
            "authorization_statement": statement,
            "activated_at": activated_at,
            **(
                {"autonomous_evidence": autonomous_evidence}
                if autonomous_evidence is not None
                else {"author_statement": statement}
            ),
        }
        public_dataset["provenance"] = {
            "manifest": PUBLIC_PROVENANCE,
            "sha256": _sha256(provenance_path),
        }
        if shared_asset_base is not None:
            public_dataset["asset_base"] = shared_asset_base
            public_dataset["shared_assets"] = {
                "immutable_asset_count": staging["immutable_asset_count"],
                "immutable_assets_sha256": staging["immutable_assets_sha256"],
            }
        dataset_path = output_dir / "dataset.json"
        if dataset_path.exists():
            dataset_path.unlink()
        dataset_path.write_text(json.dumps(public_dataset, indent=2) + "\n")

        public = _verify_public(
            output_dir,
            expected_asset_root=staging_dir if shared_asset_base else None,
            expected_asset_base=shared_asset_base,
        )
        for key in (
            "source_image_sha256",
            "boundary_sha256",
            "layers",
            "coverage",
            "immutable_asset_count",
            "immutable_assets_sha256",
        ):
            if public[key] != staging[key]:
                raise ValueError(f"Indexed public package differs from staging: {key}")

        decision = {
            "schema_version": 1,
            "kind": "indexed_public_activation_decision",
            "status": "approved_public_activation",
            "dataset_id": public_dataset["id"],
            "dataset_title": public_dataset["title"],
            "staging_artifact_dataset_id": staging["dataset"]["id"],
            "staging_artifact_title": staging["dataset"]["title"],
            "author_statement": statement,
            "approval_mode": approval_mode,
            "autonomous_evidence": autonomous_evidence,
            "activated_at": activated_at,
            "staging_assets_copied_byte_identically": shared_asset_base is None,
            "staging_assets_shared_byte_identically": shared_asset_base is not None,
            "asset_base": shared_asset_base,
            "metadata_transition_only": True,
            "staging_dataset_sha256": staging["dataset_sha256"],
            "staging_provenance_sha256": staging["provenance_sha256"],
            "staging_package_sha256": staging["package_sha256"],
            "public_dataset_sha256": public["dataset_sha256"],
            "public_provenance_sha256": public["provenance_sha256"],
            "immutable_asset_count": public["immutable_asset_count"],
            "immutable_assets_sha256": public["immutable_assets_sha256"],
            "source_image_sha256": public["source_image_sha256"],
            "boundary_sha256": public["boundary_sha256"],
            "coverage": public["coverage"],
            "layers": public["layers"],
        }
        decision_path = output_dir / DECISION
        decision_path.write_text(json.dumps(decision, indent=2) + "\n")
        return {
            **decision,
            "public_directory": str(output_dir),
            "public_activation_decision_sha256": _sha256(decision_path),
        }
    except Exception:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Activate one reviewed indexed-raster staging package."
    )
    parser.add_argument("staging_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--author-statement", required=True)
    parser.add_argument("--public-id")
    parser.add_argument("--public-title")
    parser.add_argument(
        "--asset-base",
        help=(
            "Safe root-relative deployed URL for manifest-only shared assets, "
            "for example /mapscan/data/staging/example-v1/."
        ),
    )
    arguments = parser.parse_args(argv)
    result = activate_indexed_staging_dataset(
        arguments.staging_dir,
        arguments.output_dir,
        author_statement=arguments.author_statement,
        public_id=arguments.public_id,
        public_title=arguments.public_title,
        asset_base=arguments.asset_base,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
