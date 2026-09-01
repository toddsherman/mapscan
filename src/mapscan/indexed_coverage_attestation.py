"""Bind indexed staging layers to a direct state-mask coverage audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
from PIL import Image


PROVENANCE_NAME = "autonomous-preview-provenance.json"
SUPPORTED_COVERAGE_CONTRACTS = {"full_state", "sparse_visible_evidence"}


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


def _class_values(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        values = np.asarray(image)
    if values.ndim == 3:
        values = values[..., 0]
    if values.ndim != 2:
        raise ValueError(f"Expected a 2D class raster: {path}")
    return values


def _upstream_hashes(dataset: Dict[str, Any], provenance: Dict[str, Any]) -> Dict[str, str]:
    layer_ids = [str(layer.get("id", "")) for layer in dataset.get("layers", [])]
    hashes: Dict[str, str] = {}
    class_raster = provenance.get("class_raster")
    if len(layer_ids) == 1 and isinstance(class_raster, dict):
        digest = class_raster.get("sha256")
        if isinstance(digest, str):
            hashes[layer_ids[0]] = digest
    for collection_name in ("layers", "channels"):
        collection = provenance.get(collection_name)
        if not isinstance(collection, list):
            continue
        for record in collection:
            if not isinstance(record, dict):
                continue
            layer_id = str(record.get("id", ""))
            accepted = record.get("accepted_raster")
            digest = accepted.get("sha256") if isinstance(accepted, dict) else record.get("sha256")
            if layer_id and isinstance(digest, str):
                hashes[layer_id] = digest
    return hashes


def attest_indexed_staging_coverage(
    staging_dir: Path,
    plan_path: Path,
) -> Dict[str, Any]:
    """Compute and persist a hash-bound, per-layer publication coverage contract."""

    staging_dir = staging_dir.resolve()
    plan_path = plan_path.resolve()
    dataset_path = staging_dir / "dataset.json"
    provenance_path = staging_dir / PROVENANCE_NAME
    dataset = _load_json(dataset_path)
    provenance = _load_json(provenance_path)
    plan = _load_json(plan_path)
    if dataset.get("status") != "needs_visual_review":
        raise ValueError("Coverage attestation requires a staging dataset")
    if dataset.get("provenance", {}).get("sha256") != _sha256(provenance_path):
        raise ValueError("Staging provenance hash is stale")
    if plan.get("schema_version") != 1:
        raise ValueError("Unsupported coverage plan schema")

    plan_root = plan_path.parent
    state_mask_path = (plan_root / str(plan.get("state_interior_mask"))).resolve()
    state_mask = _class_values(state_mask_path) > 0
    upstream_hashes = _upstream_hashes(dataset, provenance)
    dataset_layer_ids = [str(layer.get("id", "")) for layer in dataset.get("layers", [])]

    records: list[Dict[str, Any]] = []
    seen: set[str] = set()
    for layer in plan.get("layers", []):
        if not isinstance(layer, dict):
            raise ValueError("Coverage plan layer is invalid")
        layer_id = str(layer.get("layer_id", ""))
        if not layer_id or layer_id in seen:
            raise ValueError("Coverage plan layer identifiers must be unique")
        seen.add(layer_id)
        semantic_kind = str(layer.get("semantic_kind", "")).strip()
        coverage_contract = str(layer.get("coverage_contract", ""))
        if not semantic_kind:
            raise ValueError(f"Coverage plan layer {layer_id} has no semantic kind")
        if coverage_contract not in SUPPORTED_COVERAGE_CONTRACTS:
            raise ValueError(f"Coverage plan layer {layer_id} has an invalid contract")
        accepted_hash = str(layer.get("accepted_raster_sha256", ""))
        if upstream_hashes.get(layer_id) != accepted_hash:
            raise ValueError(f"Coverage plan layer {layer_id} is not bound to accepted evidence")

        raster_path = (plan_root / str(layer.get("publication_raster"))).resolve()
        values = _class_values(raster_path)
        if values.shape != state_mask.shape:
            raise ValueError(f"Coverage raster dimensions differ for {layer_id}")
        classified = values > 0
        outside = int(np.count_nonzero(classified & ~state_mask))
        if outside != 0:
            raise ValueError(f"Coverage raster {layer_id} contains {outside} exterior pixels")
        records.append(
            {
                "layer_id": layer_id,
                "accepted_raster_sha256": accepted_hash,
                "publication_raster_sha256": _sha256(raster_path),
                "semantic_kind": semantic_kind,
                "coverage_contract": coverage_contract,
                "colored_pixel_count_outside_state": 0,
                "nodata_pixel_count_inside_state": int(
                    np.count_nonzero(~classified & state_mask)
                ),
                "classified_pixel_count_inside_state": int(
                    np.count_nonzero(classified & state_mask)
                ),
            }
        )

    if seen != set(dataset_layer_ids) or len(seen) != len(dataset_layer_ids):
        raise ValueError("Coverage plan does not match every dataset layer")

    coverage = {
        "schema_version": 1,
        "method": "direct_mapbox_state_interior_mask_v1",
        "state_interior_mask_sha256": _sha256(state_mask_path),
        "state_interior_pixel_count": int(np.count_nonzero(state_mask)),
        "raster_width": int(state_mask.shape[1]),
        "raster_height": int(state_mask.shape[0]),
        "layers": records,
    }
    provenance["publication_coverage"] = coverage
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    dataset["provenance"]["sha256"] = _sha256(provenance_path)
    dataset_path.write_text(json.dumps(dataset, indent=2) + "\n")
    return {
        "dataset_id": dataset["id"],
        "dataset_sha256": _sha256(dataset_path),
        "provenance_sha256": _sha256(provenance_path),
        "publication_coverage": coverage,
    }
