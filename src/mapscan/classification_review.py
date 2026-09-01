"""Author-only, category-level review for classified MapScan rasters."""

from __future__ import annotations

import hashlib
import json
import mimetypes
import re
import threading
import webbrowser
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import unquote

import numpy as np
from PIL import Image, UnidentifiedImageError

from .enclosed_fill import generate_enclosed_fill_artifact
from .manual_stamp import validate_inference_exclusions, validate_stamp_operations


DECISIONS = {"approved", "rejected", "needs_revision"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_rgb(category: Dict[str, object]) -> list[int]:
    color = category.get("display_rgb", category.get("legend_rgb", [255, 0, 255]))
    if isinstance(color, list) and color and isinstance(color[0], list):
        color = color[0]
    if not isinstance(color, list) or len(color) < 3:
        return [255, 0, 255]
    return [int(color[0]), int(color[1]), int(color[2])]


def _latest_inference_root(run_dir: Path) -> Optional[Path]:
    selection_path = run_dir / "inference-selection.json"
    if selection_path.exists():
        selection = json.loads(selection_path.read_text())
        if selection.get("enabled") is False:
            return None
    candidates = [
        path
        for path in run_dir.glob("inference*")
        if path.is_dir() and (path / "inference.json").exists()
    ]
    if not candidates:
        return None

    def version(path: Path) -> tuple[int, int]:
        match = re.fullmatch(r"inference-v(\d+)", path.name)
        if match:
            return (1, int(match.group(1)))
        return (0, 0)

    return max(candidates, key=version)


def _latest_enclosed_fill_root(run_dir: Path) -> Optional[Path]:
    selection_path = run_dir / "enclosed-hole-fill-selection.json"
    if not selection_path.exists():
        return None
    selection = json.loads(selection_path.read_text())
    if selection.get("enabled") is not True:
        return None
    candidates = [
        path
        for path in run_dir.glob("enclosed-fill*")
        if path.is_dir() and (path / "enclosed-fill.json").exists()
    ]
    if not candidates:
        return None

    def version(path: Path) -> tuple[int, int]:
        match = re.fullmatch(r"enclosed-fill-v(\d+)", path.name)
        return (1, int(match.group(1))) if match else (0, 0)

    return max(candidates, key=version)


def build_classification_payload(run_dir: Path) -> Dict[str, object]:
    manifest_path = run_dir / "extraction.json"
    plan_path = run_dir / "plan.snapshot.json"
    if not manifest_path.exists() or not plan_path.exists():
        raise FileNotFoundError("Classification review needs extraction.json and plan.snapshot.json")
    source_path = run_dir / "web-mercator-source.jpg"
    if not source_path.exists():
        raise FileNotFoundError(f"Missing review source: {source_path}")
    manifest = json.loads(manifest_path.read_text())
    plan = json.loads(plan_path.read_text())
    stamp_corrections_path = run_dir / "stamp-corrections.json"
    enclosed_fill_root = _latest_enclosed_fill_root(run_dir)
    enclosed_fill_manifest = None
    if enclosed_fill_root is not None:
        candidate_manifest = json.loads(
            (enclosed_fill_root / "enclosed-fill.json").read_text()
        )
        if (
            candidate_manifest.get("extraction_manifest_sha256")
            == _sha256(manifest_path)
            and stamp_corrections_path.exists()
            and candidate_manifest.get("stamp_corrections_sha256")
            == _sha256(stamp_corrections_path)
        ):
            enclosed_fill_manifest = candidate_manifest
        else:
            enclosed_fill_root = None
    inference_root = _latest_inference_root(run_dir)
    inference_manifest_path = (
        inference_root / "inference.json" if inference_root is not None else None
    )
    inference_manifest = (
        json.loads(inference_manifest_path.read_text())
        if inference_manifest_path is not None
        else None
    )
    inference_manifest_sha256 = (
        _sha256(inference_manifest_path) if inference_manifest_path is not None else None
    )
    plan_layers = {str(layer["id"]): layer for layer in plan.get("layers", [])}
    layers = []
    for report in manifest.get("layers", []):
        layer_id = str(report["id"])
        class_path = run_dir / layer_id / "web-mercator-class-id.png"
        definition = plan_layers.get(layer_id)
        if not class_path.exists() or not isinstance(definition, dict):
            continue
        web_values = None
        try:
            web_values = np.asarray(Image.open(class_path), dtype=np.uint8)
        except (OSError, UnidentifiedImageError):
            # Small synthetic payload tests and legacy diagnostics may use
            # placeholder bytes. Current extraction manifests record these
            # counts directly, while valid legacy rasters are measured here.
            pass
        counts = report.get("web_mercator_category_pixel_counts")
        if not isinstance(counts, dict) and web_values is not None:
            counts = {
                str(category["id"]): int(np.count_nonzero(web_values == class_id))
                for class_id, category in enumerate(
                    definition.get("categories", []), 1
                )
            }
        if not isinstance(counts, dict):
            counts = report.get("category_pixel_counts", {})
        categories = []
        for class_id, category in enumerate(definition.get("categories", []), 1):
            categories.append(
                {
                    "class_id": class_id,
                    "id": str(category["id"]),
                    "label": str(category.get("label", category["id"])),
                    "display_rgb": _display_rgb(category),
                    "pixel_count": int(counts.get(str(category["id"]), 0)),
                }
            )
        extraction = report.get("extraction", {})
        valid_target_count = int(
            report.get("canonical_clip", {}).get("valid_pixel_count", 0)
        )
        final_target_count = int(
            report.get(
                "web_mercator_classified_pixel_count",
                int(np.count_nonzero(web_values)) if web_values is not None else 0,
            )
        )
        target_completion_count = int(
            report.get("web_mercator_completed_nodata_pixel_count", 0)
        )
        inference_layer_dir = (
            inference_root / layer_id
            if inference_root is not None
            else run_dir / "inference" / layer_id
        )
        inference = None
        if (
            inference_root is not None
            and isinstance(inference_manifest, dict)
            and inference_manifest_path is not None
            and (inference_layer_dir / "web-mercator-class-id-inferred.png").exists()
            and (inference_layer_dir / "web-mercator-inference-mask.png").exists()
        ):
            inference_report = next(
                (
                    item
                    for item in inference_manifest.get("layers", [])
                    if item.get("layer_id") == layer_id
                ),
                {},
            )
            inference_asset_root = inference_root.name
            inference = {
                "class_id_url": (
                    f"/asset/{inference_asset_root}/{layer_id}/"
                    "web-mercator-class-id-inferred.png"
                ),
                "mask_url": (
                    f"/asset/{inference_asset_root}/{layer_id}/"
                    "web-mercator-inference-mask.png"
                ),
                "inferred_pixel_count": inference_report.get("inferred_pixel_count"),
                "method": inference_report.get("method"),
                "warning": inference_report.get("warning"),
                "artifact_root": inference_asset_root,
                "manifest_sha256": inference_manifest_sha256,
            }
        enclosed_fill = None
        if enclosed_fill_root is not None and isinstance(enclosed_fill_manifest, dict):
            enclosed_report = next(
                (
                    item
                    for item in enclosed_fill_manifest.get("layers", [])
                    if item.get("layer_id") == layer_id
                ),
                None,
            )
            values_path = (
                enclosed_fill_root
                / layer_id
                / "web-mercator-enclosed-fill-values.png"
            )
            mask_path = (
                enclosed_fill_root
                / layer_id
                / "web-mercator-enclosed-fill-mask.png"
            )
            if (
                isinstance(enclosed_report, dict)
                and values_path.exists()
                and mask_path.exists()
            ):
                enclosed_fill = {
                    "values_url": (
                        f"/asset/{enclosed_fill_root.name}/{layer_id}/"
                        "web-mercator-enclosed-fill-values.png"
                    ),
                    "mask_url": (
                        f"/asset/{enclosed_fill_root.name}/{layer_id}/"
                        "web-mercator-enclosed-fill-mask.png"
                    ),
                    "filled_component_count": enclosed_report.get(
                        "filled_component_count"
                    ),
                    "filled_pixel_count": enclosed_report.get("filled_pixel_count"),
                    "maximum_area_exclusive": enclosed_report.get(
                        "maximum_area_exclusive"
                    ),
                    "artifact_root": enclosed_fill_root.name,
                }
        layers.append(
            {
                "id": layer_id,
                "label": layer_id.replace("-", " ").title(),
                "kind": report.get("kind"),
                "class_id_url": f"/asset/{layer_id}/web-mercator-class-id.png",
                "categories": categories,
                "zero_coverage_categories": [
                    {
                        "id": category["id"],
                        "label": category["label"],
                    }
                    for category in categories
                    if category["pixel_count"] == 0
                ],
                "metrics": {
                    "source_eligible_pixel_count": extraction.get("eligible_pixel_count"),
                    "source_classified_pixel_count": extraction.get("classified_pixel_count"),
                    "source_ambiguous_pixel_count": extraction.get("ambiguous_pixel_count"),
                    "source_nodata_pixel_count": report.get("source_nodata_pixel_count"),
                    "target_valid_pixel_count": valid_target_count or None,
                    "target_observed_pixel_count": (
                        final_target_count - target_completion_count
                        if final_target_count
                        else None
                    ),
                    "target_completion_pixel_count": target_completion_count,
                    "target_final_classified_pixel_count": final_target_count or None,
                    "target_unclassified_pixel_count": (
                        valid_target_count - final_target_count
                        if valid_target_count and final_target_count
                        else None
                    ),
                },
                "inference": inference,
                "enclosed_fill": enclosed_fill,
            }
        )
    if not layers:
        raise ValueError("No classified Web-Mercator layers are available for review")
    grid = manifest["alignment"]["inspection"]["grid"]
    decision_path = run_dir / "classification-review-decision.json"
    inference_decision_path = run_dir / "inference-review-decision.json"
    inference_exclusions_path = run_dir / "inference-exclusions.json"
    alignment_path = run_dir / "review-decision.json"
    alignment_decision = json.loads(alignment_path.read_text()) if alignment_path.exists() else None
    inference_decision = (
        json.loads(inference_decision_path.read_text())
        if inference_decision_path.exists()
        else None
    )
    if (
        not isinstance(inference_decision, dict)
        or inference_decision.get("inference_manifest_sha256")
        != inference_manifest_sha256
    ):
        inference_decision = None
    stamp_corrections = (
        json.loads(stamp_corrections_path.read_text())
        if stamp_corrections_path.exists()
        else None
    )
    if (
        not isinstance(stamp_corrections, dict)
        or stamp_corrections.get("extraction_manifest_sha256") != _sha256(manifest_path)
        or (
            inference_root is not None
            and stamp_corrections.get("inference_manifest_sha256")
            != inference_manifest_sha256
        )
    ):
        stamp_corrections = None
    inference_exclusions = (
        json.loads(inference_exclusions_path.read_text())
        if inference_exclusions_path.exists()
        else None
    )
    if (
        not isinstance(inference_exclusions, dict)
        or inference_exclusions.get("extraction_manifest_sha256")
        != _sha256(manifest_path)
        or inference_exclusions.get("inference_manifest_sha256")
        != inference_manifest_sha256
    ):
        inference_exclusions = None
    return {
        "dataset_id": manifest["dataset_id"],
        "title": manifest["title"],
        "manifest_sha256": _sha256(manifest_path),
        "width": int(grid["width"]),
        "height": int(grid["height"]),
        "source_url": "/asset/web-mercator-source.jpg",
        "warnings": [str(warning) for warning in manifest.get("warnings", [])],
        "layers": layers,
        "alignment_decision": alignment_decision,
        "decision": json.loads(decision_path.read_text()) if decision_path.exists() else None,
        "inference_decision": inference_decision,
        "stamp_corrections": stamp_corrections,
        "inference_exclusions": inference_exclusions,
    }


def write_classification_decision(
    run_dir: Path, payload: Dict[str, object], request: Dict[str, object]
) -> Dict[str, object]:
    status = str(request.get("status", ""))
    if status not in DECISIONS:
        raise ValueError(f"Decision must be one of: {', '.join(sorted(DECISIONS))}")
    notes = str(request.get("notes", "")).strip()
    if len(notes) > 10_000:
        raise ValueError("Review notes must be 10,000 characters or fewer")
    alignment = payload.get("alignment_decision")
    if status == "approved" and (
        not isinstance(alignment, dict) or alignment.get("status") != "approved"
    ):
        raise ValueError("Classification cannot be approved before alignment is approved")
    decision = {
        "schema_version": 1,
        "dataset_id": payload["dataset_id"],
        "scope": "classification",
        "status": status,
        "notes": notes,
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "extraction_manifest_sha256": payload["manifest_sha256"],
        "alignment_review": {
            "status": alignment.get("status") if isinstance(alignment, dict) else None,
            "reviewed_at": alignment.get("reviewed_at") if isinstance(alignment, dict) else None,
        },
        "evidence_policy": {
            "primary": "source_pixels_compared_with_each_legend_category_mask",
            "reconstruction": (
                "target completion remains separately masked and is reviewed as "
                "neighbor-derived evidence rather than direct observation"
            ),
            "outside_boundary": "always transparent",
        },
        "known_limitations": {
            "manifest_warnings": [str(item) for item in payload.get("warnings", [])],
            "zero_coverage_categories": [
                {
                    "layer_id": str(layer["id"]),
                    "categories": layer.get("zero_coverage_categories", []),
                }
                for layer in payload.get("layers", [])
                if layer.get("zero_coverage_categories")
            ],
        },
    }
    (run_dir / "classification-review-decision.json").write_text(
        json.dumps(decision, indent=2) + "\n"
    )
    return decision


def write_inference_decision(
    run_dir: Path, payload: Dict[str, object], request: Dict[str, object]
) -> Dict[str, object]:
    status = str(request.get("status", ""))
    if status not in DECISIONS:
        raise ValueError(f"Decision must be one of: {', '.join(sorted(DECISIONS))}")
    notes = str(request.get("notes", "")).strip()
    if len(notes) > 10_000:
        raise ValueError("Review notes must be 10,000 characters or fewer")
    classification = payload.get("decision")
    if status == "approved" and (
        not isinstance(classification, dict) or classification.get("status") != "approved"
    ):
        raise ValueError("Inference cannot be approved before classification is approved")
    inference_hashes = sorted(
        {
            str(layer["inference"]["manifest_sha256"])
            for layer in payload.get("layers", [])
            if isinstance(layer.get("inference"), dict)
        }
    )
    if not inference_hashes:
        raise ValueError("No inference artifact is available for review")
    decision = {
        "schema_version": 1,
        "dataset_id": payload["dataset_id"],
        "scope": "inference",
        "status": status,
        "notes": notes,
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "extraction_manifest_sha256": payload["manifest_sha256"],
        "inference_manifest_sha256": inference_hashes[0],
        "classification_review": {
            "status": classification.get("status") if isinstance(classification, dict) else None,
            "reviewed_at": (
                classification.get("reviewed_at") if isinstance(classification, dict) else None
            ),
        },
        "evidence_policy": {
            "primary": "neon_inference_mask_compared_with_observed_source_and_classes",
            "disclosure": "observed_and_inferred_pixels_remain_separately_toggleable",
        },
    }
    (run_dir / "inference-review-decision.json").write_text(
        json.dumps(decision, indent=2) + "\n"
    )
    return decision


def write_stamp_corrections(
    run_dir: Path, payload: Dict[str, object], request: Dict[str, object]
) -> Dict[str, object]:
    raw_operations = request.get("operations", [])
    if not isinstance(raw_operations, list):
        raise ValueError("Stamp operations must be a list")
    operations = validate_stamp_operations(
        raw_operations,
        width=int(payload["width"]),
        height=int(payload["height"]),
        layer_ids=[str(layer["id"]) for layer in payload.get("layers", [])],
    )
    inference_hashes = sorted(
        {
            str(layer["inference"]["manifest_sha256"])
            for layer in payload.get("layers", [])
            if isinstance(layer.get("inference"), dict)
        }
    )
    corrections = {
        "schema_version": 3,
        "dataset_id": payload["dataset_id"],
        "scope": "manual_clone_stamp",
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "extraction_manifest_sha256": payload["manifest_sha256"],
        "inference_manifest_sha256": inference_hashes[0] if inference_hashes else None,
        "coordinate_space": {
            "crs": "EPSG:3857 inspection raster pixels",
            "width": int(payload["width"]),
            "height": int(payload["height"]),
        },
        "policy": {
            "source": (
                "observed_for_legacy_operations_or_composite_at_operation_time"
            ),
            "target": "manual_override_patch_at_any_raster_pixel",
            "shape": "circle",
            "interaction": "one_operation_per_destination_click",
            "observed": "underlying_artifact_never_modified",
            "overlap": "last_saved_stamp_wins",
            "publication": "manual_override_applied_above_observed_and_inferred",
        },
        "operations": operations,
    }
    (run_dir / "stamp-corrections.json").write_text(
        json.dumps(corrections, indent=2) + "\n"
    )
    enclosed_selection_path = run_dir / "enclosed-hole-fill-selection.json"
    if enclosed_selection_path.exists():
        enclosed_selection = json.loads(enclosed_selection_path.read_text())
        if enclosed_selection.get("enabled") is True:
            enclosed_root = _latest_enclosed_fill_root(run_dir)
            generate_enclosed_fill_artifact(
                run_dir,
                enclosed_root or (run_dir / "enclosed-fill-v1"),
                maximum_area_exclusive=int(
                    enclosed_selection.get("maximum_area_exclusive", 50)
                ),
            )
    return corrections


def write_inference_exclusions(
    run_dir: Path, payload: Dict[str, object], request: Dict[str, object]
) -> Dict[str, object]:
    raw_operations = request.get("operations", [])
    if not isinstance(raw_operations, list):
        raise ValueError("Inference exclusion operations must be a list")
    inference_hashes = sorted(
        {
            str(layer["inference"]["manifest_sha256"])
            for layer in payload.get("layers", [])
            if isinstance(layer.get("inference"), dict)
        }
    )
    if not inference_hashes:
        raise ValueError("No inference artifact is available to edit")
    operations = validate_inference_exclusions(
        raw_operations,
        width=int(payload["width"]),
        height=int(payload["height"]),
        layer_ids=[str(layer["id"]) for layer in payload.get("layers", [])],
    )
    exclusions = {
        "schema_version": 1,
        "dataset_id": payload["dataset_id"],
        "scope": "manual_inference_exclusion",
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "extraction_manifest_sha256": payload["manifest_sha256"],
        "inference_manifest_sha256": inference_hashes[0],
        "coordinate_space": {
            "crs": "EPSG:3857 inspection raster pixels",
            "width": int(payload["width"]),
            "height": int(payload["height"]),
        },
        "policy": {
            "target": "automatic_inference_pixels_only",
            "observed": "never_modified",
            "source_artifact": "immutable",
            "publication": "subtract_exclusion_mask_from_automatic_inference",
        },
        "operations": operations,
    }
    (run_dir / "inference-exclusions.json").write_text(
        json.dumps(exclusions, indent=2) + "\n"
    )
    return exclusions


def _handler(run_dir: Path):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format_string: str, *args) -> None:
            print(f"[classification-review] {self.address_string()} {format_string % args}")

        def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/":
                self._send(APP_HTML.encode(), "text/html; charset=utf-8")
                return
            if self.path == "/session.json":
                try:
                    self._send(
                        json.dumps(build_classification_payload(run_dir)).encode(),
                        "application/json",
                    )
                except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
                    self._send(
                        json.dumps({"error": str(error)}).encode(),
                        "application/json",
                        HTTPStatus.BAD_REQUEST,
                    )
                return
            if self.path.startswith("/asset/"):
                relative = Path(unquote(self.path.removeprefix("/asset/")))
                candidate = (run_dir / relative).resolve()
                try:
                    candidate.relative_to(run_dir.resolve())
                except ValueError:
                    self._send(b"Not found", "text/plain", HTTPStatus.NOT_FOUND)
                    return
                if not candidate.is_file():
                    self._send(b"Not found", "text/plain", HTTPStatus.NOT_FOUND)
                    return
                content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
                self._send(candidate.read_bytes(), content_type)
                return
            self._send(b"Not found", "text/plain", HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            if self.path not in {
                "/decision",
                "/inference-decision",
                "/stamp-corrections",
                "/inference-exclusions",
            }:
                self._send(b"Not found", "text/plain", HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                request = json.loads(self.rfile.read(length))
                payload = build_classification_payload(run_dir)
                if self.path == "/decision":
                    decision = write_classification_decision(run_dir, payload, request)
                    response = {"ok": True, "decision": decision}
                elif self.path == "/inference-decision":
                    decision = write_inference_decision(run_dir, payload, request)
                    response = {"ok": True, "decision": decision}
                elif self.path == "/stamp-corrections":
                    corrections = write_stamp_corrections(run_dir, payload, request)
                    response = {"ok": True, "corrections": corrections}
                else:
                    exclusions = write_inference_exclusions(run_dir, payload, request)
                    response = {"ok": True, "exclusions": exclusions}
                self._send(json.dumps(response).encode(), "application/json")
            except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                self._send(
                    json.dumps({"ok": False, "error": str(error)}).encode(),
                    "application/json",
                    HTTPStatus.BAD_REQUEST,
                )

    return Handler


def serve_classification_review(
    run_dir: Path,
    host: str = "127.0.0.1",
    port: int = 8773,
    open_browser: bool = True,
) -> None:
    run_dir = run_dir.resolve()
    build_classification_payload(run_dir)
    server = ThreadingHTTPServer((host, port), _handler(run_dir))
    url = f"http://{host}:{server.server_port}"
    print(f"MapScan classification review: {url}")
    print("Press Ctrl-C to stop after review.")
    if open_browser:
        threading.Timer(0.35, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


APP_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>MapScan classification review</title>
  <style>
    :root{color-scheme:dark;--bg:#090b0f;--panel:#11151b;--line:#29313c;--ink:#f4f6f8;--muted:#9da8b5;--accent:#21d4c2}
    *{box-sizing:border-box}html,body{height:100%}body{margin:0;overflow:hidden;color:var(--ink);background:var(--bg);font:13px/1.45 ui-sans-serif,system-ui,-apple-system,sans-serif}
    header{height:64px;display:flex;align-items:center;justify-content:space-between;gap:20px;padding:10px 18px;border-bottom:1px solid var(--line);background:#0d1015}h1{margin:0;font-size:18px}.subtitle{color:var(--muted);max-width:65vw;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.badge{padding:4px 8px;border:1px solid #3b4654;border-radius:999px;color:#cbd3dc;font-size:11px;text-transform:uppercase;letter-spacing:.06em}
    main{height:calc(100% - 64px);display:grid;grid-template-columns:minmax(0,1fr) 380px}.viewer{position:relative;min-width:0;overflow:hidden;background:#161a20}canvas#canvas{width:100%;height:100%;display:block;cursor:grab;touch-action:none}canvas#canvas.dragging{cursor:grabbing}.viewer-tools{position:absolute;left:12px;bottom:12px;display:flex;align-items:center;gap:8px;padding:7px 9px;border:1px solid #34404d;border-radius:8px;background:rgba(8,11,15,.86)}
    aside{overflow:auto;padding:15px;border-left:1px solid var(--line);background:var(--panel)}section{margin-bottom:18px}h2{margin:0 0 9px;font-size:12px;text-transform:uppercase;letter-spacing:.08em;color:#c8d1dc}button,select,textarea,input{font:inherit}button{border:1px solid #465363;border-radius:6px;padding:6px 9px;color:var(--ink);background:#1a2028;cursor:pointer}button:hover{border-color:#6c7d90}select,textarea{width:100%;border:1px solid #3b4654;border-radius:6px;color:var(--ink);background:#0d1117}select{padding:7px}textarea{min-height:82px;padding:8px;resize:vertical}
    .layer{display:grid;grid-template-columns:100px 1fr 40px;gap:8px;align-items:center;margin:8px 0}.value{color:var(--muted);text-align:right;font-variant-numeric:tabular-nums}.matched-opacity{grid-column:2/4;color:var(--muted)}input[type=range]{width:100%;accent-color:var(--accent)}.category-actions,.decision-buttons,.stamp-actions,.erase-actions{display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px;margin:8px 0}.categories{display:grid;gap:5px}.category{display:grid;grid-template-columns:20px 15px minmax(0,1fr) auto 42px;gap:7px;align-items:center;padding:6px;border:1px solid #29313c;border-radius:6px;background:#0d1117}.category .swatch{width:13px;height:13px;border-radius:2px}.category .label{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.category .count{color:var(--muted);font-variant-numeric:tabular-nums}.category .solo{padding:3px 6px;font-size:11px}.stamp-actions button.active{border-color:#2ee8d4;background:#123b39}.erase-actions button.active{border-color:#fb7185;background:#431d27}.stamp-status,.erase-status{min-height:20px;color:var(--muted)}
    dl{display:grid;grid-template-columns:1fr auto;gap:5px 12px;margin:0}dt{color:var(--muted)}dd{margin:0;font-variant-numeric:tabular-nums}.instruction{color:#c9d1da}.warning{padding:9px;border-left:3px solid #f0ad3d;background:#17191b;color:#c9d1da}.decision-buttons button[data-status=approved]{border-color:#248b69}.decision-buttons button[data-status=needs_revision]{border-color:#a77619}.decision-buttons button[data-status=rejected]{border-color:#a03f49}.status{min-height:20px;margin-top:8px;color:var(--muted)}
  </style>
</head>
<body>
  <header><div><h1>MapScan classification review</h1><div class="subtitle" id="title">Loading…</div></div><span class="badge" id="review-status">not reviewed</span></header>
  <main>
    <div class="viewer"><canvas id="canvas"></canvas><div class="viewer-tools"><button id="fit">Fit</button><span id="zoom">100%</span><span>Drag to pan · wheel to zoom</span></div></div>
    <aside>
      <section><h2>Comparison</h2><div class="layer"><label>Source</label><input id="source-opacity" type="range" min="0" max="100" value="100"><span class="value">100%</span></div><div class="layer"><label>Classes</label><input id="class-opacity" type="range" min="0" max="100" value="62"><span class="value">62%</span></div><div class="layer" id="enclosed-fill-row" style="display:none"><label>Small holes</label><span class="matched-opacity">Matches Classes</span></div><div class="layer" id="inference-row" style="display:none"><label>Inferred fill</label><input id="inference-opacity" type="range" min="0" max="100" value="85"><span class="value">85%</span></div><div class="layer"><label>Manual stamp</label><span class="matched-opacity">Matches Classes</span></div><select id="layer"></select></section>
      <section><h2>Legend categories</h2><p class="instruction">Toggle categories independently or use Solo to inspect one class against the source. Transparent areas are NoData or ambiguous.</p><div class="category-actions"><button id="all">All</button><button id="none">None</button><button id="invert">Invert</button></div><div class="categories" id="categories"></div></section>
      <section id="known-limitations-section" style="display:none"><h2>Known limitations</h2><div id="known-limitations"></div></section>
      <section><h2>Clone stamp</h2><p class="instruction">Copy one complete circular class-ID patch at a time. Hold <b>A</b> and click an intact or previously painted patch to set the source, then click each destination once. You can also use <b>Set source</b>. Dragging does not paint a trail. Manual patches use the same legend colors and opacity as the observed classes.</p><div class="layer"><label>Radius</label><input id="stamp-radius" type="range" min="3" max="120" value="24"><span class="value">24px</span></div><div class="stamp-actions"><button id="stamp-mode">Clone</button><button id="stamp-source">Set source</button><button id="stamp-undo">Undo</button></div><div class="stamp-actions"><button id="stamp-clear">Clear</button><button id="stamp-save">Save stamps</button><button id="stamp-reset-source">Reset source</button></div><div class="stamp-status" id="stamp-status">No source selected. Hold A and click to set one.</div></section>
      <section id="eraser-section" style="display:none"><h2>Inference eraser</h2><p class="instruction">Remove over-inferred cyan pixels. This brush can only reject automatic inference; observed classes and the source artifact are never changed.</p><div class="layer"><label>Radius</label><input id="erase-radius" type="range" min="3" max="120" value="24"><span class="value">24px</span></div><div class="erase-actions"><button id="erase-mode">Erase inference</button><button id="erase-undo">Undo</button><button id="erase-clear">Clear</button></div><div class="erase-actions"><button id="erase-save">Save exclusions</button></div><div class="erase-status" id="erase-status">No inference rejected.</div></section>
      <section><h2>Coverage</h2><dl id="metrics"></dl><div class="warning">Precision is favored over coverage. Check for false positives in terrain, labels, borders, water, and pale background—not only missing pixels.</div><div class="warning" id="inference-warning" style="display:none;margin-top:8px">Neon cyan pixels are inferred, not observed. Confirm that they only repair defensible same-class gaps.</div></section>
      <section><h2>Classification decision</h2><textarea id="notes" placeholder="Record false positives, missing regions, confused categories, or approval evidence."></textarea><div class="decision-buttons"><button data-status="approved">Approve</button><button data-status="needs_revision">Revise</button><button data-status="rejected">Reject</button></div><div class="status" id="decision-status"></div></section>
      <section id="inference-decision-section" style="display:none"><h2>Inference decision</h2><textarea id="inference-notes" placeholder="Record overfill, underfill, or why the inferred holes are acceptable."></textarea><div class="decision-buttons inference-buttons"><button data-status="approved">Approve</button><button data-status="needs_revision">Revise</button><button data-status="rejected">Reject</button></div><div class="status" id="inference-decision-status"></div></section>
    </aside>
  </main>
<script>
const canvas=document.querySelector('#canvas'),ctx=canvas.getContext('2d');
const overlay=document.createElement('canvas'),overlayCtx=overlay.getContext('2d');
const enclosedFillOverlay=document.createElement('canvas'),enclosedFillOverlayCtx=enclosedFillOverlay.getContext('2d');
const inferenceOverlay=document.createElement('canvas'),inferenceOverlayCtx=inferenceOverlay.getContext('2d');
const manualOverlay=document.createElement('canvas'),manualOverlayCtx=manualOverlay.getContext('2d');
const view={scale:1,minScale:.1,maxScale:16,x:0,y:0,dragging:false,lastX:0,lastY:0};
const images={};let payload,currentLayer,classPixels,classImageData,enclosedFillPixels,enclosedFillMaskPixels,enclosedFillImageData,inferredClassPixels,inferenceMaskPixels,inferenceImageData,exclusionMask,excludedPixelCount=0,manualValues,manualMask,manualImageData,manualPixelCount=0,palette,active=new Set();
const stamp={enabled:false,settingSource:false,keyboardSource:false,source:null,cursor:null,operations:[]};
const erase={enabled:false,cursor:null,painting:false,lastTarget:null,operations:[]};
function loadImage(url){return new Promise((resolve,reject)=>{const image=new Image();image.onload=()=>resolve(image);image.onerror=reject;image.src=url;});}
function fit(){if(!images.source)return;view.scale=Math.min(canvas.width/images.source.naturalWidth,canvas.height/images.source.naturalHeight);view.minScale=view.scale*.5;view.maxScale=Math.max(16,view.scale*32);view.x=(canvas.width-images.source.naturalWidth*view.scale)/2;view.y=(canvas.height-images.source.naturalHeight*view.scale)/2;draw();}
function drawCircle(point,radius,color,dashed=false){if(!point)return;ctx.save();ctx.globalAlpha=1;ctx.strokeStyle=color;ctx.lineWidth=2/view.scale;ctx.setLineDash(dashed?[6/view.scale,4/view.scale]:[]);ctx.beginPath();ctx.arc(point.x,point.y,radius,0,Math.PI*2);ctx.stroke();ctx.beginPath();ctx.moveTo(point.x-5/view.scale,point.y);ctx.lineTo(point.x+5/view.scale,point.y);ctx.moveTo(point.x,point.y-5/view.scale);ctx.lineTo(point.x,point.y+5/view.scale);ctx.stroke();ctx.restore();}
function draw(){ctx.setTransform(1,0,0,1,0,0);ctx.fillStyle='#161a20';ctx.fillRect(0,0,canvas.width,canvas.height);ctx.setTransform(view.scale,0,0,view.scale,view.x,view.y);ctx.imageSmoothingEnabled=false;ctx.globalAlpha=Number(document.querySelector('#source-opacity').value)/100;if(images.source)ctx.drawImage(images.source,0,0);const classAlpha=Number(document.querySelector('#class-opacity').value)/100;ctx.globalAlpha=classAlpha;if(classPixels)ctx.drawImage(overlay,0,0);if(enclosedFillImageData)ctx.drawImage(enclosedFillOverlay,0,0);ctx.globalAlpha=Number(document.querySelector('#inference-opacity').value)/100;if(inferenceMaskPixels)ctx.drawImage(inferenceOverlay,0,0);ctx.globalAlpha=classAlpha;if(manualValues)ctx.drawImage(manualOverlay,0,0);ctx.globalAlpha=1;if(stamp.enabled){const radius=Number(document.querySelector('#stamp-radius').value);drawCircle(stamp.source,radius,'#facc15',true);drawCircle(stamp.cursor,radius,'#f472b6');}if(erase.enabled)drawCircle(erase.cursor,Number(document.querySelector('#erase-radius').value),'#fb7185');document.querySelector('#zoom').textContent=`${Math.round(view.scale/view.minScale*50)}%`;}
function resize(){const rect=canvas.getBoundingClientRect();canvas.width=Math.max(1,Math.round(rect.width));canvas.height=Math.max(1,Math.round(rect.height));fit();}new ResizeObserver(resize).observe(canvas);
function imagePoint(event){const rect=canvas.getBoundingClientRect(),x=(event.clientX-rect.left-view.x)/view.scale,y=(event.clientY-rect.top-view.y)/view.scale;if(!classPixels||x<0||y<0||x>=classPixels.width||y>=classPixels.height)return null;return{x,y};}
function updateStampStatus(message){const current=stamp.operations.filter(item=>item.layer_id===currentLayer?.id).length,source=stamp.source?`Source ${Math.round(stamp.source.x)}, ${Math.round(stamp.source.y)}.`:'No source selected. Hold A and click to set one.';document.querySelector('#stamp-status').textContent=message||`${source} ${current} stamps · ${formatCount(manualPixelCount)} override pixels.`;}
function updateEraseStatus(message){const current=erase.operations.filter(item=>item.layer_id===currentLayer?.id).length;document.querySelector('#erase-status').textContent=message||`${current} strokes · ${formatCount(excludedPixelCount)} inferred pixels rejected.`;}
function applyClone(operation,record=true){if(!classPixels)return;const width=classPixels.width,height=classPixels.height,radius=Math.max(1,Math.round(operation.radius_px)),sourceX=Math.round(operation.source[0]),sourceY=Math.round(operation.source[1]),targetX=Math.round(operation.target[0]),targetY=Math.round(operation.target[1]),diameter=radius*2+1,sourcePatch=new Uint8Array(diameter*diameter),sourceValid=new Uint8Array(diameter*diameter),sourceMode=operation.source_mode||'observed';for(let oy=-radius;oy<=radius;oy++){for(let ox=-radius;ox<=radius;ox++){if(ox*ox+oy*oy>radius*radius)continue;const sx=sourceX+ox,sy=sourceY+oy;if(sx<0||sy<0||sx>=width||sy>=height)continue;const sourceIndex=sy*width+sx,patchIndex=(oy+radius)*diameter+(ox+radius);sourcePatch[patchIndex]=sourceMode==='composite_at_operation_time'&&manualMask[sourceIndex]?manualValues[sourceIndex]:classPixels.data[sourceIndex*4];sourceValid[patchIndex]=1;}}for(let oy=-radius;oy<=radius;oy++){for(let ox=-radius;ox<=radius;ox++){const patchIndex=(oy+radius)*diameter+(ox+radius);if(!sourceValid[patchIndex])continue;const tx=targetX+ox,ty=targetY+oy;if(tx<0||ty<0||tx>=width||ty>=height)continue;const targetIndex=ty*width+tx,sourceClass=sourcePatch[patchIndex],targetOffset=targetIndex*4,colorOffset=sourceClass*4;if(!manualMask[targetIndex])manualPixelCount++;manualValues[targetIndex]=sourceClass;manualMask[targetIndex]=1;if(classImageData)classImageData.data[targetOffset+3]=0;if(enclosedFillImageData)enclosedFillImageData.data[targetOffset+3]=0;if(inferenceImageData)inferenceImageData.data[targetOffset+3]=0;if(manualImageData){manualImageData.data[targetOffset]=palette[colorOffset];manualImageData.data[targetOffset+1]=palette[colorOffset+1];manualImageData.data[targetOffset+2]=palette[colorOffset+2];manualImageData.data[targetOffset+3]=sourceClass&&active.has(sourceClass)?255:0;}}}if(record)stamp.operations.push(operation);}
function replayStamps(rebuild=true){if(!classPixels)return;manualValues=new Uint8Array(classPixels.width*classPixels.height);manualMask=new Uint8Array(classPixels.width*classPixels.height);manualImageData=null;manualPixelCount=0;for(const operation of stamp.operations){if(operation.layer_id===currentLayer.id)applyClone(operation,false);}if(rebuild)rebuildOverlay();updateStampStatus();}
function stampAt(point){if(!point||!stamp.source)return;const operation={layer_id:currentLayer.id,source:[stamp.source.x,stamp.source.y],target:[point.x,point.y],radius_px:Number(document.querySelector('#stamp-radius').value),source_mode:'composite_at_operation_time'};applyClone(operation,true);const radius=Math.ceil(operation.radius_px),x=Math.max(0,Math.round(point.x)-radius),y=Math.max(0,Math.round(point.y)-radius),width=Math.min(classPixels.width-x,radius*2+1),height=Math.min(classPixels.height-y,radius*2+1);if(classImageData)overlayCtx.putImageData(classImageData,0,0,x,y,width,height);if(enclosedFillImageData)enclosedFillOverlayCtx.putImageData(enclosedFillImageData,0,0,x,y,width,height);if(inferenceImageData)inferenceOverlayCtx.putImageData(inferenceImageData,0,0,x,y,width,height);if(manualImageData)manualOverlayCtx.putImageData(manualImageData,0,0,x,y,width,height);renderMetrics();draw();updateStampStatus();}
function applyExclusion(operation,record=true){if(!inferenceMaskPixels)return;const width=inferenceMaskPixels.width,height=inferenceMaskPixels.height,radius=Math.max(1,Math.round(operation.radius_px)),centerX=Math.round(operation.center[0]),centerY=Math.round(operation.center[1]);for(let oy=-radius;oy<=radius;oy++){for(let ox=-radius;ox<=radius;ox++){if(ox*ox+oy*oy>radius*radius)continue;const x=centerX+ox,y=centerY+oy;if(x<0||y<0||x>=width||y>=height)continue;const pixelIndex=y*width+x,targetOffset=pixelIndex*4;if(inferenceMaskPixels.data[targetOffset]===0)continue;if(!exclusionMask[pixelIndex]){exclusionMask[pixelIndex]=1;excludedPixelCount++;}if(inferenceImageData)inferenceImageData.data[targetOffset+3]=0;}}if(record)erase.operations.push(operation);}
function replayExclusions(rebuild=true){if(!classPixels)return;exclusionMask=new Uint8Array(classPixels.width*classPixels.height);inferenceImageData=null;excludedPixelCount=0;if(inferenceMaskPixels){for(const operation of erase.operations){if(operation.layer_id===currentLayer.id)applyExclusion(operation,false);}}if(rebuild)rebuildOverlay();updateEraseStatus();}
function eraseAt(point){if(!point||!inferenceMaskPixels)return;const operation={layer_id:currentLayer.id,center:[point.x,point.y],radius_px:Number(document.querySelector('#erase-radius').value)};applyExclusion(operation,true);erase.lastTarget=point;const radius=Math.ceil(operation.radius_px),x=Math.max(0,Math.round(point.x)-radius),y=Math.max(0,Math.round(point.y)-radius),width=Math.min(classPixels.width-x,radius*2+1),height=Math.min(classPixels.height-y,radius*2+1);if(inferenceImageData)inferenceOverlayCtx.putImageData(inferenceImageData,0,0,x,y,width,height);renderMetrics();draw();updateEraseStatus();}
canvas.addEventListener('wheel',event=>{event.preventDefault();const rect=canvas.getBoundingClientRect(),mx=event.clientX-rect.left,my=event.clientY-rect.top,ix=(mx-view.x)/view.scale,iy=(my-view.y)/view.scale,next=Math.max(view.minScale,Math.min(view.maxScale,view.scale*Math.exp(-event.deltaY*.0015)));view.x=mx-ix*next;view.y=my-iy*next;view.scale=next;draw();},{passive:false});
canvas.addEventListener('pointerdown',event=>{const point=imagePoint(event);if(erase.enabled&&event.button===0&&point){erase.painting=true;erase.lastTarget=null;eraseAt(point);canvas.setPointerCapture(event.pointerId);return;}if(stamp.enabled&&event.button===0&&point){if(stamp.keyboardSource||stamp.settingSource||!stamp.source){stamp.source=point;if(!stamp.keyboardSource){stamp.settingSource=false;document.querySelector('#stamp-source').classList.remove('active');}updateStampStatus(stamp.keyboardSource?'Source updated. Release A, then click once per destination.':null);draw();return;}stampAt(point);canvas.setPointerCapture(event.pointerId);return;}view.dragging=true;view.lastX=event.clientX;view.lastY=event.clientY;canvas.classList.add('dragging');canvas.setPointerCapture(event.pointerId);});
canvas.addEventListener('pointermove',event=>{const point=imagePoint(event);stamp.cursor=point;erase.cursor=point;if(erase.painting&&point){const minimum=Math.max(2,Number(document.querySelector('#erase-radius').value)*.4),distance=erase.lastTarget?Math.hypot(point.x-erase.lastTarget.x,point.y-erase.lastTarget.y):Infinity;if(distance>=minimum)eraseAt(point);else draw();return;}if(view.dragging){view.x+=event.clientX-view.lastX;view.y+=event.clientY-view.lastY;view.lastX=event.clientX;view.lastY=event.clientY;}draw();});
canvas.addEventListener('pointerup',event=>{erase.painting=false;view.dragging=false;canvas.classList.remove('dragging');if(canvas.hasPointerCapture(event.pointerId))canvas.releasePointerCapture(event.pointerId);});canvas.addEventListener('pointerleave',()=>{stamp.cursor=null;erase.cursor=null;draw();});canvas.addEventListener('dblclick',event=>{if(!stamp.enabled&&!erase.enabled)fit();});document.querySelector('#fit').onclick=fit;
function keyboardFieldActive(){const tag=document.activeElement?.tagName;return tag==='INPUT'||tag==='TEXTAREA'||tag==='SELECT';}
window.addEventListener('keydown',event=>{if(event.key.toLowerCase()!=='a'||event.metaKey||event.ctrlKey||event.altKey||keyboardFieldActive())return;event.preventDefault();if(stamp.keyboardSource)return;stamp.keyboardSource=true;stamp.settingSource=true;stamp.enabled=true;erase.enabled=false;erase.painting=false;document.querySelector('#erase-mode').classList.remove('active');document.querySelector('#stamp-mode').classList.add('active');document.querySelector('#stamp-source').classList.add('active');canvas.style.cursor='crosshair';updateStampStatus('A held: click the intact patch to set or move the source.');draw();});
window.addEventListener('keyup',event=>{if(event.key.toLowerCase()!=='a'||!stamp.keyboardSource)return;event.preventDefault();stamp.keyboardSource=false;stamp.settingSource=false;document.querySelector('#stamp-source').classList.remove('active');updateStampStatus(stamp.source?'Source set. Click once per destination.':'No source selected. Hold A and click to set one.');draw();});
window.addEventListener('blur',()=>{if(!stamp.keyboardSource)return;stamp.keyboardSource=false;stamp.settingSource=false;document.querySelector('#stamp-source').classList.remove('active');updateStampStatus();draw();});
for(const id of ['source-opacity','class-opacity','inference-opacity'])document.querySelector(`#${id}`).oninput=event=>{event.target.nextElementSibling.textContent=`${event.target.value}%`;draw();};
document.querySelector('#stamp-radius').oninput=event=>{event.target.nextElementSibling.textContent=`${event.target.value}px`;draw();};
document.querySelector('#erase-radius').oninput=event=>{event.target.nextElementSibling.textContent=`${event.target.value}px`;draw();};
function formatCount(value){return value==null?'—':Number(value).toLocaleString();}
function renderMetrics(){const m=currentLayer.metrics||{},valid=Number(m.target_valid_pixel_count||0),observed=Number(m.target_observed_pixel_count||0),completed=Number(m.target_completion_pixel_count||0),classified=Number(m.target_final_classified_pixel_count||0),unknown=m.target_unclassified_pixel_count,inferred=currentLayer.inference?.inferred_pixel_count,enclosed=currentLayer.enclosed_fill?.filled_pixel_count;document.querySelector('#metrics').innerHTML=Object.entries({'Valid target pixels':valid?formatCount(valid):'—','Directly classified on target':observed?formatCount(observed):'—','Reconstructed cartography':formatCount(completed),'Final classified':classified?formatCount(classified):'—','Still unknown inside':unknown==null?'—':formatCount(Number(unknown)),'Final classified share':valid?`${(classified/valid*100).toFixed(1)}%`:'—','Small enclosed holes':enclosed==null?'—':`${formatCount(enclosed)} (<${currentLayer.enclosed_fill.maximum_area_exclusive}px each)`,'Additional inferred fill':inferred==null?'—':`${formatCount(inferred)} (${(Number(inferred)/Math.max(observed,1)*100).toFixed(2)}% of observed)`,'Rejected inference':inferred==null?'—':formatCount(excludedPixelCount),'Retained inference':inferred==null?'—':formatCount(Math.max(0,Number(inferred)-excludedPixelCount)),'Manual stamp':formatCount(manualPixelCount)}).map(([k,v])=>`<dt>${k}</dt><dd>${v}</dd>`).join('');}
function rebuildOverlay(){if(!classPixels)return;palette=new Uint8ClampedArray(256*4);for(const category of currentLayer.categories){if(!active.has(category.class_id))continue;const offset=category.class_id*4;palette[offset]=category.display_rgb[0];palette[offset+1]=category.display_rgb[1];palette[offset+2]=category.display_rgb[2];palette[offset+3]=255;}classImageData=overlayCtx.createImageData(overlay.width,overlay.height);const source=classPixels.data,target=classImageData.data;for(let i=0;i<source.length;i+=4){const pixelIndex=i/4;if(manualMask&&manualMask[pixelIndex])continue;const offset=source[i]*4;target[i]=palette[offset];target[i+1]=palette[offset+1];target[i+2]=palette[offset+2];target[i+3]=palette[offset+3];}overlayCtx.putImageData(classImageData,0,0);enclosedFillImageData=null;if(enclosedFillPixels&&enclosedFillMaskPixels){enclosedFillImageData=enclosedFillOverlayCtx.createImageData(enclosedFillOverlay.width,enclosedFillOverlay.height);const enclosedTarget=enclosedFillImageData.data,enclosedValues=enclosedFillPixels.data,enclosedMask=enclosedFillMaskPixels.data;for(let i=0;i<enclosedMask.length;i+=4){const pixelIndex=i/4,classId=enclosedValues[i];if(enclosedMask[i]===0||(manualMask&&manualMask[pixelIndex])||!active.has(classId))continue;const offset=classId*4;enclosedTarget[i]=palette[offset];enclosedTarget[i+1]=palette[offset+1];enclosedTarget[i+2]=palette[offset+2];enclosedTarget[i+3]=255;}enclosedFillOverlayCtx.putImageData(enclosedFillImageData,0,0);}inferenceImageData=null;if(inferenceMaskPixels&&inferredClassPixels){inferenceImageData=inferenceOverlayCtx.createImageData(inferenceOverlay.width,inferenceOverlay.height);const inferredTarget=inferenceImageData.data,mask=inferenceMaskPixels.data,inferredSource=inferredClassPixels.data;for(let i=0;i<mask.length;i+=4){const pixelIndex=i/4;if(mask[i]===0||(exclusionMask&&exclusionMask[pixelIndex])||(manualMask&&manualMask[pixelIndex])||!active.has(inferredSource[i]))continue;inferredTarget[i]=0;inferredTarget[i+1]=238;inferredTarget[i+2]=238;inferredTarget[i+3]=255;}inferenceOverlayCtx.putImageData(inferenceImageData,0,0);}if(manualValues){manualImageData=manualOverlayCtx.createImageData(manualOverlay.width,manualOverlay.height);const manualTarget=manualImageData.data;for(let i=0;i<manualValues.length;i++){const classId=manualValues[i];if(!manualMask[i]||classId===0||!active.has(classId))continue;const offset=classId*4,targetOffset=i*4;manualTarget[targetOffset]=palette[offset];manualTarget[targetOffset+1]=palette[offset+1];manualTarget[targetOffset+2]=palette[offset+2];manualTarget[targetOffset+3]=255;}manualOverlayCtx.putImageData(manualImageData,0,0);}renderMetrics();draw();}
function syncChecks(){for(const input of document.querySelectorAll('.category input'))input.checked=active.has(Number(input.value));}
function renderCategories(){const root=document.querySelector('#categories');root.innerHTML='';for(const category of currentLayer.categories){const row=document.createElement('div');row.className='category';row.innerHTML=`<input type="checkbox" value="${category.class_id}" checked><span class="swatch" style="background:rgb(${category.display_rgb.join(',')})"></span><span class="label" title="${category.label}">${category.label}</span><span class="count">${formatCount(category.pixel_count)}</span><button class="solo">Solo</button>`;row.querySelector('input').onchange=event=>{event.target.checked?active.add(category.class_id):active.delete(category.class_id);rebuildOverlay();};row.querySelector('.solo').onclick=()=>{active=new Set([category.class_id]);syncChecks();rebuildOverlay();};root.appendChild(row);}}
function renderKnownLimitations(){const section=document.querySelector('#known-limitations-section'),root=document.querySelector('#known-limitations'),messages=[];const empty=currentLayer.zero_coverage_categories||[];if(empty.length)messages.push(`No distinguishable pixels were recovered for: ${empty.map(item=>item.label).join(', ')}. These legend entries remain explicit zero-coverage classes rather than guessed data.`);for(const warning of payload.warnings||[])messages.push(warning);root.innerHTML='';for(const message of messages){const row=document.createElement('div');row.className='warning';row.style.marginBottom='8px';row.textContent=message;root.appendChild(row);}section.style.display=messages.length?'block':'none';}
async function decodePixels(url){const image=await loadImage(url),surface=document.createElement('canvas'),surfaceCtx=surface.getContext('2d',{willReadFrequently:true});surface.width=image.naturalWidth;surface.height=image.naturalHeight;surfaceCtx.drawImage(image,0,0);return surfaceCtx.getImageData(0,0,surface.width,surface.height);}
async function selectLayer(){currentLayer=payload.layers.find(item=>item.id===document.querySelector('#layer').value);active=new Set(currentLayer.categories.map(item=>item.class_id));stamp.source=null;stamp.settingSource=false;stamp.keyboardSource=false;renderCategories();renderKnownLimitations();document.querySelector('#decision-status').textContent='Decoding full-resolution class IDs…';classPixels=await decodePixels(currentLayer.class_id_url);overlay.width=enclosedFillOverlay.width=inferenceOverlay.width=manualOverlay.width=classPixels.width;overlay.height=enclosedFillOverlay.height=inferenceOverlay.height=manualOverlay.height=classPixels.height;enclosedFillPixels=enclosedFillMaskPixels=inferredClassPixels=inferenceMaskPixels=null;const hasEnclosedFill=!!currentLayer.enclosed_fill,hasInference=!!currentLayer.inference;document.querySelector('#enclosed-fill-row').style.display=hasEnclosedFill?'grid':'none';document.querySelector('#inference-row').style.display=hasInference?'grid':'none';document.querySelector('#inference-warning').style.display=hasInference?'block':'none';document.querySelector('#inference-decision-section').style.display=hasInference?'block':'none';document.querySelector('#eraser-section').style.display=hasInference?'block':'none';if(hasEnclosedFill){[enclosedFillPixels,enclosedFillMaskPixels]=await Promise.all([decodePixels(currentLayer.enclosed_fill.values_url),decodePixels(currentLayer.enclosed_fill.mask_url)]);}if(hasInference){[inferredClassPixels,inferenceMaskPixels]=await Promise.all([decodePixels(currentLayer.inference.class_id_url),decodePixels(currentLayer.inference.mask_url)]);}else{erase.enabled=false;document.querySelector('#erase-mode').classList.remove('active');}replayStamps(false);replayExclusions(false);rebuildOverlay();updateStampStatus();updateEraseStatus();document.querySelector('#decision-status').textContent=payload.decision?`Saved ${payload.decision.reviewed_at}`:'';}
document.querySelector('#all').onclick=()=>{active=new Set(currentLayer.categories.map(item=>item.class_id));syncChecks();rebuildOverlay();};document.querySelector('#none').onclick=()=>{active.clear();syncChecks();rebuildOverlay();};document.querySelector('#invert').onclick=()=>{active=new Set(currentLayer.categories.filter(item=>!active.has(item.class_id)).map(item=>item.class_id));syncChecks();rebuildOverlay();};
document.querySelector('#stamp-mode').onclick=event=>{stamp.enabled=!stamp.enabled;event.target.classList.toggle('active',stamp.enabled);if(stamp.enabled){erase.enabled=false;erase.painting=false;document.querySelector('#erase-mode').classList.remove('active');}else{stamp.settingSource=false;stamp.keyboardSource=false;document.querySelector('#stamp-source').classList.remove('active');}canvas.style.cursor=stamp.enabled?'crosshair':'grab';if(stamp.enabled&&!stamp.source){stamp.settingSource=true;document.querySelector('#stamp-source').classList.add('active');updateStampStatus('Click an intact patch to set the clone source.');}else updateStampStatus();draw();};
document.querySelector('#stamp-source').onclick=event=>{stamp.enabled=true;stamp.settingSource=true;stamp.keyboardSource=false;erase.enabled=false;erase.painting=false;document.querySelector('#erase-mode').classList.remove('active');document.querySelector('#stamp-mode').classList.add('active');event.target.classList.add('active');canvas.style.cursor='crosshair';updateStampStatus('Click an intact patch to set the clone source.');};
document.querySelector('#stamp-reset-source').onclick=()=>{stamp.source=null;stamp.settingSource=true;document.querySelector('#stamp-source').classList.add('active');updateStampStatus('Click an intact patch to set the clone source.');draw();};
document.querySelector('#stamp-undo').onclick=()=>{for(let index=stamp.operations.length-1;index>=0;index--){if(stamp.operations[index].layer_id===currentLayer.id){stamp.operations.splice(index,1);break;}}replayStamps();};
document.querySelector('#stamp-clear').onclick=()=>{stamp.operations=stamp.operations.filter(item=>item.layer_id!==currentLayer.id);replayStamps();};
document.querySelector('#stamp-save').onclick=async()=>{updateStampStatus('Saving and rebuilding small-hole fill…');try{const response=await fetch('/stamp-corrections',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({operations:stamp.operations})}),result=await response.json();if(!response.ok)throw new Error(result.error||'Save failed');const refreshed=await fetch('/session.json'),nextPayload=await refreshed.json();if(!refreshed.ok)throw new Error(nextPayload.error||'Could not refresh corrections');payload=nextPayload;stamp.operations=result.corrections.operations;await selectLayer();updateStampStatus(`Saved ${result.corrections.saved_at}`);}catch(error){updateStampStatus(error.message);}};
document.querySelector('#erase-mode').onclick=event=>{erase.enabled=!erase.enabled;event.target.classList.toggle('active',erase.enabled);if(erase.enabled){stamp.enabled=false;stamp.settingSource=false;stamp.keyboardSource=false;document.querySelector('#stamp-mode').classList.remove('active');document.querySelector('#stamp-source').classList.remove('active');}canvas.style.cursor=erase.enabled?'crosshair':'grab';updateEraseStatus();draw();};
document.querySelector('#erase-undo').onclick=()=>{for(let index=erase.operations.length-1;index>=0;index--){if(erase.operations[index].layer_id===currentLayer.id){erase.operations.splice(index,1);break;}}replayExclusions();};
document.querySelector('#erase-clear').onclick=()=>{erase.operations=erase.operations.filter(item=>item.layer_id!==currentLayer.id);replayExclusions();};
document.querySelector('#erase-save').onclick=async()=>{updateEraseStatus('Saving…');try{const response=await fetch('/inference-exclusions',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({operations:erase.operations})}),result=await response.json();if(!response.ok)throw new Error(result.error||'Save failed');payload.inference_exclusions=result.exclusions;erase.operations=result.exclusions.operations;updateEraseStatus(`Saved ${result.exclusions.saved_at} · ${formatCount(excludedPixelCount)} pixels rejected.`);}catch(error){updateEraseStatus(error.message);}};
async function init(){const response=await fetch('/session.json'),result=await response.json();if(!response.ok)throw new Error(result.error||'Could not load classification review');payload=result;stamp.operations=payload.stamp_corrections?.operations||[];erase.operations=payload.inference_exclusions?.operations||[];document.querySelector('#title').textContent=payload.title;document.querySelector('#review-status').textContent=payload.decision?.status?.replace('_',' ')||'not reviewed';document.querySelector('#layer').innerHTML=payload.layers.map(item=>`<option value="${item.id}">${item.label}</option>`).join('');document.querySelector('#layer').onchange=selectLayer;if(payload.decision){document.querySelector('#notes').value=payload.decision.notes||'';document.querySelector('#decision-status').textContent=`Saved ${payload.decision.reviewed_at}`;}if(payload.inference_decision){document.querySelector('#inference-notes').value=payload.inference_decision.notes||'';document.querySelector('#inference-decision-status').textContent=`Saved ${payload.inference_decision.reviewed_at}`;}images.source=await loadImage(payload.source_url);await selectLayer();fit();}
for(const button of document.querySelectorAll('.decision-buttons button'))button.onclick=async()=>{const status=button.dataset.status,notes=document.querySelector('#notes').value;document.querySelector('#decision-status').textContent='Saving…';try{const response=await fetch('/decision',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({status,notes})}),result=await response.json();if(!response.ok)throw new Error(result.error||'Save failed');payload.decision=result.decision;document.querySelector('#review-status').textContent=status.replace('_',' ');document.querySelector('#decision-status').textContent=`Saved ${result.decision.reviewed_at}`;}catch(error){document.querySelector('#decision-status').textContent=error.message;}};
for(const button of document.querySelectorAll('.inference-buttons button'))button.onclick=async event=>{event.stopPropagation();const status=button.dataset.status,notes=document.querySelector('#inference-notes').value;document.querySelector('#inference-decision-status').textContent='Saving…';try{const response=await fetch('/inference-decision',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({status,notes})}),result=await response.json();if(!response.ok)throw new Error(result.error||'Save failed');payload.inference_decision=result.decision;document.querySelector('#inference-decision-status').textContent=`Saved ${result.decision.reviewed_at}`;}catch(error){document.querySelector('#inference-decision-status').textContent=error.message;}};
init().catch(error=>{document.querySelector('#title').textContent=error.message;});
</script>
</body>
</html>"""
