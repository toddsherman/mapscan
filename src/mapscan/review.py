"""Local full-resolution alignment and classification review interface."""

from __future__ import annotations

import hashlib
import json
import math
import mimetypes
import statistics
import threading
import webbrowser
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict
from urllib.parse import unquote

from .canonical_boundary import ACTIVE_CANONICAL_POINTER, load_active_canonical_border
from .categorical_comparison import region_pixel_bounds


DECISIONS = {"approved", "rejected", "needs_revision"}
MAX_CORRECTIONS = 200


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_review_payload(run_dir: Path) -> Dict[str, object]:
    manifest_path = run_dir / "extraction.json"
    if not manifest_path.exists():
        continuous_path = run_dir / "continuous-extraction.json"
        if continuous_path.exists():
            return _build_continuous_review_payload(run_dir, continuous_path)
        alignment_path = run_dir / "alignment.json"
        if alignment_path.exists():
            return _build_alignment_only_review_payload(run_dir, alignment_path)
        raise FileNotFoundError(f"Missing extraction manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    review = manifest.get("review")
    if not isinstance(review, dict) or not isinstance(review.get("assets"), dict):
        raise ValueError(
            "This extraction predates review assets; rerun extract-plan before review"
        )
    assets = dict(review["assets"])
    for relative in assets.values():
        if not (run_dir / str(relative)).exists():
            raise FileNotFoundError(f"Missing review asset: {run_dir / str(relative)}")

    layers = []
    for layer in manifest.get("layers", []):
        relative = f"{layer['id']}/web-mercator-preview.png"
        if (run_dir / relative).exists():
            layers.append(
                {
                    "id": layer["id"],
                    "label": str(layer["id"]).replace("-", " ").title(),
                    "url": f"/asset/{relative}",
                    "category_pixel_counts": layer.get("category_pixel_counts", {}),
                }
            )

    manifest_sha256 = _sha256(manifest_path)
    decision_path = run_dir / str(review.get("decision_path", "review-decision.json"))
    decision = json.loads(decision_path.read_text()) if decision_path.exists() else None
    corrections_path = run_dir / str(
        review.get("corrections_path", "alignment-corrections.json")
    )
    corrections = (
        json.loads(corrections_path.read_text()) if corrections_path.exists() else None
    )
    grid = manifest["alignment"]["inspection"]["grid"]
    review_regions = []
    plan_snapshot_path = run_dir / "plan.snapshot.json"
    if plan_snapshot_path.is_file():
        plan_snapshot = json.loads(plan_snapshot_path.read_text())
        for region in plan_snapshot.get("comparison_regions", []):
            left, top, right, bottom = region_pixel_bounds(region, grid)
            review_regions.append(
                {
                    "id": str(region["id"]),
                    "label": str(
                        region.get(
                            "label", str(region["id"]).replace("-", " ").title()
                        )
                    ),
                    "bounds_wgs84": region["bounds_wgs84"],
                    "pixel_bounds": [left, top, right, bottom],
                }
            )
    canonical_boundary = None
    asset_urls = {key: f"/asset/{value}" for key, value in assets.items()}
    if (
        ACTIVE_CANONICAL_POINTER.is_file()
        and grid.get("crs") is not None
        and grid.get("bounds") is not None
    ):
        canonical_manifest_path, canonical, pointer = load_active_canonical_border(
            ACTIVE_CANONICAL_POINTER
        )
        canonical_grid = canonical["source_grid"]
        for field in ("crs", "bounds"):
            if canonical_grid.get(field) != grid.get(field):
                raise ValueError(f"Active canonical boundary differs at {field}")
        asset_urls["state_overlay"] = "/canonical/overlay"
        canonical_boundary = {
            "canonical_boundary_id": canonical["canonical_boundary_id"],
            "pointer_sha256": _sha256(ACTIVE_CANONICAL_POINTER),
            "manifest_sha256": _sha256(canonical_manifest_path),
            "overlay_sha256": canonical["artifacts"]["overlay"]["sha256"],
            "mainland_component_count": canonical["topology"][
                "mainland_component_count"
            ],
            "island_component_count": canonical["topology"][
                "offshore_island_component_count"
            ],
        }
    return {
        "dataset_id": manifest["dataset_id"],
        "title": manifest["title"],
        "status": manifest["status"],
        "manifest_sha256": manifest_sha256,
        "width": grid["width"],
        "height": grid["height"],
        "alignment": manifest["alignment"],
        "county_residual": review.get("county_residual", {}),
        "county_reference": review.get("county_reference"),
        "county_diagnostic_enabled": review.get("county_diagnostic_enabled", True),
        "canonical_clip": review.get("canonical_clip"),
        "default_view": review.get("default_view", "source"),
        "assets": asset_urls,
        "canonical_boundary": canonical_boundary,
        "layers": layers,
        "review_regions": review_regions,
        "decision": decision,
        "corrections": corrections,
        "corrections_stale": bool(
            corrections
            and corrections.get("extraction_manifest_sha256") != manifest_sha256
        ),
    }


def _build_alignment_only_review_payload(
    run_dir: Path, alignment_path: Path
) -> Dict[str, object]:
    """Expose a warped original source before any data extraction is allowed."""

    alignment_source = json.loads(alignment_path.read_text())
    source_path = run_dir / "web-mercator-source.png"
    if not source_path.is_file():
        raise FileNotFoundError(
            "Alignment-only review requires web-mercator-source.png; rerun the source fit"
        )
    refit = alignment_source.get("global_refit")
    if not isinstance(refit, dict) or not isinstance(refit.get("grid"), dict):
        raise ValueError("Alignment-only review requires a global-refit grid")
    grid = refit["grid"]
    best = alignment_source.get("best", {})
    source_metrics = best.get("metrics", {})
    alignment_metrics = {
        **source_metrics,
        "control_point_median_px": source_metrics.get(
            "holdout_median_px_at_source_resolution"
        ),
        "review_correction_leave_one_out_p90_px": source_metrics.get(
            "holdout_p90_px_at_source_resolution"
        ),
    }
    canonical_manifest_path, canonical, _ = load_active_canonical_border(
        ACTIVE_CANONICAL_POINTER
    )
    canonical_grid = canonical["source_grid"]
    for field in ("crs", "bounds"):
        if canonical_grid.get(field) != grid.get(field):
            raise ValueError(f"Active canonical boundary differs at {field}")
    canonical_boundary = {
        "canonical_boundary_id": canonical["canonical_boundary_id"],
        "pointer_sha256": _sha256(ACTIVE_CANONICAL_POINTER),
        "manifest_sha256": _sha256(canonical_manifest_path),
        "overlay_sha256": canonical["artifacts"]["overlay"]["sha256"],
        "mainland_component_count": canonical["topology"][
            "mainland_component_count"
        ],
        "island_component_count": canonical["topology"][
            "offshore_island_component_count"
        ],
    }
    manifest_sha256 = _sha256(alignment_path)
    decision_path = run_dir / "review-decision.json"
    corrections_path = run_dir / "alignment-corrections.json"
    decision = json.loads(decision_path.read_text()) if decision_path.exists() else None
    corrections = (
        json.loads(corrections_path.read_text()) if corrections_path.exists() else None
    )
    return {
        "dataset_id": "california-topography-elevation-alignment",
        "title": "California Topography and Elevation · source alignment only",
        "status": alignment_source.get("status", "needs_visual_review"),
        "manifest_kind": "alignment",
        "alignment_only": True,
        "manifest_sha256": manifest_sha256,
        "width": int(grid["width"]),
        "height": int(grid["height"]),
        "alignment": {
            "mode": alignment_source.get("alignment_mode", "automatic"),
            "projection_crs": best.get(
                "projection_crs",
                alignment_source.get("reference", {}).get("crs", "—"),
            ),
            "metrics": alignment_metrics,
            "inspection": {"grid": grid},
            "source": {
                "path": str(alignment_path),
                "sha256": manifest_sha256,
            },
        },
        "county_residual": {},
        "county_reference": None,
        "county_diagnostic_enabled": False,
        "canonical_clip": None,
        "default_view": "source",
        "assets": {
            "source": "/asset/web-mercator-source.png",
            "state_overlay": "/canonical/overlay",
            "county_overlay": "/canonical/overlay",
            "county_residual": "/canonical/overlay",
        },
        "canonical_boundary": canonical_boundary,
        "layers": [
            {
                "id": "not-run",
                "label": "Data extraction not run",
                "url": "/asset/no-extraction.png",
                "category_pixel_counts": {},
            }
        ],
        "decision": decision,
        "corrections": corrections,
        "corrections_stale": bool(
            corrections
            and corrections.get("alignment_manifest_sha256") != manifest_sha256
        ),
    }


def _build_continuous_review_payload(
    run_dir: Path, manifest_path: Path
) -> Dict[str, object]:
    """Adapt a continuous-ramp run to the existing full-resolution reviewer."""

    manifest = json.loads(manifest_path.read_text())
    if manifest.get("extraction_kind") != "continuous_color_ramp":
        raise ValueError("Continuous review manifest has the wrong extraction kind")
    review = manifest.get("review")
    if not isinstance(review, dict) or not isinstance(review.get("assets"), dict):
        raise ValueError("Continuous extraction predates review assets; rerun it")
    assets = dict(review["assets"])
    for relative in assets.values():
        if not (run_dir / str(relative)).is_file():
            raise FileNotFoundError(f"Missing review asset: {run_dir / str(relative)}")
    layers = []
    for layer in review.get("layers", []):
        relative = str(layer["path"])
        if not (run_dir / relative).is_file():
            raise FileNotFoundError(f"Missing continuous review layer: {run_dir / relative}")
        layers.append(
            {
                "id": str(layer["id"]),
                "label": str(layer["label"]),
                "url": f"/asset/{relative}",
                "category_pixel_counts": {},
            }
        )
    if not layers:
        raise ValueError("Continuous extraction has no review layers")

    grid = manifest["warp"]
    alignment_path = Path(str(manifest["alignment"]))
    alignment_source = json.loads(alignment_path.read_text())
    best = alignment_source.get("best", {})
    source_metrics = best.get("metrics", {})
    alignment_metrics = {
        **source_metrics,
        "control_point_median_px": source_metrics.get(
            "holdout_median_px_at_source_resolution"
        ),
        "review_correction_leave_one_out_p90_px": source_metrics.get(
            "holdout_p90_px_at_source_resolution"
        ),
    }
    alignment = {
        "mode": alignment_source.get("alignment_mode", "automatic"),
        "projection_crs": best.get(
            "projection_crs", alignment_source.get("reference", {}).get("crs", "—")
        ),
        "metrics": alignment_metrics,
        "inspection": {"grid": grid},
        "source": {
            "path": str(alignment_path),
            "sha256": _sha256(alignment_path),
        },
    }
    canonical_manifest_path, canonical, _ = load_active_canonical_border(
        ACTIVE_CANONICAL_POINTER
    )
    canonical_grid = canonical["source_grid"]
    for field in ("crs", "bounds"):
        if canonical_grid.get(field) != grid.get(field):
            raise ValueError(f"Active canonical boundary differs at {field}")
    canonical_boundary = {
        "canonical_boundary_id": canonical["canonical_boundary_id"],
        "pointer_sha256": _sha256(ACTIVE_CANONICAL_POINTER),
        "manifest_sha256": _sha256(canonical_manifest_path),
        "overlay_sha256": canonical["artifacts"]["overlay"]["sha256"],
        "mainland_component_count": canonical["topology"]["mainland_component_count"],
        "island_component_count": canonical["topology"][
            "offshore_island_component_count"
        ],
    }
    manifest_sha256 = _sha256(manifest_path)
    decision_path = run_dir / str(review.get("decision_path", "review-decision.json"))
    corrections_path = run_dir / str(
        review.get("corrections_path", "alignment-corrections.json")
    )
    decision = json.loads(decision_path.read_text()) if decision_path.exists() else None
    corrections = (
        json.loads(corrections_path.read_text()) if corrections_path.exists() else None
    )
    target = manifest.get("target", {})
    clip = manifest.get("canonical_clip", {})
    asset_urls = {key: f"/asset/{value}" for key, value in assets.items()}
    asset_urls["state_overlay"] = "/canonical/overlay"
    return {
        "dataset_id": manifest["dataset_id"],
        "title": manifest["title"],
        "status": manifest["status"],
        "manifest_sha256": manifest_sha256,
        "width": int(grid["width"]),
        "height": int(grid["height"]),
        "alignment": alignment,
        "county_residual": {},
        "county_reference": None,
        "county_diagnostic_enabled": False,
        "canonical_clip": {
            "component_count": clip.get("component_count"),
            "source_pixel_count_removed": target.get("boundary_removed_pixel_count"),
            "source_pixels_outside_after_clip": target.get("colored_outside_pixel_count"),
        },
        "default_view": review.get("default_view", "source"),
        "assets": asset_urls,
        "canonical_boundary": canonical_boundary,
        "layers": layers,
        "decision": decision,
        "corrections": corrections,
        "corrections_stale": bool(
            corrections
            and corrections.get("extraction_manifest_sha256") != manifest_sha256
        ),
        "continuous": {
            "units": manifest.get("units", "meters above sea level"),
            "encoding": manifest.get("encoding"),
            "ramp_stops": manifest.get("ramp_stops", []),
            "special_values": manifest.get("special_values", []),
            "target": target,
        },
    }


def write_review_decision(
    run_dir: Path, payload: Dict[str, object], request: Dict[str, object]
) -> Dict[str, object]:
    status = str(request.get("status", ""))
    if status not in DECISIONS:
        raise ValueError(f"Decision must be one of: {', '.join(sorted(DECISIONS))}")
    notes = str(request.get("notes", "")).strip()
    if len(notes) > 10_000:
        raise ValueError("Review notes must be 10,000 characters or fewer")
    decision = {
        "schema_version": 1,
        "dataset_id": payload["dataset_id"],
        "status": status,
        "notes": notes,
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "review_manifest_sha256": payload["manifest_sha256"],
        **(
            {"alignment_manifest_sha256": payload["manifest_sha256"]}
            if payload.get("alignment_only")
            else {"extraction_manifest_sha256": payload["manifest_sha256"]}
        ),
        "evidence_policy": {
            "primary": "state_border_and_coastline",
            "secondary": "county_boundaries_when_present",
            "validation_only": "cartographic_city_markers",
        },
        "canonical_boundary": payload.get("canonical_boundary"),
    }
    path = run_dir / "review-decision.json"
    path.write_text(json.dumps(decision, indent=2) + "\n")
    return decision


def _validated_pixel_point(
    value: object, label: str, width: int, height: int
) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{label} must be an [x, y] pixel coordinate")
    x, y = value
    if (
        isinstance(x, bool)
        or isinstance(y, bool)
        or not isinstance(x, (int, float))
        or not isinstance(y, (int, float))
        or not math.isfinite(float(x))
        or not math.isfinite(float(y))
    ):
        raise ValueError(f"{label} must contain finite numbers")
    x, y = float(x), float(y)
    if not (0 <= x <= width - 1 and 0 <= y <= height - 1):
        raise ValueError(f"{label} lies outside the review raster")
    return x, y


def write_alignment_corrections(
    run_dir: Path, payload: Dict[str, object], request: Dict[str, object]
) -> Dict[str, object]:
    raw_corrections = request.get("corrections")
    if not isinstance(raw_corrections, list):
        raise ValueError("Corrections must be a list")
    if len(raw_corrections) > MAX_CORRECTIONS:
        raise ValueError(f"At most {MAX_CORRECTIONS} corrections may be saved")

    width, height = int(payload["width"]), int(payload["height"])
    grid = payload["alignment"]["inspection"]["grid"]
    bounds = [float(value) for value in grid["bounds"]]
    min_x, min_y, max_x, max_y = bounds
    x_denominator, y_denominator = max(width - 1, 1), max(height - 1, 1)

    def enrich(point: tuple[float, float]) -> Dict[str, object]:
        x, y = point
        normalized_x, normalized_y = x / x_denominator, y / y_denominator
        return {
            "pixel": {"x": x, "y": y},
            "normalized": {"x": normalized_x, "y": normalized_y},
            "web_mercator": {
                "x": min_x + normalized_x * (max_x - min_x),
                "y": max_y - normalized_y * (max_y - min_y),
            },
        }

    corrections = []
    magnitudes = []
    for index, raw in enumerate(raw_corrections, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"Correction {index} must be an object")
        reference = _validated_pixel_point(
            raw.get("reference"), f"Correction {index} reference point", width, height
        )
        source = _validated_pixel_point(
            raw.get("source"), f"Correction {index} source point", width, height
        )
        delta_x, delta_y = reference[0] - source[0], reference[1] - source[1]
        magnitude = math.hypot(delta_x, delta_y)
        magnitudes.append(magnitude)
        corrections.append(
            {
                "id": index,
                "reference": enrich(reference),
                "source": enrich(source),
                "required_source_displacement_px": {
                    "dx": delta_x,
                    "dy": delta_y,
                    "magnitude": magnitude,
                },
            }
        )

    record = {
        "schema_version": 2,
        "dataset_id": payload["dataset_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "review_manifest_sha256": payload["manifest_sha256"],
        **(
            {"alignment_manifest_sha256": payload["manifest_sha256"]}
            if payload.get("alignment_only")
            else {"extraction_manifest_sha256": payload["manifest_sha256"]}
        ),
        "direction": "authoritative_reference_to_current_warped_source",
        "instructions": (
            "Each arrow begins on an authoritative reference line and ends on "
            "the corresponding feature in the current warped source. The source "
            "is displaced back toward the arrow start. Zero-length arrows are fixed pins."
        ),
        "canonical_boundary": payload.get("canonical_boundary"),
        "grid": {
            "crs": grid["crs"],
            "bounds": bounds,
            "width": width,
            "height": height,
        },
        "summary": {
            "count": len(corrections),
            "median_magnitude_px": statistics.median(magnitudes) if magnitudes else 0.0,
            "max_magnitude_px": max(magnitudes, default=0.0),
        },
        "corrections": corrections,
    }
    path = run_dir / "alignment-corrections.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    return record


def _handler(run_dir: Path):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format_string: str, *args) -> None:
            print(f"[review] {self.address_string()} {format_string % args}")

        def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/":
                html = APP_HTML
                if build_review_payload(run_dir).get("alignment_only"):
                    html = html.replace(
                        'class="layer" id="classification-layer-control"',
                        'class="layer" id="classification-layer-control" hidden',
                    ).replace(
                        'id="classification-layer" aria-label="Classification layer"',
                        'id="classification-layer" aria-label="Classification layer" hidden',
                    ).replace(
                        'section id="canonical-clip"',
                        'section id="canonical-clip" hidden',
                    )
                self._send(html.encode(), "text/html; charset=utf-8")
                return
            if self.path == "/session.json":
                try:
                    body = json.dumps(build_review_payload(run_dir)).encode()
                    self._send(body, "application/json")
                except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
                    self._send(
                        json.dumps({"error": str(error)}).encode(),
                        "application/json",
                        HTTPStatus.BAD_REQUEST,
                    )
                return
            if self.path == "/canonical/overlay":
                try:
                    canonical_manifest_path, canonical, _ = load_active_canonical_border(
                        ACTIVE_CANONICAL_POINTER
                    )
                    record = canonical["artifacts"]["overlay"]
                    path = canonical_manifest_path.parent / str(record["path"])
                    if _sha256(path) != record["sha256"]:
                        raise ValueError("Active canonical overlay is stale")
                    self._send(path.read_bytes(), "image/png")
                except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
                    self._send(str(error).encode(), "text/plain", HTTPStatus.BAD_REQUEST)
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
            if self.path not in {"/decision", "/corrections"}:
                self._send(b"Not found", "text/plain", HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                request = json.loads(self.rfile.read(length))
                payload = build_review_payload(run_dir)
                if self.path == "/decision":
                    decision = write_review_decision(run_dir, payload, request)
                    response = {"ok": True, "decision": decision}
                else:
                    corrections = write_alignment_corrections(run_dir, payload, request)
                    response = {"ok": True, "corrections": corrections}
                self._send(json.dumps(response).encode(), "application/json")
            except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                self._send(
                    json.dumps({"ok": False, "error": str(error)}).encode(),
                    "application/json",
                    HTTPStatus.BAD_REQUEST,
                )

    return Handler


def serve_review(
    run_dir: Path,
    host: str = "127.0.0.1",
    port: int = 8767,
    open_browser: bool = True,
) -> None:
    run_dir = run_dir.resolve()
    build_review_payload(run_dir)
    server = ThreadingHTTPServer((host, port), _handler(run_dir))
    url = f"http://{host}:{server.server_port}"
    print(f"MapScan review: {url}")
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
  <title>MapScan alignment review</title>
  <style>
    :root { color-scheme:dark; --bg:#090b0f; --panel:#11151b; --line:#29313c; --ink:#f4f6f8; --muted:#9da8b5; --accent:#21d4c2; }
    * { box-sizing:border-box; }
    [hidden] { display:none !important; }
    html,body { height:100%; }
    body { margin:0; overflow:hidden; color:var(--ink); background:var(--bg); font:13px/1.45 ui-sans-serif,system-ui,-apple-system,sans-serif; }
    header { height:64px; display:flex; align-items:center; justify-content:space-between; gap:20px; padding:10px 18px; border-bottom:1px solid var(--line); background:#0d1015; }
    h1 { margin:0; font-size:18px; letter-spacing:-.02em; }
    .subtitle { color:var(--muted); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; max-width:65vw; }
    .badge { padding:4px 8px; border:1px solid #3b4654; border-radius:999px; color:#cbd3dc; font-size:11px; text-transform:uppercase; letter-spacing:.06em; }
    main { height:calc(100% - 64px); display:grid; grid-template-columns:minmax(0,1fr) 340px; }
    .viewer { position:relative; min-width:0; overflow:hidden; background:#161a20; }
    canvas { width:100%; height:100%; display:block; cursor:grab; touch-action:none; }
    canvas.dragging { cursor:grabbing; }
    canvas.correction-mode { cursor:crosshair; }
    .viewer-tools { position:absolute; left:12px; bottom:12px; display:flex; align-items:center; gap:8px; padding:7px 9px; border:1px solid #34404d; border-radius:8px; background:rgba(8,11,15,.86); backdrop-filter:blur(8px); }
    .viewer-tools select { width:auto; max-width:250px; padding:5px 28px 5px 8px; }
    button,select,textarea,input { font:inherit; }
    button { border:1px solid #465363; border-radius:6px; padding:6px 9px; color:var(--ink); background:#1a2028; cursor:pointer; }
    button:hover { border-color:#6c7d90; }
    button.active { border-color:var(--accent); color:#07110f; background:var(--accent); }
    button:disabled { cursor:not-allowed; opacity:.45; }
    .zoom { min-width:55px; color:#c6d0da; font-variant-numeric:tabular-nums; }
    aside { overflow:auto; padding:15px; border-left:1px solid var(--line); background:var(--panel); }
    section { margin-bottom:18px; }
    h2 { margin:0 0 9px; font-size:12px; text-transform:uppercase; letter-spacing:.08em; color:#c8d1dc; }
    .layer { display:grid; grid-template-columns:95px 1fr 38px; gap:8px; align-items:center; margin:8px 0; }
    .comparison-toggle { display:grid; grid-template-columns:1fr 1fr; gap:6px; margin:0 0 10px; }
    .comparison-hint { margin:-3px 0 10px; color:var(--muted); font-size:11px; text-align:center; }
    input[type=range] { width:100%; accent-color:var(--accent); }
    .value { text-align:right; color:var(--muted); font-variant-numeric:tabular-nums; }
    select,textarea { width:100%; border:1px solid #3b4654; border-radius:6px; color:var(--ink); background:#0d1117; }
    select { padding:7px; }
    textarea { min-height:82px; padding:8px; resize:vertical; }
    dl { display:grid; grid-template-columns:1fr auto; gap:5px 12px; margin:0; }
    dt { color:var(--muted); } dd { margin:0; font-variant-numeric:tabular-nums; }
    .warning { margin-top:9px; padding:9px; border-left:3px solid #f0ad3d; color:#c9d1da; background:#17191b; }
    .instruction { margin:0 0 9px; color:#c9d1da; }
    .direction { padding:9px; border:1px solid #3a4654; border-radius:6px; color:#e8edf2; background:#0c1117; }
    .direction b:first-child { color:#62e6d8; } .direction b:last-child { color:#ffb454; }
    .correction-actions { display:grid; grid-template-columns:1fr 1fr 1fr; gap:6px; margin-top:8px; }
    .legend { display:flex; gap:10px; margin-top:8px; color:var(--muted); }
    .swatch { display:inline-block; width:10px; height:10px; margin-right:4px; border-radius:2px; }
    .decision-buttons { display:grid; grid-template-columns:1fr 1fr 1fr; gap:6px; margin-top:8px; }
    .decision-buttons button[data-status=approved] { border-color:#248b69; }
    .decision-buttons button[data-status=needs_revision] { border-color:#a77619; }
    .decision-buttons button[data-status=rejected] { border-color:#a03f49; }
    .status { min-height:20px; margin-top:8px; color:var(--muted); }
    @media (max-width:900px) { main { grid-template-columns:1fr; grid-template-rows:minmax(360px,58vh) 1fr; overflow:auto; } aside { border-left:0; border-top:1px solid var(--line); } body { overflow:auto; } }
  </style>
</head>
<body>
  <header><div><h1>MapScan alignment review</h1><div class="subtitle" id="title">Loading…</div></div><span class="badge" id="review-status">not reviewed</span></header>
  <main>
    <div class="viewer"><canvas id="canvas"></canvas><div class="viewer-tools"><button id="reset-view">Fit</button><select id="review-region" aria-label="Review region"><option value="">Full state</option></select><span class="zoom" id="zoom">100%</span><span id="interaction-help">Drag to pan · wheel to zoom</span></div></div>
    <aside>
      <section><h2>Layers</h2><div class="comparison-toggle"><button id="show-source" class="active" aria-pressed="true">Source</button><button id="show-classification" aria-pressed="false">Extracted</button></div><div class="comparison-hint">Press Space to flip</div><div class="layer"><label for="source-opacity">Source</label><input id="source-opacity" type="range" min="0" max="100" value="100"><span class="value">100%</span></div><div class="layer" id="classification-layer-control"><label for="classification-opacity">Classes</label><input id="classification-opacity" type="range" min="0" max="100" value="0"><span class="value">0%</span></div><select id="classification-layer" aria-label="Classification layer"></select><div class="layer"><label id="state-label" for="state-opacity">State / coast</label><input id="state-opacity" type="range" min="0" max="100" value="100"><span class="value">100%</span></div><div class="layer" id="county-layer"><label id="county-label" for="county-opacity">Counties</label><input id="county-opacity" type="range" min="0" max="100" value="90"><span class="value">90%</span></div><div class="layer" id="residual-layer"><label for="residual-opacity">Residual</label><input id="residual-opacity" type="range" min="0" max="100" value="0"><span class="value">0%</span></div><div class="legend" id="residual-legend"><span><i class="swatch" style="background:#34d399"></i>≤3px</span><span><i class="swatch" style="background:#fab005"></i>3–8px</span><span><i class="swatch" style="background:#ef4444"></i>&gt;8px</span></div></section>
      <section><h2>Alignment</h2><dl id="alignment-metrics"></dl></section>
      <section id="canonical-clip"><h2>Canonical clip</h2><dl id="clip-metrics"></dl></section>
      <section id="county-diagnostic"><h2 id="county-diagnostic-title">County diagnostic</h2><dl id="county-metrics"></dl><div class="warning" id="county-warning"></div></section>
      <section><h2>Correction input</h2><p class="instruction">Add displacement arrows wherever the warped map drifts. Spread them around the state rather than concentrating them in one county.</p><div class="direction"><b>Start on reference:</b> the cyan or magenta authoritative line<br><b>End on source:</b> the matching dark feature in the map image<br>The processor moves the source back toward the start.</div><button id="correction-mode" style="width:100%;margin-top:8px">Add correction arrows</button><div class="correction-actions"><button id="undo-correction">Undo</button><button id="clear-corrections">Clear</button><button id="save-corrections">Save</button></div><div class="status" id="correction-status">No corrections</div><div class="instruction">A click without movement adds a fixed pin for an area that is already aligned.</div></section>
      <section><h2>Decision</h2><textarea id="notes" placeholder="Record local drift, category issues, or the reason for the decision."></textarea><div class="decision-buttons"><button data-status="approved">Approve</button><button data-status="needs_revision">Revise</button><button data-status="rejected">Reject</button></div><div class="status" id="decision-status"></div></section>
    </aside>
  </main>
<script>
const canvas=document.querySelector('#canvas'),ctx=canvas.getContext('2d');
const view={scale:1,x:0,y:0,minScale:.1,maxScale:16,dragging:false,lastX:0,lastY:0};
const opacity={source:1,classification:0,state:1,county:.9,residual:0};
let comparisonView='source';
const correction={enabled:false,dragging:false,draft:null,items:[],saved:false};
const images={};let payload=null;
function loadImage(url){return new Promise((resolve,reject)=>{const image=new Image();image.onload=()=>resolve(image);image.onerror=reject;image.src=url;});}
function fit(){if(!images.source)return;view.scale=Math.min(canvas.width/images.source.naturalWidth,canvas.height/images.source.naturalHeight);view.minScale=view.scale*.5;view.maxScale=Math.max(16,view.scale*32);view.x=(canvas.width-images.source.naturalWidth*view.scale)/2;view.y=(canvas.height-images.source.naturalHeight*view.scale)/2;draw();}
function fitBounds(bounds){if(!images.source)return;const [x0,y0,x1,y1]=bounds,width=Math.max(1,x1-x0),height=Math.max(1,y1-y0),padding=.9;view.scale=Math.min(canvas.width*padding/width,canvas.height*padding/height);view.minScale=Math.min(view.minScale,view.scale*.5);view.maxScale=Math.max(16,view.scale*8);view.x=(canvas.width-width*view.scale)/2-x0*view.scale;view.y=(canvas.height-height*view.scale)/2-y0*view.scale;draw();}
function drawArrow(item,index,draft=false){const [x1,y1]=item.current,[x2,y2]=item.target,dx=x2-x1,dy=y2-y1,length=Math.hypot(dx,dy),unit=1/view.scale;ctx.save();ctx.globalAlpha=1;ctx.strokeStyle=draft?'#ffffff':'#ffd43b';ctx.fillStyle=draft?'#ffffff':'#ffd43b';ctx.lineWidth=3*unit;ctx.setLineDash(draft?[7*unit,5*unit]:[]);if(length<3*unit){ctx.beginPath();ctx.arc(x1,y1,8*unit,0,Math.PI*2);ctx.stroke();ctx.beginPath();ctx.arc(x1,y1,2.5*unit,0,Math.PI*2);ctx.fill();}else{const ux=dx/length,uy=dy/length,head=13*unit;ctx.beginPath();ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);ctx.stroke();ctx.setLineDash([]);ctx.beginPath();ctx.moveTo(x2,y2);ctx.lineTo(x2-ux*head-uy*head*.55,y2-uy*head+ux*head*.55);ctx.lineTo(x2-ux*head+uy*head*.55,y2-uy*head-ux*head*.55);ctx.closePath();ctx.fill();}ctx.setLineDash([]);ctx.fillStyle='#62e6d8';ctx.beginPath();ctx.arc(x1,y1,4.5*unit,0,Math.PI*2);ctx.fill();ctx.strokeStyle='#ff9f43';ctx.lineWidth=2.5*unit;ctx.beginPath();ctx.arc(x2,y2,6*unit,0,Math.PI*2);ctx.stroke();if(index!=null){ctx.font=`bold ${13*unit}px ui-sans-serif,system-ui`;ctx.textAlign='center';ctx.textBaseline='middle';ctx.lineWidth=4*unit;ctx.strokeStyle='#090b0f';ctx.strokeText(String(index+1),x2+14*unit,y2-14*unit);ctx.fillStyle='#ffffff';ctx.fillText(String(index+1),x2+14*unit,y2-14*unit);}ctx.restore();}
function draw(){ctx.setTransform(1,0,0,1,0,0);ctx.fillStyle='#161a20';ctx.fillRect(0,0,canvas.width,canvas.height);ctx.setTransform(view.scale,0,0,view.scale,view.x,view.y);ctx.imageSmoothingEnabled=false;for(const [name,alpha] of [['source',opacity.source],['classification',opacity.classification],['county',opacity.county],['residual',opacity.residual],['state',opacity.state]]){if(images[name]&&alpha>0){ctx.globalAlpha=alpha;ctx.drawImage(images[name],0,0,payload.width,payload.height);}}ctx.globalAlpha=1;correction.items.forEach((item,index)=>drawArrow(item,index));if(correction.draft)drawArrow(correction.draft,null,true);document.querySelector('#zoom').textContent=`${Math.round(view.scale/view.minScale*50)}%`;}
function resize(){const rect=canvas.getBoundingClientRect();canvas.width=Math.max(1,Math.round(rect.width));canvas.height=Math.max(1,Math.round(rect.height));fit();}
new ResizeObserver(resize).observe(canvas);
canvas.addEventListener('wheel',event=>{event.preventDefault();const rect=canvas.getBoundingClientRect(),mx=event.clientX-rect.left,my=event.clientY-rect.top,ix=(mx-view.x)/view.scale,iy=(my-view.y)/view.scale,next=Math.max(view.minScale,Math.min(view.maxScale,view.scale*Math.exp(-event.deltaY*.0015)));view.x=mx-ix*next;view.y=my-iy*next;view.scale=next;draw();},{passive:false});
function imagePoint(event,clamp=false){const rect=canvas.getBoundingClientRect(),canvasX=(event.clientX-rect.left)*canvas.width/rect.width,canvasY=(event.clientY-rect.top)*canvas.height/rect.height;let x=(canvasX-view.x)/view.scale,y=(canvasY-view.y)/view.scale;if(clamp){x=Math.max(0,Math.min(images.source.naturalWidth-1,x));y=Math.max(0,Math.min(images.source.naturalHeight-1,y));}return [x,y];}
function insideImage(point){return point[0]>=0&&point[1]>=0&&point[0]<images.source.naturalWidth&&point[1]<images.source.naturalHeight;}
function updateCorrectionStatus(message){const count=correction.items.length,suffix=`${count} correction${count===1?'':'s'}`;document.querySelector('#correction-status').textContent=message||`${suffix}${correction.saved?' saved':' — unsaved'}`;document.querySelector('#undo-correction').disabled=count===0;document.querySelector('#clear-corrections').disabled=count===0;}
function finishPointer(event){if(canvas.hasPointerCapture(event.pointerId))canvas.releasePointerCapture(event.pointerId);canvas.classList.remove('dragging');view.dragging=false;if(correction.dragging){correction.draft.target=imagePoint(event,true);correction.items.push(correction.draft);correction.draft=null;correction.dragging=false;correction.saved=false;updateCorrectionStatus();draw();}}
canvas.addEventListener('pointerdown',event=>{if(correction.enabled){const point=imagePoint(event);if(!insideImage(point))return;correction.dragging=true;correction.draft={current:point,target:point};}else{view.dragging=true;view.lastX=event.clientX;view.lastY=event.clientY;canvas.classList.add('dragging');}canvas.setPointerCapture(event.pointerId);draw();});
canvas.addEventListener('pointermove',event=>{if(correction.dragging){correction.draft.target=imagePoint(event,true);draw();return;}if(!view.dragging)return;view.x+=event.clientX-view.lastX;view.y+=event.clientY-view.lastY;view.lastX=event.clientX;view.lastY=event.clientY;draw();});
canvas.addEventListener('pointerup',finishPointer);
canvas.addEventListener('pointercancel',event=>{correction.draft=null;correction.dragging=false;finishPointer(event);draw();});
canvas.addEventListener('dblclick',fit);document.querySelector('#reset-view').onclick=()=>{document.querySelector('#review-region').value='';fit();};
for(const [id,key] of [['source-opacity','source'],['classification-opacity','classification'],['state-opacity','state'],['county-opacity','county'],['residual-opacity','residual']]){const input=document.querySelector(`#${id}`);input.oninput=()=>{opacity[key]=Number(input.value)/100;input.nextElementSibling.textContent=`${input.value}%`;draw();};}
function setComparisonView(mode){comparisonView=mode;opacity.source=mode==='source'?1:0;opacity.classification=mode==='classification'?1:0;for(const [id,key] of [['source-opacity','source'],['classification-opacity','classification']]){const input=document.querySelector(`#${id}`),value=Math.round(opacity[key]*100);input.value=String(value);input.nextElementSibling.textContent=`${value}%`;}for(const [id,active] of [['show-source',mode==='source'],['show-classification',mode==='classification']]){const button=document.querySelector(`#${id}`);button.classList.toggle('active',active);button.setAttribute('aria-pressed',String(active));}draw();}
document.querySelector('#show-source').onclick=()=>setComparisonView('source');
document.querySelector('#show-classification').onclick=()=>setComparisonView('classification');
window.addEventListener('keydown',event=>{if(event.code!=='Space'||event.repeat)return;const tag=document.activeElement?.tagName;if(tag==='INPUT'||tag==='TEXTAREA'||tag==='SELECT'||tag==='BUTTON')return;event.preventDefault();setComparisonView(comparisonView==='source'?'classification':'source');});
document.querySelector('#review-region').onchange=event=>{if(!event.target.value){fit();return;}const region=(payload.review_regions||[]).find(item=>item.id===event.target.value);if(region)fitBounds(region.pixel_bounds);};
async function selectClassification(){const layer=payload.layers.find(item=>item.id===document.querySelector('#classification-layer').value);if(!layer)return;images.classification=await loadImage(layer.url);draw();}
function metricRows(values){return Object.entries(values).map(([label,value])=>`<dt>${label}</dt><dd>${value}</dd>`).join('');}
function setCorrectionMode(enabled){correction.enabled=enabled;const button=document.querySelector('#correction-mode');button.classList.toggle('active',enabled);button.textContent=enabled?'Finish adding arrows':'Add correction arrows';canvas.classList.toggle('correction-mode',enabled);document.querySelector('#interaction-help').textContent=enabled?'Drag reference line → matching source feature · wheel to zoom':'Drag to pan · wheel to zoom';}
document.querySelector('#correction-mode').onclick=()=>setCorrectionMode(!correction.enabled);
document.querySelector('#undo-correction').onclick=()=>{correction.items.pop();correction.saved=false;updateCorrectionStatus();draw();};
document.querySelector('#clear-corrections').onclick=()=>{correction.items=[];correction.saved=false;updateCorrectionStatus();draw();};
async function persistCorrections(){const corrections=correction.items.map(item=>({reference:item.current,source:item.target}));const response=await fetch('/corrections',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({corrections})});const result=await response.json();if(!response.ok)throw new Error(result.error||'Correction save failed');correction.saved=true;updateCorrectionStatus(`${result.corrections.summary.count} corrections saved`);return result;}
document.querySelector('#save-corrections').onclick=async()=>{const button=document.querySelector('#save-corrections');button.disabled=true;updateCorrectionStatus('Saving…');try{await persistCorrections();}catch(error){updateCorrectionStatus(error.message);}finally{button.disabled=false;}};
async function init(){const response=await fetch('/session.json');payload=await response.json();if(!response.ok)throw new Error(payload.error||'Could not load review');document.querySelector('#title').textContent=payload.title;document.querySelector('#review-status').textContent=payload.decision?.status?.replace('_',' ')||'not reviewed';setComparisonView(payload.default_view==='classification'?'classification':'source');const regionSelect=document.querySelector('#review-region');for(const region of payload.review_regions||[]){const option=document.createElement('option');option.value=region.id;option.textContent=region.label;regionSelect.appendChild(option);}if(payload.canonical_boundary)document.querySelector('#state-label').textContent='Canonical state / coast';if(payload.county_reference){document.querySelector('#county-label').textContent='County.png lines';document.querySelector('#county-diagnostic-title').textContent='County.png diagnostic';}if(payload.county_diagnostic_enabled===false){opacity.county=0;document.querySelector('#county-layer').hidden=true;document.querySelector('#residual-layer').hidden=true;document.querySelector('#residual-legend').hidden=true;document.querySelector('#county-diagnostic').hidden=true;}const select=document.querySelector('#classification-layer');select.innerHTML=payload.layers.map(layer=>`<option value="${layer.id}">${layer.label}</option>`).join('');select.onchange=selectClassification;const metrics=payload.alignment.metrics||{},solid=metrics.solid_mask_alignment_after||{},solidForward=solid.source_to_target||{};document.querySelector('#alignment-metrics').innerHTML=metricRows({'Mode':payload.alignment.mode,'CRS':payload.alignment.projection_crs,'Boundary median':solidForward.median_px!=null?`${solidForward.median_px.toFixed(2)} px`:(metrics.control_point_median_px!=null?`${metrics.control_point_median_px.toFixed(2)} px`:'—'),'Boundary P90':solidForward.p90_px!=null?`${solidForward.p90_px.toFixed(2)} px`:(metrics.review_correction_leave_one_out_p90_px!=null?`${metrics.review_correction_leave_one_out_p90_px.toFixed(2)} px`:'—'),'Within 3 px':solidForward.within_3px_fraction!=null?`${(solidForward.within_3px_fraction*100).toFixed(1)}%`:'—','Control RMS':metrics.control_point_rms_px!=null?`${metrics.control_point_rms_px.toFixed(2)} px`:'—','Control max':metrics.control_point_max_px!=null?`${metrics.control_point_max_px.toFixed(2)} px`:'—'});const clip=payload.canonical_clip||{};document.querySelector('#clip-metrics').innerHTML=metricRows({'Valid components':clip.component_count??'—','Source pixels removed':clip.source_pixel_count_removed??'—','Colored outside after clip':clip.source_pixels_outside_after_clip??'—'});const residual=payload.county_residual||{};document.querySelector('#county-metrics').innerHTML=metricRows({'Samples':residual.county_line_sample_count??'—','Median edge distance':residual.median_nearest_source_edge_px!=null?`${residual.median_nearest_source_edge_px.toFixed(2)} px`:'—','P90 edge distance':residual.p90_nearest_source_edge_px!=null?`${residual.p90_nearest_source_edge_px.toFixed(2)} px`:'—','Within 3 px':residual.within_3px_fraction!=null?`${(residual.within_3px_fraction*100).toFixed(1)}%`:'—','Within 8 px':residual.within_8px_fraction!=null?`${(residual.within_8px_fraction*100).toFixed(1)}%`:'—'});document.querySelector('#county-warning').textContent=residual.warning||'County evidence is diagnostic only.';if(payload.decision){document.querySelector('#notes').value=payload.decision.notes||'';document.querySelector('#decision-status').textContent=`Saved ${payload.decision.reviewed_at}`;}if(payload.corrections){const saved=payload.corrections,referenceFirst=saved.direction==='authoritative_reference_to_current_warped_source';correction.items=saved.corrections.map(item=>referenceFirst?{current:[item.reference.pixel.x,item.reference.pixel.y],target:[item.source.pixel.x,item.source.pixel.y]}:{current:[item.current.pixel.x,item.current.pixel.y],target:[item.target.pixel.x,item.target.pixel.y]});correction.saved=!payload.corrections_stale;updateCorrectionStatus(payload.corrections_stale?'Saved arrows target an older extraction; recheck and save them':`${correction.items.length} corrections loaded`);}else{correction.saved=false;updateCorrectionStatus('No corrections');}[images.source,images.state,images.county,images.residual]=await Promise.all([loadImage(payload.assets.source),loadImage(payload.assets.state_overlay),loadImage(payload.assets.county_overlay),loadImage(payload.assets.county_residual)]);await selectClassification();fit();}
for(const button of document.querySelectorAll('.decision-buttons button'))button.onclick=async()=>{const status=button.dataset.status,notes=document.querySelector('#notes').value;document.querySelector('#decision-status').textContent='Saving…';try{if(correction.items.length&&!correction.saved)await persistCorrections();const response=await fetch('/decision',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({status,notes})});const result=await response.json();if(!response.ok)throw new Error(result.error||'Save failed');document.querySelector('#review-status').textContent=status.replace('_',' ');document.querySelector('#decision-status').textContent=`Saved ${result.decision.reviewed_at}`;}catch(error){document.querySelector('#decision-status').textContent=error.message;}};
init().catch(error=>{document.querySelector('#title').textContent=error.message;});
</script>
</body>
</html>"""
