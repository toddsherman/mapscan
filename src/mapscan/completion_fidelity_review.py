"""Read-only review of categorical completion evidence and its information ceiling."""

from __future__ import annotations

import hashlib
import json
import mimetypes
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Sequence
from urllib.parse import unquote, urlparse

import numpy as np
from PIL import Image

from .canonical_boundary import load_active_canonical_border


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text())


def _resolve(value: object) -> Path:
    return Path(str(value)).expanduser().resolve()


def _verified(path: Path, expected: object, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    if expected and _sha256(path) != str(expected):
        raise ValueError(f"Stale {label}: {path}")
    return path


def _materialized_layer(root: Path) -> tuple[Dict[str, object], Dict[str, object]]:
    manifest_path = root / "materialization.json"
    manifest = _load(manifest_path)
    layers = manifest.get("layers", [])
    if not isinstance(layers, list) or len(layers) != 1:
        raise ValueError("Completion fidelity review requires one categorical layer")
    return manifest, layers[0]


def _artifact(root: Path, record: Dict[str, object], label: str) -> Path:
    return _verified(root / str(record["path"]), record.get("sha256"), label)


def _preview(values: np.ndarray, categories: Sequence[Dict[str, object]]) -> np.ndarray:
    rgba = np.zeros((*values.shape, 4), dtype=np.uint8)
    for class_id, category in enumerate(categories, 1):
        color = category.get("display_rgb", category.get("legend_rgb", [255, 0, 255]))
        if isinstance(color, list) and color and isinstance(color[0], list):
            color = color[0]
        selected = values == class_id
        rgba[selected, :3] = np.asarray(color[:3], dtype=np.uint8)
        rgba[selected, 3] = 235
    return rgba


def _mask_overlay(mask: np.ndarray, color: Sequence[int], alpha: int = 210) -> np.ndarray:
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    rgba[mask, :3] = np.asarray(color[:3], dtype=np.uint8)
    rgba[mask, 3] = alpha
    return rgba


def compose_conservative_surface(
    approved: np.ndarray,
    pass_one: np.ndarray,
    pass_two_values: np.ndarray,
    pass_two_mask: np.ndarray,
    allowed_approved_clear_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Compose review-only completion tiers without changing approved evidence."""

    if not (
        approved.shape == pass_one.shape == pass_two_values.shape == pass_two_mask.shape
    ):
        raise ValueError("Completion fidelity rasters use different grids")
    changed_approved = (approved > 0) & (pass_one != approved)
    allowed_clear = (
        np.zeros(approved.shape, dtype=bool)
        if allowed_approved_clear_mask is None
        else allowed_approved_clear_mask.astype(bool)
    )
    if allowed_clear.shape != approved.shape:
        raise ValueError("Approved-clear mask uses a different grid")
    invalid_approved_change = changed_approved & ~(allowed_clear & (pass_one == 0))
    if np.any(invalid_approved_change):
        raise ValueError("Conservative pass one changes approved evidence")
    output = pass_one.copy().astype(np.uint8)
    accepted = pass_two_mask.astype(bool)
    if np.any(accepted & (approved > 0)):
        raise ValueError("Stability completion overlaps approved evidence")
    output[accepted] = pass_two_values[accepted]
    return output


def build_completion_fidelity_review(
    config_path: Path,
) -> tuple[Dict[str, object], Dict[str, Path]]:
    """Verify all evidence and generate deterministic read-only review assets."""

    config_path = config_path.resolve()
    config = _load(config_path)
    if int(config.get("schema_version", 0)) != 1:
        raise ValueError("Unsupported completion fidelity review config")
    extraction_run = _resolve(config["extraction_run"])
    approved_root = _resolve(config["approved_materialization"])
    aggressive_root = _resolve(config["aggressive_materialization"])
    conservative_path = _resolve(config["conservative_audit"])
    stability_path = _resolve(config["stability_audit"])
    neighbor_path = (
        _resolve(config["neighbor_completion"])
        if config.get("neighbor_completion")
        else None
    )
    component_path = _resolve(config["component_audit"])
    alignment_path = _resolve(config["alignment_audit"])
    canonical_boundary_path = _resolve(config["canonical_boundary_audit"])
    active_canonical_pointer_path = (
        _resolve(config["active_canonical_boundary"])
        if config.get("active_canonical_boundary")
        else None
    )
    boundary_candidate_path = (
        _resolve(config["boundary_display_candidate"])
        if config.get("boundary_display_candidate")
        else None
    )
    output_dir = _resolve(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    extraction_path = extraction_run / "extraction.json"
    extraction = _load(extraction_path)
    plan_path = extraction_run / "plan.snapshot.json"
    plan = _load(plan_path)
    categories = plan["layers"][0]["categories"]
    grid = extraction["alignment"]["inspection"]["grid"]
    shape = (int(grid["height"]), int(grid["width"]))

    approved_manifest, approved_layer = _materialized_layer(approved_root)
    aggressive_manifest, aggressive_layer = _materialized_layer(aggressive_root)
    approved_class_path = _artifact(
        approved_root, approved_layer["artifacts"]["class_id"], "approved class raster"
    )
    approved_preview_path = _artifact(
        approved_root, approved_layer["artifacts"]["preview"], "approved preview"
    )
    aggressive_class_path = _artifact(
        aggressive_root,
        aggressive_layer["artifacts"]["class_id"],
        "aggressive class raster",
    )
    aggressive_preview_path = _artifact(
        aggressive_root, aggressive_layer["artifacts"]["preview"], "aggressive preview"
    )

    conservative = _load(conservative_path)
    stability = _load(stability_path)
    neighbor = _load(neighbor_path) if neighbor_path is not None else None
    component = _load(component_path)
    alignment = _load(alignment_path)
    canonical_boundary = _load(canonical_boundary_path)
    boundary_candidate = (
        _load(boundary_candidate_path) if boundary_candidate_path is not None else None
    )
    active_canonical_manifest_path = None
    active_canonical = None
    active_canonical_pointer = None
    if active_canonical_pointer_path is not None:
        (
            active_canonical_manifest_path,
            active_canonical,
            active_canonical_pointer,
        ) = load_active_canonical_border(active_canonical_pointer_path)
    if conservative.get("status") != "needs_visual_review":
        raise ValueError("Conservative completion audit is not reviewable")
    if stability.get("status") != "experimental_confidence_surface_only":
        raise ValueError("Stability surface is not explicitly audit-only")
    if neighbor is not None:
        if neighbor.get("status") != "needs_visual_review":
            raise ValueError("Neighbor completion is not reviewable")
        if neighbor.get("publication_allowed") is not False:
            raise ValueError("Neighbor completion must remain unpublished before review")
    if component.get("status") != "pass":
        raise ValueError("Boundary-component audit has not passed")
    if alignment.get("status") != "pass":
        raise ValueError("Alignment application audit has not passed")
    if alignment.get("application_mode") != "validated_noop":
        raise ValueError("This review config expects a validated no-op alignment")
    if canonical_boundary.get("status") != "pass":
        raise ValueError("Canonical boundary raster has not passed")
    if canonical_boundary.get("canonical_boundary_id") != "california-mainland-hybrid-v1":
        raise ValueError("Completion review does not use the approved canonical boundary")
    if canonical_boundary.get("target_extraction", {}).get("sha256") != _sha256(
        extraction_path
    ):
        raise ValueError("Canonical boundary raster does not bind this extraction")
    if boundary_candidate is not None:
        if boundary_candidate.get("status") != "needs_author_review":
            raise ValueError("County-detail boundary candidate is not reviewable")
        if (
            boundary_candidate.get("base_canonical_boundary_id")
            != canonical_boundary.get("canonical_boundary_id")
        ):
            raise ValueError("Boundary candidate does not derive from the canonical baseline")
        topology = boundary_candidate.get("topology", {})
        if (
            topology.get("mainland_component_count") != 1
            or topology.get("offshore_island_component_count") != 4
            or topology.get("combined_component_count") != 5
            or topology.get("county_coast_dropped_pixel_count") != 0
            or topology.get("san_francisco_bay", {}).get("exact") is not True
        ):
            raise ValueError("County-detail boundary topology has not passed")
        candidate_grid = boundary_candidate.get("grid", {})
        for field in ("crs", "bounds"):
            if candidate_grid.get(field) != grid.get(field):
                raise ValueError(f"County-detail boundary grid differs at {field}")
        if (
            int(candidate_grid.get("width", 0)) < int(grid["width"])
            or int(candidate_grid.get("height", 0)) < int(grid["height"])
        ):
            raise ValueError("County-detail boundary must retain the higher-resolution grid")
    if active_canonical is not None:
        topology = active_canonical.get("topology", {})
        if (
            topology.get("mainland_component_count") != 1
            or topology.get("offshore_island_component_count") != 4
            or topology.get("combined_component_count") != 5
            or topology.get("county_coast_dropped_pixel_count") != 0
            or topology.get("san_francisco_bay", {}).get("exact") is not True
        ):
            raise ValueError("Active canonical county-detail topology has not passed")
        active_grid = active_canonical.get("source_grid", {})
        for field in ("crs", "bounds"):
            if active_grid.get(field) != grid.get(field):
                raise ValueError(f"Active canonical boundary differs at {field}")

    approved_class_sha = _sha256(approved_class_path)
    if conservative["inputs"]["approved"]["sha256"] != approved_class_sha:
        raise ValueError("Conservative completion does not bind the approved class raster")
    if stability["inputs"]["approved_seed_evidence"]["sha256"] != approved_class_sha:
        raise ValueError("Stability completion does not bind the approved seed evidence")
    aggressive_class_sha = _sha256(aggressive_class_path)
    if conservative["inputs"]["aggressive_fixed_point"]["sha256"] != aggressive_class_sha:
        raise ValueError("Aggressive comparison does not match the conservative audit")
    if stability["inputs"]["aggressive_fixed_point"]["sha256"] != aggressive_class_sha:
        raise ValueError("Aggressive comparison does not match the stability audit")
    if component["extraction"]["sha256"] != _sha256(extraction_path):
        raise ValueError("Boundary-component audit does not bind this extraction")
    if neighbor is not None and neighbor_path is not None:
        if neighbor["inputs"]["extraction"]["sha256"] != _sha256(extraction_path):
            raise ValueError("Neighbor completion does not bind this extraction")
        if (
            neighbor["inputs"]["remaining_unknown"]["sha256"]
            != stability["artifacts"]["remaining_mask"]["sha256"]
        ):
            raise ValueError("Neighbor completion does not bind the remaining unknown mask")

    conservative_root = conservative_path.parent
    stability_root = stability_path.parent
    pass_one_path = _artifact(
        conservative_root,
        conservative["artifacts"]["conservative_class_id"],
        "conservative pass-one class raster",
    )
    pass_one_mask_path = _artifact(
        conservative_root,
        conservative["artifacts"]["accepted_mask"],
        "conservative pass-one mask",
    )
    approved_clear_path = _artifact(
        conservative_root,
        conservative["artifacts"]["boundary_clipped_mask"],
        "authoritative boundary-clipped approved mask",
    )
    pass_two_values_path = _artifact(
        stability_root,
        stability["artifacts"]["accepted_values"],
        "stability accepted values",
    )
    pass_two_mask_path = _artifact(
        stability_root,
        stability["artifacts"]["accepted_mask"],
        "stability accepted mask",
    )
    boundary_review_path = _artifact(
        stability_root,
        stability["artifacts"]["boundary_proximal_mask"],
        "boundary-proximal review mask",
    )
    unresolved_path = _artifact(
        stability_root,
        stability["artifacts"]["remaining_mask"],
        "remaining unresolved mask",
    )
    observed_path = _artifact(
        approved_root,
        approved_layer["artifacts"]["observed_mask"],
        "approved observed mask",
    )
    manual_path = _artifact(
        approved_root,
        approved_layer["artifacts"]["manual_mask"],
        "approved manual mask",
    )

    approved_values = np.asarray(Image.open(approved_class_path), dtype=np.uint8)
    pass_one_values = np.asarray(Image.open(pass_one_path), dtype=np.uint8)
    pass_two_values = np.asarray(Image.open(pass_two_values_path), dtype=np.uint8)
    pass_two_mask = np.asarray(Image.open(pass_two_mask_path)) > 0
    approved_clear_mask = np.asarray(Image.open(approved_clear_path)) > 0
    if approved_values.shape != shape:
        raise ValueError("Approved materialization and extraction grids differ")
    combined = compose_conservative_surface(
        approved_values,
        pass_one_values,
        pass_two_values,
        pass_two_mask,
        approved_clear_mask,
    )
    expected_remaining = int(stability["metrics"]["remaining_unresolved_pixel_count"])
    unresolved = np.asarray(Image.open(unresolved_path)) > 0
    if int(np.count_nonzero(unresolved)) != expected_remaining:
        raise ValueError("Unresolved mask count does not match its audit")

    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    generated: Dict[str, Path] = {}

    def save(name: str, values: np.ndarray) -> Path:
        path = assets_dir / name
        Image.fromarray(values).save(path, optimize=True)
        generated[name] = path
        return path

    conservative_preview = save("conservative-preview.png", _preview(combined, categories))
    pass_one_mask = np.asarray(Image.open(pass_one_mask_path)) > 0
    observed_mask = np.asarray(Image.open(observed_path)) > 0
    manual_mask = np.asarray(Image.open(manual_path)) > 0
    boundary_review = np.asarray(Image.open(boundary_review_path)) > 0
    save("observed-mask.png", _mask_overlay(observed_mask, [70, 150, 255], 105))
    save("manual-mask.png", _mask_overlay(manual_mask, [20, 235, 255], 210))
    save("pass-one-mask.png", _mask_overlay(pass_one_mask, [90, 255, 130], 230))
    save("pass-two-mask.png", _mask_overlay(pass_two_mask, [80, 180, 255], 245))
    save("boundary-review-mask.png", _mask_overlay(boundary_review, [255, 160, 40], 225))
    save("unresolved-mask.png", _mask_overlay(unresolved, [255, 20, 190], 205))

    neighbor_preview_path = None
    if neighbor is not None and neighbor_path is not None:
        neighbor_root = neighbor_path.parent

        def neighbor_artifact(name: str, label: str) -> Path:
            return _artifact(neighbor_root, neighbor["artifacts"][name], label)

        neighbor_preview_path = neighbor_artifact(
            "web-mercator-preview-neighbor-completed.png",
            "neighbor-completed preview",
        )
        neighbor_assumption = np.asarray(
            Image.open(
                neighbor_artifact(
                    "web-mercator-neighbor-assumption-mask.png",
                    "neighbor-assumption mask",
                )
            )
        ) > 0
        outside_unknown = np.asarray(
            Image.open(
                neighbor_artifact(
                    "web-mercator-unknown-outside-removed-mask.png",
                    "outside unknown removal mask",
                )
            )
        ) > 0
        existing_outside = np.asarray(
            Image.open(
                neighbor_artifact(
                    "web-mercator-existing-outside-removed-mask.png",
                    "outside prior-evidence removal mask",
                )
            )
        ) > 0
        manual_neighbor = np.asarray(
            Image.open(
                neighbor_artifact(
                    "web-mercator-manual-neighbor-mask.png",
                    "authored-stamp neighbor mask",
                )
            )
        ) > 0
        for label, mask in (
            ("neighbor assumption", neighbor_assumption),
            ("outside unknown", outside_unknown),
            ("outside prior evidence", existing_outside),
            ("authored-stamp neighbor", manual_neighbor),
        ):
            if mask.shape != shape:
                raise ValueError(f"{label.title()} raster uses a different grid")
        save("neighbor-assumption-mask.png", _mask_overlay(neighbor_assumption, [180, 85, 255], 195))
        save("outside-unknown-mask.png", _mask_overlay(outside_unknown, [255, 65, 45], 235))
        save("outside-existing-mask.png", _mask_overlay(existing_outside, [255, 215, 60], 245))
        save("manual-neighbor-mask.png", _mask_overlay(manual_neighbor, [0, 245, 225], 205))

    if active_canonical is not None and active_canonical_manifest_path is not None:
        mainland_record = active_canonical["artifacts"]["mainland"]
        canonical_border_path = _verified(
            active_canonical_manifest_path.parent / str(mainland_record["path"]),
            mainland_record["sha256"],
            "active canonical mainland border",
        )
        island_record = active_canonical["artifacts"]["islands"]
        island_border_path = _verified(
            active_canonical_manifest_path.parent / str(island_record["path"]),
            island_record["sha256"],
            "active canonical island borders",
        )
        for label, path in (
            ("mainland", canonical_border_path),
            ("islands", island_border_path),
        ):
            if np.asarray(Image.open(path).convert("RGBA")).shape[:2] != (
                int(active_canonical["source_grid"]["height"]),
                int(active_canonical["source_grid"]["width"]),
            ):
                raise ValueError(f"Active canonical {label} overlay uses a different grid")
        boundary_label = "Canonical county.png coast + 4 islands (lime)"
    elif boundary_candidate is not None and boundary_candidate_path is not None:
        mainland_record = boundary_candidate["artifacts"]["mainland"]
        canonical_border_path = _verified(
            boundary_candidate_path.parent / str(mainland_record["path"]),
            mainland_record["sha256"],
            "county-detail mainland border",
        )
        island_record = boundary_candidate["artifacts"]["islands"]
        island_border_path = _verified(
            boundary_candidate_path.parent / str(island_record["path"]),
            island_record["sha256"],
            "county-detail island borders",
        )
        for label, path in (
            ("mainland", canonical_border_path),
            ("islands", island_border_path),
        ):
            if np.asarray(Image.open(path).convert("RGBA")).shape[:2] != (
                int(boundary_candidate["grid"]["height"]),
                int(boundary_candidate["grid"]["width"]),
            ):
                raise ValueError(f"County-detail {label} overlay uses a different grid")
        boundary_label = "county.png coast + 4 islands (lime)"
    else:
        canonical_border_record = canonical_boundary["artifacts"]["border_vector"]
        canonical_border_path = _verified(
            canonical_boundary_path.parent / str(canonical_border_record["path"]),
            canonical_border_record["sha256"],
            "canonical mainland vector border",
        )
        island_record = component["artifacts"]["island_border"]
        island_border_path = _verified(
            component_path.parent / str(island_record["path"]),
            island_record["sha256"],
            "selected island borders",
        )
        island_border = np.asarray(Image.open(island_border_path).convert("RGBA"))[..., 3] > 0
        if island_border.shape != shape:
            raise ValueError("Boundary overlays and extraction grid differ")
        boundary_label = "Publication boundary (lime)"

    source_path = extraction_run / "web-mercator-source.jpg"
    _verified(source_path, None, "aligned source")
    assets: Dict[str, Path] = {
        "source": source_path,
        "approved": approved_preview_path,
        "aggressive": aggressive_preview_path,
        "conservative": conservative_preview,
        "observed": generated["observed-mask.png"],
        "manual": generated["manual-mask.png"],
        "pass_one": generated["pass-one-mask.png"],
        "pass_two": generated["pass-two-mask.png"],
        "boundary_review": generated["boundary-review-mask.png"],
        "unresolved": generated["unresolved-mask.png"],
        "boundary": canonical_border_path,
        "boundary_islands": island_border_path,
    }
    if neighbor_preview_path is not None:
        assets.update(
            {
                "neighbor": neighbor_preview_path,
                "neighbor_assumption": generated["neighbor-assumption-mask.png"],
                "outside_unknown": generated["outside-unknown-mask.png"],
                "outside_existing": generated["outside-existing-mask.png"],
                "manual_neighbor": generated["manual-neighbor-mask.png"],
            }
        )
    asset_urls = {key: f"/asset/{key}" for key in assets}
    metrics = {
        "approved_nonzero": int(np.count_nonzero(approved_values)),
        "initial_unresolved": int(conservative["metrics"]["initial_unresolved_pixel_count"]),
        "pass_one_accepted": int(conservative["metrics"]["accepted_completion_pixel_count"]),
        "pass_two_accepted": int(stability["metrics"]["accepted_pixel_count"]),
        "boundary_review": int(stability["metrics"]["boundary_proximal_review_pixel_count"]),
        "remaining_unresolved": expected_remaining,
        "selected_islands": int(component["selected_island_component_count"]),
        "boundary_islands": (
            int(active_canonical["topology"]["offshore_island_component_count"])
            if active_canonical is not None
            else int(boundary_candidate["topology"]["offshore_island_component_count"])
            if boundary_candidate is not None
            else int(component["selected_island_component_count"])
        ),
        "alignment_mode": alignment["application_mode"],
    }
    if neighbor is not None:
        metrics.update(
            {
                "neighbor_inside_filled": int(neighbor["metrics"]["unknown_inside_filled"]),
                "neighbor_outside_removed": int(neighbor["metrics"]["unknown_outside_removed"]),
                "existing_outside_removed": int(
                    neighbor["metrics"]["existing_classified_outside_removed"]
                ),
                "neighbor_remaining_inside": int(
                    neighbor["metrics"]["remaining_unknown_inside"]
                ),
                "manual_neighbor_pixels": int(
                    neighbor["metrics"]["neighbor_fill"]["manual_neighbor_pixel_count"]
                ),
                "manual_changed_choice_pixels": int(
                    neighbor["metrics"]["neighbor_fill"][
                        "manual_weight_changed_choice_pixel_count"
                    ]
                ),
            }
        )
    payload = {
        "schema_version": 1,
        "status": (
            "needs_visual_review"
            if neighbor is not None
            else "diagnostic_only_not_approvable"
        ),
        "title": str(config.get("title", "Categorical completion fidelity review")),
        "instruction": (
            "Compare the source and completion policies. Neighbor-filled clips exterior "
            "unknowns to transparent and assigns every retained interior unknown from "
            "nearby class evidence, with authored stamps participating at double weight. "
            "Purple marks assumptions; red marks unknowns removed outside the border."
            if neighbor is not None
            else "Compare the source and the two completion policies. The conservative view "
            "contains only approved pixels plus 560 repeatable local repairs. Magenta "
            "pixels remain unknown. The aggressive view fills every hole but is a no-go "
            "because it invents rare classes far from evidence."
        ),
        "grid": grid,
        "assets": asset_urls,
        "boundary_label": boundary_label,
        "metrics": metrics,
        "categories": [
            {
                "class_id": index,
                "label": category["label"],
                "color": category.get("display_rgb", category.get("legend_rgb")),
            }
            for index, category in enumerate(categories, 1)
        ],
        "evidence": {
            "config": {"path": str(config_path), "sha256": _sha256(config_path)},
            "extraction": {"path": str(extraction_path), "sha256": _sha256(extraction_path)},
            "approved_materialization": {
                "path": str(approved_root / "materialization.json"),
                "sha256": _sha256(approved_root / "materialization.json"),
            },
            "aggressive_materialization": {
                "path": str(aggressive_root / "materialization.json"),
                "sha256": _sha256(aggressive_root / "materialization.json"),
            },
            "conservative_audit": {"path": str(conservative_path), "sha256": _sha256(conservative_path)},
            "stability_audit": {"path": str(stability_path), "sha256": _sha256(stability_path)},
            "neighbor_completion": (
                {"path": str(neighbor_path), "sha256": _sha256(neighbor_path)}
                if neighbor_path is not None
                else None
            ),
            "component_audit": {"path": str(component_path), "sha256": _sha256(component_path)},
            "alignment_audit": {"path": str(alignment_path), "sha256": _sha256(alignment_path)},
            "canonical_boundary_audit": {
                "path": str(canonical_boundary_path),
                "sha256": _sha256(canonical_boundary_path),
            },
            "boundary_display_candidate": (
                {
                    "path": str(boundary_candidate_path),
                    "sha256": _sha256(boundary_candidate_path),
                }
                if boundary_candidate_path is not None
                else None
            ),
            "active_canonical_boundary": (
                {
                    "pointer": {
                        "path": str(active_canonical_pointer_path),
                        "sha256": _sha256(active_canonical_pointer_path),
                    },
                    "manifest": {
                        "path": str(active_canonical_manifest_path),
                        "sha256": _sha256(active_canonical_manifest_path),
                    },
                    "canonical_boundary_id": active_canonical.get(
                        "canonical_boundary_id"
                    ),
                }
                if active_canonical is not None
                and active_canonical_manifest_path is not None
                and active_canonical_pointer_path is not None
                else None
            ),
        },
    }
    session_path = output_dir / "review-session.json"
    session_path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload, assets


def serve_completion_fidelity_review(
    config_path: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8788,
    open_browser: bool = True,
) -> None:
    payload, assets = build_completion_fidelity_review(config_path)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            print(f"[completion-fidelity-review] {self.address_string()} {format % args}")

        def do_GET(self) -> None:  # noqa: N802
            route = unquote(urlparse(self.path).path)
            if route in {"/", "/index.html"}:
                body = _HTML.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
            elif route == "/session.json":
                body = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
            elif route.startswith("/asset/") and route.removeprefix("/asset/") in assets:
                path = assets[route.removeprefix("/asset/")]
                body = path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
            else:
                self.send_error(404)
                return
            self.send_header("Cache-Control", "no-store, max-age=0")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    url = f"http://{host}:{port}/"
    print(f"Completion fidelity review: {url}", flush=True)
    if open_browser:
        webbrowser.open(url)
    ThreadingHTTPServer((host, port), Handler).serve_forever()


_HTML = r"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Completion fidelity review</title><style>
:root{color-scheme:dark;font-family:Inter,ui-sans-serif,system-ui,sans-serif}*{box-sizing:border-box}body{margin:0;background:#080808;color:#f5f5f5;height:100vh;overflow:hidden}.app{display:grid;grid-template-columns:390px 1fr;height:100vh}.panel{padding:20px;border-right:1px solid #292929;background:#111;overflow:auto}.panel h1{font-size:20px;margin:0 0 7px}.badge{display:inline-block;margin:6px 0 12px;padding:4px 8px;border-radius:999px;background:#442318;color:#ffbe91;font-size:11px}.sub{font-size:12px;color:#aaa;line-height:1.45}.modes{display:grid;grid-template-columns:1fr 1fr;gap:7px;margin:16px 0 10px}button{border:1px solid #3b3b3b;background:#202020;color:#eee;padding:9px;border-radius:7px;cursor:pointer}button.active{background:#fff;color:#111;border-color:#fff}.desc{font-size:11px;color:#aaa;min-height:34px}.row{display:flex;align-items:center;gap:9px;font-size:12px;margin:8px 0}.row input[type=range]{flex:1}.metrics{border-top:1px solid #292929;margin-top:14px;padding-top:10px}.metric{display:flex;justify-content:space-between;gap:14px;padding:3px 0;font-size:12px}.metric span:first-child{color:#aaa}.legend{display:grid;grid-template-columns:1fr 1fr;gap:4px 10px;border-top:1px solid #292929;margin-top:13px;padding-top:10px}.legend-item{display:flex;align-items:center;gap:6px;font-size:10px;color:#bbb}.swatch{width:11px;height:11px;border-radius:2px}.notice{margin-top:14px;border:1px solid #4d3020;background:#25170f;color:#ffc99e;padding:10px;border-radius:7px;font-size:11px;line-height:1.45}.viewport{position:relative;overflow:hidden;background:#030303;cursor:grab}.viewport.dragging{cursor:grabbing}.stage{position:absolute;left:0;top:0;transform-origin:0 0}.stage img{position:absolute;inset:0;width:100%;height:100%;object-fit:fill;image-rendering:pixelated;pointer-events:none}#boundary{image-rendering:auto}.toolbar{position:absolute;right:14px;top:14px;display:flex;gap:6px;z-index:10}.toolbar button{background:#111d}
</style></head><body><div class="app"><aside class="panel"><h1 id="title">Completion fidelity review</h1><div class="badge">review candidate · not published</div><div class="sub" id="instruction"></div><div class="modes"><button data-mode="source">Source (S)</button><button data-mode="approved">Approved (A)</button><button data-mode="conservative">Conservative (C)</button><button data-mode="neighbor" class="active">Neighbor-filled (F)</button><button data-mode="aggressive">Aggressive (G)</button></div><div class="desc" id="description"></div><div id="toggles"></div><div class="row"><span>Overlay opacity</span><input id="opacity" type="range" min="0" max="1" value="0.78" step="0.01"></div><div class="metrics" id="metrics"></div><div class="legend" id="legend"></div><div class="notice">Neighbor-filled is a new assumption layer: observed pixels and authored stamps remain distinct evidence. Exterior removals and every interior assumption are separately inspectable before anything can be published.</div></aside><main class="viewport" id="viewport"><div class="toolbar"><button id="fit">Fit</button><button id="one">100%</button></div><div class="stage" id="stage"><img id="base"><img id="observed"><img id="manual"><img id="pass_one"><img id="pass_two"><img id="boundary_review"><img id="unresolved"><img id="neighbor_assumption"><img id="outside_unknown"><img id="outside_existing"><img id="manual_neighbor"><img id="boundary"><img id="boundary_islands"></div></main></div>
<script>
let payload,mode='neighbor',scale=1,tx=0,ty=0,drag=null;const $=s=>document.querySelector(s),stage=$('#stage'),viewport=$('#viewport'),base=$('#base');const overlayIds=['observed','manual','pass_one','pass_two','boundary_review','unresolved','neighbor_assumption','outside_unknown','outside_existing','manual_neighbor','boundary'];const labels={observed:'Observed pixels (blue)',manual:'118 authored stamps (cyan)',pass_one:'Local repairs · 384 (green)',pass_two:'Stable interior · 176 (blue)',boundary_review:'Boundary-risk proposals · 1,375 (orange)',unresolved:'Prior unknown · 19,616 (magenta)',neighbor_assumption:'Interior neighbor assumptions (purple)',outside_unknown:'Exterior unknowns removed (red)',outside_existing:'Prior classified pixels clipped outside (yellow)',manual_neighbor:'Assumptions with authored-stamp neighbors (aqua)',boundary:'Canonical state border (lime)'};const descriptions={source:'Aligned source image. Use this to judge what evidence actually exists.',approved:'Observed classification plus the 118 saved stamp edits; black/transparent gaps remain.',conservative:'Approved evidence plus 560 repeatable local repairs. Unknown areas remain transparent.',neighbor:'Every retained interior unknown is inferred from neighboring classes; exterior pixels are transparent.',aggressive:'Earlier unconstrained nearest-class counterexample for comparison.'};
function setTransform(){stage.style.transform=`translate(${tx}px,${ty}px) scale(${scale})`}function fit(){if(!payload)return;scale=Math.min(viewport.clientWidth/payload.grid.width,viewport.clientHeight/payload.grid.height);tx=(viewport.clientWidth-payload.grid.width*scale)/2;ty=(viewport.clientHeight-payload.grid.height*scale)/2;setTransform()}function one(){scale=1;tx=(viewport.clientWidth-payload.grid.width)/2;ty=(viewport.clientHeight-payload.grid.height)/2;setTransform()}
function setMode(next){mode=next;base.src=payload.assets[next];document.querySelectorAll('[data-mode]').forEach(b=>b.classList.toggle('active',b.dataset.mode===next));$('#description').textContent=descriptions[next]}
function updateOverlays(){const opacity=Number($('#opacity').value);overlayIds.forEach(id=>{const img=$('#'+id),toggle=$('#toggle-'+id);img.style.display=toggle&&toggle.checked?'block':'none';img.style.opacity=opacity});const boundaryToggle=$('#toggle-boundary');$('#boundary_islands').style.display=boundaryToggle&&boundaryToggle.checked?'block':'none';$('#boundary_islands').style.opacity=opacity}
async function init(){payload=await(await fetch('/session.json')).json();if(!payload.assets.neighbor){document.querySelector('[data-mode="neighbor"]').remove();mode='conservative'}labels.boundary=payload.boundary_label||labels.boundary;$('#title').textContent=payload.title;$('#instruction').textContent=payload.instruction;stage.style.width=payload.grid.width+'px';stage.style.height=payload.grid.height+'px';base.src=payload.assets[mode];overlayIds.forEach(id=>{if(payload.assets[id])$('#'+id).src=payload.assets[id]});$('#boundary_islands').src=payload.assets.boundary_islands;const defaults={observed:false,manual:false,pass_one:false,pass_two:false,boundary_review:false,unresolved:false,neighbor_assumption:true,outside_unknown:false,outside_existing:false,manual_neighbor:false,boundary:true};$('#toggles').innerHTML=overlayIds.filter(id=>payload.assets[id]).map(id=>`<label class="row"><input id="toggle-${id}" type="checkbox" ${defaults[id]?'checked':''}>${labels[id]}</label>`).join('');document.querySelectorAll('#toggles input').forEach(i=>i.onchange=updateOverlays);$('#opacity').oninput=updateOverlays;const m=payload.metrics,items=[['Validated alignment',m.alignment_mode.replace('_',' ')],['Approved classified',m.approved_nonzero.toLocaleString()],['Prior still unknown',m.remaining_unresolved.toLocaleString()],['Interior assumptions',(m.neighbor_inside_filled??0).toLocaleString()],['Exterior unknowns removed',(m.neighbor_outside_removed??0).toLocaleString()],['Prior exterior pixels clipped',(m.existing_outside_removed??0).toLocaleString()],['Unknown remaining inside',(m.neighbor_remaining_inside??m.remaining_unresolved).toLocaleString()],['Assumptions near authored stamps',(m.manual_neighbor_pixels??0).toLocaleString()],['Changed by stamp weighting',(m.manual_changed_choice_pixels??0).toLocaleString()],['Boundary island outlines',m.boundary_islands.toLocaleString()]];$('#metrics').innerHTML=items.map(([a,b])=>`<div class="metric"><span>${a}</span><span>${b}</span></div>`).join('');$('#legend').innerHTML=payload.categories.map(c=>`<div class="legend-item"><span class="swatch" style="background:rgb(${c.color.slice(0,3).join(',')})"></span>${c.label}</div>`).join('');document.querySelectorAll('[data-mode]').forEach(b=>b.onclick=()=>setMode(b.dataset.mode));updateOverlays();setMode(mode);fit()}
$('#fit').onclick=fit;$('#one').onclick=one;viewport.onpointerdown=e=>{drag={x:e.clientX,y:e.clientY,tx,ty};viewport.setPointerCapture(e.pointerId);viewport.classList.add('dragging')};viewport.onpointermove=e=>{if(!drag)return;tx=drag.tx+e.clientX-drag.x;ty=drag.ty+e.clientY-drag.y;setTransform()};viewport.onpointerup=()=>{drag=null;viewport.classList.remove('dragging')};viewport.onwheel=e=>{e.preventDefault();const rect=viewport.getBoundingClientRect(),mx=e.clientX-rect.left,my=e.clientY-rect.top,old=scale;scale=Math.max(.15,Math.min(8,scale*Math.exp(-e.deltaY*.0015)));tx=mx-(mx-tx)*(scale/old);ty=my-(my-ty)*(scale/old);setTransform()};window.onkeydown=e=>{if(e.target.matches('input,textarea'))return;const key=e.key.toLowerCase();if(key==='s')setMode('source');if(key==='a')setMode('approved');if(key==='c')setMode('conservative');if(key==='f'&&payload.assets.neighbor)setMode('neighbor');if(key==='g')setMode('aggressive')};window.onresize=fit;init();
</script></body></html>"""
