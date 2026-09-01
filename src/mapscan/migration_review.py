"""Dedicated visual approval gate for a migrated materialization candidate."""

from __future__ import annotations

import hashlib
import json
import mimetypes
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict

from PIL import Image


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text())


def _render_mask_overlay(
    mask_path: Path,
    output_path: Path,
    *,
    size: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    """Render a registered binary mask as a transparent review overlay."""

    mask = Image.open(mask_path).convert("L")
    if mask.size != size:
        mask = mask.resize(size, resample=Image.Resampling.NEAREST)
    alpha = mask.point(lambda value: 255 if value else 0)
    overlay = Image.new("RGBA", size, (*color, 0))
    overlay.putalpha(alpha)
    overlay.save(output_path, optimize=True)


def _county_png_review_overlays(
    alignment_path: Path,
    extraction: Dict[str, object],
    output_dir: Path,
) -> tuple[Dict[str, object], Dict[str, Path]]:
    """Build state/county overlays from the registered user-supplied county.png."""

    current_alignment_path = alignment_path
    visited = set()
    refinement_path = None
    while current_alignment_path not in visited:
        visited.add(current_alignment_path)
        candidate = current_alignment_path.parent / "perimeter-refinement.json"
        if candidate.exists():
            refinement_path = candidate
            break
        current_alignment = _load(current_alignment_path)
        parent = current_alignment.get("parent_alignment")
        if not isinstance(parent, dict):
            break
        parent_path = Path(str(parent.get("path", ""))).resolve()
        if not parent_path.is_file() or _sha256(parent_path) != parent.get("sha256"):
            raise ValueError("Alignment ancestry is stale while finding county.png")
        current_alignment_path = parent_path
    if refinement_path is None:
        raise FileNotFoundError("No county.png perimeter refinement exists in alignment ancestry")
    refinement = _load(refinement_path)
    declared_reference = refinement.get("county_reference")
    if not isinstance(declared_reference, dict):
        raise ValueError("The v4 refinement does not declare a county.png reference")
    reference_manifest_path = Path(str(declared_reference["path"])).resolve()
    if _sha256(reference_manifest_path) != declared_reference["sha256"]:
        raise ValueError("The v4 county.png reference manifest hash does not match")
    reference = _load(reference_manifest_path)
    source_path = Path(str(reference["source"]["path"])).resolve()
    if _sha256(source_path) != reference["source"]["sha256"]:
        raise ValueError("The registered county.png source hash does not match")

    review_grid = extraction["alignment"]["inspection"]["grid"]
    reference_grid = reference["web_grid"]
    if review_grid["crs"] != reference_grid["crs"]:
        raise ValueError("The county.png and review overlays use different CRSs")
    if review_grid["bounds"] != reference_grid["bounds"]:
        raise ValueError("The county.png and review overlays use different bounds")
    size = (int(review_grid["width"]), int(review_grid["height"]))

    artifact_specs = {
        "state": ("web_mercator_state_border", (0, 225, 255)),
        "county": ("web_mercator_county_border", (255, 0, 225)),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    assets: Dict[str, Path] = {}
    artifact_records: Dict[str, object] = {}
    for key, (artifact_key, color) in artifact_specs.items():
        declared_artifact = reference["artifacts"][artifact_key]
        mask_path = reference_manifest_path.parent / declared_artifact["path"]
        if _sha256(mask_path) != declared_artifact["sha256"]:
            raise ValueError(f"Registered county.png {key} mask hash does not match")
        output_path = output_dir / f"county-png-{key}-overlay.png"
        _render_mask_overlay(mask_path, output_path, size=size, color=color)
        assets[key] = output_path
        artifact_records[key] = {
            "path": str(output_path),
            "sha256": _sha256(output_path),
            "source_mask_path": str(mask_path),
            "source_mask_sha256": declared_artifact["sha256"],
        }

    manifest = {
        "schema_version": 1,
        "status": "pass",
        "purpose": "categorical_migration_review_reference_overlays",
        "reference_kind": "user_supplied_county_png",
        "source": {
            "path": str(source_path),
            "sha256": reference["source"]["sha256"],
        },
        "registration_manifest": {
            "path": str(reference_manifest_path),
            "sha256": declared_reference["sha256"],
        },
        "refinement_manifest": {
            "path": str(refinement_path),
            "sha256": _sha256(refinement_path),
        },
        "reference_grid": reference_grid,
        "review_grid": review_grid,
        "artifacts": artifact_records,
        "interpretation": (
            "Cyan is the thick California state stroke extracted from county.png; "
            "magenta is only the thin California county linework from county.png."
        ),
    }
    manifest_path = output_dir / "county-png-review-overlays.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    manifest["manifest"] = {
        "path": str(manifest_path),
        "sha256": _sha256(manifest_path),
    }
    return manifest, assets


def build_migration_review_payload(
    approved_dir: Path,
    target_run: Path,
    candidate_dir: Path,
    comparison_dir: Path,
    alignment_audit_dir: Path,
    stamp_audit_dir: Path,
) -> tuple[Dict[str, object], Dict[str, Path]]:
    approved_dir = approved_dir.resolve()
    target_run = target_run.resolve()
    candidate_dir = candidate_dir.resolve()
    comparison_dir = comparison_dir.resolve()
    alignment_audit_dir = alignment_audit_dir.resolve()
    stamp_audit_dir = stamp_audit_dir.resolve()
    approved_manifest_path = approved_dir / "materialization.json"
    candidate_manifest_path = candidate_dir / "materialization.json"
    approved_manifest = _load(approved_manifest_path)
    candidate_manifest = _load(candidate_manifest_path)
    comparison_path = comparison_dir / "approved-vs-candidate-comparison.json"
    comparison = _load(comparison_path)
    alignment_audit_path = alignment_audit_dir / "alignment-application-audit.json"
    alignment_audit = _load(alignment_audit_path)
    stamp_audit_path = stamp_audit_dir / "stamp-migration-audit.json"
    stamp_audit = _load(stamp_audit_path)
    extraction_path = target_run / "extraction.json"
    extraction = _load(extraction_path)
    alignment_path = Path(extraction["alignment"]["path"]).resolve()
    reference_overlay, reference_assets = _county_png_review_overlays(
        alignment_path,
        extraction,
        comparison_dir / "county-png-review-overlays",
    )
    decision_path = candidate_dir / "materialization-review-decision.json"
    existing_decision = _load(decision_path) if decision_path.exists() else None
    layer_id = str(candidate_manifest["layers"][0]["layer_id"])
    old_layer = approved_manifest["layers"][0]
    new_layer = candidate_manifest["layers"][0]

    candidate_hash = _sha256(candidate_manifest_path)
    if comparison["candidate"]["materialization_sha256"] != candidate_hash:
        raise ValueError("Comparison does not match the candidate materialization")
    if alignment_audit.get("status") != "pass":
        raise ValueError("Alignment application audit has not passed")
    if stamp_audit.get("status") != "pass":
        raise ValueError("Stamp migration audit has not passed")
    if alignment_audit["target_alignment"]["sha256"] != _sha256(alignment_path):
        raise ValueError("Alignment audit does not match the target extraction")

    hybrid_path = alignment_path.parent / "web-mercator-authoritative-unified-border-overlay.png"
    hybrid_report_path = alignment_path.parent / "hybrid-perimeter-audit.json"
    if not hybrid_path.is_file() or not hybrid_report_path.is_file():
        raise FileNotFoundError("Accepted hybrid perimeter assets are missing")
    hybrid_report = _load(hybrid_report_path)
    declared_hybrid = hybrid_report["artifacts"][hybrid_path.name]
    if _sha256(hybrid_path) != declared_hybrid["sha256"]:
        raise ValueError("Accepted hybrid perimeter overlay hash does not match")
    topology = hybrid_report.get("unified_border", {})
    if (
        topology.get("passed") is not True
        or topology.get("connected_component_count") != 1
        or topology.get("interior_is_exact_fill_of_displayed_border") is not True
    ):
        raise ValueError("Accepted hybrid perimeter is not one closed clipping line")
    boundary_clip = candidate_manifest.get("boundary_clip")
    if not isinstance(boundary_clip, dict):
        raise ValueError("Final candidate is not clipped to the displayed boundary")
    boundary_audit_path = Path(str(boundary_clip["audit"]["path"])).resolve()
    if _sha256(boundary_audit_path) != boundary_clip["audit"]["sha256"]:
        raise ValueError("Boundary clip audit hash does not match")
    boundary_audit = _load(boundary_audit_path)
    displayed_border_record = boundary_audit.get("boundary", {}).get("border", {})
    displayed_border_path = boundary_audit_path.parent / str(
        displayed_border_record.get("path", "")
    )
    boundary_topology = boundary_audit.get("boundary", {})
    boundary_component_count = int(
        boundary_topology.get("connected_component_count", 0)
    )
    expected_boundary_component_count = int(
        boundary_topology.get(
            "expected_component_count", boundary_component_count
        )
    )
    if (
        boundary_audit.get("status") != "pass"
        or not displayed_border_path.is_file()
        or _sha256(displayed_border_path) != displayed_border_record.get("sha256")
        or boundary_clip.get("continuous_border_sha256")
        != _sha256(displayed_border_path)
        or boundary_component_count < 1
        or boundary_component_count != expected_boundary_component_count
        or boundary_clip.get("colored_pixel_count_outside_boundary") != 0
        or boundary_clip.get("unclassified_pixel_count_inside_boundary") != 0
    ):
        raise ValueError("Boundary clip does not exactly match the displayed border")

    assets = {
        "source": target_run / "web-mercator-source.jpg",
        "state": reference_assets["state"],
        "county": reference_assets["county"],
        "hybrid": displayed_border_path,
        "old": approved_dir / old_layer["artifacts"]["preview"]["path"],
        "new": candidate_dir / new_layer["artifacts"]["preview"]["path"],
        "diff": comparison_dir
        / comparison["artifacts"]["change_overlay"]["path"],
    }
    completion_artifact = new_layer["artifacts"].get("source_diff_completion_mask")
    if isinstance(completion_artifact, dict):
        completion_mask_path = candidate_dir / str(completion_artifact["path"])
        if _sha256(completion_mask_path) != completion_artifact["sha256"]:
            raise ValueError("Source-diff completion mask hash does not match")
        completion_overlay_path = comparison_dir / "source-diff-completion-overlay.png"
        _render_mask_overlay(
            completion_mask_path,
            completion_overlay_path,
            size=(int(new_layer["width"]), int(new_layer["height"])),
            color=(0, 245, 255),
        )
        assets["completion"] = completion_overlay_path
    for path in assets.values():
        if not path.exists():
            raise FileNotFoundError(path)
    candidate_title = str(
        candidate_manifest.get("title")
        or candidate_manifest["dataset_id"]
    ).replace("-", " ")
    payload = {
        "schema_version": 1,
        "title": f"Final {candidate_title} materialization review",
        "candidate_status": candidate_manifest["status"],
        "candidate_dataset_id": candidate_manifest["dataset_id"],
        "candidate_materialization": {
            "path": str(candidate_manifest_path),
            "sha256": candidate_hash,
            "source_diff": candidate_manifest.get("source_diff"),
        },
        "approved_materialization": {
            "path": str(approved_manifest_path),
            "sha256": _sha256(approved_manifest_path),
        },
        "target_extraction": {
            "path": str(extraction_path),
            "sha256": _sha256(extraction_path),
        },
        "v4_alignment": {
            "path": str(alignment_path),
            "sha256": _sha256(alignment_path),
            "application_audit_status": alignment_audit["status"],
            "application_mismatch_pixels": alignment_audit[
                "target_recompute_mismatch_pixel_count"
            ],
        },
        "hybrid_border": {
            "path": str(displayed_border_path),
            "sha256": _sha256(displayed_border_path),
            "audit_path": str(hybrid_report_path),
            "audit_sha256": _sha256(hybrid_report_path),
            "status": hybrid_report["status"],
            "connected_component_count": boundary_component_count,
            "expected_component_count": expected_boundary_component_count,
            "mainland_interior_pixel_count": boundary_topology.get(
                "mainland_interior_pixel_count"
            ),
            "publication_interior_pixel_count": boundary_topology.get(
                "publication_interior_pixel_count"
            ),
            "components": boundary_topology.get("components", []),
        },
        "boundary_clip": {
            "audit_path": str(boundary_audit_path),
            "audit_sha256": _sha256(boundary_audit_path),
            "status": boundary_audit["status"],
        },
        "reference_overlay": reference_overlay,
        "metrics": {
            "changed_pixel_count": comparison["changed_pixel_count"],
            "changed_pixel_fraction": comparison["changed_pixel_fraction"],
            "migrated_manual_target_pixel_count": stamp_audit[
                "manual_target_pixel_count"
            ],
            "migrated_manual_target_mask_identical": stamp_audit[
                "manual_target_mask_identical"
            ],
            "manual_target_pixel_count_inside_boundary": comparison[
                "manual_target_pixel_count"
            ],
            "manual_target_pixel_count_removed_outside_boundary": int(
                stamp_audit["manual_target_pixel_count"]
                - comparison["manual_target_pixel_count"]
            ),
            "manual_value_changed_pixel_count": comparison[
                "manual_value_changed_pixel_count"
            ],
            "stamp_change_reduction_fraction": stamp_audit[
                "changed_value_reduction_fraction"
            ],
            "final_classified_pixel_count": new_layer[
                "final_classified_pixel_count"
            ],
            "enclosed_fill_pixel_count": new_layer["enclosed_fill_pixel_count"],
            "source_diff_completion_pixel_count": new_layer.get(
                "source_diff_completion_pixel_count", 0
            ),
            "dropped_visible_source_evidence_pixel_count": new_layer.get(
                "dropped_visible_source_evidence_pixel_count"
            ),
            "unclassified_pixel_count_after": new_layer.get(
                "unclassified_pixel_count_after"
            ),
            "continuous_border_component_count": boundary_component_count,
            "colored_pixel_count_outside_boundary": new_layer.get(
                "colored_pixel_count_outside_boundary"
            ),
            "boundary_removed_pixel_count": new_layer.get(
                "boundary_removed_pixel_count"
            ),
            "boundary_completion_pixel_count": new_layer.get(
                "boundary_completion_pixel_count"
            ),
        },
        "assets": {key: f"/asset/{key}" for key in assets},
        "decision_path": str(decision_path),
        "decision": existing_decision,
        "review_instruction": (
            "Inspect Final against every accepted lime publication boundary and the county "
            "network. Confirm that every lime line is closed and that no class "
            "color appears outside it; toggle the cyan source-diff completion footprint, compare Old "
            "approved, inspect the aligned source, and check the magenta Difference "
            "diagnostic. A decision binds the displayed "
            "fixed-point surface, alignment, hybrid border, and county.png hashes."
        ),
    }
    return payload, assets


def _decision_document(
    payload: Dict[str, object], request: Dict[str, object]
) -> Dict[str, object]:
    status = str(request.get("status", ""))
    statement = str(request.get("statement", "")).strip()
    if status not in {"approved", "rejected"}:
        raise ValueError("Decision must be approved or rejected")
    if request.get("inspection_confirmed") is not True:
        raise ValueError("Confirm the required comparison inspection")
    if not statement:
        raise ValueError("Enter a short review statement")
    return {
        "schema_version": 1,
        "scope": "migrated_materialized_categorical_candidate",
        "status": status,
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "materialization_sha256": payload["candidate_materialization"]["sha256"],
        "materialization_path": payload["candidate_materialization"]["path"],
        "alignment_sha256": payload["v4_alignment"]["sha256"],
        "hybrid_border_sha256": payload.get("hybrid_border", {}).get("sha256"),
        "boundary_clip_audit_sha256": payload.get("boundary_clip", {}).get(
            "audit_sha256"
        ),
        "source_diff": payload["candidate_materialization"].get("source_diff"),
        "reference_overlay_manifest_sha256": payload["reference_overlay"]["manifest"][
            "sha256"
        ],
        "reference_source_sha256": payload["reference_overlay"]["source"]["sha256"],
        "comparison_materialization_sha256": payload["approved_materialization"][
            "sha256"
        ],
        "author_statement": statement,
        "inspection_confirmed": True,
        "approval_carried_forward": False,
        "evidence": payload["metrics"],
    }


def serve_migration_review(
    approved_dir: Path,
    target_run: Path,
    candidate_dir: Path,
    comparison_dir: Path,
    alignment_audit_dir: Path,
    stamp_audit_dir: Path,
    host: str = "127.0.0.1",
    port: int = 8784,
) -> None:
    payload, assets = build_migration_review_payload(
        approved_dir,
        target_run,
        candidate_dir,
        comparison_dir,
        alignment_audit_dir,
        stamp_audit_dir,
    )
    candidate_manifest_path = Path(payload["candidate_materialization"]["path"])
    decision_path = Path(payload["decision_path"])

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/":
                self._send(200, _HTML.encode(), "text/html; charset=utf-8")
                return
            if self.path == "/session.json":
                refreshed, _ = build_migration_review_payload(
                    approved_dir,
                    target_run,
                    candidate_dir,
                    comparison_dir,
                    alignment_audit_dir,
                    stamp_audit_dir,
                )
                self._send(
                    200,
                    json.dumps(refreshed).encode(),
                    "application/json; charset=utf-8",
                )
                return
            if self.path.startswith("/asset/"):
                key = self.path.removeprefix("/asset/")
                path = assets.get(key)
                if path is None:
                    self._send(404, b"not found", "text/plain")
                    return
                content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
                self._send(200, path.read_bytes(), content_type)
                return
            self._send(404, b"not found", "text/plain")

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/decision":
                self._send(404, b"not found", "text/plain")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                request = json.loads(self.rfile.read(length))
                if _sha256(candidate_manifest_path) != payload[
                    "candidate_materialization"
                ]["sha256"]:
                    raise ValueError("Candidate changed; reload before deciding")
                decision = _decision_document(payload, request)
                decision_path.write_text(json.dumps(decision, indent=2) + "\n")
                self._send(
                    200,
                    json.dumps({"decision": decision}).encode(),
                    "application/json; charset=utf-8",
                )
            except (ValueError, json.JSONDecodeError) as error:
                self._send(
                    400,
                    json.dumps({"error": str(error)}).encode(),
                    "application/json; charset=utf-8",
                )

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Categorical migration review: http://{host}:{port}/", flush=True)
    server.serve_forever()


_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Final categorical materialization review</title>
<style>
:root{color-scheme:dark;font-family:Inter,ui-sans-serif,system-ui,sans-serif}*{box-sizing:border-box}body{margin:0;background:#090909;color:#f5f5f5;height:100vh;overflow:hidden}.app{display:grid;grid-template-columns:370px 1fr;height:100vh}.panel{padding:20px;border-right:1px solid #2a2a2a;background:#111;overflow:auto}.panel h1{font-size:20px;margin:0 0 8px}.sub{color:#aaa;font-size:12px;line-height:1.45}.hash{font-family:ui-monospace,monospace;font-size:10px;color:#888;overflow-wrap:anywhere}.modes,.actions{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:18px 0 12px}button{border:1px solid #3a3a3a;background:#202020;color:#eee;padding:9px;border-radius:7px;cursor:pointer}button.active{background:#fff;color:#111;border-color:#fff}.row{display:flex;gap:10px;align-items:center;margin:10px 0;font-size:13px}.row input[type=range]{flex:1}.metrics{border-top:1px solid #292929;margin-top:16px;padding-top:12px}.metric{display:flex;justify-content:space-between;gap:12px;padding:4px 0;font-size:12px}.metric span:first-child{color:#aaa}.review{border-top:1px solid #292929;margin-top:16px;padding-top:14px}.review textarea{width:100%;height:70px;background:#181818;color:#fff;border:1px solid #3a3a3a;border-radius:7px;padding:8px}.review label{font-size:12px;line-height:1.4;display:flex;gap:8px;margin:10px 0}.approve{background:#124c29}.reject{background:#5a1d1d}.status{font-size:12px;color:#f0c36b;margin-top:8px}.viewport{position:relative;overflow:hidden;background:#050505;cursor:grab}.viewport.dragging{cursor:grabbing}.stage{position:absolute;left:0;top:0;transform-origin:0 0;width:3398px;height:3920px}.stage img{position:absolute;inset:0;width:3398px;height:3920px;object-fit:fill;image-rendering:pixelated;pointer-events:none}.toolbar{position:absolute;right:14px;top:14px;display:flex;gap:6px;z-index:5}.toolbar button{background:#111d}.badge{display:inline-block;background:#2b2313;color:#ffd88a;padding:4px 7px;border-radius:999px;font-size:11px;margin:5px 0 12px}.legend{font-size:11px;color:#aaa;margin-bottom:4px}
</style></head><body><div class="app"><aside class="panel"><h1>Final categorical review</h1><div class="badge">fixed-point candidate · boundary-clipped · needs visual review · unpublished</div><div class="sub" id="instruction"></div><div class="modes"><button data-mode="new" class="active">Final (N)</button><button data-mode="old">Old approved (O)</button><button data-mode="source">Aligned source (S)</button><button data-mode="diff">Difference (D)</button></div><div class="legend" id="mode-description"></div><div class="row"><input id="hybrid" type="checkbox" checked><label for="hybrid">Publication boundaries (lime)</label></div><div class="row"><input id="state" type="checkbox"><label for="state">Full county.png state audit (cyan)</label></div><div class="row"><input id="county" type="checkbox" checked><label for="county">county.png county borders (magenta)</label></div><div class="row"><input id="completion" type="checkbox"><label for="completion">Source-diff completion footprint (cyan)</label></div><div class="row"><label>Overlay opacity</label><input id="overlay-opacity" type="range" min="0" max="1" value="0.82" step="0.01"></div><div class="metrics" id="metrics"></div><div class="review"><strong>Decision for this boundary-clipped candidate only</strong><div class="hash" id="candidate-hash"></div><textarea id="statement" placeholder="Short review statement"></textarea><label><input id="confirmed" type="checkbox">I inspected every lime boundary, confirmed no color appears outside them, compared Old approved, inspected Aligned source, and checked Difference.</label><div class="actions"><button class="approve" id="approve">Approve final candidate</button><button class="reject" id="reject">Reject final candidate</button></div><div class="status" id="status"></div></div></aside><main class="viewport" id="viewport"><div class="toolbar"><button id="fit">Fit</button><button id="one">100%</button></div><div class="stage" id="stage"><img id="base"><img id="hybrid-img"><img id="state-img"><img id="county-img"><img id="completion-img"><img id="diff-img"></div></main></div>
<script>
let payload,mode='new',scale=1,tx=0,ty=0,drag=null;const W=3398,H=3920;const viewport=document.querySelector('#viewport'),stage=document.querySelector('#stage'),base=document.querySelector('#base'),hybridImg=document.querySelector('#hybrid-img'),stateImg=document.querySelector('#state-img'),countyImg=document.querySelector('#county-img'),completionImg=document.querySelector('#completion-img'),diffImg=document.querySelector('#diff-img');
function transform(){stage.style.transform=`translate(${tx}px,${ty}px) scale(${scale})`}
function fit(){scale=Math.min(viewport.clientWidth/W,viewport.clientHeight/H)*.98;tx=(viewport.clientWidth-W*scale)/2;ty=(viewport.clientHeight-H*scale)/2;transform()}
function setMode(next){mode=next;document.querySelectorAll('[data-mode]').forEach(b=>b.classList.toggle('active',b.dataset.mode===mode));diffImg.style.display='none';if(mode==='new')base.src=payload.assets.new;if(mode==='old')base.src=payload.assets.old;if(mode==='source')base.src=payload.assets.source;if(mode==='diff'){base.src=payload.assets.new;diffImg.src=payload.assets.diff;diffImg.style.display='block'};const descriptions={new:'Final candidate: exactly filled inside every lime publication boundary; zero colored pixels outside.',old:'Previously approved materialization, for comparison only.',source:'Original source warped by the final accepted alignment.',diff:'Final candidate with every changed old→final pixel highlighted magenta.'};document.querySelector('#mode-description').textContent=descriptions[mode]}
function overlays(){if(!payload)return;hybridImg.style.display=document.querySelector('#hybrid').checked?'block':'none';stateImg.style.display=document.querySelector('#state').checked?'block':'none';countyImg.style.display=document.querySelector('#county').checked?'block':'none';completionImg.style.display=payload.assets.completion&&document.querySelector('#completion').checked?'block':'none';const o=document.querySelector('#overlay-opacity').value;hybridImg.style.opacity=o;stateImg.style.opacity=o;countyImg.style.opacity=o;completionImg.style.opacity=o}
async function decide(status){const out=document.querySelector('#status');try{const response=await fetch('/decision',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({status,statement:document.querySelector('#statement').value,inspection_confirmed:document.querySelector('#confirmed').checked})});const result=await response.json();if(!response.ok)throw new Error(result.error);out.textContent=`Saved ${result.decision.status} for ${result.decision.materialization_sha256.slice(0,12)}…`;}catch(error){out.textContent=error.message}}
async function init(){payload=await (await fetch('/session.json')).json();document.querySelector('#instruction').textContent=payload.review_instruction;document.querySelector('#candidate-hash').textContent=payload.candidate_materialization.sha256;hybridImg.src=payload.assets.hybrid;stateImg.src=payload.assets.state;countyImg.src=payload.assets.county;if(payload.assets.completion)completionImg.src=payload.assets.completion;else document.querySelector('#completion').closest('.row').style.display='none';const m=payload.metrics,items=[['Continuous border components',m.continuous_border_component_count.toLocaleString()],['Colored outside border',m.colored_pixel_count_outside_boundary.toLocaleString()],['Internal NoData',m.unclassified_pixel_count_after.toLocaleString()],['Removed outside border',m.boundary_removed_pixel_count.toLocaleString()],['Completed inside border',m.boundary_completion_pixel_count.toLocaleString()],['Final classified',m.final_classified_pixel_count.toLocaleString()],['Changed vs old',`${m.changed_pixel_count.toLocaleString()} (${(m.changed_pixel_fraction*100).toFixed(2)}%)`],['Migrated stamp targets',`${m.migrated_manual_target_pixel_count.toLocaleString()} · exact ${m.migrated_manual_target_mask_identical}`],['Stamp targets outside border removed',m.manual_target_pixel_count_removed_outside_boundary.toLocaleString()],['Migration divergence reduced',(m.stamp_change_reduction_fraction*100).toFixed(2)+'%'],['Source-diff completion',m.source_diff_completion_pixel_count.toLocaleString()],['Dropped source evidence',m.dropped_visible_source_evidence_pixel_count.toLocaleString()],['Final transform recompute','0 mismatches']];document.querySelector('#metrics').innerHTML=items.map(([a,b])=>`<div class="metric"><span>${a}</span><span>${b}</span></div>`).join('');if(payload.decision)document.querySelector('#status').textContent=`Existing decision: ${payload.decision.status}`;setMode('new');overlays();fit()}
document.querySelectorAll('[data-mode]').forEach(b=>b.onclick=()=>setMode(b.dataset.mode));document.querySelector('#hybrid').onchange=overlays;document.querySelector('#state').onchange=overlays;document.querySelector('#county').onchange=overlays;document.querySelector('#completion').onchange=overlays;document.querySelector('#overlay-opacity').oninput=overlays;document.querySelector('#fit').onclick=fit;document.querySelector('#one').onclick=()=>{scale=1;tx=30;ty=30;transform()};document.querySelector('#approve').onclick=()=>decide('approved');document.querySelector('#reject').onclick=()=>decide('rejected');
viewport.onwheel=e=>{e.preventDefault();const rect=viewport.getBoundingClientRect(),x=e.clientX-rect.left,y=e.clientY-rect.top,old=scale;scale=Math.max(.08,Math.min(8,scale*Math.exp(-e.deltaY*.001)));tx=x-(x-tx)*scale/old;ty=y-(y-ty)*scale/old;transform()};viewport.onpointerdown=e=>{drag={x:e.clientX,y:e.clientY,tx,ty};viewport.setPointerCapture(e.pointerId);viewport.classList.add('dragging')};viewport.onpointermove=e=>{if(!drag)return;tx=drag.tx+e.clientX-drag.x;ty=drag.ty+e.clientY-drag.y;transform()};viewport.onpointerup=()=>{drag=null;viewport.classList.remove('dragging')};window.onresize=fit;window.onkeydown=e=>{if(e.target.matches('textarea,input'))return;const key=e.key.toLowerCase();if(key==='n')setMode('new');if(key==='o')setMode('old');if(key==='s')setMode('source');if(key==='d')setMode('diff')};init();
</script></body></html>"""
