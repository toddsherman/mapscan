"""Read-only visual review for an automatic county.png fine alignment."""

from __future__ import annotations

import hashlib
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_fine_alignment_review_payload(
    run_dir: Path,
) -> tuple[Dict[str, object], Dict[str, Path]]:
    run_dir = run_dir.resolve()
    lower_colorado = (run_dir / "lower-colorado-refinement.json").exists()
    southern = (
        not lower_colorado and (run_dir / "southern-edge-refinement.json").exists()
    )
    report_path = run_dir / (
        "lower-colorado-refinement.json"
        if lower_colorado
        else (
            "southern-edge-refinement.json"
            if southern
            else "county-fine-alignment.json"
        )
    )
    report = json.loads(report_path.read_text())
    if report.get("status") != "needs_author_review":
        raise ValueError("Fine alignment has not passed automatic QA")
    if lower_colorado:
        if not report["fixed_point_gate"]["passed"]:
            raise ValueError("Lower Colorado fixed-point gate has not passed")
        if not report["independent_edge_holdouts"]["passed"]:
            raise ValueError("Lower Colorado holdouts have not passed")
        if not report["interior_county_veto"]["passed"]:
            raise ValueError("Lower Colorado interior-county veto has not passed")
        if not report["unchanged_region_veto"]["passed"]:
            raise ValueError("Lower Colorado no-change veto has not passed")
        if not report["transform_regularity"]["passed"]:
            raise ValueError("Lower Colorado transform regularity has not passed")
    elif southern:
        if not report["edge_fixed_point_gate"]["passed"]:
            raise ValueError("Southern edge fixed-point gate has not passed")
        if not report["independent_edge_holdouts"]["passed"]:
            raise ValueError("Southern edge holdouts have not passed")
        if not report["interior_county_veto"]["passed"]:
            raise ValueError("Southern edge interior-county veto has not passed")
    else:
        if not report["fixed_point_gate"]["passed"]:
            raise ValueError("Fine alignment fixed-point gate has not passed")
        if not report["independent_spatial_holdouts"]["passed"]:
            raise ValueError("Fine alignment spatial holdouts have not passed")
        if not report["transform_regularity"]["passed"]:
            raise ValueError("Fine alignment transform regularity has not passed")
        if not report["state_boundary_veto"]["passed"]:
            raise ValueError("Fine alignment state-boundary veto has not passed")

    determinism_path = run_dir / "determinism-audit.json"
    determinism = json.loads(determinism_path.read_text())
    if not determinism.get("passed"):
        raise ValueError("Fine alignment determinism audit has not passed")

    hybrid_path = run_dir / "hybrid-perimeter-audit.json"
    hybrid = None
    if lower_colorado and hybrid_path.exists():
        hybrid = json.loads(hybrid_path.read_text())
        if hybrid.get("status") != "pass_no_additional_warp":
            raise ValueError("Hybrid perimeter audit has not passed")
        if hybrid.get("decision", {}).get("additional_warp") is not False:
            raise ValueError("Hybrid perimeter audit proposed an unreviewed warp")
        hybrid_determinism_path = run_dir / "hybrid-perimeter-determinism.json"
        hybrid_determinism = json.loads(hybrid_determinism_path.read_text())
        if not hybrid_determinism.get("passed"):
            raise ValueError("Hybrid perimeter determinism audit has not passed")

    keys = {
        "before": "web-mercator-source-before.jpg",
        "fine": "web-mercator-source-after.jpg",
        "state": "web-mercator-county-png-state-overlay.png",
        "county": "web-mercator-county-png-county-overlay.png",
        "anchors_before": (
            "lower-colorado-diagnostic-before.jpg"
            if lower_colorado
            else (
                "southern-edge-diagnostic-before.jpg"
                if southern
                else "working-diagnostic-before.jpg"
            )
        ),
        "anchors_after": (
            "lower-colorado-diagnostic-after.jpg"
            if lower_colorado
            else (
                "southern-edge-diagnostic-after.jpg"
                if southern
                else "working-diagnostic-after.jpg"
            )
        ),
    }
    if lower_colorado:
        keys["census"] = "web-mercator-census-state-overlay.png"
    assets: Dict[str, Path] = {}
    for key, name in keys.items():
        path = run_dir / name
        if not path.exists():
            raise FileNotFoundError(path)
        declared = report["artifacts"].get(name)
        if declared is not None and _sha256(path) != declared["sha256"]:
            raise ValueError(f"Review asset changed: {name}")
        assets[key] = path
    if hybrid is not None:
        for key, name in (
            ("hybrid", "web-mercator-hybrid-state-overlay.png"),
            ("unified", "web-mercator-authoritative-unified-border-overlay.png"),
        ):
            asset = run_dir / name
            if not asset.exists():
                raise FileNotFoundError(asset)
            declared = hybrid["artifacts"][name]
            if _sha256(asset) != declared["sha256"]:
                raise ValueError(f"Review asset changed: {name}")
            assets[key] = asset

    if lower_colorado:
        holdout = report["independent_edge_holdouts"]
        fit = report["fit_evidence"]
        unchanged = report["unchanged_region_veto"]
        regularity = report["transform_regularity"]
        metrics = {
            "matched_before": fit["before"]["residual"],
            "matched_after": fit["after"]["residual"],
            "holdout_before": holdout["before"]["residual"],
            "holdout_after": holdout["after"]["residual"],
            "holdout_regions_passed": holdout["after"]["accepted_count"],
            "holdout_region_count": holdout["before"]["window_count"],
            "state_before": unchanged["mexico_border"]["before"]["residual"],
            "state_after": unchanged["mexico_border"]["after"]["residual"],
            "jacobian": {
                "minimum": regularity["sampling_jacobian_min"],
                "maximum": regularity["sampling_jacobian_max"],
            },
            "local_scale": None,
            "deterministic": determinism["passed"],
        }
        grid = report["grid"]
        working_grid = report["working_grid"]
        instruction = (
            "The default lime outline is one combined border: the detailed "
            "county.png shoreline through San Diego and the Tahoe hinge joined to "
            "Census land borders, including the Colorado River and Mexico border. "
            "Turn on the cyan/yellow provenance view or either complete raw "
            "reference to audit every span."
            if hybrid is not None
            else (
                "Use SE focus and Blink to inspect the lower Colorado edge. Yellow is "
                "the authoritative Census state boundary; cyan is the independently "
                "registered county.png state outline; magenta is county.png's county "
                "network. The correction was fit in green windows and validated in "
                "untouched blue windows."
            )
        )
        descriptions = {
            "fine": "Census-anchored lower-Colorado candidate at the native review grid.",
            "before": "Prior county-fine candidate before the bounded 6 px edge correction.",
            "blink": "Alternates prior and refined every 650 ms without moving the view.",
            "anchors": "Green fit and blue untouched holdout windows on the lower Colorado.",
        }
        note = (
            "The lime line contains exactly the union of the cyan authoritative "
            "coast pixels and yellow authoritative land-border pixels. At the "
            "crop-clipped San Diego/Mexico junction, an explicitly measured short "
            "connector closes the two authoritative spans; no new warp was added. "
            "The candidate remains unpublished pending visual review."
            if hybrid is not None
            else (
                "The Mexico border was not pulled toward cyan: it already agrees with "
                "the authoritative yellow Census border, while county.png is displaced "
                "there. Only the corroborated lower-Colorado residual was corrected."
            )
        )
        badge = (
            "automatic · regional hybrid perimeter · unpublished"
            if hybrid is not None
            else "automatic · Census state + county.png counties · unpublished"
        )
    elif southern:
        holdout = report["independent_edge_holdouts"]
        local_fit = report["local_fit"]
        metrics = {
            "matched_before": report["before"]["residual"],
            "matched_after": report["after"]["residual"],
            "holdout_before": holdout["before"]["residual"],
            "holdout_after": holdout["after"]["residual"],
            "holdout_regions_passed": holdout["after"]["accepted_count"],
            "holdout_region_count": holdout["before"]["accepted_count"],
            "state_before": holdout["before"]["residual"],
            "state_after": holdout["after"]["residual"],
            "jacobian": {
                "minimum": local_fit["sampled_jacobian_min"],
                "maximum": local_fit["sampled_jacobian_max"],
            },
            "local_scale": None,
            "deterministic": determinism["passed"],
        }
        grid = report["grid"]
        working_grid = report["grid"]
        instruction = (
            "Inspect Refined at 100% around the southern coast, southern border, "
            "and lower eastern edge, then Blink against the prior county-fine "
            "candidate. One contiguous half of each border segment fit the compact "
            "correction; its untouched half independently validated it."
        )
        descriptions = {
            "fine": "Refined south/southeast candidate at the full review grid.",
            "before": "Prior county-fine candidate before the compact edge correction.",
            "blink": "Alternates prior and refined every 650 ms without moving the view.",
            "anchors": "Full-resolution automatic edge controls; no manual arrows.",
        }
        note = (
            "No manual arrows or generic edges were used. Each correction has the "
            "saturated California map on its inward side and pale water/page on its "
            "outward side; gray parallel shadows cannot satisfy that signed test."
        )
        badge = "automatic · compact edge diagnostic · unpublished"
    else:
        holdout = report["independent_spatial_holdouts"]
        regularity = report["transform_regularity"]
        state_veto = report["state_boundary_veto"]
        metrics = {
            "matched_before": report["before"]["shift_residual"],
            "matched_after": report["after"]["shift_residual"],
            "holdout_before": holdout["aggregate_before"],
            "holdout_after": holdout["aggregate_after"],
            "holdout_regions_passed": sum(
                bool(item["passed"]) for item in holdout["folds"]
            ),
            "holdout_region_count": len(holdout["folds"]),
            "state_before": state_veto["before"],
            "state_after": state_veto["after"],
            "jacobian": regularity["jacobian_determinant"],
            "local_scale": regularity["local_scale"],
            "deterministic": determinism["passed"],
        }
        grid = report["native_grid"]
        working_grid = report["working_grid"]
        instruction = (
            "Inspect Fine at several zoom levels with the cyan state and magenta "
            "county.png overlays, then use Blink to compare against Before. The "
            "automatic fit used interior county junctions only; ocean/coast shadows, "
            "terrain, hazard bands, labels, and city dots were excluded."
        )
        descriptions = {
            "fine": "Fine candidate at the full review grid.",
            "before": "Rejected v4 parent before the county.png-primary correction.",
            "blink": "Alternates Before and Fine every 650 ms while preserving the view.",
            "anchors": "Working-resolution accepted and rejected automatic corrections.",
        }
        note = (
            "No manual arrows were used. State/coast evidence is an independent "
            "veto only; it cannot pull the fit toward an offshore shadow."
        )
        badge = "automatic · county.png primary · unpublished"
    payload = {
        "schema_version": 1,
        "title": "Quake automatic fine alignment",
        "status": report["status"],
        "report": {"path": str(report_path), "sha256": _sha256(report_path)},
        "candidate_alignment": report["candidate_alignment"],
        "county_reference": report["county_reference"],
        "grid": grid,
        "working_grid": working_grid,
        "assets": {key: f"/asset/{key}" for key in assets},
        "metrics": metrics,
        "evidence_policy": report["evidence_policy"],
        "instruction": instruction,
        "mode_descriptions": descriptions,
        "review_note": note,
        "badge": badge,
        "lower_colorado_refinement": lower_colorado,
        "southern_edge_refinement": southern,
        "hybrid_perimeter": (
            {
                "status": hybrid["status"],
                "coast_reference": "county.png",
                "land_border_reference": "Census 2025",
                "coast_county_png": hybrid["coast"]["county_png"][
                    "all_window_residual"
                ],
                "coast_census": hybrid["coast"]["census"][
                    "all_window_residual"
                ],
                "coast_window_count": hybrid["coast"]["county_png"][
                    "window_count"
                ],
                "coast_accepted_count": hybrid["coast"]["county_png"][
                    "accepted_count"
                ],
                "Tahoe_county_png": hybrid["Tahoe_hinge"]["county_png"][
                    "all_window_residual"
                ],
                "Tahoe_census": hybrid["Tahoe_hinge"]["census"][
                    "all_window_residual"
                ],
                "Tahoe_window_count": hybrid["Tahoe_hinge"]["county_png"][
                    "window_count"
                ],
                "Tahoe_accepted_count": hybrid["Tahoe_hinge"]["county_png"][
                    "accepted_count"
                ],
                "southern_coast_seam": hybrid["southern_coast_seam"],
                "additional_warp": hybrid["decision"]["additional_warp"],
                "unified_border": hybrid["unified_border"],
            }
            if hybrid is not None
            else None
        ),
        "publication_allowed": False,
    }
    return payload, assets


def serve_fine_alignment_review(
    run_dir: Path, host: str = "127.0.0.1", port: int = 8785
) -> None:
    payload, assets = build_fine_alignment_review_payload(run_dir)

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
                refreshed, _ = build_fine_alignment_review_payload(run_dir)
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
                kind = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
                self._send(200, path.read_bytes(), kind)
                return
            self._send(404, b"not found", "text/plain")

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"MapScan fine-alignment review: http://{host}:{port}/", flush=True)
    server.serve_forever()


_HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MapScan fine alignment</title><style>
:root{color-scheme:dark;font-family:Inter,ui-sans-serif,system-ui,sans-serif}*{box-sizing:border-box}body{margin:0;background:#090909;color:#f5f5f5;height:100vh;overflow:hidden}.app{display:grid;grid-template-columns:380px 1fr;height:100vh}.panel{padding:20px;border-right:1px solid #292929;background:#111;overflow:auto}.panel h1{font-size:20px;margin:0 0 8px}.badge{display:inline-block;background:#142f25;color:#8ff0bd;padding:5px 8px;border-radius:999px;font-size:11px;margin:5px 0 12px}.sub{color:#aaa;font-size:12px;line-height:1.45}.modes{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:18px 0 12px}button{border:1px solid #3a3a3a;background:#202020;color:#eee;padding:9px;border-radius:7px;cursor:pointer}button.active{background:#fff;color:#111;border-color:#fff}.row{display:flex;gap:10px;align-items:center;margin:10px 0;font-size:13px}.row input[type=range]{flex:1}.metrics{border-top:1px solid #292929;margin-top:16px;padding-top:12px}.metric{display:flex;justify-content:space-between;gap:12px;padding:4px 0;font-size:12px}.metric span:first-child{color:#aaa}.note{margin-top:15px;padding:10px;border:1px solid #2b4439;background:#102119;color:#aee8c8;font-size:12px;line-height:1.4;border-radius:7px}.hash{font-family:ui-monospace,monospace;font-size:10px;color:#777;overflow-wrap:anywhere;margin-top:12px}.viewport{position:relative;overflow:hidden;background:#050505;cursor:grab}.viewport.dragging{cursor:grabbing}.stage{position:absolute;left:0;top:0;transform-origin:0 0}.stage img{position:absolute;inset:0;object-fit:fill;image-rendering:pixelated;pointer-events:none}.toolbar{position:absolute;right:14px;top:14px;display:flex;gap:6px;z-index:5}.toolbar button{background:#111d}.legend{font-size:11px;color:#aaa;min-height:30px}.hidden{display:none}
</style></head><body><div class="app"><aside class="panel"><h1>MapScan fine alignment</h1><div class="badge" id="badge"></div><div class="sub" id="instruction"></div><div class="modes"><button data-mode="fine" class="active">Refined (F)</button><button data-mode="before">Before (B)</button><button data-mode="blink">Blink (L)</button><button data-mode="anchors">Evidence (A)</button></div><div class="legend" id="description"></div><div class="row" id="unified-row"><input id="unified" type="checkbox" checked><label for="unified">Combined border (lime)</label></div><div class="row" id="hybrid-row"><input id="hybrid" type="checkbox"><label for="hybrid">Provenance (cyan county.png · yellow Census)</label></div><div class="row" id="census-row"><input id="census" type="checkbox"><label for="census">Full Census state (yellow audit)</label></div><div class="row"><input id="state" type="checkbox"><label for="state">Full county.png state (cyan audit)</label></div><div class="row"><input id="county" type="checkbox"><label for="county">county.png counties (magenta)</label></div><div class="row"><label>Overlay opacity</label><input id="opacity" type="range" min="0" max="1" value="0.90" step="0.01"></div><div class="metrics" id="metrics"></div><div class="note" id="note"></div><div class="hash" id="hash"></div></aside><main class="viewport" id="viewport"><div class="toolbar"><button id="fit">Fit</button><button id="coast">Coast</button><button id="eureka">Eureka</button><button id="bay">Bay</button><button id="tahoe">Tahoe</button><button id="se">Colorado</button><button id="sandiego">San Diego</button><button id="one">100%</button></div><div class="stage" id="stage"><img id="base"><img id="state-img"><img id="county-img"><img id="census-img"><img id="hybrid-img"><img id="unified-img"></div></main></div><script>
let payload,mode='fine',scale=1,tx=0,ty=0,drag=null,blinkTimer=null,blinkFine=true,W=1,H=1;const viewport=document.querySelector('#viewport'),stage=document.querySelector('#stage'),base=document.querySelector('#base'),censusImg=document.querySelector('#census-img'),stateImg=document.querySelector('#state-img'),countyImg=document.querySelector('#county-img'),hybridImg=document.querySelector('#hybrid-img'),unifiedImg=document.querySelector('#unified-img');
function size(w,h){W=w;H=h;stage.style.width=W+'px';stage.style.height=H+'px';for(const image of stage.querySelectorAll('img')){image.style.width=W+'px';image.style.height=H+'px'}}function transform(){stage.style.transform=`translate(${tx}px,${ty}px) scale(${scale})`}function fit(){scale=Math.min(viewport.clientWidth/W,viewport.clientHeight/H)*.98;tx=(viewport.clientWidth-W*scale)/2;ty=(viewport.clientHeight-H*scale)/2;transform()}function stopBlink(){if(blinkTimer){clearInterval(blinkTimer);blinkTimer=null}}
function focusSE(){scale=Math.max(.48,Math.min(1.15,viewport.clientHeight/(H*.34)));tx=viewport.clientWidth*.55-W*.94*scale;ty=viewport.clientHeight*.56-H*.90*scale;transform()}
function focusCoast(){scale=Math.max(.38,Math.min(.92,viewport.clientHeight/(H*.58)));tx=viewport.clientWidth*.58-W*.20*scale;ty=viewport.clientHeight*.52-H*.57*scale;transform()}
function focusPoint(nx,ny,spanY=.25){scale=Math.max(.55,Math.min(1.8,viewport.clientHeight/(H*spanY)));tx=viewport.clientWidth*.55-W*nx*scale;ty=viewport.clientHeight*.52-H*ny*scale;transform()}
function renderBase(){if(mode==='fine')base.src=payload.assets.fine;if(mode==='before')base.src=payload.assets.before;if(mode==='anchors')base.src=payload.assets.anchors_after;if(mode==='blink')base.src=blinkFine?payload.assets.fine:payload.assets.before}
function setMode(next){stopBlink();mode=next;document.querySelectorAll('[data-mode]').forEach(b=>b.classList.toggle('active',b.dataset.mode===mode));document.querySelector('#description').textContent=payload.mode_descriptions[mode];if(mode==='anchors'){size(payload.working_grid.width,payload.working_grid.height)}else{size(payload.grid.width,payload.grid.height)}renderBase();if(mode==='blink'){blinkFine=true;blinkTimer=setInterval(()=>{blinkFine=!blinkFine;renderBase()},650)}overlays()}
function overlays(){const show=mode!=='anchors',opacity=document.querySelector('#opacity').value;unifiedImg.style.display=show&&payload.assets.unified&&document.querySelector('#unified').checked?'block':'none';hybridImg.style.display=show&&payload.assets.hybrid&&document.querySelector('#hybrid').checked?'block':'none';censusImg.style.display=show&&payload.assets.census&&document.querySelector('#census').checked?'block':'none';stateImg.style.display=show&&document.querySelector('#state').checked?'block':'none';countyImg.style.display=show&&document.querySelector('#county').checked?'block':'none';unifiedImg.style.opacity=opacity;hybridImg.style.opacity=opacity;censusImg.style.opacity=opacity;stateImg.style.opacity=opacity;countyImg.style.opacity=opacity}
async function init(){payload=await(await fetch('/session.json')).json();document.querySelector('#badge').textContent=payload.badge;document.querySelector('#instruction').textContent=payload.instruction;document.querySelector('#note').textContent=payload.review_note;document.querySelector('#hash').textContent='candidate '+payload.candidate_alignment.sha256;stateImg.src=payload.assets.state;countyImg.src=payload.assets.county;if(payload.assets.census)censusImg.src=payload.assets.census;else document.querySelector('#census-row').style.display='none';if(payload.assets.hybrid)hybridImg.src=payload.assets.hybrid;else document.querySelector('#hybrid-row').style.display='none';if(payload.assets.unified)unifiedImg.src=payload.assets.unified;else document.querySelector('#unified-row').style.display='none';const m=payload.metrics,lower=payload.lower_colorado_refinement,items=[];if(payload.hybrid_perimeter){const h=payload.hybrid_perimeter;items.push(['Combined border',`${h.unified_border.union_pixel_count.toLocaleString()} exact union pixels`],['Coast · county.png',`${h.coast_county_png.median_px.toFixed(2)}/${h.coast_county_png.p90_px.toFixed(2)} med/P90 px`],['Coast · Census audit',`${h.coast_census.median_px.toFixed(2)}/${h.coast_census.p90_px.toFixed(2)} med/P90 px`],['Tahoe · county.png',`${h.Tahoe_county_png.median_px.toFixed(2)}/${h.Tahoe_county_png.p90_px.toFixed(2)} med/P90 px`],['Tahoe · Census audit',`${h.Tahoe_census.median_px.toFixed(2)}/${h.Tahoe_census.p90_px.toFixed(2)} med/P90 px`],['Tahoe windows',`${h.Tahoe_accepted_count}/${h.Tahoe_window_count} pass`],['San Diego seam',`${h.southern_coast_seam.endpoint_gap_px.toFixed(1)} px gap · ${h.southern_coast_seam.connector_pixel_count} px closure`],['Additional coastal warp',h.additional_warp?'proposed':'not needed'])}items.push([(lower?'Lower Colorado fit':'Matched junctions'),`${m.matched_before.median_px.toFixed(2)}/${m.matched_before.p90_px.toFixed(2)} → ${m.matched_after.median_px.toFixed(2)}/${m.matched_after.p90_px.toFixed(2)} med/P90 px`],[(lower?'Untouched windows':'Held-out regions'),`${m.holdout_regions_passed}/${m.holdout_region_count} pass`],['Held-out residual',`${m.holdout_before.median_px.toFixed(2)}/${m.holdout_before.p90_px.toFixed(2)} → ${m.holdout_after.median_px.toFixed(2)}/${m.holdout_after.p90_px.toFixed(2)}`],[(lower?'Mexico-border veto':'State veto'),`${m.state_before.median_px.toFixed(2)}/${m.state_before.p90_px.toFixed(2)} → ${m.state_after.median_px.toFixed(2)}/${m.state_after.p90_px.toFixed(2)}`],['Jacobian determinant',`${m.jacobian.minimum.toFixed(3)}–${m.jacobian.maximum.toFixed(3)}`]);if(m.local_scale)items.push(['Local scale',`${m.local_scale.minimum.toFixed(3)}–${m.local_scale.maximum.toFixed(3)}`]);items.push(['Repeated run',m.deterministic?'byte-identical':'FAILED']);document.querySelector('#metrics').innerHTML=items.map(([a,b])=>`<div class="metric"><span>${a}</span><span>${b}</span></div>`).join('');size(payload.grid.width,payload.grid.height);setMode('fine');payload.hybrid_perimeter?fit():(lower?focusSE():fit())}
document.querySelectorAll('[data-mode]').forEach(b=>b.onclick=()=>setMode(b.dataset.mode));document.querySelector('#unified').onchange=overlays;document.querySelector('#hybrid').onchange=overlays;document.querySelector('#census').onchange=overlays;document.querySelector('#state').onchange=overlays;document.querySelector('#county').onchange=overlays;document.querySelector('#opacity').oninput=overlays;document.querySelector('#fit').onclick=fit;document.querySelector('#coast').onclick=focusCoast;document.querySelector('#eureka').onclick=()=>focusPoint(.06,.16,.22);document.querySelector('#bay').onclick=()=>focusPoint(.18,.48,.26);document.querySelector('#tahoe').onclick=()=>focusPoint(.44,.32,.23);document.querySelector('#se').onclick=focusSE;document.querySelector('#sandiego').onclick=()=>focusPoint(.70,.96,.14);document.querySelector('#one').onclick=()=>{scale=1;tx=30;ty=30;transform()};viewport.onwheel=e=>{e.preventDefault();const rect=viewport.getBoundingClientRect(),x=e.clientX-rect.left,y=e.clientY-rect.top,old=scale;scale=Math.max(.06,Math.min(8,scale*Math.exp(-e.deltaY*.001)));tx=x-(x-tx)*scale/old;ty=y-(y-ty)*scale/old;transform()};viewport.onpointerdown=e=>{drag={x:e.clientX,y:e.clientY,tx,ty};viewport.setPointerCapture(e.pointerId);viewport.classList.add('dragging')};viewport.onpointermove=e=>{if(!drag)return;tx=drag.tx+e.clientX-drag.x;ty=drag.ty+e.clientY-drag.y;transform()};viewport.onpointerup=()=>{drag=null;viewport.classList.remove('dragging')};window.onresize=()=>{if(!payload)return;payload.hybrid_perimeter?fit():(payload.lower_colorado_refinement?focusSE():fit())};window.onkeydown=e=>{const key=e.key.toLowerCase();if(key==='f')setMode('fine');if(key==='b')setMode('before');if(key==='l')setMode('blink');if(key==='a')setMode('anchors')};init();
</script></body></html>"""
