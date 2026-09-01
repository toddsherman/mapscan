"""Local paired-control-point interface for assisted map alignment."""

from __future__ import annotations

import json
import math
import mimetypes
import threading
import webbrowser
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image
from pyproj import Transformer
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import transform as transform_geometry

from .alignment import _reference_points
from .reference import iter_boundary_lines, load_california


REFERENCE_CRS = "EPSG:3310"

REFERENCE_LANDMARKS = (
    ("eureka", "Eureka", -124.1637, 40.8021),
    ("redding", "Redding", -122.3917, 40.5865),
    ("chico", "Chico", -121.8375, 39.7285),
    ("sacramento", "Sacramento", -121.4944, 38.5816),
    ("san-francisco", "San Francisco", -122.4194, 37.7749),
    ("san-jose", "San Jose", -121.8863, 37.3382),
    ("fresno", "Fresno", -119.7871, 36.7378),
    ("san-luis-obispo", "San Luis Obispo", -120.6596, 35.2828),
    ("bakersfield", "Bakersfield", -119.0187, 35.3733),
    ("santa-barbara", "Santa Barbara", -119.6982, 34.4208),
    ("los-angeles", "Los Angeles", -118.2437, 34.0522),
    ("san-diego", "San Diego", -117.1611, 32.7157),
)


def _largest_polygon(geometry):
    if isinstance(geometry, Polygon):
        return geometry
    if isinstance(geometry, MultiPolygon):
        return max(geometry.geoms, key=lambda item: item.area)
    raise TypeError(geometry.geom_type)


def _normalized_county_lines(state, counties) -> List[List[List[float]]]:
    transformer = Transformer.from_crs("EPSG:4269", REFERENCE_CRS, always_xy=True)
    projected_state = transform_geometry(transformer.transform, state)
    mainland = _largest_polygon(projected_state)
    min_x, min_y, max_x, max_y = mainland.bounds
    state_height = max_y - min_y
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    result: List[List[List[float]]] = []
    for county in counties:
        projected = transform_geometry(transformer.transform, county)
        for line in iter_boundary_lines(projected):
            coordinates = np.asarray(line.coords, dtype=np.float64)
            if len(coordinates) < 2:
                continue
            stride = max(1, math.ceil(len(coordinates) / 90))
            sampled = coordinates[::stride]
            if not np.array_equal(sampled[-1], coordinates[-1]):
                sampled = np.vstack((sampled, coordinates[-1]))
            sampled[:, 0] = (sampled[:, 0] - center_x) / state_height
            sampled[:, 1] = (center_y - sampled[:, 1]) / state_height
            result.append(np.round(sampled, 7).tolist())
    return result


def _normalized_landmarks(state) -> List[Dict[str, object]]:
    transformer = Transformer.from_crs("EPSG:4269", REFERENCE_CRS, always_xy=True)
    projected_state = transform_geometry(transformer.transform, state)
    mainland = _largest_polygon(projected_state)
    min_x, min_y, max_x, max_y = mainland.bounds
    state_height = max_y - min_y
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    landmarks: List[Dict[str, object]] = []
    for landmark_id, label, longitude, latitude in REFERENCE_LANDMARKS:
        x, y = transformer.transform(longitude, latitude)
        landmarks.append(
            {
                "id": landmark_id,
                "label": label,
                "reference": [
                    round((x - center_x) / state_height, 7),
                    round((center_y - y) / state_height, 7),
                ],
            }
        )
    return landmarks


def build_assist_payload(image_path: Path, reference_root: Path) -> Dict[str, object]:
    state, counties = load_california(reference_root)
    with Image.open(image_path) as image:
        width, height = image.size
    return {
        "source": {
            "url": "/source-image",
            "name": image_path.name,
            "width": width,
            "height": height,
        },
        "reference": {
            "name": "California — 2025 Census TIGER/Line",
            "crs": REFERENCE_CRS,
            "outline": np.round(_reference_points(state, REFERENCE_CRS, count=1000), 7).tolist(),
            "county_lines": _normalized_county_lines(state, counties),
            "landmarks": _normalized_landmarks(state),
        },
        "instructions": (
            "Click a landmark on the source, then the same landmark on the reference. "
            "Use 8–12 widely separated points when possible. Labeled city dots snap to exact "
            "reference coordinates; coastline bends and border intersections are also useful."
        ),
    }


def _fit_homography(
    reference_points: np.ndarray, source_points: np.ndarray
) -> Tuple[np.ndarray, Dict[str, float]]:
    if len(reference_points) < 4 or len(source_points) != len(reference_points):
        raise ValueError("At least four complete control-point pairs are required")
    matrix, _ = cv2.findHomography(reference_points, source_points, method=0)
    if matrix is None or not np.all(np.isfinite(matrix)):
        raise ValueError("The selected points do not define a valid projective transform")
    predicted = cv2.perspectiveTransform(
        reference_points.reshape((-1, 1, 2)).astype(np.float64), matrix
    ).reshape((-1, 2))
    residuals = np.linalg.norm(predicted - source_points, axis=1)
    return matrix, {
        "control_point_rms_px": float(np.sqrt(np.mean(np.square(residuals)))),
        "control_point_median_px": float(np.median(residuals)),
        "control_point_max_px": float(np.max(residuals)),
    }


def _write_assisted_result(
    image_path: Path,
    output_dir: Path,
    payload: Dict[str, object],
    pairs: List[Dict[str, object]],
) -> Dict[str, object]:
    control_pairs = [
        pair
        for pair in pairs
        if pair.get("role") != "validation" and not pair.get("landmark")
    ]
    validation_pairs = [pair for pair in pairs if pair not in control_pairs]
    reference_points = np.asarray(
        [pair["reference"] for pair in control_pairs], dtype=np.float64
    )
    source_points = np.asarray(
        [pair["source"] for pair in control_pairs], dtype=np.float64
    )
    matrix, metrics = _fit_homography(reference_points, source_points)
    validation_report = []
    if validation_pairs:
        validation_reference = np.asarray(
            [pair["reference"] for pair in validation_pairs], dtype=np.float64
        )
        validation_source = np.asarray(
            [pair["source"] for pair in validation_pairs], dtype=np.float64
        )
        validation_predicted = cv2.perspectiveTransform(
            validation_reference.reshape((-1, 1, 2)), matrix
        ).reshape((-1, 2))
        validation_residuals = np.linalg.norm(
            validation_predicted - validation_source, axis=1
        )
        for pair, predicted, residual in zip(
            validation_pairs, validation_predicted, validation_residuals
        ):
            validation_report.append(
                {
                    **pair,
                    "role": "validation",
                    "predicted_source": predicted.tolist(),
                    "residual_px": float(residual),
                }
            )
        metrics["validation_median_px"] = float(np.median(validation_residuals))
        metrics["validation_max_px"] = float(np.max(validation_residuals))
    report = {
        "schema_version": 1,
        "status": "diagnostic_only",
        "alignment_mode": "assisted",
        "transform_model": "projective_homography",
        "source": payload["source"],
        "reference": {
            "name": payload["reference"]["name"],
            "crs": payload["reference"]["crs"],
        },
        "control_points": control_pairs,
        "validation_points": validation_report,
        "reference_to_source_matrix": matrix.tolist(),
        "metrics": metrics,
        "warning": (
            "Control-point residual measures only the selected points. Boundary holdout "
            "inspection is still required before this transform can be accepted."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "assist-control-points.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    source_image = cv2.cvtColor(
        np.asarray(Image.open(image_path).convert("RGB")), cv2.COLOR_RGB2BGR
    )
    image = source_image.copy()
    outline = np.asarray(payload["reference"]["outline"], dtype=np.float64)
    transformed = cv2.perspectiveTransform(outline.reshape((-1, 1, 2)), matrix)
    cv2.polylines(
        image,
        [np.rint(transformed).astype(np.int32)],
        True,
        (255, 255, 0),
        max(2, round(max(image.shape[:2]) / 700)),
        cv2.LINE_AA,
    )
    for index, point in enumerate(source_points, start=1):
        location = tuple(np.rint(point).astype(int))
        cv2.circle(image, location, 7, (32, 32, 255), -1, cv2.LINE_AA)
        cv2.putText(
            image,
            str(index),
            (location[0] + 9, location[1] - 9),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (32, 32, 255),
            2,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(output_dir / "assisted-overlay.png"), image)

    # Produce the inverse view as well: source content warped into a canonical
    # California-Albers inspection canvas. This is diagnostic only; categorical
    # data will later be warped separately with nearest-neighbor resampling.
    outline = np.asarray(payload["reference"]["outline"], dtype=np.float64)
    min_u, min_v = outline.min(axis=0)
    max_u, max_v = outline.max(axis=0)
    canvas_width, canvas_height, margin = 850, 1050, 45
    canvas_scale = min(
        (canvas_width - 2 * margin) / (max_u - min_u),
        (canvas_height - 2 * margin) / (max_v - min_v),
    )
    reference_to_canvas = np.array(
        [
            [canvas_scale, 0, margin - min_u * canvas_scale],
            [0, canvas_scale, margin - min_v * canvas_scale],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    source_to_canvas = reference_to_canvas @ np.linalg.inv(matrix)
    source_rgba = cv2.cvtColor(source_image, cv2.COLOR_BGR2BGRA)
    warped = cv2.warpPerspective(
        source_rgba,
        source_to_canvas,
        (canvas_width, canvas_height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

    def canvas_points(points) -> np.ndarray:
        values = np.asarray(points, dtype=np.float64)
        return cv2.perspectiveTransform(
            values.reshape((-1, 1, 2)), reference_to_canvas
        ).astype(np.int32)

    for county_line in payload["reference"]["county_lines"]:
        cv2.polylines(warped, [canvas_points(county_line)], False, (255, 0, 255, 190), 1)
    cv2.polylines(
        warped,
        [canvas_points(payload["reference"]["outline"])],
        True,
        (255, 255, 0, 255),
        3,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(output_dir / "assisted-warped-inspection.png"), warped)
    return {"ok": True, "report": str(report_path), "metrics": metrics}


@dataclass
class AssistSession:
    image_path: Path
    output_dir: Path
    payload: Dict[str, object]


def _handler(session: AssistSession):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format_string: str, *args) -> None:
            print(f"[assist] {self.address_string()} {format_string % args}")

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
            elif self.path == "/session.json":
                self._send(json.dumps(session.payload).encode(), "application/json")
            elif self.path == "/source-image":
                content_type = mimetypes.guess_type(session.image_path.name)[0] or "image/png"
                self._send(session.image_path.read_bytes(), content_type)
            else:
                self._send(b"Not found", "text/plain", HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/save":
                self._send(b"Not found", "text/plain", HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                request = json.loads(self.rfile.read(length))
                result = _write_assisted_result(
                    session.image_path,
                    session.output_dir,
                    session.payload,
                    request.get("pairs", []),
                )
                self._send(json.dumps(result).encode(), "application/json")
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                body = json.dumps({"ok": False, "error": str(error)}).encode()
                self._send(body, "application/json", HTTPStatus.BAD_REQUEST)

    return Handler


def serve_assist(
    image_path: Path,
    reference_root: Path,
    output_dir: Path,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
) -> None:
    session = AssistSession(
        image_path=image_path.resolve(),
        output_dir=output_dir.resolve(),
        payload=build_assist_payload(image_path, reference_root),
    )
    server = ThreadingHTTPServer((host, port), _handler(session))
    url = f"http://{host}:{server.server_port}"
    print(f"MapScan assisted alignment: {url}")
    print("Press Ctrl-C to stop after saving the control points.")
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
  <title>MapScan assisted alignment</title>
  <style>
    :root { color-scheme: light; --ink:#161616; --muted:#686868; --line:#d7d7d7; --accent:#0068d9; }
    * { box-sizing:border-box; }
    body { margin:0; font:14px/1.45 ui-sans-serif,system-ui,-apple-system,sans-serif; color:var(--ink); background:#f4f4f1; }
    header { display:flex; justify-content:space-between; gap:24px; padding:18px 24px; border-bottom:1px solid var(--line); background:#fff; }
    h1 { margin:0 0 5px; font-size:20px; letter-spacing:-.02em; }
    p { margin:0; color:var(--muted); max-width:850px; }
    .actions { display:flex; align-items:center; gap:8px; white-space:nowrap; }
    button { border:1px solid #aaa; border-radius:7px; padding:8px 12px; background:#fff; color:var(--ink); cursor:pointer; }
    button.primary { color:#fff; border-color:var(--accent); background:var(--accent); }
    button:disabled { opacity:.45; cursor:not-allowed; }
    main { padding:16px 24px 24px; }
    .status { margin-bottom:12px; padding:9px 12px; border:1px solid var(--line); border-radius:8px; background:#fff; }
    .workspace { display:grid; grid-template-columns:minmax(0,1fr) minmax(360px,.63fr); gap:16px; align-items:start; }
    .panel { overflow:hidden; border:1px solid var(--line); border-radius:9px; background:#fff; }
    .panel-head { display:flex; align-items:center; justify-content:space-between; gap:12px; padding:8px 12px; border-bottom:1px solid var(--line); }
    .panel h2 { margin:0; font-size:14px; }
    .zoom { display:flex; align-items:center; gap:6px; color:var(--muted); font-size:12px; }
    .zoom select { border:1px solid #aaa; border-radius:6px; padding:4px 6px; background:#fff; }
    .canvas-wrap { position:relative; max-height:72vh; overflow:auto; background:#deded9; }
    canvas { display:block; width:100%; height:auto; cursor:crosshair; }
    .pairs { margin-top:16px; border:1px solid var(--line); border-radius:9px; background:#fff; overflow:hidden; }
    table { width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums; }
    th,td { padding:8px 10px; text-align:left; border-bottom:1px solid #eee; }
    th { color:var(--muted); font-weight:600; }
    .dot { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:7px; }
    @media (max-width:900px) { .workspace { grid-template-columns:1fr; } header { flex-direction:column; } }
  </style>
</head>
<body>
  <header>
    <div><h1>MapScan assisted alignment</h1><p id="instructions">Loading…</p></div>
    <div class="actions">
      <button id="undo">Undo</button><button id="reset">Reset</button>
      <button id="save" class="primary" disabled>Save transform</button>
    </div>
  </header>
  <main>
    <div class="status" id="status">Loading source and reference geometry…</div>
    <section class="workspace">
      <article class="panel"><div class="panel-head"><h2 id="source-title">1. Source map</h2><label class="zoom">Zoom <select id="source-zoom"><option value="100">1×</option><option value="200">2×</option><option value="400">4×</option><option value="800">8×</option></select></label></div><div class="canvas-wrap"><canvas id="source"></canvas></div></article>
      <article class="panel"><div class="panel-head"><h2 id="reference-title">2. Authoritative reference</h2><label class="zoom">Zoom <select id="reference-zoom"><option value="100">1×</option><option value="200">2×</option><option value="400">4×</option></select></label></div><div class="canvas-wrap"><canvas id="reference" width="700" height="900"></canvas></div></article>
    </section>
    <section class="pairs"><table><thead><tr><th>Pair</th><th>Source x, y</th><th>Reference u, v</th></tr></thead><tbody id="pair-list"><tr><td colspan="3">No control points yet.</td></tr></tbody></table></section>
  </main>
<script>
const palette=['#e53e3e','#3182ce','#38a169','#d69e2e','#805ad5','#dd6b20','#319795','#d53f8c'];
const state={payload:null,image:null,pairs:[],pending:null,refView:null,matrix:null};
const source=document.querySelector('#source'), reference=document.querySelector('#reference');
const sctx=source.getContext('2d'), rctx=reference.getContext('2d');
const statusEl=document.querySelector('#status'), saveButton=document.querySelector('#save');

function canvasPoint(event,canvas){const r=canvas.getBoundingClientRect();return [(event.clientX-r.left)*canvas.width/r.width,(event.clientY-r.top)*canvas.height/r.height];}
function refToCanvas(point){const v=state.refView;return [v.margin+(point[0]-v.minU)*v.scale,v.margin+(point[1]-v.minV)*v.scale];}
function canvasToRef(point){const v=state.refView;return [(point[0]-v.margin)/v.scale+v.minU,(point[1]-v.margin)/v.scale+v.minV];}
function project(point,m){const d=m[6]*point[0]+m[7]*point[1]+m[8];return [(m[0]*point[0]+m[1]*point[1]+m[2])/d,(m[3]*point[0]+m[4]*point[1]+m[5])/d];}

function solveLinear(a,b){const n=b.length,m=a.map((row,i)=>[...row,b[i]]);for(let c=0;c<n;c++){let p=c;for(let r=c+1;r<n;r++)if(Math.abs(m[r][c])>Math.abs(m[p][c]))p=r;[m[c],m[p]]=[m[p],m[c]];if(Math.abs(m[c][c])<1e-10)return null;for(let j=c;j<=n;j++)m[c][j]/=m[c][c];for(let r=0;r<n;r++){if(r===c)continue;const f=m[r][c];for(let j=c;j<=n;j++)m[r][j]-=f*m[c][j];}}return m.map(row=>row[n]);}
function fitHomography(){const controls=state.pairs.filter(pair=>pair.role!=='validation');if(controls.length<4)return null;const A=[],b=[];for(const pair of controls){const [u,v]=pair.reference,[x,y]=pair.source;A.push([u,v,1,0,0,0,-x*u,-x*v]);b.push(x);A.push([0,0,0,u,v,1,-y*u,-y*v]);b.push(y);}const ata=Array.from({length:8},()=>Array(8).fill(0)),atb=Array(8).fill(0);for(let r=0;r<A.length;r++)for(let i=0;i<8;i++){atb[i]+=A[r][i]*b[r];for(let j=0;j<8;j++)ata[i][j]+=A[r][i]*A[r][j];}const h=solveLinear(ata,atb);return h?[...h,1]:null;}

function drawPath(ctx,points,convert){if(!points.length)return;const first=convert(points[0]);ctx.beginPath();ctx.moveTo(...first);for(let i=1;i<points.length;i++)ctx.lineTo(...convert(points[i]));ctx.stroke();}
function drawSource(){if(!state.image)return;sctx.clearRect(0,0,source.width,source.height);sctx.drawImage(state.image,0,0);state.matrix=fitHomography();if(state.matrix){sctx.save();sctx.strokeStyle='#00e7e7';sctx.lineWidth=Math.max(2,source.height/600);sctx.setLineDash([10,6]);drawPath(sctx,state.payload.reference.outline,p=>project(p,state.matrix));sctx.restore();}state.pairs.forEach((pair,i)=>drawMarker(sctx,pair.source,i));if(state.pending)drawMarker(sctx,state.pending,state.pairs.length,true);}
function drawReference(){rctx.fillStyle='#f7f7f3';rctx.fillRect(0,0,reference.width,reference.height);rctx.save();rctx.strokeStyle='#c4c4be';rctx.lineWidth=1;for(const line of state.payload.reference.county_lines)drawPath(rctx,line,refToCanvas);rctx.strokeStyle='#191919';rctx.lineWidth=2.5;drawPath(rctx,state.payload.reference.outline,refToCanvas);rctx.restore();for(const landmark of state.payload.reference.landmarks||[]){const point=refToCanvas(landmark.reference);rctx.save();rctx.fillStyle='#0068d9';rctx.strokeStyle='#fff';rctx.lineWidth=1.5;rctx.beginPath();rctx.arc(point[0],point[1],4,0,Math.PI*2);rctx.fill();rctx.stroke();rctx.fillStyle='#17446f';rctx.font='11px system-ui';rctx.textAlign='left';rctx.textBaseline='bottom';rctx.fillText(landmark.label,point[0]+6,point[1]-3);rctx.restore();}state.pairs.forEach((pair,i)=>drawMarker(rctx,refToCanvas(pair.reference),i));}
function drawMarker(ctx,point,index,pending=false){const color=palette[index%palette.length];ctx.save();ctx.fillStyle=pending?'#fff':color;ctx.strokeStyle=color;ctx.lineWidth=3;ctx.beginPath();ctx.arc(point[0],point[1],9,0,Math.PI*2);ctx.fill();ctx.stroke();ctx.fillStyle=pending?color:'#fff';ctx.font='bold 12px system-ui';ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(String(index+1),point[0],point[1]+.5);ctx.restore();}
function snapReference(point){let best=null;for(const landmark of state.payload.reference.landmarks||[]){const canvas=refToCanvas(landmark.reference),distance=Math.hypot(canvas[0]-point[0],canvas[1]-point[1]);if(!best||distance<best.distance)best={distance,landmark};}return best&&best.distance<=16?{reference:best.landmark.reference,landmark:best.landmark.label}:{reference:canvasToRef(point),landmark:null};}
function update(){state.matrix=fitHomography();drawSource();drawReference();const controlCount=state.pairs.filter(pair=>pair.role!=='validation').length,validationCount=state.pairs.length-controlCount;saveButton.disabled=controlCount<4||!state.matrix;statusEl.textContent=state.pending?`Pair ${state.pairs.length+1}: now click the same landmark on the reference.`:controlCount<4?`${controlCount} geographic control${controlCount===1?'':'s'} saved. Add ${4-controlCount} more.`:`${controlCount} geographic controls${validationCount?` + ${validationCount} city validation points`:''}. Cyan uses geographic controls only.`;const body=document.querySelector('#pair-list');body.innerHTML=state.pairs.length?state.pairs.map((p,i)=>`<tr><td><span class="dot" style="background:${palette[i%palette.length]}"></span>${i+1}</td><td>${p.source.map(n=>n.toFixed(1)).join(', ')}</td><td>${p.landmark?`${p.landmark} · validation · `:''}${p.reference.map(n=>n.toFixed(5)).join(', ')}</td></tr>`).join(''):'<tr><td colspan="3">No control points yet.</td></tr>';}
source.addEventListener('click',event=>{if(state.pending){statusEl.textContent='Finish the pending pair on the reference first.';return;}state.pending=canvasPoint(event,source);update();});
reference.addEventListener('click',event=>{if(!state.pending){statusEl.textContent='Click a source landmark first.';return;}const snapped=snapReference(canvasPoint(event,reference));state.pairs.push({source:state.pending,reference:snapped.reference,landmark:snapped.landmark,role:snapped.landmark?'validation':'control'});state.pending=null;update();});
document.querySelector('#source-zoom').onchange=event=>{source.style.width=`${event.target.value}%`;};
document.querySelector('#reference-zoom').onchange=event=>{reference.style.width=`${event.target.value}%`;};
document.querySelector('#undo').onclick=()=>{if(state.pending)state.pending=null;else state.pairs.pop();update();};
document.querySelector('#reset').onclick=()=>{state.pending=null;state.pairs=[];update();};
saveButton.onclick=async()=>{saveButton.disabled=true;statusEl.textContent='Saving and validating transform…';try{const response=await fetch('/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({pairs:state.pairs})});const result=await response.json();if(!response.ok)throw new Error(result.error||'Save failed');statusEl.textContent=`Saved. Control-point RMS: ${result.metrics.control_point_rms_px.toFixed(2)} px. Diagnostic: ${result.report}`;}catch(error){statusEl.textContent=`Could not save: ${error.message}`;}finally{saveButton.disabled=false;}};

Promise.all([fetch('/session.json').then(r=>r.json()),new Promise((resolve,reject)=>{const image=new Image();image.onload=()=>resolve(image);image.onerror=reject;image.src='/source-image';})]).then(([payload,image])=>{state.payload=payload;state.image=image;source.width=payload.source.width;source.height=payload.source.height;document.querySelector('#instructions').textContent=payload.instructions;document.querySelector('#source-title').textContent=`1. Source map — ${payload.source.name}`;document.querySelector('#reference-title').textContent=`2. ${payload.reference.name}`;const all=payload.reference.outline;const us=all.map(p=>p[0]),vs=all.map(p=>p[1]),minU=Math.min(...us),maxU=Math.max(...us),minV=Math.min(...vs),maxV=Math.max(...vs),margin=35,scale=Math.min((reference.width-2*margin)/(maxU-minU),(reference.height-2*margin)/(maxV-minV));state.refView={minU,minV,maxU,maxV,margin,scale};update();}).catch(error=>{statusEl.textContent=`Failed to load: ${error.message}`;});
</script>
</body></html>"""
