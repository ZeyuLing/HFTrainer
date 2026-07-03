#!/usr/bin/env python3
"""Serve a lightweight PhysFlow Table-2 canonical NPZ comparison viewer."""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import urllib.parse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.embodied.physflow_triplet_manifest import (  # noqa: E402
    DEFAULT_BODIES,
    MESHES_BY_BODY,
    _parse_g1_body_meshes,
)


DEFAULT_SPLITS = ("lafan1_g1", "amass_test_g1", "wild_g1_clean")
DEFAULT_METHODS = ("reference", "any2track", "humanoid_gpt", "protomotions", "sonic", "beyondmimic")
METHOD_TITLES = {
    "reference": "Reference",
    "any2track": "Any2Track",
    "humanoid_gpt": "HumanoidGPT",
    "protomotions": "ProtoMotions",
    "sonic": "SONIC",
    "beyondmimic": "BeyondMimic",
}
METHOD_COLORS = {
    "reference": "#3f4752",
    "any2track": "#277da1",
    "humanoid_gpt": "#8d5fd3",
    "protomotions": "#d08b2f",
    "sonic": "#2a9d8f",
    "beyondmimic": "#c75146",
}


def _body_meta() -> list[dict[str, Any]]:
    try:
        bodies = _parse_g1_body_meshes()
        if [b["name"] for b in bodies] == DEFAULT_BODIES:
            return bodies
    except Exception:
        pass
    return [
        {
            "name": name,
            "meshes": [
                {"file": mesh, "pos": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0]}
                for mesh in MESHES_BY_BODY.get(name, [])
            ],
        }
        for name in DEFAULT_BODIES
    ]


def _npz_path(root: Path, method: str, split: str, case_id: str) -> Path:
    return root / method / split / "g1_body30" / f"{case_id}.npz"


def _list_cases(root: Path, methods: list[str], splits: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in splits:
        case_ids: set[str] = set()
        for method in methods:
            body_dir = root / method / split / "g1_body30"
            if body_dir.is_dir():
                case_ids.update(p.stem for p in body_dir.glob("*.npz") if p.is_file())
        for case_id in sorted(case_ids):
            available = [
                method for method in methods if _npz_path(root, method, split, case_id).is_file()
            ]
            if "reference" not in available or len(available) < 2:
                continue
            rows.append(
                {
                    "id": f"{split}/{case_id}",
                    "split": split,
                    "case_id": case_id,
                    "available_methods": available,
                    "num_methods": len(available),
                }
            )
    rows.sort(key=lambda r: (r["split"], -r["num_methods"], r["case_id"]))
    return rows


def _load_body(path: Path) -> dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    body_pos = np.asarray(data["body_pos"], dtype=np.float32)
    body_quat = np.asarray(data["body_quat"], dtype=np.float32)
    fps = float(np.asarray(data["fps" if "fps" in data.files else "frequency"]).reshape(-1)[0])
    body_names = [str(x) for x in np.asarray(data["body_names"], dtype=str).reshape(-1).tolist()]
    return {
        "body_pos": body_pos,
        "body_quat": body_quat,
        "fps": fps,
        "body_names": body_names,
    }


def _local_mpjpe(ref: np.ndarray, pred: np.ndarray) -> float:
    n = min(len(ref), len(pred))
    if n == 0:
        return float("nan")
    ref_local = ref[:n] - ref[:n, :1]
    pred_local = pred[:n] - pred[:n, :1]
    return float(np.linalg.norm(pred_local - ref_local, axis=-1).mean() * 1000.0)


def _metrics(ref: dict[str, Any], pred: dict[str, Any]) -> dict[str, Any]:
    ref_pos = ref["body_pos"]
    pred_pos = pred["body_pos"]
    n = min(len(ref_pos), len(pred_pos))
    if n == 0:
        return {"frames": 0, "completion": 0.0}
    global_mpjpe = float(np.linalg.norm(pred_pos[:n] - ref_pos[:n], axis=-1).mean() * 1000.0)
    root_err = float(np.linalg.norm(pred_pos[:n, 0] - ref_pos[:n, 0], axis=-1).mean())
    return {
        "frames": int(len(pred_pos)),
        "reference_frames": int(len(ref_pos)),
        "covered_frames": int(n),
        "completion": float(n / max(len(ref_pos), 1)),
        "global_mpjpe_mm": global_mpjpe,
        "local_mpjpe_mm": _local_mpjpe(ref_pos, pred_pos),
        "root_err_m": root_err,
        "fps": float(pred["fps"]),
    }


def _frames_payload(root: Path, split: str, case_id: str, method: str, bodies: list[dict[str, Any]]) -> dict[str, Any]:
    path = _npz_path(root, method, split, case_id)
    if not path.is_file():
        raise FileNotFoundError(path)
    motion = _load_body(path)
    return {
        "method": method,
        "title": METHOD_TITLES.get(method, method),
        "color": METHOD_COLORS.get(method, "#666666"),
        "split": split,
        "case_id": case_id,
        "path": str(path),
        "fps": float(motion["fps"]),
        "num_frames": int(motion["body_pos"].shape[0]),
        "body_names": motion["body_names"],
        "bodies": bodies,
        "body_pos": motion["body_pos"].round(5).tolist(),
        "body_quat": motion["body_quat"].round(6).tolist(),
    }


def _case_payload(root: Path, methods: list[str], split: str, case_id: str) -> dict[str, Any]:
    available = [method for method in methods if _npz_path(root, method, split, case_id).is_file()]
    ref = _load_body(_npz_path(root, "reference", split, case_id)) if "reference" in available else None
    metrics = {}
    if ref is not None:
        for method in available:
            if method == "reference":
                continue
            metrics[method] = _metrics(ref, _load_body(_npz_path(root, method, split, case_id)))
    return {
        "split": split,
        "case_id": case_id,
        "available_methods": available,
        "metrics": metrics,
    }


def _html() -> str:
    return r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>PhysFlow Table 2 Tracker Cases</title>
  <style>
    :root { --bg:#f5f2ea; --ink:#17201d; --muted:#68706c; --line:#c8c2b6; --panel:#fffdf7; --accent:#277da1; }
    * { box-sizing: border-box; }
    body { margin:0; min-height:100vh; background:var(--bg); color:var(--ink); font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; letter-spacing:0; }
    header { padding:18px 22px; border-bottom:1px solid var(--line); display:grid; grid-template-columns:minmax(0,1fr) auto; gap:16px; align-items:end; }
    h1 { margin:0; font-size:26px; line-height:1.1; }
    .sub { color:var(--muted); margin-top:6px; font-size:13px; }
    .status { font:12px ui-monospace, SFMono-Regular, Menlo, monospace; color:var(--muted); text-align:right; }
    main { display:grid; grid-template-columns:330px minmax(0,1fr); gap:12px; padding:12px; height:calc(100vh - 80px); }
    aside, .stage, .metrics { background:var(--panel); border:1px solid var(--line); }
    aside { overflow:hidden; display:flex; flex-direction:column; }
    .filters { padding:10px; border-bottom:1px solid var(--line); display:grid; gap:8px; }
    input, select { width:100%; min-height:34px; border:1px solid var(--line); background:#fff; padding:7px 8px; font:13px inherit; }
    #cases { overflow:auto; }
    .case { width:100%; text-align:left; border:0; border-bottom:1px solid var(--line); background:transparent; padding:10px; cursor:pointer; color:var(--ink); }
    .case.active { background:#e7f0f1; }
    .case strong { display:block; font-size:13px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .case small { display:block; margin-top:4px; color:var(--muted); font:11px ui-monospace, SFMono-Regular, Menlo, monospace; }
    .right { display:grid; grid-template-rows:minmax(0,1fr) 160px; gap:10px; min-width:0; }
    .stage { position:relative; overflow:hidden; }
    #canvas { display:block; width:100%; height:100%; }
    .toolbar { position:absolute; left:12px; top:12px; display:flex; flex-wrap:wrap; gap:8px; z-index:2; }
    button.toggle, button.control { border:1px solid var(--line); background:rgba(255,253,247,.94); color:var(--ink); min-height:34px; padding:7px 10px; cursor:pointer; font-weight:700; }
    button.toggle.off { opacity:.45; text-decoration:line-through; }
    .legend { position:absolute; right:12px; top:12px; background:rgba(255,253,247,.94); border:1px solid var(--line); padding:8px 10px; font-size:12px; line-height:1.7; z-index:2; }
    .dot { display:inline-block; width:9px; height:9px; margin-right:6px; border-radius:99px; }
    .metrics { overflow:auto; padding:10px; }
    table { width:100%; border-collapse:collapse; font-size:12px; }
    th, td { border-bottom:1px solid var(--line); padding:6px 8px; text-align:right; white-space:nowrap; }
    th:first-child, td:first-child { text-align:left; }
    .missing { color:#a34d3d; }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>PhysFlow Table 2 Tracker Case Comparison</h1>
      <div class="sub">Reference and completed tracker rollouts are rendered with the G1 mesh at 30 FPS. Methods missing for a case are left out rather than filled with stale artifacts.</div>
    </div>
    <div class="status" id="status">loading</div>
  </header>
  <main>
    <aside>
      <div class="filters">
        <select id="split"></select>
        <input id="search" placeholder="Search case id" />
      </div>
      <div id="cases"></div>
    </aside>
    <section class="right">
      <div class="stage">
        <canvas id="canvas"></canvas>
        <div class="toolbar" id="methodToggles"></div>
        <div class="legend" id="legend"></div>
      </div>
      <div class="metrics" id="metrics"></div>
    </section>
  </main>
  <script type="importmap">{"imports":{"three":"/assets/three/three.module.js","three/addons/":"/assets/three/jsm/"}}</script>
  <script type="module">
    import * as THREE from 'three';
    import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
    import { STLLoader } from 'three/addons/loaders/STLLoader.js';

    const state = { cases: [], current: null, motions: {}, visible: {}, frame: 0, playing: true, last: performance.now() };
    const canvas = document.getElementById('canvas');
    const renderer = new THREE.WebGLRenderer({canvas, antialias:true, preserveDrawingBuffer:true});
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    renderer.setClearColor(0xf5f2ea, 1);
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(45, 1, 0.05, 200);
    camera.position.set(6, -9, 4.2);
    const controls = new OrbitControls(camera, canvas);
    controls.target.set(0, 0, 1.0);
    controls.update();
    scene.add(new THREE.HemisphereLight(0xffffff, 0x89908b, 2.0));
    const sun = new THREE.DirectionalLight(0xffffff, 2.4); sun.position.set(4, -6, 8); scene.add(sun);
    const grid = new THREE.GridHelper(18, 18, 0xc0b8aa, 0xd8d2c7); grid.rotation.x = Math.PI / 2; scene.add(grid);
    const loader = new STLLoader();
    const actorRoot = new THREE.Group(); scene.add(actorRoot);
    const geoms = new Map();

    function qWXYZ(q){ return new THREE.Quaternion(q[1], q[2], q[3], q[0]); }
    function resize(){
      const w = canvas.clientWidth, h = canvas.clientHeight;
      if(canvas.width !== Math.floor(w*devicePixelRatio) || canvas.height !== Math.floor(h*devicePixelRatio)){
        renderer.setSize(w, h, false); camera.aspect = w / Math.max(h,1); camera.updateProjectionMatrix();
      }
    }
    async function geom(file){
      if(!geoms.has(file)) geoms.set(file, await loader.loadAsync('/assets/g1_mesh/' + file));
      return geoms.get(file);
    }
    async function buildActor(method, motion, offset){
      const group = new THREE.Group(); group.userData.method = method; group.position.x = offset;
      const material = new THREE.MeshStandardMaterial({color: motion.color || '#666', roughness:.72, metalness:.08});
      const bodyGroups = [];
      for(const body of motion.bodies){
        const bg = new THREE.Group(); group.add(bg); bodyGroups.push(bg);
        for(const m of body.meshes || []){
          const mesh = new THREE.Mesh(await geom(m.file), material);
          mesh.position.fromArray(m.pos || [0,0,0]);
          mesh.quaternion.copy(qWXYZ(m.quat || [1,0,0,0]));
          bg.add(mesh);
        }
      }
      group.userData.bodyGroups = bodyGroups;
      actorRoot.add(group);
      return group;
    }
    function applyFrame(group, motion, frame){
      const f = Math.min(frame, motion.num_frames - 1);
      const pos = motion.body_pos[f], quat = motion.body_quat[f];
      group.userData.bodyGroups.forEach((bg, i) => {
        bg.position.fromArray(pos[i]);
        bg.quaternion.copy(qWXYZ(quat[i]));
      });
    }
    function renderMetrics(caseInfo){
      const rows = Object.entries(caseInfo.metrics || {}).map(([m,x]) => `<tr><td>${m}</td><td>${(x.completion*100).toFixed(1)}%</td><td>${x.local_mpjpe_mm?.toFixed(1) ?? '-'}</td><td>${x.global_mpjpe_mm?.toFixed(1) ?? '-'}</td><td>${x.root_err_m?.toFixed(3) ?? '-'}</td><td>${x.frames}/${x.reference_frames}</td></tr>`).join('');
      document.getElementById('metrics').innerHTML = `<table><thead><tr><th>method</th><th>completion</th><th>local MPJPE mm</th><th>global MPJPE mm</th><th>root err m</th><th>frames</th></tr></thead><tbody>${rows || '<tr><td colspan="6">No metrics</td></tr>'}</tbody></table>`;
    }
    function renderCaseList(){
      const split = document.getElementById('split').value;
      const q = document.getElementById('search').value.toLowerCase();
      const rows = state.cases.filter(c => (!split || c.split === split) && c.case_id.toLowerCase().includes(q));
      document.getElementById('cases').innerHTML = rows.map(c => `<button class="case ${state.current?.id===c.id?'active':''}" data-id="${c.id}"><strong>${c.case_id}</strong><small>${c.split} · ${c.available_methods.join(', ')}</small></button>`).join('');
      document.querySelectorAll('.case').forEach(b => b.onclick = () => selectCase(b.dataset.id));
    }
    async function selectCase(id){
      const c = state.cases.find(x => x.id === id); if(!c) return;
      state.current = c; state.motions = {}; state.visible = {}; state.frame = 0; actorRoot.clear();
      renderCaseList(); document.getElementById('status').textContent = 'loading ' + id;
      const info = await (await fetch(`/api/case?split=${encodeURIComponent(c.split)}&case=${encodeURIComponent(c.case_id)}`)).json();
      renderMetrics(info);
      const methods = info.available_methods;
      const offsets = methods.map((_,i) => (i - (methods.length-1)/2) * 2.25);
      for(let i=0;i<methods.length;i++){
        const method = methods[i];
        state.visible[method] = true;
        const motion = await (await fetch(`/api/motion?split=${encodeURIComponent(c.split)}&case=${encodeURIComponent(c.case_id)}&method=${encodeURIComponent(method)}`)).json();
        motion.group = await buildActor(method, motion, offsets[i]);
        state.motions[method] = motion;
      }
      document.getElementById('methodToggles').innerHTML = methods.map(m => `<button class="toggle" data-method="${m}">${m}</button>`).join('') + '<button class="control" id="play">Pause</button>';
      document.querySelectorAll('.toggle').forEach(b => b.onclick = () => { const m=b.dataset.method; state.visible[m]=!state.visible[m]; b.classList.toggle('off', !state.visible[m]); state.motions[m].group.visible = state.visible[m]; });
      document.getElementById('play').onclick = () => { state.playing = !state.playing; document.getElementById('play').textContent = state.playing ? 'Pause' : 'Play'; };
      document.getElementById('legend').innerHTML = methods.map(m => `<div><span class="dot" style="background:${state.motions[m].color}"></span>${state.motions[m].title}</div>`).join('');
      document.getElementById('status').textContent = `${id} · ${methods.length} actors`;
    }
    async function init(){
      const payload = await (await fetch('/api/cases')).json();
      state.cases = payload.cases;
      const splits = [...new Set(state.cases.map(c => c.split))];
      document.getElementById('split').innerHTML = '<option value="">all splits</option>' + splits.map(s => `<option value="${s}">${s}</option>`).join('');
      document.getElementById('split').onchange = renderCaseList;
      document.getElementById('search').oninput = renderCaseList;
      renderCaseList();
      if(state.cases.length) await selectCase(state.cases[0].id);
    }
    function tick(now){
      resize();
      if(state.playing && now - state.last > 1000/30){ state.frame++; state.last = now; }
      for(const motion of Object.values(state.motions)) applyFrame(motion.group, motion, state.frame % motion.num_frames);
      controls.update(); renderer.render(scene, camera); requestAnimationFrame(tick);
    }
    init(); requestAnimationFrame(tick);
  </script>
</body>
</html>"""


class Handler(SimpleHTTPRequestHandler):
    root: Path
    methods: list[str]
    splits: list[str]
    bodies: list[dict[str, Any]]
    three_root: Path
    mesh_root: Path

    def _send_json(self, payload: Any) -> None:
        data = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_text(self, text: str, content_type: str = "text/html; charset=utf-8") -> None:
        data = text.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_file(self, path: Path) -> None:
        if not path.is_file():
            self.send_error(404, str(path))
            return
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(parsed.query)
        try:
            if parsed.path in {"/", "/index.html"}:
                self._send_text(_html())
            elif parsed.path == "/api/cases":
                cases = _list_cases(self.root, self.methods, self.splits)
                self._send_json({"root": str(self.root), "cases": cases, "methods": self.methods, "splits": self.splits})
            elif parsed.path == "/api/case":
                split = query.get("split", [""])[0]
                case_id = query.get("case", [""])[0]
                self._send_json(_case_payload(self.root, self.methods, split, case_id))
            elif parsed.path == "/api/motion":
                split = query.get("split", [""])[0]
                case_id = query.get("case", [""])[0]
                method = query.get("method", [""])[0]
                self._send_json(_frames_payload(self.root, split, case_id, method, self.bodies))
            elif parsed.path.startswith("/assets/three/"):
                self._send_file(self.three_root / parsed.path.removeprefix("/assets/three/"))
            elif parsed.path.startswith("/assets/g1_mesh/"):
                self._send_file(self.mesh_root / parsed.path.removeprefix("/assets/g1_mesh/"))
            else:
                self.send_error(404)
        except Exception as exc:  # noqa: BLE001
            self.send_error(500, repr(exc))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs/evaluation/physflow/table2_tracker"))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS))
    parser.add_argument("--three-root", type=Path, default=ROOT / "motion_annot_web/score_m2m/static/three")
    parser.add_argument(
        "--mesh-root",
        type=Path,
        default=ROOT / "hftrainer/models/motion/physflow/trackers/protomotions/vendor/protomotions/data/assets/mesh/G1",
    )
    args = parser.parse_args()

    Handler.root = args.root.resolve()
    Handler.methods = args.methods
    Handler.splits = args.splits
    Handler.bodies = _body_meta()
    Handler.three_root = args.three_root.resolve()
    Handler.mesh_root = args.mesh_root.resolve()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[physflow-table2-viewer] http://{args.host}:{args.port}/ root={Handler.root}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
