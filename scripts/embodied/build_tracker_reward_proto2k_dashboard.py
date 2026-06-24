#!/usr/bin/env python3
"""Build a static dashboard for the tracker_reward_proto_2k PhysFlow run."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any


ROOT = Path("/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer")
LOG_PATH = ROOT / "work_dirs/physflow_trreward_proto2k_fix2_0617.log"
EVAL_ROOT = ROOT / "output/physflow_verify_hymotion_g1_130k_safe"
BASE_SUMMARY = EVAL_ROOT / "base130k_frozen_eval/summary.json"
PROTO_SUMMARY = EVAL_ROOT / "proto_iter2000_frozen_eval/summary.json"
PROTO_GENERATED = EVAL_ROOT / "proto_iter2000_frozen_eval/generated/summary.json"
OUT_DIR = ROOT / "output/physflow_visualizations/tracker_reward_proto_2k"
OUT_HTML = OUT_DIR / "index.html"
THREE_SRC = ROOT / "motion_annot_web/score_m2m/static/three"
G1_MESH_SRC = (
    ROOT
    / "hftrainer/models/motion/physflow/trackers/protomotions/vendor/protomotions/data/assets/mesh/G1"
)

STEP_RE = re.compile(r"step \[(\d+)/(\d+)\]\s+(.*)")
KV_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)")

METRICS = [
    ("completion_mean", "Completion", "higher"),
    ("fall_rate", "Fall rate", "lower"),
    ("adversarial_score_mean", "Adv score", "lower"),
    ("max_joint_error_rad_mean", "Max joint err", "lower"),
    ("root_trajectory_error_mean_m", "Root traj err", "lower"),
    ("trackable_basic_rate", "Trackable basic", "higher"),
    ("foot_skate_speed_mean", "Foot skate", "lower"),
    ("joint_vel_max_mean", "Joint velocity", "lower"),
]

def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def parse_training_log(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for line in path.read_text(errors="ignore").splitlines():
        match = STEP_RE.search(line)
        if not match:
            continue
        row: dict[str, float] = {
            "step": float(match.group(1)),
            "total": float(match.group(2)),
        }
        for key, value in KV_RE.findall(match.group(3)):
            try:
                row[key] = float(value)
            except ValueError:
                pass
        rows.append(row)
    return rows


def moving_average(rows: list[dict[str, float]], key: str, window: int = 25) -> list[list[float]]:
    out: list[list[float]] = []
    vals: list[float] = []
    for row in rows:
        if key not in row:
            continue
        vals.append(row[key])
        recent = vals[-window:]
        out.append([row["step"], sum(recent) / len(recent)])
    return out


def downsample_frames(frames: list[dict[str, Any]], limit: int = 120) -> list[dict[str, Any]]:
    if len(frames) <= limit:
        chosen = frames
    else:
        chosen = [frames[round(i * (len(frames) - 1) / (limit - 1))] for i in range(limit)]
    return [{"pos": frame["body_pos"], "quat": frame["body_quat"]} for frame in chosen]


def frame_bounds(samples: list[dict[str, Any]]) -> dict[str, float]:
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    for sample in samples:
        for frame in sample["frames"]:
            for x, y, z in frame["pos"]:
                xs.append(x)
                ys.append(y)
                zs.append(z)
    if not xs:
        return {"x0": -1, "x1": 1, "y0": -1, "y1": 1, "z0": 0, "z1": 2}
    pad = 0.15
    return {
        "x0": min(xs) - pad,
        "x1": max(xs) + pad,
        "y0": min(ys) - pad,
        "y1": max(ys) + pad,
        "z0": min(zs) - pad,
        "z1": max(zs) + pad,
    }


def load_samples() -> list[dict[str, Any]]:
    summary = read_json(PROTO_GENERATED)
    records = [r for r in summary["records"] if r.get("status") == "scored"]
    ranked = sorted(records, key=lambda r: float(r["adversarial_score"]))
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rec in ranked[:3] + ranked[-3:]:
        stem = rec["output_stem"]
        if stem in seen:
            continue
        seen.add(stem)
        robot_path = Path(rec["robot_json_path"])
        robot = read_json(robot_path)
        selected.append(
            {
                "stem": stem,
                "prompt": rec["prompt"],
                "score": rec["adversarial_score"],
                "completion": rec["completion_ratio"],
                "fall": bool(rec["fall_detected"]),
                "maxJointError": rec["max_joint_error_rad"],
                "rootError": rec["root_trajectory_error_mean_m"],
                "footSkate": rec.get("kinematic", {}).get("foot_skate_speed"),
                "jointVel": rec.get("kinematic", {}).get("joint_vel_max"),
                "fps": robot.get("fps", 12.5),
                "bodies": [
                    {
                        "name": body["name"],
                        "meshes": [
                            {
                                "file": mesh["file"],
                                "pos": mesh.get("pos", [0.0, 0.0, 0.0]),
                                "quat": mesh.get("quat", [1.0, 0.0, 0.0, 0.0]),
                            }
                            for mesh in body.get("meshes", [])
                        ],
                    }
                    for body in robot["bodies"]
                ],
                "bodyNames": [body["name"] for body in robot["bodies"]],
                "meshParts": sum(len(body.get("meshes", [])) for body in robot["bodies"]),
                "frames": downsample_frames(robot["frames"]),
            }
        )
    bounds = frame_bounds(selected)
    for sample in selected:
        sample["bounds"] = bounds
    return selected


def pct_delta(base: float, value: float, direction: str) -> float:
    if base == 0:
        return 0.0
    raw = (value - base) / abs(base)
    return raw if direction == "higher" else -raw


def build_data() -> dict[str, Any]:
    base = read_json(BASE_SUMMARY)["generated"]
    proto = read_json(PROTO_SUMMARY)["generated"]
    rows = parse_training_log(LOG_PATH)
    final = rows[-1] if rows else {}
    eval_metrics = []
    for key, label, direction in METRICS:
        b = float(base[key])
        p = float(proto[key])
        eval_metrics.append(
            {
                "key": key,
                "label": label,
                "base": b,
                "proto": p,
                "delta": p - b,
                "direction": direction,
                "improvement": pct_delta(b, p, direction),
            }
        )
    return {
        "run": {
            "name": "tracker_reward_proto_2k",
            "status": "completed",
            "step": int(final.get("step", 2000)),
            "total": int(final.get("total", 2000)),
            "log": str(LOG_PATH),
            "checkpoint": str(ROOT / "work_dirs/physflow_verify_hymotion_g1_protomotions_130k_formalfix2_0617/checkpoint-iter_2000"),
            "eval": str(PROTO_SUMMARY),
        },
        "final": final,
        "series": {
            "loss": moving_average(rows, "loss"),
            "rewardBest": moving_average(rows, "reward_best_mean"),
            "rewardCand": moving_average(rows, "reward_cand_mean"),
            "nGood": moving_average(rows, "n_good"),
            "jointStd": moving_average(rows, "sel_joint_std_mean"),
        },
        "evalMetrics": eval_metrics,
        "samples": load_samples(),
    }


def copy_assets() -> None:
    asset_dir = OUT_DIR / "assets"
    for rel in [
        Path("three.module.js"),
        Path("jsm/controls/OrbitControls.js"),
        Path("jsm/loaders/STLLoader.js"),
    ]:
        src = THREE_SRC / rel
        dst = asset_dir / "three" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    mesh_dir = asset_dir / "g1_mesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    for src in sorted(G1_MESH_SRC.glob("*.stl")):
        shutil.copy2(src, mesh_dir / src.name)


def html_doc(data: dict[str, Any]) -> str:  # noqa: F811 - overrides the legacy 2D version above.
    payload = json.dumps(data, ensure_ascii=False)
    payload = payload.replace("</", "<\\/")
    template = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>tracker_reward_proto_2k G1 mesh</title>
  <style>
    :root {
      --paper: #f4f6f1;
      --ink: #141712;
      --muted: #657067;
      --line: #cbd4cc;
      --panel: #ffffff;
      --teal: #087c75;
      --rust: #bf4f2e;
      --leaf: #52783e;
      --amber: #a57812;
      --violet: #6b5aa0;
      --night: #0d1210;
      --shadow: 0 14px 42px rgba(27, 41, 32, .13);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      background:
        linear-gradient(90deg, rgba(20,23,18,.034) 1px, transparent 1px) 0 0 / 24px 24px,
        linear-gradient(0deg, rgba(20,23,18,.026) 1px, transparent 1px) 0 0 / 24px 24px,
        var(--paper);
      font-family: Avenir Next, Segoe UI, sans-serif;
      letter-spacing: 0;
    }
    header {
      display: grid;
      grid-template-columns: minmax(0, 1.15fr) minmax(300px, .85fr);
      gap: 28px;
      padding: 26px clamp(18px, 4vw, 56px) 16px;
      border-bottom: 1px solid var(--line);
    }
    h1 {
      margin: 0;
      font-family: Georgia, Cambria, serif;
      font-size: clamp(32px, 5vw, 66px);
      line-height: .94;
      font-weight: 700;
    }
    .subtitle {
      margin: 12px 0 0;
      color: var(--muted);
      max-width: 760px;
      font-size: 15px;
      line-height: 1.5;
    }
    .nav {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-bottom: 14px;
    }
    .nav a {
      border: 1px solid var(--line);
      background: #fbfcfa;
      color: var(--ink);
      min-height: 34px;
      padding: 8px 10px;
      font: 700 12px ui-monospace, SFMono-Regular, Menlo, monospace;
      text-decoration: none;
      display: inline-flex;
      align-items: center;
    }
    .nav a.active {
      background: var(--ink);
      color: #fff;
      border-color: var(--ink);
    }
    .status-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      align-content: end;
    }
    .stat {
      min-height: 82px;
      padding: 14px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,.78);
      box-shadow: var(--shadow);
    }
    .stat span {
      display: block;
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: .08em;
    }
    .stat strong {
      display: block;
      margin-top: 8px;
      font: 700 23px Georgia, serif;
      white-space: nowrap;
    }
    main {
      padding: 0 clamp(18px, 4vw, 56px) 46px;
    }
    .compare-callout {
      margin: 18px 0;
      border: 1px solid var(--line);
      background: rgba(255,255,255,.88);
      box-shadow: var(--shadow);
      padding: 13px 15px;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.45;
    }
    .compare-callout strong {
      color: var(--ink);
    }
    .compare-callout a {
      color: var(--teal);
      font-weight: 700;
      text-decoration: none;
    }
    .mesh-stage {
      position: relative;
      height: clamp(520px, 72vh, 790px);
      min-height: 500px;
      margin: 0 calc(-1 * clamp(18px, 4vw, 56px)) 26px;
      overflow: hidden;
      background:
        linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,0) 28%),
        var(--night);
      border-bottom: 1px solid rgba(255,255,255,.16);
    }
    #meshCanvas {
      width: 100%;
      height: 100%;
      display: block;
    }
    .mesh-toolbar,
    .sample-list,
    .mesh-footer {
      position: absolute;
      z-index: 2;
      color: #f6f7f1;
      backdrop-filter: blur(14px);
      background: rgba(13, 18, 16, .74);
      border: 1px solid rgba(255,255,255,.16);
      box-shadow: 0 18px 45px rgba(0,0,0,.28);
    }
    .mesh-toolbar {
      top: 16px;
      right: 16px;
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 8px;
    }
    .sample-list {
      top: 16px;
      left: 16px;
      bottom: 16px;
      width: min(334px, calc(100vw - 32px));
      overflow: auto;
      padding: 8px;
    }
    .mesh-footer {
      left: 370px;
      right: 16px;
      bottom: 16px;
      display: grid;
      grid-template-columns: minmax(180px, 1fr) minmax(320px, .8fr);
      gap: 14px;
      padding: 12px;
      align-items: center;
    }
    .sample-row {
      width: 100%;
      text-align: left;
      border: 1px solid transparent;
      border-bottom-color: rgba(255,255,255,.12);
      background: transparent;
      color: #f6f7f1;
      padding: 10px;
      font-family: Avenir Next, Segoe UI, sans-serif;
      font-weight: 700;
      cursor: pointer;
    }
    .sample-row.active {
      background: rgba(8,124,117,.23);
      border-color: rgba(109,211,199,.42);
    }
    .sample-row span,
    .sample-row small {
      display: block;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .sample-row span { white-space: nowrap; }
    .sample-row small {
      margin-top: 5px;
      color: rgba(246,247,241,.72);
      font: 11px ui-monospace, SFMono-Regular, Menlo, monospace;
      line-height: 1.35;
      display: -webkit-box;
      -webkit-line-clamp: 2;
      -webkit-box-orient: vertical;
    }
    .mesh-title {
      min-width: 0;
      font: 700 13px ui-monospace, SFMono-Regular, Menlo, monospace;
      color: #f6f7f1;
      overflow-wrap: anywhere;
    }
    .mesh-title strong {
      display: block;
      font: 700 18px Georgia, serif;
      margin-bottom: 2px;
    }
    .mesh-meta {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px;
    }
    .mini {
      min-width: 0;
      color: rgba(246,247,241,.62);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: .07em;
    }
    .mini strong {
      display: block;
      margin-top: 4px;
      color: #f6f7f1;
      font: 700 14px ui-monospace, SFMono-Regular, Menlo, monospace;
      letter-spacing: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .toolbar-group {
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }
    button, select, input[type="range"] { accent-color: var(--teal); }
    button, select {
      border: 1px solid rgba(255,255,255,.2);
      background: rgba(255,255,255,.08);
      color: #f7f8f2;
      padding: 8px 10px;
      min-height: 34px;
      font: 700 12px ui-monospace, SFMono-Regular, Menlo, monospace;
      cursor: pointer;
    }
    button.active {
      background: #f6f7f1;
      color: #0d1210;
      border-color: #f6f7f1;
    }
    input[type="range"] {
      width: 100%;
      min-width: 0;
    }
    .metrics {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 22px;
    }
    .metric {
      border: 1px solid var(--line);
      background: var(--panel);
      padding: 14px;
      min-height: 112px;
      position: relative;
      overflow: hidden;
      box-shadow: var(--shadow);
    }
    .metric::after {
      content: "";
      position: absolute;
      left: 0;
      bottom: 0;
      height: 5px;
      width: var(--w);
      background: var(--c);
    }
    .metric .label {
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: .08em;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .metric .value {
      margin-top: 10px;
      font: 700 26px ui-monospace, SFMono-Regular, Menlo, monospace;
    }
    .metric .delta {
      margin-top: 8px;
      color: var(--muted);
      font-size: 13px;
    }
    .grid {
      display: grid;
      grid-template-columns: minmax(0, 1.05fr) minmax(360px, .95fr);
      gap: 18px;
    }
    section {
      border: 1px solid var(--line);
      background: rgba(255,255,255,.86);
      box-shadow: var(--shadow);
      min-width: 0;
    }
    .section-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
    }
    h2 {
      margin: 0;
      font: 700 18px Georgia, serif;
    }
    .chart-controls {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      justify-content: flex-end;
    }
    .chart-controls button {
      color: var(--ink);
      background: #fbfcfa;
      border-color: var(--line);
    }
    .chart-controls button.active {
      background: var(--ink);
      color: #fff;
      border-color: var(--ink);
    }
    canvas.chart {
      width: 100%;
      height: 360px;
      display: block;
    }
    .logline {
      margin-top: 18px;
      padding: 12px 14px;
      border: 1px solid var(--line);
      background: #fdfefb;
      color: var(--muted);
      font: 12px ui-monospace, SFMono-Regular, Menlo, monospace;
      overflow-wrap: anywhere;
    }
    @media (max-width: 1080px) {
      header, .grid { grid-template-columns: 1fr; }
      .metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .mesh-footer {
        left: 16px;
        grid-template-columns: 1fr;
      }
      .sample-list {
        top: auto;
        right: 16px;
        bottom: 112px;
        left: 16px;
        width: auto;
        max-height: 190px;
      }
    }
    @media (max-width: 680px) {
      header {
        padding-top: 20px;
      }
      .status-grid,
      .metrics,
      .mesh-meta {
        grid-template-columns: 1fr;
      }
      h1 { font-size: 34px; }
      .mesh-stage {
        height: 780px;
      }
      .mesh-toolbar {
        left: 16px;
        right: 16px;
        justify-content: space-between;
      }
      .sample-list {
        top: 74px;
        bottom: auto;
        max-height: 230px;
      }
      .mesh-footer {
        bottom: 16px;
      }
      canvas.chart { height: 310px; }
    }
  </style>
  <script type="importmap">
    {"imports": {"three": "./assets/three/three.module.js"}}
  </script>
</head>
<body>
  <header>
    <div>
      <nav class="nav">
        <a class="active" href="./">Training + samples</a>
        <a href="../tracker_reward_proto_2k_fixed_noise_fourway/">Fixed-noise before/after</a>
        <a href="../tracker_reward_proto_2k_fixed_noise_fourway/replay_proto_eval/">Replay eval</a>
      </nav>
      <h1>tracker_reward_proto_2k</h1>
      <p class="subtitle">HYMotion-G1 mesh rollouts from the generator after tracker-reward supervision. Use the fixed-noise page for direct before/after comparison.</p>
    </div>
    <div class="status-grid">
      <div class="stat"><span>Status</span><strong>END OK</strong></div>
      <div class="stat"><span>Steps</span><strong id="stepStat">2000/2000</strong></div>
      <div class="stat"><span>Final reward</span><strong id="rewardStat">0.677</strong></div>
    </div>
  </header>
  <main>
    <div class="compare-callout">
      <strong>This page is not the direct before/after view.</strong>
      It shows training signals and selected optimized-generator rollouts. The controlled same-noise comparison is here:
      <a href="../tracker_reward_proto_2k_fixed_noise_fourway/">base generator vs optimized generator vs both tracker results</a>.
    </div>
    <div class="mesh-stage" id="meshStage">
      <canvas id="meshCanvas"></canvas>
      <div class="sample-list" id="sampleList"></div>
      <div class="mesh-toolbar">
        <button id="playBtn" class="active">Pause</button>
        <div class="toolbar-group">
          <button data-view="side" class="viewBtn active">Side</button>
          <button data-view="front" class="viewBtn">Front</button>
          <button data-view="top" class="viewBtn">Top</button>
        </div>
      </div>
      <div class="mesh-footer">
        <div>
          <div class="mesh-title" id="sampleTitle"><strong>Rollout</strong><span>loading</span></div>
          <input id="frameSlider" type="range" min="0" max="1" value="0">
        </div>
        <div class="mesh-meta" id="motionMeta"></div>
      </div>
    </div>
    <div class="metrics" id="metricTiles"></div>
    <div class="grid">
      <section>
        <div class="section-head">
          <h2>Training Signal</h2>
          <div class="chart-controls" id="curveControls"></div>
        </div>
        <canvas id="curveCanvas" class="chart"></canvas>
      </section>
      <section>
        <div class="section-head">
          <h2>Frozen Eval Delta</h2>
          <div class="chart-controls"><button id="toggleDirection" class="active">better-is-up</button></div>
        </div>
        <canvas id="deltaCanvas" class="chart"></canvas>
      </section>
    </div>
    <div class="logline" id="paths"></div>
  </main>
  <script id="dashboard-data" type="application/json">__PAYLOAD__</script>
  <script type="module">
    import * as THREE from 'three';
    import { OrbitControls } from './assets/three/jsm/controls/OrbitControls.js';
    import { STLLoader } from './assets/three/jsm/loaders/STLLoader.js';

    const data = JSON.parse(document.getElementById('dashboard-data').textContent);
    const PLAYBACK_FPS = 30;
    const colors = { loss: '#bf4f2e', rewardBest: '#087c75', rewardCand: '#a57812', nGood: '#52783e', jointStd: '#6b5aa0' };
    const labels = { loss: 'loss', rewardBest: 'reward best', rewardCand: 'reward cand', nGood: 'n_good', jointStd: 'joint std' };

    let activeSeries = 'rewardBest';
    let activeSample = 0;
    let activeFrame = 0;
    let playing = true;
    let viewMode = 'side';
    let loadToken = 0;
    let bodyGroups = [];
    let currentBounds = null;
    let lastAdvance = 0;

    const canvas = document.getElementById('meshCanvas');
    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
    renderer.setClearColor(0x0d1210, 1);
    renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    const scene = new THREE.Scene();
    scene.fog = new THREE.Fog(0x0d1210, 5.4, 10.5);

    const camera = new THREE.PerspectiveCamera(42, 1, 0.01, 80);
    camera.up.set(0, 0, 1);
    const controls = new OrbitControls(camera, canvas);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.target.set(0, 0, 0.85);

    const robotRoot = new THREE.Group();
    scene.add(robotRoot);

    const loader = new STLLoader();
    const geometryCache = new Map();
    const materialCache = new Map();

    const hemi = new THREE.HemisphereLight(0xe8fff7, 0x23332e, 2.8);
    scene.add(hemi);
    const key = new THREE.DirectionalLight(0xffffff, 2.4);
    key.position.set(2.8, -4.2, 4.0);
    scene.add(key);
    const rim = new THREE.DirectionalLight(0xd0ffe8, 1.1);
    rim.position.set(-3.2, 2.4, 2.6);
    scene.add(rim);

    const grid = new THREE.GridHelper(4.8, 24, 0x50635b, 0x27322e);
    grid.rotation.x = Math.PI / 2;
    grid.position.z = 0;
    scene.add(grid);

    const axes = new THREE.AxesHelper(0.35);
    axes.position.set(-1.9, -1.9, 0.03);
    scene.add(axes);

    window.__meshReady = false;
    window.__meshInfo = { state: 'initializing', parts: 0, missing: [] };

    document.getElementById('stepStat').textContent = `${data.run.step}/${data.run.total}`;
    document.getElementById('rewardStat').textContent = Number(data.final.reward_best_mean || 0).toFixed(3);
    document.getElementById('paths').textContent = `${data.run.log}  |  ${data.run.eval}`;

    function fmt(v) {
      const n = Number(v || 0);
      if (Math.abs(n) >= 10) return n.toFixed(2);
      if (Math.abs(n) >= 1) return n.toFixed(3);
      return n.toFixed(4);
    }

    function setQuat(target, q) {
      target.quaternion.set(q[1], q[2], q[3], q[0]);
      target.quaternion.normalize();
    }

    function materialFor(name) {
      const keyName = name.includes('left') ? 'left' : name.includes('right') ? 'right' : name.includes('torso') || name.includes('waist') ? 'core' : name.includes('head') ? 'head' : 'base';
      if (materialCache.has(keyName)) return materialCache.get(keyName);
      const colorMap = {
        left: 0x6fc6c0,
        right: 0xd4a33a,
        core: 0xd8ddd4,
        head: 0xd66a45,
        base: 0xb8c3bb,
      };
      const mat = new THREE.MeshStandardMaterial({
        color: colorMap[keyName],
        roughness: 0.62,
        metalness: 0.18,
        envMapIntensity: 0.7,
      });
      materialCache.set(keyName, mat);
      return mat;
    }

    function loadGeometry(file) {
      if (!geometryCache.has(file)) {
        const url = `assets/g1_mesh/${encodeURIComponent(file)}`;
        geometryCache.set(file, new Promise((resolve, reject) => {
          loader.load(url, (geometry) => {
            geometry.computeVertexNormals();
            resolve(geometry);
          }, undefined, reject);
        }));
      }
      return geometryCache.get(file);
    }

    function clearRobot() {
      while (robotRoot.children.length) {
        robotRoot.remove(robotRoot.children[0]);
      }
      bodyGroups = [];
    }

    async function loadRobot(sample) {
      const token = ++loadToken;
      window.__meshReady = false;
      window.__meshInfo = { state: 'loading', sample: sample.stem, parts: 0, missing: [] };
      clearRobot();
      const missing = [];
      let parts = 0;

      sample.bodies.forEach((body) => {
        const group = new THREE.Group();
        group.name = body.name;
        bodyGroups.push(group);
        robotRoot.add(group);
      });

      for (let i = 0; i < sample.bodies.length; i += 1) {
        const body = sample.bodies[i];
        const group = bodyGroups[i];
        for (const meshDef of body.meshes) {
          try {
            const geometry = await loadGeometry(meshDef.file);
            if (token !== loadToken) return;
            const mesh = new THREE.Mesh(geometry, materialFor(body.name));
            mesh.castShadow = false;
            mesh.receiveShadow = true;
            mesh.position.set(meshDef.pos[0], meshDef.pos[1], meshDef.pos[2]);
            setQuat(mesh, meshDef.quat);
            group.add(mesh);
            parts += 1;
          } catch (err) {
            missing.push(meshDef.file);
          }
        }
      }

      if (token !== loadToken) return;
      applyFrame(0);
      fitCamera(sample);
      updateSampleMeta(missing, parts);
      window.__meshReady = true;
      window.__meshInfo = { state: 'ready', sample: sample.stem, parts, missing, frames: sample.frames.length };
    }

    function boundsCenter(bounds) {
      return new THREE.Vector3(
        (bounds.x0 + bounds.x1) / 2,
        (bounds.y0 + bounds.y1) / 2,
        (bounds.z0 + bounds.z1) / 2
      );
    }

    function boundsSpan(bounds) {
      return Math.max(bounds.x1 - bounds.x0, bounds.y1 - bounds.y0, bounds.z1 - bounds.z0, 1.1);
    }

    function fitCamera(sample) {
      currentBounds = sample.bounds;
      setCameraView(viewMode);
    }

    function setCameraView(mode) {
      if (!currentBounds) return;
      viewMode = mode;
      const center = boundsCenter(currentBounds);
      const span = boundsSpan(currentBounds);
      controls.target.set(center.x, center.y, Math.max(0.72, center.z + 0.08));
      if (mode === 'front') {
        camera.up.set(0, 0, 1);
        camera.position.set(center.x, center.y - span * 2.45, center.z + span * 0.52);
      } else if (mode === 'top') {
        camera.up.set(0, 1, 0);
        camera.position.set(center.x, center.y, center.z + span * 2.55);
      } else {
        camera.up.set(0, 0, 1);
        camera.position.set(center.x + span * 2.0, center.y - span * 0.58, center.z + span * 0.62);
      }
      camera.lookAt(controls.target);
      camera.updateProjectionMatrix();
      controls.update();
    }

    function applyFrame(frameIndex) {
      const sample = data.samples[activeSample];
      if (!sample) return;
      activeFrame = Math.max(0, Math.min(frameIndex, sample.frames.length - 1));
      const frame = sample.frames[activeFrame];
      bodyGroups.forEach((group, i) => {
        const p = frame.pos[i];
        const q = frame.quat[i];
        if (!p || !q) return;
        group.position.set(p[0], p[1], p[2]);
        setQuat(group, q);
      });
      document.getElementById('frameSlider').value = activeFrame;
      updateFrameLabel();
    }

    function updateFrameLabel() {
      const sample = data.samples[activeSample];
      const title = document.getElementById('sampleTitle');
      title.innerHTML = '';
      const strong = document.createElement('strong');
      strong.textContent = sample.stem;
      const span = document.createElement('span');
      span.textContent = `frame ${activeFrame + 1}/${sample.frames.length} | 30fps playback | src ${Number(sample.fps || 30).toFixed(1)} | ${sample.meshParts} mesh parts`;
      title.appendChild(strong);
      title.appendChild(span);
    }

    function updateSampleMeta(missing = [], parts = data.samples[activeSample]?.meshParts || 0) {
      const s = data.samples[activeSample];
      const meta = document.getElementById('motionMeta');
      meta.innerHTML = `
        <div class="mini">adv score<strong>${fmt(s.score)}</strong></div>
        <div class="mini">completion<strong>${fmt(s.completion)}</strong></div>
        <div class="mini">root err<strong>${fmt(s.rootError)}</strong></div>
        <div class="mini">mesh<strong>${parts}${missing.length ? ` / miss ${missing.length}` : ''}</strong></div>`;
      updateFrameLabel();
    }

    function setSample(index) {
      activeSample = index;
      activeFrame = 0;
      const slider = document.getElementById('frameSlider');
      const sample = data.samples[activeSample];
      slider.max = sample.frames.length - 1;
      slider.value = 0;
      renderSampleList();
      updateSampleMeta([], sample.meshParts);
      loadRobot(sample);
    }

    function renderSampleList() {
      const host = document.getElementById('sampleList');
      host.innerHTML = '';
      data.samples.forEach((s, i) => {
        const row = document.createElement('button');
        row.className = 'sample-row' + (i === activeSample ? ' active' : '');
        const name = document.createElement('span');
        name.textContent = s.stem;
        const stats = document.createElement('small');
        stats.textContent = `score ${fmt(s.score)} | completion ${fmt(s.completion)} | ${s.fall ? 'fall' : 'no fall'}`;
        const prompt = document.createElement('small');
        prompt.textContent = s.prompt;
        row.appendChild(name);
        row.appendChild(stats);
        row.appendChild(prompt);
        row.onclick = () => setSample(i);
        host.appendChild(row);
      });
    }

    function initMeshControls() {
      document.getElementById('frameSlider').oninput = (event) => {
        playing = false;
        document.getElementById('playBtn').textContent = 'Play';
        document.getElementById('playBtn').classList.remove('active');
        applyFrame(Number(event.target.value));
      };
      document.getElementById('playBtn').onclick = () => {
        playing = !playing;
        document.getElementById('playBtn').textContent = playing ? 'Pause' : 'Play';
        document.getElementById('playBtn').classList.toggle('active', playing);
      };
      document.querySelectorAll('.viewBtn').forEach((btn) => {
        btn.onclick = () => {
          document.querySelectorAll('.viewBtn').forEach((b) => b.classList.remove('active'));
          btn.classList.add('active');
          setCameraView(btn.dataset.view);
        };
      });
    }

    function initTiles() {
      const host = document.getElementById('metricTiles');
      host.innerHTML = '';
      data.evalMetrics.forEach((m) => {
        const imp = m.improvement;
        const good = imp >= 0;
        const tile = document.createElement('div');
        tile.className = 'metric';
        tile.style.setProperty('--w', `${Math.min(100, Math.max(8, Math.abs(imp) * 260))}%`);
        tile.style.setProperty('--c', good ? 'var(--teal)' : 'var(--rust)');
        tile.innerHTML = `<div class="label">${m.label}</div><div class="value">${fmt(m.proto)}</div><div class="delta">base ${fmt(m.base)} | ${good ? '+' : ''}${(imp * 100).toFixed(1)}% aligned</div>`;
        host.appendChild(tile);
      });
    }

    function drawChart(canvasEl, points, key, color) {
      const ctx = canvasEl.getContext('2d');
      const dpr = window.devicePixelRatio || 1;
      const rect = canvasEl.getBoundingClientRect();
      canvasEl.width = Math.max(1, Math.floor(rect.width * dpr));
      canvasEl.height = Math.max(1, Math.floor(rect.height * dpr));
      ctx.scale(dpr, dpr);
      const w = rect.width;
      const h = rect.height;
      ctx.clearRect(0, 0, w, h);
      const pad = { l: 54, r: 20, t: 28, b: 38 };
      ctx.strokeStyle = '#cbd4cc';
      ctx.lineWidth = 1;
      ctx.font = '12px ui-monospace, Menlo, monospace';
      ctx.fillStyle = '#657067';
      for (let i = 0; i <= 4; i += 1) {
        const y = pad.t + (h - pad.t - pad.b) * i / 4;
        ctx.beginPath();
        ctx.moveTo(pad.l, y);
        ctx.lineTo(w - pad.r, y);
        ctx.stroke();
      }
      if (!points.length) return;
      const xs = points.map((p) => p[0]);
      const ys = points.map((p) => p[1]);
      const xmin = Math.min(...xs);
      const xmax = Math.max(...xs);
      let ymin = Math.min(...ys);
      let ymax = Math.max(...ys);
      if (ymin === ymax) {
        ymin -= 1;
        ymax += 1;
      }
      const xspan = xmax === xmin ? 1 : xmax - xmin;
      const yspan = ymax === ymin ? 1 : ymax - ymin;
      const xmap = (x) => pad.l + (x - xmin) / xspan * (w - pad.l - pad.r);
      const ymap = (y) => h - pad.b - (y - ymin) / yspan * (h - pad.t - pad.b);
      ctx.fillText(labels[key], pad.l, 18);
      ctx.fillText(fmt(ymax), 8, pad.t + 4);
      ctx.fillText(fmt(ymin), 8, h - pad.b);
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.4;
      ctx.beginPath();
      points.forEach((p, i) => {
        const x = xmap(p[0]);
        const y = ymap(p[1]);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
      const last = points[points.length - 1];
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(xmap(last[0]), ymap(last[1]), 4, 0, Math.PI * 2);
      ctx.fill();
    }

    function drawDelta() {
      const canvasEl = document.getElementById('deltaCanvas');
      const ctx = canvasEl.getContext('2d');
      const dpr = window.devicePixelRatio || 1;
      const rect = canvasEl.getBoundingClientRect();
      canvasEl.width = Math.max(1, Math.floor(rect.width * dpr));
      canvasEl.height = Math.max(1, Math.floor(rect.height * dpr));
      ctx.scale(dpr, dpr);
      const w = rect.width;
      const h = rect.height;
      ctx.clearRect(0, 0, w, h);
      const pad = { l: 150, r: 34, t: 26, b: 26 };
      const vals = data.evalMetrics.map((m) => m.improvement * 100);
      const maxAbs = Math.max(5, ...vals.map((v) => Math.abs(v)));
      const zero = pad.l + (w - pad.l - pad.r) / 2;
      ctx.strokeStyle = '#141712';
      ctx.beginPath();
      ctx.moveTo(zero, pad.t);
      ctx.lineTo(zero, h - pad.b);
      ctx.stroke();
      data.evalMetrics.forEach((m, i) => {
        const y = pad.t + 12 + i * ((h - pad.t - pad.b) / data.evalMetrics.length);
        const v = m.improvement * 100;
        const bw = (w - pad.l - pad.r) / 2 * Math.abs(v) / maxAbs;
        ctx.fillStyle = '#657067';
        ctx.font = '12px ui-monospace, Menlo, monospace';
        ctx.fillText(m.label, 14, y + 5);
        ctx.fillStyle = v >= 0 ? '#087c75' : '#bf4f2e';
        ctx.fillRect(v >= 0 ? zero : zero - bw, y - 8, bw, 16);
        ctx.fillText(`${v >= 0 ? '+' : ''}${v.toFixed(1)}%`, v >= 0 ? zero + bw + 8 : zero - bw - 58, y + 5);
      });
    }

    function initCurveControls() {
      const host = document.getElementById('curveControls');
      Object.keys(labels).forEach((key) => {
        const btn = document.createElement('button');
        btn.textContent = labels[key];
        btn.className = key === activeSeries ? 'active' : '';
        btn.onclick = () => {
          activeSeries = key;
          [...host.children].forEach((b) => b.classList.remove('active'));
          btn.classList.add('active');
          drawAll();
        };
        host.appendChild(btn);
      });
    }

    function resizeRenderer() {
      const rect = canvas.getBoundingClientRect();
      const width = Math.max(1, Math.floor(rect.width));
      const height = Math.max(1, Math.floor(rect.height));
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
    }

    function drawAll() {
      resizeRenderer();
      drawChart(document.getElementById('curveCanvas'), data.series[activeSeries], activeSeries, colors[activeSeries]);
      drawDelta();
    }

    function animate(now) {
      requestAnimationFrame(animate);
      const sample = data.samples[activeSample];
      if (playing && sample && window.__meshReady) {
        const interval = 1000 / PLAYBACK_FPS;
        if (!lastAdvance || now - lastAdvance >= interval) {
          activeFrame = (activeFrame + 1) % sample.frames.length;
          applyFrame(activeFrame);
          lastAdvance = now;
        }
      }
      controls.update();
      renderer.render(scene, camera);
    }

    window.addEventListener('resize', drawAll);
    initTiles();
    initCurveControls();
    initMeshControls();
    renderSampleList();
    drawAll();
    setSample(0);
    requestAnimationFrame(animate);
  </script>
</body>
</html>
"""
    return template.replace("__PAYLOAD__", payload)


def html_doc_legacy_2d(data: dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    payload = payload.replace("</", "<\\/")
    connections = json.dumps(CONNECTIONS)
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>tracker_reward_proto_2k</title>
  <style>
    :root {{
      --paper: #f5f7f4;
      --ink: #151713;
      --muted: #65706a;
      --line: #cdd4ce;
      --panel: #ffffff;
      --teal: #007d7a;
      --rust: #c44f2c;
      --leaf: #557c3e;
      --amber: #a87d19;
      --violet: #7256a8;
      --shadow: 0 14px 45px rgba(31, 45, 36, .12);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      background:
        linear-gradient(90deg, rgba(21,23,19,.035) 1px, transparent 1px) 0 0 / 24px 24px,
        linear-gradient(0deg, rgba(21,23,19,.026) 1px, transparent 1px) 0 0 / 24px 24px,
        var(--paper);
      font-family: Avenir Next, Segoe UI, sans-serif;
      letter-spacing: 0;
    }}
    header {{
      display: grid;
      grid-template-columns: 1.2fr .8fr;
      gap: 28px;
      padding: 28px clamp(18px, 4vw, 56px) 18px;
      border-bottom: 1px solid var(--line);
    }}
    h1 {{
      margin: 0;
      font-family: Georgia, Cambria, serif;
      font-size: clamp(32px, 5vw, 66px);
      line-height: .93;
      font-weight: 700;
    }}
    .subtitle {{ margin: 14px 0 0; color: var(--muted); max-width: 780px; font-size: 15px; line-height: 1.5; }}
    .status-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; align-content: end; }}
    .stat {{
      min-height: 84px;
      padding: 14px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,.72);
      box-shadow: var(--shadow);
    }}
    .stat span {{ display: block; color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .08em; }}
    .stat strong {{ display: block; margin-top: 8px; font: 700 24px Georgia, serif; }}
    main {{ padding: 24px clamp(18px, 4vw, 56px) 48px; }}
    .band {{ display: grid; gap: 18px; }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 22px;
    }}
    .metric {{
      border: 1px solid var(--line);
      background: var(--panel);
      padding: 14px;
      min-height: 112px;
      position: relative;
      overflow: hidden;
    }}
    .metric::after {{
      content: \"\";
      position: absolute;
      left: 0;
      bottom: 0;
      height: 5px;
      width: var(--w);
      background: var(--c);
    }}
    .metric .label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .08em; }}
    .metric .value {{ margin-top: 10px; font: 700 26px ui-monospace, SFMono-Regular, Menlo, monospace; }}
    .metric .delta {{ margin-top: 8px; color: var(--muted); font-size: 13px; }}
    .grid {{
      display: grid;
      grid-template-columns: minmax(0, 1.05fr) minmax(360px, .95fr);
      gap: 18px;
    }}
    section {{
      border: 1px solid var(--line);
      background: rgba(255,255,255,.86);
      box-shadow: var(--shadow);
      min-width: 0;
    }}
    .section-head {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
    }}
    h2 {{ margin: 0; font: 700 18px Georgia, serif; }}
    .controls {{ display: flex; flex-wrap: wrap; gap: 8px; }}
    button, select, input[type=\"range\"] {{ accent-color: var(--teal); }}
    button, select {{
      border: 1px solid var(--line);
      background: #fbfcfa;
      color: var(--ink);
      padding: 8px 10px;
      min-height: 34px;
      font: 700 12px ui-monospace, SFMono-Regular, Menlo, monospace;
      cursor: pointer;
    }}
    button.active {{ background: var(--ink); color: #fff; border-color: var(--ink); }}
    canvas {{ width: 100%; display: block; }}
    #curveCanvas {{ height: 360px; }}
    #deltaCanvas {{ height: 360px; }}
    .viewer {{
      display: grid;
      grid-template-columns: 260px minmax(0, 1fr);
      gap: 0;
      margin-top: 18px;
    }}
    .sample-list {{
      border-right: 1px solid var(--line);
      max-height: 520px;
      overflow: auto;
    }}
    .sample-row {{
      width: 100%;
      text-align: left;
      border: 0;
      border-bottom: 1px solid var(--line);
      background: transparent;
      padding: 12px;
      font-family: Avenir Next, Segoe UI, sans-serif;
      font-weight: 600;
    }}
    .sample-row.active {{ background: rgba(0,125,122,.11); color: var(--ink); }}
    .sample-row small {{ display: block; margin-top: 5px; color: var(--muted); font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
    .motion-panel {{ min-width: 0; }}
    #motionCanvas {{ height: 470px; background: #101410; }}
    .motion-meta {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      padding: 12px 14px;
      border-top: 1px solid var(--line);
    }}
    .mini {{ color: var(--muted); font-size: 12px; }}
    .mini strong {{ color: var(--ink); display: block; font: 700 15px ui-monospace, SFMono-Regular, Menlo, monospace; margin-top: 4px; }}
    .logline {{
      margin-top: 18px;
      padding: 12px 14px;
      border: 1px solid var(--line);
      background: #fdfefb;
      color: var(--muted);
      font: 12px ui-monospace, SFMono-Regular, Menlo, monospace;
      overflow-wrap: anywhere;
    }}
    @media (max-width: 980px) {{
      header, .grid, .viewer {{ grid-template-columns: 1fr; }}
      .metrics {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .status-grid {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
      .sample-list {{ border-right: 0; border-bottom: 1px solid var(--line); max-height: 260px; }}
    }}
    @media (max-width: 620px) {{
      .metrics, .status-grid, .motion-meta {{ grid-template-columns: 1fr; }}
      h1 {{ font-size: 34px; }}
    }}
  </style>
</head>
<body>
  <header>
    <div>
      <h1>tracker_reward_proto_2k</h1>
      <p class=\"subtitle\">PhysFlow HYMotion-G1 with the frozen ProtoMotions tracker reward. The page compares the 130k base against the 2k reward-tuned checkpoint and replays representative robot rollouts.</p>
    </div>
    <div class=\"status-grid\">
      <div class=\"stat\"><span>Status</span><strong>END OK</strong></div>
      <div class=\"stat\"><span>Steps</span><strong id=\"stepStat\">2000/2000</strong></div>
      <div class=\"stat\"><span>Final reward</span><strong id=\"rewardStat\">0.677</strong></div>
    </div>
  </header>
  <main>
    <div class=\"metrics\" id=\"metricTiles\"></div>
    <div class=\"grid\">
      <section>
        <div class=\"section-head\">
          <h2>Training Signal</h2>
          <div class=\"controls\" id=\"curveControls\"></div>
        </div>
        <canvas id=\"curveCanvas\"></canvas>
      </section>
      <section>
        <div class=\"section-head\">
          <h2>Frozen Eval Delta</h2>
          <div class=\"controls\"><button id=\"toggleDirection\" class=\"active\">better-is-up</button></div>
        </div>
        <canvas id=\"deltaCanvas\"></canvas>
      </section>
    </div>
    <section class=\"viewer\">
      <div class=\"sample-list\" id=\"sampleList\"></div>
      <div class=\"motion-panel\">
        <div class=\"section-head\">
          <h2 id=\"sampleTitle\">Rollout</h2>
          <div class=\"controls\">
            <button id=\"playBtn\" class=\"active\">pause</button>
            <button data-view=\"side\" class=\"viewBtn active\">side</button>
            <button data-view=\"front\" class=\"viewBtn\">front</button>
            <button data-view=\"top\" class=\"viewBtn\">top</button>
          </div>
        </div>
        <canvas id=\"motionCanvas\"></canvas>
        <input id=\"frameSlider\" type=\"range\" min=\"0\" max=\"1\" value=\"0\" style=\"width:100%\">
        <div class=\"motion-meta\" id=\"motionMeta\"></div>
      </div>
    </section>
    <div class=\"logline\" id=\"paths\"></div>
  </main>
  <script id=\"dashboard-data\" type=\"application/json\">{payload}</script>
  <script>
    const data = JSON.parse(document.getElementById('dashboard-data').textContent);
    const CONNECTIONS = {connections};
    const colors = {{ loss: '#c44f2c', rewardBest: '#007d7a', rewardCand: '#a87d19', nGood: '#557c3e', jointStd: '#7256a8' }};
    const labels = {{ loss: 'loss', rewardBest: 'reward best', rewardCand: 'reward cand', nGood: 'n_good', jointStd: 'joint std' }};
    let activeSeries = 'rewardBest';
    let activeSample = 0;
    let activeFrame = 0;
    let playing = true;
    let viewMode = 'side';

    document.getElementById('stepStat').textContent = `${{data.run.step}}/${{data.run.total}}`;
    document.getElementById('rewardStat').textContent = Number(data.final.reward_best_mean || 0).toFixed(3);
    document.getElementById('paths').textContent = `${{data.run.log}}  |  ${{data.run.eval}}`;

    function fmt(v) {{
      if (Math.abs(v) >= 10) return v.toFixed(2);
      if (Math.abs(v) >= 1) return v.toFixed(3);
      return v.toFixed(4);
    }}

    function initTiles() {{
      const host = document.getElementById('metricTiles');
      host.innerHTML = '';
      data.evalMetrics.forEach((m) => {{
        const imp = m.improvement;
        const good = imp >= 0;
        const tile = document.createElement('div');
        tile.className = 'metric';
        tile.style.setProperty('--w', `${{Math.min(100, Math.max(8, Math.abs(imp) * 260))}}%`);
        tile.style.setProperty('--c', good ? 'var(--teal)' : 'var(--rust)');
        tile.innerHTML = `<div class=\"label\">${{m.label}}</div><div class=\"value\">${{fmt(m.proto)}}</div><div class=\"delta\">base ${{fmt(m.base)}} | ${{good ? '+' : ''}}${{(imp * 100).toFixed(1)}}% aligned</div>`;
        host.appendChild(tile);
      }});
    }}

    function drawChart(canvas, points, key, color) {{
      const ctx = canvas.getContext('2d');
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.max(1, Math.floor(rect.width * dpr));
      canvas.height = Math.max(1, Math.floor(rect.height * dpr));
      ctx.scale(dpr, dpr);
      const w = rect.width, h = rect.height;
      ctx.clearRect(0, 0, w, h);
      const pad = {{ l: 54, r: 20, t: 28, b: 38 }};
      ctx.strokeStyle = '#cdd4ce';
      ctx.lineWidth = 1;
      ctx.font = '12px ui-monospace, Menlo, monospace';
      ctx.fillStyle = '#65706a';
      for (let i = 0; i <= 4; i++) {{
        const y = pad.t + (h - pad.t - pad.b) * i / 4;
        ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
      }}
      if (!points.length) return;
      const xs = points.map(p => p[0]);
      const ys = points.map(p => p[1]);
      const xmin = Math.min(...xs), xmax = Math.max(...xs);
      let ymin = Math.min(...ys), ymax = Math.max(...ys);
      if (ymin === ymax) {{ ymin -= 1; ymax += 1; }}
      const xmap = x => pad.l + (x - xmin) / (xmax - xmin) * (w - pad.l - pad.r);
      const ymap = y => h - pad.b - (y - ymin) / (ymax - ymin) * (h - pad.t - pad.b);
      ctx.fillText(labels[key], pad.l, 18);
      ctx.fillText(fmt(ymax), 8, pad.t + 4);
      ctx.fillText(fmt(ymin), 8, h - pad.b);
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.4;
      ctx.beginPath();
      points.forEach((p, i) => {{
        const x = xmap(p[0]), y = ymap(p[1]);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }});
      ctx.stroke();
      const last = points[points.length - 1];
      ctx.fillStyle = color;
      ctx.beginPath(); ctx.arc(xmap(last[0]), ymap(last[1]), 4, 0, Math.PI * 2); ctx.fill();
    }}

    function drawDelta() {{
      const canvas = document.getElementById('deltaCanvas');
      const ctx = canvas.getContext('2d');
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.floor(rect.width * dpr); canvas.height = Math.floor(rect.height * dpr);
      ctx.scale(dpr, dpr);
      const w = rect.width, h = rect.height;
      ctx.clearRect(0, 0, w, h);
      const pad = {{ l: 150, r: 34, t: 26, b: 26 }};
      const vals = data.evalMetrics.map(m => m.improvement * 100);
      const maxAbs = Math.max(5, ...vals.map(v => Math.abs(v)));
      const zero = pad.l + (w - pad.l - pad.r) / 2;
      ctx.strokeStyle = '#151713';
      ctx.beginPath(); ctx.moveTo(zero, pad.t); ctx.lineTo(zero, h - pad.b); ctx.stroke();
      data.evalMetrics.forEach((m, i) => {{
        const y = pad.t + 12 + i * ((h - pad.t - pad.b) / data.evalMetrics.length);
        const v = m.improvement * 100;
        const bw = (w - pad.l - pad.r) / 2 * Math.abs(v) / maxAbs;
        ctx.fillStyle = '#65706a';
        ctx.font = '12px ui-monospace, Menlo, monospace';
        ctx.fillText(m.label, 14, y + 5);
        ctx.fillStyle = v >= 0 ? '#007d7a' : '#c44f2c';
        ctx.fillRect(v >= 0 ? zero : zero - bw, y - 8, bw, 16);
        ctx.fillText(`${{v >= 0 ? '+' : ''}}${{v.toFixed(1)}}%`, v >= 0 ? zero + bw + 8 : zero - bw - 58, y + 5);
      }});
    }}

    function initControls() {{
      const host = document.getElementById('curveControls');
      Object.keys(labels).forEach(key => {{
        const btn = document.createElement('button');
        btn.textContent = labels[key];
        btn.className = key === activeSeries ? 'active' : '';
        btn.onclick = () => {{
          activeSeries = key;
          [...host.children].forEach(b => b.classList.remove('active'));
          btn.classList.add('active');
          drawAll();
        }};
        host.appendChild(btn);
      }});
    }}

    function initSamples() {{
      const host = document.getElementById('sampleList');
      host.innerHTML = '';
      data.samples.forEach((s, i) => {{
        const row = document.createElement('button');
        row.className = 'sample-row' + (i === activeSample ? ' active' : '');
        row.innerHTML = `${{s.stem}}<small>score ${{fmt(s.score)}} | completion ${{fmt(s.completion)}} | ${{s.fall ? 'fall' : 'no fall'}}</small><small>${{s.prompt}}</small>`;
        row.onclick = () => {{
          activeSample = i; activeFrame = 0;
          document.getElementById('frameSlider').max = data.samples[i].frames.length - 1;
          document.getElementById('frameSlider').value = 0;
          initSamples(); renderMotion();
        }};
        host.appendChild(row);
      }});
      document.getElementById('frameSlider').max = data.samples[activeSample].frames.length - 1;
      document.getElementById('frameSlider').oninput = (e) => {{ activeFrame = Number(e.target.value); renderMotion(); }};
      document.getElementById('playBtn').onclick = () => {{
        playing = !playing;
        document.getElementById('playBtn').textContent = playing ? 'pause' : 'play';
        document.getElementById('playBtn').classList.toggle('active', playing);
      }};
      document.querySelectorAll('.viewBtn').forEach(btn => {{
        btn.onclick = () => {{
          viewMode = btn.dataset.view;
          document.querySelectorAll('.viewBtn').forEach(b => b.classList.remove('active'));
          btn.classList.add('active');
          renderMotion();
        }};
      }});
    }}

    function project(p, bounds, w, h) {{
      const margin = 34;
      let a, b, a0, a1, b0, b1;
      if (viewMode === 'front') {{ a = p[1]; b = p[2]; a0 = bounds.y0; a1 = bounds.y1; b0 = bounds.z0; b1 = bounds.z1; }}
      else if (viewMode === 'top') {{ a = p[0]; b = p[1]; a0 = bounds.x0; a1 = bounds.x1; b0 = bounds.y0; b1 = bounds.y1; }}
      else {{ a = p[0]; b = p[2]; a0 = bounds.x0; a1 = bounds.x1; b0 = bounds.z0; b1 = bounds.z1; }}
      const x = margin + (a - a0) / (a1 - a0) * (w - margin * 2);
      const y = h - margin - (b - b0) / (b1 - b0) * (h - margin * 2);
      return [x, y];
    }}

    function renderMotion() {{
      const s = data.samples[activeSample];
      document.getElementById('sampleTitle').textContent = s.stem;
      const meta = document.getElementById('motionMeta');
      meta.innerHTML = `<div class=\"mini\">adv score<strong>${{fmt(s.score)}}</strong></div><div class=\"mini\">completion<strong>${{fmt(s.completion)}}</strong></div><div class=\"mini\">max joint err<strong>${{fmt(s.maxJointError)}}</strong></div><div class=\"mini\">root err<strong>${{fmt(s.rootError)}}</strong></div>`;
      const canvas = document.getElementById('motionCanvas');
      const ctx = canvas.getContext('2d');
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.floor(rect.width * dpr); canvas.height = Math.floor(rect.height * dpr);
      ctx.scale(dpr, dpr);
      const w = rect.width, h = rect.height;
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = '#101410'; ctx.fillRect(0, 0, w, h);
      const frame = s.frames[Math.min(activeFrame, s.frames.length - 1)];
      ctx.strokeStyle = 'rgba(245,247,244,.16)';
      ctx.lineWidth = 1;
      for (let y = 36; y < h; y += 36) {{ ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); }}
      ctx.lineCap = 'round';
      CONNECTIONS.forEach(([a, b]) => {{
        if (!frame[a] || !frame[b]) return;
        const pa = project(frame[a], s.bounds, w, h);
        const pb = project(frame[b], s.bounds, w, h);
        ctx.strokeStyle = '#dce8dc';
        ctx.lineWidth = 4;
        ctx.beginPath(); ctx.moveTo(pa[0], pa[1]); ctx.lineTo(pb[0], pb[1]); ctx.stroke();
      }});
      frame.forEach((p, idx) => {{
        const q = project(p, s.bounds, w, h);
        ctx.fillStyle = idx === 0 ? '#c44f2c' : '#f0b84d';
        ctx.beginPath(); ctx.arc(q[0], q[1], idx === 0 ? 5 : 3, 0, Math.PI * 2); ctx.fill();
      }});
      ctx.fillStyle = 'rgba(245,247,244,.75)';
      ctx.font = '12px ui-monospace, Menlo, monospace';
      ctx.fillText(`frame ${{activeFrame + 1}} / ${{s.frames.length}}`, 14, 24);
      document.getElementById('frameSlider').value = activeFrame;
    }}

    function tick() {{
      if (playing && data.samples.length) {{
        const s = data.samples[activeSample];
        activeFrame = (activeFrame + 1) % s.frames.length;
        renderMotion();
      }}
      requestAnimationFrame(() => setTimeout(tick, 1000 / 16));
    }}

    function drawAll() {{
      drawChart(document.getElementById('curveCanvas'), data.series[activeSeries], activeSeries, colors[activeSeries]);
      drawDelta();
      renderMotion();
    }}

    window.addEventListener('resize', drawAll);
    initTiles();
    initControls();
    initSamples();
    drawAll();
    tick();
  </script>
</body>
</html>
"""


def main() -> None:
    data = build_data()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    copy_assets()
    OUT_HTML.write_text(html_doc(data))
    print(OUT_HTML)


if __name__ == "__main__":
    main()
