#!/usr/bin/env python3
"""Build a four-way fixed-noise G1 dashboard for tracker-reward proto2k."""

from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path
from typing import Any


ROOT = Path("/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer")
RUN_ROOT = Path(
    os.environ.get(
        "PHYSFLOW_FIXED_NOISE_RUN_ROOT",
        str(ROOT / "output/physflow_fixed_noise_tracker_reward_proto2k_compare"),
    )
)
OUT_DIR = Path(
    os.environ.get(
        "PHYSFLOW_FIXED_NOISE_OUT_DIR",
        str(ROOT / "output/physflow_visualizations/tracker_reward_proto_2k_fixed_noise_fourway"),
    )
)
OUT_HTML = OUT_DIR / "index.html"
THREE_SRC = ROOT / "motion_annot_web/score_m2m/static/three"
G1_MESH_SRC = (
    ROOT
    / "hftrainer/models/motion/physflow/trackers/protomotions/vendor/protomotions/data/assets/mesh/G1"
)
BASE_RUN_KEY = os.environ.get("PHYSFLOW_FIXED_NOISE_BASE_KEY", "base130k")
PROTO_RUN_KEY = os.environ.get("PHYSFLOW_FIXED_NOISE_PROTO_KEY", "proto2k")
PAGE_TITLE = os.environ.get("PHYSFLOW_FIXED_NOISE_TITLE", "tracker_reward_proto_2k fixed-noise four-way")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def fmt_float(value: Any, digits: int = 4) -> float | None:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value):
        return None
    return round(value, digits)


def trackable(record: dict[str, Any]) -> bool:
    return (
        record.get("status") == "scored"
        and float(record.get("completion_ratio") or 0.0) >= 0.95
        and float(record.get("max_joint_error_rad") or 999.0) <= 0.7
        and not bool(record.get("fall_detected", True))
    )


def root_path_metrics(robot_json: Path) -> dict[str, float]:
    data = read_json(robot_json)
    roots = []
    for frame in data.get("frames", []):
        pos = frame.get("body_pos") or []
        if pos:
            roots.append(pos[0])
    fps = float(data.get("fps") or 30.0)
    if len(roots) < 2:
        return {"frames": float(len(roots)), "duration": 0.0, "xy_path": 0.0, "xy_displacement": 0.0}
    path = 0.0
    for prev, cur in zip(roots, roots[1:]):
        dx = float(cur[0]) - float(prev[0])
        dy = float(cur[1]) - float(prev[1])
        path += math.hypot(dx, dy)
    dx = float(roots[-1][0]) - float(roots[0][0])
    dy = float(roots[-1][1]) - float(roots[0][1])
    return {
        "frames": float(len(roots)),
        "duration": float(len(roots) / max(fps, 1e-6)),
        "xy_path": float(path),
        "xy_displacement": float(math.hypot(dx, dy)),
    }


def copy_assets() -> None:
    for rel in [
        Path("three.module.js"),
        Path("jsm/controls/OrbitControls.js"),
        Path("jsm/loaders/STLLoader.js"),
    ]:
        src = THREE_SRC / rel
        dst = OUT_DIR / "assets" / "three" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    mesh_dir = OUT_DIR / "assets" / "g1_mesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    for src in sorted(G1_MESH_SRC.glob("*.stl")):
        shutil.copy2(src, mesh_dir / src.name)


def rel_copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst.relative_to(OUT_DIR))


def metric_record(record: dict[str, Any], robot_json: Path, ref_json: Path) -> dict[str, Any]:
    kin = record.get("kinematic") or {}
    track_root = root_path_metrics(robot_json)
    ref_root = root_path_metrics(ref_json)
    return {
        "score": fmt_float(record.get("adversarial_score")),
        "completion": fmt_float(record.get("completion_ratio")),
        "fall": bool(record.get("fall_detected")),
        "max_joint_err": fmt_float(record.get("max_joint_error_rad")),
        "root_traj": fmt_float(record.get("root_trajectory_error_mean_m")),
        "foot_skate": fmt_float(kin.get("foot_skate_speed")),
        "joint_vel": fmt_float(kin.get("joint_vel_max")),
        "jerk": fmt_float(kin.get("jerk"), 3),
        "trackable": trackable(record),
        "ref_path": fmt_float(ref_root["xy_path"], 3),
        "track_path": fmt_float(track_root["xy_path"], 3),
        "duration": fmt_float(track_root["duration"], 3),
    }


def row_quality(base: dict[str, Any], proto: dict[str, Any]) -> dict[str, Any]:
    def num(key: str) -> float:
        value = proto.get(key)
        base_value = base.get(key)
        if value is None or base_value is None:
            return 0.0
        return float(value) - float(base_value)

    score_delta = num("score")
    root_delta = num("root_traj")
    joint_delta = num("max_joint_err")
    completion_delta = num("completion")
    fall_delta = int(bool(proto.get("fall"))) - int(bool(base.get("fall")))
    trackable_delta = int(bool(proto.get("trackable"))) - int(bool(base.get("trackable")))
    # Lower score/root/joint/fall is better; higher completion/trackable is better.
    composite = (
        -score_delta
        - 0.45 * root_delta
        - 0.35 * joint_delta
        + 0.8 * completion_delta
        - 0.8 * fall_delta
        + 0.5 * trackable_delta
    )
    if trackable_delta > 0 or (score_delta < -0.15 and fall_delta <= 0):
        verdict = "improved"
    elif trackable_delta < 0 or (score_delta > 0.15 and fall_delta >= 0):
        verdict = "worse"
    else:
        verdict = "mixed"
    return {
        "score_delta": fmt_float(score_delta),
        "root_delta": fmt_float(root_delta),
        "joint_delta": fmt_float(joint_delta),
        "completion_delta": fmt_float(completion_delta),
        "fall_delta": fall_delta,
        "trackable_delta": trackable_delta,
        "composite": fmt_float(composite),
        "verdict": verdict,
    }


def build_data() -> dict[str, Any]:
    base_summary = read_json(RUN_ROOT / "runs" / BASE_RUN_KEY / "summary.json")
    proto_summary = read_json(RUN_ROOT / "runs" / PROTO_RUN_KEY / "summary.json")
    top_summary = read_json(RUN_ROOT / "summary.json")
    manifest = read_json(RUN_ROOT / "viz/manifest.json")
    base_records = {r["output_stem"]: r for r in base_summary["records"]}
    proto_records = {r["output_stem"]: r for r in proto_summary["records"]}

    rows = []
    for stem in sorted(base_records):
        if stem not in proto_records:
            continue
        base = base_records[stem]
        proto = proto_records[stem]
        base_ref_abs = Path(base["robot_ref_path"])
        proto_ref_abs = Path(proto["robot_ref_path"])
        base_track_abs = RUN_ROOT / "runs" / BASE_RUN_KEY / "json" / f"{stem}.json"
        proto_track_abs = RUN_ROOT / "runs" / PROTO_RUN_KEY / "json" / f"{stem}.json"
        paths = {
            "base_ref": rel_copy(base_ref_abs, OUT_DIR / "data/base_ref" / f"{stem}.json"),
            "proto_ref": rel_copy(proto_ref_abs, OUT_DIR / "data/proto_ref" / f"{stem}.json"),
            "base_track": rel_copy(base_track_abs, OUT_DIR / "data/base_track" / f"{stem}.json"),
            "proto_track": rel_copy(proto_track_abs, OUT_DIR / "data/proto_track" / f"{stem}.json"),
        }
        metrics = {
            "base": metric_record(base, base_track_abs, base_ref_abs),
            "proto": metric_record(proto, proto_track_abs, proto_ref_abs),
        }
        rows.append(
            {
                "stem": stem,
                "prompt_id": base.get("prompt_id"),
                "prompt": base.get("prompt"),
                "source_index": int(base.get("source_index", -1)),
                "noise_seed": int(base.get("noise_seed", -1)),
                "paths": paths,
                "metrics": metrics,
                "delta": row_quality(metrics["base"], metrics["proto"]),
            }
        )

    counts = {
        "improved": sum(1 for r in rows if r["delta"]["verdict"] == "improved"),
        "mixed": sum(1 for r in rows if r["delta"]["verdict"] == "mixed"),
        "worse": sum(1 for r in rows if r["delta"]["verdict"] == "worse"),
        "score_improved": sum(1 for r in rows if (r["delta"]["score_delta"] or 0) < 0),
        "root_improved": sum(1 for r in rows if (r["delta"]["root_delta"] or 0) < 0),
        "joint_improved": sum(1 for r in rows if (r["delta"]["joint_delta"] or 0) < 0),
        "falls_fixed": sum(1 for r in rows if r["metrics"]["base"]["fall"] and not r["metrics"]["proto"]["fall"]),
        "new_falls": sum(1 for r in rows if not r["metrics"]["base"]["fall"] and r["metrics"]["proto"]["fall"]),
        "trackable_gained": sum(1 for r in rows if (r["delta"]["trackable_delta"] or 0) > 0),
        "trackable_lost": sum(1 for r in rows if (r["delta"]["trackable_delta"] or 0) < 0),
    }
    rows.sort(key=lambda r: (r["delta"]["verdict"] != "worse", r["delta"]["score_delta"] or 0))
    return {
        "run": {
            "title": PAGE_TITLE,
            "base": BASE_RUN_KEY,
            "proto": PROTO_RUN_KEY,
            "same_noise_policy": manifest.get("same_noise_policy"),
            "base_seed": manifest.get("base_seed"),
            "source": str(RUN_ROOT),
        },
        "summary": top_summary.get("metrics", {}),
        "counts": counts,
        "rows": rows,
    }


def html_doc(data: dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    template = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>fixed-noise proto2k four-way</title>
  <style>
    :root {
      --paper: #f3f5f1;
      --ink: #141713;
      --muted: #667068;
      --line: #c8d1ca;
      --panel: #ffffff;
      --night: #0b100e;
      --teal: #087c75;
      --gold: #b48623;
      --rust: #bd4f30;
      --blue: #426f91;
      --shadow: 0 16px 42px rgba(25, 38, 31, .14);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      background:
        linear-gradient(90deg, rgba(20,23,19,.035) 1px, transparent 1px) 0 0 / 24px 24px,
        linear-gradient(0deg, rgba(20,23,19,.026) 1px, transparent 1px) 0 0 / 24px 24px,
        var(--paper);
      font-family: Avenir Next, Segoe UI, sans-serif;
      letter-spacing: 0;
    }
    header {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(380px, .8fr);
      gap: 24px;
      padding: 24px clamp(16px, 3vw, 42px) 14px;
      border-bottom: 1px solid var(--line);
    }
    h1 {
      margin: 0;
      font: 700 clamp(30px, 4.8vw, 58px)/.96 Georgia, Cambria, serif;
    }
    .sub {
      margin: 10px 0 0;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.45;
      max-width: 900px;
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
    .stats {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      align-content: end;
    }
    .stat, .tile, .case-list, .panel-meta {
      background: rgba(255,255,255,.86);
      border: 1px solid var(--line);
      box-shadow: var(--shadow);
    }
    .stat {
      min-height: 82px;
      padding: 13px;
    }
    .stat span, .tile span {
      display: block;
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: .08em;
    }
    .stat strong {
      display: block;
      margin-top: 7px;
      font: 700 23px Georgia, Cambria, serif;
    }
    main {
      padding: 18px clamp(16px, 3vw, 42px) 42px;
    }
    .top-grid {
      display: grid;
      grid-template-columns: 310px minmax(0, 1fr);
      gap: 14px;
      align-items: start;
    }
    .case-list {
      max-height: calc(100vh - 170px);
      overflow: auto;
      position: sticky;
      top: 12px;
    }
    .case-row {
      width: 100%;
      text-align: left;
      border: 0;
      border-bottom: 1px solid var(--line);
      background: transparent;
      color: var(--ink);
      padding: 10px 12px;
      cursor: pointer;
      font-family: Avenir Next, Segoe UI, sans-serif;
      font-weight: 700;
    }
    .case-row.active { background: rgba(8,124,117,.12); }
    .case-row small {
      display: block;
      margin-top: 4px;
      color: var(--muted);
      font: 11px ui-monospace, SFMono-Regular, Menlo, monospace;
      line-height: 1.3;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .badge {
      display: inline-block;
      padding: 2px 6px;
      margin-left: 6px;
      border: 1px solid var(--line);
      font: 700 10px ui-monospace, SFMono-Regular, Menlo, monospace;
      text-transform: uppercase;
    }
    .badge.improved { color: var(--teal); }
    .badge.worse { color: var(--rust); }
    .badge.mixed { color: var(--gold); }
    .viewer-area { min-width: 0; }
    .strip {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 12px;
    }
    .tile {
      padding: 12px;
      min-height: 96px;
      overflow: hidden;
    }
    .tile strong {
      display: block;
      margin-top: 7px;
      font: 700 22px ui-monospace, SFMono-Regular, Menlo, monospace;
      white-space: nowrap;
    }
    .tile p {
      margin: 6px 0 0;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
    }
    .toolbar {
      display: flex;
      gap: 8px;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 10px;
      padding: 10px 12px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,.78);
    }
    button {
      border: 1px solid var(--line);
      background: #fbfcfa;
      color: var(--ink);
      min-height: 34px;
      padding: 8px 10px;
      font: 700 12px ui-monospace, SFMono-Regular, Menlo, monospace;
      cursor: pointer;
    }
    button.active {
      background: var(--ink);
      color: #fff;
      border-color: var(--ink);
    }
    input[type="range"] {
      width: min(520px, 48vw);
      accent-color: var(--teal);
    }
    .panel-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }
    .view-panel {
      min-width: 0;
      background: var(--night);
      border: 1px solid rgba(255,255,255,.12);
      position: relative;
      height: min(42vh, 430px);
      min-height: 320px;
      overflow: hidden;
    }
    .view-panel canvas {
      width: 100%;
      height: 100%;
      display: block;
    }
    .panel-label {
      position: absolute;
      left: 10px;
      top: 10px;
      z-index: 2;
      color: #f4f6f0;
      background: rgba(11,16,14,.7);
      border: 1px solid rgba(255,255,255,.16);
      padding: 8px 9px;
      backdrop-filter: blur(10px);
      max-width: calc(100% - 20px);
    }
    .panel-label strong {
      display: block;
      font: 700 13px ui-monospace, SFMono-Regular, Menlo, monospace;
    }
    .panel-label small {
      display: block;
      margin-top: 3px;
      color: rgba(244,246,240,.72);
      font: 11px ui-monospace, SFMono-Regular, Menlo, monospace;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .panel-meta {
      margin-top: 12px;
      padding: 12px;
      display: grid;
      grid-template-columns: 1.2fr repeat(5, minmax(0, .8fr));
      gap: 10px;
      align-items: start;
    }
    .panel-meta div { min-width: 0; }
    .panel-meta span {
      display: block;
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: .08em;
    }
    .panel-meta strong {
      display: block;
      margin-top: 5px;
      font: 700 14px ui-monospace, SFMono-Regular, Menlo, monospace;
      overflow-wrap: anywhere;
    }
    .analysis-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      margin-top: 12px;
    }
    .analysis-card {
      min-width: 0;
      background: rgba(255,255,255,.88);
      border: 1px solid var(--line);
      box-shadow: var(--shadow);
      overflow: hidden;
    }
    .analysis-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      min-height: 42px;
      padding: 10px 12px 0;
    }
    .analysis-head strong {
      font: 700 12px ui-monospace, SFMono-Regular, Menlo, monospace;
      text-transform: uppercase;
    }
    .legend {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      color: var(--muted);
      font: 11px ui-monospace, SFMono-Regular, Menlo, monospace;
    }
    .legend i {
      display: inline-block;
      width: 9px;
      height: 9px;
      margin-right: 4px;
      border-radius: 50%;
      vertical-align: -1px;
    }
    .analysis-card canvas {
      width: 100%;
      height: 210px;
      display: block;
    }
    .delta-panel {
      padding: 8px 12px 12px;
      display: grid;
      gap: 7px;
    }
    .delta-line {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 10px;
      align-items: baseline;
      padding: 6px 0;
      border-bottom: 1px solid rgba(20,23,19,.08);
      font: 12px ui-monospace, SFMono-Regular, Menlo, monospace;
    }
    .delta-line span {
      color: var(--muted);
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .delta-line strong.good { color: var(--teal); }
    .delta-line strong.bad { color: var(--rust); }
    .delta-line strong.neutral { color: var(--gold); }
    @media (max-width: 1180px) {
      header, .top-grid { grid-template-columns: 1fr; }
      .case-list { position: relative; max-height: 250px; top: 0; }
      .strip, .analysis-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    @media (max-width: 760px) {
      .stats, .strip, .panel-grid, .panel-meta, .analysis-grid { grid-template-columns: 1fr; }
      .view-panel { height: 360px; }
      input[type="range"] { width: 100%; }
      .toolbar { align-items: stretch; flex-direction: column; }
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
        <a href="../tracker_reward_proto_2k_fixed_noise_fourway/">tracker_reward four-way</a>
        <a href="../hardstable_any2track_fixed_noise/">hardstable four-way</a>
        <a href="../tracker_reward_proto_2k/">tracker reward training</a>
        <a href="../tracker_reward_proto_2k_fixed_noise_fourway/replay_proto_eval/">replay eval</a>
      </nav>
      <h1>fixed-noise proto2k four-way</h1>
      <p class="sub" id="policy"></p>
    </div>
    <div class="stats">
      <div class="stat"><span>Cases</span><strong id="caseCount">24</strong></div>
      <div class="stat"><span>Score Δ</span><strong id="scoreDelta">0</strong></div>
      <div class="stat"><span>Falls</span><strong id="fallDelta">0</strong></div>
    </div>
  </header>
  <main>
    <div class="top-grid">
      <aside class="case-list" id="caseList"></aside>
      <section class="viewer-area">
        <div class="strip" id="summaryTiles"></div>
        <div class="toolbar">
          <div>
            <button id="playBtn" class="active">Pause</button>
            <button data-view="side" class="viewBtn active">Side</button>
            <button data-view="front" class="viewBtn">Front</button>
            <button data-view="top" class="viewBtn">Top</button>
          </div>
          <input id="frameSlider" type="range" min="0" max="1000" value="0">
        </div>
        <div class="panel-grid" id="panelGrid">
          <div class="view-panel"><canvas id="baseRefCanvas"></canvas><div class="panel-label"><strong>base generator</strong><small id="baseRefLabel"></small></div></div>
          <div class="view-panel"><canvas id="protoRefCanvas"></canvas><div class="panel-label"><strong>optimized generator</strong><small id="protoRefLabel"></small></div></div>
          <div class="view-panel"><canvas id="baseTrackCanvas"></canvas><div class="panel-label"><strong>base tracker</strong><small id="baseTrackLabel"></small></div></div>
          <div class="view-panel"><canvas id="protoTrackCanvas"></canvas><div class="panel-label"><strong>optimized tracker</strong><small id="protoTrackLabel"></small></div></div>
        </div>
        <div class="analysis-grid">
          <div class="analysis-card">
            <div class="analysis-head"><strong>Generator root XY</strong><div class="legend"><span><i style="background:#426f91"></i>base</span><span><i style="background:#087c75"></i>optimized</span></div></div>
            <canvas id="generatorTrajCanvas"></canvas>
          </div>
          <div class="analysis-card">
            <div class="analysis-head"><strong>Tracker root XY</strong><div class="legend"><span><i style="background:#426f91"></i>base</span><span><i style="background:#087c75"></i>optimized</span></div></div>
            <canvas id="trackerTrajCanvas"></canvas>
          </div>
          <div class="analysis-card">
            <div class="analysis-head"><strong>Root tracking error</strong><div class="legend"><span><i style="background:#426f91"></i>base</span><span><i style="background:#087c75"></i>optimized</span></div></div>
            <canvas id="errorCurveCanvas"></canvas>
          </div>
          <div class="analysis-card">
            <div class="analysis-head"><strong>Case deltas</strong><div class="legend"><span>optimized - base</span></div></div>
            <div class="delta-panel" id="deltaPanel"></div>
          </div>
        </div>
        <div class="panel-meta" id="caseMeta"></div>
      </section>
    </div>
  </main>
  <script id="payload" type="application/json">__PAYLOAD__</script>
  <script type="module">
    import * as THREE from 'three';
    import { OrbitControls } from './assets/three/jsm/controls/OrbitControls.js';
    import { STLLoader } from './assets/three/jsm/loaders/STLLoader.js';

    const data = JSON.parse(document.getElementById('payload').textContent);
    const panels = [
      { key: 'base_ref', canvas: document.getElementById('baseRefCanvas'), label: document.getElementById('baseRefLabel'), title: 'base generator' },
      { key: 'proto_ref', canvas: document.getElementById('protoRefCanvas'), label: document.getElementById('protoRefLabel'), title: 'optimized generator' },
      { key: 'base_track', canvas: document.getElementById('baseTrackCanvas'), label: document.getElementById('baseTrackLabel'), title: 'base tracker' },
      { key: 'proto_track', canvas: document.getElementById('protoTrackCanvas'), label: document.getElementById('protoTrackLabel'), title: 'optimized tracker' },
    ];
    const loader = new STLLoader();
    const geometryCache = new Map();
    const jsonCache = new Map();
    const materialCache = new Map();
    const PLAYBACK_FPS = 30;
    let activeCase = 0;
    let timeSec = 0;
    let caseDurationSec = 1;
    let lastTickMs = null;
    let playing = true;
    let viewMode = 'side';
    let currentToken = 0;
    let globalBounds = null;
    let activeMotions = null;
    const diagnostics = {
      generator: document.getElementById('generatorTrajCanvas'),
      tracker: document.getElementById('trackerTrajCanvas'),
      error: document.getElementById('errorCurveCanvas'),
      delta: document.getElementById('deltaPanel'),
    };

    function fmt(v, digits = 3) {
      if (v === null || v === undefined || Number.isNaN(Number(v))) return 'n/a';
      return Number(v).toFixed(digits);
    }
    function sourceFps(motion) {
      return Math.max(1e-6, Number(motion && motion.fps ? motion.fps : PLAYBACK_FPS));
    }
    function motionFps(motion) {
      return PLAYBACK_FPS;
    }
    function motionDuration(motion) {
      if (!motion || !motion.frames || !motion.frames.length) return 0;
      return motion.frames.length / motionFps(motion);
    }
    function syncSlider() {
      const value = caseDurationSec > 0 ? Math.round((timeSec / caseDurationSec) * 1000) : 0;
      document.getElementById('frameSlider').value = Math.max(0, Math.min(1000, value));
    }
    function signed(v, digits = 3) {
      if (v === null || v === undefined || Number.isNaN(Number(v))) return 'n/a';
      const n = Number(v);
      return `${n >= 0 ? '+' : ''}${n.toFixed(digits)}`;
    }
    function materialFor(name, panelKey) {
      const side = name.includes('left') ? 'left' : name.includes('right') ? 'right' : name.includes('torso') || name.includes('waist') ? 'core' : 'base';
      const cacheKey = `${panelKey}:${side}`;
      if (materialCache.has(cacheKey)) return materialCache.get(cacheKey);
      const palettes = {
        base_ref: { left: 0x6f8aa4, right: 0xa98e57, core: 0xc9d2ca, base: 0xb9c1ba },
        proto_ref: { left: 0x60beb8, right: 0xd3a23b, core: 0xe0e5dc, base: 0xc1cbc3 },
        base_track: { left: 0x5f83ad, right: 0xb37e4b, core: 0xbfc8c2, base: 0xaeb8b0 },
        proto_track: { left: 0x4fc8bc, right: 0xe0a63d, core: 0xe6eadf, base: 0xcbd4cc },
      };
      const mat = new THREE.MeshStandardMaterial({
        color: palettes[panelKey][side],
        roughness: 0.62,
        metalness: 0.16,
      });
      materialCache.set(cacheKey, mat);
      return mat;
    }
    function meshBasename(file) {
      return String(file || '').split('/').pop();
    }
    function loadGeometry(file) {
      const cleanFile = meshBasename(file);
      if (!geometryCache.has(cleanFile)) {
        geometryCache.set(cleanFile, new Promise((resolve, reject) => {
          loader.load(`assets/g1_mesh/${encodeURIComponent(cleanFile)}`, (geometry) => {
            geometry.computeVertexNormals();
            resolve(geometry);
          }, undefined, reject);
        }));
      }
      return geometryCache.get(cleanFile);
    }
    async function loadJson(path) {
      if (!jsonCache.has(path)) {
        jsonCache.set(path, fetch(path).then((r) => {
          if (!r.ok) throw new Error(`${r.status} ${path}`);
          return r.json();
        }));
      }
      return jsonCache.get(path);
    }
    function setQuat(obj, q) {
      obj.quaternion.set(q[1], q[2], q[3], q[0]);
      obj.quaternion.normalize();
    }
    function makePanel(panel) {
      const renderer = new THREE.WebGLRenderer({ canvas: panel.canvas, antialias: true, preserveDrawingBuffer: true });
      renderer.setClearColor(0x0b100e, 1);
      renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
      const scene = new THREE.Scene();
      scene.fog = new THREE.Fog(0x0b100e, 5.2, 10.5);
      const camera = new THREE.PerspectiveCamera(42, 1, 0.01, 80);
      camera.up.set(0, 0, 1);
      const controls = new OrbitControls(camera, panel.canvas);
      controls.enableDamping = true;
      controls.dampingFactor = 0.08;
      controls.enablePan = false;
      const root = new THREE.Group();
      scene.add(root);
      const grid = new THREE.GridHelper(4.6, 23, 0x4d5c55, 0x25302c);
      grid.rotation.x = Math.PI / 2;
      scene.add(grid);
      const hemi = new THREE.HemisphereLight(0xecfff8, 0x1d2a26, 2.7);
      scene.add(hemi);
      const key = new THREE.DirectionalLight(0xffffff, 2.2);
      key.position.set(2.6, -4.2, 4.0);
      scene.add(key);
      const rim = new THREE.DirectionalLight(0xd2fff2, 1.0);
      rim.position.set(-3, 2.2, 2.4);
      scene.add(rim);
      Object.assign(panel, { renderer, scene, camera, controls, root, bodyGroups: [], motion: null });
    }
    panels.forEach(makePanel);

    function clearRoot(panel) {
      while (panel.root.children.length) panel.root.remove(panel.root.children[0]);
      panel.bodyGroups = [];
    }
    async function buildRobot(panel, motion, token) {
      clearRoot(panel);
      motion.bodies.forEach((body) => {
        const group = new THREE.Group();
        panel.root.add(group);
        panel.bodyGroups.push(group);
      });
      for (let i = 0; i < motion.bodies.length; i += 1) {
        const body = motion.bodies[i];
        const group = panel.bodyGroups[i];
        for (const meshDef of body.meshes || []) {
          try {
            const geometry = await loadGeometry(meshDef.file);
            if (token !== currentToken) return;
            const mesh = new THREE.Mesh(geometry, materialFor(body.name, panel.key));
            mesh.position.set(meshDef.pos[0], meshDef.pos[1], meshDef.pos[2]);
            setQuat(mesh, meshDef.quat);
            group.add(mesh);
          } catch (err) {
            console.warn(`mesh load failed: ${meshDef.file}`, err);
          }
        }
      }
      applyPanelFrame(panel, timeSec);
    }
    function motionBounds(motions) {
      const xs = [], ys = [], zs = [];
      motions.forEach((motion) => {
        motion.frames.forEach((frame) => {
          (frame.body_pos || []).forEach((p) => {
            xs.push(p[0]); ys.push(p[1]); zs.push(p[2]);
          });
        });
      });
      const pad = 0.25;
      return {
        x0: Math.min(...xs) - pad, x1: Math.max(...xs) + pad,
        y0: Math.min(...ys) - pad, y1: Math.max(...ys) + pad,
        z0: Math.min(...zs) - pad, z1: Math.max(...zs) + pad,
      };
    }
    function center(bounds) {
      return new THREE.Vector3((bounds.x0 + bounds.x1) / 2, (bounds.y0 + bounds.y1) / 2, (bounds.z0 + bounds.z1) / 2);
    }
    function span(bounds) {
      return Math.max(bounds.x1 - bounds.x0, bounds.y1 - bounds.y0, bounds.z1 - bounds.z0, 1.15);
    }
    function setCamera(panel) {
      if (!globalBounds) return;
      const c = center(globalBounds);
      const s = span(globalBounds);
      panel.controls.target.set(c.x, c.y, Math.max(0.72, c.z + 0.05));
      if (viewMode === 'front') {
        panel.camera.up.set(0, 0, 1);
        panel.camera.position.set(c.x, c.y - s * 2.25, c.z + s * 0.5);
      } else if (viewMode === 'top') {
        panel.camera.up.set(0, 1, 0);
        panel.camera.position.set(c.x, c.y, c.z + s * 2.35);
      } else {
        panel.camera.up.set(0, 0, 1);
        panel.camera.position.set(c.x + s * 1.85, c.y - s * 0.58, c.z + s * 0.6);
      }
      panel.camera.lookAt(panel.controls.target);
      panel.camera.updateProjectionMatrix();
      panel.controls.update();
    }
    function applyPanelFrame(panel, tSec) {
      const motion = panel.motion;
      if (!motion || !motion.frames.length) return;
      const fps = motionFps(motion);
      const duration = motionDuration(motion);
      const clamped = Math.max(0, Math.min(tSec, Math.max(0, duration - (1 / fps))));
      const idx = Math.max(0, Math.min(motion.frames.length - 1, Math.floor(clamped * fps + 1e-6)));
      const frame = motion.frames[idx];
      panel.bodyGroups.forEach((group, i) => {
        const p = frame.body_pos[i];
        const q = frame.body_quat[i];
        if (!p || !q) return;
        group.position.set(p[0], p[1], p[2]);
        setQuat(group, q);
      });
      const ended = tSec >= duration && duration > 0 ? ' | ended' : '';
      panel.label.textContent = `${fmt(idx / fps, 2)}s/${fmt(duration, 2)}s | frame ${idx + 1}/${motion.frames.length} | 30fps playback | src ${fmt(sourceFps(motion), 1)}${ended}`;
    }
    async function loadCase(index) {
      activeCase = index;
      timeSec = 0;
      lastTickMs = null;
      syncSlider();
      renderCaseList();
      renderCaseMeta();
      const token = ++currentToken;
      const row = data.rows[activeCase];
      const motions = await Promise.all(panels.map((panel) => loadJson(row.paths[panel.key])));
      if (token !== currentToken) return;
      activeMotions = motions;
      caseDurationSec = Math.max(0.1, ...motions.map(motionDuration));
      syncSlider();
      globalBounds = motionBounds(motions);
      for (let i = 0; i < panels.length; i += 1) {
        panels[i].motion = motions[i];
        setCamera(panels[i]);
        await buildRobot(panels[i], motions[i], token);
      }
      drawDiagnostics(row, motions);
      window.__fourwayReady = true;
      window.__fourwayInfo = {
        case: row.stem,
        timeMode: 'shared_seconds',
        durationSec: caseDurationSec,
        panels: panels.map((p) => ({
          key: p.key,
          frames: p.motion.frames.length,
          fps: PLAYBACK_FPS,
          sourceFps: sourceFps(p.motion),
          durationSec: motionDuration(p.motion),
          bodies: p.motion.bodies.length,
        })),
      };
    }
    function renderSummary() {
      const s = data.summary;
      const base = s[data.run.base] || {};
      const proto = s[data.run.proto] || {};
      const scoreDelta = (proto.adversarial_score_mean ?? 0) - (base.adversarial_score_mean ?? 0);
      const fallDelta = (proto.fall_rate ?? 0) - (base.fall_rate ?? 0);
      document.getElementById('policy').textContent = `${data.run.same_noise_policy} | playback=30fps for every panel | source=${data.run.source}`;
      document.getElementById('caseCount').textContent = data.rows.length;
      document.getElementById('scoreDelta').textContent = signed(scoreDelta, 3);
      document.getElementById('fallDelta').textContent = `${fmt((base.fall_rate ?? 0) * 100, 1)}%→${fmt((proto.fall_rate ?? 0) * 100, 1)}%`;
      const tiles = [
        ['Adversarial score', `${fmt(base.adversarial_score_mean)} → ${fmt(proto.adversarial_score_mean)}`, signed(scoreDelta), scoreDelta <= 0],
        ['Trackable basic', `${fmt((base.trackable_basic_rate ?? 0) * 100, 1)}% → ${fmt((proto.trackable_basic_rate ?? 0) * 100, 1)}%`, `${data.counts.trackable_gained} gained / ${data.counts.trackable_lost} lost`, (proto.trackable_basic_rate ?? 0) >= (base.trackable_basic_rate ?? 0)],
        ['Falls', `${fmt((base.fall_rate ?? 0) * 100, 1)}% → ${fmt((proto.fall_rate ?? 0) * 100, 1)}%`, `${data.counts.falls_fixed} fixed / ${data.counts.new_falls} new`, (proto.fall_rate ?? 0) <= (base.fall_rate ?? 0)],
        ['Root trajectory err', `${fmt(base.root_trajectory_error_mean_m)} → ${fmt(proto.root_trajectory_error_mean_m)}`, signed((proto.root_trajectory_error_mean_m ?? 0) - (base.root_trajectory_error_mean_m ?? 0)), (proto.root_trajectory_error_mean_m ?? 0) <= (base.root_trajectory_error_mean_m ?? 0)],
      ];
      const host = document.getElementById('summaryTiles');
      host.innerHTML = '';
      tiles.forEach(([label, value, note, good]) => {
        const tile = document.createElement('div');
        tile.className = 'tile';
        tile.style.borderBottom = `5px solid ${good ? 'var(--teal)' : 'var(--rust)'}`;
        tile.innerHTML = `<span>${label}</span><strong>${value}</strong><p>${note}</p>`;
        host.appendChild(tile);
      });
    }
    function renderCaseList() {
      const host = document.getElementById('caseList');
      host.innerHTML = '';
      data.rows.forEach((row, i) => {
        const btn = document.createElement('button');
        btn.className = `case-row${i === activeCase ? ' active' : ''}`;
        const d = row.delta;
        btn.innerHTML = `${row.stem}<span class="badge ${d.verdict}">${d.verdict}</span><small>score Δ ${signed(d.score_delta)} | root Δ ${signed(d.root_delta)} | trackable Δ ${signed(d.trackable_delta, 0)}</small><small>${row.prompt}</small>`;
        btn.onclick = () => loadCase(i);
        host.appendChild(btn);
      });
    }
    function renderCaseMeta() {
      const row = data.rows[activeCase];
      const base = row.metrics.base;
      const proto = row.metrics.proto;
      const d = row.delta;
      const host = document.getElementById('caseMeta');
      host.innerHTML = `
        <div><span>Case</span><strong>${row.stem} | ${row.prompt_id}<br>${row.prompt}</strong></div>
        <div><span>Noise</span><strong>${row.noise_seed}</strong></div>
        <div><span>Score</span><strong>${fmt(base.score)} → ${fmt(proto.score)}<br>${signed(d.score_delta)}</strong></div>
        <div><span>Trackable</span><strong>${base.trackable ? 'yes' : 'no'} → ${proto.trackable ? 'yes' : 'no'}<br>${signed(d.trackable_delta, 0)}</strong></div>
        <div><span>Fall</span><strong>${base.fall ? 'yes' : 'no'} → ${proto.fall ? 'yes' : 'no'}</strong></div>
        <div><span>Root err</span><strong>${fmt(base.root_traj)} → ${fmt(proto.root_traj)}<br>${signed(d.root_delta)}</strong></div>
      `;
      renderDeltaPanel(row);
    }
    function rootPath(motion) {
      return motion.frames
        .map((frame) => (frame.body_pos && frame.body_pos[0] ? [Number(frame.body_pos[0][0]), Number(frame.body_pos[0][1]), Number(frame.body_pos[0][2] || 0)] : null))
        .filter(Boolean);
    }
    function normalizePath(path) {
      if (!path.length) return [];
      const ox = path[0][0];
      const oy = path[0][1];
      return path.map((p) => [p[0] - ox, p[1] - oy, p[2]]);
    }
    function samplePath(path, t) {
      if (!path.length) return [0, 0, 0];
      const idx = Math.max(0, Math.min(path.length - 1, Math.round(t * (path.length - 1))));
      return path[idx];
    }
    function sampleMotionRootAtTime(motion, tSec) {
      const path = rootPath(motion);
      if (!path.length) return [0, 0, 0];
      const fps = motionFps(motion);
      const duration = motionDuration(motion);
      const clamped = Math.max(0, Math.min(tSec, Math.max(0, duration - (1 / fps))));
      const idx = Math.max(0, Math.min(path.length - 1, Math.floor(clamped * fps + 1e-6)));
      return path[idx];
    }
    function rootError(refMotion, trackMotion, steps = 160) {
      const values = [];
      if (!rootPath(refMotion).length || !rootPath(trackMotion).length) return values;
      const duration = Math.max(motionDuration(refMotion), motionDuration(trackMotion), 1e-6);
      for (let i = 0; i < steps; i += 1) {
        const t = steps === 1 ? 0 : (i / (steps - 1)) * duration;
        const a = sampleMotionRootAtTime(refMotion, t);
        const b = sampleMotionRootAtTime(trackMotion, t);
        const dx = a[0] - b[0];
        const dy = a[1] - b[1];
        const dz = a[2] - b[2];
        values.push(Math.sqrt(dx * dx + dy * dy + dz * dz));
      }
      return values;
    }
    function prepareCanvas(canvas) {
      const rect = canvas.getBoundingClientRect();
      const ratio = Math.min(2, window.devicePixelRatio || 1);
      const w = Math.max(240, Math.floor(rect.width));
      const h = Math.max(180, Math.floor(rect.height));
      canvas.width = Math.floor(w * ratio);
      canvas.height = Math.floor(h * ratio);
      const ctx = canvas.getContext('2d');
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      return { ctx, w, h };
    }
    function drawGrid(ctx, w, h) {
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = '#fbfcfa';
      ctx.fillRect(0, 0, w, h);
      ctx.strokeStyle = 'rgba(20,23,19,.08)';
      ctx.lineWidth = 1;
      for (let x = 20; x < w; x += 32) {
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
      }
      for (let y = 20; y < h; y += 32) {
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
      }
    }
    function pathBounds(paths) {
      const xs = [];
      const ys = [];
      paths.forEach((path) => path.forEach((p) => { xs.push(p[0]); ys.push(p[1]); }));
      if (!xs.length) return { x0: -1, x1: 1, y0: -1, y1: 1 };
      let x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
      const xPad = Math.max((x1 - x0) * .12, .12);
      const yPad = Math.max((y1 - y0) * .12, .12);
      if (Math.abs(x1 - x0) < .08) { x0 -= .1; x1 += .1; }
      if (Math.abs(y1 - y0) < .08) { y0 -= .1; y1 += .1; }
      return { x0: x0 - xPad, x1: x1 + xPad, y0: y0 - yPad, y1: y1 + yPad };
    }
    function drawPathCanvas(canvas, series) {
      const { ctx, w, h } = prepareCanvas(canvas);
      drawGrid(ctx, w, h);
      const margin = 28;
      const paths = series.map((item) => normalizePath(item.path));
      const b = pathBounds(paths);
      const sx = (w - margin * 2) / Math.max(b.x1 - b.x0, .001);
      const sy = (h - margin * 2) / Math.max(b.y1 - b.y0, .001);
      const scale = Math.min(sx, sy);
      const tx = (x) => margin + (x - b.x0) * scale;
      const ty = (y) => h - margin - (y - b.y0) * scale;
      paths.forEach((path, idx) => {
        if (path.length < 2) return;
        ctx.save();
        ctx.strokeStyle = series[idx].color;
        ctx.lineWidth = series[idx].width || 2.5;
        ctx.globalAlpha = series[idx].alpha || 1;
        if (series[idx].dash) ctx.setLineDash(series[idx].dash);
        ctx.beginPath();
        path.forEach((p, i) => {
          const x = tx(p[0]);
          const y = ty(p[1]);
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.stroke();
        ctx.setLineDash([]);
        const start = path[0];
        const end = path[path.length - 1];
        ctx.fillStyle = series[idx].color;
        ctx.beginPath(); ctx.arc(tx(start[0]), ty(start[1]), 4, 0, Math.PI * 2); ctx.fill();
        ctx.strokeStyle = series[idx].color;
        ctx.lineWidth = 2;
        ctx.beginPath(); ctx.arc(tx(end[0]), ty(end[1]), 5.5, 0, Math.PI * 2); ctx.stroke();
        ctx.restore();
      });
      ctx.fillStyle = 'rgba(20,23,19,.58)';
      ctx.font = '11px ui-monospace, SFMono-Regular, Menlo, monospace';
      ctx.fillText('start = filled dot, end = ring', 12, h - 10);
    }
    function drawErrorCanvas(canvas, baseValues, protoValues) {
      const { ctx, w, h } = prepareCanvas(canvas);
      drawGrid(ctx, w, h);
      const margin = 28;
      const maxValue = Math.max(.05, ...baseValues, ...protoValues);
      function draw(values, color) {
        if (values.length < 2) return;
        ctx.strokeStyle = color;
        ctx.lineWidth = 2.4;
        ctx.beginPath();
        values.forEach((v, i) => {
          const x = margin + (i / (values.length - 1)) * (w - margin * 2);
          const y = h - margin - (v / maxValue) * (h - margin * 2);
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.stroke();
      }
      draw(baseValues, '#426f91');
      draw(protoValues, '#087c75');
      ctx.fillStyle = 'rgba(20,23,19,.62)';
      ctx.font = '11px ui-monospace, SFMono-Regular, Menlo, monospace';
      ctx.fillText(`0`, 9, h - margin + 4);
      ctx.fillText(`${maxValue.toFixed(2)}m`, 9, margin + 4);
      const baseMean = baseValues.reduce((a, b) => a + b, 0) / Math.max(1, baseValues.length);
      const protoMean = protoValues.reduce((a, b) => a + b, 0) / Math.max(1, protoValues.length);
      ctx.fillText(`mean ${baseMean.toFixed(3)} → ${protoMean.toFixed(3)}m`, 12, h - 10);
    }
    function renderDeltaPanel(row) {
      const base = row.metrics.base;
      const proto = row.metrics.proto;
      const d = row.delta;
      const items = [
        ['score', d.score_delta, 'lower'],
        ['root traj err', d.root_delta, 'lower'],
        ['max joint err', d.joint_delta, 'lower'],
        ['completion', d.completion_delta, 'higher'],
        ['foot skate', (proto.foot_skate ?? 0) - (base.foot_skate ?? 0), 'lower'],
        ['jerk', (proto.jerk ?? 0) - (base.jerk ?? 0), 'lower'],
        ['ref path', (proto.ref_path ?? 0) - (base.ref_path ?? 0), 'neutral'],
        ['track path', (proto.track_path ?? 0) - (base.track_path ?? 0), 'neutral'],
      ];
      diagnostics.delta.innerHTML = items.map(([label, value, direction]) => {
        const n = Number(value || 0);
        let cls = 'neutral';
        if (direction === 'lower') cls = n < 0 ? 'good' : n > 0 ? 'bad' : 'neutral';
        if (direction === 'higher') cls = n > 0 ? 'good' : n < 0 ? 'bad' : 'neutral';
        return `<div class="delta-line"><span>${label}</span><strong class="${cls}">${signed(n)}</strong></div>`;
      }).join('');
    }
    function drawDiagnostics(row, motions) {
      const motionByKey = Object.fromEntries(panels.map((panel, i) => [panel.key, motions[i]]));
      drawPathCanvas(diagnostics.generator, [
        { path: rootPath(motionByKey.base_ref), color: '#426f91' },
        { path: rootPath(motionByKey.proto_ref), color: '#087c75' },
      ]);
      drawPathCanvas(diagnostics.tracker, [
        { path: rootPath(motionByKey.base_track), color: '#426f91' },
        { path: rootPath(motionByKey.proto_track), color: '#087c75' },
        { path: rootPath(motionByKey.base_ref), color: '#426f91', alpha: .32, width: 1.5, dash: [6, 5] },
        { path: rootPath(motionByKey.proto_ref), color: '#087c75', alpha: .32, width: 1.5, dash: [6, 5] },
      ]);
      drawErrorCanvas(
        diagnostics.error,
        rootError(motionByKey.base_ref, motionByKey.base_track),
        rootError(motionByKey.proto_ref, motionByKey.proto_track),
      );
      renderDeltaPanel(row);
    }
    function resize() {
      panels.forEach((panel) => {
        const rect = panel.canvas.getBoundingClientRect();
        const w = Math.max(1, Math.floor(rect.width));
        const h = Math.max(1, Math.floor(rect.height));
        panel.renderer.setSize(w, h, false);
        panel.camera.aspect = w / h;
        panel.camera.updateProjectionMatrix();
      });
      if (activeMotions) drawDiagnostics(data.rows[activeCase], activeMotions);
    }
    function animate(now) {
      requestAnimationFrame(animate);
      if (lastTickMs === null) lastTickMs = now || performance.now();
      const deltaSec = Math.min(0.1, Math.max(0, ((now || performance.now()) - lastTickMs) / 1000));
      lastTickMs = now || performance.now();
      if (playing && panels.every((p) => p.motion)) {
        timeSec += deltaSec;
        if (timeSec >= caseDurationSec) timeSec %= caseDurationSec;
        syncSlider();
      }
      panels.forEach((panel) => {
        applyPanelFrame(panel, timeSec);
        panel.controls.update();
        panel.renderer.render(panel.scene, panel.camera);
      });
    }
    document.getElementById('playBtn').onclick = () => {
      playing = !playing;
      document.getElementById('playBtn').textContent = playing ? 'Pause' : 'Play';
      document.getElementById('playBtn').classList.toggle('active', playing);
    };
    document.getElementById('frameSlider').oninput = (event) => {
      playing = false;
      document.getElementById('playBtn').textContent = 'Play';
      document.getElementById('playBtn').classList.remove('active');
      timeSec = (Number(event.target.value) / 1000) * caseDurationSec;
    };
    document.querySelectorAll('.viewBtn').forEach((btn) => {
      btn.onclick = () => {
        viewMode = btn.dataset.view;
        document.querySelectorAll('.viewBtn').forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
        panels.forEach(setCamera);
      };
    });
    window.addEventListener('resize', () => {
      resize();
      panels.forEach(setCamera);
    });
    renderSummary();
    renderCaseList();
    resize();
    loadCase(0);
    animate();
  </script>
</body>
</html>
"""
    return template.replace("fixed-noise proto2k four-way", PAGE_TITLE).replace("__PAYLOAD__", payload)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    copy_assets()
    data = build_data()
    OUT_HTML.write_text(html_doc(data))
    print(OUT_HTML)


if __name__ == "__main__":
    main()
