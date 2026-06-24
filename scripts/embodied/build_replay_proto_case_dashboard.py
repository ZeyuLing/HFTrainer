#!/usr/bin/env python3
"""Build a case-level before/after G1 mesh dashboard for replay_proto eval."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch

import sys

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2]))
from scripts.embodied.physflow_triplet_manifest import (  # noqa: E402
    DEFAULT_BODIES,
    MESHES_BY_BODY,
    _parse_g1_body_meshes,
)


ROOT = Path("/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer")
DEFAULT_EVAL_ROOT = (
    ROOT / "output/lafan1_g1_proto_baseline_eval/gt_replay_caseviz_lafan_20260618_r3"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "output/physflow_visualizations/"
    "tracker_reward_proto_2k_fixed_noise_fourway/replay_proto_cases"
)
THREE_SRC = ROOT / "motion_annot_web/score_m2m/static/three"
G1_MESH_SRC = (
    ROOT
    / "hftrainer/models/motion/physflow/trackers/protomotions/vendor/protomotions/data/assets/mesh/G1"
)

BEFORE = "protomotions_g1_bones"
AFTER = "gt_replay_after"


def _load(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def _as_np(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _bodies_meta() -> list[dict[str, Any]]:
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


def _latest_predicted(root: Path) -> Path | None:
    candidates = sorted((root / "results").glob("predicted_motion_lib_epoch_*.pt"))
    return candidates[-1] if candidates else None


def _starts_lens(lib: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    starts = _as_np(lib["length_starts"]).astype(np.int64)
    if "motion_num_frames" in lib:
        lens = _as_np(lib["motion_num_frames"]).astype(np.int64)
    else:
        total = int(lib["gts"].shape[0])
        lens = np.asarray(
            [
                int(starts[i + 1] - starts[i]) if i + 1 < len(starts) else int(total - starts[i])
                for i in range(len(starts))
            ],
            dtype=np.int64,
        )
    return starts, lens


def _slice_lib(lib: dict[str, Any], motion_id: int) -> tuple[np.ndarray, np.ndarray, float, str]:
    starts, lens = _starts_lens(lib)
    start = int(starts[motion_id])
    end = start + int(lens[motion_id])
    pos = _as_np(lib["gts"][start:end]).astype(np.float32).copy()
    quat_xyzw = _as_np(lib["grs"][start:end]).astype(np.float32).copy()
    dt = float(_as_np(lib["motion_dt"]).reshape(-1)[motion_id])
    motion_file = str(list(lib["motion_files"])[motion_id])
    return pos, quat_xyzw, dt, motion_file


def _ref_motion(path: Path, n_frames: int) -> tuple[np.ndarray, np.ndarray, int]:
    ref = _load(path)
    pos = _as_np(ref["rigid_body_pos"]).astype(np.float32)[:n_frames].copy()
    quat_xyzw = _as_np(ref["rigid_body_rot"]).astype(np.float32)[:n_frames].copy()
    fps = int(ref.get("fps", 30))
    return pos, quat_xyzw, fps


def _strip_env_offset(pos: np.ndarray, ref_xy0: np.ndarray) -> np.ndarray:
    out = pos.copy()
    out[..., :2] -= out[0, 0, :2] - ref_xy0
    return out


def _quat_to_mat_wxyz(q: np.ndarray) -> np.ndarray:
    q = q / np.linalg.norm(q, axis=-1, keepdims=True).clip(min=1e-9)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], axis=-1),
            np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], axis=-1),
            np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], axis=-1),
        ],
        axis=-2,
    ).astype(np.float32)


def _local_pos(pos: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    quat_wxyz = quat_xyzw[..., [3, 0, 1, 2]]
    root = pos[:, :1, :]
    root_rot = _quat_to_mat_wxyz(quat_wxyz[:, 0, :])
    rel = pos - root
    return np.einsum("tbc,tcd->tbd", rel, root_rot)


def _motion_metrics(ref_pos: np.ndarray, ref_quat: np.ndarray, pred_pos: np.ndarray, pred_quat: np.ndarray) -> dict[str, float]:
    frames = min(len(ref_pos), len(pred_pos))
    ref_pos = ref_pos[:frames]
    pred_pos = pred_pos[:frames]
    ref_quat = ref_quat[:frames]
    pred_quat = pred_quat[:frames]
    ref_local = _local_pos(ref_pos, ref_quat)
    pred_local = _local_pos(pred_pos, pred_quat)
    root_err = np.linalg.norm(pred_pos[:, 0, :] - ref_pos[:, 0, :], axis=-1)
    body_err = np.linalg.norm(pred_pos - ref_pos, axis=-1)
    local_err = np.linalg.norm(pred_local - ref_local, axis=-1)
    root_height_err = np.abs(pred_pos[:, 0, 2] - ref_pos[:, 0, 2])
    ref_disp = float(np.linalg.norm(ref_pos[-1, 0, :2] - ref_pos[0, 0, :2])) if frames > 1 else 0.0
    pred_disp = float(np.linalg.norm(pred_pos[-1, 0, :2] - pred_pos[0, 0, :2])) if frames > 1 else 0.0
    local_mpjpe_m = float(local_err.mean())
    root_height_err_m = float(root_height_err.mean())
    success = float(local_mpjpe_m <= 0.2 and root_height_err_m <= 0.2)
    return {
        "frames": float(frames),
        "success": success,
        "root_err_m": float(root_err.mean()),
        "root_err_max_m": float(root_err.max()),
        "mpjpe_mm": float(body_err.mean() * 1000.0),
        "local_mpjpe_mm": float(local_mpjpe_m * 1000.0),
        "root_height_err_m": root_height_err_m,
        "ref_disp_m": ref_disp,
        "track_disp_m": pred_disp,
        "disp_err_m": abs(ref_disp - pred_disp),
    }


def _series(ref_pos: np.ndarray, before_pos: np.ndarray, after_pos: np.ndarray) -> dict[str, Any]:
    frames = min(len(ref_pos), len(before_pos), len(after_pos))
    ref = ref_pos[:frames, 0, :]
    before = before_pos[:frames, 0, :]
    after = after_pos[:frames, 0, :]
    root_before = np.linalg.norm(before - ref, axis=-1)
    root_after = np.linalg.norm(after - ref, axis=-1)
    limit = min(220, frames)
    idx = np.linspace(0, frames - 1, limit).round().astype(np.int64) if frames else np.asarray([], dtype=np.int64)
    return {
        "ref_xy": ref[idx, :2].round(4).tolist(),
        "before_xy": before[idx, :2].round(4).tolist(),
        "after_xy": after[idx, :2].round(4).tolist(),
        "root_err_before": root_before[idx].round(4).tolist(),
        "root_err_after": root_after[idx].round(4).tolist(),
    }


def _write_robot_frames(path: Path, pos: np.ndarray, quat_xyzw: np.ndarray, fps: int, bodies: list[dict[str, Any]]) -> None:
    quat_wxyz = quat_xyzw[..., [3, 0, 1, 2]].astype(np.float32)
    frames = [
        {"body_pos": pos[i].astype(float).tolist(), "body_quat": quat_wxyz[i].astype(float).tolist()}
        for i in range(len(pos))
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "type": "robot_frames",
                "robot": "g1",
                "fps": int(fps),
                "num_frames": len(frames),
                "num_bodies": len(bodies),
                "bodies": bodies,
                "frames": frames,
            },
            separators=(",", ":"),
        )
    )


def _copy_assets(out_dir: Path) -> None:
    for rel in [
        Path("three.module.js"),
        Path("jsm/controls/OrbitControls.js"),
        Path("jsm/loaders/STLLoader.js"),
    ]:
        src = THREE_SRC / rel
        dst = out_dir / "assets" / "three" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    mesh_dir = out_dir / "assets" / "g1_mesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    for src in sorted(G1_MESH_SRC.glob("*.stl")):
        shutil.copy2(src, mesh_dir / src.name)


def _rel(path: Path, out_dir: Path) -> str:
    return str(path.relative_to(out_dir))


def _score_delta(before: dict[str, float], after: dict[str, float]) -> dict[str, Any]:
    root_delta = after["root_err_m"] - before["root_err_m"]
    mpjpe_delta = after["mpjpe_mm"] - before["mpjpe_mm"]
    local_delta = after["local_mpjpe_mm"] - before["local_mpjpe_mm"]
    success_delta = after["success"] - before["success"]
    composite = -root_delta * 1.0 - mpjpe_delta / 1000.0 * 0.45 - local_delta / 1000.0 * 0.55 + success_delta * 0.3
    if success_delta > 0 or composite > 0.08:
        verdict = "improved"
    elif success_delta < 0 or composite < -0.08:
        verdict = "worse"
    else:
        verdict = "mixed"
    return {
        "root_err_delta_m": root_delta,
        "mpjpe_delta_mm": mpjpe_delta,
        "local_mpjpe_delta_mm": local_delta,
        "success_delta": success_delta,
        "composite": composite,
        "verdict": verdict,
    }


def _case_rows(eval_root: Path, out_dir: Path, max_cases: int) -> list[dict[str, Any]]:
    bodies = _bodies_meta()
    before_rows: list[tuple[int, int, dict[str, Any], Path]] = []
    all_rows: list[dict[str, Any]] = []

    for shard_dir in sorted((eval_root / f"eval_{BEFORE}").glob("predicted_shard_*")):
        shard = int(shard_dir.name.rsplit("_", 1)[-1])
        before_path = _latest_predicted(shard_dir)
        after_path = _latest_predicted(eval_root / f"eval_{AFTER}" / f"predicted_shard_{shard}")
        if before_path is None or after_path is None:
            continue
        before_lib = _load(before_path)
        after_lib = _load(after_path)
        n = min(len(list(before_lib["motion_files"])), len(list(after_lib["motion_files"])))
        for motion_id in range(n):
            before_rows.append((shard, motion_id, {"before": before_lib, "after": after_lib}, before_path))

    if not before_rows:
        raise FileNotFoundError(f"No predicted_motion_lib files found under {eval_root}")

    for shard, motion_id, libs, _before_path in before_rows:
        before_pos_raw, before_quat, before_dt, before_motion_file = _slice_lib(libs["before"], motion_id)
        after_pos_raw, after_quat, after_dt, after_motion_file = _slice_lib(libs["after"], motion_id)
        ref_path = Path(before_motion_file)
        if not ref_path.exists():
            ref_path = Path(after_motion_file)
        frames = min(len(before_pos_raw), len(after_pos_raw))
        ref_pos, ref_quat, ref_fps = _ref_motion(ref_path, frames)
        frames = min(frames, len(ref_pos))
        ref_pos = ref_pos[:frames]
        ref_quat = ref_quat[:frames]
        before_pos = _strip_env_offset(before_pos_raw[:frames], ref_pos[0, 0, :2])
        after_pos = _strip_env_offset(after_pos_raw[:frames], ref_pos[0, 0, :2])
        before_quat = before_quat[:frames]
        after_quat = after_quat[:frames]

        before_metrics = _motion_metrics(ref_pos, ref_quat, before_pos, before_quat)
        after_metrics = _motion_metrics(ref_pos, ref_quat, after_pos, after_quat)
        delta = _score_delta(before_metrics, after_metrics)
        stem = ref_path.stem
        case_id = f"s{shard:02d}_m{motion_id:03d}_{stem}"
        case_dir = out_dir / "data" / case_id
        ref_json = case_dir / "reference.json"
        before_json = case_dir / "before_track.json"
        after_json = case_dir / "after_track.json"
        _write_robot_frames(ref_json, ref_pos, ref_quat, ref_fps, bodies)
        _write_robot_frames(before_json, before_pos, before_quat, int(round(1.0 / max(before_dt, 1e-9))), bodies)
        _write_robot_frames(after_json, after_pos, after_quat, int(round(1.0 / max(after_dt, 1e-9))), bodies)
        all_rows.append(
            {
                "id": case_id,
                "stem": stem,
                "shard": shard,
                "motion_id": motion_id,
                "source": str(ref_path),
                "paths": {
                    "reference": _rel(ref_json, out_dir),
                    "before": _rel(before_json, out_dir),
                    "after": _rel(after_json, out_dir),
                },
                "metrics": {
                    "before": before_metrics,
                    "after": after_metrics,
                    "delta": delta,
                    "series": _series(ref_pos, before_pos, after_pos),
                },
            }
        )

    improved = sorted(
        [r for r in all_rows if r["metrics"]["delta"]["verdict"] == "improved"],
        key=lambda r: -r["metrics"]["delta"]["composite"],
    )
    worse = sorted(
        [r for r in all_rows if r["metrics"]["delta"]["verdict"] == "worse"],
        key=lambda r: r["metrics"]["delta"]["composite"],
    )
    mixed = sorted(
        [r for r in all_rows if r["metrics"]["delta"]["verdict"] == "mixed"],
        key=lambda r: abs(r["metrics"]["delta"]["composite"]),
        reverse=True,
    )
    selected = improved[: max_cases // 2] + worse[: max(2, max_cases // 4)] + mixed[: max_cases]
    unique: dict[str, dict[str, Any]] = {}
    for row in selected:
        unique.setdefault(row["id"], row)
    rows = list(unique.values())[:max_cases]
    rows.sort(key=lambda r: (r["metrics"]["delta"]["verdict"] != "improved", -r["metrics"]["delta"]["composite"]))
    return rows


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    all_rows = rows
    return {
        "cases": len(all_rows),
        "improved": sum(1 for r in all_rows if r["metrics"]["delta"]["verdict"] == "improved"),
        "worse": sum(1 for r in all_rows if r["metrics"]["delta"]["verdict"] == "worse"),
        "mixed": sum(1 for r in all_rows if r["metrics"]["delta"]["verdict"] == "mixed"),
        "mean_root_delta_m": float(np.mean([r["metrics"]["delta"]["root_err_delta_m"] for r in all_rows])),
        "mean_mpjpe_delta_mm": float(np.mean([r["metrics"]["delta"]["mpjpe_delta_mm"] for r in all_rows])),
    }


def _html_doc(data: dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>replay proto case compare</title>
  <style>
    :root {{
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
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      background:
        linear-gradient(90deg, rgba(20,23,19,.035) 1px, transparent 1px) 0 0 / 24px 24px,
        linear-gradient(0deg, rgba(20,23,19,.026) 1px, transparent 1px) 0 0 / 24px 24px,
        var(--paper);
      font-family: Avenir Next, Segoe UI, sans-serif;
      letter-spacing: 0;
    }}
    header {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(380px, .72fr);
      gap: 24px;
      padding: 24px clamp(16px, 3vw, 42px) 14px;
      border-bottom: 1px solid var(--line);
    }}
    h1 {{
      margin: 0;
      font: 700 clamp(30px, 4.8vw, 58px)/.96 Georgia, Cambria, serif;
    }}
    .sub {{
      margin: 10px 0 0;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.45;
      max-width: 900px;
    }}
    .nav {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-bottom: 14px;
    }}
    .nav a, button {{
      border: 1px solid var(--line);
      background: #fbfcfa;
      color: var(--ink);
      min-height: 34px;
      padding: 8px 10px;
      font: 700 12px ui-monospace, SFMono-Regular, Menlo, monospace;
      text-decoration: none;
      cursor: pointer;
    }}
    .nav a.active, button.active {{
      background: var(--ink);
      color: #fff;
      border-color: var(--ink);
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      align-content: end;
    }}
    .stat, .case-list, .tile, .meta, .plot-panel {{
      background: rgba(255,255,255,.86);
      border: 1px solid var(--line);
      box-shadow: var(--shadow);
    }}
    .stat {{
      min-height: 82px;
      padding: 13px;
    }}
    .stat span, .tile span, .meta span {{
      display: block;
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: .08em;
    }}
    .stat strong {{
      display: block;
      margin-top: 7px;
      font: 700 23px Georgia, Cambria, serif;
    }}
    main {{
      padding: 18px clamp(16px, 3vw, 42px) 42px;
    }}
    .layout {{
      display: grid;
      grid-template-columns: 330px minmax(0, 1fr);
      gap: 14px;
      align-items: start;
    }}
    .case-list {{
      max-height: calc(100vh - 170px);
      overflow: auto;
      position: sticky;
      top: 12px;
    }}
    .case-row {{
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
    }}
    .case-row.active {{ background: rgba(8,124,117,.12); }}
    .case-row small {{
      display: block;
      margin-top: 4px;
      color: var(--muted);
      font: 11px ui-monospace, SFMono-Regular, Menlo, monospace;
      line-height: 1.3;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}
    .badge {{
      display: inline-block;
      padding: 2px 6px;
      margin-left: 6px;
      border: 1px solid var(--line);
      font: 700 10px ui-monospace, SFMono-Regular, Menlo, monospace;
      text-transform: uppercase;
    }}
    .badge.improved, .good {{ color: var(--teal); }}
    .badge.worse, .bad {{ color: var(--rust); }}
    .badge.mixed, .neutral {{ color: var(--gold); }}
    .strip {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 12px;
    }}
    .tile {{
      padding: 12px;
      min-height: 92px;
      overflow: hidden;
    }}
    .tile strong {{
      display: block;
      margin-top: 7px;
      font: 700 20px ui-monospace, SFMono-Regular, Menlo, monospace;
      white-space: nowrap;
    }}
    .toolbar {{
      display: flex;
      gap: 8px;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 10px;
      padding: 10px 12px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,.78);
    }}
    input[type="range"] {{
      width: min(540px, 48vw);
      accent-color: var(--teal);
    }}
    .panels {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
    }}
    .view-panel {{
      min-width: 0;
      background: var(--night);
      border: 1px solid rgba(255,255,255,.12);
      position: relative;
      height: min(52vh, 560px);
      min-height: 390px;
      overflow: hidden;
    }}
    .view-panel canvas {{
      width: 100%;
      height: 100%;
      display: block;
    }}
    .panel-label {{
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
    }}
    .panel-label strong {{
      display: block;
      font: 700 13px ui-monospace, SFMono-Regular, Menlo, monospace;
    }}
    .panel-label small {{
      display: block;
      margin-top: 3px;
      color: rgba(244,246,240,.72);
      font: 11px ui-monospace, SFMono-Regular, Menlo, monospace;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    .plots {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin-top: 12px;
    }}
    .plot-panel {{
      padding: 10px;
      min-height: 238px;
    }}
    .plot-panel strong {{
      display: block;
      font: 700 12px ui-monospace, SFMono-Regular, Menlo, monospace;
      text-transform: uppercase;
      margin-bottom: 8px;
    }}
    .plot-panel canvas {{
      width: 100%;
      height: 190px;
      display: block;
    }}
    .meta {{
      margin-top: 12px;
      padding: 12px;
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 10px;
    }}
    .meta strong {{
      display: block;
      margin-top: 5px;
      font: 700 13px ui-monospace, SFMono-Regular, Menlo, monospace;
      overflow-wrap: anywhere;
    }}
    @media (max-width: 1180px) {{
      header, .layout {{ grid-template-columns: 1fr; }}
      .case-list {{ position: relative; max-height: 260px; top: 0; }}
      .panels {{ grid-template-columns: 1fr; }}
      .view-panel {{ height: 430px; }}
    }}
    @media (max-width: 760px) {{
      .stats, .strip, .plots, .meta {{ grid-template-columns: 1fr; }}
      .toolbar {{ flex-direction: column; align-items: stretch; }}
      input[type="range"] {{ width: 100%; }}
    }}
  </style>
  <script type="importmap">
    {{"imports": {{"three": "./assets/three/three.module.js"}}}}
  </script>
</head>
<body>
  <header>
    <div>
      <nav class="nav">
        <a href="../">Four-way G1 mesh</a>
        <a href="../replay_proto_eval/">Replay eval metrics</a>
        <a class="active" href="./">Replay cases</a>
      </nav>
      <h1>replay_proto case compare</h1>
      <p class="sub">Concrete LAFAN1-G1 cases. Left is target reference, middle is original ProtoMotions tracker, right is replay-optimized tracker. All panels use G1 mesh and 30fps playback.</p>
    </div>
    <section class="stats">
      <div class="stat"><span>Cases</span><strong>{data["summary"].get("cases", 0)}</strong></div>
      <div class="stat"><span>Improved</span><strong class="good">{data["summary"].get("improved", 0)}</strong></div>
      <div class="stat"><span>Worse</span><strong class="bad">{data["summary"].get("worse", 0)}</strong></div>
    </section>
  </header>
  <main>
    <div class="layout">
      <aside class="case-list" id="caseList"></aside>
      <section>
        <div class="strip" id="tiles"></div>
        <div class="toolbar">
          <div>
            <button id="playBtn" class="active">Pause</button>
            <button data-view="side" class="viewBtn active">Side</button>
            <button data-view="front" class="viewBtn">Front</button>
            <button data-view="top" class="viewBtn">Top</button>
          </div>
          <input id="frameSlider" type="range" min="0" max="1000" value="0">
        </div>
        <div class="panels">
          <div class="view-panel"><canvas id="refCanvas"></canvas><div class="panel-label"><strong>Reference</strong><small id="refLabel"></small></div></div>
          <div class="view-panel"><canvas id="beforeCanvas"></canvas><div class="panel-label"><strong>Before tracker</strong><small id="beforeLabel"></small></div></div>
          <div class="view-panel"><canvas id="afterCanvas"></canvas><div class="panel-label"><strong>After tracker</strong><small id="afterLabel"></small></div></div>
        </div>
        <div class="plots">
          <div class="plot-panel"><strong>Root XY path</strong><canvas id="xyPlot"></canvas></div>
          <div class="plot-panel"><strong>Root error over time</strong><canvas id="errPlot"></canvas></div>
        </div>
        <div class="meta" id="meta"></div>
      </section>
    </div>
  </main>
  <script id="payload" type="application/json">{payload}</script>
  <script type="module">
    import * as THREE from 'three';
    import {{ OrbitControls }} from './assets/three/jsm/controls/OrbitControls.js';
    import {{ STLLoader }} from './assets/three/jsm/loaders/STLLoader.js';

    const DATA = JSON.parse(document.getElementById('payload').textContent);
    const panels = [
      {{ key: 'reference', canvas: document.getElementById('refCanvas'), label: document.getElementById('refLabel') }},
      {{ key: 'before', canvas: document.getElementById('beforeCanvas'), label: document.getElementById('beforeLabel') }},
      {{ key: 'after', canvas: document.getElementById('afterCanvas'), label: document.getElementById('afterLabel') }},
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
    let token = 0;
    let bounds = null;

    function fmt(v, d = 3) {{
      if (v === null || v === undefined || Number.isNaN(Number(v))) return 'n/a';
      return Number(v).toFixed(d);
    }}
    function signed(v, d = 3) {{
      if (v === null || v === undefined || Number.isNaN(Number(v))) return 'n/a';
      const n = Number(v);
      return `${{n >= 0 ? '+' : ''}}${{n.toFixed(d)}}`;
    }}
    function cls(v) {{ return v < -1e-9 ? 'good' : v > 1e-9 ? 'bad' : 'neutral'; }}
    function meshBase(file) {{ return String(file || '').split('/').pop(); }}
    function loadGeometry(file) {{
      const clean = meshBase(file);
      if (!geometryCache.has(clean)) {{
        geometryCache.set(clean, new Promise((resolve, reject) => {{
          loader.load(`assets/g1_mesh/${{encodeURIComponent(clean)}}`, (geometry) => {{
            geometry.computeVertexNormals();
            resolve(geometry);
          }}, undefined, reject);
        }}));
      }}
      return geometryCache.get(clean);
    }}
    function loadJson(path) {{
      if (!jsonCache.has(path)) {{
        jsonCache.set(path, fetch(path).then((r) => {{
          if (!r.ok) throw new Error(`${{r.status}} ${{path}}`);
          return r.json();
        }}));
      }}
      return jsonCache.get(path);
    }}
    function setQuat(obj, q) {{
      obj.quaternion.set(q[1], q[2], q[3], q[0]);
      obj.quaternion.normalize();
    }}
    function mat(bodyName, panelKey) {{
      const side = bodyName.includes('left') ? 'left' : bodyName.includes('right') ? 'right' : bodyName.includes('torso') || bodyName.includes('waist') ? 'core' : 'base';
      const key = `${{panelKey}}:${{side}}`;
      if (materialCache.has(key)) return materialCache.get(key);
      const palettes = {{
        reference: {{ left: 0xb5bdc1, right: 0xb5bdc1, core: 0xd7dcd7, base: 0xc8cec8 }},
        before: {{ left: 0x5f83ad, right: 0xb37e4b, core: 0xbfc8c2, base: 0xaeb8b0 }},
        after: {{ left: 0x4fc8bc, right: 0xe0a63d, core: 0xe6eadf, base: 0xcbd4cc }},
      }};
      const m = new THREE.MeshStandardMaterial({{ color: palettes[panelKey][side], roughness: 0.62, metalness: 0.16 }});
      materialCache.set(key, m);
      return m;
    }}
    function makePanel(panel) {{
      const renderer = new THREE.WebGLRenderer({{ canvas: panel.canvas, antialias: true, preserveDrawingBuffer: true }});
      renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
      renderer.setClearColor(0x0b100e, 1);
      const scene = new THREE.Scene();
      scene.fog = new THREE.Fog(0x0b100e, 5.4, 11);
      const camera = new THREE.PerspectiveCamera(42, 1, 0.01, 80);
      camera.up.set(0, 0, 1);
      const controls = new OrbitControls(camera, panel.canvas);
      controls.enableDamping = true;
      controls.enablePan = false;
      const root = new THREE.Group();
      scene.add(root);
      const grid = new THREE.GridHelper(4.8, 24, 0x4d5c55, 0x25302c);
      grid.rotation.x = Math.PI / 2;
      scene.add(grid);
      scene.add(new THREE.HemisphereLight(0xecfff8, 0x1d2a26, 2.8));
      const key = new THREE.DirectionalLight(0xffffff, 2.1);
      key.position.set(2.7, -4, 4);
      scene.add(key);
      const rim = new THREE.DirectionalLight(0xd2fff2, 1.0);
      rim.position.set(-3, 2.2, 2.4);
      scene.add(rim);
      Object.assign(panel, {{ renderer, scene, camera, controls, root, bodyGroups: [], motion: null }});
    }}
    panels.forEach(makePanel);

    function clear(panel) {{
      while (panel.root.children.length) panel.root.remove(panel.root.children[0]);
      panel.bodyGroups = [];
    }}
    async function buildRobot(panel, motion, t) {{
      clear(panel);
      motion.bodies.forEach(() => {{
        const group = new THREE.Group();
        panel.root.add(group);
        panel.bodyGroups.push(group);
      }});
      for (let i = 0; i < motion.bodies.length; i += 1) {{
        const body = motion.bodies[i];
        const group = panel.bodyGroups[i];
        for (const meshDef of body.meshes || []) {{
          try {{
            const geometry = await loadGeometry(meshDef.file);
            if (t !== token) return;
            const mesh = new THREE.Mesh(geometry, mat(body.name, panel.key));
            mesh.position.set(meshDef.pos[0], meshDef.pos[1], meshDef.pos[2]);
            setQuat(mesh, meshDef.quat);
            group.add(mesh);
          }} catch (err) {{
            console.warn('mesh load failed', meshDef.file, err);
          }}
        }}
      }}
      applyFrame(panel, timeSec);
    }}
    function computeBounds(motions) {{
      const xs = [], ys = [], zs = [];
      motions.forEach((motion) => motion.frames.forEach((frame) => {{
        (frame.body_pos || []).forEach((p) => {{ xs.push(p[0]); ys.push(p[1]); zs.push(p[2]); }});
      }}));
      const pad = 0.35;
      return {{
        x0: Math.min(...xs) - pad, x1: Math.max(...xs) + pad,
        y0: Math.min(...ys) - pad, y1: Math.max(...ys) + pad,
        z0: Math.min(...zs) - pad, z1: Math.max(...zs) + pad,
      }};
    }}
    function motionDuration(motion) {{
      if (!motion || !motion.frames || !motion.frames.length) return 0;
      return motion.frames.length / PLAYBACK_FPS;
    }}
    function syncSlider() {{
      const value = caseDurationSec > 0 ? Math.round((timeSec / caseDurationSec) * 1000) : 0;
      document.getElementById('frameSlider').value = Math.max(0, Math.min(1000, value));
    }}
    function frameAt(motion, tSec) {{
      const n = motion.frames.length;
      const duration = motionDuration(motion);
      const clamped = Math.max(0, Math.min(tSec, Math.max(0, duration - (1 / PLAYBACK_FPS))));
      const idx = Math.max(0, Math.min(n - 1, Math.floor(clamped * PLAYBACK_FPS + 1e-6)));
      return {{ frame: motion.frames[idx], idx, duration }};
    }}
    function applyFrame(panel, tSec) {{
      if (!panel.motion) return;
      const current = frameAt(panel.motion, tSec);
      const frame = current.frame;
      panel.bodyGroups.forEach((group, i) => {{
        const pos = frame.body_pos[i];
        const quat = frame.body_quat[i];
        group.position.set(pos[0], pos[1], pos[2]);
        setQuat(group, quat);
      }});
      const ended = tSec >= current.duration && current.duration > 0 ? ' | ended' : '';
      panel.label.textContent = `${{(current.idx / PLAYBACK_FPS).toFixed(2)}}s/${{current.duration.toFixed(2)}}s | frame ${{current.idx + 1}}/${{panel.motion.frames.length}} | 30fps playback${{ended}}`;
      setCamera(panel, frame);
    }}
    function setCamera(panel, frame = null) {{
      const fallback = bounds || {{ x0: -1, x1: 1, y0: -1, y1: 1, z0: 0, z1: 2 }};
      const root = frame && frame.body_pos && frame.body_pos[0] ? frame.body_pos[0] : [
        (fallback.x0 + fallback.x1) / 2,
        (fallback.y0 + fallback.y1) / 2,
        Math.max(0.8, (fallback.z0 + fallback.z1) / 2),
      ];
      const cx = root[0];
      const cy = root[1];
      const cz = Math.max(0.85, root[2] + 0.15);
      const span = 2.35;
      const camera = panel.camera;
      if (viewMode === 'front') camera.position.set(cx, cy - span * 1.75, cz + span * 0.3);
      else if (viewMode === 'top') camera.position.set(cx, cy, cz + span * 2.1);
      else camera.position.set(cx + span * 1.2, cy - span * 1.35, cz + span * 0.5);
      panel.controls.target.set(cx, cy, cz);
      panel.controls.update();
    }}
    async function loadCase(index) {{
      activeCase = index;
      timeSec = 0;
      lastTickMs = null;
      syncSlider();
      token += 1;
      const t = token;
      renderList();
      renderInfo();
      const row = DATA.rows[activeCase];
      const motions = await Promise.all(panels.map((panel) => loadJson(row.paths[panel.key])));
      if (t !== token) return;
      bounds = computeBounds(motions);
      caseDurationSec = Math.max(0.1, ...motions.map(motionDuration));
      syncSlider();
      panels.forEach((panel, i) => {{
        panel.motion = motions[i];
        setCamera(panel);
        buildRobot(panel, motions[i], t);
      }});
      renderPlots(row);
    }}
    function renderList() {{
      const el = document.getElementById('caseList');
      el.innerHTML = '';
      DATA.rows.forEach((row, index) => {{
        const d = row.metrics.delta;
        const btn = document.createElement('button');
        btn.className = `case-row ${{index === activeCase ? 'active' : ''}}`;
        btn.innerHTML = `${{row.stem}} <span class="badge ${{d.verdict}}">${{d.verdict}}</span><small>root ${{signed(d.root_err_delta_m)}} m / mpjpe ${{signed(d.mpjpe_delta_mm, 1)}} mm</small>`;
        btn.onclick = () => loadCase(index);
        el.appendChild(btn);
      }});
    }}
    function renderInfo() {{
      const row = DATA.rows[activeCase];
      const b = row.metrics.before;
      const a = row.metrics.after;
      const d = row.metrics.delta;
      document.getElementById('tiles').innerHTML = `
        <div class="tile"><span>Verdict</span><strong class="${{d.verdict === 'improved' ? 'good' : d.verdict === 'worse' ? 'bad' : 'neutral'}}">${{d.verdict}}</strong></div>
        <div class="tile"><span>Before root err</span><strong>${{fmt(b.root_err_m)}} m</strong></div>
        <div class="tile"><span>After root err</span><strong>${{fmt(a.root_err_m)}} m</strong></div>
        <div class="tile"><span>Delta</span><strong class="${{cls(d.root_err_delta_m)}}">${{signed(d.root_err_delta_m)}} m</strong></div>
      `;
      document.getElementById('meta').innerHTML = `
        <div><span>Case</span><strong>${{row.stem}}</strong></div>
        <div><span>Shard / motion</span><strong>${{row.shard}} / ${{row.motion_id}}</strong></div>
        <div><span>Before MPJPE</span><strong>${{fmt(b.mpjpe_mm, 1)}} mm</strong></div>
        <div><span>After MPJPE</span><strong>${{fmt(a.mpjpe_mm, 1)}} mm</strong></div>
        <div><span>MPJPE delta</span><strong class="${{cls(d.mpjpe_delta_mm)}}">${{signed(d.mpjpe_delta_mm, 1)}} mm</strong></div>
      `;
    }}
    function drawLinePlot(canvas, seriesList, yLabelMode = 'xy') {{
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.max(320, Math.floor(rect.width * dpr));
      canvas.height = Math.max(180, Math.floor(rect.height * dpr));
      const ctx = canvas.getContext('2d');
      ctx.scale(dpr, dpr);
      const w = canvas.width / dpr, h = canvas.height / dpr;
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = '#fbfcfa';
      ctx.fillRect(0, 0, w, h);
      const pad = 22;
      const points = seriesList.flatMap((s) => s.points);
      if (!points.length) return;
      const xs = yLabelMode === 'xy' ? points.map((p) => p[0]) : points.map((_, i) => i);
      const ys = yLabelMode === 'xy' ? points.map((p) => p[1]) : points;
      const minX = Math.min(...xs), maxX = Math.max(...xs);
      const minY = Math.min(...ys), maxY = Math.max(...ys);
      const sx = (x) => pad + ((x - minX) / Math.max(maxX - minX, 1e-6)) * (w - pad * 2);
      const sy = (y) => h - pad - ((y - minY) / Math.max(maxY - minY, 1e-6)) * (h - pad * 2);
      ctx.strokeStyle = '#d5ddd4';
      ctx.lineWidth = 1;
      for (let i = 0; i < 4; i += 1) {{
        const y = pad + i * (h - pad * 2) / 3;
        ctx.beginPath(); ctx.moveTo(pad, y); ctx.lineTo(w - pad, y); ctx.stroke();
      }}
      seriesList.forEach((s) => {{
        ctx.strokeStyle = s.color;
        ctx.lineWidth = s.width || 2;
        ctx.beginPath();
        s.points.forEach((p, i) => {{
          const x = yLabelMode === 'xy' ? sx(p[0]) : sx(i);
          const y = yLabelMode === 'xy' ? sy(p[1]) : sy(p);
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }});
        ctx.stroke();
      }});
    }}
    function renderPlots(row) {{
      const s = row.metrics.series;
      drawLinePlot(document.getElementById('xyPlot'), [
        {{ points: s.ref_xy, color: '#141713', width: 2.8 }},
        {{ points: s.before_xy, color: '#426f91', width: 2 }},
        {{ points: s.after_xy, color: '#087c75', width: 2 }},
      ], 'xy');
      drawLinePlot(document.getElementById('errPlot'), [
        {{ points: s.root_err_before, color: '#426f91', width: 2 }},
        {{ points: s.root_err_after, color: '#087c75', width: 2 }},
      ], 'line');
    }}
    function resize() {{
      panels.forEach((panel) => {{
        const rect = panel.canvas.getBoundingClientRect();
        panel.renderer.setSize(rect.width, rect.height, false);
        panel.camera.aspect = rect.width / Math.max(rect.height, 1);
        panel.camera.updateProjectionMatrix();
      }});
      const row = DATA.rows[activeCase];
      if (row) renderPlots(row);
    }}
    function animate(now) {{
      requestAnimationFrame(animate);
      if (lastTickMs === null) lastTickMs = now || performance.now();
      const deltaSec = Math.min(0.1, Math.max(0, ((now || performance.now()) - lastTickMs) / 1000));
      lastTickMs = now || performance.now();
      if (playing) {{
        timeSec += deltaSec;
        if (timeSec >= caseDurationSec) timeSec %= caseDurationSec;
        syncSlider();
      }}
      panels.forEach((panel) => {{
        applyFrame(panel, timeSec);
        panel.controls.update();
        panel.renderer.render(panel.scene, panel.camera);
      }});
    }}
    document.getElementById('playBtn').onclick = (e) => {{
      playing = !playing;
      e.currentTarget.textContent = playing ? 'Pause' : 'Play';
      e.currentTarget.classList.toggle('active', playing);
    }};
    document.getElementById('frameSlider').oninput = (e) => {{
      playing = false;
      document.getElementById('playBtn').textContent = 'Play';
      document.getElementById('playBtn').classList.remove('active');
      timeSec = (Number(e.currentTarget.value) / 1000) * caseDurationSec;
      panels.forEach((panel) => applyFrame(panel, timeSec));
    }};
    document.querySelectorAll('.viewBtn').forEach((btn) => {{
      btn.onclick = () => {{
        viewMode = btn.dataset.view;
        document.querySelectorAll('.viewBtn').forEach((b) => b.classList.toggle('active', b === btn));
        panels.forEach(setCamera);
      }};
    }});
    window.addEventListener('resize', resize);
    renderList();
    loadCase(0).then(() => {{ resize(); animate(); }});
  </script>
</body>
</html>
"""


def build(eval_root: Path, out_dir: Path, max_cases: int) -> dict[str, Any]:
    _copy_assets(out_dir)
    rows = _case_rows(eval_root, out_dir, max_cases=max_cases)
    data = {
        "run": {
            "eval_root": str(eval_root),
            "before": BEFORE,
            "after": AFTER,
        },
        "summary": _summary(rows),
        "rows": rows,
    }
    (out_dir / "manifest.json").write_text(json.dumps(data, indent=2) + "\n")
    (out_dir / "index.html").write_text(_html_doc(data), encoding="utf-8")
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-cases", type=int, default=12)
    args = parser.parse_args()
    data = build(args.eval_root, args.out_dir, args.max_cases)
    print(args.out_dir / "index.html")
    print(json.dumps(data["summary"], indent=2))


if __name__ == "__main__":
    main()
