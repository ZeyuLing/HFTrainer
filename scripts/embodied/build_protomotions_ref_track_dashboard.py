#!/usr/bin/env python3
"""Build G1 mesh dashboards for ProtoMotions reference-vs-tracked cases.

The input is a ProtoMotions evaluation run with:
  - reference MotionLib shards under motion_shards/
  - saved predicted MotionLib shards under eval_<method>/predicted_shard_*/
  - per-case metrics from aggregate_proto_predicted_motion_metrics.py

The output is a static website that shows the reference motion and the actual
tracker rollout for the same case, both rendered with the G1 mesh.
"""

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

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.embodied.physflow_triplet_manifest import (  # noqa: E402
    DEFAULT_BODIES,
    MESHES_BY_BODY,
    _parse_g1_body_meshes,
)


THREE_SRC = ROOT / "motion_annot_web/score_m2m/static/three"
G1_MESH_SRC = (
    ROOT
    / "hftrainer/models/motion/physflow/trackers/protomotions/vendor/protomotions/data/assets/mesh/G1"
)
LOCAL_SUCCESS_THRESH_M = 0.2
ROOT_HEIGHT_SUCCESS_THRESH_M = 0.2
ROOT_TRAJ_SUCCESS_THRESH_M = 0.5


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
    frames = int(lens[motion_id])
    pos = _as_np(lib["gts"][start : start + frames]).astype(np.float32).copy()
    quat_xyzw = _as_np(lib["grs"][start : start + frames]).astype(np.float32).copy()
    dt = float(_as_np(lib["motion_dt"]).reshape(-1)[motion_id])
    motion_file = str(list(lib.get("motion_files", []))[motion_id]) if "motion_files" in lib else ""
    return pos, quat_xyzw, dt, motion_file


def _latest_predicted(root: Path) -> Path | None:
    candidates = sorted((root / "results").glob("predicted_motion_lib_epoch_*.pt"))
    return candidates[-1] if candidates else None


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
    return np.einsum("tbc,tcd->tbd", pos - root, root_rot)


def _align_xy_to_reference(pred_pos: np.ndarray, ref_pos: np.ndarray) -> np.ndarray:
    out = pred_pos.copy()
    out[..., :2] -= out[0, 0, :2] - ref_pos[0, 0, :2]
    return out


def _metrics(
    ref_pos: np.ndarray,
    ref_quat: np.ndarray,
    pred_pos: np.ndarray,
    pred_quat: np.ndarray,
    dt: float,
) -> dict[str, float]:
    frames = min(len(ref_pos), len(pred_pos), len(ref_quat), len(pred_quat))
    ref_pos = ref_pos[:frames]
    pred_pos = pred_pos[:frames]
    ref_quat = ref_quat[:frames]
    pred_quat = pred_quat[:frames]
    ref_local = _local_pos(ref_pos, ref_quat)
    pred_local = _local_pos(pred_pos, pred_quat)
    aligned_body_err = np.linalg.norm(pred_pos - ref_pos, axis=-1)
    local_err = np.linalg.norm(pred_local - ref_local, axis=-1)
    root_err = np.linalg.norm(pred_pos[:, 0, :] - ref_pos[:, 0, :], axis=-1)
    root_height_err = np.abs(pred_pos[:, 0, 2] - ref_pos[:, 0, 2])
    local_step_vel = np.linalg.norm(np.diff(pred_local, axis=0) - np.diff(ref_local, axis=0), axis=-1)
    if frames > 2:
        local_step_acc = np.linalg.norm(
            np.diff(pred_local, n=2, axis=0) - np.diff(ref_local, n=2, axis=0),
            axis=-1,
        )
    else:
        local_step_acc = np.asarray([float("nan")], dtype=np.float32)
    safe_dt = max(float(dt), 1e-9)
    local_mpjpe_m = float(local_err.mean())
    root_height_m = float(root_height_err.mean())
    root_err_m = float(root_err.mean())
    success = (
        local_mpjpe_m <= LOCAL_SUCCESS_THRESH_M
        and root_height_m <= ROOT_HEIGHT_SUCCESS_THRESH_M
        and root_err_m <= ROOT_TRAJ_SUCCESS_THRESH_M
    )
    return {
        "frames": float(frames),
        "success": float(success),
        "aligned_global_mpjpe_mm": float(aligned_body_err.mean() * 1000.0),
        "local_mpjpe_mm": float(local_mpjpe_m * 1000.0),
        "root_err_m": root_err_m,
        "root_err_max_m": float(root_err.max()),
        "root_height_err_m": root_height_m,
        "success_root_traj_thresh_m": ROOT_TRAJ_SUCCESS_THRESH_M,
        "local_mpjve_mps": float(np.nanmean(local_step_vel) / safe_dt),
        "local_mpjae_mps2": float(np.nanmean(local_step_acc) / (safe_dt * safe_dt)),
        "ref_disp_m": float(np.linalg.norm(ref_pos[-1, 0, :2] - ref_pos[0, 0, :2])) if frames > 1 else 0.0,
        "track_disp_m": float(np.linalg.norm(pred_pos[-1, 0, :2] - pred_pos[0, 0, :2])) if frames > 1 else 0.0,
    }


def _series(ref_pos: np.ndarray, pred_pos: np.ndarray, max_points: int = 260) -> dict[str, Any]:
    frames = min(len(ref_pos), len(pred_pos))
    if frames < 1:
        return {"ref_xy": [], "track_xy": [], "root_err": []}
    idx = np.linspace(0, frames - 1, min(max_points, frames)).round().astype(np.int64)
    ref = ref_pos[:frames, 0, :]
    pred = pred_pos[:frames, 0, :]
    return {
        "ref_xy": ref[idx, :2].round(4).tolist(),
        "track_xy": pred[idx, :2].round(4).tolist(),
        "root_err": np.linalg.norm(pred[idx] - ref[idx], axis=-1).round(4).tolist(),
    }


def _write_robot_frames(path: Path, pos: np.ndarray, quat_xyzw: np.ndarray, bodies: list[dict[str, Any]]) -> None:
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
                "fps": 30,
                "source_fps_note": "normalized_to_30fps_for_visual_inspection",
                "num_frames": len(frames),
                "num_bodies": len(bodies),
                "bodies": bodies,
                "frames": frames,
            },
            separators=(",", ":"),
        )
    )


def _copy_metric_rows(metric_json: Path, method: str) -> list[dict[str, Any]]:
    data = json.loads(metric_json.read_text())
    missing = data.get("missing_predicted_motion_libs") or {}
    if missing:
        raise RuntimeError(f"{metric_json} has missing predicted shards: {missing}")
    rows = data["results"][method]["motions"]
    mapped = []
    shard = 0
    prev_mid = -1
    for global_idx, row in enumerate(rows):
        mid = int(row["motion_id"])
        if global_idx > 0 and mid <= prev_mid:
            shard += 1
        prev_mid = mid
        copied = dict(row)
        copied["global_index"] = global_idx
        copied["shard"] = shard
        copied["local_motion_id"] = mid
        mapped.append(copied)
    return mapped


def _select_rows(rows: list[dict[str, Any]], max_cases: int, include_all: bool) -> list[dict[str, Any]]:
    if include_all or max_cases <= 0 or len(rows) <= max_cases:
        return rows
    selected: dict[int, dict[str, Any]] = {}

    def add(items: list[dict[str, Any]], n: int) -> None:
        for row in items[:n]:
            selected.setdefault(int(row["global_index"]), row)

    failures = [r for r in rows if float(r.get("success", 0.0)) < 0.5]
    successes = [r for r in rows if float(r.get("success", 0.0)) >= 0.5]
    add(sorted(failures, key=lambda r: -float(r["local_mpjpe_mm"])), max_cases // 3)
    add(sorted(rows, key=lambda r: -float(r["local_mpjpe_mm"])), max_cases // 3)
    add(sorted(rows, key=lambda r: -float(r["local_mpjae_mps2"])), max_cases // 6)
    add(sorted(successes, key=lambda r: float(r["local_mpjpe_mm"])), max_cases // 6)

    if len(selected) < max_cases:
        sorted_rows = sorted(rows, key=lambda r: float(r["local_mpjpe_mm"]))
        for q in np.linspace(0, len(sorted_rows) - 1, max_cases):
            selected.setdefault(int(sorted_rows[int(round(q))]["global_index"]), sorted_rows[int(round(q))])
            if len(selected) >= max_cases:
                break

    out = list(selected.values())[:max_cases]
    out.sort(key=lambda r: (-float(r.get("local_mpjpe_mm", 0.0)), int(r["global_index"])))
    return out


def _rel(path: Path, out_dir: Path) -> str:
    return str(path.relative_to(out_dir))


def _build_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    bodies = _bodies_meta()
    metric_rows = _copy_metric_rows(args.metric_json, args.method)
    selected = _select_rows(metric_rows, args.max_cases, args.all_cases)
    by_shard: dict[int, list[dict[str, Any]]] = {}
    for row in selected:
        by_shard.setdefault(int(row["shard"]), []).append(row)

    rows: list[dict[str, Any]] = []
    eval_dir = args.eval_root / f"eval_{args.method}"
    for shard, shard_rows in sorted(by_shard.items()):
        ref_path = args.motion_base / args.shard_file_template.format(shard=shard)
        pred_path = _latest_predicted(eval_dir / f"predicted_shard_{shard}")
        if pred_path is None:
            raise FileNotFoundError(f"Missing predicted MotionLib for shard {shard}: {eval_dir}")
        ref_lib = _load(ref_path)
        pred_lib = _load(pred_path)
        for row in shard_rows:
            motion_id = int(row["local_motion_id"])
            ref_pos, ref_quat, ref_dt, motion_file = _slice_lib(ref_lib, motion_id)
            pred_pos_raw, pred_quat, pred_dt, _ = _slice_lib(pred_lib, motion_id)
            frames = min(int(row["frames"]), len(ref_pos), len(pred_pos_raw))
            ref_pos = ref_pos[:frames]
            ref_quat = ref_quat[:frames]
            pred_quat = pred_quat[:frames]
            pred_pos = _align_xy_to_reference(pred_pos_raw[:frames], ref_pos)
            metrics = _metrics(ref_pos, ref_quat, pred_pos, pred_quat, pred_dt or ref_dt)
            stem = Path(motion_file).stem if motion_file else f"motion_{row['global_index']:06d}"
            case_id = f"{args.dataset_key}_g{int(row['global_index']):05d}_s{shard:02d}_m{motion_id:04d}_{stem}"
            case_dir = args.out_dir / "data" / case_id
            ref_json = case_dir / "reference.json"
            track_json = case_dir / "tracked.json"
            _write_robot_frames(ref_json, ref_pos, ref_quat, bodies)
            _write_robot_frames(track_json, pred_pos, pred_quat, bodies)
            rows.append(
                {
                    "id": case_id,
                    "dataset": args.dataset_name,
                    "stem": stem,
                    "global_index": int(row["global_index"]),
                    "shard": shard,
                    "motion_id": motion_id,
                    "source_motion": motion_file,
                    "paths": {
                        "reference": _rel(ref_json, args.out_dir),
                        "tracked": _rel(track_json, args.out_dir),
                    },
                    "metrics": {
                        "visual_recomputed": metrics,
                        "aggregator_row": row,
                        "series": _series(ref_pos, pred_pos),
                    },
                }
            )
    rows.sort(key=lambda r: (-float(r["metrics"]["visual_recomputed"]["local_mpjpe_mm"]), r["global_index"]))
    return rows


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    m = [r["metrics"]["visual_recomputed"] for r in rows]
    return {
        "cases": len(rows),
        "success_rate": float(np.mean([x["success"] for x in m])),
        "local_mpjpe_mm_mean": float(np.mean([x["local_mpjpe_mm"] for x in m])),
        "aligned_global_mpjpe_mm_mean": float(np.mean([x["aligned_global_mpjpe_mm"] for x in m])),
        "local_mpjve_mps_mean": float(np.mean([x["local_mpjve_mps"] for x in m])),
        "local_mpjae_mps2_mean": float(np.mean([x["local_mpjae_mps2"] for x in m])),
    }


def _html_doc(data: dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{data["dataset"]} ProtoMotions reference vs tracked</title>
  <style>
    :root {{
      --paper: #f4f1ea;
      --ink: #191a17;
      --muted: #6c7169;
      --line: #c6c1b7;
      --panel: #fffefa;
      --night: #0c1110;
      --track: #0a7c72;
      --ref: #333833;
      --warn: #b95635;
      --ok: #127a52;
      --gold: #a07821;
      --shadow: 0 16px 38px rgba(36, 33, 25, .13);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      background:
        linear-gradient(90deg, rgba(25,26,23,.035) 1px, transparent 1px) 0 0 / 28px 28px,
        linear-gradient(0deg, rgba(25,26,23,.028) 1px, transparent 1px) 0 0 / 28px 28px,
        var(--paper);
      font-family: Optima, Avenir Next, Segoe UI, sans-serif;
      letter-spacing: 0;
    }}
    header {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(360px, .65fr);
      gap: 24px;
      padding: 26px clamp(16px, 3vw, 42px) 16px;
      border-bottom: 1px solid var(--line);
    }}
    h1 {{
      margin: 0;
      font: 700 clamp(28px, 4.4vw, 54px)/.98 Georgia, Cambria, serif;
      max-width: 980px;
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
      background: #fffefa;
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
      background: rgba(255,254,250,.9);
      border: 1px solid var(--line);
      box-shadow: var(--shadow);
    }}
    .stat {{ min-height: 82px; padding: 13px; }}
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
    main {{ padding: 18px clamp(16px, 3vw, 42px) 42px; }}
    .layout {{
      display: grid;
      grid-template-columns: 340px minmax(0, 1fr);
      gap: 14px;
      align-items: start;
    }}
    .case-list {{
      max-height: calc(100vh - 178px);
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
      font-family: Optima, Avenir Next, Segoe UI, sans-serif;
      font-weight: 700;
    }}
    .case-row.active {{ background: rgba(10,124,114,.13); }}
    .case-row small {{
      display: block;
      margin-top: 4px;
      color: var(--muted);
      font: 11px ui-monospace, SFMono-Regular, Menlo, monospace;
      line-height: 1.35;
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
    .badge.good {{ color: var(--ok); }}
    .badge.bad {{ color: var(--warn); }}
    .strip {{
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 12px;
    }}
    .tile {{ padding: 12px; min-height: 88px; overflow: hidden; }}
    .tile strong {{
      display: block;
      margin-top: 7px;
      font: 700 19px ui-monospace, SFMono-Regular, Menlo, monospace;
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
      background: rgba(255,254,250,.82);
    }}
    input[type="range"] {{ width: min(540px, 48vw); accent-color: var(--track); }}
    .panels {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }}
    .view-panel {{
      min-width: 0;
      background: var(--night);
      border: 1px solid rgba(255,255,255,.12);
      position: relative;
      height: min(58vh, 610px);
      min-height: 430px;
      overflow: hidden;
    }}
    .view-panel canvas {{ width: 100%; height: 100%; display: block; }}
    .panel-label {{
      position: absolute;
      left: 10px;
      top: 10px;
      z-index: 2;
      color: #f8f6ef;
      background: rgba(12,17,16,.72);
      border: 1px solid rgba(255,255,255,.16);
      padding: 8px 9px;
      backdrop-filter: blur(10px);
      max-width: calc(100% - 20px);
    }}
    .panel-label strong {{ display: block; font: 700 13px ui-monospace, SFMono-Regular, Menlo, monospace; }}
    .panel-label small {{
      display: block;
      margin-top: 3px;
      color: rgba(248,246,239,.76);
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
    .plot-panel {{ padding: 10px; min-height: 232px; }}
    .plot-panel strong {{
      display: block;
      font: 700 12px ui-monospace, SFMono-Regular, Menlo, monospace;
      text-transform: uppercase;
      margin-bottom: 8px;
    }}
    .plot-panel canvas {{ width: 100%; height: 186px; display: block; }}
    .meta {{
      margin-top: 12px;
      padding: 12px;
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
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
    @media (max-width: 780px) {{
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
      <nav class="nav" id="nav"></nav>
      <h1>{data["dataset"]} reference vs tracker rollout</h1>
      <p class="sub">Each case shows the original reference MotionLib clip and the actual ProtoMotions G1 tracker rollout. Both panels use the G1 mesh and fixed 30fps playback. Metrics are recomputed for the displayed clip after stripping the IsaacGym grid offset from the tracker rollout.</p>
    </div>
    <section class="stats">
      <div class="stat"><span>Displayed cases</span><strong>{data["summary"].get("cases", 0)}</strong></div>
      <div class="stat"><span>Success rate</span><strong>{data["summary"].get("success_rate", 0):.3f}</strong></div>
      <div class="stat"><span>Mean local MPJPE</span><strong>{data["summary"].get("local_mpjpe_mm_mean", 0):.1f}mm</strong></div>
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
          <div class="view-panel"><canvas id="refCanvas"></canvas><div class="panel-label"><strong>Reference motion</strong><small id="refLabel"></small></div></div>
          <div class="view-panel"><canvas id="trackCanvas"></canvas><div class="panel-label"><strong>Tracker rollout</strong><small id="trackLabel"></small></div></div>
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
      {{ key: 'tracked', canvas: document.getElementById('trackCanvas'), label: document.getElementById('trackLabel') }},
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
    function material(bodyName, panelKey) {{
      const side = bodyName.includes('left') ? 'left' : bodyName.includes('right') ? 'right' : bodyName.includes('torso') || bodyName.includes('waist') ? 'core' : 'base';
      const key = `${{panelKey}}:${{side}}`;
      if (materialCache.has(key)) return materialCache.get(key);
      const palettes = {{
        reference: {{ left: 0xb8b9b2, right: 0xb8b9b2, core: 0xf1eee2, base: 0xd5d4ca }},
        tracked: {{ left: 0x4fbeb3, right: 0xe1a338, core: 0xf4f3e9, base: 0xcbd5cb }},
      }};
      const m = new THREE.MeshStandardMaterial({{ color: palettes[panelKey][side], roughness: 0.62, metalness: 0.16 }});
      materialCache.set(key, m);
      return m;
    }}
    function makePanel(panel) {{
      const renderer = new THREE.WebGLRenderer({{ canvas: panel.canvas, antialias: true, preserveDrawingBuffer: true }});
      renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
      renderer.setClearColor(0x0c1110, 1);
      const scene = new THREE.Scene();
      scene.fog = new THREE.Fog(0x0c1110, 5.4, 11);
      const camera = new THREE.PerspectiveCamera(42, 1, 0.01, 80);
      camera.up.set(0, 0, 1);
      const controls = new OrbitControls(camera, panel.canvas);
      controls.enableDamping = true;
      controls.enablePan = false;
      const root = new THREE.Group();
      scene.add(root);
      const grid = new THREE.GridHelper(5.0, 25, 0x536057, 0x29312d);
      grid.rotation.x = Math.PI / 2;
      scene.add(grid);
      scene.add(new THREE.HemisphereLight(0xf5fff8, 0x1f2925, 2.8));
      const key = new THREE.DirectionalLight(0xffffff, 2.2);
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
            const mesh = new THREE.Mesh(geometry, material(body.name, panel.key));
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
    function frameAt(motion, tSec) {{
      const n = motion.frames.length;
      const duration = motionDuration(motion);
      const clamped = Math.max(0, Math.min(tSec, Math.max(0, duration - (1 / PLAYBACK_FPS))));
      const idx = Math.max(0, Math.min(n - 1, Math.floor(clamped * PLAYBACK_FPS + 1e-6)));
      return {{ frame: motion.frames[idx], idx, duration }};
    }}
    function syncSlider() {{
      const value = caseDurationSec > 0 ? Math.round((timeSec / caseDurationSec) * 1000) : 0;
      document.getElementById('frameSlider').value = Math.max(0, Math.min(1000, value));
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
      panel.label.textContent = `${{(current.idx / PLAYBACK_FPS).toFixed(2)}}s/${{current.duration.toFixed(2)}}s | frame ${{current.idx + 1}}/${{panel.motion.frames.length}} | 30fps${{ended}}`;
      setCamera(panel, frame);
    }}
    function setCamera(panel, frame = null) {{
      const fallback = bounds || {{ x0: -1, x1: 1, y0: -1, y1: 1, z0: 0, z1: 2 }};
      const root = frame && frame.body_pos && frame.body_pos[0] ? frame.body_pos[0] : [
        (fallback.x0 + fallback.x1) / 2,
        (fallback.y0 + fallback.y1) / 2,
        Math.max(0.8, (fallback.z0 + fallback.z1) / 2),
      ];
      const cx = root[0], cy = root[1], cz = Math.max(0.85, root[2] + 0.15);
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
      panels.forEach((panel, i) => {{
        panel.motion = motions[i];
        setCamera(panel);
        buildRobot(panel, motions[i], t);
      }});
      renderPlots(row);
      resize();
    }}
    function renderNav() {{
      const nav = document.getElementById('nav');
      nav.innerHTML = DATA.sibling_links.map((link) => `<a class="${{link.active ? 'active' : ''}}" href="${{link.href}}">${{link.label}}</a>`).join('');
    }}
    function renderList() {{
      const el = document.getElementById('caseList');
      el.innerHTML = '';
      DATA.rows.forEach((row, index) => {{
        const m = row.metrics.visual_recomputed;
        const btn = document.createElement('button');
        btn.className = `case-row ${{index === activeCase ? 'active' : ''}}`;
        btn.innerHTML = `${{row.stem}} <span class="badge ${{m.success ? 'good' : 'bad'}}">${{m.success ? 'pass' : 'fail'}}</span><small>local ${{fmt(m.local_mpjpe_mm, 1)}}mm | aligned global ${{fmt(m.aligned_global_mpjpe_mm, 1)}}mm | s${{row.shard}} m${{row.motion_id}}</small>`;
        btn.onclick = () => loadCase(index);
        el.appendChild(btn);
      }});
    }}
    function renderInfo() {{
      const row = DATA.rows[activeCase];
      const m = row.metrics.visual_recomputed;
      const agg = row.metrics.aggregator_row;
      const wildMetrics = [];
      if (agg.reference_frames !== undefined) wildMetrics.push(['Reference frames', `${{agg.reference_frames}}`]);
      if (agg.tracker_frames !== undefined) wildMetrics.push(['Tracker frames', `${{agg.tracker_frames}}`]);
      if (agg.overlap_frames !== undefined) wildMetrics.push(['Metric overlap frames', `${{agg.overlap_frames}}`]);
      if (agg.completion !== undefined) wildMetrics.push(['Judge completion', `${{fmt(agg.completion, 3)}}`]);
      if (agg.fall !== undefined) wildMetrics.push(['Judge fall', `${{agg.fall ? 'true' : 'false'}}`]);
      if (agg.max_joint_err_rad !== undefined) wildMetrics.push(['Max joint error', `${{fmt(agg.max_joint_err_rad, 3)}} rad`]);
      if (agg.root_traj_err_m !== undefined) wildMetrics.push(['Judge root traj error', `${{fmt(agg.root_traj_err_m, 3)}} m`]);
      if (agg.score !== undefined) wildMetrics.push(['Wild score', `${{fmt(agg.score, 3)}}`]);
      const frameLabel = agg.overlap_frames !== undefined ? 'Metric overlap' : 'Frames';
      const metaRows = [
        ['Dataset case', `${{row.dataset}} / #${{row.global_index}}`],
        ['Shard motion', `${{row.shard}} / ${{row.motion_id}}`],
        [frameLabel, `${{fmt(m.frames, 0)}} @ 30fps view`],
        ['Aggregator local MPJPE', `${{fmt(agg.local_mpjpe_mm, 1)}} mm`],
        ['Root error', `${{fmt(m.root_err_m, 3)}} m`],
        ['Root height error', `${{fmt(m.root_height_err_m, 3)}} m`],
        ['Ref displacement', `${{fmt(m.ref_disp_m, 3)}} m`],
        ['Track displacement', `${{fmt(m.track_disp_m, 3)}} m`],
        ...wildMetrics,
      ];
      document.getElementById('tiles').innerHTML = `
        <div class="tile"><span>Case success</span><strong class="${{m.success ? 'good' : 'bad'}}">${{m.success ? 'pass' : 'fail'}}</strong></div>
        <div class="tile"><span>Local MPJPE</span><strong>${{fmt(m.local_mpjpe_mm, 1)}} mm</strong></div>
        <div class="tile"><span>Aligned global MPJPE</span><strong>${{fmt(m.aligned_global_mpjpe_mm, 1)}} mm</strong></div>
        <div class="tile"><span>Velocity error</span><strong>${{fmt(m.local_mpjve_mps, 3)}} m/s</strong></div>
        <div class="tile"><span>Acceleration error</span><strong>${{fmt(m.local_mpjae_mps2, 3)}} m/s^2</strong></div>
      `;
      document.getElementById('meta').innerHTML = metaRows
        .map(([label, value]) => `<div><span>${{label}}</span><strong>${{value}}</strong></div>`)
        .join('');
    }}
    function drawPlot(canvas, seriesList, mode = 'xy') {{
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.max(320, Math.floor(rect.width * dpr));
      canvas.height = Math.max(180, Math.floor(rect.height * dpr));
      const ctx = canvas.getContext('2d');
      ctx.scale(dpr, dpr);
      const w = canvas.width / dpr, h = canvas.height / dpr;
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = '#fffefa';
      ctx.fillRect(0, 0, w, h);
      const pad = 22;
      const points = seriesList.flatMap((s) => s.points);
      if (!points.length) return;
      const xs = mode === 'xy' ? points.map((p) => p[0]) : points.map((_, i) => i);
      const ys = mode === 'xy' ? points.map((p) => p[1]) : points;
      const minX = Math.min(...xs), maxX = Math.max(...xs);
      const minY = Math.min(...ys), maxY = Math.max(...ys);
      const sx = (x) => pad + ((x - minX) / Math.max(maxX - minX, 1e-6)) * (w - pad * 2);
      const sy = (y) => h - pad - ((y - minY) / Math.max(maxY - minY, 1e-6)) * (h - pad * 2);
      ctx.strokeStyle = '#d8d0c4';
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
          const x = mode === 'xy' ? sx(p[0]) : sx(i);
          const y = mode === 'xy' ? sy(p[1]) : sy(p);
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }});
        ctx.stroke();
      }});
    }}
    function renderPlots(row) {{
      const s = row.metrics.series;
      drawPlot(document.getElementById('xyPlot'), [
        {{ points: s.ref_xy, color: '#191a17', width: 2.8 }},
        {{ points: s.track_xy, color: '#0a7c72', width: 2.2 }},
      ], 'xy');
      drawPlot(document.getElementById('errPlot'), [
        {{ points: s.root_err, color: '#b95635', width: 2.2 }},
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
    renderNav();
    renderList();
    loadCase(0).then(() => {{ resize(); animate(); }});
  </script>
</body>
</html>
"""


def build(args: argparse.Namespace) -> dict[str, Any]:
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _copy_assets(args.out_dir)
    rows = _build_rows(args)
    data = {
        "dataset": args.dataset_name,
        "dataset_key": args.dataset_key,
        "method": args.method,
        "source": {
            "eval_root": str(args.eval_root),
            "motion_base": str(args.motion_base),
            "metric_json": str(args.metric_json),
            "shard_file_template": args.shard_file_template,
        },
        "summary": _summary(rows),
        "rows": rows,
        "sibling_links": [],
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(data, indent=2) + "\n")
    (args.out_dir / "index.html").write_text(_html_doc(data), encoding="utf-8")
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--dataset-key", required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--motion-base", type=Path, required=True)
    parser.add_argument("--metric-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--method", default="protomotions_g1_bones")
    parser.add_argument("--shard-file-template", required=True)
    parser.add_argument("--max-cases", type=int, default=48)
    parser.add_argument("--all-cases", action="store_true")
    args = parser.parse_args()
    data = build(args)
    print(args.out_dir / "index.html")
    print(json.dumps(data["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
