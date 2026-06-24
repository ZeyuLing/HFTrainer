#!/usr/bin/env python3
"""Build four-column robot-frame visualizations for tracker benchmark cases.

The embodied_viz /physflow_triplet page can render arbitrary columns when the
manifest provides `column_order`.  This script joins existing OpenTrack
reference/rollout robot_frames with ProtoMotions/PhysFlow predicted MotionLib
rollouts and writes a single manifest:

  Reference | Any2Track | ProtoMotions | PhysFlow

For ProtoMotions rollouts, the evaluator stores IsaacGym body positions with
per-env XY offsets.  We remove that offset by anchoring each rollout's first
pelvis XY to the corresponding OpenTrack/reference JSON first pelvis XY.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.embodied.cursor_build_overfit_triplet_manifest import (  # noqa: E402
    _bodies_meta,
    _per_motion_frames,
    tracked_to_robot_frames,
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _first_root_xy(robot_frames_path: Path) -> np.ndarray:
    data = _load_json(robot_frames_path)
    frames = data.get("frames") or []
    if not frames:
        return np.zeros(2, dtype=np.float32)
    return np.asarray(frames[0]["body_pos"][0][:2], dtype=np.float32)


def _slice_motion(d: dict[str, Any], motion_id: int) -> tuple[np.ndarray, np.ndarray, int]:
    starts, lengths = _per_motion_frames(d)
    start = starts[motion_id]
    n = lengths[motion_id]
    pos = d["gts"][start : start + n].detach().cpu().numpy().copy()
    quat = d["grs"][start : start + n].detach().cpu().numpy().copy()
    dt = float(d["motion_dt"][motion_id])
    fps = int(round(1.0 / dt)) if dt > 0 else 50
    return pos, quat, fps


def _quat_xyzw_to_mat(q: np.ndarray) -> np.ndarray:
    x, y, z, w = np.moveaxis(q, -1, 0)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    out = np.empty(q.shape[:-1] + (3, 3), dtype=np.float32)
    out[..., 0, 0] = 1.0 - 2.0 * (yy + zz)
    out[..., 0, 1] = 2.0 * (xy - wz)
    out[..., 0, 2] = 2.0 * (xz + wy)
    out[..., 1, 0] = 2.0 * (xy + wz)
    out[..., 1, 1] = 1.0 - 2.0 * (xx + zz)
    out[..., 1, 2] = 2.0 * (yz - wx)
    out[..., 2, 0] = 2.0 * (xz - wy)
    out[..., 2, 1] = 2.0 * (yz + wx)
    out[..., 2, 2] = 1.0 - 2.0 * (xx + yy)
    return out


def _local_body_pos(pos: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    root_pos = pos[:, :1, :]
    root_rot = _quat_xyzw_to_mat(quat_xyzw[:, 0])
    return np.einsum("tbi,tij->tbj", pos - root_pos, root_rot)


def _compute_pred_metrics(pred_pos: np.ndarray, pred_quat: np.ndarray, motion_file: str) -> dict[str, float]:
    ref = torch.load(motion_file, map_location="cpu", weights_only=False)
    ref_pos = ref["rigid_body_pos"].detach().cpu().numpy().astype(np.float32)
    ref_quat = ref["rigid_body_rot"].detach().cpu().numpy().astype(np.float32)
    frames = min(len(ref_pos), len(pred_pos))
    if frames <= 1:
        return {}
    ref_pos = ref_pos[:frames]
    ref_quat = ref_quat[:frames]
    pred_pos = pred_pos[:frames]
    pred_quat = pred_quat[:frames]
    ref_local = _local_body_pos(ref_pos, ref_quat)
    pred_local = _local_body_pos(pred_pos, pred_quat)
    local_mpjpe_m = float(np.linalg.norm(pred_local - ref_local, axis=-1).mean())
    local_mpjve_m = float(
        np.linalg.norm(np.diff(pred_local, axis=0) - np.diff(ref_local, axis=0), axis=-1).mean()
    )
    root_h_m = float(np.abs(pred_pos[:, 0, 2] - ref_pos[:, 0, 2]).mean())
    return {
        "success": bool(local_mpjpe_m <= 0.2 and root_h_m <= 0.2),
        "MPJPE_mm": local_mpjpe_m * 1000.0,
        "MPJVE_mm_frame": local_mpjve_m * 1000.0,
        "RootH_mm": root_h_m * 1000.0,
    }


def _load_predicted_rollouts(eval_dir: Path) -> dict[str, tuple[dict[str, Any], int]]:
    """Return stem -> (MotionLib dict, local motion id)."""
    out: dict[str, tuple[dict[str, Any], int]] = {}
    for pt_path in sorted(eval_dir.glob("predicted_shard_*/results/predicted_motion_lib_epoch_*.pt")):
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
        for local_id, motion_file in enumerate(data.get("motion_files", [])):
            stem = Path(str(motion_file)).stem
            out[stem] = (data, local_id)
    return out


def build_lafan(args: argparse.Namespace) -> Path:
    opentrack_manifest = _load_json(args.opentrack_manifest)
    opentrack_metrics = _load_json(args.opentrack_metrics)
    opentrack_by_name = {
        row["motion"]: row for row in opentrack_metrics.get("motions", [])
    }
    proto_rollouts = _load_predicted_rollouts(args.proto_eval_dir)
    phys_rollouts = _load_predicted_rollouts(args.physflow_eval_dir)

    out_dir = args.out_dir.resolve()
    bodies = _bodies_meta()
    rows = []

    for idx, row in enumerate(opentrack_manifest.get("rows", [])):
        stem = row.get("prompt_id")
        if stem not in proto_rollouts or stem not in phys_rollouts:
            continue

        reference_col = row["columns"]["reference"]
        opentrack_col = row["columns"]["opentrack"]
        ref_path = Path(reference_col["path"])
        ref_xy0 = _first_root_xy(ref_path)

        proto_data, proto_id = proto_rollouts[stem]
        proto_pos, proto_quat, proto_fps = _slice_motion(proto_data, proto_id)
        proto_metrics = _compute_pred_metrics(
            proto_pos,
            proto_quat,
            str(proto_data["motion_files"][proto_id]),
        )
        proto_pos[..., :2] -= proto_pos[0, 0, :2] - ref_xy0
        proto_json = out_dir / "robot_frames" / "protomotions" / f"{stem}.json"
        tracked_to_robot_frames(proto_pos, proto_quat, bodies, proto_fps, proto_json)

        phys_data, phys_id = phys_rollouts[stem]
        phys_pos, phys_quat, phys_fps = _slice_motion(phys_data, phys_id)
        phys_metrics = _compute_pred_metrics(
            phys_pos,
            phys_quat,
            str(phys_data["motion_files"][phys_id]),
        )
        phys_pos[..., :2] -= phys_pos[0, 0, :2] - ref_xy0
        phys_json = out_dir / "robot_frames" / "physflow" / f"{stem}.json"
        tracked_to_robot_frames(phys_pos, phys_quat, bodies, phys_fps, phys_json)

        ot_metric = opentrack_by_name.get(stem, {})
        rows.append(
            {
                "iteration": len(rows),
                "iteration_label": f"Case {len(rows):02d} · {stem}",
                "prompt_id": stem,
                "prompt": f"LAFAN1-G1 tracker benchmark motion: {stem}",
                "category": "LAFAN-G1",
                "columns": {
                    "reference": {
                        "status": "ready",
                        "title": "Reference",
                        "path": reference_col["path"],
                        "metrics": {},
                    },
                    "opentrack": {
                        "status": "ready",
                        "title": "Any2Track",
                        "path": opentrack_col["path"],
                        "metrics": {
                            "success": ot_metric.get("success"),
                            "paper_success": ot_metric.get("paper_success"),
                            "MPJPE_mm": ot_metric.get("local_mpjpe_mm"),
                            "MPJVE_mm_frame": (
                                ot_metric.get("local_mpjve_mps") * 1000.0 / 50.0
                                if ot_metric.get("local_mpjve_mps") is not None
                                else None
                            ),
                            "RootH_mm": (
                                ot_metric.get("root_height_err_mean") * 1000.0
                                if ot_metric.get("root_height_err_mean") is not None
                                else None
                            ),
                        },
                    },
                    "protomotions": {
                        "status": "ready",
                        "title": "ProtoMotions",
                        "path": str(proto_json),
                        "metrics": proto_metrics,
                    },
                    "physflow": {
                        "status": "ready",
                        "title": "PhysFlow",
                        "path": str(phys_json),
                        "metrics": phys_metrics,
                    },
                },
            }
        )

    # Put large visual gaps first, while keeping every case available.
    def sort_key(r: dict[str, Any]) -> tuple[float, float]:
        cols = r["columns"]
        ot = cols["opentrack"]["metrics"].get("MPJPE_mm") or 0.0
        proto = cols["protomotions"]["metrics"].get("MPJPE_mm") or 0.0
        phys = cols["physflow"]["metrics"].get("MPJPE_mm") or 0.0
        return (max(proto, phys) - ot, max(proto, phys))

    rows.sort(key=sort_key, reverse=True)
    for i, r in enumerate(rows):
        r["iteration"] = i
        r["iteration_label"] = f"Case {i:02d} · {r['prompt_id']}"

    manifest = {
        "schema_version": 1,
        "project": "Tracker Benchmark Motion Visualization · LAFAN-G1",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "generated_from": {
            "opentrack_manifest": str(args.opentrack_manifest),
            "proto_eval_dir": str(args.proto_eval_dir),
            "physflow_eval_dir": str(args.physflow_eval_dir),
        },
        "group_label": "case",
        "column_order": [
            {"key": "reference", "title": "Reference", "color": "raw"},
            {"key": "opentrack", "title": "Any2Track", "color": "track"},
            {"key": "protomotions", "title": "ProtoMotions", "color": "opt"},
            {"key": "physflow", "title": "PhysFlow", "color": "track-after"},
        ],
        "rows": rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--opentrack-manifest",
        type=Path,
        default=PROJECT_ROOT
        / "output/opentrack_lafan1_g1/viz_lafan1_40x600_20260605/manifest.json",
    )
    parser.add_argument(
        "--opentrack-metrics",
        type=Path,
        default=PROJECT_ROOT
        / "output/opentrack_lafan1_g1/viz_lafan1_40x600_20260605/metrics.json",
    )
    parser.add_argument(
        "--proto-eval-dir",
        type=Path,
        default=PROJECT_ROOT
        / "output/lafan1_g1_proto_baseline_eval/physflow_0605h_rollout_metrics_v100g470_20260605/eval_protomotions_g1_bones",
    )
    parser.add_argument(
        "--physflow-eval-dir",
        type=Path,
        default=PROJECT_ROOT
        / "output/lafan1_g1_proto_baseline_eval/physflow_0605h_rollout_metrics_v100g470_20260605/eval_physflow0605h",
    )
    args = parser.parse_args()

    manifest_path = build_lafan(args)
    print(manifest_path)


if __name__ == "__main__":
    main()
