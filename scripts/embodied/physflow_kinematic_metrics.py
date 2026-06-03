#!/usr/bin/env python3
"""G1 kinematic artifact metrics for PhysFlow T2M physical-realism evaluation.

These are the *simulation-free* physical-plausibility metrics defined in the
PhysFlow paper (sec_4_experiments): foot skating, ground penetration, floating
ratio, temporal jump rate, and acceleration/jerk -- computed directly on the
generated G1 robot motion (world-frame ``rigid_body_pos``). They quantify whether
a text-to-motion output is physically plausible *as a kinematic trajectory*,
complementing the tracker-in-the-loop (MuJoCo) physical-executability metrics.

Input: ProtoMotions ``.motion`` files (``torch.load`` dict with ``rigid_body_pos``
[T, 33, 3] in a Z-up world frame, optional ``dof_pos`` [T, 29]). Foot bodies are
``left_ankle_roll_link`` (idx 7) and ``right_ankle_roll_link`` (idx 13) in the
33-body DEFAULT_BODIES ordering.

Conventions match scripts/embodied/physflow_triplet_manifest.py: G1 MuJoCo is
Z-up (root height = qpos[2]); ground plane at z = 0.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

ROOT = Path("/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer")

# foot body indices in the 33-body DEFAULT_BODIES ordering (Z-up world frame).
LEFT_FOOT_IDX = 7
RIGHT_FOOT_IDX = 13
UP = 2  # z-up

# thresholds (metres / metres-per-second), tuned for G1 @ 30 fps.
CONTACT_H = 0.07      # foot considered "in contact" below this height
SKATE_VEL = 0.05      # horizontal speed (m/s) above which a contact frame counts as sliding
FLOAT_H = 0.15        # both feet above this => airborne / floating frame
PEN_EPS = 0.02        # body below -PEN_EPS counts as ground penetration
JUMP_DISP = 0.10      # per-frame body displacement (m) above which a frame is a "jump"


def _to_np(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float().numpy()
    return np.asarray(x, dtype=np.float32)


def _jerk(positions: np.ndarray, fps: float) -> float:
    """Mean jerk magnitude over all bodies (m/s^3). Mirrors
    m2m_eval_metrics.compute_jitter_positions (axis-agnostic)."""
    if positions.shape[0] < 4:
        return 0.0
    dt = 1.0 / fps
    jerk = np.diff(positions, n=3, axis=0) / (dt ** 3)
    return float(np.mean(np.linalg.norm(jerk.reshape(jerk.shape[0], -1), axis=-1)))


def _accel(positions: np.ndarray, fps: float) -> float:
    if positions.shape[0] < 3:
        return 0.0
    dt = 1.0 / fps
    acc = np.diff(positions, n=2, axis=0) / (dt ** 2)
    return float(np.mean(np.linalg.norm(acc.reshape(acc.shape[0], -1), axis=-1)))


def g1_kinematic_metrics(motion: Dict, fps: Optional[float] = None) -> Dict[str, float]:
    """Compute simulation-free kinematic artifact metrics from one G1 motion dict."""
    body_pos = _to_np(motion["rigid_body_pos"])  # [T, 33, 3]
    if body_pos.ndim != 3 or body_pos.shape[1] <= RIGHT_FOOT_IDX:
        raise ValueError(f"unexpected rigid_body_pos shape {body_pos.shape}")
    T = body_pos.shape[0]
    fps = float(fps if fps is not None else motion.get("fps", 30))
    dt = 1.0 / fps

    lf = body_pos[:, LEFT_FOOT_IDX, :]   # [T, 3]
    rf = body_pos[:, RIGHT_FOOT_IDX, :]
    lf_h, rf_h = lf[:, UP], rf[:, UP]

    out: Dict[str, float] = {"num_frames": float(T)}

    # ---- foot skating: horizontal slip while a foot is grounded ----
    if T >= 2:
        horiz = [c for c in range(3) if c != UP]
        lf_spd = np.linalg.norm(np.diff(lf[:, horiz], axis=0), axis=-1) / dt  # [T-1]
        rf_spd = np.linalg.norm(np.diff(rf[:, horiz], axis=0), axis=-1) / dt
        lf_contact = lf_h[:-1] < CONTACT_H
        rf_contact = rf_h[:-1] < CONTACT_H
        contact_spds = np.concatenate([lf_spd[lf_contact], rf_spd[rf_contact]])
        n_contact = int(contact_spds.size)
        out["foot_skate_speed"] = float(contact_spds.mean()) if n_contact else 0.0
        out["foot_skate_ratio"] = float((contact_spds > SKATE_VEL).mean()) if n_contact else 0.0
    else:
        out["foot_skate_speed"] = 0.0
        out["foot_skate_ratio"] = 0.0

    # ---- ground penetration: lowest body point below the floor ----
    min_z = body_pos[:, :, UP].min(axis=1)  # [T] lowest body per frame
    pen = np.clip(-min_z, 0.0, None)
    out["penetration_depth"] = float(pen.mean())        # mean penetration (m), 0 if never below floor
    out["penetration_max"] = float(pen.max())
    out["penetration_ratio"] = float((min_z < -PEN_EPS).mean())

    # ---- floating ratio: both feet airborne ----
    out["float_ratio"] = float(((lf_h > FLOAT_H) & (rf_h > FLOAT_H)).mean())

    # ---- temporal jump rate: discontinuous per-frame body displacement ----
    if T >= 2:
        frame_disp = np.linalg.norm(np.diff(body_pos, axis=0), axis=-1).max(axis=1)  # [T-1] max body move/frame
        out["jump_rate"] = float((frame_disp > JUMP_DISP).mean())
        out["max_frame_disp"] = float(frame_disp.max())
    else:
        out["jump_rate"] = 0.0
        out["max_frame_disp"] = 0.0

    # ---- smoothness: acceleration / jerk ----
    out["accel"] = _accel(body_pos, fps)
    out["jerk"] = _jerk(body_pos, fps)

    # ---- articulation (anti-freeze sanity): joint angle temporal std ----
    if "dof_pos" in motion:
        dof = _to_np(motion["dof_pos"])
        if dof.ndim == 2 and dof.shape[0] > 1:
            out["joint_std"] = float(np.std(dof, axis=0).mean())
            out["joint_vel_max"] = float(np.abs(np.diff(dof, axis=0) / dt).max())

    return out


def aggregate(metrics: List[Dict[str, float]]) -> Dict[str, float]:
    if not metrics:
        return {}
    keys = set().union(*[m.keys() for m in metrics])
    return {k: float(np.mean([m[k] for m in metrics if k in m])) for k in sorted(keys)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion-dir", help="directory of .motion files")
    ap.add_argument("--motion-files", nargs="*", default=[])
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--out", default=None, help="optional per-motion + aggregate JSON")
    args = ap.parse_args()

    paths: List[Path] = [Path(p) for p in args.motion_files]
    if args.motion_dir:
        paths += sorted(Path(args.motion_dir).glob("*.motion"))
    if not paths:
        raise SystemExit("no .motion files given")

    per_motion = {}
    metrics_list = []
    for p in paths:
        try:
            m = torch.load(p, map_location="cpu")
            mm = g1_kinematic_metrics(m, fps=args.fps)
            per_motion[p.name] = mm
            metrics_list.append(mm)
        except Exception as e:  # noqa: BLE001
            per_motion[p.name] = {"error": str(e)}

    agg = aggregate(metrics_list)
    print("=== aggregate kinematic artifact metrics (n=%d) ===" % len(metrics_list))
    for k in ["foot_skate_speed", "foot_skate_ratio", "penetration_depth",
              "penetration_ratio", "float_ratio", "jump_rate", "accel", "jerk",
              "joint_std"]:
        if k in agg:
            print(f"  {k:20s} {agg[k]:.4f}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(
            {"aggregate": agg, "per_motion": per_motion}, indent=2))
        print(f"[kinematic] wrote {args.out}")


if __name__ == "__main__":
    main()
