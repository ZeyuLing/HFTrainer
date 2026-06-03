#!/usr/bin/env python3
"""Build an embodied_viz /physflow_triplet manifest for the overfit run.

Three columns per motion, all rendered as G1 robot_frames JSON:
  raw_reference        -> KIMODO reference (target, from .motion)
  optimized_reference  -> tracker rollout @ EARLY epoch  (reconstruction, early)
  tracked_rollout      -> tracker rollout @ LATE epoch    (reconstruction, final)

The tracker columns come from ProtoMotions ``predicted_motion_lib_epoch_*.pt``
(``gts`` global body positions [F,33,3], ``grs`` XYZW quats [F,33,4]) which the
in-training mimic evaluator saved. We strip the IsaacGym per-env XY grid offset
(anchor each tracker rollout's t=0 pelvis to the reference t=0 pelvis) so all
three columns live in the same world frame.

Run with py3.10 (`python3`): only torch.load + numpy, no IsaacGym.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import sys
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2]))
from scripts.embodied.physflow_triplet_manifest import (  # noqa: E402
    DEFAULT_BODIES,
    MESHES_BY_BODY,
    motion_to_robot_frames,
    _parse_g1_body_meshes,
)


def _bodies_meta():
    try:
        bodies = _parse_g1_body_meshes()
        if [b["name"] for b in bodies] == DEFAULT_BODIES:
            return bodies
    except Exception:
        pass
    return [
        {"name": n, "meshes": [{"file": m, "pos": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0]}
                                for m in MESHES_BY_BODY.get(n, [])]}
        for n in DEFAULT_BODIES
    ]


def _per_motion_frames(d):
    total = d["gts"].shape[0]
    ls = d["length_starts"].long().tolist()
    nm = len(ls)
    nf = [(ls[m + 1] - ls[m]) if m + 1 < nm else (total - ls[m]) for m in range(nm)]
    return ls, nf


def tracked_to_robot_frames(gts, grs, bodies, fps, out_path):
    """gts [T,33,3], grs [T,33,4] XYZW -> robot_frames JSON (WXYZ)."""
    pos = gts.astype(np.float32)
    quat_wxyz = grs[..., [3, 0, 1, 2]].astype(np.float32)
    frames = [{"body_pos": pos[t].tolist(), "body_quat": quat_wxyz[t].tolist()}
              for t in range(len(pos))]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "type": "robot_frames", "robot": "g1", "fps": int(fps),
        "num_frames": len(frames), "num_bodies": len(bodies),
        "bodies": bodies, "frames": frames,
    }))
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--late-epoch", type=int, default=740)
    ap.add_argument("--early-epoch", type=int, default=20)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-motions", type=int, default=99)
    args = ap.parse_args()

    rd = Path(args.results_dir)
    # Absolute out_dir so manifest column paths are absolute — the embodied_viz
    # /api/robot_frames endpoint only resolves relative paths under DATA_DIR,
    # but accepts absolute paths directly.
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    bodies = _bodies_meta()

    late = torch.load(rd / f"predicted_motion_lib_epoch_{args.late_epoch}.pt",
                      map_location="cpu", weights_only=False)
    ep = rd / f"predicted_motion_lib_epoch_{args.early_epoch}.pt"
    early = torch.load(ep, map_location="cpu", weights_only=False) if ep.exists() else None

    files = list(late["motion_files"])
    nm = min(len(files), args.max_motions)
    ls_l, nf_l = _per_motion_frames(late)
    if early is not None:
        ls_e, nf_e = _per_motion_frames(early)

    def slice_motion(d, ls, nf, m):
        s, n = ls[m], nf[m]
        return d["gts"][s:s + n].numpy().copy(), d["grs"][s:s + n].numpy().copy()

    def fps_of(d, m):
        dt = float(d["motion_dt"][m])
        return int(round(1.0 / dt)) if dt > 0 else 30

    def resample_root(root_xyz, n_to):
        """Linearly resample [Tt,3] root path onto n_to samples over [0,1] time."""
        Tt = len(root_xyz)
        if Tt == n_to or Tt < 2:
            return root_xyz[:n_to]
        ti = np.linspace(0.0, 1.0, Tt)
        to = np.linspace(0.0, 1.0, n_to)
        return np.stack([np.interp(to, ti, root_xyz[:, k]) for k in range(3)], axis=1)

    rows = []
    for m in range(nm):
        mf = files[m]
        stem = Path(mf).stem
        try:
            ref = torch.load(mf, map_location="cpu", weights_only=False)
        except Exception as e:
            print("skip", m, stem, e)
            continue
        ref_pos = ref["rigid_body_pos"].numpy()
        ref_fps = int(ref.get("fps", 30))
        ref_xy0 = ref_pos[0, 0, :2].copy()
        T_ref = len(ref_pos)

        # reference column (reuse the canonical .motion -> robot_frames)
        ref_json = out_dir / "robot_frames" / f"{stem}.reference.json"
        motion_to_robot_frames(Path(mf), ref_json)

        # late tracker column. Tracker rollout is recorded at the SIM rate
        # (motion_dt -> typically 50fps, ~198 frames for a 4s clip) while the
        # KIMODO reference is 30fps/120 frames. We keep each column at its OWN
        # native fps + full frame count (so both play the full ~4s in real
        # time), and only RESAMPLE onto the reference timeline when computing
        # trajectory-error metrics so frames are compared at matched times.
        late_fps = fps_of(late, m)
        gl, ql = slice_motion(late, ls_l, nf_l, m)
        gl[..., :2] -= (gl[0, 0, :2] - ref_xy0)  # strip env offset
        late_json = out_dir / "robot_frames" / f"{stem}.track_e{args.late_epoch}.json"
        tracked_to_robot_frames(gl, ql, bodies, late_fps, late_json)
        # per-motion root metrics (late): resample tracker root onto ref times
        rr = ref_pos[:, 0, :]                       # [T_ref,3]
        tr = resample_root(gl[:, 0, :], T_ref)      # [T_ref,3]
        root_err = float(np.mean(np.linalg.norm(tr - rr, axis=1)))
        disp_ref = float(np.linalg.norm(rr[-1, :2] - rr[0, :2]))
        disp_trk = float(np.linalg.norm(gl[-1, 0, :2] - gl[0, 0, :2]))

        # early tracker column (optional)
        if early is not None:
            early_fps = fps_of(early, m)
            ge, qe = slice_motion(early, ls_e, nf_e, m)
            ge[..., :2] -= (ge[0, 0, :2] - ref_xy0)
            early_json = out_dir / "robot_frames" / f"{stem}.track_e{args.early_epoch}.json"
            tracked_to_robot_frames(ge, qe, bodies, early_fps, early_json)
            te = resample_root(ge[:, 0, :], T_ref)
            re = ref_pos[:, 0, :]
            early_err = float(np.mean(np.linalg.norm(te - re, axis=1)))
            opt_col = {"status": "ready",
                       "title": f"Tracker @ epoch {args.early_epoch} (early)",
                       "path": str(early_json),
                       "metrics": {"root_traj_err_mean_m": round(early_err, 3)}}
        else:
            opt_col = {"status": "pending",
                       "title": f"Tracker @ epoch {args.early_epoch} (early)",
                       "path": "", "metrics": {}}

        rows.append({
            "iteration": 0,
            "iteration_label": "overfit",
            "prompt_id": stem,
            "prompt": stem.split("_", 3)[-1].replace("_", " ") if "_" in stem else stem,
            "category": "locomotion" if disp_ref > 1.0 else "in_place",
            "difficulty": None,
            "seed": None,
            "sample_idx": 0,
            "_sort_disp": disp_ref,
            "columns": {
                "raw_reference": {
                    "status": "ready", "title": "KIMODO Reference (target)",
                    "path": str(ref_json),
                    "metrics": {"root_displacement_ref_m": round(disp_ref, 3)},
                },
                "optimized_reference": opt_col,
                "tracked_rollout": {
                    "status": "ready",
                    "title": f"Tracker @ epoch {args.late_epoch} (final)",
                    "path": str(late_json),
                    "metrics": {
                        "root_traj_err_mean_m": round(root_err, 3),
                        "root_displacement_ref_m": round(disp_ref, 3),
                        "root_displacement_track_m": round(disp_trk, 3),
                        "root_displacement_error_m": round(abs(disp_ref - disp_trk), 3),
                    },
                },
            },
        })
        print(f"row {m:3d} {stem[:46]:46s} disp_ref={disp_ref:5.2f} disp_trk={disp_trk:5.2f} root_err={root_err:.3f}")

    # locomotion (most translation) first
    rows.sort(key=lambda r: -r.get("_sort_disp", 0.0))
    for r in rows:
        r.pop("_sort_disp", None)

    manifest = {
        "schema_version": 1,
        "project": "PhysFlow KIMODO-G1 Overfit — Reference vs Tracker (e20/e740)",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "generated_from": {"results_dir": str(rd),
                           "late_epoch": args.late_epoch,
                           "early_epoch": args.early_epoch},
        "rows": rows,
    }
    mp = out_dir / "manifest.json"
    mp.write_text(json.dumps(manifest, indent=2))
    print("MANIFEST_DONE", mp, "rows=", len(rows))


if __name__ == "__main__":
    main()
