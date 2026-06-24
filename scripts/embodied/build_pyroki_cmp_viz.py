#!/usr/bin/env python3
"""Build a 3-column comparison viewer payload: SMPL | GMR-retarget | PyRoki-retarget.

Pure numpy/json (no mujoco): reuses the existing GMR viz dir for SMPL meshes and the
GMR robot_frames (which already embed G1 link meshes), and generates PyRoki robot_frames
from the PyRoki AMP NPZ by reusing GMR's `bodies` mesh geometry (same robot, same body
order = STD_BODY_NAMES) and replacing only the per-frame poses.

Writes output/pyroki_cmp_viz/{robot_frames_pyroki/, manifest.json}. SMPL + GMR frames are
referenced from the existing GMR viz dir.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def pyroki_amp_to_rf(npz_path: Path, bodies: list, fps_fallback: int) -> dict:
    d = np.load(npz_path, allow_pickle=True)
    body_pos = d["body_positions"].astype(float)       # (T,30,3)
    body_rot_xyzw = d["body_rotations"].astype(float)  # (T,30,4)
    fps = int(round(float(np.asarray(d["fps"]).reshape(-1)[0]))) if "fps" in d.files else fps_fallback
    T = body_pos.shape[0]
    frames = [{
        "body_pos": body_pos[t].tolist(),
        "body_quat": body_rot_xyzw[t][:, [3, 0, 1, 2]].tolist(),  # xyzw->wxyz
    } for t in range(T)]
    return {"type": "robot_frames", "robot": "g1", "fps": fps, "num_frames": T,
            "num_bodies": len(bodies), "bodies": bodies, "frames": frames}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gmr-viz-dir", default=str(PROJECT_ROOT / "output" / "g1_amp_viz_fixed2"))
    ap.add_argument("--pyroki-amp-dir", default=str(PROJECT_ROOT / "data" / "g1_pyroki"))
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "output" / "pyroki_cmp_viz"))
    args = ap.parse_args()

    gmr_dir = Path(args.gmr_viz_dir)
    amp_dir = Path(args.pyroki_amp_dir)
    out_dir = Path(args.out_dir)
    rf_py_dir = out_dir / "robot_frames_pyroki"
    rf_py_dir.mkdir(parents=True, exist_ok=True)

    gmr_manifest = json.load(open(gmr_dir / "manifest.json"))
    rows_in = gmr_manifest["rows"]

    # index pyroki AMP npz by the 'XX_' numeric prefix
    amp_by_prefix = {}
    for p in amp_dir.glob("*.npz"):
        pref = p.name.split("_", 1)[0]
        amp_by_prefix[pref] = p

    rows_out = []
    for r in rows_in:
        name = r["name"]            # e.g. 00_S3_Jog_3_poses_origintime_0.3_10.3
        prefix = name.split("_", 1)[0]
        smpl_path = r["smpl_path"]
        gmr_path = r["g1_path"]
        amp = amp_by_prefix.get(prefix)
        if amp is None:
            print(f"[skip] {name}: no pyroki amp for prefix {prefix}")
            continue
        # reuse GMR bodies (mesh geometry) for the pyroki frames
        gmr_rf = json.load(open(gmr_path))
        bodies = gmr_rf["bodies"]
        py_rf = pyroki_amp_to_rf(amp, bodies, fps_fallback=gmr_rf.get("fps", 15))
        py_path = (rf_py_dir / f"{prefix}_{name}.json").resolve()
        json.dump(py_rf, open(py_path, "w"))
        rows_out.append({
            "name": name,
            "source": r.get("source", ""),
            "smpl_path": smpl_path,
            "gmr_path": gmr_path,
            "pyroki_path": str(py_path),
            "gmr_frames": gmr_rf.get("num_frames"),
            "pyroki_frames": py_rf["num_frames"],
        })
        print(f"[ok] {name}: gmr {gmr_rf.get('num_frames')}f, pyroki {py_rf['num_frames']}f")

    manifest = {
        "title": "SMPL vs GMR vs PyRoki (G1 retargeting)",
        "columns": ["SMPL (source)", "GMR (per-frame IK)", "PyRoki (trajectory opt)"],
        "rows": rows_out,
    }
    json.dump(manifest, open(out_dir / "manifest.json", "w"), indent=2)
    print(f"\nWrote {out_dir/'manifest.json'} with {len(rows_out)} rows")


if __name__ == "__main__":
    main()
