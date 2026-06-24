#!/usr/bin/env python3
"""Build SMPL-vs-G1 viewer payloads straight from data/g1 AMP npz outputs.

For each chosen retargeted motion (data/g1/<rel>.npz) this emits:
  * G1 robot_frames json  -- from body_positions/body_rotations in the AMP npz
                             (no re-FK needed; ground-aligned already)
  * SMPL mesh json        -- from the source AMASS npz data/hymotion_data/<rel>.npz

Writes a manifest.json consumed by /smpl_vs_g1.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "demo", str(PROJECT_ROOT / "scripts" / "embodied" / "smpl_g1_compare_demo.py")
)
_demo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_demo)

DEFAULT_G1_DIR = PROJECT_ROOT / "data" / "g1"
DEFAULT_AMASS_DIR = PROJECT_ROOT / "data" / "hymotion_data"
DEFAULT_MJCF = PROJECT_ROOT / "ref_repo" / "ProtoMotions" / "protomotions" / "data" / "assets" / "mjcf" / "g1_holo_compat.xml"
DEFAULT_OUT = PROJECT_ROOT / "output" / "g1_amp_viz"

STD_BODY_NAMES = [
    "pelvis",
    "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link",
    "left_knee_link", "left_ankle_pitch_link", "left_ankle_roll_link",
    "right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link",
    "right_knee_link", "right_ankle_pitch_link", "right_ankle_roll_link",
    "waist_yaw_link", "waist_roll_link", "torso_link",
    "left_shoulder_pitch_link", "left_shoulder_roll_link", "left_shoulder_yaw_link",
    "left_elbow_link", "left_wrist_roll_link", "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_shoulder_pitch_link", "right_shoulder_roll_link", "right_shoulder_yaw_link",
    "right_elbow_link", "right_wrist_roll_link", "right_wrist_pitch_link",
    "right_wrist_yaw_link",
]

KEYWORDS = ["walk", "run", "jog", "jump", "kick", "dance", "punch", "throw",
            "box", "wave", "turn", "squat", "spin", "sit", "crouch"]


def amp_to_robot_frames(npz_path: Path, mesh_by_name: dict) -> dict:
    d = np.load(npz_path, allow_pickle=True)
    body_names = [str(x) for x in d["body_names"]]
    body_pos = d["body_positions"].astype(float)        # (T,30,3)
    body_rot_xyzw = d["body_rotations"].astype(float)    # (T,30,4)
    fps = int(round(float(np.asarray(d["fps"]).reshape(-1)[0])))
    T = body_pos.shape[0]

    bodies = [{"name": n, "meshes": mesh_by_name.get(n, [])} for n in body_names]
    frames = []
    for t in range(T):
        quat_wxyz = body_rot_xyzw[t][:, [3, 0, 1, 2]]  # xyzw -> wxyz
        frames.append({
            "body_pos": body_pos[t].tolist(),
            "body_quat": quat_wxyz.tolist(),
        })
    return {
        "type": "robot_frames",
        "robot": "g1",
        "fps": fps,
        "num_frames": T,
        "num_bodies": len(bodies),
        "bodies": bodies,
        "frames": frames,
    }


def build_smpl_from_amass(npz: Path, target_fps: int, max_frames: int):
    d = np.load(npz, allow_pickle=True)
    poses = d["poses"].astype(np.float32)
    trans = d["trans"].astype(np.float32)
    betas = d["betas"].astype(np.float32) if "betas" in d.files else np.zeros((1, 16), np.float32)
    gender = str(d["gender"]) if "gender" in d.files else "neutral"
    src_fps = int(np.asarray(d.get("mocap_framerate", 30)).reshape(-1)[0])
    stride = max(1, round(src_fps / target_fps))
    poses, trans = poses[::stride], trans[::stride]
    if poses.shape[0] > max_frames:
        poses, trans = poses[:max_frames], trans[:max_frames]
    out_fps = max(1, round(src_fps / stride))
    return _demo.build_smpl_json(poses, trans, betas, out_fps, gender), out_fps, poses.shape[0]


def pick(g1_dir: Path, num: int, seed: int, scan_cap: int = 60000):
    """Single bounded os.walk; bucket by keyword, fill rest with random."""
    import os
    kw_hit: dict[str, Path] = {}
    pool: list[Path] = []
    rng = random.Random(seed)
    for root, _dirs, files in os.walk(g1_dir):
        for f in files:
            if not f.endswith(".npz"):
                continue
            p = Path(root) / f
            low = f.lower()
            for kw in KEYWORDS:
                if kw in low and kw not in kw_hit:
                    kw_hit[kw] = p
            if len(pool) < 3000:
                pool.append(p)
        if len(pool) >= scan_cap or len(kw_hit) >= len(KEYWORDS):
            if len(pool) >= 200:
                break
    picks = list(kw_hit.values())
    rng.shuffle(pool)
    for p in pool:
        if len(picks) >= num:
            break
        if p not in picks:
            picks.append(p)
    return picks[:num]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g1-dir", default=str(DEFAULT_G1_DIR))
    ap.add_argument("--amass-dir", default=str(DEFAULT_AMASS_DIR))
    ap.add_argument("--mjcf", default=str(DEFAULT_MJCF))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--num", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target-fps", type=int, default=30)
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--names", nargs="*", default=None,
                    help="explicit data/g1-relative npz paths")
    args = ap.parse_args()

    g1_dir = Path(args.g1_dir)
    amass_dir = Path(args.amass_dir)
    out_dir = Path(args.out_dir)
    smpl_dir = out_dir / "smpl_mesh"
    rf_dir = out_dir / "robot_frames"
    smpl_dir.mkdir(parents=True, exist_ok=True)
    rf_dir.mkdir(parents=True, exist_ok=True)

    _, bodies = _demo.load_g1_model(Path(args.mjcf))
    mesh_by_name = {b["name"]: b["meshes"] for b in bodies}

    if args.names:
        picks = [g1_dir / n for n in args.names]
    else:
        picks = pick(g1_dir, args.num, args.seed)
    print(f"[pick] {len(picks)} motions")

    rows = []
    for idx, g1_npz in enumerate(picks):
        rel = g1_npz.relative_to(g1_dir)
        amass = amass_dir / rel
        if not amass.is_file():
            print(f"[skip] no source AMASS for {rel}")
            continue
        stem = f"{idx:02d}_" + g1_npz.stem
        try:
            rf = amp_to_robot_frames(g1_npz, mesh_by_name)
            smpl_json, out_fps, nfr = build_smpl_from_amass(amass, args.target_fps, args.max_frames)
        except Exception as e:
            print(f"[fail] {stem}: {e}")
            continue
        smpl_path = (smpl_dir / f"{stem}.json").resolve()
        rf_path = (rf_dir / f"{stem}.json").resolve()
        json.dump(smpl_json, open(smpl_path, "w"))
        json.dump(rf, open(rf_path, "w"))
        rows.append({
            "name": stem,
            "source": str(rel),
            "frames": rf["num_frames"],
            "fps": rf["fps"],
            "smpl_path": str(smpl_path),
            "g1_path": str(rf_path),
        })
        print(f"[ok] {stem}: G1 {rf['num_frames']}f@{rf['fps']} | SMPL {nfr}f@{out_fps}")

    manifest = {
        "title": "data/g1 retarget: SMPL (before) vs G1 (after)",
        "g1_dir": str(g1_dir),
        "rows": rows,
    }
    man = out_dir / "manifest.json"
    json.dump(manifest, open(man, "w"), indent=2)
    print(f"[done] {len(rows)} rows -> {man}")


if __name__ == "__main__":
    main()
