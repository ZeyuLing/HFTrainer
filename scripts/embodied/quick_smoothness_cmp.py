#!/usr/bin/env python3
"""Quick smoothness comparison between kinematic and physics-sim mesh JSONs."""
import json
import numpy as np
import sys
import os

CEPH = "/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"

def load_poses_from_mesh_json(json_path):
    """Load SMPL poses from mesh JSON."""
    with open(json_path) as f:
        data = json.load(f)
    poses = []
    for frame in data["frames"]:
        body = frame[0]
        pose = np.array(body["poses"][0], dtype=np.float32)  # (156,) SMPL+H
        poses.append(pose[:72])  # only first 72 (SMPL)
    return np.array(poses)  # (T, 72)

def compute_smoothness(poses, fps=30):
    """Compute angular jerk and acceleration."""
    dt = 1.0 / fps
    vel = np.diff(poses, axis=0) / dt
    acc = np.diff(vel, axis=0) / dt
    jerk = np.diff(acc, axis=0) / dt
    mean_jerk = float(np.mean(np.abs(jerk)))
    mean_acc = float(np.mean(np.abs(acc)))
    return mean_jerk, mean_acc

def main():
    stems = sys.argv[1:] if len(sys.argv) > 1 else [
        "v4_crouch_001", "v4_turn_004", "v4_balance_003", "v4_walk_001",
        "v4_gesture_010", "v4_interact_002"
    ]

    kin_dir = f"{CEPH}/output/embodied_t2m_v4/data/smpl_mesh"
    phys_dir = f"{CEPH}/output/embodied_t2m_v4/data/smpl_mesh_physics"
    new_phys_dir = "/tmp/test_smooth_fix"

    print(f"{'Stem':30s}  {'KIN_jerk':>10s}  {'OLD_phys':>10s}  {'NEW_phys':>10s}  "
          f"{'OLD_ratio':>10s}  {'NEW_ratio':>10s}  {'Improve':>8s}")
    print("-" * 110)

    for stem in stems:
        kin_path = f"{kin_dir}/{stem}.json"
        old_phys_path = f"{phys_dir}/{stem}.json"
        new_phys_path = f"{new_phys_dir}/{stem}.json"

        if not os.path.exists(kin_path):
            print(f"  {stem}: kinematic not found")
            continue

        kin_poses = load_poses_from_mesh_json(kin_path)
        kin_jerk, _ = compute_smoothness(kin_poses)

        old_jerk_str = "N/A"
        old_ratio_str = "N/A"
        if os.path.exists(old_phys_path):
            old_poses = load_poses_from_mesh_json(old_phys_path)
            old_jerk, _ = compute_smoothness(old_poses)
            old_ratio = old_jerk / max(kin_jerk, 1e-6)
            old_jerk_str = f"{old_jerk:.1f}"
            old_ratio_str = f"{old_ratio:.3f}"
        else:
            old_jerk = None
            old_ratio = None

        new_jerk_str = "N/A"
        new_ratio_str = "N/A"
        improve_str = ""
        if os.path.exists(new_phys_path):
            new_poses = load_poses_from_mesh_json(new_phys_path)
            new_jerk, _ = compute_smoothness(new_poses)
            new_ratio = new_jerk / max(kin_jerk, 1e-6)
            new_jerk_str = f"{new_jerk:.1f}"
            new_ratio_str = f"{new_ratio:.3f}"
            if old_jerk is not None:
                pct = (old_jerk - new_jerk) / max(old_jerk, 1e-6) * 100
                improve_str = f"{pct:+.1f}%"

        print(f"{stem:30s}  {kin_jerk:10.1f}  {old_jerk_str:>10s}  {new_jerk_str:>10s}  "
              f"{old_ratio_str:>10s}  {new_ratio_str:>10s}  {improve_str:>8s}")

if __name__ == "__main__":
    main()
