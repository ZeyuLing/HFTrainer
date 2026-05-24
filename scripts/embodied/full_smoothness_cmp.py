#!/usr/bin/env python3
"""Full smoothness comparison across all physics-simulated motions."""
import json
import numpy as np
import os
import glob

CEPH = "/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"

def load_poses(json_path):
    with open(json_path) as f:
        data = json.load(f)
    poses = []
    for frame in data["frames"]:
        body = frame[0]
        pose = np.array(body["poses"][0], dtype=np.float32)
        poses.append(pose[:72])
    return np.array(poses)

def compute_jerk(poses, fps=30):
    dt = 1.0 / fps
    vel = np.diff(poses, axis=0) / dt
    acc = np.diff(vel, axis=0) / dt
    jerk = np.diff(acc, axis=0) / dt
    return float(np.mean(np.abs(jerk)))

def main():
    kin_dir = f"{CEPH}/output/embodied_t2m_v4/data/smpl_mesh"
    phys_dir = f"{CEPH}/output/embodied_t2m_v4/data/smpl_mesh_physics"

    phys_files = sorted(glob.glob(f"{phys_dir}/*.json"))
    print(f"Found {len(phys_files)} physics mesh JSONs")
    print()
    print(f"{'Stem':35s}  {'KIN':>8s}  {'PHYS':>8s}  {'Ratio':>8s}")
    print("-" * 70)

    ratios = []
    worse_cases = []
    for phys_path in phys_files:
        stem = os.path.basename(phys_path).replace(".json", "")
        kin_path = f"{kin_dir}/{stem}.json"
        if not os.path.exists(kin_path):
            print(f"{stem:35s}  {'N/A':>8s}  {'N/A':>8s}  {'N/A':>8s}")
            continue
        kin_jerk = compute_jerk(load_poses(kin_path))
        phys_jerk = compute_jerk(load_poses(phys_path))
        ratio = phys_jerk / max(kin_jerk, 1e-6)
        ratios.append(ratio)
        flag = "  <-- WORSE" if ratio > 1.0 else ""
        print(f"{stem:35s}  {kin_jerk:8.1f}  {phys_jerk:8.1f}  {ratio:8.3f}{flag}")
        if ratio > 1.0:
            worse_cases.append((stem, ratio))

    print()
    print("=" * 70)
    print(f"Total motions: {len(ratios)}")
    print(f"Mean ratio: {np.mean(ratios):.3f}")
    print(f"Median ratio: {np.median(ratios):.3f}")
    print(f"< 1.0 (smoother): {sum(r < 1.0 for r in ratios)}/{len(ratios)} ({sum(r < 1.0 for r in ratios)/len(ratios)*100:.0f}%)")
    print(f"> 1.0 (worse): {sum(r > 1.0 for r in ratios)}/{len(ratios)} ({sum(r > 1.0 for r in ratios)/len(ratios)*100:.0f}%)")
    print(f"> 2.0 (much worse): {sum(r > 2.0 for r in ratios)}/{len(ratios)}")
    print(f"Min ratio: {min(ratios):.3f}")
    print(f"Max ratio: {max(ratios):.3f}")

    if worse_cases:
        print(f"\nWorse cases (ratio > 1.0):")
        for stem, ratio in sorted(worse_cases, key=lambda x: -x[1]):
            print(f"  {stem:35s}  {ratio:.3f}")

if __name__ == "__main__":
    main()
