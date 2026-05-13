#!/usr/bin/env python3
"""Batch compare old vs new retargeting pipeline on multiple motions.

Runs the new pipeline on each motion NPZ, then compares the resulting cache
against the old (v4) cache. Produces a summary table of improvements.

Usage (on debug machine with mink/mujoco):
    cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
    python scripts/embodied/_batch_compare.py
"""
import sys
import os
import subprocess
import tempfile
import numpy as np

# Configuration
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
V4_DIR = os.path.join(PROJECT_ROOT, "output", "embodied_t2m_v4", "data")
NPZ_DIR = os.path.join(V4_DIR, "npz")
OLD_CACHE_DIR = os.path.join(V4_DIR, "caches")
PIPELINE_SCRIPT = os.path.join(PROJECT_ROOT, "scripts", "embodied", "pipeline_motion_to_robot.py")

# Diverse motion IDs to test
TEST_MOTIONS = [
    "v4_walk_005",       # walking (baseline)
    "v4_stand_001",      # standing still (should be very smooth)
    "v4_dance_002",      # dance
    "v4_gesture_003",    # hand wave
    "v4_sport_003",      # punch
    "v4_exercise_001",   # squat
    "v4_jog_001",        # jogging
    "v4_combo_005",      # walk + wave
]


def load_cache(path):
    import torch
    return torch.load(path, map_location='cpu', weights_only=False)


def compute_metrics(cache):
    """Compute jitter metrics from a ProtoMotions cache."""
    def to_np(x):
        if hasattr(x, 'numpy'):
            return x.numpy()
        return np.asarray(x)

    dof_vel = to_np(cache['dof_vel'])
    body_vel = to_np(cache['body_vel'])
    body_pos = to_np(cache['body_pos'])
    dt = float(cache['control_dt'])

    dof_accel = np.diff(dof_vel, axis=0) / dt
    body_accel = np.diff(body_vel, axis=0) / dt

    root_z = body_pos[:, 0, 2]

    return {
        'dof_vel_max': float(np.abs(dof_vel).max()),
        'dof_vel_mean': float(np.abs(dof_vel).mean()),
        'dof_accel_max': float(np.abs(dof_accel).max()),
        'dof_accel_mean': float(np.abs(dof_accel).mean()),
        'body_vel_max': float(np.abs(body_vel).max()),
        'body_vel_mean': float(np.abs(body_vel).mean()),
        'body_accel_max': float(np.abs(body_accel).max()),
        'body_accel_mean': float(np.abs(body_accel).mean()),
        'root_z_std': float(root_z.std()),
        'root_z_5f_drop': float(root_z[:5].max() - root_z[:5].min()),
    }


def run_new_pipeline(npz_path, output_path):
    """Run the new pipeline on an NPZ file."""
    cmd = [
        sys.executable, PIPELINE_SCRIPT,
        "--input", npz_path,
        "--output", output_path,
        "--fk-ground-mode", "global",
        # smoothing is on by default
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print(f"  PIPELINE FAILED: {result.stderr[-500:]}")
        return False
    return True


def main():
    tmpdir = tempfile.mkdtemp(prefix="retarget_compare_")
    print(f"Temp dir: {tmpdir}")
    print(f"Testing {len(TEST_MOTIONS)} motions\n")

    results = []
    for motion_id in TEST_MOTIONS:
        npz_path = os.path.join(NPZ_DIR, f"{motion_id}.npz")
        old_cache_path = os.path.join(OLD_CACHE_DIR, f"{motion_id}.pt")
        new_cache_path = os.path.join(tmpdir, f"{motion_id}_new.pt")

        if not os.path.exists(npz_path):
            print(f"[SKIP] {motion_id}: NPZ not found")
            continue
        if not os.path.exists(old_cache_path):
            print(f"[SKIP] {motion_id}: old cache not found")
            continue

        print(f"[{motion_id}] Running new pipeline...")
        ok = run_new_pipeline(npz_path, new_cache_path)
        if not ok:
            print(f"[FAIL] {motion_id}")
            continue

        old_cache = load_cache(old_cache_path)
        new_cache = load_cache(new_cache_path)

        old_m = compute_metrics(old_cache)
        new_m = compute_metrics(new_cache)

        results.append({
            'motion_id': motion_id,
            'old': old_m,
            'new': new_m,
        })
        print(f"  DOF accel max: {old_m['dof_accel_max']:.1f} -> {new_m['dof_accel_max']:.1f} ({old_m['dof_accel_max']/max(new_m['dof_accel_max'],1e-9):.2f}x)")
        print(f"  Body accel mean: {old_m['body_accel_mean']:.2f} -> {new_m['body_accel_mean']:.2f} ({old_m['body_accel_mean']/max(new_m['body_accel_mean'],1e-9):.2f}x)")
        print(f"  Root Z 5f drop: {old_m['root_z_5f_drop']:.4f} -> {new_m['root_z_5f_drop']:.4f}")
        print()

    if not results:
        print("No results!")
        return

    # Summary table
    print("\n" + "=" * 100)
    print("SUMMARY: Old vs New Pipeline Comparison")
    print("=" * 100)

    # Header
    metrics = ['dof_accel_max', 'dof_accel_mean', 'body_accel_max', 'body_accel_mean', 'root_z_5f_drop']
    header = f"{'Motion ID':<20}"
    for m in metrics:
        header += f" | {m:>18}"
    print(header)
    print("-" * len(header))

    # Per-motion rows (show ratio = old/new, >1 means improvement)
    for r in results:
        row = f"{r['motion_id']:<20}"
        for m in metrics:
            old_v = r['old'][m]
            new_v = r['new'][m]
            if new_v > 1e-9:
                ratio = old_v / new_v
                row += f" | {ratio:>17.2f}x"
            else:
                row += f" |              inf"
        print(row)

    # Average improvement
    print("-" * len(header))
    avg_row = f"{'AVERAGE':<20}"
    for m in metrics:
        ratios = []
        for r in results:
            old_v = r['old'][m]
            new_v = r['new'][m]
            if new_v > 1e-9:
                ratios.append(old_v / new_v)
        if ratios:
            avg_row += f" | {np.mean(ratios):>17.2f}x"
        else:
            avg_row += f" |              N/A"
    print(avg_row)

    # Absolute values table
    print(f"\n{'='*100}")
    print("ABSOLUTE VALUES (New Pipeline)")
    print("=" * 100)
    header2 = f"{'Motion ID':<20}"
    for m in metrics:
        header2 += f" | {m:>18}"
    print(header2)
    print("-" * len(header2))
    for r in results:
        row = f"{r['motion_id']:<20}"
        for m in metrics:
            val = r['new'][m]
            row += f" | {val:>18.2f}"
        print(row)

    print(f"\nTemp dir: {tmpdir}")
    print("Done!")


if __name__ == "__main__":
    main()
