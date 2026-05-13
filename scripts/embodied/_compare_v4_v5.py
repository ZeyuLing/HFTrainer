#!/usr/bin/env python3
"""Generate comprehensive comparison report: V4 (old pipeline) vs V5 (new pipeline).

Compares all ProtoMotions caches in two directories and produces:
1. Per-motion jitter metrics comparison
2. Summary statistics by motion category
3. Improvement distribution histogram data
4. Worst-case analysis (which motions need more work)

Usage:
    python scripts/embodied/_compare_v4_v5.py
"""
import os
import sys
import json
import numpy as np
from collections import defaultdict

# Adjust for cluster vs local path
if os.path.exists("/apdcephfs_cq11"):
    BASE = "/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
else:
    BASE = "/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"

V4_CACHE_DIR = os.path.join(BASE, "output/embodied_t2m_v4/data/caches")
V5_CACHE_DIR = os.path.join(BASE, "output/embodied_t2m_v5/data/caches")


def to_np(x):
    if hasattr(x, 'numpy') and callable(x.numpy):
        return x.numpy()
    return np.asarray(x)


def compute_metrics(cache):
    dof_vel = to_np(cache['dof_vel'])
    body_vel = to_np(cache['body_vel'])
    body_pos = to_np(cache['body_pos'])
    dof_pos = to_np(cache['dof_pos'])
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
        'root_z_range': float(root_z.max() - root_z.min()),
        'root_z_5f_drop': float(root_z[:5].max() - root_z[:5].min()),
        'dof_pos_range': float(dof_pos.max() - dof_pos.min()),
        'num_frames': int(dof_vel.shape[0]),
    }


def get_category(motion_id):
    """Extract motion category from ID (e.g., v4_walk_005 -> walk)."""
    # Strip 'motion_' prefix if present (from --npz-dir mode)
    if motion_id.startswith('motion_'):
        motion_id = motion_id[7:]
    parts = motion_id.split('_')
    if len(parts) >= 2:
        return parts[1]  # v4_walk_005 -> walk
    return 'unknown'


def main():
    import torch

    # Find matching caches
    v4_files = set(f for f in os.listdir(V4_CACHE_DIR) if f.endswith('.pt') and '_tracked' not in f)
    v5_files = set(f for f in os.listdir(V5_CACHE_DIR) if f.endswith('.pt') and '_tracked' not in f)

    # V5 uses motion_{stem}.pt naming from --npz-dir mode
    # Map v5 names back to v4 names
    v5_to_v4 = {}
    for v5f in v5_files:
        # motion_v4_walk_005.pt -> v4_walk_005.pt
        stem = v5f.replace('motion_', '')
        if stem in v4_files:
            v5_to_v4[v5f] = stem

    print(f"V4 caches: {len(v4_files)}")
    print(f"V5 caches: {len(v5_files)}")
    print(f"Matched pairs: {len(v5_to_v4)}")
    print()

    if not v5_to_v4:
        print("No matching cache pairs found!")
        return

    results = []
    categories = defaultdict(list)

    for v5_name, v4_name in sorted(v5_to_v4.items()):
        v4_path = os.path.join(V4_CACHE_DIR, v4_name)
        v5_path = os.path.join(V5_CACHE_DIR, v5_name)

        try:
            v4_cache = torch.load(v4_path, map_location='cpu', weights_only=False)
            v5_cache = torch.load(v5_path, map_location='cpu', weights_only=False)
        except Exception as e:
            print(f"  Error loading {v4_name}: {e}")
            continue

        v4_m = compute_metrics(v4_cache)
        v5_m = compute_metrics(v5_cache)

        motion_id = v4_name.replace('.pt', '')
        cat = get_category(motion_id)

        entry = {
            'motion_id': motion_id,
            'category': cat,
            'v4': v4_m,
            'v5': v5_m,
        }
        results.append(entry)
        categories[cat].append(entry)

    # =====================================================================
    # Per-motion comparison
    # =====================================================================
    key_metrics = ['dof_accel_max', 'dof_accel_mean', 'body_accel_max', 'body_accel_mean']

    print("=" * 120)
    print("PER-MOTION COMPARISON (improvement ratios = V4/V5, >1 means V5 is better)")
    print("=" * 120)
    header = f"{'Motion ID':<25} {'Cat':<10}"
    for m in key_metrics:
        header += f" | {m:>16}"
    print(header)
    print("-" * len(header))

    for r in results:
        row = f"{r['motion_id']:<25} {r['category']:<10}"
        for m in key_metrics:
            old = r['v4'][m]
            new = r['v5'][m]
            ratio = old / new if new > 1e-9 else float('inf')
            row += f" | {ratio:>15.2f}x"
        print(row)

    # =====================================================================
    # Category summary
    # =====================================================================
    print(f"\n{'='*120}")
    print("CATEGORY SUMMARY (average improvement ratio)")
    print("=" * 120)
    header = f"{'Category':<12} {'Count':>5}"
    for m in key_metrics:
        header += f" | {m:>16}"
    print(header)
    print("-" * len(header))

    all_ratios = {m: [] for m in key_metrics}

    for cat in sorted(categories.keys()):
        entries = categories[cat]
        row = f"{cat:<12} {len(entries):>5}"
        for m in key_metrics:
            ratios = []
            for e in entries:
                old, new = e['v4'][m], e['v5'][m]
                if new > 1e-9:
                    ratios.append(old / new)
            avg = np.mean(ratios) if ratios else 0
            all_ratios[m].extend(ratios)
            row += f" | {avg:>15.2f}x"
        print(row)

    print("-" * len(header))
    row = f"{'OVERALL':<12} {len(results):>5}"
    for m in key_metrics:
        avg = np.mean(all_ratios[m]) if all_ratios[m] else 0
        row += f" | {avg:>15.2f}x"
    print(row)

    # =====================================================================
    # Absolute values summary (V5)
    # =====================================================================
    print(f"\n{'='*120}")
    print("V5 ABSOLUTE VALUES (max and mean across all motions)")
    print("=" * 120)
    for m in key_metrics:
        v5_vals = [r['v5'][m] for r in results]
        print(f"  {m:<20}: max={max(v5_vals):>10.1f}, mean={np.mean(v5_vals):>10.1f}, median={np.median(v5_vals):>10.1f}")

    # =====================================================================
    # Worst cases (V5 still has high values)
    # =====================================================================
    print(f"\n{'='*120}")
    print("WORST CASES IN V5 (top 10 by dof_accel_max)")
    print("=" * 120)
    by_dof_accel = sorted(results, key=lambda r: r['v5']['dof_accel_max'], reverse=True)
    for r in by_dof_accel[:10]:
        v4_val = r['v4']['dof_accel_max']
        v5_val = r['v5']['dof_accel_max']
        ratio = v4_val / v5_val if v5_val > 1e-9 else float('inf')
        print(f"  {r['motion_id']:<25} {r['category']:<10} V4={v4_val:>8.1f} -> V5={v5_val:>8.1f} ({ratio:.1f}x)")

    print(f"\nWORST CASES IN V5 (top 10 by body_accel_mean)")
    print("=" * 120)
    by_body_accel = sorted(results, key=lambda r: r['v5']['body_accel_mean'], reverse=True)
    for r in by_body_accel[:10]:
        v4_val = r['v4']['body_accel_mean']
        v5_val = r['v5']['body_accel_mean']
        ratio = v4_val / v5_val if v5_val > 1e-9 else float('inf')
        print(f"  {r['motion_id']:<25} {r['category']:<10} V4={v4_val:>8.2f} -> V5={v5_val:>8.2f} ({ratio:.1f}x)")

    # =====================================================================
    # Save JSON for programmatic analysis
    # =====================================================================
    report_path = os.path.join(BASE, "output/embodied_t2m_v5/comparison_report.json")
    report = {
        'num_motions': len(results),
        'overall_improvement': {m: float(np.mean(all_ratios[m])) for m in key_metrics},
        'per_motion': results,
        'per_category': {
            cat: {
                'count': len(entries),
                'avg_improvement': {
                    m: float(np.mean([
                        e['v4'][m] / e['v5'][m]
                        for e in entries
                        if e['v5'][m] > 1e-9
                    ])) for m in key_metrics
                }
            }
            for cat, entries in categories.items()
        }
    }
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report saved to: {report_path}")


if __name__ == "__main__":
    main()
