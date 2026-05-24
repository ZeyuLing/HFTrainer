#!/usr/bin/env python3
"""Spot-check generated eval outputs for velocity spikes."""
import os, sys, glob
import numpy as np

eval_dir = '/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/work_dirs/prism_1b_tp2m_multiframe_kt_spectral/eval_hml3d_rewritten'

npz_files = sorted(glob.glob(os.path.join(eval_dir, '*.npz')))
print(f"Found {len(npz_files)} npz files")

spike_count = 0
total_checked = 0

for f in npz_files[:100]:  # Check first 100
    try:
        d = np.load(f, allow_pickle=True)
        bp = d['body_pose']
        T = bp.shape[0]
        if T < 20:
            continue
        bp_flat = bp.reshape(T, -1)
        diffs = np.diff(bp_flat, axis=0)
        vel = np.linalg.norm(diffs, axis=1)

        # Spike detection: vel[0] > 3x the mean of frames 15+
        mean_stable = vel[15:].mean()
        ratio = vel[0] / (mean_stable + 1e-8)

        total_checked += 1
        if ratio > 3.0:
            spike_count += 1
            fname = os.path.basename(f)
            print(f"  SPIKE: {fname} vel[0]={vel[0]:.3f}, stable_mean={mean_stable:.3f}, ratio={ratio:.2f}x")
            print(f"    vel[0:10]: {' '.join([f'{v:.2f}' for v in vel[:10]])}")
    except Exception as e:
        pass

print(f"\nResults: {spike_count}/{total_checked} have first-frame spike (>{3.0}x ratio)")
print(f"Spike rate: {spike_count/max(total_checked,1)*100:.1f}%")
