#!/usr/bin/env python3
"""Quick sanity check on generated NPZ files."""
import numpy as np
import glob
import sys

npz_dir = sys.argv[1] if len(sys.argv) > 1 else 'work_dirs/prism_1b_tp2m_multiframe_kt_spectral/eval_hml3d_rewritten'
files = sorted(glob.glob(f'{npz_dir}/*.npz'))
print(f'Total NPZ files: {len(files)}')

for f in files[:5]:
    data = np.load(f, allow_pickle=True)
    print(f'\n=== {f.split("/")[-1]} ===')
    for k in data.files:
        v = data[k]
        if v.dtype.kind == 'f':  # float
            print(f'  {k}: shape={v.shape} min={v.min():.4f} max={v.max():.4f} mean={v.mean():.4f} std={v.std():.4f}')
        else:
            print(f'  {k}: shape={v.shape} dtype={v.dtype} val={v}')

# Check body_pose range — axis-angle should be roughly in [-pi, pi]
print('\n\n=== QUALITY DIAGNOSTICS ===')
import math
for f in files[:10]:
    name = f.split('/')[-1]
    data = np.load(f, allow_pickle=True)

    issues = []

    if 'body_pose' in data.files:
        bp = data['body_pose']
        bp_abs_max = np.abs(bp).max()
        bp_std = bp.std()
        if bp_abs_max > 10.0:
            issues.append(f'body_pose abs_max={bp_abs_max:.2f} (>10, deformed!)')
        if bp_std < 0.01:
            issues.append(f'body_pose std={bp_std:.6f} (near-zero, static)')

    if 'transl' in data.files:
        tr = data['transl']
        tr_range = tr.max() - tr.min()
        if tr_range > 50.0:
            issues.append(f'transl range={tr_range:.2f} (>50m, exploding!)')
        if tr_range < 0.001:
            issues.append(f'transl range={tr_range:.6f} (no movement)')

    if 'global_orient' in data.files:
        go = data['global_orient']
        go_abs_max = np.abs(go).max()
        if go_abs_max > 10.0:
            issues.append(f'global_orient abs_max={go_abs_max:.2f} (>10, deformed!)')

    status = 'ISSUES: ' + '; '.join(issues) if issues else 'OK'
    print(f'  {name}: {status}')
