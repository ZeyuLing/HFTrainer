#!/usr/bin/env python3
"""Compare two ProtoMotions cache files for jitter metrics."""
import sys
import torch
import numpy as np

def to_np(x):
    if hasattr(x, 'numpy'):
        return x.numpy()
    return np.asarray(x)

def stats(cache, name):
    dof_vel = to_np(cache['dof_vel'])
    body_vel = to_np(cache['body_vel'])
    body_pos = to_np(cache['body_pos'])
    dt = float(cache['control_dt'])

    dof_accel = np.diff(dof_vel, axis=0) / dt
    body_accel = np.diff(body_vel, axis=0) / dt

    root_z = body_pos[:, 0, 2]
    root_z_drop = root_z[:5].max() - root_z[:5].min()

    print(f'=== {name} ===')
    print(f'  DOF vel max:     {np.abs(dof_vel).max():.2f} rad/s')
    print(f'  DOF vel mean:    {np.abs(dof_vel).mean():.3f} rad/s')
    print(f'  DOF accel max:   {np.abs(dof_accel).max():.1f} rad/s2')
    print(f'  DOF accel mean:  {np.abs(dof_accel).mean():.2f} rad/s2')
    print(f'  Body vel max:    {np.abs(body_vel).max():.2f} m/s')
    print(f'  Body vel mean:   {np.abs(body_vel).mean():.3f} m/s')
    print(f'  Body accel max:  {np.abs(body_accel).max():.1f} m/s2')
    print(f'  Body accel mean: {np.abs(body_accel).mean():.2f} m/s2')
    print(f'  Root Z range:    [{root_z.min():.4f}, {root_z.max():.4f}]')
    print(f'  Root Z std:      {root_z.std():.4f}')
    print(f'  Root Z first-5-frame drop: {root_z_drop:.4f}m')
    print()
    return {
        'dof_vel_max': np.abs(dof_vel).max(),
        'dof_accel_max': np.abs(dof_accel).max(),
        'body_vel_max': np.abs(body_vel).max(),
        'body_accel_max': np.abs(body_accel).max(),
    }

old_path = sys.argv[1]
new_path = sys.argv[2]

old = torch.load(old_path, map_location='cpu', weights_only=False)
new = torch.load(new_path, map_location='cpu', weights_only=False)

s_old = stats(old, 'OLD (perframe FK, no smooth)')
s_new = stats(new, 'NEW (global FK, smoothed)')

print('=== IMPROVEMENT RATIOS ===')
for k in ['dof_vel_max', 'dof_accel_max', 'body_vel_max', 'body_accel_max']:
    ratio = s_old[k] / s_new[k] if s_new[k] > 0 else float('inf')
    print(f'  {k}: {s_old[k]:.2f} -> {s_new[k]:.2f} ({ratio:.2f}x reduction)')
