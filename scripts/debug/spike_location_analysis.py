#!/usr/bin/env python3
"""Check where velocity spikes occur - at segment boundaries?"""
import os, sys
import numpy as np

sys.path.insert(0, '/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer')

gen_dir = 'work_dirs/prism_1b_tp2m_multiframe_kt_spectral/eval_hml3d_rewritten'
gen_files = sorted(os.listdir(gen_dir))[:10]

print('=== Velocity spike location analysis ===')
for fname in gen_files:
    d = np.load(os.path.join(gen_dir, fname), allow_pickle=True)
    bp = d['body_pose']  # (T, 63)
    T = bp.shape[0]

    bp_joints = bp.reshape(T, 21, 3)
    # Per-frame max joint velocity
    frame_vel = np.linalg.norm(np.diff(bp_joints, axis=0), axis=2).max(axis=1)  # (T-1,)

    mean_vel = frame_vel.mean()
    # Find top 5 spike frames
    top5 = np.argsort(frame_vel)[-5:][::-1]

    # Also compute velocity at segment boundaries (every 128 frames?)
    seg_size = 128  # typical PRISM segment
    seg_boundaries = [i for i in range(seg_size, T-1, seg_size)]

    print(f'\n{fname}: T={T}, mean_vel={mean_vel:.4f}')
    print(f'  Top 5 spike frames: {[(int(f), f"{frame_vel[f]:.4f}") for f in top5]}')
    print(f'  Potential segment boundaries ({seg_size}-frame): {seg_boundaries}')
    if seg_boundaries:
        for sb in seg_boundaries:
            if sb < len(frame_vel):
                print(f'    Frame {sb}: vel={frame_vel[sb]:.4f} ({frame_vel[sb]/mean_vel:.1f}x mean)')

    # Check velocity distribution: first 10 frames vs rest
    if T > 20:
        first10_vel = frame_vel[:10].mean()
        rest_vel = frame_vel[10:].mean()
        print(f'  First 10 frames vel: {first10_vel:.4f} ({first10_vel/mean_vel:.2f}x mean)')
        print(f'  Rest frames vel: {rest_vel:.4f} ({rest_vel/mean_vel:.2f}x mean)')

    # Check if velocity increases over time (accumulating error?)
    if T > 50:
        quarter = len(frame_vel) // 4
        q1 = frame_vel[:quarter].mean()
        q2 = frame_vel[quarter:2*quarter].mean()
        q3 = frame_vel[2*quarter:3*quarter].mean()
        q4 = frame_vel[3*quarter:].mean()
        print(f'  Velocity by quarter: Q1={q1:.4f}, Q2={q2:.4f}, Q3={q3:.4f}, Q4={q4:.4f}')
