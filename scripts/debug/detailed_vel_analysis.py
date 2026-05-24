#!/usr/bin/env python3
"""Detailed per-joint analysis of generated vs GT motions."""
import json, os, sys
import numpy as np

sys.path.insert(0, '/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer')

gen_dir = 'work_dirs/prism_1b_tp2m_multiframe_kt_spectral/eval_hml3d_rewritten'
gen_files = sorted(os.listdir(gen_dir))[:5]

print('=== Generated motions - detailed analysis ===')
for fname in gen_files:
    d = np.load(os.path.join(gen_dir, fname), allow_pickle=True)
    bp = d['body_pose']
    go = d['global_orient']
    tr = d['transl']
    T = bp.shape[0]

    bp_joints = bp.reshape(T, 21, 3)
    per_joint_vel = np.linalg.norm(np.diff(bp_joints, axis=0), axis=2)  # (T-1, 21)
    max_frame_vel = per_joint_vel.max(axis=1)
    aa_mag = np.linalg.norm(bp_joints, axis=2)

    print(f'\n{fname}:')
    print(f'  frames={T}')
    print(f'  body_pose range: [{bp.min():.3f}, {bp.max():.3f}]')
    print(f'  Per-joint vel: mean={per_joint_vel.mean():.4f}, max={per_joint_vel.max():.4f}')
    print(f'  Frame with max vel: {np.argmax(max_frame_vel)}, val={max_frame_vel.max():.4f}')
    print(f'  Axis-angle magnitude: mean={aa_mag.mean():.4f}, max={aa_mag.max():.4f}')
    print(f'  Global orient vel={np.linalg.norm(np.diff(go, axis=0), axis=1).mean():.5f}')
    print(f'  Transl vel={np.linalg.norm(np.diff(tr, axis=0), axis=1).mean():.5f}')

    mean_vel = per_joint_vel.mean()
    spike_frames = np.where(max_frame_vel > 3 * mean_vel)[0]
    print(f'  Spike frames (vel > 3x mean): {len(spike_frames)}/{T-1}')

    # Check temporal autocorrelation (should be high for smooth motion)
    # Use first joint as proxy
    j0_vel = per_joint_vel[:, 0]
    if len(j0_vel) > 10:
        autocorr = np.corrcoef(j0_vel[:-1], j0_vel[1:])[0, 1]
        print(f'  Temporal autocorrelation (joint 0 vel): {autocorr:.4f}')

# Now compare with GT
print('\n\n=== GT motions - same analysis ===')
data = json.load(open('data/motionhub/train.json'))
dl = data.get('data_list', {})
keys = list(dl.keys())[:200]
count = 0
for k in keys:
    info = dl[k]
    path = info.get('smplx_path', '')
    if not path:
        continue
    fp = os.path.join('data/motionhub', path)
    if not os.path.isfile(fp):
        continue
    d = np.load(fp, allow_pickle=True)
    if 'poses' not in d or d['poses'].shape[0] <= 10:
        continue
    poses = d['poses']
    T = poses.shape[0]
    go = poses[:, :3]
    bp = poses[:, 3:66]

    bp_joints = bp.reshape(T, 21, 3)
    per_joint_vel = np.linalg.norm(np.diff(bp_joints, axis=0), axis=2)
    max_frame_vel = per_joint_vel.max(axis=1)
    aa_mag = np.linalg.norm(bp_joints, axis=2)

    print(f'\n{k}:')
    print(f'  frames={T}')
    print(f'  body_pose range: [{bp.min():.3f}, {bp.max():.3f}]')
    print(f'  Per-joint vel: mean={per_joint_vel.mean():.4f}, max={per_joint_vel.max():.4f}')
    print(f'  Axis-angle magnitude: mean={aa_mag.mean():.4f}, max={aa_mag.max():.4f}')

    mean_vel = per_joint_vel.mean()
    spike_frames = np.where(max_frame_vel > 3 * mean_vel)[0]
    print(f'  Spike frames (vel > 3x mean): {len(spike_frames)}/{T-1}')

    j0_vel = per_joint_vel[:, 0]
    if len(j0_vel) > 10:
        autocorr = np.corrcoef(j0_vel[:-1], j0_vel[1:])[0, 1]
        print(f'  Temporal autocorrelation (joint 0 vel): {autocorr:.4f}')

    count += 1
    if count >= 5:
        break
