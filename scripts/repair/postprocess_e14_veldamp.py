#!/usr/bin/env python3
"""Post-process E14 NPZ outputs: velocity-damping approach to reduce foot skating.

When a foot is on the ground and sliding, dampen the pelvis XZ velocity
for that frame. This works on velocities (not cumulative positions), so
each frame is corrected independently — preventing the drift issue.

Algorithm:
1. Compute FK positions for all frames
2. For each frame in generated region:
   a. Check if any foot is grounded (Y < threshold) AND sliding (XZ velocity > threshold)
   b. If so, compute a damping factor: reduce pelvis XZ velocity proportionally
      to how much the foot is sliding
   c. Apply: new_trans_xz[fi] = trans_xz[fi-1] + damped_velocity
3. Smooth the result to avoid jitter
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hftrainer.evaluation.motion.m2m_eval_metrics import motion135_to_positions_np


ALL_FOOT = [7, 8, 10, 11]
GROUND_Y_THRESH = 0.08
SLIDE_SPEED_THRESH = 0.015
MAX_DAMPING = 0.9  # maximum velocity damping (0=no change, 1=full stop)


def compute_sliding_metric(positions, gen_start, gen_end):
    gen_pos = positions[gen_start:gen_end]
    N = gen_pos.shape[0]
    bad = 0
    total = 0
    for fi in range(1, N):
        for j in ALL_FOOT:
            if gen_pos[fi, j, 1] < GROUND_Y_THRESH:
                total += 1
                xz_v = np.linalg.norm(gen_pos[fi, j, [0, 2]] - gen_pos[fi-1, j, [0, 2]])
                if xz_v > SLIDE_SPEED_THRESH:
                    bad += 1
    return bad / max(1, total)


def velocity_damping_fix(motion_135, bone_offsets, gen_start, gen_end,
                          max_iters=3, target_ratio=0.15):
    """Iteratively dampen pelvis XZ velocity when feet are sliding."""
    from scipy.ndimage import gaussian_filter1d

    current = motion_135.copy()
    T = motion_135.shape[0]

    for iteration in range(max_iters):
        positions = motion135_to_positions_np(current, bone_offsets)
        ratio = compute_sliding_metric(positions, gen_start, gen_end)
        if ratio < target_ratio:
            break

        # Compute per-frame damping factor
        damping = np.zeros(T, dtype=np.float32)  # 0 = no damping, 1 = full stop

        for fi in range(max(gen_start, 1), min(gen_end, T)):
            max_slide = 0.0
            num_grounded = 0
            for j in ALL_FOOT:
                y = positions[fi, j, 1]
                if y < GROUND_Y_THRESH:
                    num_grounded += 1
                    xz_v = np.linalg.norm(
                        positions[fi, j, [0, 2]] - positions[fi-1, j, [0, 2]])
                    if xz_v > SLIDE_SPEED_THRESH:
                        # Slide amount relative to threshold
                        slide_excess = (xz_v - SLIDE_SPEED_THRESH) / SLIDE_SPEED_THRESH
                        # Weigh by ground proximity (closer to ground = more confident)
                        ground_confidence = max(0, 1.0 - y / GROUND_Y_THRESH)
                        max_slide = max(max_slide, slide_excess * ground_confidence)

            if max_slide > 0 and num_grounded > 0:
                # Damping proportional to slide severity, capped at MAX_DAMPING
                damping[fi] = min(MAX_DAMPING, max_slide * 0.5)

        # Smooth damping to prevent jitter
        damping[gen_start:gen_end] = gaussian_filter1d(
            damping[gen_start:gen_end], sigma=2.0)

        # Apply damping to pelvis XZ velocity
        fixed = current.copy()
        for fi in range(max(gen_start, 1), min(gen_end, T)):
            if damping[fi] > 0.01:
                # Current pelvis XZ velocity
                vel_x = current[fi, 0] - current[fi-1, 0]
                vel_z = current[fi, 2] - current[fi-1, 2]
                # Damped velocity
                new_vel_x = vel_x * (1 - damping[fi])
                new_vel_z = vel_z * (1 - damping[fi])
                # Apply: reconstruct position from damped velocity
                fixed[fi, 0] = fixed[fi-1, 0] + new_vel_x
                fixed[fi, 2] = fixed[fi-1, 2] + new_vel_z

        # Smooth transition at gen_end boundary
        # The damped result may drift from the original endpoint
        # Use a linear blend over the last few frames to match
        blend_len = min(10, gen_end - gen_start)
        if gen_end < T:
            # Compute the drift at gen_end
            drift_x = current[gen_end-1, 0] - fixed[gen_end-1, 0]
            drift_z = current[gen_end-1, 2] - fixed[gen_end-1, 2]
            # Linearly distribute this drift correction across blend_len frames at the end
            for bi in range(blend_len):
                fi = gen_end - blend_len + bi
                alpha = bi / blend_len  # 0 at start of blend, 1 at end
                fixed[fi, 0] += drift_x * alpha
                fixed[fi, 2] += drift_z * alpha

        current = fixed

    return current


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--pids', type=str, default=None)
    parser.add_argument('--bone-offsets', default='data/hymotion_m2m_data/bone_offsets_22.pt')
    parser.add_argument('--max-iters', type=int, default=3)
    parser.add_argument('--target-ratio', type=float, default=0.15)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    bone_offsets = torch.load(args.bone_offsets, map_location='cpu').float().numpy()
    if not args.dry_run:
        os.makedirs(args.output_dir, exist_ok=True)

    npz_files = sorted(f for f in os.listdir(args.npz_dir) if f.endswith('.npz'))
    if args.pids:
        target_pids = set(args.pids.split(','))
        npz_files = [f for f in npz_files if f.replace('.npz', '') in target_pids]

    print(f"{'pid':>5}  {'before':>10}  {'after':>10}  {'delta':>8}")
    print('-' * 40)

    improved = 0
    worsened = 0
    total_before = 0.0
    total_after = 0.0
    n_processed = 0

    for fname in npz_files:
        path = os.path.join(args.npz_dir, fname)
        d = np.load(path, allow_pickle=True)

        if 'motion_135' in d.files:
            m135 = d['motion_135'].astype(np.float32)
        else:
            continue

        layout = json.loads(bytes(d['layout_json']).decode()) if 'layout_json' in d.files else {}
        nc_a = layout.get('N_cond_a', 45)
        n_trans = layout.get('N_transition', 0)
        if n_trans == 0:
            continue
        gen_start = nc_a
        gen_end = nc_a + n_trans

        pos_before = motion135_to_positions_np(m135, bone_offsets)
        ratio_before = compute_sliding_metric(pos_before, gen_start, gen_end)

        if ratio_before < 0.05:
            pid = fname.replace('.npz', '')
            print(f'{pid}  {ratio_before:10.1%}  {"skip":>10}  {"---":>8}')
            if not args.dry_run:
                import shutil
                shutil.copy2(path, os.path.join(args.output_dir, fname))
            continue

        fixed = velocity_damping_fix(
            m135, bone_offsets, gen_start, gen_end,
            max_iters=args.max_iters, target_ratio=args.target_ratio)

        pos_after = motion135_to_positions_np(fixed, bone_offsets)
        ratio_after = compute_sliding_metric(pos_after, gen_start, gen_end)

        delta = ratio_after - ratio_before
        pid = fname.replace('.npz', '')
        marker = '✓' if delta < -0.02 else ('✗' if delta > 0.02 else '~')
        print(f'{pid}  {ratio_before:10.1%}  {ratio_after:10.1%}  {delta:+8.1%} {marker}')

        if delta < -0.02:
            improved += 1
        elif delta > 0.02:
            worsened += 1
        total_before += ratio_before
        total_after += ratio_after
        n_processed += 1

        if not args.dry_run:
            from hftrainer.pipelines.motion.differentiable_fk import motion135_to_fk
            fixed_t = torch.from_numpy(fixed).float()
            wp, _, _, _ = motion135_to_fk(
                fixed_t, torch.from_numpy(bone_offsets), 'local')
            save_kw = dict(
                motion_135=fixed,
                positions=wp.numpy(),
                translation=fixed[:, :3],
            )
            if 'layout_json' in d.files:
                save_kw['layout_json'] = d['layout_json']
            np.savez_compressed(os.path.join(args.output_dir, fname), **save_kw)

    if n_processed > 0:
        print(f'\nSummary: {n_processed} cases')
        print(f'  Improved: {improved}  Worsened: {worsened}')
        print(f'  Avg: {total_before/n_processed:.1%} -> {total_after/n_processed:.1%}')

    if not args.dry_run:
        all_npz = sorted(f for f in os.listdir(args.npz_dir) if f.endswith('.npz'))
        for fname in all_npz:
            out_path = os.path.join(args.output_dir, fname)
            if not os.path.exists(out_path):
                import shutil
                shutil.copy2(os.path.join(args.npz_dir, fname), out_path)
        print(f'\nSaved to {args.output_dir}')


if __name__ == '__main__':
    main()
