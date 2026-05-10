#!/usr/bin/env python3
"""Post-process E14: foot-pinning approach.

For each frame where a foot is grounded, compute where the pelvis should be
so that foot stays at its position from when it first contacted the ground.

Strategy:
1. Identify foot contact events: consecutive frames where foot Y < threshold
2. For each contact event, the foot's XZ should stay at its position at contact start
3. Compute what pelvis XZ offset would achieve this (based on FK relationship)
4. Since pelvis XZ maps 1:1 to all joints (it's just a global offset),
   the correction = -(foot_xz[fi] - foot_xz[contact_start])
5. Average corrections from all grounded feet
6. Apply with smooth blending at contact start/end
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


def detect_contact_events(positions, gen_start, gen_end):
    """Detect foot contact events in the generated region.

    Returns list of (joint_idx, start_frame, end_frame, anchor_xz).
    """
    T = positions.shape[0]
    events = []

    for j in ALL_FOOT:
        in_contact = False
        contact_start = -1
        anchor_xz = None

        for fi in range(gen_start, min(gen_end, T)):
            y = positions[fi, j, 1]
            if y < GROUND_Y_THRESH:
                if not in_contact:
                    in_contact = True
                    contact_start = fi
                    anchor_xz = positions[fi, j, [0, 2]].copy()
            else:
                if in_contact:
                    # End of contact event
                    if fi - contact_start >= 3:  # at least 3 frames
                        events.append((j, contact_start, fi, anchor_xz))
                    in_contact = False

        # Handle contact that extends to end
        if in_contact and gen_end - contact_start >= 3:
            events.append((j, contact_start, min(gen_end, T), anchor_xz))

    return events


def foot_pinning_fix(motion_135, bone_offsets, gen_start, gen_end, max_iters=3):
    """Pin grounded feet by adjusting pelvis XZ."""
    from scipy.ndimage import gaussian_filter1d

    current = motion_135.copy()
    T = motion_135.shape[0]

    for iteration in range(max_iters):
        positions = motion135_to_positions_np(current, bone_offsets)
        ratio = compute_sliding_metric(positions, gen_start, gen_end)
        if ratio < 0.15:
            break

        events = detect_contact_events(positions, gen_start, gen_end)
        if not events:
            break

        # Compute per-frame correction from all contact events
        corrections_x = np.zeros(T, dtype=np.float32)
        corrections_z = np.zeros(T, dtype=np.float32)
        weights = np.zeros(T, dtype=np.float32)

        for j, cs, ce, anchor_xz in events:
            for fi in range(cs, ce):
                # How much is this foot sliding from its anchor?
                curr_xz = positions[fi, j, [0, 2]]
                slide_xz = curr_xz - anchor_xz
                slide_mag = np.linalg.norm(slide_xz)

                if slide_mag > SLIDE_SPEED_THRESH:
                    # Weight by confidence: higher for lower foot, stronger for more slide
                    y = positions[fi, j, 1]
                    w = max(0, 1.0 - y / GROUND_Y_THRESH) * min(1.0, slide_mag / 0.05)

                    # Fade in/out at contact boundaries
                    fade_len = 3
                    if fi < cs + fade_len:
                        w *= (fi - cs) / fade_len
                    if fi > ce - fade_len:
                        w *= (ce - fi) / fade_len

                    corrections_x[fi] -= slide_xz[0] * w
                    corrections_z[fi] -= slide_xz[1] * w
                    weights[fi] += w

        # Normalize by total weight
        mask = weights > 0
        corrections_x[mask] /= weights[mask]
        corrections_z[mask] /= weights[mask]

        # Limit max correction per frame to prevent jumping
        MAX_CORR = 0.03  # 3cm per frame max
        mag = np.sqrt(corrections_x**2 + corrections_z**2)
        too_large = mag > MAX_CORR
        if too_large.any():
            scale = MAX_CORR / (mag[too_large] + 1e-8)
            corrections_x[too_large] *= scale
            corrections_z[too_large] *= scale

        # Smooth corrections
        sigma = 3.0
        corrections_x[gen_start:gen_end] = gaussian_filter1d(
            corrections_x[gen_start:gen_end], sigma=sigma)
        corrections_z[gen_start:gen_end] = gaussian_filter1d(
            corrections_z[gen_start:gen_end], sigma=sigma)

        # Apply corrections to pelvis XZ (column 0=X, 2=Z in translation)
        current[gen_start:gen_end, 0] += corrections_x[gen_start:gen_end]
        current[gen_start:gen_end, 2] += corrections_z[gen_start:gen_end]

    return current


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--pids', type=str, default=None)
    parser.add_argument('--bone-offsets', default='data/hymotion_m2m_data/bone_offsets_22.pt')
    parser.add_argument('--max-iters', type=int, default=3)
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

    stats = {'improved': 0, 'worsened': 0, 'total_b': 0.0, 'total_a': 0.0, 'n': 0}

    for fname in npz_files:
        path = os.path.join(args.npz_dir, fname)
        d = np.load(path, allow_pickle=True)
        if 'motion_135' not in d.files:
            continue
        m135 = d['motion_135'].astype(np.float32)
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
            continue

        fixed = foot_pinning_fix(m135, bone_offsets, gen_start, gen_end,
                                  max_iters=args.max_iters)
        pos_after = motion135_to_positions_np(fixed, bone_offsets)
        ratio_after = compute_sliding_metric(pos_after, gen_start, gen_end)

        delta = ratio_after - ratio_before
        pid = fname.replace('.npz', '')
        marker = '✓' if delta < -0.02 else ('✗' if delta > 0.02 else '~')
        print(f'{pid}  {ratio_before:10.1%}  {ratio_after:10.1%}  {delta:+8.1%} {marker}')

        stats['n'] += 1
        stats['total_b'] += ratio_before
        stats['total_a'] += ratio_after
        if delta < -0.02:
            stats['improved'] += 1
        elif delta > 0.02:
            stats['worsened'] += 1

        if not args.dry_run:
            from hftrainer.pipelines.motion.differentiable_fk import motion135_to_fk
            fixed_t = torch.from_numpy(fixed).float()
            wp, _, _, _ = motion135_to_fk(fixed_t, torch.from_numpy(bone_offsets), 'local')
            save_kw = dict(motion_135=fixed, positions=wp.numpy(), translation=fixed[:, :3])
            if 'layout_json' in d.files:
                save_kw['layout_json'] = d['layout_json']
            np.savez_compressed(os.path.join(args.output_dir, fname), **save_kw)

    n = stats['n']
    if n > 0:
        print(f'\nSummary: {n} cases')
        print(f'  Improved: {stats["improved"]}  Worsened: {stats["worsened"]}')
        print(f'  Avg: {stats["total_b"]/n:.1%} -> {stats["total_a"]/n:.1%}')


if __name__ == '__main__':
    main()
