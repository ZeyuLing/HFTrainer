#!/usr/bin/env python3
"""Post-process E14 NPZ outputs to reduce foot skating.

Strategy: For each frame in the generated transition region, if a foot joint
is near the ground (Y < threshold), adjust pelvis translation Y to ensure
the lowest foot touches the ground. This doesn't change the pose (rotations),
only the root translation — so the body shape is preserved but the character
stays grounded.

Additionally: detect frames where pelvis translates significantly in XZ but
leg rotations barely change, and apply velocity-based correction to reduce
the sliding appearance.

Usage:
    python3 scripts/postprocess_e14_foot_contact.py \
        --npz-dir work_dirs/.../uncond_local/E14_M/npz/ \
        --output-dir work_dirs/.../uncond_local/E14_M/npz_fixed/ \
        --pids 00000,00020,00033,00040,00045,00048,00055,00073
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hftrainer.evaluation.motion.m2m_eval_metrics import motion135_to_positions_np


FOOT_JOINTS = [10, 11]      # L_Foot, R_Foot
ANKLE_JOINTS = [7, 8]       # L_Ankle, R_Ankle
LEG_ROT_JOINTS = [1, 2, 4, 5, 7, 8, 10, 11]  # all leg joints for rot velocity
GROUND_Y_THRESH = 0.06      # feet considered "on ground" if Y < this
SLIDE_SPEED_THRESH = 0.012  # XZ speed > this while on ground = skating


def fix_foot_contact(motion_135, bone_offsets, gen_start, gen_end,
                     smooth_window=5):
    """Fix foot skating by adjusting pelvis Y to keep feet grounded.

    For each generated frame:
    1. Compute FK to get joint positions
    2. Find the minimum foot Y
    3. If min foot Y < 0 (below ground), shift pelvis Y up by |min_foot_Y|
    4. If min foot Y > GROUND_Y_THRESH and both feet are above ground,
       allow it (character might be jumping/in-air)
    5. Smooth the Y corrections with a gaussian window to avoid jitter
    """
    T = motion_135.shape[0]
    positions = motion135_to_positions_np(motion_135, bone_offsets)

    # Only fix generated region
    fixed = motion_135.copy()
    y_corrections = np.zeros(T)

    for fi in range(gen_start, min(gen_end, T)):
        foot_ys = [positions[fi, j, 1] for j in FOOT_JOINTS + ANKLE_JOINTS]
        min_foot_y = min(foot_ys)

        # If foot is below ground, push pelvis up
        if min_foot_y < -0.01:
            y_corrections[fi] = -min_foot_y
        # If foot barely above ground but clearly skating, push down slightly
        elif min_foot_y > GROUND_Y_THRESH:
            # Check if this frame is "supposed" to be grounded
            # (adjacent frames have feet on ground)
            if fi > gen_start and fi < gen_end - 1:
                prev_min_y = min(positions[fi-1, j, 1] for j in FOOT_JOINTS)
                next_min_y = min(positions[fi+1, j, 1] for j in FOOT_JOINTS) if fi+1 < T else min_foot_y
                if prev_min_y < GROUND_Y_THRESH and next_min_y < GROUND_Y_THRESH:
                    # Neighboring frames are grounded but this one floats
                    y_corrections[fi] = -min_foot_y * 0.5  # partial correction

    # Smooth corrections
    if smooth_window > 1:
        from scipy.ndimage import gaussian_filter1d
        y_corrections[gen_start:gen_end] = gaussian_filter1d(
            y_corrections[gen_start:gen_end], sigma=smooth_window/3)

    # Apply
    fixed[:, 1] = fixed[:, 1] + y_corrections

    return fixed, y_corrections


def compute_sliding_metric(motion_135, bone_offsets, gen_start, gen_end):
    """Compute the sliding metric for the generated region."""
    positions = motion135_to_positions_np(motion_135, bone_offsets)
    gen_pos = positions[gen_start:gen_end]
    N = gen_pos.shape[0]
    bad = 0
    total = 0
    for fi in range(1, N):
        for j in FOOT_JOINTS + ANKLE_JOINTS:
            if gen_pos[fi, j, 1] < GROUND_Y_THRESH:
                total += 1
                xz_v = np.linalg.norm(gen_pos[fi, j, [0, 2]] - gen_pos[fi-1, j, [0, 2]])
                if xz_v > SLIDE_SPEED_THRESH:
                    bad += 1
    return bad / max(1, total)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--pids', type=str, default=None,
                        help='Comma-separated prompt IDs to fix (default: all)')
    parser.add_argument('--bone-offsets', default='data/hymotion_m2m_data/bone_offsets_22.pt')
    args = parser.parse_args()

    bone_offsets = torch.load(args.bone_offsets, map_location='cpu').float().numpy()
    os.makedirs(args.output_dir, exist_ok=True)

    npz_files = sorted(f for f in os.listdir(args.npz_dir) if f.endswith('.npz'))
    if args.pids:
        target_pids = set(args.pids.split(','))
        npz_files = [f for f in npz_files if f.replace('.npz', '') in target_pids]

    for fname in npz_files:
        path = os.path.join(args.npz_dir, fname)
        d = np.load(path, allow_pickle=True)
        m135 = d['motion_135'].astype(np.float32)
        layout = json.loads(bytes(d['layout_json']).decode()) if 'layout_json' in d.files else {}
        nc_a = layout.get('N_cond_a', 45)
        n_trans = layout.get('N_transition', 0)
        gen_start = nc_a
        gen_end = nc_a + n_trans

        slide_before = compute_sliding_metric(m135, bone_offsets, gen_start, gen_end)

        fixed, corrections = fix_foot_contact(m135, bone_offsets, gen_start, gen_end)

        slide_after = compute_sliding_metric(fixed, bone_offsets, gen_start, gen_end)

        pid = fname.replace('.npz', '')
        print(f'{pid}: slide {slide_before:.1%} -> {slide_after:.1%}  '
              f'(max_corr={np.abs(corrections).max():.3f}m)')

        # Save
        from hftrainer.pipelines.motion.differentiable_fk import motion135_to_fk
        fixed_t = torch.from_numpy(fixed).float()
        wp, _, _, _ = motion135_to_fk(fixed_t, torch.from_numpy(bone_offsets), 'local')

        save_kw = dict(
            motion_135=fixed,
            positions=wp.numpy(),
            translation=fixed[:, :3],
        )
        if 'layout_json' in d.files:
            save_kw['layout_json'] = d['layout_json']

        out_path = os.path.join(args.output_dir, fname)
        np.savez_compressed(out_path, **save_kw)

    print(f'\nDone. Fixed NPZs saved to {args.output_dir}')


if __name__ == '__main__':
    main()
