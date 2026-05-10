#!/usr/bin/env python3
"""Post-process E14 NPZ outputs to reduce foot skating via pelvis XZ correction.

Core idea: When a foot is detected on the ground (Y < threshold) but has
significant XZ velocity, it means the pelvis is translating in XZ without
the legs moving — classic foot skating. Fix: dampen pelvis XZ velocity
proportionally to how much the grounded foot is sliding.

Algorithm:
1. Run FK to get joint positions for all frames
2. For each frame in the generated region:
   a. Identify ground-contact feet (Y < threshold)
   b. Compute each grounded foot's XZ displacement from previous frame
   c. If displacement > slide threshold, compute a pelvis XZ correction
      that would pin the grounded foot in place
   d. Blend corrections from multiple grounded feet
3. Smooth corrections temporally to avoid jitter
4. Recompute FK to verify improvement

Usage:
    python3 scripts/postprocess_e14_antislide.py \
        --npz-dir work_dirs/.../uncond_local/E14_M/npz/ \
        --output-dir work_dirs/.../uncond_local/E14_M/npz_fixed/ \
        [--pids 00000,00020,00033,00040,00045,00048,00055,00073]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hftrainer.evaluation.motion.m2m_eval_metrics import motion135_to_positions_np


FOOT_JOINTS = [10, 11]       # L_Foot, R_Foot
ANKLE_JOINTS = [7, 8]        # L_Ankle, R_Ankle
ALL_FOOT = [7, 8, 10, 11]    # ankles + feet
GROUND_Y_THRESH = 0.08       # feet considered "on ground" if Y < this
SLIDE_SPEED_THRESH = 0.015   # XZ speed > this while on ground = skating
CORRECTION_STRENGTH = 0.85   # how much to correct (1.0 = full lock, <1 = partial)


def compute_foot_contact(positions, threshold=GROUND_Y_THRESH):
    """Return per-frame per-foot ground contact mask.

    Returns:
        contact: (T, 4) bool array for joints [7,8,10,11]
        min_foot_y: (T,) array of minimum foot Y per frame
    """
    T = positions.shape[0]
    contact = np.zeros((T, 4), dtype=bool)
    min_foot_y = np.zeros(T)
    for fi in range(T):
        for ji, j in enumerate(ALL_FOOT):
            y = positions[fi, j, 1]
            contact[fi, ji] = y < threshold
        min_foot_y[fi] = min(positions[fi, j, 1] for j in ALL_FOOT)
    return contact, min_foot_y


def compute_pelvis_xz_correction(positions, contact, gen_start, gen_end,
                                  strength=CORRECTION_STRENGTH):
    """Compute per-frame pelvis XZ correction to reduce foot sliding.

    For each frame where a foot is on the ground and sliding, compute
    how much to shift pelvis XZ to pin that foot's XZ position.

    Returns:
        corrections: (T, 2) array of (dx, dz) corrections for pelvis
    """
    T = positions.shape[0]
    corrections = np.zeros((T, 2), dtype=np.float32)

    for fi in range(max(gen_start, 1), min(gen_end, T)):
        # Find grounded feet
        grounded = []
        for ji, j in enumerate(ALL_FOOT):
            if contact[fi, ji]:
                # Compute XZ displacement from previous frame
                dx = positions[fi, j, 0] - positions[fi-1, j, 0]
                dz = positions[fi, j, 2] - positions[fi-1, j, 2]
                xz_speed = np.sqrt(dx*dx + dz*dz)
                if xz_speed > SLIDE_SPEED_THRESH:
                    # This foot is sliding — weight by how much
                    weight = min(1.0, xz_speed / (SLIDE_SPEED_THRESH * 3))
                    # Also weight by how close to ground (lower = more confident contact)
                    ground_weight = max(0, 1.0 - positions[fi, j, 1] / GROUND_Y_THRESH)
                    grounded.append((dx, dz, weight * ground_weight, j))

        if not grounded:
            continue

        # Weighted average of corrections from all grounded sliding feet
        total_w = sum(g[2] for g in grounded)
        if total_w > 0:
            avg_dx = sum(g[0] * g[2] for g in grounded) / total_w
            avg_dz = sum(g[1] * g[2] for g in grounded) / total_w
            corrections[fi, 0] = -avg_dx * strength
            corrections[fi, 1] = -avg_dz * strength

    return corrections


def apply_corrections_cumulative(motion_135, corrections, gen_start, gen_end,
                                  smooth_sigma=2.0):
    """Apply XZ corrections to pelvis translation cumulatively.

    corrections[fi] represents the instantaneous XZ shift for frame fi.
    We accumulate them so that frame fi's pelvis is shifted by sum of
    corrections[gen_start:fi+1].
    """
    from scipy.ndimage import gaussian_filter1d

    fixed = motion_135.copy()
    T = motion_135.shape[0]

    # Smooth instantaneous corrections before accumulating
    if smooth_sigma > 0:
        for dim in range(2):
            seg = corrections[gen_start:gen_end, dim]
            corrections[gen_start:gen_end, dim] = gaussian_filter1d(seg, sigma=smooth_sigma)

    # Accumulate corrections
    cum_x = np.cumsum(corrections[:, 0])
    cum_z = np.cumsum(corrections[:, 1])

    # Apply only to generated region (with smooth fade-in/out at boundaries)
    fade_len = 5
    for fi in range(gen_start, min(gen_end, T)):
        # Fade in at start
        if fi < gen_start + fade_len:
            alpha = (fi - gen_start) / fade_len
        # Fade out at end
        elif fi >= gen_end - fade_len:
            alpha = (gen_end - fi) / fade_len
        else:
            alpha = 1.0

        fixed[fi, 0] += cum_x[fi] * alpha  # trans_x
        fixed[fi, 2] += cum_z[fi] * alpha  # trans_z

    return fixed


def compute_sliding_metric(positions, gen_start, gen_end):
    """Compute foot skating ratio and avg speed for generated region."""
    gen_pos = positions[gen_start:gen_end]
    N = gen_pos.shape[0]
    bad = 0
    total = 0
    speeds = []
    for fi in range(1, N):
        for j in ALL_FOOT:
            if gen_pos[fi, j, 1] < GROUND_Y_THRESH:
                total += 1
                xz_v = np.linalg.norm(gen_pos[fi, j, [0, 2]] - gen_pos[fi-1, j, [0, 2]])
                if xz_v > SLIDE_SPEED_THRESH:
                    bad += 1
                    speeds.append(xz_v)
    ratio = bad / max(1, total)
    avg_speed = np.mean(speeds) if speeds else 0.0
    return ratio, avg_speed


def iterative_correction(motion_135, bone_offsets, gen_start, gen_end,
                          max_iters=3, target_ratio=0.15):
    """Iteratively apply corrections until skating is below target or max_iters reached."""
    current = motion_135.copy()

    for it in range(max_iters):
        positions = motion135_to_positions_np(current, bone_offsets)
        contact, _ = compute_foot_contact(positions)
        ratio, avg_speed = compute_sliding_metric(positions, gen_start, gen_end)

        if ratio < target_ratio:
            break

        # Compute corrections for this iteration
        corrections = compute_pelvis_xz_correction(
            positions, contact, gen_start, gen_end,
            strength=min(CORRECTION_STRENGTH, 0.5 + 0.2 * it))  # ramp up strength

        # Apply corrections
        current = apply_corrections_cumulative(
            current, corrections, gen_start, gen_end,
            smooth_sigma=max(1.5, 2.5 - 0.5 * it))  # decrease smoothing

    return current


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--pids', type=str, default=None,
                        help='Comma-separated prompt IDs to fix (default: all)')
    parser.add_argument('--bone-offsets', default='data/hymotion_m2m_data/bone_offsets_22.pt')
    parser.add_argument('--max-iters', type=int, default=3)
    parser.add_argument('--target-ratio', type=float, default=0.15)
    parser.add_argument('--dry-run', action='store_true',
                        help='Only compute metrics, do not save')
    args = parser.parse_args()

    bone_offsets = torch.load(args.bone_offsets, map_location='cpu').float().numpy()
    os.makedirs(args.output_dir, exist_ok=True)

    npz_files = sorted(f for f in os.listdir(args.npz_dir) if f.endswith('.npz'))
    if args.pids:
        target_pids = set(args.pids.split(','))
        npz_files = [f for f in npz_files if f.replace('.npz', '') in target_pids]

    print(f"{'pid':>5}  {'before':>10}  {'after':>10}  {'delta':>8}  {'iters':>5}")
    print('-' * 48)

    improved = 0
    worsened = 0
    unchanged = 0
    total_before = 0
    total_after = 0

    for fname in npz_files:
        path = os.path.join(args.npz_dir, fname)
        d = np.load(path, allow_pickle=True)

        # Get motion data
        if 'motion_135' in d.files:
            m135 = d['motion_135'].astype(np.float32)
        elif 'motion' in d.files:
            m135 = d['motion'][:, :135].astype(np.float32)
        else:
            print(f'  {fname}: no motion_135 key, skipping')
            continue

        # Get layout info
        layout = json.loads(bytes(d['layout_json']).decode()) if 'layout_json' in d.files else {}
        nc_a = layout.get('N_cond_a', 45)
        n_trans = layout.get('N_transition', 0)
        if n_trans == 0:
            continue
        gen_start = nc_a
        gen_end = nc_a + n_trans

        # Compute before metric
        pos_before = motion135_to_positions_np(m135, bone_offsets)
        ratio_before, speed_before = compute_sliding_metric(pos_before, gen_start, gen_end)

        # Skip if already good
        if ratio_before < 0.05:
            pid = fname.replace('.npz', '')
            print(f'{pid}  {ratio_before:10.1%}  {"skip":>10}  {"---":>8}  {"0":>5}')
            # Still copy to output
            if not args.dry_run:
                import shutil
                shutil.copy2(path, os.path.join(args.output_dir, fname))
            continue

        # Iterative correction
        fixed = iterative_correction(
            m135, bone_offsets, gen_start, gen_end,
            max_iters=args.max_iters, target_ratio=args.target_ratio)

        # Compute after metric
        pos_after = motion135_to_positions_np(fixed, bone_offsets)
        ratio_after, speed_after = compute_sliding_metric(pos_after, gen_start, gen_end)

        delta = ratio_after - ratio_before
        pid = fname.replace('.npz', '')
        marker = '✓' if delta < -0.02 else ('✗' if delta > 0.02 else '~')
        print(f'{pid}  {ratio_before:10.1%}  {ratio_after:10.1%}  {delta:+8.1%} {marker}')

        if delta < -0.02:
            improved += 1
        elif delta > 0.02:
            worsened += 1
        else:
            unchanged += 1
        total_before += ratio_before
        total_after += ratio_after

        # Save
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

            out_path = os.path.join(args.output_dir, fname)
            np.savez_compressed(out_path, **save_kw)

    n = improved + worsened + unchanged
    if n > 0:
        print(f'\nSummary: {n} cases processed')
        print(f'  Improved: {improved} ({improved/n:.0%})')
        print(f'  Worsened: {worsened} ({worsened/n:.0%})')
        print(f'  Unchanged: {unchanged} ({unchanged/n:.0%})')
        print(f'  Avg ratio: {total_before/n:.1%} -> {total_after/n:.1%}')

    if not args.dry_run:
        # Copy non-processed files
        all_npz = sorted(f for f in os.listdir(args.npz_dir) if f.endswith('.npz'))
        for fname in all_npz:
            out_path = os.path.join(args.output_dir, fname)
            if not os.path.exists(out_path):
                import shutil
                shutil.copy2(os.path.join(args.npz_dir, fname), out_path)
        print(f'\nFixed NPZs saved to {args.output_dir}')


if __name__ == '__main__':
    main()
