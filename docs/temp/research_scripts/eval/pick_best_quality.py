#!/usr/bin/env python3
"""Pick best NPZ from multiple seeds using a combined quality score.

Score = foot_skating_ratio + alpha * slide_mismatch
where slide_mismatch = pelvis_xz_displacement / leg_rotation_velocity
(high means body moves but legs don't)
"""
import argparse, json, os, shutil, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from hftrainer.evaluation.motion.m2m_eval_metrics import motion135_to_positions_np

ALL_FOOT = [7, 8, 10, 11]
LEG_JOINTS = [1, 2, 4, 5, 7, 8, 10, 11]
GROUND_Y = 0.08
SLIDE_THRESH = 0.015


def score_npz(m135, bone_offsets, gen_start, gen_end):
    """Return a quality score (lower = better)."""
    N_gen = gen_end - gen_start
    if N_gen < 2:
        return 0.0

    positions = motion135_to_positions_np(m135, bone_offsets)
    gen_pos = positions[gen_start:gen_end]
    N = gen_pos.shape[0]

    # 1. Foot skating ratio
    bad = total = 0
    for fi in range(1, N):
        for j in ALL_FOOT:
            if gen_pos[fi, j, 1] < GROUND_Y:
                total += 1
                xz_v = np.linalg.norm(gen_pos[fi, j, [0, 2]] - gen_pos[fi - 1, j, [0, 2]])
                if xz_v > SLIDE_THRESH:
                    bad += 1
    fs_ratio = bad / max(1, total)

    # 2. Pelvis XZ displacement vs leg rotation
    # Use MAX displacement from start (not endpoint) for loop tasks where endpoint=start
    pelvis_xz_from_start = np.array([np.linalg.norm(gen_pos[fi, 0, [0, 2]] - gen_pos[0, 0, [0, 2]]) for fi in range(N)])
    pelvis_xz = pelvis_xz_from_start.max()  # max XZ distance during motion
    # Also compute total pelvis XZ path length (captures back-and-forth sliding)
    pelvis_xz_path = sum(np.linalg.norm(gen_pos[fi, 0, [0, 2]] - gen_pos[fi-1, 0, [0, 2]]) for fi in range(1, N))

    rot6d = m135[gen_start:gen_end, 3:135]
    rot_vel = np.abs(np.diff(rot6d, axis=0))
    leg_rot_vel = sum(rot_vel[:, j * 6:(j + 1) * 6].sum() for j in LEG_JOINTS)
    all_rot_vel = rot_vel.sum()
    leg_ratio = leg_rot_vel / max(1.0, all_rot_vel)

    # Mismatch: high pelvis displacement with low leg activity
    if leg_rot_vel > 0.1:
        mismatch = pelvis_xz_path / (leg_rot_vel / N_gen)  # path-based, per-frame normalized
    else:
        mismatch = pelvis_xz_path * 100  # no leg motion at all = very bad

    # 3. Foot position total velocity (more foot movement = more natural)
    foot_vel = 0
    for j in ALL_FOOT:
        for fi in range(1, N):
            foot_vel += np.linalg.norm(gen_pos[fi, j] - gen_pos[fi - 1, j])
    foot_vel_per_frame = foot_vel / max(1, N - 1)

    # Combined score: lower is better
    # fs_ratio: 0-1 (foot skating)
    # mismatch: 0-inf (body slides without legs)
    # -foot_vel: penalize lack of movement
    score = fs_ratio + 0.3 * mismatch - 0.1 * foot_vel_per_frame
    return score, fs_ratio, mismatch, leg_ratio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed-dirs', nargs='+', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--bone-offsets', default='data/hymotion_m2m_data/bone_offsets_22.pt')
    args = parser.parse_args()

    bo = torch.load(args.bone_offsets, map_location='cpu', weights_only=True).float().numpy()
    os.makedirs(args.output_dir, exist_ok=True)

    seed_dirs = {}
    for kv in args.seed_dirs:
        k, v = kv.split('=', 1)
        seed_dirs[k] = v

    all_fnames = set()
    for d in seed_dirs.values():
        for f in os.listdir(d):
            if f.endswith('.npz'):
                all_fnames.add(f)

    print(f"{'pid':>5}  {'seed':>5}  {'score':>7}  {'fs':>6}  {'mism':>6}  {'leg%':>6}")
    print('-' * 45)

    total_score_old = total_score_new = 0
    n = 0

    for fname in sorted(all_fnames):
        pid = fname.replace('.npz', '')
        candidates = []

        for skey, sdir in seed_dirs.items():
            path = os.path.join(sdir, fname)
            if not os.path.exists(path):
                continue
            d = np.load(path, allow_pickle=True)
            if 'motion_135' not in d.files or 'layout_json' not in d.files:
                continue
            m135 = d['motion_135'].astype(np.float32)
            layout = json.loads(bytes(d['layout_json']).decode())
            # Determine generated region based on task type
            if 'N_transition' in layout and layout['N_transition'] > 0:
                nc_a = layout.get('N_cond_a', 45)
                n_trans = layout['N_transition']
                gen_start = nc_a
                gen_end = nc_a + n_trans
            elif 'T_gt_eff' in layout:
                # E8-D loop task: entire motion is generated (with append)
                # Score the whole motion
                gen_start = 0
                gen_end = m135.shape[0]
            else:
                continue
            result = score_npz(m135, bo, gen_start, gen_end)
            candidates.append((skey, path, result))

        if not candidates:
            continue

        candidates.sort(key=lambda x: x[2][0])  # sort by combined score
        best_skey, best_path, (best_score, fs, mism, leg) = candidates[0]
        worst_score = candidates[-1][2][0]

        shutil.copy2(best_path, os.path.join(args.output_dir, fname))

        total_score_new += best_score
        total_score_old += candidates[0][2][0]  # s0 is always first seed
        n += 1

        print(f'{pid}  {best_skey:>5}  {best_score:7.3f}  {fs:5.1%}  {mism:6.2f}  {leg:5.1%}')

    print(f'\n{n} cases, avg score: {total_score_new / max(1, n):.3f}')

    # Distribution
    seed_counts = {}
    for fname in sorted(all_fnames):
        path = os.path.join(args.output_dir, fname)
        if os.path.exists(path):
            # Check which seed it came from by file size/content match
            pass
    print(f'\nOutput: {args.output_dir}')


if __name__ == '__main__':
    main()
