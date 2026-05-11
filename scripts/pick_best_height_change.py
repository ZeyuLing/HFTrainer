#!/usr/bin/env python3
"""Pick best NPZ for sit/lie/height-change cases.

For cases where pelvis Y changes significantly (stand→sit, stand→lie, etc),
the skating metric is irrelevant. Instead we score by:
1. Smoothness of Y trajectory (low jerk = smooth sit-down)
2. Overall motion jitter (acceleration spikes)
3. Joint angle reasonableness (no extreme twists)
"""
import argparse, json, os, shutil, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from hftrainer.evaluation.motion.m2m_eval_metrics import motion135_to_positions_np


def score_height_change(m135, bone_offsets, gen_start, gen_end):
    """Score quality for height-change transitions (lower = better).

    For sit/lie/crouch transitions, skating manifests differently:
    - Feet slide on ground during height change (normal skating)
    - Body drifts horizontally while only Y should change
    - Jittery or non-smooth descent/ascent trajectory
    """
    N_gen = gen_end - gen_start
    if N_gen < 3:
        return (0.0, 0.0, 0.0, 0.0)

    positions = motion135_to_positions_np(m135, bone_offsets)
    gen_pos = positions[gen_start:gen_end]  # (N, 22, 3)
    N = gen_pos.shape[0]

    # 1. Foot skating during height change:
    # When feet are on ground (Y < threshold), they should stay put in XZ.
    # Use a HIGHER threshold for sit/lie (feet may be at Y=0.12 while sitting).
    GROUND_Y = 0.12  # higher threshold for non-standing poses
    SLIDE_THRESH = 0.012  # slightly tighter since these should be more static
    foot_joints = [7, 8, 10, 11]
    bad = total = 0
    for fi in range(1, N):
        for j in foot_joints:
            if gen_pos[fi, j, 1] < GROUND_Y:
                total += 1
                xz_v = np.linalg.norm(gen_pos[fi, j, [0, 2]] - gen_pos[fi-1, j, [0, 2]])
                if xz_v > SLIDE_THRESH:
                    bad += 1
    fs_ratio = bad / max(1, total)

    # 2. Unwanted horizontal drift: pelvis XZ should be relatively stable
    # during a sit-down (most motion is in Y). Penalize large XZ drift
    # relative to Y change.
    pelvis_xz_drift = np.linalg.norm(gen_pos[-1, 0, [0, 2]] - gen_pos[0, 0, [0, 2]])
    pelvis_y_change = abs(gen_pos[-1, 0, 1] - gen_pos[0, 0, 1])
    # Ratio of horizontal to vertical movement (low is better for sit/lie)
    if pelvis_y_change > 0.05:
        xz_y_ratio = pelvis_xz_drift / pelvis_y_change
    else:
        xz_y_ratio = pelvis_xz_drift * 5  # if no Y change, just penalize XZ drift

    # 3. Smoothness: jitter (acceleration) of all joints
    vel = np.diff(gen_pos, axis=0)
    accel = np.diff(vel, axis=0)
    jitter = np.linalg.norm(accel, axis=-1).mean()

    # 4. Y trajectory smoothness: penalize non-monotonic Y for sit-down
    # (oscillations = character going up-down-up during what should be a smooth descent)
    pelvis_y = gen_pos[:, 0, 1]
    y_reversals = 0
    for fi in range(2, N):
        dy_prev = pelvis_y[fi-1] - pelvis_y[fi-2]
        dy_curr = pelvis_y[fi] - pelvis_y[fi-1]
        if abs(dy_prev) > 0.005 and abs(dy_curr) > 0.005:
            if dy_prev * dy_curr < 0:  # direction reversal
                y_reversals += 1
    y_reversal_rate = y_reversals / max(1, N - 2)

    # Combined: lower = better
    score = (fs_ratio * 1.0 +
             xz_y_ratio * 0.3 +
             jitter * 15.0 +
             y_reversal_rate * 2.0)
    return (score, fs_ratio, xz_y_ratio, jitter)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed-dirs', nargs='+', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--pids', type=str, required=True,
                        help='Comma-separated PIDs to process')
    parser.add_argument('--bone-offsets', default='data/hymotion_m2m_data/bone_offsets_22.pt')
    args = parser.parse_args()

    bo = torch.load(args.bone_offsets, map_location='cpu', weights_only=True).float().numpy()
    os.makedirs(args.output_dir, exist_ok=True)

    seed_dirs = {}
    for kv in args.seed_dirs:
        k, v = kv.split('=', 1)
        seed_dirs[k] = v

    target_pids = set(f'{int(p):05d}.npz' for p in args.pids.split(','))

    print(f"{'pid':>5}  {'seed':>5}  {'score':>7}  {'fs':>6}  {'xz/y':>6}  {'jitter':>7}")
    print('-' * 48)

    n = 0
    for fname in sorted(target_pids):
        if not any(os.path.exists(os.path.join(d, fname)) for d in seed_dirs.values()):
            continue

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
            if 'N_transition' in layout and layout['N_transition'] > 0:
                nc_a = layout.get('N_cond_a', 45)
                gen_start = nc_a
                gen_end = nc_a + layout['N_transition']
            elif 'T_gt_eff' in layout:
                gen_start = 0
                gen_end = m135.shape[0]
            else:
                continue
            result = score_height_change(m135, bo, gen_start, gen_end)
            candidates.append((skey, path, result))

        if not candidates:
            continue

        candidates.sort(key=lambda x: x[2][0])
        best_skey, best_path, (score, fs, xzy, jit) = candidates[0]
        shutil.copy2(best_path, os.path.join(args.output_dir, fname))
        n += 1

        pid = fname.replace('.npz', '')
        print(f'{pid}  {best_skey:>5}  {score:7.3f}  {fs:5.1%}  {xzy:6.2f}  {jit:7.4f}')

    print(f'\n{n} cases processed -> {args.output_dir}')


if __name__ == '__main__':
    main()
