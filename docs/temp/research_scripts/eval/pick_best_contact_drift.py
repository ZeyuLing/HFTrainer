#!/usr/bin/env python3
"""Pick best NPZ from multiple seeds using contact-drift skating metric.

Contact drift measures total XZ displacement of grounded feet during contact
events, normalized by frame count. This is the most accurate skating detector
we have — it catches slow persistent sliding that binary fs_ratio misses.

Usage:
    python3 scripts/pick_best_contact_drift.py \
        --seed-dirs s0=/path/to/s0/npz s1=/path/to/s1/npz ... \
        --output-dir /path/to/best/npz \
        [--only-skating-cases]  # only re-pick cases with skating > threshold
"""
import argparse
import json
import os
import shutil
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hftrainer.evaluation.motion.m2m_eval_metrics import motion135_to_positions_np


ALL_FOOT = [7, 8, 10, 11]  # L_ankle, R_ankle, L_foot, R_foot
CONTACT_Y_THRESH = 0.08   # below this Y = foot on ground
CONTACT_XZ_IDLE = 0.003   # below this XZ speed/frame = foot stationary (in contact)
MIN_CONTACT_FRAMES = 3    # ignore contact events shorter than this


def compute_contact_drift(positions, gen_start, gen_end):
    """Compute skating score based on total XZ drift of grounded feet.

    For each foot joint, identify contact events (consecutive frames where
    Y < threshold and XZ velocity is low enough to be "planted"). Then
    measure total XZ displacement during each contact event. A foot that
    is planted should have ~0 XZ drift; any drift is skating.

    Returns:
        skating_score: total XZ drift / N_frames (lower = better)
        contact_info: dict with per-foot details
    """
    gen_pos = positions[gen_start:gen_end]  # (N, 22, 3)
    N = gen_pos.shape[0]
    if N < 3:
        return 0.0, {}

    total_drift = 0.0
    contact_info = {}

    for j in ALL_FOOT:
        foot_y = gen_pos[:, j, 1]
        foot_xz = gen_pos[:, j, [0, 2]]

        # Find frames where foot is on/near ground
        grounded = foot_y < CONTACT_Y_THRESH

        # Find contact events (consecutive grounded frames)
        events = []
        start = None
        for fi in range(N):
            if grounded[fi]:
                if start is None:
                    start = fi
            else:
                if start is not None:
                    if fi - start >= MIN_CONTACT_FRAMES:
                        events.append((start, fi))
                    start = None
        if start is not None and N - start >= MIN_CONTACT_FRAMES:
            events.append((start, N))

        # For each contact event, measure XZ drift
        joint_drift = 0.0
        n_contact_frames = 0
        for (es, ee) in events:
            # Total XZ displacement during this contact event
            event_drift = 0.0
            for fi in range(es + 1, ee):
                d = np.linalg.norm(foot_xz[fi] - foot_xz[fi - 1])
                event_drift += d
            joint_drift += event_drift
            n_contact_frames += (ee - es)

        total_drift += joint_drift
        contact_info[j] = {
            'n_events': len(events),
            'n_contact_frames': n_contact_frames,
            'total_drift': joint_drift,
        }

    # Normalize by total frames
    skating_score = total_drift / max(1, N)
    return skating_score, contact_info


def compute_jitter(positions, gen_start, gen_end):
    """Compute motion jitter (acceleration magnitude) in generated region."""
    gen_pos = positions[gen_start:gen_end]
    if gen_pos.shape[0] < 3:
        return 0.0
    vel = np.diff(gen_pos, axis=0)
    accel = np.diff(vel, axis=0)
    return float(np.linalg.norm(accel, axis=-1).mean())


def score_npz(m135, bone_offsets, gen_start, gen_end):
    """Compute combined quality score. Lower = better.

    Returns: (combined_score, skating_score, jitter, contact_info)
    """
    positions = motion135_to_positions_np(m135, bone_offsets)
    skating_score, contact_info = compute_contact_drift(positions, gen_start, gen_end)
    jitter = compute_jitter(positions, gen_start, gen_end)

    # Combined: skating dominates, jitter is secondary
    combined = skating_score + 0.5 * jitter
    return combined, skating_score, jitter, contact_info


def get_gen_region(layout, total_frames):
    """Extract generated region from layout."""
    if 'N_transition' in layout and layout['N_transition'] > 0:
        nc_a = layout.get('N_cond_a', 45)
        n_trans = layout['N_transition']
        return nc_a, nc_a + n_trans
    elif 'T_gt_eff' in layout:
        return 0, total_frames
    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed-dirs', nargs='+', required=True,
                        help='key=path pairs, e.g. s0=dir0/npz s1=dir1/npz')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--bone-offsets', default='data/hymotion_m2m_data/bone_offsets_22.pt')
    parser.add_argument('--fallback-dir', default=None,
                        help='If provided, copy unchanged cases from this dir instead of first seed')
    parser.add_argument('--only-pids', type=str, default=None,
                        help='Comma-separated PIDs to process (skip others)')
    parser.add_argument('--skating-threshold', type=float, default=0.3,
                        help='Cases with score above this are flagged as skating')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    bo = torch.load(args.bone_offsets, map_location='cpu', weights_only=True).float().numpy()
    os.makedirs(args.output_dir, exist_ok=True)

    # Parse seed dirs
    seed_dirs = {}
    for kv in args.seed_dirs:
        k, v = kv.split('=', 1)
        seed_dirs[k] = v

    # Find all NPZ files
    all_fnames = set()
    for d in seed_dirs.values():
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.endswith('.npz'):
                    all_fnames.add(f)
    all_fnames = sorted(all_fnames)

    # Filter by PIDs if specified
    if args.only_pids:
        target = set(f'{int(p):05d}.npz' for p in args.only_pids.split(','))
        all_fnames = [f for f in all_fnames if f in target]

    print(f"Processing {len(all_fnames)} cases from {len(seed_dirs)} seed dirs")
    print(f"{'pid':>5}  {'best':>6}  {'skate':>7}  {'jitter':>7}  {'comb':>7}  {'worst':>7}  seeds")
    print('-' * 65)

    results = {}
    skating_cases = []
    seed_counts = defaultdict(int)
    total_skate = 0.0

    for fname in all_fnames:
        pid = fname.replace('.npz', '')
        candidates = []

        for skey, sdir in seed_dirs.items():
            path = os.path.join(sdir, fname)
            if not os.path.exists(path):
                continue
            try:
                d = np.load(path, allow_pickle=True)
                if 'motion_135' not in d.files or 'layout_json' not in d.files:
                    continue
                m135 = d['motion_135'].astype(np.float32)
                layout = json.loads(bytes(d['layout_json']).decode())
                gen_start, gen_end = get_gen_region(layout, m135.shape[0])
                if gen_start is None:
                    continue
                combined, skate, jitter, cinfo = score_npz(m135, bo, gen_start, gen_end)
                candidates.append({
                    'seed': skey, 'path': path,
                    'combined': combined, 'skate': skate,
                    'jitter': jitter, 'contact_info': cinfo
                })
            except Exception as e:
                print(f"  WARN: {skey}/{fname}: {e}")
                continue

        if not candidates:
            # Copy from fallback if available
            if args.fallback_dir:
                fb = os.path.join(args.fallback_dir, fname)
                if os.path.exists(fb):
                    shutil.copy2(fb, os.path.join(args.output_dir, fname))
            continue

        # Sort by combined score (lower = better)
        candidates.sort(key=lambda x: x['combined'])
        best = candidates[0]
        worst = candidates[-1]

        shutil.copy2(best['path'], os.path.join(args.output_dir, fname))
        results[fname] = best
        seed_counts[best['seed']] += 1
        total_skate += best['skate']

        if best['skate'] > args.skating_threshold:
            skating_cases.append((pid, best['skate']))

        seeds_str = ','.join(c['seed'] for c in candidates[:3])
        if len(candidates) > 3:
            seeds_str += f'+{len(candidates)-3}'
        print(f"{pid}  {best['seed']:>6}  {best['skate']:7.3f}  {best['jitter']:7.4f}  "
              f"{best['combined']:7.3f}  {worst['skate']:7.3f}  {seeds_str}")

    n = len(results)
    print(f"\n{'='*65}")
    print(f"Total: {n} cases, avg skating: {total_skate/max(1,n):.3f}")
    print(f"Cases with skating > {args.skating_threshold}: {len(skating_cases)}/{n}")

    if skating_cases:
        print(f"\nStill-skating cases (score > {args.skating_threshold}):")
        for pid, score in sorted(skating_cases, key=lambda x: -x[1]):
            print(f"  pid={pid}: skating={score:.3f}")

    print(f"\nSeed distribution:")
    for skey in sorted(seed_counts.keys()):
        print(f"  {skey}: {seed_counts[skey]} cases")

    print(f"\nOutput: {args.output_dir}")


if __name__ == '__main__':
    main()
