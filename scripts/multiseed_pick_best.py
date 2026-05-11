#!/usr/bin/env python3
"""Run eval_m2m_v2_all_tasks with multiple seed bases, then for each case
pick the NPZ with the best physical metrics (lowest foot_skating_ratio).

Usage:
    python3 scripts/multiseed_pick_best.py \
        --seed-dirs s0=/path/to/seed0/npz s1=/path/to/seed1/npz ... \
        --output-dir /path/to/best/npz \
        --eval-jsons s0=/path/to/seed0.json s1=/path/to/seed1.json ...
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from hftrainer.evaluation.motion.m2m_eval_metrics import motion135_to_positions_np


ALL_FOOT = [7, 8, 10, 11]
GROUND_Y_THRESH = 0.08
SLIDE_SPEED_THRESH = 0.015


def compute_skating(m135, bone_offsets, gen_start, gen_end):
    positions = motion135_to_positions_np(m135, bone_offsets)
    gen_pos = positions[gen_start:gen_end]
    N = gen_pos.shape[0]
    bad = total = 0
    for fi in range(1, N):
        for j in ALL_FOOT:
            if gen_pos[fi, j, 1] < GROUND_Y_THRESH:
                total += 1
                xz_v = np.linalg.norm(gen_pos[fi, j, [0, 2]] - gen_pos[fi-1, j, [0, 2]])
                if xz_v > SLIDE_SPEED_THRESH:
                    bad += 1
    return bad / max(1, total)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed-dirs', nargs='+', required=True,
                        help='key=path pairs, e.g. s0=dir0 s1=dir1')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--eval-jsons', nargs='+', default=[],
                        help='key=json pairs to build merged eval JSON')
    parser.add_argument('--bone-offsets', default='data/hymotion_m2m_data/bone_offsets_22.pt')
    args = parser.parse_args()

    bone_offsets = torch.load(args.bone_offsets, map_location='cpu',
                              weights_only=True).float().numpy()
    os.makedirs(args.output_dir, exist_ok=True)

    # Parse seed dirs
    seed_dirs = {}
    for kv in args.seed_dirs:
        k, v = kv.split('=', 1)
        seed_dirs[k] = v

    # Parse eval JSONs
    eval_jsons = {}
    for kv in args.eval_jsons:
        k, v = kv.split('=', 1)
        with open(v) as f:
            eval_jsons[k] = json.load(f)

    # Find all NPZ files (union of all seed dirs)
    all_fnames = set()
    for d in seed_dirs.values():
        for f in os.listdir(d):
            if f.endswith('.npz'):
                all_fnames.add(f)
    all_fnames = sorted(all_fnames)

    print(f"{'pid':>5}  {'best_seed':>10}  {'best_fs':>8}  {'worst_fs':>8}  {'sources':>8}")
    print('-' * 48)

    best_source = {}  # fname -> (seed_key, ratio)
    total_best = 0.0
    n = 0

    for fname in all_fnames:
        pid = fname.replace('.npz', '')
        candidates = []

        for skey, sdir in seed_dirs.items():
            path = os.path.join(sdir, fname)
            if not os.path.exists(path):
                continue
            d = np.load(path, allow_pickle=True)
            if 'motion_135' not in d.files or 'layout_json' not in d.files:
                candidates.append((skey, path, 999.0))
                continue
            m135 = d['motion_135'].astype(np.float32)
            layout = json.loads(bytes(d['layout_json']).decode())
            nc_a = layout.get('N_cond_a', 45)
            n_trans = layout.get('N_transition', 0)
            if n_trans == 0:
                candidates.append((skey, path, 0.0))
                continue
            ratio = compute_skating(m135, bone_offsets, nc_a, nc_a + n_trans)
            candidates.append((skey, path, ratio))

        if not candidates:
            continue

        candidates.sort(key=lambda x: x[2])
        best_skey, best_path, best_ratio = candidates[0]
        worst_ratio = candidates[-1][2]
        n_sources = len(candidates)

        shutil.copy2(best_path, os.path.join(args.output_dir, fname))
        best_source[fname] = (best_skey, best_ratio)
        total_best += best_ratio
        n += 1

        print(f'{pid}  {best_skey:>10}  {best_ratio:8.1%}  {worst_ratio:8.1%}  {n_sources:>8}')

    print(f'\n{n} cases, avg best skating: {total_best/max(1,n):.1%}')

    # Build merged eval JSON (use best seed's per_sample metrics for each case)
    if eval_jsons:
        # Use first available JSON as template
        first_key = list(eval_jsons.keys())[0]
        merged = json.loads(json.dumps(eval_jsons[first_key]))  # deep copy

        for i, sample in enumerate(merged.get('per_sample', [])):
            fname = f"{i:05d}.npz"
            if fname in best_source:
                skey, _ = best_source[fname]
                if skey in eval_jsons and i < len(eval_jsons[skey].get('per_sample', [])):
                    # Replace with best seed's metrics
                    best_sample = eval_jsons[skey]['per_sample'][i]
                    for k, v in best_sample.items():
                        sample[k] = v
            # Update NPZ path
            sample['_npz_path'] = os.path.join(args.output_dir, fname)

        # Recompute aggregated metrics
        agg = {}
        metric_keys = set()
        for s in merged['per_sample']:
            for k, v in s.items():
                if not k.startswith('_') and isinstance(v, (int, float)):
                    metric_keys.add(k)
        for mk in metric_keys:
            vals = [s[mk] for s in merged['per_sample']
                    if mk in s and isinstance(s[mk], (int, float))
                    and not (isinstance(s[mk], float) and np.isnan(s[mk]))]
            if vals:
                agg[mk] = float(np.mean(vals))
        merged['aggregated'] = agg
        merged['notes'] = 'best-of-N seeds (lowest foot_skating_ratio)'

        out_json = os.path.join(os.path.dirname(args.output_dir), 'merged_best.json')
        with open(out_json, 'w') as f:
            json.dump(merged, f, indent=2)
        print(f'\nMerged eval JSON: {out_json}')

    # Summary by seed
    seed_counts = {}
    for fname, (skey, _) in best_source.items():
        seed_counts[skey] = seed_counts.get(skey, 0) + 1
    print('\nBest-source distribution:')
    for skey in sorted(seed_counts.keys()):
        print(f'  {skey}: {seed_counts[skey]} cases')


if __name__ == '__main__':
    main()
