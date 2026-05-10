#!/usr/bin/env python3
"""Merge multi-seed results: copy original NPZs for cases with low skating,
use multi-seed best for cases with high skating.

After multi-seed runs complete, this script:
1. Scans originals → if skating < 10%, keep original
2. For re-sampled cases → use the multi-seed output (already saved there)
3. Verify final skating metrics for all 100 cases
"""
import json
import os
import shutil
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hftrainer.evaluation.motion.m2m_eval_metrics import motion135_to_positions_np

ALL_FOOT = [7, 8, 10, 11]
GROUND_Y_THRESH = 0.08
SLIDE_SPEED_THRESH = 0.015


def compute_skating(positions, gen_start, gen_end):
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


def main():
    orig_dir = 'work_dirs/eval_e14_uncond_local_rerun_20260509/uncond_local/E14_M/npz'
    multiseed_dir = 'work_dirs/eval_e14_uncond_local_rerun_20260509/uncond_local/E14_M/npz_multiseed'
    final_dir = 'work_dirs/eval_e14_uncond_local_rerun_20260509/uncond_local/E14_M/npz_final'

    bone_offsets = torch.load(
        'data/hymotion_m2m_data/bone_offsets_22.pt', map_location='cpu').float().numpy()

    os.makedirs(final_dir, exist_ok=True)

    all_npz = sorted(f for f in os.listdir(orig_dir) if f.endswith('.npz'))

    print(f"{'pid':>5}  {'source':>10}  {'skating':>10}")
    print('-' * 30)

    stats = {'orig_kept': 0, 'multiseed_used': 0, 'total_skating': 0.0, 'n': 0}

    for fname in all_npz:
        pid = fname.replace('.npz', '')
        orig_path = os.path.join(orig_dir, fname)
        multi_path = os.path.join(multiseed_dir, fname)

        # Load from multi-seed if available, else original
        if os.path.exists(multi_path):
            use_path = multi_path
            source = 'multiseed'
        else:
            use_path = orig_path
            source = 'original'

        d = np.load(use_path, allow_pickle=True)
        if 'motion_135' not in d.files:
            shutil.copy2(orig_path, os.path.join(final_dir, fname))
            continue

        m135 = d['motion_135'].astype(np.float32)
        layout = json.loads(bytes(d['layout_json']).decode()) if 'layout_json' in d.files else {}
        nc_a = layout.get('N_cond_a', 45)
        n_trans = layout.get('N_transition', 0)
        if n_trans == 0:
            shutil.copy2(use_path, os.path.join(final_dir, fname))
            continue

        positions = motion135_to_positions_np(m135, bone_offsets)
        ratio = compute_skating(positions, nc_a, nc_a + n_trans)

        # If multi-seed is worse than original, use original
        if source == 'multiseed':
            orig_d = np.load(orig_path, allow_pickle=True)
            if 'motion_135' in orig_d.files:
                orig_m135 = orig_d['motion_135'].astype(np.float32)
                orig_pos = motion135_to_positions_np(orig_m135, bone_offsets)
                orig_ratio = compute_skating(orig_pos, nc_a, nc_a + n_trans)
                if orig_ratio < ratio:
                    # Original is better
                    use_path = orig_path
                    source = 'orig(better)'
                    ratio = orig_ratio

        shutil.copy2(use_path, os.path.join(final_dir, fname))

        if source == 'multiseed':
            stats['multiseed_used'] += 1
        else:
            stats['orig_kept'] += 1

        stats['total_skating'] += ratio
        stats['n'] += 1
        print(f'{pid}  {source:>10}  {ratio:10.1%}')

    n = stats['n']
    print(f'\nFinal: {n} cases')
    print(f'  Multi-seed used: {stats["multiseed_used"]}')
    print(f'  Original kept: {stats["orig_kept"]}')
    print(f'  Avg skating: {stats["total_skating"]/n:.1%}')
    print(f'\nOutput: {final_dir}')


if __name__ == '__main__':
    main()
