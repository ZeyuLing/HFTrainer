"""Overwrite the `ours_*` fields of the existing BrokenAMASS* repair-compare
NPZs with a freshly-run combo result, keeping corrupted/gt/stablemotion and the
GT corruption_mask intact.

    ours_135    <- combo results['motion_fix'][i]   (smpldata -> m2m135)
    skel_ours   <- FK(ours_135)
    mask_joint  <- combo results['joint_masks'][i]   (real self_denoise mask)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_stablemotion_e9 import smpldata_to_m2m135  # noqa: E402
from hftrainer.evaluation.motion.m2m_eval_metrics import (  # noqa: E402
    motion135_to_positions_np,
)


def _ten(x):
    return x.float() if isinstance(x, torch.Tensor) else \
        torch.from_numpy(np.asarray(x)).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--combo', required=True, help='ours combo results.npy')
    ap.add_argument('--npz-dir', required=True, help='existing viewer npz dir')
    args = ap.parse_args()

    bone_offsets = torch.load(
        str(PROJECT_ROOT / 'data/hymotion_m2m_data/bone_offsets_22.pt'),
        map_location='cpu', weights_only=False,
    ).float()

    combo = np.load(args.combo, allow_pickle=True).item()
    fix = combo['motion_fix']
    jmasks = combo.get('joint_masks')
    lengths = np.asarray(combo['lengths']).reshape(-1)
    npz_dir = Path(args.npz_dir)

    n_done = 0
    for i in range(len(fix)):
        p = npz_dir / f'{i:05d}.npz'
        if not p.is_file():
            continue
        d = dict(np.load(p, allow_pickle=True))
        L = int(d['length'])
        sd = fix[i]
        sd_y = {k: _ten(sd[k])[:L] for k in ('poses', 'trans', 'joints')}
        o135 = smpldata_to_m2m135(sd_y, bone_offsets)
        o135 = np.asarray(o135, dtype=np.float32)[:L]
        skel_ours = motion135_to_positions_np(o135, bone_offsets.numpy())
        if jmasks is not None and i < len(jmasks):
            mj = np.asarray(jmasks[i]).astype(bool)
            if mj.shape[0] < L:
                mj = np.concatenate(
                    [mj, np.zeros((L - mj.shape[0], 22), bool)], axis=0)
            mj = mj[:L]
        else:
            mj = np.zeros((L, 22), bool)
        d['ours_135'] = o135
        d['skel_ours'] = skel_ours.astype(np.float32)
        d['mask_joint'] = mj.astype(bool)
        d['mask_coverage'] = np.float32(mj.mean())
        np.savez(p, **d)
        n_done += 1
        if n_done % 50 == 0:
            print(f'  [{n_done}]')
    print(f'[done] overwrote ours fields in {n_done} npz')


if __name__ == '__main__':
    sys.exit(main())
