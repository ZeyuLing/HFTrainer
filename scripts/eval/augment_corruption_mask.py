"""Augment the BrokenAMASS* repair-compare per-case NPZ with a ground-truth
*corruption mask*: which joints/frames of the corrupted input were actually
modified by the corruption process. This lets the viewer highlight the degraded
frames/regions when showing the corrupted (and clean) clip.

The corruption is applied directly on the motion representation (root
translation + per-joint rot6d), so the *exact* corruption label is simply
"where corrupted_135 differs from gt_135":

    joint j (rot6d) corrupted at frame t  <=>  max|c_rot6d[t,j] - g_rot6d[t,j]| > EPS
    pelvis (j=0) also corrupted           <=>  max|c_transl[t] - g_transl[t]|  > EPS

We compare in the raw 135-dim representation rather than skeleton positions on
purpose: a position-space diff propagates an upstream joint's rotation error
down the whole kinematic chain, over-stating which joints were corrupted. The
rotation/translation channels are exactly what the corruption touched.

EPS only filters the float / slerp-resampling noise floor (~1e-6..1e-5); the
diff distribution is strongly bimodal so 1e-4 cleanly separates corrupted joints
(~15-20%) from untouched ones. Genuinely clean clips (BrokenAMASS keeps some,
corrupted_135 == gt_135 to float precision) correctly stay all-blue.

Usage:
    python3 scripts/eval/augment_corruption_mask.py \
        --npz-dir output/eval/brokenamass_star_repair_compare/npz --eps 1e-4
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from hftrainer.evaluation.motion.m2m_eval_metrics import (  # noqa: E402
    motion135_to_positions_np,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz-dir', required=True)
    ap.add_argument('--eps', type=float, default=1e-4,
                    help='per-channel |corrupted - gt| above which a joint-frame '
                         'is flagged corrupted (filters slerp/float noise floor).')
    args = ap.parse_args()

    bone_offsets = torch.load(
        str(PROJECT_ROOT / 'data/hymotion_m2m_data/bone_offsets_22.pt'),
        map_location='cpu', weights_only=False,
    ).float().numpy()

    paths = sorted(glob.glob(os.path.join(args.npz_dir, '*.npz')))
    print(f'[augment] {len(paths)} cases, eps={args.eps:.0e} '
          '(exact per-joint rot6d + root-transl diff)')
    covs = []
    for i, p in enumerate(paths):
        d = dict(np.load(p, allow_pickle=True))
        c135 = np.asarray(d['corrupted_135'], np.float64)
        g135 = np.asarray(d['gt_135'], np.float64)
        T = min(c135.shape[0], g135.shape[0])
        c135, g135 = c135[:T], g135[:T]
        # exact corruption label: which joint-frames were actually modified.
        # rot6d channels [3:135] = 22 joints x 6; pelvis(j=0) is the global
        # orient, j=1..21 the body joints.
        cr = c135[:, 3:135].reshape(T, 22, 6)
        gr = g135[:, 3:135].reshape(T, 22, 6)
        corruption_mask = np.abs(cr - gr).max(axis=-1) > args.eps   # (T,22) bool
        # pelvis additionally flagged when the root translation was corrupted.
        tdiff = np.abs(c135[:, :3] - g135[:, :3]).max(axis=-1) > args.eps
        corruption_mask[:, 0] |= tdiff
        skel_g = motion135_to_positions_np(
            g135.astype(np.float32), bone_offsets)                  # (T,22,3)
        covs.append(corruption_mask.mean())
        d['skel_gt'] = skel_g.astype(np.float32)
        d['corruption_mask'] = corruption_mask.astype(bool)
        d['corruption_coverage'] = np.float32(corruption_mask.mean())
        np.savez(p, **d)
        if (i + 1) % 50 == 0 or i == len(paths) - 1:
            print(f'  [{i+1}/{len(paths)}] mean corruption coverage so far '
                  f'{np.mean(covs)*100:.1f}%')
    print(f'[done] mean corruption coverage {np.mean(covs)*100:.1f}%  '
          f'(p50 {np.percentile(covs,50)*100:.1f}%  p90 {np.percentile(covs,90)*100:.1f}%)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
