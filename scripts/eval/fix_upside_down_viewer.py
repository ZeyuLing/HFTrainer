"""Stand upside-down (gravity-flipped) clips upright in the repair-compare
viewer NPZs. Applies the same 180°-X + reground correction used at model input
(scripts.run_stablemotion_e9._upright_fix_135) to the gt / corrupted (and,
optionally, ours) 135-dim motions, recomputing their skeletons. ours is
normally overwritten from a fresh upright re-run, so by default we fix only
gt + corrupted here.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer')
sys.path.insert(0, str(ROOT))

from scripts.run_stablemotion_e9 import _upright_fix_135  # noqa: E402
from hftrainer.evaluation.motion.m2m_eval_metrics import (  # noqa: E402
    motion135_to_positions_np,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz-dir', required=True)
    ap.add_argument('--also-ours', action='store_true',
                    help='also fix ours_135 (use only if ours was NOT re-run '
                         'with upright correction).')
    args = ap.parse_args()

    bo = torch.load(
        str(ROOT / 'data/hymotion_m2m_data/bone_offsets_22.pt'),
        map_location='cpu', weights_only=False).float()
    bon = bo.numpy()

    roles = [('gt_135', 'skel_gt'), ('corrupted_135', 'skel_corrupted')]
    if args.also_ours:
        roles.append(('ours_135', 'skel_ours'))

    npz_dir = Path(args.npz_dir)
    n_fixed = {r: 0 for r, _ in roles}
    files = sorted(npz_dir.glob('*.npz'))
    for f in files:
        d = dict(np.load(f, allow_pickle=True))
        changed = False
        for m_key, s_key in roles:
            if m_key not in d:
                continue
            fixed, flipped = _upright_fix_135(d[m_key], bo)
            if flipped:
                d[m_key] = fixed.astype(np.float32)
                d[s_key] = motion135_to_positions_np(fixed, bon).astype(np.float32)
                n_fixed[m_key] += 1
                changed = True
        if changed:
            np.savez(f, **d)
    print('[done] upright-fixed clips per role:',
          {r: n_fixed[r] for r, _ in roles}, f'/ {len(files)}')


if __name__ == '__main__':
    sys.exit(main())
