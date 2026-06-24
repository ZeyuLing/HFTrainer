"""Compute the per-joint adaptive (QC-defect) repair mask for BrokenAMASS*.

This reproduces EXACTLY the mask our M2M repair uses inside
``evaluate_sample`` for the ``AUTO_qc`` setting: run the motion Quality
Checker on the (20->30 fps upsampled) corrupted clip, OR all failing
checkers' invalid masks into a per-joint per-frame defect mask, with the
same dilation params. We then reduce the (T,135) expansion back to a
per-joint (T,22) boolean and resample 30->20 fps to align with the stored
SM results frame count, so the web viewer can render the mask on a
skeleton (red = regenerated, cyan = kept).

No model inference here — the mask is a deterministic function of the
input motion, so this is faithful and cheap.

Usage (run on taiji):
    CUDA_VISIBLE_DEVICES=0 python3 scripts/eval/compute_repair_masks.py \
        --sm-results ref_repo/StableMotion/output/brokenamass_star_sm_enhanced/results.npy \
        --out ref_repo/StableMotion/output/brokenamass_star_ours/repair_masks.npy \
        --max-samples 9999
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_stablemotion_e9 import (  # noqa: E402
    smpldata_to_m2m135,
    _resample_motion135_slerp,
)
from scripts.eval.eval_m2m_v2_all_tasks import (  # noqa: E402
    _compute_qc_defect_mask,
)


def _to_torch(x):
    return x.float() if isinstance(x, torch.Tensor) else torch.from_numpy(np.asarray(x)).float()


def _per_joint_from_135_mask(mask_135: np.ndarray) -> np.ndarray:
    """(T,135)->(T,22) bool: joint j masked iff its rot6d block (col 3+6j) is set."""
    T = mask_135.shape[0]
    cols = 3 + 6 * np.arange(22)
    return mask_135[:, cols] > 0.5


def _resample_bool_nearest(mask_TJ: np.ndarray, L_out: int) -> np.ndarray:
    """Nearest-neighbour temporal resample of a (T,22) boolean mask to L_out."""
    T = mask_TJ.shape[0]
    if T == L_out:
        return mask_TJ
    idx = np.clip(np.round(np.linspace(0, T - 1, L_out)).astype(int), 0, T - 1)
    return mask_TJ[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sm-results', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-samples', type=int, default=9999)
    ap.add_argument('--src-fps', type=float, default=20.0)
    ap.add_argument('--work-fps', type=float, default=30.0)
    ap.add_argument('--device', default='cuda')
    # match eval AUTO_qc setting
    ap.add_argument('--dilate-temp', type=int, default=2)
    ap.add_argument('--no-dilate-spatial', action='store_true')
    ap.add_argument('--no-borderline', action='store_true')
    args = ap.parse_args()

    bone_offsets = torch.load(
        str(PROJECT_ROOT / 'data/hymotion_m2m_data/bone_offsets_22.pt'),
        map_location='cpu', weights_only=False,
    ).float()

    sm = np.load(args.sm_results, allow_pickle=True).item()
    corrupted = sm['motion']
    lengths = np.asarray(sm['lengths']).reshape(-1)
    N = min(len(corrupted), args.max_samples)
    print(f'[masks] {N} clips from {args.sm_results}')

    masks = []          # list of (L,22) bool
    coverage = []       # per-clip fraction of (frame,joint) cells masked
    t0 = time.time()
    for i in range(N):
        sd = {k: _to_torch(v) for k, v in corrupted[i].items()}
        T20 = sd['poses'].shape[0]
        L = int(lengths[i]) if i < len(lengths) else T20
        L = min(L, T20)
        sd = {k: v[:L] for k, v in sd.items()}

        m135_20 = smpldata_to_m2m135(sd, bone_offsets)
        T30 = max(2, int(round(L * args.work_fps / args.src_fps)))
        m135_30 = _resample_motion135_slerp(m135_20, T30)

        mask_135 = _compute_qc_defect_mask(
            m135_30, bone_offsets, motion_dim=135,
            dilate_temp=args.dilate_temp,
            dilate_spatial=not args.no_dilate_spatial,
            include_borderline=not args.no_borderline,
            device=args.device,
        )
        if mask_135 is None:
            pj30 = np.zeros((T30, 22), dtype=bool)
        else:
            pj30 = _per_joint_from_135_mask(np.asarray(mask_135))
        pj20 = _resample_bool_nearest(pj30, L)
        masks.append(pj20)
        coverage.append(float(pj20.mean()))
        if (i + 1) % 20 == 0 or i == N - 1:
            dt = time.time() - t0
            mc = float(np.mean(coverage))
            print(f'[{i+1}/{N}] cover(mean)={mc*100:.1f}% '
                  f'({dt/(i+1):.2f}s/clip)')

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, {
        'masks_joint': np.array(masks, dtype=object),
        'coverage': np.asarray(coverage, dtype=np.float32),
        'lengths': lengths[:N],
    })
    cov = np.asarray(coverage)
    print(f'[save] {out_path}')
    print(f'[coverage] mean={cov.mean()*100:.1f}%  median={np.median(cov)*100:.1f}%  '
          f'max={cov.max()*100:.1f}%  zero-clips={(cov==0).sum()}/{N}')


if __name__ == '__main__':
    sys.exit(main())
