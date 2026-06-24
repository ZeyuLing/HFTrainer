"""Generate MoGenDIT change-based adaptive masks for the BrokenAMASS* clips
and cache them where ``evaluate_sample(_use_adaptive_mask)`` looks them up.

For each SM-corrupted clip i:
    smpldata(20fps) -> m2m135 -> resample 30fps -> SMPL-H npz(poses T,156)
    -> MoGenDITRepairPipeline.compute_adaptive_mask
    -> save {joint_mask:(T30,22), trans_mask:(T30,)} to
       data/eval/hymotion_m2m/adaptive_masks_mogendit/brokenamass_star/{i:05d}.npz

run_ours sets sample['path']='brokenamass_star/{i:05d}.npz' so the cache key
matches. Masks are at 30fps (the fps our model runs at), so no resample needed
inside evaluate_sample (T_orig == T30).

Run on taiji:
    CUDA_VISIBLE_DEVICES=0 python3 scripts/eval/gen_mogendit_masks_brokenamass.py \
        --sm-results ref_repo/StableMotion/output/brokenamass_star_sm_enhanced/results.npy
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_stablemotion_e9 import (  # noqa: E402
    smpldata_to_m2m135,
    m2m135_to_smpldata_24,
    _resample_motion135_slerp,
)

CACHE_DIR = PROJECT_ROOT / 'data/eval/hymotion_m2m/adaptive_masks_mogendit/brokenamass_star'


def _to_torch(x):
    return x.float() if isinstance(x, torch.Tensor) else torch.from_numpy(np.asarray(x)).float()


def export_clip_npz(sd, bone_offsets, L, out_npz, work_fps=30.0, src_fps=20.0):
    sd = {k: _to_torch(sd[k])[:L] for k in ('poses', 'trans', 'joints')}
    m135_20 = smpldata_to_m2m135(sd, bone_offsets)
    T30 = max(2, int(round(L * work_fps / src_fps)))
    m135_30 = _resample_motion135_slerp(m135_20, T30)
    sd30 = m2m135_to_smpldata_24(m135_30, bone_offsets)
    poses156 = np.zeros((T30, 156), dtype=np.float32)
    poses156[:, :66] = sd30['poses'].reshape(T30, -1).cpu().numpy().astype(np.float32)[:, :66]
    np.savez(
        out_npz,
        poses=poses156,
        trans=sd30['trans'].cpu().numpy().astype(np.float32),
        betas=np.zeros(16, dtype=np.float32),
        gender='neutral',
        mocap_framerate=np.float32(work_fps),
    )
    return T30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sm-results', required=True)
    ap.add_argument('--max-samples', type=int, default=9999)
    ap.add_argument('--mogendit-steps', type=int, default=10)
    ap.add_argument('--joint-threshold', type=float, default=0.15)
    ap.add_argument('--max-mask-ratio', type=float, default=0.15)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    bone_offsets = torch.load(
        str(PROJECT_ROOT / 'data/hymotion_m2m_data/bone_offsets_22.pt'),
        map_location='cpu', weights_only=False,
    ).float()
    sm = np.load(args.sm_results, allow_pickle=True).item()
    corrupted = sm['motion']
    lengths = np.asarray(sm['lengths']).reshape(-1)
    N = min(len(corrupted), args.max_samples)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f'[gen] {N} clips -> {CACHE_DIR}')

    print('[load] MoGenDIT MoreDiff-0.1B ...')
    from hftrainer.pipelines.motion.mogendit_pipeline import MoGenDITRepairPipeline
    mog = MoGenDITRepairPipeline(model_name='MoreDiff-0.1B', device=args.device)
    print('[ready]')

    cover = []
    t0 = time.time()
    with tempfile.TemporaryDirectory() as td:
        for i in range(N):
            out_path = CACHE_DIR / f'{i:05d}.npz'
            if out_path.is_file() and not args.force:
                z = np.load(out_path); cover.append(float(z['joint_mask'].mean())); continue
            L = int(min(lengths[i], _to_torch(corrupted[i]['poses']).shape[0]))
            npz = f'{td}/{i:05d}.npz'
            export_clip_npz(corrupted[i], bone_offsets, L, npz)
            try:
                res = mog.compute_adaptive_mask(
                    npz, step=args.mogendit_steps,
                    joint_threshold=args.joint_threshold,
                    trans_threshold=0.05, max_mask_ratio=args.max_mask_ratio,
                )
                jm = np.asarray(res['joint_mask']).astype(bool)
                tm = np.asarray(res['trans_mask']).astype(bool)
            except Exception as e:
                print(f'  [{i}] FAIL {e}; empty mask')
                T30 = max(2, int(round(L * 30.0 / 20.0)))
                jm = np.zeros((T30, 22), dtype=bool); tm = np.zeros(T30, dtype=bool)
            np.savez_compressed(out_path, joint_mask=jm, trans_mask=tm)
            cover.append(float(jm.mean()))
            if (i + 1) % 20 == 0 or i == N - 1:
                dt = time.time() - t0
                print(f'[{i+1}/{N}] cover(mean)={np.mean(cover)*100:.1f}% '
                      f'({dt/(i+1):.2f}s/clip)')

    cov = np.asarray(cover)
    print(f'[done] cover mean={cov.mean()*100:.1f}% median={np.median(cov)*100:.1f}% '
          f'zero={(cov==0).sum()}/{len(cov)}')


if __name__ == '__main__':
    sys.exit(main())
