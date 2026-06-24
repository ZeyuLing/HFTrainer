"""Probe: does MoGenDIT's change-based adaptive mask detect BrokenAMASS*
corruption? Exports a few SM-corrupted clips to SMPL-H npz (poses T,156 @30fps),
runs MoGenDITRepairPipeline.compute_adaptive_mask, and compares the detected
per-frame mask against the ground-truth corruption labels (sm['gt_labels']).

Run on taiji:
    CUDA_VISIBLE_DEVICES=0 python3 scripts/eval/test_mogendit_mask_brokenamass.py
"""
from __future__ import annotations

import os
import sys
import tempfile
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

SM = 'ref_repo/StableMotion/output/brokenamass_star_sm_enhanced/results.npy'
CASES = [0, 3, 6, 20, 50, 100]


def _to_torch(x):
    return x.float() if isinstance(x, torch.Tensor) else torch.from_numpy(np.asarray(x)).float()


def export_clip_npz(sd, bone_offsets, L, work_fps, src_fps, out_npz):
    sd = {k: _to_torch(sd[k])[:L] for k in ('poses', 'trans', 'joints')}
    m135_20 = smpldata_to_m2m135(sd, bone_offsets)
    T30 = max(2, int(round(L * work_fps / src_fps)))
    m135_30 = _resample_motion135_slerp(m135_20, T30)
    sd30 = m2m135_to_smpldata_24(m135_30, bone_offsets)
    poses66 = sd30['poses'].reshape(T30, -1).cpu().numpy().astype(np.float32)[:, :66]
    poses156 = np.zeros((T30, 156), dtype=np.float32)
    poses156[:, :66] = poses66
    trans = sd30['trans'].cpu().numpy().astype(np.float32)
    np.savez(
        out_npz,
        poses=poses156,
        trans=trans,
        betas=np.zeros(16, dtype=np.float32),
        gender='neutral',
        mocap_framerate=np.float32(work_fps),
    )
    return T30


def gt_label_frame(sm, i):
    gl = sm['gt_labels']
    flat = np.concatenate([np.asarray(b) for b in gl], axis=0)  # (Nclip, T)
    return np.asarray(flat[i]).astype(bool)  # (T,) per-frame corrupted (20fps)


def main():
    bone_offsets = torch.load(
        str(PROJECT_ROOT / 'data/hymotion_m2m_data/bone_offsets_22.pt'),
        map_location='cpu', weights_only=False,
    ).float()
    sm = np.load(str(PROJECT_ROOT / SM), allow_pickle=True).item()
    corrupted = sm['motion']
    lengths = np.asarray(sm['lengths']).reshape(-1)

    print('[load] MoGenDIT MoreDiff-0.1B ...')
    from hftrainer.pipelines.motion.mogendit_pipeline import MoGenDITRepairPipeline
    mog = MoGenDITRepairPipeline(model_name='MoreDiff-0.1B', device='cuda:0')
    print('[ready]\n')

    with tempfile.TemporaryDirectory() as td:
        for i in CASES:
            L = int(min(lengths[i], _to_torch(corrupted[i]['poses']).shape[0]))
            npz = os.path.join(td, f'{i:05d}.npz')
            T30 = export_clip_npz(corrupted[i], bone_offsets, L, 30.0, 20.0, npz)
            res = mog.compute_adaptive_mask(
                npz, step=10, joint_threshold=0.15,
                trans_threshold=0.05, max_mask_ratio=0.15,
            )
            jm = res['joint_mask']            # (T30, 22) bool
            frame_flag30 = jm.any(axis=1)     # (T30,)
            # downsample to 20fps for gt comparison
            idx = np.clip(np.round(np.linspace(0, T30 - 1, L)).astype(int), 0, T30 - 1)
            frame_flag20 = frame_flag30[idx]
            gt = gt_label_frame(sm, i)[:L]
            inter = (frame_flag20 & gt).sum()
            recall = inter / max(gt.sum(), 1)
            prec = inter / max(frame_flag20.sum(), 1)
            print(f'case {i:3d}: T30={T30} cellcover={jm.mean()*100:4.1f}% '
                  f'flaggedframes(30fps)={frame_flag30.mean()*100:4.1f}% | '
                  f'GTcorrupt={gt.mean()*100:4.1f}%  recall={recall*100:4.1f}%  '
                  f'prec={prec*100:4.1f}%  changeMag(med)={np.median(res["change_magnitude"][:,1:]):.3f}')


if __name__ == '__main__':
    sys.exit(main())
