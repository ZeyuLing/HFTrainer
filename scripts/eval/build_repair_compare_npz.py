"""Build per-case 135-dim NPZ for the BrokenAMASS* repair 4-panel viewer.

Reads StableMotion-format results (smpldata, z-up canonical) for all four
roles and converts each to our 135-dim motion (z-up->y-up + smpldata->m2m135),
so the web viewer can render them as SMPL-H meshes on a shared timeline:

    corrupted_135      ← SM results['motion'][i]        (corrupted input)
    gt_135             ← clean results_collected['motion'][i]
    stablemotion_135   ← SM results['motion_fix'][i]
    ours_135           ← ours results['motion_fix'][i]

Usage:
    python3 scripts/eval/build_repair_compare_npz.py \
        --sm ref_repo/StableMotion/output/brokenamass_star_sm_enhanced/results.npy \
        --ours ref_repo/StableMotion/output/brokenamass_star_ours/results.npy \
        --gt ref_repo/StableMotion/output/brokenamass_star_clean_v2/results_collected.npy \
        --out-dir output/eval/brokenamass_star_repair_compare/npz
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_stablemotion_e9 import (  # noqa: E402
    smpldata_to_m2m135,
)
from hftrainer.evaluation.motion.m2m_eval_metrics import (  # noqa: E402
    motion135_to_positions_np,
)
from scripts.eval.eval_m2m_v2_all_tasks import (  # noqa: E402
    _compute_strict_adaptive_mask,
)

ADAPTIVE_MASK_DIR = (
    PROJECT_ROOT / 'data/eval/hymotion_m2m/adaptive_masks_mogendit/brokenamass_star'
)


def _ten(x):
    return x.float() if isinstance(x, torch.Tensor) else torch.from_numpy(np.asarray(x)).float()


def _temporal_dilate(jm: np.ndarray, k: int) -> np.ndarray:
    """±k frame max-pool per joint (matches _load_adaptive_mask_for_motion)."""
    if k <= 0:
        return jm
    out = jm.copy()
    for j in range(jm.shape[1]):
        col = jm[:, j].astype(bool)
        o = col.copy()
        for s in range(1, k + 1):
            o[s:] |= col[:-s]
            o[:-s] |= col[s:]
        out[:, j] = o
    return out


def load_joint_mask(case_i: int, L: int) -> np.ndarray:
    """Load the MoGenDIT mask and reproduce the STRICT mask actually used at
    inference (AUTO_strict_sdedit: dilate=2, min_blob=3, kinematic spatial
    dilation, lock_trans). Resample 30fps->20fps to (L,22) bool. The mask panel
    must reflect what was really regenerated, not the raw adaptive mask."""
    f = ADAPTIVE_MASK_DIR / f'{case_i:05d}.npz'
    if not f.is_file():
        return np.zeros((L, 22), dtype=bool)
    jm = np.load(f)['joint_mask'].astype(np.float32)   # (T30, 22) 1=generate
    T = jm.shape[0]
    # Broadcast per-joint mask to a 135-dim raw mask (trans + 22*rot6d), then
    # apply the exact strict-tightening used at inference.
    raw135 = np.zeros((T, 135), dtype=np.float32)
    raw135[:, :3] = jm[:, 0:1]
    for j in range(22):
        raw135[:, 3 + j * 6: 3 + (j + 1) * 6] = jm[:, j:j + 1]
    strict135 = _compute_strict_adaptive_mask(
        raw135, dilate=2, min_blob=3, motion_dim=135, lock_trans=True,
    )
    jstrict = np.zeros((T, 22), dtype=bool)
    for j in range(22):
        jstrict[:, j] = strict135[:, 3 + j * 6] >= 0.5
    idx = np.clip(np.round(np.linspace(0, T - 1, L)).astype(int), 0, T - 1)
    return jstrict[idx]


def smpldata_z_to_135(sd_dict, bone_offsets, L):
    # StableMotion results.npy smpldata is already y-up (verified: head-pelvis
    # spine ≈ +y, y the largest axis spread). The earlier z_up_to_y_up call was
    # a double-rotation that laid every figure down along z — removed.
    sd_y = {k: _ten(sd_dict[k])[:L] for k in ('poses', 'trans', 'joints')}
    return smpldata_to_m2m135(sd_y, bone_offsets)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sm', required=True)
    ap.add_argument('--ours', required=True)
    ap.add_argument('--gt', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--max-cases', type=int, default=300)
    args = ap.parse_args()

    bone_offsets = torch.load(
        str(PROJECT_ROOT / 'data/hymotion_m2m_data/bone_offsets_22.pt'),
        map_location='cpu', weights_only=False,
    ).float()

    sm = np.load(args.sm, allow_pickle=True).item()
    ours = np.load(args.ours, allow_pickle=True).item()
    gt = np.load(args.gt, allow_pickle=True).item()

    corrupted = sm['motion']
    sm_fix = sm['motion_fix']
    ours_fix = ours['motion_fix']
    gt_motion = gt['motion']
    lengths = np.asarray(sm['lengths']).reshape(-1)

    N = min(len(corrupted), len(sm_fix), len(ours_fix), len(gt_motion),
            args.max_cases)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'[build] {N} cases -> {out_dir}')

    for i in range(N):
        L = int(min(lengths[i], _ten(corrupted[i]['poses']).shape[0]))
        try:
            c135 = smpldata_z_to_135(corrupted[i], bone_offsets, L)
            g135 = smpldata_z_to_135(gt_motion[i], bone_offsets, L)
            s135 = smpldata_z_to_135(sm_fix[i], bone_offsets, L)
            o135 = smpldata_z_to_135(ours_fix[i], bone_offsets, L)
        except Exception as e:
            print(f'  [{i}] convert FAIL: {e}')
            continue
        # Adaptive mask (which joints/frames OUR method regenerated) + FK
        # joint positions so the viewer can render a mask-colored skeleton.
        # Prefer the REAL self_denoise mask saved by run_ours (what hymotion-m2m
        # actually regenerated); fall back to the MoGenDIT mask only if absent.
        ours_masks = ours.get('joint_masks')
        if ours_masks is not None and i < len(ours_masks):
            mask_joint = np.asarray(ours_masks[i]).astype(bool)[:L]   # (L,22)
            if mask_joint.shape[0] < L:
                pad = np.zeros((L - mask_joint.shape[0], 22), bool)
                mask_joint = np.concatenate([mask_joint, pad], axis=0)
        else:
            mask_joint = load_joint_mask(i, L)                       # (L, 22) bool
        skel_corrupted = motion135_to_positions_np(c135, bone_offsets.numpy())
        skel_ours = motion135_to_positions_np(o135, bone_offsets.numpy())
        np.savez(
            out_dir / f'{i:05d}.npz',
            corrupted_135=c135.astype(np.float32),
            gt_135=g135.astype(np.float32),
            stablemotion_135=s135.astype(np.float32),
            ours_135=o135.astype(np.float32),
            mask_joint=mask_joint.astype(bool),
            skel_corrupted=skel_corrupted.astype(np.float32),
            skel_ours=skel_ours.astype(np.float32),
            mask_coverage=np.float32(mask_joint.mean()),
            length=L,
        )
        if (i + 1) % 50 == 0 or i == N - 1:
            print(f'  [{i+1}/{N}]')
    print('[done]')


if __name__ == '__main__':
    sys.exit(main())
