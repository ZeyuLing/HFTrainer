"""Angle-metric self-denoise detection sweep.

For each case x detect_tau: run the stage-1 self-denoise projection ONCE with
detect_metric='angle', capture the physical per-joint geodesic-angle change
(radians) + translation change (meters) via a spy on
_self_denoise_joint_change, then offline re-derive the per-joint defect mask at
many angle thresholds and score coverage/recall/precision against the GT
corruption mask (corrupted-vs-clean), to see whether the MoGenDIT-style
physical metric localises corruption WITHOUT a max_mask_ratio cap.
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer')
sys.path.insert(0, str(ROOT))

from hftrainer.pipelines.motion.hymotion_m2m_pipeline import (  # noqa: E402
    HyMotionM2MPipeline,
)
from hftrainer.pipelines.motion.repair_utils import (  # noqa: E402
    compute_strict_adaptive_mask,
)
from scripts.eval.run_ours_repair_brokenamass import build_model  # noqa: E402
from scripts.run_stablemotion_e9 import (  # noqa: E402
    smpldata_to_m2m135, _resample_motion135_slerp,
)

bone_offsets = torch.load(
    str(ROOT / 'data/hymotion_m2m_data/bone_offsets_22.pt'),
    map_location='cpu', weights_only=False).float()

_cap = {}
_orig = HyMotionM2MPipeline._self_denoise_joint_change
def _spy(self, mn, stage1):
    j, t = _orig(self, mn, stage1)
    _cap['j'] = np.asarray(j).copy()
    _cap['t'] = np.asarray(t).copy()
    return j, t
HyMotionM2MPipeline._self_denoise_joint_change = _spy

bundle, pipeline, info, ckpt = build_model('uncond_local', 'cuda')

sm = np.load(str(ROOT / 'ref_repo/StableMotion/output/'
                     'brokenamass_star_sm_vanilla/results.npy'),
             allow_pickle=True).item()
corrupted = sm['motion']
lengths = np.asarray(sm['lengths']).reshape(-1)
CMP = ROOT / 'output/eval/brokenamass_star_repair_compare/npz'

NC = 15
TAUS = [0.1, 0.15, 0.2, 0.3]
THRS_RAD = [0.05, 0.10, 0.15, 0.20, 0.30, 0.45]   # ~2.9 / 5.7 / 8.6 / 11 / 17 / 26 deg

agg = {(tau, thr): {'cov': [], 'rec': [], 'prec': []}
       for tau in TAUS for thr in THRS_RAD}
chg_pct = {tau: [] for tau in TAUS}
gt_cov = []

for i in range(NC):
    sd = {k: (v.float() if torch.is_tensor(v)
              else torch.from_numpy(np.asarray(v)).float())
          for k, v in corrupted[i].items()}
    L = int(min(lengths[i], sd['poses'].shape[0]))
    sd = {k: v[:L] for k, v in sd.items()}
    m135_20 = smpldata_to_m2m135(sd, bone_offsets)
    T30 = max(2, int(round(L * 30.0 / 20.0)))
    m135_30 = _resample_motion135_slerp(m135_20, T30)
    motion_t = torch.from_numpy(np.asarray(m135_30, np.float32)).cuda()
    cm = np.load(CMP / f'{i:05d}.npz', allow_pickle=True)['corruption_mask']
    Lc = min(L, cm.shape[0])
    cm = cm[:Lc].astype(bool)
    gt_cov.append(cm.mean())
    idx = np.clip(np.round(np.linspace(0, T30 - 1, L)).astype(int), 0, T30 - 1)
    for tau in TAUS:
        torch.manual_seed(1234 + i)
        torch.cuda.manual_seed_all(1234 + i)
        np.random.seed((1234 + i) & 0xFFFFFFFF)
        _ = pipeline.infer_repair(
            motion_t, lengths=[T30], mask_source='self_denoise',
            detect_metric='angle', detect_tau=tau,
            detect_joint_thr_rad=99.0,            # result irrelevant; we spy jchg
            translation_mode='lock', mask_granularity='joint', sdedit_tau=0.5,
            strict_tighten=True)
        jchg = _cap['j'][:T30]                    # (T30,22) radians
        chg_pct[tau].append([np.percentile(jchg, p) for p in (50, 75, 90)])
        for thr in THRS_RAD:
            jflag_raw = np.zeros((T30, 135), np.float32)
            jf = jchg > thr
            for j in range(22):
                jflag_raw[jf[:, j], 3 + j * 6:3 + (j + 1) * 6] = 1.0
            tight = compute_strict_adaptive_mask(
                jflag_raw, dilate=2, min_blob=3, motion_dim=135, lock_trans=True)
            jflag = (tight[:, 3:135].reshape(-1, 22, 6) >= 0.5).any(-1)[:T30]
            jm20 = jflag[idx][:Lc]
            inter = int((jm20 & cm).sum())
            agg[(tau, thr)]['cov'].append(jm20.mean())
            agg[(tau, thr)]['rec'].append(inter / max(int(cm.sum()), 1))
            agg[(tau, thr)]['prec'].append(inter / max(int(jm20.sum()), 1))
    print(f'[{i+1}/{NC}] done', flush=True)

print(f'\n=== ANGLE detect sweep ({NC} cases). '
      f'GT corruption cov = {np.mean(gt_cov)*100:.1f}% ===')
for tau in TAUS:
    pc = np.mean(chg_pct[tau], axis=0)
    print(f'\n-- detect_tau={tau}  angle(rad) p50/p75/p90 = '
          f'{pc[0]:.3f}/{pc[1]:.3f}/{pc[2]:.3f}  '
          f'({np.degrees(pc[0]):.1f}/{np.degrees(pc[1]):.1f}/'
          f'{np.degrees(pc[2]):.1f} deg)')
    print(f"   {'thr_rad':>8} {'deg':>5} {'cov%':>7} {'recall%':>8} {'prec%':>7}")
    for thr in THRS_RAD:
        a = agg[(tau, thr)]
        print(f'   {thr:8.2f} {np.degrees(thr):5.1f} '
              f'{np.mean(a["cov"])*100:7.1f} {np.mean(a["rec"])*100:8.1f} '
              f'{np.mean(a["prec"])*100:7.1f}', flush=True)
