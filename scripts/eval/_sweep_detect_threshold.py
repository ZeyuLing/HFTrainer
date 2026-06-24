"""Self-denoise detection threshold sweep for HyMotionM2MPipeline.infer_repair.

Loads the M2M model once; for each case runs the stage-1 self-denoise projection
ONCE (detect_tau=0.5), captures the (normalized) change map mn-stage1 via a spy
on compute_ada_keep_mask, then re-derives the per-joint defect mask at many
thresholds (cheap, CPU) and scores coverage / recall / precision against the
ground-truth corruption mask (corrupted-vs-clean deviation, from the compare
NPZ). This tells us which threshold catches the real corruption.
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer')
sys.path.insert(0, str(ROOT))

import hftrainer.pipelines.motion.hymotion_m2m_pipeline as PMOD  # noqa: E402
from hftrainer.pipelines.motion.repair_utils import (  # noqa: E402
    compute_ada_keep_mask, compute_strict_adaptive_mask,
)
from scripts.eval.run_ours_repair_brokenamass import build_model  # noqa: E402
from scripts.run_stablemotion_e9 import (  # noqa: E402
    smpldata_to_m2m135, _resample_motion135_slerp,
)

bone_offsets = torch.load(
    str(ROOT / 'data/hymotion_m2m_data/bone_offsets_22.pt'),
    map_location='cpu', weights_only=False).float()

_cap = {}
_orig = PMOD.compute_ada_keep_mask
def _spy(mn, stage1, **kw):
    _cap['mn'] = np.asarray(mn).copy()
    _cap['stage1'] = np.asarray(stage1).copy()
    return _orig(mn, stage1, **kw)
PMOD.compute_ada_keep_mask = _spy

bundle, pipeline, info, ckpt = build_model('uncond_local', 'cuda')

sm = np.load(str(ROOT / 'ref_repo/StableMotion/output/'
                     'brokenamass_star_sm_vanilla/results.npy'),
             allow_pickle=True).item()
corrupted = sm['motion']
lengths = np.asarray(sm['lengths']).reshape(-1)
CMP = ROOT / 'output/eval/brokenamass_star_repair_compare/npz'

NC = 10
TAUS = [0.1, 0.2, 0.3, 0.5]
# per-tau we test both abs thresholds and topk_pct coverages
ABS_THRS = [0.02, 0.04, 0.06, 0.08, 0.12, 0.16]
TOPK = [0.2, 0.3, 0.4]               # target fraction of cells flagged


def score(mn, stage1, model_dim, T30, idx, cm, Lc, mode, val):
    raw = compute_ada_keep_mask(mn, stage1, threshold_mode=mode, threshold=val)
    tight = compute_strict_adaptive_mask(
        raw, dilate=2, min_blob=3, motion_dim=model_dim, lock_trans=True)
    jflag = (tight[:, 3:135].reshape(-1, 22, 6) >= 0.5).any(-1)[:T30]
    jm20 = jflag[idx][:Lc]
    inter = int((jm20 & cm).sum())
    return (jm20.mean(),
            inter / max(int(cm.sum()), 1),
            inter / max(int(jm20.sum()), 1))


configs = ([('abs', t) for t in ABS_THRS] + [('topk_pct', k) for k in TOPK])
agg = {(tau, m, v): {'cov': [], 'rec': [], 'prec': []}
       for tau in TAUS for (m, v) in configs}
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
            translation_mode='lock', mask_granularity='joint', sdedit_tau=0.5,
            detect_tau=tau, detect_threshold=0.07, strict_tighten=True)
        mn = _cap['mn']
        stage1 = _cap['stage1']
        model_dim = mn.shape[-1]
        ch = np.abs(mn[:T30] - stage1[:T30])
        chg_pct[tau].append([np.percentile(ch, p) for p in (50, 75, 90)])
        for (m, v) in configs:
            c, r, p = score(mn, stage1, model_dim, T30, idx, cm, Lc, m, v)
            agg[(tau, m, v)]['cov'].append(c)
            agg[(tau, m, v)]['rec'].append(r)
            agg[(tau, m, v)]['prec'].append(p)
    print(f'[{i+1}/{NC}] done', flush=True)

print(f'\n=== detect sweep ({NC} cases). GT corruption cov = '
      f'{np.mean(gt_cov)*100:.1f}% ===')
for tau in TAUS:
    pc = np.mean(chg_pct[tau], axis=0)
    print(f'\n-- detect_tau={tau}  change |mn-stage1| p50/p75/p90 = '
          f'{pc[0]:.3f}/{pc[1]:.3f}/{pc[2]:.3f}')
    print(f"   {'mode':>9} {'val':>5} {'cov%':>7} {'recall%':>8} {'prec%':>7}")
    for (m, v) in configs:
        a = agg[(tau, m, v)]
        print(f'   {m:>9} {v:5.2f} {np.mean(a["cov"])*100:7.1f} '
              f'{np.mean(a["rec"])*100:8.1f} {np.mean(a["prec"])*100:7.1f}',
              flush=True)
