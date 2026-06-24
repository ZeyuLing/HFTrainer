#!/usr/bin/env python3
"""Diagnostic: is the Guo-evaluator FID blow-up on PerMo a small-sample artifact?

Converts N real PerMo style motions to HML263, encodes them with the Guo motion
encoder, and computes FID between two disjoint halves of REAL motions for a few
sample sizes. Real-vs-real FID should be ~0 if the estimator is well-conditioned;
if it explodes at small N (e.g. 120) but settles at large N (e.g. >512), the
earlier FID=5e20 is purely the 512-dim covariance being rank-deficient.
"""
import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path("/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer")
sys.path.insert(0, str(_REPO / "ref_repo/Momask/momask-codes"))
sys.path.insert(0, str(_REPO))

import utils.motion_process  # noqa: F401,E402

from hftrainer.datasets.motion.motionhub.transforms.load_smplx import (  # noqa: E402
    process_smplx_pose, process_transl,
)
from hftrainer.motion.representation.convert import (  # noqa: E402
    motion135_to_motion272, motion272_to_hml263,
)
from hftrainer.evaluation.evaluators.humanml3d_263 import HumanML263Evaluator  # noqa: E402
from hftrainer.evaluation.evaluators.t2m_metrics import activation_stats, calc_frechet  # noqa: E402

PERMO = _REPO / "data/hymotion_data/PerMo/PerMo/20260513/motions/train"


def npz_to_135(p):
    d = np.load(p, allow_pickle=True)
    return np.asarray(d["motion_135"], np.float32)


def to263(m135):
    m272 = motion135_to_motion272(np.asarray(m135, np.float32))
    out = motion272_to_hml263(m272, joints_from="smpl_fk")
    m = out[0] if isinstance(out, tuple) else out
    return np.asarray(m, np.float32)


def main():
    ev = HumanML263Evaluator(device="cuda"); ev._ensure_loaded()
    files = sorted(p for p in PERMO.glob("*.npz") if not p.name.startswith("Neutral_"))
    rng = np.random.default_rng(0); rng.shuffle(files)
    files = files[:600]
    embs = []
    for i, p in enumerate(files):
        try:
            m263 = to263(npz_to_135(str(p)))
        except Exception as e:  # noqa: BLE001
            if len(embs) == 0 and i < 3:
                print(f"[convfail] {p.name}: {type(e).__name__}: {e}")
            continue
        L = min(len(m263), 196)
        if L < 40:
            continue
        t_eff = (L // ev.unit_length) * ev.unit_length
        m = ev._pad_norm(m263, t_eff)
        mt = torch.from_numpy(m).unsqueeze(0).float().to(ev.device)
        with torch.no_grad():
            mov = ev._movement_enc(mt[..., :-4])
            e = ev._motion_enc(mov, torch.tensor([t_eff // ev.unit_length]))
        embs.append(e.cpu().numpy()[0])
        if len(embs) >= 520:
            break
    embs = np.stack(embs)
    print(f"[info] encoded {len(embs)} real PerMo motions, emb dim {embs.shape[1]}")
    for n in [60, 120, 200, 256]:
        if 2 * n > len(embs):
            continue
        a, b = embs[:n], embs[n:2 * n]
        mu1, c1 = activation_stats(a); mu2, c2 = activation_stats(b)
        try:
            fid = float(calc_frechet(mu1, c1, mu2, c2))
        except Exception as e:  # noqa: BLE001
            fid = f"ERR {e}"
        print(f"  real-vs-real FID (N={n} per side): {fid}")


if __name__ == "__main__":
    main()
