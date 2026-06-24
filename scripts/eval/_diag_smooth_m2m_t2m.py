#!/usr/bin/env python3
"""Diagnostic: apply the official HY-Motion-T2M decode smoothing (SLERP rot6d +
Savitzky-Golay transl) to already-generated HYMotion-M2M t2m-only ``motion_135``
predictions, re-encode to MS272, and re-evaluate.

Goal: isolate how much of the FID gap between HY-Motion-T2M-1.0-Lite (smoothed,
FID ~11.7) and the M2M t2m-only specialist (unsmoothed, FID ~25.4) is caused by
the M2M pipeline NOT applying the official temporal smoothing that the T2M
pipeline applies by default.

Usage:
    PYTHONPATH=$PWD python3 scripts/eval/_diag_smooth_m2m_t2m.py \
        --m135-dir outputs/.../t2m_only_from_lite_ep18_cfg5.0/m135 \
        --out-dir  outputs/.../t2m_only_from_lite_ep18_cfg5.0/pred272_smoothed \
        --workers 8
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _smooth_one(args):
    sid, src, dst = args
    import torch
    from hftrainer.models.motion.hymotion_t2m._smoothing import (
        smooth_with_savgol,
        smooth_with_slerp,
    )
    from hftrainer.motion.representation.motion272 import motion135_to_272

    m135 = np.load(src).astype(np.float32)
    L = m135.shape[0]
    if L < 5:
        return sid, "too_short"
    transl = torch.from_numpy(m135[:, :3]).float().unsqueeze(0)        # (1,L,3)
    rot6d = torch.from_numpy(m135[:, 3:135]).float().reshape(1, L, 22, 6)
    rot6d_s = smooth_with_slerp(rot6d, sigma=1.0)                      # (1,L,22,6)
    transl_s = smooth_with_savgol(transl, window_length=11, polyorder=5)
    m135_s = np.concatenate(
        [transl_s[0].cpu().numpy(), rot6d_s[0].cpu().numpy().reshape(L, 132)],
        axis=-1,
    ).astype(np.float32)
    m272 = motion135_to_272(m135_s, rotation_space="local")
    np.save(dst, m272.astype(np.float32))
    return sid, "ok"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--m135-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    m135_dir = Path(args.m135_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(m135_dir.glob("*.npy"))
    if args.limit:
        files = files[: args.limit]
    tasks = [
        (f.stem, str(f), str(out_dir / f.name))
        for f in files
        if not (out_dir / f.name).exists()
    ]
    print(f"[smooth] {len(tasks)} to smooth (of {len(files)}); workers={args.workers}",
          flush=True)

    ok = 0
    fail = 0
    if args.workers <= 1:
        for t in tasks:
            _sid, st = _smooth_one(t)
            ok += st == "ok"
            fail += st != "ok"
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for i, (_sid, st) in enumerate(ex.map(_smooth_one, tasks, chunksize=8)):
                ok += st == "ok"
                fail += st != "ok"
                if (i + 1) % 200 == 0:
                    print(f"  [{i+1}/{len(tasks)}] ok={ok} fail={fail}", flush=True)
    print(f"[smooth] done ok={ok} fail={fail} -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
