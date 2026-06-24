#!/usr/bin/env python3
"""Verify the HY-Motion jitter is a 'rel-transl decoded as abs' bug.

HY-Motion T2M trains with transl_type='rel' (per-frame root displacement:
delta[t] = abs[t]-abs[t-1], delta[0]=0). But decode_motion_from_latent uses
latent_denorm[...,0:3] directly as ABSOLUTE translation -- no cumsum. So the
viewer plots the velocity curve as if it were a position -> high-freq jitter.

Two checks (no GPU):
1. Simulate the bug on GT: take GT abs root, diff it to 'rel', plot rel as abs,
   measure jerk. Should explode ~like HY.
2. Fix HY: cumsum the decoded HY root, measure jerk. Should drop to ~GT level.
"""
from __future__ import annotations

import argparse
import numpy as np
from pathlib import Path


def _jerk(x):
    return float(np.linalg.norm(np.diff(x, n=3, axis=0), axis=-1).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hy_dir", default="outputs/evaluation/hymotion_h3d272/hy_272")
    ap.add_argument("--n", type=int, default=60)
    args = ap.parse_args()

    from hftrainer.evaluation.evaluators.motionstreamer_272 import MotionStreamer272Evaluator
    try:
        from hftrainer.datasets.motion.representation.humanml_repr import recover_local_rotations_and_root
    except Exception:
        from hftrainer.models.motion.components.utils.humanml_repr import recover_local_rotations_and_root

    ev = MotionStreamer272Evaluator(device="cpu")
    pairs = ev.load_test_pairs()
    hy_dir = Path(args.hy_dir)

    def root_of(m):
        _, root = recover_local_rotations_and_root(np.asarray(m, np.float32))
        return np.asarray(root, np.float32)  # (T,3) absolute (in 272 space)

    def xz(a):
        return _jerk(a[:, None, [0, 2]])

    gt_xz, gt_bug_xz, hy_xz, hy_fix_xz = [], [], [], []
    gt_all, gt_bug_all, hy_all, hy_fix_all = [], [], [], []
    seen = set()
    for idx, (name, cap, gt, ml) in enumerate(pairs):
        if name in seen:
            continue
        hf = hy_dir / f"{idx:06d}.npy"
        if not hf.exists():
            continue
        seen.add(name)
        if len(seen) > args.n:
            break
        # --- GT native (abs) + simulated bug (diff -> treat as abs) ---
        rg = root_of(gt)
        rel = np.concatenate([np.zeros((1, 3), np.float32), rg[1:] - rg[:-1]], 0)
        gt_xz.append(xz(rg)); gt_bug_xz.append(xz(rel))
        gt_all.append(_jerk(rg[:, None])); gt_bug_all.append(_jerk(rel[:, None]))
        # --- HY decoded root (raw) + fix (cumsum -> abs) ---
        rh = root_of(np.load(hf))
        rh_fix = np.cumsum(rh, axis=0)
        hy_xz.append(xz(rh)); hy_fix_xz.append(xz(rh_fix))
        hy_all.append(_jerk(rh[:, None])); hy_fix_all.append(_jerk(rh_fix[:, None]))

    def m(x):
        return float(np.mean(x))

    print(f"\n=== rel-as-abs bug verification ({len(seen)} samples) ===")
    print(f"{'variant':<34}{'root_jerk':>12}{'rootXZ_jerk':>14}")
    print(f"{'GT native (abs, correct)':<34}{m(gt_all):>12.5f}{m(gt_xz):>14.5f}")
    print(f"{'GT diff->as-abs (simulated bug)':<34}{m(gt_bug_all):>12.5f}{m(gt_bug_xz):>14.5f}")
    print(f"{'HY decoded root (raw, buggy)':<34}{m(hy_all):>12.5f}{m(hy_xz):>14.5f}")
    print(f"{'HY cumsum->abs (proposed fix)':<34}{m(hy_fix_all):>12.5f}{m(hy_fix_xz):>14.5f}")
    print("\n--- key ratios ---")
    print(f"GT-bug / GT-native (XZ)  = {m(gt_bug_xz)/max(m(gt_xz),1e-9):>6.1f}x  "
          f"(should resemble HY's ~32x if bug mechanism is correct)")
    print(f"HY-raw / HY-fix (XZ)     = {m(hy_xz)/max(m(hy_fix_xz),1e-9):>6.1f}x  "
          f"(cumsum should restore smoothness)")


if __name__ == "__main__":
    main()
