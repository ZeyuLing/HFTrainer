#!/usr/bin/env python3
"""Test whether bf16 inference quantization of ABSOLUTE translation explains
HY-Motion's root-XZ jitter.

HY stats: Std(transl)=[0.58,0.14,0.80] (metre-scale => absolute, not rel).
The flow ODE runs in bf16 on the NORMALIZED motion; transl normalized ~O(1),
so bf16's ~2^-8 mantissa injects per-frame position noise after denorm.

Simulate: take GT smooth abs root (from 272 decode), normalize with HY std,
round-trip through bf16, denormalize, measure XZ jerk vs original.
"""
from __future__ import annotations
import argparse
import numpy as np
import torch
from pathlib import Path


def _jerk_xz(a):
    return float(np.linalg.norm(np.diff(a[:, [0, 2]], n=3, axis=0), axis=-1).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hy_dir", default="outputs/evaluation/hymotion_h3d272/hy_272")
    ap.add_argument("--n", type=int, default=60)
    args = ap.parse_args()
    from hftrainer.datasets.motion.representation.humanml_repr import recover_local_rotations_and_root

    std = np.array([0.5796, 0.1396, 0.804], np.float32)  # HY transl std
    mean = np.array([-0.0023, 1.0901, 0.1103], np.float32)

    def root_of(p):
        _, r = recover_local_rotations_and_root(np.load(p).astype(np.float32))
        return np.asarray(r, np.float32)

    hy = sorted(Path(args.hy_dir).glob("*.npy"))[: args.n]
    gt_raw, gt_bf16, hy_raw = [], [], []
    for p in hy:
        # We use HY's own decoded root only as the "HY observed" reference;
        # for the GT-smooth surrogate we synthesize a SMOOTH abs path by
        # low-pass filtering HY's root (removes the suspected bf16 noise), then
        # re-inject bf16 to see if jitter returns.
        rh = root_of(p)
        hy_raw.append(_jerk_xz(rh))
        # smooth surrogate: moving-average (window 5) -> "what a clean model gives"
        k = 5
        pad = np.pad(rh, ((k // 2, k // 2), (0, 0)), mode="edge")
        smooth = np.stack([np.convolve(pad[:, i], np.ones(k) / k, "valid") for i in range(3)], 1)
        gt_raw.append(_jerk_xz(smooth))
        # bf16 round-trip in normalized space
        norm = (smooth - mean) / std
        norm_q = torch.from_numpy(norm).to(torch.bfloat16).float().numpy()
        recon = norm_q * std + mean
        gt_bf16.append(_jerk_xz(recon))

    m = lambda x: float(np.mean(x))
    print(f"\n=== bf16-on-abs-transl test ({len(hy)} samples) ===")
    print(f"smoothed surrogate (clean)        XZ_jerk = {m(gt_raw):.5f}")
    print(f"smoothed + bf16 round-trip        XZ_jerk = {m(gt_bf16):.5f}")
    print(f"HY actual decoded root            XZ_jerk = {m(hy_raw):.5f}")
    print(f"\nbf16 / clean        = {m(gt_bf16)/max(m(gt_raw),1e-9):.1f}x")
    print(f"HY actual / clean   = {m(hy_raw)/max(m(gt_raw),1e-9):.1f}x")
    print("(if bf16/clean ~ HY/clean, bf16 quantization explains the jitter)")


if __name__ == "__main__":
    main()
