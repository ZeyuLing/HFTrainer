#!/usr/bin/env python3
"""Repair HML3D-263 outputs saved with feature-biased root/foot std.

Some generators output features normalized with canonical HumanML3D stats, but
our earlier export path denormalized them with the VQ/feature-biased stats where
root motion and foot-contact std are divided by 25. This keeps body pose mostly
reasonable while collapsing global root motion. The repair maps those channels
back to canonical scale without changing the remaining pose channels:

    fixed = (old - mean) / biased_std * canonical_std + mean
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--biased-mean", default="ref_repo/MotionGPT3/datasets/humanml3d/Mean.npy")
    ap.add_argument("--biased-std", default="ref_repo/MotionGPT3/datasets/humanml3d/Std.npy")
    ap.add_argument("--canonical-mean", default="ref_repo/TeSMo/dataset/HumanML3D/Mean.npy")
    ap.add_argument("--canonical-std", default="ref_repo/TeSMo/dataset/HumanML3D/Std.npy")
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def main():
    args = build_args()
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    b_mean = np.load(args.biased_mean).astype(np.float32)
    b_std = np.load(args.biased_std).astype(np.float32)
    c_mean = np.load(args.canonical_mean).astype(np.float32)
    c_std = np.load(args.canonical_std).astype(np.float32)
    if b_mean.shape != (263,) or b_std.shape != (263,) or c_mean.shape != (263,) or c_std.shape != (263,):
        raise ValueError("all mean/std files must have shape (263,)")
    if not np.allclose(b_mean, c_mean, atol=1e-6):
        raise ValueError("biased and canonical means differ; this repair assumes shared means")

    # Root rotation velocity, root x/z velocity, root height, and foot contacts.
    fix_idx = np.r_[0:4, 259:263]
    scale = np.ones(263, dtype=np.float32)
    scale[fix_idx] = c_std[fix_idx] / b_std[fix_idx]
    print(f"[setup] in={in_dir} out={out_dir} root_scale={scale[:4].tolist()} foot_scale={scale[259:263].tolist()}", flush=True)

    written = skipped = 0
    for src in sorted(in_dir.glob("*.npy")):
        dst = out_dir / src.name
        if dst.exists() and not args.overwrite:
            skipped += 1
            continue
        arr = np.load(src).astype(np.float32)
        if arr.ndim != 2 or arr.shape[-1] != 263:
            raise ValueError(f"{src}: expected (T,263), got {arr.shape}")
        fixed = (arr - b_mean[None, :]) * scale[None, :] + c_mean[None, :]
        np.save(dst, fixed.astype(np.float32))
        written += 1
        if written % 500 == 0:
            print(f"[progress] written={written} skipped={skipped}", flush=True)
    print(f"[done] written={written} skipped={skipped}", flush=True)


if __name__ == "__main__":
    main()
