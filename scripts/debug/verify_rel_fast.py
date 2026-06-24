#!/usr/bin/env python3
"""Minimal, evaluator-free check that HY-Motion jitter is 'rel decoded as abs'.

Only needs the saved HY 272 files. For each: decode root, compare raw jerk vs
cumsum(root) jerk. If cumsum restores smoothness, the model output is per-frame
velocity (transl_type='rel') that decode_motion_from_latent wrongly used as abs.
"""
from __future__ import annotations
import argparse, sys
import numpy as np
from pathlib import Path


def _jerk(x):
    return float(np.linalg.norm(np.diff(x, n=3, axis=0), axis=-1).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hy_dir", default="outputs/evaluation/hymotion_h3d272/hy_272")
    ap.add_argument("--g2z_dir", default="outputs/evaluation/motionmillion_h3d272/mm_272_len150")
    ap.add_argument("--n", type=int, default=60)
    args = ap.parse_args()

    from hftrainer.datasets.motion.representation.humanml_repr import recover_local_rotations_and_root

    def root_of(p):
        _, root = recover_local_rotations_and_root(np.load(p).astype(np.float32))
        return np.asarray(root, np.float32)

    def xz(a):
        return _jerk(a[:, None, [0, 2]])

    hy = sorted(Path(args.hy_dir).glob("*.npy"))[: args.n]
    g2z_dir = Path(args.g2z_dir)
    raw, fix, g2z = [], [], []
    for p in hy:
        rh = root_of(p)
        raw.append(xz(rh)); fix.append(xz(np.cumsum(rh, axis=0)))
        gp = g2z_dir / p.name
        if gp.exists():
            g2z.append(xz(root_of(gp)))
    m = lambda x: float(np.mean(x)) if x else float("nan")
    print(f"\n=== HY rel-as-abs fast check ({len(hy)} HY samples) ===")
    print(f"HY decoded root  rootXZ_jerk (raw, buggy)   = {m(raw):.5f}")
    print(f"HY cumsum->abs   rootXZ_jerk (proposed fix) = {m(fix):.5f}")
    print(f"Go-to-Zero       rootXZ_jerk (native 272)   = {m(g2z):.5f}")
    print(f"\nHY raw / HY cumsum (XZ) = {m(raw)/max(m(fix),1e-9):.1f}x "
          f"(>>1 => cumsum removes the jitter => model output is velocity)")
    print(f"HY cumsum vs G2Z (XZ)   = {m(fix)/max(m(g2z),1e-9):.2f}x "
          f"(~1 => fixed HY root as smooth as native-272 baseline)")


if __name__ == "__main__":
    sys.exit(main())
