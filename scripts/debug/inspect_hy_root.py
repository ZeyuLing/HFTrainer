#!/usr/bin/env python3
"""Inspect what the decoded HY-Motion root trajectory actually is."""
from __future__ import annotations
import numpy as np
from pathlib import Path

from hftrainer.datasets.motion.representation.humanml_repr import recover_local_rotations_and_root

hy = sorted(Path("outputs/evaluation/hymotion_h3d272/hy_272").glob("*.npy"))
g2z = Path("outputs/evaluation/motionmillion_h3d272/mm_272_len150")

def root_of(p):
    _, r = recover_local_rotations_and_root(np.load(p).astype(np.float32))
    return np.asarray(r, np.float32)

for p in hy[:3]:
    rh = root_of(p)
    rc = np.cumsum(rh, axis=0)
    print(f"\n### {p.name}  T={len(rh)}")
    for label, a in (("HY raw root", rh), ("HY cumsum", rc)):
        xz = a[:, [0, 2]]
        span = xz.max(0) - xz.min(0)
        step = np.linalg.norm(np.diff(xz, axis=0), axis=1)
        print(f"  {label:<12} XZspan=({span[0]:.3f},{span[1]:.3f}) "
              f"y[min,max]=({a[:,1].min():.2f},{a[:,1].max():.2f}) "
              f"|dstep| mean={step.mean():.4f} max={step.max():.4f}")
    gp = g2z / p.name
    if gp.exists():
        rg = root_of(gp)
        xz = rg[:, [0, 2]]
        span = xz.max(0) - xz.min(0)
        step = np.linalg.norm(np.diff(xz, axis=0), axis=1)
        print(f"  {'G2Z root':<12} XZspan=({span[0]:.3f},{span[1]:.3f}) "
              f"y[min,max]=({rg[:,1].min():.2f},{rg[:,1].max():.2f}) "
              f"|dstep| mean={step.mean():.4f} max={step.max():.4f}")
    # print first 8 frames of HY raw xz to see structure
    print("  HY raw xz[:8]:", np.round(rh[:8, [0, 2]], 3).tolist())
