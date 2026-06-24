#!/usr/bin/env python3
"""Decisive test: is the autograd gradient through OmniControl's recover_from_ric
correct under our torch 2.5? Compare autograd vs finite-difference for d(pelvis_pos)
w.r.t the 263-dim motion (esp. root rot/vel channels 0,1,2,3).

If autograd disagrees with finite-diff -> in-place ops break the gradient ->
that is exactly why spatial guidance (x <- x - scale*grad) fails to move the root.
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np
for _n, _v in {"bool": bool, "float": float, "int": int, "object": object,
               "str": str, "complex": complex, "unicode": str}.items():
    if not hasattr(np, _n):
        setattr(np, _n, _v)
import torch

OMNI = Path("/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/ref_repo/OmniControl")
sys.path.insert(0, str(OMNI))
from data_loaders.humanml.scripts.motion_process import recover_from_ric  # noqa: E402

torch.manual_seed(0)
print("torch", torch.__version__)

T = 40
data = torch.randn(1, T, 263, dtype=torch.float32) * 0.1   # small motion
data = data.clone()

def pelvis_path(x):
    pos = recover_from_ric(x, 22)        # (1,T,22,3)
    return pos[:, :, 0, :]               # pelvis (1,T,3)

# ---- autograd gradient of sum(pelvis) w.r.t data ----
x = data.clone().requires_grad_(True)
p = pelvis_path(x)
target = p.sum()
g_auto = torch.autograd.grad(target, x)[0]   # (1,T,263)

# ---- finite-difference on a few channels per frame ----
eps = 2e-3
def loss_only(x):
    with torch.no_grad():
        return pelvis_path(x).sum().item()

base = loss_only(data)
checks = [(5, 0), (5, 1), (5, 2), (5, 3), (10, 1), (10, 2), (20, 1), (20, 2)]
print(f"{'frame,ch':>10} | {'autograd':>12} | {'finite-diff':>12} | {'abs.err':>10}")
maxerr = 0.0
for (f, c) in checks:
    xp = data.clone(); xp[0, f, c] += eps
    xm = data.clone(); xm[0, f, c] -= eps
    fd = (loss_only(xp) - loss_only(xm)) / (2 * eps)
    ga = g_auto[0, f, c].item()
    err = abs(fd - ga)
    maxerr = max(maxerr, err)
    print(f"  ({f:>3},{c:>2}) | {ga:12.5f} | {fd:12.5f} | {err:10.5f}")

print(f"\nmax abs err over checked (frame,channel) = {maxerr:.6f}")
print("[verdict]", "AUTOGRAD BROKEN (in-place ops)" if maxerr > 1e-2
      else "autograd matches finite-diff (recover ok)")

# also report norm split: pelvis-relevant root channels vs rest, for full grad
gn = g_auto[0]                       # (T,263)
root_norm = gn[:, :4].norm().item()
rest_norm = gn[:, 4:].norm().item()
print(f"grad norm: root(0:4)={root_norm:.4f}  rest(4:)={rest_norm:.4f}")
