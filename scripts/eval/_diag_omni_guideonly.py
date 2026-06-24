#!/usr/bin/env python3
"""Isolate OmniControl's guide() loop from the model: can pure gradient guidance
pull a random normalized motion's pelvis onto a target hint? Replays the EXACT
guide math (scale=20/maxkf, grad=model_variance*grad, x-=scale*grad) at a low t.

loss drops a lot  -> guidance numerics fine; failure is in the control branch.
loss stays flat   -> guidance step size/numerics broken under this torch.
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

ROOT = Path("/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer")
OMNI = ROOT / "ref_repo/OmniControl"
CONDMDI = ROOT / "ref_repo/CondMDI"
stage = ROOT / "output/evaluation/_omni_official/_stage"
os.chdir(str(stage))
sys.path.insert(0, str(OMNI))
from data_loaders.humanml.scripts.motion_process import recover_from_ric  # noqa
import types

device = "cuda" if torch.cuda.is_available() else "cpu"
print("torch", torch.__version__, "dev", device)

mean = torch.from_numpy(np.load("dataset/HumanML3D/Mean.npy")).float().to(device)
std = torch.from_numpy(np.load("dataset/HumanML3D/Std.npy")).float().to(device)
raw_mean = torch.from_numpy(np.load("dataset/humanml_spatial_norm/Mean_raw.npy")).float().to(device)
raw_std = torch.from_numpy(np.load("dataset/humanml_spatial_norm/Std_raw.npy")).float().to(device)

T = 196
# target pelvis path: a circle on the ground (clearly different from a forward walk)
tt = np.linspace(0, 2 * np.pi, T)
target_pelvis = np.stack([np.cos(tt) - 1.0, np.zeros(T) + 0.9, np.sin(tt)], 1).astype(np.float32)  # (T,3)
hint = np.zeros((1, T, 22, 3), dtype=np.float32)
hint[0, :, 0, :] = target_pelvis
# normalize hint as the pipeline expects
hint_n = (hint - raw_mean.view(22, 3).cpu().numpy()) / raw_std.view(22, 3).cpu().numpy()
hint_n = hint_n.reshape(1, T, 66)
hint_t = torch.from_numpy(hint_n).float().to(device)

# random normalized motion (263), shape (B,263,1,T) like p_sample mean
x = (torch.randn(1, 263, 1, T, device=device) * 1.0)

n_joint = 22
mask_hint = hint_t.view(1, T, n_joint, 3).sum(-1, keepdim=True) != 0
hint_m = (hint_t * raw_std.view(1, 1, 66) + raw_mean.view(1, 1, 66)).view(1, T, n_joint, 3) * mask_hint

def gradients(x):
    with torch.enable_grad():
        x.requires_grad_(True)
        x_ = x.permute(0, 3, 2, 1).contiguous().squeeze(2)
        x_ = x_ * std + mean
        jp = recover_from_ric(x_, 22)
        loss = torch.norm((jp - hint_m) * mask_hint, dim=-1)
        grad = torch.autograd.grad([loss.sum()], [x])[0]
        grad[..., 0] = 0
    return loss, grad.detach()

def cur_err():
    with torch.no_grad():
        x_ = x.permute(0, 3, 2, 1).contiguous().squeeze(2) * std + mean
        jp = recover_from_ric(x_, 22)
        d = torch.norm((jp - hint_m) * mask_hint, dim=-1)
        return (d.sum() / mask_hint.sum()).item()

# scale = 20/max_keyframes (dense -> 196)
max_kf = mask_hint.sum(1).squeeze(-1).max(1)[0]
scale = (20.0 / max_kf).view(1, 1, 1, 1)
model_variance = torch.tensor(0.01, device=device)   # low-t floor

print(f"max_keyframes={int(max_kf.item())} scale={scale.item():.4f}")
print(f"[init] mean pelvis err = {cur_err():.4f} m")
for step in range(500):
    loss, grad = gradients(x)
    grad = model_variance * grad
    x = (x - scale * grad).detach()
    if step % 50 == 0 or step == 499:
        print(f"  step {step:3d}: loss.sum={loss.sum().item():10.2f}  grad.norm={grad.norm().item():.4f}  pelvis_err={cur_err():.4f} m")
print(f"[final] mean pelvis err = {cur_err():.4f} m  (target=circle; <0.05 => guidance works)")
