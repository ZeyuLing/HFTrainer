#!/usr/bin/env python3
"""Isolate OmniControl's spatial-guidance optimisation from the diffusion model.
Take a real GT motion x0 (normalised 263), set a FAR pelvis target at the last
frame, and run ONLY the guide gradient-descent (recover_from_ric + autograd,
no model forward). If pure GD pulls the pelvis onto the target -> the guidance
math is sound and the diffusion model fights it; if not -> grad/recover broken."""
from __future__ import annotations
import os, sys, types
from pathlib import Path
import numpy as np
for _n, _v in {"bool": bool, "float": float, "int": int, "object": object,
               "str": str, "complex": complex, "unicode": str}.items():
    if not hasattr(np, _n):
        setattr(np, _n, _v)
import torch
ROOT = Path("/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer")
OMNI = ROOT / "ref_repo/OmniControl"; CONDMDI = ROOT / "ref_repo/CondMDI"
stage = ROOT / "output/evaluation/_omni_official/_stage"
os.chdir(str(stage)); sys.path.insert(0, str(OMNI))
from data_loaders.humanml.scripts.motion_process import recover_from_ric  # noqa
SID = os.environ.get("SID", "M005213")
dev = "cuda"
mean = torch.from_numpy(np.load(str(OMNI / "dataset/t2m_mean.npy"))).float().to(dev)
std = torch.from_numpy(np.load(str(OMNI / "dataset/t2m_std.npy"))).float().to(dev)
raw_mean = torch.from_numpy(np.load(str(OMNI / "dataset/humanml_spatial_norm/Mean_raw.npy"))).float().to(dev)
raw_std = torch.from_numpy(np.load(str(OMNI / "dataset/humanml_spatial_norm/Std_raw.npy"))).float().to(dev)

m = np.load(str(CONDMDI / "dataset/HumanML3D/new_joint_vecs" / f"{SID}.npy")).astype(np.float32)
L = (min(len(m), 196) // 4) * 4
m = m[:L]
gt_joints = recover_from_ric(torch.from_numpy(m).float(), 22).numpy()
ctrl = L - 1
tp = gt_joints[ctrl, 0].copy()
print(f"[{SID}] L={L} ctrl={ctrl} GT pelvis@ctrl={tp} disp={np.linalg.norm(tp-gt_joints[0,0]):.3f} m")

# normalised motion x (263) padded to 196; x0 = GT itself (so initial pelvis = GT)
x0 = (torch.from_numpy(m).float().to(dev) - mean) / std         # (L,263)
x = torch.zeros(196, 263, device=dev); x[:L] = x0
# OmniControl tensor layout: [B,263,1,T]
x = x.permute(1, 0)[None, :, None, :].contiguous()              # (1,263,1,196)

# target: keep GT pelvis BUT move it far (add +2m in X, flip Z) to test reach
tgt = tp.copy(); tgt[0] += 2.0; tgt[2] = -tgt[2]
hint = np.zeros((196, 22, 3), np.float32); hint[ctrl, 0] = tgt
hint_t = torch.from_numpy(hint).to(dev)
mask_hint = (hint_t.abs().sum(-1, keepdim=True) != 0).view(1, 196, 22, 1)
print(f"FAR target pelvis@ctrl={tgt}  (init err={np.linalg.norm(tp-tgt):.3f} m)")

def recover_pelvis(xx):
    x_ = xx.permute(0, 3, 2, 1).squeeze(2) * std + mean      # (1,196,263)
    return recover_from_ric(x_, 22)[0, ctrl, 0]

# pure GD using the SAME math as guide() (scale=20 for 1 keyframe, var path)
scale = 20.0
for var, label, steps in [(0.01, "low-t var floor", 2000), (0.5, "high-t var", 2000)]:
    xx = x.clone()
    for it in range(steps):
        xx.requires_grad_(True)
        x_ = xx.permute(0, 3, 2, 1).squeeze(2) * std + mean
        jp = recover_from_ric(x_, 22)
        loss = torch.norm((jp - hint_t.view(1, 196, 22, 3)) * mask_hint, dim=-1).sum()
        g = torch.autograd.grad(loss, xx)[0]
        g[..., 0] = 0
        g = var * g
        xx = (xx - scale * g).detach()
        if it % 400 == 0 or it == steps - 1:
            with torch.no_grad():
                p = recover_pelvis(xx).cpu().numpy()
            print(f"  [{label}] it={it:4d} pelvis_err={np.linalg.norm(p-tgt):.4f} m  pelvis={np.round(p,3)}")
