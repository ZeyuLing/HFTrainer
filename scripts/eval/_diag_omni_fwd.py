#!/usr/bin/env python3
"""Forward-level probe: does the hint actually change OmniControl's prediction?
Prints control-branch norm and the delta between hint vs zero-hint outputs."""
from __future__ import annotations
import os, sys, types
from pathlib import Path
import numpy as np
for _n, _v in {"bool": bool, "int": int, "float": float, "complex": complex,
               "object": object, "str": str, "unicode": str}.items():
    if not hasattr(np, _n):
        setattr(np, _n, _v)
import torch

ROOT = Path("/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer")
OMNI = ROOT / "ref_repo/OmniControl"
sys.path.insert(0, str(ROOT / "scripts/eval"))
from omnicontrol_run_bodypart import _build_args, _stage
out = ROOT / "output/evaluation/_diag_omni"; out.mkdir(parents=True, exist_ok=True)
stage = _stage(out); os.chdir(str(stage))
sys.path.insert(0, str(OMNI))
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from utils.fixseed import fixseed
fixseed(0)
device = "cuda"; MF = 196
args = _build_args(str(OMNI / "save/omnicontrol_ckpt/model_humanml3d.pt"), 1.0)
model, diffusion = create_model_and_diffusion(args, types.SimpleNamespace(dataset=types.SimpleNamespace()))
load_model_wo_clip(model, torch.load(args.model_path, map_location="cpu"))
model.to(device); model.eval()

raw_mean = diffusion.raw_mean.view(1, 22, 3).numpy().astype(np.float32)
raw_std = diffusion.raw_std.view(1, 22, 3).numpy().astype(np.float32)
from utils.text_control_example import spiral_forward
path = spiral_forward(MF)[:, :3].astype(np.float32)
gtj = np.zeros((MF, 22, 3), np.float32); gtj[:, 0] = path
gn = (gtj - raw_mean) / raw_std
h = np.zeros((MF, 22, 3), np.float32); h[:, 0, :] = gn[:, 0, :]
hint = torch.from_numpy(h.reshape(1, MF, 66)).to(device)
zero = torch.zeros_like(hint)

x = torch.randn(1, 263, 1, MF, device=device)
t = torch.tensor([500], device=device)
ymask = torch.ones((1, 1, 1, MF), dtype=torch.bool, device=device)

# probe control-branch norm
caps = {}
orig = model.cmdm_forward
def probe(xx, ts, y=None, weight=1.0):
    c = orig(xx, ts, y, weight)
    sm = (y['hint'].sum(-1) != 0).float().mean().item()
    caps['ctrl'] = float(c.norm().item()); caps['seqmask_frac'] = sm
    gh = model.input_hint_block(y['hint'].float())
    caps['guided_hint'] = float(gh.norm().item())
    return c
model.cmdm_forward = probe

with torch.no_grad():
    y_h = {"mask": ymask, "lengths": torch.tensor([MF], device=device), "text": ["a person walks"], "hint": hint}
    out_h = model(x, t, y_h)
    print(f"[fwd] WITH hint: control_norm={caps['ctrl']:.3f} guided_hint_norm={caps['guided_hint']:.3f} seqmask_frac={caps['seqmask_frac']:.3f}")
    y_z = {"mask": ymask, "lengths": torch.tensor([MF], device=device), "text": ["a person walks"], "hint": zero}
    out_z = model(x, t, y_z)
    print(f"[fwd] ZERO hint: control_norm={caps['ctrl']:.3f} guided_hint_norm={caps['guided_hint']:.3f}")
    d = (out_h - out_z).norm().item(); rel = d / out_z.norm().item()
    print(f"[fwd] ||out_hint - out_zero|| = {d:.4f}  (relative {rel:.4%})  out_norm={out_z.norm().item():.2f}")
    # how much does the ROOT trajectory channels (0,1,2,3) of x0 differ?
    droot = (out_h[:, :4] - out_z[:, :4]).norm().item()
    print(f"[fwd] root-channel(0:4) delta norm = {droot:.4f}")
