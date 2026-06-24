#!/usr/bin/env python3
"""Instrument OmniControl's spatial guidance: force a single FAR controlled
keyframe (last valid frame) on a real test clip and log the guidance loss
(metres) vs diffusion timestep. Reveals whether guidance steadily pulls the
pelvis to a far target or plateaus (undershoot)."""
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
OMNI = ROOT / "ref_repo/OmniControl"
CONDMDI = ROOT / "ref_repo/CondMDI"
stage = ROOT / "output/evaluation/_omni_official/_stage"
SID = os.environ.get("SID", "M005213")  # worst clip from density=1
GUIDANCE = float(os.environ.get("GUIDANCE", "2.5"))
SCALE_MULT = float(os.environ.get("SCALE_MULT", "1.0"))
os.chdir(str(stage)); sys.path.insert(0, str(OMNI))
from utils.model_util import create_model_and_diffusion, load_model_wo_clip  # noqa
from utils.fixseed import fixseed  # noqa
from model.cfg_sampler import ClassifierFreeSampleModel  # noqa
from data_loaders.humanml.scripts.motion_process import recover_from_ric  # noqa
fixseed(0); device = "cuda"; MF = 196


def _args(mp):
    return types.SimpleNamespace(model_path=mp, dataset="humanml",
        cond_mode="both_text_spatial", arch="trans_enc", emb_trans_dec=False,
        layers=8, latent_dim=512, cond_mask_prob=0.1, noise_schedule="cosine",
        diffusion_steps=1000, sigma_small=True, guidance_param=GUIDANCE,
        lambda_rcxyz=0.0, lambda_vel=0.0, lambda_fc=0.0)


mp = str(OMNI / "save/omnicontrol_ckpt/model_humanml3d.pt")
model, diffusion = create_model_and_diffusion(_args(mp), types.SimpleNamespace(dataset=types.SimpleNamespace()))
load_model_wo_clip(model, torch.load(mp, map_location="cpu"))
if GUIDANCE != 1:
    model = ClassifierFreeSampleModel(model)
model.to(device); model.eval()

std = diffusion.std.to(device).float(); mean = diffusion.mean.to(device).float()
raw_mean = diffusion.raw_mean.view(22, 3).cpu().numpy().astype(np.float32)
raw_std = diffusion.raw_std.view(22, 3).cpu().numpy().astype(np.float32)

m = np.load(str(CONDMDI / "dataset/HumanML3D/new_joint_vecs" / f"{SID}.npy")).astype(np.float32)
L = (min(len(m), MF) // 4) * 4
m = m[:L]
joints = recover_from_ric(torch.from_numpy(m).float(), 22).numpy()
ctrl = L - 1  # FAR frame
disp = np.linalg.norm(joints[ctrl, 0] - joints[0, 0])
print(f"[clip {SID}] L={L} ctrl_frame={ctrl} target pelvis={joints[ctrl,0]} disp_from_origin={disp:.3f} m")
mask_seq = np.zeros((L, 22, 3), dtype=bool); mask_seq[ctrl, 0] = True
jn = ((joints - raw_mean.reshape(22, 3)) / raw_std.reshape(22, 3)) * mask_seq
h = np.zeros((MF, 66), dtype=np.float32); h[:L] = jn.reshape(L, 66)
hint_t = torch.from_numpy(h[None]).to(device)
ymask = torch.zeros((1, 1, 1, MF), dtype=torch.bool, device=device); ymask[0, :, :, :L] = True
tp = joints[ctrl, 0]

# monkeypatch gradients to log loss in metres at the controlled frame
orig_grad = diffusion.gradients
log = {}
SCALE0 = None
def patched_grad(x, hint, mask_hint, joint_ids=None):
    loss, grad = orig_grad(x, hint, mask_hint, joint_ids)
    return loss, grad * SCALE_MULT
diffusion.gradients = patched_grad

# also wrap guide to record per-timestep loss (metres)
orig_guide = diffusion.guide
def patched_guide(x, t, **kw):
    out = orig_guide(x, t, **kw)
    # measure pelvis err at ctrl frame on current mean
    with torch.no_grad():
        xx = out.permute(0, 3, 2, 1).squeeze(2) * std + mean
        jp = recover_from_ric(xx.float(), 22)[0, ctrl, 0].cpu().numpy()
    log[int(t[0])] = float(np.linalg.norm(jp - tp))
    return out
diffusion.guide = patched_guide

model_kwargs = {"y": {"mask": ymask, "lengths": torch.tensor([L], device=device),
                      "text": [""], "hint": hint_t}}
if GUIDANCE != 1:
    model_kwargs["y"]["scale"] = torch.ones(1, device=device) * GUIDANCE
print(f"[+] sampling SCALE_MULT={SCALE_MULT} ...", flush=True)
with torch.no_grad():
    sample = diffusion.p_sample_loop(model, (1, 263, 1, MF), clip_denoised=False,
        model_kwargs=model_kwargs, progress=True, const_noise=False)
print("\n=== guidance loss (m) at ctrl frame vs timestep t ===", flush=True)
for t in sorted(log.keys(), reverse=True):
    if t % 100 == 0 or t < 12:
        print(f"  t={t:4d}  pelvis_err={log[t]:.4f} m")
s = sample[:, :263].permute(0, 2, 3, 1).squeeze(1) * std + mean
gp = recover_from_ric(s.float(), 22)[0, ctrl, 0].cpu().numpy()
print(f"\nFINAL pelvis_err at ctrl frame = {np.linalg.norm(gp - tp):.4f} m   (target disp={disp:.3f} m)")
print(f"gen pelvis={gp}  target={tp}")
