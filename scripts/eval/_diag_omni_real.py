#!/usr/bin/env python3
"""Reproduce OmniControl's PAPER setting: pelvis trajectory control on REAL
HumanML3D test motions (NOT the synthetic spiral/weave demos). Hint = standard
recover_from_ric of a real 263 motion, joint 0, all frames (density=100),
normalized exactly like the official dataset.random_mask. Then run the official
p_sample_loop and report pelvis control error (metres) per clip + mean.

If mean error ~0.03-0.06 m -> OmniControl reproduces; our wrapper's 0.79 m must
come from the abs_3d hint frame / editing-set trajectories, not the model.
"""
from __future__ import annotations
import os, sys, types, glob, random
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

GUIDANCE = float(os.environ.get("GUIDANCE", "2.5"))
NSAMP = int(os.environ.get("NSAMP", "8"))
DENSITY = float(os.environ.get("DENSITY", "100"))  # 100 = every frame; 1/2/5 = that many frames

os.chdir(str(stage))
sys.path.insert(0, str(OMNI))
from utils.model_util import create_model_and_diffusion, load_model_wo_clip  # noqa
from utils.fixseed import fixseed  # noqa
from model.cfg_sampler import ClassifierFreeSampleModel  # noqa
from data_loaders.humanml.scripts.motion_process import recover_from_ric  # noqa

fixseed(0)
device = "cuda"
MF = 196


def _args(mp):
    return types.SimpleNamespace(
        model_path=mp, dataset="humanml", cond_mode="both_text_spatial",
        arch="trans_enc", emb_trans_dec=False, layers=8, latent_dim=512,
        cond_mask_prob=0.1, noise_schedule="cosine", diffusion_steps=1000,
        sigma_small=True, guidance_param=GUIDANCE,
        lambda_rcxyz=0.0, lambda_vel=0.0, lambda_fc=0.0)


mp = str(OMNI / "save/omnicontrol_ckpt/model_humanml3d.pt")
print("[+] build model", flush=True)
model, diffusion = create_model_and_diffusion(_args(mp), types.SimpleNamespace(dataset=types.SimpleNamespace()))
state = torch.load(mp, map_location="cpu")
load_model_wo_clip(model, state)
if GUIDANCE != 1:
    model = ClassifierFreeSampleModel(model)
model.to(device); model.eval()

mean = diffusion.mean.to(device).float()
std = diffusion.std.to(device).float()
# optional override of 263-dim training Mean/Std (debug: t2m-eval vs real training norm)
_MS = os.environ.get("MEANSTD_DIR", "")
if _MS:
    import numpy as _np
    mean = torch.from_numpy(_np.load(os.path.join(_MS, "Mean.npy")).astype("float32")).to(device)
    std = torch.from_numpy(_np.load(os.path.join(_MS, "Std.npy")).astype("float32")).to(device)
    diffusion.mean = mean.cpu(); diffusion.std = std.cpu()  # guide() uses these too
    print(f"[override] 263 Mean/Std <- {_MS}  std[:4]={_np.round(std[:4].cpu().numpy(),5)}", flush=True)
raw_mean = diffusion.raw_mean.view(22, 3).cpu().numpy().astype(np.float32)
raw_std = diffusion.raw_std.view(22, 3).cpu().numpy().astype(np.float32)

# real test motions (standard 263 new_joint_vecs)
njv = CONDMDI / "dataset/HumanML3D/new_joint_vecs"
test_ids = [l.strip() for l in open(CONDMDI / "dataset/HumanML3D/test.txt")]
random.seed(0); random.shuffle(test_ids)

texts = []   # empty (only_spatial)
hints = []   # (MF,66) normalized
gt_pel = []  # (MF,3) world pelvis (standard recover)
lens = []
picked = []
for sid in test_ids:
    p = njv / f"{sid}.npy"
    if not p.exists():
        continue
    m = np.load(str(p)).astype(np.float32)
    if m.shape[-1] != 263 or len(m) < 40:
        continue
    L = min(len(m), MF)
    # crop to multiple of 4 like the official loader
    L = (L // 4) * 4
    m = m[:L]
    # real caption (both_text_spatial needs text); texts/<sid>.txt: "caption#tokens#.."
    tp = CONDMDI / "dataset/HumanML3D/texts" / f"{sid}.txt"
    cap = ""
    if tp.exists():
        ln = tp.read_text().splitlines()
        if ln:
            cap = ln[0].split("#")[0].strip()
    joints = recover_from_ric(torch.from_numpy(m).float(), 22).numpy()  # (L,22,3) standard frame
    # build pelvis hint (joint 0) at density
    mask_seq = np.zeros((L, 22, 3), dtype=bool)
    if DENSITY in (1, 2, 5):
        k = int(DENSITY)
    else:
        k = int(L * DENSITY / 100)
    sel = np.sort(np.random.choice(L, max(1, k), replace=False))
    mask_seq[sel, 0] = True
    jn = (joints - raw_mean.reshape(22, 3)) / raw_std.reshape(22, 3)
    jn = jn * mask_seq
    h = np.zeros((MF, 66), dtype=np.float32)
    h[:L] = jn.reshape(L, 66)
    hints.append(h); texts.append(cap); gt_pel.append(np.pad(joints[:, 0, :], ((0, MF - L), (0, 0))))
    lens.append(L); picked.append(sid)
    if len(picked) >= NSAMP:
        break

N = len(picked)
print(f"[+] {N} real test clips, density={DENSITY}", flush=True)
hint_t = torch.from_numpy(np.stack(hints)).to(device)
lengths_t = torch.tensor(lens, device=device)
ymask = torch.zeros((N, 1, 1, MF), dtype=torch.bool, device=device)
for i, L in enumerate(lens):
    ymask[i, :, :, :L] = True
model_kwargs = {"y": {"mask": ymask, "lengths": lengths_t, "text": texts, "hint": hint_t}}
if GUIDANCE != 1:
    model_kwargs["y"]["scale"] = torch.ones(N, device=device) * GUIDANCE

print("[+] sampling ...", flush=True)
with torch.no_grad():
    sample = diffusion.p_sample_loop(
        model, (N, 263, 1, MF), clip_denoised=False, model_kwargs=model_kwargs,
        skip_timesteps=0, init_image=None, progress=True, dump_steps=None,
        noise=None, const_noise=False)
sample = sample[:, :263]
s = sample.permute(0, 2, 3, 1) * std + mean
joints_out = recover_from_ric(s.float(), 22).view(N, MF, 22, 3).cpu().numpy()

print("\n==== pelvis control error on REAL test trajectories (density=%g) ====" % DENSITY)
errs = []
for i, sid in enumerate(picked):
    L = lens[i]
    hh = hints[i].reshape(MF, 22, 3)
    m = (np.abs(hh).sum(-1) > 0)        # controlled (frame,joint)
    gp = joints_out[i, :, 0, :]         # generated pelvis
    tp = gt_pel[i]                      # target pelvis (world)
    fr = m[:, 0]                        # controlled frames for joint0
    e = np.linalg.norm((gp - tp)[fr], axis=-1).mean()
    errs.append(e)
    print(f"  {sid:>8}  L={L:3d}  ctrl_frames={int(fr.sum()):3d}  pelvis_err={e:.4f} m")
print(f"\nMEAN pelvis control error = {np.mean(errs):.4f} m  (paper ~0.03-0.06 m)")
print("[verdict]", "REPRODUCES (model works on real traj)" if np.mean(errs) < 0.12
      else "still high -> deeper issue")
