#!/usr/bin/env python3
"""Diagnostic: does OmniControl spatial guidance actually pull the root toward the
dense pelvis hint? Runs ONE clip with the exact run_bodypart setup and prints the
guidance loss across timesteps + final root XZ error vs hint."""
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
CONDMDI = ROOT / "ref_repo/CondMDI"
SID = sys.argv[1] if len(sys.argv) > 1 else "000019"
GUID = float(sys.argv[2]) if len(sys.argv) > 2 else 2.5

# reuse staging
sys.path.insert(0, str(ROOT / "scripts/eval"))
from omnicontrol_run_bodypart import _build_args, _stage  # noqa: E402
from bodypart_pos_common import part_joints  # noqa: E402

out = ROOT / "output/evaluation/_diag_omni"
out.mkdir(parents=True, exist_ok=True)
stage = _stage(out)
os.chdir(str(stage))
sys.path.insert(0, str(OMNI))
from utils.model_util import create_model_and_diffusion, load_model_wo_clip  # noqa: E402
from utils.fixseed import fixseed  # noqa: E402
from model.cfg_sampler import ClassifierFreeSampleModel  # noqa: E402
from data_loaders.humanml.scripts.motion_process import recover_from_ric  # noqa: E402

fixseed(0)
device = "cuda"
MF = 196
args = _build_args(str(OMNI / "save/omnicontrol_ckpt/model_humanml3d.pt"), GUID)
model, diffusion = create_model_and_diffusion(args, types.SimpleNamespace(dataset=types.SimpleNamespace()))
state = torch.load(args.model_path, map_location="cpu")
load_model_wo_clip(model, state)
if GUID != 1:
    model = ClassifierFreeSampleModel(model)
model.to(device)
model.eval()

raw_mean = diffusion.raw_mean.view(1, 22, 3).numpy().astype(np.float32)
raw_std = diffusion.raw_std.view(1, 22, 3).numpy().astype(np.float32)

jt = np.array(part_joints("root"), dtype=np.int64)
USE_SPIRAL = os.environ.get("USE_SPIRAL", "0") == "1"
if USE_SPIRAL:
    # official OmniControl dense-pelvis demo path -> isolate env/model vs our hint
    from utils.text_control_example import spiral_forward
    path = spiral_forward(MF)[:, :3].astype(np.float32)  # (MF,3) world xyz
    L = MF
    gtj = np.zeros((MF, 22, 3), dtype=np.float32); gtj[:, 0] = path
    gn = (gtj - raw_mean) / raw_std
    h = np.zeros((MF, 22, 3), dtype=np.float32)
    h[:, jt, :] = gn[:, jt, :]
    cap = "a person walks"
else:
    # GT joints hint (abs_3d recovered) for SID
    gtj = np.load(str(ROOT / f"output/evaluation/table7_traj/omnicontrol/gt_joints/{SID}.npy")).astype(np.float32)
    L = min(len(gtj), MF)
    gn = (gtj[:L] - raw_mean) / raw_std
    h = np.zeros((MF, 22, 3), dtype=np.float32)
    h[:L, jt, :] = gn[:, jt, :]
    cap = None
# optionally sparsify: keep only KEYFRAMES evenly-spaced controlled frames
KF = int(os.environ.get("KEYFRAMES", "0"))
if KF > 0:
    keep = np.linspace(0, L - 1, KF).round().astype(int)
    h2 = np.zeros_like(h); h2[keep] = h[keep]; h = h2
    print(f"[diag] sparsified to {KF} keyframes: {keep.tolist()}")
hint_t = torch.from_numpy(h.reshape(1, MF, 66)).to(device)
print(f"[diag] SID={SID} L={L} guid={GUID} hint nonzero frames={int((np.abs(h).sum((1,2))>0).sum())}")
print(f"[diag] hint root(norm) abs-mean={np.abs(gn[:,0]).mean():.3f}  raw root XZ span x={np.ptp(gtj[:L,0,0]):.2f} z={np.ptp(gtj[:L,0,2]):.2f}")

# caption
if cap is None:
    from bodypart_pos_common import load_editing_index  # noqa: E402
    cap = {str(it["source_id"]): str(it.get("caption_en", "")) for it in load_editing_index()}.get(SID, "")
print(f"[diag] caption={cap!r}")

ymask = torch.zeros((1, 1, 1, MF), dtype=torch.bool, device=device); ymask[..., :L] = True
model_kwargs = {"y": {"mask": ymask, "lengths": torch.tensor([L], device=device),
                      "text": [cap], "hint": hint_t}}
if GUID != 1:
    model_kwargs["y"]["scale"] = torch.ones(1, device=device) * GUID

# instrument gradients() to log loss + grad-norm across timesteps
orig_grad = diffusion.gradients
glog = {}
def patched(x, hint, mask_hint, joint_ids=None):
    loss, grad = orig_grad(x, hint, mask_hint, joint_ids)
    t = getattr(diffusion, "_diag_t", -1)
    glog.setdefault(int(t), []).append((float(loss.sum().item()), float(grad.norm().item())))
    return loss, grad
diffusion.gradients = patched
# wrap guide to capture scale + loss before/after the inner optimisation
orig_guide = diffusion.guide
def wrap_guide(x, t, model_kwargs=None, **kw):
    diffusion._diag_t = int(t[0].item())
    before = len(glog.get(int(t[0].item()), []))
    out = orig_guide(x, t, model_kwargs=model_kwargs, **kw)
    return out
diffusion.guide = wrap_guide
orig_psample = diffusion.p_sample
def psample(model, x, t, **kw):
    diffusion._diag_t = int(t[0].item())
    return orig_psample(model, x, t, **kw)
diffusion.p_sample = psample

# optional scale boost to test whether weak guidance is the cause
SCALE_MULT = float(os.environ.get("SCALE_MULT", "1"))
if SCALE_MULT != 1:
    _orig_cgs = diffusion.calc_grad_scale
    diffusion.calc_grad_scale = lambda mh: _orig_cgs(mh) * SCALE_MULT
    print(f"[diag] SCALE_MULT={SCALE_MULT}")

with torch.no_grad():
    sample = diffusion.p_sample_loop(
        model, (1, 263, 1, MF), clip_denoised=False, model_kwargs=model_kwargs,
        skip_timesteps=0, init_image=None, progress=True, dump_steps=None,
        noise=None, const_noise=False)

mean263 = diffusion.mean.to(device).float(); std263 = diffusion.std.to(device).float()
s = sample[:, :263].permute(0, 2, 3, 1) * std263 + mean263
joints = recover_from_ric(s.float(), 22).view(MF, 22, 3).cpu().numpy()[:L]

# guidance loss trajectory: within-call loss reduction + grad norm
ts = sorted(glog.keys(), reverse=True)
samp = [t for t in ts if t in (999, 500, 100, 50, 20, 10, 9, 5, 1, 0)]
print("[diag] per-t: ninner | loss_first->loss_last | grad_norm:")
for t in samp:
    vals = glog[t]
    lf, ll = vals[0][0], vals[-1][0]
    gn = np.mean([v[1] for v in vals])
    print(f"   t={t:4d} ninner={len(vals):3d} loss {lf:8.3f} -> {ll:8.3f}  gradnorm={gn:.4e}")
out_xz = joints[:, 0][:, [0, 2]]; gt_xz = gtj[:L, 0][:, [0, 2]]
err = np.linalg.norm(out_xz - gt_xz, axis=1)
print(f"[diag] FINAL root XZ err mean={err.mean():.3f} max={err.max():.3f}")
print(f"[diag] out root XZ span x={np.ptp(out_xz[:,0]):.2f} z={np.ptp(out_xz[:,1]):.2f} | gt span x={np.ptp(gt_xz[:,0]):.2f} z={np.ptp(gt_xz[:,1]):.2f}")
print(f"[diag] out root y range=[{joints[:,0,1].min():.2f},{joints[:,0,1].max():.2f}] gt y=[{gtj[:L,0,1].min():.2f},{gtj[:L,0,1].max():.2f}]")
