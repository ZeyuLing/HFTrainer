#!/usr/bin/env python3
"""Faithful replica of OmniControl's official ``generate.py --text_prompt predefined``
spatial-control path, run in OUR torch-2.5 env, to decide whether OmniControl's
spatial guidance actually follows the control signal here.

It reuses the released-checkpoint model/diffusion setup (identical to
``omnicontrol_run_bodypart.py``), takes the OFFICIAL predefined hints from
``utils.text_control_example.collate_all`` (pelvis dense/sparse, wrist, head,
foot, spiral, combination, in-betweening), runs the pristine ``p_sample_loop``
and reports the OFFICIAL ``simple_eval`` control error (metres) PER sample.

Low pelvis/spiral error -> guidance works here, our eval wrapper is the bug.
High error everywhere    -> env/version issue (official torch 1.7.1 vs our 2.5).
"""
from __future__ import annotations
import os, sys, types
from pathlib import Path
import numpy as np

for _n, _v in {"bool": bool, "int": int, "float": float, "complex": complex,
               "object": object, "str": str, "unicode": str}.items():
    if not hasattr(np, _n):
        setattr(np, _n, _v)
import torch  # noqa: E402

ROOT = Path("/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer")
OMNI = ROOT / "ref_repo/OmniControl"
CONDMDI = ROOT / "ref_repo/CondMDI"

GUIDANCE = float(os.environ.get("GUIDANCE", "2.5"))
COND_MODE = os.environ.get("COND_MODE", "only_spatial")  # only_spatial|both_text_spatial
SEED = int(os.environ.get("SEED", "0"))


def _stage() -> Path:
    stage = ROOT / "output/evaluation/_omni_official/_stage"
    (stage / "dataset/humanml_spatial_norm").mkdir(parents=True, exist_ok=True)
    (stage / "dataset/HumanML3D").mkdir(parents=True, exist_ok=True)

    def link(src, dst):
        if not (dst.exists() or dst.is_symlink()):
            os.symlink(str(src), str(dst))
    link(CONDMDI / "dataset/HumanML3D/Mean.npy", stage / "dataset/HumanML3D/Mean.npy")
    link(CONDMDI / "dataset/HumanML3D/Std.npy", stage / "dataset/HumanML3D/Std.npy")
    link(OMNI / "dataset/humanml_spatial_norm/Mean_raw.npy",
         stage / "dataset/humanml_spatial_norm/Mean_raw.npy")
    link(OMNI / "dataset/humanml_spatial_norm/Std_raw.npy",
         stage / "dataset/humanml_spatial_norm/Std_raw.npy")
    link(OMNI / "body_models", stage / "body_models")
    return stage


def _args(model_path):
    return types.SimpleNamespace(
        model_path=model_path, dataset="humanml", cond_mode="both_text_spatial",
        arch="trans_enc", emb_trans_dec=False, layers=8, latent_dim=512,
        cond_mask_prob=0.1, noise_schedule="cosine", diffusion_steps=1000,
        sigma_small=True, guidance_param=GUIDANCE,
        lambda_rcxyz=0.0, lambda_vel=0.0, lambda_fc=0.0,
    )


def main():
    stage = _stage()
    os.chdir(str(stage))
    sys.path.insert(0, str(OMNI))
    from utils.model_util import create_model_and_diffusion, load_model_wo_clip
    from utils.fixseed import fixseed
    from model.cfg_sampler import ClassifierFreeSampleModel
    from data_loaders.humanml.scripts.motion_process import recover_from_ric
    from utils.text_control_example import (
        pelvis_dense_text_control_example,
        pelvis_sparse_text_control_example,
        unnatural_text_control_example,
    )
    from utils.simple_eval import simple_eval

    fixseed(SEED)
    device = "cuda"
    MF = 196
    mp = str(OMNI / "save/omnicontrol_ckpt/model_humanml3d.pt")
    args = _args(mp)
    print("[+] build model+diffusion", flush=True)
    model, diffusion = create_model_and_diffusion(args, types.SimpleNamespace(dataset=types.SimpleNamespace()))
    state = torch.load(mp, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    miss_real = [k for k in missing if not k.startswith("clip_model.")]
    print(f"[ckpt] missing(non-clip)={len(miss_real)} unexpected={len(unexpected)}", flush=True)
    if miss_real[:8]:
        print("       e.g. missing:", miss_real[:8], flush=True)
    if unexpected[:8]:
        print("       e.g. unexpected:", unexpected[:8], flush=True)
    if GUIDANCE != 1:
        model = ClassifierFreeSampleModel(model)
    model.to(device)
    model.eval()

    mean263 = diffusion.mean.to(device).float()
    std263 = diffusion.std.to(device).float()

    # trajectory-relevant predefined hints only (avoid dataset-dependent inbetween).
    raw_mean_np = np.load("dataset/humanml_spatial_norm/Mean_raw.npy")
    raw_std_np = np.load("dataset/humanml_spatial_norm/Std_raw.npy")
    t0, h0, _ = pelvis_dense_text_control_example(MF, raw_mean_np, raw_std_np, index=0)
    t1, h1, _ = pelvis_sparse_text_control_example(MF, raw_mean_np, raw_std_np, index=0)
    t5, h5, _ = unnatural_text_control_example(MF, raw_mean_np, raw_std_np, index=0)
    texts = list(t0) + list(t1) + list(t5)
    hints = np.concatenate([h0, h1, h5], axis=0)   # (N, MF, 66) normalized
    labels = ["pelvis_dense", "pelvis_sparse", "spiral(unnatural)"]
    N = len(texts)
    if COND_MODE == "only_spatial":
        texts = ["" for _ in texts]

    hint_t = torch.from_numpy(hints.astype(np.float32)).to(device)   # (N,MF,66)
    lengths_t = torch.tensor([MF] * N, device=device)
    ymask = torch.ones((N, 1, 1, MF), dtype=torch.bool, device=device)
    model_kwargs = {"y": {"mask": ymask, "lengths": lengths_t,
                          "text": list(texts), "hint": hint_t}}
    if GUIDANCE != 1:
        model_kwargs["y"]["scale"] = torch.ones(N, device=device) * GUIDANCE

    print(f"[+] sampling N={N} cond_mode={COND_MODE} guidance={GUIDANCE}", flush=True)
    with torch.no_grad():
        sample = diffusion.p_sample_loop(
            model, (N, 263, 1, MF), clip_denoised=False, model_kwargs=model_kwargs,
            skip_timesteps=0, init_image=None, progress=True, dump_steps=None,
            noise=None, const_noise=False)
    sample = sample[:, :263]
    s = sample.permute(0, 2, 3, 1) * std263 + mean263       # (N,1,MF,263)
    joints = recover_from_ric(s.float(), 22)                # (N,1,MF,22,3)
    joints = joints.view(N, MF, 22, 3)                      # world xyz (m)

    # denormalize hint exactly like generate.py
    raw_mean = torch.from_numpy(np.load("dataset/humanml_spatial_norm/Mean_raw.npy")).to(device)
    raw_std = torch.from_numpy(np.load("dataset/humanml_spatial_norm/Std_raw.npy")).to(device)
    h = hint_t.clone()
    hmask = h.view(N, MF, 22, 3).sum(-1) != 0
    h = h * raw_std + raw_mean
    h = h.view(N, MF, 22, 3) * hmask.unsqueeze(-1)
    h = h.view(N, MF, -1)

    motion_np = joints.permute(0, 2, 3, 1).cpu().numpy()    # (N,22,3,MF) for simple_eval
    hint_np = h.cpu().numpy()                               # (N,MF,66)

    # dump generated pelvis path + target for inspection
    gen_pelvis = joints[:, :, 0, :].cpu().numpy()            # (N,MF,3)
    tgt_pelvis = hint_np.reshape(N, MF, 22, 3)[:, :, 0, :]   # (N,MF,3)
    np.savez(os.path.join(ROOT, "output/evaluation/_omni_pelvis_paths.npz"),
             gen=gen_pelvis, tgt=tgt_pelvis, labels=np.array(labels))
    print("[saved] output/evaluation/_omni_pelvis_paths.npz")

    print("\n==== OFFICIAL simple_eval control error (metres, lower=better) ====")
    for i in range(N):
        err = simple_eval(motion_np[i:i+1], hint_np[i:i+1], 22)
        # also report pelvis-only XZ drift for trajectory categories
        hh = hint_np[i].reshape(MF, 22, 3)
        m = (np.abs(hh).sum(-1) > 0)            # (MF,22) controlled
        mm = joints[i].cpu().numpy()
        if m.any():
            diff = np.linalg.norm((mm - hh) * m[..., None], axis=-1)
            mean_m = diff.sum() / m.sum()
        else:
            mean_m = float("nan")
        print(f"  [{i}] {labels[i]:18s} simple_eval={float(err):.4f} m   mean_joint_err={mean_m:.4f} m")
    print("\n[interpret] pelvis_dense / pelvis_sparse / spiral should be < ~0.05 m if guidance follows.")


if __name__ == "__main__":
    main()
