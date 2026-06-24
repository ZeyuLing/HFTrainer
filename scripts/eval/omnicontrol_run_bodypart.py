#!/usr/bin/env python3
"""OmniControl body-part 3D-position control for Table-6 Experiment B.

OmniControl (ICLR'24) is purpose-built for "control ANY joint at ANY frame via
its 3D position".  For ExpB we feed, as the spatial hint, the GT 3D positions of
ONE body-part's joints (--part) on EVERY frame and let the model regenerate the
rest of the body from text.  Output world joints ``(T,22,3)`` @20fps feed the
shared IK->motion_135->metrics chain (identical to CondMDI / \\ours / KIMODO).

Design (no ref_repo edits):
  * we ``chdir`` into a STAGING dir that symlinks the norm-stat files OmniControl
    loads via relative ``./dataset/...`` paths (its own ``humanml_spatial_norm`` +
    CondMDI's standard 263 ``Mean/Std`` -- identical HumanML3D normalisation), and
    put OmniControl on ``sys.path``;
  * model args are constructed from the released-model defaults (no ``args.json``
    ships with the checkpoint);
  * hints come from HumanML3D-native GT joints (``omnicontrol_gt_joints.py``).

Output: ``<out>/<source_id>.npy`` of shape ``(T,22,3)`` @20fps.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import types
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
TESMO = ROOT / "ref_repo/TeSMo"
PART_MODULE = ROOT / "scripts/eval"


def _build_args(model_path: str, guidance: float):
    return types.SimpleNamespace(
        model_path=model_path, dataset="humanml", cond_mode="both_text_spatial",
        arch="trans_enc", emb_trans_dec=False, layers=8, latent_dim=512,
        cond_mask_prob=0.1, noise_schedule="cosine", diffusion_steps=1000,
        sigma_small=True, guidance_param=guidance,
        lambda_rcxyz=0.0, lambda_vel=0.0, lambda_fc=0.0,
    )


def _stage(out_root: Path) -> Path:
    """Create a cwd staging dir with the dataset files OmniControl loads."""
    stage = out_root / "_stage"
    (stage / "dataset/HumanML3D").mkdir(parents=True, exist_ok=True)
    (stage / "dataset/humanml_spatial_norm").mkdir(parents=True, exist_ok=True)

    def link(src: Path, dst: Path):
        if dst.exists() or dst.is_symlink():
            return
        try:
            os.symlink(str(src), str(dst))
        except FileExistsError:
            # race: a sibling shard process created it first -- harmless
            pass

    # standard 263 *training* Mean/Std (root-channel std ~0.013). CondMDI's
    # ``Mean.npy`` is byte-identical to the T2M *eval* norm (root std ~0.0005);
    # using it under-scales the root rot/transl by ~25x -> the body never turns
    # and every motion drifts straight forward. OmniControl (MDM-based) was
    # trained with the canonical HumanML3D training norm shipped by TeSMo, which
    # reproduces the paper's pelvis Traj.Err (~0.058 m vs ~0.79 m with the eval norm).
    link(TESMO / "dataset/HumanML3D/Mean.npy", stage / "dataset/HumanML3D/Mean.npy")
    link(TESMO / "dataset/HumanML3D/Std.npy", stage / "dataset/HumanML3D/Std.npy")
    # OmniControl's own spatial (raw joint) norm
    link(OMNI / "dataset/humanml_spatial_norm/Mean_raw.npy",
         stage / "dataset/humanml_spatial_norm/Mean_raw.npy")
    link(OMNI / "dataset/humanml_spatial_norm/Std_raw.npy",
         stage / "dataset/humanml_spatial_norm/Std_raw.npy")
    # SMPL body model (rot2xyz construction loads ./body_models/smpl/*)
    link(OMNI / "body_models", stage / "body_models")
    return stage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=str(OMNI / "save/omnicontrol_ckpt/model_humanml3d.pt"))
    ap.add_argument("--gt-joints-dir", required=True,
                    help="dir of HumanML3D-native GT joints (T,22,3) @20fps, from omnicontrol_gt_joints.py")
    ap.add_argument("--source-id-file", required=True)
    ap.add_argument("--part", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--guidance", type=float, default=2.5)
    ap.add_argument("--max-frames", type=int, default=196)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shard-index", dest="shard", type=int)
    ap.add_argument("--seed", type=int, default=0)
    args_cli = ap.parse_args()

    sys.path.insert(0, str(PART_MODULE))
    from bodypart_pos_common import part_joints
    joints_sel = part_joints(args_cli.part)

    out = Path(args_cli.out)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    stage = _stage(out.parent)

    os.chdir(str(stage))
    sys.path.insert(0, str(OMNI))
    from utils.model_util import create_model_and_diffusion, load_model_wo_clip
    from utils.fixseed import fixseed
    from model.cfg_sampler import ClassifierFreeSampleModel
    from data_loaders.humanml.scripts.motion_process import recover_from_ric

    fixseed(args_cli.seed)
    device = "cuda"
    MF = args_cli.max_frames

    args = _build_args(args_cli.model_path, args_cli.guidance)
    data_stub = types.SimpleNamespace(dataset=types.SimpleNamespace())
    print("[+] creating OmniControl model + diffusion ...", flush=True)
    model, diffusion = create_model_and_diffusion(args, data_stub)
    state = torch.load(args_cli.model_path, map_location="cpu")
    load_model_wo_clip(model, state)
    if args_cli.guidance != 1:
        model = ClassifierFreeSampleModel(model)
    model.to(device)
    model.eval()
    print("[+] model ready", flush=True)

    raw_mean = diffusion.raw_mean.view(1, 22, 3).numpy().astype(np.float32)
    raw_std = diffusion.raw_std.view(1, 22, 3).numpy().astype(np.float32)
    mean263 = diffusion.mean.to(device).float()
    std263 = diffusion.std.to(device).float()

    # shared ids
    sp = Path(args_cli.source_id_file)
    if not sp.is_absolute():
        sp = ROOT / sp
    txt = sp.read_text()
    try:
        ids = [str(x) for x in json.loads(txt)]
    except Exception:  # noqa: BLE001
        ids = [s.strip() for s in txt.splitlines() if s.strip()]
    gtd = Path(args_cli.gt_joints_dir)
    if not gtd.is_absolute():
        gtd = ROOT / gtd
    ids = [s for s in ids if (gtd / f"{s}.npy").exists()]
    if args_cli.num_shards > 1:
        ids = ids[args_cli.shard:: args_cli.num_shards]
    ids = [s for s in ids if not (out / f"{s}.npy").exists()]
    print(f"[+] {len(ids)} clips (shard {args_cli.shard}/{args_cli.num_shards}) part={args_cli.part}", flush=True)

    # caption lookup
    from bodypart_pos_common import load_editing_index
    cap_by_sid = {str(it["source_id"]): str(it.get("caption_en", ""))
                  for it in load_editing_index()}

    bs = args_cli.batch_size
    jt = np.asarray(joints_sel, dtype=np.int64)
    n_done = 0
    for bstart in range(0, len(ids), bs):
        chunk = ids[bstart: bstart + bs]
        B = len(chunk)
        hint = np.zeros((B, MF, 22, 3), dtype=np.float32)
        lengths, caps = [], []
        for i, sid in enumerate(chunk):
            g = np.load(str(gtd / f"{sid}.npy")).astype(np.float32)  # (L,22,3)
            L = min(len(g), MF)
            gn = (g[:L] - raw_mean) / raw_std            # normalize all joints
            h = np.zeros((MF, 22, 3), dtype=np.float32)
            h[:L, jt, :] = gn[:, jt, :]                  # keep only part joints
            hint[i] = h
            lengths.append(L)
            caps.append(cap_by_sid.get(sid, ""))
        hint_t = torch.from_numpy(hint.reshape(B, MF, 66)).to(device)
        lengths_t = torch.tensor(lengths, device=device)
        ymask = torch.zeros((B, 1, 1, MF), dtype=torch.bool, device=device)
        for i, L in enumerate(lengths):
            ymask[i, :, :, :L] = True

        model_kwargs = {"y": {
            "mask": ymask, "lengths": lengths_t, "text": caps, "hint": hint_t,
        }}
        if args_cli.guidance != 1:
            model_kwargs["y"]["scale"] = torch.ones(B, device=device) * args_cli.guidance

        with torch.no_grad():
            sample = diffusion.p_sample_loop(
                model, (B, model.njoints if hasattr(model, "njoints") else 263, 1, MF),
                clip_denoised=False, model_kwargs=model_kwargs,
                skip_timesteps=0, init_image=None, progress=False,
                dump_steps=None, noise=None, const_noise=False)

        sample = sample[:, :263]                                  # [B,263,1,MF]
        s = sample.permute(0, 2, 3, 1) * std263 + mean263         # [B,1,MF,263]
        joints = recover_from_ric(s.float(), 22)                  # [B,1,MF,22,3]
        joints = joints.view(B, MF, 22, 3).cpu().numpy()
        for i, sid in enumerate(chunk):
            L = lengths[i]
            np.save(str(out / f"{sid}.npy"), joints[i, :L].astype(np.float32))
        n_done += B
        print(f"  {n_done}/{len(ids)} done", flush=True)

    print(f"[+] DONE {n_done} clips -> {out}", flush=True)


if __name__ == "__main__":
    main()
