#!/usr/bin/env python3
"""Run CondMDI in-betweening (first+last frame only) on our 4012-clip HumanML3D
test set, and dump per-clip world joints ``(T,22,3)`` for the unified 272 eval.

Manual per-clip feeding (bypasses the t2m eval dataloader) so output ids map
exactly to ``source_id``. Mirrors ``sample/conditional_synthesis.py`` math:
normalize abs-263 -> first_last keyframe mask -> imputation -> p_sample_loop ->
inv_transform -> recover_from_ric(abs_3d=True) -> world joints.

Output: ``<out>/<id>.npy`` of shape ``(T,22,3)`` (un-padded, length-cropped).

Usage::

    CUDA_VISIBLE_DEVICES=0 python3 scripts/eval/condmdi_run_inbetween.py \
        --out output/evaluation/mib_h3d_full/_condmdi_joints \
        --batch-size 16 --num-shards 1 --shard 0
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

CONDMDI = Path("/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/ref_repo/CondMDI")
os.chdir(str(CONDMDI))
sys.path.insert(0, str(CONDMDI))


def _read_first_caption(text_file: Path) -> str:
    if not text_file.exists():
        return ""
    for line in text_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("#")
        if len(parts) < 4:
            continue
        cap, ftag, ttag = parts[0], parts[2], parts[3]
        try:
            fv, tv = float(ftag), float(ttag)
        except ValueError:
            continue
        if (fv == 0.0 or fv != fv) and (tv == 0.0 or tv != tv):
            return cap
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="save/model000750000.pt")
    ap.add_argument("--data-root", default="dataset/HumanML3D")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--guidance", type=float, default=2.5)
    ap.add_argument("--max-frames", type=int, default=196)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--use-ddim", action="store_true", help="ddim100 respacing (CondMDI eval setting, ~10x faster).")
    ap.add_argument("--seed", type=int, default=0)
    args_cli = ap.parse_args()

    from utils.parser_util import cond_synt_args
    from utils.model_util import create_model_and_diffusion, load_saved_model
    from utils import dist_util
    from utils.fixseed import fixseed
    from model.cfg_sampler import ClassifierFreeSampleModel
    from data_loaders.humanml.scripts.motion_process import recover_from_ric
    from utils.editing_util import get_keyframes_mask
    import types

    fixseed(args_cli.seed)

    # Build CondMDI args by loading the checkpoint's args.json + required flags.
    sys.argv = [
        "condmdi", "--model_path", args_cli.model_path,
        "--keyframe_conditioned",
        "--abs_3d",
        "--guidance_param", str(args_cli.guidance),
        "--imputate",
        "--dataset", "humanml",
        "--keyframe_selection_scheme", "random_frames",
    ]
    args = cond_synt_args()
    # cond_synt_args restores use_ddim from the checkpoint's args.json (False),
    # so force it here AFTER parsing to actually enable ddim100 respacing.
    args.use_ddim = bool(args_cli.use_ddim)

    device = "cuda"
    dist_util.setup_dist(0)
    out = Path("/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer") / args_cli.out
    out.mkdir(parents=True, exist_ok=True)

    data_root = CONDMDI / args_cli.data_root
    mean_abs = np.load(str(data_root / "Mean_abs_3d.npy")).astype(np.float32)  # (263,)
    std_abs = np.load(str(data_root / "Std_abs_3d.npy")).astype(np.float32)
    test_ids = [s.strip() for s in (data_root / "test.txt").read_text().splitlines() if s.strip()]
    if args_cli.limit:
        test_ids = test_ids[: args_cli.limit]
    if args_cli.num_shards > 1:
        test_ids = test_ids[args_cli.shard:: args_cli.num_shards]
    # skip already-produced
    test_ids = [s for s in test_ids if not (out / f"{s}.npy").exists()]
    print(f"[+] {len(test_ids)} clips to run (shard {args_cli.shard}/{args_cli.num_shards})", flush=True)

    data_shim = types.SimpleNamespace(dataset=types.SimpleNamespace())
    print("[+] creating model + diffusion ...", flush=True)
    model, diffusion = create_model_and_diffusion(args, data_shim)
    load_saved_model(model, args_cli.model_path)
    if args_cli.guidance != 1:
        model = ClassifierFreeSampleModel(model)
    model.to(device)
    model.eval()
    print("[+] model ready", flush=True)

    mean_t = torch.from_numpy(mean_abs).to(device)
    std_t = torch.from_numpy(std_abs).to(device)
    MF = args_cli.max_frames
    bs = args_cli.batch_size

    def load_clip(sid):
        m = np.load(str(data_root / "new_joint_vecs_abs_3d" / f"{sid}.npy")).astype(np.float32)
        L = min(len(m), MF)
        cap = _read_first_caption(data_root / "texts" / f"{sid}.txt")
        return m[:L], L, cap

    n_done = 0
    for bstart in range(0, len(test_ids), bs):
        chunk = test_ids[bstart: bstart + bs]
        motions = np.zeros((len(chunk), MF, 263), dtype=np.float32)
        lengths, caps = [], []
        for i, sid in enumerate(chunk):
            m, L, cap = load_clip(sid)
            motions[i, :L] = m
            lengths.append(L)
            caps.append(cap)
        # normalize abs
        x = torch.from_numpy(motions).to(device)
        x = (x - mean_t) / std_t  # [b,MF,263]
        x = x.permute(0, 2, 1).unsqueeze(2)  # [b,263,1,MF]
        lengths_t = torch.tensor(lengths, device=device)
        ymask = torch.zeros((len(chunk), 1, 1, MF), dtype=torch.bool, device=device)
        for i, L in enumerate(lengths):
            ymask[i, :, :, :L] = True

        obs_mask, obs_joint_mask = get_keyframes_mask(
            data=x, lengths=lengths_t, edit_mode="first_last",
            feature_mode="pos_rot_vel", get_joint_mask=True)

        model_kwargs = {"obs_x0": x, "obs_mask": obs_mask, "y": {}}
        model_kwargs["y"]["mask"] = ymask
        model_kwargs["y"]["lengths"] = lengths_t
        model_kwargs["y"]["text"] = caps
        model_kwargs["y"]["diffusion_steps"] = args.diffusion_steps
        # imputation (zero_keyframe_loss=False model -> impute observed at inference)
        model_kwargs["y"]["imputate"] = 1
        model_kwargs["y"]["stop_imputation_at"] = 0
        model_kwargs["y"]["replacement_distribution"] = "conditional"
        model_kwargs["y"]["inpainted_motion"] = x
        model_kwargs["y"]["inpainting_mask"] = obs_mask
        model_kwargs["y"]["reconstruction_guidance"] = False
        if args_cli.guidance != 1:
            model_kwargs["y"]["text_scale"] = torch.ones(len(chunk), device=device) * args_cli.guidance

        with torch.no_grad():
            sample = diffusion.p_sample_loop(
                model, (len(chunk), model.njoints if hasattr(model, "njoints") else 263, 1, MF),
                clip_denoised=False, model_kwargs=model_kwargs,
                skip_timesteps=0, init_image=None, progress=False,
                dump_steps=None, noise=None, const_noise=False)

        # unnormalize + recover joints (abs_3d)
        s = sample.cpu().permute(0, 2, 3, 1)  # [b,1,MF,263]
        s = (s * std_abs) + mean_abs
        s = torch.from_numpy(np.asarray(s)).float()
        joints = recover_from_ric(s, 22, abs_3d=True)  # [b,1,MF,22,3]
        joints = joints.view(len(chunk), MF, 22, 3).numpy()

        for i, sid in enumerate(chunk):
            L = lengths[i]
            np.save(str(out / f"{sid}.npy"), joints[i, :L].astype(np.float32))
        n_done += len(chunk)
        print(f"  {n_done}/{len(test_ids)} done", flush=True)

    print(f"[+] DONE {n_done} clips -> {out}", flush=True)


if __name__ == "__main__":
    main()
