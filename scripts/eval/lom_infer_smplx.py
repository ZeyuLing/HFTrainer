#!/usr/bin/env python3
"""Batch text->motion inference for Language-of-Motion (LoM).

Mirrors ``ref_repo/language_of_motion/demo.py`` text2motion path but drives it
from our motionhub-style annotation + a precomputed ORIGINAL caption map, and
saves one SMPL-X NPZ per sample (poses[T,165] axis-angle, trans[T,3], 30 fps).
Those NPZ then flow through ``convert_smplx_npz_dir_to_135d.py`` ->
``convert_hylite135_to_motionclip_col.py`` -> the MotionCLIP evaluator.

Uses the cached ``google/flan-t5-base`` (config path is missing) and disables
flash-attention (turbot5 is not installed).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[2]
LOM_ROOT = REPO / "ref_repo" / "language_of_motion"


def _iter_entries(raw):
    data = raw["data_list"] if isinstance(raw, dict) and "data_list" in raw else raw
    if isinstance(data, dict):
        for name, entry in data.items():
            yield str(name), entry
    else:
        for i, entry in enumerate(data):
            yield str(entry.get("motion_id") or entry.get("id") or i), entry


def load_jobs(anno_file, caption_map, num_shards, shard_index, max_samples):
    cmap = json.loads(Path(caption_map).read_text())
    jobs, eligible = [], 0
    for name, _entry in _iter_entries(json.loads(Path(anno_file).read_text())):
        caption = cmap.get(name)
        if not (isinstance(caption, str) and caption.strip()):
            continue
        if eligible % num_shards == shard_index:
            jobs.append((name, caption.strip()))
            if max_samples and len(jobs) >= max_samples:
                break
        eligible += 1
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anno-file", required=True)
    ap.add_argument("--caption-map", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--flan-path", default="google/flan-t5-base")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="fp32")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    anno_file = Path(args.anno_file).resolve()
    caption_map = Path(args.caption_map).resolve()

    jobs = load_jobs(anno_file, caption_map, args.num_shards, args.shard_index, args.max_samples)
    if args.skip_existing:
        jobs = [(n, c) for n, c in jobs if not (out_dir / f"{n}.npz").exists()]
    print({"jobs": len(jobs), "out_dir": str(out_dir),
           "shard": f"{args.shard_index}/{args.num_shards}"}, flush=True)
    if not jobs:
        return

    # LoM configs use relative ./configs and ./model_files paths.
    os.chdir(LOM_ROOT)
    sys.path.insert(0, str(LOM_ROOT))
    logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger("lom_infer")

    import pytorch_lightning as pl  # noqa: WPS433
    from lom.config import parse_args  # noqa: WPS433
    from lom.models.build_model import build_model  # noqa: WPS433
    from lom.utils.load_checkpoint import load_pretrained_vae, load_pretrained_lm  # noqa: WPS433
    from lom.utils.rotation_conversions import (  # noqa: WPS433
        rotation_6d_to_matrix, rotation_6d_to_axis_angle, matrix_to_axis_angle,
        matrix_to_rotation_6d,
    )
    from lom.utils.other_tools import velocity2position  # noqa: WPS433
    from lom.data.mixed_dataset.data_tools import (  # noqa: WPS433
        JOINT_MASK_UPPER, JOINT_MASK_HAND, JOINT_MASK_LOWER,
    )

    # The Language_Motion model instantiates SMPL-X / FLAME body+face models in
    # __init__, but text2motion token generation never calls them (only mesh /
    # metric paths do, which we skip).  Stub them to avoid the licensed assets.
    import smplx  # noqa: WPS433
    _orig_create = smplx.create

    class _Stub(torch.nn.Module):
        def __init__(self, *a, **k):
            super().__init__()

        def forward(self, *a, **k):  # noqa: ANN001
            raise RuntimeError("stubbed SMPL-X/FLAME called (mesh path disabled)")

    def _safe_create(model_path, **kw):  # noqa: ANN001
        if not os.path.exists(model_path):
            return _Stub()
        return _orig_create(model_path, **kw)

    smplx.create = _safe_create
    smplx.FLAME = lambda *a, **k: _Stub()

    sys.argv = ["demo.py", "--cfg", "configs/demo_text2motion.yaml",
                "--task", "text2motion", "--text", "configs/demo_text2motion.yaml"]
    cfg = parse_args(phase="demo")
    cfg.model.params.lm.params.model_path = args.flan_path
    cfg.model.params.lm.params.flash_attention = False
    pl.seed_everything(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    motion_fps = cfg.model.params.modality_setup.params.motion_fps

    model = build_model(cfg)
    load_pretrained_vae(cfg, model, logger, phase="demo")
    load_pretrained_lm(cfg, model, logger, phase="demo")
    model.to(device)
    cast = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    if cast != torch.float32:
        model.to(cast)
    model.eval()
    # generate_direct() normally sets this; we bypass it, so set it manually.
    model.lm.device = device

    def inverse_selection_tensor(filtered_t, selection_array, n):
        sel = torch.from_numpy(selection_array).to(filtered_t.device)
        out = torch.zeros((n, 165)).to(filtered_t.device)
        idx = torch.where(sel == 1)[0]
        for i in range(n):
            out[i, idx] = filtered_t[i]
        return out

    def reconstruct(outputs):
        of = [t.float() for t in outputs["face"]]
        oh = [t.float() for t in outputs["hand"]]
        ol = [t.float() for t in outputs["lower"]]
        ou = [t.float() for t in outputs["upper"]]
        # For text->motion LoM only emits upper/lower tokens; face/hand come back as
        # length-1 placeholders ([0]).  Match Language_Motion.forward(): take the MAX
        # length across parts and zero-pad the short ones (NOT min, which collapses
        # everything to 1 token -> 4 frames).
        L = max(of[0].shape[0], oh[0].shape[0], ol[0].shape[0], ou[0].shape[0])

        def _pad(t):
            if t.shape[0] >= L:
                return t[:L]
            return torch.cat([t, torch.zeros(L - t.shape[0], dtype=t.dtype, device=t.device)], dim=0)

        rec_index_face = torch.stack([_pad(t) for t in of], dim=0)
        rec_index_hands = torch.stack([_pad(t) for t in oh], dim=0)
        rec_index_lower = torch.stack([_pad(t) for t in ol], dim=0)
        rec_index_upper = torch.stack([_pad(t) for t in ou], dim=0)
        rec_index_face = torch.clamp(rec_index_face, 0, model.lm.face_codebook_size - 1)
        rec_index_upper = torch.clamp(rec_index_upper, 0, model.lm.upper_codebook_size - 1)
        rec_index_lower = torch.clamp(rec_index_lower, 0, model.lm.lower_codebook_size - 1)
        rec_index_hands = torch.clamp(rec_index_hands, 0, model.lm.hand_codebook_size - 1)

        for vae in (model.vae_global, model.vae_face, model.vae_upper, model.vae_lower, model.vae_hand):
            vae.float()
        rec_face = model.vae_face.decode(rec_index_face.int()).float()
        rec_upper = model.vae_upper.decode(rec_index_upper.int()).float()
        rec_lower = model.vae_lower.decode(rec_index_lower.int()).float()
        rec_hands = model.vae_hand.decode(rec_index_hands.int()).float()

        rec_exps = rec_face[:, :, 6:]
        rec_pose_jaw = rec_face[:, :, :6]
        rec_pose_legs = rec_lower[:, :, :54]
        bs, n = rec_pose_jaw.shape[0], rec_pose_jaw.shape[1]

        rec_pose_upper = rec_upper.reshape(bs, n, 13, 6)
        rec_pose_upper = rotation_6d_to_axis_angle(rec_pose_upper).reshape(bs * n, 13 * 3)
        rec_pose_upper_recover = inverse_selection_tensor(rec_pose_upper.to(device), JOINT_MASK_UPPER, bs * n)

        rec_pose_lower = rec_pose_legs.reshape(bs, n, 9, 6)
        rec_pose_lower = rotation_6d_to_matrix(rec_pose_lower)
        rec_lower2global = matrix_to_rotation_6d(rec_pose_lower.clone()).reshape(bs, n, 9 * 6)
        rec_pose_lower = matrix_to_axis_angle(rec_pose_lower).reshape(bs * n, 9 * 3)
        rec_pose_lower_recover = inverse_selection_tensor(rec_pose_lower.to(device), JOINT_MASK_LOWER, bs * n)

        rec_pose_hands = rec_hands.reshape(bs, n, 30, 6)
        rec_pose_hands = rotation_6d_to_axis_angle(rec_pose_hands).reshape(bs * n, 30 * 3)
        rec_pose_hands_recover = inverse_selection_tensor(rec_pose_hands.to(device), JOINT_MASK_HAND, bs * n)

        rec_pose_jaw = rec_pose_jaw.reshape(bs * n, 6)
        rec_pose_jaw = rotation_6d_to_axis_angle(rec_pose_jaw).reshape(bs * n, 1 * 3)

        rec_pose = rec_pose_upper_recover + rec_pose_lower_recover + rec_pose_hands_recover
        rec_pose[:, 66:69] = rec_pose_jaw

        to_global = rec_lower
        if to_global.shape[2] == 54:
            to_global = F.pad(to_global, (0, 7))
        to_global[:, :, 54:57] = 0.0
        to_global[:, :, :54] = rec_lower2global
        rec_global = model.vae_global(to_global)
        rec_trans_v_s = rec_global["rec_pose"][:, :, 54:57]
        rec_x = velocity2position(rec_trans_v_s[:, :, 0:1], 1 / motion_fps,
                                  torch.zeros(rec_trans_v_s[:, 0, 0:1].shape, device=device))
        rec_z = velocity2position(rec_trans_v_s[:, :, 2:3], 1 / motion_fps,
                                  torch.zeros(rec_trans_v_s[:, 0, 2:3].shape, device=device))
        rec_y = rec_trans_v_s[:, :, 1:2]
        rec_trans = torch.cat([rec_x, rec_y, rec_z], dim=-1)

        poses = rec_pose.detach().cpu().numpy().reshape(n, 55 * 3)
        exps = rec_exps.detach().cpu().numpy().reshape(n, 100)
        trans = rec_trans.detach().cpu().numpy().reshape(n, 3)
        return poses, exps, trans

    betas = np.zeros(300, dtype=np.float32)
    bs = max(1, args.batch_size)
    with torch.no_grad():
        for start in tqdm(range(0, len(jobs), bs), desc=f"LoM[{args.shard_index}]"):
            chunk = jobs[start:start + bs]
            names = [n for n, _ in chunk]
            caps = [c for _, c in chunk]
            try:
                of, oh, ou, ol, _ = model.lm.generate_direct(input=caps, do_sample=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] batch@{start}: {exc}", flush=True)
                continue
            for i, name in enumerate(names):
                try:
                    if min(of[i].shape[0], oh[i].shape[0], ou[i].shape[0], ol[i].shape[0]) < 1:
                        raise ValueError("empty token sequence")
                    outputs = {"face": [of[i]], "hand": [oh[i]], "upper": [ou[i]], "lower": [ol[i]]}
                    poses, exps, trans = reconstruct(outputs)
                except Exception as exc:  # noqa: BLE001
                    print(f"[warn] {name}: {exc}", flush=True)
                    continue
                # Save both the raw SMPL-X NPZ layout and the split keys
                # (transl/global_orient/body_pose) consumed by
                # convert_smplx_npz_to_135d (135-dim = transl + go_rot6d + body21_rot6d).
                poses = poses.astype(np.float32)
                np.savez(out_dir / f"{name}.npz", betas=betas, poses=poses,
                         expressions=exps.astype(np.float32), trans=trans.astype(np.float32),
                         transl=trans.astype(np.float32),
                         global_orient=poses[:, 0:3].copy(),
                         body_pose=poses[:, 3:66].copy(),
                         model="smplx2020", gender="neutral", mocap_frame_rate=30)


if __name__ == "__main__":
    main()
