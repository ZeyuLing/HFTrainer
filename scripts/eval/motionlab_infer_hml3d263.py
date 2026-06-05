#!/usr/bin/env python3
"""Run MotionLab text-to-motion inference on the HumanML3D test split.

This wrapper intentionally avoids MotionLab's Lightning ``test.py`` path:
the upstream text-only test step updates metrics but does not return generated
motions to ``test_epoch_end``, so ``SAVE_PREDICTIONS`` cannot export the files
we need for cross-protocol evaluation.

The implementation below loads only the released text encoder and denoiser,
then mirrors ``RFMOTION_SEPERATE.diffusion_reverse`` for the text condition.
Predictions are saved as un-normalized HumanML3D-263 features keyed by the
reconstructed HumanML3D test ids.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm


REPO = Path(__file__).resolve().parents[2]
MOTIONLAB_ROOT = REPO / "ref_repo" / "MotionLab"
SRC_H3D272 = REPO / "ref_repo" / "MotionStreamer" / "MotionStreamer" / "humanml3d_272"
RECON_ROOT = REPO / "work_dirs" / "h3d263_eval" / "h3d263_test_recon_fk"
OFFICIAL_HML_STATS = MOTIONLAB_ROOT / "datasets" / "all"


def _read_first_caption(text_file: Path) -> str | None:
    if not text_file.exists():
        return None
    for line in text_file.read_text().splitlines():
        parts = line.strip().split("#")
        if len(parts) < 4:
            continue
        try:
            f_tag = float(parts[2])
            to_tag = float(parts[3])
        except ValueError:
            continue
        f_tag = 0.0 if np.isnan(f_tag) else f_tag
        to_tag = 0.0 if np.isnan(to_tag) else to_tag
        if f_tag == 0.0 and to_tag == 0.0 and parts[0].strip():
            return parts[0].strip()
    return None


def _load_cfg(args):
    os.chdir(str(MOTIONLAB_ROOT))
    if str(MOTIONLAB_ROOT) not in sys.path:
        sys.path.insert(0, str(MOTIONLAB_ROOT))

    from rfmotion.config import get_module_config  # noqa: WPS433

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    ckpt_cfg = None
    if isinstance(ckpt, dict):
        ckpt_cfg = ckpt.get("datamodule_hyper_parameters", {}).get("cfg")

    if args.cfg_from_checkpoint and ckpt_cfg is not None:
        cfg = OmegaConf.create(ckpt_cfg)
        cfg_assets = OmegaConf.load(args.cfg_assets)
        cfg = OmegaConf.merge(cfg, cfg_assets)
    else:
        cfg_base = OmegaConf.load("configs/base.yaml")
        cfg_exp = OmegaConf.merge(cfg_base, OmegaConf.load(args.cfg))
        cfg_model = get_module_config(cfg_exp.model, cfg_exp.model.target)
        cfg_assets = OmegaConf.load(args.cfg_assets)
        cfg = OmegaConf.merge(cfg_exp, cfg_model, cfg_assets)

    cfg.DEBUG = False
    cfg.ACCELERATOR = "gpu"
    cfg.DEVICE = [0]
    cfg.TRAIN.STAGE = "diffusion"
    cfg.TRAIN.ABLATION.VAE = False
    cfg.DATASET.NFEATS = 263
    cfg.DATASET.NJOINTS = 22
    cfg.model.denoiser.params.nfeats = 263
    cfg.METRIC.TYPE = []
    cfg.TEST.CHECKPOINTS = args.checkpoint
    if args.clip_path:
        cfg.model.clip_path = args.clip_path
        cfg.model.text_encoder.params.modelpath = args.clip_path
    return cfg


def _load_modules(cfg, device: torch.device):
    from rfmotion.config import instantiate_from_config  # noqa: WPS433
    from rfmotion.models.operator.scheduling_flow_match_euler_discrete import (  # noqa: WPS433
        FlowMatchEulerDiscreteScheduler,
    )

    text_encoder = instantiate_from_config(cfg.model.text_encoder).eval().to(device)
    denoiser = instantiate_from_config(cfg.model.denoiser).eval().to(device)
    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=cfg.model.noise_scheduler.params.num_train_timesteps
    )

    ckpt = torch.load(cfg.TEST.CHECKPOINTS, map_location="cpu")
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    den_state = {
        k.replace("denoiser.", "", 1): v
        for k, v in state.items()
        if k.startswith("denoiser.")
    }
    missing, unexpected = denoiser.load_state_dict(den_state, strict=False)
    print(f"[+] denoiser loaded: missing={len(missing)} unexpected={len(unexpected)}")
    if unexpected:
        print("    unexpected:", unexpected[:10])
    return text_encoder, denoiser, scheduler


def _make_jobs(recon_root: Path, src_h3d272: Path, max_samples: int | None,
               num_shards: int, shard_index: int):
    ids = [s.strip() for s in (recon_root / "test.txt").read_text().splitlines() if s.strip()]
    jobs = []
    for sid in ids:
        m_file = recon_root / "new_joint_vecs" / f"{sid}.npy"
        if not m_file.exists():
            continue
        length = int(np.load(m_file, mmap_mode="r").shape[0])
        if length < 40 or length >= 200:
            continue
        caption = _read_first_caption(src_h3d272 / "texts" / f"{sid}.txt")
        if not caption:
            continue
        jobs.append((sid, caption, (length // 4) * 4))
        if max_samples and len(jobs) >= max_samples:
            break
    if num_shards > 1:
        jobs = jobs[shard_index::num_shards]
    return jobs


def _sample_text_batch(
    cfg,
    text_encoder,
    denoiser,
    scheduler,
    captions,
    lengths,
    device,
    stage,
    num_steps=None,
):
    bsz = len(captions)
    max_len = max(lengths)
    target_motion = torch.zeros((bsz, max_len, 263), device=device, dtype=torch.float32)
    noisy_latents = torch.randn_like(target_motion)
    if num_steps is not None:
        steps = int(num_steps)
    elif stage == "demo":
        steps = int(cfg.model.scheduler.num_demo_steps)
    else:
        steps = int(cfg.model.scheduler.num_eval_steps)
    scheduler.set_timesteps(num_inference_steps=steps, device=device)
    text_lengths = [0] * bsz + [77] * bsz
    with torch.no_grad():
        instruction_uncond = text_encoder("reconstruct given masked source motion.")[0][0]
        instruction_text = text_encoder("generate motion by given text.")[0][0]
        instructions = torch.cat([
            instruction_uncond.repeat(bsz, 1),
            instruction_text.repeat(bsz, 1),
        ], dim=0)
        text = text_encoder([""] * bsz + list(captions))
        for t in scheduler.timesteps.to(torch.int32):
            if int(t.item()) == 0:
                continue
            latent_model_input = torch.cat([noisy_latents] * 2)
            v_pred = denoiser(
                instructions=instructions,
                hidden_states=latent_model_input,
                timestep=t,
                text=text,
                text_lengths=text_lengths,
                hint=None,
                hint_lengths=None,
                style=None,
                style_lengths=None,
                content=None,
                content_lengths=None,
                source_motion=None,
                source_lengths=None,
                source_lengths_z=None,
                target_lengths=list(lengths) + list(lengths),
                target_lengths_z=list(lengths) + list(lengths),
                return_dict=False,
            )[0]
            v_uncond, v_cond = v_pred.chunk(2)
            v_pred = v_uncond + cfg.model.text_guidance_scale * (v_cond - v_uncond)
            noisy_latents = scheduler.step(v_pred, t, noisy_latents, return_dict=False)[0]
    return noisy_latents


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="configs/config_rfmotion_text.yaml")
    parser.add_argument("--cfg-assets", default="configs/assets.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/motionflow/motionflow/motionflow.ckpt")
    parser.add_argument(
        "--no-cfg-from-checkpoint",
        dest="cfg_from_checkpoint",
        action="store_false",
        help="Use --cfg instead of the training cfg saved in the checkpoint.",
    )
    parser.set_defaults(cfg_from_checkpoint=True)
    parser.add_argument("--clip-path", default="openai/clip-vit-large-patch14")
    parser.add_argument("--recon-root", default=str(RECON_ROOT))
    parser.add_argument("--src-h3d272", default=str(SRC_H3D272))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--stats-root",
        default=str(OFFICIAL_HML_STATS),
        help="Directory containing the official HumanML3D Mean.npy/Std.npy used to denormalize model outputs.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Override the official stage step count. "
             "MotionLab eval uses cfg.model.scheduler.num_eval_steps; "
             "demo uses cfg.model.scheduler.num_demo_steps.",
    )
    parser.add_argument(
        "--stage",
        choices=["demo", "eval"],
        default="demo",
        help="Official MotionLab sampling stage. Use demo for the released visual demo "
             "(usually 201 steps) and eval for the paper evaluator path "
             "(the released checkpoint stores 21 steps).",
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir).resolve()
    recon_root = Path(args.recon_root).resolve()
    src_h3d272 = Path(args.src_h3d272).resolve()
    stats_root = Path(args.stats_root).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = _load_cfg(args)
    text_encoder, denoiser, scheduler = _load_modules(cfg, device)

    jobs = _make_jobs(
        recon_root, src_h3d272, args.max_samples,
        args.num_shards, args.shard_index,
    )
    mean = torch.from_numpy(np.load(stats_root / "Mean.npy")).float().to(device)
    std = torch.from_numpy(np.load(stats_root / "Std.npy")).float().to(device)
    print(f"[+] denorm_stats={stats_root}")
    print(f"[+] jobs={len(jobs)} shard={args.shard_index}/{args.num_shards} "
          f"out={out_dir} device={device} stage={args.stage} "
          f"steps={args.num_steps or (cfg.model.scheduler.num_demo_steps if args.stage == 'demo' else cfg.model.scheduler.num_eval_steps)}")

    written = skipped = failed = 0
    for start in tqdm(range(0, len(jobs), args.batch_size), ncols=80):
        chunk = jobs[start:start + args.batch_size]
        todo = []
        for item in chunk:
            sid = item[0]
            if args.skip_existing and (out_dir / f"{sid}.npy").exists():
                skipped += 1
            else:
                todo.append(item)
        if not todo:
            continue
        ids, captions, lengths = zip(*todo)
        try:
            pred_norm = _sample_text_batch(
                cfg,
                text_encoder,
                denoiser,
                scheduler,
                captions,
                lengths,
                device,
                args.stage,
                args.num_steps,
            )
            pred = pred_norm * std + mean
            pred = pred.detach().cpu().numpy().astype(np.float32)
            for i, (sid, _cap, length) in enumerate(todo):
                np.save(out_dir / f"{sid}.npy", pred[i, :length])
                written += 1
        except Exception as exc:  # noqa: BLE001
            failed += len(todo)
            print(f"[fail] batch={start}: {type(exc).__name__}: {exc}", flush=True)
    print(f"[done] written={written} skipped={skipped} failed={failed}")


if __name__ == "__main__":
    main()
