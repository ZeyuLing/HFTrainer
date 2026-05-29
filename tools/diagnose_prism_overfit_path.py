"""Diagnose PRISM overfit mismatch between training loss and generation.

The checks intentionally reuse the overfit dataloader branch with cached T5:

1. VAE roundtrip: GT motion -> latent -> decoded motion.
2. Teacher-forced flow: use GT latent to create x_t and measure velocity MSE.
3. Oracle x1 reconstruction: x_t - sigma * predicted_velocity.
4. Free sampling: same path as text-only generation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import hftrainer  # noqa: F401 - populate registries
from mmengine.config import Config
from torch.utils.data import DataLoader, Subset

from hftrainer.datasets.motion.motionhub.flexible_collate import flexible_collate
from hftrainer.registry import DATASETS, MODEL_BUNDLES
from hftrainer.utils.checkpoint_utils import find_latest_checkpoint, load_checkpoint
from tools.eval_prism_overfit_cached_t5 import (
    generate_with_cached_t5,
    motion_to_positions,
    rot_geodesic_rad,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/prism/prism_1b_tp2m_multiframe_kt_spectral_unified_t5cached_overfit100.py",
    )
    parser.add_argument("--checkpoint", default="auto")
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--num-samples", type=int, default=2)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def resolve_checkpoint(cfg, checkpoint: str, work_dir: Optional[str]) -> str:
    if checkpoint != "auto":
        return checkpoint
    root = work_dir or cfg.work_dir
    latest = find_latest_checkpoint(root)
    if latest is None:
        raise FileNotFoundError(f"No checkpoint-* found under {root}")
    return latest


def build_bundle(cfg, checkpoint: str, device: str):
    model_cfg = cfg.model.to_dict() if hasattr(cfg.model, "to_dict") else dict(cfg.model)
    bundle_cls = MODEL_BUNDLES.get(model_cfg["type"])
    bundle = bundle_cls.from_config(model_cfg)
    state_dict = load_checkpoint(checkpoint, map_location="cpu")
    load_report = summarize_checkpoint_load(bundle, state_dict)
    bundle.load_state_dict_selective(state_dict, strict=False)
    bundle.eval()
    return bundle.to(device), load_report


def summarize_checkpoint_load(bundle, state_dict: Dict[str, object]) -> Dict[str, object]:
    report: Dict[str, object] = {"top_keys": sorted(list(state_dict.keys()))[:20]}
    transformer_sd = state_dict.get("transformer")
    if isinstance(transformer_sd, dict):
        target = getattr(bundle.transformer, "module", bundle.transformer)
        target_keys = set(target.state_dict().keys())
        ckpt_keys = set(transformer_sd.keys())
        missing = sorted(target_keys - ckpt_keys)
        unexpected = sorted(ckpt_keys - target_keys)
        mismatched = []
        target_sd = target.state_dict()
        for key in sorted(target_keys & ckpt_keys):
            value = transformer_sd[key]
            if isinstance(value, torch.Tensor) and value.shape != target_sd[key].shape:
                mismatched.append(
                    f"{key}: ckpt {tuple(value.shape)} vs model {tuple(target_sd[key].shape)}"
                )
        report.update(
            {
                "transformer_missing_count": len(missing),
                "transformer_unexpected_count": len(unexpected),
                "transformer_mismatched_count": len(mismatched),
                "transformer_missing_head": missing[:10],
                "transformer_unexpected_head": unexpected[:10],
                "transformer_mismatched_head": mismatched[:10],
            }
        )
    return report


def build_dataset(cfg, num_samples: int):
    dataset_cfg = cfg.train_dataloader.dataset
    if hasattr(dataset_cfg, "to_dict"):
        dataset_cfg = dataset_cfg.to_dict()
    dataset = DATASETS.build(dataset_cfg)
    if num_samples > 0 and num_samples < len(dataset):
        dataset = Subset(dataset, list(range(num_samples)))
    return dataset


def valid_motion(gt: torch.Tensor, pred: torch.Tensor, valid: int):
    valid = min(valid, int(gt.shape[0]), int(pred.shape[0]))
    gt_i = gt[:valid]
    pred_i = pred[:valid]
    pred_rot = pred_i[:, 6:].reshape(valid, 22, 6)
    gt_rot = gt_i[:, 6:].reshape(valid, 22, 6)
    result = {
        "motion_l2": float((pred_i - gt_i).norm(dim=-1).mean().cpu()),
        "transl_l2": float((pred_i[:, :6] - gt_i[:, :6]).norm(dim=-1).mean().cpu()),
        "rot6d_l2": float((pred_rot - gt_rot).norm(dim=-1).mean().cpu()),
        "mpjre_deg": float(rot_geodesic_rad(pred_rot, gt_rot).mean().cpu() * 180.0 / torch.pi),
    }
    return result


def maybe_mpjpe(bundle, pred_motion: torch.Tensor, gt_motion: torch.Tensor, valid: int):
    if getattr(bundle.smpl_pose_processor, "smpl_model", None) is None:
        return None
    valid = min(valid, int(gt_motion.shape[0]), int(pred_motion.shape[0]))
    pred_j = motion_to_positions(bundle, pred_motion[:valid])
    gt_j = motion_to_positions(bundle, gt_motion[:valid])
    return float((pred_j - gt_j).norm(dim=-1).mean().cpu() * 1000.0)


@torch.no_grad()
def decode_latents(bundle, latents: torch.Tensor) -> torch.Tensor:
    denorm = latents * bundle.latents_std.to(latents) + bundle.latents_mean.to(latents)
    with torch.autocast(denorm.device.type, enabled=False):
        motion = bundle.vae.decode(denorm.float())
    motion = motion.reshape(motion.shape[0], motion.shape[1], -1)
    return bundle.smpl_pose_processor.denormalize(motion.float())


@torch.no_grad()
def diagnose(args) -> Dict[str, object]:
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    cfg = Config.fromfile(args.config)
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    checkpoint = resolve_checkpoint(cfg, args.checkpoint, args.work_dir)
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"

    bundle, load_report = build_bundle(cfg, checkpoint, device)
    dataset = build_dataset(cfg, args.num_samples)
    loader = DataLoader(
        dataset,
        batch_size=args.num_samples,
        shuffle=False,
        num_workers=0,
        collate_fn=flexible_collate,
    )
    batch = next(iter(loader))
    gt = batch["motion"].to(device=device, dtype=torch.float32)
    if gt.ndim == 2:
        gt = gt.unsqueeze(0)
    num_frames = batch["num_frames"]
    if not isinstance(num_frames, torch.Tensor):
        num_frames = torch.as_tensor(num_frames)

    latents = bundle.encode_motion(gt)
    bsz, _, latent_frames, latent_joints = latents.shape
    transformer_dtype = next(bundle.transformer.parameters()).dtype
    text_states = batch["t5_text_embeds"].to(device=device, dtype=transformer_dtype)
    text_mask = batch["t5_text_mask"].to(device=device)
    padding_mask = bundle.create_padding_mask(
        num_frames=num_frames,
        batch_size=bsz,
        latent_frames=latent_frames,
        latent_joints=latent_joints,
        device=device,
    )
    condition_mask = torch.ones(
        bsz, 1, latent_frames, latent_joints, device=device, dtype=torch.bool
    )
    transformer_module = getattr(bundle.transformer, "module", bundle.transformer)

    roundtrip = decode_latents(bundle, latents)
    roundtrip_metrics = []
    for i in range(bsz):
        valid = int(num_frames[i].item())
        item = valid_motion(gt[i], roundtrip[i], valid)
        item["mpjpe_mm"] = maybe_mpjpe(bundle, roundtrip[i], gt[i], valid)
        roundtrip_metrics.append(item)

    teacher = {}
    scheduler_timesteps = bundle.scheduler.timesteps.to(device=device)
    probe_indices = sorted(set([0, len(scheduler_timesteps) // 4, len(scheduler_timesteps) // 2, len(scheduler_timesteps) - 1]))
    for idx in probe_indices:
        timestep = scheduler_timesteps[idx].expand(bsz)
        noisy, targets = bundle.add_flow_noise(latents, timestep)
        seq_ts = bundle.create_sequence_ts(
            timestep,
            condition_mask,
            transformer_module.config.patch_size,
        )
        pred = bundle.transformer(
            hidden_states=noisy.to(dtype=transformer_dtype),
            encoder_hidden_states=text_states,
            timestep=seq_ts,
            hidden_states_mask=padding_mask,
            encoder_hidden_states_mask=text_mask,
        ).float()
        mse = F.mse_loss(pred, targets.float(), reduction="none")
        mask = padding_mask.unsqueeze(1).expand_as(mse).float()
        mse_all = float((mse * mask).sum().cpu() / (mask.sum().cpu() + 1e-6))
        sigma = bundle.scheduler.sigmas.to(device=device, dtype=latents.dtype)[idx]
        x1_hat = noisy - sigma * pred.to(noisy.dtype)
        recon = decode_latents(bundle, x1_hat)
        recon_metrics = []
        for i in range(bsz):
            valid = int(num_frames[i].item())
            item = valid_motion(gt[i], recon[i], valid)
            item["mpjpe_mm"] = maybe_mpjpe(bundle, recon[i], gt[i], valid)
            recon_metrics.append(item)
        teacher[str(int(timestep[0].item()))] = {
            "index": int(idx),
            "sigma": float(sigma.cpu()),
            "target_mse": mse_all,
            "oracle_recon": recon_metrics,
        }

    sampled = generate_with_cached_t5(
        bundle,
        batch["t5_text_embeds"],
        batch["t5_text_mask"],
        num_frames,
        num_steps=args.num_steps,
        guidance_scale=1.0,
    )
    sampled_metrics = []
    for i in range(bsz):
        valid = int(num_frames[i].item())
        item = valid_motion(gt[i], sampled[i], valid)
        item["mpjpe_mm"] = maybe_mpjpe(bundle, sampled[i], gt[i], valid)
        sampled_metrics.append(item)

    captions = batch.get("caption", [""] * bsz)
    return {
        "checkpoint": checkpoint,
        "num_samples": bsz,
        "num_steps": args.num_steps,
        "load_report": load_report,
        "latent_shape": list(latents.shape),
        "num_frames": [int(x) for x in num_frames.tolist()],
        "captions": [str(x) for x in captions],
        "roundtrip": roundtrip_metrics,
        "teacher_forced": teacher,
        "free_sampling": sampled_metrics,
    }


def main():
    args = parse_args()
    result = diagnose(args)
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)


if __name__ == "__main__":
    main()
