"""Measure PRISM loss before and after one real optimizer update."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import hftrainer  # noqa: F401
from mmengine.config import Config

from diagnose_prism_trainstep_equivalence import (
    build_bundle,
    build_loader,
    build_trainer,
    manual_train_loss,
    move_batch_to_device,
    tensor_scalar,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def batches(cfg, args, device: str):
    loader = build_loader(cfg, args.num_samples, args.batch_size)
    while True:
        for batch in loader:
            yield move_batch_to_device(batch, device)


def grad_norm(parameters) -> float:
    total = torch.zeros((), device="cuda" if torch.cuda.is_available() else "cpu")
    for param in parameters:
        if param.grad is None:
            continue
        total = total + param.grad.detach().float().pow(2).sum()
    return float(total.sqrt().detach().cpu())


def delta_norm(before: Dict[str, torch.Tensor], bundle) -> Dict[str, float]:
    total = 0.0
    samples = {}
    for name, param in bundle.transformer.named_parameters():
        if name not in before:
            continue
        diff = (param.detach().float().cpu() - before[name]).pow(2).sum().item()
        total += diff
        if name in {
            "patch_embedding.weight",
            "condition_embedder.time_embedder.linear_1.weight",
            "blocks.0.attn1.to_q.weight",
        }:
            samples[name] = diff ** 0.5
    samples["total"] = total ** 0.5
    return samples


@torch.no_grad()
def fixed_timestep_loss(bundle, trainer, batch: Dict[str, Any], timestep_index: int) -> Dict[str, float]:
    motion = batch["motion"]
    num_frames = batch.get("num_frames")
    latents = bundle.encode_motion(motion)
    batch_size, _, latent_frames, latent_joints = latents.shape
    padding_mask = bundle.create_padding_mask(
        num_frames=num_frames,
        batch_size=batch_size,
        latent_frames=latent_frames,
        latent_joints=latent_joints,
        device=latents.device,
    )
    transformer_dtype = next(bundle.transformer.parameters()).dtype
    text_states = batch["t5_text_embeds"].to(device=latents.device, dtype=transformer_dtype)
    text_mask = batch["t5_text_mask"].to(device=latents.device)
    scheduler_timesteps = bundle.scheduler.timesteps.to(device=latents.device)
    step_indices = torch.full(
        (batch_size,),
        int(timestep_index),
        device=latents.device,
        dtype=torch.long,
    )
    timesteps = scheduler_timesteps[step_indices]
    noisy_latents, targets = bundle.add_flow_noise(latents, timesteps)
    condition_mask_vae = bundle.create_condition_mask(
        latents,
        frame_condition_rate=trainer.frame_condition_rate,
        condition_num_frames=trainer.condition_num_frames,
        num_frames=num_frames,
    )
    noisy_latents = torch.where(condition_mask_vae, noisy_latents, latents)
    transformer_module = getattr(bundle.transformer, "module", bundle.transformer)
    seq_ts = bundle.create_sequence_ts(
        timesteps,
        condition_mask_vae,
        transformer_module.config.patch_size,
    )
    model_pred = bundle.transformer(
        hidden_states=noisy_latents.to(dtype=transformer_dtype),
        encoder_hidden_states=text_states,
        timestep=seq_ts,
        hidden_states_mask=padding_mask if num_frames is not None else None,
        encoder_hidden_states_mask=text_mask,
    ).float()
    mse = F.mse_loss(model_pred, targets.float(), reduction="none")
    full_mask = (
        condition_mask_vae.expand_as(mse).float()
        * padding_mask.unsqueeze(1).expand_as(mse).float()
    )
    mse_transl = mse[:, :, :, :1]
    mask_transl = full_mask[:, :, :, :1]
    loss_transl = (mse_transl * mask_transl).sum() / (mask_transl.sum() + 1e-6)
    mse_rot = mse[:, :, :, 1:]
    mask_rot = full_mask[:, :, :, 1:]
    loss_rot = (mse_rot * mask_rot).sum() / (mask_rot.sum() + 1e-6)
    loss = trainer.translation_loss_weight * loss_transl + (
        1.0 - trainer.translation_loss_weight
    ) * loss_rot
    return {
        "loss": float(loss.detach().cpu()),
        "loss_transl": float(loss_transl.detach().cpu()),
        "loss_rot": float(loss_rot.detach().cpu()),
    }


def fixed_timestep_suite(bundle, trainer, batch: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    return {
        str(idx): fixed_timestep_loss(bundle, trainer, batch, idx)
        for idx in (0, 500, 750, 999)
    }


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    bundle = build_bundle(cfg, args.checkpoint, device)
    trainer = build_trainer(cfg, bundle)
    bundle.train()
    trainer.train()
    batch_iter = batches(cfg, args, device)
    eval_batch = next(batch_iter)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    before_loss = manual_train_loss(bundle, trainer, eval_batch)
    before_fixed = fixed_timestep_suite(bundle, trainer, eval_batch)

    opt_cfg = cfg.optimizer.to_dict() if hasattr(cfg.optimizer, "to_dict") else dict(cfg.optimizer)
    opt_cfg = dict(opt_cfg)
    opt_type = opt_cfg.pop("type")
    if args.lr is not None:
        opt_cfg["lr"] = args.lr
    opt_cls = getattr(torch.optim, opt_type)
    optimizer = opt_cls(bundle.trainable_parameters(), **opt_cfg)

    watched = {
        name: param.detach().float().cpu().clone()
        for name, param in bundle.transformer.named_parameters()
        if param.requires_grad
    }

    step_rows = []
    last_loss = None
    gnorm = 0.0
    for step in range(args.steps):
        train_batch = eval_batch if step == 0 else next(batch_iter)
        torch.manual_seed(args.seed + step)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed + step)
        train_out = trainer.train_step(train_batch)
        loss = train_out["loss"]
        loss.backward()
        gnorm = grad_norm(bundle.trainable_parameters())
        optimizer.step()
        optimizer.zero_grad()
        last_loss = tensor_scalar(loss)
        step_rows.append({
            "step": step,
            "loss": last_loss,
            "loss_transl": tensor_scalar(train_out["loss_transl"]),
            "loss_rot": tensor_scalar(train_out["loss_rot"]),
            "grad_norm": gnorm,
        })

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    after_same_seed = manual_train_loss(bundle, trainer, eval_batch)
    torch.manual_seed(args.seed + 1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + 1)
    after_next_seed = manual_train_loss(bundle, trainer, eval_batch)
    after_fixed = fixed_timestep_suite(bundle, trainer, eval_batch)

    result = {
        "checkpoint": args.checkpoint,
        "optimizer": {"type": opt_type, **opt_cfg},
        "steps": int(args.steps),
        "last_train_step_loss": last_loss,
        "step_rows": step_rows,
        "before_same_seed": before_loss,
        "before_fixed_timestep": before_fixed,
        "after_same_seed": after_same_seed,
        "after_next_seed": after_next_seed,
        "after_fixed_timestep": after_fixed,
        "grad_norm": gnorm,
        "delta_norm": delta_norm(watched, bundle),
    }
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)


if __name__ == "__main__":
    main()
