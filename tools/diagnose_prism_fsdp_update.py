"""Run a tiny real Accelerate/FSDP PRISM update and replay fixed local loss."""

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
from torch.utils.data import DataLoader, Subset

from hftrainer.datasets.motion.motionhub.flexible_collate import flexible_collate
from hftrainer.registry import DATASETS
from hftrainer.runner.accelerate_runner import AccelerateRunner


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", default=None)
    parser.add_argument("--save-debug-checkpoint", action="store_true")
    return parser.parse_args()


def scalar(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().cpu())
    return float(value)


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def reduce_scalar(runner: AccelerateRunner, value: float) -> float:
    tensor = torch.tensor(float(value), device=runner.accelerator.device)
    tensor = runner.accelerator.reduce(tensor, reduction="mean")
    return float(tensor.detach().cpu())


@torch.no_grad()
def replay_loss(runner: AccelerateRunner, batch: Dict[str, Any], seed: int) -> Dict[str, float]:
    set_seed(seed)
    runner.bundle.train()
    runner.trainer.train()
    out = runner.trainer.train_step(batch)
    return {
        "loss": reduce_scalar(runner, scalar(out["loss"])),
        "loss_transl": reduce_scalar(runner, scalar(out["loss_transl"])),
        "loss_rot": reduce_scalar(runner, scalar(out["loss_rot"])),
    }


def grad_norm(runner: AccelerateRunner) -> float:
    total = torch.zeros((), device=runner.accelerator.device)
    for param in runner.bundle.trainable_parameters():
        if param.grad is not None:
            total = total + param.grad.detach().float().pow(2).sum()
    total = total.sqrt()
    total = runner.accelerator.reduce(total, reduction="mean")
    return float(total.detach().cpu())


@torch.no_grad()
def fixed_timestep_loss(
    runner: AccelerateRunner,
    batch: Dict[str, Any],
    timestep_index: int,
) -> Dict[str, float]:
    bundle = runner.bundle
    trainer = runner.trainer
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
    text_states = batch["t5_text_embeds"].to(
        device=latents.device,
        dtype=transformer_dtype,
    )
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
    pred = bundle.transformer(
        hidden_states=noisy_latents.to(dtype=transformer_dtype),
        encoder_hidden_states=text_states,
        timestep=seq_ts,
        hidden_states_mask=padding_mask if num_frames is not None else None,
        encoder_hidden_states_mask=text_mask,
    ).float()
    mse = F.mse_loss(pred, targets.float(), reduction="none")
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
        "loss": reduce_scalar(runner, scalar(loss)),
        "loss_transl": reduce_scalar(runner, scalar(loss_transl)),
        "loss_rot": reduce_scalar(runner, scalar(loss_rot)),
    }


def fixed_timestep_suite(runner: AccelerateRunner, batch: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    return {
        str(idx): fixed_timestep_loss(runner, batch, idx)
        for idx in (0, 500, 750, 999)
    }


def seeded_fixed_timestep_loss(
    runner: AccelerateRunner,
    batch: Dict[str, Any],
    timestep_index: int,
    seed: int,
) -> Dict[str, float]:
    set_seed(seed)
    return fixed_timestep_loss(runner, batch, timestep_index)


def raw_first_batch(cfg, batch_size: int, device: torch.device) -> Dict[str, Any]:
    dataset_cfg = cfg.train_dataloader.dataset
    if hasattr(dataset_cfg, "to_dict"):
        dataset_cfg = dataset_cfg.to_dict()
    dataset = DATASETS.build(dataset_cfg)
    dataset = Subset(dataset, list(range(batch_size)))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=flexible_collate,
    )
    batch = next(iter(loader))
    out = dict(batch)
    for key, value in list(out.items()):
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device)
    return out


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    runner = AccelerateRunner.from_cfg(cfg)

    trainable_modules = [
        getattr(runner.bundle, name)
        for name in runner.bundle._trainable_modules
        if isinstance(getattr(runner.bundle, name), torch.nn.Module)
    ]

    data_iter = iter(runner.train_dataloader)
    fixed_batch = next(data_iter)
    raw_eval_batch = raw_first_batch(cfg, fixed_batch["motion"].shape[0], runner.accelerator.device)

    pre = replay_loss(runner, fixed_batch, args.seed)
    pre_fixed = fixed_timestep_suite(runner, fixed_batch)
    pre_raw_fixed = fixed_timestep_suite(runner, raw_eval_batch)
    pre_raw_seeded_999 = seeded_fixed_timestep_loss(runner, raw_eval_batch, 999, args.seed)
    rows = []
    for step in range(args.steps):
        batch = fixed_batch if step == 0 else next(data_iter)
        set_seed(args.seed + step)
        runner.bundle.train()
        runner.trainer.train()
        with runner.accelerator.accumulate(*trainable_modules):
            out = runner.trainer.train_step(batch)
            loss = out["loss"]
            runner.accelerator.backward(loss)
            runner._sync_orphan_param_grads()
            gnorm = grad_norm(runner)
            if runner.max_grad_norm is not None:
                runner.accelerator.clip_grad_norm_(
                    list(runner.bundle.trainable_parameters()),
                    runner.max_grad_norm,
                )
            for opt in runner.optimizers.values():
                opt.step()
                opt.zero_grad()
            for sched in runner.lr_schedulers.values():
                sched.step()
        rows.append({
            "step": step,
            "train_loss": reduce_scalar(runner, scalar(out["loss"])),
            "train_loss_transl": reduce_scalar(runner, scalar(out["loss_transl"])),
            "train_loss_rot": reduce_scalar(runner, scalar(out["loss_rot"])),
            "grad_norm": gnorm,
        })

    post_same = replay_loss(runner, fixed_batch, args.seed)
    post_next = replay_loss(runner, fixed_batch, args.seed + 1)
    post_fixed = fixed_timestep_suite(runner, fixed_batch)
    post_raw_fixed = fixed_timestep_suite(runner, raw_eval_batch)
    post_raw_seeded_999 = seeded_fixed_timestep_loss(runner, raw_eval_batch, 999, args.seed)

    result = {
        "config": args.config,
        "steps": args.steps,
        "rank": runner.accelerator.process_index,
        "num_processes": runner.accelerator.num_processes,
        "pre_same_seed": pre,
        "pre_fixed_timestep": pre_fixed,
        "pre_raw_first_batch_fixed_timestep": pre_raw_fixed,
        "pre_raw_first_batch_seeded_999": pre_raw_seeded_999,
        "step_rows": rows,
        "post_same_seed": post_same,
        "post_next_seed": post_next,
        "post_fixed_timestep": post_fixed,
        "post_raw_first_batch_fixed_timestep": post_raw_fixed,
        "post_raw_first_batch_seeded_999": post_raw_seeded_999,
    }

    if args.save_debug_checkpoint:
        runner.current_epoch = 1
        runner.save_checkpoint()
        result["saved_checkpoint_epoch"] = 1

    runner.accelerator.wait_for_everyone()
    if runner.accelerator.is_main_process:
        text = json.dumps(result, indent=2, ensure_ascii=False)
        print(text)
        if args.output:
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as handle:
                handle.write(text)


if __name__ == "__main__":
    main()
