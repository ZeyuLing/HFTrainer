"""Compare PrismTrainer.train_step against a manual loss computation.

This diagnostic is intentionally narrow: it checks whether the loss printed by
the real trainer path can be reproduced by the hand-written teacher-forced loss
scanner for the same checkpoint, batch, random timestep and noise.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import hftrainer  # noqa: F401
from mmengine.config import Config
from torch.utils.data import DataLoader, Subset

from hftrainer.datasets.motion.motionhub.flexible_collate import flexible_collate
from hftrainer.registry import DATASETS, MODEL_BUNDLES, TRAINERS
from hftrainer.utils.checkpoint_utils import find_latest_checkpoint, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/prism/prism_1b_tp2m_multiframe_kt_spectral_unified_t5cached_overfit100.py",
    )
    parser.add_argument("--checkpoint", default="auto")
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument(
        "--model-train-mode",
        action="store_true",
        help="Run bundle/trainer in train() mode instead of the default eval() mode.",
    )
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def resolve_checkpoint(cfg, checkpoint: str, work_dir: Optional[str]) -> str:
    if checkpoint != "auto":
        return checkpoint
    latest = find_latest_checkpoint(work_dir or cfg.work_dir)
    if latest is None:
        raise FileNotFoundError("No checkpoint found")
    return latest


def build_bundle(cfg, checkpoint: str, device: str):
    model_cfg = cfg.model.to_dict() if hasattr(cfg.model, "to_dict") else dict(cfg.model)
    bundle_cls = MODEL_BUNDLES.get(model_cfg["type"])
    bundle = bundle_cls.from_config(model_cfg)
    bundle.load_state_dict_selective(load_checkpoint(checkpoint, map_location="cpu"), strict=False)
    return bundle.to(device)


def build_trainer(cfg, bundle):
    trainer_cfg = cfg.trainer.to_dict() if hasattr(cfg.trainer, "to_dict") else dict(cfg.trainer)
    return TRAINERS.build(trainer_cfg, default_args={"bundle": bundle})


def build_loader(cfg, num_samples: int, batch_size: int):
    dataset_cfg = cfg.train_dataloader.dataset
    if hasattr(dataset_cfg, "to_dict"):
        dataset_cfg = dataset_cfg.to_dict()
    dataset = DATASETS.build(dataset_cfg)
    if num_samples > 0 and num_samples < len(dataset):
        dataset = Subset(dataset, list(range(num_samples)))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=flexible_collate,
    )


def move_batch_to_device(batch: Dict[str, Any], device: str) -> Dict[str, Any]:
    out = dict(batch)
    for key, value in list(out.items()):
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device)
    return out


@torch.no_grad()
def manual_train_loss(bundle, trainer, batch: Dict[str, Any]) -> Dict[str, float]:
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

    condition_frame_mask_vae = bundle.create_condition_mask(
        latents,
        frame_condition_rate=trainer.frame_condition_rate,
        condition_num_frames=trainer.condition_num_frames,
        num_frames=num_frames,
    )

    step_indices = torch.randint(
        0,
        len(bundle.scheduler.timesteps),
        (batch_size,),
        device=latents.device,
    )
    scheduler_timesteps = bundle.scheduler.timesteps.to(device=latents.device)
    timesteps = scheduler_timesteps[step_indices]

    noisy_latents, targets = bundle.add_flow_noise(latents, timesteps)
    noisy_latents = torch.where(condition_frame_mask_vae, noisy_latents, latents)

    transformer_module = getattr(bundle.transformer, "module", bundle.transformer)
    seq_ts = bundle.create_sequence_ts(
        timesteps,
        condition_frame_mask_vae,
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
    condition_mask = condition_frame_mask_vae.expand_as(mse).float()
    padding_mask_full = padding_mask.unsqueeze(1).expand_as(mse).float()
    full_mask = condition_mask * padding_mask_full

    mse_transl = mse[:, :, :, :1]
    mask_transl = full_mask[:, :, :, :1]
    loss_transl = (mse_transl * mask_transl).sum() / (mask_transl.sum() + 1e-6)

    mse_rot = mse[:, :, :, 1:]
    mask_rot = full_mask[:, :, :, 1:]
    loss_rot = (mse_rot * mask_rot).sum() / (mask_rot.sum() + 1e-6)

    w_t = trainer.translation_loss_weight
    loss = w_t * loss_transl + (1.0 - w_t) * loss_rot
    return {
        "loss": float(loss.detach().cpu()),
        "loss_transl": float(loss_transl.detach().cpu()),
        "loss_rot": float(loss_rot.detach().cpu()),
        "step_indices": [int(x) for x in step_indices.detach().cpu().tolist()],
        "timesteps": [int(x) for x in timesteps.detach().cpu().tolist()],
    }


def tensor_scalar(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu())
    return float(value)


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    checkpoint = resolve_checkpoint(cfg, args.checkpoint, args.work_dir)
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"

    bundle = build_bundle(cfg, checkpoint, device)
    trainer = build_trainer(cfg, bundle)
    if args.model_train_mode:
        bundle.train()
        trainer.train()
    else:
        bundle.eval()
        trainer.eval()

    loader = build_loader(cfg, args.num_samples, args.batch_size)
    batch = None
    for idx, item in enumerate(loader):
        if idx == args.batch_index:
            batch = move_batch_to_device(item, device)
            break
    if batch is None:
        raise IndexError(f"batch-index {args.batch_index} out of range")

    rows = []
    for rep in range(args.repeats):
        seed = args.seed + rep
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        train_out = trainer.train_step(batch)

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        manual_out = manual_train_loss(bundle, trainer, batch)

        rows.append({
            "seed": seed,
            "trainer": {
                "loss": tensor_scalar(train_out["loss"]),
                "loss_transl": tensor_scalar(train_out["loss_transl"]),
                "loss_rot": tensor_scalar(train_out["loss_rot"]),
            },
            "manual": manual_out,
            "abs_diff": abs(tensor_scalar(train_out["loss"]) - manual_out["loss"]),
        })

    result = {
        "checkpoint": checkpoint,
        "batch_size": int(batch["motion"].shape[0]),
        "num_frames": [int(x) for x in batch["num_frames"].detach().cpu().tolist()],
        "trainer_cfg": {
            "frame_condition_rate": float(trainer.frame_condition_rate),
            "condition_num_frames": trainer.condition_num_frames,
            "prompt_drop_rate": float(trainer.prompt_drop_rate),
            "translation_loss_weight": float(trainer.translation_loss_weight),
            "use_fp16_autocast": bool(trainer.use_fp16_autocast),
        },
        "model_train_mode": bool(args.model_train_mode),
        "rows": rows,
    }
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)


if __name__ == "__main__":
    main()
