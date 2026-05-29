"""Scan exact PRISM trainer loss over fixed timesteps for overfit debugging."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import hftrainer  # noqa: F401
from mmengine.config import Config
from torch.utils.data import DataLoader, Subset

from hftrainer.datasets.motion.motionhub.flexible_collate import flexible_collate
from hftrainer.registry import DATASETS, MODEL_BUNDLES
from hftrainer.utils.checkpoint_utils import find_latest_checkpoint, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/prism/prism_1b_tp2m_multiframe_kt_spectral_unified_t5cached_overfit100.py",
    )
    parser.add_argument("--checkpoint", default="auto")
    parser.add_argument(
        "--frozen-module-checkpoint",
        default=None,
        help=(
            "Optional checkpoint used only to restore frozen latent-space "
            "modules (vae, smpl_pose_processor, and bundle latent stats)."
        ),
    )
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--model-train-mode",
        action="store_true",
        help="Set bundle.train() before scanning instead of the default eval().",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--indices", default="0,50,100,250,500,750,999")
    parser.add_argument("--random-draws", type=int, default=8)
    parser.add_argument(
        "--all-batches",
        action="store_true",
        help="Loop over the whole selected dataset and report aggregate losses.",
    )
    parser.add_argument(
        "--legacy-spectral-unified-l2",
        action="store_true",
        help=(
            "Diagnostic only: rebuild spectral_unified joint RoPE buffers with "
            "the old L2-norm spectral positions used before the toporesid fix."
        ),
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


def _filter_frozen_latent_state(state_dict):
    return {
        key: state_dict[key]
        for key in ("vae", "smpl_pose_processor", "__bundle_params__")
        if key in state_dict
    }


def build_bundle(
    cfg,
    checkpoint: str,
    device: str,
    frozen_module_checkpoint: Optional[str] = None,
):
    model_cfg = cfg.model.to_dict() if hasattr(cfg.model, "to_dict") else dict(cfg.model)
    bundle_cls = MODEL_BUNDLES.get(model_cfg["type"])
    bundle = bundle_cls.from_config(model_cfg)
    bundle.load_state_dict_selective(load_checkpoint(checkpoint, map_location="cpu"), strict=False)
    if frozen_module_checkpoint:
        bundle.load_state_dict_selective(
            _filter_frozen_latent_state(
                load_checkpoint(frozen_module_checkpoint, map_location="cpu")
            ),
            strict=False,
        )
    bundle.eval()
    return bundle.to(device)


def patch_legacy_spectral_unified_l2(bundle) -> None:
    """Recreate the pre-toporesid spectral_unified joint RoPE buffers.

    The current code computes body-token positions as 1..22 plus a small signed
    spectral residual.  Checkpoints trained before that code was loaded used
    L2-norm spectral positions instead.  This patch is intentionally local to
    this diagnostic script so we can test whether a checkpoint was trained under
    the old in-memory RoPE without changing runtime code.
    """
    import numpy as np
    from hftrainer.models.motion.prism.network.motion_rope import _compute_spectral_coords

    transformer = getattr(bundle.transformer, "module", bundle.transformer)
    rope = getattr(transformer, "rope", None)
    if rope is None or getattr(rope, "joint_pos_mode", None) != "spectral_unified":
        return

    device = rope.joint_freqs_cos.device
    j_dim = rope._j_dim
    theta = 10000.0
    freqs_dtype = torch.float64
    spectral_scale = getattr(rope, "spectral_scale", None)
    scale = spectral_scale if spectral_scale is not None else 22.0
    coords = _compute_spectral_coords(
        num_joints=22,
        num_modes=getattr(rope, "num_spectral_modes", 4),
    )
    positions = np.linalg.norm(coords, axis=1)
    max_pos = positions.max()
    if max_pos > 1e-8:
        positions = positions * (scale / max_pos)

    pos = torch.from_numpy(positions).to(device=device, dtype=freqs_dtype)
    half_dim = j_dim // 2
    freq_seq = torch.arange(0, half_dim, dtype=freqs_dtype, device=device)
    freqs = 1.0 / (theta ** (2.0 * freq_seq / j_dim))
    angles = torch.outer(pos, freqs)
    rope.joint_freqs_cos = torch.cos(angles).float().repeat_interleave(2, dim=1)
    rope.joint_freqs_sin = torch.sin(angles).float().repeat_interleave(2, dim=1)


def build_dataset(cfg, num_samples: int):
    dataset_cfg = cfg.train_dataloader.dataset
    if hasattr(dataset_cfg, "to_dict"):
        dataset_cfg = dataset_cfg.to_dict()
    dataset = DATASETS.build(dataset_cfg)
    if num_samples > 0 and num_samples < len(dataset):
        dataset = Subset(dataset, list(range(num_samples)))
    return dataset


@torch.no_grad()
def trainer_loss_at(bundle, batch: Dict[str, object], timestep_indices: torch.Tensor) -> Dict[str, float]:
    motion = batch["motion"]
    num_frames = batch["num_frames"]
    if not isinstance(num_frames, torch.Tensor):
        num_frames = torch.as_tensor(num_frames)
    latents = bundle.encode_motion(motion)
    batch_size, _, latent_frames, latent_joints = latents.shape
    device = latents.device
    padding_mask = bundle.create_padding_mask(
        num_frames=num_frames,
        batch_size=batch_size,
        latent_frames=latent_frames,
        latent_joints=latent_joints,
        device=device,
    )
    transformer_dtype = next(bundle.transformer.parameters()).dtype
    text_states = batch["t5_text_embeds"].to(device=device, dtype=transformer_dtype)
    text_mask = batch["t5_text_mask"].to(device=device)
    scheduler_timesteps = bundle.scheduler.timesteps.to(device=device)
    timesteps = scheduler_timesteps[timestep_indices.to(device)]
    noisy_latents, targets = bundle.add_flow_noise(latents, timesteps)
    condition_mask_vae = torch.ones(
        batch_size, 1, latent_frames, latent_joints, device=device, dtype=torch.bool
    )
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
        hidden_states_mask=padding_mask,
        encoder_hidden_states_mask=text_mask,
    ).float()
    mse = F.mse_loss(model_pred, targets.float(), reduction="none")
    full_mask = condition_mask_vae.expand_as(mse).float() * padding_mask.unsqueeze(1).expand_as(mse).float()
    mse_transl = mse[:, :, :, :1]
    mask_transl = full_mask[:, :, :, :1]
    loss_transl = (mse_transl * mask_transl).sum() / (mask_transl.sum() + 1e-6)
    mse_rot = mse[:, :, :, 1:]
    mask_rot = full_mask[:, :, :, 1:]
    loss_rot = (mse_rot * mask_rot).sum() / (mask_rot.sum() + 1e-6)
    loss = 0.5 * loss_transl + 0.5 * loss_rot
    pred_norm = (model_pred * full_mask).pow(2).sum().sqrt() / (full_mask.sum().sqrt() + 1e-6)
    target_norm = (targets.float() * full_mask).pow(2).sum().sqrt() / (full_mask.sum().sqrt() + 1e-6)
    return {
        "loss": float(loss.cpu()),
        "loss_transl": float(loss_transl.cpu()),
        "loss_rot": float(loss_rot.cpu()),
        "pred_rms": float(pred_norm.cpu()),
        "target_rms": float(target_norm.cpu()),
        "timesteps": [int(x) for x in timesteps.detach().cpu().tolist()],
        "sigmas": [
            float(bundle.scheduler.sigmas[int(i)].detach().cpu())
            for i in timestep_indices.detach().cpu().tolist()
        ],
    }


def _mean_dict(rows: List[Dict[str, float]]) -> Dict[str, float]:
    keys = [k for k, v in rows[0].items() if isinstance(v, (int, float))]
    return {k: float(sum(float(row[k]) for row in rows) / len(rows)) for k in keys}


@torch.no_grad()
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    cfg = Config.fromfile(args.config)
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    checkpoint = resolve_checkpoint(cfg, args.checkpoint, args.work_dir)
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    bundle = build_bundle(
        cfg,
        checkpoint,
        device,
        frozen_module_checkpoint=args.frozen_module_checkpoint,
    )
    if args.legacy_spectral_unified_l2:
        patch_legacy_spectral_unified_l2(bundle)
    if args.model_train_mode:
        bundle.train()
    dataset = build_dataset(cfg, args.num_samples)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=flexible_collate,
    )
    batches = []
    for batch in loader:
        batch["motion"] = batch["motion"].to(device=device, dtype=torch.float32)
        batches.append(batch)
        if not args.all_batches:
            break

    fixed: Dict[str, Dict[str, float]] = {}
    for idx in [int(x) for x in args.indices.split(",") if x.strip()]:
        rows = []
        for batch in batches:
            timestep_indices = torch.full((batch["motion"].shape[0],), idx, dtype=torch.long)
            rows.append(trainer_loss_at(bundle, batch, timestep_indices))
        fixed[str(idx)] = _mean_dict(rows)

    random_draws: List[Dict[str, float]] = []
    n_steps = len(bundle.scheduler.timesteps)
    for _ in range(args.random_draws):
        rows = []
        for batch in batches:
            timestep_indices = torch.randint(0, n_steps, (batch["motion"].shape[0],))
            rows.append(trainer_loss_at(bundle, batch, timestep_indices))
        random_draws.append(_mean_dict(rows))

    result = {
        "checkpoint": checkpoint,
        "frozen_module_checkpoint": args.frozen_module_checkpoint,
        "legacy_spectral_unified_l2": bool(args.legacy_spectral_unified_l2),
        "all_batches": bool(args.all_batches),
        "num_batches": len(batches),
        "batch_size": int(batches[0]["motion"].shape[0]),
        "num_frames_first_batch": [int(x) for x in batches[0]["num_frames"].tolist()],
        "fixed": fixed,
        "random_draws": random_draws,
    }
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)


if __name__ == "__main__":
    main()
