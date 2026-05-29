"""Per-sample PRISM overfit loss diagnostics.

This script recomputes the exact teacher-forced flow loss used by
``PrismTrainer.train_step`` for each sample in the overfit dataset.  It is meant
to answer a narrow debugging question: when the rank-0 training log is low, are
all 100 samples actually low under the saved checkpoint?
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import hftrainer  # noqa: F401 - populate registries
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
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--indices",
        default="0,250,500,750,999",
        help="Comma-separated scheduler timestep indices to probe.",
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
    bundle.eval()
    return bundle.to(device)


def build_dataset(cfg, num_samples: int):
    dataset_cfg = cfg.train_dataloader.dataset
    if hasattr(dataset_cfg, "to_dict"):
        dataset_cfg = dataset_cfg.to_dict()
    dataset = DATASETS.build(dataset_cfg)
    if num_samples > 0 and num_samples < len(dataset):
        dataset = Subset(dataset, list(range(num_samples)))
    return dataset


def _scalar(value: torch.Tensor) -> float:
    return float(value.detach().cpu())


@torch.no_grad()
def loss_for_batch(
    bundle,
    batch: Dict[str, object],
    timestep_index: int,
    device: torch.device | str,
) -> Dict[str, List[float]]:
    motion = batch["motion"].to(device=device, dtype=torch.float32)
    if motion.ndim == 2:
        motion = motion.unsqueeze(0)
    num_frames = batch["num_frames"]
    if not isinstance(num_frames, torch.Tensor):
        num_frames = torch.as_tensor(num_frames)

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
    timestep = scheduler_timesteps[timestep_index]
    timesteps = timestep.expand(batch_size)
    noisy_latents, targets = bundle.add_flow_noise(latents, timesteps)

    condition_mask = torch.ones(
        batch_size, 1, latent_frames, latent_joints, device=latents.device, dtype=torch.bool
    )
    transformer_module = getattr(bundle.transformer, "module", bundle.transformer)
    seq_ts = bundle.create_sequence_ts(
        timesteps,
        condition_mask,
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
    full_mask = condition_mask.expand_as(mse).float() * padding_mask.unsqueeze(1).expand_as(mse).float()
    mse_transl = mse[:, :, :, :1]
    mask_transl = full_mask[:, :, :, :1]
    mse_rot = mse[:, :, :, 1:]
    mask_rot = full_mask[:, :, :, 1:]

    losses = []
    losses_transl = []
    losses_rot = []
    pred_rms = []
    target_rms = []
    for item_idx in range(batch_size):
        lt = (mse_transl[item_idx] * mask_transl[item_idx]).sum() / (
            mask_transl[item_idx].sum() + 1e-6
        )
        lr = (mse_rot[item_idx] * mask_rot[item_idx]).sum() / (
            mask_rot[item_idx].sum() + 1e-6
        )
        loss = 0.5 * lt + 0.5 * lr
        mask_i = full_mask[item_idx]
        pred_norm = (model_pred[item_idx] * mask_i).pow(2).sum().sqrt() / (
            mask_i.sum().sqrt() + 1e-6
        )
        target_norm = (targets.float()[item_idx] * mask_i).pow(2).sum().sqrt() / (
            mask_i.sum().sqrt() + 1e-6
        )
        losses.append(_scalar(loss))
        losses_transl.append(_scalar(lt))
        losses_rot.append(_scalar(lr))
        pred_rms.append(_scalar(pred_norm))
        target_rms.append(_scalar(target_norm))

    sigma = float(bundle.scheduler.sigmas[timestep_index].detach().cpu())
    return {
        "timestep": [int(timestep.detach().cpu())] * batch_size,
        "sigma": [sigma] * batch_size,
        "loss": losses,
        "loss_transl": losses_transl,
        "loss_rot": losses_rot,
        "pred_rms": pred_rms,
        "target_rms": target_rms,
    }


def summarize(values: List[float]) -> Dict[str, float]:
    vals = sorted(float(v) for v in values)
    if not vals:
        return {}

    def q(frac: float) -> float:
        if len(vals) == 1:
            return vals[0]
        pos = frac * (len(vals) - 1)
        lo = int(pos)
        hi = min(lo + 1, len(vals) - 1)
        mix = pos - lo
        return vals[lo] * (1.0 - mix) + vals[hi] * mix

    return {
        "count": len(vals),
        "mean": sum(vals) / len(vals),
        "min": vals[0],
        "p25": q(0.25),
        "median": q(0.5),
        "p75": q(0.75),
        "p90": q(0.9),
        "max": vals[-1],
    }


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

    bundle = build_bundle(cfg, checkpoint, device)
    dataset = build_dataset(cfg, args.num_samples)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=flexible_collate,
    )
    indices = [int(x) for x in args.indices.split(",") if x.strip()]

    rows = []
    for batch_idx, batch in enumerate(loader):
        batch_size = batch["motion"].shape[0]
        captions = batch.get("caption", [""] * batch_size)
        paths = batch.get("motion_path", [""] * batch_size)
        num_frames = batch["num_frames"]
        if not isinstance(num_frames, torch.Tensor):
            num_frames = torch.as_tensor(num_frames)
        text_mask = batch["t5_text_mask"]
        per_index = {
            str(idx): loss_for_batch(bundle, batch, idx, device)
            for idx in indices
        }

        for item_idx in range(batch_size):
            row = {
                "sample_index": batch_idx * args.batch_size + item_idx,
                "motion_path": str(paths[item_idx]) if isinstance(paths, (list, tuple)) else str(paths),
                "caption": str(captions[item_idx]) if isinstance(captions, (list, tuple)) else str(captions),
                "num_frames": int(num_frames[item_idx].item()),
                "t5_tokens": int(text_mask[item_idx].sum().item()),
                "losses": {},
            }
            for idx in indices:
                data = per_index[str(idx)]
                row["losses"][str(idx)] = {
                    key: data[key][item_idx]
                    for key in ("timestep", "sigma", "loss", "loss_transl", "loss_rot", "pred_rms", "target_rms")
                }
            rows.append(row)

    summary = {}
    for idx in indices:
        key = str(idx)
        losses = [row["losses"][key]["loss"] for row in rows]
        summary[key] = summarize(losses)
        summary[key]["top_bad"] = [
            {
                "sample_index": row["sample_index"],
                "loss": row["losses"][key]["loss"],
                "motion_path": row["motion_path"],
                "caption": row["caption"],
            }
            for row in sorted(rows, key=lambda r: r["losses"][key]["loss"], reverse=True)[:10]
        ]

    result = {
        "checkpoint": checkpoint,
        "num_samples": len(rows),
        "indices": indices,
        "summary": summary,
        "samples": rows,
    }
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
