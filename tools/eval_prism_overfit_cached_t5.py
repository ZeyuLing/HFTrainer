"""Evaluate PRISM overfit checkpoints with cached T5 features.

This script uses the same dataloader branch as the overfit training config:
caption -> pre-extracted T5 embedding -> transformer. It avoids online T5
encoding so the evaluation condition matches training exactly.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
import hftrainer  # noqa: F401 - populate registries from package imports
from mmengine.config import Config
from torch.utils.data import DataLoader, Subset

from hftrainer.datasets.motion.motionhub.flexible_collate import flexible_collate
from hftrainer.models.motion.components.utils.geometry.rotation_convert import (
    rotation_6d_to_matrix,
)
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
            "modules (vae, smpl_pose_processor, and bundle latent stats). "
            "The transformer still comes from --checkpoint."
        ),
    )
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="First dataset index to evaluate (for multi-GPU sharding).",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument(
        "--decode-frames",
        type=int,
        default=0,
        help=(
            "If >0, generate this many motion frames before cropping metrics to "
            "num_frames. Use 360 to match the overfit training clip length."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default=None)
    parser.add_argument("--positions-dir", default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def resolve_checkpoint(cfg, checkpoint: str, work_dir: Optional[str]) -> str:
    if checkpoint != "auto":
        return checkpoint
    root = work_dir or cfg.work_dir
    latest = find_latest_checkpoint(root)
    if latest is None:
        raise FileNotFoundError(f"No checkpoint-* found under {root}")
    return latest


def _filter_frozen_latent_state(state_dict: Dict[str, object]) -> Dict[str, object]:
    filtered: Dict[str, object] = {}
    for key in ("vae", "smpl_pose_processor", "__bundle_params__"):
        value = state_dict.get(key)
        if value is not None:
            filtered[key] = value
    return filtered


def build_bundle(
    cfg,
    checkpoint: str,
    device: str,
    frozen_module_checkpoint: Optional[str] = None,
):
    model_cfg = cfg.model.to_dict() if hasattr(cfg.model, "to_dict") else dict(cfg.model)
    bundle_type = model_cfg.get("type")
    bundle_cls = MODEL_BUNDLES.get(bundle_type)
    if bundle_cls is None:
        raise KeyError(f"Unknown bundle type: {bundle_type}")

    bundle = bundle_cls.from_config(model_cfg)
    state_dict = load_checkpoint(checkpoint, map_location="cpu")
    bundle.load_state_dict_selective(state_dict, strict=False)
    if frozen_module_checkpoint:
        frozen_state = load_checkpoint(frozen_module_checkpoint, map_location="cpu")
        bundle.load_state_dict_selective(
            _filter_frozen_latent_state(frozen_state),
            strict=False,
        )
    bundle.eval()
    return bundle.to(device)


def build_dataset(cfg, num_samples: int, start_index: int = 0):
    dataset_cfg = cfg.train_dataloader.dataset
    if hasattr(dataset_cfg, "to_dict"):
        dataset_cfg = dataset_cfg.to_dict()
    dataset = DATASETS.build(dataset_cfg)
    total = len(dataset)
    start = max(0, start_index)
    if num_samples > 0:
        end = min(total, start + num_samples)
    else:
        end = total
    if start > 0 or end < total:
        dataset = Subset(dataset, list(range(start, end)))
    return dataset


def finite_mean(values: Iterable[float]) -> float:
    vals = list(values)
    return float(sum(vals) / max(len(vals), 1))


def rot_geodesic_rad(pred_6d: torch.Tensor, gt_6d: torch.Tensor) -> torch.Tensor:
    pred_R = rotation_6d_to_matrix(pred_6d, convention="column")
    gt_R = rotation_6d_to_matrix(gt_6d, convention="column")
    rel = pred_R.transpose(-1, -2) @ gt_R
    trace = rel.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
    cos = ((trace - 1.0) * 0.5).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    return torch.acos(cos)


def canonicalize_pair_positions(
    pred_positions: torch.Tensor, gt_positions: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one viewer transform to Pred/GT so their spatial gap stays visible."""
    pred = pred_positions.clone()
    gt = gt_positions.clone()
    floor_y = torch.minimum(pred[..., 1].amin(), gt[..., 1].amin())
    origin_xz = gt[0, 0, [0, 2]].clone()
    for pos in (pred, gt):
        pos[..., 1] = pos[..., 1] - floor_y
        pos[..., 0] = pos[..., 0] - origin_xz[0]
        pos[..., 2] = pos[..., 2] - origin_xz[1]
    return pred, gt


def safe_sample_key(batch: Dict[str, object], global_idx: int, sample_idx: int) -> str:
    motion_path = batch.get("motion_path")
    if isinstance(motion_path, (list, tuple)):
        motion_path = motion_path[sample_idx]
    if motion_path:
        stem = Path(str(motion_path)).stem
    else:
        stem = f"sample_{global_idx:04d}"
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)
    return f"{global_idx:04d}_{stem}"


def motion_to_positions(bundle, motion_vec: torch.Tensor) -> torch.Tensor:
    transl = bundle.smpl_pose_processor.inv_convert_transl(
        motion_vec[:, :6], use_rollout=True
    )
    joints = bundle.smpl_pose_processor.fk(
        transl.unsqueeze(0),
        motion_vec[:, 6:].unsqueeze(0),
        rot_type="rotation_6d",
    )
    return joints.squeeze(0)


@torch.no_grad()
def generate_with_cached_t5(
    bundle,
    text_states: torch.Tensor,
    text_mask: torch.Tensor,
    num_frames: torch.Tensor,
    num_steps: int,
    guidance_scale: float,
    decode_frames_override: int = 0,
) -> torch.Tensor:
    device = next(bundle.transformer.parameters()).device
    transformer_dtype = next(bundle.transformer.parameters()).dtype
    batch_size = text_states.shape[0]
    num_frames_max = int(num_frames.max().item())
    scale = bundle.vae.config.scale_factor_temporal
    requested_frames = max(num_frames_max, int(decode_frames_override or 0))
    if (requested_frames - 1) % scale != 0:
        decode_frames = (requested_frames // scale) * scale + 1
    else:
        decode_frames = max(1, requested_frames)
    latent_frames = (decode_frames - 1) // scale + 1
    latent_joints = 23

    text_states = text_states.to(device=device, dtype=transformer_dtype)
    text_mask = text_mask.to(device=device)

    bundle.scheduler.set_timesteps(num_steps, device=device)
    timesteps = bundle.scheduler.timesteps
    latents = torch.randn(
        batch_size,
        bundle.transformer.config.in_channels,
        latent_frames,
        latent_joints,
        device=device,
        dtype=transformer_dtype,
    )
    padding_mask = bundle.create_padding_mask(
        num_frames=num_frames.to(device),
        batch_size=batch_size,
        latent_frames=latent_frames,
        latent_joints=latent_joints,
        device=device,
    )
    condition_mask = torch.ones(
        batch_size, 1, latent_frames, latent_joints, device=device, dtype=torch.bool
    )
    transformer_module = getattr(bundle.transformer, "module", bundle.transformer)

    for t in timesteps:
        step_ts = torch.full((batch_size,), t, device=device, dtype=t.dtype)
        seq_ts = bundle.create_sequence_ts(
            step_ts,
            condition_mask,
            transformer_module.config.patch_size,
        )
        pred = bundle.transformer(
            hidden_states=latents.to(transformer_dtype),
            encoder_hidden_states=text_states,
            timestep=seq_ts,
            hidden_states_mask=padding_mask,
            encoder_hidden_states_mask=text_mask,
        )

        if guidance_scale != 1.0:
            raise NotImplementedError("Cached-T5 CFG is not implemented for this debug script.")

        latents = bundle.scheduler.step(pred, t, latents, return_dict=False)[0]

    latents = latents * bundle.latents_std.to(latents) + bundle.latents_mean.to(latents)
    device_type = latents.device.type
    with torch.autocast(device_type, enabled=False):
        motion = bundle.vae.decode(latents.float())
    motion = motion[:, :num_frames_max]
    motion = motion.reshape(batch_size, motion.shape[1], -1)
    return bundle.smpl_pose_processor.denormalize(motion.float())


def evaluate(args) -> Dict[str, object]:
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
    dataset = build_dataset(cfg, args.num_samples, start_index=args.start_index)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=flexible_collate,
    )

    transl_l2 = []
    rot6d_l2 = []
    mpjre_rad = []
    mpjpe = []
    samples = []
    positions_dir = Path(args.positions_dir) if args.positions_dir else None
    if positions_dir is not None:
        positions_dir.mkdir(parents=True, exist_ok=True)

    can_fk = getattr(bundle.smpl_pose_processor, "smpl_model", None) is not None

    for batch_idx, batch in enumerate(loader):
        gt = batch["motion"].to(device=device, dtype=torch.float32)
        if gt.ndim == 2:
            gt = gt.unsqueeze(0)
        num_frames = batch["num_frames"]
        if not isinstance(num_frames, torch.Tensor):
            num_frames = torch.as_tensor(num_frames)
        pred = generate_with_cached_t5(
            bundle,
            batch["t5_text_embeds"],
            batch["t5_text_mask"],
            num_frames,
            num_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            decode_frames_override=args.decode_frames,
        )
        gt_denorm = gt[:, : pred.shape[1]]

        for i in range(pred.shape[0]):
            valid = min(int(num_frames[i].item()), int(pred.shape[1]), int(gt_denorm.shape[1]))
            pred_i = pred[i, :valid]
            gt_i = gt_denorm[i, :valid]

            transl_pred = bundle.smpl_pose_processor.inv_convert_transl(
                pred_i[:, :6], use_rollout=False
            )
            transl_gt = bundle.smpl_pose_processor.inv_convert_transl(
                gt_i[:, :6], use_rollout=False
            )
            transl_err = (transl_pred - transl_gt).norm(dim=-1).mean()

            pred_rot = pred_i[:, 6:].reshape(valid, 22, 6)
            gt_rot = gt_i[:, 6:].reshape(valid, 22, 6)
            rot_err = (pred_rot - gt_rot).norm(dim=-1).mean()
            geo = rot_geodesic_rad(pred_rot, gt_rot).mean()

            transl_l2.append(float(transl_err.cpu()))
            rot6d_l2.append(float(rot_err.cpu()))
            mpjre_rad.append(float(geo.cpu()))

            mpjpe_val = None
            if can_fk:
                pred_joints = motion_to_positions(bundle, pred_i)
                gt_joints = motion_to_positions(bundle, gt_i)
                mpjpe_val = float((pred_joints - gt_joints).norm(dim=-1).mean().cpu())
                mpjpe.append(mpjpe_val)

                if positions_dir is not None:
                    caption = batch.get("caption", [""] * pred.shape[0])[i]
                    global_idx = args.start_index + batch_idx * args.batch_size + i
                    key = safe_sample_key(batch, global_idx, i)
                    metrics = {
                        "transl_l2": transl_l2[-1],
                        "rot6d_l2": rot6d_l2[-1],
                        "mpjre_rad": mpjre_rad[-1],
                        "mpjre_deg": mpjre_rad[-1] * 180.0 / 3.141592653589793,
                        "mpjpe": mpjpe_val,
                        "mpjpe_m": mpjpe_val,
                        "mpjpe_mm": mpjpe_val * 1000.0,
                    }
                    view_pred, view_gt = canonicalize_pair_positions(
                        pred_joints, gt_joints
                    )
                    np_pred = view_pred.detach().cpu().float().numpy()
                    np_gt = view_gt.detach().cpu().float().numpy()

                    np.savez_compressed(
                        positions_dir / f"{key}.npz",
                        pred_positions=np_pred,
                        gt_positions=np_gt,
                        caption=str(caption),
                        num_frames=int(valid),
                        metrics=metrics,
                    )

            if len(samples) < 8:
                caption = batch.get("caption", [""] * pred.shape[0])[i]
                samples.append(
                    {
                        "batch": batch_idx,
                        "caption": caption,
                        "num_frames": valid,
                        "transl_l2": transl_l2[-1],
                        "rot6d_l2": rot6d_l2[-1],
                        "mpjre_rad": mpjre_rad[-1],
                        "mpjpe": mpjpe_val,
                        "mpjpe_mm": mpjpe_val * 1000.0 if mpjpe_val is not None else None,
                    }
                )

            if args.progress:
                print(
                    f"[eval] sample={len(transl_l2)}/{len(dataset)} "
                    f"mpjpe_mm={mpjpe_val * 1000.0 if mpjpe_val is not None else 'NA'} "
                    f"mpjre_deg={mpjre_rad[-1] * 180.0 / 3.141592653589793:.4f}",
                    flush=True,
                )

    result = {
        "checkpoint": checkpoint,
        "frozen_module_checkpoint": args.frozen_module_checkpoint,
        "num_samples": len(transl_l2),
        "num_steps": args.num_steps,
        "guidance_scale": args.guidance_scale,
        "transl_l2": finite_mean(transl_l2),
        "rot6d_l2": finite_mean(rot6d_l2),
        "mpjre_rad": finite_mean(mpjre_rad),
        "mpjre_deg": finite_mean(mpjre_rad) * 180.0 / 3.141592653589793,
        "mpjpe": finite_mean(mpjpe) if mpjpe else None,
        "mpjpe_mm": finite_mean(mpjpe) * 1000.0 if mpjpe else None,
        "samples": samples,
    }
    return result


def main():
    args = parse_args()
    result = evaluate(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
