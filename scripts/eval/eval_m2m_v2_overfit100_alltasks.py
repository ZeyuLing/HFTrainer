#!/usr/bin/env python3
"""Evaluate the M2M-v2 all-task overfit-100 run.

This script intentionally reuses the training dataloader pipeline so masks,
editing source motions, text embeddings, padding and task metadata match the
overfit experiment exactly.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import hftrainer  # noqa: F401 - populate registries
from mmengine.config import Config

from hftrainer.datasets.motion.motionhub.flexible_collate import flexible_collate
from hftrainer.evaluation.motion.m2m_eval_metrics import motion135_to_positions_np
from hftrainer.models.motion.hymotion_m2m.network.geometry import rot6d_to_rotation_matrix
from hftrainer.pipelines.motion.hymotion_m2m_pipeline import HyMotionM2MPipeline
from hftrainer.registry import DATASETS, MODEL_BUNDLES
from hftrainer.utils.checkpoint_utils import find_latest_checkpoint, load_checkpoint


DEFAULT_CONFIG = "configs/hymotion_m2m/hymotion_m2m_overfit_100_caption_046b.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default="auto")
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--text-guidance-scale", type=float, default=1.0)
    parser.add_argument(
        "--replacement-guidance",
        default="skip_last",
        choices=["none", "all", "skip_last", "flow_interp"],
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--save-npz", action="store_true")
    parser.add_argument("--save-all-npz", action="store_true")
    parser.add_argument("--visual-per-task", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def resolve_checkpoint(cfg: Config, checkpoint: str, work_dir: Optional[str]) -> str:
    if checkpoint != "auto":
        return checkpoint
    root = work_dir or cfg.work_dir
    ckpt = find_latest_checkpoint(root)
    if ckpt is None:
        raise FileNotFoundError(f"No checkpoint found under {root}")
    return ckpt


def build_bundle(cfg: Config, checkpoint: str, device: str):
    model_cfg = cfg.model.to_dict() if hasattr(cfg.model, "to_dict") else dict(cfg.model)
    bundle_cls = MODEL_BUNDLES.get(model_cfg["type"])
    if bundle_cls is None:
        raise KeyError(f"Unknown model bundle: {model_cfg['type']}")
    bundle = bundle_cls.from_config(model_cfg)
    state = load_checkpoint(checkpoint, map_location="cpu")
    bundle.load_state_dict_selective(state, strict=False)
    del state
    bundle.eval()
    return bundle.to(device)


def build_dataset(cfg: Config, max_samples: int):
    dataset_cfg = cfg.train_dataloader.dataset
    if hasattr(dataset_cfg, "to_dict"):
        dataset_cfg = dataset_cfg.to_dict()
    dataset_cfg = dict(dataset_cfg)
    dataset_cfg["refetch"] = False
    dataset_cfg["verbose"] = False
    for transform in dataset_cfg.get("pipeline", []):
        if isinstance(transform, dict) and transform.get("type") == "PackInputs":
            meta_keys = list(transform.get("meta_keys", []))
            for key in ("subset", "source_motion_path"):
                if key not in meta_keys:
                    meta_keys.append(key)
            transform["meta_keys"] = meta_keys
    dataset = DATASETS.build(dataset_cfg)
    if max_samples > 0 and max_samples < len(dataset):
        dataset = Subset(dataset, list(range(max_samples)))
    return dataset


def as_length_list(value: Any) -> List[int]:
    if isinstance(value, torch.Tensor):
        return [int(x) for x in value.detach().cpu().view(-1).tolist()]
    if isinstance(value, (list, tuple)):
        out = []
        for x in value:
            if isinstance(x, torch.Tensor):
                out.append(int(x.detach().cpu().item()))
            else:
                out.append(int(x))
        return out
    return [int(value)]


def to_device_or_none(value: Any, device: str):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, (list, tuple)) and len(value) == 1 and isinstance(value[0], torch.Tensor):
        return value[0].unsqueeze(0).to(device) if value[0].ndim == 2 else value[0].to(device)
    return value


def prepare_inference_batch(bundle, batch: Dict[str, Any], device: str) -> Tuple[Dict[str, Any], torch.Tensor]:
    src_motion = batch["src_motion"].to(device=device, dtype=torch.float32)
    tgt_motion = batch["tgt_motion"].to(device=device, dtype=torch.float32)
    src_mask = batch["src_mask"].to(device=device, dtype=torch.float32)
    if src_motion.ndim == 2:
        src_motion = src_motion.unsqueeze(0)
        tgt_motion = tgt_motion.unsqueeze(0)
        src_mask = src_mask.unsqueeze(0)

    tgt_lengths = as_length_list(batch["tgt_length"])
    src_lengths = as_length_list(batch.get("src_length", batch["tgt_length"]))

    src_norm = bundle.normalize_motion(src_motion)
    tgt_norm = bundle.normalize_motion(tgt_motion)

    edit_flags = batch.get("edit_mode")
    if edit_flags is not None:
        if isinstance(edit_flags, torch.Tensor):
            keep = edit_flags.to(device=device).view(-1, 1, 1).float()
        elif isinstance(edit_flags, (list, tuple)):
            keep = torch.tensor([float(bool(x)) for x in edit_flags], device=device).view(-1, 1, 1)
        else:
            keep = torch.tensor([float(bool(edit_flags))], device=device).view(-1, 1, 1)
        src_norm = src_norm * (1.0 - src_mask * (1.0 - keep))
    else:
        src_norm = src_norm * (1.0 - src_mask)

    for i, (src_len, tgt_len) in enumerate(zip(src_lengths, tgt_lengths)):
        if src_len < src_norm.shape[1]:
            src_norm[i, src_len:] = 0.0
            src_mask[i, src_len:] = 0.0
        if tgt_len < tgt_norm.shape[1]:
            tgt_norm[i, tgt_len:] = 0.0

    infer_batch: Dict[str, Any] = {
        "src_motion": src_norm,
        "src_mask": src_mask,
        "src_length": src_lengths,
        "tgt_length": tgt_lengths,
        "clean_motion": tgt_norm,
    }

    for key in ("text_vec_raw", "text_ctxt_raw", "text_ctxt_raw_length"):
        if key in batch and batch[key] is not None:
            infer_batch[key] = to_device_or_none(batch[key], device)

    return infer_batch, tgt_motion


def joint_generation_mask(src_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map 198-dim per-channel masks to per-frame joint/rotation masks."""
    T = src_mask.shape[0]
    joint_mask = np.zeros((T, 22), dtype=bool)
    rot_mask = np.zeros((T, 22), dtype=bool)
    trans_mask = src_mask[:, 0:3].max(axis=-1) > 0.5
    joint_mask[trans_mask, :] = True
    for j in range(22):
        r0 = 3 + j * 6
        r1 = r0 + 6
        r = src_mask[:, r0:r1].max(axis=-1) > 0.5
        rot_mask[:, j] = r
        joint_mask[r, j] = True
    for j in range(1, 22):
        p0 = 135 + (j - 1) * 3
        p1 = p0 + 3
        p = src_mask[:, p0:p1].max(axis=-1) > 0.5
        joint_mask[p, j] = True
    return joint_mask, rot_mask, trans_mask


def masked_mean(values: np.ndarray, mask: Optional[np.ndarray]) -> Optional[float]:
    if mask is None:
        return float(values.mean())
    if not mask.any():
        return None
    return float(values[mask].mean())


def compute_rot_errors_deg(pred_rot6d: np.ndarray, gt_rot6d: np.ndarray) -> np.ndarray:
    pred = torch.from_numpy(pred_rot6d).float()
    gt = torch.from_numpy(gt_rot6d).float()
    pred_R = rot6d_to_rotation_matrix(pred)
    gt_R = rot6d_to_rotation_matrix(gt)
    rel = torch.matmul(pred_R, gt_R.transpose(-1, -2))
    trace = rel.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
    cos = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cos)).cpu().numpy()


def summarize(rows: List[Dict[str, Any]], key: str) -> Dict[str, Dict[str, float]]:
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get(key, "unknown"))].append(row)

    numeric = [
        "mpjpe_all_mm",
        "mpjpe_target_mm",
        "mpjpe_preserved_mm",
        "mpjre_all_deg",
        "mpjre_target_deg",
        "mpjre_preserved_deg",
        "trans_error_mm",
    ]
    out: Dict[str, Dict[str, float]] = {}
    for name, items in sorted(buckets.items()):
        rec: Dict[str, float] = {"count": float(len(items))}
        for metric in numeric:
            vals = [x[metric] for x in items if x.get(metric) is not None and math.isfinite(float(x[metric]))]
            if vals:
                arr = np.asarray(vals, dtype=np.float64)
                rec[f"{metric}_mean"] = float(arr.mean())
                rec[f"{metric}_std"] = float(arr.std())
                rec[f"{metric}_median"] = float(np.median(arr))
                rec[f"{metric}_max"] = float(arr.max())
        out[name] = rec
    return out


def write_summary_csv(path: Path, grouped: Dict[str, Dict[str, float]], group_name: str) -> None:
    fields = [group_name]
    for stats in grouped.values():
        for k in stats.keys():
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for name, stats in grouped.items():
            row = {group_name: name}
            row.update(stats)
            writer.writerow(row)


def clean_for_json(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {str(k): clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_for_json(v) for v in obj]
    return obj


def item_meta(batch: Dict[str, Any], b: int) -> Dict[str, Any]:
    def pick(key: str, default=None):
        value = batch.get(key, default)
        if isinstance(value, (list, tuple)):
            return value[b] if b < len(value) else default
        if isinstance(value, torch.Tensor):
            if value.ndim == 0:
                return value.item()
            return value[b].detach().cpu().tolist()
        return value

    return {
        "case_id": pick("overfit_source_key", f"sample_{b}"),
        "task": pick("overfit_task", "unknown"),
        "subset": pick("subset", "unknown"),
        "motion_path": pick("motion_path", ""),
        "fps": pick("fps", None),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = Config.fromfile(args.config)
    checkpoint = resolve_checkpoint(cfg, args.checkpoint, args.work_dir)
    ckpt_name = Path(checkpoint).name
    out_dir = Path(args.output_dir or Path(cfg.work_dir) / "eval_overfit_alltasks" / ckpt_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_dir = out_dir / "npz"
    if args.save_npz or args.save_all_npz:
        npz_dir.mkdir(parents=True, exist_ok=True)

    print(f"config={args.config}", flush=True)
    print(f"checkpoint={checkpoint}", flush=True)
    print(f"output_dir={out_dir}", flush=True)

    dataset = build_dataset(cfg, args.max_samples)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=flexible_collate,
    )

    bundle = build_bundle(cfg, checkpoint, args.device)
    pipeline = HyMotionM2MPipeline(
        bundle=bundle,
        num_steps=args.num_steps,
        text_guidance_scale=args.text_guidance_scale,
        replacement_guidance=args.replacement_guidance,
    )
    bone_offsets = bundle.get_bone_offsets().detach().cpu().float().numpy()

    rows: List[Dict[str, Any]] = []
    visual_candidates: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    started = time.time()

    for batch_idx, batch in enumerate(loader):
        infer_batch, gt_motion = prepare_inference_batch(bundle, batch, args.device)
        with torch.no_grad():
            output = pipeline(infer_batch)
        pred_denorm = bundle.denormalize_motion(output["latent"]).detach().cpu()
        gt_motion_cpu = gt_motion.detach().cpu()
        src_mask_cpu = infer_batch["src_mask"].detach().cpu()
        src_motion_denorm = bundle.denormalize_motion(infer_batch["src_motion"]).detach().cpu()
        lengths = infer_batch["tgt_length"]

        B = pred_denorm.shape[0]
        for b in range(B):
            T = int(lengths[b])
            pred_198 = pred_denorm[b, :T].float().numpy()
            gt_198 = gt_motion_cpu[b, :T].float().numpy()
            src_198 = src_motion_denorm[b, :T].float().numpy()
            mask_198 = src_mask_cpu[b, :T].float().numpy()

            pred_135 = pred_198[:, :135]
            gt_135 = gt_198[:, :135]
            src_135 = src_198[:, :135]

            pred_pos = motion135_to_positions_np(pred_135, bone_offsets)
            gt_pos = motion135_to_positions_np(gt_135, bone_offsets)
            pos_err = np.linalg.norm(pred_pos - gt_pos, axis=-1) * 1000.0

            joint_mask, rot_mask, trans_mask = joint_generation_mask(mask_198)
            mpjpe_all = float(pos_err.mean())
            mpjpe_target = masked_mean(pos_err, joint_mask)
            mpjpe_preserved = masked_mean(pos_err, ~joint_mask)

            pred_rot = pred_135[:, 3:135].reshape(T, 22, 6)
            gt_rot = gt_135[:, 3:135].reshape(T, 22, 6)
            rot_err = compute_rot_errors_deg(pred_rot, gt_rot)
            mpjre_all = float(rot_err.mean())
            mpjre_target = masked_mean(rot_err, rot_mask)
            mpjre_preserved = masked_mean(rot_err, ~rot_mask)
            trans_error = float(np.linalg.norm(pred_135[:, :3] - gt_135[:, :3], axis=-1).mean() * 1000.0)

            meta = item_meta(batch, b)
            sample_idx = len(rows)
            case_id = str(meta["case_id"]).replace("/", "_")
            task = str(meta["task"])
            row = {
                "index": sample_idx,
                **meta,
                "checkpoint": checkpoint,
                "num_frames": T,
                "mask_density": float(mask_198.mean()),
                "mpjpe_all_mm": mpjpe_all,
                "mpjpe_target_mm": mpjpe_target,
                "mpjpe_preserved_mm": mpjpe_preserved,
                "mpjre_all_deg": mpjre_all,
                "mpjre_target_deg": mpjre_target,
                "mpjre_preserved_deg": mpjre_preserved,
                "trans_error_mm": trans_error,
                "npz": None,
            }

            should_save = args.save_all_npz or args.save_npz
            if should_save:
                npz_path = npz_dir / f"{sample_idx:03d}_{task}_{case_id[:120]}.npz"
                np.savez_compressed(
                    npz_path,
                    pred_198=pred_198,
                    gt_198=gt_198,
                    src_198=src_198,
                    pred_135=pred_135,
                    gt_135=gt_135,
                    src_135=src_135,
                    pred_positions=pred_pos,
                    gt_positions=gt_pos,
                    src_mask=mask_198,
                    pos_error_mm=pos_err,
                    rot_error_deg=rot_err,
                )
                row["npz"] = str(npz_path)

            rows.append(row)
            visual_candidates[task].append(row)

        done = len(rows)
        if done == 1 or done % 10 == 0:
            recent = rows[-min(10, len(rows)) :]
            print(
                f"[{done}/{len(dataset)}] "
                f"recent_mpjpe={np.mean([r['mpjpe_all_mm'] for r in recent]):.2f}mm "
                f"recent_mpjre={np.mean([r['mpjre_all_deg'] for r in recent]):.3f}deg",
                flush=True,
            )

    per_sample_path = out_dir / "per_sample.jsonl"
    with per_sample_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(clean_for_json(row), ensure_ascii=False) + "\n")

    by_task = summarize(rows, "task")
    by_subset = summarize(rows, "subset")
    overall = summarize(rows, "__overall__")
    overall = {"all": summarize([{**r, "__overall__": "all"} for r in rows], "__overall__")["all"]}

    write_summary_csv(out_dir / "summary_by_task.csv", by_task, "task")
    write_summary_csv(out_dir / "summary_by_subset.csv", by_subset, "subset")

    visual_manifest = []
    for task, items in sorted(visual_candidates.items()):
        ranked = sorted(items, key=lambda x: float(x["mpjpe_all_mm"]), reverse=True)
        chosen = ranked[: max(0, args.visual_per_task)]
        for row in chosen:
            visual_manifest.append({
                "reason": f"worst_{task}",
                "task": task,
                "index": row["index"],
                "case_id": row["case_id"],
                "subset": row["subset"],
                "mpjpe_all_mm": row["mpjpe_all_mm"],
                "mpjre_all_deg": row["mpjre_all_deg"],
                "npz": row.get("npz"),
            })

    summary = {
        "config": args.config,
        "checkpoint": checkpoint,
        "output_dir": str(out_dir),
        "num_samples": len(rows),
        "num_steps": args.num_steps,
        "replacement_guidance": args.replacement_guidance,
        "text_guidance_scale": args.text_guidance_scale,
        "elapsed_sec": time.time() - started,
        "overall": overall["all"],
        "by_task": by_task,
        "by_subset": by_subset,
        "visual_manifest": visual_manifest,
    }
    with (out_dir / "summary.json").open("w") as f:
        json.dump(clean_for_json(summary), f, ensure_ascii=False, indent=2)
    with (out_dir / "visual_manifest.json").open("w") as f:
        json.dump(clean_for_json(visual_manifest), f, ensure_ascii=False, indent=2)

    print(json.dumps(clean_for_json({
        "checkpoint": checkpoint,
        "num_samples": len(rows),
        "overall": overall["all"],
        "output_dir": str(out_dir),
    }), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
