#!/usr/bin/env python3
"""Comprehensive Evaluation Report for HyMotion M2M Checkpoints.

Evaluates all 8 converged M2M models on 4 completion tasks + replacement guidance comparison.

Models (8 total):
  Baselines (4):
    - uncond_fm:      flow matching, no text, pred_type=velocity
    - uncond_jit:     flow matching, no text, pred_type=x1 (JiT)
    - caption_fm:     flow matching, text-conditioned, pred_type=velocity
    - caption_jit:    flow matching, text-conditioned, pred_type=x1 (JiT)
  Mask-Aware Noise / MAN (4):
    - uncond_fm_man:  + mask_aware_noise (V4 ablation)
    - uncond_jit_man: + mask_aware_noise (V4 ablation)
    - caption_fm_man: + mask_aware_noise (V4 ablation)
    - caption_jit_man:+ mask_aware_noise (V4 ablation)

Tasks (4):
  1. in_between:  preserve first/last 30 frames, generate middle
  2. prediction:  preserve first 90 frames, generate rest
  3. joint_edit:  preserve lower body, regenerate upper body
  4. full_gen:    all mask=1 (pure generation from noise)

Replacement Guidance Comparison (MAN models only):
  - none:        standard ODE (no replacement)
  - skip_last:   replace known regions every step except last
  - flow_interp: replace with flow-matching interpolation (1-t)*z0 + t*x_clean

Metrics:
  - rot_error:         mean |pred - gt| on generated frames (rotation space, unitless)
  - p_rot_error:       mean |pred - gt| on preserved frames (should be ~0 for good models)
  - jitter:            mean |acceleration| (lower = smoother)
  - boundary_jump:     mean |pred[boundary] - gt[boundary]| at mask edges
  - foot_skating:      xz velocity when height < 0.5m (cm/frame)

Usage:
    PYTHON=/data/home/zeyuling/miniconda3/envs/m2m_eval/bin/python
    cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
    CUDA_VISIBLE_DEVICES=0 $PYTHON scripts/eval_m2m_checkpoint_report.py \
        --num-samples 100 --num-steps 50 \
        --output-dir work_dirs/eval_report_20260327
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ──────────────────────────────────────────────────────────────────────
# Model registry
# ──────────────────────────────────────────────────────────────────────

MODELS = {
    # name: (work_dir_name, is_man, has_caption)
    "uncond_fm":       ("hymotion_m2m_completion_uncond_fm_046b",       False, False),
    "uncond_jit":      ("hymotion_m2m_completion_uncond_jit_046b",      False, False),
    "caption_fm":      ("hymotion_m2m_completion_caption_fm_046b",      False, True),
    "caption_jit":     ("hymotion_m2m_completion_caption_jit_046b",     False, True),
    "uncond_fm_man":   ("hymotion_m2m_completion_uncond_fm_man_046b",   True,  False),
    "uncond_jit_man":  ("hymotion_m2m_completion_uncond_jit_man_046b",  True,  False),
    "caption_fm_man":  ("hymotion_m2m_completion_caption_fm_man_046b",  True,  True),
    "caption_jit_man": ("hymotion_m2m_completion_caption_jit_man_046b", True,  True),
}

TASKS = ["in_between", "prediction", "joint_edit", "full_gen"]

# Replacement guidance modes to test on MAN models
REPLACEMENT_MODES = ["none", "skip_last", "flow_interp"]


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def find_latest_checkpoint(work_dir: str) -> str:
    """Find latest checkpoint by mtime."""
    work_dir = Path(work_dir)
    if not work_dir.is_dir():
        return None
    ckpts = sorted(
        [d for d in work_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda d: d.stat().st_mtime,
    )
    return str(ckpts[-1]) if ckpts else None


def find_training_config(work_dir: str) -> str:
    """Find the training config.py from work_dir's run directories."""
    work_dir = Path(work_dir)
    run_dirs = sorted(
        [d for d in work_dir.iterdir() if d.is_dir() and d.name.startswith("2026")],
        key=lambda d: d.name,
    )
    for rd in reversed(run_dirs):
        cfg_path = rd / "config.py"
        if cfg_path.is_file():
            return str(cfg_path)
    return None


def create_task_mask(motion: torch.Tensor, task_type: str) -> torch.Tensor:
    """Create evaluation mask. mask=1 means generate, 0=preserve.

    Args:
        motion: (B, T, D) motion tensor
        task_type: one of TASKS

    Returns:
        (B, T, D) binary mask
    """
    B, T, D = motion.shape
    mask = torch.ones(B, T, D, device=motion.device, dtype=motion.dtype)

    if task_type == "in_between":
        n = min(30, T // 4)
        mask[:, :n] = 0.0
        mask[:, -n:] = 0.0

    elif task_type == "prediction":
        n = min(90, T // 2)
        mask[:, :n] = 0.0

    elif task_type == "joint_edit":
        # Preserve lower body: translation + specific joints
        # Translation: dims 0:3
        # Joints: L_Hip(1), R_Hip(2), L_Knee(4), R_Knee(5),
        #         L_Ankle(7), R_Ankle(8), L_Foot(10), R_Foot(11)
        lower_dims = list(range(0, 3))
        for j in [1, 2, 4, 5, 7, 8, 10, 11]:
            lower_dims.extend(range(3 + j * 6, 3 + (j + 1) * 6))
        mask[:, :, lower_dims] = 0.0

    elif task_type == "full_gen":
        pass  # all 1

    return mask


def compute_boundary_jump(pred: np.ndarray, gt: np.ndarray, mask_temporal: np.ndarray, tgt_length: int) -> float:
    """Compute mean absolute error at mask boundary frames.

    Boundary = the first/last generated frame adjacent to a preserved frame.
    """
    # mask_temporal: (T,) boolean, True = generated
    jumps = []
    for t in range(1, min(tgt_length, len(mask_temporal))):
        # Transition from preserved to generated
        if not mask_temporal[t - 1] and mask_temporal[t]:
            jumps.append(np.abs(pred[t] - gt[t]).mean())
        # Transition from generated to preserved
        if mask_temporal[t - 1] and not mask_temporal[t]:
            jumps.append(np.abs(pred[t - 1] - gt[t - 1]).mean())
    return float(np.mean(jumps)) if jumps else 0.0


def compute_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    mask: np.ndarray,
    tgt_length: int,
) -> dict:
    """Compute all metrics for one sample.

    Args:
        pred: (T, D) predicted motion (denormalized)
        gt:   (T, D) ground truth motion (denormalized)
        mask: (T, D) binary mask (1=generated)
        tgt_length: actual sequence length
    """
    T = min(tgt_length, pred.shape[0])
    pred = pred[:T]
    gt = gt[:T]
    mask = mask[:T]

    # Temporal mask: frame is "generated" if >50% of its dims are masked
    gen_mask = mask.mean(axis=-1) > 0.5  # (T,)

    diff = np.abs(pred - gt)
    frame_err = diff.mean(axis=-1)  # (T,)

    metrics = {}

    # Generated frame error
    if gen_mask.sum() > 0:
        metrics["rot_error"] = float(frame_err[gen_mask].mean())
    else:
        metrics["rot_error"] = 0.0

    # Preserved frame error
    preserve_mask = ~gen_mask
    if preserve_mask.sum() > 0:
        metrics["p_rot_error"] = float(frame_err[preserve_mask].mean())
    else:
        metrics["p_rot_error"] = 0.0

    # Jitter (acceleration magnitude)
    if T >= 3:
        accel = pred[2:] - 2 * pred[1:-1] + pred[:-2]
        metrics["jitter"] = float(np.mean(np.abs(accel)))
    else:
        metrics["jitter"] = 0.0

    # Boundary jump
    metrics["boundary_jump"] = compute_boundary_jump(pred, gt, gen_mask, T)

    # Foot skating (simplified on translation xz)
    if T >= 2:
        transl = pred[:, :3]
        height = transl[:, 1]
        xz_vel = np.sqrt(
            (transl[1:, 0] - transl[:-1, 0]) ** 2
            + (transl[1:, 2] - transl[:-1, 2]) ** 2
        )
        low = height[:-1] < 0.5
        if low.sum() > 0:
            metrics["foot_skating"] = float(xz_vel[low].mean() * 100)  # cm/frame
        else:
            metrics["foot_skating"] = 0.0
    else:
        metrics["foot_skating"] = 0.0

    return metrics


# ──────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────

def load_model(model_name: str, device: str = "cuda"):
    """Load bundle + checkpoint for a model variant."""
    from mmengine.config import Config
    from hftrainer.models.motion.hymotion_m2m.bundle import HyMotionM2MBundle
    from hftrainer.utils.checkpoint_utils import load_checkpoint

    work_dir_name, is_man, has_caption = MODELS[model_name]
    work_dir = str(PROJECT_ROOT / "work_dirs" / work_dir_name)

    ckpt_path = find_latest_checkpoint(work_dir)
    if ckpt_path is None:
        raise FileNotFoundError(f"No checkpoint for {model_name} in {work_dir}")

    # Use training config to match normalization
    training_cfg_path = find_training_config(work_dir)
    if training_cfg_path:
        cfg = Config.fromfile(training_cfg_path)
    else:
        raise FileNotFoundError(f"No training config for {model_name}")

    bundle = HyMotionM2MBundle.from_config(cfg.model)
    bundle = bundle.to(device)
    bundle.eval()

    state_dict = load_checkpoint(ckpt_path, map_location=device)
    bundle.load_state_dict_selective(state_dict)

    return bundle, cfg, ckpt_path


# ──────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────

def build_eval_dataset(cfg, num_samples: int):
    """Build evaluation dataset from config."""
    from hftrainer.registry import DATASETS
    from torch.utils.data import DataLoader

    dataset_cfg = cfg.train_dataloader.dataset.copy()
    dataset = DATASETS.build(dataset_cfg)

    if hasattr(dataset, "__len__") and len(dataset) > num_samples:
        # Use fixed seed for reproducibility
        rng = torch.Generator().manual_seed(42)
        indices = torch.randperm(len(dataset), generator=rng)[:num_samples].tolist()
        dataset = torch.utils.data.Subset(dataset, indices)

    from hftrainer.datasets.collate import flexible_collate
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        collate_fn=flexible_collate,
    )


# ──────────────────────────────────────────────────────────────────────
# Evaluation core
# ──────────────────────────────────────────────────────────────────────

def evaluate_single_model(
    model_name: str,
    bundle,
    cfg,
    dataloader,
    num_samples: int,
    num_steps: int,
    device: str,
    replacement_modes: list = None,
):
    """Evaluate one model on all tasks with optional replacement guidance modes.

    Returns: dict[task][replacement_mode] -> aggregated metrics
    """
    from hftrainer.pipelines.motion.hymotion_m2m_pipeline import HyMotionM2MPipeline

    _, is_man, _ = MODELS[model_name]
    if replacement_modes is None:
        replacement_modes = REPLACEMENT_MODES if is_man else ["none"]

    results = {}

    for task in TASKS:
        results[task] = {}
        for rep_mode in replacement_modes:
            pipeline = HyMotionM2MPipeline(
                bundle=bundle,
                num_steps=num_steps,
                replacement_guidance=rep_mode,
            )

            task_metrics = []
            count = 0
            t0 = time.time()

            for batch_idx, batch in enumerate(dataloader):
                if count >= num_samples:
                    break

                try:
                    src_motion = batch["src_motion"].to(device)
                    tgt_motion = batch["tgt_motion"].to(device)
                    tgt_length = batch.get("tgt_length")
                    if isinstance(tgt_length, torch.Tensor):
                        tgt_length = tgt_length.tolist()

                    B, T, D = src_motion.shape

                    # Create task-specific mask
                    src_mask = create_task_mask(tgt_motion, task_type=task)

                    # Normalize for pipeline
                    src_motion_norm = bundle.normalize_motion(tgt_motion.clone())

                    # For completion: zero mask regions
                    src_motion_norm = src_motion_norm * (1 - src_mask)

                    # Prepare text embeddings if available
                    text_keys = {}
                    if "text_vec_raw" in batch and batch["text_vec_raw"] is not None:
                        text_keys["text_vec_raw"] = batch["text_vec_raw"].to(device)
                        text_keys["text_ctxt_raw"] = batch["text_ctxt_raw"].to(device)
                        text_keys["text_ctxt_raw_length"] = batch["text_ctxt_raw_length"].to(device)

                    eval_batch = {
                        "src_motion": src_motion_norm,
                        "src_mask": src_mask,
                        "tgt_length": tgt_length,
                        "src_length": tgt_length,
                        **text_keys,
                    }

                    with torch.no_grad():
                        output = pipeline(eval_batch)

                    # Get denormalized prediction
                    pred_denorm = output["latent_denorm"][0].cpu().numpy()
                    gt_denorm = tgt_motion[0].cpu().numpy()
                    mask_np = src_mask[0].cpu().numpy()
                    length = int(tgt_length[0]) if tgt_length else T

                    m = compute_metrics(pred_denorm, gt_denorm, mask_np, length)
                    task_metrics.append(m)
                    count += 1

                except Exception as e:
                    print(f"    [WARN] sample {batch_idx} error: {e}")
                    continue

            elapsed = time.time() - t0

            # Aggregate
            if task_metrics:
                agg = {}
                for key in task_metrics[0]:
                    vals = [m[key] for m in task_metrics]
                    agg[key] = {
                        "mean": round(float(np.mean(vals)), 6),
                        "std": round(float(np.std(vals)), 6),
                        "n": len(vals),
                    }
                agg["_elapsed_s"] = round(elapsed, 1)
                agg["_samples"] = count
            else:
                agg = {"_error": "no samples processed", "_elapsed_s": round(elapsed, 1)}

            results[task][rep_mode] = agg

            # Progress
            rot_err = agg.get("rot_error", {}).get("mean", "N/A")
            boundary = agg.get("boundary_jump", {}).get("mean", "N/A")
            print(
                f"  {task:15s} | rep={rep_mode:12s} | "
                f"rot_err={rot_err} boundary={boundary} | "
                f"{count} samples in {elapsed:.0f}s"
            )

    return results


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="HyMotion M2M Checkpoint Evaluation Report")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default="work_dirs/eval_report")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--models", type=str, nargs="+", default=None,
        help="Specific models to evaluate. Default: all available.",
    )
    parser.add_argument(
        "--skip-replacement", action="store_true",
        help="Skip replacement guidance comparison on MAN models (faster).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which models to evaluate
    if args.models:
        model_names = args.models
    else:
        model_names = list(MODELS.keys())

    # Filter to models with checkpoints
    available = []
    for name in model_names:
        work_dir_name = MODELS[name][0]
        ckpt = find_latest_checkpoint(str(PROJECT_ROOT / "work_dirs" / work_dir_name))
        if ckpt:
            available.append(name)
            print(f"[OK] {name:20s} -> {os.path.basename(ckpt)}")
        else:
            print(f"[SKIP] {name:20s} -> no checkpoint found")

    print(f"\nEvaluating {len(available)} models, {args.num_samples} samples, {args.num_steps} steps")
    print(f"Output: {output_dir}")
    print("=" * 80)

    all_results = {}
    dataloader_cache = {}

    for model_name in available:
        print(f"\n{'='*80}")
        print(f"Model: {model_name}")
        print(f"{'='*80}")

        try:
            bundle, cfg, ckpt_path = load_model(model_name, args.device)
            print(f"  Checkpoint: {ckpt_path}")
            print(f"  pred_type: {bundle.pred_type}")
            print(f"  mean: {bundle.mean.shape}, std: {bundle.std.shape}")

            # Cache dataloader per config type
            _, _, has_caption = MODELS[model_name]
            dl_key = "caption" if has_caption else "uncond"
            if dl_key not in dataloader_cache:
                print(f"  Building dataloader ({dl_key})...")
                dataloader_cache[dl_key] = build_eval_dataset(cfg, args.num_samples)
            dataloader = dataloader_cache[dl_key]

            # Determine replacement modes
            _, is_man, _ = MODELS[model_name]
            if is_man and not args.skip_replacement:
                rep_modes = REPLACEMENT_MODES
            else:
                rep_modes = ["none"]

            results = evaluate_single_model(
                model_name, bundle, cfg, dataloader,
                args.num_samples, args.num_steps, args.device,
                replacement_modes=rep_modes,
            )
            results["_meta"] = {
                "checkpoint": ckpt_path,
                "pred_type": bundle.pred_type,
                "is_man": is_man,
                "has_caption": has_caption,
            }
            all_results[model_name] = results

            # Save intermediate results
            with open(output_dir / f"eval_{model_name}.json", "w") as f:
                json.dump(results, f, indent=2)

            # Free GPU memory
            del bundle
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  [ERROR] {model_name}: {e}")
            traceback.print_exc()
            all_results[model_name] = {"_error": str(e)}

    # ──────────────────────────────────────────────────────────────────
    # Generate summary report
    # ──────────────────────────────────────────────────────────────────
    report_path = output_dir / "eval_report.json"
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary table
    print("\n" + "=" * 120)
    print("EVALUATION SUMMARY (replacement_guidance=none)")
    print("=" * 120)
    header = f"{'Model':<22s} {'Task':<15s} {'RotErr':<10s} {'[P]RotErr':<10s} {'Jitter':<10s} {'BndJump':<10s} {'FootSkate':<10s}"
    print(header)
    print("-" * 120)

    for model_name in available:
        if "_error" in all_results.get(model_name, {}):
            print(f"{model_name:<22s} ERROR: {all_results[model_name]['_error']}")
            continue
        for task in TASKS:
            task_data = all_results[model_name].get(task, {}).get("none", {})
            if "_error" in task_data:
                print(f"{model_name:<22s} {task:<15s} ERROR")
                continue
            rot = task_data.get("rot_error", {}).get("mean", "N/A")
            p_rot = task_data.get("p_rot_error", {}).get("mean", "N/A")
            jitter = task_data.get("jitter", {}).get("mean", "N/A")
            bnd = task_data.get("boundary_jump", {}).get("mean", "N/A")
            skate = task_data.get("foot_skating", {}).get("mean", "N/A")

            def fmt(v):
                return f"{v:.6f}" if isinstance(v, float) else str(v)

            print(
                f"{model_name:<22s} {task:<15s} {fmt(rot):<10s} {fmt(p_rot):<10s} "
                f"{fmt(jitter):<10s} {fmt(bnd):<10s} {fmt(skate):<10s}"
            )

    # MAN replacement guidance comparison
    man_models = [m for m in available if MODELS[m][1] and not args.skip_replacement]
    if man_models:
        print("\n" + "=" * 120)
        print("REPLACEMENT GUIDANCE COMPARISON (MAN models)")
        print("=" * 120)
        print(f"{'Model':<22s} {'Task':<15s} {'RepMode':<14s} {'RotErr':<10s} {'[P]RotErr':<10s} {'BndJump':<10s}")
        print("-" * 120)
        for model_name in man_models:
            for task in ["in_between", "prediction"]:  # boundary quality matters most here
                for rep_mode in REPLACEMENT_MODES:
                    task_data = all_results[model_name].get(task, {}).get(rep_mode, {})
                    rot = task_data.get("rot_error", {}).get("mean", "N/A")
                    p_rot = task_data.get("p_rot_error", {}).get("mean", "N/A")
                    bnd = task_data.get("boundary_jump", {}).get("mean", "N/A")

                    def fmt(v):
                        return f"{v:.6f}" if isinstance(v, float) else str(v)

                    print(
                        f"{model_name:<22s} {task:<15s} {rep_mode:<14s} "
                        f"{fmt(rot):<10s} {fmt(p_rot):<10s} {fmt(bnd):<10s}"
                    )

    print(f"\nFull results saved to: {report_path}")
    print(f"Per-model results in: {output_dir}/eval_*.json")


if __name__ == "__main__":
    main()
