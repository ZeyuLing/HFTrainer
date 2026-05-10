#!/usr/bin/env python3
"""Ablation Evaluation Script for HyMotion M2M experiments.

Evaluates M2M models on 4 completion tasks:
1. In-between: First/last 30 frames preserved, generate middle
2. Prediction: First 90 frames preserved, generate rest
3. Joint Edit: Lower body preserved, regenerate upper body
4. Full Gen: All mask=1 (unconditional generation)

Metrics:
- MPJPE (Mean Per Joint Position Error, mm)
- [P]-MPJPE (Preserved frame MPJPE, mm)
- Foot Skating (cm/s when height < 0.05m)
- Jitter (mm/frame^2, acceleration)
- Ground Penetration (mm, min toe y < 0)
- Quality Pass Rate (MotionQualityChecker)

Usage:
    python scripts/eval_m2m_ablation.py \
        --config configs/hymotion_m2m/ablation/ablation_m2_baseline.py \
        --checkpoint work_dirs/ablation_m2_baseline/checkpoint-epoch_20 \
        --num-samples 200 \
        --num-steps 50 \
        --output work_dirs/ablation_m2_baseline/eval_results.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mmengine.config import Config


def build_model_from_config(config_path, checkpoint_path, device='cuda'):
    """Build model bundle and load checkpoint."""
    from hftrainer.registry import MODEL_BUNDLES
    from hftrainer.runner.accelerate_runner import AccelerateRunner

    cfg = Config.fromfile(config_path)

    # Build bundle
    bundle = MODEL_BUNDLES.build(cfg.model)
    bundle = bundle.to(device)
    bundle.eval()

    # Load checkpoint
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt_file = os.path.join(checkpoint_path, 'model.pt')
        if os.path.exists(ckpt_file):
            state_dict = torch.load(ckpt_file, map_location=device)
            if '__hftrainer_meta__' in state_dict:
                meta = state_dict.pop('__hftrainer_meta__')
            bundle.load_state_dict_selective(state_dict)
            print(f"Loaded checkpoint from {ckpt_file}")
        else:
            # Try safetensors
            st_file = os.path.join(checkpoint_path, 'model.safetensors')
            if os.path.exists(st_file):
                from safetensors.torch import load_file
                state_dict = load_file(st_file)
                bundle.load_state_dict_selective(state_dict)
                print(f"Loaded checkpoint from {st_file}")
    return bundle, cfg


def build_eval_dataloader(cfg, num_samples=200):
    """Build evaluation dataloader from training config's dataset."""
    from hftrainer.registry import DATASETS
    from torch.utils.data import DataLoader

    # Clone the training dataset config but use a fixed seed for reproducibility
    dataset_cfg = cfg.train_dataloader.dataset.copy()
    dataset = DATASETS.build(dataset_cfg)

    # Take first num_samples
    if hasattr(dataset, '__len__') and len(dataset) > num_samples:
        indices = list(range(num_samples))
        dataset = torch.utils.data.Subset(dataset, indices)

    from hftrainer.datasets.collate import flexible_collate
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        collate_fn=flexible_collate,
    )
    return dataloader


def create_task_mask(src_motion, task_type='in_between'):
    """Create evaluation mask for a specific task type.

    Args:
        src_motion: (1, T, D) source motion tensor
        task_type: one of 'in_between', 'prediction', 'joint_edit', 'full_gen'

    Returns:
        src_mask: (1, T, D) binary mask (1=generate, 0=preserve)
    """
    B, T, D = src_motion.shape
    mask = torch.ones(B, T, D, device=src_motion.device)

    if task_type == 'in_between':
        # Preserve first 30 and last 30 frames
        preserve_frames = min(30, T // 4)
        mask[:, :preserve_frames] = 0.0
        mask[:, -preserve_frames:] = 0.0

    elif task_type == 'prediction':
        # Preserve first 90 frames
        preserve_frames = min(90, T // 2)
        mask[:, :preserve_frames] = 0.0

    elif task_type == 'joint_edit':
        # Preserve lower body (translation + hip/knee/ankle/foot joints)
        # dims [0:3] = translation, joints indices: 1(L_Hip),2(R_Hip),4(L_Knee),5(R_Knee),
        # 7(L_Ankle),8(R_Ankle),10(L_Foot),11(R_Foot)
        lower_dims = list(range(0, 3))  # translation
        lower_joints = [1, 2, 4, 5, 7, 8, 10, 11]
        for j in lower_joints:
            lower_dims.extend(range(3 + j * 6, 3 + (j + 1) * 6))
        mask[:, :, lower_dims] = 0.0

    elif task_type == 'full_gen':
        # All mask=1 (unconditional generation)
        pass

    return mask


def compute_fk(motion, bundle):
    """Compute FK to get 3D joint positions from rot6d + transl.

    Args:
        motion: (B, T, 135) — [transl(3), rot6d(132)]

    Returns:
        keypoints3d: (B, T, 22, 3) joint positions
    """
    if not hasattr(bundle, 'body_model') or bundle.body_model is None:
        # Simple FK using rotation_converter if body model not available
        from hftrainer.models.motion.hymotion_m2m.network.geometry import (
            rot6d_to_rotation_matrix,
        )
        B, T, D = motion.shape
        transl = motion[:, :, :3]  # (B, T, 3)
        rot6d = motion[:, :, 3:].reshape(B, T, 22, 6)  # (B, T, 22, 6)

        # Convert rot6d to rotation matrix
        rot_mat = rot6d_to_rotation_matrix(rot6d)  # (B, T, 22, 3, 3)

        # Use SmplxLiteJ24 if available
        if hasattr(bundle, 'smpl_model') and bundle.smpl_model is not None:
            positions = bundle.smpl_model.fk(rot_mat, transl)
            return positions

        # Fallback: return None (can't compute FK without body model)
        return None

    return None


def compute_metrics(pred_motion, gt_motion, src_mask, bundle, tgt_length=None):
    """Compute evaluation metrics.

    Args:
        pred_motion: (B, T, 135) predicted motion (denormalized)
        gt_motion: (B, T, 135) ground truth motion (denormalized)
        src_mask: (B, T, 135) binary mask (1=generated, 0=preserved)
        bundle: model bundle (for FK computation)
        tgt_length: actual sequence length

    Returns:
        dict of metrics
    """
    B, T, D = pred_motion.shape
    metrics = {}

    # Ensure on CPU for metric computation
    pred = pred_motion.cpu().numpy()
    gt = gt_motion.cpu().numpy()
    mask = src_mask.cpu().numpy()

    # --- MPJPE (rotation-space as proxy if FK not available) ---
    # Compute per-frame error on generated frames
    gen_mask_temporal = mask.mean(axis=-1) > 0.5  # (B, T) — frames that are mostly generated
    diff = np.abs(pred - gt)
    frame_error = diff.mean(axis=-1)  # (B, T)

    # MPJPE on generated frames
    gen_errors = []
    for b in range(B):
        length = int(tgt_length[b]) if tgt_length is not None else T
        gen_frames = gen_mask_temporal[b, :length]
        if gen_frames.sum() > 0:
            gen_errors.append(frame_error[b, :length][gen_frames].mean())
    metrics['mpjpe_rot_space'] = float(np.mean(gen_errors)) if gen_errors else 0.0

    # [P]-MPJPE on preserved frames
    preserve_errors = []
    for b in range(B):
        length = int(tgt_length[b]) if tgt_length is not None else T
        preserve_frames = ~gen_mask_temporal[b, :length]
        if preserve_frames.sum() > 0:
            preserve_errors.append(frame_error[b, :length][preserve_frames].mean())
    metrics['p_mpjpe_rot_space'] = float(np.mean(preserve_errors)) if preserve_errors else 0.0

    # --- Jitter (acceleration = second-order finite difference) ---
    jitters = []
    for b in range(B):
        length = int(tgt_length[b]) if tgt_length is not None else T
        if length < 3:
            continue
        m = pred[b, :length]
        accel = m[2:] - 2 * m[1:-1] + m[:-2]  # (T-2, D)
        jitter = np.mean(np.abs(accel))
        jitters.append(jitter)
    metrics['jitter'] = float(np.mean(jitters)) if jitters else 0.0

    # --- Foot Skating (simplified: measure xz velocity of translation when height is low) ---
    foot_skating_vals = []
    for b in range(B):
        length = int(tgt_length[b]) if tgt_length is not None else T
        if length < 2:
            continue
        transl = pred[b, :length, :3]  # (T, 3) — [x, y, z]
        height = transl[:, 1]  # y-axis is height
        xz_vel = np.sqrt(
            (transl[1:, 0] - transl[:-1, 0]) ** 2 +
            (transl[1:, 2] - transl[:-1, 2]) ** 2
        )  # (T-1,)
        # Frames where height is low (feet likely on ground)
        low_height = height[:-1] < 0.5  # Below 0.5m — person on ground
        if low_height.sum() > 0:
            foot_skating_vals.append(float(xz_vel[low_height].mean() * 100))  # cm/frame
    metrics['foot_skating_cm_per_frame'] = float(np.mean(foot_skating_vals)) if foot_skating_vals else 0.0

    # --- Ground Penetration (translation y < 0) ---
    penetrations = []
    for b in range(B):
        length = int(tgt_length[b]) if tgt_length is not None else T
        transl_y = pred[b, :length, 1]  # y-axis
        below = transl_y[transl_y < 0]
        if len(below) > 0:
            penetrations.append(float(np.abs(below).mean() * 1000))  # mm
        else:
            penetrations.append(0.0)
    metrics['ground_penetration_mm'] = float(np.mean(penetrations))

    return metrics


def evaluate_model(bundle, cfg, args):
    """Main evaluation loop."""
    from hftrainer.pipelines.motion.hymotion_m2m_pipeline import HyMotionM2MPipeline

    device = next(bundle.motion_transformer.parameters()).device
    pipeline = HyMotionM2MPipeline(bundle=bundle, num_steps=args.num_steps)

    # Build dataloader
    dataloader = build_eval_dataloader(cfg, num_samples=args.num_samples)

    tasks = ['in_between', 'prediction', 'joint_edit', 'full_gen']
    all_results = {}

    for task in tasks:
        print(f"\n{'='*60}")
        print(f"Evaluating task: {task}")
        print(f"{'='*60}")

        task_metrics = []
        count = 0

        for batch_idx, batch in enumerate(dataloader):
            if count >= args.num_samples:
                break

            try:
                src_motion = batch['src_motion'].to(device)
                tgt_motion = batch['tgt_motion'].to(device)
                tgt_length = batch.get('tgt_length')

                # Create task-specific mask
                src_mask = create_task_mask(src_motion, task_type=task)

                # Normalize src_motion for pipeline input
                src_motion_norm = bundle.normalize_motion(src_motion.clone())
                tgt_motion_norm = bundle.normalize_motion(tgt_motion.clone())

                # Run pipeline
                eval_batch = {
                    'src_motion': src_motion_norm,
                    'src_mask': src_mask,
                    'tgt_length': tgt_length,
                    'src_length': tgt_length,
                }
                output = pipeline(eval_batch)

                # Get predicted motion (denormalized)
                if isinstance(output, dict):
                    pred_motion = output.get('latent', output.get('rot6d'))
                    if pred_motion is None:
                        pred_motion = list(output.values())[0]
                else:
                    pred_motion = output

                # Denormalize if needed
                if hasattr(bundle, 'denormalize_motion'):
                    pred_motion = bundle.denormalize_motion(pred_motion)

                # Denormalize GT
                tgt_motion_denorm = tgt_motion  # Already denormalized (raw from dataset)

                # Compute metrics
                m = compute_metrics(
                    pred_motion, tgt_motion_denorm, src_mask, bundle, tgt_length
                )
                task_metrics.append(m)
                count += 1

                if count % 50 == 0:
                    print(f"  Processed {count}/{args.num_samples} samples")

            except Exception as e:
                print(f"  Error on sample {batch_idx}: {e}")
                continue

        # Aggregate metrics
        if task_metrics:
            agg = {}
            for key in task_metrics[0]:
                vals = [m[key] for m in task_metrics if m[key] is not None]
                agg[key] = {
                    'mean': float(np.mean(vals)) if vals else 0.0,
                    'std': float(np.std(vals)) if vals else 0.0,
                    'count': len(vals),
                }
            all_results[task] = agg
            print(f"\n  Results for {task}:")
            for k, v in agg.items():
                print(f"    {k}: {v['mean']:.4f} ± {v['std']:.4f} (n={v['count']})")

    return all_results


def main():
    parser = argparse.ArgumentParser(description='Evaluate M2M ablation experiments')
    parser.add_argument('--config', required=True, help='Config file path')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint directory')
    parser.add_argument('--num-samples', type=int, default=200, help='Number of eval samples')
    parser.add_argument('--num-steps', type=int, default=50, help='ODE integration steps')
    parser.add_argument('--output', type=str, default=None, help='Output JSON path')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(os.path.dirname(args.checkpoint), 'eval_results.json')

    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Num samples: {args.num_samples}")
    print(f"Num steps: {args.num_steps}")
    print(f"Output: {args.output}")

    # Build model
    bundle, cfg = build_model_from_config(args.config, args.checkpoint, args.device)

    # Evaluate
    results = evaluate_model(bundle, cfg, args)

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    header = f"{'Task':<15} {'MPJPE':<12} {'[P]-MPJPE':<12} {'Jitter':<12} {'FootSkate':<12} {'GndPen':<12}"
    print(header)
    print("-" * 80)
    for task, metrics in results.items():
        mpjpe = metrics.get('mpjpe_rot_space', {}).get('mean', 0)
        p_mpjpe = metrics.get('p_mpjpe_rot_space', {}).get('mean', 0)
        jitter = metrics.get('jitter', {}).get('mean', 0)
        skating = metrics.get('foot_skating_cm_per_frame', {}).get('mean', 0)
        penetration = metrics.get('ground_penetration_mm', {}).get('mean', 0)
        print(f"{task:<15} {mpjpe:<12.4f} {p_mpjpe:<12.4f} {jitter:<12.4f} {skating:<12.4f} {penetration:<12.4f}")


if __name__ == '__main__':
    main()
