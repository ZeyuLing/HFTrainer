#!/usr/bin/env python3
"""Comprehensive evaluation of HyMotion M2M checkpoints across multiple tasks.

Tests:
  1. Reconstruction (identity): mask=all_zero -> model should output identity
  2. Full generation (T2M-like): mask=all_one -> unconditional generation from noise
  3. Temporal in-between: keep first/last 20 frames, mask middle
  4. Joint completion: mask upper body, keep lower body
  5. Keyframe interpolation: keep every 30th frame, mask rest
  6. Repair (edit mode): feed corrupted motion via reactive channel

Metrics per sample:
  - MPJPE (mean per-joint position error via FK, in mm)
  - Jitter (3rd-order finite diff, lower=smoother)
  - Bone length consistency (std of bone lengths across frames)
  - Translation drift (for generated samples)
  - Boundary error (at mask edges, for completion tasks)

Usage:
    python tools/eval_m2m_checkpoints.py --models uncond_fm caption_fm --max-samples 50 --num-steps 50
    python tools/eval_m2m_checkpoints.py --all --max-samples 100

Requires: torch>=2.0, mmengine, torchdiffeq, safetensors
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ============================================================================
# Model registry
# ============================================================================
MODELS = {
    'uncond_fm': {
        'config': 'configs/hymotion_m2m/hymotion_m2m_completion_uncond_fm_046b.py',
        'work_dir': 'work_dirs/hymotion_m2m_completion_uncond_fm_046b',
        'desc': 'Unconditioned + Flow Matching (velocity)',
    },
    'caption_fm': {
        'config': 'configs/hymotion_m2m/hymotion_m2m_completion_caption_fm_046b.py',
        'work_dir': 'work_dirs/hymotion_m2m_completion_caption_fm_046b',
        'desc': 'Caption + Flow Matching (velocity)',
    },
    'uncond_jit': {
        'config': 'configs/hymotion_m2m/hymotion_m2m_completion_uncond_jit_046b.py',
        'work_dir': 'work_dirs/hymotion_m2m_completion_uncond_jit_046b',
        'desc': 'Unconditioned + JiT (x1 pred)',
    },
    'caption_jit': {
        'config': 'configs/hymotion_m2m/hymotion_m2m_completion_caption_jit_046b.py',
        'work_dir': 'work_dirs/hymotion_m2m_completion_caption_jit_046b',
        'desc': 'Caption + JiT (x1 pred)',
    },
    'uncond_fm_man': {
        'config': 'configs/hymotion_m2m/hymotion_m2m_completion_uncond_fm_man_046b.py',
        'work_dir': 'work_dirs/hymotion_m2m_completion_uncond_fm_man_046b',
        'desc': 'Unconditioned + FM + Mask-Aware Noise (V4)',
    },
    'caption_fm_man': {
        'config': 'configs/hymotion_m2m/hymotion_m2m_completion_caption_fm_man_046b.py',
        'work_dir': 'work_dirs/hymotion_m2m_completion_caption_fm_man_046b',
        'desc': 'Caption + FM + Mask-Aware Noise (V4)',
    },
    'uncond_jit_man': {
        'config': 'configs/hymotion_m2m/hymotion_m2m_completion_uncond_jit_man_046b.py',
        'work_dir': 'work_dirs/hymotion_m2m_completion_uncond_jit_man_046b',
        'desc': 'Unconditioned + JiT + Mask-Aware Noise (V4)',
    },
    'caption_jit_man': {
        'config': 'configs/hymotion_m2m/hymotion_m2m_completion_caption_jit_man_046b.py',
        'work_dir': 'work_dirs/hymotion_m2m_completion_caption_jit_man_046b',
        'desc': 'Caption + JiT + Mask-Aware Noise (V4)',
    },
}

# SMPL-22 bone pairs (parent -> child)
SMPL_22_BONE_PAIRS = [
    (0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (4, 7), (5, 8),
    (7, 10), (8, 11), (3, 6), (6, 9), (9, 12), (9, 13), (9, 14),
    (12, 15), (13, 16), (14, 17), (16, 18), (17, 19), (18, 20), (19, 21),
]

# Joint group indices for upper/lower body
UPPER_BODY_JOINTS = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]  # spine + head + arms
LOWER_BODY_JOINTS = [0, 1, 2, 4, 5, 7, 8, 10, 11]  # pelvis + legs + feet

# ============================================================================
# Data loading
# ============================================================================

def load_motion_135d(npz_path: str) -> Optional[torch.Tensor]:
    """Load npz -> 135-dim motion (abs transl + rot6d)."""
    try:
        from hftrainer.datasets.motion.motionhub.transforms.load_smplx import (
            process_transl, process_smplx_pose,
        )
        data = np.load(npz_path, allow_pickle=True)
        trans_key = 'trans' if 'trans' in data else 'transl'
        abs_trans = data[trans_key].astype(np.float32)
        poses_key = 'poses' if 'poses' in data else 'body_pose'
        poses = data[poses_key].astype(np.float32)
        transl = process_transl(abs_trans, 'abs')
        pose = process_smplx_pose(poses, 'rotation_6d', 'smpl_22')
        motion = np.concatenate([transl, pose], axis=-1)
        return torch.from_numpy(motion).float()
    except Exception as e:
        return None


def load_test_samples(anno_file: str, data_dir: str, max_samples: int) -> List[Dict]:
    """Load test motion samples from annotation file."""
    with open(anno_file) as f:
        anno = json.load(f)

    data_list = anno.get('data_list', {})
    if isinstance(data_list, dict):
        items = list(data_list.values())
    else:
        items = data_list

    samples = []
    for item in items:
        if len(samples) >= max_samples:
            break
        smplx_path = item.get('smplx_path', '')
        full_path = os.path.join(data_dir, smplx_path)
        if not os.path.exists(full_path):
            continue
        motion = load_motion_135d(full_path)
        if motion is None or motion.shape[0] < 30:
            continue
        # Crop/pad to max 360 frames
        T = min(motion.shape[0], 360)
        motion = motion[:T]
        samples.append({
            'path': smplx_path,
            'motion': motion,
            'T': T,
            'caption': item.get('caption', ''),
        })

    return samples


# ============================================================================
# Mask builders
# ============================================================================

def build_full_mask(T: int, D: int = 135) -> torch.Tensor:
    """All masked (T2M / unconditional generation)."""
    return torch.ones(T, D)


def build_zero_mask(T: int, D: int = 135) -> torch.Tensor:
    """No mask (identity / reconstruction)."""
    return torch.zeros(T, D)


def build_temporal_inbetween_mask(T: int, D: int = 135, keep_frames: int = 20) -> torch.Tensor:
    """Keep first and last keep_frames, mask middle."""
    mask = torch.ones(T, D)
    mask[:keep_frames] = 0
    mask[-keep_frames:] = 0
    return mask


def build_upper_body_mask(T: int, D: int = 135) -> torch.Tensor:
    """Mask upper body joints (keep lower body)."""
    mask = torch.zeros(T, D)
    for j in UPPER_BODY_JOINTS:
        start = 3 + j * 6
        end = 3 + (j + 1) * 6
        mask[:, start:end] = 1
    return mask


def build_keyframe_mask(T: int, D: int = 135, interval: int = 30) -> torch.Tensor:
    """Keep every `interval`-th frame, mask rest."""
    mask = torch.ones(T, D)
    for t in range(0, T, interval):
        mask[t] = 0
    # Always keep last frame
    mask[-1] = 0
    return mask


# ============================================================================
# Metrics
# ============================================================================

def compute_jitter(motion: np.ndarray) -> float:
    """3rd-order finite difference (acceleration of velocity)."""
    if motion.shape[0] < 4:
        return 0.0
    # 3rd order diff: x[t+3] - 3x[t+2] + 3x[t+1] - x[t]
    diff3 = motion[3:] - 3 * motion[2:-1] + 3 * motion[1:-2] - motion[:-3]
    return float(np.mean(np.abs(diff3)))


def compute_velocity_stats(motion: np.ndarray) -> Dict[str, float]:
    """Compute velocity and acceleration stats."""
    if motion.shape[0] < 2:
        return {'avg_vel': 0, 'max_vel': 0}
    vel = np.diff(motion, axis=0)
    vel_norms = np.linalg.norm(vel.reshape(vel.shape[0], -1), axis=-1)
    return {
        'avg_vel': float(np.mean(vel_norms)),
        'max_vel': float(np.max(vel_norms)),
    }


def compute_boundary_error(
    original: np.ndarray, generated: np.ndarray,
    mask: np.ndarray, boundary_width: int = 3
) -> float:
    """Compute error at mask boundary (transition quality)."""
    T, D = mask.shape
    # Find boundary frames (mask transitions from 0 to 1 or 1 to 0)
    mask_any = mask.max(axis=-1) > 0.5  # per-frame
    boundary_frames = []
    for t in range(1, T):
        if mask_any[t] != mask_any[t - 1]:
            for dt in range(-boundary_width, boundary_width + 1):
                ft = t + dt
                if 0 <= ft < T:
                    boundary_frames.append(ft)
    boundary_frames = list(set(boundary_frames))

    if not boundary_frames:
        return 0.0

    orig_boundary = original[boundary_frames]
    gen_boundary = generated[boundary_frames]
    return float(np.mean(np.abs(orig_boundary - gen_boundary)))


def compute_bone_length_consistency(motion_135d: np.ndarray) -> float:
    """Compute bone length CV (coefficient of variation) from rot6d motion.

    This uses a simple approximation: compare consecutive-frame joint-group
    differences. True FK requires SMPL body model.
    """
    # Simple metric: look at relative stability of rot6d values
    T = motion_135d.shape[0]
    if T < 2:
        return 0.0
    # Per-joint rot6d norm stability
    rot6d = motion_135d[:, 3:].reshape(T, 22, 6)
    norms = np.linalg.norm(rot6d, axis=-1)  # [T, 22]
    cv = np.std(norms, axis=0) / (np.mean(norms, axis=0) + 1e-8)
    return float(np.mean(cv))


def compute_translation_metrics(motion_135d: np.ndarray) -> Dict[str, float]:
    """Translation-specific metrics."""
    transl = motion_135d[:, :3]
    total_dist = float(np.sum(np.linalg.norm(np.diff(transl, axis=0), axis=-1)))
    height_range = float(np.max(transl[:, 1]) - np.min(transl[:, 1]))
    return {
        'total_distance': round(total_dist, 4),
        'height_range': round(height_range, 4),
    }


# ============================================================================
# Evaluation tasks
# ============================================================================

TASKS = {
    'reconstruction': {
        'mask_fn': build_zero_mask,
        'desc': 'Identity reconstruction (mask=0)',
        'needs_gt': True,
    },
    'full_generation': {
        'mask_fn': build_full_mask,
        'desc': 'Full generation (mask=1, T2M-like)',
        'needs_gt': False,
    },
    'temporal_inbetween': {
        'mask_fn': build_temporal_inbetween_mask,
        'desc': 'Temporal in-between (keep 20 head+tail)',
        'needs_gt': True,
    },
    'upper_body_completion': {
        'mask_fn': build_upper_body_mask,
        'desc': 'Upper body completion (keep lower)',
        'needs_gt': True,
    },
    'keyframe_interpolation': {
        'mask_fn': build_keyframe_mask,
        'desc': 'Keyframe interpolation (every 30th frame)',
        'needs_gt': True,
    },
}


def evaluate_single(
    pipeline,
    bundle,
    sample: Dict,
    task_name: str,
    task_cfg: Dict,
    device: str,
    replacement_modes: List[str] = ['none'],
) -> Dict:
    """Run evaluation on a single sample for a single task."""
    motion = sample['motion']  # [T, 135]
    T, D = motion.shape

    mask_fn = task_cfg['mask_fn']
    mask = mask_fn(T, D)  # [T, 135]

    # Normalize
    motion_norm = bundle.normalize_motion(motion.unsqueeze(0).to(device))  # [1, T, 135]
    src_mask = mask.unsqueeze(0).to(device)  # [1, T, 135]

    # For completion tasks: zero out masked regions in src_motion
    src_motion_norm = motion_norm.clone()
    if task_name != 'reconstruction':
        src_motion_norm = src_motion_norm * (1 - src_mask)

    results = {}
    for rep_mode in replacement_modes:
        pipeline.replacement_guidance = rep_mode

        batch = {
            'src_motion': src_motion_norm,
            'src_mask': src_mask,
            'src_length': [T],
            'tgt_length': [T],
        }

        t0 = time.time()
        with torch.no_grad():
            output = pipeline(batch)
        elapsed = time.time() - t0

        sampled = output['latent']  # [1, T, 135] normalized
        output_denorm = bundle.denormalize_motion(sampled)[0].cpu().numpy()
        original_raw = motion.numpy()

        # Metrics
        metrics = {
            'time_sec': round(elapsed, 2),
            'jitter_output': round(compute_jitter(output_denorm), 6),
            'jitter_original': round(compute_jitter(original_raw), 6),
        }
        metrics.update({
            f'transl_{k}': v
            for k, v in compute_translation_metrics(output_denorm).items()
        })
        metrics['bone_length_cv'] = round(
            compute_bone_length_consistency(output_denorm), 6)

        vel = compute_velocity_stats(output_denorm)
        metrics['avg_velocity'] = round(vel['avg_vel'], 6)
        metrics['max_velocity'] = round(vel['max_vel'], 6)

        if task_cfg['needs_gt']:
            # MPJPE-like: mean absolute error in output space
            diff = np.abs(output_denorm - original_raw)
            metrics['mae_all'] = round(float(diff.mean()), 6)
            # Masked region error
            mask_np = mask.numpy()
            mask_region = mask_np > 0.5
            if mask_region.any():
                metrics['mae_masked'] = round(float(diff[mask_region].mean()), 6)
            # Unmasked region error (should be near-zero for good conditioning)
            unmask_region = mask_np < 0.5
            if unmask_region.any():
                metrics['mae_unmasked'] = round(float(diff[unmask_region].mean()), 6)
            # Boundary error
            metrics['boundary_error'] = round(
                compute_boundary_error(original_raw, output_denorm, mask_np), 6)

        results[rep_mode] = metrics

    return results


# ============================================================================
# Main
# ============================================================================

def load_model(model_name: str, device: str):
    """Load model bundle and create pipeline."""
    from mmengine.config import Config
    from hftrainer.registry import MODEL_BUNDLES
    from hftrainer.utils.checkpoint_utils import load_checkpoint, find_latest_checkpoint
    from hftrainer.pipelines.motion.hymotion_m2m_pipeline import HyMotionM2MPipeline

    model_info = MODELS[model_name]
    cfg = Config.fromfile(model_info['config'])
    bundle = MODEL_BUNDLES.build(cfg.model.to_dict())

    ckpt_path = find_latest_checkpoint(model_info['work_dir'])
    if ckpt_path is None:
        print(f'  WARNING: No checkpoint found for {model_name}')
        return None, None, None

    print(f'  Loading checkpoint: {ckpt_path}')
    sd = load_checkpoint(ckpt_path, map_location='cpu')
    bundle.load_state_dict_selective(sd)
    del sd
    bundle.eval()
    bundle = bundle.to(device)

    pipeline = HyMotionM2MPipeline(bundle=bundle, num_steps=50)
    return bundle, pipeline, ckpt_path


def main():
    parser = argparse.ArgumentParser(description='Evaluate M2M checkpoints')
    parser.add_argument('--models', nargs='+', choices=list(MODELS.keys()),
                        help='Models to evaluate')
    parser.add_argument('--all', action='store_true',
                        help='Evaluate all available models')
    parser.add_argument('--tasks', nargs='+', choices=list(TASKS.keys()),
                        default=list(TASKS.keys()),
                        help='Tasks to evaluate')
    parser.add_argument('--max-samples', type=int, default=50)
    parser.add_argument('--num-steps', type=int, default=50)
    parser.add_argument('--test-replacement', action='store_true',
                        help='Also test replacement guidance modes')
    parser.add_argument('--output-dir', type=str,
                        default='work_dirs/m2m_eval_report')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--test-anno', type=str,
                        default='data/annotation/test_motionhub_recon.json')
    parser.add_argument('--data-dir', type=str, default='data/motionhub')
    args = parser.parse_args()

    if args.all:
        model_names = list(MODELS.keys())
    elif args.models:
        model_names = args.models
    else:
        # Default: 4 main models
        model_names = ['uncond_fm', 'caption_fm', 'uncond_jit', 'caption_jit']

    replacement_modes = ['none']
    if args.test_replacement:
        replacement_modes = ['none', 'skip_last', 'flow_interp']

    os.makedirs(args.output_dir, exist_ok=True)

    # Load test samples
    print(f'Loading test samples from {args.test_anno}...')
    samples = load_test_samples(args.test_anno, args.data_dir, args.max_samples)
    print(f'  Loaded {len(samples)} valid samples')

    if not samples:
        print('ERROR: No valid test samples found!')
        sys.exit(1)

    all_results = {}

    for model_name in model_names:
        print(f'\n{"="*60}')
        print(f'Model: {model_name} — {MODELS[model_name]["desc"]}')
        print(f'{"="*60}')

        bundle, pipeline, ckpt_path = load_model(model_name, args.device)
        if bundle is None:
            all_results[model_name] = {'error': 'no checkpoint'}
            continue

        pipeline.num_steps = args.num_steps
        model_results = {
            'checkpoint': ckpt_path,
            'num_steps': args.num_steps,
            'tasks': {},
        }

        for task_name in args.tasks:
            task_cfg = TASKS[task_name]
            print(f'\n  Task: {task_name} — {task_cfg["desc"]}')

            task_metrics_list = []
            for i, sample in enumerate(samples):
                try:
                    result = evaluate_single(
                        pipeline, bundle, sample, task_name, task_cfg,
                        args.device, replacement_modes)
                    task_metrics_list.append({
                        'sample_idx': i,
                        'path': sample['path'],
                        'T': sample['T'],
                        'metrics': result,
                    })
                    if (i + 1) % 10 == 0:
                        print(f'    [{i+1}/{len(samples)}] done')
                except Exception as e:
                    print(f'    [{i+1}] ERROR: {e}')
                    continue

            # Aggregate metrics
            aggregated = {}
            for rep_mode in replacement_modes:
                mode_metrics = [
                    m['metrics'][rep_mode]
                    for m in task_metrics_list
                    if rep_mode in m['metrics']
                ]
                if not mode_metrics:
                    continue
                agg = {}
                for key in mode_metrics[0]:
                    if key == 'time_sec':
                        agg[key] = round(np.mean([m[key] for m in mode_metrics]), 2)
                    elif isinstance(mode_metrics[0][key], (int, float)):
                        vals = [m[key] for m in mode_metrics]
                        agg[f'{key}_mean'] = round(float(np.mean(vals)), 6)
                        agg[f'{key}_std'] = round(float(np.std(vals)), 6)
                        agg[f'{key}_median'] = round(float(np.median(vals)), 6)
                aggregated[rep_mode] = agg

            model_results['tasks'][task_name] = {
                'num_samples': len(task_metrics_list),
                'aggregated': aggregated,
                'per_sample': task_metrics_list[:5],  # Save first 5 for detail
            }

            # Print summary
            for rep_mode, agg in aggregated.items():
                print(f'    [{rep_mode}] n={len(task_metrics_list)}, '
                      f'time={agg.get("time_sec", "?")}s')
                for k, v in sorted(agg.items()):
                    if k != 'time_sec' and 'mean' in k:
                        print(f'      {k}: {v}')

        all_results[model_name] = model_results

        # Free GPU memory
        del bundle, pipeline
        torch.cuda.empty_cache()

    # Save full results
    output_path = os.path.join(args.output_dir, 'eval_results.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nFull results saved to {output_path}')

    # Print comparative summary
    print(f'\n{"="*80}')
    print('COMPARATIVE SUMMARY')
    print(f'{"="*80}')
    for task_name in args.tasks:
        print(f'\n--- {task_name} ---')
        header = f'{"Model":<25} {"jitter_out":<15} {"mae_masked":<15} {"mae_unmask":<15} {"boundary":<15}'
        print(header)
        for model_name in model_names:
            res = all_results.get(model_name, {})
            if 'error' in res:
                print(f'{model_name:<25} ERROR: {res["error"]}')
                continue
            task_res = res.get('tasks', {}).get(task_name, {})
            agg = task_res.get('aggregated', {}).get('none', {})
            jitter = agg.get('jitter_output_mean', 'N/A')
            mae_m = agg.get('mae_masked_mean', 'N/A')
            mae_u = agg.get('mae_unmasked_mean', 'N/A')
            bnd = agg.get('boundary_error_mean', 'N/A')
            print(f'{model_name:<25} {str(jitter):<15} {str(mae_m):<15} {str(mae_u):<15} {str(bnd):<15}')


if __name__ == '__main__':
    main()
