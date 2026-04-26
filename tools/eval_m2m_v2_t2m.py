#!/usr/bin/env python3
"""T2M generation on Yiran subset — multi-GPU parallel inference.

Runs all 4 v2 models on 240 text prompts from 251125_yiran_subset.json.
Uncond models: null text (pure generation). Caption models: encode text at runtime.

Each GPU handles one model. Outputs NPZ + per-sample metrics + aggregate report.

Usage:
    python tools/eval_m2m_v2_t2m.py                      # all 4 models
    python tools/eval_m2m_v2_t2m.py --models uncond_local # single model
    python tools/eval_m2m_v2_t2m.py --gpus 0 1 2 3       # specify GPUs
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from multiprocessing import Process, Queue

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ============================================================================
# Constants
# ============================================================================

PROMPT_FILE = 'data/eval/t2m/251125_yiran_subset.json'
OUTPUT_DIR = 'work_dirs/m2m_v2_t2m_eval'
BONE_OFFSETS_PATH = 'data/hymotion_m2m_data/bone_offsets_22.pt'

MODELS = {
    'uncond_local': {
        'config': 'configs/hymotion_m2m_v2/hymotion_m2m_v2_uncond_local_046b.py',
        'work_dir': 'work_dirs/hymotion_m2m_v2_uncond_local_046b',
        'has_caption': False,
        'rotation_space': 'local',
    },
    'uncond_global': {
        'config': 'configs/hymotion_m2m_v2/hymotion_m2m_v2_uncond_global_046b.py',
        'work_dir': 'work_dirs/hymotion_m2m_v2_uncond_global_046b',
        'has_caption': False,
        'rotation_space': 'global',
    },
    'caption_local': {
        'config': 'configs/hymotion_m2m_v2/hymotion_m2m_v2_caption_local_046b.py',
        'work_dir': 'work_dirs/hymotion_m2m_v2_caption_local_046b',
        'has_caption': True,
        'rotation_space': 'local',
    },
    'caption_global': {
        'config': 'configs/hymotion_m2m_v2/hymotion_m2m_v2_caption_global_046b.py',
        'work_dir': 'work_dirs/hymotion_m2m_v2_caption_global_046b',
        'has_caption': True,
        'rotation_space': 'global',
    },
    # Phase 1 variants: pure T2M curriculum (no completion tasks)
    'caption_local_phase1': {
        'config': 'configs/hymotion_m2m_v2/hymotion_m2m_v2_caption_local_phase1.py',
        'work_dir': 'work_dirs/hymotion_m2m_v2_caption_local_phase1',
        'has_caption': True,
        'rotation_space': 'local',
    },
    'caption_global_phase1': {
        'config': 'configs/hymotion_m2m_v2/hymotion_m2m_v2_caption_global_phase1.py',
        'work_dir': 'work_dirs/hymotion_m2m_v2_caption_global_phase1',
        'has_caption': True,
        'rotation_space': 'global',
    },
}


# ============================================================================
# Parse prompts
# ============================================================================

def load_prompts(path: str) -> List[Dict]:
    """Parse 251125_yiran_subset.json -> list of {text, frames, id}."""
    with open(path) as f:
        data = json.load(f)
    raw = data['test_prompts_251125_yiran_subset']
    prompts = []
    for entry in raw:
        parts = entry.split('#')
        prompts.append({
            'text': parts[0],
            'frames': int(parts[1]),
            'cond': parts[2],
            'id': parts[3],
        })
    return prompts


# ============================================================================
# Single-GPU worker
# ============================================================================

def run_model_on_gpu(
    model_name: str,
    gpu_id: int,
    prompts: List[Dict],
    output_dir: str,
    num_steps: int,
    cfg_scale: float,
    result_queue: Queue,
):
    """Worker function: load model on specified GPU, run all prompts."""
    import torch
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = 'cuda:0'

    model_info = MODELS[model_name]
    model_output_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)
    npz_dir = os.path.join(model_output_dir, 'npz')
    os.makedirs(npz_dir, exist_ok=True)

    print(f'[GPU {gpu_id}] {model_name}: loading model...')

    try:
        from mmengine.config import Config
        from hftrainer.registry import MODEL_BUNDLES
        from hftrainer.utils.checkpoint_utils import load_checkpoint, find_latest_checkpoint
        from hftrainer.pipelines.motion.hymotion_m2m_pipeline import HyMotionM2MPipeline
        from hftrainer.evaluation.motion.m2m_eval_metrics import (
            compute_jitter_positions, compute_bone_length_cv,
            compute_foot_ground_metrics, compute_jitter_135,
        )
        from hftrainer.pipelines.motion.differentiable_fk import motion135_to_fk

        # Quality checker (rot6d -> axis-angle conversion needed)
        quality_checker = None
        try:
            from hftrainer.evaluation.quality_check_rules.motion_quality_checker import MotionQualityChecker
            from hftrainer.models.motion.components.utils.geometry.rotation_convert import (
                rotation_6d_to_axis_angle,
            )
            quality_checker = MotionQualityChecker(device=device)
            print(f'[GPU {gpu_id}] {model_name}: quality checker loaded ({len(quality_checker.enabled)} checkers)')
        except Exception as e:
            print(f'[GPU {gpu_id}] {model_name}: quality checker unavailable: {e}')

        def _motion135_to_checker_input(motion_135_np, rot_space='local'):
            """Convert motion_135 (3+132 rot6d) to {poses: axis-angle, trans} for quality checker."""
            transl = motion_135_np[:, :3]  # (T, 3)
            rot6d = motion_135_np[:, 3:135].reshape(-1, 22, 6)  # (T, 22, 6)
            # rot6d -> axis-angle (T, 22, 3)
            aa = rotation_6d_to_axis_angle(torch.from_numpy(rot6d).float()).numpy()
            poses = aa.reshape(-1, 66)  # (T, 66) for SMPL 22 joints
            return {'poses': poses, 'trans': transl}

        cfg = Config.fromfile(model_info['config'])
        bundle = MODEL_BUNDLES.build(cfg.model.to_dict())

        # For caption models: inject text_encoder config for runtime encoding
        # (Training uses pre-extracted embeddings, but eval needs live encoding)
        if model_info['has_caption'] and bundle._text_encoder_cfg is None:
            bundle._text_encoder_cfg = {
                'llm_type': 'qwen3_embedding',
                'max_length_llm': 512,
                'sentence_emb_type': 'clipl',
                'max_length_sentence_emb': 77,
            }
            print(f'[GPU {gpu_id}] {model_name}: injected runtime text_encoder config')

        ckpt_path = find_latest_checkpoint(model_info['work_dir'])
        if ckpt_path is None:
            msg = f'No checkpoint found at {model_info["work_dir"]}'
            print(f'[GPU {gpu_id}] {model_name}: ERROR - {msg}')
            result_queue.put({'model': model_name, 'error': msg})
            return

        print(f'[GPU {gpu_id}] {model_name}: loading {ckpt_path}')
        sd = load_checkpoint(ckpt_path, map_location='cpu')
        bundle.load_state_dict_selective(sd)
        del sd
        bundle.eval()
        bundle = bundle.to(device)

        pipeline = HyMotionM2MPipeline(
            bundle=bundle,
            num_steps=num_steps,
            text_guidance_scale=cfg_scale if model_info['has_caption'] else 1.0,
            replacement_guidance='none',  # T2M: no imputation needed (full mask)
        )

        # Load bone offsets for FK metrics
        bone_offsets = None
        if os.path.exists(BONE_OFFSETS_PATH):
            bone_offsets = torch.load(BONE_OFFSETS_PATH, map_location='cpu')

        rotation_space = model_info['rotation_space']

        print(f'[GPU {gpu_id}] {model_name}: running {len(prompts)} prompts...')

        per_sample = []
        t_start = time.time()

        for i, prompt in enumerate(prompts):
            text = prompt['text']
            T = min(prompt['frames'], 360)  # actual target length
            D = 198
            T_PAD = 360  # training always pads to 360

            # Build full mask (T2M: all generate), padded to 360
            src_mask = torch.zeros(1, T_PAD, D, device=device)
            src_mask[:, :T, :] = 1.0  # mask=1 for real frames, 0 for padding

            # Build zero src_motion (no condition), padded to 360
            src_motion = torch.zeros(1, T_PAD, D, device=device)

            batch = {
                'src_motion': src_motion,
                'src_mask': src_mask,
                'src_length': [T],
                'tgt_length': [T],  # real length — pipeline builds tgt_padding_mask from this
            }

            # For caption models: try encoding text
            if model_info['has_caption'] and text:
                try:
                    text_out = bundle.encode_text([text])
                    batch['text_vec_raw'] = text_out['text_vec_raw'].to(device)
                    batch['text_ctxt_raw'] = text_out['text_ctxt_raw'].to(device)
                    batch['text_ctxt_raw_length'] = text_out['text_ctxt_raw_length'].to(device)
                except Exception as e:
                    print(f'  WARNING: text encoding failed: {e}')  # Don't silently swallow

            # Run inference
            t0 = time.time()
            with torch.no_grad():
                output = pipeline(batch)
            elapsed = time.time() - t0

            # Decode output
            sampled = output['latent']  # (1, T_PAD, 198) normalized
            output_denorm = bundle.denormalize_motion(sampled)[0].cpu()  # (T_PAD, 198)

            # Crop to actual target length (discard padding frames)
            output_denorm = output_denorm[:T]

            # Extract 135-dim for metrics
            output_135 = output_denorm[:, :135].numpy()

            # --- Sanity checks ---
            rot6d_part = output_135[:, 3:135].reshape(T, 22, 6)
            rot6d_norms = np.linalg.norm(rot6d_part, axis=-1)  # (T, 22)
            transl = output_135[:, :3]

            sanity = {
                'rot6d_norm_mean': float(rot6d_norms.mean()),
                'rot6d_norm_std': float(rot6d_norms.std()),
                'rot6d_norm_min': float(rot6d_norms.min()),
                'rot6d_norm_max': float(rot6d_norms.max()),
                'transl_range_x': float(transl[:, 0].max() - transl[:, 0].min()),
                'transl_range_y': float(transl[:, 1].max() - transl[:, 1].min()),
                'transl_range_z': float(transl[:, 2].max() - transl[:, 2].min()),
                'output_value_range': [float(output_135.min()), float(output_135.max())],
            }

            # --- Metrics ---
            metrics = {
                'jitter_135': compute_jitter_135(output_135),
                'inference_time': round(elapsed, 2),
            }
            metrics.update(sanity)

            # FK-based metrics
            if bone_offsets is not None:
                try:
                    output_135_t = torch.from_numpy(output_135).float()
                    world_pos, _, _, _ = motion135_to_fk(
                        output_135_t, bone_offsets, rotation_space=rotation_space)
                    pos_np = world_pos.numpy()  # (T, 22, 3)

                    metrics['jitter_pos'] = compute_jitter_positions(pos_np, fps=30.0)
                    bl = compute_bone_length_cv(pos_np)
                    metrics.update(bl)
                    foot = compute_foot_ground_metrics(pos_np, fps=30.0)
                    metrics.update(foot)

                    # Position sanity
                    metrics['pos_range_y'] = float(pos_np[:, :, 1].max() - pos_np[:, :, 1].min())
                    metrics['pos_mean_y'] = float(pos_np[:, :, 1].mean())
                    metrics['pelvis_height_mean'] = float(pos_np[:, 0, 1].mean())

                    # Save NPZ (with axis-angle poses for quality checker)
                    npz_path = os.path.join(npz_dir, f'{prompt["id"]}.npz')
                    np.savez_compressed(
                        npz_path,
                        motion_135=output_135,
                        positions=pos_np,
                        translation=transl,
                    )

                    # --- Physics metrics from FK positions ---
                    # Velocity / acceleration stats
                    joint_vel = np.diff(pos_np, axis=0) * 30.0  # (T-1, 22, 3) m/s
                    joint_acc = np.diff(joint_vel, axis=0) * 30.0  # (T-2, 22, 3) m/s^2
                    vel_mag = np.linalg.norm(joint_vel, axis=-1)  # (T-1, 22)
                    acc_mag = np.linalg.norm(joint_acc, axis=-1)  # (T-2, 22)
                    metrics['avg_velocity'] = float(vel_mag.mean())
                    metrics['max_velocity'] = float(vel_mag.max())
                    metrics['avg_acceleration'] = float(acc_mag.mean())
                    metrics['max_acceleration'] = float(acc_mag.max())
                    # Pelvis-specific
                    metrics['pelvis_trans_jerk'] = float(
                        np.linalg.norm(np.diff(pos_np[:, 0, :], n=3, axis=0) * (30.0**3), axis=-1).mean()
                    ) if T > 3 else 0.0
                    # Head stability
                    if pos_np.shape[1] > 15:
                        head_vel = np.linalg.norm(np.diff(pos_np[:, 15, :], axis=0), axis=-1) * 30.0
                        metrics['head_jitter_ratio'] = float((head_vel > 0.5).mean())
                    # Self-penetration (simple capsule approximation via joint distances)
                    # Check if left/right wrist penetrates torso (simplified)
                    if pos_np.shape[1] >= 22:
                        spine_center = (pos_np[:, 3, :] + pos_np[:, 6, :] + pos_np[:, 9, :]) / 3  # spine avg
                        l_wrist_dist = np.linalg.norm(pos_np[:, 20, :] - spine_center, axis=-1)
                        r_wrist_dist = np.linalg.norm(pos_np[:, 21, :] - spine_center, axis=-1)
                        torso_radius = 0.15  # approximate
                        metrics['arm_penetration_ratio'] = float(
                            ((l_wrist_dist < torso_radius) | (r_wrist_dist < torso_radius)).mean()
                        )

                except Exception as e:
                    metrics['fk_error'] = str(e)

            # --- Quality Checker ---
            try:
                if quality_checker is not None:
                    # Convert motion_135 (rot6d) to axis-angle for checker
                    checker_data = _motion135_to_checker_input(output_135, rotation_space)
                    qc_result = quality_checker.check(checker_data)
                    metrics['qc_pass'] = 1 if qc_result.is_valid else 0
                    metrics['qc_num_failed'] = len(qc_result.failed_checks)
                    metrics['qc_num_borderline'] = len(qc_result.borderline_checks)
                    metrics['qc_failed_checks'] = qc_result.failed_checks
                    # Per-checker pass/fail as individual metrics
                    for checker_name, cr in qc_result.all_results.items():
                        metrics[f'qc_{checker_name}'] = 1 if cr['is_valid'] else 0
            except Exception as e:
                metrics['qc_error'] = str(e)

            # 198-dim position channel sanity
            if output_denorm.shape[-1] >= 198:
                pos_channel = output_denorm[:, 135:198].numpy()
                metrics['pos_channel_range'] = [float(pos_channel.min()), float(pos_channel.max())]
                metrics['pos_channel_mean'] = float(pos_channel.mean())

                # FK consistency: compare rotation-FK pos vs position channel
                if bone_offsets is not None and 'fk_error' not in metrics:
                    try:
                        # FK gives world pos; position channel is XZ-rel-pelvis, Y-abs
                        fk_joint = pos_np[:, 1:, :]  # (T, 21, 3) exclude pelvis
                        pelvis_xz = pos_np[:, 0:1, [0, 2]]  # (T, 1, 2)
                        fk_rel = fk_joint.copy()
                        fk_rel[:, :, 0] -= pelvis_xz[:, :, 0]
                        fk_rel[:, :, 2] -= pelvis_xz[:, :, 1]
                        fk_flat = fk_rel.reshape(-1, 63)
                        pos_flat = pos_channel.reshape(-1, 63)
                        fk_consistency = float(np.mean(np.abs(fk_flat - pos_flat)))
                        metrics['fk_consistency_mae'] = fk_consistency
                    except Exception:
                        pass

            per_sample.append({
                'prompt_id': prompt['id'],
                'text': text,
                'target_frames': prompt['frames'],
                'actual_frames': T,
                'metrics': metrics,
            })

            if (i + 1) % 20 == 0 or (i + 1) == len(prompts):
                elapsed_total = time.time() - t_start
                speed = (i + 1) / elapsed_total
                print(f'[GPU {gpu_id}] {model_name}: {i+1}/{len(prompts)} '
                      f'({speed:.1f} samples/min)')

        # Aggregate
        total_time = time.time() - t_start
        metric_keys = set()
        for s in per_sample:
            metric_keys.update(s['metrics'].keys())

        agg = {}
        for key in sorted(metric_keys):
            vals = [s['metrics'][key] for s in per_sample
                    if key in s['metrics'] and isinstance(s['metrics'][key], (int, float))]
            if vals:
                arr = np.array(vals)
                agg[key] = {
                    'mean': round(float(np.mean(arr)), 6),
                    'std': round(float(np.std(arr)), 6),
                    'median': round(float(np.median(arr)), 6),
                    'min': round(float(np.min(arr)), 6),
                    'max': round(float(np.max(arr)), 6),
                }

        result = {
            'model': model_name,
            'checkpoint': ckpt_path,
            'rotation_space': rotation_space,
            'has_caption': model_info['has_caption'],
            'num_prompts': len(prompts),
            'num_steps': num_steps,
            'cfg_scale': cfg_scale if model_info['has_caption'] else 1.0,
            'total_time_sec': round(total_time, 1),
            'speed_samples_per_min': round(len(prompts) / total_time * 60, 1),
            'aggregated': agg,
            'per_sample': per_sample,
        }

        # Save per-model result
        result_path = os.path.join(model_output_dir, 'result.json')
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)

        print(f'[GPU {gpu_id}] {model_name}: DONE in {total_time:.0f}s. '
              f'Saved to {result_path}')

        # Key sanity summary
        rot_norm = agg.get('rot6d_norm_mean', {}).get('mean', 'N/A')
        jitter = agg.get('jitter_pos', {}).get('mean', 'N/A')
        pelvis_h = agg.get('pelvis_height_mean', {}).get('mean', 'N/A')
        bone_cv = agg.get('bone_length_cv_mean', {}).get('mean', 'N/A')
        skating = agg.get('foot_skating_ratio', {}).get('mean', 'N/A')
        fk_cons = agg.get('fk_consistency_mae', {}).get('mean', 'N/A')

        print(f'[GPU {gpu_id}] {model_name} SUMMARY:')
        print(f'  rot6d_norm_mean={rot_norm}, jitter_pos={jitter}')
        print(f'  pelvis_height={pelvis_h}, bone_cv={bone_cv}')
        print(f'  skating={skating}, fk_consistency={fk_cons}')

        result_queue.put({'model': model_name, 'summary': {
            'rot6d_norm_mean': rot_norm,
            'jitter_pos': jitter,
            'pelvis_height_mean': pelvis_h,
            'bone_length_cv_mean': bone_cv,
            'foot_skating_ratio': skating,
            'fk_consistency_mae': fk_cons,
        }})

    except Exception as e:
        msg = f'{type(e).__name__}: {e}\n{traceback.format_exc()}'
        print(f'[GPU {gpu_id}] {model_name}: FATAL ERROR\n{msg}')
        result_queue.put({'model': model_name, 'error': msg})


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='T2M eval on Yiran subset')
    parser.add_argument('--models', nargs='+', choices=list(MODELS.keys()),
                        default=list(MODELS.keys()))
    parser.add_argument('--gpus', nargs='+', type=int, default=None,
                        help='GPU IDs to use (default: auto-assign)')
    parser.add_argument('--num-steps', type=int, default=50)
    parser.add_argument('--cfg-scale', type=float, default=5.0,
                        help='CFG scale for caption models (5.0 standard for flow matching)')
    parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR)
    parser.add_argument('--prompt-file', type=str, default=PROMPT_FILE)
    args = parser.parse_args()

    # Load prompts
    prompts = load_prompts(args.prompt_file)
    print(f'Loaded {len(prompts)} prompts from {args.prompt_file}')

    os.makedirs(args.output_dir, exist_ok=True)

    # Assign GPUs
    model_names = args.models
    if args.gpus:
        gpu_ids = args.gpus
    else:
        # Auto-assign: use GPUs 0..N-1
        gpu_ids = list(range(len(model_names)))

    if len(gpu_ids) < len(model_names):
        print(f'WARNING: {len(model_names)} models but only {len(gpu_ids)} GPUs. '
              f'Will run sequentially on GPU {gpu_ids[0]} for overflow.')
        while len(gpu_ids) < len(model_names):
            gpu_ids.append(gpu_ids[0])

    print(f'Model -> GPU assignment:')
    for m, g in zip(model_names, gpu_ids):
        print(f'  {m} -> GPU {g}')

    # Launch parallel workers
    result_queue = Queue()
    processes = []

    for model_name, gpu_id in zip(model_names, gpu_ids):
        p = Process(
            target=run_model_on_gpu,
            args=(model_name, gpu_id, prompts, args.output_dir,
                  args.num_steps, args.cfg_scale, result_queue),
        )
        p.start()
        processes.append(p)
        print(f'Started {model_name} on GPU {gpu_id} (PID {p.pid})')

    # Wait for all
    for p in processes:
        p.join()

    # Collect results
    print(f'\n{"=" * 70}')
    print('ALL MODELS COMPLETE')
    print(f'{"=" * 70}')

    results = {}
    while not result_queue.empty():
        r = result_queue.get()
        results[r['model']] = r

    # Print comparative summary
    print(f'\n{"Model":<20} {"rot6d_norm":>12} {"jitter_pos":>12} {"pelvis_h":>12} '
          f'{"bone_cv":>12} {"skating":>12} {"fk_cons":>12}')
    print('-' * 92)

    for model_name in model_names:
        r = results.get(model_name, {})
        if 'error' in r:
            print(f'{model_name:<20} ERROR: {r["error"][:60]}')
            continue
        s = r.get('summary', {})
        def _fmt(v):
            if isinstance(v, float):
                return f'{v:.4f}'
            return str(v)[:12]
        print(f'{model_name:<20} '
              f'{_fmt(s.get("rot6d_norm_mean", "N/A")):>12} '
              f'{_fmt(s.get("jitter_pos", "N/A")):>12} '
              f'{_fmt(s.get("pelvis_height_mean", "N/A")):>12} '
              f'{_fmt(s.get("bone_length_cv_mean", "N/A")):>12} '
              f'{_fmt(s.get("foot_skating_ratio", "N/A")):>12} '
              f'{_fmt(s.get("fk_consistency_mae", "N/A")):>12}')

    print(f'\nResults saved to {args.output_dir}/')
    print('\nKey sanity checks:')
    print('  rot6d_norm_mean ≈ 1.0 (valid rotation)')
    print('  pelvis_height   ≈ 0.85-1.0m (standing human)')
    print('  bone_cv         < 0.05 (stable skeleton)')
    print('  fk_consistency  ≈ 0 (rot/pos channels agree)')


if __name__ == '__main__':
    main()
