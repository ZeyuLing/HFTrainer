#!/usr/bin/env python3
"""T2M generation on Yiran subset — multi-GPU parallel inference.

Two modes:

(A) Per-model parallelism (legacy): each --models entry runs on its own GPU
    with one --cfg-scale.  Use this when comparing multiple models at one
    cfg.

(B) Per-chunk × cfg-sweep parallelism (new): split the 240 prompts into
    --prompt-chunks worker chunks, each worker loads the model once and
    runs **every cfg in --cfg-sweep sequentially** over its chunk.  Use
    this for the CFG ablation: same ckpt, same prompts, different cfg —
    the model load + cuda warmup is paid once per worker, not once per
    cfg.  All --gpus are utilised.

Outputs:
- (A) <out>/<model>/result.json
- (B) <out>/<model>/cfg{X}/result.json    + cfg{X}/npz/<id>.npz
       The main process merges per-chunk partial JSONs into a single
       result.json per cfg after all workers finish.

Usage examples:
    # Legacy: 4 models in parallel at cfg=5
    python tools/eval_m2m_v2_t2m.py

    # CFG ablation: caption_local_phase2 ep_2860 unpatched, 5 cfgs, 8 GPUs
    python tools/eval_m2m_v2_t2m.py \\
        --models caption_local_phase2 \\
        --ckpt-path work_dirs/.../checkpoint-epoch_2860 \\
        --cfg-sweep 1.0 1.5 2.5 4.0 7.5 \\
        --prompt-chunks 8 \\
        --gpus 0 1 2 3 4 5 6 7 \\
        --output-suffix _cfg_ablation_2860_unpatched
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
        'config': 'configs/hymotion_m2m/hymotion_m2m_uncond_local_046b.py',
        'work_dir': 'work_dirs/hymotion_m2m_v2_uncond_local_046b',
        'has_caption': False,
        'rotation_space': 'local',
    },
    'uncond_global': {
        'config': 'configs/hymotion_m2m/hymotion_m2m_uncond_global_046b.py',
        'work_dir': 'work_dirs/hymotion_m2m_v2_uncond_global_046b',
        'has_caption': False,
        'rotation_space': 'global',
    },
    'caption_local': {
        'config': 'configs/hymotion_m2m/hymotion_m2m_caption_local_046b.py',
        'work_dir': 'work_dirs/hymotion_m2m_v2_caption_local_046b',
        'has_caption': True,
        'rotation_space': 'local',
    },
    'caption_global': {
        'config': 'configs/hymotion_m2m/hymotion_m2m_caption_global_046b.py',
        'work_dir': 'work_dirs/hymotion_m2m_v2_caption_global_046b',
        'has_caption': True,
        'rotation_space': 'global',
    },
    # Phase 1 variants: pure T2M curriculum (no completion tasks)
    'caption_local_phase1': {
        'config': 'configs/hymotion_m2m/hymotion_m2m_caption_local_phase1.py',
        'work_dir': 'work_dirs/hymotion_m2m_v2_caption_local_phase1',
        'has_caption': True,
        'rotation_space': 'local',
    },
    'caption_global_phase1': {
        'config': 'configs/hymotion_m2m/hymotion_m2m_caption_global_phase1.py',
        'work_dir': 'work_dirs/hymotion_m2m_v2_caption_global_phase1',
        'has_caption': True,
        'rotation_space': 'global',
    },
    # Phase 2: pretrained T2M + completion / editing curriculum.  Bundle
    # architecture identical to phase1, only loss/data pipeline differs.
    'caption_local_phase2': {
        'config': 'configs/hymotion_m2m/hymotion_m2m_caption_local_phase2.py',
        'work_dir': 'work_dirs/hymotion_m2m_v2_caption_local_phase2',
        'has_caption': True,
        'rotation_space': 'local',
    },
    'caption_global_phase2': {
        'config': 'configs/hymotion_m2m/hymotion_m2m_caption_global_phase2.py',
        'work_dir': 'work_dirs/hymotion_m2m_v2_caption_global_phase2',
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

def _aggregate_per_sample(per_sample: List[Dict]) -> Dict[str, Dict[str, float]]:
    """Aggregate per-sample metrics into mean/std/median/min/max."""
    metric_keys = set()
    for s in per_sample:
        metric_keys.update(s.get('metrics', {}).keys())

    agg = {}
    for key in sorted(metric_keys):
        vals = [s['metrics'][key] for s in per_sample
                if key in s.get('metrics', {})
                and isinstance(s['metrics'][key], (int, float))]
        if vals:
            arr = np.array(vals)
            agg[key] = {
                'mean': round(float(np.mean(arr)), 6),
                'std': round(float(np.std(arr)), 6),
                'median': round(float(np.median(arr)), 6),
                'min': round(float(np.min(arr)), 6),
                'max': round(float(np.max(arr)), 6),
            }
    return agg


def run_model_on_gpu(
    model_name: str,
    gpu_id: int,
    prompts: List[Dict],
    output_dir: str,
    num_steps: int,
    cfg_scale: float,
    result_queue: Queue,
    ckpt_path_override: Optional[str] = None,
    cfg_list: Optional[List[float]] = None,
    chunk_id: int = 0,
    chunk_total: int = 1,
):
    """Worker function: load model on specified GPU, run all prompts.

    Args:
        ckpt_path_override: if non-None, load this ckpt instead of
            find_latest_checkpoint(work_dir).  Used by the cfg-sweep
            ablation to pin a pre-patch unpatched ckpt while a separate
            training task continues writing patched ckpts to the same
            work_dir.
        cfg_list: when non-empty, run the same prompt slice through every
            cfg sequentially under one model load.  Outputs go to
            ``<output_dir>/<model>/cfg{cfg:g}/`` (npz + per-chunk json).
            ``cfg_scale`` is ignored in this mode.
        chunk_id, chunk_total: ``prompts`` is the worker's prompt slice
            (already chunked by the caller).  These ints disambiguate the
            partial-json filename so the main process can merge chunks
            after all workers finish.  ``chunk_total=1`` writes a full
            result.json directly (legacy single-chunk mode).
    """
    import torch
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = 'cuda:0'

    model_info = MODELS[model_name]
    base_model_dir = os.path.join(output_dir, model_name)
    os.makedirs(base_model_dir, exist_ok=True)

    cfgs_to_run = cfg_list if cfg_list else [cfg_scale]

    print(f'[GPU {gpu_id}] {model_name} chunk{chunk_id}/{chunk_total}: '
          f'will run {len(prompts)} prompts × {len(cfgs_to_run)} cfg, '
          f'loading model...')

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

        # Quality checker
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
            transl = motion_135_np[:, :3]
            rot6d = motion_135_np[:, 3:135].reshape(-1, 22, 6)
            aa = rotation_6d_to_axis_angle(torch.from_numpy(rot6d).float()).numpy()
            poses = aa.reshape(-1, 66)
            return {'poses': poses, 'trans': transl}

        cfg = Config.fromfile(model_info['config'])
        bundle = MODEL_BUNDLES.build(cfg.model.to_dict())

        # For caption models: inject text_encoder config for runtime encoding
        if model_info['has_caption'] and bundle._text_encoder_cfg is None:
            bundle._text_encoder_cfg = {
                'llm_type': 'qwen3_embedding',
                'max_length_llm': 512,
                'sentence_emb_type': 'clipl',
                'max_length_sentence_emb': 77,
            }
            print(f'[GPU {gpu_id}] {model_name}: injected runtime text_encoder config')

        if ckpt_path_override is not None:
            ckpt_path = ckpt_path_override
            if not os.path.exists(ckpt_path):
                msg = f'ckpt_path_override does not exist: {ckpt_path}'
                print(f'[GPU {gpu_id}] {model_name}: ERROR - {msg}')
                result_queue.put({'model': model_name, 'error': msg})
                return
        else:
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

        bone_offsets = None
        if os.path.exists(BONE_OFFSETS_PATH):
            bone_offsets = torch.load(BONE_OFFSETS_PATH, map_location='cpu')

        rotation_space = model_info['rotation_space']

        # ----- nested helper: run one cfg over the worker's prompt slice -----
        def _run_one_cfg(cfg_val: float, npz_dir: str) -> Tuple[List[Dict], float]:
            """Returns (per_sample_list, elapsed_seconds)."""
            pipeline = HyMotionM2MPipeline(
                bundle=bundle,
                num_steps=num_steps,
                text_guidance_scale=cfg_val if model_info['has_caption'] else 1.0,
                replacement_guidance='none',
            )

            print(f'[GPU {gpu_id}] {model_name} chunk{chunk_id}/{chunk_total} '
                  f'cfg={cfg_val:g}: running {len(prompts)} prompts...')

            per_sample = []
            t_start = time.time()

            for i, prompt in enumerate(prompts):
                text = prompt['text']
                T = min(prompt['frames'], 360)
                D = 198
                T_PAD = 360

                src_mask = torch.zeros(1, T_PAD, D, device=device)
                src_mask[:, :T, :] = 1.0
                src_motion = torch.zeros(1, T_PAD, D, device=device)

                batch = {
                    'src_motion': src_motion,
                    'src_mask': src_mask,
                    'src_length': [T],
                    'tgt_length': [T],
                }

                if model_info['has_caption'] and text:
                    try:
                        text_out = bundle.encode_text([text])
                        batch['text_vec_raw'] = text_out['text_vec_raw'].to(device)
                        batch['text_ctxt_raw'] = text_out['text_ctxt_raw'].to(device)
                        batch['text_ctxt_raw_length'] = text_out['text_ctxt_raw_length'].to(device)
                    except Exception as e:
                        print(f'  WARNING: text encoding failed: {e}')

                t0 = time.time()
                with torch.no_grad():
                    output = pipeline(batch)
                elapsed = time.time() - t0

                sampled = output['latent']
                output_denorm = bundle.denormalize_motion(sampled)[0].cpu()
                output_denorm = output_denorm[:T]
                output_135 = output_denorm[:, :135].numpy()

                rot6d_part = output_135[:, 3:135].reshape(T, 22, 6)
                rot6d_norms = np.linalg.norm(rot6d_part, axis=-1)
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

                metrics = {
                    'jitter_135': compute_jitter_135(output_135),
                    'inference_time': round(elapsed, 2),
                }
                metrics.update(sanity)

                pos_np = None
                if bone_offsets is not None:
                    try:
                        output_135_t = torch.from_numpy(output_135).float()
                        world_pos, _, _, _ = motion135_to_fk(
                            output_135_t, bone_offsets, rotation_space=rotation_space)
                        pos_np = world_pos.numpy()

                        metrics['jitter_pos'] = compute_jitter_positions(pos_np, fps=30.0)
                        bl = compute_bone_length_cv(pos_np)
                        metrics.update(bl)
                        foot = compute_foot_ground_metrics(pos_np, fps=30.0)
                        metrics.update(foot)

                        metrics['pos_range_y'] = float(pos_np[:, :, 1].max() - pos_np[:, :, 1].min())
                        metrics['pos_mean_y'] = float(pos_np[:, :, 1].mean())
                        metrics['pelvis_height_mean'] = float(pos_np[:, 0, 1].mean())

                        npz_path = os.path.join(npz_dir, f'{prompt["id"]}.npz')
                        np.savez_compressed(
                            npz_path,
                            motion_135=output_135,
                            positions=pos_np,
                            translation=transl,
                        )

                        joint_vel = np.diff(pos_np, axis=0) * 30.0
                        joint_acc = np.diff(joint_vel, axis=0) * 30.0
                        vel_mag = np.linalg.norm(joint_vel, axis=-1)
                        acc_mag = np.linalg.norm(joint_acc, axis=-1)
                        metrics['avg_velocity'] = float(vel_mag.mean())
                        metrics['max_velocity'] = float(vel_mag.max())
                        metrics['avg_acceleration'] = float(acc_mag.mean())
                        metrics['max_acceleration'] = float(acc_mag.max())
                        metrics['pelvis_trans_jerk'] = float(
                            np.linalg.norm(np.diff(pos_np[:, 0, :], n=3, axis=0) * (30.0**3), axis=-1).mean()
                        ) if T > 3 else 0.0
                        if pos_np.shape[1] > 15:
                            head_vel = np.linalg.norm(np.diff(pos_np[:, 15, :], axis=0), axis=-1) * 30.0
                            metrics['head_jitter_ratio'] = float((head_vel > 0.5).mean())
                        if pos_np.shape[1] >= 22:
                            spine_center = (pos_np[:, 3, :] + pos_np[:, 6, :] + pos_np[:, 9, :]) / 3
                            l_wrist_dist = np.linalg.norm(pos_np[:, 20, :] - spine_center, axis=-1)
                            r_wrist_dist = np.linalg.norm(pos_np[:, 21, :] - spine_center, axis=-1)
                            torso_radius = 0.15
                            metrics['arm_penetration_ratio'] = float(
                                ((l_wrist_dist < torso_radius) | (r_wrist_dist < torso_radius)).mean()
                            )
                    except Exception as e:
                        metrics['fk_error'] = str(e)

                try:
                    if quality_checker is not None:
                        checker_data = _motion135_to_checker_input(output_135, rotation_space)
                        qc_result = quality_checker.check(checker_data)
                        metrics['qc_pass'] = 1 if qc_result.is_valid else 0
                        metrics['qc_num_failed'] = len(qc_result.failed_checks)
                        metrics['qc_num_borderline'] = len(qc_result.borderline_checks)
                        metrics['qc_failed_checks'] = qc_result.failed_checks
                        for checker_name, cr in qc_result.all_results.items():
                            metrics[f'qc_{checker_name}'] = 1 if cr['is_valid'] else 0
                except Exception as e:
                    metrics['qc_error'] = str(e)

                if output_denorm.shape[-1] >= 198:
                    pos_channel = output_denorm[:, 135:198].numpy()
                    metrics['pos_channel_range'] = [float(pos_channel.min()), float(pos_channel.max())]
                    metrics['pos_channel_mean'] = float(pos_channel.mean())

                    if bone_offsets is not None and pos_np is not None and 'fk_error' not in metrics:
                        try:
                            fk_joint = pos_np[:, 1:, :]
                            pelvis_xz = pos_np[:, 0:1, [0, 2]]
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
                    print(f'[GPU {gpu_id}] {model_name} chunk{chunk_id} cfg={cfg_val:g}: '
                          f'{i+1}/{len(prompts)} ({speed*60:.1f} samples/min)')

            return per_sample, time.time() - t_start

        # ----- iterate over cfgs (model is loaded once) -----
        for cfg_val in cfgs_to_run:
            if cfg_list:
                cfg_subdir = f'cfg{cfg_val:g}'
                cfg_md = os.path.join(base_model_dir, cfg_subdir)
            else:
                cfg_md = base_model_dir
            cfg_nd = os.path.join(cfg_md, 'npz')
            os.makedirs(cfg_nd, exist_ok=True)

            per_sample, elapsed = _run_one_cfg(cfg_val, cfg_nd)

            if chunk_total > 1:
                # Multi-chunk: write per-chunk partial; main will merge.
                partial_path = os.path.join(
                    cfg_md,
                    f'per_sample_chunk{chunk_id:02d}_of_{chunk_total:02d}.json',
                )
                with open(partial_path, 'w') as f:
                    json.dump({
                        'model': model_name,
                        'checkpoint': ckpt_path,
                        'rotation_space': rotation_space,
                        'has_caption': model_info['has_caption'],
                        'cfg_scale': cfg_val if model_info['has_caption'] else 1.0,
                        'num_steps': num_steps,
                        'chunk_id': chunk_id,
                        'chunk_total': chunk_total,
                        'chunk_size': len(prompts),
                        'chunk_elapsed_sec': round(elapsed, 1),
                        'per_sample': per_sample,
                    }, f, indent=2,
                       default=lambda x: float(x) if isinstance(x, np.floating) else x)
                print(f'[GPU {gpu_id}] {model_name} chunk{chunk_id} cfg={cfg_val:g}: '
                      f'wrote {partial_path} ({len(per_sample)} samples in {elapsed:.0f}s)')
            else:
                # Single-chunk: produce the final result.json directly.
                agg = _aggregate_per_sample(per_sample)
                result = {
                    'model': model_name,
                    'checkpoint': ckpt_path,
                    'rotation_space': rotation_space,
                    'has_caption': model_info['has_caption'],
                    'num_prompts': len(prompts),
                    'num_steps': num_steps,
                    'cfg_scale': cfg_val if model_info['has_caption'] else 1.0,
                    'total_time_sec': round(elapsed, 1),
                    'speed_samples_per_min': round(len(prompts) / elapsed * 60, 1),
                    'aggregated': agg,
                    'per_sample': per_sample,
                }
                result_path = os.path.join(cfg_md, 'result.json')
                with open(result_path, 'w') as f:
                    json.dump(result, f, indent=2,
                              default=lambda x: float(x) if isinstance(x, np.floating) else x)
                print(f'[GPU {gpu_id}] {model_name} cfg={cfg_val:g}: '
                      f'DONE in {elapsed:.0f}s. Saved {result_path}')

            result_queue.put({
                'model': model_name,
                'cfg_scale': cfg_val if model_info['has_caption'] else 1.0,
                'chunk_id': chunk_id,
                'chunk_total': chunk_total,
                'num_samples': len(per_sample),
                'elapsed_sec': round(elapsed, 1),
            })

    except Exception as e:
        msg = f'{type(e).__name__}: {e}\n{traceback.format_exc()}'
        print(f'[GPU {gpu_id}] {model_name}: FATAL ERROR\n{msg}')
        result_queue.put({
            'model': model_name,
            'chunk_id': chunk_id,
            'error': msg,
        })


# ============================================================================
# Cross-chunk merge (used only in multi-chunk mode)
# ============================================================================

def merge_chunks(model_dir: str, model_name: str, model_info: Dict,
                 cfg_val: float, num_steps: int, ckpt_path: str) -> str:
    """Merge per_sample_chunk*.json under model_dir into a single result.json."""
    chunk_paths = sorted(
        p for p in os.listdir(model_dir)
        if p.startswith('per_sample_chunk') and p.endswith('.json')
    )
    if not chunk_paths:
        return ''
    all_samples = []
    total_elapsed = 0.0
    for cp in chunk_paths:
        with open(os.path.join(model_dir, cp)) as f:
            data = json.load(f)
        all_samples.extend(data.get('per_sample', []))
        total_elapsed += data.get('chunk_elapsed_sec', 0.0)

    # Sort by prompt_id for deterministic ordering.
    all_samples.sort(key=lambda s: s.get('prompt_id', ''))

    agg = _aggregate_per_sample(all_samples)
    result = {
        'model': model_name,
        'checkpoint': ckpt_path,
        'rotation_space': model_info['rotation_space'],
        'has_caption': model_info['has_caption'],
        'num_prompts': len(all_samples),
        'num_steps': num_steps,
        'cfg_scale': cfg_val if model_info['has_caption'] else 1.0,
        'total_time_sec': round(total_elapsed, 1),
        'wall_time_sec_max_chunk': round(total_elapsed / max(len(chunk_paths), 1), 1),
        'speed_samples_per_min': round(len(all_samples) / max(total_elapsed, 1) * 60, 1) if total_elapsed > 0 else 0.0,
        'num_chunks_merged': len(chunk_paths),
        'aggregated': agg,
        'per_sample': all_samples,
    }
    out_path = os.path.join(model_dir, 'result.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2,
                  default=lambda x: float(x) if isinstance(x, np.floating) else x)
    return out_path


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='T2M eval on Yiran subset')
    parser.add_argument('--models', nargs='+', choices=list(MODELS.keys()),
                        default=list(MODELS.keys()))
    parser.add_argument('--gpus', nargs='+', type=int, default=None,
                        help='GPU IDs to use (default: 0..N-1).')
    parser.add_argument('--num-steps', type=int, default=50)
    parser.add_argument('--cfg-scale', type=float, default=5.0,
                        help='CFG scale for caption models (used when --cfg-sweep is unset).')
    parser.add_argument('--cfg-sweep', nargs='+', type=float, default=None,
                        help='Run multiple cfgs sequentially under one model load. '
                             'Each worker iterates this list inside its prompt chunk. '
                             'Used for the CFG ablation that probes whether the null=0 '
                             'uncond branch is the caption-follow bottleneck — if cfg=1.0 '
                             '(skips uncond branch entirely) outperforms cfg>1, that is '
                             'strong evidence.')
    parser.add_argument('--ckpt-path', type=str, default=None,
                        help='Override path to a single ckpt (file or directory).  '
                             'When set, applies to ALL --models, so this is normally '
                             'only meaningful with a single --models entry.')
    parser.add_argument('--prompt-chunks', type=int, default=None,
                        help='Split prompts into N chunks across GPUs.  When unset, '
                             'defaults to len(--gpus) when both --gpus and --cfg-sweep '
                             'are provided (i.e. fully utilise all listed GPUs in the '
                             'cfg-sweep ablation).  In legacy mode (no --cfg-sweep), '
                             'defaults to 1 (one worker per --models entry).')
    parser.add_argument('--output-suffix', type=str, default='',
                        help='Append to --output-dir base name (e.g. "_cfg_ablation_2860").')
    parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR)
    parser.add_argument('--prompt-file', type=str, default=PROMPT_FILE)
    args = parser.parse_args()

    prompts = load_prompts(args.prompt_file)
    print(f'Loaded {len(prompts)} prompts from {args.prompt_file}')

    if args.output_suffix:
        args.output_dir = args.output_dir.rstrip('/') + args.output_suffix
    os.makedirs(args.output_dir, exist_ok=True)

    sweep_mode = args.cfg_sweep is not None and len(args.cfg_sweep) > 0

    # Decide per-mode chunking + GPU assignment.
    if sweep_mode:
        # CFG-sweep: one worker = one prompt chunk × all cfgs.
        if args.gpus is None:
            args.gpus = list(range(8))  # assume 8 GPUs available
        n_chunks = args.prompt_chunks if args.prompt_chunks else len(args.gpus)
        if n_chunks > len(prompts):
            n_chunks = len(prompts)
        # Split prompts as evenly as possible.
        chunk_indices: List[List[int]] = [[] for _ in range(n_chunks)]
        for i in range(len(prompts)):
            chunk_indices[i % n_chunks].append(i)
        # Map each chunk to a GPU (round-robin if more chunks than gpus).
        gpu_ids = [args.gpus[i % len(args.gpus)] for i in range(n_chunks)]
        if args.ckpt_path is not None and len(args.models) > 1:
            print(f'WARNING: --ckpt-path set but {len(args.models)} models requested; '
                  f'all will load the same ckpt.')

        print(f'CFG-sweep mode: {len(args.models)} model(s) × {n_chunks} chunk(s) × '
              f'{len(args.cfg_sweep)} cfg(s) = '
              f'{len(args.models) * n_chunks} workers, '
              f'each iterates {len(args.cfg_sweep)} cfgs internally.')
        print(f'  prompts/chunk ≈ {len(prompts) // n_chunks}')
        print(f'  GPU assignment:')
        for i, (chunk, g) in enumerate(zip(chunk_indices, gpu_ids)):
            print(f'    chunk {i:02d} ({len(chunk)} prompts) -> GPU {g}')
    else:
        # Legacy: one worker per model, one cfg.
        if args.gpus is None:
            args.gpus = list(range(len(args.models)))
        if len(args.gpus) < len(args.models):
            print(f'WARNING: {len(args.models)} models but only {len(args.gpus)} GPUs.')
            while len(args.gpus) < len(args.models):
                args.gpus.append(args.gpus[0])
        n_chunks = 1
        chunk_indices = [list(range(len(prompts)))]
        gpu_ids = args.gpus[:len(args.models)]
        print(f'Legacy mode: {len(args.models)} model(s), 1 worker each.')
        print(f'  Model -> GPU:')
        for m, g in zip(args.models, gpu_ids):
            print(f'    {m} -> GPU {g}')

    # Launch workers.
    result_queue = Queue()
    processes = []

    if sweep_mode:
        for m in args.models:
            for chunk_id, (chunk, gpu) in enumerate(zip(chunk_indices, gpu_ids)):
                chunk_prompts = [prompts[i] for i in chunk]
                p = Process(
                    target=run_model_on_gpu,
                    args=(m, gpu, chunk_prompts, args.output_dir,
                          args.num_steps, args.cfg_scale, result_queue,
                          args.ckpt_path, args.cfg_sweep, chunk_id, n_chunks),
                )
                p.start()
                processes.append(p)
                print(f'Started {m} chunk{chunk_id:02d} on GPU {gpu} (PID {p.pid})')
    else:
        for m, gpu in zip(args.models, gpu_ids):
            p = Process(
                target=run_model_on_gpu,
                args=(m, gpu, prompts, args.output_dir,
                      args.num_steps, args.cfg_scale, result_queue,
                      args.ckpt_path, None, 0, 1),
            )
            p.start()
            processes.append(p)
            print(f'Started {m} on GPU {gpu} (PID {p.pid})')

    for p in processes:
        p.join()

    # Collect partial summaries.
    partial = []
    while not result_queue.empty():
        partial.append(result_queue.get())

    print(f'\n{"=" * 70}')
    print('ALL WORKERS DONE')
    print(f'{"=" * 70}')

    # Multi-chunk: merge per-cfg per-chunk partials → single result.json per cfg.
    if sweep_mode and n_chunks > 1:
        print('Merging per-chunk partials into per-cfg result.json ...')
        for m in args.models:
            model_info = MODELS[m]
            # Find ckpt_path (every worker logged it; reuse the override or
            # rely on find_latest_checkpoint's value via partial data — we
            # don't have it in queue, but per-chunk partials store it).
            for cfg_val in args.cfg_sweep:
                cfg_dir = os.path.join(args.output_dir, m, f'cfg{cfg_val:g}')
                if not os.path.isdir(cfg_dir):
                    continue
                # Pull ckpt_path from any chunk file.
                ckpt_path = ''
                for fn in os.listdir(cfg_dir):
                    if fn.startswith('per_sample_chunk'):
                        with open(os.path.join(cfg_dir, fn)) as f:
                            ckpt_path = json.load(f).get('checkpoint', '')
                        break
                merged = merge_chunks(cfg_dir, m, model_info, cfg_val,
                                      args.num_steps, ckpt_path)
                if merged:
                    print(f'  {m} cfg={cfg_val:g}: merged -> {merged}')

    # Comparative summary.
    print(f'\n{"Model/CFG":<32} {"chunks":>8} {"samples":>10} {"elapsed":>10}')
    print('-' * 64)
    by_key: Dict[Tuple[str, float], Dict] = {}
    for r in partial:
        if 'error' in r:
            print(f'{r["model"]:<32} ERROR: {r["error"][:40]}')
            continue
        key = (r['model'], r.get('cfg_scale', 0.0))
        d = by_key.setdefault(key, {'chunks': 0, 'samples': 0, 'elapsed_max': 0.0})
        d['chunks'] += 1
        d['samples'] += r.get('num_samples', 0)
        d['elapsed_max'] = max(d['elapsed_max'], r.get('elapsed_sec', 0.0))

    for (m, cfg_val), d in sorted(by_key.items()):
        label = f'{m}@cfg{cfg_val:g}'
        print(f'{label:<32} {d["chunks"]:>8} {d["samples"]:>10} '
              f'{d["elapsed_max"]:>10.0f}s (max chunk wall)')

    print(f'\nResults saved to {args.output_dir}/')


if __name__ == '__main__':
    main()
