#!/usr/bin/env python3
"""Test M2M repair inference on low-quality motion data.

Loads low_quality.json, picks a subset, runs M2M completion (full mask = refine mode),
and saves results for visual comparison.

Usage:
    python tools/test_m2m_repair.py --model uncond_flow --max-samples 20 --num-steps 50
    python tools/test_m2m_repair.py --model caption_flow --max-samples 20 --num-steps 50
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Model configs
MODELS = {
    'uncond_flow': {
        'config': 'configs/hymotion_m2m/hymotion_m2m_completion_uncond_fm_046b.py',
        'checkpoint': 'work_dirs/hymotion_m2m_completion_uncond_fm_046b/checkpoint-epoch_262',
    },
    'caption_flow': {
        'config': 'configs/hymotion_m2m/hymotion_m2m_completion_caption_fm_046b.py',
        'checkpoint': 'work_dirs/hymotion_m2m_completion_caption_fm_046b/checkpoint-epoch_259',
    },
}


def load_motion_135d(npz_path: str) -> torch.Tensor:
    """Load npz and convert to 135-dim motion (abs transl + rot6d)."""
    from hftrainer.datasets.motion.motionhub.transforms.load_smplx import (
        process_transl, process_smplx_pose,
    )
    data = np.load(npz_path, allow_pickle=True)

    # Handle different key names
    trans_key = 'trans' if 'trans' in data else 'transl'
    abs_trans = data[trans_key].astype(np.float32)

    poses_key = 'poses' if 'poses' in data else 'body_pose'
    poses = data[poses_key].astype(np.float32)

    transl = process_transl(abs_trans, 'abs')  # [T, 3]
    pose = process_smplx_pose(poses, 'rotation_6d', 'smpl_22')  # [T, 132]
    motion = np.concatenate([transl, pose], axis=-1)  # [T, 135]
    return torch.from_numpy(motion).float()


def motion_135d_to_npz(motion_135d: np.ndarray, fps: int = 30) -> dict:
    """Convert 135-dim motion back to SMPL npz format for visualization."""
    from hftrainer.models.motion.hymotion_m2m.network.geometry import (
        rot6d_to_rotation_matrix,
    )
    from hftrainer.models.motion.components.utils.geometry.rotation_convert import (
        matrix_to_axis_angle,
    )

    T = motion_135d.shape[0]
    transl = motion_135d[:, :3]  # [T, 3]

    # rot6d is in row-major convention: [R00,R01, R10,R11, R20,R21]
    rot6d = motion_135d[:, 3:].reshape(T, 22, 6)

    # rot6d_to_rotation_matrix expects row-major rot6d (the same format as
    # process_smplx_pose outputs). No reorder needed.
    rot6d_flat = torch.from_numpy(rot6d.reshape(-1, 6)).float()
    rotmat = rot6d_to_rotation_matrix(rot6d_flat)  # [T*22, 3, 3]
    aa = matrix_to_axis_angle(rotmat.numpy())  # [T*22, 3]
    aa = np.asarray(aa).reshape(T, 22, 3)

    # Pad to 55 joints (SMPL-X)
    poses_55 = np.zeros((T, 55, 3), dtype=np.float32)
    poses_55[:, :22, :] = aa
    poses_55 = poses_55.reshape(T, -1)  # [T, 165]

    return {
        'trans': transl,
        'poses': poses_55,
        'mocap_framerate': np.array(fps),
        'gender': 'neutral',
    }


def run_repair(
    model_name: str,
    low_quality_path: str,
    data_dir: str,
    max_samples: int,
    num_steps: int,
    output_dir: str,
    device: str = 'cuda',
):
    from mmengine.config import Config
    from hftrainer.registry import MODEL_BUNDLES
    from hftrainer.utils.checkpoint_utils import load_checkpoint
    from hftrainer.pipelines.motion.hymotion_m2m_pipeline import HyMotionM2MPipeline

    model_info = MODELS[model_name]

    # Load model
    print(f'Loading model: {model_name}')
    cfg = Config.fromfile(model_info['config'])
    bundle = MODEL_BUNDLES.build(cfg.model.to_dict())
    sd = load_checkpoint(model_info['checkpoint'], map_location='cpu')
    bundle.load_state_dict_selective(sd)
    del sd
    bundle.eval()
    bundle = bundle.to(device)
    print(f'  Model loaded. mean shape={bundle.mean.shape}')

    pipeline = HyMotionM2MPipeline(bundle=bundle, num_steps=num_steps)

    # Load low quality list
    with open(low_quality_path) as f:
        lq_data = json.load(f)
    lq_items = lq_data.get('items', [])
    lq_data_dir = lq_data.get('data_dir', data_dir)
    print(f'Low quality items: {len(lq_items)}')

    os.makedirs(output_dir, exist_ok=True)

    results = []
    count = 0
    for item in lq_items:
        if count >= max_samples:
            break

        rel_path = item.get('path', '')
        reasons = item.get('reasons', [])
        npz_path = os.path.join(lq_data_dir, rel_path)

        if not os.path.exists(npz_path):
            continue

        try:
            motion = load_motion_135d(npz_path)
        except Exception as e:
            print(f'  Skip {rel_path}: {e}')
            continue

        T, D = motion.shape
        if T < 10:
            continue

        count += 1
        print(f'[{count}/{max_samples}] {rel_path} (T={T}, reasons={reasons})')

        # Normalize
        motion_norm = bundle.normalize_motion(motion.unsqueeze(0).to(device))  # [1, T, 135]

        # Full mask = refine mode (model regenerates everything based on reactive)
        src_mask = torch.ones(1, T, D, device=device)  # mask everything

        # Build batch
        batch = {
            'src_motion': motion_norm,
            'src_mask': src_mask,
            'src_length': [T],
            'tgt_length': [T],
        }

        t0 = time.time()
        with torch.no_grad():
            output = pipeline(batch)
        elapsed = time.time() - t0

        # Denormalize output
        sampled = output['latent']  # [1, T, 135] in normalized space
        repaired_norm = sampled[0].cpu()
        repaired_raw = bundle.denormalize_motion(repaired_norm.unsqueeze(0).to(device))[0].cpu().numpy()

        # Also get original raw for comparison
        original_raw = bundle.denormalize_motion(motion_norm)[0].cpu().numpy()

        # Save
        sample_name = rel_path.replace('/', '_').replace('.npz', '')
        out_path = os.path.join(output_dir, f'{sample_name}_repaired.npz')

        # Get FPS from original file
        try:
            orig_data = np.load(npz_path, allow_pickle=True)
            fps = int(orig_data.get('mocap_framerate', 30))
        except Exception:
            fps = 30

        npz_dict = motion_135d_to_npz(repaired_raw, fps=fps)
        np.savez_compressed(out_path, **npz_dict)

        # Also save original for side-by-side comparison
        orig_out_path = os.path.join(output_dir, f'{sample_name}_original.npz')
        orig_npz = motion_135d_to_npz(original_raw, fps=fps)
        np.savez_compressed(orig_out_path, **orig_npz)

        # Compute simple quality metrics
        diff = np.abs(repaired_raw - original_raw)
        jitter_orig = np.mean(np.abs(original_raw[2:] - 2 * original_raw[1:-1] + original_raw[:-2]))
        jitter_repair = np.mean(np.abs(repaired_raw[2:] - 2 * repaired_raw[1:-1] + repaired_raw[:-2]))

        results.append({
            'path': rel_path,
            'reasons': reasons,
            'frames': T,
            'time_sec': round(elapsed, 2),
            'mean_diff': round(float(diff.mean()), 4),
            'jitter_original': round(float(jitter_orig), 4),
            'jitter_repaired': round(float(jitter_repair), 4),
            'output': out_path,
        })

        print(f'  Done in {elapsed:.1f}s, jitter: {jitter_orig:.4f} → {jitter_repair:.4f}')

    # Save summary
    summary_path = os.path.join(output_dir, 'repair_summary.json')
    with open(summary_path, 'w') as f:
        json.dump({
            'model': model_name,
            'num_steps': num_steps,
            'total_samples': count,
            'results': results,
        }, f, indent=2)
    print(f'\nSummary saved to {summary_path}')

    # Print aggregate stats
    if results:
        jitter_orig_avg = np.mean([r['jitter_original'] for r in results])
        jitter_rep_avg = np.mean([r['jitter_repaired'] for r in results])
        time_avg = np.mean([r['time_sec'] for r in results])
        print(f'\n=== Aggregate ===')
        print(f'  Avg jitter: {jitter_orig_avg:.4f} (original) → {jitter_rep_avg:.4f} (repaired)')
        print(f'  Avg time per sample: {time_avg:.1f}s')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=list(MODELS.keys()), default='uncond_flow')
    parser.add_argument('--max-samples', type=int, default=20)
    parser.add_argument('--num-steps', type=int, default=50)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--low-quality-json', type=str,
                        default='data/hymotion_m2m_refine_data/data_quality_list/low_quality.json')
    parser.add_argument('--data-dir', type=str, default='data/hymotion_data')
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f'work_dirs/m2m_repair_test/{args.model}'

    run_repair(
        model_name=args.model,
        low_quality_path=args.low_quality_json,
        data_dir=args.data_dir,
        max_samples=args.max_samples,
        num_steps=args.num_steps,
        output_dir=args.output_dir,
        device=args.device,
    )


if __name__ == '__main__':
    main()
