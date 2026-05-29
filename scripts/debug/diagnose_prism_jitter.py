#!/usr/bin/env python3
"""Diagnose PRISM inference jitter by:
1. Comparing temporal velocity of generated vs GT motions (from existing NPZ)
2. Running single-sample inference with different guidance scales
3. Comparing spectral vs base model

Usage (on debug machine):
    python3 scripts/debug/diagnose_prism_jitter.py \
        --config configs/prism/prism_1b_tp2m_multiframe_kt_spectral_unified.py \
        --checkpoint work_dirs/prism_1b_tp2m_multiframe_kt_spectral/checkpoint-epoch_0 \
        --eval-dir work_dirs/prism_1b_tp2m_multiframe_kt_spectral/eval_hml3d_rewritten \
        --gpu 1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
HF_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(HF_ROOT))


def compute_body_pose_velocity(body_pose: np.ndarray) -> float:
    """Compute mean frame-to-frame L2 velocity of body_pose (T, D)."""
    if body_pose.ndim == 3:
        body_pose = body_pose[0]  # (B, T, D) -> (T, D)
    diffs = np.diff(body_pose, axis=0)  # (T-1, D)
    per_frame_vel = np.linalg.norm(diffs, axis=1)  # (T-1,)
    return float(per_frame_vel.mean())


def compute_transl_velocity(transl: np.ndarray) -> float:
    """Compute mean frame-to-frame velocity of translation (T, 3)."""
    if transl.ndim == 3:
        transl = transl[0]
    diffs = np.diff(transl, axis=0)
    per_frame_vel = np.linalg.norm(diffs, axis=1)
    return float(per_frame_vel.mean())


def analyze_existing_npz(eval_dir: str, gt_dir: str, n_samples: int = 20):
    """Load existing generated NPZ and compare with GT motions."""
    eval_path = Path(eval_dir)
    gt_path = Path(gt_dir) if gt_dir else None

    gen_files = sorted(eval_path.glob('*.npz'))[:n_samples]
    if not gen_files:
        print(f"[WARN] No NPZ files found in {eval_dir}")
        return

    print(f"\n{'='*70}")
    print(f"Analyzing {len(gen_files)} generated samples from: {eval_dir}")
    print(f"{'='*70}")

    gen_vels = []
    gen_transl_vels = []
    gt_vels = []
    gt_transl_vels = []

    for npz_file in gen_files:
        data = np.load(npz_file, allow_pickle=True)
        motion_id = npz_file.stem

        # Generated motion stats
        if 'body_pose' in data:
            bp = data['body_pose']
            vel = compute_body_pose_velocity(bp)
            gen_vels.append(vel)

        if 'transl' in data:
            tr = data['transl']
            tv = compute_transl_velocity(tr)
            gen_transl_vels.append(tv)

        # Try to load GT
        if gt_path:
            gt_file = gt_path / f'{motion_id}.npz'
            if gt_file.exists():
                gt_data = np.load(gt_file, allow_pickle=True)
                if 'body_pose' in gt_data:
                    gt_vels.append(compute_body_pose_velocity(gt_data['body_pose']))
                if 'transl' in gt_data:
                    gt_transl_vels.append(compute_transl_velocity(gt_data['transl']))

    print(f"\n--- Body Pose Velocity (frame-to-frame L2 norm) ---")
    print(f"  Generated: mean={np.mean(gen_vels):.5f}, std={np.std(gen_vels):.5f}, "
          f"min={np.min(gen_vels):.5f}, max={np.max(gen_vels):.5f}")
    if gt_vels:
        print(f"  GT:        mean={np.mean(gt_vels):.5f}, std={np.std(gt_vels):.5f}, "
              f"min={np.min(gt_vels):.5f}, max={np.max(gt_vels):.5f}")
        print(f"  Ratio (gen/gt): {np.mean(gen_vels)/np.mean(gt_vels):.2f}x")

    print(f"\n--- Translation Velocity ---")
    print(f"  Generated: mean={np.mean(gen_transl_vels):.5f}, std={np.std(gen_transl_vels):.5f}")
    if gt_transl_vels:
        print(f"  GT:        mean={np.mean(gt_transl_vels):.5f}, std={np.std(gt_transl_vels):.5f}")
        print(f"  Ratio (gen/gt): {np.mean(gen_transl_vels)/np.mean(gt_transl_vels):.2f}x")

    # Per-sample breakdown (first 5)
    print(f"\n--- Per-sample (first 5) ---")
    for i, npz_file in enumerate(gen_files[:5]):
        data = np.load(npz_file, allow_pickle=True)
        bp_vel = compute_body_pose_velocity(data['body_pose']) if 'body_pose' in data else -1
        tr_vel = compute_transl_velocity(data['transl']) if 'transl' in data else -1
        n_frames = data['body_pose'].shape[-2] if 'body_pose' in data else 0
        print(f"  {npz_file.stem}: frames={n_frames}, bp_vel={bp_vel:.5f}, transl_vel={tr_vel:.5f}")
        # Print shape info
        for key in data.files:
            print(f"    {key}: {data[key].shape}")


def run_single_inference(
    config_path: str,
    checkpoint_dir: str,
    caption: str,
    num_frames: int,
    guidance_scales: list,
    device: torch.device,
):
    """Run inference for one caption with different guidance scales."""
    from mmengine.config import Config
    from hftrainer.registry import MODEL_BUNDLES
    from hftrainer.pipelines.motion.prism_pipeline import PrismPipeline

    print(f"\n{'='*70}")
    print(f"Single-sample inference test")
    print(f"  Config: {config_path}")
    print(f"  Checkpoint: {checkpoint_dir}")
    print(f"  Caption: '{caption}'")
    print(f"  Num frames: {num_frames}")
    print(f"{'='*70}")

    cfg = Config.fromfile(config_path)
    bundle = MODEL_BUNDLES.build(cfg.model)

    ckpt_path = os.path.join(checkpoint_dir, 'model.pt')
    if os.path.isfile(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        bundle.load_state_dict_selective(state_dict, strict=False)
    bundle = bundle.eval().to(device)

    pipeline = PrismPipeline(bundle=bundle)

    for gs in guidance_scales:
        print(f"\n  --- guidance_scale={gs} ---")
        torch.manual_seed(42)
        with torch.no_grad():
            result = pipeline(
                prompts=caption,
                num_frames_per_segment=num_frames,
                num_inference_steps=50,
                guidance_scale=gs,
            )

        # Analyze result
        if isinstance(result, dict):
            for key, val in result.items():
                if isinstance(val, (np.ndarray, torch.Tensor)):
                    arr = val if isinstance(val, np.ndarray) else val.cpu().numpy()
                    print(f"    {key}: shape={arr.shape}, range=[{arr.min():.4f}, {arr.max():.4f}]")
                    if key == 'body_pose':
                        vel = compute_body_pose_velocity(arr)
                        print(f"      -> body_pose velocity: {vel:.5f}")
                    elif key == 'transl':
                        vel = compute_transl_velocity(arr)
                        print(f"      -> transl velocity: {vel:.5f}")
        else:
            print(f"    Result type: {type(result)}")

    return pipeline, bundle


def test_vae_roundtrip(bundle, device):
    """Test VAE encode-decode roundtrip to verify precision."""
    print(f"\n{'='*70}")
    print(f"VAE Roundtrip Test")
    print(f"{'='*70}")

    # Create synthetic motion-like data (batch=1, T=64, 138-dim)
    # First normalize it as if it were real motion
    T = 64
    fake_motion = torch.randn(1, T, 138, device=device) * 0.1

    # Encode
    with torch.no_grad():
        latent = bundle.encode_motion(fake_motion)

    print(f"  Input shape: {fake_motion.shape}")
    print(f"  Latent shape: {latent.shape}")
    print(f"  Latent range: [{latent.min():.4f}, {latent.max():.4f}]")
    print(f"  Latent std: {latent.std():.4f}")

    # Check if latent has NaN/Inf
    if torch.isnan(latent).any():
        print("  !! WARNING: NaN in latent !!")
    if torch.isinf(latent).any():
        print("  !! WARNING: Inf in latent !!")


def check_normalization_stats(bundle):
    """Verify normalization statistics."""
    print(f"\n{'='*70}")
    print(f"Normalization Statistics Check")
    print(f"{'='*70}")

    proc = bundle.smpl_pose_processor
    mean = proc.mean
    std = proc.std

    print(f"  mean shape: {mean.shape}, range=[{mean.min():.4f}, {mean.max():.4f}]")
    print(f"  std shape: {std.shape}, range=[{std.min():.4f}, {std.max():.4f}]")

    # Check for near-zero std (would cause explosion)
    near_zero = (std.abs() < 1e-6).sum().item()
    print(f"  Near-zero std dims: {near_zero}")

    # Print first few dims (translation dims typically)
    print(f"  First 6 dims (transl):")
    print(f"    mean: {mean[0, :6].tolist()}")
    print(f"    std:  {std[0, :6].tolist()}")
    print(f"  Dims 6-12 (global_orient):")
    print(f"    mean: {mean[0, 6:12].tolist()}")
    print(f"    std:  {std[0, 6:12].tolist()}")

    # VAE latent stats
    print(f"\n  VAE latents_mean shape: {bundle.latents_mean.shape}")
    print(f"  VAE latents_mean: {bundle.latents_mean.flatten()[:8].tolist()}")
    print(f"  VAE latents_std: {bundle.latents_std.flatten()[:8].tolist()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                       default='configs/prism/prism_1b_tp2m_multiframe_kt_spectral_unified.py')
    parser.add_argument('--checkpoint', type=str,
                       default='work_dirs/prism_1b_tp2m_multiframe_kt_spectral/checkpoint-epoch_0')
    parser.add_argument('--eval-dir', type=str,
                       default='work_dirs/prism_1b_tp2m_multiframe_kt_spectral/eval_hml3d_rewritten')
    parser.add_argument('--gt-dir', type=str, default='',
                       help='GT motion NPZ directory for comparison')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--skip-inference', action='store_true',
                       help='Only analyze existing NPZ, skip new inference')
    parser.add_argument('--base-config', type=str, default='',
                       help='Base model config for comparison')
    parser.add_argument('--base-checkpoint', type=str, default='',
                       help='Base model checkpoint for comparison')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda:0')

    # Phase 1: Analyze existing generated NPZ files
    if os.path.isdir(args.eval_dir):
        analyze_existing_npz(args.eval_dir, args.gt_dir, n_samples=30)

    if args.skip_inference:
        return

    # Phase 2: Single-sample inference with different guidance scales
    caption = "a person walks forward slowly"
    num_frames = 100
    guidance_scales = [1.0, 2.5, 5.0, 7.5]

    pipeline, bundle = run_single_inference(
        args.config, args.checkpoint, caption, num_frames, guidance_scales, device
    )

    # Phase 3: Check normalization stats
    check_normalization_stats(bundle)

    # Phase 4: VAE roundtrip test
    test_vae_roundtrip(bundle, device)

    # Phase 5: Compare with base model (if provided)
    if args.base_config and args.base_checkpoint:
        print(f"\n\n{'#'*70}")
        print(f"BASE MODEL COMPARISON")
        print(f"{'#'*70}")
        run_single_inference(
            args.base_config, args.base_checkpoint, caption, num_frames,
            [1.0, 5.0], device
        )


if __name__ == '__main__':
    main()
