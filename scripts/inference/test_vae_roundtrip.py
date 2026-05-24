"""Test VAE round-trip and diagnose where the deformation occurs.

This script:
1. Loads a real training motion sample
2. Normalizes and encodes it through the VAE
3. Decodes it back
4. Checks if the round-trip preserves rotation column norms
5. Then tests what happens with the transformer's denoised output

Usage:
    python3 scripts/inference/test_vae_roundtrip.py \
        --config configs/prism/prism_1b_tp2m_multiframe.py \
        --checkpoint work_dirs/prism_1b_tp2m_multiframe/checkpoint-iter_15000
"""

import argparse
import gc
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
from einops import rearrange
from mmengine.config import Config

import hftrainer  # noqa: trigger auto-imports
from hftrainer.registry import MODEL_BUNDLES
from hftrainer.utils.checkpoint_utils import load_checkpoint
from hftrainer.models.motion.components.utils.geometry.rotation_convert import (
    rotation_6d_to_axis_angle,
)
from hftrainer.models.motion.prism.gaussian_distribution import DiagonalGaussianDistributionNd
from diffusers.utils.torch_utils import randn_tensor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()


def check_rotation_quality(pred_poses_6d, label=""):
    """Check rotation 6D column norms - they should both be ~1.0 for valid rotations."""
    # pred_poses_6d shape: [..., 6] in row-major [R00, R01, R10, R11, R20, R21]
    # After swap to column-major: [R00, R10, R20, R01, R11, R21]
    poses_cm = pred_poses_6d[..., [0, 2, 4, 1, 3, 5]]
    col1 = poses_cm[..., :3]
    col2 = poses_cm[..., 3:]
    col1_norms = col1.norm(dim=-1)
    col2_norms = col2.norm(dim=-1)
    print(f"  [{label}] Col1 norm: mean={col1_norms.mean():.4f}, std={col1_norms.std():.4f}, min={col1_norms.min():.4f}, max={col1_norms.max():.4f}")
    print(f"  [{label}] Col2 norm: mean={col2_norms.mean():.4f}, std={col2_norms.std():.4f}, min={col2_norms.min():.4f}, max={col2_norms.max():.4f}")

    # Check axis angle magnitudes
    poses_cm_flat = rearrange(poses_cm, "... j d -> (...) j d") if poses_cm.dim() > 3 else poses_cm.reshape(-1, poses_cm.shape[-2], 6) if poses_cm.dim() == 3 else poses_cm.unsqueeze(0)
    # Actually just compute from the flat form
    poses_for_aa = poses_cm.reshape(-1, 6)
    aa = rotation_6d_to_axis_angle(poses_for_aa.unsqueeze(1)).squeeze(1)
    magnitudes = aa.norm(dim=-1)
    print(f"  [{label}] Axis-angle magnitude: mean={magnitudes.mean():.4f}, std={magnitudes.std():.4f}, max={magnitudes.max():.4f}")
    return col1_norms.mean().item(), col2_norms.mean().item()


def main():
    args = parse_args()
    device = torch.device(args.device)

    print("=" * 80)
    print("VAE ROUND-TRIP + DEFORMATION DIAGNOSTIC")
    print("=" * 80)

    # Build bundle
    cfg = Config.fromfile(args.config)
    model_cfg = cfg.model.to_dict() if hasattr(cfg.model, 'to_dict') else cfg.model
    bundle_cls = MODEL_BUNDLES.get(model_cfg['type'])
    bundle = bundle_cls.from_config(model_cfg)
    bundle.eval()

    # Load checkpoint (transformer weights)
    state_dict = load_checkpoint(args.checkpoint, map_location='cpu')
    print(f'Loading checkpoint: {args.checkpoint}')
    bundle.load_state_dict_selective(state_dict)
    del state_dict
    gc.collect()

    print(f"\nVAE config:")
    print(f"  z_dim: {bundle.vae.config.z_dim}")
    print(f"  in_channels: {bundle.vae.config.in_channels}")
    print(f"  out_channels: {bundle.vae.config.out_channels}")
    print(f"  scale_factor_temporal: {bundle.vae.config.scale_factor_temporal}")
    print(f"  latents_mean (first 4): {bundle.latents_mean.flatten()[:4].tolist()}")
    print(f"  latents_std (first 4): {bundle.latents_std.flatten()[:4].tolist()}")

    smpl_processor = bundle.smpl_pose_processor
    print(f"\nSMPL Processor:")
    print(f"  mean shape: {smpl_processor.mean.shape}, first 6 (transl): {smpl_processor.mean[:6].tolist()}")
    print(f"  std shape: {smpl_processor.std.shape}, first 6 (transl): {smpl_processor.std[:6].tolist()}")
    print(f"  mean[6:12] (global_orient): {smpl_processor.mean[6:12].tolist()}")
    print(f"  std[6:12] (global_orient): {smpl_processor.std[6:12].tolist()}")
    print(f"  mean[12:18] (body_pose joint 0): {smpl_processor.mean[12:18].tolist()}")
    print(f"  std[12:18] (body_pose joint 0): {smpl_processor.std[12:18].tolist()}")

    # ================================================================
    # TEST 1: VAE round-trip with SYNTHETIC motion (identity-ish rotation)
    # ================================================================
    print("\n" + "=" * 80)
    print("TEST 1: VAE round-trip with identity-like synthetic motion")
    print("=" * 80)

    T = 129
    # Create a simple motion: identity rotation + zero translation
    # Row-major identity rotation 6D: first row [1,0], second row [0,1], third row [0,0]
    # = [R00=1, R01=0, R10=0, R11=1, R20=0, R21=0]
    identity_rot6d = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    # 22 joints (global_orient + 21 body_pose) -> [1, T, 132]
    rotations = identity_rot6d.view(1, 1, 1, 6).expand(1, T, 22, 6).reshape(1, T, 132)
    # Translation: zeros for abs_rel (6 dims)
    transl = torch.zeros(1, T, 6)
    motion_synthetic = torch.cat([transl, rotations], dim=-1)  # [1, T, 138]

    print(f"  Synthetic motion shape: {motion_synthetic.shape}")
    print(f"  Rotation values (first joint, first frame): {motion_synthetic[0, 0, 6:12].tolist()}")

    # Normalize
    motion_norm = smpl_processor.normalize(motion_synthetic)
    print(f"  After normalize - mean: {motion_norm.mean():.4f}, std: {motion_norm.std():.4f}")

    # Reshape for VAE
    motion_4d = rearrange(motion_norm, 'b t (j d) -> b t j d', d=6)  # [1, T, 23, 6]
    print(f"  Reshaped for VAE: {motion_4d.shape}")

    # Move VAE to device
    bundle.vae = bundle.vae.to(device)
    motion_4d = motion_4d.to(device)

    # Encode
    with torch.no_grad(), torch.autocast(device.type, enabled=False):
        z_raw = bundle.vae.encode(motion_4d.float())
    z = DiagonalGaussianDistributionNd(z_raw).mode()
    print(f"  Encoded latents: shape={z.shape}, mean={z.mean():.4f}, std={z.std():.4f}")

    # Normalize latents (as training does)
    latents_mean = bundle.latents_mean.to(z)
    latents_std = bundle.latents_std.to(z)
    z_norm = (z - latents_mean) / latents_std
    print(f"  Normalized latents: mean={z_norm.mean():.4f}, std={z_norm.std():.4f}")

    # Denormalize latents (as inference does)
    z_denorm = z_norm * latents_std + latents_mean
    print(f"  Denormalized latents diff from original: {(z_denorm - z).abs().max():.2e}")

    # Decode
    with torch.no_grad(), torch.autocast(device.type, enabled=False):
        motion_decoded = bundle.vae.decode(z_denorm.float())  # [B, T, J, C]
    print(f"  Decoded motion shape: {motion_decoded.shape}")

    # Reshape back to flat
    motion_flat = rearrange(motion_decoded, 'b t j d -> b t (j d)')

    # Denormalize
    motion_denorm = smpl_processor.denormalize(motion_flat.cpu())

    # Check rotation quality
    pred_poses = motion_denorm[..., 6:]  # [B, T, 132]
    pred_poses_reshaped = pred_poses.reshape(-1, 22, 6)
    check_rotation_quality(pred_poses_reshaped, "Synthetic round-trip")

    # Compare with original
    recon_error = (motion_denorm - motion_synthetic).abs()
    print(f"  Reconstruction error (full): mean={recon_error.mean():.4f}, max={recon_error.max():.4f}")
    print(f"  Reconstruction error (rotation): mean={recon_error[..., 6:].mean():.4f}")
    print(f"  Reconstruction error (transl): mean={recon_error[..., :6].mean():.4f}")

    # ================================================================
    # TEST 2: VAE round-trip with REAL training data
    # ================================================================
    print("\n" + "=" * 80)
    print("TEST 2: VAE round-trip with real training data")
    print("=" * 80)

    # Load a real motion file
    import json
    anno_file = "data/annotation/train_hq_motionhub_hymotion.json"
    if os.path.exists(anno_file):
        with open(anno_file, 'r') as f:
            anno_data = json.load(f)
        # Get first entry
        if isinstance(anno_data, list):
            first_entry = anno_data[0]
        elif isinstance(anno_data, dict):
            first_key = list(anno_data.keys())[0]
            first_entry = anno_data[first_key]
        print(f"  First annotation entry keys: {list(first_entry.keys()) if isinstance(first_entry, dict) else 'list item'}")
        print(f"  Entry: {first_entry}")
    else:
        print(f"  Annotation file not found: {anno_file}")
        # Try to load a motion file directly
        # Look for any NPZ/npy in data/motionhub
        import glob
        npz_files = glob.glob("data/motionhub/**/*.npz", recursive=True)[:5]
        print(f"  Found {len(npz_files)} npz files")
        if npz_files:
            print(f"  First: {npz_files[0]}")

    # Try to load motion using the dataset pipeline
    # Since we can't easily run the full pipeline, let's create motion from stats
    # Use the mean + some std to create realistic-looking motion
    print("\n  Creating realistic motion from stats (mean + random variation)...")
    mean = smpl_processor.mean.clone()  # [138]
    std = smpl_processor.std.clone()  # [138]

    # Create motion that's slightly varied from mean (simulating real data)
    torch.manual_seed(42)
    real_ish_motion = mean.unsqueeze(0).unsqueeze(0).expand(1, T, -1) + \
                      0.3 * std.unsqueeze(0).unsqueeze(0).expand(1, T, -1) * torch.randn(1, T, 138)

    print(f"  Realistic motion shape: {real_ish_motion.shape}")

    # Check rotation quality of this "real" motion
    real_poses = real_ish_motion[..., 6:].reshape(-1, 22, 6)
    check_rotation_quality(real_poses, "Input real-ish motion")

    # Normalize
    motion_norm_real = smpl_processor.normalize(real_ish_motion)
    print(f"  After normalize: mean={motion_norm_real.mean():.4f}, std={motion_norm_real.std():.4f}")

    # VAE round-trip
    motion_4d_real = rearrange(motion_norm_real, 'b t (j d) -> b t j d', d=6).to(device)
    with torch.no_grad(), torch.autocast(device.type, enabled=False):
        z_real = bundle.vae.encode(motion_4d_real.float())
    z_real = DiagonalGaussianDistributionNd(z_real).mode()
    print(f"  Encoded latents: mean={z_real.mean():.4f}, std={z_real.std():.4f}")

    # Normalize and denormalize (simulate training->inference path)
    z_real_norm = (z_real - latents_mean) / latents_std
    print(f"  Normalized latents: mean={z_real_norm.mean():.4f}, std={z_real_norm.std():.4f}")
    z_real_denorm = z_real_norm * latents_std + latents_mean

    with torch.no_grad(), torch.autocast(device.type, enabled=False):
        motion_decoded_real = bundle.vae.decode(z_real_denorm.float())

    motion_flat_real = rearrange(motion_decoded_real, 'b t j d -> b t (j d)')
    motion_denorm_real = smpl_processor.denormalize(motion_flat_real.cpu())

    # Check rotation quality
    real_decoded_poses = motion_denorm_real[..., 6:].reshape(-1, 22, 6)
    check_rotation_quality(real_decoded_poses, "Round-trip real-ish motion")

    # Reconstruction error
    recon_error_real = (motion_denorm_real - real_ish_motion).abs()
    print(f"  Reconstruction error (full): mean={recon_error_real.mean():.4f}, max={recon_error_real.max():.4f}")

    # ================================================================
    # TEST 3: What does the transformer produce? Decode its output.
    # ================================================================
    print("\n" + "=" * 80)
    print("TEST 3: Transformer output quality check")
    print("=" * 80)

    # Move transformer to GPU, keep VAE on GPU
    bundle.transformer = bundle.transformer.to(device, torch.bfloat16)
    torch.cuda.empty_cache()

    vae_temporal = bundle.vae.config.scale_factor_temporal
    num_joints = 23
    num_frames = 129
    num_latent_frames = (num_frames - 1) // vae_temporal + 1
    num_channels = bundle.transformer.config.in_channels

    # Encode a simple prompt
    prompt = "a person walks forward slowly"
    text_states = bundle.encode_prompt(prompt, max_sequence_length=256)
    text_states = text_states.to(device, torch.bfloat16)
    neg_text_states = bundle.encode_prompt("", max_sequence_length=256)
    neg_text_states = neg_text_states.to(device, torch.bfloat16)

    print(f"  Text states shape: {text_states.shape}")
    print(f"  Generating with 50 steps, cfg=5.0...")

    # Setup scheduler
    bundle.scheduler.set_timesteps(50, device=device)
    timesteps = bundle.scheduler.timesteps

    # Initial noise
    shape = (1, num_channels, num_latent_frames, num_joints)
    latents = randn_tensor(shape, device=device, dtype=torch.bfloat16)

    # All-ones mask (no conditioning)
    first_frame_mask = torch.ones_like(latents)
    condition = torch.zeros_like(latents)
    motion_mask = torch.ones(1, num_latent_frames, num_joints, device=device)

    # Denoising loop
    for i, t in enumerate(timesteps):
        # expand_timesteps logic
        temp_ts = (first_frame_mask[0][0] * t).flatten()
        timestep = temp_ts.unsqueeze(0)

        latent_model_input = latents.to(torch.bfloat16)

        noise_pred = bundle.transformer(
            hidden_states=latent_model_input,
            timestep=timestep,
            encoder_hidden_states=text_states,
            attention_kwargs=None,
            is_causal=False,
            hidden_states_mask=motion_mask,
        )

        # CFG
        noise_uncond = bundle.transformer(
            hidden_states=latent_model_input,
            timestep=timestep,
            encoder_hidden_states=neg_text_states,
            attention_kwargs=None,
            is_causal=False,
            hidden_states_mask=motion_mask,
        )
        noise_pred = noise_uncond + 5.0 * (noise_pred - noise_uncond)

        latents = bundle.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    print(f"  Denoised latents: shape={latents.shape}, mean={latents.float().mean():.4f}, std={latents.float().std():.4f}")

    # Now decode
    latents_float = latents.float()
    z_for_decode = latents_float * latents_std + latents_mean
    print(f"  After latent denorm: mean={z_for_decode.mean():.4f}, std={z_for_decode.std():.4f}")

    with torch.no_grad(), torch.autocast(device.type, enabled=False):
        motion_gen = bundle.vae.decode(z_for_decode.float())
    print(f"  VAE decoded motion: shape={motion_gen.shape}, mean={motion_gen.mean():.4f}, std={motion_gen.std():.4f}")

    # Reshape and denormalize
    motion_gen_flat = rearrange(motion_gen, 'b t j d -> b t (j d)')
    print(f"  Flattened: shape={motion_gen_flat.shape}")
    print(f"  Before denorm: mean={motion_gen_flat.mean():.4f}, std={motion_gen_flat.std():.4f}")

    motion_gen_denorm = smpl_processor.denormalize(motion_gen_flat.cpu())
    print(f"  After denorm: mean={motion_gen_denorm.mean():.4f}, std={motion_gen_denorm.std():.4f}")

    # Check rotation quality
    gen_poses = motion_gen_denorm[..., 6:].reshape(-1, 22, 6)
    check_rotation_quality(gen_poses, "Generated motion")

    # ================================================================
    # TEST 4: Compare normalized latents from real data vs generated
    # ================================================================
    print("\n" + "=" * 80)
    print("TEST 4: Compare latent distributions")
    print("=" * 80)

    # Real data latents (from test 2)
    print(f"  Real data normalized latents: mean={z_real_norm.mean():.4f}, std={z_real_norm.std():.4f}")
    print(f"  Generated normalized latents: mean={latents.float().mean():.4f}, std={latents.float().std():.4f}")
    print(f"  Ratio of stds: {latents.float().std() / z_real_norm.std():.4f}")

    # Per-channel comparison
    print("\n  Per-channel latent statistics:")
    print(f"  {'Ch':>3} | {'Real mean':>10} | {'Real std':>10} | {'Gen mean':>10} | {'Gen std':>10} | {'Std ratio':>10}")
    print(f"  {'-'*3}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    for ch in range(num_channels):
        r_mean = z_real_norm[0, ch].mean().item()
        r_std = z_real_norm[0, ch].std().item()
        g_mean = latents[0, ch].float().mean().item()
        g_std = latents[0, ch].float().std().item()
        ratio = g_std / (r_std + 1e-8)
        print(f"  {ch:3d} | {r_mean:10.4f} | {r_std:10.4f} | {g_mean:10.4f} | {g_std:10.4f} | {ratio:10.4f}")

    # ================================================================
    # TEST 5: What if we decode REAL latents? (ground truth test)
    # ================================================================
    print("\n" + "=" * 80)
    print("TEST 5: Decode REAL latents (encoded from real-ish motion)")
    print("=" * 80)

    # Take the real latents (already normalized) and decode them exactly as inference would
    # This tests the FULL decode pipeline path
    z_for_real_decode = z_real_norm * latents_std + latents_mean
    with torch.no_grad(), torch.autocast(device.type, enabled=False):
        motion_from_real_latents = bundle.vae.decode(z_for_real_decode.float())

    motion_from_real_flat = rearrange(motion_from_real_latents, 'b t j d -> b t (j d)')
    motion_from_real_denorm = smpl_processor.denormalize(motion_from_real_flat.cpu())

    real_latent_poses = motion_from_real_denorm[..., 6:].reshape(-1, 22, 6)
    check_rotation_quality(real_latent_poses, "Decoded from REAL latents")

    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
If TEST 1 (synthetic) shows good col norms (~1.0) → VAE works on simple input
If TEST 2 (real-ish) shows good col norms (~1.0) → VAE round-trip works
If TEST 5 (real latents through full path) shows good col norms → full decode path works
If TEST 3 (generated) shows bad col norms → issue is in what the transformer produces
If all tests show bad col norms → issue is in the VAE or normalization
    """)

    print("\nDone!")


if __name__ == '__main__':
    main()
