"""Fast PRISM inference diagnostic - runs on T4 with 15GB VRAM.

Tests key hypotheses quickly:
1. Text length mismatch (128 vs 256)
2. Model output sanity (single-step)
3. Short denoising + decode + rotation validity check

Usage:
    python3 scripts/inference/fast_diagnose.py \
        --config configs/prism/prism_1b_tp2m_multiframe.py \
        --checkpoint work_dirs/prism_1b_tp2m_multiframe/checkpoint-iter_15000
"""

import argparse
import gc
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import numpy as np
from einops import rearrange
from mmengine.config import Config

import hftrainer  # noqa
from hftrainer.registry import MODEL_BUNDLES
from hftrainer.utils.checkpoint_utils import load_checkpoint
from diffusers.utils.torch_utils import randn_tensor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()


def check_rotation_validity(rot6d_row_major, label=""):
    """Check if rot6d values represent valid rotations.
    Row-major rot6d: [R00, R01, R10, R11, R20, R21]
    Convert to columns: col1=[R00,R10,R20], col2=[R01,R11,R21]
    Valid rotation: each column has norm ~1.0
    """
    # Row-major [R00,R01,R10,R11,R20,R21] -> column arrangement
    col1 = rot6d_row_major[..., [0, 2, 4]]  # [R00, R10, R20]
    col2 = rot6d_row_major[..., [1, 3, 5]]  # [R01, R11, R21]
    col1_norms = col1.norm(dim=-1)
    col2_norms = col2.norm(dim=-1)

    is_valid = (abs(col1_norms.mean() - 1.0) < 0.3) and (abs(col2_norms.mean() - 1.0) < 0.3)
    status = "VALID" if is_valid else "DEFORMED"

    print(f"  [{label}] Col1 norms: mean={col1_norms.mean():.4f}, std={col1_norms.std():.4f}")
    print(f"  [{label}] Col2 norms: mean={col2_norms.mean():.4f}, std={col2_norms.std():.4f}")
    print(f"  [{label}] Status: {status}")
    return is_valid


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device)
    dtype = torch.bfloat16

    print("=" * 70)
    print("  FAST PRISM DIAGNOSTIC")
    print("=" * 70)

    # Build bundle
    cfg = Config.fromfile(args.config)
    model_cfg = cfg.model.to_dict() if hasattr(cfg.model, 'to_dict') else cfg.model
    bundle_cls = MODEL_BUNDLES.get(model_cfg['type'])
    bundle = bundle_cls.from_config(model_cfg)
    bundle.eval()

    # Load checkpoint
    state_dict = load_checkpoint(args.checkpoint, map_location='cpu')
    print(f'Loading checkpoint: {args.checkpoint}')
    bundle.load_state_dict_selective(state_dict)
    del state_dict
    gc.collect()

    # =========================================================================
    # CHECK 1: Latent normalization stats
    # =========================================================================
    print("\n" + "=" * 70)
    print("  CHECK 1: Latent normalization stats")
    print("=" * 70)
    print(f"  latents_mean shape: {bundle.latents_mean.shape}")
    print(f"  latents_mean values: {bundle.latents_mean.squeeze().tolist()}")
    print(f"  latents_std shape: {bundle.latents_std.shape}")
    print(f"  latents_std values: {bundle.latents_std.squeeze().tolist()}")

    if bundle.latents_std.min() < 0.01:
        print("  WARNING: latents_std has near-zero values! This would cause explosion!")
    if bundle.latents_mean.abs().max() > 10:
        print("  WARNING: latents_mean has very large values!")

    # =========================================================================
    # CHECK 2: Scheduler timesteps and sigmas
    # =========================================================================
    print("\n" + "=" * 70)
    print("  CHECK 2: Scheduler state")
    print("=" * 70)
    print(f"  Scheduler type: {type(bundle.scheduler).__name__}")
    print(f"  num_train_timesteps: {bundle.scheduler.config.num_train_timesteps}")
    print(f"  shift: {bundle.scheduler.config.shift}")

    # Training schedule (1000 steps)
    train_ts = bundle.scheduler.timesteps
    train_sigmas = bundle.scheduler.sigmas
    print(f"  Training timesteps: {len(train_ts)} values, range [{train_ts[-1]:.2f}, {train_ts[0]:.2f}]")
    print(f"  Training sigmas: {len(train_sigmas)} values, range [{train_sigmas.min():.6f}, {train_sigmas.max():.6f}]")
    print(f"  First 5 timesteps: {train_ts[:5].tolist()}")
    print(f"  Last 5 timesteps: {train_ts[-5:].tolist()}")

    # Set to inference schedule (50 steps)
    bundle.scheduler.set_timesteps(50, device='cpu')
    infer_ts = bundle.scheduler.timesteps
    infer_sigmas = bundle.scheduler.sigmas
    print(f"  Inference timesteps (50 steps): range [{infer_ts[-1]:.2f}, {infer_ts[0]:.2f}]")
    print(f"  Inference sigmas: range [{infer_sigmas.min():.6f}, {infer_sigmas.max():.6f}]")

    # =========================================================================
    # CHECK 3: Text encoding comparison (128 vs 256)
    # =========================================================================
    print("\n" + "=" * 70)
    print("  CHECK 3: Text encoding - 128 vs 256 tokens")
    print("=" * 70)

    # Encode text on CPU
    bundle.text_encoder = bundle.text_encoder.cpu()

    prompt = "a person walks forward slowly"

    # Method A: Use bundle.encode_prompt with 128
    text_128 = bundle.encode_prompt(prompt, max_sequence_length=128, dtype=dtype)
    print(f"  text_128 shape: {text_128.shape}, norm={text_128.float().norm():.4f}")
    print(f"  text_128 non-zero elements: {(text_128.abs() > 1e-6).sum().item()}")

    # Method B: Use bundle.encode_prompt with 256
    text_256 = bundle.encode_prompt(prompt, max_sequence_length=256, dtype=dtype)
    print(f"  text_256 shape: {text_256.shape}, norm={text_256.float().norm():.4f}")
    print(f"  text_256 non-zero elements: {(text_256.abs() > 1e-6).sum().item()}")

    # Check first 128 positions - should be identical (content is same)
    first_128_from_256 = text_256[:, :128, :]
    diff = (text_128 - first_128_from_256).abs().max().item()
    print(f"  Diff in first 128 positions: {diff:.8f}")
    print(f"  (Should be ~0 if text encoding is consistent)")

    # Negative text
    neg_text_128 = bundle.encode_prompt("", max_sequence_length=128, dtype=dtype)
    neg_text_256 = bundle.encode_prompt("", max_sequence_length=256, dtype=dtype)
    print(f"  neg_text_128 norm: {neg_text_128.float().norm():.6f}")
    print(f"  neg_text_256 norm: {neg_text_256.float().norm():.6f}")

    # Free text encoder
    del bundle.text_encoder
    gc.collect()
    torch.cuda.empty_cache()

    # =========================================================================
    # CHECK 4: Model single-step output sanity
    # =========================================================================
    print("\n" + "=" * 70)
    print("  CHECK 4: Transformer single-step output")
    print("=" * 70)

    bundle.transformer = bundle.transformer.to(device, dtype)
    torch.cuda.empty_cache()
    print(f"  GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Small latent: 5 frames
    num_latent_frames = 5
    num_joints = 23
    num_channels = bundle.transformer.config.in_channels
    print(f"  Latent shape: [1, {num_channels}, {num_latent_frames}, {num_joints}]")

    # Random latent (pure noise, like start of denoising)
    latents = torch.randn(1, num_channels, num_latent_frames, num_joints, device=device, dtype=dtype)

    # Set inference timesteps
    bundle.scheduler.set_timesteps(50, device=device)
    t = bundle.scheduler.timesteps[0]  # First (largest) timestep

    # Create per-token timestep
    timestep = t.unsqueeze(0).unsqueeze(1).expand(1, num_latent_frames * num_joints)

    # Test with text_256
    text_256_dev = text_256.to(device=device, dtype=dtype)
    neg_256_dev = neg_text_256.to(device=device, dtype=dtype)

    # Forward pass
    pred = bundle.transformer(
        hidden_states=latents,
        timestep=timestep,
        encoder_hidden_states=text_256_dev,
        hidden_states_mask=torch.ones(1, num_latent_frames, num_joints, device=device),
    )
    print(f"\n  With text (len=256):")
    print(f"    pred shape: {pred.shape}")
    print(f"    pred mean: {pred.float().mean():.6f}")
    print(f"    pred std: {pred.float().std():.6f}")
    print(f"    pred abs max: {pred.float().abs().max():.6f}")

    # Uncond forward pass
    pred_uncond = bundle.transformer(
        hidden_states=latents,
        timestep=timestep,
        encoder_hidden_states=neg_256_dev,
        hidden_states_mask=torch.ones(1, num_latent_frames, num_joints, device=device),
    )
    print(f"\n  Unconditional (empty text, len=256):")
    print(f"    pred mean: {pred_uncond.float().mean():.6f}")
    print(f"    pred std: {pred_uncond.float().std():.6f}")

    # CFG combination
    cfg_scale = 5.0
    pred_cfg = pred_uncond + cfg_scale * (pred - pred_uncond)
    print(f"\n  After CFG (scale={cfg_scale}):")
    print(f"    pred_cfg mean: {pred_cfg.float().mean():.6f}")
    print(f"    pred_cfg std: {pred_cfg.float().std():.6f}")
    print(f"    pred_cfg abs max: {pred_cfg.float().abs().max():.6f}")

    # Test with text_128 for comparison
    text_128_dev = text_128.to(device=device, dtype=dtype)
    neg_128_dev = neg_text_128.to(device=device, dtype=dtype)

    pred_128 = bundle.transformer(
        hidden_states=latents,
        timestep=timestep,
        encoder_hidden_states=text_128_dev,
        hidden_states_mask=torch.ones(1, num_latent_frames, num_joints, device=device),
    )
    print(f"\n  With text (len=128):")
    print(f"    pred shape: {pred_128.shape}")
    print(f"    pred mean: {pred_128.float().mean():.6f}")
    print(f"    pred std: {pred_128.float().std():.6f}")

    diff_text_len = (pred.float() - pred_128.float()).abs()
    print(f"\n  Diff between 128 and 256 text:")
    print(f"    mean diff: {diff_text_len.mean():.6f}")
    print(f"    max diff: {diff_text_len.max():.6f}")
    print(f"    relative diff (vs pred std): {diff_text_len.mean() / pred.float().std():.4f}")

    # =========================================================================
    # CHECK 5: Full short denoising (10 steps) + decode
    # =========================================================================
    print("\n" + "=" * 70)
    print("  CHECK 5: Short denoising (10 steps) + decode")
    print("=" * 70)

    # Use 17 frames (smallest reasonable for VAE with temporal_scale=4: need (T-1)//4+1 = 5 latent frames)
    num_frames = 17
    vae_temporal = bundle.vae.config.scale_factor_temporal
    num_latent_frames = (num_frames - 1) // vae_temporal + 1
    print(f"  num_frames={num_frames}, vae_temporal={vae_temporal}, num_latent_frames={num_latent_frames}")

    # Set 10-step schedule for speed
    num_steps = 10
    bundle.scheduler.set_timesteps(num_steps, device=device)
    timesteps = bundle.scheduler.timesteps
    print(f"  Denoising: {num_steps} steps, timesteps: {timesteps.tolist()}")

    # Initial noise
    shape = (1, num_channels, num_latent_frames, num_joints)
    latents = torch.randn(*shape, device=device, dtype=dtype)
    print(f"  Initial latent std: {latents.float().std():.4f}")

    # Denoise with cfg=5.0, text_len=256
    for i, t in enumerate(timesteps):
        timestep = t.unsqueeze(0).unsqueeze(1).expand(1, num_latent_frames * num_joints)

        noise_pred = bundle.transformer(
            hidden_states=latents.to(dtype),
            timestep=timestep,
            encoder_hidden_states=text_256_dev,
            hidden_states_mask=torch.ones(1, num_latent_frames, num_joints, device=device),
        )
        noise_uncond = bundle.transformer(
            hidden_states=latents.to(dtype),
            timestep=timestep,
            encoder_hidden_states=neg_256_dev,
            hidden_states_mask=torch.ones(1, num_latent_frames, num_joints, device=device),
        )
        noise_pred = noise_uncond + cfg_scale * (noise_pred - noise_uncond)
        latents = bundle.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        if i % 3 == 0 or i == num_steps - 1:
            print(f"    Step {i}: latent mean={latents.float().mean():.4f}, std={latents.float().std():.4f}")

    # Free transformer
    bundle.transformer = bundle.transformer.cpu()
    gc.collect()
    torch.cuda.empty_cache()

    # Decode
    print("\n  Decoding...")
    bundle.vae = bundle.vae.to(device)

    # Denormalize latents
    latents_mean = bundle.latents_mean.to(latents)
    latents_std = bundle.latents_std.to(latents)
    latents_denorm = latents * latents_std + latents_mean
    print(f"  Denormalized latent: mean={latents_denorm.float().mean():.4f}, std={latents_denorm.float().std():.4f}")

    # VAE decode
    device_type = device.type if hasattr(device, 'type') else 'cuda'
    with torch.autocast(device_type, enabled=False):
        motion = bundle.vae.decode(latents_denorm.float())  # [B, T, J, D]
    print(f"  Decoded motion shape: {motion.shape}")

    # Post-process
    x_dec = rearrange(motion, "b t j d -> b t (j d)")
    x_dec = bundle.smpl_pose_processor.denormalize(x_dec)

    # Check rotation validity
    poses = x_dec[..., 6:]  # Skip translation
    poses_6d = poses.reshape(-1, 22, 6)  # [T, 22, 6] in row-major
    is_valid = check_rotation_validity(poses_6d, "cfg=5.0, text=256, 10 steps")

    # Also check translation
    transl_part = x_dec[..., :6]
    print(f"  Translation (abs_rel): mean={transl_part.mean():.4f}, std={transl_part.std():.4f}")

    # =========================================================================
    # CHECK 6: Repeat with NO CFG (cfg=1.0) to isolate CFG effect
    # =========================================================================
    print("\n" + "=" * 70)
    print("  CHECK 6: Denoising WITHOUT CFG (guidance_scale=1.0)")
    print("=" * 70)

    bundle.transformer = bundle.transformer.to(device, dtype)
    bundle.vae = bundle.vae.cpu()
    gc.collect()
    torch.cuda.empty_cache()

    bundle.scheduler.set_timesteps(num_steps, device=device)
    timesteps = bundle.scheduler.timesteps
    latents_nocfg = torch.randn(*shape, device=device, dtype=dtype)

    for i, t in enumerate(timesteps):
        timestep = t.unsqueeze(0).unsqueeze(1).expand(1, num_latent_frames * num_joints)
        noise_pred = bundle.transformer(
            hidden_states=latents_nocfg.to(dtype),
            timestep=timestep,
            encoder_hidden_states=text_256_dev,
            hidden_states_mask=torch.ones(1, num_latent_frames, num_joints, device=device),
        )
        latents_nocfg = bundle.scheduler.step(noise_pred, t, latents_nocfg, return_dict=False)[0]

        if i % 3 == 0 or i == num_steps - 1:
            print(f"    Step {i}: latent mean={latents_nocfg.float().mean():.4f}, std={latents_nocfg.float().std():.4f}")

    # Decode no-CFG result
    bundle.transformer = bundle.transformer.cpu()
    gc.collect()
    torch.cuda.empty_cache()
    bundle.vae = bundle.vae.to(device)

    latents_denorm_nocfg = latents_nocfg * latents_std.to(latents_nocfg) + latents_mean.to(latents_nocfg)
    with torch.autocast(device_type, enabled=False):
        motion_nocfg = bundle.vae.decode(latents_denorm_nocfg.float())

    x_dec_nocfg = rearrange(motion_nocfg, "b t j d -> b t (j d)")
    x_dec_nocfg = bundle.smpl_pose_processor.denormalize(x_dec_nocfg)
    poses_nocfg = x_dec_nocfg[..., 6:].reshape(-1, 22, 6)
    is_valid_nocfg = check_rotation_validity(poses_nocfg, "cfg=1.0, text=256, 10 steps")

    # =========================================================================
    # CHECK 7: What does pure noise decode to? (Sanity check VAE)
    # =========================================================================
    print("\n" + "=" * 70)
    print("  CHECK 7: Decode pure noise (VAE sanity check)")
    print("=" * 70)
    noise_latent = torch.randn(*shape, device=device, dtype=torch.float32)
    # Don't denormalize - just pass raw noise as if it were a valid latent
    with torch.autocast(device_type, enabled=False):
        motion_noise = bundle.vae.decode(noise_latent)
    x_dec_noise = rearrange(motion_noise, "b t j d -> b t (j d)")
    x_dec_noise = bundle.smpl_pose_processor.denormalize(x_dec_noise)
    poses_noise = x_dec_noise[..., 6:].reshape(-1, 22, 6)
    check_rotation_validity(poses_noise, "pure noise (should be DEFORMED)")

    # Also decode zeros (perfectly "average" latent)
    zero_latent = torch.zeros(*shape, device=device, dtype=torch.float32)
    # Denormalize zeros: 0 * std + mean = mean
    zero_denorm = zero_latent * bundle.latents_std.to(device) + bundle.latents_mean.to(device)
    with torch.autocast(device_type, enabled=False):
        motion_zero = bundle.vae.decode(zero_denorm.float())
    x_dec_zero = rearrange(motion_zero, "b t j d -> b t (j d)")
    x_dec_zero = bundle.smpl_pose_processor.denormalize(x_dec_zero)
    poses_zero = x_dec_zero[..., 6:].reshape(-1, 22, 6)
    check_rotation_validity(poses_zero, "zero latent (avg pose, should be VALID)")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  CFG=5.0 denoised motion valid: {is_valid}")
    print(f"  CFG=1.0 denoised motion valid: {is_valid_nocfg}")
    print(f"  Text len 128 vs 256 max diff: {diff_text_len.max():.6f}")
    if not is_valid and is_valid_nocfg:
        print("  >>> ROOT CAUSE: CFG scale too high! Try guidance_scale <= 2.0")
    elif not is_valid and not is_valid_nocfg:
        print("  >>> Model output itself is broken. Check:")
        print("      1. Model prediction std (should be ~1.0 for velocity)")
        print("      2. Scheduler step direction (should reduce noise)")
        print("      3. Whether model converged on this data format")
    elif is_valid:
        print("  >>> Motion is VALID! Deformation may be a decode/post-processing issue")
        print("      or only appears with longer sequences (>17 frames)")


if __name__ == '__main__':
    main()
