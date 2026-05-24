"""Memory-efficient PRISM diagnostic - runs on T4 (15GB).

Tests each stage independently to find where deformation is introduced.
Only loads one large component at a time.
"""

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
from diffusers.utils.torch_utils import randn_tensor
from hftrainer.models.motion.prism.bundle import DiagonalGaussianDistributionNd

CONFIG = 'configs/prism/prism_1b_tp2m_multiframe.py'
CHECKPOINT = 'work_dirs/prism_1b_tp2m_multiframe/checkpoint-iter_15000'
PROMPT = "a person walks forward slowly"
NUM_FRAMES = 129
NUM_STEPS = 50
DEVICE = 'cuda'


def build_bundle():
    """Build bundle (all on CPU)."""
    cfg = Config.fromfile(CONFIG)
    model_cfg = cfg.model.to_dict() if hasattr(cfg.model, 'to_dict') else cfg.model
    bundle_cls = MODEL_BUNDLES.get(model_cfg['type'])
    bundle = bundle_cls.from_config(model_cfg)
    bundle.eval()
    state_dict = load_checkpoint(CHECKPOINT, map_location='cpu')
    bundle.load_state_dict_selective(state_dict)
    del state_dict
    gc.collect()
    return bundle


def test_scheduler_timesteps(bundle):
    """TEST 1: Check scheduler timestep values."""
    print("\n" + "="*80)
    print("TEST 1: SCHEDULER TIMESTEP VALUES")
    print("="*80)

    scheduler = bundle.scheduler

    # Training mode: set_timesteps(1000)
    scheduler.set_timesteps(1000)
    train_ts = scheduler.timesteps.clone()
    train_sigmas = scheduler.sigmas.clone()
    print(f"Training set_timesteps(1000):")
    print(f"  timesteps shape: {train_ts.shape}, dtype: {train_ts.dtype}")
    print(f"  timesteps range: [{train_ts.min():.4f}, {train_ts.max():.4f}]")
    print(f"  first 5 timesteps: {train_ts[:5].tolist()}")
    print(f"  last 5 timesteps: {train_ts[-5:].tolist()}")
    print(f"  sigmas shape: {train_sigmas.shape}, dtype: {train_sigmas.dtype}")
    print(f"  sigmas range: [{train_sigmas.min():.6f}, {train_sigmas.max():.6f}]")
    print(f"  first 5 sigmas: {train_sigmas[:5].tolist()}")
    print(f"  last 5 sigmas: {train_sigmas[-5:].tolist()}")

    # Inference mode: set_timesteps(50)
    scheduler.set_timesteps(50)
    infer_ts = scheduler.timesteps.clone()
    infer_sigmas = scheduler.sigmas.clone()
    print(f"\nInference set_timesteps(50):")
    print(f"  timesteps shape: {infer_ts.shape}, dtype: {infer_ts.dtype}")
    print(f"  timesteps range: [{infer_ts.min():.4f}, {infer_ts.max():.4f}]")
    print(f"  first 5 timesteps: {infer_ts[:5].tolist()}")
    print(f"  last 5 timesteps: {infer_ts[-5:].tolist()}")
    print(f"  sigmas range: [{infer_sigmas.min():.6f}, {infer_sigmas.max():.6f}]")

    # Check: Are inference timesteps a subset of training timesteps?
    found = 0
    not_found = 0
    for t in infer_ts:
        if (train_ts == t).any():
            found += 1
        else:
            not_found += 1
    print(f"\n  Inference timesteps found in training schedule: {found}/{len(infer_ts)}")
    print(f"  NOT found (require interpolation): {not_found}/{len(infer_ts)}")

    return infer_ts, infer_sigmas


def test_text_encoding(bundle):
    """TEST 2: Compare text encoding with 128 vs 256 max_seq_len."""
    print("\n" + "="*80)
    print("TEST 2: TEXT ENCODING (128 vs 256)")
    print("="*80)

    bundle.text_encoder = bundle.text_encoder.cpu()

    for max_seq_len in [128, 256]:
        text_states = bundle.encode_prompt(
            PROMPT,
            max_sequence_length=max_seq_len,
            dtype=torch.float32,
        )
        print(f"\n  max_seq_len={max_seq_len}:")
        print(f"    shape: {text_states.shape}")
        print(f"    non-zero rows: {(text_states.abs().sum(dim=-1) > 0).sum().item()}")
        print(f"    mean (non-pad): {text_states[0, :10].mean().item():.6f}")
        print(f"    std (non-pad): {text_states[0, :10].std().item():.6f}")
        print(f"    mean (all): {text_states.mean().item():.6f}")
        print(f"    max abs: {text_states.abs().max().item():.4f}")

    # Negative prompt
    neg_states = bundle.encode_prompt("", max_sequence_length=256, dtype=torch.float32)
    print(f"\n  Negative prompt (empty, max_seq_len=256):")
    print(f"    shape: {neg_states.shape}")
    print(f"    non-zero rows: {(neg_states.abs().sum(dim=-1) > 0).sum().item()}")
    print(f"    mean: {neg_states.mean().item():.6f}")

    # Save text states for later use (use 256 to match training)
    text_states_256 = bundle.encode_prompt(PROMPT, max_sequence_length=256, dtype=torch.float32)
    neg_states_256 = bundle.encode_prompt("", max_sequence_length=256, dtype=torch.float32)

    # Also get 128 for comparison
    text_states_128 = bundle.encode_prompt(PROMPT, max_sequence_length=128, dtype=torch.float32)
    neg_states_128 = bundle.encode_prompt("", max_sequence_length=128, dtype=torch.float32)

    # Free text encoder
    del bundle.text_encoder
    gc.collect()
    torch.cuda.empty_cache()

    return text_states_256, neg_states_256, text_states_128, neg_states_128


def test_transformer_forward(bundle, text_states, neg_states, text_len_label="256"):
    """TEST 3: Single transformer forward pass with known input."""
    print("\n" + "="*80)
    print(f"TEST 3: TRANSFORMER FORWARD (text_len={text_len_label})")
    print("="*80)

    device = torch.device(DEVICE)
    dtype = torch.bfloat16
    bundle.transformer = bundle.transformer.to(device, dtype)
    torch.cuda.empty_cache()

    vae_temporal = bundle.vae.config.scale_factor_temporal
    num_joints = 23
    num_latent_frames = (NUM_FRAMES - 1) // vae_temporal + 1
    num_channels = bundle.transformer.config.in_channels

    print(f"  Config: T_latent={num_latent_frames}, J={num_joints}, C={num_channels}")
    print(f"  GPU mem: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Create random latents (simulating pure noise at t=max)
    torch.manual_seed(42)
    shape = (1, num_channels, num_latent_frames, num_joints)
    latents = randn_tensor(shape, device=device, dtype=dtype)
    print(f"\n  Input latents: shape={latents.shape}, mean={latents.float().mean():.4f}, std={latents.float().std():.4f}")

    # Create per-token timestep (all same value = max timestep)
    # First get the scheduler's max timestep
    bundle.scheduler.set_timesteps(NUM_STEPS, device=device)
    max_t = bundle.scheduler.timesteps[0]
    print(f"  Max timestep (first in schedule): {max_t.item():.4f}")

    # Replicate expand_timesteps logic from inference
    first_frame_mask = torch.ones_like(latents)
    temp_ts = (first_frame_mask[0][0] * max_t).flatten()
    timestep = temp_ts.unsqueeze(0)  # [1, T_latent * J]
    print(f"  Timestep: shape={timestep.shape}, all_equal_to_max={torch.all(timestep == max_t).item()}")

    # Motion mask (all ones = all valid)
    motion_mask = torch.ones(1, num_latent_frames, num_joints, device=device)

    # Forward pass
    text_dev = text_states.to(device=device, dtype=dtype)
    with torch.no_grad():
        output = bundle.transformer(
            hidden_states=latents,
            timestep=timestep,
            encoder_hidden_states=text_dev,
            attention_kwargs=None,
            is_causal=False,
            hidden_states_mask=motion_mask,
        )

    output_f32 = output.float()
    print(f"\n  Output: shape={output.shape}")
    print(f"  Output stats: mean={output_f32.mean():.4f}, std={output_f32.std():.4f}")
    print(f"  Output abs max: {output_f32.abs().max():.4f}")
    print(f"  Output abs mean: {output_f32.abs().mean():.4f}")
    print(f"  Any NaN: {torch.isnan(output_f32).any().item()}")
    print(f"  Any Inf: {torch.isinf(output_f32).any().item()}")

    # Check if output is reasonable - it should predict velocity (noise - latents)
    # At max timestep, input is nearly pure noise, so velocity ≈ noise - 0 ≈ noise
    # The output should have similar magnitude to input
    ratio = output_f32.std() / latents.float().std()
    print(f"  Output/Input std ratio: {ratio:.4f} (should be ~1.0 for pure noise input)")

    # Also test with negative prompt
    neg_dev = neg_states.to(device=device, dtype=dtype)
    with torch.no_grad():
        output_neg = bundle.transformer(
            hidden_states=latents,
            timestep=timestep,
            encoder_hidden_states=neg_dev,
            attention_kwargs=None,
            is_causal=False,
            hidden_states_mask=motion_mask,
        )

    output_neg_f32 = output_neg.float()
    print(f"\n  Negative output stats: mean={output_neg_f32.mean():.4f}, std={output_neg_f32.std():.4f}")
    diff = (output_f32 - output_neg_f32)
    print(f"  Cond-Uncond difference: mean={diff.mean():.4f}, std={diff.std():.4f}, abs_max={diff.abs().max():.4f}")

    return output, latents, timestep, motion_mask


def test_denoising_loop(bundle, text_states, neg_states, guidance_scale=5.0, use_fp32=False, text_len_label="256"):
    """TEST 4: Full denoising loop with statistics at each step."""
    print("\n" + "="*80)
    print(f"TEST 4: DENOISING LOOP (cfg={guidance_scale}, fp32={use_fp32}, text={text_len_label})")
    print("="*80)

    device = torch.device(DEVICE)
    dtype = torch.float32 if use_fp32 else torch.bfloat16

    # Move transformer to correct dtype
    bundle.transformer = bundle.transformer.to(device, dtype)
    torch.cuda.empty_cache()

    vae_temporal = bundle.vae.config.scale_factor_temporal
    num_joints = 23
    num_latent_frames = (NUM_FRAMES - 1) // vae_temporal + 1
    num_channels = bundle.transformer.config.in_channels

    # Setup
    torch.manual_seed(42)
    shape = (1, num_channels, num_latent_frames, num_joints)
    latents = randn_tensor(shape, device=device, dtype=dtype)

    bundle.scheduler.set_timesteps(NUM_STEPS, device=device)
    timesteps = bundle.scheduler.timesteps

    text_dev = text_states.to(device=device, dtype=dtype)
    neg_dev = neg_states.to(device=device, dtype=dtype)

    motion_mask = torch.ones(1, num_latent_frames, num_joints, device=device)
    first_frame_mask = torch.ones_like(latents)

    do_cfg = guidance_scale > 1.0

    print(f"  Latents shape: {shape}, dtype: {dtype}")
    print(f"  Num steps: {len(timesteps)}")
    print(f"  Timestep range: [{timesteps[-1].item():.2f}, {timesteps[0].item():.2f}]")

    # Track statistics
    step_stats = []

    for i, t in enumerate(timesteps):
        # Per-token timestep
        temp_ts = (first_frame_mask[0][0] * t).flatten()
        ts_input = temp_ts.unsqueeze(0)

        latent_model_input = latents

        with torch.no_grad():
            noise_pred = bundle.transformer(
                hidden_states=latent_model_input,
                timestep=ts_input,
                encoder_hidden_states=text_dev,
                attention_kwargs=None,
                is_causal=False,
                hidden_states_mask=motion_mask,
            )

            if do_cfg:
                noise_uncond = bundle.transformer(
                    hidden_states=latent_model_input,
                    timestep=ts_input,
                    encoder_hidden_states=neg_dev,
                    attention_kwargs=None,
                    is_causal=False,
                    hidden_states_mask=motion_mask,
                )
                noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)

        # Scheduler step
        latents = bundle.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        # Record stats every 10 steps and at start/end
        if i < 3 or i >= len(timesteps) - 3 or i % 10 == 0:
            lat_f32 = latents.float()
            pred_f32 = noise_pred.float()
            stats = {
                'step': i, 't': t.item(),
                'lat_mean': lat_f32.mean().item(),
                'lat_std': lat_f32.std().item(),
                'lat_absmax': lat_f32.abs().max().item(),
                'pred_mean': pred_f32.mean().item(),
                'pred_std': pred_f32.std().item(),
                'pred_absmax': pred_f32.abs().max().item(),
                'nan': torch.isnan(lat_f32).any().item(),
            }
            step_stats.append(stats)
            if i < 3 or i >= len(timesteps) - 3:
                print(f"  Step {i:3d} t={t.item():7.2f}: lat[mean={stats['lat_mean']:+.4f} std={stats['lat_std']:.4f} max={stats['lat_absmax']:.2f}] "
                      f"pred[mean={stats['pred_mean']:+.4f} std={stats['pred_std']:.4f} max={stats['pred_absmax']:.2f}]")

    print(f"\n  Final latents: mean={latents.float().mean():.4f}, std={latents.float().std():.4f}, absmax={latents.float().abs().max():.2f}")

    # Check for explosion/collapse
    final_std = latents.float().std().item()
    if final_std > 10:
        print(f"  *** WARNING: Latents EXPLODED (std={final_std:.2f}) ***")
    elif final_std < 0.01:
        print(f"  *** WARNING: Latents COLLAPSED (std={final_std:.4f}) ***")
    else:
        print(f"  Latents seem reasonable (std={final_std:.4f})")

    return latents.cpu()


def test_vae_decode(bundle, latents):
    """TEST 5: Decode latents with VAE."""
    print("\n" + "="*80)
    print("TEST 5: VAE DECODE + POST-PROCESS")
    print("="*80)

    device = torch.device(DEVICE)

    # Move VAE to device (it's small enough)
    vae = bundle.vae.to(device)

    latents = latents.to(device)

    # Denormalize latents
    latents_mean = bundle.latents_mean.to(latents)
    latents_std = bundle.latents_std.to(latents)
    print(f"  latents_mean: {latents_mean.flatten().tolist()[:4]}...")
    print(f"  latents_std: {latents_std.flatten().tolist()[:4]}...")
    print(f"  Latents before denorm: mean={latents.float().mean():.4f}, std={latents.float().std():.4f}")

    latents_denorm = latents * latents_std + latents_mean
    print(f"  Latents after denorm: mean={latents_denorm.float().mean():.4f}, std={latents_denorm.float().std():.4f}")

    # VAE decode
    with torch.no_grad(), torch.autocast('cuda', enabled=False):
        motion = vae.decode(latents_denorm.float())

    print(f"\n  VAE output shape: {motion.shape}")  # Should be [B, T, J, 6]
    print(f"  VAE output: mean={motion.mean():.4f}, std={motion.std():.4f}, absmax={motion.abs().max():.2f}")

    # Denormalize motion
    smpl_processor = bundle.smpl_pose_processor
    x_dec = rearrange(motion, "b t j d -> b t (j d)")
    print(f"  Motion (before denorm): shape={x_dec.shape}, mean={x_dec.mean():.4f}, std={x_dec.std():.4f}")

    x_dec_denorm = smpl_processor.denormalize(x_dec)
    print(f"  Motion (after denorm): mean={x_dec_denorm.mean():.4f}, std={x_dec_denorm.std():.4f}")

    # Check translation
    transl_abs_rel = x_dec_denorm[..., :6]
    print(f"\n  Translation (abs_rel): mean={transl_abs_rel.mean():.4f}, std={transl_abs_rel.std():.4f}")

    # Check rotation values
    pred_poses = x_dec_denorm[..., 6:]
    print(f"  Rotation (rot6d): mean={pred_poses.mean():.4f}, std={pred_poses.std():.4f}")

    # Convert rotation to check validity
    pred_poses_r = rearrange(pred_poses, "b t (j d) -> (b t) j d", d=6)
    # Row-major -> column-major
    pred_poses_r = pred_poses_r[..., [0, 2, 4, 1, 3, 5]]

    # Check if rot6d values are in valid range (should be close to orthonormal)
    # For valid rotations, each 3-vector should have norm close to 1
    col1 = pred_poses_r[..., :3]
    col2 = pred_poses_r[..., 3:]
    norm1 = col1.norm(dim=-1)
    norm2 = col2.norm(dim=-1)
    dot_product = (col1 * col2).sum(dim=-1)

    print(f"\n  Rotation validity check:")
    print(f"    Col1 norm: mean={norm1.mean():.4f}, std={norm1.std():.4f} (should be ~1.0)")
    print(f"    Col2 norm: mean={norm2.mean():.4f}, std={norm2.std():.4f} (should be ~1.0)")
    print(f"    Dot product (orthogonality): mean={dot_product.mean():.4f}, std={dot_product.std():.4f} (should be ~0.0)")

    # Convert to axis-angle
    aa = rotation_6d_to_axis_angle(pred_poses_r)
    aa_angles = aa.norm(dim=-1)
    print(f"\n  Axis-angle magnitudes: mean={aa_angles.mean():.4f}, std={aa_angles.std():.4f}, max={aa_angles.max():.4f}")
    print(f"  (For walking: expect mean~0.1-0.5, max<3.14)")

    if aa_angles.max() > 3.14:
        print(f"  *** WARNING: Some rotations exceed pi ({aa_angles.max():.2f}) - likely invalid ***")

    return motion


def test_vae_roundtrip(bundle):
    """TEST 6: VAE encode -> decode round-trip with real data."""
    print("\n" + "="*80)
    print("TEST 6: VAE ROUND-TRIP (synthetic known motion)")
    print("="*80)

    device = torch.device(DEVICE)
    vae = bundle.vae.to(device)

    # Create a simple known motion: all zeros (T-pose) with small perturbation
    T = 33  # frames
    J = 23  # joints
    D = 6   # rot6d

    # Identity rotation in column-major 6d: [1,0,0, 0,1,0]
    # But training uses row-major: [1,0, 0,1, 0,0]
    # Row-major means [R00,R01, R10,R11, R20,R21] for identity = [1,0, 0,1, 0,0]
    identity_rot6d = torch.zeros(1, T, J, D)
    identity_rot6d[..., 0] = 1.0  # R00 = 1
    identity_rot6d[..., 3] = 1.0  # R11 = 1

    # Normalize like training does
    motion_flat = rearrange(identity_rot6d, 'b t j d -> b t (j d)').to(device)
    motion_norm = bundle.smpl_pose_processor.normalize(motion_flat)
    motion_norm_4d = rearrange(motion_norm, 'b t (j d) -> b t j d', d=6)

    print(f"  Synthetic motion: shape={identity_rot6d.shape}")
    print(f"  After normalize: mean={motion_norm_4d.mean():.4f}, std={motion_norm_4d.std():.4f}")

    # VAE encode (returns 2*z_dim channels: mean + logvar)
    with torch.no_grad(), torch.autocast('cuda', enabled=False):
        latents_raw = vae.encode(motion_norm_4d.float())

    print(f"  VAE raw encode output: shape={latents_raw.shape}")
    # Take mode (mean) from the distribution - this gives z_dim=16 channels
    latents = DiagonalGaussianDistributionNd(latents_raw).mode()
    print(f"  VAE latents (after mode): shape={latents.shape}, mean={latents.mean():.4f}, std={latents.std():.4f}")

    # Normalize latents (like training)
    latents_mean = bundle.latents_mean.to(latents)
    latents_std = bundle.latents_std.to(latents)
    latents_norm = (latents - latents_mean) / latents_std
    print(f"  Normalized latents: mean={latents_norm.mean():.4f}, std={latents_norm.std():.4f}")

    # Denormalize and decode
    latents_denorm = latents_norm * latents_std + latents_mean
    with torch.no_grad(), torch.autocast('cuda', enabled=False):
        reconstructed = vae.decode(latents_denorm.float())

    print(f"  Reconstructed: shape={reconstructed.shape}, mean={reconstructed.mean():.4f}, std={reconstructed.std():.4f}")

    # Check reconstruction error
    # Note: temporal dimension may change due to VAE downsampling
    min_t = min(motion_norm_4d.shape[1], reconstructed.shape[1])
    error = (motion_norm_4d[:, :min_t] - reconstructed[:, :min_t]).abs()
    print(f"  Reconstruction error (first {min_t} frames): mean={error.mean():.6f}, max={error.max():.4f}")

    if error.mean() > 0.1:
        print(f"  *** WARNING: High reconstruction error - VAE may have issues ***")
    else:
        print(f"  VAE round-trip looks good (error < 0.1)")

    return reconstructed


def main():
    print("="*80)
    print("PRISM INFERENCE DIAGNOSTIC (Memory-Efficient)")
    print("="*80)
    print(f"Config: {CONFIG}")
    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Device: {DEVICE}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Build bundle (all on CPU)
    print("\nBuilding bundle...")
    t0 = time.time()
    bundle = build_bundle()
    print(f"  Built in {time.time()-t0:.1f}s")

    # TEST 1: Scheduler timesteps
    test_scheduler_timesteps(bundle)

    # TEST 2: Text encoding
    text_states_256, neg_states_256, text_states_128, neg_states_128 = test_text_encoding(bundle)

    # TEST 6: VAE round-trip (before loading transformer)
    try:
        test_vae_roundtrip(bundle)
    except Exception as e:
        print(f"  TEST 6 FAILED: {e}")
        import traceback; traceback.print_exc()

    # Free VAE from GPU
    bundle.vae = bundle.vae.cpu()
    gc.collect()
    torch.cuda.empty_cache()

    # TEST 3: Transformer forward pass (with 256 text)
    test_transformer_forward(bundle, text_states_256, neg_states_256, text_len_label="256")

    # TEST 4a: Full denoising with cfg=5.0, text=256 (matching training)
    latents_cfg5_256 = test_denoising_loop(
        bundle, text_states_256, neg_states_256,
        guidance_scale=5.0, use_fp32=False, text_len_label="256"
    )

    # TEST 4b: Full denoising with cfg=1.0 (no CFG)
    latents_nocfg_256 = test_denoising_loop(
        bundle, text_states_256, neg_states_256,
        guidance_scale=1.0, use_fp32=False, text_len_label="256"
    )

    # TEST 4c: Full denoising with cfg=5.0, text=128 (current broken setting)
    latents_cfg5_128 = test_denoising_loop(
        bundle, text_states_128, neg_states_128,
        guidance_scale=5.0, use_fp32=False, text_len_label="128"
    )

    # Compare latent statistics
    print("\n" + "="*80)
    print("COMPARISON: DENOISED LATENTS")
    print("="*80)
    print(f"  cfg=5.0 text=256: mean={latents_cfg5_256.float().mean():.4f}, std={latents_cfg5_256.float().std():.4f}")
    print(f"  cfg=1.0 text=256: mean={latents_nocfg_256.float().mean():.4f}, std={latents_nocfg_256.float().std():.4f}")
    print(f"  cfg=5.0 text=128: mean={latents_cfg5_128.float().mean():.4f}, std={latents_cfg5_128.float().std():.4f}")

    # Free transformer
    bundle.transformer = bundle.transformer.cpu()
    gc.collect()
    torch.cuda.empty_cache()

    # TEST 5: Decode the best latents
    print("\n--- Decoding cfg=5.0, text=256 latents ---")
    test_vae_decode(bundle, latents_cfg5_256)

    print("\n--- Decoding cfg=1.0, text=256 latents ---")
    test_vae_decode(bundle, latents_nocfg_256)

    print("\n" + "="*80)
    print("DIAGNOSTIC COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
