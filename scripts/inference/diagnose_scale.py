"""Comprehensive diagnostic: compare training forward pass vs inference.

Goals:
1. Reproduce the exact training forward pass and verify loss matches reported ~0.16
2. Check model prediction vs target statistics at MULTIPLE timesteps
3. Identify if the underscaling is timestep-dependent
4. Test inference denoising step-by-step to find where variance collapses
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F
import numpy as np
from mmengine.config import Config
from einops import rearrange

import hftrainer
from hftrainer.registry import MODEL_BUNDLES
from hftrainer.utils.checkpoint_utils import load_checkpoint


def load_bundle(config_path, ckpt_path, device='cuda'):
    cfg = Config.fromfile(config_path)
    model_cfg = cfg.model.to_dict() if hasattr(cfg.model, 'to_dict') else cfg.model
    bundle_cls = MODEL_BUNDLES.get(model_cfg['type'])
    bundle = bundle_cls.from_config(model_cfg)
    bundle.eval()
    state_dict = load_checkpoint(ckpt_path, map_location='cpu')
    bundle.load_state_dict_selective(state_dict)
    del state_dict
    bundle = bundle.to(device)
    return bundle


@torch.no_grad()
def test_training_forward_pass(bundle, device):
    """Replicate EXACT training forward pass to verify model works correctly."""
    print("\n" + "="*80)
    print("TEST 1: Replicate Training Forward Pass")
    print("="*80)

    # Create synthetic latents (similar to what VAE would produce)
    batch_size = 2
    latent_channels = 16
    latent_frames = 9  # (33 - 1) // 4 + 1
    latent_joints = 23

    # Use real-ish latent distribution: the normalized latents should be roughly N(0,1)
    torch.manual_seed(42)
    latents = torch.randn(batch_size, latent_channels, latent_frames, latent_joints, device=device)

    # Create masks (no conditioning, no padding - most common training case: 90%)
    condition_frame_mask_vae = torch.ones(batch_size, 1, latent_frames, latent_joints,
                                          dtype=torch.bool, device=device)
    padding_mask = torch.ones(batch_size, latent_frames, latent_joints, device=device)

    # Encode a dummy prompt (use training's max_text_length=256)
    text_states = bundle.encode_prompt(
        ["a person walks forward", "a person raises hands"],
        max_sequence_length=256,
        prompt_drop_rate=0.0,
        dtype=next(bundle.transformer.parameters()).dtype,
    )

    transformer_dtype = next(bundle.transformer.parameters()).dtype

    # Test at multiple timesteps
    scheduler_timesteps = bundle.scheduler.timesteps.to(device)
    print(f"\nScheduler has {len(scheduler_timesteps)} timesteps")
    print(f"Timestep range: [{scheduler_timesteps.min().item():.2f}, {scheduler_timesteps.max().item():.2f}]")

    test_indices = [0, 100, 250, 500, 750, 900, 999]  # Low to high noise

    results = []
    for idx in test_indices:
        if idx >= len(scheduler_timesteps):
            continue
        timesteps = scheduler_timesteps[idx].unsqueeze(0).expand(batch_size)

        # Add noise (EXACT training logic)
        noisy_latents, targets = bundle.add_flow_noise(latents, timesteps)

        # Apply condition mask (no conditioning in this test)
        noisy_latents_masked = torch.where(condition_frame_mask_vae, noisy_latents, latents)

        # Create per-token timesteps
        seq_ts = bundle.create_sequence_ts(
            timesteps,
            condition_frame_mask_vae,
            bundle.transformer.config.patch_size,
        )

        # Forward pass (EXACT training logic)
        model_pred = bundle.transformer(
            hidden_states=noisy_latents_masked.to(dtype=transformer_dtype),
            encoder_hidden_states=text_states,
            timestep=seq_ts,
            hidden_states_mask=None,  # Training passes None when no padding
            encoder_hidden_states_mask=None,
        ).float()

        # Compute loss (EXACT training logic)
        mse = F.mse_loss(model_pred, targets.float(), reduction='none')
        condition_mask = condition_frame_mask_vae.expand_as(mse).float()
        pad_mask = padding_mask.unsqueeze(1).expand_as(mse).float()
        full_mask = condition_mask * pad_mask
        loss = (mse * full_mask).sum() / (full_mask.sum() + 1e-6)

        pred_std = model_pred.std().item()
        target_std = targets.float().std().item()
        cos_sim = F.cosine_similarity(
            model_pred.flatten(1), targets.float().flatten(1), dim=1
        ).mean().item()

        sigma_val = bundle.scheduler.sigmas[idx].item() if idx < len(bundle.scheduler.sigmas) else -1

        results.append({
            'idx': idx,
            'timestep': timesteps[0].item(),
            'sigma': sigma_val,
            'loss': loss.item(),
            'pred_std': pred_std,
            'target_std': target_std,
            'scale_ratio': pred_std / (target_std + 1e-8),
            'cos_sim': cos_sim,
        })

        print(f"  t_idx={idx:4d} | t={timesteps[0].item():7.2f} | σ={sigma_val:.4f} | "
              f"loss={loss.item():.4f} | pred_std={pred_std:.4f} | target_std={target_std:.4f} | "
              f"ratio={pred_std/(target_std+1e-8):.4f} | cos_sim={cos_sim:.4f}")

    return results


@torch.no_grad()
def test_inference_with_mask_vs_none(bundle, device):
    """Test if passing motion_mask (all-ones) vs None causes different outputs."""
    print("\n" + "="*80)
    print("TEST 2: hidden_states_mask=None vs hidden_states_mask=all-ones")
    print("="*80)

    transformer_dtype = next(bundle.transformer.parameters()).dtype
    batch_size = 1
    latent_channels = 16
    latent_frames = 9
    latent_joints = 23

    torch.manual_seed(123)
    latents = torch.randn(batch_size, latent_channels, latent_frames, latent_joints,
                          device=device, dtype=transformer_dtype)

    # Create per-token timestep (same as inference: all positions get same t)
    t_val = 500.0
    timestep = torch.full((batch_size, latent_frames * latent_joints), t_val,
                          device=device, dtype=transformer_dtype)

    # Text encoding
    text_states = bundle.encode_prompt(
        ["a person walks forward"],
        max_sequence_length=256,
        prompt_drop_rate=0.0,
        dtype=transformer_dtype,
    )

    # Forward with None mask
    pred_none = bundle.transformer(
        hidden_states=latents,
        encoder_hidden_states=text_states,
        timestep=timestep,
        hidden_states_mask=None,
        encoder_hidden_states_mask=None,
    ).float()

    # Forward with all-ones mask
    motion_mask = torch.ones(batch_size, latent_frames, latent_joints, device=device)
    pred_mask = bundle.transformer(
        hidden_states=latents,
        encoder_hidden_states=text_states,
        timestep=timestep,
        hidden_states_mask=motion_mask,
        encoder_hidden_states_mask=None,
    ).float()

    max_diff = (pred_none - pred_mask).abs().max().item()
    mean_diff = (pred_none - pred_mask).abs().mean().item()

    print(f"  Max absolute difference: {max_diff:.8f}")
    print(f"  Mean absolute difference: {mean_diff:.8f}")
    print(f"  pred_none std: {pred_none.std().item():.6f}")
    print(f"  pred_mask std: {pred_mask.std().item():.6f}")

    if max_diff > 0.01:
        print("  *** SIGNIFICANT DIFFERENCE DETECTED! ***")
    else:
        print("  OK: Mask vs None produces identical results")


@torch.no_grad()
def test_text_length_effect(bundle, device):
    """Test if text encoding length (128 vs 256) affects model output scale."""
    print("\n" + "="*80)
    print("TEST 3: Text Length Effect (128 vs 256)")
    print("="*80)

    transformer_dtype = next(bundle.transformer.parameters()).dtype
    batch_size = 1
    latent_channels = 16
    latent_frames = 9
    latent_joints = 23

    torch.manual_seed(456)
    latents = torch.randn(batch_size, latent_channels, latent_frames, latent_joints,
                          device=device, dtype=transformer_dtype)

    t_val = 500.0
    timestep = torch.full((batch_size, latent_frames * latent_joints), t_val,
                          device=device, dtype=transformer_dtype)

    prompt = "a person walks forward slowly and then stops"

    # Encode with 256 tokens (training)
    text_256 = bundle.encode_prompt(
        [prompt],
        max_sequence_length=256,
        prompt_drop_rate=0.0,
        dtype=transformer_dtype,
    )

    # Encode with 128 tokens (inference script)
    text_128 = bundle.encode_prompt(
        [prompt],
        max_sequence_length=128,
        prompt_drop_rate=0.0,
        dtype=transformer_dtype,
    )

    print(f"  text_256 shape: {text_256.shape}, non-zero rows: {(text_256.abs().sum(-1) > 0).sum().item()}")
    print(f"  text_128 shape: {text_128.shape}, non-zero rows: {(text_128.abs().sum(-1) > 0).sum().item()}")

    # Forward with 256
    pred_256 = bundle.transformer(
        hidden_states=latents,
        encoder_hidden_states=text_256,
        timestep=timestep,
        hidden_states_mask=None,
        encoder_hidden_states_mask=None,
    ).float()

    # Forward with 128
    pred_128 = bundle.transformer(
        hidden_states=latents,
        encoder_hidden_states=text_128,
        timestep=timestep,
        hidden_states_mask=None,
        encoder_hidden_states_mask=None,
    ).float()

    max_diff = (pred_256 - pred_128).abs().max().item()
    mean_diff = (pred_256 - pred_128).abs().mean().item()
    cos_sim = F.cosine_similarity(pred_256.flatten(), pred_128.flatten(), dim=0).item()

    print(f"  pred_256 std: {pred_256.std().item():.6f}")
    print(f"  pred_128 std: {pred_128.std().item():.6f}")
    print(f"  Max absolute difference: {max_diff:.6f}")
    print(f"  Mean absolute difference: {mean_diff:.6f}")
    print(f"  Cosine similarity: {cos_sim:.6f}")
    print(f"  Scale ratio (128/256): {pred_128.std().item() / (pred_256.std().item() + 1e-8):.6f}")


@torch.no_grad()
def test_inference_denoising_trajectory(bundle, device):
    """Run inference and track latent statistics at every step."""
    print("\n" + "="*80)
    print("TEST 4: Inference Denoising Trajectory")
    print("="*80)

    from diffusers.utils.torch_utils import randn_tensor

    transformer_dtype = next(bundle.transformer.parameters()).dtype
    batch_size = 1
    latent_channels = 16
    num_frames = 33
    latent_joints = 23
    vae_temporal = bundle.vae.config.scale_factor_temporal
    latent_frames = (num_frames - 1) // vae_temporal + 1

    # Text encoding (use 256 to match training!)
    text_states = bundle.encode_prompt(
        ["a person walks forward"],
        max_sequence_length=256,
        prompt_drop_rate=0.0,
        dtype=transformer_dtype,
    )

    # Setup scheduler for inference
    num_inference_steps = 50
    bundle.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = bundle.scheduler.timesteps

    # Prepare latents
    shape = (batch_size, latent_channels, latent_frames, latent_joints)
    torch.manual_seed(789)
    latents = randn_tensor(shape, device=device, dtype=transformer_dtype)

    # No first frame conditioning
    first_frame_mask = torch.ones_like(latents)

    print(f"\n  Initial latents std: {latents.float().std().item():.4f}")
    print(f"  Num inference steps: {num_inference_steps}")
    print(f"  Timesteps range: [{timesteps[0].item():.2f}, {timesteps[-1].item():.2f}]")
    print(f"\n  Step | Timestep | Sigma | pred_std | latent_std | step_size")
    print(f"  {'─'*75}")

    for i, t in enumerate(timesteps):
        latent_model_input = latents.to(transformer_dtype)

        # Per-token timestep (all positions get same t since no conditioning)
        temp_ts = (first_frame_mask[0][0] * t).flatten()
        ts = temp_ts.unsqueeze(0).expand(batch_size, -1)

        # Forward
        noise_pred = bundle.transformer(
            hidden_states=latent_model_input,
            encoder_hidden_states=text_states,
            timestep=ts,
            hidden_states_mask=None,
            encoder_hidden_states_mask=None,
        ).float()

        # Get sigma for this step
        sigma = bundle.scheduler.sigmas[bundle.scheduler.step_index or i].item()
        sigma_next = bundle.scheduler.sigmas[(bundle.scheduler.step_index or i) + 1].item()
        step_size = sigma_next - sigma

        pred_std = noise_pred.std().item()

        # Scheduler step
        latents = bundle.scheduler.step(noise_pred.to(transformer_dtype), t, latents, return_dict=False)[0]

        latent_std = latents.float().std().item()

        if i < 5 or i >= num_inference_steps - 5 or i % 10 == 0:
            print(f"  {i:4d} | {t.item():8.2f} | {sigma:.4f} | {pred_std:.4f} | {latent_std:.4f} | {step_size:.5f}")

    print(f"\n  Final latents std: {latents.float().std().item():.4f}")
    print(f"  Expected std for clean signal: ~1.0 (normalized latents)")

    # Reset scheduler for bundle reuse
    bundle.scheduler.set_timesteps(bundle.scheduler.config.num_train_timesteps, device=device)

    return latents


@torch.no_grad()
def test_single_step_detail(bundle, device):
    """Detailed analysis of a single denoising step at a specific noise level.

    Key insight: at sigma=σ, the noisy sample is x_t = (1-σ)x_0 + σε.
    The target velocity is v = ε - x_0.
    After one step: x_{t-1} = x_t + (σ_next - σ) * v_pred

    If v_pred is underscaled by factor α, then:
    x_{t-1} = x_t + (σ_next - σ) * α * v_true

    This test checks what α is at different noise levels.
    """
    print("\n" + "="*80)
    print("TEST 5: Detailed Single-Step Analysis")
    print("="*80)

    transformer_dtype = next(bundle.transformer.parameters()).dtype
    batch_size = 1
    latent_channels = 16
    latent_frames = 9
    latent_joints = 23

    # Use real normalized latents
    torch.manual_seed(100)
    x_0 = torch.randn(batch_size, latent_channels, latent_frames, latent_joints, device=device)
    noise = torch.randn_like(x_0)

    # Text
    text_states = bundle.encode_prompt(
        ["a person walks forward"],
        max_sequence_length=256,
        prompt_drop_rate=0.0,
        dtype=transformer_dtype,
    )

    # Test at specific sigma values
    scheduler_timesteps = bundle.scheduler.timesteps.to(device)
    scheduler_sigmas = bundle.scheduler.sigmas.to(device)

    test_indices = [50, 200, 400, 600, 800, 950]

    print(f"\n  {'Index':>6} | {'Timestep':>9} | {'Sigma':>7} | {'pred_std':>9} | {'target_std':>10} | "
          f"{'ratio':>6} | {'cos_sim':>8} | {'pred·target':>11}")
    print(f"  {'─'*90}")

    for idx in test_indices:
        if idx >= len(scheduler_timesteps):
            continue

        t = scheduler_timesteps[idx]
        sigma = scheduler_sigmas[idx]

        # Create noisy sample
        x_t = (1 - sigma) * x_0 + sigma * noise
        target = noise - x_0

        # Create per-token timestep
        condition_mask = torch.ones(batch_size, 1, latent_frames, latent_joints,
                                    dtype=torch.bool, device=device)
        seq_ts = bundle.create_sequence_ts(
            t.unsqueeze(0).expand(batch_size),
            condition_mask,
            bundle.transformer.config.patch_size,
        )

        # Forward
        pred = bundle.transformer(
            hidden_states=x_t.to(transformer_dtype),
            encoder_hidden_states=text_states,
            timestep=seq_ts,
            hidden_states_mask=None,
            encoder_hidden_states_mask=None,
        ).float()

        pred_std = pred.std().item()
        target_std = target.float().std().item()
        ratio = pred_std / (target_std + 1e-8)
        cos_sim = F.cosine_similarity(pred.flatten(), target.float().flatten(), dim=0).item()
        dot_product = (pred.flatten() * target.float().flatten()).mean().item()

        print(f"  {idx:6d} | {t.item():9.2f} | {sigma.item():.5f} | {pred_std:9.4f} | {target_std:10.4f} | "
              f"{ratio:6.4f} | {cos_sim:8.4f} | {dot_product:11.4f}")


@torch.no_grad()
def test_scheduler_sigma_format(bundle, device):
    """Verify that scheduler timesteps and sigmas are correctly formatted."""
    print("\n" + "="*80)
    print("TEST 6: Scheduler Format Verification")
    print("="*80)

    # Training scheduler (1000 steps)
    bundle.scheduler.set_timesteps(1000, device=device)
    train_ts = bundle.scheduler.timesteps
    train_sigmas = bundle.scheduler.sigmas

    print(f"  Training schedule (1000 steps):")
    print(f"    timesteps shape: {train_ts.shape}")
    print(f"    timesteps range: [{train_ts[-1].item():.4f}, {train_ts[0].item():.4f}]")
    print(f"    sigmas shape: {train_sigmas.shape}")
    print(f"    sigmas range: [{train_sigmas[-1].item():.6f}, {train_sigmas[0].item():.6f}]")
    print(f"    First 5 timesteps: {train_ts[:5].tolist()}")
    print(f"    Last 5 timesteps: {train_ts[-5:].tolist()}")
    print(f"    First 5 sigmas: {train_sigmas[:5].tolist()}")

    # Inference scheduler (50 steps)
    bundle.scheduler.set_timesteps(50, device=device)
    infer_ts = bundle.scheduler.timesteps
    infer_sigmas = bundle.scheduler.sigmas

    print(f"\n  Inference schedule (50 steps):")
    print(f"    timesteps shape: {infer_ts.shape}")
    print(f"    timesteps range: [{infer_ts[-1].item():.4f}, {infer_ts[0].item():.4f}]")
    print(f"    sigmas shape: {infer_sigmas.shape}")
    print(f"    sigmas range: [{infer_sigmas[-1].item():.6f}, {infer_sigmas[0].item():.6f}]")
    print(f"    First 5 timesteps: {infer_ts[:5].tolist()}")
    print(f"    Last 5 timesteps: {infer_ts[-5:].tolist()}")
    print(f"    First 5 sigmas: {infer_sigmas[:5].tolist()}")

    # Verify: timestep = sigma * num_train_timesteps?
    print(f"\n  Verification: timestep ≈ sigma * 1000?")
    print(f"    ts[0] = {infer_ts[0].item():.4f}, sigma[0] * 1000 = {infer_sigmas[0].item() * 1000:.4f}")
    print(f"    Match: {abs(infer_ts[0].item() - infer_sigmas[0].item() * 1000) < 1.0}")

    # Reset to training
    bundle.scheduler.set_timesteps(1000, device=device)


def main():
    config_path = "configs/prism/prism_1b_tp2m_multiframe.py"
    ckpt_path = "work_dirs/prism_1b_tp2m_multiframe/checkpoint-iter_15000"
    device = 'cuda'

    print("Loading bundle...")
    bundle = load_bundle(config_path, ckpt_path, device)
    print("Bundle loaded successfully.")

    # Run all tests
    test_scheduler_sigma_format(bundle, device)
    test_training_forward_pass(bundle, device)
    test_inference_with_mask_vs_none(bundle, device)
    test_text_length_effect(bundle, device)
    test_single_step_detail(bundle, device)
    test_inference_denoising_trajectory(bundle, device)

    print("\n" + "="*80)
    print("ALL TESTS COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
