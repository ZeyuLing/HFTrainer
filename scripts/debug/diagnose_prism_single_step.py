"""Diagnose PRISM overfit experiment: single-step prediction accuracy + error accumulation.

This script answers:
1. Does the model predict velocity accurately on training data? (single-step MSE)
2. How well can we recover x_0 from a single denoising step? (one-step x_0 recovery)
3. How much error accumulates across multi-step denoising? (per-step tracking)
4. Is there a training vs inference distribution mismatch?

Key insight: Training loss ~0.06-0.08 but inference L2 ~2.17 (near random).
If single-step is accurate, the bug is in multi-step denoising.
If single-step is inaccurate, the bug is in model loading or data pipeline.

Usage:
    python scripts/debug/diagnose_prism_single_step.py \
        --config work_dirs/prism_overfit_100/20260526_212303/config.py \
        --checkpoint work_dirs/prism_overfit_100/checkpoint-epoch_299 \
        --sample-idx 0
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def load_model(config_path: str, checkpoint_path: str, device: str = 'cuda:0'):
    """Load PRISM model from config and checkpoint (same as eval script)."""
    import hftrainer  # noqa: F401
    from hftrainer.registry import MODEL_BUNDLES
    from mmengine import Config

    cfg = Config.fromfile(config_path)
    model_cfg = cfg.model.copy()
    bundle = MODEL_BUNDLES.build(model_cfg)

    ckpt_path = os.path.join(checkpoint_path, 'model.pt') if os.path.isdir(checkpoint_path) else checkpoint_path
    if os.path.exists(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        if 'transformer' in state_dict and isinstance(state_dict['transformer'], dict):
            transformer_sd = state_dict['transformer']
            missing, unexpected = bundle.transformer.load_state_dict(transformer_sd, strict=False)
            print(f"Loaded {len(transformer_sd)} transformer keys from {ckpt_path}")
            if missing:
                print(f"  Missing keys: {len(missing)} (e.g. {missing[:3]})")
            if unexpected:
                print(f"  Unexpected keys: {len(unexpected)} (e.g. {unexpected[:3]})")
        if '__bundle_params__' in state_dict:
            bundle_params = state_dict['__bundle_params__']
            for name, val in bundle_params.items():
                if hasattr(bundle, name):
                    param = getattr(bundle, name)
                    if isinstance(param, torch.nn.Parameter):
                        param.data.copy_(val)
                    elif isinstance(param, torch.Tensor):
                        setattr(bundle, name, val)
                    print(f"  Loaded bundle param: {name} {val.shape}")
    else:
        print(f"WARNING: Checkpoint not found at {ckpt_path}")

    bundle = bundle.to(device)
    bundle.eval()
    return bundle, cfg


def load_training_sample(cfg, sample_idx: int, device: str = 'cuda:0'):
    """Load a single training sample through the EXACT same pipeline as training.

    Returns:
        motion: raw motion tensor [1, T, 138] on device
        text_embeds: [1, max_seq_length, 4096] on device
        text_mask: [1, max_seq_length] on device
        num_frames: actual frame count
        sample_key: sample identifier
    """
    anno_file = cfg.train_dataloader.dataset.anno_file
    with open(anno_file) as f:
        anno = json.load(f)

    data_list = anno['data_list']
    sample_keys = list(data_list.keys())
    key = sample_keys[sample_idx]
    sample = data_list[key]

    print(f"Sample: {key}")
    print(f"  Caption path: {sample.get('hierarchical_caption_path', 'N/A')}")
    print(f"  SMPLX path: {sample.get('smplx_path', 'N/A')}")
    print(f"  Duration: {sample.get('duration', 0):.2f}s")

    # Load motion through the same transform as training
    from hftrainer.datasets.motion.motionhub.transforms.load_smplx import LoadSmplx55

    data_dir = cfg.train_dataloader.dataset.data_dir
    smplx_path = sample.get('smplx_path', '')
    motion_path = os.path.join(data_dir, smplx_path)

    loader = LoadSmplx55(
        key='motion',
        rot_type='rotation_6d',
        transl_type='abs_rel',
        smpl_type='smpl_22',
        rot6d_convention='column',
        transl_aug_prob=0.0,
    )
    results = {
        'motion_path': motion_path,
        'num_frames': sample.get('num_frames', int(sample.get('duration', 0) * 30)),
        'fps': sample.get('fps', 30),
    }
    results = loader.transform(results)
    motion = results['motion']  # [T, 138]
    num_frames = results['num_frames']

    # Crop/pad to clip_len=360 (same as training RandomCropPadding)
    clip_len = 360
    T = motion.shape[0]
    if T > clip_len:
        motion = motion[:clip_len]
        num_frames = min(num_frames, clip_len)
    elif T < clip_len:
        # Replicate pad (same as training)
        pad_len = clip_len - T
        motion = torch.cat([motion, motion[-1:].expand(pad_len, -1)], dim=0)

    motion = motion.unsqueeze(0).to(device)  # [1, 360, 138]

    # Load T5 features
    caption_path = sample.get('hierarchical_caption_path', '')
    feature_dir = 'data/t5_feature'
    max_seq_length = 256

    full_path = os.path.normpath(os.path.join(data_dir, caption_path))
    norm_data_dir = os.path.normpath(data_dir)
    data_parent = os.path.dirname(norm_data_dir)

    if full_path.startswith(data_parent + '/'):
        rel_path = full_path[len(data_parent) + 1:]
    else:
        rel_path = full_path

    data_dir_basename = os.path.basename(norm_data_dir)
    if rel_path.startswith(data_dir_basename + '/'):
        rel_path = rel_path[len(data_dir_basename) + 1:]

    if rel_path.endswith('.json'):
        rel_path = rel_path[:-5] + '.pt'

    pt_path = os.path.join(feature_dir, rel_path)
    data = torch.load(pt_path, map_location='cpu', weights_only=False)
    embeddings = data['embeddings']
    captions = data['captions']
    seq_lens = data['seq_lens']

    idx = 0  # First variant (deterministic)
    emb = embeddings[idx]  # [seq_len, 4096]
    seq_len = seq_lens[idx]
    caption = captions[idx] if idx < len(captions) else ''

    if emb.size(0) < max_seq_length:
        pad = torch.zeros(max_seq_length - emb.size(0), emb.size(1), dtype=emb.dtype)
        emb = torch.cat([emb, pad], dim=0)
    elif emb.size(0) > max_seq_length:
        emb = emb[:max_seq_length]
        seq_len = min(seq_len, max_seq_length)

    mask = torch.zeros(max_seq_length, dtype=torch.long)
    mask[:seq_len] = 1

    text_embeds = emb.unsqueeze(0).to(device)  # [1, 256, 4096]
    text_mask = mask.unsqueeze(0).to(device)  # [1, 256]

    print(f"  Caption: '{caption[:80]}...'")
    print(f"  Motion shape: {motion.shape}, num_frames: {num_frames}")
    print(f"  Text embeds shape: {text_embeds.shape}, text mask sum: {text_mask.sum().item()}")

    return motion, text_embeds, text_mask, num_frames, key


@torch.no_grad()
def test_single_step_prediction(bundle, motion, text_embeds, text_mask, num_frames, device):
    """Test 1: Single-step velocity prediction accuracy.

    Exactly replicates training: encode → add_flow_noise → transformer forward → compare to target.
    """
    print("\n" + "="*80)
    print("TEST 1: Single-step velocity prediction (replicating training)")
    print("="*80)

    # Step 1: Encode motion to latent space (same as training)
    latents = bundle.encode_motion(motion)
    batch_size, C, T_lat, J = latents.shape
    print(f"\n  Latents shape: {latents.shape}")
    print(f"  Latents stats: mean={latents.mean():.4f}, std={latents.std():.4f}, "
          f"min={latents.min():.4f}, max={latents.max():.4f}")

    # Step 2: Create padding mask (same as training)
    num_frames_tensor = torch.tensor([num_frames], device=device)
    padding_mask = bundle.create_padding_mask(
        num_frames=num_frames_tensor,
        batch_size=batch_size,
        latent_frames=T_lat,
        latent_joints=J,
        device=device,
    )

    # Step 3: Create ALL-GENERATE condition mask (50% of training has this)
    condition_frame_mask_vae = torch.ones(batch_size, 1, T_lat, J, dtype=torch.bool, device=device)

    # Step 4: Test at multiple timestep levels
    transformer_module = getattr(bundle.transformer, 'module', bundle.transformer)
    patch_size = transformer_module.config.patch_size
    transformer_dtype = next(bundle.transformer.parameters()).dtype

    # Test at various sigma levels
    scheduler_timesteps = bundle.scheduler.timesteps.to(device=device)
    test_indices = [0, 50, 100, 200, 500, 700, 900, 999]  # From high-noise to low-noise

    print(f"\n  Testing {len(test_indices)} timestep levels (scheduler has {len(scheduler_timesteps)} steps):")
    print(f"  {'idx':>5} {'timestep':>10} {'sigma':>8} {'MSE(pred,tgt)':>14} {'RMSE':>8} "
          f"{'pred_norm':>10} {'tgt_norm':>10} {'cosine_sim':>11}")
    print(f"  {'-'*5} {'-'*10} {'-'*8} {'-'*14} {'-'*8} {'-'*10} {'-'*10} {'-'*11}")

    all_mse = []
    for step_idx in test_indices:
        if step_idx >= len(scheduler_timesteps):
            continue

        timesteps = scheduler_timesteps[step_idx:step_idx+1]  # [1]
        sigma = bundle.scheduler.sigmas[step_idx].item()

        # Add noise (same as training)
        noisy_latents, targets = bundle.add_flow_noise(latents, timesteps)

        # Apply condition mask (all-generate: noisy_latents unchanged)
        noisy_latents = torch.where(condition_frame_mask_vae, noisy_latents, latents)

        # Create per-token timesteps (same as training)
        ts_seq = bundle.create_sequence_ts(timesteps, condition_frame_mask_vae, patch_size)

        # Forward pass
        noisy_latents_input = noisy_latents.to(dtype=transformer_dtype)
        model_pred = bundle.transformer(
            hidden_states=noisy_latents_input,
            encoder_hidden_states=text_embeds.to(dtype=transformer_dtype),
            timestep=ts_seq,
            hidden_states_mask=padding_mask,
            encoder_hidden_states_mask=text_mask,
        )
        model_pred = model_pred.float()
        targets_float = targets.float()

        # Compute metrics
        mse = F.mse_loss(model_pred, targets_float).item()
        rmse = mse ** 0.5
        pred_norm = model_pred.norm().item() / model_pred.numel() ** 0.5
        tgt_norm = targets_float.norm().item() / targets_float.numel() ** 0.5

        # Cosine similarity (flatten to 1D)
        cos_sim = F.cosine_similarity(
            model_pred.flatten().unsqueeze(0),
            targets_float.flatten().unsqueeze(0)
        ).item()

        all_mse.append(mse)
        print(f"  {step_idx:>5} {timesteps[0].item():>10.2f} {sigma:>8.4f} "
              f"{mse:>14.6f} {rmse:>8.4f} {pred_norm:>10.4f} {tgt_norm:>10.4f} {cos_sim:>11.6f}")

    mean_mse = np.mean(all_mse)
    print(f"\n  Mean MSE across timesteps: {mean_mse:.6f} (training loss was ~0.06-0.08)")
    print(f"  Conclusion: {'MATCH' if mean_mse < 0.15 else 'MISMATCH'} with training loss")

    return mean_mse


@torch.no_grad()
def test_x0_recovery(bundle, motion, text_embeds, text_mask, num_frames, device):
    """Test 2: One-step x_0 recovery accuracy.

    Given noisy x_t at timestep σ, model predicts v.
    x_0_est = x_t - σ * v_pred (since x_t = x_0 + σ*v, so x_0 = x_t - σ*v)
    """
    print("\n" + "="*80)
    print("TEST 2: One-step x_0 recovery (can model reconstruct the clean latent?)")
    print("="*80)

    latents = bundle.encode_motion(motion)
    batch_size, C, T_lat, J = latents.shape

    num_frames_tensor = torch.tensor([num_frames], device=device)
    padding_mask = bundle.create_padding_mask(
        num_frames=num_frames_tensor,
        batch_size=batch_size,
        latent_frames=T_lat,
        latent_joints=J,
        device=device,
    )
    condition_frame_mask_vae = torch.ones(batch_size, 1, T_lat, J, dtype=torch.bool, device=device)

    transformer_module = getattr(bundle.transformer, 'module', bundle.transformer)
    patch_size = transformer_module.config.patch_size
    transformer_dtype = next(bundle.transformer.parameters()).dtype

    scheduler_timesteps = bundle.scheduler.timesteps.to(device=device)
    test_indices = [0, 100, 300, 500, 700, 900, 999]

    print(f"\n  Recovery of clean latent x_0 from single model forward pass:")
    print(f"  Using: x_0_est = x_t - sigma * v_pred")
    print(f"  {'idx':>5} {'sigma':>8} {'MSE(x0_est,x0)':>16} {'rel_error':>12} {'x0_norm':>10}")
    print(f"  {'-'*5} {'-'*8} {'-'*16} {'-'*12} {'-'*10}")

    for step_idx in test_indices:
        if step_idx >= len(scheduler_timesteps):
            continue

        timesteps = scheduler_timesteps[step_idx:step_idx+1]
        sigma = bundle.scheduler.sigmas[step_idx].item()

        # Fix the random seed for reproducibility
        torch.manual_seed(42 + step_idx)
        noisy_latents, targets = bundle.add_flow_noise(latents, timesteps)
        noisy_latents = torch.where(condition_frame_mask_vae, noisy_latents, latents)

        ts_seq = bundle.create_sequence_ts(timesteps, condition_frame_mask_vae, patch_size)

        model_pred = bundle.transformer(
            hidden_states=noisy_latents.to(dtype=transformer_dtype),
            encoder_hidden_states=text_embeds.to(dtype=transformer_dtype),
            timestep=ts_seq,
            hidden_states_mask=padding_mask,
            encoder_hidden_states_mask=text_mask,
        ).float()

        # Recover x_0: since x_t = (1-σ)*x_0 + σ*noise, and v = noise - x_0
        # Then x_0 = x_t - σ*v (because x_t = x_0 + σ*(noise-x_0) = (1-σ)*x_0 + σ*noise)
        # Wait: v = noise - x_0. x_t = (1-σ)*x_0 + σ*noise = x_0 + σ*(noise - x_0) = x_0 + σ*v
        # Therefore: x_0 = x_t - σ*v
        x0_est = noisy_latents.float() - sigma * model_pred

        mse_x0 = F.mse_loss(x0_est, latents.float()).item()
        x0_norm = latents.float().norm().item() / latents.numel() ** 0.5
        rel_error = (mse_x0 ** 0.5) / (x0_norm + 1e-8)

        print(f"  {step_idx:>5} {sigma:>8.4f} {mse_x0:>16.6f} {rel_error:>12.4f} {x0_norm:>10.4f}")

    print(f"\n  Note: rel_error > 1 means x_0 estimate is worse than predicting zeros.")
    print(f"  For a perfectly memorized model, all MSE values should be near-zero.")


@torch.no_grad()
def test_full_denoising(bundle, motion, text_embeds, text_mask, num_frames,
                        num_inference_steps=50, device='cuda:0'):
    """Test 3: Full multi-step denoising with per-step error tracking.

    Run the EXACT same denoising loop as eval, but track per-step statistics.
    """
    print("\n" + "="*80)
    print(f"TEST 3: Full {num_inference_steps}-step denoising (tracking error accumulation)")
    print("="*80)

    # Encode ground truth to latent
    latents_gt = bundle.encode_motion(motion)
    batch_size, C, T_lat, J = latents_gt.shape

    transformer_module = getattr(bundle.transformer, 'module', bundle.transformer)
    patch_size = transformer_module.config.patch_size
    transformer_dtype = next(bundle.transformer.parameters()).dtype

    # All-generate condition mask
    condition_mask = torch.ones(batch_size, C, T_lat, J, dtype=torch.float32, device=device)

    # Setup scheduler for inference
    bundle.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = bundle.scheduler.timesteps

    # Start from fixed noise
    torch.manual_seed(42)
    latents = torch.randn_like(latents_gt)

    print(f"\n  GT latent stats: mean={latents_gt.mean():.4f}, std={latents_gt.std():.4f}")
    print(f"  Initial noise stats: mean={latents.mean():.4f}, std={latents.std():.4f}")
    print(f"  Scheduler timesteps: {timesteps[:5].tolist()}...{timesteps[-5:].tolist()}")
    print(f"  Scheduler sigmas: {bundle.scheduler.sigmas[:5].tolist()}...{bundle.scheduler.sigmas[-5:].tolist()}")

    print(f"\n  {'step':>5} {'t':>8} {'sigma':>8} {'dt':>10} "
          f"{'lat_mean':>9} {'lat_std':>8} {'pred_norm':>10} "
          f"{'MSE_vs_GT':>10} {'x0_est_MSE':>11}")
    print(f"  {'-'*5} {'-'*8} {'-'*8} {'-'*10} {'-'*9} {'-'*8} {'-'*10} {'-'*10} {'-'*11}")

    for i, t in enumerate(timesteps):
        noisy_latents = torch.where(condition_mask.bool(), latents, torch.zeros_like(latents))

        t_batch = t.unsqueeze(0).expand(batch_size)
        ts_expanded = bundle.create_sequence_ts(
            t_batch, condition_mask[:, :1].bool(), patch_size
        )

        # Single conditional forward (no CFG)
        model_pred = bundle.transformer(
            hidden_states=noisy_latents.to(dtype=transformer_dtype),
            encoder_hidden_states=text_embeds.to(dtype=transformer_dtype),
            timestep=ts_expanded,
            hidden_states_mask=None,
            encoder_hidden_states_mask=text_mask,
        ).float()

        # Compute per-step metrics BEFORE scheduler step
        pred_norm = model_pred.norm().item() / model_pred.numel() ** 0.5

        # x_0 estimate from current step: x_0 = x_t - σ*v
        sigma_current = bundle.scheduler.sigmas[i].item()
        x0_est = latents.float() - sigma_current * model_pred
        x0_est_mse = F.mse_loss(x0_est, latents_gt.float()).item()

        # Scheduler step
        step_output = bundle.scheduler.step(model_pred, t, latents)
        latents = step_output.prev_sample

        # Compute dt
        sigma_next = bundle.scheduler.sigmas[i + 1].item()
        dt = sigma_next - sigma_current

        # Distance to GT after step
        mse_vs_gt = F.mse_loss(latents, latents_gt.float()).item()

        if i % max(1, num_inference_steps // 20) == 0 or i == num_inference_steps - 1:
            print(f"  {i:>5} {t.item():>8.2f} {sigma_current:>8.4f} {dt:>10.6f} "
                  f"{latents.mean().item():>9.4f} {latents.std().item():>8.4f} "
                  f"{pred_norm:>10.4f} {mse_vs_gt:>10.4f} {x0_est_mse:>11.4f}")

    # Final comparison
    final_mse = F.mse_loss(latents, latents_gt.float()).item()
    print(f"\n  Final MSE (latent space): {final_mse:.6f}")
    print(f"  Final RMSE (latent space): {final_mse**0.5:.6f}")

    # Decode both and compare in motion space
    latents_denorm = latents * bundle.latents_std.to(latents) + bundle.latents_mean.to(latents)
    gt_denorm = latents_gt * bundle.latents_std.to(latents_gt) + bundle.latents_mean.to(latents_gt)

    device_type = latents_denorm.device.type
    with torch.autocast(device_type, enabled=False):
        pred_motion = bundle.vae.decode(latents_denorm.float())  # [B, T, J, C]
        gt_motion_decoded = bundle.vae.decode(gt_denorm.float())  # [B, T, J, C]

    # L2 in motion space (same metric as eval)
    T_valid = min(num_frames, pred_motion.shape[1], gt_motion_decoded.shape[1])
    pred_np = pred_motion[0, :T_valid].cpu().numpy()
    gt_np = gt_motion_decoded[0, :T_valid].cpu().numpy()

    l2_error = np.sqrt(((pred_np - gt_np) ** 2).sum(axis=-1))  # [T, J]
    mean_l2 = l2_error.mean()
    max_l2 = l2_error.max()

    print(f"\n  Motion space L2 error (decoded through VAE):")
    print(f"    Mean L2: {mean_l2:.4f}")
    print(f"    Max L2: {max_l2:.4f}")
    print(f"    (Eval reported: mean_l2=2.17 for 50 steps, guidance=1.0)")

    return final_mse, mean_l2


@torch.no_grad()
def test_vae_roundtrip(bundle, motion, num_frames, device):
    """Test 4: VAE encode-decode roundtrip accuracy.

    If VAE itself has errors, they'll compound with denoising errors.
    """
    print("\n" + "="*80)
    print("TEST 4: VAE encode-decode roundtrip (baseline reconstruction quality)")
    print("="*80)

    # Encode
    latents = bundle.encode_motion(motion)

    # Decode (denormalize then decode)
    latents_denorm = latents * bundle.latents_std.to(latents) + bundle.latents_mean.to(latents)
    device_type = latents_denorm.device.type
    with torch.autocast(device_type, enabled=False):
        reconstructed = bundle.vae.decode(latents_denorm.float())  # [B, T, J, C]

    # Compare to normalized input
    motion_normalized = bundle.smpl_pose_processor.normalize(motion.float())
    from einops import rearrange
    motion_reshaped = rearrange(motion_normalized, 'b t (j d) -> b t j d', d=6)

    T_valid = min(num_frames, reconstructed.shape[1], motion_reshaped.shape[1])
    pred_np = reconstructed[0, :T_valid].cpu().numpy()
    gt_np = motion_reshaped[0, :T_valid].cpu().numpy()

    l2_error = np.sqrt(((pred_np - gt_np) ** 2).sum(axis=-1))  # [T, J]
    mean_l2 = l2_error.mean()
    max_l2 = l2_error.max()

    print(f"\n  VAE roundtrip L2 error (encode → mode → decode):")
    print(f"    Mean L2: {mean_l2:.6f}")
    print(f"    Max L2: {max_l2:.6f}")
    print(f"    (This is the FLOOR — denoising cannot be better than this)")

    # Also check latent statistics
    print(f"\n  Latent statistics (post-normalization):")
    print(f"    Shape: {latents.shape}")
    print(f"    Mean: {latents.mean():.4f} (should be ~0)")
    print(f"    Std: {latents.std():.4f} (should be ~1)")
    print(f"    Per-channel mean: {latents.mean(dim=(0,2,3)).tolist()}")
    print(f"    Per-channel std: {latents.std(dim=(0,2,3)).tolist()}")

    return mean_l2


@torch.no_grad()
def test_few_steps(bundle, motion, text_embeds, text_mask, num_frames, device):
    """Test 5: Compare different numbers of inference steps.

    Tests: 1 step, 5 steps, 10 steps, 20 steps, 50 steps
    Shows how error accumulates with more steps.
    """
    print("\n" + "="*80)
    print("TEST 5: Comparing different numbers of inference steps")
    print("="*80)

    latents_gt = bundle.encode_motion(motion)
    batch_size, C, T_lat, J = latents_gt.shape

    transformer_module = getattr(bundle.transformer, 'module', bundle.transformer)
    patch_size = transformer_module.config.patch_size
    transformer_dtype = next(bundle.transformer.parameters()).dtype

    condition_mask = torch.ones(batch_size, C, T_lat, J, dtype=torch.float32, device=device)

    step_counts = [1, 2, 5, 10, 20, 50]

    print(f"\n  {'steps':>6} {'final_latent_MSE':>16} {'motion_mean_L2':>15} {'motion_max_L2':>14}")
    print(f"  {'-'*6} {'-'*16} {'-'*15} {'-'*14}")

    for n_steps in step_counts:
        bundle.scheduler.set_timesteps(n_steps, device=device)
        timesteps = bundle.scheduler.timesteps

        # Same starting noise
        torch.manual_seed(42)
        latents = torch.randn_like(latents_gt)

        for i, t in enumerate(timesteps):
            noisy_latents = torch.where(condition_mask.bool(), latents, torch.zeros_like(latents))
            t_batch = t.unsqueeze(0).expand(batch_size)
            ts_expanded = bundle.create_sequence_ts(
                t_batch, condition_mask[:, :1].bool(), patch_size
            )

            model_pred = bundle.transformer(
                hidden_states=noisy_latents.to(dtype=transformer_dtype),
                encoder_hidden_states=text_embeds.to(dtype=transformer_dtype),
                timestep=ts_expanded,
                hidden_states_mask=None,
                encoder_hidden_states_mask=text_mask,
            ).float()

            latents = bundle.scheduler.step(model_pred, t, latents).prev_sample

        # Latent MSE
        latent_mse = F.mse_loss(latents, latents_gt.float()).item()

        # Decode and compute motion L2
        latents_denorm = latents * bundle.latents_std.to(latents) + bundle.latents_mean.to(latents)
        gt_denorm = latents_gt * bundle.latents_std.to(latents_gt) + bundle.latents_mean.to(latents_gt)
        device_type = latents_denorm.device.type
        with torch.autocast(device_type, enabled=False):
            pred_motion = bundle.vae.decode(latents_denorm.float())
            gt_motion = bundle.vae.decode(gt_denorm.float())

        T_valid = min(num_frames, pred_motion.shape[1], gt_motion.shape[1])
        pred_np = pred_motion[0, :T_valid].cpu().numpy()
        gt_np = gt_motion[0, :T_valid].cpu().numpy()
        l2_error = np.sqrt(((pred_np - gt_np) ** 2).sum(axis=-1))

        print(f"  {n_steps:>6} {latent_mse:>16.6f} {l2_error.mean():>15.4f} {l2_error.max():>14.4f}")

    # Reset scheduler back
    bundle.scheduler.set_timesteps(bundle.scheduler.config.num_train_timesteps, device=device)


@torch.no_grad()
def test_deterministic_prediction(bundle, motion, text_embeds, text_mask, num_frames, device):
    """Test 6: Verify model gives same output for same input (determinism check).

    If the model gives different outputs for the same input, there's a stochasticity bug.
    """
    print("\n" + "="*80)
    print("TEST 6: Determinism check (same input → same output?)")
    print("="*80)

    latents = bundle.encode_motion(motion)
    batch_size, C, T_lat, J = latents.shape

    condition_frame_mask_vae = torch.ones(batch_size, 1, T_lat, J, dtype=torch.bool, device=device)

    transformer_module = getattr(bundle.transformer, 'module', bundle.transformer)
    patch_size = transformer_module.config.patch_size
    transformer_dtype = next(bundle.transformer.parameters()).dtype

    # Pick a specific timestep
    scheduler_timesteps = bundle.scheduler.timesteps.to(device=device)
    timesteps = scheduler_timesteps[500:501]

    # Fix noise
    torch.manual_seed(123)
    noisy_latents, targets = bundle.add_flow_noise(latents, timesteps)
    noisy_latents = torch.where(condition_frame_mask_vae, noisy_latents, latents)
    ts_seq = bundle.create_sequence_ts(timesteps, condition_frame_mask_vae, patch_size)

    # Forward pass 1
    pred1 = bundle.transformer(
        hidden_states=noisy_latents.to(dtype=transformer_dtype),
        encoder_hidden_states=text_embeds.to(dtype=transformer_dtype),
        timestep=ts_seq,
        hidden_states_mask=None,
        encoder_hidden_states_mask=text_mask,
    ).float()

    # Forward pass 2 (same input)
    pred2 = bundle.transformer(
        hidden_states=noisy_latents.to(dtype=transformer_dtype),
        encoder_hidden_states=text_embeds.to(dtype=transformer_dtype),
        timestep=ts_seq,
        hidden_states_mask=None,
        encoder_hidden_states_mask=text_mask,
    ).float()

    diff = (pred1 - pred2).abs().max().item()
    print(f"\n  Max absolute difference between two forward passes: {diff:.2e}")
    print(f"  Conclusion: {'DETERMINISTIC' if diff < 1e-5 else 'NON-DETERMINISTIC (BUG!)'}")


def main():
    parser = argparse.ArgumentParser(description='Diagnose PRISM single-step prediction')
    parser.add_argument('--config', type=str,
                       default='work_dirs/prism_overfit_100/20260526_212303/config.py',
                       help='Config file path')
    parser.add_argument('--checkpoint', type=str,
                       default='work_dirs/prism_overfit_100/checkpoint-epoch_299',
                       help='Checkpoint path')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--sample-idx', type=int, default=0,
                       help='Index of training sample to test')
    parser.add_argument('--num-inference-steps', type=int, default=50)
    args = parser.parse_args()

    # Disable fused SDP (V100 compatibility)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)

    print("="*80)
    print("PRISM OVERFIT DIAGNOSTIC: Single-Step Prediction + Error Accumulation")
    print("="*80)
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Sample index: {args.sample_idx}")

    # Load model
    print("\nLoading model...")
    bundle, cfg = load_model(args.config, args.checkpoint, args.device)

    # Load training sample
    print("\nLoading training sample...")
    motion, text_embeds, text_mask, num_frames, sample_key = load_training_sample(
        cfg, args.sample_idx, args.device
    )

    # Run tests
    test_deterministic_prediction(bundle, motion, text_embeds, text_mask, num_frames, args.device)
    vae_l2 = test_vae_roundtrip(bundle, motion, num_frames, args.device)
    mean_mse = test_single_step_prediction(bundle, motion, text_embeds, text_mask, num_frames, args.device)
    test_x0_recovery(bundle, motion, text_embeds, text_mask, num_frames, args.device)
    _, motion_l2 = test_full_denoising(bundle, motion, text_embeds, text_mask, num_frames,
                                        args.num_inference_steps, args.device)
    test_few_steps(bundle, motion, text_embeds, text_mask, num_frames, args.device)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"  VAE roundtrip L2: {vae_l2:.6f} (floor)")
    print(f"  Single-step MSE: {mean_mse:.6f} (vs training loss ~0.06-0.08)")
    print(f"  Full denoising L2: {motion_l2:.4f} (eval reports ~2.17)")

    if mean_mse > 0.5:
        print("\n  DIAGNOSIS: Model predictions are INACCURATE at single-step level.")
        print("  → Bug is in model loading, data pipeline, or forward pass.")
        print("  → Check: checkpoint loaded correctly? Same data processing as training?")
    elif motion_l2 > 1.0 and mean_mse < 0.15:
        print("\n  DIAGNOSIS: Single-step accurate but multi-step denoising fails.")
        print("  → Bug is in the denoising loop / scheduler integration.")
        print("  → Check: scheduler config mismatch? Error accumulation? Scheduler step sign?")
    else:
        print("\n  DIAGNOSIS: Both single-step and multi-step look reasonable.")
        print("  → Model may need more training or the L2 metric is noisy.")


if __name__ == '__main__':
    main()
