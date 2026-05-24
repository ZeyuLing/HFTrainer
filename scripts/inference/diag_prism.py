"""Diagnostic script to isolate PRISM inference deformation bug.

Tests:
1. VAE round-trip (encode real motion -> decode -> compare)
2. Transformer output magnitude check
3. Full denoising with stats per step
4. Compare max_text_length=128 vs 256
5. Test with official PrismPipeline
"""

import gc
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
from einops import rearrange
from mmengine.config import Config

import hftrainer  # noqa: trigger auto-imports
from hftrainer.registry import MODEL_BUNDLES
from hftrainer.utils.checkpoint_utils import load_checkpoint
from diffusers.utils.torch_utils import randn_tensor


def main():
    config_path = 'configs/prism/prism_1b_tp2m_multiframe.py'
    ckpt_path = 'work_dirs/prism_1b_tp2m_multiframe/checkpoint-iter_15000'
    device = torch.device('cuda')

    print("=" * 80)
    print("PRISM INFERENCE DIAGNOSTIC")
    print("=" * 80)

    # Build bundle
    cfg = Config.fromfile(config_path)
    model_cfg = cfg.model.to_dict() if hasattr(cfg.model, 'to_dict') else cfg.model
    bundle_cls = MODEL_BUNDLES.get(model_cfg['type'])
    bundle = bundle_cls.from_config(model_cfg)
    bundle.eval()

    # Load checkpoint
    print(f"\n[1] Loading checkpoint: {ckpt_path}")
    state_dict = load_checkpoint(ckpt_path, map_location='cpu')

    # Check what's in the state dict
    print(f"  State dict type: {type(state_dict)}")
    print(f"  Total keys: {len(state_dict)}")

    # Check for transformer keys
    transformer_keys = [k for k in state_dict.keys() if k.startswith('transformer.')]
    vae_keys = [k for k in state_dict.keys() if k.startswith('vae.')]
    text_enc_keys = [k for k in state_dict.keys() if k.startswith('text_encoder.')]
    bundle_keys = [k for k in state_dict.keys() if not any(k.startswith(p) for p in ['transformer.', 'vae.', 'text_encoder.'])]

    print(f"  Transformer keys: {len(transformer_keys)}")
    print(f"  VAE keys: {len(vae_keys)}")
    print(f"  Text encoder keys: {len(text_enc_keys)}")
    print(f"  Bundle-level keys: {bundle_keys}")

    # Load the state dict
    bundle.load_state_dict_selective(state_dict)
    del state_dict
    gc.collect()

    # Check latents_mean/std values
    print(f"\n  latents_mean: {bundle.latents_mean.flatten().tolist()}")
    print(f"  latents_std: {bundle.latents_std.flatten().tolist()}")

    # =========================================================================
    # TEST 1: VAE Round-Trip
    # =========================================================================
    print("\n" + "=" * 80)
    print("[TEST 1] VAE Round-Trip")
    print("=" * 80)

    # Create a synthetic "reasonable" motion: identity rotations + small translation
    # rot6d for identity rotation: [1,0,0,1,0,0] (first two columns of I)
    # After normalize, this should be near training distribution mean
    num_frames = 33
    num_joints = 23
    feat_dim = 6

    # Generate a simple "walking" motion in raw format
    # Use dummy normalized values near zero (training distribution center)
    dummy_motion = torch.randn(1, num_frames, num_joints * feat_dim) * 0.1  # Small values near mean
    dummy_motion = dummy_motion.float()

    # Move VAE to device
    bundle.vae = bundle.vae.to(device)

    # Encode
    dummy_norm = dummy_motion.to(device)
    dummy_reshaped = rearrange(dummy_norm, 'b t (j d) -> b t j d', d=6)

    with torch.no_grad(), torch.autocast('cuda', enabled=False):
        encoded = bundle.vae.encode(dummy_reshaped.float())

    from hftrainer.models.motion.prism.gaussian_distribution import DiagonalGaussianDistributionNd
    latents = DiagonalGaussianDistributionNd(encoded).mode()

    print(f"  Input motion shape: {dummy_reshaped.shape}")
    print(f"  Encoded latents shape: {latents.shape}")
    print(f"  Latents stats: mean={latents.mean():.4f}, std={latents.std():.4f}, "
          f"min={latents.min():.4f}, max={latents.max():.4f}")

    # Normalize latents (as done in training)
    latents_norm = (latents - bundle.latents_mean.to(latents)) / bundle.latents_std.to(latents)
    print(f"  Normalized latents stats: mean={latents_norm.mean():.4f}, std={latents_norm.std():.4f}")

    # Denormalize latents (as done in inference)
    latents_denorm = latents_norm * bundle.latents_std.to(latents_norm) + bundle.latents_mean.to(latents_norm)

    # Check round-trip of normalization
    norm_error = (latents - latents_denorm).abs().max()
    print(f"  Latent norm/denorm round-trip error: {norm_error:.2e}")

    # Decode
    with torch.no_grad(), torch.autocast('cuda', enabled=False):
        decoded = bundle.vae.decode(latents.float())

    print(f"  Decoded motion shape: {decoded.shape}")
    print(f"  Decoded stats: mean={decoded.mean():.4f}, std={decoded.std():.4f}")

    # Compare
    recon_error = (dummy_reshaped.to(device) - decoded).abs().mean()
    print(f"  VAE reconstruction error (MAE): {recon_error:.4f}")

    # Test with normalized latents round-trip
    with torch.no_grad(), torch.autocast('cuda', enabled=False):
        decoded_from_norm = bundle.vae.decode(latents_denorm.float())

    norm_recon_error = (dummy_reshaped.to(device) - decoded_from_norm).abs().mean()
    print(f"  VAE recon error after norm/denorm: {norm_recon_error:.4f}")
    print(f"  [{'PASS' if norm_recon_error < 0.5 else 'FAIL'}] VAE round-trip")

    # =========================================================================
    # TEST 2: Load real training data and test VAE
    # =========================================================================
    print("\n" + "=" * 80)
    print("[TEST 2] Real Data VAE Round-Trip")
    print("=" * 80)

    # Try to load a real motion sample
    anno_file = 'data/annotation/train_hq_motionhub_hymotion.json'
    data_dir = 'data/motionhub'

    real_motion = None
    if os.path.exists(anno_file):
        import json
        with open(anno_file, 'r') as f:
            annotations = json.load(f)

        # Find a sample with the right format
        for anno in annotations[:20]:
            motion_path = os.path.join(data_dir, anno.get('smplx', ''))
            if os.path.exists(motion_path):
                try:
                    data = np.load(motion_path)
                    print(f"  Found motion: {motion_path}")
                    print(f"  Keys: {list(data.keys())}")
                    break
                except:
                    continue
        else:
            print("  Could not find a valid motion file in first 20 annotations")
    else:
        print(f"  Annotation file not found: {anno_file}")

    # =========================================================================
    # TEST 3: Transformer Output Check
    # =========================================================================
    print("\n" + "=" * 80)
    print("[TEST 3] Transformer Forward Pass Sanity Check")
    print("=" * 80)

    # Move transformer to GPU
    bundle.vae = bundle.vae.cpu()
    torch.cuda.empty_cache()

    dtype = torch.bfloat16
    bundle.transformer = bundle.transformer.to(device, dtype)

    vae_temporal = bundle.vae.config.scale_factor_temporal
    num_latent_frames = (num_frames - 1) // vae_temporal + 1
    in_channels = bundle.transformer.config.in_channels

    print(f"  num_latent_frames={num_latent_frames}, num_joints={num_joints}, in_channels={in_channels}")

    # Create random latent input (simulating noisy latents at t=1000)
    latent_input = torch.randn(1, in_channels, num_latent_frames, num_joints, device=device, dtype=dtype)

    # Create text embeddings (match training: max_seq_len=256)
    bundle.text_encoder = bundle.text_encoder.to('cpu')

    prompt = "a person walks forward slowly"
    for max_len in [128, 256]:
        inputs = bundle.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_len,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            text_output = bundle.text_encoder(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
            )
            text_states = text_output.last_hidden_state

            # Apply the same encode_prompt logic
            seq_len = inputs.attention_mask.gt(0).sum(dim=1).long()[0]
            text_states = text_states[:, :seq_len, :]
            if text_states.shape[1] < max_len:
                padding = text_states.new_zeros(1, max_len - text_states.shape[1], text_states.shape[2])
                text_states = torch.cat([text_states, padding], dim=1)

        text_states_dev = text_states.to(device=device, dtype=dtype)

        # Create all-ones mask
        motion_mask = torch.ones(1, num_latent_frames, num_joints, device=device)

        # Create timestep (expand_timesteps=True)
        t = torch.tensor([999.0], device=device)  # High noise level
        first_frame_mask = torch.ones(1, 1, num_latent_frames, num_joints, device=device)
        temp_ts = (first_frame_mask[0][0] * t).flatten()
        timestep = temp_ts.unsqueeze(0)

        with torch.no_grad():
            output = bundle.transformer(
                hidden_states=latent_input,
                timestep=timestep,
                encoder_hidden_states=text_states_dev,
                attention_kwargs=None,
                is_causal=False,
                hidden_states_mask=motion_mask,
            )

        print(f"\n  max_text_length={max_len}:")
        print(f"    Text states shape: {text_states.shape}")
        print(f"    Input latent shape: {latent_input.shape}")
        print(f"    Timestep shape: {timestep.shape}, values: min={timestep.min():.1f}, max={timestep.max():.1f}")
        print(f"    Output shape: {output.shape}")
        print(f"    Output stats: mean={output.float().mean():.4f}, std={output.float().std():.4f}")
        print(f"    Output range: [{output.float().min():.4f}, {output.float().max():.4f}]")

    # =========================================================================
    # TEST 4: Single-Step Denoising from Known Latent
    # =========================================================================
    print("\n" + "=" * 80)
    print("[TEST 4] Denoising from Nearly-Clean Latent")
    print("=" * 80)

    # Use a clean latent (the one we encoded from dummy data)
    clean_latent = latents_norm.to(device, dtype)  # Normalized latent

    # Add small noise (low timestep = almost clean)
    scheduler = bundle.scheduler
    scheduler.set_timesteps(50, device=device)

    # Find low timestep (near clean)
    print(f"  Scheduler timesteps (first 5): {scheduler.timesteps[:5].tolist()}")
    print(f"  Scheduler timesteps (last 5): {scheduler.timesteps[-5:].tolist()}")

    # Simulate what training does: add noise at a low timestep
    low_t = scheduler.timesteps[-1]  # Smallest timestep (least noise)
    print(f"  Using low timestep: {low_t}")

    noise = torch.randn_like(clean_latent)
    sigma = scheduler.sigmas[-1]  # sigma for smallest timestep
    print(f"  Sigma at low t: {sigma:.6f}")

    noisy_latent = (1 - sigma) * clean_latent + sigma * noise

    print(f"  Clean latent stats: mean={clean_latent.float().mean():.4f}, std={clean_latent.float().std():.4f}")
    print(f"  Noisy latent stats: mean={noisy_latent.float().mean():.4f}, std={noisy_latent.float().std():.4f}")

    # Single forward pass
    motion_mask = torch.ones(1, num_latent_frames, num_joints, device=device)
    first_frame_mask = torch.ones(1, 1, num_latent_frames, num_joints, device=device)
    temp_ts = (first_frame_mask[0][0] * low_t).flatten()
    timestep = temp_ts.unsqueeze(0)

    # Use max_len=256 text (matching training)
    inputs_256 = bundle.tokenizer(
        prompt, padding="max_length", max_length=256, truncation=True, return_tensors="pt",
    )
    with torch.no_grad():
        text_output = bundle.text_encoder(
            input_ids=inputs_256.input_ids, attention_mask=inputs_256.attention_mask,
        )
        text_states_256 = text_output.last_hidden_state
        seq_len = inputs_256.attention_mask.gt(0).sum(dim=1).long()[0]
        text_states_256 = text_states_256[:, :seq_len, :]
        if text_states_256.shape[1] < 256:
            pad = text_states_256.new_zeros(1, 256 - text_states_256.shape[1], text_states_256.shape[2])
            text_states_256 = torch.cat([text_states_256, pad], dim=1)

    text_states_dev = text_states_256.to(device=device, dtype=dtype)

    with torch.no_grad():
        pred = bundle.transformer(
            hidden_states=noisy_latent,
            timestep=timestep,
            encoder_hidden_states=text_states_dev,
            attention_kwargs=None,
            is_causal=False,
            hidden_states_mask=motion_mask,
        )

    print(f"  Model prediction stats: mean={pred.float().mean():.4f}, std={pred.float().std():.4f}")

    # The predicted target in training is: noise - latents (velocity field)
    # So the expected prediction ≈ noise - clean_latent
    expected = noise - clean_latent
    print(f"  Expected target (noise-latent) stats: mean={expected.float().mean():.4f}, std={expected.float().std():.4f}")

    # Compare
    pred_error = (pred.float() - expected.float()).abs().mean()
    print(f"  Prediction error vs expected: {pred_error:.4f}")
    print(f"  [{'PASS' if pred_error < 2.0 else 'FAIL'}] Single-step prediction (error < 2.0)")

    # =========================================================================
    # TEST 5: Full Denoising Loop (50 steps) with Stats Monitoring
    # =========================================================================
    print("\n" + "=" * 80)
    print("[TEST 5] Full Denoising Loop (50 steps)")
    print("=" * 80)

    scheduler.set_timesteps(50, device=device)
    timesteps_sched = scheduler.timesteps

    latents_loop = torch.randn(1, in_channels, num_latent_frames, num_joints, device=device, dtype=dtype)
    print(f"  Initial noise stats: mean={latents_loop.float().mean():.4f}, std={latents_loop.float().std():.4f}")

    for step_idx, t in enumerate(timesteps_sched):
        first_frame_mask = torch.ones(1, 1, num_latent_frames, num_joints, device=device)
        latent_model_input = latents_loop.to(dtype)
        temp_ts = (first_frame_mask[0][0] * t).flatten()
        ts = temp_ts.unsqueeze(0)

        with torch.no_grad():
            noise_pred = bundle.transformer(
                hidden_states=latent_model_input,
                timestep=ts,
                encoder_hidden_states=text_states_dev,
                attention_kwargs=None,
                is_causal=False,
                hidden_states_mask=motion_mask,
            )

        latents_loop = scheduler.step(noise_pred, t, latents_loop, return_dict=False)[0]

        if step_idx % 10 == 0 or step_idx == len(timesteps_sched) - 1:
            print(f"  Step {step_idx:2d} (t={t:.1f}): "
                  f"latent mean={latents_loop.float().mean():.4f}, "
                  f"std={latents_loop.float().std():.4f}, "
                  f"pred mean={noise_pred.float().mean():.4f}, "
                  f"pred std={noise_pred.float().std():.4f}")

    print(f"\n  Final denoised latent: mean={latents_loop.float().mean():.4f}, std={latents_loop.float().std():.4f}")

    # Compare to what a real latent looks like
    print(f"  Reference clean latent: mean={clean_latent.float().mean():.4f}, std={clean_latent.float().std():.4f}")

    # =========================================================================
    # TEST 6: Decode the Denoised Latent and Check Motion Stats
    # =========================================================================
    print("\n" + "=" * 80)
    print("[TEST 6] Decode Denoised Latent -> Motion")
    print("=" * 80)

    # Move VAE back
    bundle.transformer = bundle.transformer.cpu()
    torch.cuda.empty_cache()
    bundle.vae = bundle.vae.to(device)

    # Denormalize
    final_latent = latents_loop.float()
    latents_mean = bundle.latents_mean.to(final_latent)
    latents_std = bundle.latents_std.to(final_latent)
    denorm_latent = final_latent * latents_std + latents_mean

    print(f"  Denormalized latent: mean={denorm_latent.mean():.4f}, std={denorm_latent.std():.4f}")

    with torch.no_grad(), torch.autocast('cuda', enabled=False):
        motion_decoded = bundle.vae.decode(denorm_latent.float())

    print(f"  Decoded motion shape: {motion_decoded.shape}")
    print(f"  Decoded motion stats: mean={motion_decoded.mean():.4f}, std={motion_decoded.std():.4f}")
    print(f"  Decoded motion range: [{motion_decoded.min():.4f}, {motion_decoded.max():.4f}]")

    # Post-process
    smpl_processor = bundle.smpl_pose_processor
    x_dec = rearrange(motion_decoded, "b t j d -> b t (j d)")
    print(f"  Flattened shape: {x_dec.shape}")

    # Denormalize using stats
    x_denorm = smpl_processor.denormalize(x_dec)
    print(f"  Denormalized motion stats: mean={x_denorm.mean():.4f}, std={x_denorm.std():.4f}")
    print(f"  Denormalized motion range: [{x_denorm.min():.4f}, {x_denorm.max():.4f}]")

    # Split transl and pose
    transl_abs_rel = x_denorm[..., :6]
    pred_poses = x_denorm[..., 6:]

    print(f"  Translation (abs_rel) stats: mean={transl_abs_rel.mean():.4f}, std={transl_abs_rel.std():.4f}")
    print(f"  Pose (rot6d) stats: mean={pred_poses.mean():.4f}, std={pred_poses.std():.4f}")
    print(f"  Pose range: [{pred_poses.min():.4f}, {pred_poses.max():.4f}]")

    # Check if rot6d values are reasonable (should be near [-1, 1] for valid rotations)
    rot6d_reasonable = pred_poses.abs().max() < 5.0
    print(f"  [{'PASS' if rot6d_reasonable else 'FAIL'}] Rot6d values in reasonable range (|max| < 5.0)")

    # =========================================================================
    # TEST 7: Compare Training vs Inference Timestep Construction
    # =========================================================================
    print("\n" + "=" * 80)
    print("[TEST 7] Timestep Construction Comparison")
    print("=" * 80)

    # Training path: create_sequence_ts
    # condition_frame_mask_vae is all-True for T2M (no frame conditioning at inference)
    # So target_ts = ori_ts broadcasted to [B, T', J] then flattened

    # Simulate training timestep for T2M (no conditioning):
    ori_ts = torch.tensor([500.0], device=device)  # Single timestep
    # In training: condition_frame_mask_vae is all True (everything is being denoised)
    # target_ts = ori_ts.unsqueeze(1).unsqueeze(2).expand(1, num_latent_frames, num_joints)
    # = all 500.0, flattened to [1, T'*J]
    train_ts = ori_ts.unsqueeze(0).unsqueeze(1).expand(1, num_latent_frames, num_joints)
    train_ts_flat = train_ts.flatten(1)

    # Inference path: expand_timesteps
    t_infer = torch.tensor(500.0, device=device)
    first_frame_mask_infer = torch.ones(1, 1, num_latent_frames, num_joints, device=device)
    temp_ts_infer = (first_frame_mask_infer[0][0] * t_infer).flatten()
    infer_ts = temp_ts_infer.unsqueeze(0)

    ts_match = torch.allclose(train_ts_flat, infer_ts)
    print(f"  Training timestep shape: {train_ts_flat.shape}, values: all {train_ts_flat[0,0]:.1f}")
    print(f"  Inference timestep shape: {infer_ts.shape}, values: all {infer_ts[0,0]:.1f}")
    print(f"  [{'PASS' if ts_match else 'FAIL'}] Timestep construction matches")

    # =========================================================================
    # TEST 8: Check if PrismPipeline produces same results
    # =========================================================================
    print("\n" + "=" * 80)
    print("[TEST 8] Official PrismPipeline Test")
    print("=" * 80)

    try:
        from hftrainer.pipelines.motion.prism_pipeline import PrismPipeline

        # Reload transformer
        bundle.vae = bundle.vae.cpu()
        torch.cuda.empty_cache()

        # Rebuild bundle fresh for pipeline test
        cfg2 = Config.fromfile(config_path)
        model_cfg2 = cfg2.model.to_dict() if hasattr(cfg2.model, 'to_dict') else cfg2.model
        bundle2 = bundle_cls.from_config(model_cfg2)
        bundle2.eval()

        state_dict2 = load_checkpoint(ckpt_path, map_location='cpu')
        bundle2.load_state_dict_selective(state_dict2)
        del state_dict2
        gc.collect()

        bundle2 = bundle2.to(device)

        pipeline = PrismPipeline(bundle2)
        result = pipeline(
            prompts="a person walks forward slowly",
            num_frames_per_segment=33,
            num_inference_steps=50,
            guidance_scale=5.0,
        )
        print(f"  PrismPipeline result type: {type(result)}")
        if isinstance(result, dict):
            for k, v in result.items():
                if isinstance(v, (torch.Tensor, np.ndarray)):
                    v_arr = np.array(v) if not isinstance(v, np.ndarray) else v
                    print(f"    {k}: shape={v_arr.shape}, mean={v_arr.mean():.4f}, std={v_arr.std():.4f}")
                else:
                    print(f"    {k}: {type(v)}")
        print(f"  [INFO] PrismPipeline ran successfully")
    except Exception as e:
        print(f"  [SKIP] PrismPipeline failed: {e}")

    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
