"""Critical diagnostic: test if model predicts correct velocity on real data.

This test loads real training data, encodes through VAE, adds noise at a known
timestep, and checks if the model's prediction matches the expected target
(velocity = noise - clean_latent).

If this test passes → the bug is in scheduler integration/accumulation
If this test fails → the model weights are corrupted or loaded wrong

Usage:
    python3 scripts/inference/test_single_step.py \
        --config configs/prism/prism_1b_tp2m_multiframe.py \
        --checkpoint work_dirs/prism_1b_tp2m_multiframe/checkpoint-iter_15000
"""

import argparse
import gc
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
from einops import rearrange
from mmengine.config import Config

import hftrainer  # noqa
from hftrainer.registry import MODEL_BUNDLES
from hftrainer.utils.checkpoint_utils import load_checkpoint
from hftrainer.models.motion.prism.bundle import _get_sigmas


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device)
    dtype = torch.bfloat16

    print("=" * 70)
    print("  SINGLE-STEP VELOCITY PREDICTION TEST")
    print("=" * 70)

    # =====================================================================
    # 1. Build and load model
    # =====================================================================
    cfg = Config.fromfile(args.config)
    model_cfg = cfg.model.to_dict() if hasattr(cfg.model, 'to_dict') else cfg.model
    bundle_cls = MODEL_BUNDLES.get(model_cfg['type'])
    bundle = bundle_cls.from_config(model_cfg)
    bundle.eval()

    state_dict = load_checkpoint(args.checkpoint, map_location='cpu')
    print(f'Loading checkpoint: {args.checkpoint}')
    bundle.load_state_dict_selective(state_dict)
    del state_dict
    gc.collect()

    # =====================================================================
    # 2. Load real training data
    # =====================================================================
    print("\n--- Loading training data ---")
    data_dir = "/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/data"
    anno_path = os.path.join(data_dir, "annotation/train_hymotion_400h.json")

    with open(anno_path, 'r') as f:
        anno = json.load(f)

    data_list = anno['data_list']
    first_key = list(data_list.keys())[0]
    motion_rel_path = data_list[first_key]['smplx_path']
    anno_dir = os.path.join(data_dir, "annotation")
    motion_path = os.path.normpath(os.path.join(anno_dir, motion_rel_path))
    caption = data_list[first_key].get('caption', 'a person walks forward')

    print(f"  Motion: {motion_path}")
    print(f"  Caption: {caption}")

    from hftrainer.datasets.motion.motionhub.transforms.load_smplx import (
        process_smplx_pose, process_transl
    )

    npz_data = np.load(motion_path, allow_pickle=True)
    abs_trans = np.asarray(npz_data["trans"], dtype=np.float32)
    poses = np.asarray(npz_data["poses"], dtype=np.float32)

    pose_rot6d = process_smplx_pose(poses, "rotation_6d", "smpl_22")
    transl = process_transl(abs_trans, "abs_rel")
    motion_vec = np.concatenate([transl, pose_rot6d], axis=-1)
    motion_tensor = torch.from_numpy(motion_vec).float()

    # Take 129 frames
    target_frames = 129
    T = motion_tensor.shape[0]
    if T >= target_frames:
        motion_tensor = motion_tensor[:target_frames]
    else:
        pad = motion_tensor[-1:].expand(target_frames - T, -1)
        motion_tensor = torch.cat([motion_tensor, pad], dim=0)

    print(f"  Motion shape: {motion_tensor.shape}")

    # =====================================================================
    # 3. Encode through VAE to get clean latents (replicate train_step)
    # =====================================================================
    print("\n--- Encoding motion through VAE ---")
    bundle.vae = bundle.vae.to(device)

    # Normalize motion (same as bundle.encode_motion)
    motion_norm = bundle.smpl_pose_processor.normalize(motion_tensor.unsqueeze(0))
    motion_4d = rearrange(motion_norm, 'b t (j d) -> b t j d', d=6).to(device)

    from hftrainer.models.motion.prism.gaussian_distribution import DiagonalGaussianDistributionNd
    device_type = device.type
    with torch.autocast(device_type, enabled=False):
        z_raw = bundle.vae.encode(motion_4d.float())
    lat = DiagonalGaussianDistributionNd(z_raw)
    gt_latents = lat.mode()  # [1, 16, T_latent, 23]

    # Normalize latents
    latents_mean = bundle.latents_mean.to(gt_latents)
    latents_std = bundle.latents_std.to(gt_latents)
    gt_latents_norm = (gt_latents - latents_mean) / latents_std

    print(f"  GT latent shape: {gt_latents_norm.shape}")
    print(f"  GT latent mean: {gt_latents_norm.mean():.4f}, std: {gt_latents_norm.std():.4f}")
    print(f"  GT latent per-channel std: {gt_latents_norm.squeeze(0).std(dim=(1,2)).cpu().tolist()[:4]}...")

    # Free VAE
    bundle.vae = bundle.vae.cpu()
    gc.collect()
    torch.cuda.empty_cache()

    # =====================================================================
    # 4. Encode text (matching training: max_length=256)
    # =====================================================================
    print("\n--- Encoding text ---")
    bundle.text_encoder = bundle.text_encoder.to(device)

    # Use training max_text_length from config
    max_text_length = 256  # from config: trainer.max_text_length=256
    text_states = bundle.encode_prompt(
        caption, max_sequence_length=max_text_length, dtype=dtype
    )
    print(f"  Text states shape: {text_states.shape}")
    print(f"  Text states norm: {text_states.float().norm():.4f}")

    bundle.text_encoder = bundle.text_encoder.cpu()
    gc.collect()
    torch.cuda.empty_cache()

    # =====================================================================
    # 5. Move transformer to GPU and test
    # =====================================================================
    print("\n--- Loading transformer ---")
    bundle.transformer = bundle.transformer.to(device, dtype)
    torch.cuda.empty_cache()
    print(f"  GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Check weight statistics
    total_params = 0
    zero_layers = 0
    for name, param in bundle.transformer.named_parameters():
        total_params += 1
        if param.abs().max() < 1e-8:
            zero_layers += 1
            print(f"  WARNING: Zero layer: {name}")
    print(f"  Total params: {total_params}, Zero layers: {zero_layers}")

    # =====================================================================
    # 6. Single-step velocity prediction test
    # =====================================================================
    print("\n" + "=" * 70)
    print("  SINGLE-STEP VELOCITY PREDICTION TEST")
    print("=" * 70)

    # Use the training scheduler (1000 steps - same as training)
    bundle.scheduler.set_timesteps(bundle.scheduler.config.num_train_timesteps)
    B, C, T_lat, J = gt_latents_norm.shape

    text_states_dev = text_states.to(device=device, dtype=dtype)

    # Test at multiple noise levels
    test_timestep_indices = [50, 200, 500, 800, 950]

    for idx in test_timestep_indices:
        t = bundle.scheduler.timesteps[idx:idx+1].to(device)
        sigma = _get_sigmas(bundle.scheduler, t, n_dim=4, dtype=gt_latents_norm.dtype)

        # Add noise (exact training procedure)
        noise = torch.randn_like(gt_latents_norm)
        noisy_latents = (1 - sigma) * gt_latents_norm + sigma * noise
        target = noise - gt_latents_norm  # velocity target

        # Create per-token timestep (expand_timesteps=True, matching training)
        timestep_expanded = t.unsqueeze(1).expand(B, T_lat * J)

        # Model prediction
        pred = bundle.transformer(
            hidden_states=noisy_latents.to(dtype),
            timestep=timestep_expanded,
            encoder_hidden_states=text_states_dev,
            hidden_states_mask=None,  # training passes None when no padding
        ).float()

        # Metrics
        mse = ((pred - target) ** 2).mean().item()
        cos_sim = torch.nn.functional.cosine_similarity(
            pred.flatten(), target.flatten(), dim=0
        ).item()
        pred_norm = pred.norm().item()
        target_norm = target.norm().item()
        pred_std = pred.std().item()
        target_std = target.std().item()

        # Single Euler step check: does it move toward clean latent?
        # x_t - sigma_t * v_pred should ≈ (1-sigma_t) * x_0
        # So (x_t - sigma_t * v_pred) / (1-sigma_t) should ≈ x_0
        sigma_val = sigma.squeeze().item()
        if sigma_val < 1.0:
            x0_estimate = (noisy_latents - sigma * pred) / (1 - sigma)
            x0_mse = ((x0_estimate - gt_latents_norm) ** 2).mean().item()
        else:
            x0_mse = float('nan')

        print(f"\n  t_idx={idx:4d} (t={t.item():.1f}, sigma={sigma_val:.4f}):")
        print(f"    Target:  std={target_std:.4f}, norm={target_norm:.4f}")
        print(f"    Pred:    std={pred_std:.4f}, norm={pred_norm:.4f}")
        print(f"    MSE(pred, target): {mse:.6f}")
        print(f"    Cosine sim: {cos_sim:.6f}")
        print(f"    x0_estimate MSE vs GT: {x0_mse:.6f}")

    # =====================================================================
    # 7. Multi-step denoising test with diagnostics
    # =====================================================================
    print("\n" + "=" * 70)
    print("  MULTI-STEP DENOISING (50 steps, cfg=1.0)")
    print("=" * 70)

    bundle.scheduler.set_timesteps(50, device=device)
    timesteps = bundle.scheduler.timesteps

    # Print scheduler sigmas
    sigmas = bundle.scheduler.sigmas
    print(f"  Scheduler sigmas: len={len(sigmas)}")
    print(f"    First 5: {sigmas[:5].tolist()}")
    print(f"    Last 5: {sigmas[-5:].tolist()}")
    print(f"    Range: [{sigmas.min():.6f}, {sigmas.max():.6f}]")

    # Start from KNOWN noisy version of GT (not random noise)
    # This tests if the model can recover GT from its noised version
    sigma_start = sigmas[0]  # First (largest) sigma
    noise_for_test = torch.randn_like(gt_latents_norm)
    latents = (1 - sigma_start) * gt_latents_norm + sigma_start * noise_for_test
    latents = latents.to(device)

    print(f"\n  Starting from noisy GT (sigma_start={sigma_start:.4f})")
    print(f"  Initial latent: mean={latents.float().mean():.4f}, std={latents.float().std():.4f}")
    print(f"  GT latent:      mean={gt_latents_norm.mean():.4f}, std={gt_latents_norm.std():.4f}")

    for i, t in enumerate(timesteps):
        timestep_expanded = t.unsqueeze(0).unsqueeze(1).expand(B, T_lat * J)

        pred = bundle.transformer(
            hidden_states=latents.to(dtype),
            timestep=timestep_expanded,
            encoder_hidden_states=text_states_dev,
            hidden_states_mask=None,
        )

        latents = bundle.scheduler.step(pred, t, latents, return_dict=False)[0]

        if i % 10 == 0 or i == len(timesteps) - 1:
            mse_to_gt = ((latents.float() - gt_latents_norm) ** 2).mean().item()
            print(f"    Step {i:3d}: latent std={latents.float().std():.4f}, "
                  f"MSE to GT={mse_to_gt:.6f}")

    final_mse = ((latents.float() - gt_latents_norm) ** 2).mean().item()
    print(f"\n  Final MSE to GT: {final_mse:.6f}")
    print(f"  Final latent std: {latents.float().std():.4f}")
    print(f"  GT latent std: {gt_latents_norm.std():.4f}")

    # =====================================================================
    # 8. Also test from pure random noise (standard inference)
    # =====================================================================
    print("\n" + "=" * 70)
    print("  STANDARD INFERENCE (50 steps, from random noise, cfg=1.0)")
    print("=" * 70)

    bundle.scheduler.set_timesteps(50, device=device)
    timesteps = bundle.scheduler.timesteps
    shape = gt_latents_norm.shape
    latents_rand = torch.randn(*shape, device=device, dtype=dtype)

    print(f"  Initial random noise: std={latents_rand.float().std():.4f}")

    for i, t in enumerate(timesteps):
        timestep_expanded = t.unsqueeze(0).unsqueeze(1).expand(B, T_lat * J)

        pred = bundle.transformer(
            hidden_states=latents_rand.to(dtype),
            timestep=timestep_expanded,
            encoder_hidden_states=text_states_dev,
            hidden_states_mask=None,
        )

        latents_rand = bundle.scheduler.step(pred, t, latents_rand, return_dict=False)[0]

        if i % 10 == 0 or i == len(timesteps) - 1:
            print(f"    Step {i:3d}: latent std={latents_rand.float().std():.4f}, "
                  f"pred std={pred.float().std():.4f}")

    print(f"\n  Final latent std: {latents_rand.float().std():.4f}")
    print(f"  Expected (from GT): ~{gt_latents_norm.std():.4f}")

    if latents_rand.float().std() < 0.5:
        print("\n  >>> VARIANCE COLLAPSE CONFIRMED!")
        print("  >>> Model's predictions are causing latent to shrink")
        print("  >>> Need to check: model output scale vs expected velocity scale")

        # Diagnostic: what's the model predicting at the LAST timestep?
        t_last = timesteps[-1]
        sigma_last = bundle.scheduler.sigmas[-2]  # second to last (last is 0)
        print(f"\n  At last timestep t={t_last:.1f}, sigma={sigma_last:.6f}:")
        print(f"    Expected pred ≈ noise - x0, std ≈ {(2**0.5):.4f} (if noise and x0 independent)")
        print(f"    Actual pred std: {pred.float().std():.4f}")
    else:
        print("\n  >>> Variance looks reasonable!")

    # =====================================================================
    # 9. Manual Euler step verification
    # =====================================================================
    print("\n" + "=" * 70)
    print("  MANUAL EULER STEP vs SCHEDULER STEP")
    print("=" * 70)
    # Verify the scheduler.step() does what we think
    bundle.scheduler.set_timesteps(50, device=device)
    test_latent = torch.randn(*shape, device=device, dtype=dtype)
    t_test = bundle.scheduler.timesteps[0]  # first timestep

    # Get sigma for this step
    sigma_curr = bundle.scheduler.sigmas[0]
    sigma_next = bundle.scheduler.sigmas[1]

    # Model prediction
    timestep_expanded = t_test.unsqueeze(0).unsqueeze(1).expand(B, T_lat * J)
    pred_test = bundle.transformer(
        hidden_states=test_latent.to(dtype),
        timestep=timestep_expanded,
        encoder_hidden_states=text_states_dev,
        hidden_states_mask=None,
    )

    # Scheduler step
    latents_from_scheduler = bundle.scheduler.step(pred_test, t_test, test_latent, return_dict=False)[0]

    # Manual step: prev_sample = sample + (sigma_next - sigma) * model_output
    latents_manual = test_latent + (sigma_next - sigma_curr) * pred_test

    diff = (latents_from_scheduler.float() - latents_manual.float()).abs().max().item()
    print(f"  sigma_curr={sigma_curr:.6f}, sigma_next={sigma_next:.6f}")
    print(f"  (sigma_next - sigma_curr) = {(sigma_next - sigma_curr):.6f}")
    print(f"  Max diff (scheduler vs manual): {diff:.10f}")
    print(f"  Scheduler and manual step {'MATCH' if diff < 1e-4 else 'DIFFER!'}")

    print("\n" + "=" * 70)
    print("  DONE")
    print("=" * 70)


if __name__ == '__main__':
    main()
