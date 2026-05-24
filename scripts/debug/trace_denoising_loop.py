"""Trace full 50-step denoising loop with known x_0 target.

Starting from the SAME noisy sample used in oracle test, runs the full 50-step
inference loop and tracks latent evolution vs. the ideal trajectory.

This identifies exactly WHERE the loop diverges from expected behavior.

Usage:
    cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
    python3 scripts/debug/trace_denoising_loop.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import gc
import numpy as np
import torch
from einops import rearrange
from mmengine.config import Config

import hftrainer  # noqa
from hftrainer.registry import MODEL_BUNDLES
from hftrainer.utils.checkpoint_utils import load_checkpoint
from hftrainer.models.motion.prism.gaussian_distribution import DiagonalGaussianDistributionNd


def cosine_similarity(a, b):
    return (a * b).sum() / (a.norm() * b.norm() + 1e-8)


def main():
    device = torch.device('cuda')
    config_path = "configs/prism/prism_1b_tp2m_multiframe.py"
    ckpt_path = "work_dirs/prism_1b_tp2m_multiframe/checkpoint-iter_15000"

    # ========== Load Bundle ==========
    print("=" * 70)
    print("TRACE DENOISING LOOP: Full 50-step with known x_0")
    print("=" * 70)

    print("\n[1] Loading bundle...")
    cfg = Config.fromfile(config_path)
    model_cfg = cfg.model.to_dict() if hasattr(cfg.model, 'to_dict') else cfg.model
    bundle_cls = MODEL_BUNDLES.get(model_cfg['type'])
    bundle = bundle_cls.from_config(model_cfg)
    bundle.eval()

    state_dict = load_checkpoint(ckpt_path, map_location='cpu')
    bundle.load_state_dict_selective(state_dict)
    del state_dict
    gc.collect()

    # ========== Get x_0 from real data ==========
    print("\n[2] Encoding real motion to get known x_0...")
    from hftrainer.datasets.motion.motionhub.transforms.load_smplx import process_smplx_pose, process_transl

    base_dir = "data/motionhub/amass_sup/smplx_55"
    motion_file = None
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if f.endswith('.npz'):
                motion_file = os.path.join(root, f)
                break
        if motion_file:
            break

    data = np.load(motion_file, allow_pickle=True)
    trans = np.asarray(data["trans"], dtype=np.float32)[:129]
    poses = np.asarray(data["poses"], dtype=np.float32)[:129]

    transl_processed = process_transl(trans, "abs_rel")
    pose_processed = process_smplx_pose(poses, "rotation_6d", "smpl_22")
    motion_vec = np.concatenate([transl_processed, pose_processed], axis=-1)
    motion_tensor = torch.from_numpy(motion_vec).float().unsqueeze(0)

    smpl_proc = bundle.smpl_pose_processor
    motion_norm = smpl_proc.normalize(motion_tensor)
    motion_for_vae = rearrange(motion_norm, "b t (j d) -> b t j d", d=6)

    with torch.no_grad():
        latents_enc = bundle.vae.encode(motion_for_vae.float())
    latents_mode = DiagonalGaussianDistributionNd(latents_enc).mode()
    x_0 = (latents_mode - bundle.latents_mean) / bundle.latents_std
    x_0 = x_0.to(device)

    print(f"  x_0: shape={x_0.shape}, std={x_0.std():.4f}")

    # ========== Get text embeddings ==========
    print("\n[3] Encoding text...")
    prompt = "a person walks forward slowly"
    text_states = bundle.encode_prompt(prompt, max_sequence_length=256, dtype=torch.bfloat16).to(device)
    neg_text_states = bundle.encode_prompt("", max_sequence_length=256, dtype=torch.bfloat16).to(device)

    # ========== Move transformer to GPU ==========
    print("\n[4] Moving transformer to GPU...")
    bundle.transformer = bundle.transformer.to(device, torch.bfloat16)
    bundle.transformer.eval()
    torch.cuda.empty_cache()

    # ========== Setup 50-step schedule ==========
    scheduler = bundle.scheduler
    scheduler.set_timesteps(50, device=device)
    timesteps = scheduler.timesteps
    sigmas = scheduler.sigmas

    print(f"\n  50-step schedule:")
    print(f"  Timesteps: {timesteps[:5].tolist()} ... {timesteps[-5:].tolist()}")
    print(f"  Sigmas: {sigmas[:5].tolist()} ... {sigmas[-5:].tolist()}")

    # ========== Create known noisy sample ==========
    torch.manual_seed(42)
    noise = torch.randn_like(x_0)
    target_velocity = noise - x_0  # true target at any sigma

    batch_size = 1
    _, C, T_lat, J = x_0.shape
    N = T_lat * J
    motion_mask = torch.ones(batch_size, T_lat, J, device=device)

    # ========== Run inference-style denoising starting from known noise ==========
    # Start from the EXACT x_T that would give x_0 after perfect denoising
    # At sigma=1.0: x_T = (1-1)*x_0 + 1*noise = noise
    latents = noise.clone()

    print(f"\n{'='*90}")
    print(f"DENOISING LOOP (50 steps, CFG=1.0, NO guidance)")
    print(f"Starting latents std={latents.std():.4f}, target x_0 std={x_0.std():.4f}")
    print(f"{'='*90}")
    print(f"{'Step':>4} {'t':>8} {'σ':>7} {'σ_next':>7} | {'lat_std':>8} {'ideal_std':>9} | {'pred_std':>9} {'cos_pred_v':>10} {'cos_lat_x0':>10}")
    print("-" * 100)

    scheduler._step_index = 0  # Reset step index

    for i, t in enumerate(timesteps):
        # Ideal latent at current sigma: x_t_ideal = (1-σ)*x_0 + σ*noise
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        ideal_latent = (1 - sigma) * x_0 + sigma * noise
        ideal_std = ideal_latent.std().item()

        # Create timestep tensor [B, N]
        ts_tensor = t.unsqueeze(0).unsqueeze(1).expand(batch_size, N).contiguous()

        # Model prediction (NO CFG for clarity)
        with torch.no_grad():
            pred = bundle.transformer(
                hidden_states=latents.to(torch.bfloat16),
                timestep=ts_tensor,
                encoder_hidden_states=text_states.to(torch.bfloat16),
                attention_kwargs=None,
                is_causal=False,
                hidden_states_mask=motion_mask,
            ).float()

        # Scheduler step
        dt = sigma_next - sigma
        latents = latents.float() + dt * pred

        # Metrics
        lat_std = latents.std().item()
        pred_std = pred.std().item()
        cos_pred_v = cosine_similarity(pred.flatten(), target_velocity.flatten()).item()
        cos_lat_x0 = cosine_similarity(latents.flatten(), x_0.flatten()).item()

        if i % 5 == 0 or i == len(timesteps) - 1:
            print(f"{i:4d} {t.item():8.2f} {sigma.item():7.4f} {sigma_next.item():7.4f} | "
                  f"{lat_std:8.4f} {ideal_std:9.4f} | "
                  f"{pred_std:9.4f} {cos_pred_v:10.4f} {cos_lat_x0:10.4f}")

    print(f"\n  FINAL latent std: {latents.std():.4f} (target x_0 std: {x_0.std():.4f})")
    print(f"  FINAL cos(latent, x_0): {cosine_similarity(latents.flatten(), x_0.flatten()):.4f}")
    print(f"  MSE(latent, x_0): {(latents - x_0).pow(2).mean():.4f}")

    # ========== Now compare: what does the INFERENCE PIPELINE produce? ==========
    print(f"\n\n{'='*90}")
    print(f"COMPARISON: Standard inference from SAME noise (cfg=5.0)")
    print(f"{'='*90}")

    # Reset scheduler
    scheduler.set_timesteps(50, device=device)
    scheduler._step_index = None

    latents2 = noise.clone()
    guidance_scale = 5.0

    for i, t in enumerate(timesteps):
        ts_tensor = t.unsqueeze(0).unsqueeze(1).expand(batch_size, N).contiguous()

        with torch.no_grad():
            noise_pred = bundle.transformer(
                hidden_states=latents2.to(torch.bfloat16),
                timestep=ts_tensor,
                encoder_hidden_states=text_states.to(torch.bfloat16),
                attention_kwargs=None,
                is_causal=False,
                hidden_states_mask=motion_mask,
            ).float()

            noise_uncond = bundle.transformer(
                hidden_states=latents2.to(torch.bfloat16),
                timestep=ts_tensor,
                encoder_hidden_states=neg_text_states.to(torch.bfloat16),
                attention_kwargs=None,
                is_causal=False,
                hidden_states_mask=motion_mask,
            ).float()

        # CFG
        noise_pred_cfg = noise_uncond + guidance_scale * (noise_pred - noise_uncond)

        # Manual Euler step (same as scheduler)
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        dt = sigma_next - sigma
        latents2 = latents2.float() + dt * noise_pred_cfg

        if i % 10 == 0 or i == len(timesteps) - 1:
            print(f"  Step {i:3d} | σ={sigma.item():.4f}→{sigma_next.item():.4f} | "
                  f"uncond_std={noise_uncond.std():.4f} cond_std={noise_pred.std():.4f} "
                  f"cfg_std={noise_pred_cfg.std():.4f} | lat_std={latents2.std():.4f}")

    print(f"\n  CFG=5 FINAL: latent std={latents2.std():.4f}, cos(lat,x_0)={cosine_similarity(latents2.flatten(), x_0.flatten()):.4f}")

    # ========== Critical comparison: with vs without known x_0 ==========
    print(f"\n\n{'='*90}")
    print(f"KEY COMPARISON: Does model predict SAME thing for known vs unknown noise?")
    print(f"{'='*90}")
    print(f"(If model output is same for known noise and random noise, model doesn't use x_0 info)")

    # Test with different random noise (not containing x_0 info)
    torch.manual_seed(123)
    random_noise = torch.randn_like(x_0)

    scheduler.set_timesteps(50, device=device)
    t_mid = timesteps[25]  # Mid-point
    sigma_mid = sigmas[25]

    # Noisy from known x_0
    x_t_known = (1 - sigma_mid) * x_0 + sigma_mid * noise
    # Pure random (no x_0 info - just noise)
    x_t_random = random_noise.clone()

    ts_tensor = t_mid.unsqueeze(0).unsqueeze(1).expand(batch_size, N).contiguous()

    with torch.no_grad():
        pred_known = bundle.transformer(
            hidden_states=x_t_known.to(torch.bfloat16),
            timestep=ts_tensor,
            encoder_hidden_states=text_states.to(torch.bfloat16),
            attention_kwargs=None,
            is_causal=False,
            hidden_states_mask=motion_mask,
        ).float()

        pred_random = bundle.transformer(
            hidden_states=x_t_random.to(torch.bfloat16),
            timestep=ts_tensor,
            encoder_hidden_states=text_states.to(torch.bfloat16),
            attention_kwargs=None,
            is_causal=False,
            hidden_states_mask=motion_mask,
        ).float()

    true_v_known = noise - x_0
    print(f"\n  At sigma={sigma_mid.item():.4f} (step 25 of 50):")
    print(f"  Input with known x_0: pred_std={pred_known.std():.4f}, cos_to_true_v={cosine_similarity(pred_known.flatten(), true_v_known.flatten()):.4f}")
    print(f"  Input with random noise: pred_std={pred_random.std():.4f}")
    print(f"  cos(pred_known, pred_random): {cosine_similarity(pred_known.flatten(), pred_random.flatten()):.4f}")
    print(f"  → If model is working: predictions should differ significantly for different inputs")
    print(f"  → If model ignores input: predictions would be similar regardless of input")


if __name__ == "__main__":
    main()
