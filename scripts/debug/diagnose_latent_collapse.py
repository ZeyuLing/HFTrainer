"""Targeted diagnosis: WHY do denoised latents collapse to std≈0.5?

Hypothesis: Model predictions at high sigma point towards zero (mean),
causing latent shrinkage. This test measures:
1. cos(pred, input_latents) at each step → if ≈1.0, model predicts v≈latents, causing shrinkage
2. cos(pred_cond, pred_uncond) → if ≈1.0, text conditioning is weak
3. Actual sigma schedule with shift=5.0

Usage:
    cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
    python3 scripts/debug/diagnose_latent_collapse.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import gc
import torch
from mmengine.config import Config
from diffusers.utils.torch_utils import randn_tensor

import hftrainer  # noqa
from hftrainer.registry import MODEL_BUNDLES
from hftrainer.utils.checkpoint_utils import load_checkpoint


def cos_sim(a, b):
    return torch.nn.functional.cosine_similarity(
        a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)
    ).item()


def main():
    device = torch.device('cuda')
    config_path = "configs/prism/prism_1b_tp2m_multiframe.py"
    ckpt_path = "work_dirs/prism_1b_tp2m_multiframe/checkpoint-iter_15000"

    print("=" * 80)
    print("DIAGNOSIS: Why do denoised latents collapse?")
    print("=" * 80)

    # Load
    cfg = Config.fromfile(config_path)
    model_cfg = cfg.model.to_dict() if hasattr(cfg.model, 'to_dict') else cfg.model
    bundle_cls = MODEL_BUNDLES.get(model_cfg['type'])
    bundle = bundle_cls.from_config(model_cfg)
    bundle.eval()
    state_dict = load_checkpoint(ckpt_path, map_location='cpu')
    bundle.load_state_dict_selective(state_dict)
    del state_dict
    gc.collect()

    # Text
    text_states = bundle.encode_prompt("a person walks forward slowly", max_sequence_length=256, dtype=torch.bfloat16).to(device)
    neg_text_states = bundle.encode_prompt("", max_sequence_length=256, dtype=torch.bfloat16).to(device)

    # Transformer to GPU
    bundle.transformer = bundle.transformer.to(device, torch.bfloat16)
    bundle.transformer.eval()
    torch.cuda.empty_cache()

    # Setup
    batch_size = 1
    T_lat = 33
    J = 23
    N = T_lat * J
    C = bundle.transformer.config.in_channels

    scheduler = bundle.scheduler
    scheduler.set_timesteps(50, device=device)
    timesteps = scheduler.timesteps
    sigmas = scheduler.sigmas

    print(f"\n[SIGMA SCHEDULE] shift=5.0, 50 steps:")
    print(f"  First 10 sigmas: {[f'{s:.4f}' for s in sigmas[:10].tolist()]}")
    print(f"  Last 10 sigmas:  {[f'{s:.4f}' for s in sigmas[-10:].tolist()]}")
    print(f"  First 5 timesteps: {timesteps[:5].tolist()}")
    print(f"  Last 5 timesteps:  {timesteps[-5:].tolist()}")

    # Print step sizes (dt = sigma_next - sigma)
    print(f"\n  Step sizes (|dt|):")
    for i in range(0, 50, 5):
        dt = (sigmas[i+1] - sigmas[i]).item()
        print(f"    Step {i:2d}: σ={sigmas[i]:.4f} → {sigmas[i+1]:.4f}, dt={dt:.5f}")

    # Initialize latents
    torch.manual_seed(42)
    latents = torch.randn(batch_size, C, T_lat, J, device=device, dtype=torch.bfloat16)
    initial_latents = latents.clone()

    motion_mask = torch.ones(batch_size, T_lat, J, device=device)

    # First frame mask handling (same as pipeline when no frame conditioning)
    first_frame_mask = torch.ones_like(latents)
    condition = torch.zeros_like(latents)

    print(f"\n{'='*80}")
    print(f"DENOISING LOOP - Tracking predictions at each step")
    print(f"{'='*80}")
    print(f"{'Step':>4} {'σ':>7} {'σ_next':>7} {'|dt|':>7} | {'lat_std':>8} {'pred_std':>9} | "
          f"{'cos(p,lat)':>10} {'cos(c,u)':>8} {'cond_std':>9} {'uncond_std':>10} {'cfg_std':>8}")
    print("-" * 115)

    guidance_scale = 5.0

    for i, t in enumerate(timesteps):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        dt = sigma_next - sigma

        # Prepare input (same as pipeline)
        latent_model_input = (
            (1 - first_frame_mask) * condition + first_frame_mask * latents
        ).to(torch.bfloat16)
        temp_ts = (first_frame_mask[0][0] * t).flatten()
        timestep = temp_ts.unsqueeze(0).expand(batch_size, -1)

        with torch.no_grad():
            # Conditional prediction
            noise_pred_cond = bundle.transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=text_states,
                attention_kwargs=None,
                is_causal=False,
                hidden_states_mask=motion_mask,
            ).float()

            # Unconditional prediction
            noise_pred_uncond = bundle.transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=neg_text_states,
                attention_kwargs=None,
                is_causal=False,
                hidden_states_mask=motion_mask,
            ).float()

        # CFG
        noise_pred_cfg = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

        # Scheduler step
        latents = scheduler.step(noise_pred_cfg.to(torch.bfloat16), t, latents, return_dict=False)[0]

        # Metrics
        lat_std = latents.float().std().item()
        pred_std = noise_pred_cfg.std().item()
        cond_std = noise_pred_cond.std().item()
        uncond_std = noise_pred_uncond.std().item()
        cos_pred_lat = cos_sim(noise_pred_cfg, latent_model_input.float())
        cos_cond_uncond = cos_sim(noise_pred_cond, noise_pred_uncond)

        if i % 5 == 0 or i == len(timesteps) - 1:
            print(f"{i:4d} {sigma.item():7.4f} {sigma_next.item():7.4f} {abs(dt.item()):7.5f} | "
                  f"{lat_std:8.4f} {pred_std:9.4f} | "
                  f"{cos_pred_lat:10.4f} {cos_cond_uncond:8.4f} {cond_std:9.4f} {uncond_std:10.4f} {noise_pred_cfg.std().item():8.4f}")

    print(f"\n  FINAL: latent std = {latents.float().std():.4f}")
    print(f"  cos(final_latents, initial_noise) = {cos_sim(latents.float(), initial_latents.float()):.4f}")

    # ========== Key Analysis ==========
    print(f"\n{'='*80}")
    print(f"ANALYSIS")
    print(f"{'='*80}")
    print(f"""
    If cos(pred, latents) ≈ 1.0 at high sigma:
      → Model predicts v ≈ latents (i.e., v ≈ ε when latents≈ε)
      → Step: latents_new = latents + dt*latents = (1+dt)*latents
      → Since dt<0: latents SHRINK each step
      → This is "mode collapse to zero" - model hasn't learned enough

    If cos(cond, uncond) ≈ 1.0:
      → Text conditioning is WEAK - model gives same output regardless of text
      → CFG amplification has no effect
      → This is a model quality issue, not an inference bug

    If cos(pred, latents) ≈ 0 and predictions have good std:
      → Model IS trying to denoise but in wrong direction
      → Likely an inference bug (sign error, wrong formula, etc.)
    """)

    # ========== Test with CFG=1 (no guidance) ==========
    print(f"\n{'='*80}")
    print(f"COMPARISON: CFG=1.0 (no guidance)")
    print(f"{'='*80}")

    scheduler.set_timesteps(50, device=device)
    torch.manual_seed(42)
    latents_nocfg = torch.randn(batch_size, C, T_lat, J, device=device, dtype=torch.bfloat16)

    for i, t in enumerate(timesteps):
        latent_model_input = (
            (1 - first_frame_mask) * condition + first_frame_mask * latents_nocfg
        ).to(torch.bfloat16)
        temp_ts = (first_frame_mask[0][0] * t).flatten()
        timestep = temp_ts.unsqueeze(0).expand(batch_size, -1)

        with torch.no_grad():
            noise_pred = bundle.transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=text_states,
                attention_kwargs=None,
                is_causal=False,
                hidden_states_mask=motion_mask,
            ).float()

        latents_nocfg = scheduler.step(noise_pred.to(torch.bfloat16), t, latents_nocfg, return_dict=False)[0]

        if i % 10 == 0 or i == len(timesteps) - 1:
            print(f"  Step {i:3d}: lat_std={latents_nocfg.float().std():.4f}, pred_std={noise_pred.std():.4f}")

    print(f"  FINAL (CFG=1): latent std = {latents_nocfg.float().std():.4f}")

    # ========== Crucial test: does the model learn text at all? ==========
    print(f"\n{'='*80}")
    print(f"TEXT CONDITIONING STRENGTH TEST")
    print(f"{'='*80}")

    # Test with various prompts at sigma=1.0 (first step)
    scheduler.set_timesteps(50, device=device)
    torch.manual_seed(42)
    test_latents = torch.randn(batch_size, C, T_lat, J, device=device, dtype=torch.bfloat16)

    t_first = timesteps[0]
    temp_ts = (first_frame_mask[0][0] * t_first).flatten()
    ts = temp_ts.unsqueeze(0).expand(batch_size, -1)

    prompts = [
        "a person walks forward slowly",
        "a person jumps high",
        "a person sits down on a chair",
        "",  # empty/negative
    ]

    preds = []
    for prompt in prompts:
        text_enc = bundle.encode_prompt(prompt, max_sequence_length=256, dtype=torch.bfloat16).to(device)
        with torch.no_grad():
            pred = bundle.transformer(
                hidden_states=test_latents,
                timestep=ts,
                encoder_hidden_states=text_enc,
                attention_kwargs=None,
                is_causal=False,
                hidden_states_mask=motion_mask,
            ).float()
        preds.append(pred)
        print(f"  '{prompt[:40]:40s}': pred_std={pred.std():.4f}")

    print(f"\n  Pairwise cosine similarities:")
    for i in range(len(prompts)):
        for j in range(i+1, len(prompts)):
            cs = cos_sim(preds[i], preds[j])
            print(f"    cos('{prompts[i][:20]}', '{prompts[j][:20]}') = {cs:.4f}")


if __name__ == "__main__":
    main()
