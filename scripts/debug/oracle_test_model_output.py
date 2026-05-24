"""Oracle test: verify model predictions match expected targets on known data.

This test definitively determines whether the loaded model weights work correctly:
1. Load real training data → encode to get x_0 latent
2. Create known noise ε
3. Compute x_t = (1-σ)*x_0 + σ*ε at known sigma
4. Create proper timestep tensor (same as training)
5. Pass through transformer
6. Compare prediction to expected target (ε - x_0)

If the model predicts correctly: issue is in inference loop construction
If the model predicts incorrectly: issue is in checkpoint loading or model state

Usage:
    cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
    python3 scripts/debug/oracle_test_model_output.py
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
    """Cosine similarity between two tensors."""
    return (a * b).sum() / (a.norm() * b.norm() + 1e-8)


def main():
    device = torch.device('cuda')
    config_path = "configs/prism/prism_1b_tp2m_multiframe.py"
    ckpt_path = "work_dirs/prism_1b_tp2m_multiframe/checkpoint-iter_15000"

    # ========== Load Bundle ==========
    print("=" * 70)
    print("ORACLE TEST: Model prediction vs known target")
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

    # ========== Encode real motion to get x_0 ==========
    print("\n[2] Encoding real motion to get x_0 latent...")

    from hftrainer.datasets.motion.motionhub.transforms.load_smplx import process_smplx_pose, process_transl

    # Find real motion files
    base_dir = "data/motionhub/amass_sup/smplx_55"
    motion_file = None
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if f.endswith('.npz'):
                motion_file = os.path.join(root, f)
                break
        if motion_file:
            break

    if motion_file is None:
        print("ERROR: No motion file found!")
        return

    print(f"  Using: {os.path.basename(motion_file)}")
    data = np.load(motion_file, allow_pickle=True)
    trans = np.asarray(data["trans"], dtype=np.float32)[:129]
    poses = np.asarray(data["poses"], dtype=np.float32)[:129]

    # Process through training pipeline
    transl_processed = process_transl(trans, "abs_rel")
    pose_processed = process_smplx_pose(poses, "rotation_6d", "smpl_22")
    motion_vec = np.concatenate([transl_processed, pose_processed], axis=-1)
    motion_tensor = torch.from_numpy(motion_vec).float().unsqueeze(0)

    # Encode motion (same as training)
    smpl_proc = bundle.smpl_pose_processor
    motion_norm = smpl_proc.normalize(motion_tensor)
    motion_for_vae = rearrange(motion_norm, "b t (j d) -> b t j d", d=6)

    with torch.no_grad():
        latents_enc = bundle.vae.encode(motion_for_vae.float())
    latents_mode = DiagonalGaussianDistributionNd(latents_enc).mode()
    x_0 = (latents_mode - bundle.latents_mean) / bundle.latents_std  # normalized x_0

    print(f"  x_0 shape: {x_0.shape}")
    print(f"  x_0 stats: mean={x_0.mean():.4f}, std={x_0.std():.4f}, range=[{x_0.min():.3f}, {x_0.max():.3f}]")

    # ========== Encode text ==========
    print("\n[3] Encoding text prompt...")
    prompt = "a person walks forward slowly"
    # Use the same method as inference
    text_states = bundle.encode_prompt(
        prompt, max_sequence_length=256, dtype=torch.bfloat16,
    )
    neg_text_states = bundle.encode_prompt(
        "", max_sequence_length=256, dtype=torch.bfloat16,
    )
    print(f"  text_states shape: {text_states.shape}")

    # ========== Move to GPU ==========
    print("\n[4] Moving transformer to GPU...")
    bundle.transformer = bundle.transformer.to(device, torch.bfloat16)
    bundle.transformer.eval()
    x_0 = x_0.to(device)
    text_states = text_states.to(device)
    neg_text_states = neg_text_states.to(device)
    torch.cuda.empty_cache()

    # ========== Test at multiple sigma values ==========
    print("\n[5] Testing model predictions at multiple sigma values...")
    print("=" * 70)

    # The scheduler was already set_timesteps(1000) in bundle.__init__
    scheduler = bundle.scheduler
    scheduler_timesteps = scheduler.timesteps.to(device)
    scheduler_sigmas = scheduler.sigmas.to(device)

    print(f"  Scheduler timesteps: {len(scheduler_timesteps)} steps")
    print(f"  Timestep range: [{scheduler_timesteps[-1]:.2f}, {scheduler_timesteps[0]:.2f}]")
    print(f"  Sigma range: [{scheduler_sigmas[-2]:.4f}, {scheduler_sigmas[0]:.4f}]")

    # Test at different positions in the schedule
    test_indices = [0, 100, 250, 500, 750, 900, 999]  # covers full range

    batch_size = 1
    _, C, T_lat, J = x_0.shape
    N = T_lat * J  # number of tokens

    # Create motion padding mask (all ones = all valid, same as inference)
    motion_mask = torch.ones(batch_size, T_lat, J, device=device)

    torch.manual_seed(42)
    noise = torch.randn_like(x_0)  # fixed noise ε

    expected_target = noise - x_0  # v = ε - x_0
    expected_std = expected_target.std().item()
    print(f"\n  Known target (ε - x_0): std={expected_std:.4f}")
    print(f"  x_0 std={x_0.std():.4f}, noise std={noise.std():.4f}")
    print(f"  Expected velocity std ≈ sqrt(noise_std² + x0_std²) = {(noise.std()**2 + x_0.std()**2).sqrt():.4f}")

    print(f"\n{'Step':>6} {'Timestep':>10} {'Sigma':>8} | {'Pred_std':>10} {'Target_std':>11} | {'Cosine':>8} {'MSE':>10} {'Ratio':>7}")
    print("-" * 90)

    for idx in test_indices:
        t = scheduler_timesteps[idx]
        sigma = scheduler_sigmas[idx]

        # Create noisy latent: x_t = (1-σ)*x_0 + σ*ε
        sigma_4d = sigma.view(1, 1, 1, 1)
        x_t = (1 - sigma_4d) * x_0 + sigma_4d * noise

        # Create timestep tensor [B, N] - all same value (unconditioned case)
        # Same as training with no frame conditioning
        timestep = t.unsqueeze(0).unsqueeze(1).expand(batch_size, N).contiguous()

        # Cast to model dtype
        x_t_input = x_t.to(torch.bfloat16)

        # Run through transformer
        with torch.no_grad():
            pred = bundle.transformer(
                hidden_states=x_t_input,
                timestep=timestep,
                encoder_hidden_states=text_states.to(torch.bfloat16),
                attention_kwargs=None,
                is_causal=False,
                hidden_states_mask=motion_mask,
            ).float()

        # Compare to target
        target = expected_target.float()
        pred_float = pred.float()

        pred_std = pred_float.std().item()
        cos_sim = cosine_similarity(pred_float.flatten(), target.flatten()).item()
        mse = (pred_float - target).pow(2).mean().item()
        ratio = pred_std / (expected_std + 1e-8)

        print(f"{idx:6d} {t.item():10.2f} {sigma.item():8.4f} | "
              f"{pred_std:10.4f} {expected_std:11.4f} | "
              f"{cos_sim:8.4f} {mse:10.4f} {ratio:7.3f}")

    # ========== Also test: TRAINING mode vs EVAL mode ==========
    print("\n\n" + "=" * 70)
    print("TEST: TRAINING MODE vs EVAL MODE")
    print("=" * 70)

    # Pick a mid-range timestep (index 500, ~mid-schedule)
    idx = 500
    t = scheduler_timesteps[idx]
    sigma = scheduler_sigmas[idx]
    sigma_4d = sigma.view(1, 1, 1, 1)
    x_t = (1 - sigma_4d) * x_0 + sigma_4d * noise
    timestep = t.unsqueeze(0).unsqueeze(1).expand(batch_size, N).contiguous()
    x_t_input = x_t.to(torch.bfloat16)

    # Eval mode (current)
    bundle.transformer.eval()
    with torch.no_grad():
        pred_eval = bundle.transformer(
            hidden_states=x_t_input,
            timestep=timestep,
            encoder_hidden_states=text_states.to(torch.bfloat16),
            attention_kwargs=None,
            is_causal=False,
            hidden_states_mask=motion_mask,
        ).float()

    # Train mode
    bundle.transformer.train()
    with torch.no_grad():
        pred_train = bundle.transformer(
            hidden_states=x_t_input,
            timestep=timestep,
            encoder_hidden_states=text_states.to(torch.bfloat16),
            attention_kwargs=None,
            is_causal=False,
            hidden_states_mask=motion_mask,
        ).float()
    bundle.transformer.eval()

    print(f"\n  At step idx={idx}, t={t.item():.2f}, sigma={sigma.item():.4f}:")
    print(f"  Eval mode: pred_std={pred_eval.std():.4f}, cos_sim_to_target={cosine_similarity(pred_eval.flatten(), expected_target.flatten()):.4f}")
    print(f"  Train mode: pred_std={pred_train.std():.4f}, cos_sim_to_target={cosine_similarity(pred_train.flatten(), expected_target.flatten()):.4f}")
    diff = (pred_eval - pred_train).abs().max().item()
    print(f"  Max diff between train/eval mode: {diff:.6f}")

    # ========== Test: different text prompts ==========
    print("\n\n" + "=" * 70)
    print("TEST: EFFECT OF TEXT PROMPT")
    print("=" * 70)

    bundle.transformer.eval()
    # Test with positive prompt, negative prompt, and random prompt
    for label, tstates in [
        ("positive prompt", text_states),
        ("empty/negative prompt", neg_text_states),
    ]:
        with torch.no_grad():
            pred_t = bundle.transformer(
                hidden_states=x_t_input,
                timestep=timestep,
                encoder_hidden_states=tstates.to(torch.bfloat16),
                attention_kwargs=None,
                is_causal=False,
                hidden_states_mask=motion_mask,
            ).float()
        cos = cosine_similarity(pred_t.flatten(), expected_target.flatten()).item()
        print(f"  {label:30s}: pred_std={pred_t.std():.4f}, cos_sim={cos:.4f}")

    # ========== Test: FP32 vs BF16 ==========
    print("\n\n" + "=" * 70)
    print("TEST: FP32 vs BF16 MODEL")
    print("=" * 70)

    # FP32 test
    bundle.transformer = bundle.transformer.float()
    with torch.no_grad():
        pred_fp32 = bundle.transformer(
            hidden_states=x_t.float(),
            timestep=timestep,
            encoder_hidden_states=text_states.float(),
            attention_kwargs=None,
            is_causal=False,
            hidden_states_mask=motion_mask,
        ).float()

    cos_fp32 = cosine_similarity(pred_fp32.flatten(), expected_target.flatten()).item()
    print(f"  FP32: pred_std={pred_fp32.std():.4f}, cos_sim={cos_fp32:.4f}")
    print(f"  BF16: pred_std={pred_eval.std():.4f}, cos_sim={cosine_similarity(pred_eval.flatten(), expected_target.flatten()):.4f}")
    bf16_fp32_diff = (pred_fp32 - pred_eval).abs().max().item()
    print(f"  Max diff bf16 vs fp32: {bf16_fp32_diff:.6f}")

    # ========== FINAL DIAGNOSIS ==========
    print("\n\n" + "=" * 70)
    print("DIAGNOSIS SUMMARY")
    print("=" * 70)

    # Check if predictions are systematically too small
    final_pred_std = pred_eval.std().item()
    final_cos = cosine_similarity(pred_eval.flatten(), expected_target.flatten()).item()

    print(f"\n  Model prediction std:  {final_pred_std:.4f}")
    print(f"  Expected target std:   {expected_std:.4f}")
    print(f"  Ratio (pred/target):   {final_pred_std / expected_std:.4f}")
    print(f"  Cosine similarity:     {final_cos:.4f}")

    if final_pred_std < expected_std * 0.3:
        print(f"\n  ⚠️  Model predictions are DRAMATICALLY too small ({final_pred_std/expected_std:.1%} of expected)")
        print(f"  → This indicates weights are corrupted/partial or output projection is wrong")
        if final_cos > 0.5:
            print(f"  → BUT direction is correct (cos={final_cos:.3f}), so it might be a SCALING issue")
            print(f"    (e.g., output projection weight scaled down, or missing final layer norm)")
        else:
            print(f"  → Direction is also wrong (cos={final_cos:.3f}), model is not learning this correctly")
    elif abs(final_pred_std / expected_std - 1.0) < 0.3 and final_cos > 0.5:
        print(f"\n  ✓ Model predictions look CORRECT")
        print(f"  → Bug must be in how inference constructs inputs or scheduler steps")
    else:
        print(f"\n  ⚠️  Model predictions are somewhat off")
        print(f"  → Need further investigation")

    # ========== Additional: check if scheduler in inference mode produces different results ==========
    print("\n\n" + "=" * 70)
    print("TEST: SCHEDULER STATE - TRAINING (1000 steps) vs INFERENCE (50 steps)")
    print("=" * 70)

    # In training, scheduler has 1000 timesteps
    # Let's verify what happens if we re-init with 50 steps like inference does
    bundle.transformer = bundle.transformer.to(torch.bfloat16)

    # Save current state
    training_timesteps = scheduler.timesteps.clone()
    training_sigmas = scheduler.sigmas.clone()

    # Switch to inference mode (50 steps)
    scheduler.set_timesteps(50, device=device)
    inference_timesteps = scheduler.timesteps.clone()
    inference_sigmas = scheduler.sigmas.clone()

    print(f"  Training timesteps (1000): first={training_timesteps[0]:.2f}, last={training_timesteps[-1]:.2f}")
    print(f"  Inference timesteps (50):  first={inference_timesteps[0]:.2f}, last={inference_timesteps[-1]:.2f}")
    print(f"  Training sigmas range: [{training_sigmas.min():.6f}, {training_sigmas.max():.6f}]")
    print(f"  Inference sigmas range: [{inference_sigmas.min():.6f}, {inference_sigmas.max():.6f}]")

    # Test: does the model prediction differ when scheduler state is 50 vs 1000?
    # This matters because _get_sigmas uses scheduler.timesteps for lookup!
    # Let's test with a timestep value that exists in both
    t_val = inference_timesteps[25]  # mid-point of inference schedule
    print(f"\n  Testing with t={t_val:.2f} (exists in inference schedule):")

    sigma_inf = inference_sigmas[25]
    sigma_4d = sigma_inf.view(1, 1, 1, 1)
    x_t_test = (1 - sigma_4d) * x_0 + sigma_4d * noise
    timestep_test = t_val.unsqueeze(0).unsqueeze(1).expand(batch_size, N).contiguous()

    with torch.no_grad():
        pred_inf = bundle.transformer(
            hidden_states=x_t_test.to(torch.bfloat16),
            timestep=timestep_test,
            encoder_hidden_states=text_states.to(torch.bfloat16),
            attention_kwargs=None,
            is_causal=False,
            hidden_states_mask=motion_mask,
        ).float()

    target_at_t = noise - x_0
    cos_inf = cosine_similarity(pred_inf.flatten(), target_at_t.flatten()).item()
    print(f"  Prediction std: {pred_inf.std():.4f}, cos_sim to target: {cos_inf:.4f}")
    print(f"  (Target std should still be {target_at_t.std():.4f})")

    # Restore training state
    scheduler.set_timesteps(1000, device=device)


if __name__ == "__main__":
    main()
