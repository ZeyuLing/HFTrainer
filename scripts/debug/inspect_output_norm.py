"""Inspect the transformer's output normalization to find WHY predictions are too small.

The model output goes through:
1. LayerNorm (elementwise_affine=False) → normalizes to std≈1
2. Multiply by (1 + scale) where scale = scale_shift_table[0,1,:] + temb
3. Add shift where shift = scale_shift_table[0,0,:] + temb
4. proj_out: Linear(1536, 16)

If (1 + scale) is systematically < 1.0, output is attenuated.
If proj_out weights are too small, output is attenuated.

This script hooks into the model to capture intermediate values at each stage.

Usage:
    cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
    python3 scripts/debug/inspect_output_norm.py
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


def main():
    device = torch.device('cuda')
    config_path = "configs/prism/prism_1b_tp2m_multiframe.py"
    ckpt_path = "work_dirs/prism_1b_tp2m_multiframe/checkpoint-iter_15000"

    print("=" * 80)
    print("INSPECT OUTPUT NORMALIZATION: Where does attenuation happen?")
    print("=" * 80)

    # Load bundle
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

    # ========== Inspect scale_shift_table directly ==========
    print("\n[2] Inspecting scale_shift_table (learned parameter)")
    sst = bundle.transformer.scale_shift_table.data  # [1, 2, 1536]
    print(f"  Shape: {sst.shape}")
    print(f"  scale_shift_table[0, 0, :] (shift bias): mean={sst[0,0].mean():.6f}, std={sst[0,0].std():.6f}")
    print(f"  scale_shift_table[0, 1, :] (scale bias): mean={sst[0,1].mean():.6f}, std={sst[0,1].std():.6f}")
    print(f"  scale_shift_table range: [{sst.min():.6f}, {sst.max():.6f}]")

    # ========== Inspect proj_out weights ==========
    print("\n[3] Inspecting proj_out (final linear projection)")
    proj_w = bundle.transformer.proj_out.weight.data  # [16, 1536]
    proj_b = bundle.transformer.proj_out.bias.data if bundle.transformer.proj_out.bias is not None else None
    print(f"  proj_out weight shape: {proj_w.shape}")
    print(f"  proj_out weight: mean={proj_w.mean():.6f}, std={proj_w.std():.6f}")
    print(f"  proj_out weight abs mean: {proj_w.abs().mean():.6f}")
    if proj_b is not None:
        print(f"  proj_out bias: mean={proj_b.mean():.6f}, std={proj_b.std():.6f}")
    else:
        print(f"  proj_out bias: None")

    # Expected output std from proj_out given unit-variance input:
    # std ≈ sqrt(sum(w_i^2)) = sqrt(fan_in * w_std^2) = sqrt(1536) * w_std
    expected_proj_std = (1536 ** 0.5) * proj_w.std().item()
    print(f"  Expected output std (if input std=1): {expected_proj_std:.4f}")

    # ========== Move to GPU and run ==========
    print("\n[4] Moving transformer to GPU...")
    bundle.transformer = bundle.transformer.to(device, torch.bfloat16)
    bundle.transformer.eval()
    torch.cuda.empty_cache()

    # Encode text
    text_states = bundle.encode_prompt("a person walks forward slowly", max_sequence_length=256, dtype=torch.bfloat16).to(device)

    # Setup inputs
    batch_size = 1
    T_lat = 33
    J = 23
    N = T_lat * J  # 759
    C = bundle.transformer.config.in_channels  # 16

    torch.manual_seed(42)
    latents = torch.randn(batch_size, C, T_lat, J, device=device, dtype=torch.bfloat16)

    # Create timestep tensors for different sigma values
    scheduler = bundle.scheduler
    scheduler.set_timesteps(50, device=device)
    timesteps = scheduler.timesteps
    sigmas = scheduler.sigmas

    motion_mask = torch.ones(batch_size, T_lat, J, device=device)

    # ========== Hook into the model to capture intermediates ==========
    print("\n[5] Running model with hooks to capture intermediate values")
    print("=" * 80)

    # We'll manually trace through the output norm
    # Instead of hooks, let's modify the forward to capture values

    transformer = bundle.transformer

    # Test at multiple timestep values
    test_timestep_indices = [0, 10, 25, 40, 49]  # different points in schedule

    for tidx in test_timestep_indices:
        t = timesteps[tidx]
        sigma = sigmas[tidx]

        # Prepare input same as inference pipeline
        timestep_tensor = torch.full((batch_size, N), t.item(), device=device, dtype=torch.bfloat16)

        print(f"\n  --- Timestep index={tidx}, t={t.item():.2f}, σ={sigma.item():.4f} ---")

        # Run condition_embedder manually to get temb
        with torch.no_grad():
            # Flatten timestep for condition_embedder
            ts_flat = timestep_tensor.flatten()  # [B*N]
            ts_seq_len = N

            temb, timestep_proj, _ = transformer.condition_embedder(
                ts_flat,
                text_states,
                timestep_seq_len=ts_seq_len,
            )

        # temb shape should be [B, N, inner_dim]
        print(f"    temb: shape={temb.shape}, mean={temb.float().mean():.4f}, std={temb.float().std():.4f}")
        print(f"    temb range: [{temb.float().min():.4f}, {temb.float().max():.4f}]")

        # Compute scale and shift as the output norm would
        sst_device = transformer.scale_shift_table.to(temb.device)
        combined = sst_device.unsqueeze(0) + temb.unsqueeze(2)  # [B, N, 2, inner_dim]
        shift, scale = combined.chunk(2, dim=2)
        shift = shift.squeeze(2).float()
        scale = scale.squeeze(2).float()

        print(f"    scale (raw): mean={scale.mean():.4f}, std={scale.std():.4f}, range=[{scale.min():.4f}, {scale.max():.4f}]")
        print(f"    shift (raw): mean={shift.mean():.4f}, std={shift.std():.4f}, range=[{shift.min():.4f}, {shift.max():.4f}]")

        # The actual multiplier is (1 + scale)
        multiplier = 1 + scale
        print(f"    (1 + scale): mean={multiplier.mean():.4f}, std={multiplier.std():.4f}, range=[{multiplier.min():.4f}, {multiplier.max():.4f}]")
        print(f"    (1 + scale) abs mean: {multiplier.abs().mean():.4f}")

        # What fraction of elements have (1+scale) < 0.5?
        frac_below_half = (multiplier.abs() < 0.5).float().mean().item()
        frac_negative = (multiplier < 0).float().mean().item()
        print(f"    Fraction |1+scale| < 0.5: {frac_below_half:.4f}")
        print(f"    Fraction (1+scale) < 0: {frac_negative:.4f}")

    # ========== Full forward pass comparison ==========
    print(f"\n\n{'='*80}")
    print("FULL FORWARD PASS: Compare prediction magnitude")
    print("=" * 80)

    # Get a known target for reference
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

    print(f"\n  x_0 (real latent): std={x_0.std():.4f}")

    # Create known noisy sample at sigma=0.5 (mid-range)
    torch.manual_seed(42)
    noise = torch.randn_like(x_0)
    target = noise - x_0  # true velocity target
    print(f"  target (ε - x_0): std={target.std():.4f}")

    # Test at a specific sigma
    scheduler.set_timesteps(1000, device=device)
    test_idx = 500
    t_test = scheduler.timesteps[test_idx]
    sigma_test = scheduler.sigmas[test_idx]

    x_t = (1 - sigma_test) * x_0 + sigma_test * noise
    timestep_test = torch.full((batch_size, N), t_test.item(), device=device, dtype=torch.bfloat16)

    print(f"\n  Testing at step_idx={test_idx}, t={t_test.item():.2f}, σ={sigma_test.item():.4f}")
    print(f"  x_t std: {x_t.std():.4f}")

    with torch.no_grad():
        pred = transformer(
            hidden_states=x_t.to(torch.bfloat16),
            timestep=timestep_test,
            encoder_hidden_states=text_states,
            attention_kwargs=None,
            is_causal=False,
            hidden_states_mask=motion_mask,
        ).float()

    print(f"\n  Model prediction: std={pred.std():.4f}, mean={pred.mean():.4f}")
    print(f"  Target: std={target.std():.4f}")
    print(f"  Ratio (pred/target): {pred.std() / target.std():.4f}")

    cos = torch.nn.functional.cosine_similarity(
        pred.flatten().unsqueeze(0), target.float().flatten().unsqueeze(0)
    ).item()
    print(f"  Cosine similarity: {cos:.4f}")

    # ========== Now check if it's temb or proj_out causing the issue ==========
    print(f"\n\n{'='*80}")
    print("ISOLATING THE CAUSE: temb vs proj_out vs hidden_states")
    print("=" * 80)

    # Run the full forward up to just before output norm
    # We need to hook into the model

    # Register a hook on norm_out to capture its input
    norm_input_captured = {}
    def hook_norm_out_input(module, input, output):
        norm_input_captured['input'] = input[0].float().detach()
        norm_input_captured['output'] = output.float().detach()

    hook_handle = transformer.norm_out.register_forward_hook(hook_norm_out_input)

    # Also hook proj_out
    proj_input_captured = {}
    def hook_proj_out_input(module, input, output):
        proj_input_captured['input'] = input[0].float().detach()
        proj_input_captured['output'] = output.float().detach()

    hook_handle2 = transformer.proj_out.register_forward_hook(hook_proj_out_input)

    with torch.no_grad():
        pred2 = transformer(
            hidden_states=x_t.to(torch.bfloat16),
            timestep=timestep_test,
            encoder_hidden_states=text_states,
            attention_kwargs=None,
            is_causal=False,
            hidden_states_mask=motion_mask,
        ).float()

    hook_handle.remove()
    hook_handle2.remove()

    print(f"\n  [Stage A] Hidden states BEFORE norm_out (after all transformer blocks):")
    print(f"    shape: {norm_input_captured['input'].shape}")
    print(f"    std: {norm_input_captured['input'].std():.4f}")
    print(f"    mean: {norm_input_captured['input'].mean():.4f}")

    print(f"\n  [Stage B] After norm_out (LayerNorm, no affine):")
    print(f"    std: {norm_input_captured['output'].std():.4f}")
    print(f"    mean: {norm_input_captured['output'].mean():.4f}")

    print(f"\n  [Stage C] Input to proj_out (after scale/shift):")
    print(f"    shape: {proj_input_captured['input'].shape}")
    print(f"    std: {proj_input_captured['input'].std():.4f}")
    print(f"    mean: {proj_input_captured['input'].mean():.4f}")

    print(f"\n  [Stage D] Output of proj_out (final model output pre-unpatchify):")
    print(f"    shape: {proj_input_captured['output'].shape}")
    print(f"    std: {proj_input_captured['output'].std():.4f}")
    print(f"    mean: {proj_input_captured['output'].mean():.4f}")

    print(f"\n  [Stage E] Final output (after unpatchify):")
    print(f"    std: {pred2.std():.4f}")

    # Compute what the std SHOULD be at each stage
    print(f"\n  EXPECTED FLOW (if model predicts target correctly):")
    print(f"    target std = {target.std():.4f}")
    print(f"    After unpatchify: same as proj_out output (just reshape)")
    print(f"    proj_out output: should be {target.std():.4f}")
    print(f"    proj_out input: should be {target.std().item() / expected_proj_std * 1536**0.5:.4f}" if expected_proj_std > 0 else "    N/A")

    # Check attenuation ratios
    print(f"\n  ATTENUATION ANALYSIS:")
    ratio_bc = proj_input_captured['input'].std() / norm_input_captured['output'].std()
    ratio_cd = proj_input_captured['output'].std() / proj_input_captured['input'].std()
    print(f"    B→C (norm_out → scale/shift): ratio = {ratio_bc:.4f}")
    print(f"       This is effectively |(1 + scale)|_rms = {ratio_bc:.4f}")
    print(f"    C→D (proj_out linear): ratio = {ratio_cd:.4f}")
    print(f"    Overall B→D: {proj_input_captured['output'].std() / norm_input_captured['output'].std():.4f}")
    print(f"    Overall A→E (full): {pred2.std() / norm_input_captured['input'].std():.4f}")

    if ratio_bc < 0.5:
        print(f"\n  ❌ OUTPUT NORM IS THE BOTTLENECK! (1+scale) multiplier is ~{ratio_bc:.3f}")
        print(f"     temb values cause (1+scale) to be too small")
    elif ratio_cd < 0.5:
        print(f"\n  ❌ PROJ_OUT IS THE BOTTLENECK! Linear projection attenuates by {ratio_cd:.3f}")
    else:
        print(f"\n  → Neither stage individually causes >2x attenuation")
        print(f"     Combined effect: {ratio_bc * ratio_cd:.4f}")

    # ========== Check: what does all-zeros timestep (t=0) give? ==========
    print(f"\n\n{'='*80}")
    print("SANITY CHECK: temb at t=0 vs t=500 vs t=1000")
    print("=" * 80)

    for t_val in [0.0, 100.0, 500.0, 900.0, 999.0]:
        ts = torch.full((batch_size * N,), t_val, device=device, dtype=torch.bfloat16)
        with torch.no_grad():
            temb_test, _, _ = transformer.condition_embedder(
                ts, text_states, timestep_seq_len=N
            )
        temb_f = temb_test.float()
        # Compute scale
        sst_dev = transformer.scale_shift_table.to(temb_test.device).float()
        combined = sst_dev.unsqueeze(0) + temb_f.unsqueeze(2)
        _, sc = combined.chunk(2, dim=2)
        sc = sc.squeeze(2)
        mult = 1 + sc
        print(f"  t={t_val:6.1f}: temb mean={temb_f.mean():.4f} std={temb_f.std():.4f} | "
              f"(1+scale) mean={mult.mean():.4f} std={mult.std():.4f} abs_mean={mult.abs().mean():.4f}")


if __name__ == "__main__":
    main()
