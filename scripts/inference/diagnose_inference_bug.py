"""Comprehensive diagnostic for PRISM inference deformation bug.

Tests multiple hypotheses:
1. Text length mismatch (128 vs 256)
2. CFG scale issues (no CFG vs CFG=5.0)
3. Model single-step accuracy on real data
4. Latent statistics at each denoising step
5. expand_timesteps=True vs False

Usage:
    python3 scripts/inference/diagnose_inference_bug.py \
        --config configs/prism/prism_1b_tp2m_multiframe.py \
        --checkpoint work_dirs/prism_1b_tp2m_multiframe/checkpoint-iter_15000
"""

import argparse
import gc
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
from hftrainer.models.motion.components.utils.geometry.rotation_convert import (
    rotation_6d_to_axis_angle,
    rotation_6d_to_matrix,
)
from diffusers.utils.torch_utils import randn_tensor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()


def print_section(title):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")


def check_rotation_validity(rot6d_row_major, label=""):
    """Check if rot6d values are valid rotations.

    For valid rotations:
    - Convert from row-major [R00,R01,R10,R11,R20,R21] to column-major [R00,R10,R20,R01,R11,R21]
    - Each column (3 elements) should have norm ~1.0
    """
    # Convert row-major to column-major
    col_major = rot6d_row_major[..., [0, 2, 4, 1, 3, 5]]
    col1 = col_major[..., :3]
    col2 = col_major[..., 3:]
    col1_norms = col1.norm(dim=-1)
    col2_norms = col2.norm(dim=-1)
    print(f"  [{label}] Column 1 norms: mean={col1_norms.mean():.4f}, std={col1_norms.std():.4f}, "
          f"min={col1_norms.min():.4f}, max={col1_norms.max():.4f}")
    print(f"  [{label}] Column 2 norms: mean={col2_norms.mean():.4f}, std={col2_norms.std():.4f}, "
          f"min={col2_norms.min():.4f}, max={col2_norms.max():.4f}")
    print(f"  [{label}] Valid rotation (both norms ~1.0): "
          f"col1={'YES' if abs(col1_norms.mean() - 1.0) < 0.2 else 'NO'}, "
          f"col2={'YES' if abs(col2_norms.mean() - 1.0) < 0.2 else 'NO'}")


@torch.no_grad()
def test_single_step_accuracy(bundle, device, dtype):
    """Test 1: Load real training data, add noise, check model prediction accuracy."""
    print_section("TEST 1: Single-Step Prediction Accuracy on Real Training Data")

    # Find a real training data file
    data_dir = "/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/data"
    # Load annotation to find a motion file
    import json
    anno_path = os.path.join(data_dir, "annotation/train_hymotion_400h.json")
    if not os.path.exists(anno_path):
        print("  [SKIP] Training annotation not found")
        return None

    with open(anno_path, 'r') as f:
        anno = json.load(f)

    # Get first motion path
    data_list = anno['data_list']
    first_key = list(data_list.keys())[0]
    motion_rel_path = data_list[first_key]['smplx_path']
    motion_path = os.path.join(data_dir, motion_rel_path)

    if not os.path.exists(motion_path):
        print(f"  [SKIP] Motion file not found: {motion_path}")
        return None

    print(f"  Loading motion: {motion_path}")

    # Load and process through the LoadSmplx55 transform (same as training)
    from hftrainer.datasets.motion.motionhub.transforms.load_smplx import (
        process_smplx_pose, process_transl
    )

    npz_data = np.load(motion_path, allow_pickle=True)
    abs_trans = np.asarray(npz_data["trans"], dtype=np.float32)
    poses = np.asarray(npz_data["poses"], dtype=np.float32)

    # Process: axis_angle -> rotation_6d (row-major), smpl_22
    pose_rot6d = process_smplx_pose(poses, "rotation_6d", "smpl_22")  # [T, 132]
    transl = process_transl(abs_trans, "abs_rel")  # [T, 6]
    motion_vec = np.concatenate([transl, pose_rot6d], axis=-1)  # [T, 138]
    motion_tensor = torch.from_numpy(motion_vec).float()

    print(f"  Motion shape: {motion_tensor.shape}")
    T = motion_tensor.shape[0]

    # Check rotation validity of raw data
    pose_part = motion_tensor[:, 6:].reshape(T, 22, 6)
    check_rotation_validity(pose_part, "GT training data")

    # Take first 129 frames (or pad)
    target_frames = 129
    if T >= target_frames:
        motion_tensor = motion_tensor[:target_frames]
    else:
        # Pad with repeat of last frame
        pad = motion_tensor[-1:].expand(target_frames - T, -1)
        motion_tensor = torch.cat([motion_tensor, pad], dim=0)

    # Normalize
    motion_norm = bundle.smpl_pose_processor.normalize(motion_tensor.unsqueeze(0))  # [1, T, 138]

    # Reshape for VAE: [1, T, 138] -> [1, T, 23, 6]
    motion_4d = rearrange(motion_norm, 'b t (j d) -> b t j d', d=6)

    # VAE encode
    device_type = device.type if hasattr(device, 'type') else 'cuda'
    motion_4d_dev = motion_4d.to(device)
    with torch.autocast(device_type, enabled=False):
        z_raw = bundle.vae.encode(motion_4d_dev.float())

    from hftrainer.models.motion.prism.gaussian_distribution import DiagonalGaussianDistributionNd
    lat = DiagonalGaussianDistributionNd(z_raw)
    gt_latents = lat.mode()  # [1, 16, T_latent, 23]

    # Normalize latents
    latents_mean = bundle.latents_mean.to(gt_latents)
    latents_std = bundle.latents_std.to(gt_latents)
    gt_latents_norm = (gt_latents - latents_mean) / latents_std

    print(f"\n  GT Latent stats (normalized):")
    print(f"    Shape: {gt_latents_norm.shape}")
    print(f"    Mean: {gt_latents_norm.mean():.4f}")
    print(f"    Std: {gt_latents_norm.std():.4f}")
    print(f"    Per-channel std: {gt_latents_norm.squeeze(0).std(dim=(1,2)).cpu().tolist()}")

    # Now test single-step prediction at various noise levels
    print(f"\n  Testing model prediction accuracy at different noise levels:")

    # Get scheduler timesteps (training uses full 1000)
    bundle.scheduler.set_timesteps(1000, device=device)
    all_timesteps = bundle.scheduler.timesteps

    # Test at t=100 (low noise), t=500 (medium), t=900 (high noise)
    test_indices = [100, 500, 900]

    for idx in test_indices:
        t = all_timesteps[idx:idx+1]
        noise = torch.randn_like(gt_latents_norm)

        # Get sigma for this timestep
        from hftrainer.models.motion.prism.bundle import _get_sigmas
        sigma = _get_sigmas(bundle.scheduler, t, n_dim=4, dtype=gt_latents_norm.dtype)

        # Create noisy latent
        noisy_latent = (1 - sigma) * gt_latents_norm + sigma * noise
        target = noise - gt_latents_norm  # velocity

        # Create per-token timestep (expand_timesteps=True, no conditioning)
        B, C, T_lat, J = noisy_latent.shape
        timestep_expanded = t.unsqueeze(1).expand(B, T_lat * J)

        # Model prediction
        noisy_input = noisy_latent.to(dtype)
        pred = bundle.transformer(
            hidden_states=noisy_input,
            timestep=timestep_expanded,
            encoder_hidden_states=torch.zeros(1, 256, 4096, device=device, dtype=dtype),  # uncond
            hidden_states_mask=torch.ones(1, T_lat, J, device=device),
        ).float()

        # Compare
        mse = ((pred - target) ** 2).mean().item()
        cos_sim = torch.nn.functional.cosine_similarity(
            pred.flatten(), target.flatten(), dim=0
        ).item()
        pred_std = pred.std().item()
        target_std = target.std().item()

        print(f"    t_idx={idx:4d} (t={t.item():.1f}, sigma={sigma.squeeze().item():.4f}): "
              f"MSE={mse:.4f}, cos_sim={cos_sim:.4f}, "
              f"pred_std={pred_std:.4f}, target_std={target_std:.4f}")

    return gt_latents_norm


@torch.no_grad()
def test_denoising_variants(bundle, gt_latents, device, dtype):
    """Test 2: Run denoising with different configurations and compare results."""
    print_section("TEST 2: Denoising Loop Variants")

    prompt = "a person walks forward slowly"
    num_frames = 129
    num_joints = 23
    vae_temporal = bundle.vae.config.scale_factor_temporal
    num_latent_frames = (num_frames - 1) // vae_temporal + 1
    num_channels = bundle.transformer.config.in_channels

    # Encode text with CORRECT length (256, matching training)
    text_states_256 = bundle.encode_prompt(
        prompt, max_sequence_length=256,
        dtype=dtype,
    ).to(device)  # [1, 256, 4096]

    neg_text_states_256 = bundle.encode_prompt(
        "", max_sequence_length=256,
        dtype=dtype,
    ).to(device)

    # Also encode with 128 (what the lowmem script used)
    text_states_128 = bundle.encode_prompt(
        prompt, max_sequence_length=128,
        dtype=dtype,
    ).to(device)

    neg_text_states_128 = bundle.encode_prompt(
        "", max_sequence_length=128,
        dtype=dtype,
    ).to(device)

    print(f"  Text states 256 shape: {text_states_256.shape}")
    print(f"  Text states 128 shape: {text_states_128.shape}")
    print(f"  Text states 256 norm: {text_states_256.float().norm():.4f}")
    print(f"  Text states 128 norm: {text_states_128.float().norm():.4f}")

    # Fixed seed for comparison
    generator = torch.Generator(device=device).manual_seed(42)

    configs = [
        {"name": "cfg=1.0, text_len=256, expand_ts=True", "cfg": 1.0, "text": text_states_256, "neg": neg_text_states_256, "expand": True},
        {"name": "cfg=5.0, text_len=256, expand_ts=True", "cfg": 5.0, "text": text_states_256, "neg": neg_text_states_256, "expand": True},
        {"name": "cfg=2.0, text_len=256, expand_ts=True", "cfg": 2.0, "text": text_states_256, "neg": neg_text_states_256, "expand": True},
        {"name": "cfg=5.0, text_len=128, expand_ts=True", "cfg": 5.0, "text": text_states_128, "neg": neg_text_states_128, "expand": True},
        {"name": "cfg=5.0, text_len=256, expand_ts=False", "cfg": 5.0, "text": text_states_256, "neg": neg_text_states_256, "expand": False},
    ]

    results = {}

    for config in configs:
        print(f"\n  Running: {config['name']}")

        # Reset seed for fair comparison
        generator = torch.Generator(device=device).manual_seed(42)
        shape = (1, num_channels, num_latent_frames, num_joints)
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        # Set up scheduler
        bundle.scheduler.set_timesteps(50, device=device)
        timesteps = bundle.scheduler.timesteps

        do_cfg = config['cfg'] > 1.0

        # Track latent stats per step
        step_stats = []

        for i, t in enumerate(timesteps):
            if config['expand']:
                # Per-token timestep (all same since no conditioning)
                B, C, T_lat, J = latents.shape
                timestep = t.unsqueeze(0).unsqueeze(1).expand(1, T_lat * J)
            else:
                timestep = t.expand(1)

            latent_input = latents.to(dtype)

            # Forward pass
            noise_pred = bundle.transformer(
                hidden_states=latent_input,
                timestep=timestep,
                encoder_hidden_states=config['text'],
                hidden_states_mask=torch.ones(1, num_latent_frames, num_joints, device=device),
            )

            if do_cfg:
                noise_uncond = bundle.transformer(
                    hidden_states=latent_input,
                    timestep=timestep,
                    encoder_hidden_states=config['neg'],
                    hidden_states_mask=torch.ones(1, num_latent_frames, num_joints, device=device),
                )
                noise_pred = noise_uncond + config['cfg'] * (noise_pred - noise_uncond)

            # Record model output stats
            if i % 10 == 0 or i == len(timesteps) - 1:
                step_stats.append({
                    'step': i,
                    't': t.item(),
                    'pred_mean': noise_pred.float().mean().item(),
                    'pred_std': noise_pred.float().std().item(),
                    'latent_mean': latents.float().mean().item(),
                    'latent_std': latents.float().std().item(),
                })

            # Scheduler step
            latents = bundle.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        # Final latent stats
        final_latent = latents.float()
        results[config['name']] = {
            'latent': final_latent.cpu(),
            'mean': final_latent.mean().item(),
            'std': final_latent.std().item(),
            'step_stats': step_stats,
        }

        print(f"    Final latent: mean={final_latent.mean():.4f}, std={final_latent.std():.4f}")
        for ss in step_stats:
            print(f"    Step {ss['step']:3d} (t={ss['t']:.1f}): "
                  f"pred_std={ss['pred_std']:.4f}, latent_std={ss['latent_std']:.4f}")

    # Compare with GT latent stats
    if gt_latents is not None:
        print(f"\n  GT latent reference: mean={gt_latents.mean():.4f}, std={gt_latents.std():.4f}")

    return results


@torch.no_grad()
def test_decode_and_validate(bundle, results, device):
    """Test 3: Decode the best result and validate rotation quality."""
    print_section("TEST 3: Decode and Validate Rotation Quality")

    for name, result in results.items():
        print(f"\n  Decoding: {name}")
        latents = result['latent'].to(device)

        # Denormalize latents
        latents_mean = bundle.latents_mean.to(latents)
        latents_std = bundle.latents_std.to(latents)
        latents_denorm = latents * latents_std + latents_mean

        # VAE decode
        device_type = device.type if hasattr(device, 'type') else 'cuda'
        with torch.autocast(device_type, enabled=False):
            motion = bundle.vae.decode(latents_denorm.float())  # [B, T, J, D]

        # Denormalize motion
        x_dec = rearrange(motion, "b t j d -> b t (j d)")
        x_dec = bundle.smpl_pose_processor.denormalize(x_dec)

        # Extract poses (skip first 6 dims = translation)
        poses = x_dec[..., 6:]  # [1, T, 132]
        poses_6d = poses.reshape(-1, 22, 6)  # [T, 22, 6] in row-major

        check_rotation_validity(poses_6d, name)

        # Also check translation reasonableness
        transl_abs_rel = x_dec[..., :6]
        transl = bundle.smpl_pose_processor.inv_convert_transl(transl_abs_rel)
        print(f"  Translation range: x=[{transl[...,0].min():.2f}, {transl[...,0].max():.2f}], "
              f"y=[{transl[...,1].min():.2f}, {transl[...,1].max():.2f}], "
              f"z=[{transl[...,2].min():.2f}, {transl[...,2].max():.2f}]")


@torch.no_grad()
def test_text_conditioning_effect(bundle, device, dtype):
    """Test 4: Check if text conditioning has any effect."""
    print_section("TEST 4: Text Conditioning Effect")

    prompts = [
        "a person walks forward slowly",
        "a person jumps high into the air",
        "a person sits down on a chair",
    ]

    num_latent_frames = 33
    num_joints = 23
    num_channels = bundle.transformer.config.in_channels

    # Encode all prompts
    text_states_list = []
    for p in prompts:
        ts = bundle.encode_prompt(p, max_sequence_length=256, dtype=dtype).to(device)
        text_states_list.append(ts)

    # Use same noise for all
    generator = torch.Generator(device=device).manual_seed(123)
    shape = (1, num_channels, num_latent_frames, num_joints)
    latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

    # Single step prediction with different text conditioning
    bundle.scheduler.set_timesteps(50, device=device)
    t = bundle.scheduler.timesteps[0]  # First (highest noise) timestep

    timestep = t.unsqueeze(0).unsqueeze(1).expand(1, num_latent_frames * num_joints)

    preds = []
    for i, ts in enumerate(text_states_list):
        pred = bundle.transformer(
            hidden_states=latents.to(dtype),
            timestep=timestep,
            encoder_hidden_states=ts,
            hidden_states_mask=torch.ones(1, num_latent_frames, num_joints, device=device),
        ).float()
        preds.append(pred)
        print(f"  Prompt '{prompts[i][:30]}...': pred mean={pred.mean():.4f}, std={pred.std():.4f}")

    # Check if predictions differ across prompts
    for i in range(len(preds)):
        for j in range(i+1, len(preds)):
            diff = (preds[i] - preds[j]).abs()
            print(f"  Diff between prompt {i} and {j}: mean={diff.mean():.6f}, max={diff.max():.6f}")


def main():
    args = parse_args()
    device = torch.device(args.device)
    dtype = torch.bfloat16

    print("="*80)
    print("  PRISM INFERENCE DIAGNOSTIC")
    print("="*80)

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

    # Move everything to GPU
    bundle.transformer = bundle.transformer.to(device, dtype)
    bundle.vae = bundle.vae.to(device)
    bundle.text_encoder = bundle.text_encoder.to(device)
    torch.cuda.empty_cache()

    # Run tests
    gt_latents = test_single_step_accuracy(bundle, device, dtype)

    test_text_conditioning_effect(bundle, device, dtype)

    results = test_denoising_variants(bundle, gt_latents, device, dtype)

    test_decode_and_validate(bundle, results, device)

    print_section("DIAGNOSTIC COMPLETE")
    print("\n  Summary of findings:")
    if gt_latents is not None:
        print(f"  - GT latent std: {gt_latents.std():.4f}")
    for name, r in results.items():
        print(f"  - {name}: final std={r['std']:.4f}")


if __name__ == '__main__':
    main()
