"""Partial denoising diagnostic: encode GT, add partial noise, denoise.

Tests whether the model has learned anything by checking if it can recover
motion from various noise levels. If it recovers well at low noise but fails
at high noise, the model is partially trained.

Usage:
    python3 scripts/inference/diagnose_partial_denoise.py \
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

import hftrainer  # noqa: trigger auto-imports
from hftrainer.registry import MODEL_BUNDLES
from hftrainer.utils.checkpoint_utils import load_checkpoint
from hftrainer.models.motion.components.utils.geometry.rotation_convert import (
    rotation_6d_to_axis_angle,
    axis_angle_to_rotation_6d,
)
from hftrainer.models.motion.prism.gaussian_distribution import DiagonalGaussianDistributionNd
from diffusers.utils.torch_utils import randn_tensor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--gt-npz', default=None)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output-dir', default=None)
    return parser.parse_args()


@torch.no_grad()
def encode_gt_to_latents(bundle, gt_npz_path, device):
    """Encode ground truth NPZ to VAE latents."""
    data = dict(np.load(gt_npz_path, allow_pickle=True))

    transl_gt = data.get('transl', data.get('trans', None))
    poses_gt = data.get('poses', None)

    if transl_gt is None or poses_gt is None:
        print(f"  ERROR: NPZ missing 'transl'/'trans' or 'poses'")
        return None

    T = transl_gt.shape[0]
    max_frames = 129
    if T > max_frames:
        transl_gt = transl_gt[:max_frames]
        poses_gt = poses_gt[:max_frames]
        T = max_frames

    n_joints_in = poses_gt.shape[1] // 3
    sel = list(range(22))
    aa = poses_gt.reshape(T, n_joints_in, 3)[:, sel, :]  # [T, 22, 3]

    # Convert axis-angle to rotation_6d (column-major)
    aa_flat = aa.reshape(T * 22, 3)
    rot6d_col = axis_angle_to_rotation_6d(torch.from_numpy(aa_flat).float())
    rot6d_col = rot6d_col.reshape(T, 22, 6)

    # Column-major to row-major
    rot6d_row = rot6d_col[:, :, [0, 3, 1, 4, 2, 5]]
    poses_132 = rot6d_row.reshape(T, 22 * 6)

    # Translation abs_rel
    transl_t = torch.from_numpy(transl_gt).float()
    rel_t = torch.zeros_like(transl_t)
    rel_t[1:] = transl_t[1:] - transl_t[:-1]
    transl_abs_rel = torch.cat([transl_t, rel_t], dim=-1)  # [T, 6]

    # Full motion vector: [T, 138]
    motion_vec = torch.cat([transl_abs_rel, poses_132], dim=-1).unsqueeze(0)  # [1, T, 138]

    # Normalize
    smpl_processor = bundle.smpl_pose_processor
    motion_norm = smpl_processor.normalize(motion_vec)

    # Reshape for VAE
    motion_reshaped = rearrange(motion_norm, 'b t (j d) -> b t j d', d=6)  # [1, T, 23, 6]

    # Encode
    vae = bundle.vae.to(device)
    motion_dev = motion_reshaped.to(device).float()

    with torch.autocast(device.type if hasattr(device, 'type') else 'cuda', enabled=False):
        latents_raw = vae.encode(motion_dev)
    latents = DiagonalGaussianDistributionNd(latents_raw).mode()

    # Normalize latents
    latents_mean = bundle.latents_mean.to(latents)
    latents_std = bundle.latents_std.to(latents)
    latents_normed = (latents - latents_mean) / latents_std

    print(f"  GT latents shape: {latents_normed.shape}")
    print(f"  GT latents stats: mean={latents_normed.mean():.4f}, std={latents_normed.std():.4f}")
    print(f"  GT latents per-channel std: {latents_normed.std(dim=(2,3)).squeeze()[:5]}")

    return latents_normed


@torch.no_grad()
def denoise_from_noisy(bundle, gt_latents, noise_level, device, dtype=torch.bfloat16,
                       num_inference_steps=50, prompt="a person walks forward"):
    """Add noise at given level to GT latents, then denoise.

    noise_level: float 0-1, where 0 = no noise (pure GT), 1 = full noise
    """
    transformer = bundle.transformer.to(device, dtype)

    # Find the timestep closest to our desired noise level
    # In flow matching: noisy = (1-sigma)*clean + sigma*noise
    # sigma corresponds to noise_level
    bundle.scheduler.set_timesteps(num_inference_steps, device=device)
    all_timesteps = bundle.scheduler.timesteps
    all_sigmas = bundle.scheduler.sigmas.to(device)

    # Find the timestep index where sigma is closest to our desired noise_level
    # scheduler.sigmas is in descending order (1.0 -> 0.0)
    # We want to start denoising from the timestep corresponding to noise_level
    schedule_timesteps = bundle.scheduler.timesteps

    # Find start index
    start_idx = 0
    for i, t in enumerate(schedule_timesteps):
        # Get sigma for this timestep
        step_idx = (bundle.scheduler.timesteps == t).nonzero().item()
        sigma = all_sigmas[step_idx].item()
        if sigma <= noise_level:
            start_idx = i
            break

    if noise_level >= 0.99:
        start_idx = 0  # Start from maximum noise

    print(f"  noise_level={noise_level:.2f}, starting from step {start_idx}/{len(schedule_timesteps)}")
    print(f"  (will run {len(schedule_timesteps) - start_idx} denoising steps)")

    # Add noise at the desired level
    noise = torch.randn_like(gt_latents)
    noisy_latents = (1 - noise_level) * gt_latents + noise_level * noise
    latents = noisy_latents.to(device)

    print(f"  Noisy latents stats: mean={latents.mean():.4f}, std={latents.std():.4f}")

    # Encode prompt
    tokenizer = bundle.tokenizer
    text_encoder = bundle.text_encoder

    # Check if text_encoder is still available
    if text_encoder is not None:
        text_encoder = text_encoder.to(device, dtype)
        inputs = tokenizer(
            prompt, padding="max_length", max_length=256,
            truncation=True, return_tensors="pt",
        )
        text_output = text_encoder(
            input_ids=inputs.input_ids.to(device),
            attention_mask=inputs.attention_mask.to(device),
        )
        text_states = text_output.last_hidden_state * inputs.attention_mask.unsqueeze(-1).float().to(device)
        text_states = text_states.to(dtype)
    else:
        print("  ERROR: text_encoder not available")
        return None

    # Motion mask
    batch_size = 1
    motion_mask = torch.ones(batch_size, latents.shape[2], latents.shape[3], device=device)

    # Only denoise from start_idx onwards
    timesteps_to_run = schedule_timesteps[start_idx:]

    for t in timesteps_to_run:
        # Per-token timestep (expand_timesteps=True, no conditioning)
        first_frame_mask = torch.ones(1, 1, latents.shape[2], latents.shape[3], device=device)
        temp_ts = (first_frame_mask[0][0] * t).flatten()
        timestep = temp_ts.unsqueeze(0).expand(batch_size, -1)

        latent_model_input = latents.to(dtype)

        noise_pred = transformer(
            hidden_states=latent_model_input,
            timestep=timestep,
            encoder_hidden_states=text_states,
            attention_kwargs=None,
            is_causal=False,
            hidden_states_mask=motion_mask,
        )

        latents = bundle.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    return latents


@torch.no_grad()
def decode_latents_to_height(bundle, latents, device):
    """Decode latents and compute body height via FK."""
    vae = bundle.vae.to(device)
    smpl_processor = bundle.smpl_pose_processor

    # Denormalize latents
    latents_mean = bundle.latents_mean.to(latents)
    latents_std = bundle.latents_std.to(latents)
    latents_denorm = latents * latents_std + latents_mean

    # Decode
    with torch.autocast(device.type if hasattr(device, 'type') else 'cuda', enabled=False):
        motion = vae.decode(latents_denorm.float())  # [B, T, J, 6]

    # Post-process
    x_dec = rearrange(motion, "b t j d -> b t (j d)")
    x_dec = smpl_processor.denormalize(x_dec)

    transl_abs_rel = x_dec[..., :6]
    transl = smpl_processor.inv_convert_transl(transl_abs_rel)
    pred_poses = x_dec[..., 6:]

    pred_poses = rearrange(pred_poses, "b t (j d) -> (b t) j d", d=6)
    pred_poses = pred_poses[..., [0, 2, 4, 1, 3, 5]]
    pred_poses = rotation_6d_to_axis_angle(pred_poses)
    pred_poses = rearrange(pred_poses, "(b t) j d -> b t (j d)", b=1)

    pred_smplx_dict = smpl_processor.transl_pose_to_smplx_dict(
        transl.squeeze(0), pred_poses.squeeze(0),
        mocap_framerate=30.0, gender='neutral', rot_type="axis_angle",
    )
    pred_smplx_dict = smpl_processor.normalize_smplx_dict(pred_smplx_dict)

    # FK check
    transl_np = pred_smplx_dict['transl']
    global_orient = pred_smplx_dict['global_orient']
    body_pose = pred_smplx_dict['body_pose']

    T = transl_np.shape[0]
    transl_t = torch.from_numpy(transl_np).float()
    go_t = torch.from_numpy(global_orient).float()
    bp_t = torch.from_numpy(body_pose).float()
    betas = torch.zeros(T, 10)

    smpl_model = smpl_processor.smpl_model
    try:
        dev = next(smpl_model.parameters()).device
    except StopIteration:
        dev = next(smpl_model.buffers()).device

    all_joints = []
    chunk_size = 32
    for i in range(0, T, chunk_size):
        output = smpl_model(
            body_pose=bp_t[i:i+chunk_size].to(dev),
            betas=betas[i:i+chunk_size].to(dev),
            global_orient=go_t[i:i+chunk_size].to(dev),
            transl=transl_t[i:i+chunk_size].to(dev),
        )
        joints = output[1] if isinstance(output, tuple) else output
        all_joints.append(joints.cpu())

    joints = torch.cat(all_joints, dim=0)
    heights = joints[:, :, 1].max(dim=1).values - joints[:, :, 1].min(dim=1).values

    return heights.mean().item(), latents.std().item()


def main():
    args = parse_args()
    device = torch.device(args.device)

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

    # Find GT NPZ
    gt_npz = args.gt_npz
    if gt_npz is None:
        candidates = [
            "data/hymotion_data/Academic/20250916/motions/HumanML3D-HumanEva/S3_Box_1_poses_origintime_13.55_22.7.npz",
        ]
        for c in candidates:
            if os.path.isfile(c):
                gt_npz = c
                break

    if not gt_npz or not os.path.isfile(gt_npz):
        print(f"ERROR: No GT NPZ found. Pass --gt-npz <path>")
        return

    print(f"\n{'='*60}")
    print(f"  Partial Denoising Diagnostic")
    print(f"  GT NPZ: {gt_npz}")
    print(f"{'='*60}")

    # Step 1: Encode GT
    print("\n--- Step 1: Encoding GT to latents ---")
    gt_latents = encode_gt_to_latents(bundle, gt_npz, device)
    if gt_latents is None:
        return

    # Step 2: Compute GT body height (decode without any noise)
    print("\n--- Step 2: GT decode check (no noise) ---")
    gt_height, gt_std = decode_latents_to_height(bundle, gt_latents.to(device), device)
    print(f"  GT decoded height: {gt_height:.3f}m")
    print(f"  GT latent std: {gt_std:.4f}")

    # Step 3: Test various noise levels
    noise_levels = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]

    print(f"\n{'='*60}")
    print(f"  Testing noise levels: {noise_levels}")
    print(f"  Expected height: ~{gt_height:.2f}m")
    print(f"{'='*60}")

    results = []
    for nl in noise_levels:
        print(f"\n--- Noise level: {nl:.2f} ---")
        denoised = denoise_from_noisy(
            bundle, gt_latents.to(device), nl, device,
            num_inference_steps=50,
            prompt="a person is boxing",
        )
        if denoised is not None:
            height, lat_std = decode_latents_to_height(bundle, denoised, device)
            ratio = height / gt_height if gt_height > 0 else 0
            results.append((nl, height, lat_std, ratio))
            status = "PASS" if 0.8 < ratio < 1.2 else "FAIL"
            print(f"  Result: height={height:.3f}m, latent_std={lat_std:.4f}, "
                  f"ratio={ratio:.3f} [{status}]")

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Noise':<8} {'Height':<10} {'Lat Std':<10} {'Ratio':<8} {'Status'}")
    print(f"  {'-----':<8} {'------':<10} {'-------':<10} {'-----':<8} {'------'}")
    print(f"  {'0.00':<8} {gt_height:<10.3f} {gt_std:<10.4f} {'1.000':<8} {'GT'}")
    for nl, h, ls, r in results:
        status = "PASS" if 0.8 < r < 1.2 else "FAIL"
        print(f"  {nl:<8.2f} {h:<10.3f} {ls:<10.4f} {r:<8.3f} {status}")

    # Conclusion
    print(f"\n  INTERPRETATION:")
    passing = [r for r in results if 0.8 < r[3] < 1.2]
    failing = [r for r in results if not (0.8 < r[3] < 1.2)]

    if len(passing) == len(results):
        print("  Model recovers at ALL noise levels → inference pipeline has a bug")
    elif len(passing) == 0:
        print("  Model FAILS at ALL noise levels → model is undertrained/broken")
    else:
        max_passing_nl = max(r[0] for r in passing)
        min_failing_nl = min(r[0] for r in failing)
        print(f"  Model works up to noise_level={max_passing_nl:.2f}, fails at {min_failing_nl:.2f}")
        print(f"  → Model is PARTIALLY trained. Cannot generate from pure noise yet.")
        print(f"  → Need more training iterations (currently at 15K).")

    # Save results if output dir specified
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        results_path = os.path.join(args.output_dir, 'partial_denoise_results.txt')
        with open(results_path, 'w') as f:
            f.write(f"GT height: {gt_height:.3f}m\n")
            f.write(f"GT latent std: {gt_std:.4f}\n\n")
            f.write(f"{'Noise':<8} {'Height':<10} {'Lat Std':<10} {'Ratio':<8} {'Status'}\n")
            for nl, h, ls, r in results:
                status = "PASS" if 0.8 < r < 1.2 else "FAIL"
                f.write(f"{nl:<8.2f} {h:<10.3f} {ls:<10.4f} {r:<8.3f} {status}\n")
        print(f"\n  Results saved to: {results_path}")


if __name__ == '__main__':
    main()
