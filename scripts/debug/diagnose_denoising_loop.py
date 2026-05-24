"""Diagnose PRISM denoising loop: track latent statistics at every step.

Tests:
1. Full denoising with CFG=5.0 (same as inference) - check if final latents are reasonable
2. Full denoising with CFG=1.0 (no guidance) - isolate CFG issue
3. Check float32 vs bfloat16 precision differences
4. After denoising, decode and check body height

Usage:
    cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
    python3 scripts/debug/diagnose_denoising_loop.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import gc
import numpy as np
import torch
from einops import rearrange
from mmengine.config import Config
from scipy.spatial.transform import Rotation as R

import hftrainer  # noqa
from hftrainer.registry import MODEL_BUNDLES
from hftrainer.utils.checkpoint_utils import load_checkpoint
from hftrainer.models.motion.components.utils.geometry.rotation_convert import (
    rotation_6d_to_axis_angle,
)
from diffusers.utils.torch_utils import randn_tensor


SMPL_22_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]


def load_j_template():
    template_path = "/apdcephfs_cq11/share_1467498/home/zeyuling/motion_vis_web/static/assets/j_template_22.npy"
    return np.load(template_path).astype(np.float32)


def numpy_fk_22(transl, poses_aa, J_template):
    T = transl.shape[0]
    aa = poses_aa.reshape(T, 22, 3)
    local_rots = R.from_rotvec(aa.reshape(-1, 3)).as_matrix().astype(np.float32).reshape(T, 22, 3, 3)
    offsets = np.zeros((22, 3), dtype=np.float32)
    offsets[0] = J_template[0]
    for j in range(1, 22):
        offsets[j] = J_template[j] - J_template[SMPL_22_PARENTS[j]]
    positions = np.zeros((T, 22, 3), dtype=np.float32)
    rotations = np.zeros((T, 22, 3, 3), dtype=np.float32)
    rotations[:, 0] = local_rots[:, 0]
    positions[:, 0] = transl + offsets[0]
    for j in range(1, 22):
        p = SMPL_22_PARENTS[j]
        rotations[:, j] = np.matmul(rotations[:, p], local_rots[:, j])
        offset_rot = np.einsum("tij,j->ti", rotations[:, p], offsets[j])
        positions[:, j] = positions[:, p] + offset_rot
    return positions


def compute_body_height(positions):
    head = positions[:, 15, :]
    feet = np.minimum(
        np.minimum(positions[:, 7, :], positions[:, 8, :]),
        np.minimum(positions[:, 10, :], positions[:, 11, :])
    )
    height = head[:, 1] - feet[:, 1]
    return float(np.mean(height))


def decode_latents(bundle, latents, device):
    """Decode latents to body height."""
    vae = bundle.vae.to(device)
    latents_mean = bundle.latents_mean.to(latents)
    latents_std = bundle.latents_std.to(latents)
    latents_denorm = latents * latents_std + latents_mean

    with torch.autocast(device.type, enabled=False):
        motion = vae.decode(latents_denorm.float())

    smpl_proc = bundle.smpl_pose_processor
    x_dec = rearrange(motion, "b t j d -> b t (j d)")
    x_dec = smpl_proc.denormalize(x_dec)

    transl_abs_rel = x_dec[..., :6]
    transl = smpl_proc.inv_convert_transl(transl_abs_rel)
    pred_poses = x_dec[..., 6:]
    pred_poses = rearrange(pred_poses, "b t (j d) -> (b t) j d", d=6)
    pred_poses = pred_poses[..., [0, 2, 4, 1, 3, 5]]
    pred_poses = rotation_6d_to_axis_angle(pred_poses)
    pred_poses = rearrange(pred_poses, "(b t) j d -> b t (j d)", b=1)

    J_template = load_j_template()
    transl_np = transl.squeeze(0).cpu().numpy()
    poses_np = pred_poses.squeeze(0).cpu().numpy()
    joints = numpy_fk_22(transl_np, poses_np, J_template)
    return compute_body_height(joints)


@torch.no_grad()
def run_denoising(
    transformer, scheduler, text_states, neg_text_states,
    num_frames, num_joints, vae_temporal, num_steps, guidance_scale,
    device, dtype, seed=42, label="",
):
    """Run full denoising loop and track statistics."""
    batch_size = 1
    num_latent_frames = (num_frames - 1) // vae_temporal + 1
    num_channels = transformer.config.in_channels

    shape = (batch_size, num_channels, num_latent_frames, num_joints)
    generator = torch.Generator(device=device).manual_seed(seed)
    latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

    first_frame_mask = torch.ones_like(latents)
    condition = torch.zeros_like(latents)

    motion_mask = torch.ones(batch_size, num_latent_frames, num_joints, device=device)

    scheduler.set_timesteps(num_steps, device=device)
    timesteps = scheduler.timesteps

    do_cfg = guidance_scale > 1.0
    text_dev = text_states.to(device=device, dtype=dtype)
    neg_dev = neg_text_states.to(device=device, dtype=dtype) if do_cfg else None

    print(f"\n{'='*60}")
    print(f"  Denoising: {label}")
    print(f"  dtype={dtype}, cfg={guidance_scale}, steps={num_steps}, seed={seed}")
    print(f"  Initial latents: mean={latents.mean():.4f}, std={latents.std():.4f}")
    print(f"{'='*60}")

    for i, t in enumerate(timesteps):
        # Prepare input (expand_timesteps=True)
        latent_model_input = (
            (1 - first_frame_mask) * condition + first_frame_mask * latents
        ).to(dtype)
        temp_ts = (first_frame_mask[0][0] * t).flatten()
        timestep = temp_ts.unsqueeze(0).expand(batch_size, -1)

        # Forward pass
        noise_pred = transformer(
            hidden_states=latent_model_input,
            timestep=timestep,
            encoder_hidden_states=text_dev,
            attention_kwargs=None,
            is_causal=False,
            hidden_states_mask=motion_mask,
        )

        # CFG
        if do_cfg:
            noise_uncond = transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=neg_dev,
                attention_kwargs=None,
                is_causal=False,
                hidden_states_mask=motion_mask,
            )
            noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)

        # Scheduler step
        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        # Log every 10 steps + first + last
        if i % 10 == 0 or i == len(timesteps) - 1:
            sigma = scheduler.sigmas[i]
            pred_std = noise_pred.std().item()
            lat_std = latents.std().item()
            lat_mean = latents.mean().item()
            print(f"  Step {i:3d} | t={t:.1f} σ={sigma:.4f} | "
                  f"pred: std={pred_std:.4f} | "
                  f"latents: mean={lat_mean:.4f} std={lat_std:.4f}")

    print(f"  FINAL latents: mean={latents.mean():.4f}, std={latents.std():.4f}, "
          f"range=[{latents.min():.3f}, {latents.max():.3f}]")
    return latents


def main():
    device = torch.device('cuda')
    config_path = "configs/prism/prism_1b_tp2m_multiframe.py"
    ckpt_path = "work_dirs/prism_1b_tp2m_multiframe/checkpoint-iter_15000"

    print("Loading bundle...")
    cfg = Config.fromfile(config_path)
    model_cfg = cfg.model.to_dict() if hasattr(cfg.model, 'to_dict') else cfg.model
    bundle_cls = MODEL_BUNDLES.get(model_cfg['type'])
    bundle = bundle_cls.from_config(model_cfg)
    bundle.eval()

    state_dict = load_checkpoint(ckpt_path, map_location='cpu')
    bundle.load_state_dict_selective(state_dict)
    del state_dict
    gc.collect()

    # Encode text (CPU)
    print("Encoding text on CPU...")
    prompt = "a person walks forward slowly"
    text_states = bundle.encode_prompt(
        prompt, max_sequence_length=256, dtype=torch.bfloat16,
    )
    neg_text_states = bundle.encode_prompt(
        "", max_sequence_length=256, dtype=torch.bfloat16,
    )
    print(f"  text_states shape: {text_states.shape}")
    print(f"  neg_text_states shape: {neg_text_states.shape}")
    print(f"  neg_text_states norm: {neg_text_states.norm():.4f}")
    print(f"  neg_text_states[0, 0, :5]: {neg_text_states[0, 0, :5].tolist()}")
    print(f"  neg_text_states[0, 1, :5]: {neg_text_states[0, 1, :5].tolist()}")

    # Move transformer to GPU
    print("Moving transformer to GPU...")
    bundle.transformer = bundle.transformer.to(device, torch.bfloat16)
    torch.cuda.empty_cache()

    vae_temporal = bundle.vae.config.scale_factor_temporal
    num_joints = 23
    num_frames = 129

    # Test 1: CFG=5.0, bf16 (same as inference script)
    latents_cfg5 = run_denoising(
        bundle.transformer, bundle.scheduler, text_states, neg_text_states,
        num_frames, num_joints, vae_temporal,
        num_steps=50, guidance_scale=5.0, device=device, dtype=torch.bfloat16,
        seed=42, label="CFG=5.0, bf16 (SAME AS INFERENCE)",
    )

    # Test 2: CFG=1.0 (NO guidance), bf16
    latents_cfg1 = run_denoising(
        bundle.transformer, bundle.scheduler, text_states, neg_text_states,
        num_frames, num_joints, vae_temporal,
        num_steps=50, guidance_scale=1.0, device=device, dtype=torch.bfloat16,
        seed=42, label="CFG=1.0 (NO GUIDANCE), bf16",
    )

    # Test 3: CFG=5.0, float32
    bundle.transformer = bundle.transformer.float()
    latents_cfg5_fp32 = run_denoising(
        bundle.transformer, bundle.scheduler, text_states.float(), neg_text_states.float(),
        num_frames, num_joints, vae_temporal,
        num_steps=50, guidance_scale=5.0, device=device, dtype=torch.float32,
        seed=42, label="CFG=5.0, FLOAT32",
    )

    # Decode all and check body heights
    print("\n" + "=" * 60)
    print("DECODING AND CHECKING BODY HEIGHTS")
    print("=" * 60)

    bundle.transformer = bundle.transformer.cpu()
    del bundle.transformer
    gc.collect()
    torch.cuda.empty_cache()

    for name, lat in [
        ("CFG=5.0 bf16", latents_cfg5),
        ("CFG=1.0 bf16", latents_cfg1),
        ("CFG=5.0 fp32", latents_cfg5_fp32),
    ]:
        lat_dev = lat.to(device).float()
        height = decode_latents(bundle, lat_dev, device)
        print(f"  {name}: body_height = {height:.4f} m "
              f"(latent std={lat.float().std():.4f})")

    print(f"\n  EXPECTED: ~1.5-1.7m for a standing/walking human")
    print(f"  If all heights are bad: model/training issue")
    print(f"  If CFG=1.0 is good but CFG=5.0 is bad: CFG amplifying errors")
    print(f"  If fp32 is good but bf16 is bad: precision issue")

    # Also decode with zero latents as sanity check
    zero_lat = torch.zeros(1, 16, 33, 23, device=device)
    zero_height = decode_latents(bundle, zero_lat, device)
    print(f"\n  Zero-latent decode height: {zero_height:.4f} m (baseline)")


if __name__ == "__main__":
    main()
