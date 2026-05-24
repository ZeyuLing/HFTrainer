"""Check what latent statistics the VAE produces for REAL training data.

Critical test:
- If training latents have std≈1.0, then denoised latents with std≈0.5 means model under-denoises
- If training latents have std≈0.5, then denoising is correct but decode pipeline has a bug

Also checks:
- Zero-latent decode FK height (should give reasonable body if decode pipeline is correct)
- Real-data round-trip FK height (encode → decode → FK)

Usage:
    cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
    python3 scripts/debug/check_training_latent_stats.py
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


def main():
    device = torch.device('cpu')  # Run on CPU to avoid OOM
    config_path = "configs/prism/prism_1b_tp2m_multiframe.py"
    ckpt_path = "work_dirs/prism_1b_tp2m_multiframe/checkpoint-iter_15000"

    # ========== Load Bundle ==========
    print("=" * 70)
    print("LOADING BUNDLE")
    print("=" * 70)
    cfg = Config.fromfile(config_path)
    model_cfg = cfg.model.to_dict() if hasattr(cfg.model, 'to_dict') else cfg.model
    bundle_cls = MODEL_BUNDLES.get(model_cfg['type'])
    bundle = bundle_cls.from_config(model_cfg)
    bundle.eval()

    state_dict = load_checkpoint(ckpt_path, map_location='cpu')
    bundle.load_state_dict_selective(state_dict)
    del state_dict
    gc.collect()

    latents_mean = bundle.latents_mean
    latents_std = bundle.latents_std
    print(f"  latents_mean: {latents_mean.flatten().tolist()}")
    print(f"  latents_std: {latents_std.flatten().tolist()}")

    smpl_proc = bundle.smpl_pose_processor
    vae = bundle.vae
    vae_temporal = vae.config.scale_factor_temporal
    print(f"  VAE z_dim={vae.config.z_dim}, scale_factor_temporal={vae_temporal}")
    print(f"  smpl_proc mean shape: {smpl_proc.mean.shape}")
    print(f"  smpl_proc std shape: {smpl_proc.std.shape}")

    # ========== Load Real Training Data ==========
    print("\n" + "=" * 70)
    print("LOADING REAL TRAINING DATA")
    print("=" * 70)

    # Use multiple motion files for better statistics
    motion_files = []
    base_dir = "data/motionhub/amass_sup/smplx_55"
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if f.endswith('.npz'):
                motion_files.append(os.path.join(root, f))
            if len(motion_files) >= 10:
                break
        if len(motion_files) >= 10:
            break

    print(f"  Found {len(motion_files)} motion files")

    # Process motions through the same pipeline as training
    from hftrainer.datasets.motion.motionhub.transforms.load_smplx import process_smplx_pose, process_transl

    J_template = load_j_template()
    all_latent_stats = []

    for i, mpath in enumerate(motion_files[:5]):
        print(f"\n  [{i+1}] Processing: {os.path.basename(mpath)}")
        data = np.load(mpath, allow_pickle=True)
        trans = np.asarray(data["trans"], dtype=np.float32)
        poses = np.asarray(data["poses"], dtype=np.float32)
        T = trans.shape[0]

        if T < 33:  # Need at least enough frames for VAE
            print(f"    Skipping (T={T} too short)")
            continue

        # Trim to fixed length for fair comparison (129 frames = same as inference)
        T_use = min(T, 129)
        trans_use = trans[:T_use]
        poses_use = poses[:T_use]

        # Original FK height
        poses_66 = poses_use[:, :66]
        orig_joints = numpy_fk_22(trans_use, poses_66, J_template)
        orig_height = compute_body_height(orig_joints)
        print(f"    Original FK height: {orig_height:.4f} m")

        # Process through training pipeline
        transl_processed = process_transl(trans_use, "abs_rel")  # [T, 6]
        pose_processed = process_smplx_pose(poses_use, "rotation_6d", "smpl_22")  # [T, 132]
        motion_vec = np.concatenate([transl_processed, pose_processed], axis=-1)  # [T, 138]

        # Normalize (same as training)
        motion_tensor = torch.from_numpy(motion_vec).float().unsqueeze(0)  # [1, T, 138]
        motion_norm = smpl_proc.normalize(motion_tensor)

        # Reshape for VAE: [1, T, 138] -> [1, T, 23, 6]
        motion_for_vae = rearrange(motion_norm, "b t (j d) -> b t j d", d=6)

        # VAE encode
        with torch.no_grad():
            latents_enc = vae.encode(motion_for_vae.float())  # returns [B, 2*z_dim, T_lat, J]

        # Get mode (first half of channels)
        from hftrainer.models.motion.prism.gaussian_distribution import DiagonalGaussianDistributionNd
        latents_mode = DiagonalGaussianDistributionNd(latents_enc).mode()

        # Apply latent normalization (same as bundle.encode_motion)
        latents_normalized = (latents_mode - latents_mean) / latents_std

        lat_mean = latents_normalized.mean().item()
        lat_std = latents_normalized.std().item()
        lat_min = latents_normalized.min().item()
        lat_max = latents_normalized.max().item()
        print(f"    Normalized latent: mean={lat_mean:.4f}, std={lat_std:.4f}, range=[{lat_min:.3f}, {lat_max:.3f}]")
        all_latent_stats.append((lat_mean, lat_std, lat_min, lat_max))

        # Also decode back and check FK height (round-trip test)
        latents_denorm = latents_normalized * latents_std + latents_mean
        with torch.no_grad():
            motion_recon = vae.decode(latents_denorm.float())

        x_recon = rearrange(motion_recon, "b t j d -> b t (j d)")
        x_recon_denorm = smpl_proc.denormalize(x_recon)

        recon_transl_abs_rel = x_recon_denorm[..., :6]
        recon_transl = smpl_proc.inv_convert_transl(recon_transl_abs_rel)
        recon_poses = x_recon_denorm[..., 6:]
        recon_poses = rearrange(recon_poses, "b t (j d) -> (b t) j d", d=6)
        recon_poses = recon_poses[..., [0, 2, 4, 1, 3, 5]]
        recon_poses_aa = rotation_6d_to_axis_angle(recon_poses)
        recon_poses_aa = rearrange(recon_poses_aa, "(b t) j d -> b t (j d)", b=1)

        recon_transl_np = recon_transl.squeeze(0).numpy()
        recon_poses_np = recon_poses_aa.squeeze(0).numpy()
        T_recon = min(recon_transl_np.shape[0], T_use)
        recon_joints = numpy_fk_22(recon_transl_np[:T_recon], recon_poses_np[:T_recon], J_template)
        recon_height = compute_body_height(recon_joints)
        print(f"    Round-trip FK height: {recon_height:.4f} m (original: {orig_height:.4f})")

    # ========== Summary ==========
    print("\n" + "=" * 70)
    print("TRAINING LATENT STATISTICS SUMMARY")
    print("=" * 70)
    if all_latent_stats:
        means = [s[0] for s in all_latent_stats]
        stds = [s[1] for s in all_latent_stats]
        print(f"  Across {len(all_latent_stats)} samples:")
        print(f"    Mean of means: {np.mean(means):.4f}")
        print(f"    Mean of stds:  {np.mean(stds):.4f}")
        print(f"    Std of stds:   {np.std(stds):.4f}")
        print()
        print(f"  DENOISED latent std was ≈0.5")
        print(f"  If training latent std ≈ 1.0 → model under-denoises (training bug)")
        print(f"  If training latent std ≈ 0.5 → denoising correct, bug is in decode")
    else:
        print("  NO VALID SAMPLES PROCESSED!")

    # ========== Zero-latent decode test ==========
    print("\n" + "=" * 70)
    print("ZERO-LATENT DECODE TEST")
    print("=" * 70)
    T_latent = (129 - 1) // vae_temporal + 1
    num_joints = 23
    z_dim = vae.config.z_dim

    zero_latents = torch.zeros(1, z_dim, T_latent, num_joints)
    # Denormalize: z = z_norm * std + mean  (zero → just mean)
    zero_denorm = zero_latents * latents_std + latents_mean
    print(f"  Zero-latent after denorm: mean={zero_denorm.mean():.6f}, std={zero_denorm.std():.6f}")
    print(f"  (This is just latents_mean broadcast: {latents_mean.flatten()[:4].tolist()}...)")

    with torch.no_grad():
        zero_motion = vae.decode(zero_denorm.float())
    print(f"  VAE decode output: shape={zero_motion.shape}, range=[{zero_motion.min():.4f}, {zero_motion.max():.4f}]")

    x_zero = rearrange(zero_motion, "b t j d -> b t (j d)")
    x_zero_denorm = smpl_proc.denormalize(x_zero)
    print(f"  After motion denorm: range=[{x_zero_denorm.min():.4f}, {x_zero_denorm.max():.4f}]")

    # Detailed breakdown of zero-latent decoded motion
    print(f"\n  Motion breakdown (frame 0):")
    frame0 = x_zero_denorm[0, 0, :].numpy()
    print(f"    transl_abs_rel[:6]: {frame0[:6]}")
    print(f"    global_orient (rot6d) [:6]: {frame0[6:12]}")
    print(f"    first body joint (rot6d) [:6]: {frame0[12:18]}")

    zero_transl_abs_rel = x_zero_denorm[..., :6]
    zero_transl = smpl_proc.inv_convert_transl(zero_transl_abs_rel)
    print(f"  Translation: shape={zero_transl.shape}, Y_mean={zero_transl[0, :, 1].mean():.4f}")
    print(f"  Translation frame 0: {zero_transl[0, 0, :].tolist()}")

    zero_poses = x_zero_denorm[..., 6:]
    zero_poses_r = rearrange(zero_poses, "b t (j d) -> (b t) j d", d=6)
    zero_poses_col = zero_poses_r[..., [0, 2, 4, 1, 3, 5]]
    zero_poses_aa = rotation_6d_to_axis_angle(zero_poses_col)
    zero_poses_aa = rearrange(zero_poses_aa, "(b t) j d -> b t (j d)", b=1)
    print(f"  Axis-angle: range=[{zero_poses_aa.min():.4f}, {zero_poses_aa.max():.4f}]")

    zero_transl_np = zero_transl.squeeze(0).numpy()
    zero_poses_np = zero_poses_aa.squeeze(0).numpy()
    zero_joints = numpy_fk_22(zero_transl_np, zero_poses_np, J_template)
    zero_height = compute_body_height(zero_joints)
    print(f"\n  ** Zero-latent body height: {zero_height:.4f} m **")
    print(f"     Expected: ~1.5-1.7m")

    # Also test with a latent that has std=1 (random normal)
    print("\n  Testing with random N(0,1) latents (simulating perfect denoising result):")
    torch.manual_seed(42)
    rand_latents = torch.randn(1, z_dim, T_latent, num_joints)
    rand_denorm = rand_latents * latents_std + latents_mean
    with torch.no_grad():
        rand_motion = vae.decode(rand_denorm.float())
    x_rand = rearrange(rand_motion, "b t j d -> b t (j d)")
    x_rand_denorm = smpl_proc.denormalize(x_rand)
    rand_transl_abs_rel = x_rand_denorm[..., :6]
    rand_transl = smpl_proc.inv_convert_transl(rand_transl_abs_rel)
    rand_poses = x_rand_denorm[..., 6:]
    rand_poses_r = rearrange(rand_poses, "b t (j d) -> (b t) j d", d=6)
    rand_poses_col = rand_poses_r[..., [0, 2, 4, 1, 3, 5]]
    rand_poses_aa = rotation_6d_to_axis_angle(rand_poses_col)
    rand_poses_aa = rearrange(rand_poses_aa, "(b t) j d -> b t (j d)", b=1)
    rand_transl_np = rand_transl.squeeze(0).numpy()
    rand_poses_np = rand_poses_aa.squeeze(0).numpy()
    rand_joints = numpy_fk_22(rand_transl_np, rand_poses_np, J_template)
    rand_height = compute_body_height(rand_joints)
    print(f"  ** Random N(0,1) latent body height: {rand_height:.4f} m **")
    print(f"     (random garbage, but should still have ~human-scale if decode is correct)")

    # ========== CRITICAL: Test bundle.encode_motion directly ==========
    print("\n" + "=" * 70)
    print("BUNDLE.encode_motion DIRECT TEST")
    print("=" * 70)
    # Load first valid motion
    if motion_files:
        mpath = motion_files[0]
        data = np.load(mpath, allow_pickle=True)
        trans = np.asarray(data["trans"], dtype=np.float32)[:129]
        poses = np.asarray(data["poses"], dtype=np.float32)[:129]
        T = trans.shape[0]

        transl_processed = process_transl(trans, "abs_rel")
        pose_processed = process_smplx_pose(poses, "rotation_6d", "smpl_22")
        motion_vec = np.concatenate([transl_processed, pose_processed], axis=-1)
        motion_tensor = torch.from_numpy(motion_vec).float().unsqueeze(0)

        with torch.no_grad():
            bundle_latents = bundle.encode_motion(motion_tensor)
        print(f"  bundle.encode_motion output:")
        print(f"    shape: {bundle_latents.shape}")
        print(f"    mean: {bundle_latents.mean():.4f}")
        print(f"    std: {bundle_latents.std():.4f}")
        print(f"    range: [{bundle_latents.min():.3f}, {bundle_latents.max():.3f}]")


if __name__ == "__main__":
    main()
