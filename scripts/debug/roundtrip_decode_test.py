"""Definitive round-trip test: real motion → encode → x_0 → decode_motion → post_process_motion.

This bypasses the denoising loop entirely. If the output is deformed, the bug is in
the decode path. If the output is correct, the bug is in the denoising loop.

Additionally tests with latent_std=0.5 (simulating the collapsed inference output)
to check if the deformation is simply caused by out-of-distribution latents.

Usage:
    cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
    python3 scripts/debug/roundtrip_decode_test.py
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
from hftrainer.models.motion.prism.gaussian_distribution import DiagonalGaussianDistributionNd
from hftrainer.models.motion.components.utils.geometry.rotation_convert import (
    rotation_6d_to_axis_angle,
)


SMPL_22_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]


def load_j_template():
    template_path = "/apdcephfs_cq11/share_1467498/home/zeyuling/motion_vis_web/static/assets/j_template_22.npy"
    return np.load(template_path).astype(np.float32)


def numpy_fk_22(transl, poses_aa, J_template):
    """Forward kinematics for SMPL 22-joint model."""
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
    """Compute body height from joint positions."""
    head = positions[:, 15, :]
    feet = np.minimum(
        np.minimum(positions[:, 7, :], positions[:, 8, :]),
        np.minimum(positions[:, 10, :], positions[:, 11, :])
    )
    height = head[:, 1] - feet[:, 1]
    return float(np.mean(height))


def compute_limb_lengths(positions):
    """Compute average limb lengths to check body proportions."""
    # Left arm: shoulder(13) -> elbow(16) -> wrist(18)
    # Right arm: shoulder(14) -> elbow(17) -> wrist(19)
    # Left leg: hip(1) -> knee(4) -> ankle(7)
    # Right leg: hip(2) -> knee(5) -> ankle(8)
    limbs = {
        "L_upper_arm": (13, 16),
        "L_lower_arm": (16, 18),
        "R_upper_arm": (14, 17),
        "R_lower_arm": (17, 19),
        "L_upper_leg": (1, 4),
        "L_lower_leg": (4, 7),
        "R_upper_leg": (2, 5),
        "R_lower_leg": (5, 8),
        "spine": (0, 9),
    }
    result = {}
    for name, (j1, j2) in limbs.items():
        lengths = np.linalg.norm(positions[:, j2] - positions[:, j1], axis=-1)
        result[name] = float(np.mean(lengths))
    return result


def main():
    device = torch.device('cpu')  # CPU is fine for this test
    config_path = "configs/prism/prism_1b_tp2m_multiframe.py"
    ckpt_path = "work_dirs/prism_1b_tp2m_multiframe/checkpoint-iter_15000"

    print("=" * 80)
    print("DEFINITIVE ROUND-TRIP TEST: encode → x_0 → decode → FK")
    print("Bypasses denoising loop entirely")
    print("=" * 80)

    # ========== Load Bundle ==========
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

    J_template = load_j_template()

    # ========== Load real motion ==========
    print("\n[2] Loading real motion data...")
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

    if motion_file is None:
        print("ERROR: No motion file found!")
        return

    print(f"  File: {os.path.basename(motion_file)}")
    data = np.load(motion_file, allow_pickle=True)
    trans_orig = np.asarray(data["trans"], dtype=np.float32)[:129]
    poses_orig = np.asarray(data["poses"], dtype=np.float32)[:129]
    T_orig = trans_orig.shape[0]
    print(f"  Frames: {T_orig}")

    # ========== Original FK (ground truth) ==========
    print("\n[3] Computing ORIGINAL FK (ground truth)...")
    poses_aa_orig = poses_orig[:, :66]  # First 22 joints * 3
    joints_orig = numpy_fk_22(trans_orig, poses_aa_orig, J_template)
    height_orig = compute_body_height(joints_orig)
    limbs_orig = compute_limb_lengths(joints_orig)
    print(f"  Original body height: {height_orig:.4f} m")
    print(f"  Original limb lengths:")
    for name, length in limbs_orig.items():
        print(f"    {name:15s}: {length:.4f} m")

    # ========== Encode to latent (training pipeline) ==========
    print("\n[4] Encoding through training pipeline (process → normalize → VAE encode)...")
    transl_processed = process_transl(trans_orig, "abs_rel")  # [T, 6]
    pose_processed = process_smplx_pose(poses_orig, "rotation_6d", "smpl_22")  # [T, 132]
    motion_vec = np.concatenate([transl_processed, pose_processed], axis=-1)  # [T, 138]
    motion_tensor = torch.from_numpy(motion_vec).float().unsqueeze(0)  # [1, T, 138]

    smpl_proc = bundle.smpl_pose_processor
    motion_norm = smpl_proc.normalize(motion_tensor)
    motion_for_vae = rearrange(motion_norm, "b t (j d) -> b t j d", d=6)

    with torch.no_grad():
        latents_enc = bundle.vae.encode(motion_for_vae.float())
    latents_mode = DiagonalGaussianDistributionNd(latents_enc).mode()
    x_0 = (latents_mode - bundle.latents_mean) / bundle.latents_std
    print(f"  x_0 shape: {x_0.shape}")
    print(f"  x_0 stats: mean={x_0.mean():.4f}, std={x_0.std():.4f}")

    # ========== TEST 1: Perfect x_0 → decode → post_process → FK ==========
    print("\n" + "=" * 80)
    print("TEST 1: PERFECT x_0 → decode_motion → post_process_motion → FK")
    print("=" * 80)

    # Use the exact same decode pipeline as inference
    # We call the same functions directly
    latents_std = bundle.latents_std
    latents_mean = bundle.latents_mean
    vae = bundle.vae

    # Step 1: Denormalize latent → VAE decode (same as decode_motion)
    latents_denorm = x_0 * latents_std + latents_mean
    with torch.no_grad():
        x_dec = vae.decode(latents_denorm.float())
    print(f"  VAE decode output: shape={x_dec.shape}")  # [B, T, J, C]

    # Step 2: Apply post_process_motion logic manually (WITHOUT fix_first_chunk)
    x_flat = rearrange(x_dec, "b t j d -> b t (j d)")
    x_denorm = smpl_proc.denormalize(x_flat)
    print(f"  After denormalize: shape={x_denorm.shape}, range=[{x_denorm.min():.4f}, {x_denorm.max():.4f}]")

    # Translation
    transl_abs_rel = x_denorm[..., :6]
    transl_recon = smpl_proc.inv_convert_transl(transl_abs_rel)
    print(f"  Translation: shape={transl_recon.shape}, Y_mean={transl_recon[0, :, 1].mean():.4f}")

    # Poses
    pred_poses = x_denorm[..., 6:]
    pred_poses = rearrange(pred_poses, "b t (j d) -> (b t) j d", d=6)
    # Row-major → column-major permutation
    pred_poses = pred_poses[..., [0, 2, 4, 1, 3, 5]]
    pred_poses_aa = rotation_6d_to_axis_angle(pred_poses)
    pred_poses_aa = rearrange(pred_poses_aa, "(b t) j d -> b t (j d)", b=1)

    # FK
    transl_np = transl_recon.squeeze(0).numpy()
    poses_np = pred_poses_aa.squeeze(0).numpy()
    T_recon = transl_np.shape[0]
    joints_recon = numpy_fk_22(transl_np[:T_recon], poses_np[:T_recon], J_template)
    height_recon = compute_body_height(joints_recon)
    limbs_recon = compute_limb_lengths(joints_recon)

    print(f"\n  Round-trip body height: {height_recon:.4f} m (original: {height_orig:.4f} m)")
    print(f"  Height error: {abs(height_recon - height_orig):.4f} m ({abs(height_recon - height_orig)/height_orig*100:.1f}%)")
    print(f"\n  Limb length comparison:")
    print(f"    {'Limb':15s} {'Original':>10s} {'Round-trip':>12s} {'Error%':>8s}")
    for name in limbs_orig:
        orig = limbs_orig[name]
        recon = limbs_recon[name]
        err_pct = abs(recon - orig) / (orig + 1e-8) * 100
        print(f"    {name:15s} {orig:10.4f} {recon:12.4f} {err_pct:7.1f}%")

    # ========== TEST 2: Same with fix_first_chunk (as inference actually does) ==========
    print("\n" + "=" * 80)
    print("TEST 2: Same with fix_first_chunk=True (matching inference)")
    print("=" * 80)

    # Repeat decode
    with torch.no_grad():
        x_dec2 = vae.decode(latents_denorm.float())

    x_flat2 = rearrange(x_dec2, "b t j d -> b t (j d)")
    x_denorm2 = smpl_proc.denormalize(x_flat2)
    transl_abs_rel2 = x_denorm2[..., :6]
    transl_recon2 = smpl_proc.inv_convert_transl(transl_abs_rel2)

    pred_poses2 = x_denorm2[..., 6:]
    pred_poses2 = rearrange(pred_poses2, "b t (j d) -> (b t) j d", d=6)
    pred_poses2 = pred_poses2[..., [0, 2, 4, 1, 3, 5]]
    pred_poses_aa2 = rotation_6d_to_axis_angle(pred_poses2)
    pred_poses_aa2 = rearrange(pred_poses_aa2, "(b t) j d -> b t (j d)", b=1)

    # Apply fix_first_chunk logic
    T = pred_poses_aa2.shape[1]
    scale = 4  # vae_scale_factor_temporal
    min_fix = 3 * scale  # 12
    max_fix = min(T // 3, 10 * scale)

    if T >= min_fix + 16:
        diffs = pred_poses_aa2[:, 1:] - pred_poses_aa2[:, :-1]
        vel = diffs.norm(dim=-1).squeeze(0)
        stable_start = max(max_fix + 8, int(T * 0.6))
        stable_vel_median = vel[stable_start:].median()
        spike_threshold = 2.0 * stable_vel_median
        n_fix = min_fix
        for i in range(min_fix, min(max_fix, len(vel))):
            if vel[i] > spike_threshold:
                n_fix = i + 2
        n_fix = min(n_fix, max_fix)
        print(f"  fix_first_chunk: n_fix={n_fix}, stable_vel_median={stable_vel_median:.4f}, threshold={spike_threshold:.4f}")
    else:
        n_fix = min(scale, T // 3)
        print(f"  fix_first_chunk: short motion, n_fix={n_fix}")

    if n_fix > 0 and T > n_fix + 4:
        anchor_idx = n_fix
        n_ref = min(16, T - anchor_idx - 1)
        anchor = pred_poses_aa2[:, anchor_idx]
        ref_vel = (pred_poses_aa2[:, anchor_idx + n_ref] - pred_poses_aa2[:, anchor_idx]) / n_ref
        n_blend = min(8, n_fix // 2)
        n_hard = n_fix - n_blend
        for i in range(n_hard):
            pred_poses_aa2[:, i] = anchor - (n_fix - i) * ref_vel
        if n_blend > 0:
            original_poses = pred_poses_aa2[:, n_hard:n_fix].clone()
            for i in range(n_blend):
                extrap = anchor - (n_blend - i) * ref_vel
                alpha = 0.5 * (1.0 - torch.cos(torch.tensor((i + 1) / (n_blend + 1) * 3.14159265)))
                pred_poses_aa2[:, n_hard + i] = (1 - alpha) * extrap + alpha * original_poses[:, i]

        # Fix translation too
        T_tr = transl_recon2.shape[1]
        if T_tr > n_fix + 4:
            tr_anchor = transl_recon2[:, anchor_idx]
            tr_ref_vel = (transl_recon2[:, anchor_idx + n_ref] - transl_recon2[:, anchor_idx]) / n_ref
            for i in range(n_hard):
                transl_recon2[:, i] = tr_anchor - (n_fix - i) * tr_ref_vel
            if n_blend > 0:
                original_tr = transl_recon2[:, n_hard:n_fix].clone()
                for i in range(n_blend):
                    extrap_tr = tr_anchor - (n_blend - i) * tr_ref_vel
                    alpha = 0.5 * (1.0 - torch.cos(torch.tensor((i + 1) / (n_blend + 1) * 3.14159265)))
                    transl_recon2[:, n_hard + i] = (1 - alpha) * extrap_tr + alpha * original_tr[:, i]

    transl_np2 = transl_recon2.squeeze(0).numpy()
    poses_np2 = pred_poses_aa2.squeeze(0).numpy()
    joints_recon2 = numpy_fk_22(transl_np2, poses_np2, J_template)
    height_recon2 = compute_body_height(joints_recon2)
    limbs_recon2 = compute_limb_lengths(joints_recon2)

    print(f"\n  With fix_first_chunk body height: {height_recon2:.4f} m (original: {height_orig:.4f})")
    print(f"  Height error: {abs(height_recon2 - height_orig):.4f} m ({abs(height_recon2 - height_orig)/height_orig*100:.1f}%)")

    # ========== TEST 3: Simulate collapsed latents (std=0.5) ==========
    print("\n" + "=" * 80)
    print("TEST 3: COLLAPSED LATENTS (simulating inference output with std≈0.5)")
    print("=" * 80)

    # Scale x_0 to have std=0.5 (simulating what inference produces)
    x_0_collapsed = x_0 * (0.5 / x_0.std())
    print(f"  Collapsed x_0: std={x_0_collapsed.std():.4f}")

    latents_collapsed_denorm = x_0_collapsed * latents_std + latents_mean
    with torch.no_grad():
        x_dec_collapsed = vae.decode(latents_collapsed_denorm.float())

    x_flat_c = rearrange(x_dec_collapsed, "b t j d -> b t (j d)")
    x_denorm_c = smpl_proc.denormalize(x_flat_c)
    transl_abs_rel_c = x_denorm_c[..., :6]
    transl_recon_c = smpl_proc.inv_convert_transl(transl_abs_rel_c)

    pred_poses_c = x_denorm_c[..., 6:]
    pred_poses_c = rearrange(pred_poses_c, "b t (j d) -> (b t) j d", d=6)
    pred_poses_c = pred_poses_c[..., [0, 2, 4, 1, 3, 5]]
    pred_poses_aa_c = rotation_6d_to_axis_angle(pred_poses_c)
    pred_poses_aa_c = rearrange(pred_poses_aa_c, "(b t) j d -> b t (j d)", b=1)

    transl_np_c = transl_recon_c.squeeze(0).numpy()
    poses_np_c = pred_poses_aa_c.squeeze(0).numpy()
    joints_collapsed = numpy_fk_22(transl_np_c, poses_np_c, J_template)
    height_collapsed = compute_body_height(joints_collapsed)
    limbs_collapsed = compute_limb_lengths(joints_collapsed)

    print(f"\n  Collapsed-latent body height: {height_collapsed:.4f} m (original: {height_orig:.4f})")
    print(f"  Height error: {abs(height_collapsed - height_orig):.4f} m ({abs(height_collapsed - height_orig)/height_orig*100:.1f}%)")
    print(f"\n  Limb length comparison (collapsed vs original):")
    print(f"    {'Limb':15s} {'Original':>10s} {'Collapsed':>12s} {'Error%':>8s}")
    for name in limbs_orig:
        orig = limbs_orig[name]
        coll = limbs_collapsed[name]
        err_pct = abs(coll - orig) / (orig + 1e-8) * 100
        print(f"    {name:15s} {orig:10.4f} {coll:12.4f} {err_pct:7.1f}%")

    # ========== TEST 4: Random noise latents (std=1.0) - what inference starts with ==========
    print("\n" + "=" * 80)
    print("TEST 4: RANDOM NOISE LATENTS (std=1.0, random garbage)")
    print("=" * 80)

    torch.manual_seed(42)
    x_0_random = torch.randn_like(x_0)
    print(f"  Random x_0: std={x_0_random.std():.4f}")

    latents_random_denorm = x_0_random * latents_std + latents_mean
    with torch.no_grad():
        x_dec_random = vae.decode(latents_random_denorm.float())

    x_flat_r = rearrange(x_dec_random, "b t j d -> b t (j d)")
    x_denorm_r = smpl_proc.denormalize(x_flat_r)
    transl_abs_rel_r = x_denorm_r[..., :6]
    transl_recon_r = smpl_proc.inv_convert_transl(transl_abs_rel_r)

    pred_poses_r = x_denorm_r[..., 6:]
    pred_poses_r = rearrange(pred_poses_r, "b t (j d) -> (b t) j d", d=6)
    pred_poses_r = pred_poses_r[..., [0, 2, 4, 1, 3, 5]]
    pred_poses_aa_r = rotation_6d_to_axis_angle(pred_poses_r)
    pred_poses_aa_r = rearrange(pred_poses_aa_r, "(b t) j d -> b t (j d)", b=1)

    transl_np_r = transl_recon_r.squeeze(0).numpy()
    poses_np_r = pred_poses_aa_r.squeeze(0).numpy()
    joints_random = numpy_fk_22(transl_np_r, poses_np_r, J_template)
    height_random = compute_body_height(joints_random)
    print(f"  Random-latent body height: {height_random:.4f} m (should be garbage)")

    # ========== SUMMARY ==========
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"""
  Original (ground truth):      height = {height_orig:.4f} m
  Round-trip (perfect x_0):     height = {height_recon:.4f} m  (VAE reconstruction loss)
  Round-trip (w/ fix_first):    height = {height_recon2:.4f} m
  Collapsed latents (std=0.5):  height = {height_collapsed:.4f} m  (inference-like)
  Random noise (std=1.0):       height = {height_random:.4f} m  (garbage)
    """)

    if abs(height_recon - height_orig) / height_orig < 0.15:
        print("  ✓ DECODE PATH IS CORRECT (round-trip produces valid body)")
        print("  → Bug is in denoising loop / model quality at high sigma")
        if abs(height_collapsed - height_orig) / height_orig > 0.3:
            print("  → AND collapsed latents (std=0.5) produce deformed output")
            print("  → The deformation is caused by latent collapse during inference")
            print("  → Root cause: model fails at high sigma → ODE trajectory divergence → std shrinkage")
        else:
            print("  → BUT collapsed latents still produce reasonable output")
            print("  → Deformation may have other causes (fix_first_chunk corruption?)")
    else:
        print("  ❌ DECODE PATH HAS A BUG!")
        print("  → Even perfect x_0 latent produces deformed output")
        print("  → Need to investigate decode pipeline (rot6d, normalization, VAE)")

    # ========== TEST 5: Direct comparison of axis-angle values ==========
    print("\n" + "=" * 80)
    print("TEST 5: AXIS-ANGLE VALUE COMPARISON (first frame)")
    print("=" * 80)

    print(f"  Original axis-angle (frame 0, global orient): {poses_aa_orig[0, :3]}")
    print(f"  Round-trip axis-angle (frame 0, global orient): {poses_np[0, :3]}")
    print(f"  Difference: {np.abs(poses_aa_orig[0, :3] - poses_np[0, :3])}")
    print()
    print(f"  Original axis-angle (frame 0, joint 1): {poses_aa_orig[0, 3:6]}")
    print(f"  Round-trip axis-angle (frame 0, joint 1): {poses_np[0, 3:6]}")
    print(f"  Difference: {np.abs(poses_aa_orig[0, 3:6] - poses_np[0, 3:6])}")
    print()
    # Check all joints MSE
    T_cmp = min(T_orig, T_recon)
    mse_poses = np.mean((poses_aa_orig[:T_cmp, :66] - poses_np[:T_cmp, :66]) ** 2)
    mse_transl = np.mean((trans_orig[:T_cmp] - transl_np[:T_cmp]) ** 2)
    print(f"  MSE poses (all frames, 22 joints): {mse_poses:.6f}")
    print(f"  MSE translation (all frames): {mse_transl:.6f}")

    # ========== TEST 6: Check if smpl_proc.denormalize introduces scaling ==========
    print("\n" + "=" * 80)
    print("TEST 6: NORMALIZATION ROUND-TRIP CHECK")
    print("=" * 80)

    # motion_tensor is [1, T, 138] (raw, before normalize)
    # motion_norm is [1, T, 138] (after normalize)
    motion_roundtrip = smpl_proc.denormalize(motion_norm)
    norm_err = (motion_tensor - motion_roundtrip).abs().max().item()
    print(f"  normalize → denormalize max error: {norm_err:.8f}")
    if norm_err < 1e-5:
        print("  ✓ Normalization round-trip is lossless")
    else:
        print("  ❌ Normalization round-trip has errors!")

    # Check specific values
    print(f"\n  Original motion_vec[0, :6] (transl_abs_rel): {motion_tensor[0, 0, :6].numpy()}")
    print(f"  After norm→denorm[0, :6]: {motion_roundtrip[0, 0, :6].numpy()}")
    print(f"  Norm mean[:6]: {smpl_proc.mean[0, 0, :6].numpy()}")
    print(f"  Norm std[:6]: {smpl_proc.std[0, 0, :6].numpy()}")


if __name__ == "__main__":
    main()
