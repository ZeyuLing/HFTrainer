"""Diagnose PRISM inference deformation bug.

Tests multiple hypotheses:
1. Stats mismatch (normalization mean/std dimensions)
2. VAE round-trip on real data (encode → decode → FK check)
3. Zero-latent decode (average motion should look normal)
4. Latent statistics check (are latents_mean/std reasonable?)
5. Post-processing correctness (axis-angle conversion + transl recovery)

Usage:
    cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
    python3 scripts/debug/diagnose_prism_decode.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

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
    axis_angle_to_rotation_6d,
)


# SMPL-22 parent chain for FK
SMPL_22_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]


def load_j_template():
    """Load the 22-joint rest pose template."""
    template_path = "/apdcephfs_cq11/share_1467498/home/zeyuling/motion_vis_web/static/assets/j_template_22.npy"
    if os.path.isfile(template_path):
        return np.load(template_path).astype(np.float32)
    # Fallback
    smplx_path = "/apdcephfs_cq11/share_1467498/home/zeyuling/versatilemotion/checkpoints/smpl_models/smplx/SMPLX_NEUTRAL.npz"
    data = np.load(smplx_path, allow_pickle=True)
    J_reg = np.asarray(data["J_regressor"], dtype=np.float64)
    v_template = np.asarray(data["v_template"], dtype=np.float64)
    return (J_reg @ v_template)[:22].astype(np.float32)


def numpy_fk_22(transl, poses_aa, J_template):
    """Pure numpy FK to get 22-joint positions. poses_aa: [T, 66] axis-angle."""
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
    """Compute body height from joint positions [T, 22, 3]. Returns avg height."""
    # Head (joint 15) to lowest foot (joints 7=left_ankle, 8=right_ankle, 10=left_foot, 11=right_foot)
    head = positions[:, 15, :]
    feet = np.minimum(
        np.minimum(positions[:, 7, :], positions[:, 8, :]),
        np.minimum(positions[:, 10, :], positions[:, 11, :])
    )
    height = head[:, 1] - feet[:, 1]  # Y-up
    return float(np.mean(height))


def main():
    print("=" * 70)
    print("PRISM INFERENCE DEFORMATION DIAGNOSIS")
    print("=" * 70)

    config_path = "configs/prism/prism_1b_tp2m_multiframe.py"
    ckpt_path = "work_dirs/prism_1b_tp2m_multiframe/checkpoint-iter_15000"

    # =================== 1. Load bundle ===================
    print("\n[1] Loading bundle and checkpoint...")
    cfg = Config.fromfile(config_path)
    model_cfg = cfg.model.to_dict() if hasattr(cfg.model, 'to_dict') else cfg.model
    bundle_cls = MODEL_BUNDLES.get(model_cfg['type'])
    bundle = bundle_cls.from_config(model_cfg)
    bundle.eval()

    state_dict = load_checkpoint(ckpt_path, map_location='cpu')
    bundle.load_state_dict_selective(state_dict)
    del state_dict
    print("  Bundle loaded successfully.")

    # =================== 2. Check stats dimensions ===================
    print("\n[2] Checking normalization stats...")
    smpl_proc = bundle.smpl_pose_processor
    mean = smpl_proc.mean
    std = smpl_proc.std
    print(f"  mean shape: {mean.shape}, std shape: {std.shape}")
    print(f"  Expected: 138 dims (6 transl_abs_rel + 6 global_orient + 126 body_pose)")
    assert mean.shape[0] == 138, f"STATS MISMATCH! Got {mean.shape[0]} dims, expected 138"
    print(f"  ✓ Stats are 138-dim (correct for smpl_22 + rot6d + abs_rel)")

    # Print stats ranges
    print(f"\n  Stats breakdown:")
    print(f"    transl_abs mean: {mean[:3].tolist()}")
    print(f"    transl_abs std:  {std[:3].tolist()}")
    print(f"    transl_vel mean: {mean[3:6].tolist()}")
    print(f"    transl_vel std:  {std[3:6].tolist()}")
    print(f"    global_orient mean: {mean[6:12].tolist()}")
    print(f"    global_orient std:  {std[6:12].tolist()}")
    print(f"    body_pose mean range: [{mean[12:].min():.4f}, {mean[12:].max():.4f}]")
    print(f"    body_pose std range:  [{std[12:].min():.4f}, {std[12:].max():.4f}]")

    # =================== 3. Check latent normalization ===================
    print("\n[3] Checking VAE latent normalization...")
    latents_mean = bundle.latents_mean
    latents_std = bundle.latents_std
    print(f"  latents_mean shape: {latents_mean.shape}")
    print(f"  latents_std shape:  {latents_std.shape}")
    print(f"  latents_mean range: [{latents_mean.min():.4f}, {latents_mean.max():.4f}]")
    print(f"  latents_std range:  [{latents_std.min():.4f}, {latents_std.max():.4f}]")
    print(f"  VAE z_dim: {bundle.vae.config.z_dim}")
    print(f"  VAE scale_factor_temporal: {bundle.vae.config.scale_factor_temporal}")

    # =================== 4. Test: Decode zero latents (average motion) ===================
    print("\n[4] Decoding ZERO latents (should give 'average motion')...")
    device = torch.device('cpu')

    # Zero latents → denormalize → VAE decode → post-process
    T_frames = 129
    vae_temporal = bundle.vae.config.scale_factor_temporal
    T_latent = (T_frames - 1) // vae_temporal + 1
    num_joints = 23
    z_dim = bundle.vae.config.z_dim

    zero_latents = torch.zeros(1, z_dim, T_latent, num_joints)
    print(f"  Latent shape: {zero_latents.shape}")

    # Denormalize latents
    denorm_latents = zero_latents * latents_std + latents_mean
    print(f"  After latent denorm: range [{denorm_latents.min():.4f}, {denorm_latents.max():.4f}]")

    # VAE decode
    with torch.no_grad(), torch.autocast('cpu', enabled=False):
        motion_decoded = bundle.vae.decode(denorm_latents.float())
    print(f"  VAE output shape: {motion_decoded.shape}")  # should be [1, T, 23, 6]
    print(f"  VAE output range: [{motion_decoded.min():.4f}, {motion_decoded.max():.4f}]")

    # Post-process
    x_dec = rearrange(motion_decoded, "b t j d -> b t (j d)")
    print(f"  Flattened shape: {x_dec.shape}")  # [1, T, 138]

    # Denormalize motion
    x_denorm = smpl_proc.denormalize(x_dec)
    print(f"  After motion denorm: range [{x_denorm.min():.4f}, {x_denorm.max():.4f}]")

    # Extract translation
    transl_abs_rel = x_denorm[..., :6]
    transl = smpl_proc.inv_convert_transl(transl_abs_rel)
    print(f"  Translation shape: {transl.shape}")
    print(f"  Translation range: [{transl.min():.4f}, {transl.max():.4f}]")
    print(f"  Translation Y (height above ground): mean={transl[0, :, 1].mean():.4f}")

    # Extract poses and convert
    pred_poses = x_denorm[..., 6:]  # [1, T, 132]
    pred_poses_reshaped = rearrange(pred_poses, "b t (j d) -> (b t) j d", d=6)
    # Row-major → column-major permutation
    pred_poses_col = pred_poses_reshaped[..., [0, 2, 4, 1, 3, 5]]
    pred_poses_aa = rotation_6d_to_axis_angle(pred_poses_col)  # [B*T, 22, 3]
    pred_poses_aa = rearrange(pred_poses_aa, "(b t) j d -> b t (j d)", b=1)
    print(f"  Axis-angle poses shape: {pred_poses_aa.shape}")  # [1, T, 66]
    print(f"  Axis-angle range: [{pred_poses_aa.min():.4f}, {pred_poses_aa.max():.4f}]")

    # FK to check body height
    J_template = load_j_template()
    transl_np = transl.squeeze(0).numpy()
    poses_np = pred_poses_aa.squeeze(0).numpy()
    joints = numpy_fk_22(transl_np, poses_np, J_template)
    body_height = compute_body_height(joints)
    print(f"\n  ** Body height from zero-latent decode: {body_height:.4f} m **")
    print(f"     (Expected: ~1.5-1.7m for normal standing human)")
    if body_height < 1.0 or body_height > 2.5:
        print(f"  ⚠️  ABNORMAL body height! Bug is likely in decode/post-processing pipeline.")
    else:
        print(f"  ✓ Body height is reasonable. Bug is likely in denoising (model output).")

    # =================== 5. Load real training data and check round-trip ===================
    print("\n[5] Testing VAE round-trip on REAL training data...")

    # Find a real training sample
    import json
    anno_file = "data/annotation/train_hq_motionhub_hymotion.json"
    with open(anno_file, 'r') as f:
        anno = json.load(f)
    # Pick first sample
    sample = anno[0]
    motion_path = os.path.join("data/motionhub", sample.get("smplx", sample.get("motion_path", "")))
    print(f"  Loading sample: {motion_path}")

    if not os.path.isfile(motion_path):
        print(f"  ⚠️  File not found, trying alternate paths...")
        # Try common patterns
        for k in ["smplx", "motion_path", "npz_path"]:
            if k in sample:
                alt = os.path.join("data/motionhub", sample[k])
                if os.path.isfile(alt):
                    motion_path = alt
                    break
        if not os.path.isfile(motion_path):
            print(f"  Cannot find training sample. Skipping round-trip test.")
            print(f"  Sample keys: {list(sample.keys())}")
            print(f"  Sample: {sample}")
            return

    # Load raw data
    data = np.load(motion_path, allow_pickle=True)
    abs_trans = np.asarray(data["trans"], dtype=np.float32)
    poses_raw = np.asarray(data["poses"], dtype=np.float32)
    T = abs_trans.shape[0]
    print(f"  Raw data: T={T}, trans shape={abs_trans.shape}, poses shape={poses_raw.shape}")

    # Compute original FK height for reference
    poses_66 = poses_raw[:, :66] if poses_raw.shape[1] >= 66 else np.pad(poses_raw, ((0,0),(0, 66-poses_raw.shape[1])))
    orig_joints = numpy_fk_22(abs_trans, poses_66, J_template)
    orig_height = compute_body_height(orig_joints)
    print(f"  Original body height (FK): {orig_height:.4f} m")

    # Process through training pipeline (same as LoadSmplx55)
    from hftrainer.datasets.motion.motionhub.transforms.load_smplx import process_smplx_pose, process_transl

    transl_processed = process_transl(abs_trans, "abs_rel")  # [T, 6]
    pose_processed = process_smplx_pose(poses_raw, "rotation_6d", "smpl_22")  # [T, 132]
    motion_vec = np.concatenate([transl_processed, pose_processed], axis=-1)  # [T, 138]
    print(f"  Processed motion_vec shape: {motion_vec.shape}")
    print(f"  motion_vec range: [{motion_vec.min():.4f}, {motion_vec.max():.4f}]")

    # Normalize
    motion_tensor = torch.from_numpy(motion_vec).float().unsqueeze(0)  # [1, T, 138]
    motion_norm = smpl_proc.normalize(motion_tensor)
    print(f"  After normalize: range [{motion_norm.min():.4f}, {motion_norm.max():.4f}]")

    # Reshape for VAE: [1, T, 138] -> [1, T, 23, 6]
    motion_for_vae = rearrange(motion_norm, "b t (j d) -> b t j d", d=6)
    print(f"  VAE input shape: {motion_for_vae.shape}")

    # VAE encode
    with torch.no_grad(), torch.autocast('cpu', enabled=False):
        latents_enc = bundle.vae.encode(motion_for_vae.float())
    print(f"  Encoded latents shape: {latents_enc.shape}")

    # Apply latent normalization (same as bundle.encode_motion)
    from hftrainer.models.motion.prism.gaussian_distribution import DiagonalGaussianDistributionNd
    latents_mode = DiagonalGaussianDistributionNd(latents_enc).mode()
    latents_normalized = (latents_mode - latents_mean) / latents_std
    print(f"  Normalized latents range: [{latents_normalized.min():.4f}, {latents_normalized.max():.4f}]")

    # Now decode (same as inference)
    latents_denorm = latents_normalized * latents_std + latents_mean
    with torch.no_grad(), torch.autocast('cpu', enabled=False):
        motion_reconstructed = bundle.vae.decode(latents_denorm.float())
    print(f"  Reconstructed motion shape: {motion_reconstructed.shape}")

    # Post-process (same as inference script)
    x_recon = rearrange(motion_reconstructed, "b t j d -> b t (j d)")
    x_recon_denorm = smpl_proc.denormalize(x_recon)

    recon_transl_abs_rel = x_recon_denorm[..., :6]
    recon_transl = smpl_proc.inv_convert_transl(recon_transl_abs_rel)
    recon_poses = x_recon_denorm[..., 6:]  # [1, T, 132]
    recon_poses_reshaped = rearrange(recon_poses, "b t (j d) -> (b t) j d", d=6)
    recon_poses_col = recon_poses_reshaped[..., [0, 2, 4, 1, 3, 5]]
    recon_poses_aa = rotation_6d_to_axis_angle(recon_poses_col)
    recon_poses_aa = rearrange(recon_poses_aa, "(b t) j d -> b t (j d)", b=1)

    recon_transl_np = recon_transl.squeeze(0).numpy()
    recon_poses_np = recon_poses_aa.squeeze(0).numpy()

    # Trim to match original T (VAE may pad)
    T_recon = recon_transl_np.shape[0]
    T_orig = abs_trans.shape[0]
    T_use = min(T_recon, T_orig)

    recon_joints = numpy_fk_22(recon_transl_np[:T_use], recon_poses_np[:T_use], J_template)
    recon_height = compute_body_height(recon_joints)
    print(f"\n  ** Round-trip body height: {recon_height:.4f} m (original: {orig_height:.4f} m) **")
    if abs(recon_height - orig_height) < 0.3:
        print(f"  ✓ Round-trip preserves body proportions → Bug is in DENOISING (model output)")
        print(f"    The DiT model is generating bad latents. Check:")
        print(f"    - Checkpoint loading (are weights correct?)")
        print(f"    - Scheduler mismatch (shift, timesteps)")
        print(f"    - CFG scale too high?")
    else:
        print(f"  ⚠️ Round-trip BREAKS body proportions → Bug is in DECODE/POST-PROCESSING")
        print(f"    Check: VAE weights, normalization stats, or rotation conversion")

    # =================== 6. Check actual generated output ===================
    print("\n[6] Checking previously generated output files...")
    output_dir = "work_dirs/prism_1b_tp2m_multiframe/eval_fix256"
    if os.path.isdir(output_dir):
        npz_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.npz')])
        for fname in npz_files[:2]:
            fpath = os.path.join(output_dir, fname)
            gen_data = np.load(fpath, allow_pickle=True)
            gen_transl = np.asarray(gen_data.get("transl", gen_data.get("trans", np.zeros((1,3)))))
            gen_go = np.asarray(gen_data.get("global_orient", np.zeros((1,3))))
            gen_bp = np.asarray(gen_data.get("body_pose", np.zeros((1,63))))
            T_gen = gen_transl.shape[0]
            gen_poses_66 = np.concatenate([gen_go.reshape(T_gen, -1)[:, :3], gen_bp.reshape(T_gen, -1)[:, :63]], axis=1)
            gen_joints = numpy_fk_22(gen_transl, gen_poses_66, J_template)
            gen_height = compute_body_height(gen_joints)
            print(f"  {fname}: T={T_gen}, body_height={gen_height:.4f}m, transl_Y_mean={gen_transl[:, 1].mean():.4f}")
    else:
        print(f"  Output dir not found: {output_dir}")

    # =================== 7. Cross-check: compare encode_motion roundtrip ===================
    print("\n[7] Testing bundle.encode_motion vs manual encode...")
    # Use the same motion but go through bundle.encode_motion
    # The bundle.encode_motion expects [B, T, 138] (already normalized!)
    # Actually no - let's check:
    # bundle.encode_motion signature:
    #   motion = motion.float()  # input
    #   motion = self.smpl_pose_processor.normalize(motion)
    #   motion = rearrange(motion, 'b t (j d) -> b t j d', d=6)
    #   latents = self.vae.encode(motion.float())
    # So it expects RAW (unnormalized) motion [B, T, 138]!

    # Let's compare
    with torch.no_grad():
        bundle_latents = bundle.encode_motion(motion_tensor)
    print(f"  bundle.encode_motion output shape: {bundle_latents.shape}")
    print(f"  bundle.encode_motion output range: [{bundle_latents.min():.4f}, {bundle_latents.max():.4f}]")
    print(f"  Manual encode range: [{latents_normalized.min():.4f}, {latents_normalized.max():.4f}]")
    diff = (bundle_latents - latents_normalized).abs().max()
    print(f"  Max difference: {diff:.6f}")
    if diff < 1e-3:
        print(f"  ✓ Manual encode matches bundle.encode_motion")
    else:
        print(f"  ⚠️ MISMATCH between manual and bundle encode!")

    print("\n" + "=" * 70)
    print("DIAGNOSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
