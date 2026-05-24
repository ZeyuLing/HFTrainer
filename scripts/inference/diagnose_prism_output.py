"""Diagnostic script to isolate PRISM inference deformation bug.

Tests:
1. Load generated NPZ → FK → check body proportions
2. Load real training data → full encode→decode round-trip → check proportions
3. Compare to isolate whether bug is in denoising vs post-processing

Usage:
    python3 scripts/inference/diagnose_prism_output.py \
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
    axis_angle_to_matrix,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--generated-npz', default=None,
                        help='Path to a generated NPZ to diagnose')
    parser.add_argument('--gt-npz', default=None,
                        help='Path to a ground-truth training NPZ for round-trip test')
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()


def check_smplx_dict_quality(smplx_dict, label=""):
    """Check body proportions from SMPLX dict using FK."""
    print(f"\n{'='*60}")
    print(f"  Quality Check: {label}")
    print(f"{'='*60}")

    transl = smplx_dict.get('transl', None)
    global_orient = smplx_dict.get('global_orient', None)
    body_pose = smplx_dict.get('body_pose', None)

    if transl is None or global_orient is None or body_pose is None:
        print("  ERROR: Missing required keys (transl, global_orient, body_pose)")
        print(f"  Available keys: {list(smplx_dict.keys())}")
        return

    print(f"  Shape: transl={transl.shape}, global_orient={global_orient.shape}, body_pose={body_pose.shape}")
    print(f"  Transl range: X=[{transl[:,0].min():.3f}, {transl[:,0].max():.3f}], "
          f"Y=[{transl[:,1].min():.3f}, {transl[:,1].max():.3f}], "
          f"Z=[{transl[:,2].min():.3f}, {transl[:,2].max():.3f}]")

    # Check axis-angle magnitudes
    T = body_pose.shape[0]
    poses_per_joint = body_pose.reshape(T, -1, 3)
    magnitudes = np.linalg.norm(poses_per_joint, axis=-1)
    print(f"  Body pose axis-angle magnitudes: mean={magnitudes.mean():.3f}, "
          f"max={magnitudes.max():.3f}, >π count={int((magnitudes > np.pi).sum())}")

    # Translation velocity
    if T > 1:
        vel = np.diff(transl, axis=0)
        vel_mag = np.linalg.norm(vel, axis=-1)
        print(f"  Translation velocity: mean={vel_mag.mean():.4f} m/frame, "
              f"max={vel_mag.max():.4f} m/frame")
        if vel_mag.max() > 0.5:
            print("  WARNING: Very high translation velocity (>0.5 m/frame at 30fps = 15 m/s)")

    return transl, body_pose


def run_fk_check(bundle, smplx_dict, label=""):
    """Run forward kinematics and check body height."""
    print(f"\n  FK Check for: {label}")

    transl = torch.from_numpy(smplx_dict['transl']).float()
    global_orient = torch.from_numpy(smplx_dict['global_orient']).float()
    body_pose = torch.from_numpy(smplx_dict['body_pose']).float()

    T = transl.shape[0]

    if bundle.smpl_pose_processor.smpl_model is not None:
        smpl_model = bundle.smpl_pose_processor.smpl_model
        # SmplxLiteV437Coco17 forward - uses buffers not parameters
        try:
            device = next(smpl_model.parameters()).device
        except StopIteration:
            device = next(smpl_model.buffers()).device

        # Default betas (neutral shape)
        betas = torch.zeros(T, 10)

        # Process in chunks to avoid OOM
        chunk_size = 32
        all_joints = []
        for i in range(0, T, chunk_size):
            chunk_t = transl[i:i+chunk_size].to(device)
            chunk_go = global_orient[i:i+chunk_size].to(device)
            chunk_bp = body_pose[i:i+chunk_size].to(device)
            chunk_betas = betas[i:i+chunk_size].to(device)

            with torch.no_grad():
                output = smpl_model(
                    body_pose=chunk_bp,
                    betas=chunk_betas,
                    global_orient=chunk_go,
                    transl=chunk_t,
                )
            # output is tuple: (verts437, coco17_joints)
            if isinstance(output, tuple):
                joints = output[1]  # coco17_joints: [chunk, 17, 3]
            else:
                joints = output
            all_joints.append(joints.cpu())

        joints = torch.cat(all_joints, dim=0)  # [T, 17, 3]

        # COCO17 joint indices: 0=nose, 5=left_shoulder, 6=right_shoulder,
        # 11=left_hip, 12=right_hip, 15=left_ankle, 16=right_ankle
        # Body height = max_y - min_y across all joints
        heights = joints[:, :, 1].max(dim=1).values - joints[:, :, 1].min(dim=1).values
        print(f"    Body height: mean={heights.mean():.3f}m, min={heights.min():.3f}m, max={heights.max():.3f}m")
        print(f"    Expected: ~1.5-1.8m for adult")

        # Ankle heights (should be near ground)
        ankle_y = joints[:, [15, 16], 1]  # [T, 2]
        print(f"    Ankle Y: mean={ankle_y.mean():.3f}m, min={ankle_y.min():.3f}m")
        print(f"    Expected: near 0m (ground)")

        # Hip height
        hip_y = joints[:, [11, 12], 1].mean(dim=1)
        print(f"    Hip Y: mean={hip_y.mean():.3f}m")
        print(f"    Expected: ~0.85-1.0m")

        # Shoulder span
        shoulder_span = (joints[:, 5, :] - joints[:, 6, :]).norm(dim=-1)
        print(f"    Shoulder span: mean={shoulder_span.mean():.3f}m")
        print(f"    Expected: ~0.35-0.45m")

        if heights.mean() < 1.0:
            print("    *** FAIL: Body height too small - motion is DEFORMED ***")
        elif heights.mean() > 2.5:
            print("    *** FAIL: Body height too large - motion is DEFORMED ***")
        else:
            print("    *** PASS: Body proportions look reasonable ***")

        return heights.mean().item()
    else:
        print("    SMPL model not available, skipping FK")
        return None


@torch.no_grad()
def vae_roundtrip_test(bundle, gt_npz_path, device):
    """Load GT data → encode → decode → post-process → FK check.

    This tests the decode pipeline in isolation. If this produces correct
    body proportions, the bug is in the denoising loop. If this also
    produces deformed output, the bug is in the decode pipeline.
    """
    print(f"\n{'='*60}")
    print(f"  VAE Round-Trip Test")
    print(f"  GT NPZ: {gt_npz_path}")
    print(f"{'='*60}")

    # Load ground truth NPZ
    data = dict(np.load(gt_npz_path, allow_pickle=True))
    print(f"  NPZ keys: {list(data.keys())}")

    transl_gt = data.get('transl', data.get('trans', None))
    poses_gt = data.get('poses', None)

    if transl_gt is None or poses_gt is None:
        print("  ERROR: NPZ missing 'transl'/'trans' or 'poses' key")
        return

    print(f"  GT shapes: transl={transl_gt.shape}, poses={poses_gt.shape}")
    T = transl_gt.shape[0]

    # Limit frames for memory
    max_frames = 129
    if T > max_frames:
        transl_gt = transl_gt[:max_frames]
        poses_gt = poses_gt[:max_frames]
        T = max_frames

    # === Step 1: Convert GT to training format (replicate LoadSmplx55) ===
    # poses_gt is [T, 156] for SMPL-H (52 joints * 3) or [T, 165] for SMPL-X (55 joints * 3)
    n_joints_in = poses_gt.shape[1] // 3
    print(f"  Input joints: {n_joints_in}")

    # Select SMPL-22 joints: global_orient (1) + body (21) = 22 joints
    # SMPL joint indices for smpl_22
    if n_joints_in == 52:
        # SMPL-H: first 22 joints
        sel = list(range(22))
    elif n_joints_in == 55:
        # SMPL-X 55: first 22 joints
        sel = list(range(22))
    elif n_joints_in == 24:
        # SMPL: first 22 joints
        sel = list(range(22))
    else:
        print(f"  WARNING: Unexpected joint count {n_joints_in}, using first 22")
        sel = list(range(22))

    # Reshape to [T, J, 3] and select joints
    aa = poses_gt.reshape(T, n_joints_in, 3)[:, sel, :]  # [T, 22, 3]

    # Convert axis-angle to rotation_6d (column-major)
    aa_flat = aa.reshape(T * 22, 3)
    rot6d_col = axis_angle_to_rotation_6d(torch.from_numpy(aa_flat).float())  # [T*22, 6]
    rot6d_col = rot6d_col.reshape(T, 22, 6)

    # Permute to row-major: col_major[0,3,1,4,2,5] -> row_major
    rot6d_row = rot6d_col[:, :, [0, 3, 1, 4, 2, 5]]
    poses_138 = rot6d_row.reshape(T, 22 * 6)  # [T, 132]

    # Convert translation to abs_rel format: [abs_x, abs_y, abs_z, rel_x, rel_y, rel_z]
    transl_t = torch.from_numpy(transl_gt).float()  # [T, 3]
    rel_t = torch.zeros_like(transl_t)
    rel_t[1:] = transl_t[1:] - transl_t[:-1]
    transl_abs_rel = torch.cat([transl_t, rel_t], dim=-1)  # [T, 6]

    # Full motion vector: [T, 138]
    motion_vec = torch.cat([transl_abs_rel, poses_138], dim=-1).unsqueeze(0)  # [1, T, 138]
    print(f"  Motion vector shape: {motion_vec.shape}")

    # === Step 2: Normalize ===
    smpl_processor = bundle.smpl_pose_processor
    motion_norm = smpl_processor.normalize(motion_vec)
    print(f"  Normalized motion stats: mean={motion_norm.mean():.4f}, std={motion_norm.std():.4f}")

    # === Step 3: Reshape for VAE: [B, T, J, 6] → [B, T, 23, 6] ===
    # 138 = 23 joints * 6 (6 transl + 22*6 rotation → treated as 23 "joints" each with 6 channels)
    motion_reshaped = rearrange(motion_norm, 'b t (j d) -> b t j d', d=6)  # [1, T, 23, 6]
    print(f"  Reshaped for VAE: {motion_reshaped.shape}")

    # === Step 4: VAE Encode ===
    vae = bundle.vae.to(device)
    motion_dev = motion_reshaped.to(device).float()

    from hftrainer.models.motion.prism.gaussian_distribution import DiagonalGaussianDistributionNd

    device_type = device.type if hasattr(device, 'type') else 'cuda'
    with torch.autocast(device_type, enabled=False):
        latents_raw = vae.encode(motion_dev)
    latents = DiagonalGaussianDistributionNd(latents_raw).mode()

    # Normalize latents
    latents_mean = bundle.latents_mean.to(latents)
    latents_std = bundle.latents_std.to(latents)
    latents_normed = (latents - latents_mean) / latents_std
    print(f"  Encoded latents shape: {latents_normed.shape}")
    print(f"  Latents stats: mean={latents_normed.mean():.4f}, std={latents_normed.std():.4f}")

    # === Step 5: VAE Decode (denormalize latents first) ===
    latents_denorm = latents_normed * latents_std + latents_mean
    with torch.autocast(device_type, enabled=False):
        motion_decoded = vae.decode(latents_denorm.float())  # [B, T, J, 6]
    print(f"  Decoded motion shape: {motion_decoded.shape}")

    # === Step 6: Post-process (same as prism_backend.post_process_motion) ===
    x_dec = rearrange(motion_decoded, "b t j d -> b t (j d)")  # [1, T_dec, 138]
    print(f"  Flattened decoded: {x_dec.shape}")

    # Denormalize
    x_dec = smpl_processor.denormalize(x_dec)

    transl_abs_rel_dec = x_dec[..., :6]
    transl_dec = smpl_processor.inv_convert_transl(transl_abs_rel_dec)
    pred_poses = x_dec[..., 6:]

    # Convert rot6d to axis-angle
    pred_poses = rearrange(pred_poses, "b t (j d) -> (b t) j d", d=6)
    # Row-major → column-major for rotation_6d_to_axis_angle
    pred_poses = pred_poses[..., [0, 2, 4, 1, 3, 5]]
    pred_poses = rotation_6d_to_axis_angle(pred_poses)
    pred_poses = rearrange(pred_poses, "(b t) j d -> b t (j d)", b=1)

    # Convert to smplx_dict
    pred_smplx_dict = smpl_processor.transl_pose_to_smplx_dict(
        transl_dec.squeeze(0),
        pred_poses.squeeze(0),
        mocap_framerate=30.0,
        gender='neutral',
        rot_type="axis_angle",
    )
    pred_smplx_dict = smpl_processor.normalize_smplx_dict(pred_smplx_dict)

    # === Step 7: FK check ===
    print("\n  --- Round-trip result ---")
    check_smplx_dict_quality(pred_smplx_dict, "VAE Round-Trip")
    rt_height = run_fk_check(bundle, pred_smplx_dict, "VAE Round-Trip")

    # === Also check the original GT directly ===
    print("\n  --- Original GT (for comparison) ---")
    gt_smplx_dict = {
        'transl': transl_gt[:T],
        'global_orient': poses_gt[:T, :3],
        'body_pose': poses_gt[:T, 3:66],  # 21 joints * 3
    }
    gt_smplx_dict_norm = smpl_processor.normalize_smplx_dict(gt_smplx_dict)
    check_smplx_dict_quality(gt_smplx_dict_norm, "Original GT")
    gt_height = run_fk_check(bundle, gt_smplx_dict_norm, "Original GT")

    if rt_height and gt_height:
        ratio = rt_height / gt_height
        print(f"\n  Height ratio (round-trip / GT): {ratio:.3f}")
        if abs(ratio - 1.0) < 0.1:
            print("  *** CONCLUSION: Decode pipeline is CORRECT. Bug is in DENOISING LOOP. ***")
        else:
            print("  *** CONCLUSION: Decode pipeline has a BUG. ***")

    return pred_smplx_dict


def main():
    args = parse_args()
    device = torch.device(args.device)

    # Build bundle
    cfg = Config.fromfile(args.config)
    model_cfg = cfg.model.to_dict() if hasattr(cfg.model, 'to_dict') else cfg.model
    bundle_cls = MODEL_BUNDLES.get(model_cfg['type'])
    bundle = bundle_cls.from_config(model_cfg)
    bundle.eval()

    # Load checkpoint (only need VAE + smpl_processor, not transformer)
    state_dict = load_checkpoint(args.checkpoint, map_location='cpu')
    print(f'Loading checkpoint: {args.checkpoint}')
    bundle.load_state_dict_selective(state_dict)
    del state_dict
    gc.collect()

    # === Test 1: Check generated NPZ if provided ===
    if args.generated_npz:
        gen_data = dict(np.load(args.generated_npz, allow_pickle=True))
        print(f"\n{'='*60}")
        print(f"  Checking Generated NPZ: {args.generated_npz}")
        print(f"{'='*60}")
        check_smplx_dict_quality(gen_data, "Generated Motion")
        run_fk_check(bundle, gen_data, "Generated Motion")
    else:
        # Default: check all generated files
        eval_dir = "work_dirs/prism_1b_tp2m_multiframe/eval_after_fix_v2"
        if os.path.isdir(eval_dir):
            npz_files = sorted([f for f in os.listdir(eval_dir) if f.endswith('.npz')])
            if npz_files:
                gen_path = os.path.join(eval_dir, npz_files[0])
                gen_data = dict(np.load(gen_path, allow_pickle=True))
                print(f"\n{'='*60}")
                print(f"  Checking Generated NPZ: {gen_path}")
                print(f"{'='*60}")
                check_smplx_dict_quality(gen_data, "Generated Motion")
                run_fk_check(bundle, gen_data, "Generated Motion")

    # === Test 2: VAE round-trip on GT data ===
    gt_npz = args.gt_npz
    if gt_npz is None:
        # Try to find a GT NPZ
        candidates = [
            "data/hymotion_data/Academic/20250916/motions/HumanML3D-HumanEva/S3_Box_1_poses_origintime_13.55_22.7.npz",
        ]
        for c in candidates:
            if os.path.isfile(c):
                gt_npz = c
                break

    if gt_npz and os.path.isfile(gt_npz):
        vae_roundtrip_test(bundle, gt_npz, device)
    else:
        print(f"\n  WARNING: No GT NPZ found for round-trip test.")
        print(f"  Tried: {gt_npz}")
        print(f"  Pass --gt-npz <path> to specify one.")


if __name__ == '__main__':
    main()
