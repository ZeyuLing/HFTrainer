#!/usr/bin/env python3
"""VAE encode-decode roundtrip test for PRISM.

Tests whether VAE can faithfully reconstruct a GT motion.
Also tests the rot6d convention by comparing:
1. Direct conversion (no permutation)
2. With [0,2,4,1,3,5] permutation

If the VAE roundtrip works without permutation but fails with it,
the permutation is the bug.
"""

import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from einops import rearrange

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def main():
    device = torch.device('cuda:0')
    config_path = 'configs/prism/prism_1b_tp2m_multiframe_kt_spectral.py'
    checkpoint_dir = 'work_dirs/prism_1b_tp2m_multiframe_kt_spectral/checkpoint-epoch_2'
    gt_npz_path = 'data/motionhub/humanact12/smplx_55/010541.npz'
    output_dir = 'work_dirs/prism_1b_tp2m_multiframe_kt_spectral/vae_roundtrip_test'
    os.makedirs(output_dir, exist_ok=True)

    # ==================== 1. Build model ====================
    print('[1] Building PRISM bundle...')
    from mmengine.config import Config
    from hftrainer.registry import MODEL_BUNDLES

    cfg = Config.fromfile(config_path)
    bundle = MODEL_BUNDLES.build(cfg.model)

    ckpt_path = os.path.join(checkpoint_dir, 'model.pt')
    print(f'    Loading checkpoint: {ckpt_path}')
    state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    bundle.load_state_dict_selective(state_dict, strict=False)
    del state_dict
    gc.collect()

    # Move only VAE + processor to GPU (we don't need transformer/text_encoder)
    bundle.vae = bundle.vae.to(device, torch.float32).eval()
    bundle.smpl_pose_processor = bundle.smpl_pose_processor.to(device, torch.float32)

    from hftrainer.models.motion.prism.gaussian_distribution import DiagonalGaussianDistributionNd
    from hftrainer.models.motion.components.utils.geometry.rotation_convert import (
        rotation_6d_to_axis_angle,
    )

    # ==================== 2. Load GT motion and convert ====================
    print('[2] Loading GT motion and converting to rot6d...')
    gt_smplx = dict(np.load(gt_npz_path, allow_pickle=True))
    print(f'    GT keys: {list(gt_smplx.keys())}')
    print(f'    body_pose shape: {gt_smplx.get("body_pose", np.zeros(1)).shape}')
    print(f'    transl shape: {gt_smplx.get("transl", np.zeros(1)).shape}')

    # Convert GT to motion vector using the processor (axis_angle -> rot6d, column-major)
    motion_vec = bundle.smpl_pose_processor.smplx_dict_to_motion_vector(gt_smplx)
    # motion_vec shape: (T, D) where D = 6(transl) + 23*6(poses) = 144
    print(f'    motion_vec shape: {motion_vec.shape}')
    print(f'    motion_vec[:3, :6] (transl part): {motion_vec[:3, :6]}')

    T = motion_vec.shape[0]
    # Limit to max 196 frames (VAE constraint)
    if T > 196:
        motion_vec = motion_vec[:196]
        T = 196
    print(f'    Using T={T} frames')

    # ==================== 3. VAE encode ====================
    print('[3] VAE encode...')
    with torch.no_grad():
        motion_input = motion_vec.unsqueeze(0).to(device)  # (1, T, D)
        # This does: normalize -> rearrange to (B,T,J,6) -> encode -> mode
        latents = bundle.encode_motion(motion_input)
        print(f'    Latent shape: {latents.shape}')

    # ==================== 4. VAE decode ====================
    print('[4] VAE decode...')
    latents_mean = torch.tensor(
        bundle.vae.config.latents_mean, dtype=torch.float32, device=device
    ).view(1, bundle.vae.config.z_dim, 1, 1)
    latents_std = torch.tensor(
        bundle.vae.config.latents_std, dtype=torch.float32, device=device
    ).view(1, bundle.vae.config.z_dim, 1, 1)

    with torch.no_grad():
        # Un-normalize latents (reverse of encode_motion normalization)
        decoded_latents = latents * latents_std + latents_mean
        motion_decoded = bundle.vae.decode(decoded_latents.float())  # (B, T, J, D=6)
        print(f'    Decoded motion shape: {motion_decoded.shape}')

    # ==================== 5. Post-process (WITHOUT permutation) ====================
    print('[5] Post-processing WITHOUT rot6d permutation...')
    with torch.no_grad():
        x_dec = rearrange(motion_decoded, 'b t j d -> b t (j d)').float()
        x_dec = bundle.smpl_pose_processor.denormalize(x_dec)

        # Compare denormalized output with original motion_vec
        T_dec = x_dec.shape[1]
        gt_on_device = motion_vec[:T_dec].unsqueeze(0).to(device)
        diff = (x_dec - gt_on_device).abs()
        print(f'    VAE output T={T_dec} (input was T={T})')
        print(f'    Reconstruction error (L1):')
        print(f'      Overall mean: {diff.mean().item():.6f}')
        print(f'      Transl part (first 6 dims): {diff[..., :6].mean().item():.6f}')
        print(f'      Pose part (6: dims): {diff[..., 6:].mean().item():.6f}')
        print(f'      Max error: {diff.max().item():.6f}')

        # Convert back to SMPL — NO permutation
        transl_abs_rel = x_dec[..., :6]
        transl = bundle.smpl_pose_processor.inv_convert_transl(transl_abs_rel)
        pred_poses = x_dec[..., 6:]

        pred_poses_shaped = rearrange(pred_poses, "b t (j d) -> (b t) j d", d=6)
        # DIRECT conversion — no permutation
        pred_poses_aa = rotation_6d_to_axis_angle(pred_poses_shaped)
        pred_poses_aa = rearrange(pred_poses_aa, "(b t) j d -> b t (j d)", b=1)

        smplx_dict_no_perm = bundle.smpl_pose_processor.transl_pose_to_smplx_dict(
            transl.squeeze(0).float(),
            pred_poses_aa.squeeze(0).float(),
            mocap_framerate=30.0,
            gender='neutral',
            rot_type='axis_angle',
        )
        smplx_dict_no_perm = bundle.smpl_pose_processor.normalize_smplx_dict(smplx_dict_no_perm)

    # Save
    out_no_perm = os.path.join(output_dir, 'roundtrip_NO_permutation.npz')
    np.savez_compressed(out_no_perm, **{
        k: v.detach().cpu().numpy().astype(np.float32) if isinstance(v, torch.Tensor)
        else (v.astype(np.float32) if isinstance(v, np.ndarray) else v)
        for k, v in smplx_dict_no_perm.items()
    })
    print(f'    Saved: {out_no_perm}')

    # ==================== 6. Post-process WITH permutation [0,2,4,1,3,5] ====================
    print('[6] Post-processing WITH rot6d permutation [0,2,4,1,3,5]...')
    with torch.no_grad():
        pred_poses_shaped2 = rearrange(pred_poses, "b t (j d) -> (b t) j d", d=6)
        # Apply the "row-major to column-major" permutation
        pred_poses_perm = pred_poses_shaped2[..., [0, 2, 4, 1, 3, 5]]
        pred_poses_aa2 = rotation_6d_to_axis_angle(pred_poses_perm)
        pred_poses_aa2 = rearrange(pred_poses_aa2, "(b t) j d -> b t (j d)", b=1)

        smplx_dict_with_perm = bundle.smpl_pose_processor.transl_pose_to_smplx_dict(
            transl.squeeze(0).float(),
            pred_poses_aa2.squeeze(0).float(),
            mocap_framerate=30.0,
            gender='neutral',
            rot_type='axis_angle',
        )
        smplx_dict_with_perm = bundle.smpl_pose_processor.normalize_smplx_dict(smplx_dict_with_perm)

    out_with_perm = os.path.join(output_dir, 'roundtrip_WITH_permutation.npz')
    np.savez_compressed(out_with_perm, **{
        k: v.detach().cpu().numpy().astype(np.float32) if isinstance(v, torch.Tensor)
        else (v.astype(np.float32) if isinstance(v, np.ndarray) else v)
        for k, v in smplx_dict_with_perm.items()
    })
    print(f'    Saved: {out_with_perm}')

    # ==================== 7. Save GT as reference ====================
    print('[7] Saving GT reference...')
    out_gt = os.path.join(output_dir, 'gt_original.npz')
    gt_pack = {}
    for k, v in gt_smplx.items():
        if isinstance(v, np.ndarray):
            if v.dtype.kind == 'f' or v.dtype.kind == 'i':
                gt_pack[k] = v.astype(np.float32)
            else:
                gt_pack[k] = v
        elif isinstance(v, (int, float)):
            gt_pack[k] = np.array(v, dtype=np.float32)
        elif isinstance(v, str):
            gt_pack[k] = np.array(v)
    np.savez_compressed(out_gt, **gt_pack)
    print(f'    Saved: {out_gt}')

    # ==================== 8. Compare body_pose ranges ====================
    print('\n[8] Body pose comparison:')
    T_out = bp_no_perm.shape[0] if not isinstance(
        smplx_dict_no_perm.get('body_pose'), type(None)) else T
    gt_bp = gt_smplx.get('body_pose', np.zeros((T, 63)))

    bp_no_perm = smplx_dict_no_perm.get('body_pose', np.zeros((T, 63)))
    if isinstance(bp_no_perm, torch.Tensor):
        bp_no_perm = bp_no_perm.cpu().numpy()
    T_out = bp_no_perm.shape[0]
    gt_bp = gt_bp[:T_out]
    print(f'    GT body_pose range: [{gt_bp.min():.3f}, {gt_bp.max():.3f}]')
    print(f'    NO permutation body_pose range: [{bp_no_perm.min():.3f}, {bp_no_perm.max():.3f}]')

    bp_with_perm = smplx_dict_with_perm.get('body_pose', np.zeros((T, 63)))
    if isinstance(bp_with_perm, torch.Tensor):
        bp_with_perm = bp_with_perm.cpu().numpy()
    print(f'    WITH permutation body_pose range: [{bp_with_perm.min():.3f}, {bp_with_perm.max():.3f}]')

    # L1 error vs GT
    err_no_perm = np.abs(bp_no_perm - gt_bp).mean()
    err_with_perm = np.abs(bp_with_perm[:T_out] - gt_bp).mean()
    print(f'\n    L1 error vs GT body_pose:')
    print(f'      NO permutation: {err_no_perm:.6f}')
    print(f'      WITH permutation: {err_with_perm:.6f}')
    print(f'      >>> {"NO perm is BETTER" if err_no_perm < err_with_perm else "WITH perm is BETTER"} <<<')

    print(f'\n[+] All files saved to: {output_dir}')
    print('    View them at: http://<IP>:8084/browse?path=' + os.path.abspath(output_dir))


if __name__ == '__main__':
    main()
