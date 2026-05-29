#!/usr/bin/env python3
"""Test VAE roundtrip specifically for translation channels.

Answers the question: Can the VAE faithfully reconstruct GT translation
after encode->decode? If not, the transformer cannot possibly produce
correct translation regardless of its latent predictions.

Tests on the same samples that showed worst drift in inference.
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
    config_path = 'configs/prism/prism_1b_tp2m_multiframe_kt_spectral_unified.py'
    checkpoint_dir = 'work_dirs/prism_1b_tp2m_multiframe_kt_spectral/checkpoint-epoch_2'

    # Target samples (same as test_rollout_vs_abs.py)
    target_samples = [
        {'name': 'humanml3d_1059', 'num_frames': 121},  # sits, raises hand
        {'name': 'humanml3d_194', 'num_frames': 193},   # squats
        {'name': 'humanml3d_942', 'num_frames': 89},    # cross-legged eating
        {'name': 'humanml3d_927', 'num_frames': 193},   # jogging
        {'name': 'humanml3d_1063', 'num_frames': 133},  # walking
    ]

    # ==================== Build model ====================
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

    # Move only VAE + processor to GPU
    bundle.vae = bundle.vae.to(device, torch.float32).eval()
    bundle.smpl_pose_processor = bundle.smpl_pose_processor.to(device, torch.float32)

    from hftrainer.models.motion.prism.gaussian_distribution import DiagonalGaussianDistributionNd

    # VAE latent normalization constants
    latents_mean = torch.tensor(
        bundle.vae.config.latents_mean, dtype=torch.float32, device=device
    ).view(1, bundle.vae.config.z_dim, 1, 1)
    latents_std = torch.tensor(
        bundle.vae.config.latents_std, dtype=torch.float32, device=device
    ).view(1, bundle.vae.config.z_dim, 1, 1)

    vae_temporal = bundle.vae.config.scale_factor_temporal
    print(f'    VAE temporal compression: {vae_temporal}x')
    print(f'    VAE z_dim: {bundle.vae.config.z_dim}')

    # Load GT samples
    meta = json.loads(Path('data/annotation/test_hml3d.json').read_text())

    print('\n[2] Running VAE roundtrip on target samples...')
    print('=' * 80)

    for sample in target_samples:
        name = sample['name']
        num_frames = sample['num_frames']

        entry = meta['data_list'].get(name, {})
        gt_path = entry.get('smplx_path', '')
        if not gt_path:
            print(f'\n--- {name}: No smplx_path in metadata, skipping ---')
            continue

        gt_full = os.path.join('data/motionhub', gt_path)
        if not os.path.exists(gt_full):
            print(f'\n--- {name}: File not found: {gt_full}, skipping ---')
            continue

        print(f'\n--- {name} ({num_frames} frames) ---')

        # Load GT and convert to motion vector
        gt_data = dict(np.load(gt_full, allow_pickle=True))
        motion_vec = bundle.smpl_pose_processor.smplx_dict_to_motion_vector(gt_data)
        # motion_vec: (T, 138) tensor on device

        T_orig = motion_vec.shape[0]
        T_use = min(T_orig, num_frames, 196)  # VAE max 196
        motion_vec = motion_vec[:T_use]
        print(f'  motion_vec shape: {motion_vec.shape} (using {T_use} of {T_orig} frames)')

        # Extract GT translation info BEFORE normalization
        gt_transl_raw = motion_vec[:, :6].clone()  # [T, 6] abs_rel
        gt_abs_pos = gt_transl_raw[:, :3].cpu().numpy()
        gt_rel_vel = gt_transl_raw[:, 3:6].cpu().numpy()

        print(f'  GT abs position frame 0: {gt_abs_pos[0]}')
        print(f'  GT abs position frame -1: {gt_abs_pos[-1]}')
        print(f'  GT displacement: {np.linalg.norm(gt_abs_pos[-1] - gt_abs_pos[0]):.4f} m')
        print(f'  GT rel velocity mean: {gt_rel_vel.mean(axis=0)}')
        print(f'  GT rel velocity std: {gt_rel_vel.std(axis=0)}')

        # ==================== VAE encode ====================
        with torch.no_grad():
            motion_input = motion_vec.unsqueeze(0).to(device)  # (1, T, 138)
            latents = bundle.encode_motion(motion_input)
            print(f'  Latent shape: {latents.shape}')  # (1, Z, T_latent, 23)

        # ==================== VAE decode ====================
        with torch.no_grad():
            # Un-normalize latents
            decoded_latents = latents * latents_std + latents_mean
            motion_decoded = bundle.vae.decode(decoded_latents.float())  # (B, T, J, D=6)

            # Flatten and denormalize
            x_dec = rearrange(motion_decoded, 'b t j d -> b t (j d)').float()
            x_dec = bundle.smpl_pose_processor.denormalize(x_dec)

        # ==================== Compare translation ====================
        T_dec = x_dec.shape[1]
        print(f'  Decoded T={T_dec} (input T={T_use}, ratio={T_use/T_dec:.2f})')

        # Get denormalized abs_rel
        dec_transl_abs_rel = x_dec[0, :, :6].cpu().numpy()  # [T_dec, 6]
        dec_abs_pos = dec_transl_abs_rel[:, :3]
        dec_rel_vel = dec_transl_abs_rel[:, 3:6]

        # Compare with GT (using min of both lengths)
        T_cmp = min(T_use, T_dec)
        gt_cmp = gt_transl_raw[:T_cmp].cpu().numpy()

        # Absolute position error
        abs_pos_err = np.abs(dec_abs_pos[:T_cmp] - gt_cmp[:, :3])
        print(f'\n  === TRANSLATION RECONSTRUCTION ERROR ===')
        print(f'  Abs position L1 error (per-frame mean): {abs_pos_err.mean():.6f} m')
        print(f'  Abs position L1 error (max): {abs_pos_err.max():.6f} m')
        print(f'  Abs position L1 per-axis mean: X={abs_pos_err[:, 0].mean():.6f}, '
              f'Y={abs_pos_err[:, 1].mean():.6f}, Z={abs_pos_err[:, 2].mean():.6f}')

        # Relative velocity error
        rel_vel_err = np.abs(dec_rel_vel[:T_cmp] - gt_cmp[:, 3:6])
        print(f'  Rel velocity L1 error (per-frame mean): {rel_vel_err.mean():.6f} m/frame')
        print(f'  Rel velocity L1 error (max): {rel_vel_err.max():.6f} m/frame')

        # Displacement after roundtrip
        dec_disp = np.linalg.norm(dec_abs_pos[T_cmp-1] - dec_abs_pos[0])
        gt_disp = np.linalg.norm(gt_abs_pos[T_cmp-1] - gt_abs_pos[0])
        print(f'\n  Displacement comparison:')
        print(f'    GT:        {gt_disp:.4f} m')
        print(f'    Decoded:   {dec_disp:.4f} m')
        print(f'    Ratio:     {dec_disp / max(gt_disp, 0.001):.2f}x')

        # Also test rollout reconstruction
        with torch.no_grad():
            transl_abs_rel_t = x_dec[..., :6]
            transl_rollout = bundle.smpl_pose_processor.inv_convert_transl(transl_abs_rel_t)
            t_rollout = transl_rollout[0].cpu().numpy()
            disp_rollout = np.linalg.norm(t_rollout[T_cmp-1] - t_rollout[0])
            print(f'    Rollout:   {disp_rollout:.4f} m')

        # Pose reconstruction error
        gt_poses = motion_vec[:T_cmp, 6:].cpu().numpy()
        dec_poses = x_dec[0, :T_cmp, 6:].cpu().numpy()
        pose_err = np.abs(dec_poses - gt_poses)
        print(f'\n  Pose reconstruction L1 error: {pose_err.mean():.6f}')
        print(f'  Pose reconstruction max error: {pose_err.max():.6f}')

        # Normalized space comparison (to see if error is pre or post normalization)
        with torch.no_grad():
            motion_norm = bundle.smpl_pose_processor.normalize(motion_input)
            x_dec_norm = bundle.smpl_pose_processor.normalize(x_dec[:, :T_use])
            norm_diff = (x_dec_norm[:, :T_cmp] - motion_norm[:, :T_cmp]).abs()
            print(f'\n  === NORMALIZED SPACE ERROR (pre-denorm) ===')
            print(f'  Overall L1: {norm_diff.mean().item():.6f}')
            print(f'  Transl dims [0:6] L1: {norm_diff[..., :6].mean().item():.6f}')
            print(f'  Pose dims [6:] L1: {norm_diff[..., 6:].mean().item():.6f}')
            print(f'  Transl abs [0:3] L1: {norm_diff[..., :3].mean().item():.6f}')
            print(f'  Transl rel [3:6] L1: {norm_diff[..., 3:6].mean().item():.6f}')

    print('\n' + '=' * 80)
    print('[+] VAE roundtrip test complete.')
    print('\nInterpretation:')
    print('  - If reconstruction error is LOW (< 0.01 m for transl):')
    print('    -> VAE is faithful, problem is in transformer latent prediction')
    print('  - If reconstruction error is HIGH (> 0.1 m for transl):')
    print('    -> VAE itself cannot represent translation well, fundamental issue')


if __name__ == '__main__':
    main()
