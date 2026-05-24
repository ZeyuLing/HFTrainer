#!/usr/bin/env python3
"""Verify the first-frame velocity spike fix.

Tests that linear extrapolation of frame 0 from frames 1-2 eliminates
the velocity discontinuity without distorting the motion.
"""
import os, sys, json
import numpy as np
import torch

sys.path.insert(0, '/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def compute_per_frame_vel(arr):
    """Per-frame max-joint velocity from (T, D) array reshaped to (T, J, 6)."""
    T = arr.shape[0]
    if arr.ndim == 2:
        J = arr.shape[1] // 6
        arr = arr.reshape(T, J, 6)
    diffs = np.diff(arr, axis=0)
    frame_vel = np.linalg.norm(diffs, axis=2).max(axis=1)
    return frame_vel


def fix_first_frame(motion: torch.Tensor) -> torch.Tensor:
    """Apply first-frame fix: linear extrapolation from frames 1-2."""
    if motion.shape[1] >= 3:
        motion[:, 0] = 2.0 * motion[:, 1] - motion[:, 2]
    return motion


def main():
    device = torch.device('cuda:0')

    from hftrainer.registry import HF_MODELS, MODELS
    from einops import rearrange
    from hftrainer.models.motion.prism.gaussian_distribution import DiagonalGaussianDistributionNd

    # Build VAE
    vae_cfg = dict(
        type='AutoencoderKLPrism2DTK',
        from_pretrained=dict(pretrained_model_name_or_path='checkpoints/vermo_vae'),
    )
    vae = HF_MODELS.build(vae_cfg).eval().to(device)
    print(f"VAE loaded. z_dim={vae.config.z_dim}")

    # Build processor
    proc_cfg = dict(
        type='SMPLPoseProcessor',
        do_normalize=True,
        stats_file='data/statistic/smplx55_stats_hymotion_aug.json',
        rot_type='rotation_6d',
        transl_type='abs_rel',
        smpl_type='smpl_22',
    )
    processor = MODELS.build(proc_cfg)

    # Load GT motions
    train_json = 'data/motionhub/train.json'
    with open(train_json, 'r') as f:
        data_list = json.load(f).get('data_list', {})

    gt_samples = []
    for k, info in list(data_list.items())[:200]:
        path = info.get('smplx_path', '') or info.get('motion_path', '')
        if not path:
            continue
        fp = os.path.join('data/motionhub', path)
        if not os.path.isfile(fp):
            continue
        d = np.load(fp, allow_pickle=True)
        if 'poses' not in d or d['poses'].shape[0] < 60:
            continue
        gt_samples.append((k, fp, d))
        if len(gt_samples) >= 5:
            break

    print(f"\nFound {len(gt_samples)} GT samples")
    print(f"\n{'='*70}")
    print("=== Comparing: Original VAE Decode vs Fixed (linear extrapolation) ===")
    print(f"{'='*70}")

    for name, fp, d in gt_samples:
        poses = d['poses']
        T = poses.shape[0]
        T = min((T // 4) * 4, 128)
        if T < 16:
            continue

        go = d['poses'][:T, :3]
        bp = d['poses'][:T, 3:66]
        transl_key = 'trans' if 'trans' in d else 'transl'
        transl = d[transl_key][:T] if transl_key in d else np.zeros((T, 3))

        smplx_dict = {'global_orient': go, 'body_pose': bp, 'transl': transl}
        try:
            motion_vec = processor.smplx_dict_to_motion_vector(smplx_dict)
        except Exception as e:
            print(f"  Skip {name}: {e}")
            continue

        print(f"\n--- {name} (T={T}) ---")

        motion_vec_norm = processor.normalize(motion_vec.unsqueeze(0))
        x_vae = rearrange(motion_vec_norm, 'b t (j d) -> b t j d', d=6)

        with torch.no_grad():
            x_gpu = x_vae.float().to(device)
            z = vae.encode(x_gpu)
            dist = DiagonalGaussianDistributionNd(z)
            latent = dist.mode()

            # Decode WITHOUT fix
            decoded_raw = vae.decode(latent)

            # Decode WITH fix (simulate)
            decoded_fixed = decoded_raw.clone()
            decoded_fixed = fix_first_frame(decoded_fixed)

        # Compare
        original_np = x_vae.cpu().numpy()[0]
        raw_np = decoded_raw.cpu().numpy()[0]
        fixed_np = decoded_fixed.cpu().numpy()[0]

        T_out = raw_np.shape[0]
        orig_flat = original_np[:T_out].reshape(T_out, -1)
        raw_flat = raw_np.reshape(T_out, -1)
        fixed_flat = fixed_np.reshape(T_out, -1)

        orig_vel = compute_per_frame_vel(orig_flat)
        raw_vel = compute_per_frame_vel(raw_flat)
        fixed_vel = compute_per_frame_vel(fixed_flat)

        print(f"  {'Fr':<4} {'GT_vel':<10} {'Raw_vel':<10} {'Fixed_vel':<10} {'Raw/GT':<10} {'Fixed/GT':<10}")
        for i in range(min(8, len(orig_vel))):
            r_raw = raw_vel[i] / (orig_vel[i] + 1e-8)
            r_fix = fixed_vel[i] / (orig_vel[i] + 1e-8)
            print(f"  {i:<4} {orig_vel[i]:<10.5f} {raw_vel[i]:<10.5f} {fixed_vel[i]:<10.5f} {r_raw:<10.3f} {r_fix:<10.3f}")

        # Summary
        print(f"\n  Summary:")
        print(f"    Raw vel (frame 0):   {raw_vel[0]:.5f} ({raw_vel[0]/(orig_vel[0]+1e-8):.2f}x GT)")
        print(f"    Fixed vel (frame 0): {fixed_vel[0]:.5f} ({fixed_vel[0]/(orig_vel[0]+1e-8):.2f}x GT)")
        print(f"    Raw mean vel:   {raw_vel.mean():.5f}")
        print(f"    Fixed mean vel: {fixed_vel.mean():.5f}")
        print(f"    GT mean vel:    {orig_vel.mean():.5f}")

        # Check reconstruction error
        raw_err = np.linalg.norm((raw_np - original_np[:T_out]).reshape(T_out, -1), axis=1)
        fixed_err = np.linalg.norm((fixed_np - original_np[:T_out]).reshape(T_out, -1), axis=1)
        print(f"    Recon error (raw, frame0):   {raw_err[0]:.6f}")
        print(f"    Recon error (fixed, frame0): {fixed_err[0]:.6f}")
        print(f"    Recon error (raw, mean):     {raw_err.mean():.6f}")
        print(f"    Recon error (fixed, mean):   {fixed_err.mean():.6f}")

    # ============ Test with random latent ============
    print(f"\n\n{'='*70}")
    print("=== RANDOM LATENT decode: Raw vs Fixed ===")
    print(f"{'='*70}")

    latents_mean = torch.tensor(vae.config.latents_mean).float()
    latents_std = torch.tensor(vae.config.latents_std).float()
    z_dim = vae.config.z_dim
    T_latent = 32
    J_latent = 23

    for trial in range(3):
        torch.manual_seed(42 + trial)
        rand_latent = torch.randn(1, z_dim, T_latent, J_latent).to(device)
        rand_latent = rand_latent * latents_std.view(1, -1, 1, 1).to(device) + latents_mean.view(1, -1, 1, 1).to(device)

        with torch.no_grad():
            decoded_raw = vae.decode(rand_latent)
            decoded_fixed = fix_first_frame(decoded_raw.clone())

        raw_np = decoded_raw.cpu().numpy()[0]
        fixed_np = decoded_fixed.cpu().numpy()[0]
        T_out = raw_np.shape[0]

        raw_vel = compute_per_frame_vel(raw_np.reshape(T_out, -1))
        fixed_vel = compute_per_frame_vel(fixed_np.reshape(T_out, -1))

        print(f"\n  Trial {trial}: T_out={T_out}")
        print(f"    Raw:   first4={raw_vel[:4].mean():.5f}, rest={raw_vel[4:].mean():.5f}, ratio={raw_vel[:4].mean()/(raw_vel[4:].mean()+1e-8):.3f}")
        print(f"    Fixed: first4={fixed_vel[:4].mean():.5f}, rest={fixed_vel[4:].mean():.5f}, ratio={fixed_vel[:4].mean()/(fixed_vel[4:].mean()+1e-8):.3f}")
        print(f"    Frame 0: raw={raw_vel[0]:.5f}, fixed={fixed_vel[0]:.5f}")

    print(f"\n\n{'='*70}")
    print("CONCLUSION: If Fixed/GT ratios are close to 1.0 and Fixed first4/rest")
    print("ratio is close to 1.0, the fix successfully eliminates the velocity spike.")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
