#!/usr/bin/env python3
"""Test if the VAE decoder introduces first-frame velocity artifacts.

Loads the VAE from its pretrained checkpoint (checkpoints/vermo_vae),
encodes GT motion, decodes it, and compares per-frame velocity.
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


def main():
    device = torch.device('cuda:0')

    from hftrainer.registry import HF_MODELS, MODELS

    # Build VAE from pretrained
    vae_cfg = dict(
        type='AutoencoderKLPrism2DTK',
        from_pretrained=dict(pretrained_model_name_or_path='checkpoints/vermo_vae'),
    )
    print("Building VAE from pretrained...")
    vae = HF_MODELS.build(vae_cfg)
    vae = vae.eval().to(device)
    print(f"VAE loaded. z_dim={vae.config.z_dim}, scale_factor_temporal={vae.config.scale_factor_temporal}")
    print(f"  temporal_downsample={vae.config.temporal_downsample}")

    # Build processor (without smpl_model to save memory)
    proc_cfg = dict(
        type='SMPLPoseProcessor',
        do_normalize=True,
        stats_file='data/statistic/smplx55_stats_hymotion_aug.json',
        rot_type='rotation_6d',
        transl_type='abs_rel',
        smpl_type='smpl_22',
    )
    processor = MODELS.build(proc_cfg)
    print(f"Processor: transl_type={processor.transl_type}, rot_type={processor.rot_type}")
    print(f"  mean shape: {processor.mean.shape}, std shape: {processor.std.shape}")

    # Load GT motions
    train_json = 'data/motionhub/train.json'
    with open(train_json, 'r') as f:
        data_list = json.load(f).get('data_list', {})

    gt_samples = []
    for k, info in data_list.items():
        if 'humanml3d' not in k.lower():
            continue
        path = info.get('smplx_path', '')
        if not path:
            path = info.get('motion_path', '')
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

    # If no humanml3d samples, try any
    if not gt_samples:
        print("No humanml3d samples found, trying any...")
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
    print("=== VAE Encode-Decode Roundtrip: Does first frame get corrupted? ===")
    print(f"{'='*70}")

    from einops import rearrange
    from hftrainer.models.motion.prism.gaussian_distribution import DiagonalGaussianDistributionNd

    for name, fp, d in gt_samples:
        poses = d['poses']
        T = poses.shape[0]
        T = (T // 4) * 4
        if T < 16:
            continue
        T = min(T, 128)

        go = d['poses'][:T, :3]
        bp = d['poses'][:T, 3:66]
        transl_key = 'trans' if 'trans' in d else 'transl'
        transl = d[transl_key][:T] if transl_key in d else np.zeros((T, 3))

        smplx_dict = {
            'global_orient': go,
            'body_pose': bp,
            'transl': transl,
        }
        try:
            motion_vec = processor.smplx_dict_to_motion_vector(smplx_dict)  # (T, D) tensor
        except Exception as e:
            print(f"  Skip {name}: {e}")
            continue

        print(f"\n--- {name} (T={T}) ---")

        # Normalize
        motion_vec_norm = processor.normalize(motion_vec.unsqueeze(0))  # (1, T, D)

        # Reshape for VAE: [B, T, J, C] where D=J*6
        J = motion_vec_norm.shape[-1] // 6
        x_vae = rearrange(motion_vec_norm, 'b t (j d) -> b t j d', d=6)
        print(f"  Input: shape={x_vae.shape}, range=[{x_vae.min():.3f}, {x_vae.max():.3f}]")

        with torch.no_grad():
            x_gpu = x_vae.float().to(device)
            z = vae.encode(x_gpu)
            dist = DiagonalGaussianDistributionNd(z)
            latent = dist.mode()
            print(f"  Latent: {latent.shape}")
            decoded = vae.decode(latent)

        decoded_np = decoded.cpu().numpy()[0]  # (T, J, C)
        original_np = x_vae.cpu().numpy()[0]

        T_out = decoded_np.shape[0]
        orig_flat = original_np[:T_out].reshape(T_out, -1)
        recon_flat = decoded_np.reshape(T_out, -1)

        orig_vel = compute_per_frame_vel(orig_flat)
        recon_vel = compute_per_frame_vel(recon_flat)

        print(f"  Orig vel: mean={orig_vel.mean():.5f}, max={orig_vel.max():.5f}")
        print(f"  Recon vel: mean={recon_vel.mean():.5f}, max={recon_vel.max():.5f}")

        # Reconstruction error per frame
        recon_err = np.linalg.norm((decoded_np[:T_out] - original_np[:T_out]).reshape(T_out, -1), axis=1)

        print(f"\n  Frame-by-frame (first 15):")
        print(f"  {'Fr':<4} {'OrigVel':<10} {'ReconVel':<10} {'Ratio':<8} {'ReconErr':<10}")
        for i in range(min(15, len(orig_vel))):
            ratio = recon_vel[i] / (orig_vel[i] + 1e-8)
            print(f"  {i:<4} {orig_vel[i]:<10.5f} {recon_vel[i]:<10.5f} {ratio:<8.3f} {recon_err[i]:<10.6f}")

        # Summary
        print(f"\n  Recon error: first4={recon_err[:4].mean():.6f}, frames4-8={recon_err[4:8].mean():.6f}, rest={recon_err[8:].mean():.6f}")
        if T_out > 10:
            print(f"  Recon vel: first4={recon_vel[:4].mean():.5f}, frames4-8={recon_vel[4:8].mean():.5f}, rest={recon_vel[8:].mean():.5f}")

    # ============ Test with random latent to simulate inference ============
    print(f"\n\n{'='*70}")
    print("=== RANDOM LATENT decode (simulating diffusion output) ===")
    print(f"{'='*70}")

    # Get latents stats from the VAE config
    latents_mean = torch.tensor(vae.config.latents_mean).float() if vae.config.latents_mean else torch.zeros(vae.config.z_dim)
    latents_std = torch.tensor(vae.config.latents_std).float() if vae.config.latents_std else torch.ones(vae.config.z_dim)
    print(f"Latents mean: {latents_mean[:4].tolist()}")
    print(f"Latents std: {latents_std[:4].tolist()}")

    z_dim = vae.config.z_dim
    T_latent = 32  # -> ~128 motion frames
    # 138 dims = 23 joints * 6 (rot6d)
    J_latent = 23

    for trial in range(3):
        torch.manual_seed(42 + trial)
        rand_latent = torch.randn(1, z_dim, T_latent, J_latent).to(device)
        # Scale to real distribution
        rand_latent = rand_latent * latents_std.view(1, -1, 1, 1).to(device) + latents_mean.view(1, -1, 1, 1).to(device)

        with torch.no_grad():
            decoded = vae.decode(rand_latent)

        decoded_np = decoded.cpu().numpy()[0]
        T_out = decoded_np.shape[0]
        vel = compute_per_frame_vel(decoded_np.reshape(T_out, -1))

        print(f"\n  Trial {trial}: T_out={T_out}")
        print(f"    Vel: mean={vel.mean():.5f}, max={vel.max():.5f}")
        first4 = vel[:4].mean()
        rest = vel[4:].mean()
        print(f"    First4={first4:.5f}, Rest={rest:.5f}, Ratio={first4/(rest+1e-8):.3f}")
        print(f"    Frame vels: {[f'{v:.4f}' for v in vel[:12]]}")


if __name__ == '__main__':
    main()
