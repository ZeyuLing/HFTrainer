#!/usr/bin/env python3
"""
Latent Statistics Analysis: 1D (monolithic) vs 2D (joint-factorized) VAE
========================================================================
Computes the following metrics for the PRISM TMM2026 paper:

1. **Per-joint latent distribution** (mean, std) for 2D VAE; per-channel for 1D VAE.
   → Verifies that per-joint KL regularization maps heterogeneous inputs to shared N(0,I).

2. **Flow-matching velocity target magnitudes** ||z_0 - z_1|| per joint (2D) vs per channel (1D).
   → Shows that 2D has balanced velocity targets while 1D has heterogeneous ones.

3. **Condition number** κ = max(||v||) / min(||v||) across joints/channels.
   → Quantifies how "balanced" the learning problem is.

4. **Per-joint input statistics** before and after encoding.
   → Shows the normalization effect of the VAE.

Usage:
    python scripts/analysis/analyze_vae_latent_1d_vs_2d.py \
        --output_dir papers/PRISM_TMM2026/analysis_results \
        --num_samples 500
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)


# ==================== SMPL Joint Names ====================
SMPL_22_JOINT_NAMES = [
    "pelvis",       # 0  (global orient)
    "left_hip",     # 1
    "right_hip",    # 2
    "spine1",       # 3
    "left_knee",    # 4
    "right_knee",   # 5
    "spine2",       # 6
    "left_ankle",   # 7
    "right_ankle",  # 8
    "spine3",       # 9
    "left_foot",    # 10
    "right_foot",   # 11
    "neck",         # 12
    "left_collar",  # 13
    "right_collar", # 14
    "head",         # 15
    "left_shoulder", # 16
    "right_shoulder",# 17
    "left_elbow",   # 18
    "right_elbow",  # 19
    "left_wrist",   # 20
    "right_wrist",  # 21
]

# For the 2D VAE: K=23 tokens = [root_transl, global_orient, 21 body joints]
TOKEN_NAMES_2D = ["root_transl"] + SMPL_22_JOINT_NAMES  # 23 tokens


# ==================== Data Loading ====================
def load_motion_data(data_dir, annotation_path, num_samples=500, clip_len=120, seed=42):
    """Load motion data from MotionHub, convert to both 1D and 2D format."""
    from hftrainer.datasets.motion.motionhub.transforms.load_smplx import (
        process_smplx_pose, process_transl
    )

    rng = np.random.RandomState(seed)

    with open(annotation_path, 'r') as f:
        ann = json.load(f)
    data_list = ann['data_list']
    keys = list(data_list.keys())
    rng.shuffle(keys)

    motions_1d = []  # [B, T, 138]
    motions_2d = []  # [B, T, 23, 6]
    loaded = 0
    skipped = 0

    # Annotation paths are like ../hymotion_data/Academic/20250916/motions/...
    # Resolved relative to annotation dir (data/annotation/), this becomes:
    # data/hymotion_data/Academic/20250916/motions/...
    annotation_dir = os.path.dirname(os.path.abspath(annotation_path))

    for key in keys:
        if loaded >= num_samples:
            break
        item = data_list[key]
        smplx_path = item['smplx_path']

        # Resolve relative path from annotation directory
        abs_path = os.path.normpath(os.path.join(annotation_dir, smplx_path))

        if not os.path.exists(abs_path):
            skipped += 1
            if skipped <= 3:
                print(f"  Warning: file not found: {abs_path}")
            continue

        try:
            data = np.load(abs_path, allow_pickle=True)
            trans = np.asarray(data['trans'], dtype=np.float32)
            poses = np.asarray(data['poses'], dtype=np.float32)
        except Exception:
            continue

        T = trans.shape[0]
        if T < clip_len:
            continue

        # Random crop
        start = rng.randint(0, T - clip_len + 1)
        trans_clip = trans[start:start+clip_len]
        poses_clip = poses[start:start+clip_len]

        # Process pose: smpl_22, rotation_6d → [T, 22*6=132]
        pose_6d = process_smplx_pose(poses_clip, rot_type='rotation_6d', out_type='smpl_22')  # [T, 132]
        # Process translation: abs_rel → [T, 6]
        transl_6d = process_transl(trans_clip, transl_type='abs_rel')  # [T, 6]

        # 1D format: [T, 138] = [transl(6) + pose(132)]
        motion_1d = np.concatenate([transl_6d, pose_6d], axis=-1)  # [T, 138]

        # 2D format: [T, 23, 6]
        # Token 0: root_transl [6] = [abs_x, abs_y, abs_z, delta_x, delta_y, delta_z]
        # Token 1: global_orient [6] = 6D rotation of pelvis
        # Token 2-22: body joints [6] each = 6D rotation
        motion_2d = np.zeros((clip_len, 23, 6), dtype=np.float32)
        motion_2d[:, 0, :] = transl_6d  # root translation token
        pose_6d_reshaped = pose_6d.reshape(clip_len, 22, 6)
        motion_2d[:, 1:, :] = pose_6d_reshaped  # orient + 21 body joints

        motions_1d.append(motion_1d)
        motions_2d.append(motion_2d)
        loaded += 1

    print(f"Loaded {loaded} motion clips of {clip_len} frames each. (skipped {skipped} missing files)")
    motions_1d = np.stack(motions_1d, axis=0)  # [B, T, 138]
    motions_2d = np.stack(motions_2d, axis=0)  # [B, T, 23, 6]
    return motions_1d, motions_2d


# ==================== Model Loading ====================
def load_vae_2d(ckpt_dir, device='cuda'):
    """Load the 2D joint-factorized VAE (HuggingFace format)."""
    from hftrainer.models.motion.prism.autoencoder_kl_2d import AutoencoderKLPrism2DTK
    model = AutoencoderKLPrism2DTK.from_pretrained(ckpt_dir)
    model = model.to(device).eval()
    return model


def load_vae_1d(ckpt_path, device='cuda'):
    """Load the 1D monolithic VAE (mmengine checkpoint format)."""
    from hftrainer.models.motion.prism.autoencoder_kl_1d import AutoencoderKLPrism1D
    model = AutoencoderKLPrism1D(
        base_dim=96,
        in_channels=138,
        out_channels=138,
        z_dim=16,
        is_residual=False,
        num_res_blocks=2,
        temporal_downsample=(False, True, True),
    )
    # Load from mmengine checkpoint (model is nested under 'state_dict' key, with 'vae.' prefix)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt

    # Strip 'vae.' prefix if present (from SmplVAE1DTrainer wrapper)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('vae.'):
            new_state_dict[k[4:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)
    model = model.to(device).eval()
    return model


# ==================== Analysis Functions ====================
@torch.no_grad()
def analyze_latent_distributions(vae_1d, vae_2d, motions_1d, motions_2d, device='cuda', batch_size=32):
    """
    Encode motion through both VAEs and compute latent distribution statistics.

    Returns dict with:
      - latent_mean_1d: [16, T'] per-channel mean
      - latent_std_1d: [16, T'] per-channel std
      - latent_mean_2d: [16, T', 23] per-channel-per-joint mean
      - latent_std_2d: [16, T', 23] per-channel-per-joint std
      - posterior_mean_1d: [B, 16, T'] encoder posterior means
      - posterior_logvar_1d: [B, 16, T'] encoder posterior logvars
      - posterior_mean_2d: [B, 16, T', 23] encoder posterior means
      - posterior_logvar_2d: [B, 16, T', 23] encoder posterior logvars
    """
    B = motions_1d.shape[0]
    all_z_1d = []
    all_mu_1d = []
    all_logvar_1d = []
    all_z_2d = []
    all_mu_2d = []
    all_logvar_2d = []

    for start in range(0, B, batch_size):
        end = min(start + batch_size, B)

        # 1D VAE: input [B, T, 138]
        # encode() returns raw tensor [B, 2*z_dim, T/4] with mean+logvar packed in dim=1
        x_1d = torch.from_numpy(motions_1d[start:end]).to(device)
        enc_1d_raw = vae_1d.encode(x_1d)  # [B, 32, T/4]
        mu_1d, logvar_1d = torch.chunk(enc_1d_raw, 2, dim=1)  # each [B, 16, T/4]
        logvar_1d = torch.clamp(logvar_1d, -30.0, 20.0)
        z_1d = mu_1d + torch.exp(0.5 * logvar_1d) * torch.randn_like(mu_1d)

        all_z_1d.append(z_1d.cpu())
        all_mu_1d.append(mu_1d.cpu())
        all_logvar_1d.append(logvar_1d.cpu())

        # 2D VAE: input [B, T, 23, 6]
        # encode() returns raw tensor [B, 2*z_dim, T/4, 23] with mean+logvar packed in dim=1
        x_2d = torch.from_numpy(motions_2d[start:end]).to(device)
        enc_2d_raw = vae_2d.encode(x_2d)  # [B, 32, T/4, 23]
        mu_2d, logvar_2d = torch.chunk(enc_2d_raw, 2, dim=1)  # each [B, 16, T/4, 23]
        logvar_2d = torch.clamp(logvar_2d, -30.0, 20.0)
        z_2d = mu_2d + torch.exp(0.5 * logvar_2d) * torch.randn_like(mu_2d)

        all_z_2d.append(z_2d.cpu())
        all_mu_2d.append(mu_2d.cpu())
        all_logvar_2d.append(logvar_2d.cpu())

        print(f"  Encoded batch {start//batch_size + 1}/{(B + batch_size - 1)//batch_size}")

    z_1d = torch.cat(all_z_1d, dim=0)            # [B, 16, T']
    mu_1d = torch.cat(all_mu_1d, dim=0)          # [B, 16, T']
    logvar_1d = torch.cat(all_logvar_1d, dim=0)  # [B, 16, T']
    z_2d = torch.cat(all_z_2d, dim=0)            # [B, 16, T', 23]
    mu_2d = torch.cat(all_mu_2d, dim=0)          # [B, 16, T', 23]
    logvar_2d = torch.cat(all_logvar_2d, dim=0)  # [B, 16, T', 23]

    return {
        'z_1d': z_1d.numpy(),
        'z_2d': z_2d.numpy(),
        'mu_1d': mu_1d.numpy(),
        'logvar_1d': logvar_1d.numpy(),
        'mu_2d': mu_2d.numpy(),
        'logvar_2d': logvar_2d.numpy(),
    }


@torch.no_grad()
def compute_velocity_targets(latent_data):
    """
    Compute flow-matching velocity target magnitudes.

    For flow matching with linear interpolation: z_t = (1-t)*z_0 + t*z_1
    The velocity field target is v = z_0 - z_1 (transport from noise z_1 to clean z_0).
    Since z_1 ~ N(0, I), the velocity magnitude ||z_0 - z_1|| ≈ ||z_0|| when z_0 >> 1,
    but more precisely we sample z_1 and compute the actual velocity targets.

    Returns velocity magnitude statistics per joint (2D) or per channel (1D).
    """
    z_1d = latent_data['z_1d']  # [B, 16, T']
    z_2d = latent_data['z_2d']  # [B, 16, T', 23]

    # Sample noise
    rng = np.random.RandomState(42)
    noise_1d = rng.randn(*z_1d.shape).astype(np.float32)
    noise_2d = rng.randn(*z_2d.shape).astype(np.float32)

    # Velocity target: v = z_0 - z_1 (clean latent - noise)
    v_1d = z_1d - noise_1d  # [B, 16, T']
    v_2d = z_2d - noise_2d  # [B, 16, T', 23]

    # 1D: per-channel velocity magnitude (average over B and T')
    # |v_1d|: for each channel c, compute ||v||_2 over c-dim (but 1D has only 1 dim per channel)
    # Actually: compute per-channel L2 norm = |v_1d[:, c, :]| mean
    v_mag_1d_per_channel = np.sqrt(np.mean(v_1d ** 2, axis=(0, 2)))  # [16]

    # 2D: per-joint velocity magnitude (average over B and T')
    # For each joint k, velocity is a 16-dim vector across channels.
    # L2 norm per (sample, time, joint): [B, T', 23]
    v_mag_2d_per_joint_per_sample = np.sqrt(np.sum(v_2d ** 2, axis=1))  # [B, T', 23] sum over C=16
    v_mag_2d_per_joint = np.mean(v_mag_2d_per_joint_per_sample, axis=(0, 1))  # [23]
    v_mag_2d_per_joint_std = np.std(v_mag_2d_per_joint_per_sample, axis=(0, 1))  # [23]

    # 1D: per-channel velocity, each channel is a scalar per (sample, time)
    # L2 norm doesn't make sense per-channel for 1D since it's 1D per channel.
    # Instead, compute per-"virtual-joint" groups for 1D:
    # channels 0-5: root_transl (6 dims), channels 6-11: global_orient (6 dims), ...
    # But 1D latent is 16 channels, not 138. The 1D VAE compresses all 138 dims into 16 latent channels.
    # So we can't do per-joint grouping for 1D latent.
    # Instead, report per-channel statistics.
    v_mag_1d_per_channel_per_sample = np.abs(v_1d)  # [B, 16, T']
    v_mag_1d_per_channel_mean = np.mean(v_mag_1d_per_channel_per_sample, axis=(0, 2))  # [16]
    v_mag_1d_per_channel_std = np.std(v_mag_1d_per_channel_per_sample, axis=(0, 2))  # [16]

    # Condition number: max/min velocity magnitude ratio
    # For 2D: ratio of joint-level velocity magnitudes
    kappa_2d = np.max(v_mag_2d_per_joint) / (np.min(v_mag_2d_per_joint) + 1e-8)
    # For 1D: ratio of channel-level velocity magnitudes
    kappa_1d = np.max(v_mag_1d_per_channel_mean) / (np.min(v_mag_1d_per_channel_mean) + 1e-8)

    return {
        'v_mag_2d_per_joint': v_mag_2d_per_joint,           # [23]
        'v_mag_2d_per_joint_std': v_mag_2d_per_joint_std,   # [23]
        'v_mag_1d_per_channel': v_mag_1d_per_channel_mean,  # [16]
        'v_mag_1d_per_channel_std': v_mag_1d_per_channel_std, # [16]
        'kappa_2d': kappa_2d,
        'kappa_1d': kappa_1d,
    }


def compute_input_statistics(motions_1d, motions_2d):
    """
    Compute per-joint and per-channel input statistics BEFORE encoding.
    Shows the heterogeneity of the raw input space.
    """
    # 1D: [B, T, 138] — compute per-dim statistics
    B, T, D = motions_1d.shape

    # Per-dim mean and std (over all samples and frames)
    input_mean_1d = np.mean(motions_1d, axis=(0, 1))  # [138]
    input_std_1d = np.std(motions_1d, axis=(0, 1))    # [138]

    # Group by joint for meaningful statistics:
    # dims 0-5: root_transl, dims 6-11: global_orient, dims 12-17: joint1, ...
    input_std_per_joint_1d = np.zeros(23)
    for j in range(23):
        joint_data = motions_1d[:, :, j*6:(j+1)*6]  # [B, T, 6]
        input_std_per_joint_1d[j] = np.std(joint_data)

    # 2D: [B, T, 23, 6] — per-joint statistics
    input_mean_2d = np.mean(motions_2d, axis=(0, 1))  # [23, 6]
    input_std_2d = np.std(motions_2d, axis=(0, 1))    # [23, 6]
    input_std_per_joint_2d = np.std(motions_2d.reshape(-1, 23, 6), axis=(0, 2))  # [23]

    # Range per joint (max - min over entire dataset)
    input_range_per_joint = np.zeros(23)
    for j in range(23):
        joint_data = motions_2d[:, :, j, :]
        input_range_per_joint[j] = np.max(joint_data) - np.min(joint_data)

    return {
        'input_mean_1d': input_mean_1d,
        'input_std_1d': input_std_1d,
        'input_std_per_joint_1d': input_std_per_joint_1d,
        'input_mean_2d': input_mean_2d,
        'input_std_2d': input_std_2d,
        'input_std_per_joint_2d': input_std_per_joint_2d,
        'input_range_per_joint': input_range_per_joint,
    }


def compute_posterior_statistics(latent_data):
    """
    Analyze the encoder posterior: how close is each joint's posterior to N(0,1)?

    For 2D VAE: per-joint KL = 0.5 * (mu^2 + exp(logvar) - logvar - 1)
    For 1D VAE: per-channel KL
    """
    mu_1d = latent_data['mu_1d']          # [B, 16, T']
    logvar_1d = latent_data['logvar_1d']  # [B, 16, T']
    mu_2d = latent_data['mu_2d']          # [B, 16, T', 23]
    logvar_2d = latent_data['logvar_2d']  # [B, 16, T', 23]

    # Per-channel KL for 1D
    kl_1d_per_channel = 0.5 * (mu_1d**2 + np.exp(logvar_1d) - logvar_1d - 1)  # [B, 16, T']
    kl_1d_per_channel_mean = np.mean(kl_1d_per_channel, axis=(0, 2))  # [16]

    # Per-joint KL for 2D (averaged over channels and time)
    kl_2d_per_joint = 0.5 * (mu_2d**2 + np.exp(logvar_2d) - logvar_2d - 1)  # [B, 16, T', 23]
    kl_2d_per_joint_mean = np.mean(kl_2d_per_joint, axis=(0, 1, 2))  # [23] (avg over B, C, T')

    # Per-joint posterior statistics (how close to N(0,1))
    posterior_mean_per_joint_2d = np.mean(mu_2d, axis=(0, 1, 2))  # [23] should be ~0
    posterior_std_per_joint_2d = np.mean(np.exp(0.5 * logvar_2d), axis=(0, 1, 2))  # [23] should be ~1

    # Per-channel posterior statistics for 1D
    posterior_mean_per_channel_1d = np.mean(mu_1d, axis=(0, 2))  # [16] should be ~0
    posterior_std_per_channel_1d = np.mean(np.exp(0.5 * logvar_1d), axis=(0, 2))  # [16] should be ~1

    # Latent sample statistics (z = mu + eps * sigma)
    z_1d = latent_data['z_1d']  # [B, 16, T']
    z_2d = latent_data['z_2d']  # [B, 16, T', 23]

    z_mean_per_channel_1d = np.mean(z_1d, axis=(0, 2))  # [16]
    z_std_per_channel_1d = np.std(z_1d, axis=(0, 2))    # [16]
    z_mean_per_joint_2d = np.mean(z_2d, axis=(0, 1, 2))  # [23]
    z_std_per_joint_2d = np.std(z_2d, axis=(0, 1, 2))    # [23]

    return {
        'kl_1d_per_channel': kl_1d_per_channel_mean,
        'kl_2d_per_joint': kl_2d_per_joint_mean,
        'posterior_mean_per_joint_2d': posterior_mean_per_joint_2d,
        'posterior_std_per_joint_2d': posterior_std_per_joint_2d,
        'posterior_mean_per_channel_1d': posterior_mean_per_channel_1d,
        'posterior_std_per_channel_1d': posterior_std_per_channel_1d,
        'z_mean_per_channel_1d': z_mean_per_channel_1d,
        'z_std_per_channel_1d': z_std_per_channel_1d,
        'z_mean_per_joint_2d': z_mean_per_joint_2d,
        'z_std_per_joint_2d': z_std_per_joint_2d,
    }


# ==================== Visualization ====================
def generate_plots(input_stats, latent_data, posterior_stats, velocity_stats, output_dir):
    """Generate all visualization plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    os.makedirs(output_dir, exist_ok=True)

    # Color scheme
    C_1D = '#e74c3c'  # red for 1D (monolithic)
    C_2D = '#2ecc71'  # green for 2D (joint-factorized)

    # ========== Figure 1: Input space heterogeneity ==========
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Per-joint input std
    ax = axes[0]
    bars = ax.bar(range(23), input_stats['input_std_per_joint_2d'],
                  color=[C_1D if i == 0 else '#3498db' for i in range(23)],
                  alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(23))
    ax.set_xticklabels(TOKEN_NAMES_2D, rotation=90, fontsize=7)
    ax.set_ylabel('Standard Deviation', fontsize=11)
    ax.set_title('(a) Input Space: Per-Joint Std (before encoding)', fontsize=12)
    ax.axhline(y=input_stats['input_std_per_joint_2d'][1:].mean(), color='gray',
               linestyle='--', alpha=0.5, label='Mean (joints only)')
    ax.legend(fontsize=9)
    # Highlight root is different
    ax.annotate('Translation\n(meters)', xy=(0, input_stats['input_std_per_joint_2d'][0]),
                xytext=(2, input_stats['input_std_per_joint_2d'][0] * 1.2),
                arrowprops=dict(arrowstyle='->', color=C_1D),
                fontsize=9, color=C_1D, ha='center')

    # Per-joint input range
    ax = axes[1]
    bars = ax.bar(range(23), input_stats['input_range_per_joint'],
                  color=[C_1D if i == 0 else '#3498db' for i in range(23)],
                  alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(23))
    ax.set_xticklabels(TOKEN_NAMES_2D, rotation=90, fontsize=7)
    ax.set_ylabel('Value Range (max - min)', fontsize=11)
    ax.set_title('(b) Input Space: Per-Joint Value Range', fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_input_heterogeneity.pdf'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'fig_input_heterogeneity.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved fig_input_heterogeneity.pdf/png")

    # ========== Figure 2: Latent space normalization (2D VAE per-joint) ==========
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Per-joint latent mean
    ax = axes[0]
    ax.bar(range(23), posterior_stats['z_mean_per_joint_2d'],
           color=C_2D, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xticks(range(23))
    ax.set_xticklabels(TOKEN_NAMES_2D, rotation=90, fontsize=7)
    ax.set_ylabel('Mean', fontsize=11)
    ax.set_title('(a) 2D VAE: Per-Joint Latent Mean (target: 0)', fontsize=12)
    ax.set_ylim(-0.5, 0.5)

    # Per-joint latent std
    ax = axes[1]
    ax.bar(range(23), posterior_stats['z_std_per_joint_2d'],
           color=C_2D, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Target: 1.0')
    ax.set_xticks(range(23))
    ax.set_xticklabels(TOKEN_NAMES_2D, rotation=90, fontsize=7)
    ax.set_ylabel('Standard Deviation', fontsize=11)
    ax.set_title('(b) 2D VAE: Per-Joint Latent Std (target: 1)', fontsize=12)
    ax.legend(fontsize=9)

    # Per-joint KL divergence
    ax = axes[2]
    ax.bar(range(23), posterior_stats['kl_2d_per_joint'],
           color=C_2D, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(23))
    ax.set_xticklabels(TOKEN_NAMES_2D, rotation=90, fontsize=7)
    ax.set_ylabel('KL Divergence', fontsize=11)
    ax.set_title('(c) 2D VAE: Per-Joint KL(posterior || N(0,I))', fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_2d_latent_normalization.pdf'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'fig_2d_latent_normalization.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved fig_2d_latent_normalization.pdf/png")

    # ========== Figure 3: 1D vs 2D latent statistics comparison ==========
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1D per-channel std
    ax = axes[0]
    x = range(16)
    ax.bar(x, posterior_stats['z_std_per_channel_1d'],
           color=C_1D, alpha=0.8, edgecolor='black', linewidth=0.5, label='1D (monolithic)')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Target: 1.0')
    ax.set_xticks(x)
    ax.set_xticklabels([f'ch{i}' for i in range(16)], fontsize=8)
    ax.set_ylabel('Standard Deviation', fontsize=11)
    ax.set_title('(a) 1D VAE: Per-Channel Latent Std', fontsize=12)
    ax.legend(fontsize=9)

    # 2D per-joint std (same as above but for comparison)
    ax = axes[1]
    ax.bar(range(23), posterior_stats['z_std_per_joint_2d'],
           color=C_2D, alpha=0.8, edgecolor='black', linewidth=0.5, label='2D (joint-factorized)')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Target: 1.0')
    ax.set_xticks(range(23))
    ax.set_xticklabels(TOKEN_NAMES_2D, rotation=90, fontsize=7)
    ax.set_ylabel('Standard Deviation', fontsize=11)
    ax.set_title('(b) 2D VAE: Per-Joint Latent Std', fontsize=12)
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_1d_vs_2d_latent_std.pdf'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'fig_1d_vs_2d_latent_std.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved fig_1d_vs_2d_latent_std.pdf/png")

    # ========== Figure 4: Velocity target magnitudes ==========
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 2D per-joint velocity magnitude
    ax = axes[0]
    ax.bar(range(23), velocity_stats['v_mag_2d_per_joint'],
           yerr=velocity_stats['v_mag_2d_per_joint_std'],
           color=C_2D, alpha=0.8, edgecolor='black', linewidth=0.5,
           capsize=2, label='2D VAE')
    ax.set_xticks(range(23))
    ax.set_xticklabels(TOKEN_NAMES_2D, rotation=90, fontsize=7)
    ax.set_ylabel('Velocity Magnitude ||v||', fontsize=11)
    ax.set_title(f'(a) 2D: Per-Joint Velocity Target ||\u03b5 - z₀|| (κ = {velocity_stats["kappa_2d"]:.2f})',
                 fontsize=12)
    ax.legend(fontsize=9)

    # 1D per-channel velocity magnitude
    ax = axes[1]
    ax.bar(range(16), velocity_stats['v_mag_1d_per_channel'],
           yerr=velocity_stats['v_mag_1d_per_channel_std'],
           color=C_1D, alpha=0.8, edgecolor='black', linewidth=0.5,
           capsize=2, label='1D VAE')
    ax.set_xticks(range(16))
    ax.set_xticklabels([f'ch{i}' for i in range(16)], fontsize=8)
    ax.set_ylabel('Velocity Magnitude |v|', fontsize=11)
    ax.set_title(f'(b) 1D: Per-Channel Velocity Target |ε - z₀| (κ = {velocity_stats["kappa_1d"]:.2f})',
                 fontsize=12)
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_velocity_targets.pdf'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'fig_velocity_targets.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved fig_velocity_targets.pdf/png")

    # ========== Figure 5: Key comparison bar chart (paper-ready) ==========
    fig, ax = plt.subplots(figsize=(8, 5))

    # Condition number comparison
    bars = ax.bar(['1D Monolithic\n(κ = {:.2f})'.format(velocity_stats['kappa_1d']),
                   '2D Joint-Factorized\n(κ = {:.2f})'.format(velocity_stats['kappa_2d'])],
                  [velocity_stats['kappa_1d'], velocity_stats['kappa_2d']],
                  color=[C_1D, C_2D], alpha=0.8, edgecolor='black', linewidth=1)
    ax.set_ylabel('Condition Number κ = max(||v||) / min(||v||)', fontsize=12)
    ax.set_title('Velocity Target Balance: Lower κ → More Balanced Learning', fontsize=13)
    # Add value annotations
    for bar, val in zip(bars, [velocity_stats['kappa_1d'], velocity_stats['kappa_2d']]):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                f'{val:.2f}', ha='center', va='bottom', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_condition_number.pdf'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'fig_condition_number.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved fig_condition_number.pdf/png")

    # ========== Figure 6: Combined normalization effect (input → latent) ==========
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Top: input std per joint
    ax = axes[0]
    ax.bar(range(23), input_stats['input_std_per_joint_2d'],
           color='#95a5a6', alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(23))
    ax.set_xticklabels(TOKEN_NAMES_2D, rotation=45, fontsize=8, ha='right')
    ax.set_ylabel('Std', fontsize=11)
    ax.set_title('Input Space: Per-Joint Std (heterogeneous)', fontsize=12)
    # Show coefficient of variation
    cv_input = np.std(input_stats['input_std_per_joint_2d']) / np.mean(input_stats['input_std_per_joint_2d'])
    ax.text(0.95, 0.9, f'CV = {cv_input:.3f}', transform=ax.transAxes,
            fontsize=12, ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.5))

    # Bottom: latent std per joint (from 2D VAE)
    ax = axes[1]
    ax.bar(range(23), posterior_stats['z_std_per_joint_2d'],
           color=C_2D, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Target: 1.0')
    ax.set_xticks(range(23))
    ax.set_xticklabels(TOKEN_NAMES_2D, rotation=45, fontsize=8, ha='right')
    ax.set_ylabel('Std', fontsize=11)
    ax.set_title('Latent Space (2D VAE): Per-Joint Std (normalized → ≈1)', fontsize=12)
    cv_latent = np.std(posterior_stats['z_std_per_joint_2d']) / np.mean(posterior_stats['z_std_per_joint_2d'])
    ax.text(0.95, 0.9, f'CV = {cv_latent:.3f}', transform=ax.transAxes,
            fontsize=12, ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.5))
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_normalization_effect.pdf'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'fig_normalization_effect.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved fig_normalization_effect.pdf/png")


def print_summary(input_stats, posterior_stats, velocity_stats):
    """Print a text summary of all results."""
    print("\n" + "="*80)
    print("LATENT STATISTICS ANALYSIS: 1D (Monolithic) vs 2D (Joint-Factorized) VAE")
    print("="*80)

    print("\n--- Input Space Heterogeneity ---")
    print(f"Per-joint input std (23 joints):")
    for i, name in enumerate(TOKEN_NAMES_2D):
        print(f"  {name:>20s}: std={input_stats['input_std_per_joint_2d'][i]:.4f}  "
              f"range={input_stats['input_range_per_joint'][i]:.4f}")
    std_cv = np.std(input_stats['input_std_per_joint_2d']) / np.mean(input_stats['input_std_per_joint_2d'])
    print(f"  Coefficient of variation of stds: {std_cv:.4f}")
    print(f"  Root transl std / Joint rotation std ratio: "
          f"{input_stats['input_std_per_joint_2d'][0] / np.mean(input_stats['input_std_per_joint_2d'][1:]):.2f}x")

    print("\n--- 2D VAE: Per-Joint Latent Statistics (should be ≈ N(0,1)) ---")
    for i, name in enumerate(TOKEN_NAMES_2D):
        print(f"  {name:>20s}: mean={posterior_stats['z_mean_per_joint_2d'][i]:+.4f}  "
              f"std={posterior_stats['z_std_per_joint_2d'][i]:.4f}  "
              f"KL={posterior_stats['kl_2d_per_joint'][i]:.4f}")
    cv_latent = np.std(posterior_stats['z_std_per_joint_2d']) / np.mean(posterior_stats['z_std_per_joint_2d'])
    print(f"  Coefficient of variation of latent stds: {cv_latent:.4f}")

    print("\n--- 1D VAE: Per-Channel Latent Statistics ---")
    for i in range(16):
        print(f"  ch{i:2d}: mean={posterior_stats['z_mean_per_channel_1d'][i]:+.4f}  "
              f"std={posterior_stats['z_std_per_channel_1d'][i]:.4f}  "
              f"KL={posterior_stats['kl_1d_per_channel'][i]:.4f}")
    cv_1d = np.std(posterior_stats['z_std_per_channel_1d']) / np.mean(posterior_stats['z_std_per_channel_1d'])
    print(f"  Coefficient of variation of latent stds: {cv_1d:.4f}")

    print("\n--- Velocity Target Magnitudes ---")
    print(f"2D VAE (per-joint): {velocity_stats['v_mag_2d_per_joint']}")
    print(f"  Condition number κ_2D = {velocity_stats['kappa_2d']:.4f}")
    print(f"1D VAE (per-channel): {velocity_stats['v_mag_1d_per_channel']}")
    print(f"  Condition number κ_1D = {velocity_stats['kappa_1d']:.4f}")
    print(f"  κ_1D / κ_2D = {velocity_stats['kappa_1d'] / velocity_stats['kappa_2d']:.2f}x")

    print("\n--- Key Finding ---")
    print(f"Input space CV: {std_cv:.4f} (heterogeneous)")
    print(f"2D latent CV:   {cv_latent:.4f} (normalized)")
    print(f"1D latent CV:   {cv_1d:.4f}")
    print(f"Velocity κ ratio (1D/2D): {velocity_stats['kappa_1d'] / velocity_stats['kappa_2d']:.2f}x")
    print("="*80)


# ==================== Main ====================
def main():
    parser = argparse.ArgumentParser(description='Analyze 1D vs 2D VAE latent statistics')
    parser.add_argument('--output_dir', type=str,
                        default='papers/PRISM_TMM2026/analysis_results',
                        help='Output directory for plots and data')
    parser.add_argument('--num_samples', type=int, default=500,
                        help='Number of motion clips to analyze')
    parser.add_argument('--clip_len', type=int, default=120,
                        help='Clip length in frames')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for VAE encoding')
    parser.add_argument('--vae_2d_ckpt', type=str,
                        default='checkpoints/vermo_vae',
                        help='Path to 2D VAE checkpoint (HF format)')
    parser.add_argument('--vae_1d_ckpt', type=str,
                        default='../versatilemotion/work_dirs/smpl_vae1d_nostatic_aug_hq/iter_13000.pth',
                        help='Path to 1D VAE checkpoint (mmengine .pth)')
    parser.add_argument('--data_dir', type=str,
                        default='data/motionhub',
                        help='Motion data directory')
    parser.add_argument('--annotation', type=str,
                        default='data/annotation/train_hymotion_400h.json',
                        help='Annotation file')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: Load motion data
    print("="*60)
    print("Step 1: Loading motion data...")
    print("="*60)
    motions_1d, motions_2d = load_motion_data(
        args.data_dir, args.annotation,
        num_samples=args.num_samples,
        clip_len=args.clip_len,
    )
    print(f"  1D shape: {motions_1d.shape}")
    print(f"  2D shape: {motions_2d.shape}")

    # Step 2: Compute input statistics
    print("\n" + "="*60)
    print("Step 2: Computing input statistics...")
    print("="*60)
    input_stats = compute_input_statistics(motions_1d, motions_2d)

    # Step 3: Load VAE models and encode
    print("\n" + "="*60)
    print("Step 3: Loading VAE models and encoding...")
    print("="*60)
    print("  Loading 2D VAE...")
    vae_2d = load_vae_2d(args.vae_2d_ckpt, device=args.device)
    print("  Loading 1D VAE...")
    vae_1d = load_vae_1d(args.vae_1d_ckpt, device=args.device)
    print("  Encoding through both VAEs...")
    latent_data = analyze_latent_distributions(
        vae_1d, vae_2d, motions_1d, motions_2d,
        device=args.device, batch_size=args.batch_size,
    )

    # Step 4: Compute posterior statistics
    print("\n" + "="*60)
    print("Step 4: Computing posterior statistics...")
    print("="*60)
    posterior_stats = compute_posterior_statistics(latent_data)

    # Step 5: Compute velocity targets
    print("\n" + "="*60)
    print("Step 5: Computing velocity target magnitudes...")
    print("="*60)
    velocity_stats = compute_velocity_targets(latent_data)

    # Step 6: Print summary
    print_summary(input_stats, posterior_stats, velocity_stats)

    # Step 7: Generate plots
    print("\n" + "="*60)
    print("Step 7: Generating plots...")
    print("="*60)
    generate_plots(input_stats, latent_data, posterior_stats, velocity_stats, args.output_dir)

    # Step 8: Save raw data
    print("\n" + "="*60)
    print("Step 8: Saving raw data...")
    print("="*60)
    np.savez(os.path.join(args.output_dir, 'analysis_data.npz'),
             # Input stats
             input_std_per_joint=input_stats['input_std_per_joint_2d'],
             input_range_per_joint=input_stats['input_range_per_joint'],
             # Posterior stats
             z_mean_per_joint_2d=posterior_stats['z_mean_per_joint_2d'],
             z_std_per_joint_2d=posterior_stats['z_std_per_joint_2d'],
             z_mean_per_channel_1d=posterior_stats['z_mean_per_channel_1d'],
             z_std_per_channel_1d=posterior_stats['z_std_per_channel_1d'],
             kl_2d_per_joint=posterior_stats['kl_2d_per_joint'],
             kl_1d_per_channel=posterior_stats['kl_1d_per_channel'],
             # Velocity stats
             v_mag_2d_per_joint=velocity_stats['v_mag_2d_per_joint'],
             v_mag_2d_per_joint_std=velocity_stats['v_mag_2d_per_joint_std'],
             v_mag_1d_per_channel=velocity_stats['v_mag_1d_per_channel'],
             v_mag_1d_per_channel_std=velocity_stats['v_mag_1d_per_channel_std'],
             kappa_2d=velocity_stats['kappa_2d'],
             kappa_1d=velocity_stats['kappa_1d'],
             # Token names
             token_names=np.array(TOKEN_NAMES_2D),
    )
    print(f"  Saved analysis_data.npz to {args.output_dir}")

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()
