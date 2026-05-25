#!/usr/bin/env python3
"""Diagnose latent distribution difference between translation token (J=0) and body joints (J=1:22).

Key question: Does the transformer face a fundamentally harder task for translation
because its latent distribution differs from body joints, yet shares the same
flow matching noise schedule and normalization?

Tests:
1. Encode many GT motions through VAE
2. Compare latent statistics for token 0 (translation) vs tokens 1-22 (body joints)
3. Check if per-channel normalization equalizes both or leaves a gap
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

    # ==================== Build model ====================
    print('[1] Building PRISM bundle (VAE only)...')
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

    # VAE latent normalization constants
    latents_mean = torch.tensor(
        bundle.vae.config.latents_mean, dtype=torch.float32, device=device
    ).view(1, bundle.vae.config.z_dim, 1, 1)
    latents_std = torch.tensor(
        bundle.vae.config.latents_std, dtype=torch.float32, device=device
    ).view(1, bundle.vae.config.z_dim, 1, 1)

    vae_temporal = bundle.vae.config.scale_factor_temporal
    z_dim = bundle.vae.config.z_dim
    print(f'    VAE z_dim: {z_dim}, temporal: {vae_temporal}x')
    print(f'    Latent normalization constants:')
    print(f'      mean: {latents_mean.flatten().cpu().numpy()}')
    print(f'      std:  {latents_std.flatten().cpu().numpy()}')

    # Load test samples
    meta = json.loads(Path('data/annotation/test_hml3d.json').read_text())
    data_list = meta['data_list']

    # Get first N samples with valid paths
    N_SAMPLES = 50
    valid_samples = []
    for name, entry in data_list.items():
        gt_path = entry.get('smplx_path', '')
        if not gt_path:
            continue
        gt_full = os.path.join('data/motionhub', gt_path)
        if not os.path.exists(gt_full):
            continue
        num_frames = int(entry.get('num_frames', 0))
        if num_frames < 24:
            continue
        valid_samples.append({
            'name': name,
            'num_frames': min(num_frames, 196),
            'path': gt_full,
        })
        if len(valid_samples) >= N_SAMPLES:
            break

    print(f'\n[2] Encoding {len(valid_samples)} GT motions through VAE...')

    # Collect all latents
    all_latents_raw = []       # Before per-channel normalization
    all_latents_normed = []    # After per-channel normalization

    for i, sample in enumerate(valid_samples):
        if (i + 1) % 10 == 0:
            print(f'    [{i+1}/{len(valid_samples)}]')

        gt_data = dict(np.load(sample['path'], allow_pickle=True))
        motion_vec = bundle.smpl_pose_processor.smplx_dict_to_motion_vector(gt_data)

        T_use = min(motion_vec.shape[0], sample['num_frames'], 196)
        motion_vec = motion_vec[:T_use]

        with torch.no_grad():
            motion_input = motion_vec.unsqueeze(0).to(device)  # (1, T, 138)
            # Use the bundle's encode_motion which does normalize + VAE encode + latent norm
            latents_normed = bundle.encode_motion(motion_input)  # (1, Z, T_latent, 23)

            # Also get raw latents BEFORE normalization
            # Reverse the normalization to get raw
            latents_raw = latents_normed * latents_std + latents_mean

            all_latents_raw.append(latents_raw.cpu())
            all_latents_normed.append(latents_normed.cpu())

    print(f'\n[3] Analyzing latent distributions...')
    print('=' * 90)

    # Concatenate all latents: each is (1, Z=16, T_latent, J=23)
    # We want to compare J=0 (translation) vs J=1:23 (body joints)

    # Collect per-token stats
    transl_raw_vals = []      # All raw latent values for token 0
    body_raw_vals = []        # All raw latent values for tokens 1-22
    transl_normed_vals = []   # All normed latent values for token 0
    body_normed_vals = []     # All normed latent values for tokens 1-22

    # Also per-channel stats
    transl_per_ch_raw = [[] for _ in range(z_dim)]
    body_per_ch_raw = [[] for _ in range(z_dim)]
    transl_per_ch_normed = [[] for _ in range(z_dim)]
    body_per_ch_normed = [[] for _ in range(z_dim)]

    for lr, ln in zip(all_latents_raw, all_latents_normed):
        # lr shape: (1, Z, T_latent, 23)
        lr = lr.squeeze(0)  # (Z, T_latent, 23)
        ln = ln.squeeze(0)

        # Token 0 = translation
        transl_r = lr[:, :, 0]   # (Z, T_latent)
        body_r = lr[:, :, 1:]    # (Z, T_latent, 22)
        transl_n = ln[:, :, 0]
        body_n = ln[:, :, 1:]

        transl_raw_vals.append(transl_r.flatten())
        body_raw_vals.append(body_r.flatten())
        transl_normed_vals.append(transl_n.flatten())
        body_normed_vals.append(body_n.flatten())

        for c in range(z_dim):
            transl_per_ch_raw[c].append(transl_r[c].flatten())
            body_per_ch_raw[c].append(body_r[c].flatten())
            transl_per_ch_normed[c].append(transl_n[c].flatten())
            body_per_ch_normed[c].append(body_n[c].flatten())

    # Aggregate
    transl_raw_all = torch.cat(transl_raw_vals)
    body_raw_all = torch.cat(body_raw_vals)
    transl_normed_all = torch.cat(transl_normed_vals)
    body_normed_all = torch.cat(body_normed_vals)

    print(f'\n  === OVERALL STATISTICS ===')
    print(f'  Total latent elements: transl={len(transl_raw_all)}, body={len(body_raw_all)}')
    print(f'\n  --- RAW latents (before per-channel normalization) ---')
    print(f'  Translation (token 0):')
    print(f'    mean={transl_raw_all.mean():.6f}, std={transl_raw_all.std():.6f}')
    print(f'    min={transl_raw_all.min():.4f}, max={transl_raw_all.max():.4f}')
    print(f'    abs_mean={transl_raw_all.abs().mean():.6f}')
    print(f'  Body joints (tokens 1-22):')
    print(f'    mean={body_raw_all.mean():.6f}, std={body_raw_all.std():.6f}')
    print(f'    min={body_raw_all.min():.4f}, max={body_raw_all.max():.4f}')
    print(f'    abs_mean={body_raw_all.abs().mean():.6f}')
    print(f'  RATIO (body_std / transl_std): {body_raw_all.std() / transl_raw_all.std():.4f}')

    print(f'\n  --- NORMALIZED latents (after per-channel norm, what transformer sees) ---')
    print(f'  Translation (token 0):')
    print(f'    mean={transl_normed_all.mean():.6f}, std={transl_normed_all.std():.6f}')
    print(f'    min={transl_normed_all.min():.4f}, max={transl_normed_all.max():.4f}')
    print(f'    abs_mean={transl_normed_all.abs().mean():.6f}')
    print(f'  Body joints (tokens 1-22):')
    print(f'    mean={body_normed_all.mean():.6f}, std={body_normed_all.std():.6f}')
    print(f'    min={body_normed_all.min():.4f}, max={body_normed_all.max():.4f}')
    print(f'    abs_mean={body_normed_all.abs().mean():.6f}')
    print(f'  RATIO (body_std / transl_std): {body_normed_all.std() / transl_normed_all.std():.4f}')

    print(f'\n  === PER-CHANNEL ANALYSIS (what matters for flow matching) ===')
    print(f'  {"Ch":>3} | {"Transl_mean":>12} {"Transl_std":>12} | {"Body_mean":>12} {"Body_std":>12} | {"Std_ratio":>10} {"Mean_diff":>10}')
    print(f'  {"-"*3}-+-{"-"*12}-{"-"*12}-+-{"-"*12}-{"-"*12}-+-{"-"*10}-{"-"*10}')

    ch_std_ratios = []
    ch_mean_diffs = []

    for c in range(z_dim):
        t_vals = torch.cat(transl_per_ch_normed[c])
        b_vals = torch.cat(body_per_ch_normed[c])

        t_mean, t_std = t_vals.mean().item(), t_vals.std().item()
        b_mean, b_std = b_vals.mean().item(), b_vals.std().item()

        ratio = b_std / (t_std + 1e-8)
        mean_diff = abs(t_mean - b_mean)
        ch_std_ratios.append(ratio)
        ch_mean_diffs.append(mean_diff)

        flag = ' ***' if abs(ratio - 1.0) > 0.3 or mean_diff > 0.3 else ''
        print(f'  {c:3d} | {t_mean:12.6f} {t_std:12.6f} | {b_mean:12.6f} {b_std:12.6f} | {ratio:10.4f} {mean_diff:10.4f}{flag}')

    print(f'\n  Summary of per-channel std ratios (body/transl):')
    ch_std_ratios = np.array(ch_std_ratios)
    print(f'    Mean ratio: {ch_std_ratios.mean():.4f}')
    print(f'    Min ratio:  {ch_std_ratios.min():.4f} (ch {ch_std_ratios.argmin()})')
    print(f'    Max ratio:  {ch_std_ratios.max():.4f} (ch {ch_std_ratios.argmax()})')
    print(f'    Channels with ratio < 0.7 or > 1.5: {np.sum((ch_std_ratios < 0.7) | (ch_std_ratios > 1.5))}')

    # Check percentile distributions
    print(f'\n  === PERCENTILE COMPARISON (normalized latents) ===')
    for pct in [1, 5, 25, 50, 75, 95, 99]:
        t_pct = torch.quantile(transl_normed_all, pct / 100).item()
        b_pct = torch.quantile(body_normed_all, pct / 100).item()
        print(f'    P{pct:02d}: transl={t_pct:8.4f}, body={b_pct:8.4f}, diff={t_pct - b_pct:8.4f}')

    # Check: for stationary motions, what happens to translation latents?
    print(f'\n  === STATIONARY vs LOCOMOTION COMPARISON ===')
    # Identify motions by displacement
    displacements = []
    for i, sample in enumerate(valid_samples):
        lr = all_latents_raw[i].squeeze(0)  # (Z, T_latent, 23)
        # Quick proxy: use raw latents of translation token, channel with most variance
        gt_data = dict(np.load(sample['path'], allow_pickle=True))
        transl = gt_data.get('transl', gt_data.get('trans', np.zeros((2, 3))))
        if len(transl) > 1:
            disp = np.linalg.norm(transl[-1] - transl[0])
        else:
            disp = 0.0
        displacements.append(disp)

    displacements = np.array(displacements)
    stationary_mask = displacements < 0.1  # < 10cm total displacement
    locomotion_mask = displacements > 0.5  # > 50cm total displacement

    n_stat = stationary_mask.sum()
    n_loco = locomotion_mask.sum()
    print(f'  Stationary (disp < 0.1m): {n_stat} samples')
    print(f'  Locomotion (disp > 0.5m): {n_loco} samples')

    if n_stat > 0:
        stat_transl = torch.cat([all_latents_normed[i].squeeze(0)[:, :, 0].flatten()
                                 for i in range(len(valid_samples)) if stationary_mask[i]])
        stat_body = torch.cat([all_latents_normed[i].squeeze(0)[:, :, 1:].flatten()
                               for i in range(len(valid_samples)) if stationary_mask[i]])
        print(f'\n  Stationary motions:')
        print(f'    Transl latent: mean={stat_transl.mean():.6f}, std={stat_transl.std():.6f}, '
              f'abs_mean={stat_transl.abs().mean():.6f}')
        print(f'    Body latent:   mean={stat_body.mean():.6f}, std={stat_body.std():.6f}, '
              f'abs_mean={stat_body.abs().mean():.6f}')

    if n_loco > 0:
        loco_transl = torch.cat([all_latents_normed[i].squeeze(0)[:, :, 0].flatten()
                                 for i in range(len(valid_samples)) if locomotion_mask[i]])
        loco_body = torch.cat([all_latents_normed[i].squeeze(0)[:, :, 1:].flatten()
                               for i in range(len(valid_samples)) if locomotion_mask[i]])
        print(f'\n  Locomotion motions:')
        print(f'    Transl latent: mean={loco_transl.mean():.6f}, std={loco_transl.std():.6f}, '
              f'abs_mean={loco_transl.abs().mean():.6f}')
        print(f'    Body latent:   mean={loco_body.mean():.6f}, std={loco_body.std():.6f}, '
              f'abs_mean={loco_body.abs().mean():.6f}')

    # Final: Check what the model_pred noise targets look like for translation vs body
    # In flow matching with shift=5.0, the target is: (noise - x0) / sigma
    # If translation latents have very different magnitude from body latents,
    # the target signal-to-noise ratio differs per token.
    print(f'\n  === FLOW MATCHING IMPLICATIONS ===')
    t_std = transl_normed_all.std().item()
    b_std = body_normed_all.std().item()
    print(f'  Translation latent std: {t_std:.6f}')
    print(f'  Body joint latent std:  {b_std:.6f}')
    print(f'  Ratio: {t_std / b_std:.4f}')
    print(f'')
    print(f'  In flow matching, the velocity target v = (x1 - x0) where x0=data, x1=noise~N(0,1)')
    print(f'  If transl_std differs from body_std:')
    print(f'    - Equal noise (unit Gaussian) applied to both')
    print(f'    - If transl_std << body_std: noise dominates translation signal early')
    print(f'    - If transl_std >> body_std: translation signal persists longer')
    print(f'')
    print(f'  Current situation: transl_std/body_std = {t_std/b_std:.4f}')
    if abs(t_std / b_std - 1.0) > 0.3:
        print(f'  ⚠️  SIGNIFICANT DISTRIBUTION MISMATCH detected!')
        print(f'      This means flow matching noise schedule is suboptimal for one of the token types.')
    else:
        print(f'  ✓ Distribution roughly matched — noise schedule should work similarly for both.')

    print(f'\n  === CONCLUSIONS ===')
    print(f'  If transl and body have similar normalized distributions:')
    print(f'    → Problem is NOT in latent statistics, likely in transformer attention/architecture')
    print(f'  If transl has very different distribution (much smaller/larger std):')
    print(f'    → Flow matching noise schedule is suboptimal for translation')
    print(f'    → Consider per-token normalization or different noise schedules')

    print('\n' + '=' * 90)
    print('[+] Done.')


if __name__ == '__main__':
    main()
