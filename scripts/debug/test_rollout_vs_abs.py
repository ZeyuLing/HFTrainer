#!/usr/bin/env python3
"""Quick test: compare use_rollout=True vs False for translation reconstruction.

Re-runs inference on a few samples and saves both versions to compare.
Specifically targets the worst-case stationary motions where drift is catastrophic.

Usage:
    python3 scripts/debug/test_rollout_vs_abs.py \
        --config configs/prism/prism_1b_tp2m_multiframe_kt_spectral.py \
        --checkpoint work_dirs/prism_1b_tp2m_multiframe_kt_spectral/checkpoint-epoch_2 \
        --output-dir work_dirs/prism_1b_tp2m_multiframe_kt_spectral/debug_rollout_test
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from einops import rearrange

SCRIPT_DIR = Path(__file__).resolve().parent
HF_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(HF_ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        default='configs/prism/prism_1b_tp2m_multiframe_kt_spectral.py')
    parser.add_argument('--checkpoint', type=str,
                        default='work_dirs/prism_1b_tp2m_multiframe_kt_spectral/checkpoint-epoch_2')
    parser.add_argument('--output-dir', type=str,
                        default='work_dirs/prism_1b_tp2m_multiframe_kt_spectral/debug_rollout_test')
    parser.add_argument('--num-inference-steps', type=int, default=50)
    parser.add_argument('--guidance-scale', type=float, default=5.0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda:0')
    dtype = torch.bfloat16

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Target samples: worst stationary motions
    target_samples = [
        {'name': 'humanml3d_1059', 'num_frames': 121},  # sits, raises hand (worst: 413x)
        {'name': 'humanml3d_194', 'num_frames': 193},   # squats (120x)
        {'name': 'humanml3d_942', 'num_frames': 89},    # cross-legged eating (74x)
        {'name': 'humanml3d_927', 'num_frames': 193},   # jogging (should be OK)
        {'name': 'humanml3d_1063', 'num_frames': 133},  # walking (should be OK)
    ]

    # Load captions
    captions = json.loads(Path('data/annotation/test_hml3d_rewritten.json').read_text())
    for s in target_samples:
        s['caption'] = captions.get(s['name'], f"motion {s['name']}")

    # Build model
    print('[+] Building model...')
    from mmengine.config import Config
    from hftrainer.registry import MODEL_BUNDLES

    cfg = Config.fromfile(args.config)
    bundle = MODEL_BUNDLES.build(cfg.model)

    ckpt_path = os.path.join(args.checkpoint, 'model.pt')
    print(f'[+] Loading checkpoint: {ckpt_path}')
    state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    bundle.load_state_dict_selective(state_dict, strict=False)
    del state_dict
    gc.collect()

    # Encode text on GPU, then offload
    print('[+] Encoding text prompts...')
    bundle.text_encoder = bundle.text_encoder.to(device, dtype).eval()

    all_text_states = []
    with torch.no_grad():
        neg_text_states = bundle.encode_prompt(
            [''], max_sequence_length=256, prompt_drop_rate=0.0, dtype=dtype,
        ).cpu()
        for s in target_samples:
            text_states = bundle.encode_prompt(
                [s['caption']], max_sequence_length=256, prompt_drop_rate=0.0, dtype=dtype,
            )
            all_text_states.append(text_states.cpu())

    bundle.text_encoder = bundle.text_encoder.to('cpu')
    torch.cuda.empty_cache()
    gc.collect()

    # Load transformer + VAE
    print('[+] Loading transformer + VAE to GPU...')
    bundle.transformer = bundle.transformer.to(device, dtype).eval()
    bundle.vae = bundle.vae.to(device, torch.float32).eval()
    bundle.smpl_pose_processor = bundle.smpl_pose_processor.to(device, torch.float32)

    from diffusers.utils.torch_utils import randn_tensor
    from hftrainer.models.motion.components.utils.geometry.rotation_convert import (
        rotation_6d_to_axis_angle,
    )

    vae_temporal = bundle.vae.config.scale_factor_temporal
    num_channels = bundle.transformer.config.in_channels
    num_joints = 23

    latents_mean = torch.tensor(
        bundle.vae.config.latents_mean, dtype=dtype, device=device
    ).view(1, bundle.vae.config.z_dim, 1, 1)
    latents_std = torch.tensor(
        bundle.vae.config.latents_std, dtype=dtype, device=device
    ).view(1, bundle.vae.config.z_dim, 1, 1)

    torch.cuda.empty_cache()
    free_mem = torch.cuda.mem_get_info(0)[0] / 1024**3
    print(f'    GPU free: {free_mem:.1f} GB')

    print(f'[+] Running inference on {len(target_samples)} samples...')

    for i, sample in enumerate(target_samples):
        name = sample['name']
        num_frames = sample['num_frames']
        print(f'\n--- [{i+1}/{len(target_samples)}] {name}: "{sample["caption"][:50]}..." ({num_frames} frames) ---')

        with torch.no_grad():
            text_states = all_text_states[i].to(device, dtype)
            neg_text = neg_text_states.to(device, dtype)

            num_latent_frames = (num_frames - 1) // vae_temporal + 1
            latents = randn_tensor(
                (1, num_channels, num_latent_frames, num_joints),
                device=device, dtype=dtype,
            )

            bundle.scheduler.set_timesteps(args.num_inference_steps, device=device)
            timesteps = bundle.scheduler.timesteps

            motion_mask = torch.ones(1, num_latent_frames, num_joints, device=device)

            for t in timesteps:
                latent_model_input = latents.to(dtype)
                temp_ts = (torch.ones(num_latent_frames, num_joints, device=device) * t).flatten()
                timestep = temp_ts.unsqueeze(0)

                model_pred = bundle.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=text_states,
                    hidden_states_mask=motion_mask,
                )
                if hasattr(model_pred, 'sample'):
                    model_pred = model_pred.sample

                if args.guidance_scale > 1.0:
                    noise_uncond = bundle.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=neg_text,
                        hidden_states_mask=motion_mask,
                    )
                    if hasattr(noise_uncond, 'sample'):
                        noise_uncond = noise_uncond.sample
                    model_pred = noise_uncond + args.guidance_scale * (model_pred - noise_uncond)

                latents = bundle.scheduler.step(model_pred, t, latents).prev_sample

            # Decode VAE
            decoded_latents = latents * latents_std.to(latents) + latents_mean.to(latents)
            with torch.autocast('cuda', enabled=False):
                motion = bundle.vae.decode(decoded_latents.float())

            # x_dec is [B, T, 23, 6] -> [B, T, 138]
            x_dec = rearrange(motion, 'b t j d -> b t (j d)').float()
            x_dec = bundle.smpl_pose_processor.denormalize(x_dec)

            # ==== KEY COMPARISON: abs_rel dims [0:6] ====
            transl_abs_rel = x_dec[..., :6]  # [1, T, 6]
            print(f'  Denormalized transl_abs_rel shape: {transl_abs_rel.shape}')
            print(f'  Abs position (dims 0:3) frame 0: {transl_abs_rel[0, 0, :3].cpu().numpy()}')
            print(f'  Abs position (dims 0:3) frame -1: {transl_abs_rel[0, -1, :3].cpu().numpy()}')
            print(f'  Rel velocity (dims 3:6) frame 1: {transl_abs_rel[0, 1, 3:6].cpu().numpy()}')
            print(f'  Rel velocity (dims 3:6) mean: {transl_abs_rel[0, :, 3:6].mean(dim=0).cpu().numpy()}')

            # Method 1: use_rollout=True (current behavior)
            transl_rollout = bundle.smpl_pose_processor.inv_convert_transl(transl_abs_rel)
            # Method 2: use_rollout=False (direct abs position)
            transl_abs = bundle.smpl_pose_processor.inv_convert_transl(transl_abs_rel, use_rollout=False)

            t_rollout = transl_rollout[0].cpu().numpy()
            t_abs = transl_abs[0].cpu().numpy()

            disp_rollout = np.linalg.norm(t_rollout[-1] - t_rollout[0])
            disp_abs = np.linalg.norm(t_abs[-1] - t_abs[0])

            print(f'  Displacement (rollout): {disp_rollout:.4f} m')
            print(f'  Displacement (abs):     {disp_abs:.4f} m')

            # Load GT for comparison
            meta = json.loads(Path('data/annotation/test_hml3d.json').read_text())
            entry = meta['data_list'].get(name, {})
            gt_path = entry.get('smplx_path', '')
            if gt_path:
                gt_full = os.path.join('data/motionhub', gt_path)
                if os.path.exists(gt_full):
                    gt_data = dict(np.load(gt_full))
                    gt_transl = gt_data.get('transl', gt_data.get('trans', np.zeros((2, 3))))
                    n = min(len(t_rollout), len(gt_transl))
                    gt_disp = np.linalg.norm(gt_transl[n-1] - gt_transl[0])
                    print(f'  GT displacement:        {gt_disp:.4f} m')
                    print(f'  Ratio (rollout/GT):     {disp_rollout/max(gt_disp, 0.001):.1f}x')
                    print(f'  Ratio (abs/GT):         {disp_abs/max(gt_disp, 0.001):.1f}x')

            # Also save both versions as NPZ for visual comparison
            # Post-process poses
            pred_poses = x_dec[..., 6:]
            pred_poses_rearr = rearrange(pred_poses, "b t (j d)-> (b t) j d", d=6)
            pred_poses_aa = rotation_6d_to_axis_angle(pred_poses_rearr)
            pred_poses_aa = rearrange(pred_poses_aa, "(b t) j d -> b t (j d)", b=1)

            for method, transl_tensor in [('rollout', transl_rollout), ('abs', transl_abs)]:
                smplx_dict = bundle.smpl_pose_processor.transl_pose_to_smplx_dict(
                    transl_tensor.squeeze(0).float(),
                    pred_poses_aa.squeeze(0).float(),
                    mocap_framerate=30.0,
                    gender='neutral',
                    rot_type='axis_angle',
                )
                smplx_dict = bundle.smpl_pose_processor.normalize_smplx_dict(smplx_dict)

                out_path = os.path.join(args.output_dir, f'{name}_{method}.npz')
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                pack = {}
                for k, v in smplx_dict.items():
                    if isinstance(v, np.ndarray):
                        pack[k] = v.astype(np.float32, copy=False)
                    elif isinstance(v, torch.Tensor):
                        pack[k] = v.detach().cpu().numpy().astype(np.float32)
                    else:
                        pack[k] = v
                np.savez_compressed(out_path, **pack)

            # Also save raw abs_rel values for analysis
            raw_data = {
                'abs_rel_6d': transl_abs_rel[0].cpu().numpy(),  # [T, 6]
                'transl_rollout': t_rollout,  # [T, 3]
                'transl_abs': t_abs,  # [T, 3]
            }
            np.savez(os.path.join(args.output_dir, f'{name}_raw.npz'), **raw_data)

    print('\n[+] Done! Results saved to:', args.output_dir)


if __name__ == '__main__':
    main()
