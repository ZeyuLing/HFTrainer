#!/usr/bin/env python3
"""PRISM T2M eval — Low-memory single-GPU version with CPU offloading.

Loads text encoder to GPU for encoding, then offloads to CPU before denoising.
This allows running on GPUs with ~15GB VRAM (e.g., Tesla T4).

Usage:
    python3 scripts/eval/eval_prism_t2m_hml3d_lowmem.py \
        --config configs/prism/prism_1b_tp2m_multiframe_kt_spectral.py \
        --checkpoint work_dirs/prism_1b_tp2m_multiframe_kt_spectral/checkpoint-epoch_2 \
        --output-dir work_dirs/prism_1b_tp2m_multiframe_kt_spectral/eval_hml3d_epoch2_50samples \
        --max-samples 50
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
HF_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(HF_ROOT))


def load_test_samples(
    rewritten_file: Path,
    meta_file: Path,
    max_samples: Optional[int] = None,
    min_frames: int = 24,
    max_frames: int = 196,
) -> List[Dict]:
    """Load test samples by combining rewritten captions with metadata."""
    captions = json.loads(rewritten_file.read_text())
    meta = json.loads(meta_file.read_text())
    data_list = meta['data_list']

    samples = []
    for motion_id, caption in captions.items():
        if motion_id not in data_list:
            continue
        entry = data_list[motion_id]
        num_frames = int(entry.get('num_frames', 0))
        if num_frames < min_frames:
            continue
        num_frames = min(num_frames, max_frames)

        samples.append({
            'name': motion_id,
            'caption': caption,
            'num_frames': num_frames,
        })

        if max_samples and len(samples) >= max_samples:
            break

    return samples


def save_smplx_npz(out_path: str, smplx_dict: Dict):
    """Save smplx_dict to compressed NPZ."""
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


def main():
    parser = argparse.ArgumentParser(
        description='PRISM T2M eval — Low-memory single-GPU version.',
    )
    parser.add_argument('--config', type=str,
                        default='configs/prism/prism_1b_tp2m_multiframe_kt_spectral.py')
    parser.add_argument('--checkpoint', type=str,
                        default='work_dirs/prism_1b_tp2m_multiframe_kt_spectral/checkpoint-epoch_2')
    parser.add_argument('--anno-file', type=str,
                        default='data/annotation/test_hml3d_rewritten.json')
    parser.add_argument('--meta-file', type=str,
                        default='data/annotation/test_hml3d.json')
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--max-samples', type=int, default=50)
    parser.add_argument('--num-inference-steps', type=int, default=50)
    parser.add_argument('--guidance-scale', type=float, default=5.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--min-frames', type=int, default=24)
    parser.add_argument('--max-frames', type=int, default=196)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = str(Path(args.checkpoint).parent / 'eval_hml3d_epoch2_50samples')

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda:0')

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load test samples
    print(f'[+] Loading test samples...')
    samples = load_test_samples(
        rewritten_file=Path(args.anno_file),
        meta_file=Path(args.meta_file),
        max_samples=args.max_samples,
        min_frames=args.min_frames,
        max_frames=args.max_frames,
    )
    print(f'    Total: {len(samples)} samples')

    # Filter already done
    remaining = []
    for s in samples:
        out_path = Path(args.output_dir) / f"{s['name']}.npz"
        if not out_path.exists():
            remaining.append(s)
    print(f'    Already done: {len(samples) - len(remaining)}, remaining: {len(remaining)}')

    if not remaining:
        print('[+] All samples already generated!')
        return

    # ==== Build model components individually with CPU offloading ====
    print('[+] Building model...')
    from mmengine.config import Config
    from hftrainer.registry import MODEL_BUNDLES

    cfg = Config.fromfile(args.config)
    bundle_cfg = cfg.model

    # Build bundle on CPU first
    bundle = MODEL_BUNDLES.build(bundle_cfg)

    # Load checkpoint
    ckpt_path = os.path.join(args.checkpoint, 'model.pt')
    print(f'[+] Loading checkpoint: {ckpt_path}')
    state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    bundle.load_state_dict_selective(state_dict, strict=False)
    del state_dict
    gc.collect()

    # Step 1: Text encoder on GPU (temporarily) to encode all prompts
    print('[+] Encoding text prompts on GPU...')
    dtype = torch.bfloat16

    # Move text encoder + tokenizer to GPU
    bundle.text_encoder = bundle.text_encoder.to(device, dtype).eval()

    all_text_states = []
    neg_text_states = None

    with torch.no_grad():
        # Encode negative prompt
        neg_text_states = bundle.encode_prompt(
            [''],
            max_sequence_length=256,
            prompt_drop_rate=0.0,
            dtype=dtype,
        ).to(device)

        # Encode all prompts
        for s in remaining:
            text_states = bundle.encode_prompt(
                [s['caption']],
                max_sequence_length=256,
                prompt_drop_rate=0.0,
                dtype=dtype,
            )
            all_text_states.append(text_states.cpu())

    # Move negative prompt to CPU for now
    neg_text_states_cpu = neg_text_states.cpu()
    del neg_text_states

    # Offload text encoder to CPU to free GPU memory
    print('[+] Offloading text encoder to CPU...')
    bundle.text_encoder = bundle.text_encoder.to('cpu')
    torch.cuda.empty_cache()
    gc.collect()

    # Step 2: Move transformer + VAE + SMPL to GPU for denoising
    print('[+] Loading transformer + VAE to GPU...')
    bundle.transformer = bundle.transformer.to(device, dtype).eval()
    bundle.vae = bundle.vae.to(device, torch.float32).eval()
    # SMPL processor must stay float32 for FK operations
    bundle.smpl_pose_processor = bundle.smpl_pose_processor.to(device, torch.float32)

    # Set up scheduler
    from diffusers import FlowMatchEulerDiscreteScheduler
    bundle.scheduler = bundle.scheduler

    torch.cuda.empty_cache()
    gc.collect()

    free_mem = torch.cuda.mem_get_info(0)[0] / 1024**3
    print(f'    GPU free after loading: {free_mem:.1f} GB')

    # Step 3: Build pipeline-like denoising
    from hftrainer.pipelines.motion.prism_backend import PrismARPipeline
    # We can't use PrismARPipeline directly since it puts everything on GPU.
    # Instead, run inference manually using the bundle components.

    from diffusers.utils.torch_utils import randn_tensor
    from einops import rearrange
    from hftrainer.models.motion.prism.gaussian_distribution import DiagonalGaussianDistributionNd
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

    print(f'[+] Starting inference ({len(remaining)} samples)...')
    n_success = 0
    n_fail = 0
    t_start = time.time()

    for i, sample in enumerate(remaining):
        name = sample['name']
        num_frames = sample['num_frames']

        out_path = Path(args.output_dir) / f'{name}.npz'
        if out_path.exists():
            n_success += 1
            continue

        try:
            with torch.no_grad():
                # Get pre-encoded text states
                text_states = all_text_states[i].to(device, dtype)
                neg_text = neg_text_states_cpu.to(device, dtype)

                # Latent shape
                num_latent_frames = (num_frames - 1) // vae_temporal + 1

                # Random noise
                latents = randn_tensor(
                    (1, num_channels, num_latent_frames, num_joints),
                    device=device, dtype=dtype,
                )

                # Scheduler
                bundle.scheduler.set_timesteps(args.num_inference_steps, device=device)
                timesteps = bundle.scheduler.timesteps

                # Motion mask: all valid positions (no padding)
                motion_mask = torch.ones(
                    1, num_latent_frames, num_joints, device=device
                )

                # Denoising loop with expand_timesteps=True (matches training)
                for t in timesteps:
                    latent_model_input = latents.to(dtype)

                    # Per-token timestep: [B, T_latent * J]
                    # For pure T2M (no conditioning), all positions get same t
                    temp_ts = (torch.ones(num_latent_frames, num_joints, device=device) * t).flatten()
                    timestep = temp_ts.unsqueeze(0)  # [1, T_latent * J]

                    # Conditional prediction
                    model_pred = bundle.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=text_states,
                        hidden_states_mask=motion_mask,
                    )
                    if hasattr(model_pred, 'sample'):
                        model_pred = model_pred.sample

                    # CFG
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
                    motion = bundle.vae.decode(decoded_latents.float())  # [1, T, J, D]

                # Post-process to SMPL-X (cast to float32 for FK/normalize)
                x_dec = rearrange(motion, 'b t j d -> b t (j d)').float()
                x_dec = bundle.smpl_pose_processor.denormalize(x_dec)
                transl_abs_rel = x_dec[..., :6]
                transl = bundle.smpl_pose_processor.inv_convert_transl(transl_abs_rel)
                pred_poses = x_dec[..., 6:]

                pred_poses = rearrange(pred_poses, "b t (j d)-> (b t) j d", d=6)
                # Training data already uses column-major 6D convention [R00,R10,R20,R01,R11,R21]
                # (matrix_to_rotation_6d uses _stack_cols01 → columns of rotation matrix).
                # rotation_6d_to_axis_angle expects column-major input — no permutation needed.
                pred_poses = rotation_6d_to_axis_angle(pred_poses)
                pred_poses = rearrange(pred_poses, "(b t) j d -> b t (j d)", b=1)

                smplx_dict = bundle.smpl_pose_processor.transl_pose_to_smplx_dict(
                    transl.squeeze(0).float(),
                    pred_poses.squeeze(0).float(),
                    mocap_framerate=30.0,
                    gender='neutral',
                    rot_type='axis_angle',
                )
                smplx_dict = bundle.smpl_pose_processor.normalize_smplx_dict(smplx_dict)

            save_smplx_npz(str(out_path), smplx_dict)
            n_success += 1

        except Exception as e:
            import traceback
            print(f'[!] Failed: {name}: {e}')
            traceback.print_exc()
            n_fail += 1

        # Progress
        if (i + 1) % 5 == 0:
            elapsed = time.time() - t_start
            avg_t = elapsed / (i + 1)
            eta = avg_t * (len(remaining) - i - 1)
            print(f'[{i + 1}/{len(remaining)}] ok={n_success} fail={n_fail} '
                  f'avg={avg_t:.1f}s/sample ETA={eta/60:.0f}min')

    elapsed = time.time() - t_start
    print(f'\n[+] Done: {n_success} ok, {n_fail} fail, {elapsed:.0f}s total')
    print(f'    Output: {args.output_dir}')


if __name__ == '__main__':
    main()
