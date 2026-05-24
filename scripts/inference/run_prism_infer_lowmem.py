"""Memory-efficient PRISM inference script for low-VRAM GPUs.

Strategy: encode text on CPU, then offload text_encoder, load transformer+VAE
to GPU for denoising, finally decode and save.

This script replicates the EXACT denoising logic from PrismARPipeline.generate_single_segment
including hidden_states_mask, is_causal, and expand_timesteps handling.

Usage:
    python3 scripts/inference/run_prism_infer_lowmem.py \
        --config configs/prism/prism_1b_tp2m_multiframe.py \
        --checkpoint work_dirs/prism_1b_tp2m_multiframe/checkpoint-iter_15000 \
        --output-dir work_dirs/prism_1b_tp2m_multiframe/eval_after_fix
"""

import argparse
import gc
import os
import sys
import time

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
)
from diffusers.utils.torch_utils import randn_tensor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--num-frames', type=int, default=129)
    parser.add_argument('--num-steps', type=int, default=50)
    parser.add_argument('--guidance-scale', type=float, default=5.0)
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()


# Test prompts
PROMPTS = [
    "a person walks forward slowly",
    "a person raises both hands above the head",
    "a person kicks with the right leg",
    "a person sits down on a chair",
    "a person waves hello with the right hand",
]


def encode_text_on_cpu(bundle, prompts, max_seq_len=256):
    """Encode prompts using text_encoder on CPU, return embeddings.

    CORRECTED VERSION: Uses exact same logic as bundle.encode_prompt():
    - Computes actual sequence lengths from attention_mask
    - Trims to actual seq_len (removing padded tokens)
    - Pads with explicit new_zeros (not masked multiplication)
    - Uses max_seq_len=256 (matches training max_text_length=256)

    This ensures identical embeddings to training pipeline.
    """
    # Move text encoder to CPU for encoding
    bundle.text_encoder = bundle.text_encoder.cpu()
    torch.cuda.empty_cache()

    all_text_states = []
    neg_text_states = None

    for prompt in prompts:
        inputs = bundle.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_seq_len,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = inputs.input_ids.cpu()
        attention_mask = inputs.attention_mask.cpu()
        
        # Compute actual sequence length (KEY FIX 1 - matches training line 181)
        seq_lens = attention_mask.gt(0).sum(dim=1).long()

        with torch.no_grad():
            text_output = bundle.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            hidden_states = text_output.last_hidden_state  # [1, max_seq_len, D]
            
            # Trim to actual seq_len (KEY FIX 2 - matches training line 185)
            # This removes padded tokens, not just masking them out
            hidden_states = hidden_states[:, :seq_lens[0], :]  # [1, seq_len, D]
            
            # Pad with explicit zeros (KEY FIX 3 - matches training lines 186-192)
            # This creates EXACT zeros using new_zeros(), not encoder_output * 0
            # The difference matters: encoder_output * 0 ≈ [-1.3e-8, 1.5e-8, ...] (noisy)
            # while new_zeros() produces [0.0, 0.0, ...] (exact)
            if hidden_states.shape[1] < max_seq_len:
                padding = hidden_states.new_zeros(
                    hidden_states.shape[0],
                    max_seq_len - hidden_states.shape[1],
                    hidden_states.shape[2]
                )
                hidden_states = torch.cat([hidden_states, padding], dim=1)
            
            all_text_states.append(hidden_states)

    # Negative prompt (empty string) - apply same fixes
    inputs = bundle.tokenizer(
        "",
        padding="max_length",
        max_length=max_seq_len,
        truncation=True,
        return_tensors="pt",
    )
    
    # Compute seq_len for negative prompt
    seq_lens_neg = inputs.attention_mask.gt(0).sum(dim=1).long()
    
    with torch.no_grad():
        neg_output = bundle.text_encoder(
            input_ids=inputs.input_ids.cpu(),
            attention_mask=inputs.attention_mask.cpu(),
        )
        neg_text_states = neg_output.last_hidden_state  # [1, max_seq_len, D]
        
        # Trim to actual seq_len
        neg_text_states = neg_text_states[:, :seq_lens_neg[0], :]
        
        # Pad with explicit zeros
        if neg_text_states.shape[1] < max_seq_len:
            padding = neg_text_states.new_zeros(
                neg_text_states.shape[0],
                max_seq_len - neg_text_states.shape[1],
                neg_text_states.shape[2]
            )
            neg_text_states = torch.cat([neg_text_states, padding], dim=1)

    # Free text encoder from memory
    del bundle.text_encoder
    gc.collect()
    torch.cuda.empty_cache()

    return all_text_states, neg_text_states




@torch.no_grad()
def generate_single_segment(
    transformer,
    scheduler,
    text_states,
    negative_text_states,
    num_frames,
    num_joints,
    vae_temporal,
    num_inference_steps,
    guidance_scale,
    device,
    dtype,
    expand_timesteps=True,
    is_causal=False,
):
    """Denoise one motion segment.

    Replicates the exact logic from PrismARPipeline.generate_single_segment,
    including hidden_states_mask, is_causal, expand_timesteps, and
    first_frame_mask handling.
    """
    batch_size = 1
    num_latent_frames = (num_frames - 1) // vae_temporal + 1
    num_channels = transformer.config.in_channels

    # --- Prepare latents (matches PrismARPipeline.prepare_latents) ---
    shape = (batch_size, num_channels, num_latent_frames, num_joints)
    latents = randn_tensor(shape, device=device, dtype=dtype)

    # No first_frame conditioning for basic T2M generation
    condition = torch.zeros_like(latents)
    first_frame_mask = torch.ones_like(latents)  # All 1s = denoise everything

    # --- Create motion mask (critical for attention) ---
    # All positions valid during inference (no padding)
    motion_mask = torch.ones(
        batch_size, latents.shape[2], latents.shape[3], device=device
    )

    # --- Scheduler setup ---
    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps

    do_cfg = guidance_scale > 1.0 and negative_text_states is not None
    text_states_dev = text_states.to(device=device, dtype=dtype)
    neg_states_dev = negative_text_states.to(device=device, dtype=dtype) if do_cfg else None

    # --- Denoising loop (exact replica of PrismARPipeline) ---
    for t in timesteps:
        # Timestep handling (expand_timesteps logic)
        if expand_timesteps:
            # With expand_timesteps=True, timestep is per-joint:
            # first_frame_mask[0][0] is shape [T_latent, J], multiply by scalar t
            # Since first_frame_mask is all-ones (no conditioning), this gives t for all positions
            latent_model_input = (
                (1 - first_frame_mask) * condition + first_frame_mask * latents
            ).to(dtype)
            temp_ts = (first_frame_mask[0][0] * t).flatten()  # [T_latent * J]
            timestep = temp_ts.unsqueeze(0).expand(batch_size, -1)  # [B, T_latent * J]
        else:
            latent_model_input = latents.to(dtype)
            timestep = t.expand(batch_size)

        # Forward pass with ALL required parameters
        noise_pred = transformer(
            hidden_states=latent_model_input,
            timestep=timestep,
            encoder_hidden_states=text_states_dev,
            attention_kwargs=None,
            is_causal=is_causal,
            hidden_states_mask=motion_mask,
        )

        # CFG
        if do_cfg:
            noise_uncond = transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=neg_states_dev,
                attention_kwargs=None,
                is_causal=is_causal,
                hidden_states_mask=motion_mask,
            )
            noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)

        # Scheduler step
        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    return latents


@torch.no_grad()
def decode_and_postprocess(bundle, latents, device):
    """Decode latents -> motion -> smplx_dict.

    Replicates PrismARPipeline.decode_motion + post_process_motion.
    """
    # Move VAE to device
    vae = bundle.vae.to(device)

    # Denormalize latents (latent-space normalization)
    latents_mean = bundle.latents_mean.to(latents)
    latents_std = bundle.latents_std.to(latents)
    latents = latents * latents_std + latents_mean

    # Decode (VAE runs in fp32)
    device_type = device.type if hasattr(device, 'type') else 'cuda'
    with torch.autocast(device_type, enabled=False):
        motion = vae.decode(latents.float())  # [B, T, J=23, D=6]

    # Post-process motion (matches PrismARPipeline.post_process_motion)
    smpl_processor = bundle.smpl_pose_processor

    # Flatten: [B, T, 23, 6] -> [B, T, 138]
    x_dec = rearrange(motion, "b t j d -> b t (j d)")
    # Denormalize using stats (138-dim: 6 transl + 22*6 rotation)
    x_dec = smpl_processor.denormalize(x_dec)

    transl_abs_rel = x_dec[..., :6]
    transl = smpl_processor.inv_convert_transl(transl_abs_rel)
    pred_poses = x_dec[..., 6:]

    pred_poses = rearrange(pred_poses, "b t (j d) -> (b t) j d", d=6)
    # Training data uses column-major 6D convention [R00,R10,R20,R01,R11,R21].
    # rotation_6d_to_axis_angle expects column-major input — no permutation needed.
    pred_poses = rotation_6d_to_axis_angle(pred_poses)
    pred_poses = rearrange(pred_poses, "(b t) j d -> b t (j d)", b=1)

    pred_smplx_dict = smpl_processor.transl_pose_to_smplx_dict(
        transl.squeeze(0),
        pred_poses.squeeze(0),
        mocap_framerate=30.0,
        gender='neutral',
        rot_type="axis_angle",
    )

    pred_smplx_dict = smpl_processor.normalize_smplx_dict(pred_smplx_dict)
    return pred_smplx_dict


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # Build bundle
    cfg = Config.fromfile(args.config)
    model_cfg = cfg.model.to_dict() if hasattr(cfg.model, 'to_dict') else cfg.model
    bundle_cls = MODEL_BUNDLES.get(model_cfg['type'])
    bundle = bundle_cls.from_config(model_cfg)
    bundle.eval()

    # Load checkpoint
    state_dict = load_checkpoint(args.checkpoint, map_location='cpu')
    print(f'Loading checkpoint: {args.checkpoint}')
    bundle.load_state_dict_selective(state_dict)
    del state_dict
    gc.collect()

    # Check expand_timesteps config (from PrismARPipeline default)
    expand_timesteps = True
    is_causal = False
    print(f"  expand_timesteps={expand_timesteps}, is_causal={is_causal}")

    # Step 1: Encode all text on CPU
    print("Step 1: Encoding text prompts on CPU...")
    t0 = time.time()
    all_text_states, neg_text_states = encode_text_on_cpu(bundle, PROMPTS)
    print(f"  Done in {time.time()-t0:.1f}s")

    # Step 2: Move transformer to GPU
    print("Step 2: Moving transformer to GPU...")
    dtype = torch.bfloat16
    bundle.transformer = bundle.transformer.to(device, dtype)
    torch.cuda.empty_cache()
    print(f"  GPU memory after transformer load: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    vae_temporal = bundle.vae.config.scale_factor_temporal
    num_joints = 23  # 1 transl joint + 22 body joints in PRISM latent space

    # Step 3: Generate motions
    print(f"Step 3: Generating {len(PROMPTS)} motions ({args.num_steps} steps, cfg={args.guidance_scale})...")
    all_latents = []
    for i, prompt in enumerate(PROMPTS):
        print(f"  [{i+1}/{len(PROMPTS)}] '{prompt}'")
        t0 = time.time()
        latents = generate_single_segment(
            transformer=bundle.transformer,
            scheduler=bundle.scheduler,
            text_states=all_text_states[i],
            negative_text_states=neg_text_states,
            num_frames=args.num_frames,
            num_joints=num_joints,
            vae_temporal=vae_temporal,
            num_inference_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            device=device,
            dtype=dtype,
            expand_timesteps=expand_timesteps,
            is_causal=is_causal,
        )
        all_latents.append(latents.cpu())
        print(f"    Generated in {time.time()-t0:.1f}s")

    # Step 4: Free transformer, move VAE to GPU for decoding
    print("Step 4: Decoding motions with VAE...")
    bundle.transformer = bundle.transformer.cpu()
    del bundle.transformer
    gc.collect()
    torch.cuda.empty_cache()

    for i, (prompt, latents) in enumerate(zip(PROMPTS, all_latents)):
        print(f"  [{i+1}/{len(PROMPTS)}] Decoding '{prompt}'...")
        latents_dev = latents.to(device)
        smplx_dict = decode_and_postprocess(bundle, latents_dev, device)

        # Save
        fname = f"motion_{i:02d}.npz"
        out_path = os.path.join(args.output_dir, fname)
        bundle.smpl_pose_processor.save_smplx_npz(out_path, smplx_dict)
        print(f"    Saved: {out_path}")

    # Save a prompt list for reference
    prompt_file = os.path.join(args.output_dir, "prompts.txt")
    with open(prompt_file, 'w') as f:
        for i, p in enumerate(PROMPTS):
            f.write(f"{i:02d}: {p}\n")
    print(f"\nDone! Results saved to: {args.output_dir}")
    print(f"Prompt list: {prompt_file}")


if __name__ == '__main__':
    main()
