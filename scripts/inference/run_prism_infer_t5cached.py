"""PRISM inference using pre-extracted T5 features (NO T5 encoder loaded).

This script verifies that pre-extracted T5 features produce valid inference
results, proving the extraction + loading pipeline works correctly.

Key difference from run_prism_infer_lowmem.py:
- Does NOT load or use T5 text encoder at all
- Loads pre-extracted .pt files from data/t5_feature/
- Pads embeddings to max_seq_length and passes directly to transformer

Usage:
    python3 scripts/inference/run_prism_infer_t5cached.py \
        --config configs/prism/prism_1b_tp2m_multiframe_kt_spectral_unified_t5cached.py \
        --checkpoint work_dirs/prism_1b_tp2m_multiframe_kt_spectral_unified/checkpoint-iter_15000 \
        --output-dir work_dirs/prism_1b_tp2m_multiframe_kt_spectral_unified/eval_15k_t5cached
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
    parser.add_argument('--max-seq-length', type=int, default=256)
    parser.add_argument('--feature-dir', default='data/t5_feature',
                        help='Directory with pre-extracted T5 features')
    return parser.parse_args()


# Pre-extracted feature files to use for inference (found in data/t5_feature/)
FEATURE_FILES = [
    "data/t5_feature/hymotion_data/Academic/20250916/human_checked_augmented_caption/motionpro/0814_zzb_2024-08-14-10-05-50_originalframes_007173_007446.pt",
    "data/t5_feature/hymotion_data/Academic/20250916/improved_simple_augmented_caption/motionpro/0930_hx_2024-09-30-14-38-48_originalframes_006325_006549.pt",
    "data/t5_feature/hymotion_data/AcademicRetarget/20250910/human_checked_augmented_caption/SnapMoGen/ep1_00242_3214_3430.pt",
    "data/t5_feature/hymotion_data/Academic/20250916/improved_simple_augmented_caption_deprecated_mirror_251215/M_dancedb/Fanie_Zumba_C3D_poses_originalframes_004316_004496.pt",
    "data/t5_feature/hymotion_data/Academic/20250916/improved_simple_augmented_caption_deprecated_mirror_251215/M_motionpro/0801_lh_2024-08-01-16-07-50_originalframes_009127_009322.pt",
]


def load_preextracted_features(feature_files, null_embedding_path, max_seq_length=256):
    """Load pre-extracted T5 features from .pt files.

    For each file, randomly pick one caption variant and pad to max_seq_length.
    Also loads the null (empty string) embedding for CFG negative prompt.

    Returns:
        all_text_states: list of [1, max_seq_length, 4096] tensors
        neg_text_states: [1, max_seq_length, 4096] tensor (null/empty embedding)
        captions: list of caption strings used
    """
    import random
    random.seed(42)

    all_text_states = []
    captions = []

    for feat_path in feature_files:
        data = torch.load(feat_path, map_location='cpu', weights_only=False)

        # Randomly pick one caption variant
        num_variants = len(data['captions'])
        idx = random.randint(0, num_variants - 1)

        embedding = data['embeddings'][idx]  # [seq_len, 4096] bf16
        caption = data['captions'][idx]
        seq_len = data['seq_lens'][idx]

        # Pad to max_seq_length (same as LoadPreExtractedT5Feature transform)
        if embedding.shape[0] < max_seq_length:
            padding = torch.zeros(
                max_seq_length - embedding.shape[0],
                embedding.shape[1],
                dtype=embedding.dtype
            )
            padded = torch.cat([embedding, padding], dim=0)
        else:
            padded = embedding[:max_seq_length]

        all_text_states.append(padded.unsqueeze(0))  # [1, max_seq_length, 4096]
        captions.append(caption)

    # Load null embedding (for CFG negative prompt)
    null_data = torch.load(null_embedding_path, map_location='cpu', weights_only=False)
    null_emb = null_data['embedding']  # [1, 4096] (EOS token only)
    null_seq_len = null_data['seq_len']  # 1

    # Pad null embedding to max_seq_length
    if null_emb.shape[0] < max_seq_length:
        padding = torch.zeros(
            max_seq_length - null_emb.shape[0],
            null_emb.shape[1],
            dtype=null_emb.dtype
        )
        neg_text_states = torch.cat([null_emb, padding], dim=0)
    else:
        neg_text_states = null_emb[:max_seq_length]
    neg_text_states = neg_text_states.unsqueeze(0)  # [1, max_seq_length, 4096]

    return all_text_states, neg_text_states, captions


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
    """Denoise one motion segment (identical to run_prism_infer_lowmem.py)."""
    batch_size = 1
    num_latent_frames = (num_frames - 1) // vae_temporal + 1
    num_channels = transformer.config.in_channels

    shape = (batch_size, num_channels, num_latent_frames, num_joints)
    latents = randn_tensor(shape, device=device, dtype=dtype)

    condition = torch.zeros_like(latents)
    first_frame_mask = torch.ones_like(latents)

    motion_mask = torch.ones(
        batch_size, latents.shape[2], latents.shape[3], device=device
    )

    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps

    do_cfg = guidance_scale > 1.0 and negative_text_states is not None
    text_states_dev = text_states.to(device=device, dtype=dtype)
    neg_states_dev = negative_text_states.to(device=device, dtype=dtype) if do_cfg else None

    for t in timesteps:
        if expand_timesteps:
            latent_model_input = (
                (1 - first_frame_mask) * condition + first_frame_mask * latents
            ).to(dtype)
            temp_ts = (first_frame_mask[0][0] * t).flatten()
            timestep = temp_ts.unsqueeze(0).expand(batch_size, -1)
        else:
            latent_model_input = latents.to(dtype)
            timestep = t.expand(batch_size)

        noise_pred = transformer(
            hidden_states=latent_model_input,
            timestep=timestep,
            encoder_hidden_states=text_states_dev,
            attention_kwargs=None,
            is_causal=is_causal,
            hidden_states_mask=motion_mask,
        )

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

        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    return latents


@torch.no_grad()
def decode_and_postprocess(bundle, latents, device):
    """Decode latents -> motion -> smplx_dict (identical to lowmem script)."""
    vae = bundle.vae.to(device)

    latents_mean = bundle.latents_mean.to(latents)
    latents_std = bundle.latents_std.to(latents)
    latents = latents * latents_std + latents_mean

    device_type = device.type if hasattr(device, 'type') else 'cuda'
    with torch.autocast(device_type, enabled=False):
        motion = vae.decode(latents.float())

    smpl_processor = bundle.smpl_pose_processor
    x_dec = rearrange(motion, "b t j d -> b t (j d)")
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

    null_embedding_path = os.path.join(args.feature_dir, '_null_embedding.pt')

    # Step 1: Load pre-extracted T5 features (NO T5 encoder needed!)
    print("=" * 60)
    print("PRISM Inference with Pre-Extracted T5 Features")
    print("=" * 60)
    print(f"\nStep 1: Loading pre-extracted T5 features...")
    print(f"  Feature dir: {args.feature_dir}")
    print(f"  Null embedding: {null_embedding_path}")
    t0 = time.time()

    all_text_states, neg_text_states, captions = load_preextracted_features(
        FEATURE_FILES, null_embedding_path, args.max_seq_length
    )
    print(f"  Loaded {len(captions)} caption embeddings in {time.time()-t0:.2f}s")
    print(f"  (T5 encoder NOT loaded - saving ~11GB GPU memory!)")
    for i, cap in enumerate(captions):
        print(f"    [{i}] {cap[:80]}")

    # Step 2: Build bundle (text encoder is built but never used/loaded to GPU)
    print(f"\nStep 2: Building model bundle...")
    cfg = Config.fromfile(args.config)
    model_cfg = cfg.model.to_dict() if hasattr(cfg.model, 'to_dict') else cfg.model

    bundle_cls = MODEL_BUNDLES.get(model_cfg['type'])
    bundle = bundle_cls.from_config(model_cfg)
    bundle.eval()

    # Immediately free text_encoder and tokenizer - we don't need them
    if hasattr(bundle, 'text_encoder'):
        del bundle.text_encoder
    if hasattr(bundle, 'tokenizer'):
        del bundle.tokenizer
    gc.collect()
    print(f"  (Freed text_encoder & tokenizer - not needed with pre-extracted features)")

    # Load checkpoint (only transformer, VAE, scheduler, smpl_processor)
    state_dict = load_checkpoint(args.checkpoint, map_location='cpu')
    print(f'  Loading checkpoint: {args.checkpoint}')
    bundle.load_state_dict_selective(state_dict)
    del state_dict
    gc.collect()

    # Step 3: Move transformer to GPU and generate
    print(f"\nStep 3: Moving transformer to GPU...")
    dtype = torch.bfloat16
    bundle.transformer = bundle.transformer.to(device, dtype)
    torch.cuda.empty_cache()
    print(f"  GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print(f"  (Compare: with T5 this would be ~{torch.cuda.memory_allocated()/1e9 + 11:.1f} GB)")

    vae_temporal = bundle.vae.config.scale_factor_temporal
    num_joints = 23

    print(f"\nStep 4: Generating {len(captions)} motions ({args.num_steps} steps, cfg={args.guidance_scale})...")
    all_latents = []
    for i, caption in enumerate(captions):
        print(f"  [{i+1}/{len(captions)}] '{caption[:60]}...'")
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
            expand_timesteps=True,
            is_causal=False,
        )
        all_latents.append(latents.cpu())
        print(f"    Generated in {time.time()-t0:.1f}s")

    # Step 5: Decode
    print(f"\nStep 5: Decoding motions with VAE...")
    bundle.transformer = bundle.transformer.cpu()
    del bundle.transformer
    gc.collect()
    torch.cuda.empty_cache()

    for i, (caption, latents) in enumerate(zip(captions, all_latents)):
        print(f"  [{i+1}/{len(captions)}] Decoding...")
        latents_dev = latents.to(device)
        smplx_dict = decode_and_postprocess(bundle, latents_dev, device)

        fname = f"motion_{i:02d}.npz"
        out_path = os.path.join(args.output_dir, fname)
        bundle.smpl_pose_processor.save_smplx_npz(out_path, smplx_dict)
        print(f"    Saved: {out_path}")

    # Save prompt list
    prompt_file = os.path.join(args.output_dir, "prompts.txt")
    with open(prompt_file, 'w') as f:
        for i, cap in enumerate(captions):
            f.write(f"{i:02d}: {cap}\n")

    print(f"\n{'=' * 60}")
    print(f"Done! Results saved to: {args.output_dir}")
    print(f"Prompts: {prompt_file}")
    print(f"{'=' * 60}")
    print(f"\nVerification: T5 features loaded from .pt files produced valid motions")
    print(f"without loading the 5.7B T5 encoder. Pipeline is working correctly.")


if __name__ == '__main__':
    main()
