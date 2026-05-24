"""Test inference with the pretrained versatilemotion PRISM model via hf_trainer pipeline.

This verifies the rot6d convention fix by running inference with a model that was
trained CORRECTLY (column-major data with column-major stats in versatilemotion)
through our fixed pipeline.

If the output is NOT deformed, the fix is confirmed.

Usage:
    cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
    python3 scripts/debug/test_inference_pretrained.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import numpy as np

import hftrainer  # noqa - register modules
from hftrainer.registry import MODEL_BUNDLES


def main():
    device = torch.device('cuda')
    dtype = torch.bfloat16
    pretrained_root = '/apdcephfs_cq11/share_1467498/home/zeyuling/versatilemotion/opensource/prism/pretrained_models/prism_1.4b'
    output_dir = 'work_dirs/prism_1b_tp2m_multiframe/eval_pretrained_fix'
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("TEST: Inference with pretrained model through fixed pipeline")
    print("=" * 80)

    # Construct bundle config manually
    bundle_cfg = {
        'type': 'PrismBundle',
        'transformer': {
            'type': 'PrismTransformerMotionModel',
            'from_pretrained': {
                'pretrained_model_name_or_path': os.path.join(pretrained_root, 'transformer'),
            },
        },
        'vae': {
            'type': 'AutoencoderKLPrism2DTK',
            'from_pretrained': {
                'pretrained_model_name_or_path': os.path.join(pretrained_root, 'vae'),
            },
        },
        'tokenizer': {
            'type': 'T5Tokenizer',
            'from_pretrained': {
                'pretrained_model_name_or_path': os.path.join(pretrained_root, 'tokenizer'),
            },
        },
        'text_encoder': {
            'type': 'UMT5EncoderModel',
            'from_pretrained': {
                'pretrained_model_name_or_path': os.path.join(pretrained_root, 'text_encoder'),
            },
        },
        'scheduler': {
            'type': 'FlowMatchEulerDiscreteScheduler',
            'num_train_timesteps': 1000,
            'shift': 5.0,
            'use_dynamic_shifting': False,
        },
        'smpl_pose_processor': {
            'type': 'SMPLPoseProcessor',
            'do_normalize': True,
            'stats_file': os.path.join(pretrained_root, 'stats.json'),
            'rot_type': 'rotation_6d',
            'transl_type': 'abs_rel',
            'smpl_type': 'smpl_22',
            'smpl_model': {
                'type': 'SmplxLiteV437Coco17',
                'model_path': 'checkpoints/smpl_models/smplx',
                'smplx2smpl_path': 'checkpoints/smpl_models/smplx2smpl_sparse.pt',
                'coco17_regressor_path': 'checkpoints/smpl_models/smpl_coco17_J_regressor.pt',
                'smplx_verts437_path': 'checkpoints/smpl_models/smplx_verts437.pt',
                'gender': 'neutral',
                'num_betas': 10,
            },
        },
    }

    print("\n[1] Building bundle...")
    bundle_cls = MODEL_BUNDLES.get(bundle_cfg['type'])
    bundle = bundle_cls.from_config(bundle_cfg)
    bundle.eval()

    # Move to device with bf16 to save memory
    print("[2] Moving to GPU (bf16 for text_encoder + transformer)...")
    bundle.text_encoder = bundle.text_encoder.to(device, dtype)
    bundle.transformer = bundle.transformer.to(device, dtype)
    bundle.vae = bundle.vae.to(device, torch.float32)
    bundle = bundle.to(device)

    # Import the backend directly instead of PrismPipeline wrapper
    # (the wrapper calls .to() again which triggers OOM)
    from hftrainer.pipelines.motion.prism_backend import PrismARPipeline

    print("[3] Creating inference pipeline (direct backend)...")
    # Manually build PrismARPipeline without the redundant .to() calls
    # by passing already-loaded modules
    pipeline = PrismARPipeline.__new__(PrismARPipeline)
    from diffusers import DiffusionPipeline
    DiffusionPipeline.__init__(pipeline)
    pipeline.register_modules(
        vae=bundle.vae,
        text_encoder=bundle.text_encoder,
        tokenizer=bundle.tokenizer,
        transformer=bundle.transformer,
        scheduler=bundle.scheduler,
    )
    pipeline.register_to_config(expand_timesteps=True, is_causal=False)
    pipeline.smpl_processor = bundle.smpl_pose_processor
    pipeline.latents_mean = torch.tensor(
        bundle.vae.config.latents_mean, dtype=torch.float32, device=device
    ).view(1, bundle.vae.config.z_dim, 1, 1)
    pipeline.latents_std = torch.tensor(
        bundle.vae.config.latents_std, dtype=torch.float32, device=device
    ).view(1, bundle.vae.config.z_dim, 1, 1)
    pipeline.vae_scale_factor_temporal = bundle.vae.config.scale_factor_temporal
    pipeline._kafs_alpha_map = None
    pipeline._kafs_mode = "none"

    # Free bundle reference to save memory
    del bundle
    torch.cuda.empty_cache()

    print(f"    GPU memory used: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    # Generate motions with different prompts
    prompts = [
        "a person walks forward slowly",
        "a person raises both hands above their head",
        "a person sits down on a chair",
    ]

    for i, prompt in enumerate(prompts):
        print(f"\n[4.{i}] Generating: '{prompt}'")
        # PrismARPipeline.__call__ returns smplx_dict
        smplx_dict = pipeline(
            prompts=prompt,
            num_frames_per_segment=129,
            num_inference_steps=50,
            guidance_scale=2.0,
            normalize=True,
            max_sequence_length=256,
        )

        # Save result as npz
        output_path = os.path.join(output_dir, f'motion_{i}.npz')
        save_dict = {}
        for k, v in smplx_dict.items():
            if torch.is_tensor(v):
                save_dict[k] = v.cpu().numpy()
            elif isinstance(v, np.ndarray):
                save_dict[k] = v
            else:
                save_dict[k] = np.array(v)
        np.savez(output_path, **save_dict)
        print(f"    Saved to {output_path}")

        # Basic sanity checks
        if 'body_pose' in smplx_dict:
            bp = smplx_dict['body_pose']
            if torch.is_tensor(bp):
                bp = bp.cpu().numpy()
            print(f"    body_pose shape: {bp.shape}, range: [{bp.min():.3f}, {bp.max():.3f}]")
            if np.isnan(bp).any():
                print("    WARNING: NaN values in body_pose!")
            if np.abs(bp).max() > 10.0:
                print(f"    WARNING: Extreme values in body_pose (max abs: {np.abs(bp).max():.1f})")
            else:
                print(f"    PASS: body_pose values look reasonable (max abs: {np.abs(bp).max():.3f})")

        if 'transl' in smplx_dict:
            tr = smplx_dict['transl']
            if torch.is_tensor(tr):
                tr = tr.cpu().numpy()
            print(f"    transl shape: {tr.shape}, range: [{tr.min():.3f}, {tr.max():.3f}]")
            if np.abs(tr).max() > 50.0:
                print(f"    WARNING: Extreme translation (max abs: {np.abs(tr).max():.1f})")
            else:
                print(f"    PASS: transl values look reasonable")

    print(f"\n{'='*80}")
    print(f"Done! Results saved to {output_dir}/")
    print(f"Visualize at: http://21.6.58.73:8080/")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
