"""Run inference through the OFFICIAL PrismARPipeline backend.

This confirms that the standalone script matches the official pipeline.
If both produce deformed output, the model is definitively undertrained.

Usage:
    python3 scripts/inference/run_official_pipeline.py \
        --config configs/prism/prism_1b_tp2m_multiframe.py \
        --checkpoint work_dirs/prism_1b_tp2m_multiframe/checkpoint-iter_15000 \
        --output-dir work_dirs/prism_1b_tp2m_multiframe/eval_official_pipeline
"""

import argparse
import gc
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
from mmengine.config import Config

import hftrainer  # noqa: trigger auto-imports
from hftrainer.registry import MODEL_BUNDLES
from hftrainer.utils.checkpoint_utils import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--guidance-scale', type=float, default=2.0)
    parser.add_argument('--num-steps', type=int, default=50)
    return parser.parse_args()


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

    # Move everything to device
    bundle = bundle.to(device)
    torch.cuda.empty_cache()

    # Use official PrismPipeline
    from hftrainer.pipelines.motion.prism_pipeline import PrismPipeline
    pipeline = PrismPipeline(bundle)

    prompts = [
        "a person walks forward slowly",
        "a person raises both hands above the head",
    ]

    for i, prompt in enumerate(prompts):
        print(f"\n[{i+1}/{len(prompts)}] Generating: '{prompt}'")
        print(f"  guidance_scale={args.guidance_scale}, num_steps={args.num_steps}")

        result = pipeline(
            prompts=prompt,
            num_frames_per_segment=129,
            num_joints=23,
            num_inference_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
        )

        # result should contain smplx_dict or motion data
        print(f"  Result type: {type(result)}")
        if isinstance(result, dict):
            print(f"  Result keys: {list(result.keys())}")
            # Try to save
            smplx_dict = result.get('smplx_dict', result.get('motion', None))
            if smplx_dict is not None and isinstance(smplx_dict, dict):
                out_path = os.path.join(args.output_dir, f"motion_{i:02d}.npz")
                bundle.smpl_pose_processor.save_smplx_npz(out_path, smplx_dict)
                print(f"  Saved: {out_path}")

                # Quick height check
                if 'transl' in smplx_dict:
                    t = smplx_dict['transl']
                    print(f"  Transl shape: {t.shape}, range Y: [{t[:,1].min():.3f}, {t[:,1].max():.3f}]")
            else:
                print(f"  Could not extract smplx_dict from result")
                # Save whatever we got
                out_path = os.path.join(args.output_dir, f"result_{i:02d}.pt")
                torch.save(result, out_path)
                print(f"  Saved raw result: {out_path}")
        else:
            print(f"  Unexpected result type: {type(result)}")


if __name__ == '__main__':
    main()
