#!/usr/bin/env python3
"""Quick one-sample test to verify fix_first_chunk is actually applied."""
import os, sys
sys.path.insert(0, '/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import numpy as np


def main():
    device = torch.device('cuda:0')
    from mmengine.config import Config
    from hftrainer.registry import MODEL_BUNDLES
    from hftrainer.pipelines.motion.prism_pipeline import PrismPipeline

    config_path = 'configs/prism/prism_1b_tp2m_multiframe_kt_spectral_unified.py'
    ckpt_dir = 'work_dirs/prism_1b_tp2m_multiframe_kt_spectral/checkpoint-epoch_0'

    print("Loading model...")
    cfg = Config.fromfile(config_path)
    bundle = MODEL_BUNDLES.build(cfg.model)
    ckpt_path = os.path.join(ckpt_dir, 'model.pt')
    if os.path.isfile(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        bundle.load_state_dict_selective(state_dict, strict=False)
    bundle = bundle.eval().to(device)
    pipeline = PrismPipeline(bundle=bundle)
    print("Model loaded.\n")

    # Generate one sample - should trigger [FIX_DEBUG] prints
    print("Generating sample with seed=42, T=197...")
    torch.manual_seed(42)
    with torch.no_grad():
        result = pipeline(
            prompts="A person walks forward slowly and then turns left.",
            num_frames_per_segment=197,
            num_inference_steps=50,
            guidance_scale=5.0,
        )

    bp = result['body_pose']
    if isinstance(bp, torch.Tensor):
        bp = bp.numpy()
    T = bp.shape[0]
    print(f"\nOutput body_pose shape: {bp.shape}")

    bp_flat = bp.reshape(T, -1)
    diffs = np.diff(bp_flat, axis=0)
    vel = np.linalg.norm(diffs, axis=1)

    vel_str = ' '.join(['%.3f' % v for v in vel[:30]])
    print(f"Vel[0:30]: {vel_str}")
    print(f"Mean vel (15+): {vel[15:].mean():.3f}")
    ratio = vel[0] / (vel[15:].mean() + 1e-8)
    print(f"vel[0]/mean(15+) = {ratio:.2f}x")

    if ratio > 3.0:
        print("\nFAILED - spike still present!")
    else:
        print("\nPASSED - no spike!")


if __name__ == '__main__':
    main()
