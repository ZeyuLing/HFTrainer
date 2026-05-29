#!/usr/bin/env python3
"""Quick PRISM jitter test: guidance_scale comparison on single sample."""
import os
import sys
import numpy as np
import torch

sys.path.insert(0, '/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer')
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from mmengine.config import Config
from hftrainer.registry import MODEL_BUNDLES
from hftrainer.pipelines.motion.prism_pipeline import PrismPipeline


def compute_vel(arr):
    """Mean frame-to-frame L2 velocity."""
    if arr.ndim == 3:
        arr = arr[0]
    return float(np.linalg.norm(np.diff(arr, axis=0), axis=1).mean())


def main():
    device = torch.device('cuda:0')
    config_path = 'configs/prism/prism_1b_tp2m_multiframe_kt_spectral_unified.py'
    ckpt_dir = 'work_dirs/prism_1b_tp2m_multiframe_kt_spectral/checkpoint-epoch_0'

    print("Loading model...")
    cfg = Config.fromfile(config_path)
    bundle = MODEL_BUNDLES.build(cfg.model)

    ckpt_path = os.path.join(ckpt_dir, 'model.pt')
    state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    bundle.load_state_dict_selective(state_dict, strict=False)
    bundle = bundle.eval().to(device)
    print("Model loaded.")

    # Print normalization stats
    proc = bundle.smpl_pose_processor
    print(f"\nNormalization stats:")
    print(f"  mean shape: {proc.mean.shape}, first 6: {proc.mean.flatten()[:6].tolist()}")
    print(f"  std shape: {proc.std.shape}, first 6: {proc.std.flatten()[:6].tolist()}")
    print(f"  min std: {proc.std.min():.6f}")
    print(f"  VAE latents_mean: {bundle.latents_mean.flatten()[:4].tolist()}")
    print(f"  VAE latents_std: {bundle.latents_std.flatten()[:4].tolist()}")

    pipeline = PrismPipeline(bundle=bundle)

    caption = "a person walks forward slowly"
    num_frames = 100

    for gs in [1.0, 2.5, 5.0, 7.5]:
        print(f"\n--- guidance_scale={gs} ---")
        torch.manual_seed(42)
        with torch.no_grad():
            result = pipeline(
                prompts=caption,
                num_frames_per_segment=num_frames,
                num_inference_steps=50,
                guidance_scale=gs,
            )

        if isinstance(result, dict):
            bp = result.get('body_pose')
            tr = result.get('transl')
            if bp is not None:
                if isinstance(bp, torch.Tensor):
                    bp = bp.cpu().numpy()
                print(f"  body_pose: shape={bp.shape}, range=[{bp.min():.3f}, {bp.max():.3f}], vel={compute_vel(bp):.5f}")
            if tr is not None:
                if isinstance(tr, torch.Tensor):
                    tr = tr.cpu().numpy()
                print(f"  transl: shape={tr.shape}, range=[{tr.min():.3f}, {tr.max():.3f}], vel={compute_vel(tr):.5f}")
            # Global orient
            go = result.get('global_orient')
            if go is not None:
                if isinstance(go, torch.Tensor):
                    go = go.cpu().numpy()
                print(f"  global_orient: shape={go.shape}, range=[{go.min():.3f}, {go.max():.3f}], vel={compute_vel(go):.5f}")

    # Also test base model (non-spectral)
    print("\n\n" + "="*70)
    print("BASE MODEL (non-spectral) COMPARISON")
    print("="*70)

    base_config = 'configs/prism/prism_1b_tp2m_multiframe.py'
    base_ckpt = 'work_dirs/prism_1b_tp2m_multiframe/checkpoint-iter_15000'

    if os.path.isfile(os.path.join(base_ckpt, 'model.pt')):
        cfg2 = Config.fromfile(base_config)
        bundle2 = MODEL_BUNDLES.build(cfg2.model)
        sd2 = torch.load(os.path.join(base_ckpt, 'model.pt'), map_location='cpu', weights_only=False)
        bundle2.load_state_dict_selective(sd2, strict=False)
        bundle2 = bundle2.eval().to(device)
        pipeline2 = PrismPipeline(bundle=bundle2)

        for gs in [1.0, 5.0]:
            print(f"\n--- BASE guidance_scale={gs} ---")
            torch.manual_seed(42)
            with torch.no_grad():
                result = pipeline2(
                    prompts=caption,
                    num_frames_per_segment=num_frames,
                    num_inference_steps=50,
                    guidance_scale=gs,
                )
            if isinstance(result, dict):
                bp = result.get('body_pose')
                tr = result.get('transl')
                if bp is not None:
                    if isinstance(bp, torch.Tensor):
                        bp = bp.cpu().numpy()
                    print(f"  body_pose: shape={bp.shape}, range=[{bp.min():.3f}, {bp.max():.3f}], vel={compute_vel(bp):.5f}")
                if tr is not None:
                    if isinstance(tr, torch.Tensor):
                        tr = tr.cpu().numpy()
                    print(f"  transl: shape={tr.shape}, range=[{tr.min():.3f}, {tr.max():.3f}], vel={compute_vel(tr):.5f}")
    else:
        print(f"  Base checkpoint not found: {base_ckpt}/model.pt")


if __name__ == '__main__':
    main()
