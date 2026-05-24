#!/usr/bin/env python3
"""Quick end-to-end test: generate one motion and check velocity profile.

This tests the full pipeline (including the first-frame fix) on a single prompt.
"""
import os, sys
import numpy as np
import torch

sys.path.insert(0, '/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def main():
    device = torch.device('cuda:0')

    from mmengine.config import Config
    from hftrainer.registry import MODEL_BUNDLES
    from hftrainer.pipelines.motion.prism_pipeline import PrismPipeline

    config_path = 'configs/prism/prism_1b_tp2m_multiframe_kt_spectral.py'
    ckpt_dir = 'work_dirs/prism_1b_tp2m_multiframe_kt_spectral/checkpoint-epoch_0'

    print("Loading model...")
    cfg = Config.fromfile(config_path)
    bundle = MODEL_BUNDLES.build(cfg.model)

    ckpt_path = os.path.join(ckpt_dir, 'model.pt')
    if os.path.isfile(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        bundle.load_state_dict_selective(state_dict, strict=False)
    else:
        from safetensors.torch import load_file
        st_path = os.path.join(ckpt_dir, 'model.safetensors')
        st_dict = load_file(st_path)
        bundle.transformer.load_state_dict(st_dict, strict=False)

    bundle = bundle.eval().to(device)
    pipeline = PrismPipeline(bundle=bundle)
    print("Model loaded.")

    # Generate
    prompt = "A person walks forward slowly and then turns left."
    num_frames = 120

    print(f"\nGenerating: '{prompt}' ({num_frames} frames)")
    torch.manual_seed(42)
    with torch.no_grad():
        smplx_dict = pipeline(
            prompts=prompt,
            num_frames_per_segment=num_frames,
            num_inference_steps=50,
            guidance_scale=5.0,
        )

    # Analyze result
    bp = smplx_dict['body_pose']
    go = smplx_dict['global_orient']
    tr = smplx_dict['transl']
    T = bp.shape[0]

    if isinstance(bp, torch.Tensor):
        bp = bp.numpy()
        go = go.numpy()
        tr = tr.numpy()

    print(f"\nResult: T={T}")
    print(f"  body_pose range: [{bp.min():.3f}, {bp.max():.3f}]")
    print(f"  transl range: [{tr.min():.3f}, {tr.max():.3f}]")

    # Per-frame velocity analysis
    bp_joints = bp.reshape(T, -1, 3)
    frame_vel = np.linalg.norm(np.diff(bp_joints, axis=0), axis=2).max(axis=1)
    transl_vel = np.linalg.norm(np.diff(tr, axis=0), axis=1)

    print(f"\n  Body pose velocity:")
    print(f"    mean={frame_vel.mean():.5f}, max={frame_vel.max():.5f}")
    print(f"    first 10 frames: {[f'{v:.4f}' for v in frame_vel[:10]]}")
    print(f"    frames 10-20: {[f'{v:.4f}' for v in frame_vel[10:20]]}")

    # Check first-frame spike
    mean_vel = frame_vel.mean()
    first_vel = frame_vel[0]
    ratio = first_vel / mean_vel
    print(f"\n  First-frame velocity ratio: {ratio:.3f}x mean")

    if ratio > 3.0:
        print(f"  ❌ SPIKE STILL PRESENT (ratio > 3x)")
    elif ratio > 2.0:
        print(f"  ⚠️  Mild spike (ratio > 2x)")
    else:
        print(f"  ✅ No spike (ratio < 2x)")

    print(f"\n  Translation velocity:")
    print(f"    mean={transl_vel.mean():.5f}, max={transl_vel.max():.5f}")
    print(f"    first 5: {[f'{v:.4f}' for v in transl_vel[:5]]}")

    # Save for visual inspection
    out_path = '/tmp/prism_e2e_test_fixed.npz'
    pack = {}
    for k, v in smplx_dict.items():
        if isinstance(v, np.ndarray):
            pack[k] = v.astype(np.float32)
        elif isinstance(v, torch.Tensor):
            pack[k] = v.detach().cpu().numpy().astype(np.float32)
        else:
            pack[k] = v
    np.savez_compressed(out_path, **pack)
    print(f"\n  Saved: {out_path}")


if __name__ == '__main__':
    main()
