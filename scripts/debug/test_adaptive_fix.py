#!/usr/bin/env python3
"""Test the adaptive first-frame velocity spike fix with proper discontinuity metric."""
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
    else:
        from safetensors.torch import load_file
        st_path = os.path.join(ckpt_dir, 'model.safetensors')
        st_dict = load_file(st_path)
        bundle.transformer.load_state_dict(st_dict, strict=False)
    bundle = bundle.eval().to(device)
    pipeline = PrismPipeline(bundle=bundle)
    print("Model loaded.\n")

    # Test with multiple seeds and lengths
    test_cases = [
        (42, 197, "A person walks forward slowly and then turns left."),
        (123, 301, "A person walks forward slowly and then turns left."),
        (7, 81, "A man jumps up."),
        (100, 250, "A person waves their hand and walks backward."),
        (200, 150, "Someone does a cartwheel."),
        (300, 120, "A person stands up from sitting."),
        (500, 180, "A man kicks his right leg forward."),
    ]

    pass_count = 0
    fail_count = 0

    for seed, num_frames, prompt in test_cases:
        print(f"=== seed={seed}, T={num_frames} ===")
        torch.manual_seed(seed)
        with torch.no_grad():
            smplx_dict = pipeline(
                prompts=prompt,
                num_frames_per_segment=num_frames,
                num_inference_steps=50,
                guidance_scale=5.0,
            )

        bp = smplx_dict['body_pose']
        if isinstance(bp, torch.Tensor):
            bp = bp.numpy()
        T = bp.shape[0]
        bp_flat = bp.reshape(T, -1)

        # Compute velocity (frame-to-frame)
        diffs = np.diff(bp_flat, axis=0)
        vel = np.linalg.norm(diffs, axis=1)

        # Compute acceleration (velocity change between adjacent frames)
        # This is the TRUE spike metric: a sudden discontinuity
        accel = np.abs(np.diff(vel))

        # Metrics:
        # 1. Max acceleration in first 25 frames (jerk = spike indicator)
        # 2. Compare to median acceleration in stable region
        stable_start = max(30, T // 2)
        stable_accel_median = np.median(accel[stable_start:])
        max_accel_first25 = accel[:25].max() if len(accel) > 25 else accel.max()
        accel_ratio = max_accel_first25 / (stable_accel_median + 1e-8)

        # 3. Also check: is the velocity constant in the extrapolated region?
        # (std of velocity in first 12 frames should be very low)
        vel_std_first12 = vel[:12].std()
        vel_mean_first12 = vel[:12].mean()
        cv_first12 = vel_std_first12 / (vel_mean_first12 + 1e-8)  # coefficient of variation

        # Print velocity profile
        vel_str = ' '.join([f'{v:.2f}' for v in vel[:25]])
        print(f"  Vel[0:25]: {vel_str}")
        print(f"  First12 mean={vel_mean_first12:.3f}, std={vel_std_first12:.4f}, CV={cv_first12:.3f}")
        print(f"  Max accel (first25): {max_accel_first25:.3f}")
        print(f"  Stable accel median: {stable_accel_median:.3f}")
        print(f"  Accel ratio: {accel_ratio:.2f}x")

        # PASS criteria:
        # - CV of first 12 frames < 0.1 (velocity is constant in extrapolated region)
        # - Accel ratio < 5x (no extreme discontinuity at boundary)
        has_spike = False
        reasons = []
        if cv_first12 > 0.1:
            has_spike = True
            reasons.append(f"CV={cv_first12:.3f}>0.1")
        if accel_ratio > 5.0:
            has_spike = True
            reasons.append(f"accel_ratio={accel_ratio:.2f}>5.0")

        if has_spike:
            print(f"  FAIL: {', '.join(reasons)}\n")
            fail_count += 1
        else:
            print(f"  PASS\n")
            pass_count += 1

    print(f"\n{'='*50}")
    print(f"Results: {pass_count}/{pass_count+fail_count} passed, {fail_count} failed")
    if fail_count == 0:
        print("All tests PASSED! Adaptive fix is working correctly.")
    else:
        print(f"Issues in {fail_count} cases - investigate further.")


if __name__ == '__main__':
    main()
