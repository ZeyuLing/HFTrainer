#!/usr/bin/env python3
"""Compare fixed vs unfixed velocity profiles for validation.
Directly calls post_process_motion to bypass __call__ signature."""
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

    # Test the problematic cases with and without fix
    test_cases = [
        (7, 81, "A man jumps up."),
        (42, 197, "A person walks forward slowly and then turns left."),
        (123, 301, "A person walks forward slowly and then turns left."),
    ]

    for seed, num_frames, prompt in test_cases:
        print(f"\n{'='*60}")
        print(f"=== seed={seed}, T={num_frames}: '{prompt}' ===")

        # Generate latent (same for both)
        # We need to intercept the motion before post_process.
        # Monkey-patch post_process to save raw motion
        backend = pipeline.backend
        raw_motions = []
        orig_post_process = backend.post_process_motion

        def capture_post_process(motion_pred, **kwargs):
            raw_motions.append(motion_pred.clone())
            return orig_post_process(motion_pred, **kwargs)

        # Generate WITH fix (default)
        backend.post_process_motion = capture_post_process
        torch.manual_seed(seed)
        with torch.no_grad():
            result_fixed = pipeline(
                prompts=prompt,
                num_frames_per_segment=num_frames,
                num_inference_steps=50,
                guidance_scale=5.0,
            )

        # Now decode WITHOUT fix using the same raw motion
        raw_motion = raw_motions[0]  # captured before post-processing
        with torch.no_grad():
            result_unfixed = orig_post_process(
                raw_motion.clone(),
                use_static=False,
                use_smooth=False,
                normalize=True,
                fix_first_chunk=False,
                mocap_framerate=30.0,
                gender="neutral",
            )

        # Restore
        backend.post_process_motion = orig_post_process

        for label, result in [("FIXED", result_fixed), ("UNFIXED", result_unfixed)]:
            bp = result['body_pose']
            if isinstance(bp, torch.Tensor):
                bp = bp.numpy()
            T = bp.shape[0]
            bp_flat = bp.reshape(T, -1)
            diffs = np.diff(bp_flat, axis=0)
            vel = np.linalg.norm(diffs, axis=1)

            # Print first 30 frames of velocity
            vel_str = ' '.join([f'{v:.2f}' for v in vel[:30]])
            print(f"\n  [{label}] Vel[0:30]: {vel_str}")
            print(f"  [{label}] Mean vel: {vel.mean():.3f}, Max vel: {vel.max():.3f}")
            print(f"  [{label}] Vel[0]={vel[0]:.3f}")

            # The key metric: first-frame velocity ratio
            if T > 30:
                ratio_first = vel[0] / (vel[15:].mean() + 1e-8)
                print(f"  [{label}] vel[0]/mean(15+) = {ratio_first:.2f}x")

    print("\n\nConclusion: Compare UNFIXED vel[0:5] vs FIXED vel[0:5].")
    print("UNFIXED should show large spikes; FIXED should be smooth.")


if __name__ == '__main__':
    main()
