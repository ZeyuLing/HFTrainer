#!/usr/bin/env python3
"""Debug: trace the fix_first_chunk logic to understand why spikes persist."""
import torch, numpy as np, sys, os
sys.path.insert(0, '/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from mmengine.config import Config
from hftrainer.registry import MODEL_BUNDLES
from hftrainer.pipelines.motion.prism_pipeline import PrismPipeline
from einops import rearrange
from hftrainer.models.motion.components.utils.geometry.rotation_convert import rotation_6d_to_axis_angle


def main():
    config_path = 'configs/prism/prism_1b_tp2m_multiframe_kt_spectral_unified.py'
    ckpt_dir = 'work_dirs/prism_1b_tp2m_multiframe_kt_spectral/checkpoint-epoch_0'

    print("Loading model...")
    cfg = Config.fromfile(config_path)
    bundle = MODEL_BUNDLES.build(cfg.model)
    ckpt_path = os.path.join(ckpt_dir, 'model.pt')
    if os.path.isfile(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        bundle.load_state_dict_selective(state_dict, strict=False)
    bundle = bundle.eval().cuda()
    pipeline = PrismPipeline(bundle=bundle)
    print("Model loaded.\n")

    backend = pipeline.backend

    # Monkey-patch post_process_motion to add debug tracing
    orig_pp = backend.post_process_motion

    def debug_pp(x_dec, use_static=False, use_smooth=False, normalize=True,
                 fix_first_chunk=True, mocap_framerate=30.0, gender="neutral"):
        print(f"\n{'='*60}")
        print(f"DEBUG post_process_motion called: fix_first_chunk={fix_first_chunk}")

        # Replicate the logic manually to trace
        x_dec_work = rearrange(x_dec, "b t j d -> b t (j d)")
        x_dec_work = backend.smpl_processor.denormalize(x_dec_work)
        transl_abs_rel = x_dec_work[..., :6]
        transl = backend.smpl_processor.inv_convert_transl(transl_abs_rel)
        pred_poses = x_dec_work[..., 6:]

        pred_poses = rearrange(pred_poses, "b t (j d)-> (b t) j d", d=6)
        pred_poses = pred_poses[..., [0, 2, 4, 1, 3, 5]]
        pred_poses = rotation_6d_to_axis_angle(pred_poses)
        pred_poses = rearrange(pred_poses, "(b t) j d -> b t (j d)", b=1)

        T = pred_poses.shape[1]
        scale = backend.vae_scale_factor_temporal
        min_fix = 3 * scale
        max_fix = min(T // 3, 10 * scale)

        print(f"  T={T}, scale={scale}, min_fix={min_fix}, max_fix={max_fix}")
        print(f"  T >= min_fix+16: {T >= min_fix + 16}")

        if T >= min_fix + 16:
            # Compute velocity BEFORE fix
            diffs = pred_poses[:, 1:] - pred_poses[:, :-1]
            vel = diffs.norm(dim=-1).squeeze(0)

            stable_start = max(max_fix + 8, int(T * 0.6))
            stable_vel_median = vel[stable_start:].median().item()
            spike_threshold = 2.0 * stable_vel_median

            print(f"  stable_start={stable_start}, stable_vel_median={stable_vel_median:.4f}")
            print(f"  spike_threshold={spike_threshold:.4f}")

            vel_first = ' '.join(['%.3f' % v.item() for v in vel[:30]])
            print(f"  vel BEFORE fix [0:30]: {vel_first}")

            # Scan for spikes
            n_fix = min_fix
            spikes_found = []
            for i in range(min_fix, min(max_fix, len(vel))):
                if vel[i] > spike_threshold:
                    n_fix = i + 2
                    spikes_found.append((i, vel[i].item()))
            n_fix = min(n_fix, max_fix)

            print(f"  Spikes in scan range [{min_fix},{min(max_fix,len(vel))}]: {spikes_found[:10]}")
            print(f"  n_fix={n_fix}")

            if n_fix > 0 and T > n_fix + 4:
                anchor_idx = n_fix
                n_ref = min(16, T - anchor_idx - 1)
                n_ref = max(n_ref, 1)
                anchor = pred_poses[:, anchor_idx]
                ref_vel = (pred_poses[:, anchor_idx + n_ref] - pred_poses[:, anchor_idx]) / n_ref
                ref_vel_norm = ref_vel.norm().item()

                n_blend = min(8, n_fix // 2)
                n_hard = n_fix - n_blend

                print(f"  anchor_idx={anchor_idx}, n_ref={n_ref}, ref_vel_norm={ref_vel_norm:.4f}")
                print(f"  n_hard={n_hard}, n_blend={n_blend}")
                print(f"  Expected vel in hard zone: {ref_vel_norm:.4f}")

        # Now call original to get actual result
        result = orig_pp(x_dec, use_static=use_static, use_smooth=use_smooth,
                        normalize=normalize, fix_first_chunk=fix_first_chunk,
                        mocap_framerate=mocap_framerate, gender=gender)

        # Check output velocity
        bp = result['body_pose']
        if isinstance(bp, torch.Tensor):
            bp_np = bp.numpy()
        else:
            bp_np = bp
        T_out = bp_np.shape[0]
        bp_flat = bp_np.reshape(T_out, -1)
        diffs_out = np.diff(bp_flat, axis=0)
        vel_out = np.linalg.norm(diffs_out, axis=1)
        vel_out_str = ' '.join(['%.3f' % v for v in vel_out[:30]])
        print(f"  vel AFTER fix [0:30]: {vel_out_str}")
        ratio = vel_out[0] / (vel_out[15:].mean() + 1e-8)
        print(f"  vel[0]/mean(15+) = {ratio:.2f}x")
        if ratio > 3.0:
            print(f"  STILL HAS SPIKE!")
        else:
            print(f"  OK - no spike")

        return result

    backend.post_process_motion = debug_pp

    # Test with seed 42 (known spike case from prior testing)
    for seed in [42, 7, 123]:
        print(f"\n\n{'#'*60}")
        print(f"### SEED={seed} ###")
        torch.manual_seed(seed)
        with torch.no_grad():
            result = pipeline(
                prompts="A person walks forward slowly and then turns left.",
                num_frames_per_segment=197,
                num_inference_steps=50,
                guidance_scale=5.0,
            )


if __name__ == '__main__':
    main()
