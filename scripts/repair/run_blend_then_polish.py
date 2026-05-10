#!/usr/bin/env python3
"""Blend-then-Polish experiment: pure_blend + model smoothing.

Takes the pure_blend output (which already looks good) and runs a
light model pass to smooth the transition boundaries where blend
weight drops to 0.

Strategy: feed the pure_blend result as clean_motion to the M2M pipeline.
The model sees it via VACE conditioning and replacement guidance,
and only "polishes" the generated (mask=1) regions.
"""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger('blend_polish')

from scripts.eval_keyframe_pose_guidance import (
    load_before_after_pairs, select_keyposes, build_imputation_batch,
    compute_metrics, find_latest_checkpoint, load_m2m_bundle,
    NUM_KEYPOSES, MIN_KEYPOSE_DIFF, BEFORE_DIR, AFTER_DIR, MAN_MODELS,
)
from scripts.run_pure_blend_baseline import pure_blend


@torch.no_grad()
def blend_then_polish(
    bundle, before_motion, after_motion, keypose_indices,
    batch_info, device='cuda:0', num_steps=50, sdedit_strength=0.15,
):
    """Pure blend → global SDEdit polish (no mask).

    1. Run pure_blend to get the well-corrected motion
    2. Global SDEdit: start from (1-s)*blended + s*noise, denoise entire motion
       No mask, no replacement guidance — model smooths everything uniformly
    3. Light postprocess: reinforce keypose precision
    """
    from hftrainer.pipelines.motion.hymotion_m2m_pipeline import HyMotionM2MPipeline

    # Step 1: pure blend
    blended, equiv_info = pure_blend(before_motion, after_motion, keypose_indices)

    # Step 2: global SDEdit — manual ODE with no mask
    T = batch_info['num_frames']
    blended_t = torch.from_numpy(blended).float().unsqueeze(0).to(device)
    normalized_blended = bundle.normalize_motion(blended_t)

    # VACE context: use blended as the full observed input (mask=0 everywhere)
    # This tells the model the entire motion is "context"
    all_zeros_mask = torch.zeros_like(blended_t)
    vace_context = bundle.prepare_vace_input(
        src_motion=normalized_blended,
        ref_pose=None,
        src_mask=all_zeros_mask,
    )

    # Text: unconditional
    B = 1
    vtxt_input = bundle.null_vtxt_feat.expand(B, 1, -1)
    ctxt_input = bundle.null_ctxt_input.expand(B, 1, -1)
    ctxt_length = torch.tensor([1], device=device).expand(B)

    from hftrainer.pipelines.motion.hymotion_m2m_pipeline import _length_to_mask
    tgt_padding_mask = _length_to_mask(
        torch.tensor([T], dtype=torch.long, device=device), T
    )
    ctxt_mask_temporal = _length_to_mask(ctxt_length, 1).expand(B, -1)

    D = normalized_blended.shape[-1]

    def fn(t_val, x):
        x_input = torch.cat([x, vace_context], dim=-1)
        x_pred = bundle.predict_flow(
            x_input=x_input,
            ctxt_input=ctxt_input,
            vtxt_input=vtxt_input,
            timesteps=t_val.expand(x_input.shape[0]),
            x_mask_temporal=tgt_padding_mask,
            ctxt_mask_temporal=ctxt_mask_temporal,
        )
        if bundle.pred_type == 'x1':
            t_eps = 0.05
            x_pred = (x_pred - x) / (1.0 - t_val).clamp_min(t_eps)
        return x_pred

    # SDEdit: start ODE from t=sdedit_strength instead of t=0
    # y0 = (1-s)*x1 + s*noise (interpolation at t=s on the flow)
    z = torch.randn(B, T, D, device=device, dtype=normalized_blended.dtype)
    s = sdedit_strength
    y0 = (1 - s) * normalized_blended + s * z

    t_grid = torch.linspace(s, 1.0, num_steps + 1, device=device, dtype=normalized_blended.dtype)
    x = y0
    for i in range(len(t_grid) - 1):
        dt = t_grid[i + 1] - t_grid[i]
        v = fn(t_grid[i], x)
        x = x + v * dt

    sampled = x
    result_dict = bundle.decode_motion_from_latent(sampled)
    output_denorm = bundle.denormalize_motion(sampled)
    model_output = output_denorm.squeeze(0).cpu().numpy()

    # Preserve translation from blend
    model_output[:, :3] = blended[:, :3]

    # Step 3: reinforce keypose neighborhood from pure_blend
    output = model_output.copy()
    sorted_kp = sorted(keypose_indices)
    boundaries = [0] + sorted_kp + [T - 1]
    for idx_i, ki in enumerate(sorted_kp):
        left_dist = ki - boundaries[idx_i]
        right_dist = boundaries[idx_i + 2] - ki
        half_gap = min(left_dist, right_dist) // 2
        RADIUS = max(min(half_gap, 15), 3)
        for f in range(max(0, ki - RADIUS), min(T, ki + RADIUS + 1)):
            d = abs(f - ki)
            t_w = d / (RADIUS + 1)
            w = 0.5 * (1 + np.cos(np.pi * t_w))
            output[f, 3:] = w * blended[f, 3:] + (1 - w) * model_output[f, 3:]

    for ki in keypose_indices:
        output[ki, 3:] = after_motion[ki, 3:]

    return output, blended, equiv_info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num-cases', type=int, default=None)
    parser.add_argument('--num-steps', type=int, default=50)
    args = parser.parse_args()

    device = f'cuda:{args.gpu}'
    before_dir = os.path.join(str(PROJECT_ROOT), BEFORE_DIR)
    after_dir = os.path.join(str(PROJECT_ROOT), AFTER_DIR)
    pairs = load_before_after_pairs(before_dir, after_dir, max_pairs=args.num_cases)

    if not pairs:
        logger.error('No pairs loaded')
        return

    # Load M2M model (local rot only)
    model_name, config_path, work_dir, rot_space = MAN_MODELS[0]
    ckpt_path = find_latest_checkpoint(os.path.join(str(PROJECT_ROOT), work_dir))
    bundle = load_m2m_bundle(
        os.path.join(str(PROJECT_ROOT), config_path), ckpt_path, device=device,
    )

    output_dir = PROJECT_ROOT / "output" / "eval_keyframe_pose_v3" / "local_rot" / "blend_then_polish"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for case_idx, pair in enumerate(pairs):
        before_motion = pair['before_motion']
        after_motion = pair['after_motion']
        T = pair['num_frames']

        kp_indices, diffs = select_keyposes(
            before_motion, after_motion, k=NUM_KEYPOSES, min_diff=MIN_KEYPOSE_DIFF,
        )
        case_key = f'case{case_idx:03d}_{pair["filename"].replace(".npz", "")}'

        try:
            batch_info = build_imputation_batch(
                before_motion, after_motion, kp_indices, mode='keyframe_only',
            )

            t0 = time.time()
            output, blended, equiv_info = blend_then_polish(
                bundle, before_motion, after_motion, kp_indices,
                batch_info, device=device, num_steps=args.num_steps,
            )
            elapsed = time.time() - t0

            metrics = compute_metrics(
                output, before_motion, after_motion, kp_indices, batch_info['src_mask'],
            )

            np.savez_compressed(
                str(output_dir / f'{case_key}.npz'),
                output_motion=output,
                before_motion=before_motion,
                after_motion=after_motion,
                composite_motion=batch_info['composite_motion'],
                src_mask=batch_info['src_mask'],
                keypose_indices=np.array(kp_indices),
                equiv_frames=np.array(sorted(set(sum(equiv_info.values(), [])))),
                correction_diffs=diffs,
            )

            results.append({
                'case_key': case_key, 'filename': pair['filename'],
                'num_frames': T, 'keypose_indices': kp_indices,
                'elapsed_sec': elapsed, **metrics,
            })

            logger.info(
                f'  {case_key}: kf={metrics["kf_mpjpe"]:.4f} glob={metrics["global_mpjpe"]:.4f} '
                f'src={metrics["src_mpjpe"]:.4f} smooth={metrics["overall_smoothness"]:.4f} ({elapsed:.1f}s)'
            )
        except Exception as e:
            logger.error(f'  {case_key}: FAILED - {e}')
            traceback.print_exc()

    # Aggregate
    metric_keys = ['kf_mpjpe', 'global_mpjpe', 'src_mpjpe', 'boundary_smoothness',
                   'overall_smoothness', 'foot_skating']
    agg = {}
    for mk in metric_keys:
        vals = [r[mk] for r in results if mk in r]
        agg[f'{mk}_mean'] = float(np.mean(vals))

    print(f'\n=== BLEND-THEN-POLISH (N={len(results)}) ===')
    for k, v in agg.items():
        print(f'  {k}: {v:.4f}')

    with open(output_dir / 'results.json', 'w') as f:
        json.dump({'aggregate': agg, 'cases': results}, f, indent=2)

    # Update eval_summary
    summary_path = PROJECT_ROOT / "output" / "eval_keyframe_pose_v3" / "local_rot" / "eval_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        summary['comparison'] = [c for c in summary.get('comparison', []) if c.get('variant') != 'blend_then_polish']
        summary['comparison'].append({
            'variant': 'blend_then_polish',
            'model': model_name, 'imp_mode': 'keyframe_only', 'rep_mode': 'skip_last',
            'sdedit_strength': 0.0, 'rotation_space': 'local', 'checkpoint': os.path.basename(ckpt_path),
            'n_cases': len(results),
            'kf_mpjpe': agg['kf_mpjpe_mean'], 'global_mpjpe': agg['global_mpjpe_mean'],
            'src_mpjpe': agg['src_mpjpe_mean'], 'bnd_smooth': agg['boundary_smoothness_mean'],
            'overall_smooth': agg['overall_smoothness_mean'], 'foot_skate': agg['foot_skating_mean'],
            'time_sec': np.mean([r['elapsed_sec'] for r in results]),
        })
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

    print(f'\nSaved to {output_dir}')


if __name__ == '__main__':
    main()
