#!/usr/bin/env python3
"""Hybrid: Pure Blend + Boundary-only Model Polish.

Best of both worlds:
- Pure blend propagates keypose correction (proven good)
- Model only smooths the narrow band where blend weight drops to 0
- Everything else kept as pure blend output

This avoids the model destroying blend quality while leveraging
its ability to generate natural transitions at discontinuities.
"""
import argparse, json, os, sys, time, traceback
from pathlib import Path
import numpy as np, torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger('hybrid')

from scripts.eval_keyframe_pose_guidance import (
    load_before_after_pairs, select_keyposes, compute_metrics,
    find_latest_checkpoint, load_m2m_bundle,
    NUM_KEYPOSES, MIN_KEYPOSE_DIFF, BEFORE_DIR, AFTER_DIR, MAN_MODELS, D,
)
from scripts.run_pure_blend_baseline import pure_blend


BOUNDARY_BAND = 8  # ±8 frames around each blend boundary


@torch.no_grad()
def hybrid_blend_polish(bundle, pipeline, before, after, kp_indices, device):
    """Pure blend + boundary-only model polish."""
    T = before.shape[0]

    # Step 1: pure blend
    blended, equiv_info = pure_blend(before, after, kp_indices)

    # Step 2: build mask — only the narrow band at blend boundaries
    sorted_kp = sorted(kp_indices)
    boundaries_list = [0] + sorted_kp + [T - 1]
    src_mask = np.zeros((T, D), dtype=np.float32)

    for i_kp, ki in enumerate(sorted_kp):
        left_dist = ki - boundaries_list[i_kp]
        right_dist = boundaries_list[i_kp + 2] - ki
        max_r = min(left_dist, right_dist) // 2
        TR = max(min(max_r, 40), 8)  # pure_blend's TEMPORAL_RADIUS
        # Boundaries are at ki-TR and ki+TR
        for bnd in [ki - TR, ki + TR]:
            for f in range(max(0, bnd - BOUNDARY_BAND), min(T, bnd + BOUNDARY_BAND + 1)):
                src_mask[f] = 1.0

    # Keypose frames always observed
    for ki in kp_indices:
        src_mask[ki] = 0.0

    # Step 3: model polish on boundary band only
    composite = blended.copy()
    for ki in kp_indices:
        composite[ki] = after[ki].copy()

    composite_t = torch.from_numpy(composite).float().unsqueeze(0).to(device)
    src_mask_t = torch.from_numpy(src_mask).float().unsqueeze(0).to(device)
    blended_t = torch.from_numpy(blended).float().unsqueeze(0).to(device)

    norm_composite = bundle.normalize_motion(composite_t)
    norm_blended = bundle.normalize_motion(blended_t)
    vace_input = norm_composite * (1 - src_mask_t)

    batch = {
        'src_motion': vace_input, 'src_mask': src_mask_t,
        'src_length': [T], 'tgt_length': [T],
        'clean_motion': norm_blended,
    }

    result = pipeline(batch)
    output_denorm = bundle.denormalize_motion(result['latent'])
    final = composite_t * (1 - src_mask_t) + output_denorm * src_mask_t
    final[:, :, :3] = blended_t[:, :, :3]
    output = final.squeeze(0).cpu().numpy()

    # Force exact keypose
    for ki in kp_indices:
        output[ki, 3:] = after[ki, 3:]

    return output, blended, equiv_info, src_mask


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
        return

    model_name, config_path, work_dir, _ = MAN_MODELS[0]
    ckpt_path = find_latest_checkpoint(os.path.join(str(PROJECT_ROOT), work_dir))
    bundle = load_m2m_bundle(os.path.join(str(PROJECT_ROOT), config_path), ckpt_path, device=device)

    from hftrainer.pipelines.motion.hymotion_m2m_pipeline import HyMotionM2MPipeline
    pipeline = HyMotionM2MPipeline(bundle=bundle, num_steps=args.num_steps, replacement_guidance='skip_last')

    variant_name = 'hybrid_blend_boundary_polish'
    output_dir = PROJECT_ROOT / "output" / "eval_keyframe_pose_v3" / "local_rot" / variant_name
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for case_idx, pair in enumerate(pairs):
        before = pair['before_motion']
        after = pair['after_motion']
        T = pair['num_frames']
        kp, diffs = select_keyposes(before, after, k=NUM_KEYPOSES, min_diff=MIN_KEYPOSE_DIFF)
        case_key = f'case{case_idx:03d}_{pair["filename"].replace(".npz", "")}'

        try:
            t0 = time.time()
            output, blended, equiv_info, src_mask = hybrid_blend_polish(
                bundle, pipeline, before, after, kp, device,
            )
            elapsed = time.time() - t0

            metrics = compute_metrics(output, before, after, kp, src_mask)

            np.savez_compressed(
                str(output_dir / f'{case_key}.npz'),
                output_motion=output, before_motion=before, after_motion=after,
                composite_motion=blended, src_mask=src_mask,
                keypose_indices=np.array(kp),
                equiv_frames=np.array(sorted(set(sum(equiv_info.values(), [])))),
                correction_diffs=diffs,
            )

            results.append({
                'case_key': case_key, 'filename': pair['filename'],
                'num_frames': T, 'keypose_indices': kp,
                'elapsed_sec': elapsed, **metrics,
            })

            logger.info(f'  {case_key}: glob={metrics["global_mpjpe"]:.4f} '
                        f'smooth={metrics["overall_smoothness"]:.4f} '
                        f'bnd={metrics["boundary_smoothness"]:.4f} '
                        f'foot={metrics["foot_skating"]:.4f} ({elapsed:.1f}s)')
        except Exception as e:
            logger.error(f'  {case_key}: FAILED - {e}')
            traceback.print_exc()

    if not results:
        return

    metric_keys = ['kf_mpjpe', 'global_mpjpe', 'src_mpjpe', 'boundary_smoothness',
                   'overall_smoothness', 'foot_skating']
    agg = {f'{mk}_mean': float(np.mean([r[mk] for r in results])) for mk in metric_keys}

    print(f'\n=== HYBRID (N={len(results)}) ===')
    for k, v in agg.items():
        print(f'  {k}: {v:.4f}')

    with open(output_dir / 'results.json', 'w') as f:
        json.dump({'aggregate': agg, 'cases': results}, f, indent=2)

    # Update summary
    summary_path = output_dir.parent / "eval_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        summary['comparison'] = [c for c in summary.get('comparison', []) if c.get('variant') != variant_name]
        summary['comparison'].append({
            'variant': variant_name, 'model': model_name,
            'imp_mode': 'hybrid_boundary', 'rep_mode': 'skip_last',
            'sdedit_strength': 0.0, 'rotation_space': 'local',
            'checkpoint': os.path.basename(ckpt_path),
            'n_cases': len(results),
            'kf_mpjpe': agg['kf_mpjpe_mean'], 'global_mpjpe': agg['global_mpjpe_mean'],
            'src_mpjpe': agg['src_mpjpe_mean'], 'bnd_smooth': agg['boundary_smoothness_mean'],
            'overall_smooth': agg['overall_smoothness_mean'], 'foot_skate': agg['foot_skating_mean'],
            'time_sec': float(np.mean([r['elapsed_sec'] for r in results])),
        })
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

    print(f'\nSaved to {output_dir}')


if __name__ == '__main__':
    main()
