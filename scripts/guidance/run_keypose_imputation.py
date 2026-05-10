#!/usr/bin/env python3
"""Keypose imputation with local_edit + flow_interp.

Based on eval report findings: best config is
  uncond_jit_man + local_edit + flow_interp

Strategy:
- local_edit: only mask a window around each keypose (±EDIT_RADIUS),
  keep everything outside as observed. The model only generates the
  transition region, not the entire motion.
- flow_interp: known regions follow flow-matching interpolation path
  ((1-t)*z0 + t*x_clean) during ODE — train-consistent for MAN models.
- Also try: anchor_inbetween (mask everything except first+last+keypose)

Compare against pure_blend baseline.
"""
import argparse, json, os, sys, time, traceback
from pathlib import Path
import numpy as np, torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger('kp_impute')

from scripts.eval_keyframe_pose_guidance import (
    load_before_after_pairs, select_keyposes, compute_metrics,
    find_latest_checkpoint, load_m2m_bundle,
    NUM_KEYPOSES, MIN_KEYPOSE_DIFF, BEFORE_DIR, AFTER_DIR, MAN_MODELS, D,
)
from scripts.run_pure_blend_baseline import pure_blend


def build_local_edit_batch(before_motion, after_motion, keypose_indices, edit_radius=30):
    """local_edit: mask ±edit_radius frames around each keypose, keep rest observed.

    Keypose frames themselves are observed (mask=0).
    """
    T = before_motion.shape[0]
    composite = before_motion.copy()
    src_mask = np.zeros((T, D), dtype=np.float32)  # 0 = observed

    for ki in keypose_indices:
        # Mask frames in [ki-edit_radius, ki+edit_radius] EXCEPT ki itself
        for f in range(max(0, ki - edit_radius), min(T, ki + edit_radius + 1)):
            if f != ki:
                src_mask[f] = 1.0
        # Set keypose frame from after
        composite[ki] = after_motion[ki].copy()

    return {
        'composite_motion': composite,
        'src_mask': src_mask,
        'before_motion': before_motion,
        'after_motion': after_motion,
        'keypose_indices': keypose_indices,
        'num_frames': T,
    }


def build_anchor_inbetween_batch(before_motion, after_motion, keypose_indices):
    """anchor_inbetween: keep first + last + keypose frames, mask rest."""
    T = before_motion.shape[0]
    composite = before_motion.copy()
    src_mask = np.ones((T, D), dtype=np.float32)  # 1 = to-generate

    # Anchor: first and last frames observed
    src_mask[0] = 0.0
    src_mask[-1] = 0.0
    # Keypose frames observed
    for ki in keypose_indices:
        composite[ki] = after_motion[ki].copy()
        src_mask[ki] = 0.0

    return {
        'composite_motion': composite,
        'src_mask': src_mask,
        'before_motion': before_motion,
        'after_motion': after_motion,
        'keypose_indices': keypose_indices,
        'num_frames': T,
    }


@torch.no_grad()
def run_imputation(bundle, batch_info, device, rep_mode='flow_interp', num_steps=50):
    """Run M2M imputation pipeline."""
    from hftrainer.pipelines.motion.hymotion_m2m_pipeline import HyMotionM2MPipeline

    pipeline = HyMotionM2MPipeline(
        bundle=bundle, num_steps=num_steps, replacement_guidance=rep_mode,
    )

    composite = torch.from_numpy(batch_info['composite_motion']).float().unsqueeze(0).to(device)
    src_mask = torch.from_numpy(batch_info['src_mask']).float().unsqueeze(0).to(device)
    before = torch.from_numpy(batch_info['before_motion']).float().unsqueeze(0).to(device)
    T = batch_info['num_frames']

    normalized_composite = bundle.normalize_motion(composite)
    # clean_motion = full composite (with keypose replaced), NOT zeroed
    clean_motion = normalized_composite.clone()
    # VACE input: zero out masked regions
    vace_input = normalized_composite * (1 - src_mask)

    infer_batch = {
        'src_motion': vace_input,
        'src_mask': src_mask,
        'src_length': [T],
        'tgt_length': [T],
        'clean_motion': clean_motion,
    }

    result = pipeline(infer_batch)
    output_denorm = bundle.denormalize_motion(result['latent'])

    # Merge: keep observed from composite, generated from model
    final = composite * (1 - src_mask) + output_denorm * src_mask
    # Preserve translation
    final[:, :, :3] = before[:, :, :3]
    return final.squeeze(0).cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num-cases', type=int, default=None)
    parser.add_argument('--num-steps', type=int, default=50)
    parser.add_argument('--model', default='uncond_fm_man',
                        choices=['uncond_fm_man', 'uncond_jit_man'])
    args = parser.parse_args()

    device = f'cuda:{args.gpu}'
    before_dir = os.path.join(str(PROJECT_ROOT), BEFORE_DIR)
    after_dir = os.path.join(str(PROJECT_ROOT), AFTER_DIR)
    pairs = load_before_after_pairs(before_dir, after_dir, max_pairs=args.num_cases)
    if not pairs:
        return

    # Find the right model
    model_map = {
        'uncond_fm_man': ('uncond_fm_man',
                          'configs/hymotion_m2m/hymotion_m2m_completion_uncond_fm_man_046b.py',
                          'work_dirs/hymotion_m2m_completion_uncond_fm_man_046b'),
        'uncond_jit_man': ('uncond_jit_man',
                           'configs/hymotion_m2m/hymotion_m2m_completion_uncond_jit_man_046b.py',
                           'work_dirs/hymotion_m2m_completion_uncond_jit_man_046b'),
    }
    model_name, config_path, work_dir = model_map[args.model]
    ckpt_path = find_latest_checkpoint(os.path.join(str(PROJECT_ROOT), work_dir))
    if not ckpt_path:
        logger.error(f'No checkpoint for {model_name}')
        return
    bundle = load_m2m_bundle(os.path.join(str(PROJECT_ROOT), config_path), ckpt_path, device=device)

    # Configs to evaluate
    configs = [
        ('local_edit_flow_interp', 'flow_interp', build_local_edit_batch, {'edit_radius': 30}),
        ('local_edit_skip_last',   'skip_last',   build_local_edit_batch, {'edit_radius': 30}),
        ('anchor_inbetween_flow_interp', 'flow_interp', build_anchor_inbetween_batch, {}),
        ('anchor_inbetween_skip_last',   'skip_last',   build_anchor_inbetween_batch, {}),
    ]

    base_dir = PROJECT_ROOT / "output" / "eval_keyframe_pose_v3" / "local_rot"

    for config_name, rep_mode, batch_fn, batch_kwargs in configs:
        variant_name = f'{model_name}__{config_name}'
        variant_dir = base_dir / variant_name
        variant_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f'\n=== {variant_name} ===')
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
                batch_info = batch_fn(before_motion, after_motion, kp_indices, **batch_kwargs)

                t0 = time.time()
                output = run_imputation(bundle, batch_info, device, rep_mode, args.num_steps)
                elapsed = time.time() - t0

                # Force exact keypose
                for ki in kp_indices:
                    output[ki, 3:] = after_motion[ki, 3:]

                metrics = compute_metrics(output, before_motion, after_motion, kp_indices, batch_info['src_mask'])

                # Compute equiv_frames (simple: keypose ± edit_radius)
                equiv = set()
                for ki in kp_indices:
                    for f in range(max(0, ki - 30), min(T, ki + 31)):
                        equiv.add(f)

                np.savez_compressed(
                    str(variant_dir / f'{case_key}.npz'),
                    output_motion=output, before_motion=before_motion,
                    after_motion=after_motion, composite_motion=batch_info['composite_motion'],
                    src_mask=batch_info['src_mask'],
                    keypose_indices=np.array(kp_indices),
                    equiv_frames=np.array(sorted(equiv)),
                    correction_diffs=diffs,
                )

                results.append({
                    'case_key': case_key, 'filename': pair['filename'],
                    'num_frames': T, 'keypose_indices': kp_indices,
                    'elapsed_sec': elapsed, **metrics,
                })

                logger.info(f'  {case_key}: kf={metrics["kf_mpjpe"]:.4f} glob={metrics["global_mpjpe"]:.4f} '
                            f'src={metrics["src_mpjpe"]:.4f} bnd={metrics["boundary_smoothness"]:.4f} '
                            f'smooth={metrics["overall_smoothness"]:.4f} ({elapsed:.1f}s)')

            except Exception as e:
                logger.error(f'  {case_key}: FAILED - {e}')
                traceback.print_exc()

        if results:
            metric_keys = ['kf_mpjpe', 'global_mpjpe', 'src_mpjpe', 'boundary_smoothness',
                           'overall_smoothness', 'foot_skating']
            agg = {f'{mk}_mean': float(np.mean([r[mk] for r in results])) for mk in metric_keys}
            logger.info(f'\n  -> {variant_name} (N={len(results)}):')
            for k, v in agg.items():
                logger.info(f'     {k}: {v:.4f}')
            with open(variant_dir / 'results.json', 'w') as f:
                json.dump({'aggregate': agg, 'cases': results}, f, indent=2)

            # Update summary
            summary_path = base_dir / "eval_summary.json"
            if summary_path.exists():
                with open(summary_path) as f:
                    summary = json.load(f)
                summary['comparison'] = [c for c in summary.get('comparison', []) if c.get('variant') != variant_name]
                summary['comparison'].append({
                    'variant': variant_name, 'model': model_name,
                    'imp_mode': config_name, 'rep_mode': rep_mode,
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

    print('\nDone!')


if __name__ == '__main__':
    main()
