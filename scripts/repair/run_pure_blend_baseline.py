#!/usr/bin/env python3
"""Generate pure-blending baseline for keypose evaluation.

No model inference — just applies correction propagation
(temporal + similarity dual-weight) from keypose frames.
This is the "blending-only" approach that reportedly looked good.
"""
import os
import sys
import json
import time
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval_keyframe_pose_guidance import (
    load_before_after_pairs,
    select_keyposes,
    build_imputation_batch,
    compute_metrics,
    NUM_KEYPOSES, MIN_KEYPOSE_DIFF,
    BEFORE_DIR, AFTER_DIR,
)


def pure_blend(
    before_motion: np.ndarray,
    after_motion: np.ndarray,
    keypose_indices: list,
) -> tuple:
    """Pure correction blending: dual-weight (temporal + similarity).

    This is the original postprocess logic that worked well, applied
    directly to before_motion (no model involved).
    """
    T, D = before_motion.shape
    result = before_motion.copy()
    equiv_frames_dict = {}

    sorted_kp = sorted(keypose_indices)
    boundaries = [0] + sorted_kp + [T - 1]
    max_r = 0
    for i, ki_idx in enumerate(sorted_kp):
        left_dist = ki_idx - boundaries[i]
        right_dist = boundaries[i + 2] - ki_idx
        max_r = max(max_r, min(left_dist, right_dist) // 2)
    TEMPORAL_RADIUS = max(min(max_r, 40), 8)

    for ki in keypose_indices:
        correction = after_motion[ki, 3:] - before_motion[ki, 3:]

        # Weight source 1: Temporal proximity (cosine falloff)
        temporal_weight = np.zeros(T, dtype=np.float32)
        for f in range(T):
            d = abs(f - ki)
            if d <= TEMPORAL_RADIUS:
                t = d / (TEMPORAL_RADIUS + 1)
                temporal_weight[f] = 0.5 * (1 + np.cos(np.pi * t))
        temporal_weight[ki] = 1.0

        # Weight source 2: Pose similarity (for cyclic motions)
        body = before_motion[:, 9:135]
        kp_pose = body[ki]
        dists = np.array([np.linalg.norm(body[f] - kp_pose) for f in range(T)])
        corr_norm = np.linalg.norm(correction)

        vel = np.array([np.linalg.norm(body[f] - body[f-1]) for f in range(1, T)])
        static_frac = (vel < 0.03).mean()
        is_static = static_frac > 0.9 and vel.max() < 0.1

        max_dist = max(corr_norm * 1.5, np.percentile(dists, 40))
        if is_static:
            max_dist = np.percentile(dists, 60)

        similarity_weight = np.zeros(T, dtype=np.float32)
        for f in range(T):
            if dists[f] < max_dist:
                t = dists[f] / max_dist
                similarity_weight[f] = 0.5 * (1 + np.cos(np.pi * t))

        frame_weight = np.maximum(temporal_weight, similarity_weight)

        # Temporal smoothing: prevent sudden drops (±0.03/frame)
        for f in range(1, T):
            if frame_weight[f] < frame_weight[f-1] - 0.03:
                frame_weight[f] = frame_weight[f-1] - 0.03
        for f in range(T-2, -1, -1):
            if frame_weight[f] < frame_weight[f+1] - 0.03:
                frame_weight[f] = frame_weight[f+1] - 0.03

        equiv = sorted([int(f) for f in range(T) if frame_weight[f] > 0.3])
        if ki not in equiv:
            equiv = sorted(equiv + [ki])
        equiv_frames_dict[ki] = equiv

        # Apply: before + w * correction
        for f in range(T):
            result[f, 3:] = before_motion[f, 3:] + frame_weight[f] * correction

    # Force exact keypose
    for ki in keypose_indices:
        result[ki, 3:] = after_motion[ki, 3:]

    # Preserve translation
    result[:, :3] = before_motion[:, :3]

    return result, equiv_frames_dict


def main():
    before_dir = os.path.join(str(PROJECT_ROOT), BEFORE_DIR)
    after_dir = os.path.join(str(PROJECT_ROOT), AFTER_DIR)
    pairs = load_before_after_pairs(before_dir, after_dir)

    output_dir = PROJECT_ROOT / "output" / "eval_keyframe_pose_v3" / "local_rot" / "pure_blend"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for case_idx, pair in enumerate(pairs):
        before_motion = pair['before_motion']
        after_motion = pair['after_motion']
        T = pair['num_frames']

        kp_indices, diffs = select_keyposes(
            before_motion, after_motion,
            k=NUM_KEYPOSES, min_diff=MIN_KEYPOSE_DIFF,
        )

        case_key = f'case{case_idx:03d}_{pair["filename"].replace(".npz", "")}'

        batch_info = build_imputation_batch(
            before_motion, after_motion, kp_indices, mode='keyframe_only',
        )

        t0 = time.time()
        output, equiv_info = pure_blend(before_motion, after_motion, kp_indices)
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

        print(f'  {case_key}: kf={metrics["kf_mpjpe"]:.4f} glob={metrics["global_mpjpe"]:.4f} '
              f'src={metrics["src_mpjpe"]:.4f} smooth={metrics["overall_smoothness"]:.4f}')

    # Aggregate
    metric_keys = ['kf_mpjpe', 'global_mpjpe', 'src_mpjpe', 'boundary_smoothness',
                   'overall_smoothness', 'foot_skating']
    agg = {}
    for mk in metric_keys:
        vals = [r[mk] for r in results]
        agg[f'{mk}_mean'] = float(np.mean(vals))

    print(f'\n=== PURE BLEND (N={len(results)}) ===')
    for k, v in agg.items():
        print(f'  {k}: {v:.4f}')

    # Save summary
    with open(output_dir / 'results.json', 'w') as f:
        json.dump({'aggregate': agg, 'cases': results}, f, indent=2)

    # Update eval_summary.json
    summary_path = PROJECT_ROOT / "output" / "eval_keyframe_pose_v3" / "local_rot" / "eval_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        # Add or update pure_blend entry
        summary['comparison'] = [c for c in summary.get('comparison', []) if c.get('variant') != 'pure_blend']
        summary['comparison'].append({
            'variant': 'pure_blend',
            'model': 'none',
            'imp_mode': 'keyframe_only',
            'rep_mode': 'none',
            'sdedit_strength': 0.0,
            'rotation_space': 'local',
            'checkpoint': None,
            'n_cases': len(results),
            'kf_mpjpe': agg['kf_mpjpe_mean'],
            'global_mpjpe': agg['global_mpjpe_mean'],
            'src_mpjpe': agg['src_mpjpe_mean'],
            'bnd_smooth': agg['boundary_smoothness_mean'],
            'overall_smooth': agg['overall_smoothness_mean'],
            'foot_skate': agg['foot_skating_mean'],
            'time_sec': 0.0,
        })
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

    print(f'\nSaved to {output_dir}')


if __name__ == '__main__':
    main()
