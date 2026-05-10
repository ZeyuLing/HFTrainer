#!/usr/bin/env python3
"""CJGame repair evaluation: 5 configs + GT comparison.

Scans CJGame NPZ data, filters to quality-problematic samples via MotionQualityChecker,
runs 5 repair configs (MoGenDIT ada_denoise + 4 M2M variants), and compares with
hand-cleaned _cleaned.npz ground truth.

Output is compatible with the m2m_repair_compare web tool.

Usage:
    python3 tools/repair_eval_cjgame.py --device cuda
    python3 tools/repair_eval_cjgame.py --device cuda --configs uncond_fm_man uncond_jit_man
    python3 tools/repair_eval_cjgame.py --device cuda --skip-mask  # reuse existing masks
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = Path('data/lightai_data/CJGame_MB/npz_split')
OUTPUT_DIR = Path('output/m2m_repair_eval_cjgame')

# M2M repair configs: name -> (config_path, checkpoint_dir, replacement_guidance)
M2M_CONFIGS = {
    'uncond_fm_man': {
        'config': 'configs/hymotion_m2m/hymotion_m2m_completion_uncond_fm_man_046b.py',
        'checkpoint': 'work_dirs/hymotion_m2m_completion_uncond_fm_man_046b/checkpoint-epoch_1000',
        'replacement_guidance': 'skip_last',
    },
    'uncond_jit_man': {
        'config': 'configs/hymotion_m2m/hymotion_m2m_completion_uncond_jit_man_046b.py',
        'checkpoint': 'work_dirs/hymotion_m2m_completion_uncond_jit_man_046b/checkpoint-epoch_796',
        'replacement_guidance': 'skip_last',
    },
    'uncond_fm_man_globalrot': {
        'config': 'configs/hymotion_m2m/hymotion_m2m_completion_uncond_fm_man_globalrot_046b.py',
        'checkpoint': 'work_dirs/hymotion_m2m_completion_uncond_fm_man_globalrot_046b/checkpoint-epoch_459',
        'replacement_guidance': 'skip_last',
    },
    'uncond_jit_man_globalrot': {
        'config': 'configs/hymotion_m2m/hymotion_m2m_completion_uncond_jit_man_globalrot_046b.py',
        'checkpoint': 'work_dirs/hymotion_m2m_completion_uncond_jit_man_globalrot_046b/checkpoint-epoch_339',
        'replacement_guidance': 'skip_last',
    },
}

# ---------------------------------------------------------------------------
# Motion I/O helpers (from tools/test_m2m_repair.py)
# ---------------------------------------------------------------------------

def load_motion_135d(npz_path: str) -> torch.Tensor:
    """Load npz and convert to 135-dim motion (abs transl + rot6d)."""
    from hftrainer.datasets.motion.motionhub.transforms.load_smplx import (
        process_transl, process_smplx_pose,
    )
    data = np.load(npz_path, allow_pickle=True)

    trans_key = 'trans' if 'trans' in data else 'transl'
    abs_trans = data[trans_key].astype(np.float32)

    poses_key = 'poses' if 'poses' in data else 'body_pose'
    poses = data[poses_key].astype(np.float32)

    transl = process_transl(abs_trans, 'abs')  # [T, 3]
    pose = process_smplx_pose(poses, 'rotation_6d', 'smpl_22')  # [T, 132]
    motion = np.concatenate([transl, pose], axis=-1)  # [T, 135]
    return torch.from_numpy(motion).float()


def motion_135d_to_npz(motion_135d: np.ndarray, fps: int = 30) -> dict:
    """Convert 135-dim motion back to SMPL npz format."""
    from hftrainer.models.motion.hymotion_m2m.network.geometry import (
        rot6d_to_rotation_matrix,
    )
    from hftrainer.models.motion.components.utils.geometry.rotation_convert import (
        matrix_to_axis_angle,
    )

    T = motion_135d.shape[0]
    transl = motion_135d[:, :3]
    rot6d = motion_135d[:, 3:].reshape(T, 22, 6)

    # rot6d_to_rotation_matrix expects row-major rot6d (the same format as
    # process_smplx_pose outputs). No reorder needed — the function does
    # view(3,2) which reads rows, then takes columns 0/1 as R's first two columns.
    rot6d_flat = torch.from_numpy(rot6d.reshape(-1, 6)).float()
    rotmat = rot6d_to_rotation_matrix(rot6d_flat)
    aa = matrix_to_axis_angle(rotmat.numpy())
    aa = np.asarray(aa).reshape(T, 22, 3)

    poses_55 = np.zeros((T, 55, 3), dtype=np.float32)
    poses_55[:, :22, :] = aa
    poses_55 = poses_55.reshape(T, -1)

    return {
        'trans': transl,
        'poses': poses_55,
        'mocap_framerate': np.array(fps),
        'gender': 'neutral',
    }


# ---------------------------------------------------------------------------
# Step 1: Scan data, find pairs with _cleaned
# ---------------------------------------------------------------------------

def scan_cleaned_pairs(data_dir: Path) -> List[Dict]:
    """Find original files that have a _cleaned counterpart."""
    pairs = []
    all_files = sorted(data_dir.glob('*.npz'))
    cleaned_set = {f.name for f in all_files if '_cleaned' in f.name}

    for f in all_files:
        if '_cleaned' in f.name:
            continue
        cleaned_name = f.stem + '_cleaned.npz'
        if cleaned_name in cleaned_set:
            pairs.append({
                'original': f.name,
                'cleaned': cleaned_name,
            })
    logger.info(f'Found {len(pairs)} original-cleaned pairs in {data_dir}')
    return pairs


# ---------------------------------------------------------------------------
# Step 2: Quality check to filter problematic samples
# ---------------------------------------------------------------------------

def filter_quality_problems(
    pairs: List[Dict],
    data_dir: Path,
    device: str = 'cuda',
) -> List[Dict]:
    """Run MotionQualityChecker on originals, keep those with failed checks."""
    from hftrainer.evaluation.quality_check_rules.motion_quality_checker import (
        MotionQualityChecker,
    )
    checker = MotionQualityChecker(device=device)
    logger.info('MotionQualityChecker initialized, checking %d samples...', len(pairs))

    problematic = []
    for i, pair in enumerate(pairs):
        orig_path = str(data_dir / pair['original'])
        try:
            result = checker.check_from_file(orig_path)
        except Exception as e:
            logger.warning('Checker failed for %s: %s', pair['original'], e)
            continue

        if result.failed_checks:
            pair['before_failed'] = result.failed_checks
            pair['before_borderline'] = result.borderline_checks
            problematic.append(pair)

        if (i + 1) % 200 == 0:
            logger.info('  checked %d/%d, found %d problematic', i + 1, len(pairs), len(problematic))

    logger.info('Quality filtering done: %d / %d have failed checks', len(problematic), len(pairs))
    return problematic


# ---------------------------------------------------------------------------
# Step 3: Compute adaptive masks via MoGenDIT
# ---------------------------------------------------------------------------

def compute_all_adaptive_masks(
    samples: List[Dict],
    data_dir: Path,
    mask_dir: Path,
    device: str = 'cuda',
) -> None:
    """Compute MoGenDIT adaptive masks for all samples."""
    from hftrainer.pipelines.motion.mogendit_pipeline import MoGenDITRepairPipeline

    mask_dir.mkdir(parents=True, exist_ok=True)

    # Check which masks already exist
    to_compute = []
    for s in samples:
        mask_path = mask_dir / s['original']
        if mask_path.exists():
            continue
        to_compute.append(s)

    if not to_compute:
        logger.info('All %d adaptive masks already exist, skipping.', len(samples))
        return

    logger.info('Computing adaptive masks for %d samples (skipping %d existing)...',
                len(to_compute), len(samples) - len(to_compute))

    pipeline = MoGenDITRepairPipeline(device=device)

    for i, s in enumerate(to_compute):
        orig_path = str(data_dir / s['original'])
        mask_path = mask_dir / s['original']
        try:
            mask_result = pipeline.compute_adaptive_mask(orig_path)
            np.savez_compressed(
                str(mask_path),
                joint_mask=mask_result['joint_mask'],
                trans_mask=mask_result['trans_mask'],
            )
        except Exception as e:
            logger.warning('Mask computation failed for %s: %s', s['original'], e)
            continue

        if (i + 1) % 20 == 0:
            logger.info('  masks computed: %d/%d', i + 1, len(to_compute))

    logger.info('Adaptive mask computation done.')

    # Cleanup MoGenDIT pipeline to free GPU
    del pipeline
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Step 4: MoGenDIT ada_denoise repair
# ---------------------------------------------------------------------------

def run_mogendit_repair(
    samples: List[Dict],
    data_dir: Path,
    output_dir: Path,
    device: str = 'cuda',
) -> List[Dict]:
    """Run MoGenDIT ada_denoise repair on all samples."""
    from hftrainer.pipelines.motion.mogendit_pipeline import MoGenDITRepairPipeline

    config_name = 'mogendit_ada_denoise'
    repaired_dir = output_dir / config_name / 'repaired'
    repaired_dir.mkdir(parents=True, exist_ok=True)

    # Check which already repaired
    to_repair = []
    for s in samples:
        out_path = repaired_dir / s['original']
        if out_path.exists():
            continue
        to_repair.append(s)

    if not to_repair:
        logger.info('[%s] All %d samples already repaired, skipping inference.',
                     config_name, len(samples))
    else:
        logger.info('[%s] Repairing %d samples (skipping %d existing)...',
                     config_name, len(to_repair), len(samples) - len(to_repair))

        pipeline = MoGenDITRepairPipeline(device=device)

        for i, s in enumerate(to_repair):
            orig_path = str(data_dir / s['original'])
            out_path = str(repaired_dir / s['original'])
            try:
                pipeline.repair_npz(orig_path, out_path, mode='ada_denoise', step=10)
            except Exception as e:
                logger.warning('[%s] Failed %s: %s', config_name, s['original'], e)
                continue

            if (i + 1) % 10 == 0:
                logger.info('[%s] repaired %d/%d', config_name, i + 1, len(to_repair))

        del pipeline
        torch.cuda.empty_cache()

    # Evaluate all samples
    return evaluate_config(config_name, samples, data_dir, output_dir, device)


# ---------------------------------------------------------------------------
# Step 5: M2M repair with adaptive mask
# ---------------------------------------------------------------------------

def expand_mask_to_135d(joint_mask: np.ndarray, trans_mask: np.ndarray) -> np.ndarray:
    """Expand (T,22) joint_mask + (T,) trans_mask to (T,135) mask.

    135d layout: [trans(3), rot6d_joint0(6), rot6d_joint1(6), ..., rot6d_joint21(6)]
    """
    T = joint_mask.shape[0]
    mask_135 = np.zeros((T, 135), dtype=np.float32)

    # Translation: frames 0-2
    for t in range(T):
        mask_135[t, 0:3] = float(trans_mask[t])

    # Per-joint rot6d: 22 joints × 6 dims
    for j in range(22):
        start = 3 + j * 6
        end = start + 6
        mask_135[:, start:end] = joint_mask[:, j:j+1].astype(np.float32)

    return mask_135


def run_m2m_repair(
    config_name: str,
    config_info: Dict,
    samples: List[Dict],
    data_dir: Path,
    mask_dir: Path,
    output_dir: Path,
    num_steps: int = 50,
    device: str = 'cuda',
) -> List[Dict]:
    """Run M2M repair with adaptive mask for one config."""
    from mmengine.config import Config
    from hftrainer.registry import MODEL_BUNDLES
    from hftrainer.utils.checkpoint_utils import load_checkpoint
    from hftrainer.pipelines.motion.hymotion_m2m_pipeline import HyMotionM2MPipeline

    repaired_dir = output_dir / config_name / 'repaired'
    repaired_dir.mkdir(parents=True, exist_ok=True)

    # Check which already repaired
    to_repair = []
    for s in samples:
        out_path = repaired_dir / s['original']
        if out_path.exists():
            continue
        to_repair.append(s)

    if not to_repair:
        logger.info('[%s] All %d samples already repaired, skipping inference.',
                     config_name, len(samples))
    else:
        logger.info('[%s] Loading model from %s ...', config_name, config_info['checkpoint'])
        cfg = Config.fromfile(config_info['config'])
        bundle = MODEL_BUNDLES.build(cfg.model.to_dict())
        sd = load_checkpoint(config_info['checkpoint'], map_location='cpu')
        bundle.load_state_dict_selective(sd)
        del sd
        bundle.eval()
        bundle = bundle.to(device)

        replacement_guidance = config_info.get('replacement_guidance', 'skip_last')
        pipeline = HyMotionM2MPipeline(
            bundle=bundle,
            num_steps=num_steps,
            replacement_guidance=replacement_guidance,
        )

        logger.info('[%s] Repairing %d samples (skipping %d existing)...',
                     config_name, len(to_repair), len(samples) - len(to_repair))

        for i, s in enumerate(to_repair):
            orig_path = str(data_dir / s['original'])
            mask_path = mask_dir / s['original']
            out_path = repaired_dir / s['original']

            try:
                # Load motion
                motion = load_motion_135d(orig_path)  # [T, 135]
                T, D = motion.shape
                MAX_FRAMES = 360

                # Load adaptive mask
                if not mask_path.exists():
                    logger.warning('[%s] No mask for %s, skipping', config_name, s['original'])
                    continue
                mask_data = np.load(str(mask_path))
                joint_mask = mask_data['joint_mask']  # (T_mask, 22)
                trans_mask = mask_data['trans_mask']   # (T_mask,)

                # Align lengths (mask may differ slightly from motion)
                T_mask = min(T, joint_mask.shape[0])
                joint_mask = joint_mask[:T_mask]
                trans_mask = trans_mask[:T_mask]
                if T_mask < T:
                    # Pad mask with False (no repair) for extra frames
                    joint_mask = np.pad(joint_mask, ((0, T - T_mask), (0, 0)))
                    trans_mask = np.pad(trans_mask, (0, T - T_mask))

                # Clamp to max_frames (model trained on 360)
                T_use = min(T, MAX_FRAMES)

                mask_135 = expand_mask_to_135d(joint_mask[:T_use], trans_mask[:T_use])
                mask_135_t = torch.from_numpy(mask_135).float().unsqueeze(0).to(device)  # [1, T_use, 135]

                # Normalize motion (clamp to T_use)
                motion_norm = bundle.normalize_motion(
                    motion[:T_use].unsqueeze(0).to(device)
                )  # [1, T_use, 135]

                # clean_motion = full normalized (for replacement guidance)
                clean_motion = motion_norm.clone()
                # src_motion = zeroed in masked regions (for VACE input)
                src_motion = motion_norm * (1 - mask_135_t)

                # Pad to MAX_FRAMES (model trained on fixed-length 360)
                if T_use < MAX_FRAMES:
                    pad_len = MAX_FRAMES - T_use
                    src_motion = torch.nn.functional.pad(src_motion, (0, 0, 0, pad_len), value=0)
                    clean_motion = torch.nn.functional.pad(clean_motion, (0, 0, 0, pad_len), value=0)
                    mask_135_t = torch.nn.functional.pad(mask_135_t, (0, 0, 0, pad_len), value=0)

                batch = {
                    'src_motion': src_motion,
                    'clean_motion': clean_motion,
                    'src_mask': mask_135_t,
                    'src_length': [T_use],
                    'tgt_length': [T_use],
                }

                with torch.no_grad():
                    output = pipeline(batch)

                # Get repaired latent, truncate to T_use, and denormalize
                repaired_latent = output['latent'][:, :T_use]  # [1, T_use, 135]
                repaired_raw = bundle.denormalize_motion(repaired_latent)[0].cpu().numpy()

                # Blend: keep original in unmasked regions, use model output in masked
                mask_135_np = expand_mask_to_135d(joint_mask[:T_use], trans_mask[:T_use])
                original_raw = motion[:T_use].numpy()
                combined = original_raw * (1 - mask_135_np) + repaired_raw * mask_135_np

                # If original motion was longer than MAX_FRAMES, append unrepaired tail
                if T > T_use:
                    combined = np.concatenate([combined, motion[T_use:].numpy()], axis=0)

                # Save as NPZ
                try:
                    orig_data = np.load(orig_path, allow_pickle=True)
                    fps = int(orig_data.get('mocap_framerate', 30))
                except Exception:
                    fps = 30

                npz_dict = motion_135d_to_npz(combined, fps=fps)
                np.savez_compressed(str(out_path), **npz_dict)

            except Exception as e:
                logger.warning('[%s] Failed %s: %s', config_name, s['original'], e)
                continue

            if (i + 1) % 10 == 0:
                logger.info('[%s] repaired %d/%d', config_name, i + 1, len(to_repair))

        # Cleanup
        del pipeline, bundle
        torch.cuda.empty_cache()

    # Evaluate all samples
    return evaluate_config(config_name, samples, data_dir, output_dir, device)


# ---------------------------------------------------------------------------
# Step 6: Evaluate repaired results
# ---------------------------------------------------------------------------

def evaluate_config(
    config_name: str,
    samples: List[Dict],
    data_dir: Path,
    output_dir: Path,
    device: str = 'cuda',
) -> List[Dict]:
    """Run quality checker + MPJPE vs GT on repaired samples for one config."""
    from hftrainer.evaluation.quality_check_rules.motion_quality_checker import (
        MotionQualityChecker,
    )

    repaired_dir = output_dir / config_name / 'repaired'
    checker = MotionQualityChecker(device=device)

    details = []
    improved_count = 0
    degraded_count = 0
    same_count = 0

    for s in samples:
        repaired_path = repaired_dir / s['original']
        cleaned_path = data_dir / s['cleaned']

        if not repaired_path.exists():
            continue

        before_failed = s.get('before_failed', [])

        # Quality check on repaired
        try:
            after_result = checker.check_from_file(str(repaired_path))
            after_failed = after_result.failed_checks
            after_valid = after_result.is_valid
        except Exception as e:
            logger.warning('[%s] Checker failed on repaired %s: %s',
                           config_name, s['original'], e)
            after_failed = []
            after_valid = False

        # MPJPE vs GT (_cleaned.npz) in 135d space
        mpjpe_unmasked = None
        try:
            repaired_motion = load_motion_135d(str(repaired_path))
            gt_motion = load_motion_135d(str(cleaned_path))
            T_min = min(repaired_motion.shape[0], gt_motion.shape[0])
            diff = torch.abs(repaired_motion[:T_min] - gt_motion[:T_min])
            mpjpe_unmasked = float(diff.mean())
        except Exception as e:
            logger.warning('[%s] MPJPE failed for %s: %s',
                           config_name, s['original'], e)

        # Determine improvement
        before_n = len(before_failed)
        after_n = len(after_failed)
        if after_n < before_n:
            improved = True
            improved_count += 1
        elif after_n > before_n:
            improved = False
            degraded_count += 1
        else:
            improved = None  # same number of failures
            same_count += 1

        # Get frame count
        try:
            d = np.load(str(repaired_path), allow_pickle=True)
            num_frames = int(d['poses'].shape[0])
        except Exception:
            num_frames = 0

        # Mask ratio
        mask_path = output_dir / 'adaptive_masks' / s['original']
        mask_ratio = 0.0
        if mask_path.exists():
            try:
                md = np.load(str(mask_path))
                jm = md['joint_mask']
                mask_ratio = round(float(jm.sum()) / max(jm.size, 1), 4)
            except Exception:
                pass

        detail = {
            'path': s['original'],
            'num_frames': num_frames,
            'mask_ratio': mask_ratio,
            'before_failed': before_failed,
            'after_valid': after_valid,
            'after_failed': after_failed,
            'improved': improved,
            'mpjpe_unmasked': round(mpjpe_unmasked, 6) if mpjpe_unmasked is not None else None,
        }
        details.append(detail)

    # Save repair_stats.json
    stats = {
        'config': config_name,
        'total': len(details),
        'improved': improved_count,
        'degraded': degraded_count,
        'same': same_count,
        'details': details,
    }

    stats_path = output_dir / config_name / 'repair_stats.json'
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    logger.info('[%s] Evaluation done: %d total, %d improved, %d degraded, %d same',
                config_name, len(details), improved_count, degraded_count, same_count)

    if details:
        mpjpe_vals = [d['mpjpe_unmasked'] for d in details if d['mpjpe_unmasked'] is not None]
        if mpjpe_vals:
            logger.info('[%s] MPJPE vs GT: mean=%.6f, median=%.6f',
                        config_name, np.mean(mpjpe_vals), np.median(mpjpe_vals))

    return details


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='CJGame repair evaluation')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--data-dir', type=str, default=str(DATA_DIR))
    parser.add_argument('--output-dir', type=str, default=str(OUTPUT_DIR))
    parser.add_argument('--num-steps', type=int, default=50,
                        help='ODE steps for M2M inference')
    parser.add_argument('--skip-mask', action='store_true',
                        help='Skip adaptive mask computation (reuse existing)')
    parser.add_argument('--skip-mogendit', action='store_true',
                        help='Skip MoGenDIT repair')
    parser.add_argument('--configs', nargs='*', default=None,
                        help='M2M config names to run (default: all)')
    parser.add_argument('--eval-only', action='store_true',
                        help='Skip inference, only run evaluation on existing results')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    mask_dir = output_dir / 'adaptive_masks'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which M2M configs to run
    if args.configs:
        m2m_configs = {k: v for k, v in M2M_CONFIGS.items() if k in args.configs}
        unknown = set(args.configs) - set(M2M_CONFIGS.keys())
        if unknown:
            logger.warning('Unknown configs: %s (available: %s)',
                           unknown, list(M2M_CONFIGS.keys()))
    else:
        m2m_configs = M2M_CONFIGS

    # Step 1: Scan data for original-cleaned pairs
    sample_list_path = output_dir / 'sample_list.json'
    if sample_list_path.exists():
        logger.info('Loading existing sample_list.json ...')
        with open(sample_list_path) as f:
            samples = json.load(f)
        logger.info('Loaded %d problematic samples from cache.', len(samples))
    else:
        pairs = scan_cleaned_pairs(data_dir)

        # Step 2: Quality check to filter problematic ones
        samples = filter_quality_problems(pairs, data_dir, device=args.device)

        # Save sample list
        with open(sample_list_path, 'w') as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
        logger.info('Saved sample_list.json with %d samples.', len(samples))

    if not samples:
        logger.error('No problematic samples found. Exiting.')
        return

    logger.info('=== %d samples to evaluate ===', len(samples))

    if args.eval_only:
        logger.info('--eval-only mode: skipping all inference, running evaluation only.')

        # Evaluate MoGenDIT
        if not args.skip_mogendit:
            mogendit_repaired = output_dir / 'mogendit_ada_denoise' / 'repaired'
            if mogendit_repaired.is_dir():
                evaluate_config('mogendit_ada_denoise', samples, data_dir, output_dir, args.device)

        # Evaluate M2M configs
        for config_name in m2m_configs:
            m2m_repaired = output_dir / config_name / 'repaired'
            if m2m_repaired.is_dir():
                evaluate_config(config_name, samples, data_dir, output_dir, args.device)

        logger.info('Evaluation-only mode complete.')
        return

    # Step 3: Compute adaptive masks
    if not args.skip_mask:
        compute_all_adaptive_masks(samples, data_dir, mask_dir, device=args.device)
    else:
        logger.info('Skipping adaptive mask computation (--skip-mask).')

    # Step 4: MoGenDIT ada_denoise repair
    if not args.skip_mogendit:
        logger.info('=== Running MoGenDIT ada_denoise ===')
        run_mogendit_repair(samples, data_dir, output_dir, device=args.device)

    # Step 5: M2M repairs
    for config_name, config_info in m2m_configs.items():
        logger.info('=== Running M2M config: %s ===', config_name)
        run_m2m_repair(
            config_name=config_name,
            config_info=config_info,
            samples=samples,
            data_dir=data_dir,
            mask_dir=mask_dir,
            output_dir=output_dir,
            num_steps=args.num_steps,
            device=args.device,
        )

    # Summary
    logger.info('\n=== All configs done ===')
    logger.info('Output directory: %s', output_dir)
    all_configs = []
    if not args.skip_mogendit:
        all_configs.append('mogendit_ada_denoise')
    all_configs.extend(m2m_configs.keys())
    for cfg in all_configs:
        stats_path = output_dir / cfg / 'repair_stats.json'
        if stats_path.exists():
            with open(stats_path) as f:
                stats = json.load(f)
            mpjpe_vals = [d['mpjpe_unmasked'] for d in stats.get('details', [])
                          if d.get('mpjpe_unmasked') is not None]
            mpjpe_str = f', MPJPE={np.mean(mpjpe_vals):.6f}' if mpjpe_vals else ''
            logger.info('  %s: total=%d, improved=%d, degraded=%d%s',
                        cfg, stats['total'], stats['improved'], stats['degraded'], mpjpe_str)


if __name__ == '__main__':
    main()
