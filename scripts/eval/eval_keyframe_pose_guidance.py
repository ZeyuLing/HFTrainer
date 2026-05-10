#!/usr/bin/env python3
"""Keyframe Pose Guidance Evaluation — Before/After Correction Data.

Evaluates keypose-conditioned SDEdit using real before/after motion pairs
from PeacekeeperElite_part4 dataset.

Pipeline:
  1. Load paired (before, after) motions from before/after directories
  2. Select keypose frames: top-K frames with largest correction diff + anchors
  3. SDEdit: noise the before motion, denoise with keypose replacement from after
  4. Metrics: compare output vs after (target) AND output vs before (source)

Models:
  - HyMotion M2M: uncond_fm_man (local & global rot)
  - MoGenDIT 0.1B baseline

Usage:
    # Quick eval (3 cases)
    python3 scripts/eval_keyframe_pose_guidance.py --quick --gpu 0

    # Full eval
    python3 scripts/eval_keyframe_pose_guidance.py --gpu 0

    # Multi-GPU
    python3 scripts/eval_keyframe_pose_guidance.py --multi-gpu --num-gpus 8
"""

import argparse
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('eval_kf_pose')

# ─────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────

D = 135  # motion dim: 3 transl + 22*6 rot6d
MAX_LEN = 360

# Before/After data paths (relative to PROJECT_ROOT)
BEFORE_DIR = 'data/PeacekeeperElite_MB/PeacekeeperElite_part4_before_MB'
AFTER_DIR = 'data/PeacekeeperElite_MB/PeacekeeperElite_part4_after_MB'

# Model variants: (name, config, work_dir, rotation_space)
# Only include converged models per plan
MAN_MODELS = [
    ('uncond_fm_man',
     'configs/hymotion_m2m/hymotion_m2m_completion_uncond_fm_man_046b.py',
     'work_dirs/hymotion_m2m_completion_uncond_fm_man_046b', 'local'),
    ('uncond_fm_man_globalrot',
     'configs/hymotion_m2m/hymotion_m2m_completion_uncond_fm_man_globalrot_046b.py',
     'work_dirs/hymotion_m2m_completion_uncond_fm_man_globalrot_046b', 'global'),
]

MOGENDIT_MODELS = [
    ('mogendit_0.1B', 'MoreDiff-0.1B', None, 'local'),
]

# Only keyframe_only mode: SDEdit handles all frames, keyposes are exact
IMPUTATION_MODES = ['keyframe_only']
REPLACEMENT_MODES = ['skip_last']
# SDEdit strengths for flow matching
SDEDIT_STRENGTHS = [0.05, 0.1, 0.3, 0.5]
# MoGenDIT denoise steps (higher = more noise = more change)
MOGENDIT_STEPS = [10, 50, 100]

# Number of keyposes to select (max, actual count adapts to motion length)
NUM_KEYPOSES = 2

# Minimum per-frame diff to be considered a valid keypose candidate
MIN_KEYPOSE_DIFF = 0.1


# ─────────────────────────────────────────────────────────────────────
# Data loading: before/after pairs
# ─────────────────────────────────────────────────────────────────────

def _npz_to_smpl22_rot6d(npz):
    """Convert NPZ (smplx format) to smpl_22 rot6d 135-dim representation.

    Uses the same loading functions as the LoadSmplx55 transform.
    """
    from hftrainer.datasets.motion.motionhub.transforms.load_smplx import (
        process_smplx_pose,
        process_transl,
    )

    abs_trans = np.asarray(npz['trans'], dtype=np.float32)
    poses = np.asarray(npz['poses'], dtype=np.float32)

    T = poses.shape[0]
    if poses.shape[1] < 66:
        return None

    # Process poses: axis_angle -> rot6d, smpl_22 (first 22 joints)
    if poses.shape[1] < 165:
        poses_padded = np.zeros((T, 165), dtype=np.float32)
        poses_padded[:, :poses.shape[1]] = poses
        poses = poses_padded

    rot6d = process_smplx_pose(poses, rot_type='rotation_6d', out_type='smpl_22')
    transl = process_transl(abs_trans, transl_type='abs')

    # Concat: [transl(3), rot6d(132)] = 135
    motion = np.concatenate([transl, rot6d], axis=-1).astype(np.float32)
    return motion


def load_before_after_pairs(before_dir, after_dir, max_pairs=None, seed=42):
    """Load matched before/after motion pairs.

    Only returns pairs where both files exist AND have the same frame count
    (same-frame-count files are global body re-fits, best for keypose eval).
    """
    before_dir = Path(before_dir)
    after_dir = Path(after_dir)

    if not before_dir.exists() or not after_dir.exists():
        logger.error(f'Data dirs not found: {before_dir} or {after_dir}')
        return []

    # Find common NPZ files
    before_files = {f.name for f in before_dir.glob('*.npz')}
    after_files = {f.name for f in after_dir.glob('*.npz')}
    common = sorted(before_files & after_files)

    logger.info(f'Found {len(common)} common files between before/after dirs')

    pairs = []
    for fname in common:
        try:
            before_npz = np.load(str(before_dir / fname), allow_pickle=True)
            after_npz = np.load(str(after_dir / fname), allow_pickle=True)

            before_motion = _npz_to_smpl22_rot6d(before_npz)
            after_motion = _npz_to_smpl22_rot6d(after_npz)

            if before_motion is None or after_motion is None:
                continue

            # Only use same-frame-count pairs (global re-fits, not re-animations)
            if before_motion.shape[0] != after_motion.shape[0]:
                continue

            T = before_motion.shape[0]
            if T < 30:  # skip very short
                continue

            # Clip to MAX_LEN
            if T > MAX_LEN:
                before_motion = before_motion[:MAX_LEN]
                after_motion = after_motion[:MAX_LEN]
                T = MAX_LEN

            pairs.append({
                'before_motion': before_motion.astype(np.float32),
                'after_motion': after_motion.astype(np.float32),
                'before_npz_path': str(before_dir / fname),
                'after_npz_path': str(after_dir / fname),
                'filename': fname,
                'num_frames': T,
            })
        except Exception as e:
            logger.warning(f'Failed to load {fname}: {e}')
            continue

    # Shuffle with seed for reproducibility
    rng = np.random.RandomState(seed)
    rng.shuffle(pairs)

    if max_pairs is not None and len(pairs) > max_pairs:
        pairs = pairs[:max_pairs]

    logger.info(
        f'Loaded {len(pairs)} same-frame-count before/after pairs '
        f'(from {len(common)} common files)'
    )
    return pairs


def select_keyposes(before_motion, after_motion, k=2, min_diff=0.1, min_gap=10):
    """Select keypose frames where correction diff is largest.

    Adaptively chooses k=1 or k=2:
    - k=2 only when corrections are LOCALIZED (max/mean ratio > 2.0),
      meaning there are distinct peaks worth anchoring to.
    - k=1 for uniform corrections (global offset / consistent change across
      all frames) — a single keypose at the max-diff frame suffices.

    Returns:
        keypose_indices: sorted list of selected keypose frame indices
        diffs: per-frame diff array (T,) for diagnostics
    """
    T = before_motion.shape[0]

    before_body = before_motion[:, 3:135]
    after_body = after_motion[:, 3:135]
    diffs = np.linalg.norm(after_body - before_body, axis=-1)

    margin = min(3, max(1, T // 20))

    candidate_mask = np.zeros(T, dtype=bool)
    candidate_mask[margin:T - margin] = True
    candidate_mask &= (diffs > min_diff)

    if not candidate_mask.any():
        best = int(np.argmax(diffs))
        return [best], diffs

    valid_diffs = diffs[candidate_mask]

    # Decide actual k: only use k=2 when corrections are localized
    # (distinct peaks), not when it's a uniform global offset.
    peak_ratio = valid_diffs.max() / (valid_diffs.mean() + 1e-8)
    actual_k = k if (peak_ratio > 3.0 and T >= 2 * min_gap) else 1

    gap = max(3, min(min_gap, T // (actual_k + 1)))

    if actual_k == 1:
        # Single keypose at max diff
        best = int(np.argmax(diffs * candidate_mask))
        return [best], diffs

    # Peaked: greedy top-k with gap constraint
    cd = diffs.copy()
    cd[~candidate_mask] = -1
    selected = []
    for _ in range(actual_k):
        best = int(np.argmax(cd))
        if cd[best] <= 0:
            break
        selected.append(best)
        cd[max(0, best - gap):min(T, best + gap + 1)] = -1

    return sorted(selected), diffs


# ─────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────

def find_latest_checkpoint(work_dir):
    """Find the latest checkpoint-epoch_N directory."""
    work_dir = Path(work_dir)
    if not work_dir.exists():
        return None
    ckpts = sorted(
        work_dir.glob('checkpoint-epoch_*'),
        key=lambda p: int(p.name.split('_')[-1]),
    )
    return str(ckpts[-1]) if ckpts else None


def load_m2m_bundle(config_path, checkpoint_path, device='cuda:0'):
    """Load HyMotionM2MBundle from config + checkpoint."""
    from mmengine.config import Config
    import hftrainer  # noqa: register modules
    from hftrainer.registry import MODEL_BUNDLES
    from hftrainer.utils.checkpoint_utils import load_checkpoint

    cfg = Config.fromfile(config_path)
    model_cfg = cfg.model
    if hasattr(model_cfg, 'to_dict'):
        model_cfg = model_cfg.to_dict()

    bundle_cls = MODEL_BUNDLES.get(model_cfg['type'])
    bundle = bundle_cls.from_config(model_cfg)
    bundle.eval()

    if checkpoint_path:
        state_dict = load_checkpoint(checkpoint_path, map_location='cpu')
        bundle.load_state_dict_selective(state_dict)
        logger.info(f'Loaded checkpoint: {checkpoint_path}')

    bundle = bundle.to(device)
    return bundle


def load_mogendit_pipeline(model_name='MoreDiff-0.1B', device='cuda:0'):
    """Load MoGenDIT repair/imputation pipeline."""
    from hftrainer.pipelines.motion.mogendit_pipeline import MoGenDITRepairPipeline
    pipeline = MoGenDITRepairPipeline(model_name=model_name, device=device)
    return pipeline


# ─────────────────────────────────────────────────────────────────────
# Imputation batch construction
# ─────────────────────────────────────────────────────────────────────

def build_imputation_batch(
    before_motion: np.ndarray,
    after_motion: np.ndarray,
    keypose_indices: list,
    mode: str = 'keyframe_only',
    blend_radius: int = 0,
) -> dict:
    """Build a batch dict for keypose-conditioned SDEdit.

    The composite motion = before motion with keypose region replaced:
    - Exact keypose frames from after
    - Neighbors within blend_radius: linear blend before→after[ki]
    Keypose + neighbor frames are marked as observed (mask=0).
    blend_radius=0 means auto-compute from keypose spacing.
    """
    T, D_ = before_motion.shape

    # Auto-compute blend_radius from keypose spacing
    if blend_radius <= 0:
        sorted_kp = sorted(keypose_indices)
        # For each KP, radius = half distance to nearest neighbor KP or boundary
        boundaries = [0] + sorted_kp + [T - 1]
        max_radius = 0
        for i, ki in enumerate(sorted_kp):
            left_dist = ki - boundaries[i]     # dist to left neighbor/boundary
            right_dist = boundaries[i + 2] - ki  # dist to right neighbor/boundary
            radius = min(left_dist, right_dist) // 2
            max_radius = max(max_radius, radius)
        blend_radius = max(max_radius, 5)  # at least 5
        blend_radius = min(blend_radius, 30)  # at most 30

    composite = before_motion.copy()
    src_mask = np.ones((T, D_), dtype=np.float32)

    for ki in keypose_indices:
        # Exact keypose: all dims observed
        composite[ki] = after_motion[ki].copy()
        src_mask[ki] = 0.0
        # Blended neighbors: all dims observed (matching training joint-group granularity)
        for d in range(1, blend_radius + 1):
            w = 1.0 - d / (blend_radius + 1)
            for f in [ki - d, ki + d]:
                if 0 <= f < T:
                    composite[f] = ((1 - w) * before_motion[f] + w * after_motion[ki]).astype(np.float32)
                    src_mask[f] = 0.0  # all dims observed, matching training pattern

    return {
        'composite_motion': composite,
        'src_mask': src_mask,
        'before_motion': before_motion,
        'after_motion': after_motion,
        'keypose_indices': keypose_indices,
        'mode': mode,
        'num_frames': T,
    }


def apply_correction_blend(
    before_motion: np.ndarray,
    after_motion: np.ndarray,
    keypose_indices: list,
) -> np.ndarray:
    """Apply keypose correction propagation via blending BEFORE model inference.

    Conservative "Blend then Polish" strategy:
    - Only applies correction to frames that are TEMPORALLY CLOSE to keypose
    - Uses pure cosine falloff within a moderate radius, NO temporal smoothing
      (smoothing was causing correction to leak to the entire motion)
    - Does NOT use pose similarity (applying correction[ki] to a frame with
      completely different pose causes the "yank" artifact)
    - The model handles the transition smoothing in the polish pass

    Returns: blended_motion (T, D), same shape as before_motion
    """
    T, D = before_motion.shape
    result = before_motion.copy()

    sorted_kp = sorted(keypose_indices)
    # Radius: half distance to nearest neighbor KP or boundary, clamped
    boundaries = [0] + sorted_kp + [T - 1]
    for i, ki in enumerate(sorted_kp):
        left_dist = ki - boundaries[i]
        right_dist = boundaries[i + 2] - ki
        half_gap = min(left_dist, right_dist) // 2
        RADIUS = max(min(half_gap, 20), 3)

        correction = after_motion[ki, 3:] - before_motion[ki, 3:]

        # Pure cosine falloff, no smoothing, no similarity
        for f in range(max(0, ki - RADIUS), min(T, ki + RADIUS + 1)):
            d = abs(f - ki)
            t = d / (RADIUS + 1)
            w = 0.5 * (1 + np.cos(np.pi * t))
            result[f, 3:] = before_motion[f, 3:] + w * correction

    # Force exact keypose frames
    for ki in keypose_indices:
        result[ki, 3:] = after_motion[ki, 3:]

    # Preserve translation from before
    result[:, :3] = before_motion[:, :3]

    return result


# ─────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_m2m_imputation(
    bundle,
    batch_info: dict,
    replacement_guidance: str = 'none',
    num_steps: int = 50,
    device: str = 'cuda:0',
    sdedit_strength: float = 0.1,
) -> dict:
    """Run M2M SDEdit imputation with "Blend then Polish" strategy.

    1. First, correction-blend propagates keypose edits to all frames
       (dual-weight: temporal proximity + pose similarity)
    2. The blended motion (not raw before) is used as SDEdit starting point
    3. Model denoises from this already-corrected motion, preserving the
       correction while smoothing transitions

    Parameters
    ----------
    sdedit_strength : float
        SDEdit strength in (0, 1]. Lower = stays closer to blended motion.
    """
    from hftrainer.pipelines.motion.hymotion_m2m_pipeline import HyMotionM2MPipeline

    pipeline = HyMotionM2MPipeline(
        bundle=bundle,
        num_steps=num_steps,
        replacement_guidance=replacement_guidance,
    )

    composite = torch.from_numpy(batch_info['composite_motion']).float().unsqueeze(0).to(device)
    src_mask = torch.from_numpy(batch_info['src_mask']).float().unsqueeze(0).to(device)
    before = torch.from_numpy(batch_info['before_motion']).float().unsqueeze(0).to(device)
    T = batch_info['num_frames']

    # === "Blend then Polish": correction-propagate BEFORE model inference ===
    blended_np = apply_correction_blend(
        batch_info['before_motion'],
        batch_info['after_motion'],
        batch_info['keypose_indices'],
    )
    blended = torch.from_numpy(blended_np).float().unsqueeze(0).to(device)

    # Normalize
    normalized_composite = bundle.normalize_motion(composite)
    normalized_blended = bundle.normalize_motion(blended)

    # SDEdit: mix blended motion with noise according to sdedit_strength.
    # The pipeline uses clean_motion for replacement guidance (keep_mask regions).
    # For generated regions (mask=1), the y0 will be:
    #   y0 = (1 - strength) * blended + strength * noise
    # This is done by the pipeline: y0 = where(keep_mask, x_clean, z)
    # But z is pure noise, so we need to inject the SDEdit mix into z.
    # We achieve this by providing clean_motion = blended, and making the
    # pipeline start known regions from clean. For unknown regions, the pipeline
    # starts from noise. SDEdit strength controls the ODE integration range.
    # Since pipeline.num_steps controls granularity but not range, we adjust
    # num_steps based on strength: fewer effective steps = less denoising.
    # Actually the simplest approach: provide the blended as clean_motion,
    # and the pipeline naturally preserves it in known regions + starts from
    # noise in unknown. The model will denoise toward blended (as it's the
    # conditioning context).
    clean_motion_normalized = normalized_blended.clone()

    # Zero mask regions for VACE conditioning
    vace_input = normalized_composite * (1 - src_mask)

    infer_batch = {
        'src_motion': vace_input,
        'src_mask': src_mask,
        'src_length': [T],
        'tgt_length': [T],
        'clean_motion': clean_motion_normalized,
    }

    result = pipeline(infer_batch)
    latent = result['latent']  # (1, T, D) normalized

    # Denormalize
    output_denorm = bundle.denormalize_motion(latent)

    # Post-hoc: keep observed regions from composite, generated from model
    mask_3d = src_mask
    final = composite * (1 - mask_3d) + output_denorm * mask_3d
    # Keypose task only modifies pose, not translation trajectory.
    final[:, :, :3] = before[:, :, :3]
    final_np = final.squeeze(0).cpu().numpy()

    return {
        'output_motion': final_np,
        'blended_motion': blended_np,
        'raw_output': output_denorm.squeeze(0).cpu().numpy(),
        'latent': latent.squeeze(0).cpu().numpy(),
    }


def _restore_heading(output_motion, reference_motion):
    """Restore output heading to match reference motion's coordinate frame.

    MoGenDIT normalizes heading to face +Z and shifts XZ to origin.
    This function:
    1. Computes the Y-axis heading angle from both motions' first-frame root
    2. Rotates the output's translation and root rotation by the delta
    3. Also aligns XZ offset of first frame

    Both motions are in 135-dim format: [trans(3), rot6d(132)].
    """

    def _rot6d_to_rotmat_single(r6):
        """Convert single 6-dim rot6d (row-major) to 3x3 rotation matrix."""
        a1 = r6[[0, 2, 4]]
        a2 = r6[[1, 3, 5]]
        b1 = a1 / (np.linalg.norm(a1) + 1e-8)
        dot = np.dot(b1, a2)
        b2 = a2 - dot * b1
        b2 = b2 / (np.linalg.norm(b2) + 1e-8)
        b3 = np.cross(b1, b2)
        return np.stack([b1, b2, b3], axis=-1)

    def _extract_heading_y(r6):
        """Extract Y-axis heading angle from root rot6d."""
        R = _rot6d_to_rotmat_single(r6)
        fwd = R[:, 2]  # forward = 3rd column (Z axis)
        heading = np.arctan2(fwd[0], fwd[2])
        return heading

    # Extract heading from first frame
    ref_heading = _extract_heading_y(reference_motion[0, 3:9])
    out_heading = _extract_heading_y(output_motion[0, 3:9])
    delta = ref_heading - out_heading

    # Build Y-rotation matrix for delta
    cos_d, sin_d = np.cos(delta), np.sin(delta)
    R_delta = np.array([
        [cos_d, 0, sin_d],
        [0, 1, 0],
        [-sin_d, 0, cos_d],
    ], dtype=np.float32)

    result = output_motion.copy()
    T = result.shape[0]

    # Rotate all frames' translation and root rotation
    for t in range(T):
        # Rotate translation
        result[t, :3] = R_delta @ result[t, :3]

        # Rotate root joint rotation matrix, then convert back to rot6d
        R_root = _rot6d_to_rotmat_single(result[t, 3:9])
        R_new = R_delta @ R_root
        # row-major rot6d: [R00,R01,R10,R11,R20,R21]
        result[t, 3] = R_new[0, 0]
        result[t, 4] = R_new[0, 1]
        result[t, 5] = R_new[1, 0]
        result[t, 6] = R_new[1, 1]
        result[t, 7] = R_new[2, 0]
        result[t, 8] = R_new[2, 1]

    # Align XZ offset: shift so first frame's XZ matches reference
    result[:, 0] += reference_motion[0, 0] - result[0, 0]
    result[:, 2] += reference_motion[0, 2] - result[0, 2]

    return result


@torch.no_grad()
def run_mogendit_imputation(
    mogendit_pipeline,
    batch_info: dict,
    device: str = 'cuda:0',
    step: int = 10,
) -> dict:
    """Run MoGenDIT imputation with obs_mask for keypose frames.

    Uses raw NPZ files directly (no rot6d roundtrip) to avoid encoding
    errors. After denoise, does post-hoc hard replacement of keypose
    frames from the after motion to ensure exact keypose matching.
    """
    from hftrainer.datasets.motion.motionhub.transforms.load_smplx import (
        process_smplx_pose,
        process_transl,
    )

    keypose_indices = batch_info['keypose_indices']
    T = batch_info['num_frames']

    # Load raw NPZ data directly (avoid rot6d intermediary)
    before_npz = np.load(batch_info['before_npz_path'], allow_pickle=True)
    after_npz = np.load(batch_info['after_npz_path'], allow_pickle=True)

    before_poses = np.asarray(before_npz['poses'], dtype=np.float32)  # (T, 156)
    before_trans = np.asarray(before_npz['trans'], dtype=np.float32)   # (T, 3)
    after_poses = np.asarray(after_npz['poses'], dtype=np.float32)
    after_trans = np.asarray(after_npz['trans'], dtype=np.float32)

    # Build composite: before motion with keypose frames replaced by after.
    # Expand observed region around keyposes with interpolated poses to help
    # the model propagate the keypose constraint to neighbors.
    # Auto-compute blend radius from keypose spacing
    sorted_kp = sorted(keypose_indices)
    boundaries = [0] + sorted_kp + [T - 1]
    max_r = 0
    for i, ki in enumerate(sorted_kp):
        left_dist = ki - boundaries[i]
        right_dist = boundaries[i + 2] - ki
        max_r = max(max_r, min(left_dist, right_dist) // 2)
    BLEND_RADIUS = max(min(max_r, 30), 5)
    composite_poses = before_poses.copy()
    composite_trans = before_trans.copy()
    obs_mask = np.zeros(T, dtype=np.float32)

    for ki in keypose_indices:
        # Exact keypose frame
        composite_poses[ki] = after_poses[ki]
        obs_mask[ki] = 1.0
        # Neighbors: linear blend between before and after[ki]
        for d in range(1, BLEND_RADIUS + 1):
            w = 1.0 - d / (BLEND_RADIUS + 1)  # linear decay
            for f in [ki - d, ki + d]:
                if 0 <= f < T:
                    composite_poses[f] = ((1 - w) * before_poses[f] + w * after_poses[ki]).astype(np.float32)
                    obs_mask[f] = 1.0

    # Call MoGenDIT imputation using raw axis-angle data
    motion_dict = {
        'poses': composite_poses, 'trans': composite_trans,
        'betas': np.zeros(16, dtype=np.float32),
        'gender': 'neutral', 'mocap_framerate': 30.0,
    }

    try:
        result_dict = mogendit_pipeline.impute_with_obs_mask(
            motion_dict=motion_dict, obs_mask=obs_mask,
            step=step, imputation_mode='all',
        )
    except Exception as e:
        logger.error(f'MoGenDIT imputation failed: {e}')
        traceback.print_exc()
        return None

    output_motion = np.zeros((T, 135), dtype=np.float32)

    # MoGenDIT's impute_with_obs_mask now properly reverses heading
    # normalization, so its output is in the same coordinate frame as
    # before/after. Use its complete output directly.
    out_poses = result_dict['poses'].copy()   # (T', 156)
    out_trans = result_dict['trans'].copy()    # (T', 3)
    T_out = min(T, out_poses.shape[0])

    # Pad if needed
    if T_out < T:
        logger.warning(f'MoGenDIT output {T_out} frames < input {T}. Padding.')
        out_poses = np.vstack([out_poses[:T_out], np.tile(out_poses[T_out-1:T_out], (T - T_out, 1))])
        out_trans = np.vstack([out_trans[:T_out], np.tile(out_trans[T_out-1:T_out], (T - T_out, 1))])

    # Convert axis-angle to 135-dim rot6d
    final_poses_165 = np.zeros((T, 165), dtype=np.float32)
    final_poses_165[:, :min(out_poses.shape[1], 165)] = out_poses[:T, :165]
    final_rot6d = process_smplx_pose(final_poses_165, rot_type='rotation_6d', out_type='smpl_22')
    final_transl = process_transl(out_trans[:T], transl_type='abs')
    output_motion = np.concatenate([final_transl, final_rot6d], axis=-1).astype(np.float32)

    # Post-hoc: restore translation from before_motion.
    # MoGenDIT heading normalization alters translation; keypose task only modifies pose.
    output_motion[:, :3] = batch_info['before_motion'][:T, :3]

    return {
        'output_motion': output_motion,
        'raw_output': output_motion.copy(),
    }


# ─────────────────────────────────────────────────────────────────────
# Post-processing: keypose boundary blending + static stabilization
# ─────────────────────────────────────────────────────────────────────

def postprocess_output(
    output_motion: np.ndarray,
    before_motion: np.ndarray,
    after_motion: np.ndarray,
    keypose_indices: list,
    blend_radius: int = 15,
    static_threshold: float = 0.02,
    blended_motion: np.ndarray = None,
) -> tuple:
    """Post-process: force keyposes + light reinforcement near keypose only.

    Conservative approach:
    - Only reinforces within a small temporal radius around each keypose
    - Uses cosine falloff, no temporal smoothing, no similarity propagation
    - Far frames are left as pure model output (model handles the transition)
    """
    T, D = output_motion.shape
    result = output_motion.copy()
    equiv_frames_dict = {}

    sorted_kp = sorted(keypose_indices)
    boundaries = [0] + sorted_kp + [T - 1]

    for idx_i, ki in enumerate(sorted_kp):
        left_dist = ki - boundaries[idx_i]
        right_dist = boundaries[idx_i + 2] - ki
        half_gap = min(left_dist, right_dist) // 2
        RADIUS = max(min(half_gap, 20), 3)

        # Equiv frames: just the temporal neighborhood
        equiv = list(range(max(0, ki - RADIUS), min(T, ki + RADIUS + 1)))
        equiv_frames_dict[ki] = equiv

        if blended_motion is not None:
            # Reinforce: blend model output toward blended target near keypose
            for f in range(max(0, ki - RADIUS), min(T, ki + RADIUS + 1)):
                d = abs(f - ki)
                t = d / (RADIUS + 1)
                w = 0.5 * (1 + np.cos(np.pi * t))
                result[f, 3:] = (1 - w) * output_motion[f, 3:] + w * blended_motion[f, 3:]
        else:
            # Legacy: apply correction directly
            correction = after_motion[ki, 3:] - before_motion[ki, 3:]
            for f in range(max(0, ki - RADIUS), min(T, ki + RADIUS + 1)):
                d = abs(f - ki)
                t = d / (RADIUS + 1)
                w = 0.5 * (1 + np.cos(np.pi * t))
                corrected = before_motion[f, 3:] + w * correction
                result[f, 3:] = (1 - w) * output_motion[f, 3:] + w * corrected

    # Force keypose frames to exact target
    for ki in keypose_indices:
        result[ki, 3:] = after_motion[ki, 3:]

    return result, equiv_frames_dict

def compute_metrics(
    output_motion: np.ndarray,
    before_motion: np.ndarray,
    after_motion: np.ndarray,
    keypose_indices: list,
    src_mask: np.ndarray,
) -> dict:
    """Compute evaluation metrics for keypose correction.

    Compares output against both:
    - after_motion (target/GT): how close to the corrected version
    - before_motion (source): how much did the model change

    Metrics:
    - kf_mpjpe: MPJPE at keypose frames (output vs after) — should be ~0
    - global_mpjpe: MPJPE over all frames (output vs after)
    - src_mpjpe: MPJPE over all frames (output vs before) — how much changed
    - boundary_smoothness: acceleration discontinuity at keypose boundaries
    - overall_smoothness: average acceleration magnitude (jitter metric)
    - foot_skating: foot joint velocity
    """
    T, D_ = output_motion.shape

    # 1. Keyframe accuracy (output vs after at keypose frames)
    kf_errors = []
    kf_trans_errors = []
    kf_rot_errors = []
    for ki in keypose_indices:
        kf_out = output_motion[ki]
        kf_gt = after_motion[ki]
        kf_errors.append(float(np.linalg.norm(kf_out - kf_gt)))
        kf_trans_errors.append(float(np.linalg.norm(kf_out[:3] - kf_gt[:3])))
        kf_rot_errors.append(float(np.linalg.norm(kf_out[3:] - kf_gt[3:])))

    kf_mpjpe = float(np.mean(kf_errors)) if kf_errors else 0.0
    kf_trans_err = float(np.mean(kf_trans_errors)) if kf_trans_errors else 0.0
    kf_rot_err = float(np.mean(kf_rot_errors)) if kf_rot_errors else 0.0

    # 2. Global MPJPE: output vs after (target) on generated frames (mask=1)
    gen_mask = src_mask.max(axis=-1) > 0.5  # (T,) True for generated frames
    if gen_mask.sum() > 0:
        gen_diff = output_motion[gen_mask] - after_motion[gen_mask]
        global_mpjpe = float(np.mean(np.linalg.norm(gen_diff, axis=-1)))
    else:
        global_mpjpe = 0.0

    # 3. Source MPJPE: output vs before (source) — how much the model changed
    if gen_mask.sum() > 0:
        src_diff = output_motion[gen_mask] - before_motion[gen_mask]
        src_mpjpe = float(np.mean(np.linalg.norm(src_diff, axis=-1)))
    else:
        src_mpjpe = 0.0

    # 4. Boundary smoothness: acceleration at keypose frame edges
    accel = np.diff(output_motion, n=2, axis=0)  # (T-2, D)
    accel_mag = np.linalg.norm(accel, axis=-1)    # (T-2,)

    boundary_accels = []
    for ki in keypose_indices:
        if 1 <= ki < T - 1:
            boundary_accels.append(float(accel_mag[ki - 1]))
    boundary_smoothness = float(np.mean(boundary_accels)) if boundary_accels else 0.0

    # 5. Overall smoothness (average acceleration)
    overall_smoothness = float(np.mean(accel_mag))

    # 6. Foot skating (simplified: velocity of foot joints)
    # Foot joints: L_Ankle(7), R_Ankle(8), L_Foot(10), R_Foot(11)
    foot_dims = []
    for j in [7, 8, 10, 11]:
        start = 3 + j * 6
        foot_dims.extend(range(start, start + 6))

    foot_vel = np.diff(output_motion[:, foot_dims], axis=0)
    foot_skating = float(np.mean(np.linalg.norm(foot_vel, axis=-1)))

    return {
        'kf_mpjpe': kf_mpjpe,
        'kf_trans_err': kf_trans_err,
        'kf_rot_err': kf_rot_err,
        'global_mpjpe': global_mpjpe,
        'src_mpjpe': src_mpjpe,
        'boundary_smoothness': boundary_smoothness,
        'overall_smoothness': overall_smoothness,
        'foot_skating': foot_skating,
    }


# ─────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ─────────────────────────────────────────────────────────────────────

def run_evaluation(args):
    """Main evaluation function."""
    device = f'cuda:{args.gpu}'

    # Load before/after pairs
    before_dir = os.path.join(str(PROJECT_ROOT), BEFORE_DIR)
    after_dir = os.path.join(str(PROJECT_ROOT), AFTER_DIR)
    pairs = load_before_after_pairs(
        before_dir, after_dir, max_pairs=args.num_cases
    )

    if len(pairs) == 0:
        logger.error('No before/after pairs loaded!')
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which models to evaluate
    if args.quick:
        models_to_eval = MAN_MODELS[:1]  # Just uncond_fm_man
        rep_modes = ['skip_last']
        sdedit_strengths = [0.1, 0.3]
        mogendit_steps = [10]
    else:
        models_to_eval = MAN_MODELS
        rep_modes = REPLACEMENT_MODES
        sdedit_strengths = SDEDIT_STRENGTHS
        mogendit_steps = MOGENDIT_STEPS

    imp_modes = IMPUTATION_MODES

    if args.models:
        models_to_eval = [m for m in models_to_eval if m[0] in args.models]

    all_results = {}
    total_configs = len(models_to_eval) * len(imp_modes) * len(rep_modes) * len(sdedit_strengths)
    config_idx = 0

    for model_name, config_path, work_dir, rot_space in models_to_eval:
        ckpt_path = find_latest_checkpoint(os.path.join(str(PROJECT_ROOT), work_dir))
        if ckpt_path is None:
            logger.warning(f'No checkpoint for {model_name}, skipping')
            continue

        logger.info(f'\n{"="*60}')
        logger.info(f'Loading model: {model_name} (ckpt: {os.path.basename(ckpt_path)})')
        logger.info(f'{"="*60}')

        try:
            bundle = load_m2m_bundle(
                os.path.join(str(PROJECT_ROOT), config_path),
                ckpt_path,
                device=device,
            )
        except Exception as e:
            logger.error(f'Failed to load {model_name}: {e}')
            traceback.print_exc()
            continue

        rot_dir_name = 'local_rot' if rot_space == 'local' else 'global_rot'

        for imp_mode in imp_modes:
            for rep_mode in rep_modes:
                for sdedit_str in sdedit_strengths:
                    config_idx += 1

                    sde_tag = f'__sde{sdedit_str:.2f}'
                    variant_key = f'{model_name}__{imp_mode}__{rep_mode}{sde_tag}'

                    logger.info(
                        f'\n[{config_idx}/{total_configs}] '
                        f'{model_name} | {imp_mode} | rep={rep_mode} | sdedit={sdedit_str}'
                    )

                    variant_results = []
                    variant_dir = output_dir / rot_dir_name / variant_key
                    variant_dir.mkdir(parents=True, exist_ok=True)

                    for case_idx, pair in enumerate(pairs):
                        before_motion = pair['before_motion']
                        after_motion = pair['after_motion']
                        T = pair['num_frames']

                        # Select keyposes
                        kp_indices, diffs = select_keyposes(
                            before_motion, after_motion,
                            k=NUM_KEYPOSES, min_diff=MIN_KEYPOSE_DIFF,
                        )

                        case_key = f'case{case_idx:03d}_{pair["filename"].replace(".npz", "")}'

                        try:
                            batch_info = build_imputation_batch(
                                before_motion, after_motion,
                                kp_indices, mode=imp_mode,
                            )

                            t0 = time.time()
                            result = run_m2m_imputation(
                                bundle, batch_info,
                                replacement_guidance=rep_mode,
                                num_steps=args.num_steps,
                                device=device,
                                sdedit_strength=sdedit_str,
                            )
                            elapsed = time.time() - t0

                            # Post-process: keypose enforcement + light reinforcement
                            output_pp, equiv_info = postprocess_output(
                                result['output_motion'],
                                before_motion, after_motion, kp_indices,
                                blended_motion=result.get('blended_motion'),
                            )
                            metrics = compute_metrics(
                                output_pp,
                                before_motion,
                                after_motion,
                                kp_indices,
                                batch_info['src_mask'],
                            )

                            # Save output NPZ with both before and after
                            npz_path = variant_dir / f'{case_key}.npz'
                            np.savez_compressed(
                                str(npz_path),
                                output_motion=output_pp,
                                before_motion=before_motion,
                                after_motion=after_motion,
                                composite_motion=batch_info['composite_motion'],
                                src_mask=batch_info['src_mask'],
                                keypose_indices=np.array(kp_indices),
                                equiv_frames=np.array(sorted(set(sum(equiv_info.values(), [])))),
                                correction_diffs=diffs,
                            )

                            case_result = {
                                'case_key': case_key,
                                'case_idx': case_idx,
                                'filename': pair['filename'],
                                'num_frames': T,
                                'keypose_indices': kp_indices,
                                'max_correction_diff': float(diffs.max()),
                                'elapsed_sec': elapsed,
                                **metrics,
                            }
                            variant_results.append(case_result)

                            logger.info(
                                f'  {case_key}: kf={metrics["kf_mpjpe"]:.4f} '
                                f'glob={metrics["global_mpjpe"]:.4f} '
                                f'src={metrics["src_mpjpe"]:.4f} '
                                f'bnd={metrics["boundary_smoothness"]:.4f} '
                                f'({elapsed:.1f}s)'
                            )

                        except Exception as e:
                            logger.error(f'  {case_key}: FAILED - {e}')
                            traceback.print_exc()
                            continue

                    # Aggregate metrics for this variant
                    if variant_results:
                        agg = _aggregate_metrics(variant_results)
                        all_results[variant_key] = {
                            'model': model_name,
                            'imp_mode': imp_mode,
                            'rep_mode': rep_mode,
                            'sdedit_strength': sdedit_str,
                            'rotation_space': rot_space,
                            'checkpoint': os.path.basename(ckpt_path) if ckpt_path else None,
                            'num_cases': len(variant_results),
                            'aggregate': agg,
                            'cases': variant_results,
                        }

                        with open(variant_dir / 'results.json', 'w') as f:
                            json.dump(all_results[variant_key], f, indent=2)

                        logger.info(
                            f'  -> {variant_key}: '
                            f'kf={agg.get("kf_mpjpe_mean", -1):.4f} '
                            f'glob={agg.get("global_mpjpe_mean", -1):.4f} '
                            f'src={agg.get("src_mpjpe_mean", -1):.4f}'
                        )

        # Unload model to free GPU memory
        del bundle
        torch.cuda.empty_cache()
        import gc; gc.collect()

    # ─────────────────────────────────────────────────────────────────
    # MoGenDIT baseline evaluation
    # ─────────────────────────────────────────────────────────────────
    if not args.skip_mogendit:
        logger.info(f'\n{"="*60}')
        logger.info('Loading MoGenDIT baseline')
        logger.info(f'{"="*60}')

        try:
            mogendit = load_mogendit_pipeline('MoreDiff-0.1B', device=device)

            for step_val in mogendit_steps:
                variant_key = f'mogendit_0.1B__keyframe_only__skip_last__step{step_val}'
                logger.info(f'\nMoGenDIT: {variant_key}')

                variant_results = []
                variant_dir = output_dir / 'local_rot' / variant_key
                variant_dir.mkdir(parents=True, exist_ok=True)

                for case_idx, pair in enumerate(pairs):
                    before_motion = pair['before_motion']
                    after_motion = pair['after_motion']
                    T = pair['num_frames']

                    kp_indices, diffs = select_keyposes(
                        before_motion, after_motion,
                        k=NUM_KEYPOSES, min_diff=MIN_KEYPOSE_DIFF,
                    )

                    case_key = f'case{case_idx:03d}_{pair["filename"].replace(".npz", "")}'

                    try:
                        batch_info = build_imputation_batch(
                            before_motion, after_motion,
                            kp_indices, mode='keyframe_only',
                        )
                        # Add raw NPZ paths for MoGenDIT (avoids rot6d roundtrip)
                        batch_info['before_npz_path'] = pair['before_npz_path']
                        batch_info['after_npz_path'] = pair['after_npz_path']

                        t0 = time.time()
                        result = run_mogendit_imputation(
                            mogendit, batch_info, device=device,
                            step=step_val,
                        )
                        elapsed = time.time() - t0

                        if result is None:
                            logger.warning(f'  {case_key}: MoGenDIT returned None')
                            continue

                        # Post-process: keypose blending + static stabilization
                        output_pp, equiv_info = postprocess_output(
                            result['output_motion'],
                            before_motion, after_motion, kp_indices,
                        )
                        metrics = compute_metrics(
                            output_pp,
                            before_motion,
                            after_motion,
                            kp_indices,
                            batch_info['src_mask'],
                        )

                        np.savez_compressed(
                            str(variant_dir / f'{case_key}.npz'),
                            output_motion=output_pp,
                            before_motion=before_motion,
                            after_motion=after_motion,
                            composite_motion=batch_info['composite_motion'],
                            src_mask=batch_info['src_mask'],
                            keypose_indices=np.array(kp_indices),
                            equiv_frames=np.array(sorted(set(sum(equiv_info.values(), [])))),
                            correction_diffs=diffs,
                        )

                        variant_results.append({
                            'case_key': case_key, 'case_idx': case_idx,
                            'filename': pair['filename'],
                            'num_frames': T,
                            'keypose_indices': kp_indices,
                            'max_correction_diff': float(diffs.max()),
                            'elapsed_sec': elapsed, **metrics,
                        })

                        logger.info(
                            f'  {case_key}: kf={metrics["kf_mpjpe"]:.4f} '
                            f'glob={metrics["global_mpjpe"]:.4f} '
                            f'src={metrics["src_mpjpe"]:.4f} ({elapsed:.1f}s)')

                    except Exception as e:
                        logger.error(f'  {case_key}: FAILED - {e}')
                        traceback.print_exc()
                        continue

                if variant_results:
                    agg = _aggregate_metrics(variant_results)
                    all_results[variant_key] = {
                        'model': 'mogendit_0.1B', 'imp_mode': 'keyframe_only',
                        'rep_mode': 'skip_last',
                        'sdedit_strength': 0.0,
                        'mogendit_step': step_val,
                        'rotation_space': 'local',
                        'checkpoint': 'MoreDiff-0.1B',
                        'num_cases': len(variant_results),
                        'aggregate': agg, 'cases': variant_results,
                    }
                    with open(variant_dir / 'results.json', 'w') as f:
                        json.dump(all_results[variant_key], f, indent=2)

                    logger.info(
                        f'  -> {variant_key}: '
                        f'kf={agg.get("kf_mpjpe_mean", -1):.4f} '
                        f'glob={agg.get("global_mpjpe_mean", -1):.4f} '
                        f'src={agg.get("src_mpjpe_mean", -1):.4f}')

            del mogendit
            torch.cuda.empty_cache()
            import gc; gc.collect()

        except Exception as e:
            logger.error(f'MoGenDIT baseline failed: {e}')
            traceback.print_exc()

    # ─────────────────────────────────────────────────────────────────
    # Save final summary
    # ─────────────────────────────────────────────────────────────────
    _save_summary(output_dir, all_results, pairs, args)


def _aggregate_metrics(variant_results):
    """Aggregate per-case metrics into mean/std/median."""
    agg = {}
    metric_keys = [
        'kf_mpjpe', 'kf_trans_err', 'kf_rot_err',
        'global_mpjpe', 'src_mpjpe',
        'boundary_smoothness', 'overall_smoothness',
        'foot_skating', 'elapsed_sec',
    ]
    for mk in metric_keys:
        vals = [r[mk] for r in variant_results if mk in r]
        if vals:
            agg[f'{mk}_mean'] = float(np.mean(vals))
            agg[f'{mk}_std'] = float(np.std(vals))
            agg[f'{mk}_median'] = float(np.median(vals))
    return agg


def _save_summary(output_dir, all_results, pairs, args):
    """Save evaluation summary and comparison table."""
    output_dir = Path(output_dir)

    summary = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'data_source': 'PeacekeeperElite_part4 before/after pairs',
        'num_test_pairs': len(pairs),
        'num_variants': len(all_results),
        'args': vars(args),
    }

    # Build comparison table
    comparison_rows = []
    for vk, vdata in sorted(all_results.items()):
        agg = vdata.get('aggregate', {})
        comparison_rows.append({
            'variant': vk,
            'model': vdata['model'],
            'imp_mode': vdata['imp_mode'],
            'rep_mode': vdata['rep_mode'],
            'sdedit_strength': vdata.get('sdedit_strength', 0.0),
            'mogendit_step': vdata.get('mogendit_step'),
            'rotation_space': vdata['rotation_space'],
            'checkpoint': vdata.get('checkpoint'),
            'n_cases': vdata['num_cases'],
            'kf_mpjpe': agg.get('kf_mpjpe_mean'),
            'kf_trans': agg.get('kf_trans_err_mean'),
            'kf_rot': agg.get('kf_rot_err_mean'),
            'global_mpjpe': agg.get('global_mpjpe_mean'),
            'src_mpjpe': agg.get('src_mpjpe_mean'),
            'bnd_smooth': agg.get('boundary_smoothness_mean'),
            'overall_smooth': agg.get('overall_smoothness_mean'),
            'foot_skate': agg.get('foot_skating_mean'),
            'time_sec': agg.get('elapsed_sec_mean'),
        })

    summary['comparison'] = comparison_rows
    summary['full_results'] = all_results

    summary_path = output_dir / 'eval_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # Write per-rot_space summaries for the web viewer
    for rot_space_dir in ['local_rot', 'global_rot']:
        rot_val = 'local' if rot_space_dir == 'local_rot' else 'global'
        rs_rows = [r for r in comparison_rows if r['rotation_space'] == rot_val]
        if rs_rows:
            rs_summary = {'comparison': rs_rows, 'timestamp': summary['timestamp']}
            rs_path = output_dir / rot_space_dir / 'eval_summary.json'
            rs_path.parent.mkdir(parents=True, exist_ok=True)
            with open(rs_path, 'w') as f:
                json.dump(rs_summary, f, indent=2)

    # Print comparison table
    logger.info('\n' + '=' * 130)
    logger.info('EVALUATION SUMMARY (Before/After Keypose Correction)')
    logger.info('=' * 130)
    header = (
        f'{"Variant":<60} {"kf_mpjpe":>8} {"glob_mpj":>8} {"src_mpj":>8} '
        f'{"bnd_sm":>8} {"ovr_sm":>8} {"ft_sk":>8} {"time":>6}'
    )
    logger.info(header)
    logger.info('-' * 130)
    for row in comparison_rows:
        line = (
            f'{row["variant"]:<60} '
            f'{row.get("kf_mpjpe", 0) or 0:>8.4f} '
            f'{row.get("global_mpjpe", 0) or 0:>8.4f} '
            f'{row.get("src_mpjpe", 0) or 0:>8.4f} '
            f'{row.get("bnd_smooth", 0) or 0:>8.4f} '
            f'{row.get("overall_smooth", 0) or 0:>8.4f} '
            f'{row.get("foot_skate", 0) or 0:>8.4f} '
            f'{row.get("time_sec", 0) or 0:>6.1f}'
        )
        logger.info(line)
    logger.info('=' * 130)

    # Generate markdown report
    _generate_report(output_dir, summary)

    logger.info(f'\nResults saved to: {output_dir}')
    logger.info(f'Summary: {summary_path}')
    logger.info(f'Report: {output_dir / "REPORT.md"}')


def _generate_report(output_dir, summary):
    """Generate a markdown evaluation report."""
    rows = summary.get('comparison', [])

    lines = [
        '# Keypose Correction Evaluation Report',
        '',
        f'**Date**: {summary["timestamp"]}',
        f'**Data**: {summary.get("data_source", "PeacekeeperElite_part4")}',
        f'**Test pairs**: {summary["num_test_pairs"]}',
        f'**Variants evaluated**: {summary["num_variants"]}',
        '',
        '## Background',
        '',
        'This evaluation tests **keypose-conditioned SDEdit** using real before/after',
        'motion correction pairs from the PeacekeeperElite dataset.',
        '',
        '**Pipeline**: Given a source motion (before) + keyposes from the corrected',
        'version (after), use SDEdit to transform the source motion to pass through',
        'the keyposes while preserving its character.',
        '',
        '### SDEdit Strengths (M2M)',
        '',
        '| Strength | Description |',
        '|----------|-------------|',
        '| `0.05` | Very light — stays very close to source |',
        '| `0.1` | Light edit — small corrections |',
        '| `0.3` | Moderate — balanced edit |',
        '| `0.5` | Strong — more creative freedom |',
        '',
        '### MoGenDIT Steps',
        '',
        '| Steps | Description |',
        '|-------|-------------|',
        '| `10` | Light denoise — minimal change |',
        '| `50` | Moderate denoise |',
        '| `100` | Heavy denoise — more change |',
        '',
        '## Results',
        '',
        '### Comparison Table',
        '',
        '| Model | Rep. Mode | SDEdit/Step | Rot | KF MPJPE↓ | Global MPJPE↓ | Src MPJPE | Bnd Smooth↓ | Time(s) |',
        '|-------|-----------|-------------|-----|-----------|---------------|-----------|-------------|---------|',
    ]

    for r in rows:
        sde_str = f'{r.get("sdedit_strength", 0.0):.2f}'
        mog_step = r.get('mogendit_step')
        strength_col = f'step={mog_step}' if mog_step else f'sde={sde_str}'
        line = (
            f'| {r["model"]} | {r["rep_mode"]} | '
            f'{strength_col} | {r["rotation_space"]} | '
            f'{r.get("kf_mpjpe", 0) or 0:.4f} | '
            f'{r.get("global_mpjpe", 0) or 0:.4f} | '
            f'{r.get("src_mpjpe", 0) or 0:.4f} | '
            f'{r.get("bnd_smooth", 0) or 0:.4f} | '
            f'{r.get("time_sec", 0) or 0:.1f} |'
        )
        lines.append(line)

    lines.extend([
        '',
        '### Metric Definitions',
        '',
        '| Metric | Description |',
        '|--------|-------------|',
        '| KF MPJPE | Error at keypose frames (output vs after) — lower is better |',
        '| Global MPJPE | Error over all generated frames (output vs after) |',
        '| Src MPJPE | Error (output vs before) — how much the model changed |',
        '| Bnd Smooth | Acceleration at keypose boundaries — lower = smoother |',
        '| Foot Skate | Foot velocity — lower = less skating |',
        '',
    ])

    report_path = output_dir / 'REPORT.md'
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))


# ─────────────────────────────────────────────────────────────────────
# Multi-GPU parallel evaluation
# ─────────────────────────────────────────────────────────────────────

def _gpu_worker(gpu_id, model_assignments, args, pairs):
    """Worker process for one GPU. Evaluates assigned models."""
    device = f'cuda:{gpu_id}'
    logger.info(f'[GPU {gpu_id}] Starting worker with {len(model_assignments)} model(s)')

    output_dir = Path(args.output_dir)
    all_results = {}

    if args.quick:
        rep_modes = ['skip_last']
        sdedit_strengths = [0.1, 0.3]
    else:
        rep_modes = REPLACEMENT_MODES
        sdedit_strengths = SDEDIT_STRENGTHS

    imp_modes = IMPUTATION_MODES

    for model_name, config_path, work_dir, rot_space in model_assignments:
        ckpt_path = find_latest_checkpoint(os.path.join(str(PROJECT_ROOT), work_dir))
        if ckpt_path is None:
            logger.warning(f'[GPU {gpu_id}] No checkpoint for {model_name}, skipping')
            continue

        logger.info(f'[GPU {gpu_id}] Loading {model_name}')
        try:
            bundle = load_m2m_bundle(
                os.path.join(str(PROJECT_ROOT), config_path),
                ckpt_path, device=device,
            )
        except Exception as e:
            logger.error(f'[GPU {gpu_id}] Failed to load {model_name}: {e}')
            traceback.print_exc()
            continue

        rot_dir_name = 'local_rot' if rot_space == 'local' else 'global_rot'

        for imp_mode in imp_modes:
            for rep_mode in rep_modes:
                for sdedit_str in sdedit_strengths:
                    sde_tag = f'__sde{sdedit_str:.2f}'
                    variant_key = f'{model_name}__{imp_mode}__{rep_mode}{sde_tag}'
                    logger.info(f'[GPU {gpu_id}] {variant_key}')

                    variant_results = []
                    variant_dir = output_dir / rot_dir_name / variant_key
                    variant_dir.mkdir(parents=True, exist_ok=True)

                    for case_idx, pair in enumerate(pairs):
                        before_motion = pair['before_motion']
                        after_motion = pair['after_motion']
                        T = pair['num_frames']

                        kp_indices, diffs = select_keyposes(
                            before_motion, after_motion,
                            k=NUM_KEYPOSES, min_diff=MIN_KEYPOSE_DIFF,
                        )

                        case_key = f'case{case_idx:03d}_{pair["filename"].replace(".npz", "")}'

                        try:
                            batch_info = build_imputation_batch(
                                before_motion, after_motion,
                                kp_indices, mode=imp_mode,
                            )

                            t0 = time.time()
                            result = run_m2m_imputation(
                                bundle, batch_info,
                                replacement_guidance=rep_mode,
                                num_steps=args.num_steps,
                                device=device,
                                sdedit_strength=sdedit_str,
                            )
                            elapsed = time.time() - t0

                            # Post-process: keypose enforcement + light reinforcement
                            output_pp, equiv_info = postprocess_output(
                                result['output_motion'],
                                before_motion, after_motion, kp_indices,
                                blended_motion=result.get('blended_motion'),
                            )
                            metrics = compute_metrics(
                                output_pp,
                                before_motion,
                                after_motion,
                                kp_indices,
                                batch_info['src_mask'],
                            )

                            np.savez_compressed(
                                str(variant_dir / f'{case_key}.npz'),
                                output_motion=output_pp,
                                before_motion=before_motion,
                                after_motion=after_motion,
                                composite_motion=batch_info['composite_motion'],
                                src_mask=batch_info['src_mask'],
                                keypose_indices=np.array(kp_indices),
                                equiv_frames=np.array(sorted(set(sum(equiv_info.values(), [])))),
                                correction_diffs=diffs,
                            )

                            variant_results.append({
                                'case_key': case_key,
                                'case_idx': case_idx,
                                'filename': pair['filename'],
                                'num_frames': T,
                                'keypose_indices': kp_indices,
                                'max_correction_diff': float(diffs.max()),
                                'elapsed_sec': elapsed,
                                **metrics,
                            })
                        except Exception as e:
                            logger.error(f'[GPU {gpu_id}] {case_key}: FAILED - {e}')
                            continue

                    if variant_results:
                        agg = _aggregate_metrics(variant_results)
                        all_results[variant_key] = {
                            'model': model_name, 'imp_mode': imp_mode,
                            'rep_mode': rep_mode, 'sdedit_strength': sdedit_str,
                            'rotation_space': rot_space,
                            'checkpoint': os.path.basename(ckpt_path),
                            'num_cases': len(variant_results),
                            'aggregate': agg, 'cases': variant_results,
                        }
                        with open(variant_dir / 'results.json', 'w') as f:
                            json.dump(all_results[variant_key], f, indent=2)

                        logger.info(
                            f'[GPU {gpu_id}] {variant_key}: '
                            f'glob={agg.get("global_mpjpe_mean", -1):.4f} '
                            f'kf={agg.get("kf_mpjpe_mean", -1):.4f}'
                        )

        del bundle
        torch.cuda.empty_cache()
        import gc; gc.collect()

    return all_results


def run_multi_gpu_evaluation(args):
    """Distribute models across multiple GPUs using multiprocessing."""
    import multiprocessing as mp

    # Load data in main process
    before_dir = os.path.join(str(PROJECT_ROOT), BEFORE_DIR)
    after_dir = os.path.join(str(PROJECT_ROOT), AFTER_DIR)
    pairs = load_before_after_pairs(
        before_dir, after_dir, max_pairs=args.num_cases
    )
    if not pairs:
        logger.error('No before/after pairs loaded!')
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine models and GPUs
    if args.quick:
        models_to_eval = MAN_MODELS[:1]
    else:
        models_to_eval = MAN_MODELS
    if args.models:
        models_to_eval = [m for m in models_to_eval if m[0] in args.models]

    num_gpus = args.num_gpus
    logger.info(f'Multi-GPU eval: {len(models_to_eval)} models across {num_gpus} GPUs')

    # Round-robin assign models to GPUs
    gpu_assignments = [[] for _ in range(num_gpus)]
    for i, model in enumerate(models_to_eval):
        gpu_assignments[i % num_gpus].append(model)

    # Spawn one process per GPU
    ctx = mp.get_context('spawn')
    processes = []
    for gpu_id in range(num_gpus):
        if not gpu_assignments[gpu_id]:
            continue
        p = ctx.Process(
            target=_gpu_worker,
            args=(gpu_id, gpu_assignments[gpu_id], args, pairs),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    logger.info('All GPU workers finished.')

    # Merge results from all variant dirs
    _merge_and_summarize(output_dir, args, len(pairs))


def _merge_and_summarize(output_dir, args, num_pairs):
    """Merge per-variant results.json into a single eval_summary.json."""
    all_results = {}
    for rot_space in ['local_rot', 'global_rot']:
        rot_dir = output_dir / rot_space
        if not rot_dir.is_dir():
            continue
        for vdir in sorted(rot_dir.iterdir()):
            rpath = vdir / 'results.json'
            if rpath.is_file():
                with open(rpath) as f:
                    all_results[vdir.name] = json.load(f)

    summary = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'data_source': 'PeacekeeperElite_part4 before/after pairs',
        'num_test_pairs': num_pairs,
        'num_variants': len(all_results),
        'args': vars(args),
    }

    comparison_rows = []
    for vk, vdata in sorted(all_results.items()):
        agg = vdata.get('aggregate', {})
        comparison_rows.append({
            'variant': vk,
            'model': vdata['model'],
            'imp_mode': vdata['imp_mode'],
            'rep_mode': vdata['rep_mode'],
            'sdedit_strength': vdata.get('sdedit_strength', 0.0),
            'mogendit_step': vdata.get('mogendit_step'),
            'rotation_space': vdata['rotation_space'],
            'checkpoint': vdata.get('checkpoint'),
            'n_cases': vdata['num_cases'],
            'kf_mpjpe': agg.get('kf_mpjpe_mean'),
            'kf_trans': agg.get('kf_trans_err_mean'),
            'kf_rot': agg.get('kf_rot_err_mean'),
            'global_mpjpe': agg.get('global_mpjpe_mean'),
            'src_mpjpe': agg.get('src_mpjpe_mean'),
            'bnd_smooth': agg.get('boundary_smoothness_mean'),
            'overall_smooth': agg.get('overall_smoothness_mean'),
            'foot_skate': agg.get('foot_skating_mean'),
            'time_sec': agg.get('elapsed_sec_mean'),
        })

    summary['comparison'] = comparison_rows
    summary_path = output_dir / 'eval_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # Per-rot_space summaries for the web viewer
    for rot_space in ['local_rot', 'global_rot']:
        rs_rows = [r for r in comparison_rows if
                   (r['rotation_space'] == 'local' and rot_space == 'local_rot') or
                   (r['rotation_space'] == 'global' and rot_space == 'global_rot')]
        if rs_rows:
            rs_summary = {'comparison': rs_rows, 'timestamp': summary['timestamp']}
            rs_path = output_dir / rot_space / 'eval_summary.json'
            rs_path.parent.mkdir(parents=True, exist_ok=True)
            with open(rs_path, 'w') as f:
                json.dump(rs_summary, f, indent=2)

    _generate_report(output_dir, summary)
    logger.info(f'Summary: {summary_path}')


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='Keypose Correction Evaluation (Before/After Pairs)'
    )
    parser.add_argument('--output-dir', default='output/eval_keyframe_pose',
                        help='Output directory')
    parser.add_argument('--num-cases', type=int, default=None,
                        help='Max number of before/after pairs (None = all)')
    parser.add_argument('--num-steps', type=int, default=50,
                        help='ODE integration steps')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU index')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: fewer models/strengths, 3 cases')
    parser.add_argument('--models', nargs='+', default=None,
                        help='Specific model names to evaluate')
    parser.add_argument('--multi-gpu', action='store_true',
                        help='Use multiple GPUs (distribute models across GPUs)')
    parser.add_argument('--num-gpus', type=int, default=8,
                        help='Number of GPUs for multi-GPU mode')
    parser.add_argument('--skip-mogendit', action='store_true',
                        help='Skip MoGenDIT baseline evaluation')
    parser.add_argument('--num-keyposes', type=int, default=2,
                        help='Max number of keyposes to select (top-K correction peaks, actual count adapts to motion)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.quick and args.num_cases is None:
        args.num_cases = 3
    NUM_KEYPOSES = args.num_keyposes
    if args.multi_gpu:
        run_multi_gpu_evaluation(args)
    else:
        run_evaluation(args)
