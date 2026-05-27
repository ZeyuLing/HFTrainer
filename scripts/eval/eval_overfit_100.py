#!/usr/bin/env python3
"""Overfitting Verification — Evaluate overfit-100 model on its own training set.

Purpose:
  After the overfit model converges (loss → ~0), this script tests if it can
  faithfully reproduce training motions given various condition modes:
    1. text_only      — Full generation with caption (mask=1 everywhere)
    2. text_frame     — In-betweening: first/last 20% observed + caption
    3. text_upper     — Upper body edit: lower body preserved, upper generated
    4. text_lower     — Lower body edit: upper body preserved, lower generated
    5. frame_head     — Head completion: only first frame observed
    6. frame_tail     — Tail completion: only last frame observed
    7. keyframe_periodic — Sparse keyframes: every 10th frame observed
    8. trans_only     — Trajectory condition: only root translation observed

  If the model is correctly implemented, MPJPE should be near-zero for all tasks
  on the 100-sample training set.

Usage:
    # Text-only generation
    python scripts/eval/eval_overfit_100.py --mode text_only

    # Text + frame condition (in-betweening)
    python scripts/eval/eval_overfit_100.py --mode text_frame

    # Text + upper body edit
    python scripts/eval/eval_overfit_100.py --mode text_upper

    # Lower body edit
    python scripts/eval/eval_overfit_100.py --mode text_lower

    # Head completion (continuation from first frame)
    python scripts/eval/eval_overfit_100.py --mode frame_head

    # Tail completion (generate prefix given last frame)
    python scripts/eval/eval_overfit_100.py --mode frame_tail

    # Periodic keyframe interpolation
    python scripts/eval/eval_overfit_100.py --mode keyframe_periodic

    # Trajectory-only condition
    python scripts/eval/eval_overfit_100.py --mode trans_only

    # All modes
    python scripts/eval/eval_overfit_100.py --mode all

    # Specify checkpoint epoch
    python scripts/eval/eval_overfit_100.py --mode all --epoch 5000

    # Save NPZ for visualization
    python scripts/eval/eval_overfit_100.py --mode all --save-npz

Requires: torch>=2.0, mmengine, safetensors
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# =============================================================================
# Constants
# =============================================================================

CONFIG_PATH = 'configs/hymotion_m2m_v2/hymotion_m2m_v2_overfit_100_caption_046b.py'
WORK_DIR = 'work_dirs/hymotion_m2m_v2_overfit_100_v2'
ANNO_FILE = 'data/annotation/overfit_100_caption_20260526.json'
MOTION_DATA_DIR = 'data/motionhub'
BONE_OFFSETS_PATH = 'data/hymotion_m2m_data/bone_offsets_22.pt'

MOTION_DIM = 198
T_PAD = 360  # Training always pads to 360

# SMPL-22 joint groups for mask construction
# Group layout: [transl(3)] + [joint_0_rot(6)] + ... + [joint_21_rot(6)] + [joint_1_pos(3)] + ... + [joint_21_pos(3)]
# 23 atomic groups: 1 (trans) + 22 (joint rotations)
# Position channels (63 dims) are coupled with their rotation counterparts
N_JOINTS = 22
TRANSL_SLICE = slice(0, 3)
ROT_SLICE = slice(3, 135)  # 22*6 = 132
POS_SLICE = slice(135, 198)  # 21*3 = 63 (pelvis excluded)

# Upper body joints (SMPL-22): neck(12), l_collar(13), r_collar(14), head(15),
# l_shoulder(16), r_shoulder(17), l_elbow(18), r_elbow(19), l_wrist(20), r_wrist(21)
UPPER_BODY_JOINTS = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
# Lower body joints: pelvis(0), l_hip(1), r_hip(2), spine1(3), l_knee(4),
# r_knee(5), spine2(6), l_ankle(7), r_ankle(8), spine3(9), l_foot(10), r_foot(11)
LOWER_BODY_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]


def build_mask_text_only(T: int, D: int = 198) -> np.ndarray:
    """Full generation mask: all dims are target (mask=1)."""
    mask = np.ones((T, D), dtype=np.float32)
    return mask


def build_mask_inbetweening(T: int, D: int = 198, obs_ratio: float = 0.2) -> np.ndarray:
    """In-betweening: first and last `obs_ratio` fraction are observed (mask=0)."""
    mask = np.ones((T, D), dtype=np.float32)
    n_obs = max(1, int(T * obs_ratio))
    mask[:n_obs, :] = 0.0  # First 20% observed
    mask[-n_obs:, :] = 0.0  # Last 20% observed
    return mask


def build_mask_upper_body_edit(T: int, D: int = 198) -> np.ndarray:
    """Upper body edit: upper body joints are target (mask=1), lower body preserved (mask=0)."""
    mask = np.zeros((T, D), dtype=np.float32)
    # Mark upper body rotation dims as target
    for j in UPPER_BODY_JOINTS:
        rot_start = 3 + j * 6
        rot_end = 3 + (j + 1) * 6
        mask[:, rot_start:rot_end] = 1.0
    # Mark corresponding position dims as target (joints 1-21 mapped to pos dims)
    # Position layout: joint 1 at pos_start+0, joint 2 at pos_start+3, ...
    # Position channel excludes pelvis (joint 0), so joint j maps to index j-1
    for j in UPPER_BODY_JOINTS:
        if j == 0:
            continue  # Pelvis has no position channel
        pos_idx = j - 1  # 0-indexed in position block
        pos_start = 135 + pos_idx * 3
        pos_end = 135 + (pos_idx + 1) * 3
        mask[:, pos_start:pos_end] = 1.0
    return mask


def build_mask_lower_body_edit(T: int, D: int = 198) -> np.ndarray:
    """Lower body edit: lower body joints are target (mask=1), upper body preserved (mask=0).

    This is the symmetric counterpart of upper_body_edit.
    Translation (root) is also generated since lower body includes pelvis.
    """
    mask = np.zeros((T, D), dtype=np.float32)
    # Translation is generated (pelvis = root motion)
    mask[:, :3] = 1.0
    # Mark lower body rotation dims as target
    for j in LOWER_BODY_JOINTS:
        rot_start = 3 + j * 6
        rot_end = 3 + (j + 1) * 6
        mask[:, rot_start:rot_end] = 1.0
    # Mark corresponding position dims as target
    for j in LOWER_BODY_JOINTS:
        if j == 0:
            continue  # Pelvis has no position channel
        pos_idx = j - 1
        pos_start = 135 + pos_idx * 3
        pos_end = 135 + (pos_idx + 1) * 3
        mask[:, pos_start:pos_end] = 1.0
    return mask


def build_mask_head_completion(T: int, D: int = 198, obs_frames: int = 1) -> np.ndarray:
    """Head completion: only first `obs_frames` frames observed, rest generated.

    Use case: given a starting pose, generate the continuation.
    """
    mask = np.ones((T, D), dtype=np.float32)
    mask[:obs_frames, :] = 0.0  # First frame(s) observed
    return mask


def build_mask_tail_completion(T: int, D: int = 198, obs_frames: int = 1) -> np.ndarray:
    """Tail completion: only last `obs_frames` frames observed, rest generated.

    Use case: given an ending pose, generate what comes before it.
    """
    mask = np.ones((T, D), dtype=np.float32)
    mask[-obs_frames:, :] = 0.0  # Last frame(s) observed
    return mask


def build_mask_periodic_keyframes(T: int, D: int = 198, period: int = 10) -> np.ndarray:
    """Periodic keyframe condition: every `period` frames are observed, rest generated.

    Use case: sparse keyframe interpolation — given every Nth frame, fill in between.
    """
    mask = np.ones((T, D), dtype=np.float32)
    for t in range(0, T, period):
        mask[t, :] = 0.0  # Keyframe observed
    return mask


def build_mask_trans_only(T: int, D: int = 198) -> np.ndarray:
    """Trajectory-only condition: only root translation (3 dims) is observed, rest generated.

    Use case: given a trajectory path, generate full body motion along it.
    """
    mask = np.ones((T, D), dtype=np.float32)
    mask[:, :3] = 0.0  # Translation (dims 0-2) observed
    return mask


def build_mask_keyframe_rot_only(T: int, D: int = 198, period: int = 10) -> np.ndarray:
    """Sparse keyframes with only ROTATION channels observed.

    At every `period` frames, joint rotations (dims 3:135) are observed.
    Translation and FK positions are always generated.
    Use case: given sparse rotation keyframes + text, generate full motion.
    """
    mask = np.ones((T, D), dtype=np.float32)
    for t in range(0, T, period):
        mask[t, 3:135] = 0.0  # Only rotation channels observed at keyframes
    return mask


def build_mask_keyframe_pos_only(T: int, D: int = 198, period: int = 10) -> np.ndarray:
    """Sparse keyframes with only FK POSITION channels observed.

    At every `period` frames, FK positions (dims 135:198) are observed.
    Translation and rotations are always generated.
    Use case: given sparse position keyframes + text, generate full motion.
    """
    mask = np.ones((T, D), dtype=np.float32)
    for t in range(0, T, period):
        mask[t, 135:198] = 0.0  # Only position channels observed at keyframes
    return mask


def build_mask_style_edit(T: int, D: int = 198) -> np.ndarray:
    """Style editing: trajectory (translation + root rotation) is preserved, body regenerated.

    Preserves root translation (dims 0:3) and root joint rotation (dims 3:9) for ALL frames.
    All other body rotations (dims 9:135) and FK positions (dims 135:198) are generated.
    Use case: keep path/facing direction, re-generate body articulation style from text.
    """
    mask = np.ones((T, D), dtype=np.float32)
    mask[:, 0:3] = 0.0   # Translation preserved (all frames)
    mask[:, 3:9] = 0.0   # Root rotation preserved (all frames)
    return mask


def load_annotation() -> Dict:
    """Load the overfit-100 annotation file."""
    with open(ANNO_FILE, 'r') as f:
        data = json.load(f)
    return data


def load_bone_offsets() -> np.ndarray:
    """Load SMPL-22 bone offsets for FK computation."""
    if os.path.exists(BONE_OFFSETS_PATH):
        if BONE_OFFSETS_PATH.endswith('.pt'):
            data = torch.load(BONE_OFFSETS_PATH, map_location='cpu', weights_only=False)
            if isinstance(data, torch.Tensor):
                return data.numpy()
            return np.array(data)
        return np.load(BONE_OFFSETS_PATH)
    # Fallback: try to find in mean_std_dir
    alt_path = 'data/hymotion_m2m_data/_stats_198dim/bone_offsets.npy'
    if os.path.exists(alt_path):
        return np.load(alt_path)
    raise FileNotFoundError(
        f"Bone offsets not found at {BONE_OFFSETS_PATH} or {alt_path}")


def motion_135_to_198(motion_135: np.ndarray, bone_offsets: np.ndarray) -> np.ndarray:
    """Convert 135-dim to 198-dim by appending FK position channels."""
    from hftrainer.evaluation.motion.m2m_eval_metrics import motion135_to_positions_np
    positions = motion135_to_positions_np(motion_135, bone_offsets)  # (T, 22, 3)
    joint_pos = positions[:, 1:, :]  # (T, 21, 3) — exclude pelvis
    pelvis_xz = positions[:, 0:1, [0, 2]]  # (T, 1, 2)
    joint_pos_rel = joint_pos.copy()
    joint_pos_rel[:, :, 0] -= pelvis_xz[:, :, 0]
    joint_pos_rel[:, :, 2] -= pelvis_xz[:, :, 1]
    pos_flat = joint_pos_rel.reshape(-1, 63)
    return np.concatenate([motion_135, pos_flat], axis=-1).astype(np.float32)


def load_motion_npz(motion_path: str) -> Optional[np.ndarray]:
    """Load a 135-dim motion from NPZ file.

    Handles both:
      1. Pre-computed format: NPZ with 'motion_135' key (T, 135)
      2. Raw SMPL format: NPZ with 'poses' (T, 165) + 'trans' (T, 3) axis-angle

    For raw SMPL format, replicates LoadSmplx55 logic:
      - poses (T, 165) axis-angle → rot6d (T, 22*6=132) row-major
      - trans (T, 3) absolute
      - concat [trans, rot6d] → (T, 135)

    Returns (T, 135) numpy array or None if loading fails.
    """
    from hftrainer.datasets.motion.motionhub.transforms.load_smplx import (
        process_smplx_pose,
        process_transl,
    )

    # Resolve path
    if not os.path.isabs(motion_path):
        full_path = os.path.join(MOTION_DATA_DIR, motion_path)
    else:
        full_path = motion_path

    if not os.path.exists(full_path):
        # Try alternate location (annotation uses ../hymotion_data/ relative paths)
        alt_path = os.path.normpath(os.path.join('data/annotation', motion_path))
        if os.path.exists(alt_path):
            full_path = alt_path
        else:
            alt_path2 = os.path.join('data/hymotion_data', motion_path)
            if os.path.exists(alt_path2):
                full_path = alt_path2
            else:
                print(f"  Warning: motion file not found: {motion_path}")
                return None

    try:
        npz = np.load(full_path, allow_pickle=True)

        # Case 1: Pre-computed motion_135 (e.g., PerMo)
        if 'motion_135' in npz:
            return np.asarray(npz['motion_135'], dtype=np.float32)

        # Case 2: Raw SMPL format — 'poses' + 'trans'
        # This is the format used by our overfit-100 dataset
        if 'poses' in npz and 'trans' in npz:
            abs_trans = np.asarray(npz['trans'], dtype=np.float32)  # (T, 3)
            poses = np.asarray(npz['poses'], dtype=np.float32)  # (T, 165) or (T, 55*3)

            # Convert poses (axis-angle) to rot6d, SMPL-22, row-major convention
            # Same as LoadSmplx55(rot_type='rotation_6d', smpl_type='smpl_22')
            rot6d = process_smplx_pose(
                poses, rot_type='rotation_6d', out_type='smpl_22',
                rot6d_convention='row',
            )  # (T, 132)

            # Process translation (absolute)
            transl = process_transl(abs_trans, transl_type='abs')  # (T, 3)

            # Concat [trans(3), rot6d(132)] = 135-dim
            motion_135 = np.concatenate([transl, rot6d], axis=-1)  # (T, 135)
            return motion_135.astype(np.float32)

        # Case 3: Legacy dict format with 'smplx' or 'motion' key
        if 'smplx' in npz:
            data = npz['smplx']
            if isinstance(data, np.ndarray) and data.dtype == object:
                data = data.item()
        elif 'motion' in npz:
            data = npz['motion']
            if isinstance(data, np.ndarray) and data.dtype == object:
                data = data.item()
        else:
            print(f"  Warning: unrecognized NPZ format (keys: {list(npz.keys())}): {full_path}")
            return None

        if isinstance(data, np.ndarray):
            if data.shape[-1] == 135:
                return data.astype(np.float32)
            elif data.shape[-1] == 198:
                return data[:, :135].astype(np.float32)

        return None
    except Exception as e:
        print(f"  Warning: failed to load {full_path}: {e}")
        return None


def load_text_embedding(entry: Dict) -> Optional[Dict[str, torch.Tensor]]:
    """Load pre-extracted text embedding for a sample.

    Returns dict with 'text_vec_raw', 'text_ctxt_raw', 'text_ctxt_raw_length'
    or None if not found.
    """
    # Try to get caption path from annotation
    # Priority: hierarchical_caption_path > caption_path (same as dataset class)
    caption_path = entry.get('hierarchical_caption_path', '') or entry.get('caption_path', '')
    if not caption_path:
        caption_info = entry.get('caption', {})
        if isinstance(caption_info, str):
            return None
        caption_path = caption_info.get('path', '') if isinstance(caption_info, dict) else ''
    if not caption_path:
        return None
    # Resolve relative path (annotation uses ../hymotion_data/... relative to data/annotation/)
    if not os.path.isabs(caption_path):
        caption_path = os.path.normpath(os.path.join('data/annotation', caption_path))

    # Use the same mapping as LoadPreExtractedTextEmbedding
    from hftrainer.datasets.motion.motionhub.transforms.load_text import (
        _caption_path_to_embedding_path,
    )

    try:
        embed_path = _caption_path_to_embedding_path(caption_path)
        if embed_path and os.path.exists(embed_path):
            data = torch.load(embed_path, map_location='cpu', weights_only=False)
            result = data['result'][0]  # First caption variant
            return {
                'text_vec_raw': result['text_embedding']['text_vec_raw'],  # (1, 1, 768)
                'text_ctxt_raw': result['text_embedding']['text_ctxt_raw'],  # (1, seq, 4096)
                'text_ctxt_raw_length': result['text_embedding']['text_ctxt_raw_length'],  # (1,)
            }
    except Exception as e:
        print(f"  Warning: failed to load text embedding for {caption_path}: {e}")

    return None


def load_model(device: str, epoch: Optional[int] = None):
    """Load the overfit-100 model bundle and create pipeline.

    Returns: bundle, pipeline, ckpt_path
    """
    from mmengine.config import Config
    from hftrainer.registry import MODEL_BUNDLES
    from hftrainer.utils.checkpoint_utils import load_checkpoint, find_latest_checkpoint
    from hftrainer.pipelines.motion.hymotion_m2m_pipeline import HyMotionM2MPipeline

    cfg = Config.fromfile(CONFIG_PATH)
    bundle = MODEL_BUNDLES.build(cfg.model.to_dict())

    if epoch is not None:
        ckpt_path = os.path.join(WORK_DIR, f'checkpoint-epoch_{epoch}')
        if not os.path.exists(ckpt_path):
            # Try the timestamped subdirectory
            subdirs = [d for d in os.listdir(WORK_DIR) if d.startswith('2026')]
            if subdirs:
                ckpt_path = os.path.join(WORK_DIR, subdirs[0], f'checkpoint-epoch_{epoch}')
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    else:
        # Find latest checkpoint
        ckpt_path = find_latest_checkpoint(WORK_DIR)
        if ckpt_path is None:
            # Try subdirectory
            subdirs = sorted([d for d in os.listdir(WORK_DIR) if d.startswith('2026')])
            if subdirs:
                ckpt_path = find_latest_checkpoint(os.path.join(WORK_DIR, subdirs[-1]))
        if ckpt_path is None:
            raise FileNotFoundError(f"No checkpoint found in {WORK_DIR}")

    print(f"Loading checkpoint: {ckpt_path}")
    sd = load_checkpoint(ckpt_path, map_location='cpu')
    bundle.load_state_dict_selective(sd)
    del sd
    bundle.eval()
    bundle = bundle.to(device)

    pipeline = HyMotionM2MPipeline(
        bundle=bundle,
        num_steps=50,
        replacement_guidance='skip_last',  # MAN imputation for known regions
    )
    return bundle, pipeline, ckpt_path


def compute_mpjpe(pred_positions: np.ndarray, gt_positions: np.ndarray,
                  mask: Optional[np.ndarray] = None) -> float:
    """Compute Mean Per-Joint Position Error (mm).

    Args:
        pred_positions: (T, 22, 3) predicted joint positions in meters.
        gt_positions: (T, 22, 3) ground truth positions.
        mask: (T, D) optional mask. If provided, only compute on generated dims.

    Returns:
        MPJPE in millimeters.
    """
    diff = pred_positions - gt_positions  # (T, 22, 3)
    per_joint_error = np.linalg.norm(diff, axis=-1)  # (T, 22)
    return float(per_joint_error.mean() * 1000)  # Convert m to mm


def compute_mpjre(pred_rot6d: np.ndarray, gt_rot6d: np.ndarray) -> float:
    """Compute Mean Per-Joint Rotation Error (degrees).

    Uses geodesic distance between rotation matrices.

    Args:
        pred_rot6d: (T, 22, 6) predicted rotations in 6D representation.
        gt_rot6d: (T, 22, 6) ground truth rotations.

    Returns:
        MPJRE in degrees.
    """
    from scipy.spatial.transform import Rotation

    T, J = pred_rot6d.shape[0], pred_rot6d.shape[1]
    errors = []

    for t in range(T):
        for j in range(J):
            # Convert 6D to rotation matrix
            pred_r6 = pred_rot6d[t, j]
            gt_r6 = gt_rot6d[t, j]

            # 6D → 3x3: first two columns, normalize, cross product for third
            pred_mat = _rot6d_to_matrix(pred_r6)
            gt_mat = _rot6d_to_matrix(gt_r6)

            # Geodesic distance
            R_diff = pred_mat @ gt_mat.T
            # Clamp trace for numerical stability
            trace = np.clip(np.trace(R_diff), -1.0, 3.0)
            angle = np.arccos(np.clip((trace - 1) / 2, -1.0, 1.0))
            errors.append(angle)

    return float(np.degrees(np.mean(errors)))


def _rot6d_to_matrix(r6d: np.ndarray) -> np.ndarray:
    """Convert 6D rotation to 3x3 rotation matrix (row-major convention)."""
    a1 = r6d[:3]
    a2 = r6d[3:6]
    # Normalize first column
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    # Second column: orthogonalize and normalize
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / (np.linalg.norm(b2) + 1e-8)
    # Third column: cross product
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)  # (3, 3)


def evaluate_single_sample(
    bundle,
    pipeline,
    entry: Dict,
    motion_key: str,
    bone_offsets: np.ndarray,
    mode: str,
    device: str,
) -> Optional[Dict[str, Any]]:
    """Evaluate a single sample in the given mode.

    Args:
        bundle: loaded model bundle
        pipeline: inference pipeline
        entry: annotation entry dict
        motion_key: key in data_list
        bone_offsets: (22, 3) bone offsets
        mode: 'text_only', 'text_frame', 'text_upper'
        device: torch device string

    Returns:
        dict with metrics and optional output, or None on failure.
    """
    # Load motion
    motion_path = entry.get('smplx_path', entry.get('motion_path', ''))
    motion_135 = load_motion_npz(motion_path)
    if motion_135 is None:
        return None

    T_raw = motion_135.shape[0]
    T = min(T_raw, T_PAD)
    motion_135 = motion_135[:T]

    # Convert to 198-dim
    motion_198 = motion_135_to_198(motion_135, bone_offsets)

    # Build mask based on mode
    if mode == 'text_only':
        mask = build_mask_text_only(T, MOTION_DIM)
    elif mode == 'text_frame':
        mask = build_mask_inbetweening(T, MOTION_DIM, obs_ratio=0.2)
    elif mode == 'text_upper':
        mask = build_mask_upper_body_edit(T, MOTION_DIM)
    elif mode == 'text_lower':
        mask = build_mask_lower_body_edit(T, MOTION_DIM)
    elif mode == 'frame_head':
        mask = build_mask_head_completion(T, MOTION_DIM, obs_frames=1)
    elif mode == 'frame_tail':
        mask = build_mask_tail_completion(T, MOTION_DIM, obs_frames=1)
    elif mode == 'keyframe_periodic':
        mask = build_mask_periodic_keyframes(T, MOTION_DIM, period=10)
    elif mode == 'trans_only':
        mask = build_mask_trans_only(T, MOTION_DIM)
    elif mode == 'keyframe_rot':
        mask = build_mask_keyframe_rot_only(T, MOTION_DIM, period=10)
    elif mode == 'keyframe_pos':
        mask = build_mask_keyframe_pos_only(T, MOTION_DIM, period=10)
    elif mode == 'style_edit':
        mask = build_mask_style_edit(T, MOTION_DIM)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Load text embedding
    text_embedding = load_text_embedding(entry)

    # Normalize motion
    motion_t = torch.from_numpy(motion_198).float().unsqueeze(0).to(device)
    motion_norm = bundle.normalize_motion(motion_t)  # (1, T, 198)

    # Build src_mask tensor
    mask_t = torch.from_numpy(mask).float().unsqueeze(0).to(device)  # (1, T, 198)

    # src_motion = normalized motion * (1 - mask) → known regions preserved
    src_motion_norm = motion_norm * (1 - mask_t)
    clean_motion = motion_norm.clone()

    # Pad to T_PAD if needed
    if T < T_PAD:
        pad_len = T_PAD - T
        src_motion_norm = torch.nn.functional.pad(src_motion_norm, (0, 0, 0, pad_len))
        mask_t = torch.nn.functional.pad(mask_t, (0, 0, 0, pad_len))
        clean_motion = torch.nn.functional.pad(clean_motion, (0, 0, 0, pad_len))

    # Build inference batch
    batch = {
        'src_motion': src_motion_norm,
        'src_mask': mask_t,
        'src_length': [T],
        'tgt_length': [T],
        'clean_motion': clean_motion,
    }

    # Add text conditioning if available
    if text_embedding is not None:
        text_vec = text_embedding['text_vec_raw'].to(device)
        text_ctxt = text_embedding['text_ctxt_raw'].to(device)
        text_ctxt_len = text_embedding['text_ctxt_raw_length'].to(device)
        # Reshape if needed (remove extra batch dim from storage)
        if text_vec.dim() == 3:
            text_vec = text_vec.squeeze(0)  # (1, 768)
        if text_ctxt.dim() == 3 and text_ctxt.shape[0] == 1:
            pass  # Already (1, seq, 4096)
        elif text_ctxt.dim() == 2:
            text_ctxt = text_ctxt.unsqueeze(0)
        if text_ctxt_len.dim() == 0:
            text_ctxt_len = text_ctxt_len.unsqueeze(0)

        batch['text_vec_raw'] = text_vec
        batch['text_ctxt_raw'] = text_ctxt
        batch['text_ctxt_raw_length'] = text_ctxt_len

    # Run inference
    pipeline.replacement_guidance = 'skip_last'
    pipeline.text_guidance_scale = 1.0

    with torch.no_grad():
        output = pipeline(batch)

    sampled_norm = output['latent']  # (1, T_PAD, 198) normalized

    # Denormalize
    output_denorm = bundle.denormalize_motion(sampled_norm)[0, :T].cpu().numpy()  # (T, 198)

    # Extract 135-dim for FK
    out_135 = output_denorm[:, :135]
    gt_135 = motion_135

    # Compute FK positions for MPJPE
    from hftrainer.evaluation.motion.m2m_eval_metrics import motion135_to_positions_np
    pred_positions = motion135_to_positions_np(out_135, bone_offsets)  # (T, 22, 3)
    gt_positions = motion135_to_positions_np(gt_135, bone_offsets)  # (T, 22, 3)

    # Compute MPJPE (full body)
    mpjpe = compute_mpjpe(pred_positions, gt_positions)

    # Compute MPJPE on target region only (for partial masks)
    if mode == 'text_upper':
        target_joints = UPPER_BODY_JOINTS
        mpjpe_target = compute_mpjpe(
            pred_positions[:, target_joints],
            gt_positions[:, target_joints])
        mpjpe_preserved = compute_mpjpe(
            pred_positions[:, LOWER_BODY_JOINTS],
            gt_positions[:, LOWER_BODY_JOINTS])
    elif mode == 'text_lower':
        target_joints = LOWER_BODY_JOINTS
        mpjpe_target = compute_mpjpe(
            pred_positions[:, target_joints],
            gt_positions[:, target_joints])
        mpjpe_preserved = compute_mpjpe(
            pred_positions[:, UPPER_BODY_JOINTS],
            gt_positions[:, UPPER_BODY_JOINTS])
    elif mode == 'text_frame':
        # Target = middle portion (not first/last 20%)
        n_obs = max(1, int(T * 0.2))
        mid_slice = slice(n_obs, T - n_obs)
        mpjpe_target = compute_mpjpe(
            pred_positions[mid_slice], gt_positions[mid_slice])
        mpjpe_preserved = compute_mpjpe(
            np.concatenate([pred_positions[:n_obs], pred_positions[-n_obs:]], axis=0),
            np.concatenate([gt_positions[:n_obs], gt_positions[-n_obs:]], axis=0))
    elif mode == 'frame_head':
        # Target = all frames except first
        mpjpe_target = compute_mpjpe(
            pred_positions[1:], gt_positions[1:])
        mpjpe_preserved = compute_mpjpe(
            pred_positions[:1], gt_positions[:1])
    elif mode == 'frame_tail':
        # Target = all frames except last
        mpjpe_target = compute_mpjpe(
            pred_positions[:-1], gt_positions[:-1])
        mpjpe_preserved = compute_mpjpe(
            pred_positions[-1:], gt_positions[-1:])
    elif mode == 'keyframe_periodic':
        # Target = non-keyframe frames; preserved = keyframe frames
        period = 10
        keyframe_idx = list(range(0, T, period))
        target_idx = [t for t in range(T) if t not in keyframe_idx]
        if target_idx:
            mpjpe_target = compute_mpjpe(
                pred_positions[target_idx], gt_positions[target_idx])
        else:
            mpjpe_target = 0.0
        mpjpe_preserved = compute_mpjpe(
            pred_positions[keyframe_idx], gt_positions[keyframe_idx])
    elif mode == 'trans_only':
        # Only translation is preserved; all joints are generated
        mpjpe_target = mpjpe
        # Preserved = translation only (check root position)
        mpjpe_preserved = float(
            np.linalg.norm(out_135[:, :3] - gt_135[:, :3], axis=-1).mean() * 1000)
    elif mode == 'keyframe_rot':
        # Target = non-keyframe frames; preserved = rotation at keyframe frames
        period = 10
        keyframe_idx = list(range(0, T, period))
        target_idx = [t for t in range(T) if t not in keyframe_idx]
        if target_idx:
            mpjpe_target = compute_mpjpe(
                pred_positions[target_idx], gt_positions[target_idx])
        else:
            mpjpe_target = 0.0
        # Preserved: check rotation error at keyframes only
        kf_pred_rot = out_135[keyframe_idx, 3:135].reshape(-1, 22, 6)
        kf_gt_rot = gt_135[keyframe_idx, 3:135].reshape(-1, 22, 6)
        mpjpe_preserved = compute_mpjpe(
            pred_positions[keyframe_idx], gt_positions[keyframe_idx])
    elif mode == 'keyframe_pos':
        # Target = non-keyframe frames; preserved = FK position at keyframe frames
        period = 10
        keyframe_idx = list(range(0, T, period))
        target_idx = [t for t in range(T) if t not in keyframe_idx]
        if target_idx:
            mpjpe_target = compute_mpjpe(
                pred_positions[target_idx], gt_positions[target_idx])
        else:
            mpjpe_target = 0.0
        # Preserved: check position error at keyframes
        mpjpe_preserved = compute_mpjpe(
            pred_positions[keyframe_idx], gt_positions[keyframe_idx])
    elif mode == 'style_edit':
        # Trajectory (trans + root rot) preserved; body articulation generated
        mpjpe_target = mpjpe  # Full body (since most joints regenerated)
        # Preserved: root translation + root rotation
        trans_err = float(
            np.linalg.norm(out_135[:, :3] - gt_135[:, :3], axis=-1).mean() * 1000)
        # Root rotation error (first joint, dims 3:9)
        root_rot_pred = out_135[:, 3:9].reshape(-1, 1, 6)
        root_rot_gt = gt_135[:, 3:9].reshape(-1, 1, 6)
        mpjpe_preserved = trans_err  # Report translation preservation
    else:
        mpjpe_target = mpjpe
        mpjpe_preserved = 0.0

    # Compute MPJRE
    pred_rot6d = out_135[:, 3:135].reshape(T, 22, 6)
    gt_rot6d = gt_135[:, 3:135].reshape(T, 22, 6)
    mpjre = compute_mpjre(pred_rot6d, gt_rot6d)

    # Translation error
    trans_error = float(np.linalg.norm(out_135[:, :3] - gt_135[:, :3], axis=-1).mean() * 1000)

    return {
        'motion_key': motion_key,
        'motion_path': motion_path,
        'T': T,
        'mpjpe': mpjpe,
        'mpjpe_target': mpjpe_target,
        'mpjpe_preserved': mpjpe_preserved,
        'mpjre': mpjre,
        'trans_error_mm': trans_error,
        'has_text': text_embedding is not None,
        'pred_135': out_135,
        'gt_135': gt_135,
        'pred_positions': pred_positions,
        'gt_positions': gt_positions,
    }


def run_evaluation(
    mode: str,
    epoch: Optional[int] = None,
    save_npz: bool = False,
    max_samples: int = 100,
    device: str = 'cuda:0',
):
    """Run the full overfitting evaluation.

    Args:
        mode: 'text_only', 'text_frame', 'text_upper', or 'all'
        epoch: specific checkpoint epoch (None = latest)
        save_npz: whether to save NPZ files for visualization
        max_samples: max samples to evaluate
        device: torch device
    """
    print(f"\n{'='*70}")
    print(f"  Overfitting Verification — Mode: {mode}")
    print(f"{'='*70}\n")

    # Load model
    print("Loading model...")
    bundle, pipeline, ckpt_path = load_model(device, epoch=epoch)
    print(f"  Checkpoint: {ckpt_path}")

    # Load bone offsets
    bone_offsets = load_bone_offsets()
    print(f"  Bone offsets: {bone_offsets.shape}")

    # Load annotation
    anno = load_annotation()
    data_list = anno['data_list']
    print(f"  Annotation: {len(data_list)} samples")

    # Determine modes to evaluate
    ALL_MODES = ['text_only', 'text_frame', 'text_upper', 'text_lower',
                 'frame_head', 'frame_tail', 'keyframe_periodic', 'trans_only',
                 'keyframe_rot', 'keyframe_pos', 'style_edit']
    modes = ALL_MODES if mode == 'all' else [mode]

    for eval_mode in modes:
        print(f"\n--- Evaluating mode: {eval_mode} ---")

        results = []
        n_success = 0
        n_fail = 0
        n_no_text = 0

        for i, (key, entry) in enumerate(data_list.items()):
            if i >= max_samples:
                break

            result = evaluate_single_sample(
                bundle, pipeline, entry, key,
                bone_offsets, eval_mode, device)

            if result is None:
                n_fail += 1
                continue

            if not result['has_text']:
                n_no_text += 1

            results.append(result)
            n_success += 1

            # Progress reporting
            if (i + 1) % 10 == 0:
                avg_mpjpe = np.mean([r['mpjpe'] for r in results])
                avg_mpjre = np.mean([r['mpjre'] for r in results])
                print(f"  [{i+1}/{min(len(data_list), max_samples)}] "
                      f"MPJPE={avg_mpjpe:.1f}mm  MPJRE={avg_mpjre:.2f}°")

        # Print summary
        if results:
            mpjpes = [r['mpjpe'] for r in results]
            mpjres = [r['mpjre'] for r in results]
            mpjpe_targets = [r['mpjpe_target'] for r in results]
            mpjpe_preserveds = [r['mpjpe_preserved'] for r in results]
            trans_errors = [r['trans_error_mm'] for r in results]

            print(f"\n{'='*50}")
            print(f"  Mode: {eval_mode}")
            print(f"  Samples: {n_success} success / {n_fail} failed / {n_no_text} without text")
            print(f"  Checkpoint: {ckpt_path}")
            print(f"{'='*50}")
            print(f"  MPJPE (full body):   {np.mean(mpjpes):7.1f} ± {np.std(mpjpes):5.1f} mm")
            print(f"  MPJPE (target):      {np.mean(mpjpe_targets):7.1f} ± {np.std(mpjpe_targets):5.1f} mm")
            print(f"  MPJPE (preserved):   {np.mean(mpjpe_preserveds):7.1f} ± {np.std(mpjpe_preserveds):5.1f} mm")
            print(f"  MPJRE:               {np.mean(mpjres):7.2f} ± {np.std(mpjres):5.2f} °")
            print(f"  Translation Error:   {np.mean(trans_errors):7.1f} ± {np.std(trans_errors):5.1f} mm")
            print(f"{'='*50}")

            # Expected thresholds for a well-overfit model:
            mean_mpjpe = np.mean(mpjpes)
            if mean_mpjpe < 5.0:
                verdict = "EXCELLENT — near-perfect reproduction"
            elif mean_mpjpe < 20.0:
                verdict = "GOOD — strong overfitting"
            elif mean_mpjpe < 50.0:
                verdict = "FAIR — partial overfitting, may need more training"
            else:
                verdict = "POOR — model has NOT overfit, check implementation"
            print(f"  Verdict: {verdict}")
            print()

            # Save NPZ if requested
            if save_npz:
                out_dir = os.path.join(WORK_DIR, 'eval_overfit', eval_mode)
                os.makedirs(out_dir, exist_ok=True)
                for r in results:
                    npz_path = os.path.join(out_dir, f"{r['motion_key']}.npz")
                    np.savez_compressed(
                        npz_path,
                        pred_135=r['pred_135'],
                        gt_135=r['gt_135'],
                        pred_positions=r['pred_positions'],
                        gt_positions=r['gt_positions'],
                        mpjpe=r['mpjpe'],
                        mpjre=r['mpjre'],
                    )
                print(f"  Saved {len(results)} NPZ files to {out_dir}")

                # Also save summary JSON
                summary = {
                    'mode': eval_mode,
                    'checkpoint': ckpt_path,
                    'n_samples': n_success,
                    'mean_mpjpe_mm': float(np.mean(mpjpes)),
                    'std_mpjpe_mm': float(np.std(mpjpes)),
                    'mean_mpjre_deg': float(np.mean(mpjres)),
                    'std_mpjre_deg': float(np.std(mpjres)),
                    'mean_trans_error_mm': float(np.mean(trans_errors)),
                    'per_sample': [
                        {
                            'key': r['motion_key'],
                            'mpjpe': r['mpjpe'],
                            'mpjre': r['mpjre'],
                            'mpjpe_target': r['mpjpe_target'],
                            'mpjpe_preserved': r['mpjpe_preserved'],
                            'trans_error_mm': r['trans_error_mm'],
                            'T': r['T'],
                            'has_text': r['has_text'],
                        }
                        for r in results
                    ],
                }
                summary_path = os.path.join(out_dir, 'summary.json')
                with open(summary_path, 'w') as f:
                    json.dump(summary, f, indent=2)
                print(f"  Summary saved to {summary_path}")
        else:
            print(f"\n  No successful evaluations for mode={eval_mode}")


def main():
    parser = argparse.ArgumentParser(description='Overfitting verification for M2M v2')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['text_only', 'text_frame', 'text_upper', 'text_lower',
                                 'frame_head', 'frame_tail', 'keyframe_periodic',
                                 'trans_only', 'keyframe_rot', 'keyframe_pos',
                                 'style_edit', 'all'],
                        help='Evaluation mode')
    parser.add_argument('--epoch', type=int, default=None,
                        help='Checkpoint epoch (default: latest)')
    parser.add_argument('--save-npz', action='store_true',
                        help='Save NPZ output for visualization')
    parser.add_argument('--max-samples', type=int, default=100,
                        help='Max samples to evaluate')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device for inference')
    args = parser.parse_args()

    run_evaluation(
        mode=args.mode,
        epoch=args.epoch,
        save_npz=args.save_npz,
        max_samples=args.max_samples,
        device=args.device,
    )


if __name__ == '__main__':
    main()
