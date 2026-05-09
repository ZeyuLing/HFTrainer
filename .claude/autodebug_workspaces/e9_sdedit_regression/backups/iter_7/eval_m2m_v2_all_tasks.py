#!/usr/bin/env python3
"""HyMotion M2M v2 comprehensive evaluation across 16 tasks (E1-E16).

Evaluates 4 model variants:
  - uncond_local:   No text, local rotation
  - uncond_global:  No text, global rotation
  - caption_local:  Text-conditioned, local rotation
  - caption_global: Text-conditioned, global rotation

Usage:
    # Run Phase 1 tasks (E2, E3, E4, E5) on all models
    python tools/eval_m2m_v2_all_tasks.py --tasks E2 E3 E5 --max-samples 50

    # Run specific task with specific setting
    python tools/eval_m2m_v2_all_tasks.py --tasks E2 --settings A B --models uncond_local

    # Run all tasks (full evaluation)
    python tools/eval_m2m_v2_all_tasks.py --all-tasks --max-samples 100

    # With replacement guidance for MAN imputation
    python tools/eval_m2m_v2_all_tasks.py --tasks E2 --replacement-guidance skip_last

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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ============================================================================
# Caption embedding cache — pre-extracted by
#   scripts/extract_eval_caption_embeddings.py
# Loading avoids bundle.encode_text() (which fails for caption_* bundles that
# were trained with LoadPreExtractedTextEmbedding and therefore ship without a
# runtime text_encoder config).
# ============================================================================

CAPTION_EMBED_CACHE_PATH = Path(__file__).resolve().parent.parent / \
    'data' / 'eval' / 'm2m_v2' / 'caption_embeddings' / 'cache.pt'

_CAPTION_EMBED_CACHE: Optional[Dict[str, Dict[str, torch.Tensor]]] = None


def _load_caption_embed_cache() -> Dict[str, Dict[str, torch.Tensor]]:
    global _CAPTION_EMBED_CACHE
    if _CAPTION_EMBED_CACHE is not None:
        return _CAPTION_EMBED_CACHE
    if not CAPTION_EMBED_CACHE_PATH.is_file():
        print(f'  WARNING: caption embedding cache not found at '
              f'{CAPTION_EMBED_CACHE_PATH}. Caption models will run '
              f'unconditioned. Run scripts/extract_eval_caption_embeddings.py '
              f'first.')
        _CAPTION_EMBED_CACHE = {}
        return _CAPTION_EMBED_CACHE
    data = torch.load(str(CAPTION_EMBED_CACHE_PATH), map_location='cpu',
                      weights_only=False)
    cache = data.get('cache', {}) if isinstance(data, dict) else {}
    if not isinstance(cache, dict):
        cache = {}
    print(f'  Loaded {len(cache)} caption embeddings from '
          f'{CAPTION_EMBED_CACHE_PATH}')
    _CAPTION_EMBED_CACHE = cache
    return _CAPTION_EMBED_CACHE


def _lookup_caption_embedding(caption: str
                              ) -> Optional[Dict[str, torch.Tensor]]:
    """Return {'text_vec_raw', 'text_ctxt_raw', 'text_ctxt_raw_length'} or None."""
    cache = _load_caption_embed_cache()
    key = caption.strip()
    if not key or key not in cache:
        return None
    entry = cache[key]
    return {
        'text_vec_raw': entry['text_vec_raw'],
        'text_ctxt_raw': entry['text_ctxt_raw'],
        'text_ctxt_raw_length': entry['text_ctxt_raw_length'],
    }


# ============================================================================
# Model registry — v2 (198-dim, two-tier condition sampler)
# ============================================================================

V2_MODELS = {
    'uncond_local': {
        'config': 'configs/hymotion_m2m_v2/hymotion_m2m_v2_uncond_local_046b.py',
        'work_dir': 'work_dirs/hymotion_m2m_v2_uncond_local_046b',
        'desc': 'v2 Unconditioned + Local rotation',
        'has_caption': False,
        'rotation_space': 'local',
    },
    'uncond_global': {
        'config': 'configs/hymotion_m2m_v2/hymotion_m2m_v2_uncond_global_046b.py',
        'work_dir': 'work_dirs/hymotion_m2m_v2_uncond_global_046b',
        'desc': 'v2 Unconditioned + Global rotation',
        'has_caption': False,
        'rotation_space': 'global',
    },
    'caption_local': {
        'config': 'configs/hymotion_m2m_v2/hymotion_m2m_v2_caption_local_046b.py',
        'work_dir': 'work_dirs/hymotion_m2m_v2_caption_local_046b',
        'desc': 'v2 Caption + Local rotation (mixed training)',
        'has_caption': True,
        'rotation_space': 'local',
    },
    'caption_global': {
        'config': 'configs/hymotion_m2m_v2/hymotion_m2m_v2_caption_global_046b.py',
        'work_dir': 'work_dirs/hymotion_m2m_v2_caption_global_046b',
        'desc': 'v2 Caption + Global rotation (mixed training)',
        'has_caption': True,
        'rotation_space': 'global',
    },
    # Phase 1 variants: pure T2M curriculum (no completion tasks)
    'caption_local_phase1': {
        'config': 'configs/hymotion_m2m_v2/hymotion_m2m_v2_caption_local_phase1.py',
        'work_dir': 'work_dirs/hymotion_m2m_v2_caption_local_phase1',
        'desc': 'v2 Caption + Local rotation (Phase 1: pure T2M)',
        'has_caption': True,
        'rotation_space': 'local',
    },
    'caption_global_phase1': {
        'config': 'configs/hymotion_m2m_v2/hymotion_m2m_v2_caption_global_phase1.py',
        'work_dir': 'work_dirs/hymotion_m2m_v2_caption_global_phase1',
        'desc': 'v2 Caption + Global rotation (Phase 1: pure T2M)',
        'has_caption': True,
        'rotation_space': 'global',
    },
    # Phase 2 variants: T2M base + completion curriculum (longer training)
    'caption_local_phase2': {
        'config': 'configs/hymotion_m2m_v2/hymotion_m2m_v2_caption_local_phase2.py',
        'work_dir': 'work_dirs/hymotion_m2m_v2_caption_local_phase2',
        'desc': 'v2 Caption + Local rotation (Phase 2: T2M + completion)',
        'has_caption': True,
        'rotation_space': 'local',
    },
    'caption_global_phase2': {
        'config': 'configs/hymotion_m2m_v2/hymotion_m2m_v2_caption_global_phase2.py',
        'work_dir': 'work_dirs/hymotion_m2m_v2_caption_global_phase2',
        'desc': 'v2 Caption + Global rotation (Phase 2: T2M + completion)',
        'has_caption': True,
        'rotation_space': 'global',
    },
}

# Also allow running v1 models for comparison
V1_MODELS = {
    'v1_uncond_fm_man': {
        'config': 'configs/hymotion_m2m/hymotion_m2m_completion_uncond_fm_man_046b.py',
        'work_dir': 'work_dirs/hymotion_m2m_completion_uncond_fm_man_046b',
        'desc': 'v1 Unconditioned FM+MAN',
        'has_caption': False,
        'rotation_space': 'local',
        'motion_dim': 135,
    },
    'v1_uncond_fm_man_globalrot': {
        'config': 'configs/hymotion_m2m/hymotion_m2m_completion_uncond_fm_man_globalrot_046b.py',
        'work_dir': 'work_dirs/hymotion_m2m_completion_uncond_fm_man_globalrot_046b',
        'desc': 'v1 Unconditioned FM+MAN+GlobalRot',
        'has_caption': False,
        'rotation_space': 'global',
        'motion_dim': 135,
    },
}

ALL_MODELS = {**V2_MODELS, **V1_MODELS}

# 198-dim layout
MOTION_DIM_V2 = 198
TRANSL_DIM = 3
ROT6D_DIM = 132  # 22 joints * 6
POSITION_DIM = 63  # 21 joints * 3 (pelvis excluded)

# Eval data directory
EVAL_DATA_DIR = 'data/eval/m2m_v2'
EVAL_DATA_DIR_LEGACY = 'data/eval/hymotion_m2m'
MOTION_DATA_DIR = 'data/motionhub'

# E9 Motion Repair: precomputed MoGenDIT adaptive masks live here, produced by
# scripts/compute_adaptive_masks_for_eval.py.  Indexed by motion_path relative
# to data/hymotion_data/ (i.e. the "data/hymotion_data/" prefix is stripped).
E9_ADAPTIVE_MASK_DIR = 'data/eval/hymotion_m2m/adaptive_masks_mogendit'

# SMPL-22 parent array (pelvis=root, -1). Used for kinematic spatial dilation
# of adaptive masks: if joint j is marked defective at frame f, also mark its
# parent and children to cover IK propagation that may affect neighbors.
SMPL22_PARENTS = [
    -1,  # 0  pelvis
    0,   # 1  l_hip
    0,   # 2  r_hip
    0,   # 3  spine1
    1,   # 4  l_knee
    2,   # 5  r_knee
    3,   # 6  spine2
    4,   # 7  l_ankle
    5,   # 8  r_ankle
    6,   # 9  spine3
    7,   # 10 l_foot
    8,   # 11 r_foot
    9,   # 12 neck
    9,   # 13 l_collar
    9,   # 14 r_collar
    12,  # 15 head
    13,  # 16 l_shoulder
    14,  # 17 r_shoulder
    16,  # 18 l_elbow
    17,  # 19 r_elbow
    18,  # 20 l_wrist
    19,  # 21 r_wrist
]


def _load_adaptive_mask_for_motion(
    motion_path: str,
    T: int,
    D: int = 198,
    no_trans_mask: bool = True,
    temporal_dilate: int = 3,
) -> Optional[np.ndarray]:
    """Load MoGenDIT adaptive mask for a motion and expand it to the model's
    native motion dimension.

    Reads ``{E9_ADAPTIVE_MASK_DIR}/{motion_path without data/hymotion_data/ prefix}``
    which stores ``joint_mask: (T_orig, 22) bool`` and
    ``trans_mask: (T_orig,) bool``.

    Expands the (T, 22+1) sparse mask to HyMotion M2M v2 198-dim layout:
      channel 0..2     : translation XYZ                (driven by trans_mask)
      channel 3..134   : 22 joints × rot6d              (driven by joint_mask[:, j])
      channel 135..197 : 21 joints × position (j=1..21) (driven by joint_mask[:, j])
                         (pelvis is excluded from the position block)

    Supports D=135 (legacy, rot-only) for backward compat with v1 models.

    **Training-distribution alignment** (2026-04-21): The model was trained on
    M7 scattered_joint strategy which has two features the raw MoGenDIT mask
    lacks:
      1. M7 NEVER masks translation (col 0).  Training data has trans always
         observed in the scattered-repair pattern.
      2. M7 applies per-spot temporal dilation of 1-8 frames each side.
         Training mask has dilated "blobs" around each flag point, not
         single-frame point flags.
    Without these, the raw MoGenDIT mask is OOD for the model: per-frame
    isolated flags with translation churn — producing 4-5× higher jitter
    than C_full (all-1 mask, which IS a trained pattern, M5).

    Args:
        no_trans_mask: if True, zero out translation channels in the final
            mask (match M7 training).  Default True.
        temporal_dilate: radius (frames each side) of temporal dilation on
            joint_mask / trans_mask.  Default 3 (M7 uses 1-8, median ~4).
            Set 0 to disable.

    Returns None if no adaptive mask is cached for this motion.
    """
    if not motion_path:
        return None
    if D not in (135, 198):
        raise ValueError(f'Unsupported motion dim for adaptive mask: {D}')

    project_root = Path(__file__).resolve().parent.parent
    mp_cache_rel = motion_path
    prefix = 'data/hymotion_data/'
    if mp_cache_rel.startswith(prefix):
        mp_cache_rel = mp_cache_rel[len(prefix):]
    mask_path = project_root / E9_ADAPTIVE_MASK_DIR / mp_cache_rel
    if not mask_path.is_file():
        return None

    data = np.load(str(mask_path), allow_pickle=True)
    joint_mask = data['joint_mask'].astype(np.float32)  # (T_orig, 22)
    if 'trans_mask' in data.files:
        trans_mask = data['trans_mask'].astype(np.float32)  # (T_orig,)
    else:
        trans_mask = np.zeros(joint_mask.shape[0], dtype=np.float32)

    # ------------------------------------------------------------------
    # Training-distribution alignment (2026-04-21)
    # ------------------------------------------------------------------
    # 1. Match M7 convention: translation is never masked during training.
    if no_trans_mask:
        trans_mask = np.zeros_like(trans_mask)

    # 2. Temporal dilation (1-D max-pool along time axis).
    if temporal_dilate > 0:
        T_orig = joint_mask.shape[0]
        k = temporal_dilate
        # joint_mask: (T_orig, 22) -> per-joint 1-D dilation
        jm_dilated = np.zeros_like(joint_mask)
        for j in range(joint_mask.shape[1]):
            col = joint_mask[:, j].astype(bool)
            # simple max-pool via rolling OR
            out = col.copy()
            for s in range(1, k + 1):
                out[s:] |= col[:-s]
                out[:-s] |= col[s:]
            jm_dilated[:, j] = out.astype(np.float32)
        joint_mask = jm_dilated
        # trans_mask would also be dilated if not zeroed, but we already
        # zero it above in the default path.

    T_orig = joint_mask.shape[0]
    T_mask = min(T, T_orig)

    mask = np.zeros((T, D), dtype=np.float32)
    # Translation channels (0..2)
    for d in range(3):
        mask[:T_mask, d] = trans_mask[:T_mask]
    # Rotation channels (3..134): 22 joints × 6 rot6d dims
    for j in range(22):
        start = 3 + j * 6
        jc = joint_mask[:T_mask, j]
        for d in range(6):
            mask[:T_mask, start + d] = jc

    # Position channels (135..197): 21 joints (j=1..21, pelvis excluded)
    if D == 198:
        for j in range(1, 22):
            start = 135 + (j - 1) * 3
            jc = joint_mask[:T_mask, j]
            for d in range(3):
                mask[:T_mask, start + d] = jc

    return mask


# ============================================================================
# Data loading
# ============================================================================

def load_motion_135d(npz_path: str) -> Optional[np.ndarray]:
    """Load npz -> 135-dim motion (abs transl + rot6d)."""
    try:
        from hftrainer.datasets.motion.motionhub.transforms.load_smplx import (
            process_transl, process_smplx_pose,
        )
        data = np.load(npz_path, allow_pickle=True)
        trans_key = 'trans' if 'trans' in data else 'transl'
        abs_trans = data[trans_key].astype(np.float32)
        poses_key = 'poses' if 'poses' in data else 'body_pose'
        poses = data[poses_key].astype(np.float32)
        transl = process_transl(abs_trans, 'abs')
        pose = process_smplx_pose(poses, 'rotation_6d', 'smpl_22')
        motion = np.concatenate([transl, pose], axis=-1)
        return motion.astype(np.float32)
    except Exception as e:
        return None


def motion_135_to_198(
    motion_135: np.ndarray,
    bone_offsets: np.ndarray,
) -> np.ndarray:
    """Convert 135-dim to 198-dim by appending FK position channels.

    198-dim = 135 (trans + rot6d) + 63 (21 joints * 3D position).
    Position is: XZ relative to pelvis, Y absolute. Pelvis excluded.

    Args:
        motion_135: (T, 135) motion.
        bone_offsets: (22, 3) bone offsets.

    Returns:
        (T, 198) motion.
    """
    from hftrainer.evaluation.motion.m2m_eval_metrics import motion135_to_positions_np

    positions = motion135_to_positions_np(motion_135, bone_offsets)  # (T, 22, 3)

    # Exclude pelvis (joint 0), keep joints 1-21
    joint_pos = positions[:, 1:, :]  # (T, 21, 3)

    # XZ relative to pelvis
    pelvis_xz = positions[:, 0:1, [0, 2]]  # (T, 1, 2)
    joint_pos_rel = joint_pos.copy()
    joint_pos_rel[:, :, 0] -= pelvis_xz[:, :, 0]  # X relative
    joint_pos_rel[:, :, 2] -= pelvis_xz[:, :, 1]  # Z relative
    # Y stays absolute

    pos_flat = joint_pos_rel.reshape(-1, 63)  # (T, 63)
    return np.concatenate([motion_135, pos_flat], axis=-1).astype(np.float32)


def load_eval_samples(
    anno_file: str,
    motion_data_dir: str,
    max_samples: int,
    min_frames: int = 30,
    max_frames: int = 360,
    require_caption: bool = False,
    bone_offsets: Optional[np.ndarray] = None,
    convert_to_198: bool = False,
    task_id: Optional[str] = None,
) -> List[Dict]:
    """Load evaluation samples from annotation JSON.

    Args:
        anno_file: path to eval JSON file.
        motion_data_dir: base directory for motion NPZ files.
        max_samples: maximum number of samples.
        min_frames: minimum frame count.
        max_frames: crop to this length. For E9, this is overridden — we keep
            the full motion length so sliding-window repair can process the
            whole sequence instead of truncating long motions.
        require_caption: only include samples with caption.
        bone_offsets: if converting to 198-dim.
        convert_to_198: whether to compute 198-dim representation.
        task_id: optional task id. Used to bypass max_frames cropping for E9
            (long-motion repair uses sliding windows over the full sequence).

    Returns:
        List of sample dicts with keys: motion, motion_198, T, caption, path, fps.
    """
    with open(anno_file) as f:
        anno = json.load(f)

    data_list = anno.get('data_list', {})
    if isinstance(data_list, dict):
        items = list(data_list.values())
    else:
        items = data_list

    samples = []
    for item in items:
        if len(samples) >= max_samples:
            break

        caption = item.get('caption', item.get('caption_en', item.get('text_caption', ''))) or ''
        if require_caption and not caption:
            continue

        motion_path = item.get('motion_path', item.get('smplx_path', ''))

        # E14 transition: no motion_path, uses motion_a_path/motion_b_path instead
        # E9 repair: may have relative paths based on project root
        if not motion_path and ('motion_a_path' in item or 'target_motion_path' in item):
            # Transition/target tasks: create a dummy motion from first source
            src = item.get('motion_a_path', item.get('motion_path', ''))
            if not src:
                src = item.get('target_motion_path', '')
            if src:
                motion_path = src

        # E1 (T2M): pure generation, no input motion needed — create blank motion
        if not motion_path:
            num_frames = item.get('num_frames', 120)
            T = min(num_frames, max_frames)
            motion = np.zeros((T, 135), dtype=np.float32)
            sample = {
                'path': '',
                'motion': motion,
                'T': T,
                'caption': caption,
                'fps': item.get('fps', 30),
                'source': item.get('source', ''),
                'num_frames_orig': num_frames,
            }
            for extra_key in ('motion_a_path', 'motion_b_path', 'target_motion_path'):
                if extra_key in item:
                    sample[extra_key] = item[extra_key]
            if convert_to_198 and bone_offsets is not None:
                sample['motion_198'] = motion_135_to_198(motion, bone_offsets)
            samples.append(sample)
            continue

        # Support absolute paths (new datalists) and relative paths (legacy)
        if os.path.isabs(motion_path):
            full_path = motion_path
        else:
            # Try motion_data_dir first, then project root
            full_path = os.path.join(motion_data_dir, motion_path)
            if not os.path.exists(full_path):
                full_path = os.path.abspath(motion_path)
        if not os.path.exists(full_path):
            continue

        motion = load_motion_135d(full_path)
        if motion is None or motion.shape[0] < min_frames:
            continue

        # For E9 Motion Repair we keep the full motion length — inference
        # will run in 2 sliding windows over the complete sequence and
        # blend the overlap, matching MoGenDIT's long-motion protocol.
        # Other tasks continue to crop to max_frames (360) to fit within
        # the training context.
        if task_id == 'E9':
            T = motion.shape[0]
        else:
            T = min(motion.shape[0], max_frames)
        motion = motion[:T]

        sample = {
            'path': motion_path,
            'motion': motion,
            'T': T,
            'caption': caption,
            'fps': item.get('fps', 30),
            'source': item.get('source', ''),
            'num_frames_orig': item.get('num_frames', T),
        }

        # Preserve extra paths for transition/target tasks (E14/E15/E16)
        for extra_key in ('motion_a_path', 'motion_b_path', 'target_motion_path'):
            if extra_key in item:
                sample[extra_key] = item[extra_key]

        if convert_to_198 and bone_offsets is not None:
            sample['motion_198'] = motion_135_to_198(motion, bone_offsets)

        samples.append(sample)

    return samples


# ============================================================================
# Model loading
# ============================================================================

def load_model(model_name: str, device: str):
    """Load model bundle and create pipeline.

    Returns:
        bundle, pipeline, ckpt_path, model_info
    """
    from mmengine.config import Config
    from hftrainer.registry import MODEL_BUNDLES
    from hftrainer.utils.checkpoint_utils import load_checkpoint, find_latest_checkpoint
    from hftrainer.pipelines.motion.hymotion_m2m_pipeline import HyMotionM2MPipeline

    model_info = ALL_MODELS[model_name]
    cfg = Config.fromfile(model_info['config'])
    bundle = MODEL_BUNDLES.build(cfg.model.to_dict())

    ckpt_path = find_latest_checkpoint(model_info['work_dir'])
    if ckpt_path is None:
        print(f'  WARNING: No checkpoint found for {model_name} at {model_info["work_dir"]}')
        return None, None, None, model_info

    print(f'  Loading checkpoint: {ckpt_path}')
    sd = load_checkpoint(ckpt_path, map_location='cpu')
    bundle.load_state_dict_selective(sd)
    del sd
    bundle.eval()
    bundle = bundle.to(device)

    pipeline = HyMotionM2MPipeline(
        bundle=bundle,
        num_steps=50,
        replacement_guidance='none',  # Will be overridden per task
    )
    return bundle, pipeline, ckpt_path, model_info


# ============================================================================
# Per-sample evaluation
# ============================================================================

def evaluate_sample(
    bundle,
    pipeline,
    sample: Dict,
    task,
    setting_name: str,
    model_info: Dict,
    bone_offsets: np.ndarray,
    device: str,
    replacement_guidance: str = 'skip_last',
    text_guidance_scale: float = 1.0,
    num_steps: int = 50,
) -> Dict:
    """Evaluate a single sample on a single task + setting.

    Returns dict with all computed metrics.
    """
    from hftrainer.evaluation.motion.m2m_eval_metrics import compute_all_metrics
    from hftrainer.evaluation.motion.m2m_eval_tasks import (
        build_keyframe_nonuniform_mask,
        build_keyframe_adaptive_mask,
        detect_keyframes_from_motion,
        build_loop_completion_mask,
        build_transition_mask,
        build_transition_to_target_first_mask,
        build_transition_to_target_last_mask,
        compute_transition_length,
        extract_ee_constraints_from_gt,
        detect_foot_contact_frames,
    )

    motion_dim = model_info.get('motion_dim', MOTION_DIM_V2)
    rotation_space = model_info.get('rotation_space', 'local')
    constraint_info = None  # For E4 end-effector; set in mask building

    # Get motion in the right dimension
    motion_135 = sample['motion']  # (T, 135) always available
    T = sample['T']

    if motion_dim == 198:
        if 'motion_198' in sample:
            motion_raw = sample['motion_198']
        else:
            motion_raw = motion_135_to_198(motion_135, bone_offsets)
    else:
        motion_raw = motion_135

    # ---- Special handling for E8-B/C/D: loop completion ----
    setting_kwargs = task.settings[setting_name].mask_kwargs
    gt_motion_135 = motion_135  # save original GT for metrics

    if task.task_id == 'E8' and '_loop_append' in setting_kwargs:
        N_append = setting_kwargs['_loop_append']
        T_gt = T
        T_total = T_gt + N_append

        # Build extended motion: [GT(T_gt) | zeros(N_append-1) | first_frame(1)]
        first_frame = motion_135[0:1]  # (1, 135)
        pad_frames = np.zeros((N_append - 1, 135), dtype=np.float32)
        motion_135 = np.concatenate([motion_135, pad_frames, first_frame], axis=0)
        T = T_total

        if motion_dim == 198:
            first_frame_198 = motion_raw[0:1]
            pad_frames_198 = np.zeros((N_append - 1, 198), dtype=np.float32)
            motion_raw = np.concatenate([motion_raw, pad_frames_198, first_frame_198], axis=0)
        else:
            motion_raw = motion_135

        # Build loop completion mask
        mask = build_loop_completion_mask(T_total, 135, T_gt=T_gt, N_append=N_append)

    # ---- Special handling for E14: transition stitching ----
    elif task.task_id == 'E14' and '_use_transition_data' in setting_kwargs:
        N_cond = setting_kwargs.get('_cond_frames', 15)

        # Load motion_a and motion_b
        motion_a_path = sample.get('motion_a_path', '')
        motion_b_path = sample.get('motion_b_path', '')
        if not os.path.isabs(motion_a_path):
            motion_a_path = os.path.join(MOTION_DATA_DIR, motion_a_path)
        if not os.path.isabs(motion_b_path):
            motion_b_path = os.path.join(MOTION_DATA_DIR, motion_b_path)

        motion_a = load_motion_135d(motion_a_path)
        motion_b = load_motion_135d(motion_b_path)
        if motion_a is None or motion_b is None:
            return {}, None

        # ------------------------------------------------------------------
        # v2 transition redesign (2026-04-20):
        #   1. Place B in world coords AFTER A (forward_step + small yaw),
        #      instead of letting B stay at the origin — this avoids the
        #      "return to origin" artefact in the transition region.
        #   2. Canonicalize the (A_tail | pad | B_head) segment so that the
        #      anchor frame (start of A_tail) sits at the origin facing +Z,
        #      matching the canonical training distribution.
        #   3. After inference, decanonicalize the output back to world
        #      coordinates before computing metrics.
        # ------------------------------------------------------------------
        import torch as _torch
        from hftrainer.pipelines.motion.transition_utils import (
            canonicalize_segment, decanonicalize_segment,
            place_b_after_a,
        )

        motion_a_t = _torch.from_numpy(motion_a).float()
        motion_b_t = _torch.from_numpy(motion_b).float()

        # Step 1: place B in world coords, continuing A's forward direction
        forward_step = float(setting_kwargs.get('_forward_step', 1.0))
        yaw_offset_deg = float(setting_kwargs.get('_yaw_offset_deg', 0.0))
        motion_b_world = place_b_after_a(
            motion_a_t, motion_b_t,
            forward_step=forward_step,
            yaw_offset_deg=yaw_offset_deg,
        )
        motion_b_world_np = motion_b_world.numpy()

        # Take tail of A and head of (world-placed) B
        a_tail = motion_a[-N_cond:]               # (N_cond, 135) world coords
        b_head = motion_b_world_np[:N_cond]       # (N_cond, 135) world coords

        # Transition length based on world-coords gap (not "both at origin" trick)
        from hftrainer.evaluation.motion.m2m_eval_metrics import motion135_to_positions_np
        pos_a = motion135_to_positions_np(motion_a, bone_offsets)
        pos_b_world = motion135_to_positions_np(motion_b_world_np, bone_offsets)
        pos_a_end = pos_a[-1, 0]
        pos_b_start = pos_b_world[0, 0]
        N_transition = compute_transition_length(pos_a_end, pos_b_start)

        # Build world-space transition context: [A_tail | pad | B_head_world]
        transition_pad = np.zeros((N_transition, 135), dtype=np.float32)
        world_segment = np.concatenate([a_tail, transition_pad, b_head], axis=0)
        T = world_segment.shape[0]

        # Step 2: canonicalize so anchor (A_tail[0]) is at origin, heading +Z
        world_segment_t = _torch.from_numpy(world_segment).float()
        canon_segment_t, R_canon, offset_canon = canonicalize_segment(
            world_segment_t, anchor_frame=0)
        motion_135 = canon_segment_t.numpy()
        gt_motion_135 = motion_135  # no GT for transition region

        # Stash the canonicalization transform so inference output can be
        # mapped back to world coordinates before metrics / viz.
        _transition_canon_info = dict(
            R_canon=R_canon, offset_canon=offset_canon,
            motion_a_full=motion_a, motion_b_world_full=motion_b_world_np,
            N_cond=N_cond, N_transition=N_transition,
        )

        if motion_dim == 198:
            motion_raw = motion_135_to_198(motion_135, bone_offsets)
        else:
            motion_raw = motion_135

        mask = build_transition_mask(
            T, 135, N_cond_a=N_cond, N_transition=N_transition, N_cond_b=N_cond)

    # ---- Special handling for E15: transition to target first frame ----
    elif task.task_id == 'E15' and '_use_target_first' in setting_kwargs:
        N_cond_tail = setting_kwargs.get('_cond_tail_frames', 15)

        # Load target motion to get its first frame
        target_path = sample.get('target_motion_path', '')
        if not os.path.isabs(target_path):
            target_path = os.path.join(MOTION_DATA_DIR, target_path)
        target_motion = load_motion_135d(target_path)
        if target_motion is None:
            return {}, None

        # ------------------------------------------------------------------
        # v2 transition redesign (2026-04-20): target is a static pose.
        # We treat the target as a 1-frame "B motion" and place it forward
        # of motion_tail, then canonicalize the whole segment.
        # ------------------------------------------------------------------
        import torch as _torch
        from hftrainer.pipelines.motion.transition_utils import (
            canonicalize_segment, place_b_after_a,
        )

        # motion_135 here is the current sample's motion (source). Take its tail.
        motion_tail = motion_135[-N_cond_tail:]  # (N_cond_tail, 135) world coords
        target_first_canon = target_motion[0:1]  # (1, 135) canonical (source pose)

        # Place target_first forward of motion_tail (same trick as E14)
        forward_step = float(setting_kwargs.get('_forward_step', 1.0))
        motion_tail_t = _torch.from_numpy(motion_tail).float()
        target_first_t = _torch.from_numpy(target_first_canon).float()
        target_first_world = place_b_after_a(
            motion_tail_t, target_first_t,
            forward_step=forward_step,
            yaw_offset_deg=0.0,
        ).numpy()

        # Adaptive transition length
        from hftrainer.evaluation.motion.m2m_eval_metrics import motion135_to_positions_np
        pos_tail = motion135_to_positions_np(motion_tail, bone_offsets)
        pos_target_world = motion135_to_positions_np(target_first_world, bone_offsets)
        N_transition = compute_transition_length(
            pos_tail[-1, 0], pos_target_world[0, 0])

        # Build sequence: [motion_tail | zeros(transition) | target_first_world]
        transition_pad = np.zeros((N_transition, 135), dtype=np.float32)
        world_segment = np.concatenate(
            [motion_tail, transition_pad, target_first_world], axis=0)
        T = world_segment.shape[0]

        # Canonicalize the whole segment
        world_segment_t = _torch.from_numpy(world_segment).float()
        canon_segment_t, R_canon, offset_canon = canonicalize_segment(
            world_segment_t, anchor_frame=0)
        motion_135 = canon_segment_t.numpy()
        gt_motion_135 = motion_135

        _transition_canon_info = dict(
            R_canon=R_canon, offset_canon=offset_canon,
            motion_a_full=None, motion_b_world_full=None,
            N_cond=N_cond_tail, N_transition=N_transition,
        )

        if motion_dim == 198:
            motion_raw = motion_135_to_198(motion_135, bone_offsets)
        else:
            motion_raw = motion_135

        mask = build_transition_to_target_first_mask(
            T, 135, N_cond_tail=N_cond_tail, N_transition=N_transition)

    # ---- Special handling for E16: transition from target last frame ----
    elif task.task_id == 'E16' and '_use_target_last' in setting_kwargs:
        N_cond_head = setting_kwargs.get('_cond_head_frames', 15)

        # Load target motion to get its last frame
        target_path = sample.get('target_motion_path', '')
        if not os.path.isabs(target_path):
            target_path = os.path.join(MOTION_DATA_DIR, target_path)
        target_motion = load_motion_135d(target_path)
        if target_motion is None:
            return {}, None

        # ------------------------------------------------------------------
        # v2 transition redesign (2026-04-20): target_last is a static pose
        # sitting in front (as the "anchor"), motion_head is placed forward
        # of it in world coords; the whole segment is then canonicalized
        # so target_last (the anchor frame) is at origin facing +Z.
        # ------------------------------------------------------------------
        import torch as _torch
        from hftrainer.pipelines.motion.transition_utils import (
            canonicalize_segment, place_b_after_a,
        )

        target_last = target_motion[-1:]                          # (1, 135) canonical
        motion_head = sample['motion'][:N_cond_head]              # (N_cond_head, 135) canonical

        # Place motion_head forward of target_last in world coords
        forward_step = float(setting_kwargs.get('_forward_step', 1.0))
        target_last_t = _torch.from_numpy(target_last).float()
        motion_head_t = _torch.from_numpy(motion_head).float()
        motion_head_world = place_b_after_a(
            target_last_t, motion_head_t,
            forward_step=forward_step,
            yaw_offset_deg=0.0,
        ).numpy()

        # Adaptive transition length
        from hftrainer.evaluation.motion.m2m_eval_metrics import motion135_to_positions_np
        pos_target = motion135_to_positions_np(target_last, bone_offsets)
        pos_head_world = motion135_to_positions_np(motion_head_world, bone_offsets)
        N_transition = compute_transition_length(
            pos_target[-1, 0], pos_head_world[0, 0])

        # Build sequence: [target_last | zeros(transition) | motion_head_world]
        transition_pad = np.zeros((N_transition, 135), dtype=np.float32)
        world_segment = np.concatenate(
            [target_last, transition_pad, motion_head_world], axis=0)
        T = world_segment.shape[0]

        # Canonicalize
        world_segment_t = _torch.from_numpy(world_segment).float()
        canon_segment_t, R_canon, offset_canon = canonicalize_segment(
            world_segment_t, anchor_frame=0)
        motion_135 = canon_segment_t.numpy()
        gt_motion_135 = motion_135

        _transition_canon_info = dict(
            R_canon=R_canon, offset_canon=offset_canon,
            motion_a_full=None, motion_b_world_full=None,
            N_cond=N_cond_head, N_transition=N_transition,
        )

        if motion_dim == 198:
            motion_raw = motion_135_to_198(motion_135, bone_offsets)
        else:
            motion_raw = motion_135

        mask = build_transition_to_target_last_mask(
            T, 135, N_transition=N_transition, N_cond_head=N_cond_head)

    # ---- Standard mask building ----
    else:
        D = motion_dim
        if task.task_id == 'E3' and setting_name == 'D':
            # E3-D: SPARSE adaptive keyframe — keep only the strongest
            # acceleration peaks, no uniform filler. Approximately 1
            # keyframe per second of motion at 30 fps.
            keyframe_indices = detect_keyframes_from_motion(
                motion_135, bone_offsets,
                sparse=True,
                target_density=1.0 / 30.0,
                peak_distance=10,
            )
            mask = build_keyframe_adaptive_mask(T, D, keyframe_indices=keyframe_indices)
        elif task.task_id == 'E4':
            mask_result = task.build_mask(T, D, setting_name)
            if isinstance(mask_result, tuple):
                mask, constraint_info = mask_result
            else:
                mask = mask_result
                constraint_info = None
        elif task.task_id == 'E6':
            gt_positions = _get_positions(motion_135, bone_offsets)
            contact_frames = detect_foot_contact_frames(gt_positions)
            if setting_name == 'A':
                mask = task.mask_builder(T, D, contact_frames=contact_frames)
            else:
                mask = task.build_mask(T, D, setting_name)
        elif task.task_id == 'E9' and setting_kwargs.get('_use_adaptive_mask'):
            # MoGenDIT adaptive mask — built directly at the model's native
            # motion dim so mask channels align one-to-one with motion
            # channels (no 135→198 broadcast hack). No dilation applied.
            # Use sample['path'] (datalist-relative motion_path) as cache key.
            sample_mp = sample.get('path', '') or sample.get('motion_path', '')
            adaptive = _load_adaptive_mask_for_motion(
                sample_mp, T, D=motion_dim,
            )
            if adaptive is None:
                # No cached adaptive mask — fall back to full mask so eval
                # never silently diverges from expected behavior.
                print(f'    [warn] No adaptive mask cached for '
                      f'{sample_mp[-60:]}, using full mask')
                mask = np.ones((T, motion_dim), dtype=np.float32)
            else:
                mask = adaptive
        else:
            mask = task.build_mask(T, D, setting_name)

    # Finalize mask dimension
    D = motion_dim

    # For 198-dim, expand 135-dim mask to 198-dim
    if D == 198 and mask.shape[-1] == 135:
        # Position channels follow the same mask pattern as rotation channels
        # For each joint j (1-21), position dims [135+j*3 : 135+j*3+3] follow
        # the same mask as rotation dims [3+j*6 : 3+j*6+6]
        pos_mask = np.zeros((T, 63), dtype=np.float32)
        for j in range(21):
            rot_mask_val = mask[:, 3 + (j + 1) * 6]  # joint j+1 (skipping pelvis)
            pos_mask[:, j * 3:(j + 1) * 3] = rot_mask_val[:, None]
        mask = np.concatenate([mask, pos_mask], axis=-1)

    # Normalize
    # CRITICAL: motion_raw is in LOCAL rotation space (from dataset), but global
    # models use mean/std computed from GLOBAL rotation data. We must convert
    # motion_raw to global rotation space before normalization.
    if rotation_space == 'global':
        import torch as _torch
        from hftrainer.datasets.motion.motionhub.transforms.fk_utils import (
            local_to_global_rot6d_torch as _l2g,
        )
        _mr = motion_raw.copy()
        _rot_local = _torch.from_numpy(_mr[:, 3:135].reshape(T, 22, 6)).float()
        _rot_global = _l2g(_rot_local)
        _mr[:, 3:135] = _rot_global.reshape(T, 132).numpy()
        if motion_dim == 198:
            # Position channels (135:198) don't change with rotation space
            pass
        motion_raw = _mr

    motion_norm = bundle.normalize_motion(
        torch.from_numpy(motion_raw).float().unsqueeze(0).to(device))
    src_mask = torch.from_numpy(mask).float().unsqueeze(0).to(device)

    # Pad to 360 frames (training always pads to 360). For E9 repair, motions
    # may be LONGER than 360 — handled below via sliding-window stitching.
    T_PAD = 360
    if T < T_PAD:
        pad_len = T_PAD - T
        # Replicate-pad motion (same as training RandomCropPadding)
        motion_norm = torch.nn.functional.pad(
            motion_norm, (0, 0, 0, pad_len), mode='constant', value=0.0)
        # Pad mask with zeros (padding frames = not masked for generation)
        src_mask = torch.nn.functional.pad(
            src_mask, (0, 0, 0, pad_len), mode='constant', value=0.0)

    # Prepare src_motion: zero out masked regions (completion/inpainting mode)
    # vs keep LQ values (editing mode). Per-setting _editing_mode overrides the
    # task-level task.is_editing default (used by E9 to test inpaint vs edit).
    is_editing_effective = setting_kwargs.get('_editing_mode', task.is_editing)
    if is_editing_effective:
        src_motion_norm = motion_norm.clone()  # editing: keep LQ values
    else:
        src_motion_norm = motion_norm * (1 - src_mask)

    # Prepare clean_motion for MAN imputation
    clean_motion = motion_norm.clone()

    # ---- Text conditioning (shared by all windows in E9 long-motion path) ----
    # Pre-compute the text embedding ONCE so the per-window loop can reuse it.
    text_fields = None
    if model_info.get('has_caption') and sample.get('caption'):
        cap = sample['caption']
        cached = _lookup_caption_embedding(cap)
        if cached is not None:
            dev = bundle.device if hasattr(bundle, 'device') else device
            text_fields = {
                'text_vec_raw': cached['text_vec_raw'].to(dev),
                'text_ctxt_raw': cached['text_ctxt_raw'].to(dev),
                'text_ctxt_raw_length': cached['text_ctxt_raw_length'].to(dev),
            }
        else:
            try:
                text_out = bundle.encode_text([cap])
                text_fields = {
                    'text_vec_raw': text_out['vtxt_input'],
                    'text_ctxt_raw': text_out['ctxt_input'],
                    'text_ctxt_raw_length': text_out['ctxt_length'],
                }
            except Exception as e:
                print(f'  WARNING: text conditioning unavailable for '
                      f'caption[:60]={cap[:60]!r} — falling back to uncond. '
                      f'Reason: {e}')

    # Set pipeline parameters
    pipeline.replacement_guidance = replacement_guidance
    pipeline.text_guidance_scale = text_guidance_scale
    pipeline.num_steps = num_steps
    # Per-setting SDEdit-style partial-noise start (currently used by E9
    # A_adaptive_inpaint to align with MoGenDIT's ada_denoise path). Default
    # 0 → standard init (masked region starts from pure noise).
    pipeline.sdedit_tau = float(setting_kwargs.get('_sdedit_tau', 0.0))

    # ──────────────────────────────────────────────────────────────
    # Inference: short path (T ≤ T_PAD) vs sliding-window (T > T_PAD)
    # ──────────────────────────────────────────────────────────────
    # For E9 Motion Repair on long sequences, the model's 360-frame context
    # is not enough. We run inference in 2 overlapping windows:
    #   Window A: frames [0 : T_PAD]
    #   Window B: frames [T - T_PAD : T]
    # and linearly blend the outputs across the overlap region
    #   overlap = [T - T_PAD, T_PAD]   (length = 2*T_PAD - T)
    # Weight A = 1 at overlap start, 0 at overlap end; B is 1 - A.
    # Non-overlap segments of A / B are copied as-is.
    #
    # Rationale: MoGenDIT uses the same strategy (see mogendit_pipeline.py,
    # use_windowed=True, window_size=224, prev_padding=20). Our window size
    # matches the training context (360) for fidelity, and 2 windows are
    # enough to cover all observed E9 samples (max observed = 590 frames).
    needs_windowed = (task.task_id == 'E9') and (T > T_PAD)

    def _run_single_window(mn_win, sm_win, src_clean_win, T_win):
        """Run one pipeline call with T_win ≤ T_PAD. mn_win/sm_win/src_clean_win
        are already padded to (1, T_PAD, D)."""
        is_editing_local = setting_kwargs.get('_editing_mode', task.is_editing)
        if is_editing_local:
            src_motion_win = mn_win.clone()
        else:
            src_motion_win = mn_win * (1 - sm_win)
        batch_win = {
            'src_motion': src_motion_win,
            'src_mask': sm_win,
            'src_length': [T_win],
            'tgt_length': [T_win],
            'clean_motion': src_clean_win.clone(),
        }
        if text_fields is not None:
            batch_win.update(text_fields)
        with torch.no_grad():
            out_win = pipeline(batch_win)
        # out_win['latent']: (1, T_PAD, D) normalized; return first T_win only
        return out_win['latent'][:, :T_win, :]

    t0 = time.time()
    if not needs_windowed:
        # Standard single-pass inference
        batch = {
            'src_motion': src_motion_norm,
            'src_mask': src_mask,
            'src_length': [T],
            'tgt_length': [T],
            'clean_motion': clean_motion,
        }
        if text_fields is not None:
            batch.update(text_fields)
        with torch.no_grad():
            output = pipeline(batch)
        sampled_norm = output['latent']  # (1, T_PAD, D) normalized
    else:
        # 2-window sliding-window inference for E9 long motions.
        # motion_norm/src_mask/clean_motion are full length (1, T, D).
        starts = [0, T - T_PAD]
        window_outputs = []
        for start in starts:
            end = start + T_PAD
            mn_win = motion_norm[:, start:end, :].contiguous()
            sm_win = src_mask[:, start:end, :].contiguous()
            sc_win = clean_motion[:, start:end, :].contiguous()
            samp_win = _run_single_window(mn_win, sm_win, sc_win, T_PAD)
            window_outputs.append((start, samp_win))
        # Blend: A's contribution tapers from 1→0 across the overlap;
        # B's tapers from 0→1 on the same region.
        startA, outA = window_outputs[0]  # covers [0, T_PAD)
        startB, outB = window_outputs[1]  # covers [T-T_PAD, T)
        D = outA.shape[-1]
        sampled_full = torch.zeros((1, T, D), dtype=outA.dtype, device=outA.device)
        # Non-overlap prefix from A: [0, startB)
        sampled_full[:, 0:startB, :] = outA[:, 0:startB, :]
        # Overlap region: [startB, T_PAD)  (inside A's frame), same as
        # [0, T_PAD - startB)  inside B's frame.
        overlap_len = T_PAD - startB
        if overlap_len > 0:
            w = torch.linspace(1.0, 0.0, steps=overlap_len,
                               dtype=outA.dtype, device=outA.device)
            w = w.view(1, overlap_len, 1)
            a_over = outA[:, startB:T_PAD, :]
            b_over = outB[:, 0:overlap_len, :]
            sampled_full[:, startB:T_PAD, :] = a_over * w + b_over * (1.0 - w)
        # Non-overlap suffix from B: [T_PAD, T)
        sampled_full[:, T_PAD:T, :] = outB[:, overlap_len:T_PAD, :]
        sampled_norm = sampled_full
    elapsed = time.time() - t0

    # Get output motion (sampled_norm is (1, T_eff, D); T_eff = T_PAD for short
    # path or T for windowed path — both handled by the crop-to-T below).
    output_denorm = bundle.denormalize_motion(sampled_norm)[0].cpu().numpy()

    # Crop to actual target length (discard padding frames)
    output_denorm = output_denorm[:T]

    # For metrics: always work with 135-dim + FK
    if motion_dim == 198:
        output_135 = output_denorm[:, :135]
    else:
        output_135 = output_denorm

    # Overwrite condition regions with original GT to ensure condition frames
    # are identical across models (different mean/std cause drift after denorm)
    #
    # ROTATION SPACE NOTE:
    #   denormalize_motion() returns data in the MODEL's native rotation space:
    #   - local model → output_135 is LOCAL rot6d
    #   - global model → output_135 is GLOBAL rot6d
    #   GT motion_135 is ALWAYS LOCAL (from dataset).
    #   For global models, GT must be converted to global before replacing.
    #   motion135_to_fk(..., rotation_space='global') later converts global→local.
    mask_135 = mask[:, :135] if mask.shape[-1] > 135 else mask
    cond_mask = (mask_135 < 0.5)  # mask=0 means condition (kept)
    if cond_mask.any():
        output_135 = output_135.copy()
        if rotation_space == 'global':
            import torch as _torch
            from hftrainer.datasets.motion.motionhub.transforms.fk_utils import (
                local_to_global_rot6d_torch,
            )
            rot6d_local = _torch.from_numpy(
                motion_135[:, 3:135].reshape(T, 22, 6)).float()
            rot6d_global = local_to_global_rot6d_torch(rot6d_local)
            gt_global_135 = motion_135.copy()
            gt_global_135[:, 3:135] = rot6d_global.reshape(T, 132).numpy()
            output_135[cond_mask] = gt_global_135[cond_mask]
        else:
            output_135[cond_mask] = motion_135[cond_mask]

    # ---- Decanonicalize transition output back to world coordinates ----
    # For E14/E15/E16, inference ran in canonical space (anchor at origin,
    # heading +Z). Map the output back to world coords and stitch with the
    # full A / B motions so metrics and visualization are in world space.
    _canon_info = locals().get('_transition_canon_info', None)
    if (task.task_id in ('E14', 'E15', 'E16') and _canon_info is not None):
        import torch as _torch
        from hftrainer.pipelines.motion.transition_utils import (
            decanonicalize_segment,
        )
        R_canon = _canon_info['R_canon']
        offset_canon = _canon_info['offset_canon']

        # Decanonicalize the whole segment (all T frames)
        out_t = _torch.from_numpy(output_135).float()
        out_world_t = decanonicalize_segment(out_t, R_canon, offset_canon)
        output_135 = out_world_t.numpy()

        # Also decanonicalize the GT segment so metrics compare in world coords
        gt_t = _torch.from_numpy(gt_motion_135).float()
        gt_world_t = decanonicalize_segment(gt_t, R_canon, offset_canon)
        gt_motion_135 = gt_world_t.numpy()

        # For E14: stitch with the full motion_a prefix and motion_b_world
        # suffix so the visualized output covers the entire path, not just
        # the transition window.
        if task.task_id == 'E14':
            motion_a_full = _canon_info['motion_a_full']
            motion_b_full = _canon_info['motion_b_world_full']
            N_cond = _canon_info['N_cond']
            if motion_a_full is not None and motion_b_full is not None:
                prefix = motion_a_full[:-N_cond]  # all of A except its tail
                suffix = motion_b_full[N_cond:]   # all of B except its head
                output_135 = np.concatenate([prefix, output_135, suffix], axis=0)
                gt_motion_135 = np.concatenate([prefix, gt_motion_135, suffix], axis=0)

    # Compute metrics
    metrics = compute_all_metrics(
        pred_motion=output_135,
        gt_motion=gt_motion_135 if task.needs_gt else None,
        mask=mask_135 if task.needs_gt else None,
        bone_offsets=bone_offsets,
        rotation_space=rotation_space,
        fps=sample.get('fps', 30),
        compute_fk=True,
    )

    # Task-specific metrics
    if task.task_id == 'E4' and constraint_info is not None:
        from hftrainer.evaluation.motion.m2m_eval_metrics import (
            compute_end_effector_error, motion135_to_positions_np,
        )
        pred_pos = motion135_to_positions_np(output_135, bone_offsets)
        gt_pos = motion135_to_positions_np(gt_motion_135, bone_offsets)
        # Get constraint positions from GT
        ee_constraints = extract_ee_constraints_from_gt(
            gt_pos,
            constraint_info['joint_names'],
            frame_interval=task.settings[setting_name].mask_kwargs.get('frame_interval', 10),
        )
        ee_metrics = compute_end_effector_error(
            pred_pos, ee_constraints[0], ee_constraints[1], ee_constraints[2])
        metrics.update(ee_metrics)

    if task.task_id == 'E5':
        from hftrainer.evaluation.motion.m2m_eval_metrics import compute_heading_error
        try:
            heading = compute_heading_error(output_135, gt_motion_135, bone_offsets)
            metrics['heading_error'] = heading
        except Exception:
            pass

    if task.task_id == 'E8':
        from hftrainer.evaluation.motion.m2m_eval_metrics import compute_loop_continuity
        # DO NOT hard-overwrite output_135[-1] = output_135[0] — that creates
        # a visible jump when the model has not actually converged to the
        # loop. The mask already locks frame 0 and frame T-1 as known anchors
        # (see build_loop_mask / build_loop_completion_mask), so a correct
        # model output will have output_135[-1] ≈ output_135[0] naturally.
        # compute_loop_continuity measures how close they really are.
        loop = compute_loop_continuity(output_135, bone_offsets,
                                       fps=sample.get('fps', 30))
        metrics.update(loop)
        # For loop completion (B/C/D): also compute boundary acceleration jump
        # at the transition point (T_gt boundary)
        if '_loop_append' in setting_kwargs:
            N_append = setting_kwargs['_loop_append']
            T_gt = T - N_append
            # Boundary accel jump at the GT->generated transition
            from hftrainer.evaluation.motion.m2m_eval_metrics import motion135_to_positions_np
            pred_pos = motion135_to_positions_np(output_135, bone_offsets)
            if T_gt > 1 and T > T_gt + 1:
                vel = np.diff(pred_pos, axis=0) * sample.get('fps', 30)
                acc = np.diff(vel, axis=0) * sample.get('fps', 30)
                # Boundary frame = T_gt - 1 (last GT frame's acceleration vs first generated frame)
                if T_gt - 1 < acc.shape[0] and T_gt < acc.shape[0]:
                    boundary_jump = np.linalg.norm(acc[T_gt] - acc[T_gt - 1], axis=-1).mean()
                    metrics['boundary_accel_jump_loop'] = float(boundary_jump)

    # E14: transition stitching — boundary metrics at both A/B boundaries
    if task.task_id == 'E14':
        from hftrainer.evaluation.motion.m2m_eval_metrics import motion135_to_positions_np
        pred_pos = motion135_to_positions_np(output_135, bone_offsets)
        N_cond = setting_kwargs.get('_cond_frames', 15)
        fps_val = sample.get('fps', 30)
        vel = np.diff(pred_pos, axis=0) * fps_val
        acc = np.diff(vel, axis=0) * fps_val
        # Boundary at A->transition: frame N_cond-1
        if N_cond - 1 < acc.shape[0] and N_cond < acc.shape[0]:
            jump_a = np.linalg.norm(acc[N_cond] - acc[N_cond - 1], axis=-1).mean()
            metrics['boundary_accel_jump_a'] = float(jump_a)
        # Boundary at transition->B: frame T - N_cond - 1
        b_boundary = T - N_cond - 1
        if 0 < b_boundary < acc.shape[0] and b_boundary + 1 < acc.shape[0]:
            jump_b = np.linalg.norm(acc[b_boundary + 1] - acc[b_boundary], axis=-1).mean()
            metrics['boundary_accel_jump_b'] = float(jump_b)
        metrics['transition_length'] = int(T - 2 * N_cond)

    # E15: transition to target first frame — MPJPE at last frame
    if task.task_id == 'E15':
        from hftrainer.evaluation.motion.m2m_eval_metrics import motion135_to_positions_np
        pred_pos = motion135_to_positions_np(output_135, bone_offsets)
        gt_pos = motion135_to_positions_np(gt_motion_135, bone_offsets)
        # Last frame should match target first frame
        mpjpe_last = np.sqrt(np.sum(
            (pred_pos[-1] - gt_pos[-1]) ** 2, axis=-1)).mean()
        metrics['mpjpe_last_frame'] = float(mpjpe_last)
        N_cond_tail = setting_kwargs.get('_cond_tail_frames', 15)
        metrics['transition_length'] = int(T - N_cond_tail - 1)

    # E16: transition from target last frame — MPJPE at first frame
    if task.task_id == 'E16':
        from hftrainer.evaluation.motion.m2m_eval_metrics import motion135_to_positions_np
        pred_pos = motion135_to_positions_np(output_135, bone_offsets)
        gt_pos = motion135_to_positions_np(gt_motion_135, bone_offsets)
        # First frame should match target last frame
        mpjpe_first = np.sqrt(np.sum(
            (pred_pos[0] - gt_pos[0]) ** 2, axis=-1)).mean()
        metrics['mpjpe_first_frame'] = float(mpjpe_first)
        N_cond_head = setting_kwargs.get('_cond_head_frames', 15)
        metrics['transition_length'] = int(T - N_cond_head - 1)

    if task.task_id == 'E9' and motion_dim == 198:
        from hftrainer.evaluation.motion.m2m_eval_metrics import compute_fk_consistency
        fk_cons = compute_fk_consistency(output_denorm, bone_offsets, pos_start_dim=135)
        metrics['fk_consistency'] = fk_cons

    metrics['inference_time'] = round(elapsed, 2)
    return metrics, output_135


def _get_positions(motion_135: np.ndarray, bone_offsets: np.ndarray) -> np.ndarray:
    """Helper to get FK positions."""
    from hftrainer.evaluation.motion.m2m_eval_metrics import motion135_to_positions_np
    return motion135_to_positions_np(motion_135, bone_offsets)


# ============================================================================
# Report generation
# ============================================================================

def print_comparative_table(
    all_results: Dict[str, Dict],
    task_id: str,
    setting_name: str,
    model_names: List[str],
):
    """Print a comparative table for one task+setting across models."""
    key_metrics = [
        'mpjpe_masked', 'mpjpe_unmasked', 'jitter_pos',
        'bone_length_cv_mean', 'foot_skating_ratio',
        'trajectory_ade', 'trajectory_fde',
        'ee_error_mean', 'heading_error',
        'loop_position_error', 'loop_velocity_error',
        'boundary_accel_jump', 'boundary_accel_jump_loop',
        'boundary_accel_jump_a', 'boundary_accel_jump_b',
        'mpjpe_last_frame', 'mpjpe_first_frame',
        'transition_length', 'inference_time',
    ]

    # Find which metrics are present
    available_metrics = []
    for m_name in model_names:
        task_res = all_results.get(m_name, {}).get('tasks', {}).get(
            f'{task_id}_{setting_name}', {})
        agg = task_res.get('aggregated', {})
        for km in key_metrics:
            km_mean = f'{km}'
            if km_mean in agg and km_mean not in available_metrics:
                available_metrics.append(km_mean)

    if not available_metrics:
        return

    # Header
    col_w = 16
    header = f'{"Model":<25}'
    for m in available_metrics:
        header += f'{m:<{col_w}}'
    print(header)
    print('-' * len(header))

    for m_name in model_names:
        task_res = all_results.get(m_name, {}).get('tasks', {}).get(
            f'{task_id}_{setting_name}', {})
        agg = task_res.get('aggregated', {})
        if not agg:
            print(f'{m_name:<25} (no results)')
            continue

        row = f'{m_name:<25}'
        for m in available_metrics:
            val = agg.get(m, {}).get('mean', 'N/A')
            if isinstance(val, float):
                row += f'{val:<{col_w}.4f}'
            else:
                row += f'{str(val):<{col_w}}'
        print(row)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='HyMotion M2M v2 comprehensive evaluation')
    parser.add_argument('--models', nargs='+', choices=list(ALL_MODELS.keys()),
                        default=list(V2_MODELS.keys()),
                        help='Models to evaluate')
    parser.add_argument('--tasks', nargs='+',
                        help='Task IDs to evaluate (E1-E12)')
    parser.add_argument('--all-tasks', action='store_true',
                        help='Run all 12 tasks')
    parser.add_argument('--settings', nargs='+',
                        help='Sub-settings to run (A, B, C, D, default)')
    parser.add_argument('--max-samples', type=int, default=50,
                        help='Max samples per task')
    parser.add_argument('--num-steps', type=int, default=50,
                        help='ODE integration steps')
    parser.add_argument('--replacement-guidance', type=str, default='skip_last',
                        choices=['none', 'all', 'skip_last', 'flow_interp'],
                        help='Replacement guidance mode for MAN imputation')
    parser.add_argument('--text-guidance-scale', type=float, default=5.0,
                        help='CFG scale for text-conditioned models (5.0 standard for flow matching)')
    parser.add_argument('--output-dir', type=str,
                        default='work_dirs/m2m_v2_eval_report')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save-npz', action='store_true',
                        help='Save output NPZ files for visualization')
    parser.add_argument('--motion-data-dir', type=str, default=MOTION_DATA_DIR)
    parser.add_argument('--use-rewritten', action='store_true',
                        help='Prefer the rewritten datalist variant '
                             '(eval_e*_rewritten.json) for caption-carrying '
                             'tasks. Produced by scripts/rewrite_eval_captions.py.')
    args = parser.parse_args()

    from hftrainer.evaluation.motion.m2m_eval_tasks import EVAL_TASKS, get_task
    from hftrainer.pipelines.motion.differentiable_fk import motion135_to_fk

    # Determine tasks
    if args.all_tasks:
        task_ids = list(EVAL_TASKS.keys())
    elif args.tasks:
        task_ids = args.tasks
    else:
        # Default Phase 1: E2, E3, E5 (KIMODO comparable)
        task_ids = ['E2', 'E3', 'E5']

    os.makedirs(args.output_dir, exist_ok=True)

    # Load bone offsets
    bone_offsets_path = 'data/hymotion_m2m_data/bone_offsets_22.pt'
    if os.path.exists(bone_offsets_path):
        bone_offsets = torch.load(bone_offsets_path, map_location='cpu').numpy()
    else:
        print(f'WARNING: bone_offsets not found at {bone_offsets_path}')
        print('Run: python tools/precompute_bone_offsets.py')
        sys.exit(1)

    all_results = {}

    for model_name in args.models:
        print(f'\n{"=" * 70}')
        print(f'Model: {model_name} — {ALL_MODELS[model_name]["desc"]}')
        print(f'{"=" * 70}')

        bundle, pipeline, ckpt_path, model_info = load_model(
            model_name, args.device)
        if bundle is None:
            all_results[model_name] = {'error': 'no checkpoint'}
            continue

        model_dim = model_info.get('motion_dim', MOTION_DIM_V2)
        convert_198 = (model_dim == 198)

        model_results = {
            'checkpoint': ckpt_path,
            'model': model_name,
            'desc': model_info['desc'],
            'rotation_space': model_info['rotation_space'],
            'motion_dim': model_dim,
            'num_steps': args.num_steps,
            'replacement_guidance': args.replacement_guidance,
            'tasks': {},
        }

        for task_id in task_ids:
            task = get_task(task_id)

            # Skip caption-requiring tasks for unconditioned models
            if task.needs_caption and not model_info.get('has_caption', False):
                print(f'\n  Skipping {task_id} ({task.name}) — requires caption')
                continue

            # Skip caption-enabled models on tasks that are not caption-aware
            # (e.g. E9 Motion Repair, E14 Transition — no semantic text). These
            # tasks produced visibly distorted outputs from caption models in
            # earlier eval rounds, and caption does not add value.
            task_caption_aware = getattr(task, 'caption_aware', True)
            if (not task_caption_aware) and model_info.get('has_caption', False):
                print(f'\n  Skipping {task_id} ({task.name}) for caption model '
                      f'{model_name} — task is not caption-aware')
                continue

            # Determine settings to run
            if args.settings:
                settings = [s for s in args.settings if s in task.settings]
            else:
                settings = list(task.settings.keys())

            for setting_name in settings:
                task_key = f'{task_id}_{setting_name}'
                print(f'\n  Task: {task_key} — {task.name} ({task.settings[setting_name].description})')

                # Load evaluation data.
                # Priority (highest first):
                #   1. {base}_rewritten.json  (if --use-rewritten)
                #   2. {EVAL_DATA_DIR}/{data_file}
                #   3. {EVAL_DATA_DIR_LEGACY}/{data_file}
                eval_file = None
                if args.use_rewritten:
                    base = os.path.splitext(task.data_file)[0]
                    rewritten_file = os.path.join(EVAL_DATA_DIR, base + '_rewritten.json')
                    if os.path.exists(rewritten_file):
                        eval_file = rewritten_file
                if eval_file is None:
                    eval_file = os.path.join(EVAL_DATA_DIR, task.data_file)
                    if not os.path.exists(eval_file):
                        eval_file = os.path.join(EVAL_DATA_DIR_LEGACY, task.data_file)
                if not os.path.exists(eval_file):
                    print(f'    WARNING: eval file not found: {task.data_file}')
                    continue
                if args.use_rewritten and not eval_file.endswith('_rewritten.json'):
                    print(f'    [note] no rewritten datalist for {task.task_id}, '
                          f'falling back to {os.path.basename(eval_file)}')

                samples = load_eval_samples(
                    eval_file,
                    args.motion_data_dir,
                    args.max_samples,
                    require_caption=task.needs_caption,
                    bone_offsets=bone_offsets if convert_198 else None,
                    convert_to_198=convert_198,
                    task_id=task.task_id,
                )
                print(f'    Loaded {len(samples)} samples')

                if not samples:
                    print('    WARNING: No valid samples!')
                    continue

                # Filter for E2-B: long sequences only
                if task_id == 'E2' and setting_name == 'B':
                    samples = [s for s in samples if s['T'] > 200]
                    print(f'    Filtered to {len(samples)} long sequences (>200 frames)')

                # Run evaluation
                from hftrainer.evaluation.motion.m2m_eval_metrics import aggregate_metrics

                per_sample_metrics = []
                npz_dir = None
                if args.save_npz:
                    npz_dir = os.path.join(args.output_dir, model_name, task_key, 'npz')
                    os.makedirs(npz_dir, exist_ok=True)

                for i, sample in enumerate(samples):
                    try:
                        metrics, output_135 = evaluate_sample(
                            bundle, pipeline, sample, task, setting_name,
                            model_info, bone_offsets, args.device,
                            replacement_guidance=args.replacement_guidance,
                            text_guidance_scale=(
                                args.text_guidance_scale
                                if model_info.get('has_caption') else 1.0),
                            num_steps=args.num_steps,
                        )

                        # Save NPZ with FK positions for visualization
                        if npz_dir is not None and output_135 is not None:
                            try:
                                output_135_t = torch.from_numpy(output_135).float()
                                wp, _, _, _ = motion135_to_fk(
                                    output_135_t, torch.from_numpy(bone_offsets),
                                    rotation_space=model_info.get('rotation_space', 'local'))
                                pos_np = wp.numpy()
                                npz_path = os.path.join(npz_dir, f'{i:05d}.npz')
                                np.savez_compressed(npz_path, motion_135=output_135,
                                                    positions=pos_np, translation=output_135[:, :3])
                                metrics['_npz_path'] = npz_path
                            except Exception:
                                pass

                        # Store sample info for DB import
                        metrics['_sample_idx'] = i
                        metrics['_caption'] = sample.get('caption', '')
                        metrics['_num_frames'] = sample.get('T', 0)
                        per_sample_metrics.append(metrics)

                        if (i + 1) % 10 == 0:
                            print(f'    [{i + 1}/{len(samples)}] done')
                    except Exception as e:
                        print(f'    [{i + 1}] ERROR: {e}')
                        import traceback
                        traceback.print_exc()
                        continue

                # Aggregate
                aggregated = aggregate_metrics(per_sample_metrics)

                model_results['tasks'][task_key] = {
                    'task_id': task_id,
                    'setting': setting_name,
                    'num_samples': len(per_sample_metrics),
                    'aggregated': aggregated,
                    'per_sample': per_sample_metrics,  # full list for DB import
                }

                # Print summary
                print(f'    Results (n={len(per_sample_metrics)}):')
                for metric_name in task.default_metrics:
                    if metric_name in aggregated:
                        m = aggregated[metric_name]
                        print(f'      {metric_name}: {m["mean"]:.4f} ± {m["std"]:.4f} '
                              f'(med={m["median"]:.4f})')

        all_results[model_name] = model_results

        # Free GPU memory
        del bundle, pipeline
        torch.cuda.empty_cache()

    # ---- Save results ----
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(args.output_dir, f'eval_v2_{timestamp}.json')

    # Make results JSON-serializable
    def _make_serializable(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    serializable = json.loads(json.dumps(all_results, default=_make_serializable))
    with open(output_file, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f'\n\nFull results saved to {output_file}')

    # ---- Comparative summary ----
    print(f'\n{"=" * 80}')
    print('COMPARATIVE SUMMARY')
    print(f'{"=" * 80}')

    for task_id in task_ids:
        task = get_task(task_id)
        settings_run = args.settings or list(task.settings.keys())
        for setting_name in settings_run:
            if setting_name not in task.settings:
                continue
            task_key = f'{task_id}_{setting_name}'
            print(f'\n--- {task_key}: {task.name} ({task.settings[setting_name].description}) ---')
            print_comparative_table(all_results, task_id, setting_name, args.models)

    # ---- Per-task best model summary ----
    print(f'\n\n{"=" * 80}')
    print('BEST MODEL PER TASK (by MPJPE or primary metric)')
    print(f'{"=" * 80}')

    for task_id in task_ids:
        task = get_task(task_id)
        primary_metric = task.default_metrics[0] if task.default_metrics else 'jitter_pos'
        settings_run = args.settings or list(task.settings.keys())

        for setting_name in settings_run:
            if setting_name not in task.settings:
                continue
            task_key = f'{task_id}_{setting_name}'

            best_model = None
            best_val = float('inf')

            for m_name in args.models:
                task_res = all_results.get(m_name, {}).get('tasks', {}).get(task_key, {})
                agg = task_res.get('aggregated', {})
                if primary_metric in agg:
                    val = agg[primary_metric].get('mean', float('inf'))
                    if val < best_val:
                        best_val = val
                        best_model = m_name

            if best_model:
                print(f'  {task_key}: {best_model} ({primary_metric}={best_val:.4f})')
            else:
                print(f'  {task_key}: no results')


if __name__ == '__main__':
    main()
