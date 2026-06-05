#!/usr/bin/env python3
"""HyMotion M2M v2 comprehensive evaluation across E1-E16.

Evaluates 4 model variants:
  - uncond_local:   No text, local rotation
  - uncond_global:  No text, global rotation
  - caption_local:  Text-conditioned, local rotation
  - caption_global: Text-conditioned, global rotation

Usage:
    # Run Phase 1 tasks (E2, E3, E4, E5) on all models
    python scripts/eval/eval_m2m_v2_all_tasks.py --tasks E2 E3 E5 --max-samples 50

    # Run specific task with specific setting
    python scripts/eval/eval_m2m_v2_all_tasks.py --tasks E2 --settings A B --models uncond_local

    # Run all tasks (full evaluation)
    python scripts/eval/eval_m2m_v2_all_tasks.py --all-tasks --max-samples 100

    # With replacement guidance for MAN imputation
    python scripts/eval/eval_m2m_v2_all_tasks.py --tasks E2 --replacement-guidance skip_last

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

CAPTION_EMBED_CACHE_PATH = Path(__file__).resolve().parents[2] / \
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
        'config': 'configs/hymotion_m2m/hymotion_m2m_uncond_local_046b.py',
        'work_dir': 'work_dirs/hymotion_m2m_v2_uncond_local_046b',
        'desc': 'v2 Unconditioned + Local rotation',
        'has_caption': False,
        'rotation_space': 'local',
    },
    'uncond_global': {
        'config': 'configs/hymotion_m2m/hymotion_m2m_uncond_global_046b.py',
        'work_dir': 'work_dirs/hymotion_m2m_v2_uncond_global_046b',
        'desc': 'v2 Unconditioned + Global rotation',
        'has_caption': False,
        'rotation_space': 'global',
    },
    'caption_local': {
        'config': 'configs/hymotion_m2m/hymotion_m2m_caption_local_046b.py',
        'work_dir': 'work_dirs/hymotion_m2m_v2_caption_local_046b',
        'desc': 'v2 Caption + Local rotation (mixed training)',
        'has_caption': True,
        'rotation_space': 'local',
    },
    'caption_global': {
        'config': 'configs/hymotion_m2m/hymotion_m2m_caption_global_046b.py',
        'work_dir': 'work_dirs/hymotion_m2m_v2_caption_global_046b',
        'desc': 'v2 Caption + Global rotation (mixed training)',
        'has_caption': True,
        'rotation_space': 'global',
    },
    # Phase 1 variants: pure T2M curriculum (no completion tasks)
    'caption_local_phase1': {
        'config': 'configs/hymotion_m2m/hymotion_m2m_caption_local_phase1.py',
        'work_dir': 'work_dirs/hymotion_m2m_v2_caption_local_phase1',
        'desc': 'v2 Caption + Local rotation (Phase 1: pure T2M)',
        'has_caption': True,
        'rotation_space': 'local',
    },
    'caption_global_phase1': {
        'config': 'configs/hymotion_m2m/hymotion_m2m_caption_global_phase1.py',
        'work_dir': 'work_dirs/hymotion_m2m_v2_caption_global_phase1',
        'desc': 'v2 Caption + Global rotation (Phase 1: pure T2M)',
        'has_caption': True,
        'rotation_space': 'global',
    },
    # Phase 2 variants: T2M base + completion curriculum (longer training)
    'caption_local_phase2': {
        'config': 'configs/hymotion_m2m/hymotion_m2m_caption_local_phase2.py',
        'work_dir': 'work_dirs/hymotion_m2m_v2_caption_local_phase2',
        'desc': 'v2 Caption + Local rotation (Phase 2: T2M + completion)',
        'has_caption': True,
        'rotation_space': 'local',
    },
    'caption_global_phase2': {
        'config': 'configs/hymotion_m2m/hymotion_m2m_caption_global_phase2.py',
        'work_dir': 'work_dirs/hymotion_m2m_v2_caption_global_phase2',
        'desc': 'v2 Caption + Global rotation (Phase 2: T2M + completion)',
        'has_caption': True,
        'rotation_space': 'global',
    },
    # Phase 0 root-representation ablations trained on the 2026-05-14 data.
    'M2M_v2_KIMODO_root_caption_permo_resume_E4': {
        'config': 'configs/hymotion_m2m/hymotion_m2m_kimodo_caption_permo_resume_046b.py',
        'work_dir': 'work_dirs/hymotion_m2m_v2_kimodo_caption_permo_resume_E4',
        'desc': 'v2 KIMODO Root + Caption + PerMo Resume (E4)',
        'has_caption': True,
        'rotation_space': 'local',
    },
    'smpl_caption_E2': {
        'config': 'configs/hymotion_m2m/hymotion_m2m_smpl_caption_046b.py',
        'work_dir': 'work_dirs/hymotion_m2m_v2_smpl_caption_E2',
        'desc': 'v2 SMPL Root + Caption (E2)',
        'has_caption': True,
        'rotation_space': 'local',
    },
    'smpl_caption_resume_E2': {
        'config': 'configs/hymotion_m2m/hymotion_m2m_smpl_caption_resume_046b.py',
        'work_dir': 'work_dirs/hymotion_m2m_v2_smpl_caption_resume_E2',
        'desc': 'v2 SMPL Root + Caption Resume (E2)',
        'has_caption': True,
        'rotation_space': 'local',
    },
    'M2M_v2_KIMODO_root_uncond_E3': {
        'config': 'configs/hymotion_m2m/hymotion_m2m_kimodo_uncond_046b.py',
        'work_dir': 'work_dirs/hymotion_m2m_v2_kimodo_uncond_E3',
        'desc': 'v2 KIMODO Root + Unconditioned (E3)',
        'has_caption': False,
        'rotation_space': 'local',
    },
    'smpl_uncond_E1': {
        'config': 'configs/hymotion_m2m/hymotion_m2m_smpl_uncond_046b.py',
        'work_dir': 'work_dirs/hymotion_m2m_v2_smpl_uncond_E1',
        'desc': 'v2 SMPL Root + Unconditioned (E1)',
        'has_caption': False,
        'rotation_space': 'local',
    },
    # --- Final experiment checkpoints (editfix continuations) ---
    # These point at single-checkpoint symlink dirs so find_latest_checkpoint
    # locks the exact epoch (the real work_dirs have higher epochs).
    'kimodo_caption_editfix_ep240': {
        'config': 'configs/hymotion_m2m/hymotion_m2m_kimodo_caption_permo_046b.py',
        'work_dir': 'work_dirs/_eval_kimodo_editfix_ep240',
        'desc': 'v2 KIMODO Root + Caption + PerMo, editfix from ep890 (ep240)',
        'has_caption': True,
        'rotation_space': 'local',
    },
    'smpl_caption_editfix_ep230': {
        'config': 'configs/hymotion_m2m/hymotion_m2m_smpl_caption_046b.py',
        'work_dir': 'work_dirs/_eval_smpl_editfix_ep230',
        'desc': 'v2 SMPL Root + Caption, editfix from ep870 (ep230)',
        'has_caption': True,
        'rotation_space': 'local',
    },
    'kimodo_caption_editfix_latest': {
        'config': 'configs/hymotion_m2m/hymotion_m2m_kimodo_caption_permo_046b.py',
        'work_dir': 'work_dirs/hymotion_m2m_v2_kimodo_caption_permo_E4plus_editfix_from890_20260528',
        'desc': 'v2 KIMODO Root + Caption + PerMo, editfix latest checkpoint',
        'has_caption': True,
        'rotation_space': 'local',
    },
    'smpl_caption_editfix_latest': {
        'config': 'configs/hymotion_m2m/hymotion_m2m_smpl_caption_046b.py',
        'work_dir': 'work_dirs/hymotion_m2m_v2_smpl_caption_editfix_from870_20260528',
        'desc': 'v2 SMPL Root + Caption, editfix latest checkpoint',
        'has_caption': True,
        'rotation_space': 'local',
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


def _estimate_a_end_velocity(motion_a: np.ndarray, window: int = 5) -> np.ndarray:
    """Estimate motion A's instantaneous pelvis velocity at its last frame.

    Used by E14 to decide whether a motion ends "in motion" (velocity > thr)
    or "static" (velocity ≈ 0), and to extrapolate where B should sit in the
    'velocity' placement mode.

    Args:
        motion_a: (T, 135) motion A, world coords.
        window: number of trailing frames to average (smoothing).

    Returns:
        (3,) velocity vector in m/frame.
    """
    T = motion_a.shape[0]
    if T < 2:
        return np.zeros(3, dtype=np.float32)
    w = max(2, min(window + 1, T))
    trans_tail = motion_a[-w:, 0:3]          # (w, 3)
    # mean of successive frame-diffs
    vel = (trans_tail[1:] - trans_tail[:-1]).mean(axis=0)  # (3,)
    return vel.astype(np.float32)


_PLACE_B_FOOT_JOINTS = (7, 8, 10, 11)  # SMPL-22: ankles + feet
_PLACE_B_FLOOR_WIN = 5                 # frames averaged for floor estimation


def _foot_floor_y(pos: np.ndarray, side: str = 'tail',
                  win: int = _PLACE_B_FLOOR_WIN) -> float:
    """Return min foot Y over a `win`-frame window at `side` of (T, 22, 3)."""
    feet = list(_PLACE_B_FOOT_JOINTS)
    n = min(win, pos.shape[0])
    if side == 'head':
        return float(pos[:n, feet, 1].min())
    return float(pos[-n:, feet, 1].min())


def _place_b_custom(
    motion_a: np.ndarray,
    motion_b: np.ndarray,
    placement: str,
    N_transition: int,
    forward_step: float = 1.0,
    yaw_offset_deg: float = 0.0,
    rotation_space: str = "local",
    bone_offsets: Optional[np.ndarray] = None,
    y_align: str = "floor",
) -> np.ndarray:
    """Place motion B in world using one of three strategies.

    - 'forward'  → legacy: B_xz = A_end_xz + forward_step * A_fwd_dir
    - 'overlap'  → B_xz = A_end_xz  (y preserved from B's own start)
    - 'velocity' → B_xz = A_end_xz + A_tail_velocity.xz * N_transition
                   (i.e. "if A kept moving at its current speed for
                    N_transition frames, B should land where A would be")

    Y-alignment between A and B (2026-04-26, ``y_align`` parameter):

      * ``'foot'`` / ``'floor'``  → align B so its first-frame **lowest
        foot joint Y** matches A's last-frame lowest foot joint Y. This
        is the floor-plane match: stand-on-stand, crouch-on-crouch,
        sit-on-sit all stay grounded regardless of the absolute pelvis
        height baseline of either clip. **Default.**
      * ``'pelvis'``              → match A_end pelvis Y to B_start
        pelvis Y. Cleaner for locomotion (similar postures), but
        teleports the foot through the floor when A and B differ in
        posture.
      * ``'preserve_b'``          → legacy behaviour (≤ 2026-04-26):
        keep B's own pelvis Y. Causes visible floating in M-setting on
        clips with mismatched floor baselines.

    Args:
        motion_a: (T_a, 135) world-coords motion A.
        motion_b: (T_b, 135) canonical motion B (starts at origin facing +Z).
        placement: 'forward' | 'overlap' | 'velocity'.
        N_transition: number of transition frames (used by 'velocity' mode).
        forward_step: distance in meters (only used by 'forward' mode).
        yaw_offset_deg: additional yaw turn applied to B.
        rotation_space: kept for API compatibility but IGNORED — we always
            treat the input motion as LOCAL rot6d here, because this helper
            is only ever called on raw data loaded from NPZ files, which is
            always in local rotation space. The local→global conversion
            happens LATER, just before feeding the model (see line ~2450
            of `evaluate_sample`). Using `rotation_space='global'` here
            would incorrectly yaw-rotate body joints that are still local,
            which then gets compounded by `local_to_global_rot6d_torch`
            and completely destroys the pose.
        bone_offsets: (22, 3) SMPL-22 bone offsets, REQUIRED for
            ``y_align='foot'``. When ``None`` we silently fall back to
            ``y_align='preserve_b'`` (legacy).
        y_align: 'foot' (default) | 'pelvis' | 'preserve_b'. See above.

    Returns:
        (T_b, 135) motion B placed in world coords.
    """
    import torch as _torch
    from hftrainer.pipelines.motion.transition_utils import (
        build_yaw_rotation_matrix, extract_yaw_from_root_rot6d,
        apply_rigid_transform_to_motion,
    )

    motion_a_t = _torch.from_numpy(motion_a).float()
    motion_b_t = _torch.from_numpy(motion_b).float()

    a_end_yaw = extract_yaw_from_root_rot6d(motion_a_t[-1, 3:9])
    a_end_pos = motion_a_t[-1, 0:3]
    b_start_yaw = extract_yaw_from_root_rot6d(motion_b_t[0, 3:9])

    yaw_offset = _torch.as_tensor(
        yaw_offset_deg * 3.141592653589793 / 180.0,
        dtype=a_end_yaw.dtype, device=a_end_yaw.device)
    target_yaw = a_end_yaw + yaw_offset
    delta_yaw = target_yaw - b_start_yaw
    R_B = build_yaw_rotation_matrix(delta_yaw)

    # Target B[0] XZ in world coords.
    if placement == 'overlap':
        # B.xz = A_end.xz. No horizontal offset — B overlaps A's endpoint.
        target_b0_xz = a_end_pos.clone()
    elif placement == 'velocity':
        # B.xz = A_end.xz + A_tail_velocity.xz * N_transition.
        # i.e. "if A kept moving at current velocity, where would pelvis be
        # after N_transition frames?"  Captures locomotion continuity.
        vel_np = _estimate_a_end_velocity(motion_a)
        vel = _torch.from_numpy(vel_np).to(a_end_pos)
        target_b0_xz = a_end_pos + vel * float(N_transition)
    else:  # 'forward'
        forward_dir = _torch.stack([
            _torch.sin(a_end_yaw),
            _torch.zeros_like(a_end_yaw),
            _torch.cos(a_end_yaw),
        ])
        target_b0_xz = a_end_pos + forward_step * forward_dir

    # ── Preliminary B placement: only pelvis XZ is fixed, Y still B's own.
    # We need this intermediate motion to do the foot-floor / pelvis Y
    # alignment in world coords AFTER the yaw rotation has been applied
    # (since the post-rotation foot heights reflect what the renderer /
    # network actually sees).
    b0_pos = motion_b_t[0, 0:3]
    target_b0_xy0 = _torch.stack([target_b0_xz[0], b0_pos[1], target_b0_xz[2]])
    offset_xy0 = target_b0_xy0 - _torch.einsum('ij,j->i', R_B, b0_pos)
    motion_b_xy0 = apply_rigid_transform_to_motion(
        motion_b_t, R_B, offset_xy0, rotation_space='local')

    # ── Y alignment ──────────────────────────────────────────────────
    if y_align in ('foot', 'floor') and bone_offsets is not None:
        try:
            from hftrainer.evaluation.motion.m2m_eval_metrics import (
                motion135_to_positions_np as _m2p_np,
            )
            mb_xy0_np = motion_b_xy0.numpy()
            pos_a = _m2p_np(motion_a, bone_offsets)
            pos_b = _m2p_np(mb_xy0_np, bone_offsets)
            a_floor = _foot_floor_y(pos_a, side='tail')
            b_floor = _foot_floor_y(pos_b, side='head')
            delta_y = a_floor - b_floor
        except Exception:
            delta_y = 0.0
    elif y_align == 'pelvis':
        delta_y = float((a_end_pos[1] - b0_pos[1]).item())
    else:  # 'preserve_b' or fallback when bone_offsets missing
        delta_y = 0.0

    if abs(delta_y) > 1e-5:
        motion_b_world = motion_b_xy0.clone()
        motion_b_world[:, 1] = motion_b_world[:, 1] + delta_y
        return motion_b_world.numpy()

    return motion_b_xy0.numpy()


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

    project_root = Path(__file__).resolve().parents[2]
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
# E9 v2 repair helpers (2026-04-22 redesign)
# ============================================================================

def _compute_ada_keep_mask(
    motion_norm_lq: np.ndarray,       # (T, D) normalized LQ
    denoised_stage1: np.ndarray,      # (T, D) normalized stage-1 output
    threshold_mode: str,              # 'abs' or 'topk_pct'
    threshold: float,                 # abs threshold OR top-k fraction
) -> np.ndarray:
    """Stage-2 of MoGenDIT ada_denoise: build a keep_mask from the
    model's stage-1 change pattern.

    Logic (mirrors ``MoGenDIT.motion_process.motion_refiner`` lines
    339-361):

        change      = |motion_lq - denoised_stage1|   (elementwise)
        low_change  = change <= threshold              # or topk_pct
        → per-channel "clean" flag

    Aggregation strategy: because the model's 198-dim layout is
    [trans(3) + rot6d(132) + pos(63)], we aggregate to per-joint
    per-frame: a joint is "clean at frame t" iff ALL of its
    channels (rot6d 6 + pos 3) are low-change.  Translation (dims
    0:3) is kept as a single group.  This is stricter than
    per-channel and avoids artifacts where 3 out of 6 rot6d dims
    flip state.

    Returns a ``mask`` (T, D) with 1=generate, 0=keep.  Intended to
    be fed back into the pipeline with ``replacement_guidance='skip_last'``
    and ``clean_motion=LQ``, so low-change regions stay anchored to LQ.

    The MoGenDIT caller ORed this with the external base mask; we do
    that in the caller (this function only produces the "stage-2
    extension").
    """
    D = motion_norm_lq.shape[-1]
    change = np.abs(motion_norm_lq - denoised_stage1)  # (T, D)

    if threshold_mode == 'abs':
        thr = float(threshold)
    elif threshold_mode == 'topk_pct':
        # Pick the threshold so that exactly `threshold` fraction of
        # (T, D) cells have change > threshold. Any value with
        # change > thr is "high change" (will be regenerated); rest
        # is "low change" (kept).
        thr = float(np.quantile(change.ravel(), 1.0 - float(threshold)))
    else:
        raise ValueError(f'Unknown _ada_threshold_mode: {threshold_mode!r}')

    low_change_chan = (change <= thr)  # (T, D) bool, True = clean

    # Per-joint aggregation. For 198-dim we have:
    #   dim 0:3   translation (group 1)
    #   dim 3:135 rot6d, 22 joints * 6 dim each (groups 2..23)
    #   dim 135:198 pos, 21 joints * 3 dim each (follow rot6d groups 3..23)
    # For 135-dim just trans + rot6d.
    out_mask = np.ones((motion_norm_lq.shape[0], D), dtype=np.float32)

    # Translation: all 3 dims must agree
    trans_clean = low_change_chan[:, :3].all(axis=-1)  # (T,)
    out_mask[trans_clean, :3] = 0.0

    # Rot6d per joint (22 joints, dims 3 + j*6 : 3 + (j+1)*6)
    for j in range(22):
        s, e = 3 + j * 6, 3 + (j + 1) * 6
        j_clean = low_change_chan[:, s:e].all(axis=-1)  # (T,)
        out_mask[j_clean, s:e] = 0.0

    # Position channels (198-dim only): follow rot6d cleanness per joint.
    # pos channels are for joints 1..21 (no pelvis), dims 135+(j-1)*3..
    if D >= 198:
        for j in range(1, 22):
            rot_s, rot_e = 3 + j * 6, 3 + (j + 1) * 6
            j_clean_rot = low_change_chan[:, rot_s:rot_e].all(axis=-1)  # (T,)
            ps, pe = 135 + (j - 1) * 3, 135 + j * 3
            j_clean_pos = low_change_chan[:, ps:pe].all(axis=-1)
            j_clean = j_clean_rot & j_clean_pos
            out_mask[j_clean, ps:pe] = 0.0
            # Re-write rot6d section too — both pos AND rot must agree
            # for this joint to be kept. Stricter than rot-only pass.
            out_mask[j_clean, rot_s:rot_e] = 0.0

    return out_mask


# SMPL-22 parent array used for strict-mask spatial dilation (kinematic
# neighbor propagation). Duplicated from SMPL22_PARENTS earlier in file
# but kept local to document that strict_mask uses it as a symmetric
# neighborhood (parent AND children included).
def _smpl22_neighbors() -> List[List[int]]:
    parents = SMPL22_PARENTS
    children: List[List[int]] = [[] for _ in range(len(parents))]
    for child, parent in enumerate(parents):
        if parent >= 0:
            children[parent].append(child)
    neigh: List[List[int]] = []
    for j in range(len(parents)):
        nbs = set()
        if parents[j] >= 0:
            nbs.add(parents[j])
        nbs.update(children[j])
        neigh.append(sorted(nbs))
    return neigh


def _compute_strict_adaptive_mask(
    adaptive_raw: np.ndarray,         # (T, D) raw MoGenDIT adaptive mask
    dilate: int = 2,                  # temporal dilation radius (frames)
    min_blob: int = 3,                # minimum blob size (frames) to keep
    motion_dim: int = 198,
) -> np.ndarray:
    """Tighten the raw MoGenDIT adaptive mask using 3 post-processing
    steps, all in the "1=generate" convention:

    1) Per-joint aggregation: joint j at frame t is flagged iff ALL of
       its channels in the raw mask agree (reduces per-channel noise).
    2) Spatial dilation to kinematic neighbors (parent + children)
       — a bad joint usually drags its neighbors too.
    3) Temporal dilation by ±`dilate` frames.
    4) Blob filtering: drop per-joint temporal runs shorter than
       ``min_blob`` frames (isolated single-frame flags are noise).

    Returns a new (T, D) float mask with 1=generate, 0=keep.
    """
    T, D = adaptive_raw.shape
    # --- Step 1: per-joint aggregation to (T, 22) bool ------------
    joint_flag = np.zeros((T, 22), dtype=bool)
    # Translation goes to "joint 0" (pelvis) via OR with any trans dim
    trans_any = (adaptive_raw[:, :3] >= 0.5).any(axis=-1)  # (T,)
    joint_flag[:, 0] |= trans_any
    # Rot6d per joint
    for j in range(22):
        s, e = 3 + j * 6, 3 + (j + 1) * 6
        # Use ANY (less strict) since raw mask already has dropout per dim
        joint_flag[:, j] |= (adaptive_raw[:, s:e] >= 0.5).any(axis=-1)
    # Pos channels (if 198d)
    if D >= 198:
        for j in range(1, 22):
            ps, pe = 135 + (j - 1) * 3, 135 + j * 3
            joint_flag[:, j] |= (adaptive_raw[:, ps:pe] >= 0.5).any(axis=-1)

    # --- Step 2: kinematic spatial dilation -----------------------
    # 2026-04-23: EXCLUDE upper-chain small joints (neck=12, head=15,
    # wrists=20/21, collars=13/14) from being both sources AND sinks of
    # propagation. Reason: the adaptive detector often fires on head due
    # to small pose noise, then kinematic propagation spreads the flag to
    # neck → both get regenerated, and the model invents a completely new
    # head/neck rotation (visible as "sudden head lift / turn" not in LQ).
    # Case 00165 (foot_sliding): raw mask had 46 head-flagged frames and 0
    # neck frames; after propagation both became 77 frames → model
    # generated 8° single-frame head rotation deltas not in LQ.
    #
    # Big-joint propagation (hips↔knees↔ankles, spine chain) is kept since
    # those defects genuinely spread across the chain.
    _NO_PROPAGATE = {12, 13, 14, 15, 20, 21}  # neck, collars, head, wrists
    neigh = _smpl22_neighbors()
    joint_flag_sp = joint_flag.copy()
    for j in range(22):
        if j in _NO_PROPAGATE:
            continue  # don't broadcast from these joints
        for nb in neigh[j]:
            if nb in _NO_PROPAGATE:
                continue  # don't receive into these joints
            joint_flag_sp[:, nb] |= joint_flag[:, j]
    joint_flag = joint_flag_sp

    # --- Step 3: temporal dilation --------------------------------
    if dilate > 0:
        jf = joint_flag.copy()
        for s in range(1, dilate + 1):
            jf[s:] |= joint_flag[:-s]
            jf[:-s] |= joint_flag[s:]
        joint_flag = jf

    # --- Step 4: blob filter — drop runs shorter than min_blob ----
    if min_blob > 1:
        for j in range(22):
            col = joint_flag[:, j]
            if not col.any():
                continue
            # Find runs of True
            runs = []
            i = 0
            while i < T:
                if col[i]:
                    k = i
                    while k < T and col[k]:
                        k += 1
                    runs.append((i, k))  # [i, k)
                    i = k
                else:
                    i += 1
            for (s, e) in runs:
                if (e - s) < min_blob:
                    col[s:e] = False
            joint_flag[:, j] = col

    # --- Step 5: map back to (T, D) with joint-group broadcasting --
    out_mask = np.zeros((T, D), dtype=np.float32)
    # Trans follows joint 0
    out_mask[:, :3] = joint_flag[:, 0:1].astype(np.float32)
    for j in range(22):
        s, e = 3 + j * 6, 3 + (j + 1) * 6
        out_mask[:, s:e] = joint_flag[:, j:j+1].astype(np.float32)
    if D >= 198:
        for j in range(1, 22):
            ps, pe = 135 + (j - 1) * 3, 135 + j * 3
            out_mask[:, ps:pe] = joint_flag[:, j:j+1].astype(np.float32)

    return out_mask


def _gaussian_temporal_smooth(
    x: torch.Tensor,
    sigma: float,
    protect_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """1-D Gaussian temporal smoothing along axis 1 of a (B, T, D) tensor.

    Used for the D_strict_mask_*_smooth family: MAN training assumed
    ``x_t[keep] = x1[keep]`` where x1 is a clean GT motion; at inference
    the "keep" region is LQ (jittery). Pre-smoothing LQ before feeding
    it as ``clean_motion`` reduces the high-frequency energy the model
    has to reconcile against, at the cost of losing some genuine LQ
    detail on the protected region.

    Args:
        x: (B, T, D) float tensor.
        sigma: Gaussian std in frames. Kernel radius = ceil(3*sigma).
        protect_mask: optional (B, T, D) float mask in [0, 1]. Where
            protect_mask==1 (i.e. "defective, will be regenerated") the
            smoothing is skipped (value passed through unchanged) —
            smoothing there doesn't help since the model overwrites it
            anyway, and smoothing across defect boundaries would bleed
            bad values into clean neighbors.

    Returns smoothed tensor, same shape/dtype/device as ``x``.
    """
    if sigma <= 0.0:
        return x
    T = x.shape[1]
    radius = max(1, int(round(3.0 * sigma)))
    # Build 1-D Gaussian kernel
    offsets = torch.arange(
        -radius, radius + 1, dtype=x.dtype, device=x.device)
    kernel = torch.exp(-(offsets ** 2) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum()
    # Reshape x to (B*D, 1, T) for conv1d
    B, _, D = x.shape
    x_flat = x.permute(0, 2, 1).reshape(B * D, 1, T)
    w = kernel.view(1, 1, -1)
    x_pad = torch.nn.functional.pad(
        x_flat, (radius, radius), mode='replicate')
    y_flat = torch.nn.functional.conv1d(x_pad, w)  # (B*D, 1, T)
    y = y_flat.reshape(B, D, T).permute(0, 2, 1).contiguous()
    if protect_mask is not None:
        # Preserve original values where mask==1 (generate region)
        y = torch.where(protect_mask > 0.5, x, y)
    return y



# checkers (some of which load ML classifiers / body models) takes ~5s;
# reusing the same instance keeps per-sample overhead under 50ms.
_QC_CHECKER_CACHE: Dict[str, Any] = {'checker': None}


def _run_quality_checker(
    motion_135: np.ndarray,
    bone_offsets: Any,
    device: str = 'cuda',
) -> Optional[Dict[str, Any]]:
    """Run the full MotionQualityChecker suite on a 135-dim motion and
    return a summary dict compatible with per_sample metrics.

    Converts rot6d (dims 3:135, 22 joints * 6) back to axis-angle for
    the checker input, uses the output's translation (dims 0:3) as-is.
    Shapes the output as ``{poses: (T, 22, 3), trans: (T, 3)}`` which is
    what ``MotionQualityChecker.check`` expects.

    Returns None on failure.
    """
    from hftrainer.evaluation.quality_check_rules.motion_quality_checker import (
        MotionQualityChecker,
    )
    from hftrainer.models.motion.components.utils.geometry.rotation_convert import (
        rotation_6d_to_matrix, matrix_to_axis_angle,
    )

    if _QC_CHECKER_CACHE['checker'] is None:
        qc_device = device if str(device).startswith('cuda') else 'cpu'
        _QC_CHECKER_CACHE['checker'] = MotionQualityChecker(device=qc_device)
    checker = _QC_CHECKER_CACHE['checker']

    T = motion_135.shape[0]
    trans = motion_135[:, :3].astype(np.float32)          # (T, 3)
    rot6d_flat = motion_135[:, 3:135].reshape(T, 22, 6)
    # ⚠️ HyMotion's rot6d is row-major (after `load_smplx.py` reorder).
    # `rotation_6d_to_matrix` expects column-major, so we must reorder
    # columns `[0,2,4,1,3,5]` BEFORE conversion, per
    # `hftrainer/models/motion/CLAUDE.md` §Rotation 6D Convention.
    # Without this the produced axis-angle is wrong (180°-ish off for
    # large rotations) and the QC checker flags everything as broken.
    rot6d_col = rot6d_flat[..., [0, 2, 4, 1, 3, 5]]
    mat = rotation_6d_to_matrix(rot6d_col)                # (T, 22, 3, 3)
    # matrix_to_axis_angle wants (N, 3, 3) — flatten the leading dims.
    mat_flat = mat.reshape(-1, 3, 3)
    # Re-orthogonalize via SVD polar decomposition. `rotation_6d_to_matrix`
    # uses Gram-Schmidt which produces a degenerate (det=0) matrix if
    # the two 6d columns happen to be parallel (happens occasionally in
    # pad regions / unconstrained joints after model output). The QC
    # `matrix_to_axis_angle` helper asserts det > 0 and will raise
    # otherwise, so we polar-project every matrix to the nearest valid
    # rotation before converting.
    # mat_flat: (N, 3, 3). Use numpy SVD (CPU only, ~1ms for T~400*22).
    try:
        U, _S, Vt = np.linalg.svd(mat_flat)
        # Ensure det > 0 by flipping the sign of the last column of U
        # when det(U @ Vt) < 0 (matches scipy.spatial.transform convention).
        det = np.linalg.det(U @ Vt)
        U[..., :, -1] *= np.sign(det)[..., None]
        mat_flat = (U @ Vt).astype(np.float32)
    except Exception:
        pass
    aa_flat = matrix_to_axis_angle(mat_flat)              # (T*22, 3)
    aa = aa_flat.reshape(T, 22, 3)

    motion_dict = {
        'poses': aa.astype(np.float32),
        'trans': trans,
    }
    result = checker.check(motion_dict)
    # Preserve raw AggregatedCheckResult alongside the dict so callers
    # that need `.all_results[name]['invalid_mask']` (e.g. the QC-mask
    # inference path) can still reach it via `_raw` key. Most callers
    # only need the dict view.
    d = result.to_dict()
    d['_raw'] = result
    return d


def _compute_qc_defect_mask(
    motion_135: np.ndarray,
    bone_offsets: Any,
    motion_dim: int = 198,
    dilate_temp: int = 2,           # temporal ± radius
    dilate_spatial: bool = True,    # propagate to SMPL22 kinematic neighbors
    include_borderline: bool = True,
    device: str = 'cuda',
) -> Optional[np.ndarray]:
    """Run QC on the LQ motion and build a per-joint per-frame defect mask.

    Rationale (2026-04-22): MoGenDIT's adaptive mask only flags 15-30%
    of cells, but E9 LQ samples fail QC on anatomical issues (neck bent
    180°, spine hyper-extended, ankle distortion) that affect **every
    frame** — such persistent defects are invisible to change-based
    adaptive masks. Using each failing checker's own ``invalid_mask``
    (T, 22) as the source of truth gives a much more faithful mask for
    E9 repair.

    Returns: (T, motion_dim) float mask, 1=generate (defective), 0=keep.
    Returns None if QC run fails.
    """
    try:
        r_dict = _run_quality_checker(motion_135, bone_offsets, device=device)
    except Exception as e:
        print(f'    [warn] QC for mask build failed: {e!r}')
        return None
    r = r_dict.get('_raw')
    if r is None:
        return None

    T = motion_135.shape[0]
    defect_joints = np.zeros((T, 22), dtype=bool)

    failed_names = list(r.failed_checks)
    if include_borderline:
        failed_names = failed_names + list(r.borderline_checks)
    # Checkers whose failure implies root (pelvis + translation) is
    # itself unreliable. When any of these fail, we mark pelvis for the
    # entire sequence so the model is free to rebuild the body pose
    # on a clean root — otherwise the frozen LQ pelvis cascades errors
    # through every other joint's world-space position.
    _ROOT_TAINT_CHECKERS = {
        'joint_jump',            # root_rotation / root_translation jumps
        'first_frame_rotation_velocity',
        'translation_velocity',
        'rotation_velocity',     # often spikes on root
    }
    # 2026-04-27: anatomical defects whose geometric trigger is intermittent
    # but whose underlying joint configuration is wrong on every frame. When
    # any of these fail we OR the full kinematic chain across all T frames,
    # not just the per-frame invalid_mask. This was driven by case 00181
    # (arm_penetration): the shoulder root was misposed throughout, but the
    # line-segment distance only crossed threshold on ~30% of frames.
    _GLOBAL_TAINT_CHAIN_PER_CHECKER = {
        # arm_penetration → bilateral shoulder chain (collar→shoulder→
        # elbow→wrist) on ALL frames. Includes pelvis-spine-neck because
        # arm-torso interactions implicate spine/torso pose too.
        'arm_penetration': [3, 6, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21],
    }
    pelvis_globally_tainted = False
    global_taint_joints: set = set()
    for name in failed_names:
        res = r.all_results.get(name, {})
        im = res.get('invalid_mask', None)
        if im is None:
            defect_joints[:, :] = True
            continue
        im_arr = np.asarray(im)
        if im_arr.ndim != 2 or im_arr.shape[0] == 0:
            defect_joints[:, :] = True
            continue
        tcap = min(T, im_arr.shape[0])
        defect_joints[:tcap] |= im_arr[:tcap, :22].astype(bool)
        if name in _ROOT_TAINT_CHECKERS:
            pelvis_globally_tainted = True
        if name in _GLOBAL_TAINT_CHAIN_PER_CHECKER:
            global_taint_joints.update(_GLOBAL_TAINT_CHAIN_PER_CHECKER[name])

    if pelvis_globally_tainted:
        defect_joints[:, 0] = True
    if global_taint_joints:
        # OR full-frame coverage for the kinematic chain implicated by
        # checkers like arm_penetration (every frame's shoulder is wrong
        # even when only a subset of frames triggers the geometric test).
        for j in global_taint_joints:
            defect_joints[:, j] = True

    # Spatial dilation: SMPL22 kinematic neighbors
    if dilate_spatial:
        neigh = _smpl22_neighbors()
        dj_sp = defect_joints.copy()
        for j in range(22):
            for nb in neigh[j]:
                dj_sp[:, nb] |= defect_joints[:, j]
        defect_joints = dj_sp

    # Temporal dilation
    if dilate_temp > 0:
        dj = defect_joints.copy()
        for s in range(1, dilate_temp + 1):
            dj[s:] |= defect_joints[:-s]
            dj[:-s] |= defect_joints[s:]
        defect_joints = dj

    # Expand to (T, motion_dim) with joint-group broadcast
    out = np.zeros((T, motion_dim), dtype=np.float32)
    out[:, :3] = defect_joints[:, 0:1].astype(np.float32)
    for j in range(22):
        s, e = 3 + j * 6, 3 + (j + 1) * 6
        out[:, s:e] = defect_joints[:, j:j+1].astype(np.float32)
    if motion_dim >= 198:
        for j in range(1, 22):
            ps, pe = 135 + (j - 1) * 3, 135 + j * 3
            out[:, ps:pe] = defect_joints[:, j:j+1].astype(np.float32)

    return out


# ============================================================================
# Data loading
# ============================================================================

def load_motion_135d(
    npz_path: str,
    bone_offsets: Optional[np.ndarray] = None,
    canonical: bool = True,
) -> Optional[np.ndarray]:
    """Load npz -> 135-dim motion (abs transl + rot6d).

    Args:
        npz_path: path to a SMPL-X NPZ file with ``trans``/``poses`` keys.
        bone_offsets: optional (22, 3) SMPL-22 bone offsets used by FK to
            ground-anchor the motion. Required when ``canonical=True``.
        canonical: if True (default) and ``bone_offsets`` is provided, run
            :func:`canonicalize_motion_135d_np` to enforce the training-data
            input distribution (frame-0 ``tx=tz=0`` + all-frame
            ``y_min=0``). Audited 2026-04-27 as the actual training
            distribution; the default test set floats ~14.6cm above ground
            without this step (~4σ OOD vs training).

    Returns:
        (T, 135) motion array, or None on load failure.
    """
    try:
        from hftrainer.datasets.motion.motionhub.transforms.load_smplx import (
            process_transl, process_smplx_pose,
        )
        data = np.load(npz_path, allow_pickle=True)
        if 'motion_135' in data.files:
            motion = data['motion_135'].astype(np.float32)
        else:
            trans_key = 'trans' if 'trans' in data else 'transl'
            abs_trans = data[trans_key].astype(np.float32)
            poses_key = 'poses' if 'poses' in data else 'body_pose'
            poses = data[poses_key].astype(np.float32)
            transl = process_transl(abs_trans, 'abs')
            pose = process_smplx_pose(poses, 'rotation_6d', 'smpl_22')
            motion = np.concatenate([transl, pose], axis=-1).astype(np.float32)

        if canonical and bone_offsets is not None and motion.shape[0] > 0:
            from hftrainer.evaluation.motion.m2m_eval_metrics import (
                canonicalize_motion_135d_np,
            )
            motion = canonicalize_motion_135d_np(motion, bone_offsets)

        return motion
    except Exception:
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

        motion = load_motion_135d(full_path, bone_offsets=bone_offsets)
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

        # Preserve extra paths for transition/target/editing tasks.
        for extra_key in (
            'motion_a_path',
            'motion_b_path',
            'target_motion_path',
            'source_motion_path',
        ):
            if extra_key in item:
                sample[extra_key] = item[extra_key]

        # Real semantic-editing pairs (E16 style_edit) provide a separate
        # source motion. The target ``motion_path`` remains the GT for metrics;
        # ``source_motion_path`` is the motion shown to the model through the
        # editing/reactive channel. This mirrors LoadEditingSource in training.
        source_motion_path = item.get('source_motion_path', '')
        if source_motion_path:
            if os.path.isabs(source_motion_path):
                source_full_path = source_motion_path
            else:
                source_full_path = os.path.join(motion_data_dir, source_motion_path)
                if not os.path.exists(source_full_path):
                    source_full_path = os.path.abspath(source_motion_path)
            if os.path.exists(source_full_path):
                source_motion = load_motion_135d(
                    source_full_path, bone_offsets=bone_offsets)
                if source_motion is not None and source_motion.shape[0] >= min_frames:
                    # Match LoadEditingSource training behavior: source
                    # motion is cropped/padded to the target clip length,
                    # while the target remains the metric reference.
                    if source_motion.shape[0] >= T:
                        source_motion = source_motion[:T]
                    else:
                        pad = np.repeat(
                            source_motion[-1:, :],
                            T - source_motion.shape[0],
                            axis=0,
                        )
                        source_motion = np.concatenate([source_motion, pad], axis=0)
                    sample['source_motion'] = source_motion
                    if convert_to_198 and bone_offsets is not None:
                        sample['source_motion_198'] = motion_135_to_198(
                            source_motion, bone_offsets)

        if convert_to_198 and bone_offsets is not None:
            sample['motion_198'] = motion_135_to_198(motion, bone_offsets)

        samples.append(sample)

    # ------------------------------------------------------------------
    # E13 multi-prompt autoregressive: attach a (caption, num_frames) pool
    # drawn from the FULL datalist so evaluate_sample can chain N segments
    # with per-segment text + per-segment duration. Segment k's length is
    # taken from the matching pool entry's own motion length (capped to
    # max_frames), so different samples produce differently-paced chains.
    # ------------------------------------------------------------------
    if task_id == 'E13' and samples:
        caption_pool: List[str] = []
        length_pool: List[int] = []
        for it in items:
            cap = it.get('caption', it.get('caption_en',
                         it.get('text_caption', ''))) or ''
            if not cap:
                continue
            nf = int(it.get('num_frames', 0) or 0)
            if nf <= 0:
                continue
            # Cap to max_frames so a single pool entry can never exceed the
            # training context (360). Floor to a sane minimum so very short
            # items (<30f) don't produce degenerate segments.
            nf = max(min(nf, max_frames), 30)
            caption_pool.append(cap)
            length_pool.append(nf)
        if caption_pool:
            for k, s in enumerate(samples):
                s['_caption_pool'] = caption_pool
                s['_length_pool'] = length_pool
                s['_caption_base_idx'] = k % len(caption_pool)

    if samples and bone_offsets is not None:
        from hftrainer.evaluation.motion.m2m_eval_metrics import (
            motion135_to_positions_np,
        )
        tx0_list, tz0_list, ymin_list = [], [], []
        for s in samples[: min(50, len(samples))]:
            m = s.get('motion')
            if m is None or m.shape[0] == 0:
                continue
            try:
                pos = motion135_to_positions_np(m, bone_offsets)
                tx0_list.append(float(m[0, 0]))
                tz0_list.append(float(m[0, 2]))
                ymin_list.append(float(pos[..., 1].min()))
            except Exception:
                continue
        if tx0_list:
            tx_arr = np.array(tx0_list); tz_arr = np.array(tz0_list); y_arr = np.array(ymin_list)
            print(
                f'  [canonical-check] task={task_id} n={len(tx_arr)} '
                f'tx0=[{tx_arr.mean():+.4f}±{tx_arr.std():.4f}] '
                f'tz0=[{tz_arr.mean():+.4f}±{tz_arr.std():.4f}] '
                f'y_min=[{y_arr.mean():+.4f}±{y_arr.std():.4f}] '
                f'(target: ~0,0,0)'
            )

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

    # Allow overriding work_dir via env var (e.g. to use a specific checkpoint epoch)
    _override_key = f'_EVAL_WORK_DIR__{model_name}'.upper()
    _work_dir = os.environ.get(_override_key, model_info['work_dir'])
    if _work_dir != model_info['work_dir']:
        print(f'  [override] work_dir: {model_info["work_dir"]} -> {_work_dir}')
    ckpt_path = find_latest_checkpoint(_work_dir)
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


def _evaluate_e13_multiprompt_chain(
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
    num_prompts: int = 3,
    overlap_frames: int = 5,
) -> Tuple[Dict, Optional[np.ndarray]]:
    """E13 multi-prompt autoregressive chaining.

    Generates ``num_prompts`` segments. Each segment k>=1 is conditioned on
    the last ``overlap_frames`` frames of segment k-1 (mask=0 → kept), and
    the rest is generated under segment k's own caption.

    Segment duration is taken directly from the corresponding datalist
    entry's ``num_frames`` field so that different pool entries contribute
    their natural durations (rather than a fixed constant).

    The concatenated output drops the overlap prefix from segments k>=1
    (those frames already live in the previous segment). Segment frame
    ranges and captions are returned via ``_segment_captions`` and
    ``_segment_ranges`` so the dashboard can color-code the body mesh.
    """
    from hftrainer.evaluation.motion.m2m_eval_metrics import compute_all_metrics

    motion_dim = model_info.get('motion_dim', MOTION_DIM_V2)
    rotation_space = model_info.get('rotation_space', 'local')

    # ── Build the (caption, duration) chain from the sample's pools ──
    cap_pool: List[str] = sample.get('_caption_pool') or []
    len_pool: List[int] = sample.get('_length_pool') or []
    base_idx = int(sample.get('_caption_base_idx', 0))
    fallback_cap = sample.get('caption', '') or ''
    fallback_T = int(sample.get('T', 120) or 120)

    captions: List[str] = []
    durations: List[int] = []
    for k in range(num_prompts):
        if cap_pool:
            j = (base_idx + k) % len(cap_pool)
            cap_k = cap_pool[j]
            dur_k = len_pool[j] if j < len(len_pool) else fallback_T
        else:
            cap_k = fallback_cap
            dur_k = fallback_T
        captions.append(cap_k or fallback_cap)
        durations.append(int(dur_k))
    # First segment: pin to the sample's own caption + length (keeps eval
    # reproducible per base sample).
    if fallback_cap:
        captions[0] = fallback_cap
        durations[0] = max(30, fallback_T)

    T_PAD = 360
    ov = max(0, int(overlap_frames))

    # ── Single-segment inference helper (with optional prefix) ──
    def _run_one_segment(seg_caption: str, seg_T: int,
                         prefix_denorm: Optional[np.ndarray]) -> np.ndarray:
        """Generate ``seg_T`` frames under ``seg_caption``. If ``prefix_denorm``
        is given, its last ``ov`` frames are copied into the segment head as
        a condition (mask=0), so the generated body pose continues smoothly.

        ⚠️ Canonical alignment (2026-04-22 fix for "long static between
        segments"): the previous implementation fed the prefix's ABSOLUTE
        world coords to the model. After segment 0 walks 2-3m away from
        origin, segment 1's prefix pelvis trans is e.g. (2.3, 0, 1.1) with
        an arbitrary heading — way outside the v2 training distribution
        (which anchors frame 0 at origin, heading +Z). The model's safest
        fit under that OOD condition is to stay near-static, producing the
        "long idle between segments" the user reported.

        Fix: canonicalize the prefix so its LAST frame (the true seam with
        this segment) sits at origin with heading +Z. Inference runs in
        canonical space, then the segment output is decanonicalized back
        to the same world frame as the prefix so segments stitch without
        discontinuity.
        """
        seg_T_eff = max(ov + 1, min(int(seg_T), T_PAD))  # keep at least 1 gen frame
        local_ov = ov if prefix_denorm is not None else 0

        # Canonical transform for this segment. For segment 0 (no prefix)
        # we still anchor on the synthesized src's frame 0, but since
        # src_135 is all zeros the transform is identity — no-op.
        import torch as _torch
        from hftrainer.pipelines.motion.transition_utils import (
            canonicalize_segment, decanonicalize_segment,
        )

        src_135 = np.zeros((seg_T_eff, 135), dtype=np.float32)
        mask_135 = np.ones((seg_T_eff, 135), dtype=np.float32)  # 1 = generate
        seg_canon_info = None
        if local_ov > 0:
            # Copy prefix (world coords) into the segment head.
            src_135[:local_ov] = prefix_denorm[-local_ov:]
            mask_135[:local_ov] = 0.0
            # Canonicalize the whole segment using the LAST prefix frame
            # as anchor — that's the seam where this segment has to
            # continue from. After this, src_135[local_ov - 1] sits at
            # origin facing +Z, which is the v2 training distribution.
            src_t = _torch.from_numpy(src_135).float()
            src_canon_t, R_canon, offset_canon = canonicalize_segment(
                src_t, anchor_frame=local_ov - 1)
            src_135 = src_canon_t.numpy()
            seg_canon_info = (R_canon, offset_canon)

        if motion_dim == 198:
            src_raw = motion_135_to_198(src_135, bone_offsets)
            pos_mask = np.zeros((seg_T_eff, 63), dtype=np.float32)
            for j in range(21):
                rot_mask_val = mask_135[:, 3 + (j + 1) * 6]
                pos_mask[:, j * 3:(j + 1) * 3] = rot_mask_val[:, None]
            mask_full = np.concatenate([mask_135, pos_mask], axis=-1)
        else:
            src_raw = src_135
            mask_full = mask_135

        if rotation_space == 'global':
            from hftrainer.datasets.motion.motionhub.transforms.fk_utils import (
                local_to_global_rot6d_torch as _l2g,
            )
            _rot_local = torch.from_numpy(
                src_raw[:, 3:135].reshape(seg_T_eff, 22, 6)).float()
            _rot_global = _l2g(_rot_local)
            src_raw = src_raw.copy()
            src_raw[:, 3:135] = _rot_global.reshape(seg_T_eff, 132).numpy()

        motion_norm = bundle.normalize_motion(
            torch.from_numpy(src_raw).float().unsqueeze(0).to(device))
        src_mask_t = torch.from_numpy(mask_full).float().unsqueeze(0).to(device)
        if seg_T_eff < T_PAD:
            pad_len = T_PAD - seg_T_eff
            # Pad motion_norm / src_mask to T_PAD to match the training
            # context (v2 is trained with T=360 always). Training-dist
            # convention for pad frames:
            #   - src_motion[pad] = 0  (F.pad default)
            #   - src_mask[pad]   = 0  ("known / not generated")
            #   - attention on pad is masked by tgt_padding_mask
            #   - loss on pad is zeroed
            # i.e. the model never actually looks at pad. So at inference
            # we just match that: zero-pad both motion_norm and src_mask.
            # The value we replicate for motion_norm is irrelevant because
            # the pipeline's tgt_padding_mask masks out attention over pad,
            # AND (as of 2026-04-24) the pipeline also excludes pad from
            # the replacement-guidance keep_mask so pad is not pinned to
            # any mean-pose value during ODE integration.
            #
            # Historical note: an earlier version replicated motion_norm's
            # last frame into the pad region and used src_mask=0 there,
            # which caused the pipeline's replacement step to lock pad to
            # the training-set mean pose (normalize(zeros)) and bled into
            # the tail of the valid region as "每段尾帧静止". A follow-up
            # tried src_mask=1 (free-generate pad), which sort-of worked
            # but was itself OOD vs training (model never sees mask=1 on
            # pad). The current form matches training exactly.
            motion_norm = torch.nn.functional.pad(
                motion_norm, (0, 0, 0, pad_len), mode='constant', value=0.0)
            src_mask_t = torch.nn.functional.pad(
                src_mask_t, (0, 0, 0, pad_len), mode='constant', value=0.0)

        text_fields = None
        cached = _lookup_caption_embedding(seg_caption) if seg_caption else None
        if cached is not None:
            dev = bundle.device if hasattr(bundle, 'device') else device
            text_fields = {
                'text_vec_raw': cached['text_vec_raw'].to(dev),
                'text_ctxt_raw': cached['text_ctxt_raw'].to(dev),
                'text_ctxt_raw_length': cached['text_ctxt_raw_length'].to(dev),
            }
        elif seg_caption:
            try:
                text_out = bundle.encode_text([seg_caption])
                text_fields = {
                    'text_vec_raw': text_out['vtxt_input'],
                    'text_ctxt_raw': text_out['ctxt_input'],
                    'text_ctxt_raw_length': text_out['ctxt_length'],
                }
            except Exception as e:
                print(f'      [E13] caption encode failed: {e!r}')

        src_motion_norm = motion_norm * (1 - src_mask_t)
        batch = {
            'src_motion': src_motion_norm,
            'src_mask': src_mask_t,
            'src_length': [seg_T_eff],
            'tgt_length': [seg_T_eff],
            'clean_motion': motion_norm.clone(),
        }
        if text_fields is not None:
            batch.update(text_fields)

        pipeline.replacement_guidance = replacement_guidance
        pipeline.text_guidance_scale = text_guidance_scale
        pipeline.num_steps = num_steps
        pipeline.sdedit_tau = 0.0

        with torch.no_grad():
            out = pipeline(batch)
        sampled_norm = out['latent']
        output_denorm_full = bundle.denormalize_motion(sampled_norm)[0].cpu().numpy()
        output_denorm = output_denorm_full[:seg_T_eff]

        if motion_dim == 198:
            out_135 = np.concatenate([
                output_denorm[:, :3],
                output_denorm[:, 3:135],
            ], axis=-1)
        else:
            out_135 = output_denorm[:, :135]

        if rotation_space == 'global':
            from hftrainer.datasets.motion.motionhub.transforms.fk_utils import (
                global_to_local_rot6d_torch as _g2l,
            )
            _rg = torch.from_numpy(
                out_135[:, 3:135].reshape(seg_T_eff, 22, 6)).float()
            _rl = _g2l(_rg)
            out_135[:, 3:135] = _rl.reshape(seg_T_eff, 132).numpy()

        # Decanonicalize the segment output back to the world frame of
        # this segment's prefix. For k=0 (no prefix), seg_canon_info is
        # None and the output stays in canonical space (that IS world for
        # segment 0 since there's no prior frame to align with).
        #
        # IMPORTANT: by this point we've already converted the output back
        # to LOCAL rot6d (line ~1385, the global→local branch above). So
        # decanon must treat the data as local — otherwise it would
        # yaw-rotate body joints that are stored in parent-relative form,
        # producing a double-rotation artifact identical to the E14 canon
        # bug. Hard-code 'local' regardless of the model's rotation_space.
        if seg_canon_info is not None:
            R_canon, offset_canon = seg_canon_info
            out_t = _torch.from_numpy(out_135).float()
            out_world_t = decanonicalize_segment(
                out_t, R_canon, offset_canon, rotation_space='local')
            out_135 = out_world_t.numpy().astype(np.float32)

        return out_135.astype(np.float32)

    # ── Chain segments ──
    t_start = time.time()
    segments_135: List[np.ndarray] = []
    segment_ranges: List[List[int]] = []

    cur_end = 0
    prev_segment: Optional[np.ndarray] = None
    for k, (cap, dur) in enumerate(zip(captions, durations)):
        seg = _run_one_segment(cap, dur, prefix_denorm=prev_segment)
        # For k>=1 drop the first `ov` frames (they are condition = prev tail).
        if k == 0:
            emitted = seg
        else:
            emitted = seg[ov:] if ov > 0 else seg
        seg_start = cur_end
        seg_end = cur_end + emitted.shape[0]
        segment_ranges.append([seg_start, seg_end])
        cur_end = seg_end
        segments_135.append(emitted)
        prev_segment = seg  # full segment (incl. overlap) used as next prefix

    output_135 = np.concatenate(segments_135, axis=0).astype(np.float32)
    elapsed = time.time() - t_start

    # ── Metrics (no proper GT for multi-prompt chain) ──
    T_out = output_135.shape[0]
    gt_arr = sample.get('motion')
    if gt_arr is None:
        gt_135 = output_135.copy()
    else:
        # Loose GT: sample's own motion, truncated or zero-padded to T_out.
        gt_135 = np.asarray(gt_arr, dtype=np.float32)
        if gt_135.shape[0] >= T_out:
            gt_135 = gt_135[:T_out]
        else:
            pad = np.zeros((T_out - gt_135.shape[0], 135), dtype=np.float32)
            gt_135 = np.concatenate([gt_135, pad], axis=0)

    # Mask: everything is "generated" except the prefix anchors. Build a
    # per-frame scalar mask (1 = generated, 0 = kept as prefix) and expand
    # to (T_out, 135). The first segment is all generated; for k>=1 the
    # first `ov` frames of that segment were prefix (condition) — but we
    # already DROPPED them from `output_135`, so every frame in the final
    # sequence is "generated" from the model's perspective. Use mask=1.
    metrics_mask = np.ones((T_out, 135), dtype=np.float32)
    try:
        # output_135 has already been converted back to LOCAL rotation space
        # at the end of each segment (see _run_one_segment), so metrics FK
        # always uses the local path regardless of the model's own space.
        metrics = compute_all_metrics(
            output_135, gt_135, metrics_mask, bone_offsets,
            rotation_space='local',
            fps=float(sample.get('fps', 30)),
        )
    except Exception as e:
        print(f'      [E13] metrics failed: {e!r}')
        metrics = {}

    metrics['inference_time'] = elapsed
    metrics['_segment_captions'] = list(captions)
    metrics['_segment_ranges'] = segment_ranges
    metrics['_segment_overlap_frames'] = ov
    metrics['_num_frames'] = T_out

    return metrics, output_135


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
        build_start_pose_prepend_mask,
        compute_transition_length,
        extract_ee_constraints_from_gt,
        detect_foot_contact_frames,
    )

    motion_dim = model_info.get('motion_dim', MOTION_DIM_V2)
    rotation_space = model_info.get('rotation_space', 'local')
    constraint_info = None  # For E4 end-effector; set in mask building
    keyframe_indices = None  # populated for E3 adaptive so the dashboard can
                             # render non-uniform condition frames; persisted
                             # via metrics['_keyframe_indices'] and the NPZ
                             # 'keyframe_indices' array.

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

    source_motion_raw = None
    if 'source_motion' in sample:
        source_135 = sample['source_motion'][:T]
        if motion_dim == 198:
            source_motion_raw = sample.get('source_motion_198')
            if source_motion_raw is None:
                source_motion_raw = motion_135_to_198(source_135, bone_offsets)
            source_motion_raw = source_motion_raw[:T]
        else:
            source_motion_raw = source_135

    # ---- Special handling for E8-B/C/D: loop completion ----
    setting_kwargs = task.settings[setting_name].mask_kwargs
    gt_motion_135 = motion_135  # save original GT for metrics

    # ---- Special handling for E13: multi-prompt autoregressive chaining ----
    # Each sample produces N segments, each driven by a different caption.
    # Segment k>=1 overlaps with the last `overlap_frames` frames of segment
    # k-1 (those frames act as condition so the body pose is continuous).
    if task.task_id == 'E13' and model_info.get('has_caption', False):
        num_prompts = int(setting_kwargs.get('num_prompts', 3))
        overlap_frames = int(setting_kwargs.get('overlap_frames', 5))
        return _evaluate_e13_multiprompt_chain(
            bundle, pipeline, sample, task, setting_name, model_info,
            bone_offsets, device,
            replacement_guidance=replacement_guidance,
            text_guidance_scale=text_guidance_scale,
            num_steps=num_steps,
            num_prompts=num_prompts,
            overlap_frames=overlap_frames,
        )

    if task.task_id == 'E8' and '_loop_append' in setting_kwargs:
        # ------------------------------------------------------------------
        # E8-D redesign (2026-04-22, direction A): "loop completion" that
        # respects the model's 360-frame training ceiling AND avoids the
        # condition/transition position-jump artefact.
        #
        # Previous buggy behaviour:
        #   * `motion = [GT(T_gt), zeros(89), GT[0]]` was shoved into the
        #     pipeline at full length. When T_gt+90 > 360, nothing trimmed
        #     it — pipeline silently received >360 frames, triggering
        #     model OOD / truncation.
        #   * No canonical alignment — GT's world pelvis trans / heading is
        #     OOD relative to training (anchor-at-origin, heading +Z).
        #   * Target anchor = GT[0] sat far from GT_end in world coords →
        #     generated transition had to "teleport back to origin",
        #     producing visible position jumps at the GT→transition boundary.
        #
        # Direction A fix:
        #   (i)   Clip condition to at most (360 - N_append) frames from GT
        #         tail. Dropped GT prefix surfaces later as a gray context
        #         on the dashboard (see /api/source_motions/E8).
        #   (ii)  Canonicalize the built segment using the NETWORK INPUT's
        #         frame 0 as anchor. This matches the training distribution:
        #         every clip fed to the model starts at origin facing +Z,
        #         regardless of where that frame came from in the original
        #         source motion.
        #   (iii) Leave the target anchor as GT[0] in world coords; after
        #         canonicalization it sits wherever the loop has to close
        #         to. The model generates ``N_append`` frames to reach it.
        #   (iv)  Output is decanonicalized back to world via the shared
        #         `_transition_canon_info` path used by E14/E15/E16.
        # ------------------------------------------------------------------
        N_append = setting_kwargs['_loop_append']
        # ── Dynamic _loop_append (2026-04-23) ─────────────────────────
        # If `_loop_append == 'auto'` or a negative int, derive it from the
        # GT first↔last pelvis distance (same speed-based rule as E14/15/16).
        # The fixed 90-frame default is still respected for positive ints
        # so existing configs don't change behaviour.
        if isinstance(N_append, str) and N_append == 'auto':
            use_dynamic = True
        elif isinstance(N_append, int) and N_append <= 0:
            use_dynamic = True
        else:
            use_dynamic = False
        if use_dynamic:
            from hftrainer.evaluation.motion.m2m_eval_metrics import (
                motion135_to_positions_np as _fk_np,
            )
            pos_first = _fk_np(motion_135[0:1], bone_offsets)[0, 0]
            pos_last = _fk_np(motion_135[-1:], bone_offsets)[0, 0]
            joints_last = _fk_np(motion_135[-1:], bone_offsets)[0]   # (22,3)
            joints_first = _fk_np(motion_135[0:1], bone_offsets)[0]  # (22,3)
            N_append = compute_transition_length(
                pos_last, pos_first,
                speed_per_frame=float(setting_kwargs.get(
                    '_transition_speed', 0.015)),
                min_frames=int(setting_kwargs.get('_transition_min', 30)),
                max_frames=int(setting_kwargs.get('_transition_max', 120)),
                joints_a_end=joints_last, joints_b_start=joints_first,
                pose_speed_per_frame=float(setting_kwargs.get(
                    '_pose_speed', 0.015)),
                motion_a_end_135=motion_135[-1],
                motion_b_start_135=motion_135[0],
                joint_angle_speed_per_frame=float(setting_kwargs.get(
                    '_joint_angle_speed', 0.20)),
            )
            print(f'    [E8-D] dynamic N_append={N_append} '
                  f'(dist={np.linalg.norm(pos_last - pos_first):.2f}m, '
                  f'pose_mean={np.linalg.norm(joints_last - joints_first, axis=-1).mean():.2f}m)')
            # Persist resolved int value so downstream metrics code can use
            # setting_kwargs['_loop_append'] (see boundary_accel_jump_loop
            # computation below around line ~2740).
            setting_kwargs = dict(setting_kwargs)
            setting_kwargs['_loop_append'] = N_append
        T_gt_full = T  # original GT length
        T_PAD_MAX = 360  # training ceiling

        # ── N_cond_gt ablation (2026-04-23 v2) ──────────────────────────
        # Caller may request a specific GT-tail length as condition (mirrors
        # the E14 N_cond axis). Choices:
        #   '_cond_gt_tail_frames' = int → fixed tail length.
        #   '_cond_gt_policy' = 'adaptive' → compute_cond_length rule on
        #                                    GT tail.
        #   neither → keep as much GT as fits under the 360 budget (old).
        cond_gt_tail_req = setting_kwargs.get('_cond_gt_tail_frames', None)
        cond_gt_policy = setting_kwargs.get('_cond_gt_policy', None)
        if cond_gt_policy == 'adaptive':
            from hftrainer.evaluation.motion.m2m_eval_tasks import (
                compute_cond_length,
            )
            T_gt_eff_req = compute_cond_length(
                motion_135, T_src=T_gt_full, N_transition=N_append,
                side='tail')
        elif cond_gt_tail_req is not None:
            T_gt_eff_req = int(cond_gt_tail_req)
        else:
            T_gt_eff_req = T_gt_full

        # Clip GT condition so (T_gt_eff + N_append) <= T_PAD_MAX.
        T_gt_eff = max(1, min(T_gt_eff_req, T_gt_full, T_PAD_MAX - N_append))
        # Dropped prefix (shown as gray context in the dashboard):
        n_dropped_prefix = T_gt_full - T_gt_eff

        gt_tail_135 = motion_135[-T_gt_eff:]          # (T_gt_eff, 135) world
        first_frame_135 = motion_135[0:1]             # (1, 135) world — loop target
        pad_frames_135 = np.zeros((N_append - 1, 135), dtype=np.float32)

        # World-space context segment: [GT_tail | pad | GT[0]]
        segment_world = np.concatenate(
            [gt_tail_135, pad_frames_135, first_frame_135], axis=0)
        T_total = segment_world.shape[0]            # = T_gt_eff + N_append

        # Canonicalize with anchor = network-input frame 0 (the start of the
        # fed GT tail). The canonical frame should depend only on what is sent
        # into the network, not on this segment's index in the original motion.
        import torch as _torch
        from hftrainer.pipelines.motion.transition_utils import (
            canonicalize_segment,
        )
        segment_world_t = _torch.from_numpy(segment_world).float()
        canon_segment_t, R_canon, offset_canon = canonicalize_segment(
            segment_world_t, anchor_frame=0)
        motion_135 = canon_segment_t.numpy()
        T = T_total
        gt_motion_135 = motion_135                  # looped-anchor GT in canon

        # Store canonicalization info so the shared decanonicalize branch
        # (below, near line ~1660) maps output back to world coords.
        # motion_a_full / motion_b_world_full are the dashboard context:
        #   motion_a_full = GT full (world coords) → gray prefix + condition
        #   motion_b_world_full = None (E8 has no "B" motion)
        _transition_canon_info = dict(
            R_canon=R_canon, offset_canon=offset_canon,
            motion_a_full=sample['motion'],           # original GT (world)
            motion_b_world_full=None,
            N_cond=T_gt_eff,
            N_transition=N_append,
            n_dropped_prefix=n_dropped_prefix,
        )

        # Recompute 198-dim channels from canonicalized 135-dim motion, since
        # rot6d (and hence position channel) changed after the rigid transform.
        if motion_dim == 198:
            motion_raw = motion_135_to_198(motion_135, bone_offsets)
        else:
            motion_raw = motion_135

        # Build loop completion mask: [0...0 | 1...1 | 0] with
        #   condition region = T_gt_eff frames (GT tail in canonical space)
        #   generate region  = N_append - 1 frames
        #   last 1 frame     = 0 (target = canonicalized GT[0])
        mask = build_loop_completion_mask(
            T_total, 135, T_gt=T_gt_eff, N_append=N_append)

    # ---- Special handling for E8-A: pure loop (first=last anchor) ----
    # Correct semantics (per user 2026-04-21): E8 loop animation asks the
    # model to generate a sequence that STARTS and ENDS on the same pose.
    # We take the GT's first frame as BOTH the first and the last frame
    # anchor, and let the model fill all intermediate frames.
    #
    # Previous implementation incorrectly anchored frame T-1 to motion[T-1]
    # (GT's own last frame), which is typically a different pose from the
    # first frame — so the "loop" was never actually a loop.
    elif task.task_id == 'E8' and '_loop_append' not in setting_kwargs:
        # Replace motion's last frame with its own first frame so the
        # condition anchors at both ends produce a true loop.
        first_frame_135 = motion_135[0:1]
        motion_135 = motion_135.copy()
        motion_135[-1:] = first_frame_135
        gt_motion_135 = motion_135  # also sync GT so metrics use the looped anchor
        if motion_dim == 198:
            first_frame_198 = motion_raw[0:1]
            motion_raw = motion_raw.copy()
            motion_raw[-1:] = first_frame_198
        else:
            motion_raw = motion_135
        # Build the loop mask: frames 0 and T-1 are condition (both now
        # anchored to the first-frame pose); frames 1..T-2 are generated.
        mask = task.build_mask(T, 135, setting_name)

    # ---- Special handling for E14: transition stitching ----
    elif task.task_id == 'E14' and '_use_transition_data' in setting_kwargs:
        # 2026-04-23 redesign: _cond_frames is only used when _context_policy
        # is absent (legacy); new runs drive N_cond_a / N_cond_b from the
        # context policy so the 360-frame budget is used efficiently.
        context_policy = setting_kwargs.get('_context_policy', None)
        legacy_N_cond = setting_kwargs.get('_cond_frames', 15)

        # Load motion_a and motion_b
        motion_a_path = sample.get('motion_a_path', '')
        motion_b_path = sample.get('motion_b_path', '')
        # Resolve paths (2026-04-24): support both legacy relative paths
        # (resolved against MOTION_DATA_DIR) and new repo-relative paths
        # (data/hymotion_data/...). Also handle abs paths.
        def _resolve_motion_path(p):
            if not p:
                return p
            if os.path.isabs(p):
                return p
            # Try as-is first (repo-root-relative, e.g., data/hymotion_data/...)
            if os.path.exists(p):
                return p
            # Try MOTION_DATA_DIR prefix (legacy)
            legacy = os.path.join(MOTION_DATA_DIR, p)
            if os.path.exists(legacy):
                return legacy
            return p  # let downstream fail gracefully
        motion_a_path = _resolve_motion_path(motion_a_path)
        motion_b_path = _resolve_motion_path(motion_b_path)

        motion_a = load_motion_135d(motion_a_path, bone_offsets=bone_offsets)
        motion_b = load_motion_135d(motion_b_path, bone_offsets=bone_offsets)
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
        )

        motion_a_t = _torch.from_numpy(motion_a).float()
        motion_b_t = _torch.from_numpy(motion_b).float()

        # ── Step 1: decide placement strategy for motion B ─────────────
        # _placement ∈ {'overlap','velocity','forward'} (default 'forward'
        # for backward compatibility). 'overlap' and 'velocity' are the
        # new 2026-04-23 v5 options requested by user — see E14 settings
        # L (overlap) and M (velocity) in m2m_eval_tasks.py.
        placement = setting_kwargs.get('_placement', 'forward')
        forward_step = float(setting_kwargs.get('_forward_step', 1.0))
        yaw_offset_deg = float(setting_kwargs.get('_yaw_offset_deg', 0.0))

        # ── Step 2: compute N_transition using an overlap-estimate for B ─
        # For 'velocity' mode, N_transition depends on where B ends up,
        # which depends on N_transition itself (circular). We break the
        # cycle by first estimating N_transition under 'overlap' placement
        # (B_xz = A_end_xz), then re-placing with 'velocity' if needed.
        motion_b_world_overlap = _place_b_custom(
            motion_a, motion_b, placement='overlap',
            N_transition=1, yaw_offset_deg=yaw_offset_deg,
            bone_offsets=bone_offsets)

        from hftrainer.evaluation.motion.m2m_eval_metrics import motion135_to_positions_np
        pos_a = motion135_to_positions_np(motion_a, bone_offsets)
        pos_b_overlap = motion135_to_positions_np(motion_b_world_overlap, bone_offsets)
        pos_a_end = pos_a[-1, 0]
        pos_b_start = pos_b_overlap[0, 0]
        joints_a_end = pos_a[-1]
        joints_b_start = pos_b_overlap[0]
        if '_transition_frames' in setting_kwargs:
            N_transition = int(setting_kwargs['_transition_frames'])
        else:
            N_transition = compute_transition_length(
                pos_a_end, pos_b_start,
                speed_per_frame=float(setting_kwargs.get(
                    '_transition_speed', 0.015)),
                min_frames=int(setting_kwargs.get('_transition_min', 30)),
                max_frames=int(setting_kwargs.get('_transition_max', 120)),
                joints_a_end=joints_a_end, joints_b_start=joints_b_start,
                pose_speed_per_frame=float(setting_kwargs.get(
                    '_pose_speed', 0.015)),
                motion_a_end_135=motion_a[-1],
                motion_b_start_135=motion_b_world_overlap[0],
                joint_angle_speed_per_frame=float(setting_kwargs.get(
                    '_joint_angle_speed', 0.20)),
            )

        # ── Step 3: actually place B with the chosen strategy ────────────
        motion_b_world_np = _place_b_custom(
            motion_a, motion_b, placement=placement,
            N_transition=N_transition,
            forward_step=forward_step,
            yaw_offset_deg=yaw_offset_deg,
            bone_offsets=bone_offsets)
        pos_b_world = motion135_to_positions_np(motion_b_world_np, bone_offsets)
        pos_b_start = pos_b_world[0, 0]

        # Log placement decision
        _a_vel = _estimate_a_end_velocity(motion_a)
        _a_speed = float(np.linalg.norm(_a_vel[[0, 2]]))  # xz speed m/frame
        _dist = float(np.linalg.norm(pos_a_end[[0, 2]] - pos_b_start[[0, 2]]))
        print(f'    [E14] placement={placement} '
              f'A_end_speed={_a_speed:.3f}m/frame A→B_dist={_dist:.2f}m '
              f'N_transition={N_transition}')

        # ── Context policy (2026-04-23 v2) ─────────────────────────────
        # N_cond ablation, replacing the old minimal/balanced/max budget
        # splitter. Based on 50-sample E14 analysis:
        #   - Training distribution caps cond at ~5 frames per side.
        #   - Long cond (60+) causes collapse on 3/50 samples even with
        #     clean input (OOD, not quality).
        # New policies:
        #   'fixed'    → N_cond_a / N_cond_b from setting kwargs
        #                (_n_cond_a_frames / _n_cond_b_frames).
        #   'adaptive' → compute_cond_length rule per side (base=5, quality-
        #                and horizon-aware, clamped [3,10]).
        #   (legacy minimal/balanced/max kept as fallback for old DBs.)
        MAX_FRAMES = 360
        len_a = int(motion_a.shape[0])
        len_b = int(motion_b_world_np.shape[0])
        if context_policy == 'fixed':
            want_a = int(setting_kwargs.get('_n_cond_a_frames', 5))
            want_b = int(setting_kwargs.get('_n_cond_b_frames', 5))
            N_cond_a = min(want_a, len_a)
            N_cond_b = min(want_b, len_b)
        elif context_policy == 'adaptive':
            from hftrainer.evaluation.motion.m2m_eval_tasks import (
                compute_cond_length,
            )
            N_cond_a = compute_cond_length(
                motion_a, T_src=len_a, N_transition=N_transition, side='tail')
            N_cond_b = compute_cond_length(
                motion_b_world_np, T_src=len_b, N_transition=N_transition,
                side='head')
        elif context_policy == 'minimal':
            N_cond_a = min(5, len_a)
            N_cond_b = min(5, len_b)
        elif context_policy == 'balanced':
            budget = max(0, MAX_FRAMES - N_transition)
            half = budget // 2
            N_cond_a = min(half, len_a)
            N_cond_b = min(half, len_b)
            # Ensure at least 5 frames each side even if one of them is very
            # short (so the model always has some pose history).
            N_cond_a = max(5, N_cond_a) if len_a >= 5 else len_a
            N_cond_b = max(5, N_cond_b) if len_b >= 5 else len_b
        elif context_policy == 'max':
            budget = max(0, MAX_FRAMES - N_transition)
            # Greedy: give A whatever fits, then B the remainder.
            N_cond_a = min(budget, len_a)
            N_cond_b = min(max(0, budget - N_cond_a), len_b)
            # If B got starved, try to rebalance so B has ≥15 frames.
            if N_cond_b < 15 and len_b >= 15 and N_cond_a > 15:
                give_b = min(len_b, budget // 2)
                N_cond_a = min(budget - give_b, len_a)
                N_cond_b = give_b
            # Final floor: at least 5 each if possible.
            N_cond_a = max(5, N_cond_a) if len_a >= 5 else len_a
            N_cond_b = max(5, N_cond_b) if len_b >= 5 else len_b
        else:
            # Legacy symmetric 5/15/30 behaviour.
            N_cond_a = min(legacy_N_cond, len_a)
            N_cond_b = min(legacy_N_cond, len_b)

        # Final sanity: keep total ≤ MAX_FRAMES. If overflow, shrink largest.
        total = N_cond_a + N_transition + N_cond_b
        if total > MAX_FRAMES:
            overflow = total - MAX_FRAMES
            if N_cond_a >= N_cond_b:
                N_cond_a = max(5, N_cond_a - overflow)
            else:
                N_cond_b = max(5, N_cond_b - overflow)
            # If still over (pathological: huge N_transition), clip N_trans.
            total = N_cond_a + N_transition + N_cond_b
            if total > MAX_FRAMES:
                N_transition = max(15, MAX_FRAMES - N_cond_a - N_cond_b)

        # Take tail of A and head of (world-placed) B.
        a_tail = motion_a[-N_cond_a:]                # (N_cond_a, 135)
        b_head = motion_b_world_np[:N_cond_b]        # (N_cond_b, 135)
        print(f'    [E14] policy={context_policy} '
              f'N_cond_a={N_cond_a} N_transition={N_transition} '
              f'N_cond_b={N_cond_b} total={N_cond_a + N_transition + N_cond_b} '
              f'(dist={np.linalg.norm(pos_a_end - pos_b_start):.2f}m)')

        # Build world-space transition context: [A_tail | pad | B_head_world]
        transition_pad = np.zeros((N_transition, 135), dtype=np.float32)
        world_segment = np.concatenate([a_tail, transition_pad, b_head], axis=0)
        T = world_segment.shape[0]

        # Step 2 (2026-04-26): canonicalize so the FIRST frame of the
        # network input (= A_tail[0]) sits at origin facing +Z. This
        # exactly matches the training distribution where every clip is
        # canonicalized at frame 0. The previous "boundary anchor" (last
        # frame of A_tail at origin) put A_tail[0..N_cond_a-2] in the
        # -Z half-plane, which is OOD: the v3 mask sampler trains on
        # clips where every conditioned frame sits in +Z. Empirically
        # this caused the cond→gen boundary to receive less attention
        # than expected because the model "saw" a partially-rotated
        # input. Anchoring at frame 0 instead of N_cond_a-1 brings the
        # eval segment into the same canonical pose distribution as
        # training.
        anchor_idx = 0
        world_segment_t = _torch.from_numpy(world_segment).float()
        # ROTATION SPACE: world_segment is built from raw NPZ 135-d tensors,
        # i.e. LOCAL rot6d (parent-relative). local→global conversion happens
        # LATER at line ~2450, just before feeding the model. So canon must
        # treat the body joints as local (no yaw-rotation on body, only on
        # pelvis). Hard-code 'local' regardless of model's rotation_space.
        # The matching decanon at line ~3112 runs on MODEL-space output,
        # which IS global for global models, and there rotation_space=
        # rotation_space is correct.
        canon_segment_t, R_canon, offset_canon = canonicalize_segment(
            world_segment_t, anchor_frame=anchor_idx,
            rotation_space='local')
        motion_135 = canon_segment_t.numpy()
        gt_motion_135 = motion_135  # no GT for transition region

        # Stash the canonicalization transform so inference output can be
        # mapped back to world coordinates before metrics / viz.
        _transition_canon_info = dict(
            R_canon=R_canon, offset_canon=offset_canon,
            motion_a_full=motion_a, motion_b_world_full=motion_b_world_np,
            N_cond=N_cond_a,  # legacy key: upstream uses this as "N_cond_a"
            N_cond_a=N_cond_a, N_cond_b=N_cond_b,
            N_transition=N_transition,
        )

        if motion_dim == 198:
            motion_raw = motion_135_to_198(motion_135, bone_offsets)
        else:
            motion_raw = motion_135

        mask = build_transition_mask(
            T, 135, N_cond_a=N_cond_a, N_transition=N_transition,
            N_cond_b=N_cond_b)

    # ---- Special handling for E15 (2026-04-21 redefinition): prepend a
    # transition from a target start pose P into an existing motion A ----
    elif task.task_id == 'E15' and '_use_start_pose' in setting_kwargs:
        # 2026-04-23 redesign: mirror E14's place_b_after_a + canonicalize
        # pattern. Previous implementation had three bugs:
        #   1. P's Y was raw target[0].trans.Y (target motion's world Y at
        #      its own clip origin), NOT aligned to A[0]'s Y. Result:
        #      when target was standing (Y≈1.0m) and A started crouching
        #      (Y≈0.74m), P floated 0.26m above A — a visible height
        #      "teleport" at the P→A boundary.
        #   2. P's yaw was raw target[0] yaw (unrelated to A's heading).
        #      canonicalize(anchor=0=P) then rotated the WHOLE segment so
        #      that P's yaw → +Z. A, riding along, ended up facing
        #      wherever (P.yaw - A.yaw) mapped it — sideways instead of
        #      forward. In canonical space A[0] could land at (0.99, *,
        #      -0.08) instead of (0, *, 1.0).
        #   3. P's XZ was set to A[0] - backward_dir*1.0 in A's ORIGINAL
        #      world frame; after canonicalize(P) this didn't yield a
        #      clean "P at origin, A ahead at +Z, 1m apart" layout.
        #
        # Correct layout:
        #   Step 1: canonicalize target[0] as a 1-frame clip → P at
        #           (0, target_Y, 0), facing +Z.
        #   Step 2: place_b_after_a(P, A, forward_step) → A[0] at
        #           (0, A_own_Y, +forward_step), A's heading follows P's
        #           (=+Z) while preserving A's own Y profile.
        #   Step 3: concatenate [P | pad | A_placed]; final canonicalize
        #           is near-identity because step 1+2 already produced a
        #           canonical-style segment. Kept to preserve the
        #           _transition_canon_info roundtrip convention.
        target_path = sample.get('target_motion_path', '')
        # Resolve (same logic as E14, 2026-04-24)
        if not os.path.isabs(target_path) and not os.path.exists(target_path):
            legacy = os.path.join(MOTION_DATA_DIR, target_path)
            if os.path.exists(legacy):
                target_path = legacy
        target_motion = load_motion_135d(target_path)
        if target_motion is None:
            return {}, None

        import torch as _torch
        from hftrainer.pipelines.motion.transition_utils import (
            canonicalize_segment, place_b_after_a,
        )

        motion_a_full = motion_135  # (len_A, 135), world coords from dataset
        if motion_a_full.shape[0] < 2:
            return {}, None

        from hftrainer.evaluation.motion.m2m_eval_metrics import motion135_to_positions_np

        # ── Step 1: canonical single-frame P ────────────────────────────
        # P = target[0]. canonicalize_segment snaps P's pelvis XZ to (0,0)
        # and rotates so P's yaw = +Z. Y (pelvis height) is preserved —
        # that's P's own pose height, independent of A.
        P_single = target_motion[0:1].copy()
        # P_single is raw local rot6d (loaded from dataset).
        P_canon_t, _Rp, _Op = canonicalize_segment(
            _torch.from_numpy(P_single).float(), anchor_frame=0,
            rotation_space='local')
        P_canon = P_canon_t.numpy()  # (1, 135), P's pelvis XZ=(0,0), yaw=+Z

        # ── Step 2: place A at P's location (overlap) ─────────────────────
        # User requested E15 to be "in-place transition": P and A[0] both
        # at world XZ = (0, 0), differing only in Y (pelvis height).
        # Rationale:
        #   - P is a static target pose, no locomotion implied.
        #   - Forcing B 1m in front of P (old behavior) made the model
        #     always generate a walk-in, but E15's semantic is "go from
        #     this start pose to the first pose of A" — a postural
        #     transition, not a locomotion transition.
        #   - Placing A at P's same XZ lets the model focus on the
        #     posture change (e.g. T-pose → crouch) without also having
        #     to travel a meter.
        #
        # Result layout:
        #   frame 0 (P):   pelvis = (0, P.Y, 0)
        #   frame N (A0):  pelvis = (0, A0.Y, 0)    ← same XZ as P
        #   frame N+i:     pelvis = A[i] rotated + translated
        yaw_offset_deg = float(setting_kwargs.get('_yaw_offset_deg', 0.0))
        motion_a_placed = _place_b_custom(
            P_canon,
            motion_a_full,
            placement='overlap',
            N_transition=1,  # unused by overlap
            yaw_offset_deg=yaw_offset_deg,
            # E15 explicitly wants P.Y ≠ A[0].Y (T-pose vs crouch). Skip
            # foot-floor alignment here so the postural Y-gap is preserved.
            y_align='preserve_b',
        )

        # ── Step 3: adaptive N_transition (root dist + pose diff) ───────
        P_joints = motion135_to_positions_np(P_canon, bone_offsets)[0]       # (22,3)
        A0_joints = motion135_to_positions_np(
            motion_a_placed[0:1], bone_offsets)[0]                            # (22,3)
        P_pelvis_xyz = P_joints[0]
        A0_pelvis_xyz = A0_joints[0]
        if '_prepend_N' in setting_kwargs:
            N_transition = int(setting_kwargs['_prepend_N'])
        else:
            N_transition = compute_transition_length(
                P_pelvis_xyz, A0_pelvis_xyz,
                speed_per_frame=float(setting_kwargs.get('_transition_speed', 0.015)),
                min_frames=int(setting_kwargs.get('_transition_min', 15)),
                max_frames=int(setting_kwargs.get('_transition_max', 90)),
                joints_a_end=P_joints, joints_b_start=A0_joints,
                pose_speed_per_frame=float(setting_kwargs.get('_pose_speed', 0.015)),
                motion_a_end_135=P_canon[0],
                motion_b_start_135=motion_a_placed[0],
                joint_angle_speed_per_frame=float(setting_kwargs.get('_joint_angle_speed', 0.20)),
            )
        print(f'    [E15] speed={setting_kwargs.get("_transition_speed", 0.015):.4f} '
              f'N_prepend={N_transition} '
              f'(dist={np.linalg.norm(P_pelvis_xyz - A0_pelvis_xyz):.2f}m, '
              f'len_A={motion_a_placed.shape[0]}, '
              f'P_Y={P_pelvis_xyz[1]:.2f}m A[0]_Y={A0_pelvis_xyz[1]:.2f}m)')

        # ── Step 3b: N_cond_A ablation (2026-04-23 v4, CORRECTED) ──────
        # Only truncate the A portion FED TO THE MODEL (_model_a). The full
        # motion_a_placed is preserved in canon_info.motion_b_world_full so
        # the dashboard can decanon + display all of A. This matches E14's
        # pattern: E14 sends only `N_cond_b` tail frames of motion_b to the
        # model but stores `motion_b_world_full` for visualization.
        #
        # Rationale: E15's task definition says "output = P + transition +
        # full A". But the *model input* can be any prefix of A. Short A in
        # the model input ∈ training-distribution (training uses ≤5 cond
        # frames per side); long A puts it OOD. Ablating this axis tells us
        # whether the model benefits from extra A-frames or not.
        motion_a_placed_full = motion_a_placed  # preserved for dashboard
        n_cond_a_policy = setting_kwargs.get('_n_cond_a_policy', None)
        n_cond_a_frames = setting_kwargs.get('_n_cond_a_frames', None)
        if n_cond_a_policy == 'adaptive':
            from hftrainer.evaluation.motion.m2m_eval_tasks import (
                compute_cond_length,
            )
            model_K = compute_cond_length(
                motion_a_placed_full,
                T_src=int(motion_a_placed_full.shape[0]),
                N_transition=N_transition,
                side='head',
            )
            motion_a_model = motion_a_placed_full[:model_K]
            print(f'    [E15] N_cond_A adaptive={model_K} (full A={len(motion_a_placed_full)} preserved for viz)')
        elif n_cond_a_frames is not None:
            model_K = int(min(int(n_cond_a_frames), motion_a_placed_full.shape[0]))
            motion_a_model = motion_a_placed_full[:model_K]
            print(f'    [E15] N_cond_A fixed={model_K} (full A={len(motion_a_placed_full)} preserved for viz)')
        else:
            motion_a_model = motion_a_placed_full  # no truncation

        motion_a_placed = motion_a_model  # used below for world_segment

        # ── Step 4: 360-frame ceiling guard (truncate A's tail) ─────────
        T_total = N_transition + motion_a_placed.shape[0]
        MAX_FRAMES = 360
        if T_total > MAX_FRAMES:
            keep_A = MAX_FRAMES - N_transition
            if keep_A <= 1:
                print(f'    [E15] SKIP sample: N_transition={N_transition} '
                      f'leaves no room for A under {MAX_FRAMES}-frame window')
                return {}, None
            motion_a_placed = motion_a_placed[:keep_A]
            T_total = N_transition + motion_a_placed.shape[0]
            print(f'    [E15] A truncated to {keep_A} frames so that '
                  f'N_transition({N_transition}) + A({keep_A}) <= {MAX_FRAMES}')

        # ── Step 5: assemble [P | pad | A_placed] + final canonicalize ──
        # Segment is already canonical-aligned (P at origin, A starts at
        # +forward_step*(+Z)). The final canonicalize is near-identity but
        # we keep it so the R_canon/offset_canon round-trip is symmetric
        # with E14/E8-D.
        transition_pad = np.zeros(
            (N_transition - 1, 135), dtype=np.float32) if N_transition > 1 \
            else np.zeros((0, 135), dtype=np.float32)
        world_segment = np.concatenate(
            [P_canon, transition_pad, motion_a_placed], axis=0)
        T = world_segment.shape[0]
        assert T == T_total

        world_segment_t = _torch.from_numpy(world_segment).float()
        # world_segment is all raw local rot6d — hard-code 'local'.
        canon_segment_t, R_canon, offset_canon = canonicalize_segment(
            world_segment_t, anchor_frame=0, rotation_space='local')
        motion_135 = canon_segment_t.numpy()
        gt_motion_135 = motion_135

        _transition_canon_info = dict(
            R_canon=R_canon, offset_canon=offset_canon,
            motion_a_full=None,
            # For E15 the "after-transition" motion is A. We store its FULL
            # world-placed form here so the dashboard can decanon + stitch
            # the frames the model didn't see back into the visualization.
            motion_b_world_full=motion_a_placed_full,
            N_cond=1, N_transition=N_transition,
            _e15_N_transition=N_transition,
            _e15_len_A=motion_a_placed.shape[0],
            _e15_len_A_full=motion_a_placed_full.shape[0],
        )

        if motion_dim == 198:
            motion_raw = motion_135_to_198(motion_135, bone_offsets)
        else:
            motion_raw = motion_135

        mask = build_start_pose_prepend_mask(
            T, 135, N_transition=N_transition)

    # ---- (legacy) Old E15 path kept but no longer reachable since E15
    # settings no longer set `_use_target_first`. Left for reference.
    elif task.task_id == 'E15' and '_use_target_first' in setting_kwargs:
        N_cond_tail = setting_kwargs.get('_cond_tail_frames', 15)

        # Load target motion to get its first frame
        target_path = sample.get('target_motion_path', '')
        # Resolve (same logic as E14, 2026-04-24)
        if not os.path.isabs(target_path) and not os.path.exists(target_path):
            legacy = os.path.join(MOTION_DATA_DIR, target_path)
            if os.path.exists(legacy):
                target_path = legacy
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
        if '_transition_frames' in setting_kwargs:
            N_transition = int(setting_kwargs['_transition_frames'])
        else:
            N_transition = compute_transition_length(
                pos_tail[-1, 0], pos_target_world[0, 0],
                speed_per_frame=float(setting_kwargs.get(
                    '_transition_speed', 0.015)),
                min_frames=int(setting_kwargs.get('_transition_min', 30)),
                max_frames=int(setting_kwargs.get('_transition_max', 120)),
            )

        # Build sequence: [motion_tail | zeros(transition) | target_first_world]
        transition_pad = np.zeros((N_transition, 135), dtype=np.float32)
        world_segment = np.concatenate(
            [motion_tail, transition_pad, target_first_world], axis=0)
        T = world_segment.shape[0]

        # Canonicalize the whole segment (raw local rot6d → hard-code 'local').
        world_segment_t = _torch.from_numpy(world_segment).float()
        canon_segment_t, R_canon, offset_canon = canonicalize_segment(
            world_segment_t, anchor_frame=0, rotation_space='local')
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
        # Resolve (same logic as E14, 2026-04-24)
        if not os.path.isabs(target_path) and not os.path.exists(target_path):
            legacy = os.path.join(MOTION_DATA_DIR, target_path)
            if os.path.exists(legacy):
                target_path = legacy
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
        if '_transition_frames' in setting_kwargs:
            N_transition = int(setting_kwargs['_transition_frames'])
        else:
            N_transition = compute_transition_length(
                pos_target[-1, 0], pos_head_world[0, 0],
                speed_per_frame=float(setting_kwargs.get(
                    '_transition_speed', 0.015)),
                min_frames=int(setting_kwargs.get('_transition_min', 30)),
                max_frames=int(setting_kwargs.get('_transition_max', 120)),
            )

        # Build sequence: [target_last | zeros(transition) | motion_head_world]
        transition_pad = np.zeros((N_transition, 135), dtype=np.float32)
        world_segment = np.concatenate(
            [target_last, transition_pad, motion_head_world], axis=0)
        T = world_segment.shape[0]

        # Canonicalize (raw local rot6d → hard-code 'local').
        world_segment_t = _torch.from_numpy(world_segment).float()
        canon_segment_t, R_canon, offset_canon = canonicalize_segment(
            world_segment_t, anchor_frame=0, rotation_space='local')
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
        if task.task_id == 'E3' and setting_name in ('adaptive', 'D'):
            # E3 'adaptive' (legacy alias 'D'): SPARSE adaptive keyframe —
            # keep only the strongest acceleration peaks, no uniform filler.
            # Approximately 1 keyframe per second of motion at 30 fps.
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
                raise FileNotFoundError(
                    'No adaptive mask cached for E9 adaptive setting: '
                    f'{sample_mp}. Run scripts/compute_adaptive_masks_for_eval.py '
                    'with the same eval datalist before inference.'
                )
            mask = adaptive
        elif task.task_id == 'E9' and setting_kwargs.get('_qc_defect_mask'):
            # D_qc_mask_*: run the motion Quality Checker on the LQ motion
            # itself, OR all failing checkers' invalid_masks into a
            # per-joint per-frame defect mask, expand to motion_dim.
            # This replaces MoGenDIT's change-based adaptive mask with the
            # QC-rule-based ground truth. Rationale documented in
            # `_compute_qc_defect_mask`.
            qc_mask = _compute_qc_defect_mask(
                motion_135, bone_offsets, motion_dim=motion_dim,
                dilate_temp=int(setting_kwargs.get('_qc_dilate_temp', 2)),
                dilate_spatial=bool(setting_kwargs.get('_qc_dilate_spatial', True)),
                include_borderline=bool(setting_kwargs.get('_qc_include_borderline', True)),
                device=device,
            )
            if qc_mask is None:
                print(f'    [warn] QC defect mask build failed, using full mask')
                mask = np.ones((T, motion_dim), dtype=np.float32)
            else:
                mask = qc_mask
        elif task.task_id == 'E9' and setting_kwargs.get('_union_mask'):
            # 2026-04-26: union of MoGenDIT adaptive mask AND QC-rule mask.
            # User feedback: adaptive mask alone misses persistent
            # anatomical defects (neck/spine bent), QC mask alone misses
            # change-based artefacts (frame-boundary jumps). Combining the
            # two with logical OR gives broader coverage with no extra
            # tunable knobs. This is the new default for E9 going forward.
            sample_mp = sample.get('path', '') or sample.get('motion_path', '')
            adaptive = _load_adaptive_mask_for_motion(
                sample_mp, T, D=motion_dim,
                temporal_dilate=int(setting_kwargs.get('_union_adaptive_dilate', 0)),
            )
            qc_mask = _compute_qc_defect_mask(
                motion_135, bone_offsets, motion_dim=motion_dim,
                dilate_temp=int(setting_kwargs.get('_qc_dilate_temp', 2)),
                dilate_spatial=bool(setting_kwargs.get('_qc_dilate_spatial', True)),
                include_borderline=bool(setting_kwargs.get('_qc_include_borderline', True)),
                device=device,
            )
            if adaptive is None and qc_mask is None:
                print('    [warn] both adaptive and QC mask unavailable; '
                      'using full mask')
                mask = np.ones((T, motion_dim), dtype=np.float32)
            elif adaptive is None:
                print('    [info] adaptive mask missing for '
                      f'{sample_mp[-50:]}, using QC mask only')
                mask = qc_mask
            elif qc_mask is None:
                print('    [info] QC mask unavailable, using adaptive only')
                mask = adaptive
            else:
                # Both shapes are (T, motion_dim) float32 in {0, 1}.
                a = adaptive.astype(np.float32)
                q = qc_mask.astype(np.float32)
                # Defensive: align T dim in case of off-by-one drift.
                tmin = min(a.shape[0], q.shape[0])
                a = a[:tmin]
                q = q[:tmin]
                mask = np.maximum(a, q)
                if a.shape[0] != q.shape[0]:
                    print(f'    [info] union mask trimmed to T={tmin} '
                          f'(adaptive={a.shape[0]}, qc={q.shape[0]})')
        elif task.task_id == 'E9' and setting_kwargs.get('_strict_adaptive_mask'):
            # D_strict_mask_*: load raw MoGenDIT mask and post-process it
            # (per-joint aggregation + kinematic spatial dilation +
            # temporal dilation + blob filter) to get a tighter, more
            # reliable "definitely defective" flag. Combined with
            # replacement_guidance='skip_last' + clean_motion=LQ at
            # inference time so unflagged regions stay exactly at LQ.
            sample_mp = sample.get('path', '') or sample.get('motion_path', '')
            adaptive_raw = _load_adaptive_mask_for_motion(
                sample_mp, T, D=motion_dim, temporal_dilate=0,
            )
            if adaptive_raw is None:
                raise FileNotFoundError(
                    'No adaptive mask cached for E9 strict adaptive setting: '
                    f'{sample_mp}. Run scripts/compute_adaptive_masks_for_eval.py '
                    'with the same eval datalist before inference.'
                )
            mask = _compute_strict_adaptive_mask(
                adaptive_raw,
                dilate=int(setting_kwargs.get('_strict_dilate', 2)),
                min_blob=int(setting_kwargs.get('_strict_min_blob', 3)),
                motion_dim=motion_dim,
            )
        elif task.task_id == 'E9' and setting_kwargs.get('_ada_denoise'):
            # D_ada_denoise_*: two-stage MoGenDIT ada_denoise.
            # Stage 1 needs mask=all-1 (full regeneration) — build that
            # here as the initial mask. After Stage 1 inference completes,
            # we intercept the output, compute the change-based keep_mask,
            # then re-run inference with the new mask. That second pass
            # is triggered inside the inference block below.
            mask = np.ones((T, motion_dim), dtype=np.float32)
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
        if source_motion_raw is not None:
            _sr = source_motion_raw.copy()
            _src_rot_local = _torch.from_numpy(
                _sr[:, 3:135].reshape(_sr.shape[0], 22, 6)).float()
            _src_rot_global = _l2g(_src_rot_local)
            _sr[:, 3:135] = _src_rot_global.reshape(_sr.shape[0], 132).numpy()
            source_motion_raw = _sr

    motion_norm = bundle.normalize_motion(
        torch.from_numpy(motion_raw).float().unsqueeze(0).to(device))
    source_motion_norm = None
    if source_motion_raw is not None:
        source_motion_norm = bundle.normalize_motion(
            torch.from_numpy(source_motion_raw).float().unsqueeze(0).to(device))
    src_mask = torch.from_numpy(mask).float().unsqueeze(0).to(device)

    # Pad to 360 frames (training always pads to 360). For E9 repair, motions
    # may be LONGER than 360 — handled below via sliding-window stitching.
    T_PAD = 360
    if T < T_PAD:
        pad_len = T_PAD - T
        # ── train/infer parity (2026-04-26 fix) ─────────────────────────
        # Although the dataset transform is `RandomCropPadding(pad_mode=
        # 'replicate')`, the *trainer* unconditionally OVERWRITES the pad
        # region with zeros AFTER normalization (see
        # `HyMotionM2MTrainer._prepare_and_forward`, L116-121):
        #
        #     if src_len < L_src:
        #         src_motion[i, src_len:] = 0.0
        #         src_mask[i, src_len:]   = 0.0
        #
        # i.e. the value the model actually sees in the pad region is
        # `0` in normalized space, which equals the unnormalized MEAN
        # POSE — *not* the replicated last frame. So inference must
        # match: zero-pad both motion and mask in normalized space.
        #
        # In practice the difference is tiny because the attention mask
        # is hard `-inf` (see hymotion_mmdit._canonical_mask), so pad
        # tokens are completely invisible to valid tokens. But matching
        # training distribution exactly is cheap and removes one OOD
        # variable when debugging boundary artifacts.
        motion_norm = torch.nn.functional.pad(
            motion_norm, (0, 0, 0, pad_len), mode='constant', value=0.0)
        if source_motion_norm is not None:
            source_motion_norm = torch.nn.functional.pad(
                source_motion_norm, (0, 0, 0, pad_len), mode='constant', value=0.0)
        src_mask = torch.nn.functional.pad(
            src_mask, (0, 0, 0, pad_len), mode='constant', value=0.0)

    # Prepare src_motion: zero out masked regions (completion/inpainting mode)
    # vs keep LQ values (editing mode). Per-setting _editing_mode overrides the
    # task-level task.is_editing default (used by E9 to test inpaint vs edit).
    is_editing_effective = setting_kwargs.get('_editing_mode', task.is_editing)
    if is_editing_effective:
        # editing: keep source/LQ values in the reactive channel. For real
        # style-edit pairs this is the neutral source motion; for synthetic
        # local edits it falls back to the GT/source motion itself.
        src_motion_norm = (
            source_motion_norm.clone()
            if source_motion_norm is not None else motion_norm.clone()
        )
    else:
        src_motion_norm = motion_norm * (1 - src_mask)

    # Prepare clean_motion for MAN imputation
    clean_motion = motion_norm.clone()

    # Optional: Gaussian temporal pre-smooth of LQ before imputation.
    # Addresses the D_strict_mask jitter/抽搐 issue — MAN training used
    # clean x1 for x_t[keep], but here clean_motion=LQ carries jitter,
    # which the model tries to reconcile with generated regions at blob
    # boundaries, amplifying jerk. Pre-smoothing reduces the high-freq
    # energy of the "known" signal while leaving generated regions alone
    # (protect_mask=src_mask: sigma is only applied where mask==0=keep).
    # σ≈1 frame @ 30 fps removes ~5 Hz+ while keeping gross kinematics.
    _presmooth_sigma = float(setting_kwargs.get('_presmooth_clean_sigma', 0.0))
    if _presmooth_sigma > 0.0:
        clean_motion = _gaussian_temporal_smooth(
            clean_motion, sigma=_presmooth_sigma, protect_mask=src_mask)

    # Optional: replace the imputation target with a "manifold-projected LQ"
    # produced by a pre-pass of SDEdit-from-LQ (τ=0.5, mask=all ones,
    # skip_last). This is the same idea as D_ada_denoise Stage 1, but here
    # we DO NOT use it to rebuild the mask — we only use it to replace
    # `clean_motion` so the skip_last imputation pulls the keep-region
    # toward a smooth manifold instead of toward raw LQ (which carries
    # the exact jitter strict_mask was supposed to repair).
    #
    # Motivation (2026-04-23): D_strict_mask_d2_b3_bsmooth still shows
    # local jumps because mask=0 (keep) frames are locked to LQ via
    # `x_t[keep] = LQ` at every denoise step; those LQ values contain the
    # original jitter, which the model then tries to reconcile with the
    # smooth generated region at mask boundaries. D_ada_denoise avoids
    # this because |LQ − stage1| is small in its keep region, so locking
    # to LQ ≈ locking to Stage1 (smooth).
    #
    # Blend: clean_anchor = α·stage1 + (1-α)·LQ (default α=1.0).
    _anchor_to_stage1 = bool(setting_kwargs.get('_anchor_to_stage1', False))
    _manifold_blend_alpha = float(setting_kwargs.get('_manifold_blend_alpha', 1.0))
    _manifold_sdedit_tau = float(setting_kwargs.get('_manifold_sdedit_tau', 0.5))
    # Cache Stage 1 output — D_ada_denoise path can also reuse this
    # (same computation) by setting `_ada_reuse_manifold=True`, but the
    # default ada_denoise path still runs its own Stage 1 for backward
    # compatibility. Here we only set the cache; actual reuse is
    # triggered downstream.
    _stage1_manifold_latent = None

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
    # Per-setting replacement_guidance override. A_adaptive_inpaint requires
    # 'skip_last' so that the pipeline's SDEdit branch (use_replacement=True)
    # actually activates — without this override the function-level default
    # ('none') propagates and y0 becomes pure noise, silently bypassing
    # SDEdit τ=0.5. See docs/temp/e9_a_d_inpaint_bug_20260422.md.
    if '_replacement_guidance' in setting_kwargs:
        pipeline.replacement_guidance = str(
            setting_kwargs['_replacement_guidance'])

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

    def _run_single_window(mn_win, sm_win, src_clean_win, T_win,
                           editing_override=None):
        """Run one pipeline call with T_win ≤ T_PAD. mn_win/sm_win/src_clean_win
        are already padded to (1, T_PAD, D). ``editing_override`` lets the
        ada_denoise Stage-1 caller force non-editing mode regardless of the
        setting's `_editing_mode`."""
        if editing_override is not None:
            is_editing_local = editing_override
        else:
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

    def _run_inference_pass(mask_tensor, editing_override=None):
        """Run one full pipeline pass (single window OR sliding-window
        blending) with the current mask_tensor. Reads motion_norm /
        clean_motion / pipeline / text_fields from the enclosing scope.

        Returns ``sampled_norm`` — a torch tensor (1, T_eff, D) where
        T_eff = T_PAD if not windowed, else T.
        """
        if editing_override is not None:
            is_editing_here = editing_override
        else:
            is_editing_here = setting_kwargs.get('_editing_mode', task.is_editing)
        if is_editing_here:
            src_motion_here = motion_norm.clone()
        else:
            src_motion_here = motion_norm * (1 - mask_tensor)

        if not needs_windowed:
            batch_local = {
                'src_motion': src_motion_here,
                'src_mask': mask_tensor,
                'src_length': [T],
                'tgt_length': [T],
                'clean_motion': clean_motion,
            }
            if text_fields is not None:
                batch_local.update(text_fields)
            with torch.no_grad():
                out_local = pipeline(batch_local)
            return out_local['latent']
        # Sliding-window path
        starts_local = [0, T - T_PAD]
        window_outputs_local = []
        for start_local in starts_local:
            end_local = start_local + T_PAD
            mn_win = motion_norm[:, start_local:end_local, :].contiguous()
            sm_win = mask_tensor[:, start_local:end_local, :].contiguous()
            sc_win = clean_motion[:, start_local:end_local, :].contiguous()
            samp_win = _run_single_window(
                mn_win, sm_win, sc_win, T_PAD,
                editing_override=editing_override)
            window_outputs_local.append((start_local, samp_win))
        _, outA = window_outputs_local[0]
        startB, outB = window_outputs_local[1]
        D_local = outA.shape[-1]
        sampled_full_local = torch.zeros(
            (1, T, D_local), dtype=outA.dtype, device=outA.device)
        sampled_full_local[:, 0:startB, :] = outA[:, 0:startB, :]
        overlap_len_local = T_PAD - startB
        if overlap_len_local > 0:
            w_local = torch.linspace(
                1.0, 0.0, steps=overlap_len_local,
                dtype=outA.dtype, device=outA.device).view(1, overlap_len_local, 1)
            a_over = outA[:, startB:T_PAD, :]
            b_over = outB[:, 0:overlap_len_local, :]
            sampled_full_local[:, startB:T_PAD, :] = \
                a_over * w_local + b_over * (1.0 - w_local)
        sampled_full_local[:, T_PAD:T, :] = outB[:, overlap_len_local:T_PAD, :]
        return sampled_full_local

    t0 = time.time()
    # Hoist this flag here so the manifold-projection block below can check
    # it (ada_denoise has its own Stage 1; we skip duplicate projection).
    ada_denoise = bool(setting_kwargs.get('_ada_denoise', False))

    # ── Optional manifold-projection of LQ for strict_mask imputation ──
    # When `_anchor_to_stage1=True`, do a single SDEdit pre-pass over the
    # full motion to get a manifold-projected version of LQ, then use it
    # as the `clean_motion` target for the main strict_mask inference.
    # This prevents the keep-region from locking to raw (jittery) LQ.
    if _anchor_to_stage1 and not ada_denoise:
        _prev_repl_mn = pipeline.replacement_guidance
        _prev_tau_mn = pipeline.sdedit_tau
        pipeline.replacement_guidance = 'skip_last'
        pipeline.sdedit_tau = _manifold_sdedit_tau
        # CRITICAL: pipeline activates SDEdit ONLY when src_mask has BOTH
        # 0s and 1s (line 293-295 of hymotion_m2m_pipeline.py). An all-ones
        # mask degenerates to use_replacement=False → y0=pure_noise, which
        # produces an unconditional generation unrelated to LQ (the exact
        # failure mode that D_ada_denoise_t010 exhibits). To actually
        # project LQ onto the manifold we need a near-all-ones mask with
        # at least one "keep" cell so the SDEdit branch fires with
        # y0 = (1-τ)*z + τ*LQ on the masked region. Keep frame 0 channel 0
        # as the single anchor point.
        _proj_mask_t = torch.ones_like(src_mask)
        _proj_mask_t[:, 0, 0] = 0.0  # 1 keep cell → SDEdit branch activates
        _stage1_manifold_latent = _run_inference_pass(
            _proj_mask_t, editing_override=False)
        pipeline.replacement_guidance = _prev_repl_mn
        pipeline.sdedit_tau = _prev_tau_mn
        # Blend clean_motion: α·stage1 + (1-α)·LQ
        if _manifold_blend_alpha >= 0.999:
            clean_motion = _stage1_manifold_latent
        else:
            clean_motion = (
                _manifold_blend_alpha * _stage1_manifold_latent
                + (1.0 - _manifold_blend_alpha) * clean_motion
            )

    # ── Optional Stage 1 for D_ada_denoise settings ──────────────
    # D_ada_denoise_*: run an exploratory full-regen pass first to
    # detect which joints/frames the model wants to change. Use the
    # difference between LQ and denoised_stage1 to build a tighter
    # keep_mask, then run Stage 3 with replacement_guidance='skip_last'
    # so the "clean" regions get anchored back to LQ.
    # (Stage 1 forces mask=all-1, no editing mode, no replacement.)
    # NOTE: `ada_denoise` was hoisted above for the manifold-projection
    # pre-check; no need to re-evaluate.
    if ada_denoise:
        # Stage 1: SDEdit-from-LQ manifold projection (revised 2026-04-23).
        #
        # ── History ──
        # The original Stage 1 followed MoGenDIT motion_refiner.py literally:
        #   mask = ones; mask[:, 0:1, :] = 0
        #   replacement_guidance = 'skip_last'
        #   sdedit_tau = 0.0
        # MoGenDIT works that way because it was trained with 50% motion
        # degradation → clean pairs, so "only frame 0 known" triggers a
        # repair behavior. M2M was NOT trained on motion degradation → clean
        # pairs; given the same mask, M2M treats it as a T2M generation
        # prompt anchored at frame 0. Stage 1 output then has no meaningful
        # relationship to LQ, so |LQ − stage1| is dominated by "two
        # different motions" noise, threshold always fires, Stage 3 ends
        # up regenerating everywhere. See docs/temp/e9_a_d_inpaint_bug_20260422.md.
        #
        # ── Revised Stage 1 (2026-04-23) ──
        #   mask = all ones (every frame/channel is "generate"), so the
        #       normal x_t[keep] imputation is a no-op.
        #   replacement_guidance = 'skip_last' (kept so use_replacement=True
        #       in the pipeline, activating the SDEdit branch)
        #   sdedit_tau = 0.5  (start from 0.5*LQ + 0.5*noise at t=0.5)
        # Running 50% of the ODE starting from a half-noised LQ is an
        # implicit manifold projection: the model pulls LQ back toward the
        # nearest clean motion. |LQ − stage1| then highlights frames/joints
        # the model considers inconsistent with the clean-motion manifold.
        # That is the defect signal Stage 2 needs.
        prev_repl = pipeline.replacement_guidance
        prev_tau = pipeline.sdedit_tau
        pipeline.replacement_guidance = 'skip_last'
        pipeline.sdedit_tau = 0.5
        # CRITICAL (2026-04-23): an all-ones mask makes the pipeline's
        # `use_replacement` check fail (src_mask.sum() == src_mask.numel()),
        # falling back to y0=pure_noise — Stage 1 becomes an uncond gen
        # unrelated to LQ. This was the hidden reason D_ada_denoise_t010
        # "HQ doesn't follow LQ": Stage 1 was a noise-gen, Stage 2 saw
        # huge change everywhere, Stage 3 also got all-ones → another
        # noise-gen. Leaving ONE cell as keep lets SDEdit activate so
        # Stage 1 is a true LQ-anchored manifold projection.
        stage1_mask_t = torch.ones_like(src_mask)
        stage1_mask_t[:, 0, 0] = 0.0  # single keep cell → SDEdit activates
        sampled_norm_stage1 = _run_inference_pass(
            stage1_mask_t, editing_override=False)

        # Stage 2: analyze change in normalized space, build new keep_mask
        # sampled_norm_stage1 is (1, T_eff, D); motion_norm is (1, T_eff, D)
        # where T_eff = T for windowed, T_PAD for short.
        lq_np = motion_norm[0].detach().cpu().numpy()              # (T_eff, D)
        st1_np = sampled_norm_stage1[0].detach().cpu().numpy()     # (T_eff, D)
        T_eff = lq_np.shape[0]
        # For short-path, lq_np is padded to T_PAD with the replicated
        # last frame; we still compute change over the full T_eff since
        # mask keeps pad frames as 0 downstream anyway.
        new_mask_full = _compute_ada_keep_mask(
            motion_norm_lq=lq_np,
            denoised_stage1=st1_np,
            threshold_mode=str(setting_kwargs.get('_ada_threshold_mode', 'abs')),
            threshold=float(setting_kwargs.get('_ada_threshold', 0.1)),
        )  # (T_eff, D), 1=generate, 0=keep
        # Keep the pad region of the mask at 0 (known/hold last frame)
        # just like the original src_mask did — DO NOT let ada_denoise's
        # "change over pad" flip these to 1 (it usually does since the
        # model's stage-1 pad region is free-denoise output, unrelated
        # to the replicated LQ frame).
        if T < T_eff:
            new_mask_full[T:] = 0.0
        # Upload new mask to device
        stage3_mask_t = torch.from_numpy(
            new_mask_full).float().unsqueeze(0).to(src_mask.device)

        # Restore pipeline params for Stage 3.
        # IMPORTANT (2026-04-23): Stage 3's mask=1 (regenerate) region by
        # default starts from pure noise (`y0 = torch.where(keep, x_clean, z)`
        # in pipeline). This means the generated region has NO anchor to LQ
        # — it just obeys the skip_last imputation at the boundary. For
        # repair, that often produces a "plausible but completely different
        # motion" in the masked region (user feedback: "完全不 follow LQ").
        # Use `_ada_stage3_sdedit_tau > 0` to start Stage 3 from
        # τ*noise + (1-τ)*LQ on the masked region, anchoring the generation
        # to LQ's content. Default keeps prior behavior (τ=0).
        pipeline.replacement_guidance = prev_repl if prev_repl != 'none' else 'skip_last'
        stage3_tau = float(setting_kwargs.get('_ada_stage3_sdedit_tau', prev_tau))
        pipeline.sdedit_tau = stage3_tau
        # Swap src_mask for the new mask — downstream cond_overwrite and
        # post-hoc branches should use the stage-3 mask too (it's what
        # the final output actually came from).
        src_mask = stage3_mask_t
        # numpy copy for downstream cond_mask — crop back to T frames
        # (motion_135 has T frames; mask is (T_eff, D) where T_eff may
        # equal T_PAD for short-path inference).
        mask = new_mask_full[:T]
        sampled_norm = _run_inference_pass(stage3_mask_t)
    elif not needs_windowed:
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
        # ── DEBUG (2026-05-09): verify condition replacement correctness for E14
        if task.task_id == 'E14' and os.environ.get('DEBUG_E14_DECANON'):
            _diff_post_replace = np.abs(output_135[:mask_135.shape[0]][cond_mask] - motion_135[cond_mask]).max()
            print(f'    [DEBUG E14] post-replace cond diff: {_diff_post_replace:.8f}')
            print(f'    [DEBUG E14] output_135[0,:3]={output_135[0,:3]}  motion_135[0,:3]={motion_135[0,:3]}')
            print(f'    [DEBUG E14] cond_mask.sum()={cond_mask.sum()} mask.shape={mask.shape} motion_135.shape={motion_135.shape}')

    # ---- Post-process boundary smoothing (2026-04-23) -------------------
    # When strict_mask has sharp 0↔1 boundaries, the imputation step
    # (skip_last: `x[keep] = LQ[keep]` per ODE step) produces a hard
    # discontinuity in the generated output at blob edges: keep frames
    # carry LQ content exactly, but one frame later (in the generated
    # region) the model has produced a different velocity/pose. Result:
    # a per-joint acceleration spike of O(0.1 m/frame²) at every blob
    # boundary — exactly what the user observed as "local jumps".
    #
    # Fix: blend a Gaussian-smoothed version of the output back into
    # the narrow band around each 0/1 transition in the mask. The blend
    # weight is a tent that peaks at the boundary (±`radius` frames),
    # and is 0 elsewhere — so condition regions remain exactly LQ and
    # bulk generated regions stay unchanged; only the boundary frames
    # (where the discontinuity lives) are smoothed.
    bs_radius = int(setting_kwargs.get('_boundary_smooth_radius', 0))
    if bs_radius > 0 and mask_135 is not None:
        cond_mask_frame = (mask_135 < 0.5).any(axis=-1) \
            if mask_135.ndim == 2 else (mask_135 < 0.5).any(axis=(0,))
        # Find transitions (0→1 or 1→0) along the time axis
        transitions = np.diff(
            cond_mask_frame.astype(np.int8), prepend=0
        ) != 0
        if transitions.any():
            T_o = output_135.shape[0]
            # Build tent weight: 1.0 at each transition, linearly falls
            # to 0 over `bs_radius` frames on each side, OR'd across
            # multiple boundaries.
            w = np.zeros(T_o, dtype=np.float32)
            idxs = np.where(transitions)[0]
            for idx in idxs:
                for d in range(-bs_radius, bs_radius + 1):
                    t = idx + d
                    if 0 <= t < T_o:
                        tent = 1.0 - abs(d) / max(bs_radius, 1)
                        w[t] = max(w[t], tent)
            # Gaussian-smooth output along time
            sigma_bs = float(setting_kwargs.get(
                '_boundary_smooth_sigma', max(1.0, bs_radius * 0.5)))
            radius_kern = max(1, int(round(3.0 * sigma_bs)))
            offs = np.arange(-radius_kern, radius_kern + 1, dtype=np.float32)
            kernel = np.exp(-offs * offs / (2.0 * sigma_bs * sigma_bs))
            kernel /= kernel.sum()
            # conv along time (replicate padding)
            pad = np.pad(output_135, ((radius_kern, radius_kern), (0, 0)),
                         mode='edge')
            smoothed = np.zeros_like(output_135)
            for i, k in enumerate(kernel):
                smoothed += k * pad[i:i + T_o]
            # Per-frame blend: output = w * smoothed + (1-w) * output
            w2 = w[:, None]
            output_135 = (w2 * smoothed + (1.0 - w2) * output_135).astype(np.float32)

    # ---- Post-proc 1: accel-spike median filter (2026-04-23) ----
    # Detect isolated frames whose per-channel acceleration is an outlier
    # (> mean + k·std) and replace those frames (only!) with a 3-tap
    # temporal median. This kills spike-type jitter WITHOUT touching
    # normal high-frequency motion. Only activates when the setting asks.
    asm_k = float(setting_kwargs.get('_accel_spike_k', 0.0))
    if asm_k > 0.0 and output_135.shape[0] >= 5:
        try:
            x = output_135  # (T, D)
            # Second difference ≈ acceleration
            accel = np.abs(x[2:] - 2 * x[1:-1] + x[:-2])  # (T-2, D)
            # Aggregate per frame: max across channels
            accel_frame = accel.max(axis=-1)  # (T-2,)
            mu = accel_frame.mean()
            sd = accel_frame.std()
            thr = mu + asm_k * sd
            spike_mask = accel_frame > thr  # (T-2,) True where frame is outlier
            if spike_mask.any():
                # Map back: accel index t corresponds to frame t+1
                # (center of the 3-frame stencil).
                spike_idxs = np.where(spike_mask)[0] + 1
                # 3-tap median per spike frame, per channel.
                for t in spike_idxs:
                    if 0 < t < x.shape[0] - 1:
                        x[t] = np.median(x[t - 1:t + 2], axis=0)
                output_135 = x

        except Exception as e:
            print(f'    [warn] accel-spike filter failed: {e!r}')

    # ---- Post-proc 2: Savitzky-Golay global smoother (2026-04-23) ----
    # Full-sequence Savgol filter. Preserves local peak shapes better than
    # Gaussian (which blurs them uniformly). Good for light jitter removal
    # when motion has lots of fast transitions. window must be odd.
    sg_window = int(setting_kwargs.get('_savgol_window', 0))
    sg_poly = int(setting_kwargs.get('_savgol_poly', 3))
    if sg_window >= 5 and output_135.shape[0] >= sg_window:
        try:
            from scipy.signal import savgol_filter
            if sg_window % 2 == 0:
                sg_window += 1
            output_135 = savgol_filter(
                output_135, window_length=sg_window, polyorder=sg_poly,
                axis=0, mode='nearest').astype(np.float32)
        except Exception as e:
            print(f'    [warn] savgol filter failed: {e!r}')

    # ---- E9 two-stage post-hoc replacement (MoGenDIT ada_denoise style) ----
    # When the setting asks for `_post_hoc_replace_with_adaptive`, inference
    # was run in C_full mode (mask=all 1, clean full regeneration).  Now
    # we load the *adaptive* mask and use it ONLY as a post-hoc blending
    # mask: wherever the adaptive mask flagged defective regions, use the
    # model's regenerated output; wherever it flagged clean regions, revert
    # to the original LQ motion.
    #
    # Rationale: the model was never trained on "adaptive sparse point"
    # masks as *input* conditioning, so feeding it an adaptive mask at
    # inference is OOD and amplifies jitter.  By contrast, C_full is a
    # trained pattern (M5 full_mask) and produces clean output.  The
    # adaptive mask is much more reliable as a *post-hoc* selector (it
    # accurately flags *where* the defects are, per MoGenDIT's change
    # detection).  This mirrors MoGenDIT `ada_denoise`'s two-stage design.
    if setting_kwargs.get('_post_hoc_replace_with_adaptive', False):
        sample_mp = sample.get('path', '') or sample.get('motion_path', '')
        adaptive_198 = _load_adaptive_mask_for_motion(
            sample_mp, T, D=motion_dim,
        )
        if adaptive_198 is not None:
            adaptive_135 = (
                adaptive_198[:, :135] if adaptive_198.shape[-1] > 135
                else adaptive_198
            )
            # Per-frame "defective" flag: any joint/dim flagged → defective.
            frame_defect = (adaptive_135 >= 0.5).any(axis=1).astype(np.float32)  # (T,)
            # Smooth the 0/1 indicator with a triangular kernel so the
            # output transitions between LQ (frame_defect=0 → w_C=0) and
            # C_full output (frame_defect=1 → w_C=1) are continuous,
            # preventing accel jumps at binary switch boundaries (which
            # dominated the jitter metric at ~4000).
            blend_radius = int(setting_kwargs.get('_blend_radius', 5))
            if blend_radius > 0:
                # Build triangular kernel of width 2*blend_radius+1
                k = blend_radius
                kernel = np.concatenate([
                    np.arange(1, k + 1, dtype=np.float32),
                    np.array([k + 1], dtype=np.float32),
                    np.arange(k, 0, -1, dtype=np.float32),
                ])
                kernel = kernel / kernel.sum()
                # Dilate defective flag first (OR filter over blend_radius
                # each side) so C_full output is used for the full "around
                # defect" window before we smooth. This ensures the defective
                # region itself is cleanly replaced, not blended-out.
                defect_dilated = frame_defect.copy()
                for s in range(1, k + 1):
                    defect_dilated[s:] = np.maximum(defect_dilated[s:], frame_defect[:-s])
                    defect_dilated[:-s] = np.maximum(defect_dilated[:-s], frame_defect[s:])
                w_C = np.convolve(defect_dilated, kernel, mode='same')
                # Clamp to [0, 1]
                w_C = np.clip(w_C, 0.0, 1.0)
            else:
                w_C = frame_defect

            if (w_C < 1.0).any():
                if rotation_space == 'global':
                    import torch as _torch
                    from hftrainer.datasets.motion.motionhub.transforms.fk_utils import (
                        local_to_global_rot6d_torch,
                    )
                    rot6d_local = _torch.from_numpy(
                        motion_135[:, 3:135].reshape(T, 22, 6)).float()
                    rot6d_global = local_to_global_rot6d_torch(rot6d_local)
                    lq_for_blend = motion_135.copy()
                    lq_for_blend[:, 3:135] = rot6d_global.reshape(T, 132).numpy()
                else:
                    lq_for_blend = motion_135
                # Per-frame linear blend between C_full output and LQ.
                w = w_C.reshape(T, 1)
                output_135 = output_135 * w + lq_for_blend * (1.0 - w)
                output_135 = output_135.astype(np.float32)
        else:
            # No adaptive mask cached — fall through, keep C_full output.
            pass

    # ---- Decanonicalize transition output back to world coordinates ----
    # For E14/E15/E16, inference ran in canonical space (anchor at origin,
    # heading +Z). Map the output back to world coords and stitch with the
    # full A / B motions so metrics and visualization are in world space.
    _canon_info = locals().get('_transition_canon_info', None)
    if (task.task_id in ('E14', 'E15', 'E16', 'E8') and _canon_info is not None):
        import torch as _torch
        from hftrainer.pipelines.motion.transition_utils import (
            decanonicalize_segment,
        )
        R_canon = _canon_info['R_canon']
        offset_canon = _canon_info['offset_canon']

        # Decanonicalize the whole segment (all T frames). `rotation_space`
        # tells apply_rigid_transform_to_motion whether body joints are
        # world-referenced (global) — if so they must also be yaw-rotated.
        out_t = _torch.from_numpy(output_135).float()
        out_world_t = decanonicalize_segment(
            out_t, R_canon, offset_canon, rotation_space=rotation_space)
        output_135 = out_world_t.numpy()
        # ── DEBUG (2026-05-09): check decanon result
        if os.environ.get('DEBUG_E14_DECANON'):
            print(f'    [DEBUG E14] after decanon output_135[0,:3]={output_135[0,:3]}')
            print(f'    [DEBUG E14] R_canon={R_canon}, offset_canon={offset_canon}')

        # Also decanonicalize the GT segment so metrics compare in world coords
        gt_t = _torch.from_numpy(gt_motion_135).float()
        gt_world_t = decanonicalize_segment(
            gt_t, R_canon, offset_canon, rotation_space=rotation_space)
        gt_motion_135 = gt_world_t.numpy()

        # For E14: stitch with the full motion_a prefix and motion_b_world
        # suffix so the visualized output covers the entire path, not just
        # the transition window.
        # NOTE: this used to be enabled but caused the dashboard to display
        # the A prefix twice (once from backend stitching, once from
        # `/api/source_motions` stitching on the frontend) and created a
        # height mismatch between the two copies. The frontend already
        # handles source-motion context via stitchSourceMotions(); output_135
        # should therefore contain only the transition window.
        # Guard with `_backend_stitch=True` in setting_kwargs to opt-in again
        # (not set by any current setting).
        if task.task_id == 'E14' and setting_kwargs.get('_backend_stitch', False):
            motion_a_full = _canon_info['motion_a_full']
            motion_b_full = _canon_info['motion_b_world_full']
            N_cond = _canon_info['N_cond']
            if motion_a_full is not None and motion_b_full is not None:
                prefix = motion_a_full[:-N_cond]  # all of A except its tail
                suffix = motion_b_full[N_cond:]   # all of B except its head
                output_135 = np.concatenate([prefix, output_135, suffix], axis=0)
                gt_motion_135 = np.concatenate([prefix, gt_motion_135, suffix], axis=0)

    # ---- Stash layout metadata so NPZ can expose it to the dashboard ----
    # 2026-04-23: with dynamic N_cond_a / N_cond_b / N_transition / N_append,
    # the frontend can no longer infer the frame layout from `currentSetting`.
    # Attach the resolved numbers so /api/npz/<path> can return them for the
    # stitchSourceMotions() frontend path.
    _layout = None
    if task.task_id == 'E14' and _canon_info is not None:
        _layout = {
            'task': 'E14',
            'N_cond_a': int(_canon_info.get('N_cond_a', _canon_info.get('N_cond', 0))),
            'N_transition': int(_canon_info.get('N_transition', 0)),
            'N_cond_b': int(_canon_info.get('N_cond_b', _canon_info.get('N_cond', 0))),
        }
    elif task.task_id == 'E15' and _canon_info is not None:
        _layout = {
            'task': 'E15',
            'N_transition': int(_canon_info.get('N_transition', 0)),
            'len_A': int(_canon_info.get('_e15_len_A', 0)),
            # full A length = original motion_a_placed length before truncation.
            # Dashboard reads this to know how many trailing A frames to
            # stitch back via source_motions API.
            'len_A_full': int(_canon_info.get('_e15_len_A_full',
                                              _canon_info.get('_e15_len_A', 0))),
            'N_cond_A': int(_canon_info.get('_e15_len_A', 0)),
        }
    elif task.task_id == 'E8' and _canon_info is not None:
        _layout = {
            'task': 'E8',
            'T_gt_eff': int(_canon_info.get('N_cond', 0)),
            'N_append': int(_canon_info.get('N_transition', 0)),
            'n_dropped_prefix': int(_canon_info.get('n_dropped_prefix', 0)),
        }

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
        
        # Use dynamically computed N_cond_a and N_cond_b from setup.
        # These were stored in _transition_canon_info (accessible via _canon_info)
        # and differ from the static '_cond_frames' setting.
        N_cond_a = 15  # default fallback
        N_cond_b = 15  # default fallback
        if _canon_info is not None:
            N_cond_a = int(_canon_info.get('N_cond_a', _canon_info.get('N_cond', 15)))
            N_cond_b = int(_canon_info.get('N_cond_b', _canon_info.get('N_cond', 15)))
        
        pred_pos = motion135_to_positions_np(output_135, bone_offsets)
        fps_val = sample.get('fps', 30)
        vel = np.diff(pred_pos, axis=0) * fps_val
        acc = np.diff(vel, axis=0) * fps_val
        
        # Boundary at A->transition: frame N_cond_a-1
        if N_cond_a - 1 < acc.shape[0] and N_cond_a < acc.shape[0]:
            jump_a = np.linalg.norm(acc[N_cond_a] - acc[N_cond_a - 1], axis=-1).mean()
            metrics['boundary_accel_jump_a'] = float(jump_a)
        
        # Boundary at transition->B: frame T - N_cond_b - 1
        b_boundary = T - N_cond_b - 1
        if 0 < b_boundary < acc.shape[0] and b_boundary + 1 < acc.shape[0]:
            jump_b = np.linalg.norm(acc[b_boundary + 1] - acc[b_boundary], axis=-1).mean()
            metrics['boundary_accel_jump_b'] = float(jump_b)
        metrics['transition_length'] = int(T - N_cond_a - N_cond_b)

    # E15: new semantics (2026-04-21) — prepend transition from start pose P.
    #   frame 0 = P (locked), frame N_transition = A[0] (locked).
    #   mpjpe_first_frame measures how well pred[0] matches P (should be ~0
    #     due to replacement guidance).
    #   boundary_accel_jump measures the join between transition and A[0].
    if task.task_id == 'E15':
        from hftrainer.evaluation.motion.m2m_eval_metrics import motion135_to_positions_np
        pred_pos = motion135_to_positions_np(output_135, bone_offsets)
        gt_pos = motion135_to_positions_np(gt_motion_135, bone_offsets)
        # First frame should match P (stored in gt_motion_135[0])
        mpjpe_first = np.sqrt(np.sum(
            (pred_pos[0] - gt_pos[0]) ** 2, axis=-1)).mean()
        metrics['mpjpe_first_frame'] = float(mpjpe_first)
        N_trans = setting_kwargs.get('_prepend_N', 30)
        metrics['transition_length'] = int(N_trans)
        # Boundary jump at the transition → A join.
        if pred_pos.shape[0] > N_trans + 1:
            vel = np.diff(pred_pos, axis=0) * sample.get('fps', 30)
            acc = np.diff(vel, axis=0) * sample.get('fps', 30)
            if N_trans - 1 < acc.shape[0] and N_trans < acc.shape[0]:
                jump = np.linalg.norm(
                    acc[N_trans] - acc[N_trans - 1], axis=-1).mean()
                metrics['boundary_accel_jump'] = float(jump)

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

    # ── E9-only: Quality-Checker pass rate ───────────────────────
    # User target (2026-04-22): "正确的推理方案应该能做到 60% 以上的修复率".
    # Run the same MotionQualityChecker that the motion-annotation web
    # tool uses to classify LQ/HQ, and expose the aggregate verdict so
    # per-setting QC pass rates are visible in the dashboard. Cached on
    # the module to avoid re-instantiating the 20 sub-checkers per sample.
    if task.task_id == 'E9':
        try:
            qc_result = _run_quality_checker(output_135, bone_offsets, device=device)
            if qc_result is not None:
                metrics['qc_pass'] = float(qc_result['is_valid'])   # 1 / 0
                metrics['qc_num_failed'] = float(len(qc_result['failed_checks']))
                metrics['qc_num_borderline'] = float(len(qc_result['borderline_checks']))
                # Per-checker PASS flag (1 = checker PASSED, 0 = failed).
                # Aggregated across samples → pass rate.  Unified 2026-04-23:
                # all QC metrics now share the "higher is better" convention
                # (qc_pass, qc_<checker>).  Previously this was fail rate
                # which disagreed with qc_pass direction.
                for ch_name, ch_info in qc_result['per_checker'].items():
                    metrics[f'qc_{ch_name}'] = (
                        1.0 if ch_info.get('is_valid', True) else 0.0
                    )
        except Exception as e:
            print(f'    [warn] QC checker failed: {e!r}')

    metrics['inference_time'] = round(elapsed, 2)
    # Stash GT motion + editing mask so the save_npz path can persist them for
    # the multi-task mesh viewer (condition / GT / pred). These are big arrays
    # and MUST be popped before the metrics dict is JSON-serialized for DB
    # import (done right after the NPZ write in the main loop).
    #   _src_mask: (T, motion_dim) editing mask, 0 = known/condition, 1 = generate.
    #   _gt_motion_135: (T, 135) ground-truth target the model was asked to match
    #     (equals pred for no-GT tasks like T2M / free transition regions).
    try:
        metrics['_gt_motion_135'] = np.asarray(gt_motion_135, dtype=np.float32)
        metrics['_src_mask'] = np.asarray(mask, dtype=np.float32)
    except Exception:
        pass
    # Expose transition/prepend/loop layout so save_npz path can embed it
    # and the dashboard can cut gray context at the correct frame.
    if _layout is not None:
        metrics['_layout'] = _layout
    if keyframe_indices is not None:
        try:
            metrics['_keyframe_indices'] = [int(x) for x in
                                            np.asarray(keyframe_indices).reshape(-1).tolist()]
        except Exception:
            pass
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
                        help='Task IDs to evaluate (E1-E16)')
    parser.add_argument('--all-tasks', action='store_true',
                        help='Run all registered tasks')
    parser.add_argument('--settings', nargs='+',
                        help='Sub-settings to run (A, B, C, D, default)')
    parser.add_argument('--max-samples', type=int, default=50,
                        help='Max samples per task')
    parser.add_argument('--num-steps', type=int, default=50,
                        help='ODE integration steps')
    parser.add_argument('--replacement-guidance', type=str, default='skip_last',
                        choices=['none', 'all', 'skip_last', 'flow_interp'],
                        help='Replacement guidance mode for MAN imputation')
    parser.add_argument('--text-guidance-scale', type=float, default=1.0,
                        help='CFG scale for text-conditioned M2M models. '
                             'Default 1.0 avoids over-guidance artifacts in '
                             'motion-conditioned completion/editing tasks.')
    parser.add_argument('--output-dir', type=str,
                        default='work_dirs/m2m_v2_eval_report')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save-npz', action='store_true',
                        help='Save output NPZ files for visualization')
    parser.add_argument('--motion-data-dir', type=str, default=MOTION_DATA_DIR)
    parser.add_argument('--data-file-override', type=str, default=None,
                        help='If set, replaces task.data_file for ALL tasks '
                             'with this filename (still resolved under '
                             'EVAL_DATA_DIR / EVAL_DATA_DIR_LEGACY). Used '
                             'for ablation runs where you want to point eval '
                             'at a custom datalist without editing m2m_eval_tasks.py.')
    parser.add_argument('--use-rewritten', action='store_true',
                        help='Prefer the rewritten datalist variant '
                             '(eval_e*_rewritten.json) for caption-carrying '
                             'tasks. Produced by scripts/rewrite_eval_captions.py.')
    parser.add_argument('--run-caption-nonaware', action='store_true',
                        help='Also run caption models on tasks marked as not '
                             'caption-aware, using the task inputs without '
                             'semantic caption conditioning.')
    parser.add_argument('--allow-uncond-caption-required', action='store_true',
                        help='Allow unconditioned models to run settings that '
                             'normally require captions. The caption is loaded '
                             'for bookkeeping but ignored by the model.')
    parser.add_argument('--include-disabled-settings', action='store_true',
                        help='Run settings marked _disabled in the task registry. '
                             'Use only for explicit backfill/rerun jobs.')
    parser.add_argument('--caption-override-mode',
                        choices=['none', 'blank', 'shuffle'],
                        default='none',
                        help='Diagnostic only: override loaded sample captions '
                             'after caption-required filtering. "blank" forces '
                             'null text; "shuffle" deterministically rotates '
                             'captions across samples to test caption sensitivity.')
    parser.add_argument('--seed-base', type=lambda x: int(x, 0),
                        default=0xE4A10000,
                        help='Base seed for per-sample random state '
                             '(seed = seed_base + sample_idx). '
                             'Default: 0xE4A10000. Change to get different samples.')
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

            # NOTE 2026-04-25: per-task caption skip moved INTO the
            # setting loop so per-setting `use_caption` overrides can take
            # effect (E2 v2 introduces pre20_uncond/post20_uncond/
            # mid60_uncond — these MUST run for uncond models even though
            # the task itself has needs_caption=True).

            # Skip caption-enabled models on tasks that are not caption-aware
            # (e.g. E9 Motion Repair, E14 Transition — no semantic text). These
            # tasks produced visibly distorted outputs from caption models in
            # earlier eval rounds, and caption does not add value.
            task_caption_aware = getattr(task, 'caption_aware', True)
            if ((not task_caption_aware) and model_info.get('has_caption', False)
                    and not args.run_caption_nonaware):
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
                # 2026-04-22: skip settings explicitly marked as disabled in
                # the registry (e.g. E9 D_qc_mask_* after user reported the
                # QC invalid_mask is inaccurate, or E9 B_post_replace after
                # user reported Stage 1 pure-noise generation ignored LQ
                # entirely). Kept as stubs so explicit `--settings` runs
                # still resolve but are no-ops.
                if (task.settings[setting_name].mask_kwargs.get('_disabled', False)
                        and not args.include_disabled_settings):
                    print(f'\n  Task: {task_key} — SKIPPED (setting marked '
                          f'_disabled in registry)')
                    continue

                # ── Per-setting caption policy (THREE-state, 2026-04-26) ──
                # _setting_uc:
                #   True  -> REQUIRE   : caption is mandatory; skip uncond models;
                #                        caption-aware models pass caption through.
                #   False -> FORCE_BLANK: blank captions for ALL models (used by
                #                        the *_uncond twin settings to isolate
                #                        the marginal value of caption on a
                #                        caption-aware model at fixed mask).
                #   None  -> INHERIT from task.needs_caption:
                #                        True  -> REQUIRE
                #                        False -> NEUTRAL: caption-aware models
                #                                pass caption through if present;
                #                                uncond models simply ignore it.
                # NOTE: Earlier code conflated NEUTRAL with FORCE_BLANK, which
                # silently zeroed captions for all caption-aware runs on
                # E2 pre20/post20/mid60 and E3 *. That is fixed here.
                _setting_uc = getattr(task.settings[setting_name],
                                      'use_caption', None)
                if _setting_uc is True:
                    caption_policy = 'require'
                elif _setting_uc is False:
                    caption_policy = 'blank'
                elif task.needs_caption:
                    caption_policy = 'require'
                else:
                    caption_policy = 'neutral'

                # Skip uncond models on settings that REQUIRE caption.
                if (caption_policy == 'require' and
                        not model_info.get('has_caption', False) and
                        not args.allow_uncond_caption_required):
                    print(f'\n  Task: {task_key} — SKIPPED for {model_name} '
                          f'(requires caption; setting.use_caption=True)')
                    continue
                print(f'\n  Task: {task_key} — {task.name} ({task.settings[setting_name].description})')

                # Load evaluation data.
                # Priority (highest first):
                #   1. setting.mask_kwargs['_data_file'] — per-setting override
                #      (e.g. E14 L uses static50, M uses move50)
                #   2. {base}_rewritten.json  (if --use-rewritten)
                #   3. {EVAL_DATA_DIR}/{data_file}
                #   4. {EVAL_DATA_DIR_LEGACY}/{data_file}
                _setting_obj = task.settings[setting_name]
                _per_setting_df = _setting_obj.mask_kwargs.get('_data_file')
                if args.data_file_override:
                    effective_data_file = args.data_file_override
                elif _per_setting_df:
                    effective_data_file = _per_setting_df
                else:
                    effective_data_file = task.data_file

                eval_file = None
                if args.use_rewritten:
                    base = os.path.splitext(effective_data_file)[0]
                    rewritten_file = os.path.join(EVAL_DATA_DIR, base + '_rewritten.json')
                    if os.path.exists(rewritten_file):
                        eval_file = rewritten_file
                if eval_file is None:
                    eval_file = os.path.join(EVAL_DATA_DIR, effective_data_file)
                    if not os.path.exists(eval_file):
                        eval_file = os.path.join(EVAL_DATA_DIR_LEGACY, effective_data_file)
                if not os.path.exists(eval_file):
                    print(f'    WARNING: eval file not found: {effective_data_file}')
                    continue
                if args.use_rewritten and not eval_file.endswith('_rewritten.json'):
                    print(f'    [note] no rewritten datalist for {task.task_id}, '
                          f'falling back to {os.path.basename(eval_file)}')

                samples = load_eval_samples(
                    eval_file,
                    args.motion_data_dir,
                    args.max_samples,
                    require_caption=(caption_policy == 'require'),
                    bone_offsets=bone_offsets if convert_198 else None,
                    convert_to_198=convert_198,
                    task_id=task.task_id,
                )
                print(f'    Loaded {len(samples)} samples '
                      f'(caption_policy={caption_policy})')

                # ── Force-blank caption ONLY for explicit FORCE_BLANK settings ──
                # NEUTRAL policy keeps captions intact for caption-aware models
                # (uncond models inherently ignore them). This restores the
                # pre20/post20/mid60 vs pre20_uncond/post20_uncond/mid60_uncond
                # comparison that earlier conflation broke.
                if caption_policy == 'blank':
                    for _s in samples:
                        _s['caption'] = ''

                if not samples:
                    print('    WARNING: No valid samples!')
                    continue

                if args.caption_override_mode != 'none':
                    if args.caption_override_mode == 'blank':
                        for _s in samples:
                            _s['caption_original'] = _s.get('caption', '')
                            _s['caption'] = ''
                    elif args.caption_override_mode == 'shuffle':
                        caps = [_s.get('caption', '') for _s in samples]
                        if len(caps) > 1:
                            shuffled = caps[1:] + caps[:1]
                        else:
                            shuffled = caps
                        for _s, _cap in zip(samples, shuffled):
                            _s['caption_original'] = _s.get('caption', '')
                            _s['caption'] = _cap
                    print(f'    [caption_override] mode={args.caption_override_mode}')

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
                        # 2026-04-22: seed per-sample deterministically so the
                        # SAME (sample_idx, task, setting) always produces the
                        # SAME output across runs, and across models (the seed
                        # depends only on sample_idx, not on model). Previously
                        # torch.randn inside pipeline used the global PRNG →
                        # every rerun produced a different motion, which broke
                        # cross-model comparison in the dashboard (switching
                        # models on the same case showed different generations).
                        seed = args.seed_base + i
                        torch.manual_seed(seed)
                        if torch.cuda.is_available():
                            torch.cuda.manual_seed_all(seed)
                        np.random.seed(seed & 0xFFFFFFFF)
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
                                # Add layout metadata (2026-04-23) so the
                                # dashboard can cut gray prefix/suffix at the
                                # correct *dynamic* frame count instead of
                                # using the old hard-coded 5/15/30.
                                _save_kw = dict(
                                    motion_135=output_135,
                                    positions=pos_np,
                                    translation=output_135[:, :3],
                                )
                                if 'source_motion' in sample:
                                    _save_kw['source_motion_135'] = np.asarray(
                                        sample['source_motion'], dtype=np.float32)
                                    _save_kw['source_translation'] = np.asarray(
                                        sample['source_motion'][:, :3],
                                        dtype=np.float32)
                                _layout = metrics.get('_layout', None)
                                if _layout is not None:
                                    # Serialize layout as JSON bytes (uint8
                                    # array) so np.load can read it without
                                    # allow_pickle=True.
                                    _save_kw['layout_json'] = np.frombuffer(
                                        json.dumps(_layout).encode('utf-8'),
                                        dtype=np.uint8)
                                _kfi = metrics.get('_keyframe_indices', None)
                                if _kfi:
                                    _save_kw['keyframe_indices'] = np.asarray(
                                        _kfi, dtype=np.int32)
                                # Multi-task mesh viewer extras: GT target,
                                # editing mask (0=condition/known, 1=generate),
                                # caption text, and the task/setting tag so the
                                # viewer can render condition / GT / pred meshes
                                # with per-joint condition coloring.
                                _gt = metrics.get('_gt_motion_135', None)
                                if _gt is not None:
                                    _save_kw['gt_motion_135'] = np.asarray(
                                        _gt, dtype=np.float32)
                                _sm = metrics.get('_src_mask', None)
                                if _sm is not None:
                                    _save_kw['src_mask'] = np.asarray(
                                        _sm, dtype=np.float32)
                                _cap = sample.get('caption', '') or ''
                                _save_kw['caption'] = np.array(str(_cap))
                                _save_kw['task_key'] = np.array(str(task_key))
                                np.savez_compressed(npz_path, **_save_kw)
                                metrics['_npz_path'] = npz_path
                            except Exception:
                                pass
                        # Drop the big GT/mask arrays before the metrics dict is
                        # JSON-serialized for DB import (they would bloat the
                        # eval_v2_*.json by T*333 floats per sample).
                        metrics.pop('_gt_motion_135', None)
                        metrics.pop('_src_mask', None)

                        # Store sample info for DB import
                        metrics['_sample_idx'] = i
                        # For E13 multi-prompt: _caption = " | "-joined chain
                        # so legacy single-caption readers still see something
                        # meaningful. The structured list lives in
                        # metrics['_segment_captions'].
                        if '_segment_captions' in metrics:
                            metrics['_caption'] = ' | '.join(
                                metrics['_segment_captions'])
                        else:
                            metrics['_caption'] = sample.get('caption', '')
                        if '_num_frames' not in metrics:
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

    # ---- Quality-signal auto-flags (2026-04-23) ----
    # After every eval, cross-check against the canonical-input failure-mode
    # thresholds in hftrainer/models/motion/CLAUDE.md. Flags in the console
    # help catch canonicalization / N_cond-truncation regressions early.
    _QUALITY_THRESHOLDS = {
        # task_id: {setting_pattern_substring: {metric: max_ok}}
        'E14': {'*': {'jitter_pos': 1000.0, 'foot_skating_ratio': 0.35,
                     'boundary_accel_jump': 70.0}},
        'E15': {'*': {'jitter_pos': 800.0, 'foot_skating_ratio': 0.30,
                     'boundary_accel_jump': 8.0}},
        'E8':  {'D':  {'loop_position_error': 0.05, 'jitter_pos': 500.0,
                       'foot_skating_ratio': 0.30}},
    }

    print(f'\n\n{"=" * 80}')
    print('QUALITY AUTO-FLAGS (canonical-input regression detector)')
    print(f'{"=" * 80}')
    any_flag = False
    for task_id in task_ids:
        if task_id not in _QUALITY_THRESHOLDS:
            continue
        rules = _QUALITY_THRESHOLDS[task_id]
        task = get_task(task_id)
        settings_run = args.settings or list(task.settings.keys())
        for setting_name in settings_run:
            if setting_name not in task.settings:
                continue
            # Pick matching rule (most specific first, then '*' wildcard)
            thresholds = None
            for pat, th in rules.items():
                if pat != '*' and pat in setting_name:
                    thresholds = th; break
            if thresholds is None:
                thresholds = rules.get('*')
            if not thresholds:
                continue
            for m_name in args.models:
                task_key = f'{task_id}_{setting_name}'
                agg = all_results.get(m_name, {}).get('tasks', {}).get(
                    task_key, {}).get('aggregated', {})
                for metric, limit in thresholds.items():
                    if metric not in agg:
                        continue
                    val = agg[metric].get('mean')
                    if val is None:
                        continue
                    if val > limit:
                        any_flag = True
                        print(f'  ⚠️  {m_name:15s} {task_key:25s} '
                              f'{metric}={val:.4f} > {limit:.4f}  '
                              f'(possible canonical/cond bug?)')
    if not any_flag:
        print('  ✓ All metrics within expected bounds. No regressions detected.')
    print('  (Thresholds defined at the top of '
          'hftrainer/models/motion/CLAUDE.md §Quality-signal thresholds)')


if __name__ == '__main__':
    main()
