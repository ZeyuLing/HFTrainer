"""Canonical motion-repair mask utilities (shared by the pipeline and eval).

This module is the single source of truth for the adaptive-mask post-processing
used by HyMotion-M2M motion repair. Both
:meth:`hftrainer.pipelines.motion.hymotion_m2m_pipeline.HyMotionM2MPipeline.infer_repair`
and the offline eval (``scripts/eval/eval_m2m_v2_all_tasks.py``) import from here,
so there is exactly one place that defines "how a raw defect mask is tightened".

All functions are pure numpy and operate on the 135/198-dim motion layout:

    dim 0:3     translation XYZ        (pelvis / joint 0 group)
    dim 3:135   rot6d, 22 joints * 6   (joints 0..21)
    dim 135:198 pos, 21 joints * 3     (joints 1..21, 198-dim only)

Mask convention everywhere: ``1 = generate/regenerate``, ``0 = keep (lock to LQ)``.
"""

from __future__ import annotations

from typing import List, Optional, Set

import numpy as np

# SMPL-22 kinematic parents (─1 = root). Used for the kinematic-neighbour
# spatial dilation of the defect mask.
SMPL22_PARENTS: List[int] = [
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

# Upper-chain small joints excluded from BOTH emitting and receiving spatial
# propagation. The change detector often fires on the head from small pose
# noise; kinematic propagation would then drag the neck along and the model
# invents a whole new head/neck rotation not present in the LQ input.
DEFAULT_NO_PROPAGATE: Set[int] = {12, 13, 14, 15, 20, 21}  # neck, collars, head, wrists


def motion_135_to_198(
    motion_135: np.ndarray,           # (T, 135) local transl(3) + rot6d(132)
    bone_offsets: np.ndarray,         # (22, 3) SMPL-22 bone offsets
) -> np.ndarray:
    """Expand 135-dim motion to the 198-dim v2 representation by appending the
    63 FK joint-position channels (21 joints * 3, joints 1..21, no pelvis).

    Positions are pelvis-relative in X/Z and absolute in Y (height), matching
    ``scripts/eval/eval_m2m_v2_all_tasks.motion_135_to_198`` exactly so the
    model's mean/std and channel layout line up with training.
    """
    from hftrainer.evaluation.motion.m2m_eval_metrics import (
        motion135_to_positions_np,
    )
    positions = motion135_to_positions_np(
        motion_135.astype(np.float32), bone_offsets.astype(np.float32))  # (T,22,3)
    T = positions.shape[0]
    joint_pos = positions[:, 1:, :].copy()          # (T,21,3)
    pelvis_xz = positions[:, 0:1, [0, 2]]           # (T,1,2)
    joint_pos[:, :, 0] -= pelvis_xz[:, :, 0]        # X relative to pelvis
    joint_pos[:, :, 2] -= pelvis_xz[:, :, 1]        # Z relative to pelvis
    pos_flat = joint_pos.reshape(T, 63)
    return np.concatenate(
        [motion_135.astype(np.float32), pos_flat.astype(np.float32)], axis=-1)


def smpl22_neighbors() -> List[List[int]]:
    """Symmetric kinematic neighbourhood (parent AND children) per joint."""
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


def compute_strict_adaptive_mask(
    adaptive_raw: np.ndarray,         # (T, D) raw mask, 1=generate
    dilate: int = 2,                  # temporal dilation radius (frames)
    min_blob: int = 3,                # minimum temporal run (frames) to keep
    motion_dim: int = 135,
    lock_trans: bool = False,         # if True, never mask translation (M7 conv)
    no_propagate: Optional[Set[int]] = None,
) -> np.ndarray:
    """Tighten a raw defect mask into a reliable "definitely defective" mask.

    Steps (all in the ``1=generate`` convention):

    1. Per-joint aggregation: a joint at frame t is flagged iff ANY of its
       channels in the raw mask are flagged (raw mask already has per-dim
       dropout, so OR de-noises sensibly).
    2. Kinematic spatial dilation to parent+children joints (a bad joint
       usually drags its neighbours). Upper-chain small joints
       (``no_propagate``) neither emit nor receive propagation.
    3. Temporal dilation by ``±dilate`` frames.
    4. Blob filter: drop per-joint temporal runs shorter than ``min_blob``
       frames (isolated single-frame flags are noise).
    5. Map back to ``(T, D)`` with per-joint-group broadcasting.

    ``lock_trans`` zeroes the translation columns at the end (M7 convention:
    translation is never masked, so the global trajectory stays locked to LQ).
    Spatial dilation can otherwise propagate a hips/spine flag onto the pelvis
    and re-mask translation, letting the model regenerate the global path —
    catastrophic when a clean GT exists (root drifts ~50cm on BrokenAMASS*).
    """
    if no_propagate is None:
        no_propagate = DEFAULT_NO_PROPAGATE
    T, D = adaptive_raw.shape

    # --- Step 1: per-joint aggregation to (T, 22) bool ----------------
    joint_flag = np.zeros((T, 22), dtype=bool)
    trans_any = (adaptive_raw[:, :3] >= 0.5).any(axis=-1)
    joint_flag[:, 0] |= trans_any
    for j in range(22):
        s, e = 3 + j * 6, 3 + (j + 1) * 6
        joint_flag[:, j] |= (adaptive_raw[:, s:e] >= 0.5).any(axis=-1)
    if D >= 198:
        for j in range(1, 22):
            ps, pe = 135 + (j - 1) * 3, 135 + j * 3
            joint_flag[:, j] |= (adaptive_raw[:, ps:pe] >= 0.5).any(axis=-1)

    # --- Step 2: kinematic spatial dilation ---------------------------
    neigh = smpl22_neighbors()
    joint_flag_sp = joint_flag.copy()
    for j in range(22):
        if j in no_propagate:
            continue
        for nb in neigh[j]:
            if nb in no_propagate:
                continue
            joint_flag_sp[:, nb] |= joint_flag[:, j]
    joint_flag = joint_flag_sp

    # --- Step 3: temporal dilation ------------------------------------
    if dilate > 0:
        jf = joint_flag.copy()
        for s in range(1, dilate + 1):
            jf[s:] |= joint_flag[:-s]
            jf[:-s] |= joint_flag[s:]
        joint_flag = jf

    # --- Step 4: blob filter ------------------------------------------
    if min_blob > 1:
        for j in range(22):
            col = joint_flag[:, j]
            if not col.any():
                continue
            i = 0
            while i < T:
                if col[i]:
                    k = i
                    while k < T and col[k]:
                        k += 1
                    if (k - i) < min_blob:
                        col[i:k] = False
                    i = k
                else:
                    i += 1
            joint_flag[:, j] = col

    # --- Step 5: map back to (T, D) -----------------------------------
    out_mask = np.zeros((T, D), dtype=np.float32)
    out_mask[:, :3] = joint_flag[:, 0:1].astype(np.float32)
    for j in range(22):
        s, e = 3 + j * 6, 3 + (j + 1) * 6
        out_mask[:, s:e] = joint_flag[:, j:j + 1].astype(np.float32)
    if D >= 198:
        for j in range(1, 22):
            ps, pe = 135 + (j - 1) * 3, 135 + j * 3
            out_mask[:, ps:pe] = joint_flag[:, j:j + 1].astype(np.float32)

    if lock_trans:
        out_mask[:, :3] = 0.0

    return out_mask


def compute_ada_keep_mask(
    motion_norm_lq: np.ndarray,       # (T, D) normalized LQ
    denoised_stage1: np.ndarray,      # (T, D) normalized stage-1 projection
    threshold_mode: str = 'abs',      # 'abs' or 'topk_pct'
    threshold: float = 0.1,           # abs threshold OR top-k fraction
) -> np.ndarray:
    """Self-detection (ada_denoise Stage-2): build a defect mask from the
    model's own change pattern.

        change      = |motion_lq - denoised_stage1|     (normalized space)
        high_change = change > threshold                → "this cell is defective"

    Per-joint aggregation: a joint is "clean at frame t" iff ALL of its
    channels (rot6d 6 [+ pos 3]) are low-change; otherwise it is flagged.
    Translation (dims 0:3) is treated as a single group.

    Returns ``(T, D)`` mask with 1=generate, 0=keep.
    """
    D = motion_norm_lq.shape[-1]
    change = np.abs(motion_norm_lq - denoised_stage1)

    if threshold_mode == 'abs':
        thr = float(threshold)
    elif threshold_mode == 'topk_pct':
        thr = float(np.quantile(change.ravel(), 1.0 - float(threshold)))
    else:
        raise ValueError(f'Unknown threshold_mode: {threshold_mode!r}')

    low_change_chan = (change <= thr)  # True = clean

    out_mask = np.ones((motion_norm_lq.shape[0], D), dtype=np.float32)

    trans_clean = low_change_chan[:, :3].all(axis=-1)
    out_mask[trans_clean, :3] = 0.0

    for j in range(22):
        s, e = 3 + j * 6, 3 + (j + 1) * 6
        j_clean = low_change_chan[:, s:e].all(axis=-1)
        out_mask[j_clean, s:e] = 0.0

    if D >= 198:
        for j in range(1, 22):
            rot_s, rot_e = 3 + j * 6, 3 + (j + 1) * 6
            j_clean_rot = low_change_chan[:, rot_s:rot_e].all(axis=-1)
            ps, pe = 135 + (j - 1) * 3, 135 + j * 3
            j_clean_pos = low_change_chan[:, ps:pe].all(axis=-1)
            j_clean = j_clean_rot & j_clean_pos
            out_mask[j_clean, ps:pe] = 0.0
            out_mask[j_clean, rot_s:rot_e] = 0.0

    return out_mask


def joint_mask_to_dim_mask(
    joint_flag: np.ndarray,           # (T, 22) bool, 1=defective
    motion_dim: int = 135,
    translation_mode: str = 'lock',   # 'lock' | 'detected' | 'all'
    valid_len: Optional[int] = None,  # frames [valid_len:] forced to keep (0)
) -> np.ndarray:
    """Expand a per-joint defect flag to a ``(T, D)`` generate-mask, applying
    the translation policy.

    translation_mode:
      - 'lock'     : translation never regenerated (cols 0:3 = 0).
      - 'detected' : translation regenerated only where pelvis (joint 0) is
                     flagged.
      - 'all'      : translation regenerated on every valid frame.
    """
    T = joint_flag.shape[0]
    out = np.zeros((T, motion_dim), dtype=np.float32)
    for j in range(22):
        s, e = 3 + j * 6, 3 + (j + 1) * 6
        out[:, s:e] = joint_flag[:, j:j + 1].astype(np.float32)
    if motion_dim >= 198:
        for j in range(1, 22):
            ps, pe = 135 + (j - 1) * 3, 135 + j * 3
            out[:, ps:pe] = joint_flag[:, j:j + 1].astype(np.float32)

    if translation_mode == 'lock':
        out[:, :3] = 0.0
    elif translation_mode == 'detected':
        out[:, :3] = joint_flag[:, 0:1].astype(np.float32)
    elif translation_mode == 'all':
        out[:, :3] = 1.0
    else:
        raise ValueError(f'Unknown translation_mode: {translation_mode!r}')

    if valid_len is not None and valid_len < T:
        out[valid_len:] = 0.0
    return out
