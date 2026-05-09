"""HyMotion M2M v2 evaluation task definitions (E1-E16).

Each task defines:
  - mask_builder: callable(motion, T, D, setting, **kwargs) -> mask (T, D)
  - default_metrics: list of metric names to compute
  - data_file: which eval JSON to use
  - settings: dict of sub-settings (A, B, C, D)
  - needs_gt: whether GT is needed for metrics
  - needs_caption: whether text caption is required

Mask convention: 0=known (keep), 1=generate (model fills in).
Masks are built on 23-group grid then expanded to 135-dim.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.signal import find_peaks

# Import joint group constants from universal_mask
from hftrainer.datasets.motion.motionhub.transforms.universal_mask import (
    NUM_JOINT_GROUPS,
    TRANSL_DIM,
    JOINT_ROT_DIM,
    TOTAL_DIM,
    UPPER_BODY,
    LOWER_BODY,
    LEFT_ARM,
    RIGHT_ARM,
    LEFT_LEG,
    RIGHT_LEG,
    SPINE_HEAD,
    FEET,
    TRANSLATION,
    ALL_JOINTS_NO_TRANSL,
    expand_grid_to_mask,
)


# =====================================================================
# Mask Builders
# =====================================================================

def _grid_to_mask_np(grid: np.ndarray) -> np.ndarray:
    """Convert (T, 23) grid to (T, 135) mask via expand_grid_to_mask.

    Returns numpy array.
    """
    mask_tensor = expand_grid_to_mask(grid)
    return mask_tensor.numpy()


def build_full_mask(T: int, D: int = 135, **kwargs) -> np.ndarray:
    """E1/T2M: all masked (pure generation). mask=1 everywhere."""
    return np.ones((T, D), dtype=np.float32)


def build_inbetween_mask(
    T: int,
    D: int = 135,
    keep_start: int = 5,
    keep_end: int = 5,
    **kwargs,
) -> np.ndarray:
    """E2: In-betweening. Keep first/last N frames, mask middle."""
    grid = np.ones((T, NUM_JOINT_GROUPS), dtype=np.float32)
    grid[:keep_start] = 0
    grid[-keep_end:] = 0
    return _grid_to_mask_np(grid)


def build_keyframe_mask(
    T: int,
    D: int = 135,
    interval: int = 30,
    **kwargs,
) -> np.ndarray:
    """E3: Sparse keyframe interpolation. Keep every interval-th frame."""
    grid = np.ones((T, NUM_JOINT_GROUPS), dtype=np.float32)
    for t in range(0, T, interval):
        grid[t] = 0
    # Always keep last frame
    grid[-1] = 0
    return _grid_to_mask_np(grid)


def build_keyframe_nonuniform_mask(
    T: int,
    D: int = 135,
    seed: int = 42,
    min_gap: int = 10,
    max_gap: int = 90,
    **kwargs,
) -> np.ndarray:
    """E3-D: Non-uniform keyframe spacing."""
    rng = np.random.RandomState(seed)
    grid = np.ones((T, NUM_JOINT_GROUPS), dtype=np.float32)
    grid[0] = 0
    t = 0
    while t < T:
        gap = rng.randint(min_gap, max_gap + 1)
        t += gap
        if t < T:
            grid[t] = 0
    grid[-1] = 0
    return _grid_to_mask_np(grid)


def build_end_effector_mask(
    T: int,
    D: int = 135,
    joint_names: List[str] = None,
    frame_interval: int = 10,
    **kwargs,
) -> Tuple[np.ndarray, Dict]:
    """E4: End-effector position constraint.

    Since M2M v1 uses 135-dim (no position channel), end-effector constraints
    require FK+IK. Here we build a mask that keeps specified joint groups at
    sparse frames and masks everything else.

    Returns:
        mask: (T, 135)
        constraint_info: dict with frames/joints for post-hoc error computation.
    """
    if joint_names is None:
        joint_names = ['r_wrist']

    # Joint name to 23-group index (+1 because group 0 = translation)
    from hftrainer.evaluation.motion.m2m_eval_metrics import JOINT_NAME_TO_IDX
    joint_group_indices = [JOINT_NAME_TO_IDX[name] + 1 for name in joint_names]

    grid = np.ones((T, NUM_JOINT_GROUPS), dtype=np.float32)

    constraint_frames = []
    constraint_joints = []
    for t in range(0, T, frame_interval):
        for jg in joint_group_indices:
            grid[t, jg] = 0
        constraint_frames.extend([t] * len(joint_names))
        constraint_joints.extend([JOINT_NAME_TO_IDX[name] for name in joint_names])

    mask = _grid_to_mask_np(grid)

    constraint_info = {
        'frames': np.array(constraint_frames),
        'joints': np.array(constraint_joints),
        'joint_names': joint_names,
    }
    return mask, constraint_info


def build_text_keypose_mask(
    T: int,
    D: int = 135,
    keep_start: int = 1,
    keep_end: int = 1,
    **kwargs,
) -> np.ndarray:
    """E4-D/E: Text + keypose first/last frame constraint.

    Given a text prompt and a keypose P, use P as first frame and/or last frame.
    The model generates text-guided motion between the two keyposes.

    This is essentially in-betweening with text guidance, where the
    'known' frames come from a selected keypose rather than GT continuation.

    Args:
        keep_start: number of frames to keep at start (0=no start constraint).
        keep_end: number of frames to keep at end (0=no end constraint).
    """
    grid = np.ones((T, NUM_JOINT_GROUPS), dtype=np.float32)
    if keep_start > 0:
        grid[:keep_start] = 0
    if keep_end > 0:
        grid[-keep_end:] = 0
    return _grid_to_mask_np(grid)


def build_trajectory_mask(
    T: int,
    D: int = 135,
    mode: str = 'dense',
    interval: int = 30,
    include_heading: bool = False,
    trans_axes: str = 'xyz',  # kept for compat; only 'xyz' respects training distribution
    **kwargs,
) -> np.ndarray:
    """E5: Trajectory following. Keep the **whole** translation joint-group at
    selected frames; leave everything else generated.

    ⚠️ IMPORTANT: the M2M model was trained with **joint-group** granularity
    (translation group = dims 0..2 all-or-nothing, see docs/motion/CLAUDE.md
    §Inference Practical Guide §1). Masking dims 0 and 2 but not 1 is an
    out-of-distribution pattern the model never saw during training, which
    manifests as floating / jumping results. So we always keep the full XYZ
    translation at condition frames. "XZ-only" semantics should be enforced
    via **post-hoc replacement** of the Y channel after inference, not by a
    per-dim mask.

    Returned mask is (T, D) with 1 = generate, 0 = keep.
    """
    # Always keep the entire translation group (dims 0..2) at condition frames.
    # `trans_axes` is retained in the signature for backward-compat; issuing a
    # warning-like note in the docstring. Any value that drops Y would silently
    # yield OOD behaviour so we simply ignore non-xyz selections.
    keep_dims = [0, 1, 2]

    mask = np.ones((T, D), dtype=np.float32)

    def _keep_trans_at(frame_indices):
        for f in frame_indices:
            for d in keep_dims:
                mask[f, d] = 0.0

    if mode == 'dense':
        _keep_trans_at(range(T))
    elif mode == 'sparse':
        _keep_trans_at(list(range(0, T, interval)) + [T - 1])
    elif mode == 'heading_only':
        # heading = pelvis rotation group, channels 3..8 (6D)
        for t in range(0, T, interval):
            mask[t, 3:9] = 0.0
        mask[-1, 3:9] = 0.0
    elif mode == 'trajectory_heading':
        _keep_trans_at(range(T))
        if include_heading:
            mask[:, 3:9] = 0.0  # pelvis rotation every frame

    return mask


def build_foot_ground_mask(
    T: int,
    D: int = 135,
    contact_frames: Optional[np.ndarray] = None,
    constraint_type: str = 'rotation',
    position_axes: str = 'xyz',
    **kwargs,
) -> np.ndarray:
    """E6: Foot ground constraint.

    Supports two fundamentally different constraint modes:

    1. **rotation** (135-dim): Keep the entire ankle rotation joint group at
       contact frames. This constrains the ankle rotation (rot6d) so that
       the foot orientation matches GT ground contact.

    2. **position** (198-dim): Keep specific position channel dimensions of
       ankle joints at contact frames. This directly constrains the world-space
       position of the ankle. Sub-variants:
       - 'y': only ankle Y position (height=0, XZ free)
       - 'xz': only ankle XZ position (horizontal lock, Y free)
       - 'xyz': full 3D position constraint

    For position constraints, the mask must be 198-dim (not 135-dim) because
    position channels live in dims [135:198]. The mask is built directly at
    198-dim level with per-dim control.

    Args:
        contact_frames: array of frame indices where foot touches ground.
            If None, constrain all frames.
        constraint_type: 'rotation' (135-dim joint group) or 'position' (198-dim per-dim).
        position_axes: for position mode, which axes to constrain: 'y', 'xz', or 'xyz'.
    """
    if constraint_type == 'rotation':
        # 135-dim: constrain ankle rotation joint groups
        grid = np.ones((T, NUM_JOINT_GROUPS), dtype=np.float32)
        # Ankle groups: L_Ankle=8, R_Ankle=9 in 23-group space
        ankle_groups = [8, 9]

        if contact_frames is not None:
            for t in contact_frames:
                if t < T:
                    for ag in ankle_groups:
                        grid[int(t), ag] = 0
        else:
            grid[:, ankle_groups] = 0

        return _grid_to_mask_np(grid)

    elif constraint_type == 'position':
        # 198-dim: directly constrain ankle position channels
        # Position channel layout: dims [135:198] = 21 joints × 3 (XYZ)
        # Joint order in position channel: joints 1-21 (pelvis excluded)
        # L_Ankle = joint 7 -> position index 6 (0-indexed, pelvis excluded)
        # R_Ankle = joint 8 -> position index 7
        l_ankle_pos_start = 135 + 6 * 3  # dim 153
        r_ankle_pos_start = 135 + 7 * 3  # dim 156

        mask = np.ones((T, 198), dtype=np.float32)

        # Determine which dims to constrain per ankle
        constrain_offsets = []
        if position_axes == 'y':
            constrain_offsets = [1]  # Y only
        elif position_axes == 'xz':
            constrain_offsets = [0, 2]  # X and Z
        elif position_axes == 'xyz':
            constrain_offsets = [0, 1, 2]  # all
        else:
            raise ValueError(f"Unknown position_axes: {position_axes}")

        def _set_ankle_mask(mask, frames, start_dim, offsets):
            for t in frames:
                if t < T:
                    for off in offsets:
                        mask[int(t), start_dim + off] = 0

        if contact_frames is not None:
            _set_ankle_mask(mask, contact_frames, l_ankle_pos_start, constrain_offsets)
            _set_ankle_mask(mask, contact_frames, r_ankle_pos_start, constrain_offsets)
        else:
            for off in constrain_offsets:
                mask[:, l_ankle_pos_start + off] = 0
                mask[:, r_ankle_pos_start + off] = 0

        return mask

    else:
        raise ValueError(f"Unknown constraint_type: {constraint_type}")


def build_first_frame_mask(
    T: int,
    D: int = 135,
    **kwargs,
) -> np.ndarray:
    """E7: First-frame continuation. Keep frame 0, mask rest."""
    grid = np.ones((T, NUM_JOINT_GROUPS), dtype=np.float32)
    grid[0] = 0  # keep first frame
    return _grid_to_mask_np(grid)


def build_loop_mask(
    T: int,
    D: int = 135,
    trajectory_mode: str = 'none',
    waypoint_interval: int = 30,
    **kwargs,
) -> np.ndarray:
    """E8: Loop animation. Keep first and last frame (same pose).

    Supports additional trajectory constraints:
    - 'none': only first=last frame constraint (classic loop)
    - 'dense': every frame has root XZ trajectory constraint (dense loop trajectory)
    - 'waypoints': sparse waypoints at every interval frames (sparse loop trajectory)

    For trajectory modes, translation group is additionally constrained.
    """
    grid = np.ones((T, NUM_JOINT_GROUPS), dtype=np.float32)
    grid[0] = 0
    grid[-1] = 0

    if trajectory_mode == 'dense':
        # Dense trajectory: constrain translation every frame
        grid[:, 0] = 0  # translation group
    elif trajectory_mode == 'waypoints':
        # Sparse waypoints
        for t in range(0, T, waypoint_interval):
            grid[t, 0] = 0
        grid[-1, 0] = 0  # ensure last frame translation is also kept
    elif trajectory_mode != 'none':
        raise ValueError(f"Unknown trajectory_mode: {trajectory_mode}")

    return _grid_to_mask_np(grid)


def build_loop_completion_mask(
    T_total: int,
    D: int = 135,
    T_gt: int = 0,
    N_append: int = 30,
    **kwargs,
) -> np.ndarray:
    """E8-B/C/D: Loop completion mask.

    Given T_gt frames of GT (condition) + N_append-1 frames to generate + 1 frame
    target (first frame pose, as loop constraint).

    Mask layout: [0,0,...,0, 1,1,...,1, 0]
      - GT portion (T_gt frames) = 0 (condition)
      - Appended N_append-1 frames = 1 (generate)
      - Last 1 frame = 0 (target = first frame pose)

    Total length T_total = T_gt + N_append.
    """
    grid = np.zeros((T_total, NUM_JOINT_GROUPS), dtype=np.float32)
    # GT portion: already 0
    # Appended frames to generate: T_gt to T_gt+N_append-2 (inclusive)
    if N_append > 1:
        grid[T_gt:T_gt + N_append - 1] = 1.0
    # Last frame = 0 (loop target = first frame)
    # grid[-1] is already 0
    return _grid_to_mask_np(grid)


def compute_transition_length(
    pos_a_end: np.ndarray,
    pos_b_start: np.ndarray,
    speed_per_frame: float = 0.015,
    min_frames: int = 30,
    max_frames: int = 120,
) -> int:
    """Compute adaptive transition length based on root position displacement.

    Args:
        pos_a_end: (3,) root position of the last frame of motion A.
        pos_b_start: (3,) root position of the first frame of motion B.
        speed_per_frame: assumed normal walking speed in meters/frame (~0.015m/frame at 30fps).
        min_frames: minimum transition length (1s at 30fps).
        max_frames: maximum transition length (4s at 30fps).

    Returns:
        Transition length in frames.
    """
    dist = np.linalg.norm(pos_a_end - pos_b_start)
    raw_frames = int(dist / speed_per_frame)
    return max(min_frames, min(max_frames, raw_frames))


def build_transition_mask(
    T_total: int,
    D: int = 135,
    N_cond_a: int = 15,
    N_transition: int = 30,
    N_cond_b: int = 15,
    **kwargs,
) -> np.ndarray:
    """E14: Transition stitching mask.

    Sequence layout: [A_tail(N_cond_a) | transition(N_transition) | B_head(N_cond_b)]
    Mask: A_tail=0, transition=1, B_head=0.
    """
    grid = np.zeros((T_total, NUM_JOINT_GROUPS), dtype=np.float32)
    # Transition region: from N_cond_a to N_cond_a + N_transition - 1
    grid[N_cond_a:N_cond_a + N_transition] = 1.0
    return _grid_to_mask_np(grid)


def build_transition_to_target_first_mask(
    T_total: int,
    D: int = 135,
    N_cond_tail: int = 15,
    N_transition: int = 30,
    **kwargs,
) -> np.ndarray:
    """E15: Transition to target first frame mask.

    Sequence layout: [motion_tail(N_cond_tail) | transition(N_transition) | target_first(1)]
    Mask: motion_tail=0, transition=1, target_first=0.
    """
    grid = np.zeros((T_total, NUM_JOINT_GROUPS), dtype=np.float32)
    # Transition region: from N_cond_tail to N_cond_tail + N_transition - 1
    grid[N_cond_tail:N_cond_tail + N_transition] = 1.0
    return _grid_to_mask_np(grid)


def build_transition_to_target_last_mask(
    T_total: int,
    D: int = 135,
    N_transition: int = 30,
    N_cond_head: int = 15,
    **kwargs,
) -> np.ndarray:
    """E16: Transition from target last frame mask.

    Sequence layout: [target_last(1) | transition(N_transition) | motion_head(N_cond_head)]
    Mask: target_last=0, transition=1, motion_head=0.
    """
    grid = np.zeros((T_total, NUM_JOINT_GROUPS), dtype=np.float32)
    # Transition region: from 1 to 1 + N_transition - 1
    grid[1:1 + N_transition] = 1.0
    return _grid_to_mask_np(grid)


def build_repair_mask(
    T: int,
    D: int = 135,
    defect_mask: Optional[np.ndarray] = None,
    **kwargs,
) -> np.ndarray:
    """E9: Repair mask from checker-detected defects.

    If defect_mask is provided (T, 23 grid or T, 135 full), use it.
    Otherwise default to full mask (regenerate everything).
    """
    if defect_mask is not None:
        if defect_mask.shape[-1] == NUM_JOINT_GROUPS:
            return _grid_to_mask_np(defect_mask)
        elif defect_mask.shape[-1] == D:
            return defect_mask.astype(np.float32)
    return np.ones((T, D), dtype=np.float32)


def build_part_level_mask(
    T: int,
    D: int = 135,
    keep_part: str = 'upper',
    **kwargs,
) -> np.ndarray:
    """E10: Part-level control. Keep specified body part, regenerate rest.

    Args:
        keep_part: 'upper' (keep upper body, regen lower) or
                   'lower' (keep lower body, regen upper) or
                   'spine_only' (keep pelvis + spine chain + head, regen 4 limbs) or
                   'root_only' (legacy: keep translation + pelvis, regen all pose).
    """
    grid = np.ones((T, NUM_JOINT_GROUPS), dtype=np.float32)

    if keep_part == 'upper':
        # Keep upper body (including translation, spine, arms, head)
        for g in UPPER_BODY:
            grid[:, g] = 0
        # Also keep pelvis rotation (group 1) for coherent root
        grid[:, 1] = 0
    elif keep_part == 'lower':
        # Keep lower body
        for g in LOWER_BODY:
            grid[:, g] = 0
        grid[:, 0] = 0  # keep translation
        grid[:, 1] = 0  # keep pelvis
    elif keep_part == 'spine_only':
        # Keep: translation + pelvis + spine chain + head (SPINE_HEAD groups)
        # SPINE_HEAD already contains Spine1/2/3 + Neck + Head (in group indexing).
        # Regen: all 4 limbs (hips, knees, ankles, feet, shoulders, elbows,
        # wrists, and the clavicles). A distinct and challenging part-level
        # control scenario that does NOT overlap with E5 (trajectory).
        grid[:, 0] = 0   # translation
        grid[:, 1] = 0   # pelvis
        for g in SPINE_HEAD:
            grid[:, g] = 0
    elif keep_part == 'root_only':
        # Legacy setting (kept for backward compat with old DB rows).
        # Current registry uses C_spine_only instead; this branch should only
        # be hit for historical results imported before the 2026-04-20 rename.
        grid[:, 0] = 0  # translation
        grid[:, 1] = 0  # pelvis rotation
    else:
        raise ValueError(f"Unknown keep_part: {keep_part}")

    return _grid_to_mask_np(grid)


def build_multi_prompt_mask(
    T: int,
    D: int = 135,
    overlap_frames: int = 5,
    **kwargs,
) -> np.ndarray:
    """E13: Multi-prompt autoregressive generation.

    For each segment after the first, the mask keeps the first `overlap_frames`
    frames (copied from the tail of the previous segment) and generates the rest.

    For the first segment: mask=all_1 (pure generation).

    Args:
        overlap_frames: number of overlap frames from previous segment tail.
    """
    grid = np.ones((T, NUM_JOINT_GROUPS), dtype=np.float32)
    if overlap_frames > 0:
        grid[:overlap_frames] = 0  # keep overlap region from previous segment
    return _grid_to_mask_np(grid)


# =====================================================================
# Adaptive Keyframe Detection (for E3-D)
# =====================================================================

def detect_keyframes_from_motion(
    motion_135: np.ndarray,
    bone_offsets: np.ndarray,
    min_keyframes: int = 5,
    max_gap: int = 60,
    peak_distance: int = 10,
    sparse: bool = False,
    target_density: float = 1.0 / 30.0,
) -> np.ndarray:
    """Detect keyframes from motion based on joint acceleration peaks.

    Performs FK to get world-space joint positions, then computes per-frame
    acceleration magnitude. Keyframes are placed at local acceleration peaks
    (i.e. where the motion changes most rapidly).

    Two modes:
      - Default (``sparse=False``): always includes first+last frame,
        supplements with uniform spacing to reach ``min_keyframes``, and
        inserts midpoints whenever two consecutive keyframes are more than
        ``max_gap`` apart. This guarantees dense coverage but dilutes the
        "adaptive" nature because uniform filler dominates long sequences.
      - Sparse (``sparse=True``): ONLY returns the top-K frames with the
        highest acceleration peaks. K is chosen as ``max(3, round(T *
        target_density))``. No uniform filler, no midpoint insertion, first/
        last frame not forced. This is what E3-D actually tests — whether
        the model can recover long intermediate segments from a handful of
        motion-salient keyframes.

    Args:
        motion_135: (T, 135) motion representation (transl + rot6d).
        bone_offsets: (22, 3) bone offsets for FK.
        min_keyframes: minimum number of keyframes (dense mode only).
        max_gap: maximum allowed gap between keyframes (dense mode only).
        peak_distance: minimum distance between peaks in find_peaks.
        sparse: if True, use pure peak-only selection (E3-D semantics).
        target_density: fraction of frames to keep as keyframes in sparse
            mode. 1/30 means ~1 keyframe per second at 30 fps.

    Returns:
        keyframe_indices: (K,) sorted array of keyframe frame indices.
    """
    from hftrainer.evaluation.motion.m2m_eval_metrics import motion135_to_positions_np

    T = motion_135.shape[0]
    if T <= 2:
        return np.array([0, T - 1] if T > 1 else [0])

    # FK: motion -> joint positions
    positions = motion135_to_positions_np(motion_135, bone_offsets)  # (T, 22, 3)

    # Compute velocity and acceleration
    velocity = np.diff(positions, axis=0)      # (T-1, 22, 3)
    acceleration = np.diff(velocity, axis=0)   # (T-2, 22, 3)

    # Per-frame acceleration norm: mean across joints of per-joint L2 norm
    acc_norms = np.linalg.norm(acceleration, axis=-1)  # (T-2, 22)
    acc_per_frame = np.mean(acc_norms, axis=-1)         # (T-2,)

    # Find peaks in acceleration signal
    if len(acc_per_frame) > 0:
        peaks, properties = find_peaks(acc_per_frame, distance=peak_distance)
        # Map peak indices back to original frame indices (+1 offset from double diff)
        peak_frames = peaks + 1
    else:
        peak_frames = np.array([], dtype=int)

    # ------------------------------------------------------------------
    # Sparse mode (E3-D): keep ONLY the strongest peaks, no filler.
    # ------------------------------------------------------------------
    if sparse:
        K = max(3, int(round(T * target_density)))
        if len(peak_frames) == 0:
            # No peaks at all → fall back to uniformly sampling K frames
            # across the middle of the clip (not filler on top of peaks).
            return np.linspace(0, T - 1, K, dtype=int)
        # Sort peaks by acceleration magnitude descending, take top K.
        peak_vals = acc_per_frame[peaks]
        order = np.argsort(-peak_vals)
        top_peaks = sorted(int(peak_frames[o]) for o in order[:K])
        return np.array(top_peaks, dtype=int)

    # ------------------------------------------------------------------
    # Dense mode (legacy): peaks + first/last + uniform filler.
    # ------------------------------------------------------------------
    # Always include first and last frame
    keyframes = set([0, T - 1])
    for pf in peak_frames:
        keyframes.add(int(pf))

    # If too few keyframes, supplement with uniform spacing
    if len(keyframes) < min_keyframes:
        n_needed = min_keyframes - len(keyframes)
        uniform_step = max(1, T // (n_needed + 1))
        for i in range(1, n_needed + 1):
            f = i * uniform_step
            if f < T:
                keyframes.add(f)

    # Sort keyframes
    keyframes_sorted = sorted(keyframes)

    # Fill gaps > max_gap with midpoints
    filled = [keyframes_sorted[0]]
    for i in range(1, len(keyframes_sorted)):
        gap = keyframes_sorted[i] - filled[-1]
        while gap > max_gap:
            mid = filled[-1] + gap // 2
            filled.append(mid)
            gap = keyframes_sorted[i] - filled[-1]
        filled.append(keyframes_sorted[i])

    return np.array(sorted(set(filled)), dtype=int)


def build_keyframe_adaptive_mask(
    T: int,
    D: int = 135,
    keyframe_indices: Optional[np.ndarray] = None,
    **kwargs,
) -> np.ndarray:
    """E3-D: Adaptive keyframe mask from detected keyframes.

    Frames at keyframe_indices have grid=0 (condition/keep),
    all other frames have grid=1 (generate).

    Args:
        T: sequence length.
        D: feature dimension (135 or 198).
        keyframe_indices: array of frame indices to keep as keyframes.
    """
    grid = np.ones((T, NUM_JOINT_GROUPS), dtype=np.float32)
    if keyframe_indices is not None:
        for idx in keyframe_indices:
            if 0 <= idx < T:
                grid[idx] = 0
    return _grid_to_mask_np(grid)


# =====================================================================
# Task Definitions
# =====================================================================

class TaskSetting:
    """A specific sub-setting for an evaluation task."""

    def __init__(
        self,
        name: str,
        description: str,
        mask_kwargs: Dict[str, Any],
    ):
        self.name = name
        self.description = description
        self.mask_kwargs = mask_kwargs


class EvalTask:
    """Definition of a single evaluation task (E1-E16)."""

    def __init__(
        self,
        task_id: str,
        name: str,
        description: str,
        mask_builder: Callable,
        data_file: str,
        settings: Dict[str, TaskSetting],
        default_metrics: List[str],
        needs_gt: bool = True,
        needs_caption: bool = False,
        caption_aware: bool = True,
        kimodo_comparable: bool = False,
        is_editing: bool = False,
    ):
        self.task_id = task_id
        self.name = name
        self.description = description
        self.mask_builder = mask_builder
        self.data_file = data_file
        self.settings = settings
        self.default_metrics = default_metrics
        self.needs_gt = needs_gt
        self.needs_caption = needs_caption
        # caption_aware = does this task benefit from text conditioning?
        #   True (default): caption models should run with caption;
        #                   uncond models still run without.
        #   False: the task is text-unrelated (e.g. E9 Motion Repair,
        #          E14 Transition). Caption models SHOULD NOT run at all
        #          here — they use extra capacity on irrelevant semantics
        #          and in practice produce visible distortion (see E9).
        self.caption_aware = caption_aware or needs_caption
        self.kimodo_comparable = kimodo_comparable
        self.is_editing = is_editing

    def build_mask(self, T: int, D: int, setting_name: str, **extra_kwargs) -> Any:
        """Build mask for given setting."""
        setting = self.settings[setting_name]
        kwargs = {**setting.mask_kwargs, **extra_kwargs}
        return self.mask_builder(T, D, **kwargs)


# =====================================================================
# Task Registry
# =====================================================================

def _build_tasks() -> Dict[str, EvalTask]:
    """Build all evaluation tasks (E1-E16)."""

    tasks = {}

    # --- E1: Text-to-Motion ---
    tasks['E1'] = EvalTask(
        task_id='E1',
        name='Text-to-Motion',
        description='Pure generation from text. mask=all_1.',
        mask_builder=build_full_mask,
        data_file='eval_e1_t2m.json',  # Has captions
        settings={
            'default': TaskSetting('default', 'Standard T2M generation', {}),
        },
        default_metrics=['jitter_pos', 'bone_length_cv_mean', 'foot_skating_ratio'],
        needs_gt=False,
        needs_caption=True,
        kimodo_comparable=True,
    )

    # --- E2: Motion In-Betweening ---
    tasks['E2'] = EvalTask(
        task_id='E2',
        name='Motion In-Betweening',
        description='Keep first/last N frames, generate middle.',
        mask_builder=build_inbetween_mask,
        data_file='eval_e2_inbetween.json',
        settings={
            'A': TaskSetting('A', 'Classic: 5+5 frames', {'keep_start': 5, 'keep_end': 5}),
            'B': TaskSetting('B', 'Long-range: 5+5, seq>200', {'keep_start': 5, 'keep_end': 5}),
            'C': TaskSetting('C', 'Asymmetric: 30+5 frames', {'keep_start': 30, 'keep_end': 5}),
            'D': TaskSetting('D', 'Transition stitching: connect two different motions',
                             {'keep_start': 15, 'keep_end': 15,
                              '_use_transition_data': True}),
        },
        default_metrics=[
            'mpjpe_masked', 'mpjpe_unmasked', 'boundary_accel_jump',
            'jitter_pos', 'bone_length_cv_mean', 'foot_skating_ratio',
        ],
        needs_gt=True,
        kimodo_comparable=True,
    )

    # --- E3: Sparse Keyframe Interpolation ---
    tasks['E3'] = EvalTask(
        task_id='E3',
        name='Keyframe Interpolation',
        description='Keep every K-th frame keyframe, interpolate rest.',
        mask_builder=build_keyframe_mask,
        data_file='eval_e3_keyframe.json',
        settings={
            'A': TaskSetting('A', 'Every 30 frames (1s@30fps)', {'interval': 30}),
            'B': TaskSetting('B', 'Every 60 frames (2s)', {'interval': 60}),
            'C': TaskSetting('C', 'Every 15 frames (0.5s)', {'interval': 15}),
            'D': TaskSetting('D', 'Adaptive keyframes from motion acceleration peaks',
                             {'_use_adaptive_keyframes': True}),
        },
        default_metrics=[
            'mpjpe_masked', 'mpjpe_unmasked', 'jitter_pos',
            'bone_length_cv_mean', 'foot_skating_ratio',
        ],
        needs_gt=True,
        kimodo_comparable=True,
    )

    # --- E4: End-Effector Position Constraint ---
    # Purely end-effector: sparse world-position constraints on hand/foot
    # joints at selected frames. Frame-level keypose constraints belong to E7
    # (Bi-directional Pose Completion) and are intentionally NOT duplicated
    # here.
    tasks['E4'] = EvalTask(
        task_id='E4',
        name='End-Effector Constraint',
        description='Constrain end-effector (hand/foot) world positions at '
                    'sparse frames. Pure EE — no full-body keypose constraints.',
        mask_builder=build_end_effector_mask,
        data_file='eval_e4_end_effector.json',
        settings={
            'A_rhand_sparse': TaskSetting(
                'A_rhand_sparse', 'Right hand every 10 frames',
                {'joint_names': ['r_wrist'], 'frame_interval': 10}),
            'B_ankles_sparse': TaskSetting(
                'B_ankles_sparse', 'Both ankles every 15 frames',
                {'joint_names': ['l_ankle', 'r_ankle'], 'frame_interval': 15}),
            'C_rhand_lfoot': TaskSetting(
                'C_rhand_lfoot', 'Right hand + left foot every 15 frames',
                {'joint_names': ['r_wrist', 'l_foot'], 'frame_interval': 15}),
            'D_both_hands': TaskSetting(
                'D_both_hands', 'Both hands every 10 frames',
                {'joint_names': ['l_wrist', 'r_wrist'], 'frame_interval': 10}),
            'E_all4_sparse': TaskSetting(
                'E_all4_sparse',
                'Both hands + both ankles every 20 frames',
                {'joint_names': ['l_wrist', 'r_wrist', 'l_ankle', 'r_ankle'],
                 'frame_interval': 20}),
            'F_rhand_dense': TaskSetting(
                'F_rhand_dense', 'Right hand every 5 frames (dense)',
                {'joint_names': ['r_wrist'], 'frame_interval': 5}),
        },
        default_metrics=[
            'ee_error_mean', 'ee_error_max', 'jitter_pos',
            'bone_length_cv_mean', 'mpjpe_masked',
        ],
        needs_gt=True,
        kimodo_comparable=True,
    )

    # --- E5: Trajectory Following ---
    # Root translation constraints.  XZ-only (planar) is the usual case; XYZ
    # adds vertical anchoring (for jumping/stairs); heading adds pelvis yaw.
    tasks['E5'] = EvalTask(
        task_id='E5',
        name='Trajectory Following',
        description='Follow root translation trajectory. The whole XYZ '
                    'translation group is locked at condition frames '
                    '(training distribution requires joint-group alignment; '
                    'per-axis masking causes OOD floating/jumping). Settings '
                    'vary density and whether pelvis heading is co-conditioned.',
        mask_builder=build_trajectory_mask,
        data_file='eval_e5_trajectory.json',
        settings={
            'A': TaskSetting(
                'A', 'Dense XYZ trajectory (every frame)',
                {'mode': 'dense'}),
            'B': TaskSetting(
                'B', 'Sparse XYZ waypoints (every 30 frames)',
                {'mode': 'sparse', 'interval': 30}),
            'C': TaskSetting(
                'C', 'XYZ trajectory + pelvis heading every frame',
                {'mode': 'trajectory_heading', 'include_heading': True}),
        },
        default_metrics=[
            'trajectory_ade', 'trajectory_fde', 'foot_skating_ratio',
            'jitter_pos', 'bone_length_cv_mean',
        ],
        needs_gt=True,
        kimodo_comparable=True,
    )

    # --- E6: Foot Ground Constraint ---
    # --- E6: Foot Ground Constraint ---
    # Simplified to a single, unambiguous setting: at GT foot-contact frames,
    # lock ankle/foot world position (XYZ). This is the "sticky foot" use case.
    tasks['E6'] = EvalTask(
        task_id='E6',
        name='Foot Ground Constraint',
        description='Lock ankle position (XYZ) at GT foot-contact frames. '
                    '198-dim only — position channels live at 135..197.',
        mask_builder=build_foot_ground_mask,
        data_file='eval_e6_foot_ground.json',
        settings={
            'pos_contact': TaskSetting(
                'pos_contact',
                'Foot-pos-contact: XYZ lock at GT contact frames',
                {'contact_frames': None, 'constraint_type': 'position',
                 'position_axes': 'xyz'}),
        },
        default_metrics=[
            'foot_penetration', 'foot_float', 'foot_skating_ratio',
            'jitter_pos', 'bone_length_cv_mean',
        ],
        needs_gt=True,
        kimodo_comparable=False,
    )

    # --- E7: First-Frame Continuation ---
    # Single setting: given frame 0 + text caption, generate the rest.
    # Historical B_tail / C_both settings have been removed — they were
    # functionally identical to E16 / E15 respectively (same data files,
    # their _keep_last/_keep_first kwargs were never consumed by the driver).
    tasks['E7'] = EvalTask(
        task_id='E7',
        name='First-Frame Continuation',
        description='Keep frame 0 + text caption, generate the rest of the '
                    'motion. For tail-anchor or both-anchor variants use E16 '
                    'and E15 respectively.',
        mask_builder=build_first_frame_mask,
        data_file='eval_e7_first_frame.json',
        settings={
            'default': TaskSetting(
                'default',
                'Head anchor: frame 0 + text \u2192 generate rest',
                {}),
        },
        default_metrics=[
            'mpjpe_unmasked', 'jitter_pos', 'bone_length_cv_mean',
            'foot_skating_ratio',
        ],
        needs_gt=True,
        needs_caption=True,
        kimodo_comparable=False,
    )

    # --- E8: Loop Animation ---
    tasks['E8'] = EvalTask(
        task_id='E8',
        name='Loop Animation',
        description='Generate looping animation. Setting A: pure loop (first=last). '
                    'Settings B/C/D: loop completion — given full GT, append N frames '
                    'to return to first frame pose.',
        mask_builder=build_loop_mask,
        data_file='eval_e8_loop.json',
        settings={
            'A': TaskSetting('A', 'Classic loop (first=last only)', {}),
            'D': TaskSetting('D', 'Loop completion: append 90 frames (3s)',
                             {'_loop_append': 90}),
        },
        default_metrics=[
            'loop_position_error', 'loop_velocity_error',
            'jitter_pos', 'bone_length_cv_mean', 'boundary_accel_jump',
        ],
        needs_gt=False,
        needs_caption=True,
        kimodo_comparable=False,
    )

    # --- E9: Motion Repair ---
    # Three settings combine two axes:
    #   mask source × repair mode
    # A_adaptive_inpaint: MoGenDIT adaptive mask + SDEdit-style partial-noise
    #                     start. Aligns with MoGenDIT's `ada_denoise` path —
    #                     instead of zeroing masked regions, the masked region
    #                     is initialized as `(1-τ)*LQ + τ*noise` (τ = sdedit_tau)
    #                     and the ODE runs from t = 1-τ onward. replacement
    #                     guidance locks unmasked regions to LQ per step.
    #                     No spatial/temporal dilation — the cached MoGenDIT
    #                     mask is used as-is.
    # A_adaptive_edit:    Same adaptive mask, but editing mode — src_motion keeps
    #                     full LQ values and is_editing overrides replacement
    #                     guidance so unmasked regions are NOT locked. Lets the
    #                     model modify the whole sequence when needed.
    # C_full_inpaint:     Ablation upper bound — mask=all_1, full regeneration
    #                     without LQ anchoring.
    tasks['E9'] = EvalTask(
        task_id='E9',
        name='Motion Repair',
        description='Repair defective motion. Adaptive mask from MoGenDIT '
                    '(A_*) or full regeneration (C_*). Adaptive inpaint uses '
                    'SDEdit-style partial-noise start (no zeroing) to align '
                    'with MoGenDIT ada_denoise.',
        mask_builder=build_repair_mask,
        data_file='eval_e9_repair.json',
        settings={
            'A_adaptive_inpaint': TaskSetting(
                'A_adaptive_inpaint',
                'MoGenDIT adaptive mask + SDEdit partial-noise inpainting '
                '(τ=0.5, lock unmasked to LQ)',
                {
                    '_use_adaptive_mask': True,
                    '_editing_mode': False,
                    '_sdedit_tau': 0.5,
                },
            ),
            'A_adaptive_inpaint_notau': TaskSetting(
                'A_adaptive_inpaint_notau',
                'DEBUG: MoGenDIT adaptive mask + inpainting WITHOUT SDEdit '
                '(τ=0; mask=1 regions start from pure noise)',
                {
                    '_use_adaptive_mask': True,
                    '_editing_mode': False,
                    '_sdedit_tau': 0.0,
                },
            ),
            'A_adaptive_edit': TaskSetting(
                'A_adaptive_edit',
                'MoGenDIT adaptive mask + editing (LQ as src, no lock)',
                {
                    '_use_adaptive_mask': True,
                    '_editing_mode': True,
                    '_sdedit_tau': 0.0,
                },
            ),
            'C_full_inpaint': TaskSetting(
                'C_full_inpaint',
                'Full regeneration (mask=all_1) — upper bound ablation',
                {
                    '_use_adaptive_mask': False,
                    '_editing_mode': False,
                    '_sdedit_tau': 0.0,
                },
            ),
        },
        default_metrics=[
            'jitter_pos', 'bone_length_cv_mean', 'foot_skating_ratio',
            'foot_penetration',
        ],
        needs_gt=False,
        caption_aware=False,  # repair is not text-guided; uncond-only
        kimodo_comparable=False,
        # Task-level default is editing (kept for backward compat with any
        # external callers). The v2 driver respects per-setting _editing_mode
        # when present, which overrides this default.
        is_editing=True,
    )

    # --- E10: Part-Level Control ---
    # The mask is applied on the ROTATION channels (rot6d groups for the
    # selected joints) and on the translation group. For 198-dim bundles the
    # position channels (135..197) are NOT explicitly locked — they are
    # implicitly consistent because the model is trained with FK consistency
    # loss, so predicted positions track whatever rot6d + translation it has
    # been constrained to. See docs/motion/CLAUDE.md §Mask Patterns.
    tasks['E10'] = EvalTask(
        task_id='E10',
        name='Part-Level Control',
        description='Rotation-level mask on a body part: keep that part\'s '
                    'rotations (rot6d groups) fixed, regenerate the rest. '
                    'Position channels in 198d are implicit via FK consistency.',
        mask_builder=build_part_level_mask,
        data_file='eval_e10_part_control.json',
        settings={
            'A_upper': TaskSetting(
                'A_upper',
                'Keep upper-body rotations + pelvis, regen lower rotations',
                {'keep_part': 'upper'}),
            'B_lower': TaskSetting(
                'B_lower',
                'Keep lower-body rotations + root translation, regen upper',
                {'keep_part': 'lower'}),
            # 2026-04-20: replaced C_root_only (redundant with E5 trajectory)
            # with C_spine_only — a harder, distinct part-level setting.
            'C_spine_only': TaskSetting(
                'C_spine_only',
                'Keep pelvis + spine chain + head rotations, regen all 4 limbs',
                {'keep_part': 'spine_only'}),
        },
        default_metrics=[
            'mpjpe_unmasked', 'jitter_pos', 'bone_length_cv_mean',
            'foot_skating_ratio',
        ],
        needs_gt=True,
        kimodo_comparable=False,
    )

    # --- E11 removed (2026-04-20) ---
    # Previously: Caption-conditioned Completion with (inbetween, keyframe)
    # settings. Each setting was semantically identical to E2-{A,B,C} / E3-{A,B,C}
    # executed with a caption-enabled model (which is already the default for
    # caption_* bundles). We therefore drop E11 to avoid running the same
    # computation twice. Existing rewritten captions cover the E2/E3 datalists.

    # --- E12 removed (2026-04-20) ---
    # Was a placeholder meta-task ("Run all sub-tasks") for Local vs Global
    # ablation; it never had its own data path or metrics. Ablations are
    # expressed by picking different models in the dashboard, not by adding a
    # phantom task that duplicates every other task.

    # --- E13: Multi-Prompt Autoregressive Generation ---
    tasks['E13'] = EvalTask(
        task_id='E13',
        name='Multi-Prompt Generation',
        description='Given N text descriptions, autoregressively generate '
                    'arbitrarily long motion by chaining segments. Each segment '
                    'is conditioned on a text prompt and overlaps with the '
                    'previous segment tail for continuity.',
        mask_builder=build_multi_prompt_mask,
        data_file='eval_e13_multi_prompt.json',  # Use caption data, chain multiple
        settings={
            'A': TaskSetting('A', '3 prompts, 5-frame overlap',
                             {'num_prompts': 3, 'overlap_frames': 5}),
            'B': TaskSetting('B', '5 prompts, 5-frame overlap',
                             {'num_prompts': 5, 'overlap_frames': 5}),
            'C': TaskSetting('C', '10 prompts, 10-frame overlap (long sequence)',
                             {'num_prompts': 10, 'overlap_frames': 10}),
        },
        default_metrics=[
            'jitter_pos', 'bone_length_cv_mean', 'foot_skating_ratio',
            'segment_boundary_smoothness', 'total_duration',
        ],
        needs_gt=False,
        needs_caption=True,
        kimodo_comparable=False,
    )

    # --- E14: Transition Stitching ---
    tasks['E14'] = EvalTask(
        task_id='E14',
        name='Transition Stitching',
        description='Given tail of motion A and head of motion B, generate a '
                    'transition segment in between. Transition length is adaptive '
                    'based on root displacement distance.',
        mask_builder=build_transition_mask,
        data_file='eval_e2_transition.json',
        settings={
            'A': TaskSetting('A', 'Minimal context: 5 frames each side',
                             {'_cond_frames': 5, '_use_transition_data': True}),
            'B': TaskSetting('B', 'Medium context: 15 frames each side',
                             {'_cond_frames': 15, '_use_transition_data': True}),
            'C': TaskSetting('C', 'Full context: 30 frames each side',
                             {'_cond_frames': 30, '_use_transition_data': True}),
        },
        default_metrics=[
            'boundary_accel_jump', 'jitter_pos', 'foot_skating_ratio',
            'trajectory_ade',
        ],
        needs_gt=True,
        caption_aware=False,  # transition data has no captions
        kimodo_comparable=True,
    )

    # --- E15: Transition to Target First Frame ---
    tasks['E15'] = EvalTask(
        task_id='E15',
        name='Transition to Target First Frame',
        description='Given current motion tail + target first frame pose, '
                    'generate transition from current motion to the target pose. '
                    'Transition length is adaptive based on root displacement.',
        mask_builder=build_transition_to_target_first_mask,
        data_file='eval_e7_target_d.json',
        settings={
            'A': TaskSetting('A', 'Minimal context: tail 5 frames + target first',
                             {'_cond_tail_frames': 5, '_use_target_first': True}),
            'B': TaskSetting('B', 'Medium context: tail 15 frames + target first',
                             {'_cond_tail_frames': 15, '_use_target_first': True}),
            'C': TaskSetting('C', 'Full context: tail 30 frames + target first',
                             {'_cond_tail_frames': 30, '_use_target_first': True}),
        },
        default_metrics=[
            'mpjpe_last_frame', 'jitter_pos', 'boundary_accel_jump',
        ],
        needs_gt=True,
        kimodo_comparable=False,
    )

    # --- E16: Transition to Target Last Frame ---
    tasks['E16'] = EvalTask(
        task_id='E16',
        name='Transition from Target Last Frame',
        description='Given target last frame pose + subsequent motion head, '
                    'generate transition from target pose to subsequent motion. '
                    'Transition length is adaptive based on root displacement.',
        mask_builder=build_transition_to_target_last_mask,
        data_file='eval_e7_target_c.json',
        settings={
            'A': TaskSetting('A', 'Minimal context: target last + head 5 frames',
                             {'_cond_head_frames': 5, '_use_target_last': True}),
            'B': TaskSetting('B', 'Medium context: target last + head 15 frames',
                             {'_cond_head_frames': 15, '_use_target_last': True}),
            'C': TaskSetting('C', 'Full context: target last + head 30 frames',
                             {'_cond_head_frames': 30, '_use_target_last': True}),
        },
        default_metrics=[
            'mpjpe_first_frame', 'jitter_pos', 'boundary_accel_jump',
        ],
        needs_gt=True,
        kimodo_comparable=False,
    )

    return tasks


# Module-level task registry
EVAL_TASKS: Dict[str, EvalTask] = _build_tasks()


def get_task(task_id: str) -> EvalTask:
    """Get task definition by ID (E1-E16)."""
    if task_id not in EVAL_TASKS:
        raise KeyError(f"Unknown task: {task_id}. Available: {list(EVAL_TASKS.keys())}")
    return EVAL_TASKS[task_id]


def list_tasks() -> List[str]:
    """List all task IDs."""
    return list(EVAL_TASKS.keys())


# =====================================================================
# Detect foot contact frames from GT motion
# =====================================================================

def detect_foot_contact_frames(
    positions: np.ndarray,
    ground_y: float = 0.0,
    contact_threshold: float = 0.05,
) -> np.ndarray:
    """Detect frames where foot is in contact with ground.

    Args:
        positions: (T, 22, 3) world-space joint positions.
        ground_y: ground Y coordinate.
        contact_threshold: height threshold for contact.

    Returns:
        contact_frames: (N,) array of frame indices.
    """
    ankle_foot_idx = [7, 8, 10, 11]  # L/R ankle, L/R foot
    foot_y = positions[:, ankle_foot_idx, 1]  # (T, 4)
    min_foot_y = foot_y.min(axis=1)  # (T,)
    contact = min_foot_y < ground_y + contact_threshold
    return np.where(contact)[0]


# =====================================================================
# Helper: extract end-effector GT positions from motion
# =====================================================================

def extract_ee_constraints_from_gt(
    gt_positions: np.ndarray,
    joint_names: List[str],
    frame_interval: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract end-effector world positions from GT motion for constraint.

    Args:
        gt_positions: (T, 22, 3) GT world positions.
        joint_names: list of joint names to constrain.
        frame_interval: every N frames.

    Returns:
        constraint_positions: (N, 3)
        constraint_frames: (N,)
        constraint_joints: (N,)
    """
    from hftrainer.evaluation.motion.m2m_eval_metrics import JOINT_NAME_TO_IDX

    positions = []
    frames = []
    joints = []

    T = gt_positions.shape[0]
    for t in range(0, T, frame_interval):
        for name in joint_names:
            j = JOINT_NAME_TO_IDX[name]
            positions.append(gt_positions[t, j])
            frames.append(t)
            joints.append(j)

    return (
        np.array(positions),
        np.array(frames),
        np.array(joints),
    )
