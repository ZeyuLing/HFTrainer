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
    keep_start: int = 0,
    keep_end: int = 0,
    keep_start_frac: Optional[float] = None,
    keep_end_frac: Optional[float] = None,
    **kwargs,
) -> np.ndarray:
    """E2: Motion in-betweening ablation mask.

    Keep first ``keep_start`` and/or last ``keep_end`` frames as known
    context; mask the rest for generation. When ``keep_start_frac`` /
    ``keep_end_frac`` are provided they override the integer counts and
    are computed as ``ceil(T * frac)`` so the ratio is preserved across
    variable sequence lengths.

    The six v2 settings use this single builder:

    ============= =============== =============== ==========
    setting       keep_start_f    keep_end_f      meaning
    ============= =============== =============== ==========
    start_1f      (1 frame)       -               start pose only
    end_1f        -               (1 frame)       end pose only
    both_1f       (1 frame)       (1 frame)       start + end
    pre20         0.20            -               first 20% -> predict rest
    post20        -               0.20            last 20% -> predict rest
    mid60         0.20            0.20            front/back 20% -> predict middle
    ============= =============== =============== ==========
    """
    if keep_start_frac is not None:
        keep_start = int(np.ceil(T * float(keep_start_frac)))
    if keep_end_frac is not None:
        keep_end = int(np.ceil(T * float(keep_end_frac)))
    keep_start = max(0, min(int(keep_start), T))
    keep_end = max(0, min(int(keep_end), T))
    if keep_start + keep_end > T:
        keep_end = max(0, T - keep_start)
    grid = np.ones((T, NUM_JOINT_GROUPS), dtype=np.float32)
    if keep_start > 0:
        grid[:keep_start] = 0
    if keep_end > 0:
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
    """E4: End-effector POSITION constraint — lock the target joint's world
    position at sparse constraint frames.

    Earlier implementation masked the joint's rot6d dims only. That's not
    actually a world-position constraint: the end-effector's world position
    under FK is
        pelvis_trans + parent_chain_rot × bone_offset_chain + wrist_local_rot
    so locking just `wrist_rot6d` lets pelvis (and every intermediate joint)
    drift freely, and the end-effector world position drifts with them.
    Empirically E4 runs (both HyMotion M2M and KIMODO) showed ee_error that
    barely reflected condition at all.

    Fix (2026-04-22): exploit the 198-dim layout to lock
        * pelvis translation (dim[0:3])             → root is frozen
        * target joint position channel             → joint position
          (dim[135 + (j-1)*3 : 135 + j*3],             relative to pelvis
           pelvis excluded from channel layout)        is frozen
    and DO NOT lock rot6d (model is free to rotate as long as FK matches).
    After FK: wrist_world = pelvis_trans + pos_channel which matches GT
    exactly at every constraint frame.

    Returns a 198-dim mask directly so the 135→198 auto-expansion in the
    eval loop (which copies rot6d mask into position channels) is skipped.

    Returns:
        mask: (T, 198) with 1 = generated, 0 = kept as condition.
        constraint_info: dict with frames/joints for post-hoc error
            computation (unchanged so the 3D viewer keeps working).
    """
    if joint_names is None:
        joint_names = ['r_wrist']

    from hftrainer.evaluation.motion.m2m_eval_metrics import JOINT_NAME_TO_IDX

    # 198-dim layout (v2):
    #   [0:3]            pelvis abs translation
    #   [3:135]          22 joint rot6d (pelvis + 21 body joints)
    #   [135:198]        21 body-joint position channels (pelvis excluded),
    #                    ordered joint_idx 1..21. For joint_idx j (1..21):
    #                        pos_start = 135 + (j - 1) * 3
    # For v1 (135-dim) the position channels don't exist — we can only lock
    # the joint's rot6d (the old, weaker behaviour). Fall back for that case.
    if D >= 198:
        return _build_ee_mask_198(T, joint_names, frame_interval)
    return _build_ee_mask_135_legacy(T, joint_names, frame_interval)


def _build_ee_mask_198(T: int, joint_names: List[str], frame_interval: int
                       ) -> Tuple[np.ndarray, Dict]:
    """198-dim E4 end-effector mask (KIMODO-style position imputation).

    **2026-04-22 v2 rewrite**: the earlier version masked pelvis translation
    + EE rot6d + EE pos channel together. That mask pattern is OOD — in
    training (see `condition_sampler_v2.sample_tier2_end_effector`, T2-5
    pattern, active at 10% of Tier-2 = 4% global weight) **only the EE
    position channel is masked, rot6d and pelvis trans stay 1 (generated)**.
    The old pattern produced body-sinking artefacts on cond frames (foot
    min Y -0.16m, pelvis -0.17m vs +0.87m on gen frames) because the model
    was given an unseen mask configuration.

    New pattern (matches T2-5 exactly):
      * EE position channel only:   dim[135 + (j-1)*3 : 135 + j*3]  → mask=0 (keep)
      * Everything else                                             → mask=1 (generate)

    Including:
      - pelvis translation (dim 0:3)        = generate (NOT locked)
      - all joint rot6d (dim 3:135)         = generate
      - other joints' position (dim 135:198) = generate

    Why this works:
      - VACE ``inactive`` channel delivers the GT pos value to the model
        on cond frames for the EE joint's position dims only.
      - Model was trained with exactly this pattern (T2-5) → in-distribution.
      - Model learns to *produce* rot6d that matches the constrained pos
        channel (implicit inverse kinematics), because during training the
        GT rot6d IS what produces the (known) GT pos — so the model has
        seen this pos→rot dependence thousands of times.
      - Pelvis trans is NOT locked → model free to move root, avoiding the
        "body sinks when cond frame forces wrist low" failure mode.

    Caveats:
      - World-space X/Z precision depends on model's ability to place pelvis
        consistently (XZ relative encoding means: if pred pelvis moves, the
        R_Wrist world X/Z moves with it). T2-5 training had this same
        ambiguity, so model should have learnt to pick a stable pelvis.
      - World-space Y is absolute in Scheme D → exact height control.
      - For best world-space X/Z control, can optionally add pelvis-trans
        cond at the same frames (see ``also_lock_pelvis_trans`` flag below)
        — but that departs from T2-5 into OOD territory, use sparingly.

    Returns:
        mask: (T, 198) with 1 = generated, 0 = kept as condition.
        constraint_info: dict with frames/joints for post-hoc error
            computation (unchanged so the 3D viewer keeps working).
    """
    from hftrainer.evaluation.motion.m2m_eval_metrics import JOINT_NAME_TO_IDX
    D_full = 198
    mask = np.ones((T, D_full), dtype=np.float32)

    constraint_frames, constraint_joints = [], []
    for t in range(0, T, frame_interval):
        for name in joint_names:
            j = JOINT_NAME_TO_IDX[name]  # 0..21, 0 = pelvis
            if j == 0:
                # Pelvis has no position channel in Scheme D (redundant
                # with translation). Skip — can't do EE control on root.
                continue
            # ONLY lock the joint's position channel (Scheme D: XZ relative
            # to pelvis, Y absolute). Matches T2-5 training exactly.
            pos_start = 135 + (j - 1) * 3
            mask[t, pos_start: pos_start + 3] = 0
        constraint_frames.extend([t] * len(joint_names))
        constraint_joints.extend([JOINT_NAME_TO_IDX[name] for name in joint_names])

    info = {
        'frames': np.array(constraint_frames),
        'joints': np.array(constraint_joints),
        'joint_names': joint_names,
    }
    return mask, info


def _build_ee_mask_135_legacy(T: int, joint_names: List[str], frame_interval: int
                              ) -> Tuple[np.ndarray, Dict]:
    """Legacy 135-dim fallback: can only lock rot6d (weak constraint).
    Produced by the previous implementation; kept for v1-only models.
    """
    from hftrainer.evaluation.motion.m2m_eval_metrics import JOINT_NAME_TO_IDX

    joint_group_indices = [JOINT_NAME_TO_IDX[name] + 1 for name in joint_names]
    grid = np.ones((T, NUM_JOINT_GROUPS), dtype=np.float32)
    constraint_frames, constraint_joints = [], []
    for t in range(0, T, frame_interval):
        for jg in joint_group_indices:
            grid[t, jg] = 0
        constraint_frames.extend([t] * len(joint_names))
        constraint_joints.extend([JOINT_NAME_TO_IDX[name] for name in joint_names])

    mask = _grid_to_mask_np(grid)
    info = {
        'frames': np.array(constraint_frames),
        'joints': np.array(constraint_joints),
        'joint_names': joint_names,
    }
    return mask, info


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
    trans_axes: str = 'xz',  # v2 training only masks X and Z (see note below)
    **kwargs,
) -> np.ndarray:
    """E5: Trajectory following. Keep pelvis XZ trajectory at selected frames;
    Y (height) is left as generate, matching v2 training distribution.

    ⚠️ Training/inference distribution alignment (2026-04-21):
    M2M v2 training (`condition_sampler_v2.sample_tier2_trajectory` and the
    25% trajectory overlay in `sample_tier2_edit_repair`) **only sets
    mask[f, 0]=0 and mask[f, 2]=0**, i.e. constrains pelvis X and Z.
    Y (dim 1) is always left as generate so the model can adjust height
    freely for natural pose dynamics. Previously this mask also kept Y
    (keep_dims=[0,1,2]) which was an OOD pattern the v2 model never saw
    during training — sparse waypoints flipped between ground-contact and
    floating because Y was fixed on sparse frames but free elsewhere.

    Per-dim inconsistency (Y free while X/Z constrained) IS the actual
    training distribution — the prior note about "joint-group granularity"
    applied to v1 (PrepareM2MUniversalMask) but NOT v2.

    Returned mask is (T, D) with 1 = generate, 0 = keep.
    """
    # v2 training matches: constrain X and Z translation only; leave Y free.
    if trans_axes == 'xyz':
        keep_dims = [0, 1, 2]
    else:
        keep_dims = [0, 2]

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
    joints_a_end: np.ndarray = None,
    joints_b_start: np.ndarray = None,
    pose_speed_per_frame: float = 0.008,
    motion_a_end_135: np.ndarray = None,
    motion_b_start_135: np.ndarray = None,
    joint_angle_speed_per_frame: float = 0.20,
    leg_angle_speed_per_frame: float = 0.12,
) -> int:
    """Compute adaptive transition length.

    Three terms, whichever is largest wins:

      (1) root-displacement term
            N_root = ||pelvis_A_end - pelvis_B_start|| / speed_per_frame
          Captures how far the character has to travel (walk, run, jump).

      (2) joint-position term (FK-based pose reconfiguration in world units)
            1. Remove pelvis translation + yaw (canonical body frame).
            2. Mean per-joint Euclidean distance between aligned A and B.
            3. N_pos = mean(dist) / pose_speed_per_frame
          Catches gross body rearrangement (lie→stand: knees move ~1m).

      (3) joint-angle term (2026-04-23, rotation-based) [PRIMARY for
          posture changes]
            1. For each joint j compute geodesic SO(3) angle between its
               rot6d in A and B: θ_j = acos((trace(R_a^T R_b) - 1) / 2).
            2. Weighted mean across joints (big joints like hips/knees/
               spine weigh more; small joints like wrists/head/fingers
               weigh less). Sit-down / stand-up / lie-down / crouch are
               dominated by hip + knee + spine angles, so weighted sum
               specifically catches these.
            3. N_angle = weighted_mean_angle / joint_angle_speed_per_frame
          Why angle AND position? Positions are confounded by bone lengths
          and can be near-zero when a big joint rotates near its parent
          (e.g., wrist flex). Angles give a direct "how much rotation
          does the body need to produce" measure. For stiff body changes
          (sit, lie, crouch, fall, get up) angles dominate; for locomotion
          angles are small (walking steps are small local oscillations).

      (4) leg-only angle term (2026-04-27, anti foot-skating).
          The whole-body term in (3) underestimates pure-leg complexity
          because the spine/arms/head dilute the budget. For sit-cross-
          legged, kneel, step-and-pivot, short-distance walks etc. the
          legs (hips×2, knees×2, ankles×2, feet×2) move a lot while the
          torso barely moves. We add a *separate* term that only sums
          leg-joint angles, with a tighter rad/frame budget so that 1 rad
          of leg motion alone takes ~8 frames instead of ~5.

          Diagnostic on 50 E14 samples: this term lifts 17/50 (34%)
          samples by a median +0 / p90 +32 / max +48 frames — exactly the
          ones flagged by the user (short root distance + complex leg
          articulation). Without (4) those samples got 30-90 frames; with
          (4) they get 60-120, removing the foot-sliding artefact caused
          by the model being forced to compress a 1.5-2 s leg motion into
          1 s of frames.

    Args:
        pos_a_end / pos_b_start: (3,) pelvis world positions.
        speed_per_frame: root translation speed (~0.015 m/frame = walk).
        min_frames / max_frames: clamp range.
        joints_a_end / joints_b_start: (22, 3) FK joint positions.
        pose_speed_per_frame: per-joint speed (0.008 m/frame ≈ slow limb).
        motion_a_end_135 / motion_b_start_135: (135,) raw 135-dim vectors.
            When provided, enables the joint-angle and leg-angle terms.
        joint_angle_speed_per_frame: whole-body rotation budget per frame
            (rad/frame). 0.20 rad/frame ≈ 6 rad/s, which maps a 10-rad
            sit-down budget to ~50 frames (1.67 s). Raise for quicker
            transitions, lower for slower.
        leg_angle_speed_per_frame: leg-only rotation budget per frame
            (rad/frame). 0.12 rad/frame is calibrated so that
            sit-cross-legged (~17 rad of weighted leg angle) takes ~140
            frames (clamped to 120) and short-step-pivot (~5 rad) takes
            ~40 frames. Tighter than (3) because legs do fine-grained
            contact-sensitive motion that needs more frames to plan.

    Returns:
        Transition length in frames, clamped to [min_frames, max_frames].
    """
    # Term 1: root displacement
    dist = np.linalg.norm(pos_a_end - pos_b_start)
    n_root = int(dist / max(1e-6, speed_per_frame))

    # Term 2: joint position change (FK-space)
    n_pos = 0
    if joints_a_end is not None and joints_b_start is not None:
        try:
            n_pos = _pose_change_frames(
                joints_a_end, joints_b_start, pose_speed_per_frame)
        except Exception:
            n_pos = 0

    # Term 3: joint angle change (rotation-space, weighted by joint size)
    n_angle = 0
    if motion_a_end_135 is not None and motion_b_start_135 is not None:
        try:
            n_angle = _joint_angle_change_frames(
                motion_a_end_135, motion_b_start_135,
                joint_angle_speed_per_frame)
        except Exception:
            n_angle = 0

    # Term 4: leg-only weighted angle SUM (anti foot-skating)
    n_leg_angle = 0
    if motion_a_end_135 is not None and motion_b_start_135 is not None:
        try:
            n_leg_angle = _leg_angle_change_frames(
                motion_a_end_135, motion_b_start_135,
                leg_angle_speed_per_frame)
        except Exception:
            n_leg_angle = 0

    raw = max(n_root, n_pos, n_angle, n_leg_angle)
    return max(min_frames, min(max_frames, raw))


# ---------------------------------------------------------------------------
# Condition frame count (2026-04-23): adaptive N_cond rule
# ---------------------------------------------------------------------------
#
# Empirical findings from 50-sample E14 analysis:
#   - Correlation(cond_jerk, gen_jerk) ≈ 0 → cond quality alone doesn't
#     predict gen quality.
#   - Setting A (N_cond=5+5) wins 40% of samples; B (30+30) 34%; C (60+60)
#     26%. Minimal context is most often best.
#   - Training distribution: `sample_tier2_inbetween` uses
#     randint(1, min(6, T//4)) → max 5 cond frames per side. N_cond > 5 is
#     strictly OOD; N_cond > 30 causes collapse on some samples (clean cond
#     + long context → generation jerk 350-680, vs 150 with A).
#
# Rule design:
#   base = 5  (match training distribution mode)
#   if cond source is noisy (jerk > 3):      base -= 2  (trim bad tail)
#   if transition is long (N_transition>=90): base += 2 (slight extra anchor)
#   hard cap: N_cond <= min(10, T_src // 4)  (training envelope × 2)
#   floor:    N_cond >= 3                    (need some pose history)
#
# Returns N_cond in [3, 10].

def _estimate_cond_jerk(motion_tail_135: "np.ndarray | None") -> float:
    """Quick jerk estimator on a 135-dim motion segment.

    Computes mean of third-difference of pelvis translation (dims 0:3)
    as a scalar m/frame^3. Returns 0.0 if input is too short / None.
    The downstream threshold (3) is calibrated from the 50-sample
    E14 analysis: clean motion_gen_arena clips sit at ~0-1, noisy
    LLM-generated clips at 3-15 (per-frame third-difference norm).
    """
    if motion_tail_135 is None:
        return 0.0
    m = np.asarray(motion_tail_135, dtype=np.float32)
    if m.ndim != 2 or m.shape[0] < 4 or m.shape[-1] < 3:
        return 0.0
    p = m[:, 0:3]                                        # (T, 3)
    d3 = p[3:] - 3 * p[2:-1] + 3 * p[1:-2] - p[:-3]      # (T-3, 3)
    return float(np.linalg.norm(d3, axis=-1).mean())


def compute_cond_length(
    source_motion_135: "np.ndarray | None",
    T_src: int,
    N_transition: int,
    base: int = 5,
    min_cond: int = 3,
    max_cond: int = 10,
    noisy_jerk_threshold: float = 3.0,
    long_transition_threshold: int = 90,
    side: str = 'tail',
) -> int:
    """Adaptive N_cond rule (2026-04-23).

    Args:
        source_motion_135: (T, 135) full source motion from which a slice
            will be taken as condition. If None, jerk term is skipped.
        T_src: length of the source motion (bound for N_cond <= T_src//4*2).
        N_transition: planned transition length (frames). Longer transitions
            get +2 cond frames for a slightly stronger anchor.
        base / min_cond / max_cond: clamp configuration.
        noisy_jerk_threshold: pelvis jerk (m/frame^3) above which N_cond is
            trimmed by 2. Calibrated from E14 50-sample analysis: clean
            motion_gen_arena clips sit at ~0-1, noisy LLM-generated clips
            at 3-15.
        long_transition_threshold: N_transition ≥ this value triggers the
            +2 long-horizon boost.
        side: 'tail' (use last ~2*base frames to estimate jerk, for motion_a)
            or 'head' (use first ~2*base frames, for motion_b / P target).

    Returns:
        N_cond in [min_cond, min(max_cond, T_src // 4 * 2)].
    """
    # Estimate local jerk on a small window near the boundary.
    window = min(max(8, 2 * base), T_src) if T_src > 0 else 0
    tail = None
    if source_motion_135 is not None and window > 0:
        m = np.asarray(source_motion_135, dtype=np.float32)
        if m.ndim == 2 and m.shape[0] >= 4:
            if side == 'tail':
                tail = m[-window:]
            else:
                tail = m[:window]
    jerk = _estimate_cond_jerk(tail)

    n = base
    if jerk > noisy_jerk_threshold:
        n -= 2
    if N_transition >= long_transition_threshold:
        n += 2

    hard_cap = min(max_cond, max(min_cond, T_src // 2))
    n = max(min_cond, min(hard_cap, n))
    return int(n)


# SMPL-22 joint weights for angle-based transition length.
#
# Bigger joints (hip/knee/spine/pelvis) dominate "sit/stand/lie/crouch"
# transitions — if they rotate a lot the transition must be long. Smaller
# joints (wrists, ankles, head, neck) rotate a lot during almost every
# motion without signalling "hard to do quickly", so they get lower weight.
#
# Indices follow SMPL-22 convention:
#   0 pelvis, 1 L_hip, 2 R_hip, 3 spine1, 4 L_knee, 5 R_knee, 6 spine2,
#   7 L_ankle, 8 R_ankle, 9 spine3, 10 L_foot, 11 R_foot, 12 neck,
#   13 L_collar, 14 R_collar, 15 head, 16 L_shoulder, 17 R_shoulder,
#   18 L_elbow, 19 R_elbow, 20 L_wrist, 21 R_wrist
_JOINT_ANGLE_WEIGHTS = np.array([
    2.0,  # 0 pelvis (root rot)
    2.0, 2.0,  # 1,2 hips — dominates sit/stand/crouch
    1.5,  # 3 spine1 — dominates bend/lie
    1.5, 1.5,  # 4,5 knees — dominates sit/stand/kneel
    1.5,  # 6 spine2
    0.5, 0.5,  # 7,8 ankles — small, high wobble
    1.5,  # 9 spine3
    0.3, 0.3,  # 10,11 feet — small
    0.8,  # 12 neck
    0.7, 0.7,  # 13,14 collars — small
    0.5,  # 15 head — small, high wobble
    1.2, 1.2,  # 16,17 shoulders
    0.9, 0.9,  # 18,19 elbows
    0.5, 0.5,  # 20,21 wrists — small, high wobble
], dtype=np.float32)


def _rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """Row-major rot6d → 3x3 rotation matrix via Gram-Schmidt.

    Matches the convention in hftrainer/pipelines/motion/transition_utils.py
    (row-major: rot6d layout = [R00, R01, R10, R11, R20, R21], i.e. first
    two COLUMNS of R flattened row by row). Accepts shape (..., 6).
    """
    rot6d = np.asarray(rot6d, dtype=np.float32)
    # Reshape to (..., 3, 2): col0, col1
    *batch, _ = rot6d.shape
    m = rot6d.reshape(*batch, 3, 2)
    col0 = m[..., 0]
    col1 = m[..., 1]
    # Normalize col0
    n0 = np.linalg.norm(col0, axis=-1, keepdims=True)
    col0 = col0 / np.maximum(n0, 1e-8)
    # Make col1 orthogonal to col0, then normalize
    dot = np.sum(col0 * col1, axis=-1, keepdims=True)
    col1 = col1 - dot * col0
    n1 = np.linalg.norm(col1, axis=-1, keepdims=True)
    col1 = col1 / np.maximum(n1, 1e-8)
    # col2 = col0 × col1
    col2 = np.cross(col0, col1, axis=-1)
    R = np.stack([col0, col1, col2], axis=-1)  # (..., 3, 3)
    return R


def _joint_angle_change_frames(
    motion_a: np.ndarray,
    motion_b: np.ndarray,
    joint_angle_speed_per_frame: float,
) -> int:
    """Weighted SO(3) angle change across 22 joints → transition frames.

    Decision: use weighted SUM across joints, not mean. A sit-down touches
    only ~5 joints (pelvis, hips, knees, spine) but each by ~1.2 rad. A
    mean dilutes this by the 17 joints that don't move — giving a tiny
    N_angle (14 frames) that doesn't reflect the 30-50 frames a sit-down
    realistically needs.

    Semantics of SUM:
      sum_angle = Σ_j w_j * θ_j        (rad, per whole-body transition)
      N_angle   = sum_angle / speed    (frames, where speed is radians of
                                        whole-body rotation budget per frame)

    Typical budgets:
      - Quick arm raise (elbow+shoulder ~ 1 rad each, w≈1): Σ ≈ 2 rad
      - Sit down (hips 2×1.2rad w=2, knees 2×1.5rad w=1.5, spine ~0.3rad w=1.5):
                 Σ ≈ 4.8 + 4.5 + 0.5 ≈ 10 rad
      - Lie down (add pelvis 1.5rad w=2, plus sit-like adjustments): Σ ≈ 15 rad

    Speed tuning: 0.20 rad/frame ≈ 6 rad/s — a body can rotate ~1 full
    joint through 180° per second comfortably. For sit-down Σ=10 rad →
    N=50 frames (1.67 s). For arm raise Σ=2 rad → N=10 frames (0.33 s).
    """
    a = np.asarray(motion_a, dtype=np.float32).reshape(-1)
    b = np.asarray(motion_b, dtype=np.float32).reshape(-1)
    if a.shape[-1] < 135 or b.shape[-1] < 135:
        return 0

    rot6d_a = a[3:135].reshape(22, 6)
    rot6d_b = b[3:135].reshape(22, 6)

    R_a = _rot6d_to_matrix(rot6d_a)
    R_b = _rot6d_to_matrix(rot6d_b)

    # Relative rotation R_rel = R_a^T @ R_b ;  angle = acos((tr - 1) / 2)
    R_rel = np.einsum('jab,jbc->jac', R_a.transpose(0, 2, 1), R_b)
    tr = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
    cos_theta = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
    theta = np.arccos(cos_theta)                                    # (22,)

    # Weighted SUM — whole-body rotation budget.
    w = _JOINT_ANGLE_WEIGHTS
    weighted_sum_angle = float((theta * w).sum())

    return int(weighted_sum_angle / max(1e-6, joint_angle_speed_per_frame))


# Leg-only weights (SMPL-22 indices 1,2 hips; 4,5 knees; 7,8 ankles;
# 10,11 feet). Hip / knee dominate gross leg motion; ankle / feet handle
# fine contact and orientation. Weights below are calibrated by sweeping
# (foot_skating, jitter) on E14 50 samples — see 2026-04-27 ablation.
_LEG_JOINT_INDICES = np.array([1, 2, 4, 5, 7, 8, 10, 11], dtype=np.int64)
_LEG_JOINT_WEIGHTS = np.array(
    [3.0, 3.0,   # hips     — pelvis-relative thigh swing
     3.0, 3.0,   # knees    — flexion / extension
     2.0, 2.0,   # ankles   — foot orientation, contact pre-shaping
     1.0, 1.0],  # feet     — toe / heel articulation
    dtype=np.float32,
)


def _leg_angle_change_frames(
    motion_a: np.ndarray,
    motion_b: np.ndarray,
    leg_angle_speed_per_frame: float,
) -> int:
    """Lower-body weighted SO(3) angle SUM → transition frames.

    Mirrors :func:`_joint_angle_change_frames` but on the 8 leg joints
    only, with a tighter rad/frame budget. This is needed because the
    whole-body term dilutes leg-only motions (sit-cross-legged, kneel,
    step-and-pivot) where the torso barely moves.

    Calibration:
      - Sit-cross-legged: hip-flex 1.2rad×2×3 + hip-IR 0.5rad×2×3 +
        knee 1.6rad×2×3 + ankle 0.8rad×2×2 + foot 0.5rad×2×1
        ≈ 17 rad → 142 frames (clamped to 120 by max_frames).
      - Step-and-pivot: hip-yaw 0.3rad×2×3 + knee 0.5rad×2×3 + ankle
        0.5rad×2×2 + foot 0.3rad×2×1 ≈ 5.4 rad → 45 frames.
      - Static handshake (tiny leg motion): ≤ 1 rad → ≤ 8 frames
        (subsumed by min_frames=30 floor).
    """
    a = np.asarray(motion_a, dtype=np.float32).reshape(-1)
    b = np.asarray(motion_b, dtype=np.float32).reshape(-1)
    if a.shape[-1] < 135 or b.shape[-1] < 135:
        return 0

    rot6d_a = a[3:135].reshape(22, 6)
    rot6d_b = b[3:135].reshape(22, 6)
    R_a = _rot6d_to_matrix(rot6d_a)
    R_b = _rot6d_to_matrix(rot6d_b)
    R_rel = np.einsum('jab,jbc->jac', R_a.transpose(0, 2, 1), R_b)
    tr = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
    cos_theta = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
    theta = np.arccos(cos_theta)  # (22,)

    leg_theta = theta[_LEG_JOINT_INDICES]
    weighted_sum = float((leg_theta * _LEG_JOINT_WEIGHTS).sum())
    return int(weighted_sum / max(1e-6, leg_angle_speed_per_frame))





def _pose_change_frames(
    joints_a: np.ndarray,
    joints_b: np.ndarray,
    pose_speed_per_frame: float,
) -> int:
    """Per-joint retargeting distance in a yaw/translation-aligned frame.

    joints_a, joints_b: (22, 3) world joint positions.
    """
    ja = np.asarray(joints_a, dtype=np.float32).reshape(22, 3)
    jb = np.asarray(joints_b, dtype=np.float32).reshape(22, 3)

    # 1) Remove pelvis translation so both poses share origin at pelvis.
    pa = ja[0].copy()
    pb = jb[0].copy()
    ja_c = ja - pa
    jb_c = jb - pb

    # 2) Remove yaw: rotate each so the vector pelvis→(mid-shoulder) projects
    #    onto the +Z axis in the XZ plane. Mid-shoulder index approx = 16/17
    #    (left/right shoulder) averaged. Falls back to spine3 (=9) if missing.
    def _yaw_align(joints: np.ndarray) -> np.ndarray:
        # Use shoulder midpoint (joint 16 L shoulder, 17 R shoulder in SMPL-22)
        if joints.shape[0] > 17:
            fwd = 0.5 * (joints[16] + joints[17])
        else:
            fwd = joints[min(joints.shape[0] - 1, 9)]
        dx, dz = float(fwd[0]), float(fwd[2])
        theta = np.arctan2(dx, max(1e-6, dz))  # angle to +Z axis
        c, s = np.cos(-theta), np.sin(-theta)
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
        return joints @ R.T

    ja_al = _yaw_align(ja_c)
    jb_al = _yaw_align(jb_c)

    # 3) Mean per-joint distance.
    per_joint = np.linalg.norm(ja_al - jb_al, axis=-1)  # (22,)
    mean_dist = float(per_joint.mean())

    return int(mean_dist / max(1e-6, pose_speed_per_frame))


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


def build_start_pose_prepend_mask(
    T_total: int,
    D: int = 135,
    N_transition: int = 30,
    **kwargs,
) -> np.ndarray:
    """E15 (2026-04-21 redefinition): Prepend a transition from a target
    start pose P into an existing motion A.

    Sequence layout: [P(1) | transition(N_transition - 1) | A_full(len_A)]
      - frame 0 = P (condition, mask=0)
      - frames 1..N_transition-1 = generated (mask=1)
      - frames N_transition..T_total-1 = A (condition, mask=0)

    Args:
        T_total: N_transition + len(A). The caller sizes this from the item.
        D: unused (joint-group mask expanded externally).
        N_transition: number of prepended frames (including frame 0 = P).

    Returns:
        (T_total, D) mask where 1 = generate, 0 = keep.
    """
    grid = np.zeros((T_total, NUM_JOINT_GROUPS), dtype=np.float32)
    # Generated region: frames 1..N_transition-1 (inclusive).
    # Frame 0 is condition (P), frame N_transition is condition (A[0]).
    if N_transition > 1:
        grid[1:N_transition] = 1.0
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
        keep_part: one of
          - 'upper' — keep upper body (spine/arms/head + pelvis), regen lower
          - 'lower' — keep lower body (legs/feet + pelvis + trans), regen upper
          - 'spine_only' — keep pelvis + spine chain + head, regen all 4 limbs
          - 'root_only' — legacy: keep translation + pelvis
          - 'arms_only' — keep both arms + shoulders (no spine / legs)
          - 'legs_only' — keep both legs + feet + pelvis (no arms / spine)
          - 'left_arm' / 'right_arm' — keep one arm
          - 'left_leg' / 'right_leg' — keep one leg + that foot
          - 'feet_only' — keep both ankles + feet (stance fix)
          - 'no_feet' — keep everything EXCEPT feet (regen feet only)
    """
    grid = np.ones((T, NUM_JOINT_GROUPS), dtype=np.float32)

    if keep_part == 'upper':
        for g in UPPER_BODY:
            grid[:, g] = 0
        grid[:, 1] = 0  # pelvis rot
    elif keep_part == 'lower':
        for g in LOWER_BODY:
            grid[:, g] = 0
        grid[:, 0] = 0  # translation
        grid[:, 1] = 0  # pelvis rot
    elif keep_part == 'spine_only':
        grid[:, 0] = 0
        grid[:, 1] = 0
        for g in SPINE_HEAD:
            grid[:, g] = 0
    elif keep_part == 'root_only':
        grid[:, 0] = 0
        grid[:, 1] = 0
    elif keep_part == 'arms_only':
        # Keep: both arms (L/R). Regen: spine, legs, feet, head.
        # NOTE: no pelvis/translation kept so root regenerates freely
        # → useful to probe "can the model invent plausible lower body
        # from arm motion alone?" (e.g. gesture-driven walking).
        for g in LEFT_ARM + RIGHT_ARM:
            grid[:, g] = 0
    elif keep_part == 'legs_only':
        # Keep: both legs + pelvis + translation. Regen: arms, spine, head.
        grid[:, 0] = 0
        grid[:, 1] = 0
        for g in LEFT_LEG + RIGHT_LEG + FEET:
            grid[:, g] = 0
    elif keep_part == 'left_arm':
        for g in LEFT_ARM:
            grid[:, g] = 0
    elif keep_part == 'right_arm':
        for g in RIGHT_ARM:
            grid[:, g] = 0
    elif keep_part == 'left_leg':
        grid[:, 0] = 0
        grid[:, 1] = 0
        for g in LEFT_LEG:
            grid[:, g] = 0
    elif keep_part == 'right_leg':
        grid[:, 0] = 0
        grid[:, 1] = 0
        for g in RIGHT_LEG:
            grid[:, g] = 0
    elif keep_part == 'feet_only':
        # Keep only ankles+feet (stance fix scenario)
        for g in FEET:
            grid[:, g] = 0
    elif keep_part == 'no_feet':
        # Keep everything except feet — regenerate foot placement while
        # rest stays. Tests whether model can fix foot sliding while
        # preserving all other motion.
        grid[:, 0] = 0
        grid[:, 1] = 0
        for g in UPPER_BODY:
            grid[:, g] = 0
        for g in LEFT_LEG + RIGHT_LEG:
            grid[:, g] = 0
        # Leave FEET as mask=1 (regenerate)
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
        use_caption: Optional[bool] = None,
    ):
        self.name = name
        self.description = description
        self.mask_kwargs = mask_kwargs
        # Per-setting caption switch.
        #   None  -> inherit from EvalTask.needs_caption (default behavior;
        #            caption-required tasks skip uncond models, caption-
        #            optional tasks let both caption + uncond models run).
        #   True  -> this setting REQUIRES caption (uncond models skipped
        #            on this setting; caption is force-loaded).
        #   False -> this setting is run UNCONDITIONALLY (caption is force-
        #            blanked, even for caption models, so the resulting row
        #            is a true caption-free baseline on the same mask).
        # The 2026-04-25 E2 ablation uses False for {pre20,post20,mid60}_uncond
        # so we can read the marginal value of caption directly off the
        # dashboard against their cond twins.
        self.use_caption = use_caption


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
        default_metrics=['jitter_pos', 'foot_skating_ratio'],
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
        # Captions have been passed through the project's deployed
        # rewriter (Qwen3-30B-A3B-GRPO at 11.216.46.236:8080); the file
        # below is the rewriter-output mirror of eval_e2_inbetween_v2.
        data_file='eval_e2_inbetween_v2_rewritten.json',
        settings={
            # --- 2026-04-24 v2 ablation (refreshed 2026-04-25) ---
            # Shared 220-motion held-out HQ pool sourced from Private/.
            # Six mask patterns per motion; the three long-context masks
            # (pre20, post20, mid60) additionally have *_uncond twins
            # so we can read the marginal value of caption on identical
            # masks (caption row vs caption-blanked row, same mask, same
            # samples). The three 1-frame anchor settings stay caption-
            # required because 1 frame of context is too underspecified
            # for an unconditional baseline to be informative.
            'start_1f': TaskSetting(
                'start_1f', 'Anchor start only: first 1 frame known',
                {'keep_start': 1, 'keep_end': 0}, use_caption=True),
            'end_1f': TaskSetting(
                'end_1f', 'Anchor end only: last 1 frame known',
                {'keep_start': 0, 'keep_end': 1}, use_caption=True),
            'both_1f': TaskSetting(
                'both_1f', 'Anchor both ends: first + last 1 frame',
                {'keep_start': 1, 'keep_end': 1}, use_caption=True),
            # 2026-04-26: pre20/post20/mid60 set to use_caption=None so
            # BOTH caption-aware and uncond models run on these settings.
            # Caption-aware models will use the rewritten caption; uncond
            # models will simply not see text. The *_uncond twins below
            # additionally force-blank the caption for caption-aware
            # models, isolating "marginal value of caption" at fixed mask.
            'pre20': TaskSetting(
                'pre20', 'Given front 20% -> predict the rest',
                {'keep_start_frac': 0.20, 'keep_end': 0}, use_caption=None),
            'post20': TaskSetting(
                'post20', 'Given back 20% -> predict the rest',
                {'keep_start': 0, 'keep_end_frac': 0.20}, use_caption=None),
            'mid60': TaskSetting(
                'mid60', 'Given first & last 20% -> predict middle 60%',
                {'keep_start_frac': 0.20, 'keep_end_frac': 0.20},
                use_caption=None),
            # --- caption-free twins (2026-04-25) ---
            # Same mask, same sample, but caption is forced empty regardless
            # of model type. Lets us measure the marginal value of caption.
            'pre20_uncond': TaskSetting(
                'pre20_uncond',
                'Given front 20% -> predict the rest (NO caption)',
                {'keep_start_frac': 0.20, 'keep_end': 0}, use_caption=False),
            'post20_uncond': TaskSetting(
                'post20_uncond',
                'Given back 20% -> predict the rest (NO caption)',
                {'keep_start': 0, 'keep_end_frac': 0.20}, use_caption=False),
            'mid60_uncond': TaskSetting(
                'mid60_uncond',
                'Given first & last 20% -> predict middle 60% (NO caption)',
                {'keep_start_frac': 0.20, 'keep_end_frac': 0.20},
                use_caption=False),
        },
        default_metrics=[
            'mpjpe_masked', 'mpjpe_unmasked', 'boundary_accel_jump',
            'jitter_pos', 'foot_skating_ratio',
        ],
        needs_gt=True,
        # 2026-04-26: needs_caption=False so settings with use_caption=None
        # (pre20/post20/mid60) accept uncond models. The 1-frame anchor
        # settings remain caption-required via their explicit
        # use_caption=True per-setting flag.
        needs_caption=False,
        caption_aware=True,
        kimodo_comparable=True,
    )

    # --- E3: Sparse Keyframe Interpolation ---
    # 2026-04-26: unified setting names. Every uniform-interval setting is
    # now spelled `every_<K>f` (K = frames between anchors); the adaptive
    # variant is `adaptive`. The legacy A/B/C/D codes are gone — the dashboard
    # DB has been migrated. Captioned models read the _rewritten file.
    tasks['E3'] = EvalTask(
        task_id='E3',
        name='Keyframe Interpolation',
        description=(
            'Sparse keyframe interpolation: keep every K-th frame as a '
            'pose anchor and predict the rest. v2 (2026-04-25) sources '
            '240 motions from the held-out Private pool, stratified by '
            'action category and pelvis-speed bucket; captions go '
            'through the rewriter (12-20 word "A person..." form).'
        ),
        mask_builder=build_keyframe_mask,
        # Captions have been passed through the deployed rewriter
        # (Qwen3-30B-A3B-GRPO @ 11.216.46.236:8080); this file mirrors
        # eval_e3_keyframe_v2.json with rewriter-output captions.
        data_file='eval_e3_keyframe_v2_rewritten.json',
        settings={
            'every_5f': TaskSetting(
                'every_5f', 'Every 5 frames (very dense, ~6fps anchors)',
                {'interval': 5}),
            'every_10f': TaskSetting(
                'every_10f', 'Every 10 frames (dense, 3fps anchors)',
                {'interval': 10}),
            'every_15f': TaskSetting(
                'every_15f', 'Every 15 frames (0.5s @ 30fps, 2fps anchors)',
                {'interval': 15}),
            'every_30f': TaskSetting(
                'every_30f', 'Every 30 frames (1s @ 30fps, 1fps anchors)',
                {'interval': 30}),
            'every_60f': TaskSetting(
                'every_60f', 'Every 60 frames (2s @ 30fps, 0.5fps anchors)',
                {'interval': 60}),
            'adaptive': TaskSetting(
                'adaptive',
                'Adaptive keyframes at motion-acceleration peaks (~1/s)',
                {'_use_adaptive_keyframes': True}),
        },
        default_metrics=[
            'mpjpe_masked', 'mpjpe_unmasked', 'jitter_pos',
            'foot_skating_ratio',
        ],
        needs_gt=True,
        # Caption is OPTIONAL on E3: caption-aware models will use the
        # rewritten caption when present, uncond models simply skip text.
        # Per-setting use_caption left as None (inherit) so the eval
        # routes are: caption_* models -> with caption, uncond_* -> uncond.
        needs_caption=False,
        caption_aware=True,
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
            'ee_error_mean', 'ee_error_p50', 'ee_error_p95', 'ee_error_max',
            'ee_hit_rate_5cm', 'ee_hit_rate_10cm',
            'jitter_pos', 'mpjpe_masked',
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
        description='Follow root translation trajectory on the ground plane. '
                    'Pelvis X and Z are locked at condition frames; Y (height) '
                    'is free-generated so the model can produce natural pose '
                    'dynamics (matches v2 training distribution — see '
                    'condition_sampler_v2.sample_tier2_trajectory). Settings '
                    'vary density and whether pelvis heading is co-conditioned.',
        mask_builder=build_trajectory_mask,
        data_file='eval_e5_trajectory.json',
        settings={
            'A': TaskSetting(
                'A', 'Dense XZ trajectory (every frame)',
                {'mode': 'dense'}),
            'B': TaskSetting(
                'B', 'Sparse XZ waypoints (every 30 frames)',
                {'mode': 'sparse', 'interval': 30}),
            'C': TaskSetting(
                'C', 'XZ trajectory + pelvis heading every frame',
                {'mode': 'trajectory_heading', 'include_heading': True}),
        },
        default_metrics=[
            'trajectory_ade', 'trajectory_fde', 'foot_skating_ratio',
            'jitter_pos', ],
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
            'jitter_pos', ],
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
            'mpjpe_unmasked', 'jitter_pos', 'foot_skating_ratio',
        ],
        needs_gt=True,
        needs_caption=True,
        kimodo_comparable=False,
    )

    # --- E8: Loop Animation (v2 redesign 2026-04-26) ---
    # User redesign: drop all condition/transition-length ablations,
    # keep ONLY two settings, each tied to a specific model class:
    #   A — pure loop, caption-aware models only.
    #         Anchor: frame 0 of the GT. T_loop = sample.num_frames
    #         (the test case's natural duration; once-and-for-all
    #         per-sample adaptive value — no fixed-length axis).
    #         The eval pipeline replaces motion_135[-1] with motion_135[0]
    #         so frame 0 and frame T-1 carry the same target pose, the
    #         model has to fill the rest and close the loop.
    #   D — loop completion, uncond models only.
    #         Full GT prefix as condition (no tail-only ablation), then
    #         N_append additional frames bringing the body back to GT[0].
    #         N_append = compute_transition_length 3-term rule on
    #         (motion[-1] -> motion[0]): root displacement, joint position
    #         change, joint angle change. Clamped [30, 150].
    #
    # Removed (vs the 2026-04-23 cohort): D_cond5, D_cond15,
    # D_cond_adaptive, D_fixed90 — these were N_cond / N_append axis
    # ablations that the user explicitly asked to retire in favour of a
    # single adaptive scheme.
    #
    # Caption requirements (drives model-router skipping):
    #   A.use_caption = True   -> caption-aware models only
    #   D.use_caption = False  -> uncond models only (caption blanked
    #                             even if a caption-aware model is
    #                             routed here)
    tasks['E8'] = EvalTask(
        task_id='E8',
        name='Loop Animation',
        description='Generate looping animation. Setting A: pure loop — '
                    'frame 0 anchored at both ends, model fills the '
                    'intermediate frames (caption-aware models only). '
                    'Setting D: loop completion — given full GT, append '
                    'N_append adaptive frames to return to motion[0] '
                    '(uncond models only).',
        mask_builder=build_loop_mask,
        data_file='eval_e8_loop_v2.json',
        settings={
            'A': TaskSetting(
                'A',
                'Pure loop: frame 0 anchored at both ends, T_loop = '
                'sample.num_frames (natural duration). Caption-aware '
                'models only.',
                {},
                use_caption=True,
            ),
            'D': TaskSetting(
                'D',
                'Loop completion: full GT condition + N_append adaptive '
                'frames closing back to motion[0]. N_append from '
                'compute_transition_length 3-term rule (root + joint pos + '
                'joint angle), clamped [30, 150]. Uncond models only.',
                {'_loop_append': 'auto',
                 '_transition_min': 30,
                 '_transition_max': 150},
                use_caption=False,
            ),
        },
        default_metrics=[
            'loop_position_error', 'loop_velocity_error',
            'jitter_pos', 'boundary_accel_jump',
        ],
        needs_gt=False,
        # Per-setting use_caption flags drive routing; task-level needs_caption
        # is False (D is uncond) but A's True flag escalates to caption-required
        # at the setting level via the three-state caption_policy in
        # eval_m2m_v2_all_tasks.py.
        needs_caption=False,
        caption_aware=True,
        # Re-enabled 2026-04-26 after build_constraints_e8 was rewritten to
        # match the v2 redesign (Setting A: pure loop with frame[0]==frame[-1];
        # Setting D: loop completion with adaptive N_append). KIMODO uses
        # SOMA-30 skeleton and replays the same per-sample anchor frames.
        kimodo_comparable=True,
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
        # 2026-04-26: switched from eval_e9_repair.json (215 stale cases) to
        # the v2 selection — 389 cases, severity=fail (live re-checked),
        # mask-coverage ranked, ≥20 per defect type. See data_file meta for
        # selection rules. The v1 datalist is retained as
        # eval_e9_repair.json.bak_unbalanced for reference.
        data_file='eval_e9_repair_v2.json',
        settings={
            'A_adaptive_inpaint': TaskSetting(
                'A_adaptive_inpaint',
                '[DEPRECATED 2026-04-23] MoGenDIT adaptive mask is '
                'inaccurate (user-confirmed) and the SDEdit-inpaint '
                'combination adds little over D_strict_mask_*_sdedit*. '
                'Kept as DISABLED stub so historical DB rows still '
                'have a setting name to reference.',
                {
                    '_use_adaptive_mask': True,
                    '_editing_mode': False,
                    '_sdedit_tau': 0.5,
                    '_replacement_guidance': 'skip_last',
                    '_disabled': True,
                },
            ),
            'A_adaptive_inpaint_notau': TaskSetting(
                'A_adaptive_inpaint_notau',
                '[DEPRECATED 2026-04-23] Same reason as A_adaptive_inpaint.',
                {
                    '_use_adaptive_mask': True,
                    '_editing_mode': False,
                    '_sdedit_tau': 0.0,
                    '_replacement_guidance': 'skip_last',
                    '_disabled': True,
                },
            ),
            'B_post_replace': TaskSetting(
                'B_post_replace',
                '[DEPRECATED 2026-04-22] REMOVED — Stage 1 was pure-noise '
                'generation ignoring LQ entirely; post-hoc blend replaced '
                'whole defective frames with C_full output. Result looked '
                'nothing like LQ input. Do not re-enable without a new '
                'Stage 1 that anchors to LQ (e.g. SDEdit from LQ).',
                {
                    '_use_adaptive_mask': False,
                    '_editing_mode': False,
                    '_sdedit_tau': 0.0,
                    '_post_hoc_replace_with_adaptive': True,
                    '_disabled': True,  # skipped by eval loop
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

            # ── E_union_mask — 2026-04-26 NEW DEFAULT ─────────────────
            # User-requested: take the UNION of MoGenDIT's change-based
            # adaptive mask and the QC-checker per-joint invalid_mask.
            # Adaptive alone misses persistent anatomical defects;
            # QC alone misses change-based artefacts. Union catches both.
            # Uses skip_last replacement guidance and clean_motion=LQ so
            # un-flagged regions stay locked at LQ — same protocol as the
            # D_strict_mask variants.
            'E_union_mask': TaskSetting(
                'E_union_mask',
                'Union of MoGenDIT adaptive mask AND QC invalid_mask. '
                'Spatial+temporal dilation applied on the QC half '
                '(d=2, kinematic neighbour propagate). Adaptive half '
                'kept at temporal_dilate=0 to avoid over-grow.',
                {
                    '_union_mask': True,
                    '_qc_dilate_temp': 2,
                    '_qc_dilate_spatial': True,
                    '_qc_include_borderline': True,
                    '_union_adaptive_dilate': 0,
                    '_editing_mode': False,
                },
            ),

            # ── D_* — Redesigned 2026-04-22 ──────────────────────────
            # All A_* / B_ / C_ settings above produced unsatisfying
            # repair quality (user report: "大多数问题都无法修复").
            # The D_ family is a clean-slate design that (1) removes the
            # dependency on MoGenDIT's pre-cached adaptive mask being
            # reliable and (2) implements MoGenDIT's full `ada_denoise`
            # two-stage detection + strict-mask variants.
            #
            # D_ada_denoise_* — two-stage detect-then-repair (revised 2026-04-23):
            #   Stage 1: mask=all-1, replacement_guidance='skip_last',
            #            sdedit_tau=0.5, clean_motion=LQ → SDEdit-from-LQ
            #            manifold projection. The model pulls LQ toward the
            #            nearest clean motion; stage-1 output ≈ "model's
            #            best guess of a clean version of LQ".
            #   Stage 2: change = |LQ_normalized − denoised_stage1|,
            #            aggregate to per-joint (all 6 rot6d dims must
            #            all be low-change to count as clean). Build
            #            keep_mask: "this joint at this frame is clean".
            #   Stage 3: re-run with the expanded keep_mask,
            #            replacement_guidance='skip_last',
            #            clean_motion=LQ, sdedit_tau=0 (standard inpaint).
            #   Threshold strategy: '_ada_threshold_mode' in
            #     {'abs', 'topk_pct'}, '_ada_threshold' is the value.
            # NOTE: the earlier "Stage 1 = frame-0 anchor only" design was
            # replaced because M2M was not trained on motion-degradation
            # pairs and treated that mask as a T2M prompt, producing a
            # free-generation output uncorrelated with LQ.
            'D_ada_denoise_t005': TaskSetting(
                'D_ada_denoise_t005',
                'MoGenDIT ada_denoise (2-stage): |change|<=0.05 → keep',
                {
                    '_ada_denoise': True,
                    '_ada_threshold_mode': 'abs',
                    '_ada_threshold': 0.05,
                    '_editing_mode': False,
                },
            ),
            'D_ada_denoise_t010': TaskSetting(
                'D_ada_denoise_t010',
                '[DEPRECATED 2026-04-23 FAILED AFTER PIPELINE FIX] '
                'MoGenDIT ada_denoise (2-stage). Earlier measured '
                'jitter=546/QC=82% turned out to be an artefact of the '
                'all-ones-mask pipeline bug: Stage 1 silently ran as '
                'pure-noise uncond gen (not an LQ→manifold projection), '
                'so Stage 3 mask was garbage and the final output was a '
                'smooth RANDOM motion that happened to pass QC. After '
                'fixing Stage 1 to keep one anchor cell so SDEdit '
                'actually activates, real jitter=1748 — worse than '
                'strict_mask baseline. Disabled.',
                {
                    '_disabled': True,
                    '_ada_denoise': True,
                    '_ada_threshold_mode': 'abs',
                    '_ada_threshold': 0.10,
                    '_editing_mode': False,
                },
            ),
            'D_ada_denoise_t010_s3tau05': TaskSetting(
                'D_ada_denoise_t010_s3tau05',
                '[DEPRECATED 2026-04-23] MoGenDIT ada_denoise + Stage 3 '
                'SDEdit τ=0.5. 实测 jitter 1974 / QC 37.7%，比 baseline '
                '(1504/61%) 更差；LQ-anchor 效果已被 strict_mask_bsmooth '
                '覆盖。Disabled 以保持 dashboard 整洁。',
                {
                    '_disabled': True,
                    '_ada_denoise': True,
                    '_ada_threshold_mode': 'abs',
                    '_ada_threshold': 0.10,
                    '_editing_mode': False,
                    '_ada_stage3_sdedit_tau': 0.5,
                },
            ),
            'D_ada_denoise_t010_s3tau03': TaskSetting(
                'D_ada_denoise_t010_s3tau03',
                '[DEPRECATED 2026-04-23] MoGenDIT ada_denoise + Stage 3 '
                'SDEdit τ=0.3。实测 jitter 2320 / QC 28.4%，worst among '
                'ada variants。Disabled。',
                {
                    '_disabled': True,
                    '_ada_denoise': True,
                    '_ada_threshold_mode': 'abs',
                    '_ada_threshold': 0.10,
                    '_editing_mode': False,
                    '_ada_stage3_sdedit_tau': 0.3,
                },
            ),
            'D_ada_denoise_t020': TaskSetting(
                'D_ada_denoise_t020',
                'MoGenDIT ada_denoise (2-stage): |change|<=0.20 → keep '
                '(aggressive — modify more)',
                {
                    '_ada_denoise': True,
                    '_ada_threshold_mode': 'abs',
                    '_ada_threshold': 0.20,
                    '_editing_mode': False,
                },
            ),
            'D_ada_denoise_top20': TaskSetting(
                'D_ada_denoise_top20',
                'MoGenDIT ada_denoise (2-stage): top-20% per-sample '
                'change threshold (adaptive to motion scale)',
                {
                    '_ada_denoise': True,
                    '_ada_threshold_mode': 'topk_pct',
                    '_ada_threshold': 0.20,
                    '_editing_mode': False,
                },
            ),
            'D_ada_denoise_top30': TaskSetting(
                'D_ada_denoise_top30',
                'MoGenDIT ada_denoise (2-stage): top-30% per-sample '
                'change threshold',
                {
                    '_ada_denoise': True,
                    '_ada_threshold_mode': 'topk_pct',
                    '_ada_threshold': 0.30,
                    '_editing_mode': False,
                },
            ),
            # D_strict_mask_* — tighten cached MoGenDIT mask before
            # single-shot denoise. Parameters:
            #   dilate:  spatial/temporal dilation radius (frames)
            #   min_blob: minimum blob area (frames×joints) to keep,
            #             smaller blobs are suppressed (treated as clean)
            # Always uses replacement_guidance='skip_last' and LQ as
            # clean_motion so unmasked regions stay locked.
            'D_strict_mask_d1_b2': TaskSetting(
                'D_strict_mask_d1_b2',
                'Strict MoGenDIT mask + skip_last: temporal dilate=1, '
                'min_blob=2 (conservative)',
                {
                    '_strict_adaptive_mask': True,
                    '_strict_dilate': 1,
                    '_strict_min_blob': 2,
                    '_editing_mode': False,
                },
            ),
            'D_strict_mask_d2_b3': TaskSetting(
                'D_strict_mask_d2_b3',
                'Strict MoGenDIT mask + skip_last: temporal dilate=2, '
                'min_blob=3 (recommended balance)',
                {
                    '_strict_adaptive_mask': True,
                    '_strict_dilate': 2,
                    '_strict_min_blob': 3,
                    '_editing_mode': False,
                },
            ),
            'D_strict_mask_d3_b5': TaskSetting(
                'D_strict_mask_d3_b5',
                'Strict MoGenDIT mask + skip_last: temporal dilate=3, '
                'min_blob=5 (aggressive)',
                {
                    '_strict_adaptive_mask': True,
                    '_strict_dilate': 3,
                    '_strict_min_blob': 5,
                    '_editing_mode': False,
                },
            ),

            # D_strict_mask_d2_b3_edit — jitter mitigation via editing mode
            # (2026-04-23). Same strict mask as d2_b3 but with
            # _editing_mode=True, so the model sees LQ (with its jitter)
            # in the src_motion/reactive channel and is trained to denoise
            # it. MAN training + MoGenDIT-style editing_prob=15% means the
            # model has seen "corrupted LQ in known channel" during
            # training, which is closer to the E9 inference distribution
            # than completion mode (where MAN training assumed clean GT).
            'D_strict_mask_d2_b3_edit': TaskSetting(
                'D_strict_mask_d2_b3_edit',
                '[DEPRECATED 2026-04-23] Editing mode INCREASED jitter '
                '(1769 vs 1504 baseline) and decreased QC pass rate. '
                'Disabled.',
                {
                    '_strict_adaptive_mask': True,
                    '_strict_dilate': 2,
                    '_strict_min_blob': 3,
                    '_editing_mode': True,
                    '_disabled': True,
                },
            ),
            # D_strict_mask_d2_b3_smooth — jitter mitigation via LQ pre-smooth
            # (2026-04-23). Same strict mask as d2_b3 in completion mode,
            # but clean_motion (= LQ) is pre-filtered by a 1-D Gaussian in
            # time (sigma=1 frame ≈ 5 Hz cutoff @ 30 fps) before imputation.
            # Smoothing is only applied on the "keep" region (mask==0);
            # generated regions are untouched. Reduces the high-frequency
            # OOD signal the model has to reconcile at blob boundaries.
            'D_strict_mask_d2_b3_smooth1': TaskSetting(
                'D_strict_mask_d2_b3_smooth1',
                'Strict MoGenDIT mask (d=2, b=3) + skip_last + Gaussian '
                'pre-smooth of LQ keep-region (σ=1 frame). Keeps '
                'completion mode.',
                {
                    '_strict_adaptive_mask': True,
                    '_strict_dilate': 2,
                    '_strict_min_blob': 3,
                    '_editing_mode': False,
                    '_presmooth_clean_sigma': 1.0,
                },
            ),
            # D_strict_mask_d2_b3_bsmooth — boundary smoothing (2026-04-23).
            # Adds a post-hoc Gaussian blend on a narrow band (±radius
            # frames) around every mask 0↔1 transition to kill the
            # acceleration spike at blob boundaries. Observed issue:
            # smooth1 reduced jitter mean but left frame-107-style
            # "local jumps" intact (pelvis+limbs all spike at the same
            # transition frame). This smooths those specific frames.
            'D_strict_mask_d2_b3_bsmooth': TaskSetting(
                'D_strict_mask_d2_b3_bsmooth',
                'Strict mask + pre-smooth LQ keep (σ=1) + POST-SMOOTH '
                'output at mask boundaries (radius=3, σ=2). Reduces '
                'local velocity spikes at blob edges.',
                {
                    '_strict_adaptive_mask': True,
                    '_strict_dilate': 2,
                    '_strict_min_blob': 3,
                    '_editing_mode': False,
                    '_presmooth_clean_sigma': 1.0,
                    '_boundary_smooth_radius': 3,
                    '_boundary_smooth_sigma': 2.0,
                },
            ),
            # D_strict_mask_d2_b3_bsmooth_<post-proc> — stackable post-processing
            # on top of bsmooth (2026-04-23). These are "provably safe"
            # smoothers that don't touch normal high-frequency motion:
            #   * accelK3: detect outlier frames (|accel| > μ+3σ aggregated
            #     across channels) and replace them (only) with a 3-tap
            #     temporal median. Kills spike-shaped jumps, preserves rest.
            #   * savgol7_p3: full-sequence Savitzky-Golay (window=7,
            #     poly=3) over the output. Preserves peak shape better
            #     than Gaussian; suppresses high-freq noise uniformly.
            #   * accel_savgol: stack both (savgol first, then median
            #     spike removal on residual outliers).
            'D_strict_mask_d2_b3_bsmooth_accelK3': TaskSetting(
                'D_strict_mask_d2_b3_bsmooth_accelK3',
                'bsmooth + accel-spike 3-tap median (k=3σ outlier detection).',
                {
                    '_strict_adaptive_mask': True,
                    '_strict_dilate': 2,
                    '_strict_min_blob': 3,
                    '_editing_mode': False,
                    '_presmooth_clean_sigma': 1.0,
                    '_boundary_smooth_radius': 3,
                    '_boundary_smooth_sigma': 2.0,
                    '_accel_spike_k': 3.0,
                },
            ),
            'D_strict_mask_d2_b3_bsmooth_savgol7': TaskSetting(
                'D_strict_mask_d2_b3_bsmooth_savgol7',
                '[DEPRECATED 2026-04-23] Pure subset of combo — per-case '
                'analysis (N=215) showed 0 unique passes vs combo; combo '
                'strictly dominates. Disabled.',
                {
                    '_disabled': True,
                    '_strict_adaptive_mask': True,
                    '_strict_dilate': 2,
                    '_strict_min_blob': 3,
                    '_editing_mode': False,
                    '_presmooth_clean_sigma': 1.0,
                    '_boundary_smooth_radius': 3,
                    '_boundary_smooth_sigma': 2.0,
                    '_savgol_window': 7,
                    '_savgol_poly': 3,
                },
            ),
            'D_strict_mask_d2_b3_bsmooth_savgol5': TaskSetting(
                'D_strict_mask_d2_b3_bsmooth_savgol5',
                'bsmooth + tighter Savitzky-Golay (window=5, poly=3). '
                'Less aggressive smoothing to preserve fast motion.',
                {
                    '_strict_adaptive_mask': True,
                    '_strict_dilate': 2,
                    '_strict_min_blob': 3,
                    '_editing_mode': False,
                    '_presmooth_clean_sigma': 1.0,
                    '_boundary_smooth_radius': 3,
                    '_boundary_smooth_sigma': 2.0,
                    '_savgol_window': 5,
                    '_savgol_poly': 3,
                },
            ),
            'D_strict_mask_d2_b3_bsmooth_combo': TaskSetting(
                'D_strict_mask_d2_b3_bsmooth_combo',
                'bsmooth + Savgol(w=7) + accel-spike median (k=3σ). Both '
                'post-processors stacked — Savgol smooths globally, then '
                'residual spikes get median-filtered.',
                {
                    '_strict_adaptive_mask': True,
                    '_strict_dilate': 2,
                    '_strict_min_blob': 3,
                    '_editing_mode': False,
                    '_presmooth_clean_sigma': 1.0,
                    '_boundary_smooth_radius': 3,
                    '_boundary_smooth_sigma': 2.0,
                    '_accel_spike_k': 3.0,
                    '_savgol_window': 7,
                    '_savgol_poly': 3,
                },
            ),
            # D_strict_mask_d2_b3_manifold* — [FAILED 2026-04-23]
            # Attempted fusion of strict_mask + ada_denoise by replacing
            # clean_motion with an SDEdit τ=0.5 manifold projection of LQ.
            # Empirical jitter was WORSE than bsmooth baseline (3545 / 3514
            # / 2310 vs 1350). Root cause: with only one keep cell to
            # activate SDEdit, the projection drifts away from LQ, so the
            # strict_mask keep-region gets locked to a wrong target.
            'D_strict_mask_d2_b3_manifold': TaskSetting(
                'D_strict_mask_d2_b3_manifold',
                '[DEPRECATED 2026-04-23 FAILED] Strict mask + SDEdit-τ=0.5 '
                'manifold projection as clean_motion. Measured jitter=3545 '
                '(vs bsmooth 1350). Disabled.',
                {
                    '_disabled': True,
                    '_strict_adaptive_mask': True,
                    '_strict_dilate': 2,
                    '_strict_min_blob': 3,
                    '_editing_mode': False,
                    '_anchor_to_stage1': True,
                    '_manifold_sdedit_tau': 0.5,
                    '_manifold_blend_alpha': 1.0,
                    '_replacement_guidance': 'skip_last',
                },
            ),
            'D_strict_mask_d2_b3_manifold_bsmooth': TaskSetting(
                'D_strict_mask_d2_b3_manifold_bsmooth',
                '[DEPRECATED 2026-04-23 FAILED] Manifold anchor + boundary '
                'smoothing. Measured jitter=3514. Disabled.',
                {
                    '_disabled': True,
                    '_strict_adaptive_mask': True,
                    '_strict_dilate': 2,
                    '_strict_min_blob': 3,
                    '_editing_mode': False,
                    '_anchor_to_stage1': True,
                    '_manifold_sdedit_tau': 0.5,
                    '_manifold_blend_alpha': 1.0,
                    '_replacement_guidance': 'skip_last',
                    '_boundary_smooth_radius': 3,
                    '_boundary_smooth_sigma': 2.0,
                },
            ),
            'D_strict_mask_d2_b3_manifold_a05': TaskSetting(
                'D_strict_mask_d2_b3_manifold_a05',
                '[DEPRECATED 2026-04-23 FAILED] Manifold α=0.5 blend. '
                'Measured jitter=2310. Still worse than bsmooth. Disabled.',
                {
                    '_disabled': True,
                    '_strict_adaptive_mask': True,
                    '_strict_dilate': 2,
                    '_strict_min_blob': 3,
                    '_editing_mode': False,
                    '_anchor_to_stage1': True,
                    '_manifold_sdedit_tau': 0.5,
                    '_manifold_blend_alpha': 0.5,
                    '_replacement_guidance': 'skip_last',
                    '_boundary_smooth_radius': 3,
                    '_boundary_smooth_sigma': 2.0,
                },
            ),
            'D_strict_mask_d2_b3_bsmooth_tight': TaskSetting(
                'D_strict_mask_d2_b3_bsmooth_tight',
                '[DEPRECATED 2026-04-23] Boundary smooth radius=2, σ=1. '
                '实测 jitter 1356 / QC 69.3%，与 bsmooth (1350/69.3%) '
                '差异可忽略；保留 bsmooth 即可。Disabled。',
                {
                    '_disabled': True,
                    '_strict_adaptive_mask': True,
                    '_strict_dilate': 2,
                    '_strict_min_blob': 3,
                    '_editing_mode': False,
                    '_presmooth_clean_sigma': 1.0,
                    '_boundary_smooth_radius': 2,
                    '_boundary_smooth_sigma': 1.0,
                },
            ),
            'D_strict_mask_d2_b3_smooth2': TaskSetting(
                'D_strict_mask_d2_b3_smooth2',
                '[DEPRECATED 2026-04-23] σ=2 was identical to σ=1 in '
                'empirical results (user-confirmed); we keep only smooth1.',
                {
                    '_strict_adaptive_mask': True,
                    '_strict_dilate': 2,
                    '_strict_min_blob': 3,
                    '_editing_mode': False,
                    '_presmooth_clean_sigma': 2.0,
                    '_disabled': True,
                },
            ),
            # D_strict_mask_d2_b3_sdedit{05,03} — SDEdit on masked region
            # (2026-04-23). Default behavior starts masked-region from pure
            # noise z; with _sdedit_tau the start becomes
            # (1-τ)*z + τ*LQ for the masked region, so the generated
            # content is anchored to LQ's structure. Keep region still
            # locked to LQ via skip_last. Use when you want HQ to closely
            # follow LQ rather than invent "plausibly different" repair.
            'D_strict_mask_d2_b3_sdedit05': TaskSetting(
                'D_strict_mask_d2_b3_sdedit05',
                '[DEPRECATED 2026-04-23] Strict mask + SDEdit τ=0.5. '
                '实测 jitter 1966 / QC 24.7%，虽然 follow LQ 更紧但 jitter '
                'QC 双差于 baseline；LQ-follow 需求已被 bsmooth 在 QC 上覆盖。'
                'Disabled。',
                {
                    '_disabled': True,
                    '_strict_adaptive_mask': True,
                    '_strict_dilate': 2,
                    '_strict_min_blob': 3,
                    '_editing_mode': False,
                    '_sdedit_tau': 0.5,
                    '_replacement_guidance': 'skip_last',
                },
            ),
            'D_strict_mask_d2_b3_sdedit03': TaskSetting(
                'D_strict_mask_d2_b3_sdedit03',
                '[DEPRECATED 2026-04-23] Strict mask + SDEdit τ=0.3. '
                '实测 jitter 2169 / QC 17.2%，worst strict variant。'
                'Disabled。',
                {
                    '_disabled': True,
                    '_strict_adaptive_mask': True,
                    '_strict_dilate': 2,
                    '_strict_min_blob': 3,
                    '_editing_mode': False,
                    '_sdedit_tau': 0.3,
                    '_replacement_guidance': 'skip_last',
                },
            ),

            # D_qc_mask_* — Use the motion Quality Checker's invalid_mask
            # as the source of truth for which (frame, joint) cells are
            # defective. Addresses the failure mode where MoGenDIT's
            # change-based adaptive mask misses persistent anatomical
            # defects (neck bent 180°, spine hyperext, ankle distortion)
            # that affect every frame and produce zero change signal.
            #
            # Each failing QC checker has a per-frame per-joint
            # invalid_mask; we OR them all, kinematically dilate to
            # parent+children joints, temporally dilate ±N frames, and
            # feed the result to the pipeline with replacement_guidance
            # 'skip_last' + clean_motion=LQ. Unmasked regions lock to LQ;
            # masked regions generate freely from noise.
            # ── D_qc_mask_* [DEPRECATED 2026-04-22] ──────────────────
            # Removed per user request. Direct quote: "仅依靠 quality
            # checker 作为 mask 一定是不够的，我已经肉眼检查过 quality
            # checker 的 mask，是绝对不准的". The QC invalid_mask approach
            # misses many visible defects (checker recall is low) and
            # flags too many false positives (precision is bad on
            # borderline cases). Keep setting stubs here for history but
            # mark `_disabled` so the eval loop skips them.
            'D_qc_mask_d1': TaskSetting(
                'D_qc_mask_d1',
                '[DEPRECATED] QC invalid_mask is inaccurate per user review.',
                {'_disabled': True},
            ),
            'D_qc_mask_d2': TaskSetting(
                'D_qc_mask_d2',
                '[DEPRECATED] QC invalid_mask is inaccurate per user review.',
                {'_disabled': True},
            ),
            'D_qc_mask_d3': TaskSetting(
                'D_qc_mask_d3',
                '[DEPRECATED] QC invalid_mask is inaccurate per user review.',
                {'_disabled': True},
            ),
            'D_qc_mask_strict': TaskSetting(
                'D_qc_mask_strict',
                '[DEPRECATED] QC invalid_mask is inaccurate per user review.',
                {'_disabled': True},
            ),
        },
        default_metrics=[
            'jitter_pos', 'foot_skating_ratio',
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
            # 2026-04-23: additional granular part settings ------------------
            'D_arms_only': TaskSetting(
                'D_arms_only',
                'Keep only both arms (L/R shoulders, elbows, wrists). '
                'Regen spine + legs + feet + head. Tests gesture→full-body.',
                {'keep_part': 'arms_only'}),
            'E_legs_only': TaskSetting(
                'E_legs_only',
                'Keep only both legs + feet + pelvis. Regen arms, spine, head. '
                'Tests locomotion→upper body inference.',
                {'keep_part': 'legs_only'}),
            'F_left_arm': TaskSetting(
                'F_left_arm',
                'Keep left arm (shoulder/elbow/wrist). Regen everything else. '
                'Asymmetric — tests single-limb grounding.',
                {'keep_part': 'left_arm'}),
            'G_right_arm': TaskSetting(
                'G_right_arm',
                'Keep right arm. Regen everything else.',
                {'keep_part': 'right_arm'}),
            'H_left_leg': TaskSetting(
                'H_left_leg',
                'Keep left leg + pelvis + translation. Regen everything else.',
                {'keep_part': 'left_leg'}),
            'I_right_leg': TaskSetting(
                'I_right_leg',
                'Keep right leg + pelvis + translation.',
                {'keep_part': 'right_leg'}),
            'J_feet_only': TaskSetting(
                'J_feet_only',
                'Keep only ankles + feet. Regen everything else. Use case: '
                'constrain stance, generate torso/arms freely.',
                {'keep_part': 'feet_only'}),
            'K_no_feet': TaskSetting(
                'K_no_feet',
                'Keep everything EXCEPT feet. Regen only feet — stress-test '
                'foot sliding fix while rest of motion stays intact.',
                {'keep_part': 'no_feet'}),
        },
        default_metrics=[
            'mpjpe_unmasked', 'jitter_pos', 'foot_skating_ratio',
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
    # Segment k's duration = datalist entry (base_idx+k)'s own num_frames
    # (capped to 360). Only the overlap length N varies across settings,
    # isolating the effect of "how much prior context do we hand off?".
    tasks['E13'] = EvalTask(
        task_id='E13',
        name='Multi-Prompt Generation',
        description='Given N text descriptions, autoregressively generate '
                    'arbitrarily long motion by chaining segments. Each '
                    'segment uses the previous segment\'s last N frames as '
                    'a prefix condition; segment duration is taken from the '
                    'corresponding datalist entry\'s own motion length.',
        mask_builder=build_multi_prompt_mask,
        data_file='eval_e13_multi_prompt.json',
        settings={
            'A': TaskSetting('A', '4 prompts, 1-frame prefix (minimal anchor)',
                             {'num_prompts': 4, 'overlap_frames': 1}),
            'B': TaskSetting('B', '4 prompts, 10-frame prefix (~0.3s)',
                             {'num_prompts': 4, 'overlap_frames': 10}),
            'C': TaskSetting('C', '4 prompts, 30-frame prefix (~1s context)',
                             {'num_prompts': 4, 'overlap_frames': 30}),
        },
        default_metrics=[
            'jitter_pos', 'foot_skating_ratio',
            'segment_boundary_smoothness', 'total_duration',
        ],
        needs_gt=False,
        needs_caption=True,
        kimodo_comparable=False,
    )

    # --- E14: Transition Stitching ---
    # 2026-04-23 (v5 redesign, user-driven): two ORTHOGONAL placement
    # modes instead of N_cond ablation. User observation: the previous
    # "forward_step=1m" hard-coded placement was arbitrary and circular
    # (N_transition depended on a distance we just made up). Replace with:
    #
    #   L (overlap):   B.xz = A_end.xz, only Y differs. Pure postural
    #                  transition, no locomotion. Closest to training
    #                  distribution's sample_tier2_inbetween geometry.
    #   M (move):      B.xz = A_end.xz + A_tail_velocity.xz * N_transition.
    #                  If A ends in motion (walking/running), B lands where
    #                  A would naturally be after N_transition frames at
    #                  that velocity. Forces the model to generate locomotion
    #                  to reach B. If A ends static, B ≈ overlap (velocity≈0).
    #
    # N_cond uses adaptive rule (compute_cond_length) in both settings.
    tasks['E14'] = EvalTask(
        task_id='E14',
        name='Transition Stitching',
        description='Stitch two motions A and B. Settings vary WHERE B is '
                    'placed: L=overlap (B.xz=A_end.xz, postural transition '
                    'only) vs M=move (B positioned along A\'s end-velocity '
                    'so the model must generate locomotion to reach it). '
                    'N_transition and N_cond both adaptive.',
        mask_builder=build_transition_mask,
        data_file='eval_e2_transition_by_velocity.json',
        settings={
            'L': TaskSetting(
                'L',
                '[Overlap] B.xz = A_end.xz. Postural transition only — no '
                'locomotion required. Uses 100 hq400h static samples (A '
                'ends with pelvis xz speed ≤ 0.0004 m/frame). 2026-04-27: '
                'switched to leg-aware c45+t120 (fixed N_cond=45+45, '
                'N_transition adaptive in [30,120] with leg-only angle '
                'term in compute_transition_length). This was the best '
                'cell across the 9-cell ablation grid (lowest '
                'foot_skating + boundary_accel_jump on n=50 paired t).',
                {
                    '_use_transition_data': True,
                    '_placement': 'overlap',
                    '_context_policy': 'fixed',
                    '_n_cond_a_frames': 45,
                    '_n_cond_b_frames': 45,
                    '_transition_min': 30,
                    '_transition_max': 120,
                    '_data_file': 'eval_e14_hq400h_static100.json',
                },
            ),
            'M': TaskSetting(
                'M',
                '[Move] B.xz = A_end.xz + A_tail_velocity * N_transition. '
                'Model must generate locomotion to reach B. Uses 100 '
                'hq400h walk/jog samples (A ends with pelvis xz speed in '
                '[0.004, 0.020] m/frame, stable tail). 2026-04-27: '
                'switched to leg-aware c45+t120 (fixed N_cond=45+45, '
                'N_transition adaptive in [30,120] with leg-only angle '
                'term in compute_transition_length). This was the best '
                'cell across the 9-cell ablation grid + leg-aware '
                'extension (foot_skating −4% vs leg-blind c45+t120, '
                'transition_length +12 frames *** on lifted samples).',
                {
                    '_use_transition_data': True,
                    '_placement': 'velocity',
                    '_context_policy': 'fixed',
                    '_n_cond_a_frames': 45,
                    '_n_cond_b_frames': 45,
                    '_transition_min': 30,
                    '_transition_max': 120,
                    '_data_file': 'eval_e14_hq400h_move100.json',
                },
            ),
            # ─────────────────────────────────────────────────────────
            # 2026-04-27: foot-skating ablation matrix (N_cond × N_t).
            # Goal: identify whether the slipping artefact at high
            # locomotion distance (E14 setting M, ~1-3 m) is driven by
            # (a) too-short cond (model can't read cadence in 7 frames),
            # (b) too-long N_transition forcing the model to inch root
            #     across at sub-walking speed, or
            # (c) training distribution simply lacks long-locomotion
            #     stitched transitions and no inference-time knob fixes
            #     it.
            # 3×3 grid: cond ∈ {5, 15, 30} × N_t_max ∈ {60, 120, 180}.
            # All inherit M's velocity placement + move100 datalist;
            # N_cond uses 'fixed' policy (matches name); N_transition
            # uses adaptive root/pose/angle but with the per-setting
            # max clamp.  N_transition_min stays at 30 throughout.
            # ─────────────────────────────────────────────────────────
            **{
                f'M_c{nc}_t{tmax}': TaskSetting(
                    f'M_c{nc}_t{tmax}',
                    f'[Move ablation] N_cond={nc}+{nc}, N_transition '
                    f'clamp [30, {tmax}] (vs baseline M cond=adaptive, '
                    f'N_t_max=120). Cell of foot-skating ablation grid '
                    f'(2026-04-27).',
                    {
                        '_use_transition_data': True,
                        '_placement': 'velocity',
                        '_context_policy': 'fixed',
                        '_n_cond_a_frames': nc,
                        '_n_cond_b_frames': nc,
                        '_transition_min': 30,
                        '_transition_max': tmax,
                        '_data_file': 'eval_e14_hq400h_move100.json',
                    },
                )
                for nc in (5, 15, 30)
                for tmax in (60, 120, 180)
            },
            # 2026-04-27 (extension): the 3×3 grid above is monotone in
            # N_cond at every tmax row but does NOT saturate at c30. Push
            # the N_cond axis further with t120 fixed (since t120/t180 are
            # near-identical in the base grid). Cap at 90 because
            # clip_len=360 ⇒ 90+180+90 = 360 is the training-seen ceiling.
            **{
                f'M_c{nc}_t120': TaskSetting(
                    f'M_c{nc}_t120',
                    f'[Move ablation extension] N_cond={nc}+{nc}, '
                    f'N_transition clamp [30, 120]. Probes whether the '
                    f'monotone improvement in (c5,c15,c30) saturates.',
                    {
                        '_use_transition_data': True,
                        '_placement': 'velocity',
                        '_context_policy': 'fixed',
                        '_n_cond_a_frames': nc,
                        '_n_cond_b_frames': nc,
                        '_transition_min': 30,
                        '_transition_max': 120,
                        '_data_file': 'eval_e14_hq400h_move100.json',
                    },
                )
                for nc in (45, 60, 75, 90)
            },
            # 2026-04-27 (leg-aware extension): with the new leg-only
            # angle term in compute_transition_length, 8/50 samples get
            # capped at t120; raise max to expose what the formula
            # actually wants for those.
            **{
                f'M_c45_t{tmax}': TaskSetting(
                    f'M_c45_t{tmax}',
                    f'[Leg-aware] N_cond=45+45, N_transition '
                    f'clamp [30, {tmax}]. Tests whether releasing the '
                    f'cap helps the leg-heavy samples that the new '
                    f'leg-angle term flagged as needing >120 frames.',
                    {
                        '_use_transition_data': True,
                        '_placement': 'velocity',
                        '_context_policy': 'fixed',
                        '_n_cond_a_frames': 45,
                        '_n_cond_b_frames': 45,
                        '_transition_min': 30,
                        '_transition_max': tmax,
                        '_data_file': 'eval_e14_hq400h_move100.json',
                    },
                )
                for tmax in (150, 180, 240)
            },
        },
        default_metrics=[
            'jitter_pos', 'boundary_accel_jump_a', 'boundary_accel_jump_b',
            'boundary_accel_jump', 'foot_skating_ratio', 'foot_penetration',
        ],
        needs_gt=True,
        caption_aware=False,
        kimodo_comparable=True,
    )

    # --- E15: Prepend to Start Pose (2026-04-27 v2 simplification) ---
    # 2026-04-27: datalist replaced by `eval_e15_prepend_v2.json` (200
    # paired (A, T) items, stratified by category x speed; see
    # tools/build_e15_prepend_v2_data.py). Settings collapsed into a
    # single `default` config + a small sweep grid that is removed once
    # the optimal (N_cond_A, transition_speed) pair is locked.
    #
    # Background:
    #   * A = full motion to prepend before; T = motion whose first
    #     frame defines the desired start pose P = T[0].
    #   * P and A[0] are placed at the same world xz=(0,0); only Y
    #     (pelvis height) may differ. This makes E15 a purely postural
    #     transition.
    #   * The model only sees [P | pad | A[:K]]; the dashboard still
    #     visualizes the full A by stitching the un-fed tail back.
    #
    # Sweep axes (auto-discovered settings, removed after winner lock):
    #   * `sweep_fast`   speed=0.022 m/frame, N_cond_A adaptive
    #   * `sweep_slow`   speed=0.010 m/frame, N_cond_A adaptive
    #   * `sweep_ncond5` speed=0.015 m/frame, N_cond_A fixed=5
    #   * `sweep_ncond60`speed=0.015 m/frame, N_cond_A fixed=60
    #
    # `default` = speed 0.015 + adaptive N_cond_A clamped [15, 90].
    # KIMODO comparison enabled (2026-04-27) via the new
    # build_constraints_e15 prepend implementation in
    # `tools/run_kimodo_all_tasks.py`.
    tasks['E15'] = EvalTask(
        task_id='E15',
        name='Prepend to Start Pose',
        description='Given a full motion A and a desired start pose P, '
                    'prepend N transition frames before A such that the '
                    'sequence starts at P and smoothly reaches A[0]. '
                    'P and A[0] are both at world xz=(0,0); only Y differs '
                    '(P and A[0] may have different pelvis heights, e.g. '
                    'T-pose vs crouch). Output = P + transition + full A. '
                    'v2 (2026-04-27) uses 200 paired (A, T) test cases '
                    'stratified by action category x pelvis speed.',
        mask_builder=build_start_pose_prepend_mask,
        data_file='eval_e15_prepend_v2.json',
        settings={
            'default': TaskSetting(
                'default',
                'Production setting (locked 2026-04-27 from sweep on 30 '
                'samples): N_cond_A=60 fixed, transition speed=0.015 '
                'm/frame with N_transition clamp [15, 90]. Sweep showed '
                'N_cond_A=60 vs adaptive (≈5-7) reduces jitter_pos by '
                '~28% and foot_skating_ratio by ~27% with negligible '
                'cost on boundary_accel_jump (+5%). Speed (0.010 vs '
                '0.022) had little effect, so 0.015 is kept.',
                {
                    '_use_start_pose': True,
                    '_transition_speed': 0.015,
                    '_transition_min': 15,
                    '_transition_max': 90,
                    '_n_cond_a_frames': 60,
                },
            ),
            'sweep_fast': TaskSetting(
                'sweep_fast',
                '[Sweep] Fast transition: speed=0.022 m/frame, '
                'N_cond_A adaptive, clamp [15, 90].',
                {
                    '_use_start_pose': True,
                    '_transition_speed': 0.022,
                    '_transition_min': 15,
                    '_transition_max': 90,
                    '_n_cond_a_policy': 'adaptive',
                },
            ),
            'sweep_slow': TaskSetting(
                'sweep_slow',
                '[Sweep] Slow transition: speed=0.010 m/frame, '
                'N_cond_A adaptive, clamp [30, 180].',
                {
                    '_use_start_pose': True,
                    '_transition_speed': 0.010,
                    '_transition_min': 30,
                    '_transition_max': 180,
                    '_n_cond_a_policy': 'adaptive',
                },
            ),
            'sweep_ncond5': TaskSetting(
                'sweep_ncond5',
                '[Sweep] N_cond_A=5 (short A context), speed=0.015.',
                {
                    '_use_start_pose': True,
                    '_transition_speed': 0.015,
                    '_transition_min': 15,
                    '_transition_max': 90,
                    '_n_cond_a_frames': 5,
                },
            ),
            'sweep_ncond60': TaskSetting(
                'sweep_ncond60',
                '[Sweep] N_cond_A=60 (long A context), speed=0.015.',
                {
                    '_use_start_pose': True,
                    '_transition_speed': 0.015,
                    '_transition_min': 15,
                    '_transition_max': 90,
                    '_n_cond_a_frames': 60,
                },
            ),
        },
        default_metrics=[
            'mpjpe_first_frame', 'jitter_pos', 'boundary_accel_jump',
            'foot_skating_ratio',
        ],
        needs_gt=False,
        caption_aware=False,
        kimodo_comparable=True,
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
