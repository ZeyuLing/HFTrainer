"""Condition sampler v2: two-tier architecture for 198-dim M2M v2 training.

Implements the condition sampling strategy from §4.3 of the design doc:

Tier 1 (60%): Parametric random — three orthogonal axes sampled independently:
  - Temporal: Markov chain for frame-level known/generate decisions
  - Spatial: Beta(1,6) per-joint independent Bernoulli
  - Channel: per-joint rot/pos channel decisions with per-dim position control

Tier 2 (40%): High-frequency patterns for common animation tasks:
  T2-1: Pure generation (mask all 1)
  T2-2: In-between (keep first/last frames)
  T2-3: Prefix (keep first N frames)
  T2-4: Sparse keyframes
  T2-5: End-effector position
  T2-6: Trajectory (translation XZ)
  T2-7: Foot grounding (ankle Y=0)
  T2-8: Edit/repair mode (placeholder, actual corruption in transform)

Output: (T, 198) condition mask + (T, 198) reactive + edit_mode flag.

198-dim layout:
    [0:3]      translation
    [3:135]    22 joints × 6D rot6d
    [135:198]  21 joints × 3D position (pelvis excluded)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

# -----------------------------------------------------------------------
# 198-dim layout constants
# -----------------------------------------------------------------------
MOTION_DIM = 198
TRANSL_DIM = 3
ROT6D_TOTAL = 132       # 22 * 6
POS_TOTAL = 63           # 21 * 3 (pelvis excluded)
NUM_JOINTS = 22
NUM_POS_JOINTS = 21      # joints 1..21 (pelvis excluded)

# Joint index to position base: joint j (j=1..21) position starts at 135 + (j-1)*3
def _joint_rot_slice(j: int) -> slice:
    """Return slice for joint j's rotation dims in 198-dim."""
    return slice(3 + j * 6, 3 + (j + 1) * 6)

def _joint_pos_slice(j: int) -> Optional[slice]:
    """Return slice for joint j's position dims (None for pelvis j=0)."""
    if j == 0:
        return None
    return slice(135 + (j - 1) * 3, 135 + j * 3)


# -----------------------------------------------------------------------
# Tier 1: Parametric random sampling
# -----------------------------------------------------------------------

def sample_temporal_markov(T: int, rng: np.random.RandomState) -> np.ndarray:
    """Markov chain generating (T,) binary sequence. 0=known, 1=generate."""
    p_start_known = rng.uniform(0.0, 1.0)
    p_stay_known = rng.beta(2, 2)
    p_stay_gen = rng.beta(2, 2)

    seq = np.zeros(T, dtype=np.int32)
    seq[0] = 0 if rng.random() < p_start_known else 1
    for i in range(1, T):
        if seq[i - 1] == 0:  # known
            seq[i] = 0 if rng.random() < p_stay_known else 1
        else:  # generate
            seq[i] = 1 if rng.random() < p_stay_gen else 0
    return seq


def sample_spatial_bernoulli(rng: np.random.RandomState) -> List[int]:
    """Per-joint independent Bernoulli, p_joint ~ Beta(1,6). Returns joint indices."""
    p_joint = rng.beta(1, 6)  # E=0.14, ~2-3 joints typically
    selected = [j for j in range(NUM_JOINTS) if rng.random() < p_joint]
    if len(selected) == 0:
        selected = [rng.randint(0, NUM_JOINTS)]
    return selected


def sample_channel(rng: np.random.RandomState) -> Tuple[bool, Tuple[bool, bool, bool]]:
    """Per-joint channel decision: (rot_keep, (pos_x_keep, pos_y_keep, pos_z_keep))."""
    rot_keep = rng.random() < 0.6
    pos_keep_prob = rng.beta(2, 1)
    px = rng.random() < pos_keep_prob
    py = rng.random() < pos_keep_prob
    pz = rng.random() < pos_keep_prob

    if not rot_keep and not any([px, py, pz]):
        py = True  # at least one channel

    return rot_keep, (px, py, pz)


def sample_translation(
    known_frames: np.ndarray,
    mask: np.ndarray,
    rng: np.random.RandomState,
) -> None:
    """Translation [0:3] independent sampling. Modifies mask in-place."""
    trans_keep = rng.random() < 0.2
    if not trans_keep:
        return

    pos_keep_prob = rng.beta(2, 1)
    tx = rng.random() < pos_keep_prob
    ty = rng.random() < pos_keep_prob
    tz = rng.random() < pos_keep_prob
    if not any([tx, ty, tz]):
        tx, tz = True, True  # fallback: at least XZ

    heading_keep = rng.random() < 0.3

    for f in known_frames:
        if tx:
            mask[f, 0] = 0
        if ty:
            mask[f, 1] = 0
        if tz:
            mask[f, 2] = 0
        if heading_keep:
            mask[f, 3:9] = 0  # root rot6d


def sample_tier1(T: int, rng: np.random.RandomState) -> np.ndarray:
    """Tier 1: parametric random sampling. Returns (T, 198) mask."""
    mask = np.ones((T, MOTION_DIM), dtype=np.float32)

    # 1. Temporal
    temporal_seq = sample_temporal_markov(T, rng)
    known_frames = np.where(temporal_seq == 0)[0]
    if len(known_frames) == 0:
        return mask

    # 2. Spatial (22 joints)
    per_frame_spatial = rng.random() < 0.1
    shared_joints = sample_spatial_bernoulli(rng) if not per_frame_spatial else None

    for f in known_frames:
        joints = sample_spatial_bernoulli(rng) if per_frame_spatial else shared_joints
        for j in joints:
            rot_keep, (px, py, pz) = sample_channel(rng)
            # Rotation [3+j*6 : 3+(j+1)*6]
            if rot_keep:
                mask[f, 3 + j * 6: 3 + (j + 1) * 6] = 0
            # Position (only joints 1..21)
            if j > 0:
                pos_base = 135 + (j - 1) * 3
                if px:
                    mask[f, pos_base] = 0
                if py:
                    mask[f, pos_base + 1] = 0
                if pz:
                    mask[f, pos_base + 2] = 0

    # 3. Translation (independent)
    sample_translation(known_frames, mask, rng)

    return mask


# -----------------------------------------------------------------------
# Tier 2: High-frequency pattern templates
# -----------------------------------------------------------------------

# End-effector joint indices (SMPL-22 indexing)
EE_WRISTS = [20, 21]  # L_Wrist, R_Wrist
EE_ANKLES = [7, 8]     # L_Ankle, R_Ankle
EE_ALL = EE_ANKLES + EE_WRISTS

# Tier 2 pattern names and weights
T2_PATTERNS = [
    'pure_gen', 'inbetween', 'prefix', 'keyframes',
    'end_effector', 'trajectory', 'foot_ground', 'edit_repair',
]
T2_WEIGHTS = np.array([0.20, 0.20, 0.125, 0.125, 0.125, 0.10, 0.075, 0.05])
T2_WEIGHTS = T2_WEIGHTS / T2_WEIGHTS.sum()


def _keep_full_frame(mask: np.ndarray, f: int) -> None:
    """Set frame f to all known (mask=0) across all 198 dims."""
    mask[f, :] = 0


def sample_tier2_pure_gen(T: int, mask: np.ndarray, rng: np.random.RandomState) -> bool:
    """T2-1: Pure generation — mask all 1."""
    # mask already all 1
    return False


def sample_tier2_inbetween(T: int, mask: np.ndarray, rng: np.random.RandomState) -> bool:
    """T2-2: In-between — keep first and last N frames."""
    n_start = rng.randint(1, min(6, max(2, T // 4)))
    n_end = rng.randint(1, min(6, max(2, T // 4)))
    for f in range(n_start):
        _keep_full_frame(mask, f)
    for f in range(max(n_start, T - n_end), T):
        _keep_full_frame(mask, f)
    return False


def sample_tier2_prefix(T: int, mask: np.ndarray, rng: np.random.RandomState) -> bool:
    """T2-3: Prefix — keep first N frames."""
    n_keep = rng.randint(1, max(2, T // 2))
    for f in range(n_keep):
        _keep_full_frame(mask, f)
    return False


def sample_tier2_keyframes(T: int, mask: np.ndarray, rng: np.random.RandomState) -> bool:
    """T2-4: Sparse keyframes — K frames fully known."""
    K = min(max(1, rng.geometric(p=0.15)), T)
    frames = sorted(rng.choice(T, size=min(K, T), replace=False))
    for f in frames:
        _keep_full_frame(mask, f)
    return False


def sample_tier2_end_effector(T: int, mask: np.ndarray, rng: np.random.RandomState) -> bool:
    """T2-5: End-effector position constraints."""
    n_ee = rng.randint(1, min(5, len(EE_ALL) + 1))
    ee_joints = rng.choice(EE_ALL, size=n_ee, replace=False).tolist()
    K = min(max(1, rng.geometric(p=0.1)), T)
    frames = sorted(rng.choice(T, size=min(K, T), replace=False))
    for f in frames:
        for j in ee_joints:
            ps = _joint_pos_slice(j)
            if ps is not None:
                mask[f, ps] = 0  # position XYZ
    return False


def sample_tier2_trajectory(T: int, mask: np.ndarray, rng: np.random.RandomState) -> bool:
    """T2-6: Trajectory — translation XZ on dense/sparse frames."""
    K = rng.randint(max(1, T // 10), T + 1)
    frames = sorted(rng.choice(T, size=min(K, T), replace=False))
    for f in frames:
        mask[f, 0] = 0  # trans X
        mask[f, 2] = 0  # trans Z
        if rng.random() < 0.4:
            mask[f, 3:9] = 0  # root rotation (heading)
    return False


def sample_tier2_foot_ground(T: int, mask: np.ndarray, rng: np.random.RandomState) -> bool:
    """T2-7: Foot grounding — ankle pos_Y=0 constraint."""
    ankle_joints = [7, 8]
    K = rng.randint(max(1, T // 5), T + 1)
    frames = sorted(rng.choice(T, size=min(K, T), replace=False))
    for f in frames:
        for j in ankle_joints:
            pos_base = 135 + (j - 1) * 3
            mask[f, pos_base + 1] = 0  # pos_Y only
    return False


def sample_tier2_edit_repair(T: int, mask: np.ndarray, rng: np.random.RandomState) -> bool:
    """T2-8: Edit/repair mode. Returns True to signal editing mode."""
    # Actual corruption is applied by the transform (PrepareM2Mv2Condition).
    # Here we return edit_mode=True so the transform knows to apply corruption.
    return True


_T2_FN = {
    'pure_gen': sample_tier2_pure_gen,
    'inbetween': sample_tier2_inbetween,
    'prefix': sample_tier2_prefix,
    'keyframes': sample_tier2_keyframes,
    'end_effector': sample_tier2_end_effector,
    'trajectory': sample_tier2_trajectory,
    'foot_ground': sample_tier2_foot_ground,
    'edit_repair': sample_tier2_edit_repair,
}


def sample_tier2(T: int, rng: np.random.RandomState,
                 tier2_weights: Optional[Dict[str, float]] = None) -> Tuple[np.ndarray, bool]:
    """Tier 2: sample from high-frequency patterns.

    Args:
        T: Number of frames.
        rng: Random state.
        tier2_weights: Optional dict mapping pattern names to weights.
            If provided, overrides the default T2_WEIGHTS.

    Returns:
        mask: (T, 198) condition mask.
        edit_mode: True if edit/repair pattern was selected.
    """
    if tier2_weights is not None:
        patterns = list(tier2_weights.keys())
        weights = np.array([tier2_weights[p] for p in patterns])
        weights = weights / weights.sum()
        pattern = rng.choice(patterns, p=weights)
    else:
        pattern = rng.choice(T2_PATTERNS, p=T2_WEIGHTS)
    mask = np.ones((T, MOTION_DIM), dtype=np.float32)
    edit_mode = _T2_FN[pattern](T, mask, rng)
    return mask, edit_mode


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

def sample_condition(
    T: int,
    rng: np.random.RandomState,
    tier2_prob: float = 0.4,
    editing_prob: float = 0.15,
    tier2_weights: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, bool]:
    """Two-tier condition sampling.

    Args:
        T: Number of frames.
        rng: Random state.
        tier2_prob: Probability of using Tier 2.
        editing_prob: Probability of editing mode (Tier 1 only).
        tier2_weights: Optional dict mapping Tier 2 pattern names to weights.

    Returns:
        mask: (T, 198) float32 condition mask, 0=known, 1=generate.
        edit_mode: True if this sample should use editing mode.
    """
    use_tier2 = rng.random() < tier2_prob
    edit_mode = False

    if use_tier2:
        mask, edit_mode = sample_tier2(T, rng, tier2_weights=tier2_weights)
    else:
        mask = sample_tier1(T, rng)

        # Editing mode overlay for Tier 1 completion samples
        if not edit_mode and rng.random() < editing_prob:
            edit_mode = True

    # 25% chance to overlay trajectory constraint
    if rng.random() < 0.25:
        K = rng.randint(max(1, T // 10), T + 1)
        frames = sorted(rng.choice(T, size=min(K, T), replace=False))
        for f in frames:
            mask[f, 0] = 0  # trans X
            mask[f, 2] = 0  # trans Z

    return mask, edit_mode


# -----------------------------------------------------------------------
# Mask expansion utilities for 198-dim
# -----------------------------------------------------------------------

def expand_tj_mask_to_198(
    tj_mask: np.ndarray,
    trans_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Expand (T, 22) joint mask to (T, 198) full mask.

    Args:
        tj_mask: (T, 22) per-joint mask, 1=corrupted/generate.
        trans_mask: Optional (T,) translation mask.

    Returns:
        (T, 198) mask.
    """
    T = tj_mask.shape[0]
    mask = np.zeros((T, MOTION_DIM), dtype=np.float32)

    # Translation
    if trans_mask is not None:
        for f in range(T):
            if trans_mask[f] > 0.5:
                mask[f, 0:3] = 1.0

    # Per-joint: rotation + position
    for j in range(NUM_JOINTS):
        # Rotation
        for f in range(T):
            if tj_mask[f, j] > 0.5:
                mask[f, 3 + j * 6: 3 + (j + 1) * 6] = 1.0
        # Position (joints 1..21 only)
        if j > 0:
            pos_base = 135 + (j - 1) * 3
            for f in range(T):
                if tj_mask[f, j] > 0.5:
                    mask[f, pos_base: pos_base + 3] = 1.0

    return mask


def apply_mask_perturbation(
    mask: np.ndarray,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Apply over-mask perturbation to a corruption mask.

    Only makes the mask larger (never smaller) — ensures corrupted regions
    are always included in the generation mask.

    Perturbation types:
        precise (25%): keep mask as-is
        dilated_small (25%): temporal dilation ±2-5 frames
        dilated_large (15%): temporal dilation ±5-15 frames
        joint_expand (20%): expand to kinematic neighbors
        full_seq (15%): extend corrupted joints to all frames

    Args:
        mask: (T, 198) base mask.
        rng: Random state.

    Returns:
        (T, 198) perturbed mask (always >= original).
    """
    T = mask.shape[0]
    mode = rng.choice(
        ['precise', 'dilated_small', 'dilated_large', 'joint_expand', 'full_seq'],
        p=[0.25, 0.25, 0.15, 0.20, 0.15],
    )

    if mode == 'precise':
        return mask

    result = mask.copy()

    if mode in ('dilated_small', 'dilated_large'):
        if mode == 'dilated_small':
            dilate = rng.randint(2, 6)
        else:
            dilate = rng.randint(5, 16)
        # Apply temporal dilation to each dimension
        for d in range(MOTION_DIM):
            col = mask[:, d]
            if col.sum() == 0:
                continue
            active = np.where(col > 0.5)[0]
            for f in active:
                start = max(0, f - dilate)
                end = min(T, f + dilate + 1)
                result[start:end, d] = 1.0

    elif mode == 'joint_expand':
        # Expand each corrupted joint to its kinematic parent and children
        from hftrainer.datasets.motion.motionhub.transforms.fk_utils import SMPL22_PARENTS
        children = [[] for _ in range(NUM_JOINTS)]
        for j, p in enumerate(SMPL22_PARENTS):
            if p >= 0:
                children[p].append(j)

        for f in range(T):
            for j in range(NUM_JOINTS):
                rs = _joint_rot_slice(j)
                if mask[f, rs].max() > 0.5:
                    # Expand to parent
                    p = SMPL22_PARENTS[j]
                    if p >= 0:
                        result[f, _joint_rot_slice(p)] = 1.0
                        ps = _joint_pos_slice(p)
                        if ps is not None:
                            result[f, ps] = 1.0
                    # Expand to children
                    for c in children[j]:
                        result[f, _joint_rot_slice(c)] = 1.0
                        ps = _joint_pos_slice(c)
                        if ps is not None:
                            result[f, ps] = 1.0

    elif mode == 'full_seq':
        # For each corrupted joint, extend to all frames
        for j in range(NUM_JOINTS):
            rs = _joint_rot_slice(j)
            if mask[:, rs.start].max() > 0.5:
                result[:, rs] = 1.0
                ps = _joint_pos_slice(j)
                if ps is not None:
                    result[:, ps] = 1.0

    return result
