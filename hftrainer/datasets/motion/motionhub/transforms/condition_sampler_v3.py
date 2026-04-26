"""Condition sampler v3: Universal Rank-K Boolean Tensor Prior for M2M v2 masks.

This module replaces the template-mixture design of ``condition_sampler_v2``
with a single mathematically universal prior. See
``docs/design/mask_prior_rank_k.md`` for the full specification and coverage
analysis.

Core idea
---------
Every mask ``M ∈ {0,1}^(T × 198)`` is written as a Boolean-OR of ``K``
rank-1 atoms::

    M = ⋁_{k=1..K} (t_k ⊗ d_k)

where ``t_k ∈ {0,1}^T`` picks a frame subset (temporal pattern) and
``d_k ∈ {0,1}^198`` picks a dimension subset (spatial/channel pattern).

Temporal prior ``πT`` mixes 6 primitives (all / empty / interval /
periodic / renewal / markov).

Dimensional prior ``πD`` is hierarchical:
  - ``kind ∈ {rot_only, pos_only, trans_only, mixed}``
  - each kind samples joints from an **anatomical dictionary** (17 groups)
    or a Bernoulli path, and channel subsets ``C ⊂ {x,y,z}``.

Convention
----------
Mask semantics stays identical to v2: ``1 = generate, 0 = known``. The
rank-1 atoms describe the *known* region (lock mask ``L``); the returned
generate-mask is ``M = 1 - L``.

Public entry points
-------------------
``sample_condition_v3(T, rng, **hparams) -> (mask[T, 198], edit_mode)``
    Drop-in replacement for :func:`condition_sampler_v2.sample_condition`.

``sample_mask_rank_k(T, rng, **hparams) -> np.ndarray``
    The pure mask sampler (exposed for testing / coverage audit).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# 198-dim layout constants (kept identical to condition_sampler_v2)
# ---------------------------------------------------------------------------
MOTION_DIM = 198
TRANSL_DIM = 3          # [0:3]
ROT6D_TOTAL = 132        # [3:135]  22 joints × 6
POS_TOTAL = 63           # [135:198] 21 joints × 3 (pelvis excluded)
NUM_JOINTS = 22
NUM_POS_JOINTS = 21      # joints 1..21 (pelvis excluded)

ROT_START = 3
ROT_END = 135
POS_START = 135
POS_END = 198


def _rot_slice(joint: int) -> slice:
    """Return the 6-D slice for joint `joint`'s rot6d in 198-dim."""
    return slice(ROT_START + joint * 6, ROT_START + (joint + 1) * 6)


def _pos_slice(joint: int) -> Optional[slice]:
    """Return the 3-D slice for joint `joint`'s position, or None for pelvis."""
    if joint == 0:
        return None
    return slice(POS_START + (joint - 1) * 3, POS_START + joint * 3)


# ---------------------------------------------------------------------------
# Anatomical joint dictionary (17 entries). Indexing matches fk_utils.SMPL22_PARENTS.
# ---------------------------------------------------------------------------
# Keep `end_effectors` for backward-compatibility with v2 naming; `hands_feet`
# is the corrected superset that actually covers E4 setting C.
ANATOMICAL_GROUPS: Dict[str, Tuple[int, ...]] = {
    'all':             tuple(range(22)),
    'pelvis':          (0,),
    'spine_chain':     (0, 3, 6, 9, 12, 15),
    'upper_body':      (3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21),
    'lower_body':      (0, 1, 2, 4, 5, 7, 8, 10, 11),
    'arms':            (13, 14, 16, 17, 18, 19, 20, 21),
    'legs':            (1, 2, 4, 5, 7, 8, 10, 11),
    'left_arm':        (13, 16, 18, 20),
    'right_arm':       (14, 17, 19, 21),
    'left_leg':        (1, 4, 7, 10),
    'right_leg':       (2, 5, 8, 11),
    'ankles':          (7, 8),
    'feet':            (10, 11),
    'wrists':          (20, 21),
    'hands_feet':      (10, 11, 20, 21),
    'end_effectors':   (7, 8, 20, 21),
    'head':            (15,),
}
ANATOMICAL_KEYS: List[str] = list(ANATOMICAL_GROUPS.keys())

# Non-uniform weights for anatomical groups. `end_effectors`, `hands_feet`,
# `ankles`, `wrists`, `upper_body`, `lower_body`, `spine_chain`, `all`,
# `pelvis` are common in eval; `head` and per-limb subsets are rarer.
# This up-weights typical eval signatures while keeping every group with
# non-zero support.
ANATOMICAL_WEIGHTS: Dict[str, float] = {
    'all':             2.0,
    'pelvis':          1.5,
    'spine_chain':     1.5,
    'upper_body':      2.0,
    'lower_body':      2.0,
    'arms':            1.5,
    'legs':            1.5,
    'left_arm':        1.0,
    'right_arm':       1.0,
    'left_leg':        1.0,
    'right_leg':       1.0,
    'ankles':          2.0,
    'feet':            1.5,
    'wrists':          2.0,
    'hands_feet':      2.0,
    'end_effectors':   2.0,
    'head':            0.5,
}
_ANATOMICAL_W_ARR = np.array([ANATOMICAL_WEIGHTS[k] for k in ANATOMICAL_KEYS],
                             dtype=np.float64)
_ANATOMICAL_W_ARR = _ANATOMICAL_W_ARR / _ANATOMICAL_W_ARR.sum()


# ---------------------------------------------------------------------------
# πT: temporal primitive distributions
# ---------------------------------------------------------------------------

TEMPORAL_PRIMITIVES = ('all', 'empty', 'interval', 'periodic', 'renewal', 'markov')

# `PERIODIC_STEPS` are *anchor* periods that match the most common eval
# settings (E3 every 15/30/60, E4 every 5/10/15/20). They are used 70 %
# of the time; the other 30 % draws ``p`` uniformly from
# ``[PERIODIC_STEP_RANGE]`` so that *any* integer period in support of a
# real motion task (e.g. arbitrary p=7 or p=12 future tasks) is reachable.
PERIODIC_STEPS = (5, 10, 15, 20, 30, 60)
PERIODIC_STEP_RANGE = (2, 90)        # inclusive bounds for the random branch
PERIODIC_ANCHOR_PROB = 0.70          # mixture weight on the anchor set

# Default temporal primitive weights. Tuned from coverage audit: interval
# and periodic are up-weighted because they map directly to the most
# common eval mask patterns (E2/E3/E4/E7/E15). `empty` is down-weighted
# because an empty atom contributes nothing to the union.
DEFAULT_TEMPORAL_WEIGHTS: Dict[str, float] = {
    'all': 2.0,
    'empty': 0.3,
    'interval': 3.5,
    'periodic': 4.0,
    'renewal': 1.5,
    'markov': 1.0,
}


def _t_all(T: int, rng: np.random.RandomState) -> np.ndarray:
    return np.ones(T, dtype=np.uint8)


def _t_empty(T: int, rng: np.random.RandomState) -> np.ndarray:
    return np.zeros(T, dtype=np.uint8)


def _t_interval(T: int, rng: np.random.RandomState) -> np.ndarray:
    """Contiguous window ``[a, a+ℓ)``.

    Length mode (40/30/30 mixture so that short/medium/long all have
    material probability, instead of log-uniform which over-weights very
    short lengths):

        * 40 %  ``ℓ ~ Uniform[1, max(1, T // 10)]`` — single-frame and
          short anchors (E2 start_1f/end_1f, E7).
        * 30 %  ``ℓ ~ Uniform[max(1, T // 10), max(2, T // 3)]`` — mid-
          range (E2 pre20/post20, E15 prepend).
        * 30 %  ``ℓ ~ Uniform[max(2, T // 3), T]`` — long interval (E2
          pre50+, E15 long-prepend, loose stitching).

    Start-position mode (after length is chosen):

        * 1/3 prefix (``a = 0``) — E2.start_*, E2.pre*, E7, E15
        * 1/3 suffix (``a = T - ℓ``) — E2.end_*, E2.post*
        * 1/3 uniform middle — interior mask
    """
    t = np.zeros(T, dtype=np.uint8)
    if T <= 0:
        return t
    short_hi = max(1, T // 10)
    mid_hi = max(2, T // 3)
    u_len = rng.random()
    if u_len < 0.40:
        ell = int(rng.randint(1, short_hi + 1))
    elif u_len < 0.70:
        ell = int(rng.randint(short_hi, mid_hi + 1))
    else:
        ell = int(rng.randint(mid_hi, T + 1))
    ell = max(1, min(ell, T))

    u_pos = rng.random()
    if u_pos < 1.0 / 3.0:
        a = 0
    elif u_pos < 2.0 / 3.0:
        a = T - ell
    else:
        a = rng.randint(0, T - ell + 1)
    t[a:a + ell] = 1
    return t


def _t_periodic(T: int, rng: np.random.RandomState) -> np.ndarray:
    """Every p-th frame from phase φ.

    p is drawn from a mixture:
      * w.p. ``PERIODIC_ANCHOR_PROB``  : pick from the anchor set
        ``PERIODIC_STEPS`` (eval-aligned: 5/10/15/20/30/60),
      * otherwise                      : ``p ~ DiscreteUniform[2, max_p]``
        where ``max_p = min(PERIODIC_STEP_RANGE[1], max(2, T // 2))``.

    This ensures every integer period in [2, T//2] has non-zero
    probability, so future eval tasks with arbitrary p (e.g. p=7, p=12)
    are inside the prior support, while still favouring the empirically
    common values during training.
    """
    t = np.zeros(T, dtype=np.uint8)
    if T <= 0:
        return t
    lo, hi = PERIODIC_STEP_RANGE
    max_p = max(lo, min(hi, max(2, T // 2)))
    if rng.random() < PERIODIC_ANCHOR_PROB:
        anchors = [s for s in PERIODIC_STEPS if lo <= s <= max_p]
        if anchors:
            p = int(rng.choice(anchors))
        else:
            p = int(rng.randint(lo, max_p + 1))
    else:
        p = int(rng.randint(lo, max_p + 1))
    phi = rng.randint(0, p)
    idx = np.arange(phi, T, p, dtype=np.int64)
    idx = idx[idx < T]
    if len(idx) > 0:
        t[idx] = 1
    return t


def _t_renewal(T: int, rng: np.random.RandomState) -> np.ndarray:
    """i.i.d. gaps g_i ~ Geometric(ρ), ρ ~ log-uniform in [0.02, 0.5]."""
    t = np.zeros(T, dtype=np.uint8)
    if T <= 0:
        return t
    rho = float(np.exp(rng.uniform(np.log(0.02), np.log(0.5))))
    # We sample ~ T*rho hits on average; over-sample then clip
    n_expected = max(1, int(np.ceil(T * rho * 2 + 4)))
    gaps = rng.geometric(p=rho, size=n_expected)
    pos = np.cumsum(gaps) - 1  # gap=1 means the next frame is the next hit
    pos = pos[(pos >= 0) & (pos < T)]
    if len(pos) > 0:
        t[pos] = 1
    else:
        # Guarantee at least 1 hit so the atom is not vacuous
        t[rng.randint(0, T)] = 1
    return t


def _t_markov(T: int, rng: np.random.RandomState) -> np.ndarray:
    """2-state Markov chain, p_stay_0 / p_stay_1 ~ Beta(2,2)."""
    t = np.zeros(T, dtype=np.uint8)
    if T <= 0:
        return t
    p_stay_0 = float(rng.beta(2.0, 2.0))
    p_stay_1 = float(rng.beta(2.0, 2.0))
    state = int(rng.random() < 0.5)
    for i in range(T):
        t[i] = state
        if state == 0:
            state = 0 if rng.random() < p_stay_0 else 1
        else:
            state = 1 if rng.random() < p_stay_1 else 0
    return t


_T_FN = {
    'all': _t_all,
    'empty': _t_empty,
    'interval': _t_interval,
    'periodic': _t_periodic,
    'renewal': _t_renewal,
    'markov': _t_markov,
}


def sample_temporal(
    T: int,
    rng: np.random.RandomState,
    primitive_weights: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, str]:
    """Sample one temporal pattern ``t ∈ {0,1}^T``.

    Returns
    -------
    (t, primitive_name) pair; ``primitive_name`` is the chosen πT primitive
    (useful for diagnostics / coverage audit).
    """
    primitives = TEMPORAL_PRIMITIVES
    weights = primitive_weights if primitive_weights is not None else DEFAULT_TEMPORAL_WEIGHTS
    w = np.array([weights.get(p, 0.0) for p in primitives], dtype=np.float64)
    w = w / w.sum()
    name = primitives[int(rng.choice(len(primitives), p=w))]
    return _T_FN[name](T, rng), name


# ---------------------------------------------------------------------------
# πD: dimensional priors
# ---------------------------------------------------------------------------

KINDS = ('rot_only', 'pos_only', 'trans_only', 'mixed', 'all_dim')
# Kind weights are tuned from the coverage audit: the `all_dim` branch is
# essential because many eval tasks (E2/E3/E7/E8/E15) lock *all* 198 dims
# at the selected frames, and it is exponentially unlikely that a composed
# `mixed` atom happens to hit rot+pos+trans simultaneously with `all`
# joints and all xyz channels.
DEFAULT_KIND_WEIGHTS = {
    'rot_only': 0.22,
    'pos_only': 0.30,  # up-weighted: E4/E6 end-effector + foot-ground
    'trans_only': 0.10,
    'mixed': 0.18,
    'all_dim': 0.20,
}

# Joint-subset selection mode inside a `rot_only` / `pos_only` atom.
#   anatomical: draw one of ANATOMICAL_GROUPS
#   bernoulli:  Beta(1.5, 4) prior with Bernoulli(p) inclusion (small random set)
#   single:     one random joint
JOINT_SUBSET_MODES = ('anatomical', 'bernoulli', 'single')
DEFAULT_SUBSET_WEIGHTS = (0.5, 0.3, 0.2)


def _pick_joint_subset(
    rng: np.random.RandomState,
    include_pelvis: bool,
    subset_weights: Sequence[float] = DEFAULT_SUBSET_WEIGHTS,
) -> List[int]:
    """Sample a joint subset under the three-mode joint prior.

    ``include_pelvis``: if False, restrict to joints 1..21 (used for
    position-only atoms).
    """
    mode = JOINT_SUBSET_MODES[int(rng.choice(3, p=np.asarray(subset_weights) /
                                             np.sum(subset_weights)))]

    if mode == 'anatomical':
        key = ANATOMICAL_KEYS[int(rng.choice(len(ANATOMICAL_KEYS), p=_ANATOMICAL_W_ARR))]
        joints = list(ANATOMICAL_GROUPS[key])
        if not include_pelvis:
            joints = [j for j in joints if j > 0]
        if not joints:  # anatomical group was just pelvis: fallback
            joints = [rng.randint(1, NUM_JOINTS)]
        return joints

    joint_range = list(range(NUM_JOINTS)) if include_pelvis else list(range(1, NUM_JOINTS))

    if mode == 'bernoulli':
        p = float(rng.beta(1.5, 4.0))
        selected = [j for j in joint_range if rng.random() < p]
        if not selected:
            selected = [joint_range[rng.randint(0, len(joint_range))]]
        return selected

    # mode == 'single'
    return [joint_range[rng.randint(0, len(joint_range))]]


def _pick_xyz_subset(rng: np.random.RandomState) -> Tuple[bool, bool, bool]:
    """Sample a non-empty subset of ``{x, y, z}``.

    Weighted so the full ``xyz`` subset and the ``xz`` subset (both
    essential for E4 end-effector and E5 trajectory) appear with
    probability ≳ 0.25 each; the remaining 5 subsets split the rest.

    Weight table (before normalisation):
        (x,y,z) = 1_1_1  → 4   (E4)
        (x,_,z) = 1_0_1  → 4   (E5 trans_xz)
        (_,y,_) = 0_1_0  → 2   (E6 foot_y / vertical-only)
        (x,_,_) / (_,_,z) / (x,y,_) / (_,y,z) → 1 each
    """
    # (x, y, z) tuples with weights.
    subsets = (
        (1, 1, 1, 4.0),
        (1, 0, 1, 4.0),
        (0, 1, 0, 2.0),
        (1, 0, 0, 1.0),
        (0, 0, 1, 1.0),
        (1, 1, 0, 1.0),
        (0, 1, 1, 1.0),
    )
    weights = np.array([s[3] for s in subsets], dtype=np.float64)
    weights = weights / weights.sum()
    idx = int(rng.choice(len(subsets), p=weights))
    x, y, z, _ = subsets[idx]
    return bool(x), bool(y), bool(z)


def _atom_rot_only(rng: np.random.RandomState) -> np.ndarray:
    """One rank-1 atom locking rot6d of a joint subset (all frames)."""
    d = np.zeros(MOTION_DIM, dtype=np.uint8)
    joints = _pick_joint_subset(rng, include_pelvis=True)
    for j in joints:
        s = _rot_slice(j)
        d[s.start:s.stop] = 1
    return d


def _atom_pos_only(rng: np.random.RandomState) -> np.ndarray:
    """One rank-1 atom locking position channels of a joint subset."""
    d = np.zeros(MOTION_DIM, dtype=np.uint8)
    joints = _pick_joint_subset(rng, include_pelvis=False)
    px, py, pz = _pick_xyz_subset(rng)
    for j in joints:
        s = _pos_slice(j)
        if s is None:
            continue
        base = s.start
        if px: d[base + 0] = 1
        if py: d[base + 1] = 1
        if pz: d[base + 2] = 1
    return d


def _atom_trans_only(rng: np.random.RandomState) -> np.ndarray:
    """One rank-1 atom locking translation channels."""
    d = np.zeros(MOTION_DIM, dtype=np.uint8)
    tx, ty, tz = _pick_xyz_subset(rng)
    if tx: d[0] = 1
    if ty: d[1] = 1
    if tz: d[2] = 1
    return d


def _atom_mixed(
    rng: np.random.RandomState,
    sub_include_prob: float = 0.5,
) -> np.ndarray:
    """Mixed atom = OR of rot / pos / trans sub-atoms (each with dropout)."""
    d = np.zeros(MOTION_DIM, dtype=np.uint8)
    if rng.random() < sub_include_prob:
        d |= _atom_rot_only(rng)
    if rng.random() < sub_include_prob:
        d |= _atom_pos_only(rng)
    if rng.random() < sub_include_prob:
        d |= _atom_trans_only(rng)
    # Guarantee non-empty
    if d.sum() == 0:
        d |= _atom_rot_only(rng)
    return d


def _atom_all_dim(rng: np.random.RandomState) -> np.ndarray:
    """Lock every dim (``d = 1_198``).

    This is the correct primitive for eval tasks that pin entire frames
    (E2 inbetween, E3 keyframes, E7 first-frame, E15 prepend). Composing
    rot + pos + trans manually inside ``mixed`` cannot hit this
    sub-manifold with meaningful probability because each sub-atom picks a
    *subset* of its dims.
    """
    return np.ones(MOTION_DIM, dtype=np.uint8)


_KIND_FN = {
    'rot_only': _atom_rot_only,
    'pos_only': _atom_pos_only,
    'trans_only': _atom_trans_only,
    'mixed': _atom_mixed,
    'all_dim': _atom_all_dim,
}


def sample_dimensional(
    rng: np.random.RandomState,
    kind_weights: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, str]:
    """Sample one dimensional atom ``d ∈ {0,1}^198``.

    Returns
    -------
    (d, kind) pair; ``kind`` is the chosen dimensional kind (useful for
    diagnostics).
    """
    if kind_weights is None:
        kind_weights = DEFAULT_KIND_WEIGHTS
    kinds = KINDS
    w = np.array([kind_weights.get(k, 0.0) for k in kinds], dtype=np.float64)
    w = w / w.sum()
    kind = kinds[int(rng.choice(len(kinds), p=w))]
    return _KIND_FN[kind](rng), kind


# ---------------------------------------------------------------------------
# πK: number-of-atoms prior
# ---------------------------------------------------------------------------

# Probability over K ∈ {0, 1, 2, 3, 4}.
DEFAULT_K_WEIGHTS = (0.10, 0.55, 0.25, 0.07, 0.03)


def sample_k(
    rng: np.random.RandomState,
    k_weights: Sequence[float] = DEFAULT_K_WEIGHTS,
) -> int:
    w = np.asarray(k_weights, dtype=np.float64)
    w = w / w.sum()
    return int(rng.choice(len(w), p=w))


# ---------------------------------------------------------------------------
# Main sampler
# ---------------------------------------------------------------------------

def sample_mask_rank_k(
    T: int,
    rng: np.random.RandomState,
    k_weights: Sequence[float] = DEFAULT_K_WEIGHTS,
    temporal_weights: Optional[Dict[str, float]] = None,
    kind_weights: Optional[Dict[str, float]] = None,
    return_trace: bool = False,
):
    """Sample a mask ``M[T, 198]`` under the Rank-K prior.

    Convention: output mask uses ``1 = generate, 0 = known`` (same as v2).

    Parameters
    ----------
    T : int
        Number of frames.
    rng : np.random.RandomState
        Source of randomness.
    k_weights : sequence
        Probability over K ∈ {0, 1, ..., len(k_weights) - 1}.
    temporal_weights : dict, optional
        Optional override for πT primitive weights.
    kind_weights : dict, optional
        Optional override for πD kind weights.
    return_trace : bool
        If True, also return a diagnostic dict (``K``, atoms, primitive
        names).

    Returns
    -------
    mask : np.ndarray, shape (T, 198), dtype float32
        ``1`` = generate, ``0`` = known (conditioned).
    trace : dict, optional
        When ``return_trace=True``.
    """
    K = sample_k(rng, k_weights)
    lock = np.zeros((T, MOTION_DIM), dtype=np.uint8)

    atoms: List[Dict] = []
    for _ in range(K):
        t_vec, t_name = sample_temporal(T, rng, primitive_weights=temporal_weights)
        d_vec, d_kind = sample_dimensional(rng, kind_weights=kind_weights)
        if t_vec.sum() == 0 or d_vec.sum() == 0:
            if return_trace:
                atoms.append({
                    't_primitive': t_name, 'd_kind': d_kind,
                    't_sum': int(t_vec.sum()), 'd_sum': int(d_vec.sum()),
                    'active': False,
                })
            continue
        # Boolean OR accumulation: lock[t, d] = 1 wherever atom fires.
        lock |= np.outer(t_vec, d_vec).astype(np.uint8)
        if return_trace:
            atoms.append({
                't_primitive': t_name, 'd_kind': d_kind,
                't_sum': int(t_vec.sum()), 'd_sum': int(d_vec.sum()),
                'active': True,
            })

    mask = (1 - lock).astype(np.float32)  # 1 = generate
    if return_trace:
        trace = {'K': K, 'atoms': atoms}
        return mask, trace
    return mask


def sample_condition_v3(
    T: int,
    rng: np.random.RandomState,
    editing_prob: float = 0.08,
    k_weights: Sequence[float] = DEFAULT_K_WEIGHTS,
    temporal_weights: Optional[Dict[str, float]] = None,
    kind_weights: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, bool]:
    """Drop-in replacement for ``condition_sampler_v2.sample_condition``.

    Returns ``(mask, edit_mode)`` with the same semantics as v2:
      - ``mask`` is float32 ``(T, 198)``, ``1 = generate, 0 = known``.
      - ``edit_mode`` is True iff the caller should apply the corruptor
        pipeline; the returned mask then serves as the over-mask
        (always ⊇ corrupted region).

    The ``edit_mode`` is sampled **independently** of the Rank-K mask
    (see design doc §7).
    """
    mask = sample_mask_rank_k(
        T, rng,
        k_weights=k_weights,
        temporal_weights=temporal_weights,
        kind_weights=kind_weights,
    )
    edit_mode = bool(rng.random() < editing_prob)
    return mask, edit_mode
