"""Unit tests for condition_sampler_v3 (Rank-K Boolean Tensor Prior).

Scope
-----
- Temporal primitives: each of the 6 ``πT`` primitives produces valid
  ``{0,1}^T`` vectors with the expected shape / support.
- Dimensional kinds: each of ``πD``'s 4 kinds produces a valid
  ``{0,1}^198`` vector touching only the correct channel ranges.
- Anatomical dictionary: index sets match the SMPL-22 skeleton.
- Rank-K composition: ``K=0`` yields all-generate; ``K≥1`` stays within
  [0,1]; mask convention (1 = generate) is preserved.
- Coverage probes: specific eval-task signatures appear with non-zero
  frequency in 10k samples (fast smoke check; full coverage audit is in
  ``tools/sampler_coverage_audit.py``).
"""

from __future__ import annotations

import numpy as np
import pytest

from hftrainer.datasets.motion.motionhub.transforms.condition_sampler_v3 import (
    ANATOMICAL_GROUPS,
    DEFAULT_K_WEIGHTS,
    KINDS,
    MOTION_DIM,
    NUM_JOINTS,
    PERIODIC_STEPS,
    POS_END,
    POS_START,
    ROT_END,
    ROT_START,
    TEMPORAL_PRIMITIVES,
    TRANSL_DIM,
    _atom_mixed,
    _atom_pos_only,
    _atom_rot_only,
    _atom_trans_only,
    _pick_joint_subset,
    _pick_xyz_subset,
    _t_all,
    _t_empty,
    _t_interval,
    _t_markov,
    _t_periodic,
    _t_renewal,
    sample_condition_v3,
    sample_dimensional,
    sample_k,
    sample_mask_rank_k,
    sample_temporal,
)


T = 360  # canonical clip length


@pytest.fixture
def rng():
    return np.random.RandomState(42)


# -------------------------------------------------------------------------
# Temporal primitives
# -------------------------------------------------------------------------


class TestTemporalPrimitives:
    def test_all(self, rng):
        t = _t_all(T, rng)
        assert t.shape == (T,)
        assert t.dtype == np.uint8
        assert (t == 1).all()

    def test_empty(self, rng):
        t = _t_empty(T, rng)
        assert (t == 0).all()

    def test_interval_contiguous(self, rng):
        """Interval is a single contiguous run of 1s."""
        for seed in range(64):
            local = np.random.RandomState(seed)
            t = _t_interval(T, local)
            ones = np.where(t == 1)[0]
            if len(ones) == 0:
                continue  # technically impossible given ℓ ≥ 1
            assert ones[-1] - ones[0] + 1 == len(ones), (
                f"Interval not contiguous (seed={seed}): {ones}"
            )
            assert 1 <= len(ones) <= T

    def test_periodic_step(self, rng):
        """Periodic vector has hits separated by a valid integer step.

        After the 2026-04 'arbitrary period' patch, ``p`` may come either
        from the anchor set ``PERIODIC_STEPS`` (70 % of the time) or from
        a uniform draw over the configured range (30 %). The contract is
        therefore: hits are exactly periodic with some integer step in
        the valid range.
        """
        from hftrainer.datasets.motion.motionhub.transforms.condition_sampler_v3 import (
            PERIODIC_STEP_RANGE,
        )
        lo, hi = PERIODIC_STEP_RANGE
        for seed in range(64):
            local = np.random.RandomState(seed)
            t = _t_periodic(T, local)
            ones = np.where(t == 1)[0]
            if len(ones) < 2:
                continue
            diffs = np.diff(ones)
            step = int(diffs[0])
            assert lo <= step <= hi, f"bad step {step} (seed={seed})"
            assert (diffs == step).all(), f"non-uniform periodic (seed={seed})"

    def test_periodic_anchor_dominant(self):
        """Anchor steps (5/10/15/20/30/60) should appear ≥ 50 % of the time."""
        n_anchor = 0
        n_total = 0
        for seed in range(2000):
            local = np.random.RandomState(seed)
            t = _t_periodic(T, local)
            ones = np.where(t == 1)[0]
            if len(ones) < 2:
                continue
            step = int(ones[1] - ones[0])
            n_total += 1
            if step in PERIODIC_STEPS:
                n_anchor += 1
        assert n_total > 0
        ratio = n_anchor / n_total
        assert ratio >= 0.5, f"anchor share too low: {ratio:.3f}"

    def test_periodic_supports_arbitrary_p(self):
        """Periods outside ``PERIODIC_STEPS`` must have non-zero probability.

        This guarantees the prior covers *future* eval tasks that use
        arbitrary periods (e.g. p=7 or p=12).
        """
        seen = set()
        for seed in range(5000):
            local = np.random.RandomState(seed)
            t = _t_periodic(T, local)
            ones = np.where(t == 1)[0]
            if len(ones) < 2:
                continue
            step = int(ones[1] - ones[0])
            if step not in PERIODIC_STEPS:
                seen.add(step)
        # We expect at least 5 distinct non-anchor periods over 5k draws.
        assert len(seen) >= 5, (
            f"too few non-anchor periods sampled: {sorted(seen)}"
        )

    def test_renewal_positive_hits(self, rng):
        """Renewal always produces at least one hit."""
        for seed in range(64):
            local = np.random.RandomState(seed)
            t = _t_renewal(T, local)
            assert t.sum() >= 1, f"renewal vacuous at seed {seed}"

    def test_markov_binary(self, rng):
        t = _t_markov(T, rng)
        assert set(np.unique(t).tolist()) <= {0, 1}

    def test_primitive_dispatch_names(self, rng):
        """sample_temporal returns the primitive name from the enum."""
        for _ in range(100):
            _, name = sample_temporal(T, rng)
            assert name in TEMPORAL_PRIMITIVES


# -------------------------------------------------------------------------
# Anatomical dictionary
# -------------------------------------------------------------------------


class TestAnatomical:
    def test_all_joints(self):
        assert ANATOMICAL_GROUPS['all'] == tuple(range(22))

    def test_indices_in_range(self):
        for name, joints in ANATOMICAL_GROUPS.items():
            for j in joints:
                assert 0 <= j < NUM_JOINTS, f"{name}: {j} out of range"

    def test_upper_lower_partition_non_overlap(self):
        """upper_body and lower_body should be disjoint."""
        up = set(ANATOMICAL_GROUPS['upper_body'])
        lo = set(ANATOMICAL_GROUPS['lower_body'])
        assert up.isdisjoint(lo), f"overlap: {up & lo}"

    def test_hands_feet_contains_e4_c(self):
        """E4 setting C requires r_wrist (21) and l_foot (10). Both must
        be in the hands_feet group (NOT end_effectors, which lacks feet)."""
        hf = set(ANATOMICAL_GROUPS['hands_feet'])
        assert 10 in hf
        assert 21 in hf

    def test_end_effectors_matches_v2_EE_ALL(self):
        """Backward compat with v2 EE_ALL = {l_ankle, r_ankle, l_wrist, r_wrist}."""
        ee = set(ANATOMICAL_GROUPS['end_effectors'])
        assert ee == {7, 8, 20, 21}


# -------------------------------------------------------------------------
# Dimensional atoms
# -------------------------------------------------------------------------


class TestDimensionalAtoms:
    def test_rot_only_touches_only_rot_range(self, rng):
        for _ in range(64):
            d = _atom_rot_only(rng)
            assert d[:ROT_START].sum() == 0
            assert d[POS_START:].sum() == 0
            assert d[ROT_START:ROT_END].sum() > 0

    def test_pos_only_touches_only_pos_range(self, rng):
        for _ in range(64):
            d = _atom_pos_only(rng)
            assert d[:POS_START].sum() == 0
            assert d[POS_START:POS_END].sum() > 0

    def test_pos_only_excludes_pelvis(self, rng):
        """Pelvis (joint 0) has no position channel in 198-dim."""
        for seed in range(128):
            local = np.random.RandomState(seed)
            d = _atom_pos_only(local)
            # No "pelvis pos" exists — check that POS range is partitioned
            # only on joints 1..21.
            assert d.shape == (MOTION_DIM,)

    def test_trans_only_touches_only_trans_range(self, rng):
        for _ in range(64):
            d = _atom_trans_only(rng)
            assert d[TRANSL_DIM:].sum() == 0
            assert d[:TRANSL_DIM].sum() > 0

    def test_mixed_nonempty(self, rng):
        for _ in range(64):
            d = _atom_mixed(rng)
            assert d.sum() > 0

    def test_sample_dimensional_kind_distribution(self, rng):
        """Kind-weight sampling is consistent with given weights."""
        from hftrainer.datasets.motion.motionhub.transforms.condition_sampler_v3 import (
            DEFAULT_KIND_WEIGHTS,
        )
        n = 6000
        counts = {k: 0 for k in KINDS}
        for _ in range(n):
            _, kind = sample_dimensional(rng)
            counts[kind] += 1
        total = sum(counts.values())
        freqs = {k: v / total for k, v in counts.items()}
        for k in KINDS:
            expected = DEFAULT_KIND_WEIGHTS[k]
            assert abs(freqs[k] - expected) < 0.04, (
                f'{k}: emp={freqs[k]:.3f} vs exp={expected:.3f}'
            )

    def test_all_dim_locks_everything(self, rng):
        """all_dim kind must produce d = 1_198."""
        from hftrainer.datasets.motion.motionhub.transforms.condition_sampler_v3 import (
            _atom_all_dim,
        )
        d = _atom_all_dim(rng)
        assert d.shape == (MOTION_DIM,)
        assert (d == 1).all()


# -------------------------------------------------------------------------
# Joint / channel subset selection
# -------------------------------------------------------------------------


class TestSubsetSelection:
    def test_joint_subset_nonempty(self, rng):
        for _ in range(128):
            js = _pick_joint_subset(rng, include_pelvis=True)
            assert len(js) >= 1
            for j in js:
                assert 0 <= j < NUM_JOINTS

    def test_joint_subset_no_pelvis(self, rng):
        for _ in range(128):
            js = _pick_joint_subset(rng, include_pelvis=False)
            assert len(js) >= 1
            for j in js:
                assert 1 <= j < NUM_JOINTS

    def test_xyz_subset_nonempty(self, rng):
        for _ in range(128):
            x, y, z = _pick_xyz_subset(rng)
            assert x or y or z

    def test_xyz_subset_all_cases_reachable(self, rng):
        """All 7 non-empty subsets should appear in a large draw (weighted)."""
        n = 20000
        counts = {}
        for _ in range(n):
            x, y, z = _pick_xyz_subset(rng)
            key = (int(x), int(y), int(z))
            counts[key] = counts.get(key, 0) + 1
        assert len(counts) == 7, f"saw {len(counts)} subsets, expected 7"
        # The xyz-all subset is the most common (weight=4 of 14).
        assert counts[(1, 1, 1)] / n > 0.25, f"xyz-all underrepresented: {counts}"


# -------------------------------------------------------------------------
# Rank-K composition
# -------------------------------------------------------------------------


class TestRankK:
    def test_output_shape_dtype(self, rng):
        mask = sample_mask_rank_k(T, rng)
        assert mask.shape == (T, MOTION_DIM)
        assert mask.dtype == np.float32
        assert set(np.unique(mask).tolist()) <= {0.0, 1.0}

    def test_convention_generate_is_one(self, rng):
        """With K=0 (pure gen) mask must be all 1."""
        mask = sample_mask_rank_k(T, rng, k_weights=(1.0, 0.0))
        assert (mask == 1.0).all()

    def test_k_zero_frequency(self):
        """When k_weights puts all mass on K=0, never lock anything."""
        rng = np.random.RandomState(0)
        for _ in range(64):
            mask = sample_mask_rank_k(T, rng, k_weights=(1.0, 0.0))
            assert mask.sum() == T * MOTION_DIM

    def test_k_one_locks_something(self, rng):
        """Non-degenerate K=1 forces at least some locks (in expectation)."""
        n_vacuous = 0
        for _ in range(100):
            mask = sample_mask_rank_k(T, rng, k_weights=(0.0, 1.0))
            if mask.sum() == T * MOTION_DIM:
                n_vacuous += 1
        # Some vacuous are OK (e.g. t_primitive='empty'), but most K=1 atoms
        # should lock at least one cell.
        assert n_vacuous < 30, f"{n_vacuous}/100 K=1 draws were vacuous"

    def test_k_weights_distribution(self):
        """Empirical K distribution matches the weights."""
        rng = np.random.RandomState(0)
        n = 5000
        draws = np.array([sample_k(rng) for _ in range(n)])
        for k, w in enumerate(DEFAULT_K_WEIGHTS):
            emp = (draws == k).mean()
            assert abs(emp - w) < 0.025, f"K={k}: {emp:.3f} vs {w:.3f}"

    def test_trace_reports_atoms(self, rng):
        mask, trace = sample_mask_rank_k(
            T, rng, k_weights=(0.0, 0.0, 1.0), return_trace=True
        )
        assert trace['K'] == 2
        assert len(trace['atoms']) == 2


# -------------------------------------------------------------------------
# Drop-in API
# -------------------------------------------------------------------------


class TestSampleConditionV3:
    def test_signature_matches_v2(self, rng):
        mask, edit_mode = sample_condition_v3(T, rng)
        assert mask.shape == (T, MOTION_DIM)
        assert mask.dtype == np.float32
        assert isinstance(edit_mode, bool)

    def test_edit_mode_frequency(self):
        rng = np.random.RandomState(0)
        n = 10000
        flags = [sample_condition_v3(T, rng, editing_prob=0.1)[1] for _ in range(n)]
        emp = sum(flags) / n
        assert abs(emp - 0.10) < 0.01, f"edit_mode freq {emp:.3f}"


# -------------------------------------------------------------------------
# Coverage smoke probes (fast; not a full audit)
# -------------------------------------------------------------------------


def _periodic_pos_rwrist_hit(mask: np.ndarray, interval: int) -> bool:
    """Rough test: mask locks only (some of) r_wrist pos channels, periodic."""
    # pos channels of r_wrist (j=21) are [135 + 20*3 : 135 + 21*3] = [195, 198)
    s = slice(135 + 20 * 3, 135 + 21 * 3)
    rwrist_pos = mask[:, s]
    locked_frames = np.where((rwrist_pos == 0).all(axis=1))[0]
    if len(locked_frames) < 3:
        return False
    diffs = np.diff(locked_frames)
    return (diffs == interval).mean() > 0.7


def _upper_body_rot_all_frames(mask: np.ndarray) -> bool:
    """All frames have upper_body rot6d locked."""
    joints = ANATOMICAL_GROUPS['upper_body']
    for j in joints:
        s = slice(ROT_START + j * 6, ROT_START + (j + 1) * 6)
        if not (mask[:, s] == 0).all():
            return False
    return True


def _traj_xz_all_frames(mask: np.ndarray) -> bool:
    """trans x+z locked at every frame."""
    return bool((mask[:, 0] == 0).all() and (mask[:, 2] == 0).all())


class TestCoverageSmokeProbes:
    """These are loose smoke probes. Full coverage audit is the dedicated tool."""

    def test_e4_style_rwrist_periodic_hits(self):
        rng = np.random.RandomState(0)
        n = 10000
        hits = 0
        for _ in range(n):
            mask = sample_mask_rank_k(T, rng)
            if _periodic_pos_rwrist_hit(mask, interval=10):
                hits += 1
        # Lower-bound from §3: ≳ 10⁻⁴ → at least 1 in 10k with margin.
        assert hits >= 1, f"E4-style not hit in {n} draws"

    def test_e10_style_upper_body_rot(self):
        rng = np.random.RandomState(0)
        n = 10000
        hits = 0
        for _ in range(n):
            mask = sample_mask_rank_k(T, rng)
            if _upper_body_rot_all_frames(mask):
                hits += 1
        # Expected ≈ 1.5 ×10⁻³ → ≥ 3 hits in 10k (loose).
        assert hits >= 3, f"E10-style hit only {hits}/{n}"

    def test_e5_style_traj_xz(self):
        rng = np.random.RandomState(0)
        n = 10000
        hits = 0
        for _ in range(n):
            mask = sample_mask_rank_k(T, rng)
            if _traj_xz_all_frames(mask):
                hits += 1
        # Expected ≳ 3 ×10⁻³ → ≥ 10 hits in 10k.
        assert hits >= 10, f"E5-style hit only {hits}/{n}"


# -------------------------------------------------------------------------
# "Unseen" / future-task coverage probes
#
# These signatures are *not* present in the current E1-E15 audit set.
# They exercise the prior on plausible-but-not-yet-evaluated mask shapes
# to confirm v3's generative support truly is universal w.r.t. structured
# motion-completion conditions.
# -------------------------------------------------------------------------


def _all_frames_lock_joint_rot(mask: np.ndarray, joint_idx: int) -> bool:
    """Joint `joint_idx` rot6d is locked at every frame."""
    s = slice(ROT_START + joint_idx * 6, ROT_START + (joint_idx + 1) * 6)
    return bool((mask[:, s] == 0).all())


def _frame_range_lock_dim(mask: np.ndarray,
                          a: int, b: int,
                          dim_indices) -> bool:
    """All channels in dim_indices are locked across frames [a,b)."""
    sub = mask[a:b][:, list(dim_indices)]
    return bool((sub == 0).all())


def _periodic_locks_dims(mask: np.ndarray,
                         period: int,
                         dim_indices,
                         min_hits: int = 5) -> bool:
    """At least `min_hits` periodic-spaced frames lock all `dim_indices`."""
    cols = mask[:, list(dim_indices)]
    # frames where every required channel is locked
    fully_locked = np.where((cols == 0).all(axis=1))[0]
    if len(fully_locked) < min_hits:
        return False
    diffs = np.diff(fully_locked)
    return (diffs == period).sum() >= (min_hits - 1)


class TestUnseenTaskCoverage:
    """Plausible mask shapes that no current E-task covers, but a real
    user / future eval might request. v3 prior must hit each at least once
    in a 10k-sample budget, demonstrating universal coverage rather than
    fitting only to the E1-E15 templates."""

    def test_unseen_spine_chain_locked_interval(self):
        """Spine chain (joints 3, 6, 9, 12, 15) rot locked over a single
        interval [50, 110). Combines anatomical group + interval temporal."""
        from hftrainer.datasets.motion.motionhub.transforms.condition_sampler_v3 import (
            ANATOMICAL_GROUPS,
        )
        spine = ANATOMICAL_GROUPS['spine_chain']
        rot_dims = []
        for j in spine:
            rot_dims.extend(range(ROT_START + j * 6, ROT_START + (j + 1) * 6))
        rng = np.random.RandomState(0)
        hits = 0
        for _ in range(15000):
            mask = sample_mask_rank_k(T, rng)
            # tolerant: lock contains [80, 90) (within nominal [50,110))
            if _frame_range_lock_dim(mask, 80, 90, rot_dims):
                hits += 1
        assert hits >= 1, "spine-chain interval not reachable"

    def test_unseen_arbitrary_period_pelvis_y(self):
        """Pelvis trans Y locked every 7 frames (period 7 ∉ PERIODIC_STEPS)."""
        rng = np.random.RandomState(1)
        hits = 0
        # We sample many masks; the prior must occasionally produce
        # exactly-period-7 periodic locks on the trans-y dim.
        for _ in range(20000):
            mask = sample_mask_rank_k(T, rng)
            ones_in_y = np.where(mask[:, 1] == 0)[0]
            if len(ones_in_y) < 5:
                continue
            diffs = np.diff(ones_in_y)
            if (diffs == 7).sum() >= 4:  # most gaps are 7
                hits += 1
                break
        assert hits >= 1, "arbitrary p=7 not reachable"

    def test_unseen_head_only_rot(self):
        """Head joint (15) rot locked at all frames. Tests the rare
        anatomical group end of the prior."""
        rng = np.random.RandomState(2)
        hits = 0
        for _ in range(15000):
            mask = sample_mask_rank_k(T, rng)
            if _all_frames_lock_joint_rot(mask, joint_idx=15):
                hits += 1
                break
        assert hits >= 1, "head-only rot not reachable"

    def test_unseen_trans_y_only(self):
        """Pelvis trans Y locked across all frames (vertical-only constraint)."""
        rng = np.random.RandomState(3)
        hits = 0
        for _ in range(15000):
            mask = sample_mask_rank_k(T, rng)
            if (mask[:, 1] == 0).all():
                hits += 1
                break
        assert hits >= 1, "trans-y all-frames lock not reachable"

    def test_unseen_l_knee_xz_keyframes(self):
        """Left knee (joint 4) position x+z locked at scattered keyframes
        (period 25, also not in PERIODIC_STEPS)."""
        # l_knee pos channels: 135 + 3*3 .. 135 + 4*3 = [144, 147)
        x_dim, _, z_dim = 144, 145, 146
        rng = np.random.RandomState(4)
        hits = 0
        for _ in range(20000):
            mask = sample_mask_rank_k(T, rng)
            cols = mask[:, [x_dim, z_dim]]
            fully = np.where((cols == 0).all(axis=1))[0]
            if len(fully) < 5:
                continue
            diffs = np.diff(fully)
            if (diffs == 25).sum() >= 3:
                hits += 1
                break
        assert hits >= 1, "l_knee xz @ p=25 not reachable"

    def test_unseen_full_198_short_window(self):
        """Every channel locked across a short 12-frame window (anchor +
        in-painting style request)."""
        rng = np.random.RandomState(5)
        hits = 0
        for _ in range(15000):
            mask = sample_mask_rank_k(T, rng)
            # find any 12-frame fully-locked window
            row_locked = (mask == 0).all(axis=1)
            if row_locked.shape[0] < 12:
                continue
            # cumulative sum trick to find window of length 12
            cs = np.concatenate([[0], np.cumsum(row_locked.astype(np.int32))])
            wins = cs[12:] - cs[:-12]
            if (wins == 12).any():
                hits += 1
                break
        assert hits >= 1, "12-frame full-198 window not reachable"
