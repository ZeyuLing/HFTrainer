"""Unit tests for condition_sampler_v2.

Tests cover:
- Temporal Markov chain (various parameter regimes)
- Spatial Bernoulli (joint count distribution)
- Channel decisions (rot/pos combinations)
- Translation independence
- Tier 1 coverage (100K samples, no dead zones)
- Tier 2 patterns (each pattern individually)
- OOD check (typical inference patterns covered)
"""

import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.RandomState(42)


class TestTemporalMarkov:
    def test_basic_shape(self, rng):
        from hftrainer.datasets.motion.motionhub.transforms.condition_sampler_v2 import (
            sample_temporal_markov,
        )
        T = 100
        seq = sample_temporal_markov(T, rng)
        assert seq.shape == (T,)
        assert set(np.unique(seq)) <= {0, 1}

    def test_all_generate_possible(self):
        """p_start_known=0, p_stay_gen=1 → all generate."""
        from hftrainer.datasets.motion.motionhub.transforms.condition_sampler_v2 import (
            sample_temporal_markov,
        )
        # With enough samples, should see all-generate at least once
        T = 50
        found_all_gen = False
        for seed in range(1000):
            rng = np.random.RandomState(seed)
            seq = sample_temporal_markov(T, rng)
            if seq.sum() == T:
                found_all_gen = True
                break
        assert found_all_gen, "All-generate pattern never appeared in 1000 trials"

    def test_various_patterns(self):
        """Various patterns should emerge: mostly-early known, mostly-late known, sparse."""
        from hftrainer.datasets.motion.motionhub.transforms.condition_sampler_v2 import (
            sample_temporal_markov,
        )
        T = 100
        has_early_heavy = False   # Most known frames in first half
        has_late_heavy = False    # Most known frames in second half
        has_sparse = False

        for seed in range(5000):
            rng = np.random.RandomState(seed)
            seq = sample_temporal_markov(T, rng)
            known_frames = np.where(seq == 0)[0]
            if len(known_frames) == 0:
                continue

            n_known = len(known_frames)
            n_first_half = (known_frames < T // 2).sum()
            n_second_half = n_known - n_first_half

            # Early-heavy: >80% of known frames in first half
            if n_known > 5 and n_first_half / n_known > 0.8:
                has_early_heavy = True
            # Late-heavy: >80% of known frames in second half
            if n_known > 5 and n_second_half / n_known > 0.8:
                has_late_heavy = True
            # Sparse: <20% known, scattered
            if n_known < T * 0.2 and n_known > 3:
                has_sparse = True

        assert has_early_heavy, "Never saw early-heavy known pattern"
        assert has_late_heavy, "Never saw late-heavy known pattern"
        assert has_sparse, "Never saw sparse pattern"


class TestSpatialBernoulli:
    def test_at_least_one_joint(self, rng):
        from hftrainer.datasets.motion.motionhub.transforms.condition_sampler_v2 import (
            sample_spatial_bernoulli,
        )
        for _ in range(100):
            joints = sample_spatial_bernoulli(rng)
            assert len(joints) >= 1

    def test_all_joints_valid(self, rng):
        from hftrainer.datasets.motion.motionhub.transforms.condition_sampler_v2 import (
            sample_spatial_bernoulli,
        )
        for _ in range(100):
            joints = sample_spatial_bernoulli(rng)
            assert all(0 <= j < 22 for j in joints)

    def test_distribution_biased_sparse(self):
        """Beta(1,6) should produce mostly 1-5 joints."""
        from hftrainer.datasets.motion.motionhub.transforms.condition_sampler_v2 import (
            sample_spatial_bernoulli,
        )
        counts = []
        rng = np.random.RandomState(42)
        for _ in range(10000):
            joints = sample_spatial_bernoulli(rng)
            counts.append(len(joints))
        median = np.median(counts)
        assert 1 <= median <= 5, f"Median joint count {median} not in expected range [1,5]"


class TestChannel:
    def test_at_least_one_channel(self, rng):
        from hftrainer.datasets.motion.motionhub.transforms.condition_sampler_v2 import (
            sample_channel,
        )
        for _ in range(1000):
            rot_keep, (px, py, pz) = sample_channel(rng)
            assert rot_keep or px or py or pz, "At least one channel must be kept"

    def test_rot_and_pos_combinations(self):
        """All rot/pos combinations should appear."""
        from hftrainer.datasets.motion.motionhub.transforms.condition_sampler_v2 import (
            sample_channel,
        )
        rng = np.random.RandomState(42)
        has_rot_only = False
        has_pos_only = False
        has_both = False

        for _ in range(10000):
            rot_keep, (px, py, pz) = sample_channel(rng)
            has_pos = px or py or pz
            if rot_keep and not has_pos:
                has_rot_only = True
            if not rot_keep and has_pos:
                has_pos_only = True
            if rot_keep and has_pos:
                has_both = True

        assert has_rot_only, "Never saw rotation-only"
        assert has_pos_only, "Never saw position-only"
        assert has_both, "Never saw rotation+position"


class TestTranslation:
    def test_translation_independent(self):
        """Translation constraint should appear independently of joint selection."""
        from hftrainer.datasets.motion.motionhub.transforms.condition_sampler_v2 import (
            sample_tier1, MOTION_DIM,
        )
        rng = np.random.RandomState(42)
        T = 100
        has_trans_no_pelvis = False

        for _ in range(5000):
            mask = sample_tier1(T, rng)
            # Check if translation known but pelvis rotation not known at some frame
            for f in range(T):
                trans_known = mask[f, 0:3].min() == 0
                pelvis_rot_known = mask[f, 3:9].min() == 0
                if trans_known and not pelvis_rot_known:
                    has_trans_no_pelvis = True
                    break
            if has_trans_no_pelvis:
                break

        assert has_trans_no_pelvis, "Translation never appeared independent of pelvis rotation"


class TestTier1Coverage:
    def test_no_dead_zones(self):
        """100K samples should show non-zero probability for diverse patterns."""
        from hftrainer.datasets.motion.motionhub.transforms.condition_sampler_v2 import (
            sample_condition, MOTION_DIM,
        )
        rng = np.random.RandomState(42)
        T = 60
        N = 100000

        pure_gen = 0
        has_position_only = 0
        has_rotation_only = 0
        has_translation = 0
        has_single_joint = 0
        has_editing = 0

        for _ in range(N):
            mask, edit_mode = sample_condition(T, rng)

            if mask.sum() == mask.size:
                pure_gen += 1
            if edit_mode:
                has_editing += 1

            # Check single-joint constraint at any frame
            for f in range(0, T, 10):  # sample every 10th frame for speed
                known_joints = []
                for j in range(22):
                    if mask[f, 3 + j * 6: 3 + (j + 1) * 6].min() == 0:
                        known_joints.append(j)
                if len(known_joints) == 1:
                    has_single_joint += 1
                    break

            if mask[:, 0:3].min(axis=-1).min() == 0:
                has_translation += 1

        # Validate coverage (all should be > 0.1%)
        threshold = N * 0.001
        assert pure_gen > threshold, f"Pure generation too rare: {pure_gen}/{N}"
        assert has_translation > threshold, f"Translation constraint too rare: {has_translation}/{N}"
        assert has_single_joint > threshold, f"Single-joint constraint too rare: {has_single_joint}/{N}"
        assert has_editing > threshold, f"Editing mode too rare: {has_editing}/{N}"


class TestTier2Patterns:
    def test_pure_gen(self):
        from hftrainer.datasets.motion.motionhub.transforms.condition_sampler_v2 import (
            sample_tier2_pure_gen, MOTION_DIM,
        )
        rng = np.random.RandomState(42)
        T = 100
        mask = np.ones((T, MOTION_DIM), dtype=np.float32)
        edit = sample_tier2_pure_gen(T, mask, rng)
        assert mask.sum() == mask.size, "Pure gen should be all 1"
        assert not edit

    def test_inbetween(self):
        from hftrainer.datasets.motion.motionhub.transforms.condition_sampler_v2 import (
            sample_tier2_inbetween, MOTION_DIM,
        )
        rng = np.random.RandomState(42)
        T = 100
        mask = np.ones((T, MOTION_DIM), dtype=np.float32)
        edit = sample_tier2_inbetween(T, mask, rng)
        assert not edit
        # First and last frames should be known (0)
        assert mask[0, :].sum() == 0, "First frame should be all known"
        assert mask[-1, :].sum() == 0, "Last frame should be all known"
        # Some middle frames should be generate (1)
        assert mask[T // 2, :].sum() > 0, "Middle frames should have generate regions"

    def test_prefix(self):
        from hftrainer.datasets.motion.motionhub.transforms.condition_sampler_v2 import (
            sample_tier2_prefix, MOTION_DIM,
        )
        rng = np.random.RandomState(42)
        T = 100
        mask = np.ones((T, MOTION_DIM), dtype=np.float32)
        edit = sample_tier2_prefix(T, mask, rng)
        assert not edit
        assert mask[0, :].sum() == 0, "First frame should be known"

    def test_end_effector(self):
        from hftrainer.datasets.motion.motionhub.transforms.condition_sampler_v2 import (
            sample_tier2_end_effector, MOTION_DIM,
        )
        rng = np.random.RandomState(42)
        T = 100
        mask = np.ones((T, MOTION_DIM), dtype=np.float32)
        edit = sample_tier2_end_effector(T, mask, rng)
        assert not edit
        # Some position dims should be known
        assert mask[:, 135:].min() == 0, "End-effector should set some position dims to 0"

    def test_trajectory(self):
        from hftrainer.datasets.motion.motionhub.transforms.condition_sampler_v2 import (
            sample_tier2_trajectory, MOTION_DIM,
        )
        rng = np.random.RandomState(42)
        T = 100
        mask = np.ones((T, MOTION_DIM), dtype=np.float32)
        edit = sample_tier2_trajectory(T, mask, rng)
        assert not edit
        # Translation X and Z should be known at some frames
        assert mask[:, 0].min() == 0, "Trans X should be known at some frames"
        assert mask[:, 2].min() == 0, "Trans Z should be known at some frames"

    def test_foot_ground(self):
        from hftrainer.datasets.motion.motionhub.transforms.condition_sampler_v2 import (
            sample_tier2_foot_ground, MOTION_DIM,
        )
        rng = np.random.RandomState(42)
        T = 100
        mask = np.ones((T, MOTION_DIM), dtype=np.float32)
        edit = sample_tier2_foot_ground(T, mask, rng)
        assert not edit
        # Ankle position Y should be known at some frames
        # L_Ankle is joint 7, position starts at 135 + (7-1)*3 = 153, Y = 154
        assert mask[:, 154].min() == 0, "L_Ankle Y should be known"

    def test_edit_repair_returns_edit_mode(self):
        from hftrainer.datasets.motion.motionhub.transforms.condition_sampler_v2 import (
            sample_tier2_edit_repair, MOTION_DIM,
        )
        rng = np.random.RandomState(42)
        T = 100
        mask = np.ones((T, MOTION_DIM), dtype=np.float32)
        edit = sample_tier2_edit_repair(T, mask, rng)
        assert edit, "Edit repair should return edit_mode=True"


class TestOODCheck:
    """Verify typical inference condition patterns are covered by training."""

    def test_inference_patterns_covered(self):
        """Check that each typical inference pattern appears with non-zero probability."""
        from hftrainer.datasets.motion.motionhub.transforms.condition_sampler_v2 import (
            sample_condition, MOTION_DIM,
        )
        rng = np.random.RandomState(42)
        T = 60
        N = 50000

        # Track pattern coverage
        covered = {
            'pure_gen': False,          # All mask=1
            'inbetween': False,         # First+last known, middle generate
            'prefix': False,            # First N known
            'single_joint_pos': False,  # Single joint position only
            'trajectory_xz': False,     # Translation XZ known
            'foot_y': False,            # Ankle Y known
            'rotation_only': False,     # Rotation known, position generate
        }

        for _ in range(N):
            mask, _ = sample_condition(T, rng)

            # Pure generation
            if mask.sum() == mask.size:
                covered['pure_gen'] = True

            # In-between
            if mask[0, :].sum() == 0 and mask[-1, :].sum() == 0 and mask[T // 2, :].max() == 1:
                covered['inbetween'] = True

            # Prefix
            if mask[0, :].sum() == 0 and mask[-1, :].max() == 1:
                covered['prefix'] = True

            # Single joint position only (any frame)
            for f in range(0, T, 20):
                for j in range(1, 22):
                    pos_base = 135 + (j - 1) * 3
                    pos_known = mask[f, pos_base:pos_base + 3].min() == 0
                    rot_gen = mask[f, 3 + j * 6: 3 + (j + 1) * 6].max() == 1
                    if pos_known and rot_gen:
                        covered['single_joint_pos'] = True
                        break

            # Trajectory XZ
            trans_x_known = mask[:, 0].min() == 0
            trans_z_known = mask[:, 2].min() == 0
            if trans_x_known and trans_z_known:
                covered['trajectory_xz'] = True

            # Foot Y (ankle Y position only)
            ankle_y_idx = 135 + (7 - 1) * 3 + 1  # L_Ankle Y
            if mask[:, ankle_y_idx].min() == 0:
                covered['foot_y'] = True

            # Rotation only (some joint has rotation known but position all generate)
            for f in range(0, T, 20):
                for j in range(1, 22):
                    rot_known = mask[f, 3 + j * 6: 3 + (j + 1) * 6].max() == 0
                    pos_base = 135 + (j - 1) * 3
                    pos_gen = mask[f, pos_base:pos_base + 3].min() == 1
                    if rot_known and pos_gen:
                        covered['rotation_only'] = True
                        break

            if all(covered.values()):
                break

        for pattern, found in covered.items():
            assert found, f"Inference pattern '{pattern}' never covered in {N} samples"


class TestMaskPerturbation:
    def test_over_mask_only(self):
        """Mask perturbation should only make mask larger."""
        from hftrainer.datasets.motion.motionhub.transforms.condition_sampler_v2 import (
            apply_mask_perturbation, MOTION_DIM,
        )
        rng = np.random.RandomState(42)
        T = 50

        for _ in range(100):
            base_mask = (rng.rand(T, MOTION_DIM) > 0.7).astype(np.float32)
            perturbed = apply_mask_perturbation(base_mask, rng)
            assert (perturbed >= base_mask).all(), "Perturbation should never decrease mask"

    def test_all_modes_work(self):
        """Each perturbation mode should produce valid output."""
        from hftrainer.datasets.motion.motionhub.transforms.condition_sampler_v2 import (
            apply_mask_perturbation, MOTION_DIM,
        )
        T = 50
        base_mask = np.zeros((T, MOTION_DIM), dtype=np.float32)
        base_mask[10:20, 3:9] = 1.0  # Small corruption area

        for seed in range(100):
            rng = np.random.RandomState(seed)
            perturbed = apply_mask_perturbation(base_mask, rng)
            assert perturbed.shape == (T, MOTION_DIM)
            assert (perturbed >= base_mask).all()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
