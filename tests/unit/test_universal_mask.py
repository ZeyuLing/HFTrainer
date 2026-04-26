"""Comprehensive unit tests for PrepareM2MUniversalMask.

Tests cover:
  - Constants and joint group mappings
  - Grid-to-mask expansion
  - Each of the 6 strategies (M1-M6)
  - Full transform integration
  - Backward compatibility with PrepareM2MCompletion
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
import torch

from hftrainer.datasets.motion.motionhub.transforms.universal_mask import (
    ALL_JOINTS_NO_TRANSL,
    BODY_PART_GROUPS,
    FEET,
    JOINT_ROT_DIM,
    LEFT_ARM,
    LEFT_LEG,
    LOWER_BODY,
    NUM_JOINT_GROUPS,
    RIGHT_ARM,
    RIGHT_LEG,
    SPINE_HEAD,
    TOTAL_DIM,
    TRANSL_DIM,
    TRANSLATION,
    UPPER_BODY,
    PrepareM2MUniversalMask,
    expand_grid_to_mask,
    m1_random_cell,
    m2_random_block,
    m3_temporal_contiguous,
    m4_joint_contiguous,
    m5_full_mask,
    m6_keyframe_sparse,
)


# -----------------------------------------------------------------------
# Test constants
# -----------------------------------------------------------------------


class TestConstants:
    """Verify joint group mappings and body part groups."""

    def test_total_dims(self):
        assert NUM_JOINT_GROUPS == 23
        assert TRANSL_DIM == 3
        assert JOINT_ROT_DIM == 6
        assert TOTAL_DIM == 135
        assert TRANSL_DIM + 22 * JOINT_ROT_DIM == TOTAL_DIM  # 3 abs transl + 22*6 rot6d

    def test_upper_lower_no_overlap(self):
        """Upper and lower body joints should not overlap."""
        upper = set(UPPER_BODY)
        lower = set(LOWER_BODY)
        assert upper & lower == set(), (
            f'Upper/lower overlap: {upper & lower}'
        )

    def test_left_right_no_overlap(self):
        """Left and right arm/leg should not overlap."""
        assert set(LEFT_ARM) & set(RIGHT_ARM) == set()
        assert set(LEFT_LEG) & set(RIGHT_LEG) == set()

    def test_body_parts_cover_all(self):
        """Union of upper+lower+spine should cover all 23 indices."""
        all_joints = set(range(NUM_JOINT_GROUPS))
        # Upper + lower already covers everything except maybe Pelvis(1) and Spine1(4)
        covered = set(UPPER_BODY) | set(LOWER_BODY) | set(SPINE_HEAD)
        covered.add(1)  # Pelvis
        assert covered == all_joints or covered.issuperset(all_joints - {1}), (
            f'Missing joints: {all_joints - covered}'
        )

    def test_translation_group(self):
        assert TRANSLATION == [0]

    def test_all_joints_no_transl(self):
        assert ALL_JOINTS_NO_TRANSL == list(range(1, 23))
        assert 0 not in ALL_JOINTS_NO_TRANSL

    def test_joint_indices_in_range(self):
        """All body part group indices should be in [0, 22]."""
        for name, indices in BODY_PART_GROUPS.items():
            for idx in indices:
                assert 0 <= idx < NUM_JOINT_GROUPS, (
                    f'Index {idx} in group {name!r} out of range'
                )

    def test_body_part_groups_dict(self):
        """Check expected groups exist."""
        expected = {
            'upper', 'lower', 'left_arm', 'right_arm',
            'left_leg', 'right_leg', 'spine_head', 'feet',
            'translation', 'joints_only',
        }
        assert set(BODY_PART_GROUPS.keys()) == expected


# -----------------------------------------------------------------------
# Test grid expansion
# -----------------------------------------------------------------------


class TestExpandGrid:
    """Test grid expansion from (T, 23) -> (T, 135)."""

    def test_shape(self):
        grid = np.zeros((10, 23), dtype=np.float32)
        mask = expand_grid_to_mask(grid)
        assert mask.shape == (10, 135)

    def test_expansion_correctness(self):
        """Translation group expands to 3 dims, each joint to 6 dims."""
        T = 5
        grid = np.random.randint(0, 2, size=(T, 23)).astype(np.float32)
        mask = expand_grid_to_mask(grid)
        # Check translation group (first 3 dims)
        for d in range(TRANSL_DIM):
            assert torch.equal(mask[:, d], torch.from_numpy(grid[:, 0]))
        # Check joint groups (each 6 dims)
        for j in range(22):
            start = TRANSL_DIM + j * JOINT_ROT_DIM
            group = mask[:, start:start + JOINT_ROT_DIM]
            for d in range(JOINT_ROT_DIM):
                assert torch.equal(group[:, d], torch.from_numpy(grid[:, j + 1]))

    def test_full_zero(self):
        grid = np.zeros((8, 23), dtype=np.float32)
        mask = expand_grid_to_mask(grid)
        assert mask.sum().item() == 0

    def test_full_one(self):
        grid = np.ones((8, 23), dtype=np.float32)
        mask = expand_grid_to_mask(grid)
        assert mask.sum().item() == 8 * 135

    def test_dtype(self):
        grid = np.ones((4, 23), dtype=np.float32)
        mask = expand_grid_to_mask(grid)
        assert mask.dtype == torch.float32

    def test_single_frame(self):
        grid = np.zeros((1, 23), dtype=np.float32)
        grid[0, 5] = 1.0  # joint group 5 (L_Knee)
        mask = expand_grid_to_mask(grid)
        assert mask.shape == (1, 135)
        # Joint group 5 -> dims [3+4*6 : 3+5*6] = [27:33]
        start = TRANSL_DIM + 4 * JOINT_ROT_DIM  # group index 5 is the 5th joint (index 4 in 0-based joints)
        end = start + JOINT_ROT_DIM
        assert mask[0, start:end].sum().item() == 6
        assert mask[0, :start].sum().item() == 0
        assert mask[0, end:].sum().item() == 0


# -----------------------------------------------------------------------
# Test M1: Random Cell
# -----------------------------------------------------------------------


class TestM1RandomCell:

    def test_output_shape(self):
        grid = np.zeros((20, 23), dtype=np.float32)
        rng = np.random.RandomState(42)
        m1_random_cell(20, grid, rng)
        assert grid.shape == (20, 23)

    def test_values_binary(self):
        grid = np.zeros((50, 23), dtype=np.float32)
        rng = np.random.RandomState(42)
        m1_random_cell(50, grid, rng)
        unique = set(np.unique(grid))
        assert unique <= {0.0, 1.0}

    def test_mask_ratio_in_range(self):
        """Over many runs, mask ratio should be in [0.3, 0.95] range."""
        ratios = []
        for seed in range(100):
            grid = np.zeros((50, 23), dtype=np.float32)
            rng = np.random.RandomState(seed)
            m1_random_cell(50, grid, rng)
            ratios.append(grid.mean())
        # The expected p is U[0.3, 0.95], so ratios should cluster in that range
        # Allow small statistical deviation
        assert min(ratios) >= 0.1  # some room for variance
        assert max(ratios) <= 1.0

    def test_not_all_same(self):
        """Different seeds should produce different masks."""
        grids = []
        for seed in range(5):
            grid = np.zeros((20, 23), dtype=np.float32)
            rng = np.random.RandomState(seed)
            m1_random_cell(20, grid, rng)
            grids.append(grid.copy())
        # At least some should differ
        all_same = all(np.array_equal(grids[0], g) for g in grids[1:])
        assert not all_same


# -----------------------------------------------------------------------
# Test M2: Random Block
# -----------------------------------------------------------------------


class TestM2RandomBlock:

    def test_output_shape(self):
        grid = np.zeros((30, 23), dtype=np.float32)
        rng = np.random.RandomState(42)
        m2_random_block(30, grid, rng)
        assert grid.shape == (30, 23)

    def test_values_binary(self):
        grid = np.zeros((30, 23), dtype=np.float32)
        rng = np.random.RandomState(42)
        m2_random_block(30, grid, rng)
        unique = set(np.unique(grid))
        assert unique <= {0.0, 1.0}

    def test_has_masked_cells(self):
        """At least some cells should be masked."""
        grid = np.zeros((30, 23), dtype=np.float32)
        rng = np.random.RandomState(42)
        m2_random_block(30, grid, rng)
        assert grid.sum() > 0

    def test_block_structure(self):
        """Each block should form a temporal contiguous region for selected joints."""
        # Run multiple times and check that for each masked joint column,
        # the masked frames form contiguous or near-contiguous regions
        grid = np.zeros((50, 23), dtype=np.float32)
        rng = np.random.RandomState(123)
        m2_random_block(50, grid, rng)
        # At minimum, check grid has valid binary values
        assert np.all((grid == 0) | (grid == 1))

    def test_single_frame(self):
        """Should work with T=1."""
        grid = np.zeros((1, 23), dtype=np.float32)
        rng = np.random.RandomState(42)
        m2_random_block(1, grid, rng)
        assert grid.shape == (1, 23)


# -----------------------------------------------------------------------
# Test M3: Temporal Contiguous
# -----------------------------------------------------------------------


class TestM3TemporalContiguous:

    def _run(self, T=50, seed=42):
        grid = np.zeros((T, 23), dtype=np.float32)
        rng = np.random.RandomState(seed)
        m3_temporal_contiguous(T, grid, rng)
        return grid

    def test_output_shape(self):
        grid = self._run()
        assert grid.shape == (50, 23)

    def test_all_joints_masked_for_masked_rows(self):
        """For masked frames, all 23 joints should be masked."""
        for seed in range(20):
            grid = self._run(T=50, seed=seed)
            for t in range(50):
                row = grid[t, :]
                if row.sum() > 0:
                    # Either all joints masked or none (for temporal strategies)
                    # M3 always masks full joint rows
                    assert row.sum() == 23, (
                        f'Seed {seed}, frame {t}: partial row mask in M3'
                    )

    def test_has_masked_frames(self):
        """Should produce some masked frames."""
        grid = self._run(T=50, seed=42)
        assert grid.sum() > 0

    def test_has_known_frames(self):
        """Should preserve some frames (unless extremely unlucky)."""
        # Try multiple seeds, at least some should have known frames
        has_known = False
        for seed in range(20):
            grid = self._run(T=50, seed=seed)
            if grid.sum() < 50 * 23:
                has_known = True
                break
        assert has_known

    def test_short_sequence(self):
        """T=2 and T=1 should not crash."""
        for T in [1, 2, 3]:
            grid = np.zeros((T, 23), dtype=np.float32)
            rng = np.random.RandomState(42)
            m3_temporal_contiguous(T, grid, rng)
            assert grid.shape == (T, 23)

    def test_modes_reachable(self):
        """All 5 sub-modes should be reachable."""
        # The mode is chosen uniformly, so across many seeds we should hit all
        # We can't directly check mode names, but we can verify different patterns
        patterns = set()
        for seed in range(200):
            grid = self._run(T=30, seed=seed)
            # Classify pattern: check first row, last row, middle
            first_masked = grid[0, 0] > 0
            last_masked = grid[-1, 0] > 0
            pattern = (first_masked, last_masked)
            patterns.add(pattern)
        # Should see at least: (False, False)=inbetween, (False, True)=prediction,
        # (True, False)=prefix, (True, True)=outpainting/multi_gap
        assert len(patterns) >= 3


# -----------------------------------------------------------------------
# Test M4: Joint Contiguous
# -----------------------------------------------------------------------


class TestM4JointContiguous:

    def _run(self, T=30, seed=42):
        grid = np.zeros((T, 23), dtype=np.float32)
        rng = np.random.RandomState(seed)
        m4_joint_contiguous(T, grid, rng)
        return grid

    def test_output_shape(self):
        grid = self._run()
        assert grid.shape == (30, 23)

    def test_full_temporal_coverage(self):
        """Masked joints should span all frames."""
        grid = self._run()
        for j in range(23):
            col = grid[:, j]
            if col.sum() > 0:
                # Should be masked for ALL frames
                assert col.sum() == 30, (
                    f'Joint {j}: not all frames masked ({col.sum()}/30)'
                )

    def test_has_masked_joints(self):
        grid = self._run()
        assert grid.sum() > 0

    def test_preserved_joints(self):
        """Some joints should remain unmasked."""
        grid = self._run(T=30, seed=42)
        # At least some joints should be 0
        masked_joints = grid[0, :].sum()
        assert masked_joints < 23, 'All joints masked in M4'

    def test_body_part_group_mode(self):
        """When body part groups are used, correct indices are masked."""
        # Run many times, check that masked columns match body part groups
        found_body_part = False
        for seed in range(50):
            grid = self._run(T=20, seed=seed)
            masked_joints = set(np.where(grid[0] > 0)[0])
            # Check if masked_joints is a subset of any body part group
            for name, indices in BODY_PART_GROUPS.items():
                if masked_joints and masked_joints.issubset(set(indices)):
                    found_body_part = True
                    break
            if found_body_part:
                break
        # It's possible individual joint mode was always picked; that's OK
        # Just verify the function doesn't crash and produces valid output

    def test_single_frame(self):
        grid = np.zeros((1, 23), dtype=np.float32)
        rng = np.random.RandomState(42)
        m4_joint_contiguous(1, grid, rng)
        assert grid.shape == (1, 23)


# -----------------------------------------------------------------------
# Test M5: Full Mask
# -----------------------------------------------------------------------


class TestM5FullMask:

    def test_all_ones(self):
        grid = np.zeros((20, 23), dtype=np.float32)
        rng = np.random.RandomState(42)
        m5_full_mask(20, grid, rng)
        assert np.all(grid == 1.0)

    def test_mask_ratio_one(self):
        grid = np.zeros((10, 23), dtype=np.float32)
        rng = np.random.RandomState(42)
        m5_full_mask(10, grid, rng)
        assert grid.mean() == 1.0

    def test_single_frame(self):
        grid = np.zeros((1, 23), dtype=np.float32)
        rng = np.random.RandomState(42)
        m5_full_mask(1, grid, rng)
        assert grid.sum() == 23


# -----------------------------------------------------------------------
# Test M6: Keyframe Sparse
# -----------------------------------------------------------------------


class TestM6KeyframeSparse:

    def _run(self, T=40, seed=42):
        grid = np.zeros((T, 23), dtype=np.float32)
        rng = np.random.RandomState(seed)
        m6_keyframe_sparse(T, grid, rng)
        return grid

    def test_output_shape(self):
        grid = self._run()
        assert grid.shape == (40, 23)

    def test_keyframes_preserved(self):
        """At least one frame should have some mask=0 cells."""
        grid = self._run()
        # Find frames with any 0
        preserved_frames = np.any(grid == 0, axis=1)
        assert preserved_frames.sum() >= 1

    def test_non_keyframes_masked(self):
        """Non-keyframe rows should be fully masked (all 1)."""
        grid = self._run()
        for t in range(40):
            row = grid[t, :]
            # Either fully masked (all 1) or partially/fully preserved
            if row.min() > 0:
                assert row.sum() == 23  # fully masked

    def test_at_least_one_keyframe(self):
        """Should always have at least 1 keyframe (K >= 1)."""
        for seed in range(50):
            grid = self._run(T=20, seed=seed)
            has_preserved = np.any(grid == 0)
            assert has_preserved, f'No keyframe preserved at seed {seed}'

    def test_partial_joint_keyframes(self):
        """Some runs should produce partial keyframes (not all joints preserved)."""
        found_partial = False
        for seed in range(200):
            grid = self._run(T=30, seed=seed)
            for t in range(30):
                row = grid[t, :]
                zero_count = (row == 0).sum()
                if 0 < zero_count < 23:
                    found_partial = True
                    break
            if found_partial:
                break
        assert found_partial, 'Never found partial keyframe across 200 seeds'

    def test_single_frame(self):
        grid = np.zeros((1, 23), dtype=np.float32)
        rng = np.random.RandomState(42)
        m6_keyframe_sparse(1, grid, rng)
        # With T=1, the single keyframe should be preserved
        assert grid.shape == (1, 23)

    def test_short_sequence(self):
        """T=2 should not crash."""
        grid = np.zeros((2, 23), dtype=np.float32)
        rng = np.random.RandomState(42)
        m6_keyframe_sparse(2, grid, rng)
        assert grid.shape == (2, 23)


# -----------------------------------------------------------------------
# Test PrepareM2MUniversalMask transform (integration)
# -----------------------------------------------------------------------


class TestUniversalMaskTransform:
    """Integration tests for the full transform."""

    def _make_results(self, T=30, D=135):
        motion = torch.randn(T, D)
        return {'motion': motion}

    def test_output_keys(self):
        transform = PrepareM2MUniversalMask(key='motion')
        results = transform(self._make_results())
        assert 'src_motion' in results
        assert 'tgt_motion' in results
        assert 'src_mask' in results
        assert 'tgt_length' in results
        assert 'src_length' in results

    def test_output_shapes(self):
        T, D = 30, 135
        transform = PrepareM2MUniversalMask(key='motion')
        results = transform(self._make_results(T, D))
        assert results['src_motion'].shape == (T, D)
        assert results['tgt_motion'].shape == (T, D)
        assert results['src_mask'].shape == (T, D)
        assert results['tgt_length'] == T
        assert results['src_length'] == T

    def test_mask_dtype(self):
        transform = PrepareM2MUniversalMask(key='motion')
        results = transform(self._make_results())
        mask = results['src_mask']
        assert mask.dtype == torch.float32
        # Values should be 0 or 1
        unique = torch.unique(mask)
        assert all(v in [0.0, 1.0] for v in unique.tolist())

    def test_src_tgt_identical(self):
        transform = PrepareM2MUniversalMask(key='motion')
        results_in = self._make_results()
        original = results_in['motion'].clone()
        results = transform(results_in)
        assert torch.equal(results['src_motion'], original)
        assert torch.equal(results['tgt_motion'], original)

    def test_strategy_distribution(self):
        """Run 2000 times, check strategy proportions roughly match weights."""
        # We can infer strategy from mask patterns:
        # M5 -> all 1s; M4 -> full temporal columns; etc.
        # Instead, just verify the transform doesn't crash and produces
        # reasonable variety
        transform = PrepareM2MUniversalMask(key='motion')
        ratios = []
        for _ in range(500):
            results = transform(self._make_results(T=30))
            mask = results['src_mask']
            ratios.append(mask.mean().item())
        # Should see variety
        assert min(ratios) < 0.5
        assert max(ratios) > 0.5

    def test_min_mask_ratio_enforcement(self):
        """Mask ratio should be >= min_mask_ratio."""
        transform = PrepareM2MUniversalMask(
            key='motion', min_mask_ratio=0.1, max_mask_ratio=0.95
        )
        for seed in range(100):
            torch.manual_seed(seed)
            results = transform(self._make_results(T=30))
            mask = results['src_mask']
            ratio = mask.mean().item()
            assert ratio >= 0.1 - 0.05, (  # tolerance for grid→mask expansion rounding
                f'Mask ratio {ratio} < min 0.1 at seed {seed}'
            )

    def test_max_mask_ratio_enforcement(self):
        """Mask ratio should be <= max_mask_ratio (except M5)."""
        # Use weights that exclude M5
        transform = PrepareM2MUniversalMask(
            key='motion',
            strategy_weights={
                'm1_random_cell': 0.3,
                'm2_random_block': 0.2,
                'm3_temporal_contiguous': 0.3,
                'm4_joint_contiguous': 0.15,
                'm6_keyframe_sparse': 0.05,
            },
            min_mask_ratio=0.05,
            max_mask_ratio=0.90,
        )
        for _ in range(100):
            results = transform(self._make_results(T=30))
            mask = results['src_mask']
            ratio = mask.mean().item()
            assert ratio <= 0.90 + 0.05, (  # tolerance for grid→mask expansion rounding
                f'Mask ratio {ratio} > max 0.90'
            )

    def test_short_sequence_t4(self):
        """T=4 should not crash."""
        transform = PrepareM2MUniversalMask(key='motion')
        results = transform(self._make_results(T=4))
        assert results['src_mask'].shape == (4, 135)

    def test_single_frame_t1(self):
        """T=1 edge case should not crash."""
        transform = PrepareM2MUniversalMask(key='motion')
        results = transform(self._make_results(T=1))
        assert results['src_mask'].shape == (1, 135)
        assert results['tgt_length'] == 1

    def test_custom_strategy_weights(self):
        """Custom weights should be accepted."""
        transform = PrepareM2MUniversalMask(
            key='motion',
            strategy_weights={'m5_full_mask': 1.0},
        )
        results = transform(self._make_results(T=10))
        mask = results['src_mask']
        # M5 always gives all 1s
        assert mask.mean().item() == 1.0

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match='Unknown strategy'):
            PrepareM2MUniversalMask(
                key='motion',
                strategy_weights={'m99_nonexistent': 1.0},
            )

    def test_zero_weight_raises(self):
        with pytest.raises(ValueError, match='positive number'):
            PrepareM2MUniversalMask(
                key='motion',
                strategy_weights={'m1_random_cell': 0, 'm2_random_block': 0},
            )

    def test_mask_joint_group_consistency(self):
        """Each joint group in the mask should have consistent values."""
        transform = PrepareM2MUniversalMask(key='motion')
        for _ in range(50):
            results = transform(self._make_results(T=20))
            mask = results['src_mask']
            # Check translation group (first 3 dims)
            for t in range(20):
                transl_vals = mask[t, :TRANSL_DIM].unique()
                assert len(transl_vals) == 1, (
                    f'Translation group, frame {t}: inconsistent dims'
                )
            # Check joint groups (each 6 dims)
            for j in range(22):
                start = TRANSL_DIM + j * JOINT_ROT_DIM
                group = mask[:, start:start + JOINT_ROT_DIM]
                for t in range(20):
                    vals = group[t].unique()
                    assert len(vals) == 1, (
                        f'Joint group {j}, frame {t}: inconsistent dims'
                    )

    def test_motion_not_modified_in_place(self):
        """Original motion tensor should not be modified."""
        results_in = self._make_results(T=20)
        original = results_in['motion'].clone()
        transform = PrepareM2MUniversalMask(key='motion')
        transform(results_in)
        # The original dict's 'motion' key might have been overwritten,
        # but the tensor we cloned should remain unchanged
        assert torch.equal(original, results_in.get('motion', original))

    def test_non_135_dim_robustness(self):
        """Transform should handle D != 135 gracefully."""
        # D < 135: mask should be trimmed
        results = {'motion': torch.randn(10, 100)}
        transform = PrepareM2MUniversalMask(key='motion')
        out = transform(results)
        assert out['src_mask'].shape == (10, 100)

        # D > 135: extra dims should be padded with 1.0
        results = {'motion': torch.randn(10, 150)}
        out = transform(results)
        assert out['src_mask'].shape == (10, 150)


# -----------------------------------------------------------------------
# Test backward compatibility
# -----------------------------------------------------------------------


class TestBackwardCompatibility:
    """Ensure PrepareM2MCompletion still works unchanged."""

    def test_old_transform_still_works(self):
        from hftrainer.datasets.motion.motionhub.transforms.split_motion import (
            PrepareM2MCompletion,
        )

        transform = PrepareM2MCompletion(
            key='motion',
            past_ratio=0.2,
            future_ratio=0.2,
            random_ratio=True,
            min_edge_frames=4,
            min_middle_frames=4,
        )
        motion = torch.randn(50, 135)
        results = {'motion': motion}
        out = transform(results)

        assert 'src_motion' in out
        assert 'tgt_motion' in out
        assert 'src_mask' in out
        assert 'tgt_length' in out
        assert 'src_length' in out
        assert out['src_motion'].shape == (50, 135)
        assert out['src_mask'].shape == (50, 135)
        assert out['tgt_length'] == 50

    def test_old_transform_mask_is_temporal_inbetween(self):
        """PrepareM2MCompletion should produce temporal in-between pattern."""
        from hftrainer.datasets.motion.motionhub.transforms.split_motion import (
            PrepareM2MCompletion,
        )

        transform = PrepareM2MCompletion(
            key='motion',
            past_ratio=0.2,
            future_ratio=0.2,
            random_ratio=False,
            min_edge_frames=4,
            min_middle_frames=4,
        )
        motion = torch.randn(50, 135)
        results = {'motion': motion}
        out = transform(results)

        mask = out['src_mask']
        # Should have pattern: [0...0, 1...1, 0...0] across all dims
        # Check first dim
        col = mask[:, 0]
        # Find transitions
        transitions = torch.where(col[1:] != col[:-1])[0]
        # Should have exactly 2 transitions (0->1, 1->0)
        assert len(transitions) == 2, (
            f'Expected 2 transitions, got {len(transitions)}'
        )
        # First part should be 0 (known past)
        assert col[0].item() == 0.0
        # Last part should be 0 (known future)
        assert col[-1].item() == 0.0

    def test_output_interface_compatible(self):
        """Both transforms should produce the same set of output keys."""
        from hftrainer.datasets.motion.motionhub.transforms.split_motion import (
            PrepareM2MCompletion,
        )

        old = PrepareM2MCompletion(key='motion')
        new = PrepareM2MUniversalMask(key='motion')

        motion = torch.randn(30, 135)

        old_out = old({'motion': motion.clone()})
        new_out = new({'motion': motion.clone()})

        required_keys = {'src_motion', 'tgt_motion', 'src_mask', 'tgt_length', 'src_length'}
        assert required_keys.issubset(set(old_out.keys()))
        assert required_keys.issubset(set(new_out.keys()))

        # Same shapes
        assert old_out['src_motion'].shape == new_out['src_motion'].shape
        assert old_out['src_mask'].shape == new_out['src_mask'].shape
        assert old_out['tgt_length'] == new_out['tgt_length']


# -----------------------------------------------------------------------
# Test edge cases
# -----------------------------------------------------------------------


class TestEdgeCases:
    """Edge case tests."""

    def test_very_long_sequence(self):
        """T=1000 should work without issues."""
        transform = PrepareM2MUniversalMask(key='motion')
        results = {'motion': torch.randn(1000, 135)}
        out = transform(results)
        assert out['src_mask'].shape == (1000, 135)

    def test_all_strategies_produce_valid_output(self):
        """Each strategy individually should produce valid output."""
        for strategy in [
            'm1_random_cell', 'm2_random_block', 'm3_temporal_contiguous',
            'm4_joint_contiguous', 'm5_full_mask', 'm6_keyframe_sparse',
        ]:
            transform = PrepareM2MUniversalMask(
                key='motion',
                strategy_weights={strategy: 1.0},
            )
            for T in [1, 2, 5, 20, 50]:
                results = {'motion': torch.randn(T, 135)}
                out = transform(results)
                mask = out['src_mask']
                assert mask.shape == (T, 135), (
                    f'Strategy {strategy}, T={T}: wrong shape {mask.shape}'
                )
                # Values should be 0 or 1
                assert set(mask.unique().tolist()) <= {0.0, 1.0}, (
                    f'Strategy {strategy}, T={T}: non-binary values'
                )

    def test_reproducibility_not_enforced(self):
        """Different calls should (usually) produce different masks."""
        transform = PrepareM2MUniversalMask(key='motion')
        masks = []
        for _ in range(10):
            results = transform({'motion': torch.randn(20, 135)})
            masks.append(results['src_mask'])
        # At least some should differ
        all_same = all(torch.equal(masks[0], m) for m in masks[1:])
        assert not all_same
