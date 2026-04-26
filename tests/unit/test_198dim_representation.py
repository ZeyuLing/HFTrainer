"""Unit tests for 198-dim motion representation.

Tests cover:
- 198-dim encoding/decoding
- FK consistency (position matches FK output)
- Stats correctness
- Compute198DimPosition transform
"""

import numpy as np
import pytest
import torch


class TestMotion198Representation:
    @pytest.fixture
    def bone_offsets(self):
        path = 'data/hymotion_m2m_data/bone_offsets_22.pt'
        try:
            return torch.load(path, map_location='cpu').float()
        except FileNotFoundError:
            pytest.skip(f"Bone offsets not found: {path}")

    def test_135_to_198_shape(self, bone_offsets):
        from hftrainer.datasets.motion.motionhub.transforms.compute_198dim import (
            motion135_to_198,
        )
        motion = torch.randn(10, 135)
        result = motion135_to_198(motion, bone_offsets)
        assert result.shape == (10, 198)

    def test_first_135_preserved(self, bone_offsets):
        from hftrainer.datasets.motion.motionhub.transforms.compute_198dim import (
            motion135_to_198,
        )
        motion = torch.randn(10, 135)
        result = motion135_to_198(motion, bone_offsets)
        assert torch.allclose(result[:, :135], motion)

    def test_198_to_135_roundtrip(self, bone_offsets):
        from hftrainer.datasets.motion.motionhub.transforms.compute_198dim import (
            motion135_to_198, motion198_to_135,
        )
        motion = torch.randn(10, 135)
        result = motion135_to_198(motion, bone_offsets)
        back = motion198_to_135(result)
        assert torch.allclose(back, motion)

    def test_position_channels_scheme_d(self, bone_offsets):
        """Position channels should follow Scheme D: XZ relative to pelvis, Y absolute."""
        from hftrainer.datasets.motion.motionhub.transforms.compute_198dim import (
            compute_position_channels,
        )
        from hftrainer.pipelines.motion.differentiable_fk import motion135_to_fk

        motion = torch.randn(5, 135)
        pos_63 = compute_position_channels(motion, bone_offsets)

        # Verify with manual FK
        with torch.no_grad():
            world_pos, _, _, _ = motion135_to_fk(motion, bone_offsets)

        pelvis_world = world_pos[:, 0:1, :]
        expected = world_pos[:, 1:, :].clone()
        expected[..., 0] -= pelvis_world[..., 0]
        expected[..., 2] -= pelvis_world[..., 2]
        expected_flat = expected.reshape(5, 63)

        assert torch.allclose(pos_63, expected_flat, atol=1e-5)

    def test_identity_rotation_tpose(self, bone_offsets):
        """Identity rotation should produce T-pose positions."""
        from hftrainer.datasets.motion.motionhub.transforms.compute_198dim import (
            compute_position_channels,
        )

        # Identity rot6d = [1, 0, 0, 1, 0, 0] (row-major, first two columns of I)
        T = 1
        motion = torch.zeros(T, 135)
        motion[:, 0:3] = 0  # zero translation
        for j in range(22):
            # Row-major identity: [R00,R01, R10,R11, R20,R21] = [1,0, 0,1, 0,0]
            motion[:, 3 + j * 6] = 1.0      # R00
            motion[:, 3 + j * 6 + 3] = 1.0  # R11

        pos_63 = compute_position_channels(motion, bone_offsets)
        # All joints should have reasonable T-pose positions
        assert pos_63.shape == (1, 63)
        # L_Hip (joint 1) X should be positive (left side)
        l_hip_x = pos_63[0, 0]
        r_hip_x = pos_63[0, 3]
        assert l_hip_x > 0, f"L_Hip X={l_hip_x} should be positive"
        assert r_hip_x < 0, f"R_Hip X={r_hip_x} should be negative"

    def test_batch_processing(self, bone_offsets):
        from hftrainer.datasets.motion.motionhub.transforms.compute_198dim import (
            motion135_to_198,
        )
        motion = torch.randn(4, 20, 135)  # batch of 4
        result = motion135_to_198(motion, bone_offsets)
        assert result.shape == (4, 20, 198)


class TestStats198:
    def test_local_stats_shape(self):
        try:
            mean = np.load('data/hymotion_m2m_data/_stats_198dim/Mean.npy')
            std = np.load('data/hymotion_m2m_data/_stats_198dim/Std.npy')
        except FileNotFoundError:
            pytest.skip("198-dim stats not found")

        assert mean.shape == (198,)
        assert std.shape == (198,)

    def test_global_stats_shape(self):
        try:
            mean = np.load('data/hymotion_m2m_data/_stats_198dim_global_rot/Mean.npy')
            std = np.load('data/hymotion_m2m_data/_stats_198dim_global_rot/Std.npy')
        except FileNotFoundError:
            pytest.skip("198-dim global rot stats not found")

        assert mean.shape == (198,)
        assert std.shape == (198,)

    def test_first_135_match_201(self):
        """First 135 dims of 198-dim stats should match 201-dim stats."""
        try:
            mean_198 = np.load('data/hymotion_m2m_data/_stats_198dim/Mean.npy')
            mean_201 = np.load('data/hymotion_m2m_data/_stats_201dim/Mean.npy')
        except FileNotFoundError:
            pytest.skip("Stats files not found")

        assert np.allclose(mean_198[:135], mean_201[:135])


class TestFKConsistencyLoss:
    @pytest.fixture
    def bone_offsets(self):
        try:
            return torch.load('data/hymotion_m2m_data/bone_offsets_22.pt', map_location='cpu').float()
        except FileNotFoundError:
            pytest.skip("Bone offsets not found")

    def test_gt_input_near_zero(self, bone_offsets):
        """FK loss should be near zero when position channels are FK-consistent."""
        from hftrainer.datasets.motion.motionhub.transforms.compute_198dim import (
            motion135_to_198, motion198_fk_loss,
        )

        motion_135 = torch.randn(2, 10, 135)
        motion_198 = motion135_to_198(motion_135.reshape(-1, 135), bone_offsets).reshape(2, 10, 198)

        # Use identity normalization (mean=0, std=1)
        mean = torch.zeros(198)
        std = torch.ones(198)

        loss = motion198_fk_loss(motion_198, mean, std, bone_offsets)
        assert loss.item() < 1e-4, f"FK loss should be ~0 for consistent input, got {loss.item()}"

    def test_gradient_flows(self, bone_offsets):
        """FK loss should allow gradient flow to rotation channels."""
        from hftrainer.datasets.motion.motionhub.transforms.compute_198dim import (
            motion198_fk_loss,
        )

        motion = torch.randn(1, 5, 198, requires_grad=True)
        mean = torch.zeros(198)
        std = torch.ones(198)

        loss = motion198_fk_loss(motion, mean, std, bone_offsets)
        loss.backward()
        assert motion.grad is not None
        assert motion.grad.abs().sum() > 0

    def test_t_squared_weighting(self, bone_offsets):
        """Loss should be smaller at t=0 due to t² weighting."""
        from hftrainer.datasets.motion.motionhub.transforms.compute_198dim import (
            motion198_fk_loss,
        )

        motion = torch.randn(2, 5, 198)
        mean = torch.zeros(198)
        std = torch.ones(198)

        loss_no_weight = motion198_fk_loss(motion, mean, std, bone_offsets, timesteps=None)
        loss_t0 = motion198_fk_loss(motion, mean, std, bone_offsets,
                                     timesteps=torch.tensor([0.01, 0.01]))
        loss_t1 = motion198_fk_loss(motion, mean, std, bone_offsets,
                                     timesteps=torch.tensor([1.0, 1.0]))

        assert loss_t0.item() < loss_t1.item(), "t=0 loss should be smaller than t=1"


class TestCompute198DimTransform:
    def test_transform_single_person(self):
        from hftrainer.datasets.motion.motionhub.transforms.compute_198dim import (
            Compute198DimPosition,
        )
        transform = Compute198DimPosition(key='motion')
        results = {'motion': torch.randn(20, 135)}
        try:
            out = transform.transform(results)
            assert out['motion'].shape == (20, 198)
        except FileNotFoundError:
            pytest.skip("Bone offsets not found")

    def test_transform_multi_person(self):
        from hftrainer.datasets.motion.motionhub.transforms.compute_198dim import (
            Compute198DimPosition,
        )
        transform = Compute198DimPosition(key='motion')
        results = {'motion': torch.randn(2, 15, 135)}
        try:
            out = transform.transform(results)
            assert out['motion'].shape == (2, 15, 198)
        except FileNotFoundError:
            pytest.skip("Bone offsets not found")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
