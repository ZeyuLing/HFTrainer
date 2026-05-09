"""Unit tests for FK consistency loss integration."""

import pytest
import torch


class TestFKConsistencyLossIntegration:
    """Test that FK consistency loss integrates properly with M2MLoss."""

    def test_m2m_loss_fk_consistency_param(self):
        """M2MLoss should accept fk_consistency_weight."""
        from hftrainer.models.motion.hymotion_m2m.network.m2m_loss import M2MLoss

        loss_fn = M2MLoss(
            loss_type='mse',
            velocity_weight=1.0,
            fk_consistency_weight=0.1,
            fk_consistency_warmup_steps=1000,
        )
        assert loss_fn.fk_consistency_weight == 0.1
        assert loss_fn.fk_consistency_warmup_steps == 1000

    def test_m2m_loss_component_mean_velocity_reduction(self):
        """Component reduction should average semantic groups, not raw dims."""
        from hftrainer.models.motion.hymotion_m2m.network.m2m_loss import M2MLoss

        pred = torch.zeros(1, 1, 198)
        gt = torch.zeros(1, 1, 198)
        gt[..., 0:3] = 1.0
        gt[..., 3:9] = 2.0
        gt[..., 9:135] = 3.0
        gt[..., 135:198] = 4.0

        loss_fn = M2MLoss(
            loss_type='mse',
            velocity_weight=1.0,
            velocity_loss_reduction='component_mean',
            trans_dim_weight=1.0,
        )
        result = loss_fn(
            pred_vel=pred,
            gt_vel=gt,
            data_mask_temporal=torch.ones(1, 1),
        )

        # MSE per component: trans=1, root_rot=4, body_rot=9, pos=16.
        assert torch.allclose(result['velocity'], torch.tensor(7.5))

    def test_m2m_loss_component_mean_skips_empty_generation_groups(self):
        """A fully known component should not dilute active component means."""
        from hftrainer.models.motion.hymotion_m2m.network.m2m_loss import M2MLoss

        pred = torch.zeros(1, 1, 198)
        gt = torch.zeros(1, 1, 198)
        gt[..., 0:3] = 1.0
        gt[..., 3:9] = 2.0
        gt[..., 9:135] = 3.0
        gt[..., 135:198] = 4.0
        generation_mask = torch.ones(1, 1, 198)
        generation_mask[..., 135:198] = 0.0

        loss_fn = M2MLoss(
            loss_type='mse',
            velocity_weight=1.0,
            velocity_loss_reduction='component_mean',
            trans_dim_weight=1.0,
        )
        result = loss_fn(
            pred_vel=pred,
            gt_vel=gt,
            data_mask_temporal=torch.ones(1, 1),
            generation_mask=generation_mask,
        )

        # Pos has no generated cells, so active components are 1, 4, and 9.
        assert torch.allclose(result['velocity'], torch.tensor((1.0 + 4.0 + 9.0) / 3.0))

    def test_m2m_loss_component_mean_excludes_padding(self):
        """Padded frames must not contribute to component-mean reduction."""
        from hftrainer.models.motion.hymotion_m2m.network.m2m_loss import M2MLoss

        pred = torch.zeros(1, 2, 198)
        gt = torch.zeros(1, 2, 198)
        gt[:, 0, 0:3] = 1.0
        gt[:, 0, 3:9] = 2.0
        gt[:, 0, 9:135] = 3.0
        gt[:, 0, 135:198] = 4.0
        # The padded frame is deliberately huge; the loss should be identical
        # to the single valid frame if padding is masked correctly.
        gt[:, 1, :] = 1000.0

        loss_fn = M2MLoss(
            loss_type='mse',
            velocity_weight=1.0,
            velocity_loss_reduction='component_mean',
            trans_dim_weight=1.0,
        )
        result = loss_fn(
            pred_vel=pred,
            gt_vel=gt,
            data_mask_temporal=torch.tensor([[1.0, 0.0]]),
        )

        assert torch.allclose(result['velocity'], torch.tensor(7.5))

    def test_m2m_loss_fk_consistency_in_forward(self):
        """M2MLoss should include fk_consistency in loss dict when provided."""
        from hftrainer.models.motion.hymotion_m2m.network.m2m_loss import M2MLoss

        loss_fn = M2MLoss(
            loss_type='mse',
            velocity_weight=1.0,
            fk_consistency_weight=0.5,
            fk_consistency_warmup_steps=100,
        )

        B, L, D = 2, 10, 198
        pred_vel = torch.randn(B, L, D)
        gt_vel = torch.randn(B, L, D)
        mask = torch.ones(B, L)
        fk_loss = torch.tensor(0.1)

        result = loss_fn(
            pred_vel=pred_vel,
            gt_vel=gt_vel,
            data_mask_temporal=mask,
            global_step=200,  # past warmup
            fk_consistency_loss=fk_loss,
        )

        assert 'fk_consistency' in result
        # 0.5 * 1.0 (warmup=1 at step 200) * 0.1 = 0.05
        assert abs(result['fk_consistency'].item() - 0.05) < 1e-5

    def test_m2m_loss_fk_warmup(self):
        """FK loss should be scaled by warmup factor."""
        from hftrainer.models.motion.hymotion_m2m.network.m2m_loss import M2MLoss

        loss_fn = M2MLoss(
            loss_type='mse',
            velocity_weight=1.0,
            fk_consistency_weight=1.0,
            fk_consistency_warmup_steps=1000,
        )

        B, L, D = 2, 10, 198
        mask = torch.ones(B, L)
        fk_loss = torch.tensor(1.0)

        # At step 500: warmup = 0.5
        result = loss_fn(
            pred_vel=torch.randn(B, L, D),
            gt_vel=torch.randn(B, L, D),
            data_mask_temporal=mask,
            global_step=500,
            fk_consistency_loss=fk_loss,
        )
        assert abs(result['fk_consistency'].item() - 0.5) < 1e-5

    def test_m2m_loss_no_fk_when_weight_zero(self):
        """FK loss should not appear when weight=0."""
        from hftrainer.models.motion.hymotion_m2m.network.m2m_loss import M2MLoss

        loss_fn = M2MLoss(
            loss_type='mse',
            velocity_weight=1.0,
            fk_consistency_weight=0.0,
        )

        B, L, D = 2, 10, 198
        result = loss_fn(
            pred_vel=torch.randn(B, L, D),
            gt_vel=torch.randn(B, L, D),
            data_mask_temporal=torch.ones(B, L),
            global_step=100,
            fk_consistency_loss=torch.tensor(1.0),
        )
        assert 'fk_consistency' not in result


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
