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

    def test_m2m_loss_modality_mean_respects_sparse_channel_groups(self):
        """Sparse channel supervision should be averaged inside each modality first."""
        from hftrainer.models.motion.hymotion_m2m.network.m2m_loss import M2MLoss

        pred = torch.zeros(1, 1, 198)
        gt = torch.zeros(1, 1, 198)
        # Root rotation has 6 active channels with MSE=1.
        gt[..., 3:9] = 1.0
        # Joint position supervises only x/z subchannels with MSE=9.
        gt[..., 135:198:3] = 3.0
        gt[..., 137:198:3] = 3.0

        generation_mask = torch.zeros(1, 1, 198)
        generation_mask[..., 3:9] = 1.0
        generation_mask[..., 135:198:3] = 1.0
        generation_mask[..., 137:198:3] = 1.0

        loss_fn = M2MLoss(
            loss_type='mse',
            velocity_weight=1.0,
            velocity_loss_reduction='modality_mean',
            trans_dim_weight=1.0,
        )
        result = loss_fn(
            pred_vel=pred,
            gt_vel=gt,
            data_mask_temporal=torch.ones(1, 1),
            generation_mask=generation_mask,
        )

        # Active modality means: root_rot=1, joint_pos=9.  The final scalar
        # should be (1 + 9) / 2, not an element-count-weighted mean.
        assert torch.allclose(result['velocity'], torch.tensor(5.0))
        assert torch.allclose(result['velocity_root_rot'], torch.tensor(1.0))
        assert torch.allclose(result['velocity_joint_pos'], torch.tensor(9.0))

    def test_m2m_loss_component_mean_alias_matches_modality_mean(self):
        """component_mean remains a backward-compatible alias."""
        from hftrainer.models.motion.hymotion_m2m.network.m2m_loss import M2MLoss

        pred = torch.zeros(1, 1, 198)
        gt = torch.zeros(1, 1, 198)
        gt[..., 0:3] = 1.0
        gt[..., 3:9] = 2.0
        gt[..., 9:135] = 3.0
        gt[..., 135:198] = 4.0

        results = []
        for reduction in ('component_mean', 'modality_mean'):
            loss_fn = M2MLoss(
                loss_type='mse',
                velocity_weight=1.0,
                velocity_loss_reduction=reduction,
                trans_dim_weight=1.0,
            )
            results.append(loss_fn(
                pred_vel=pred,
                gt_vel=gt,
                data_mask_temporal=torch.ones(1, 1),
            ))

        assert torch.allclose(results[0]['velocity'], results[1]['velocity'])
        assert results[1]['velocity_root_rot'].item() == 4.0

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


    def test_component_mean_produces_per_component_detached_keys(self):
        """component_mean reduction must emit velocity_{trans,root_rot,body_rot,joint_pos}."""
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

        # Per-component keys must exist
        for key in ('velocity_trans', 'velocity_root_rot',
                     'velocity_body_rot', 'velocity_joint_pos'):
            assert key in result, f"Missing key: {key}"
            # Values must be detached (no grad_fn)
            assert not result[key].requires_grad, f"{key} should be detached"
            assert result[key].grad_fn is None, f"{key} should have no grad_fn"

        # Verify per-component MSE values: trans=1, root_rot=4, body_rot=9, joint_pos=16
        assert torch.allclose(result['velocity_trans'], torch.tensor(1.0))
        assert torch.allclose(result['velocity_root_rot'], torch.tensor(4.0))
        assert torch.allclose(result['velocity_body_rot'], torch.tensor(9.0))
        assert torch.allclose(result['velocity_joint_pos'], torch.tensor(16.0))

    def test_element_mean_has_no_per_component_keys(self):
        """element_mean reduction must NOT produce velocity_{trans,...} keys."""
        from hftrainer.models.motion.hymotion_m2m.network.m2m_loss import M2MLoss

        pred = torch.zeros(1, 1, 198)
        gt = torch.ones(1, 1, 198)

        loss_fn = M2MLoss(
            loss_type='mse',
            velocity_weight=1.0,
            velocity_loss_reduction='element_mean',
            trans_dim_weight=1.0,
        )
        result = loss_fn(
            pred_vel=pred,
            gt_vel=gt,
            data_mask_temporal=torch.ones(1, 1),
        )

        for key in ('velocity_trans', 'velocity_root_rot',
                     'velocity_body_rot', 'velocity_joint_pos'):
            assert key not in result, f"Unexpected key with element_mean: {key}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
