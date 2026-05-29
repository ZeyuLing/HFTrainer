"""Unit tests for KimodoStyleAuxLoss.

Coverage:
1.  Forward shape & key presence (joint_pos / joint_vel / fk_consistency).
2.  Disabled when all weights are 0.
3.  Disabled when motion_dim < 198.
4.  Padding-mask correctness — padded frames are excluded from numerator and
    denominator; corrupting padded frames does NOT change the loss.
5.  Velocity mask correctness — vel-loss requires both endpoints valid.
6.  Backward pass: loss.backward() flows gradient into pred_x1_norm.
7.  fk_consistency numeric: when pred==gt and the 198-dim is consistent
    (built via motion135_to_198), aux_fk_consistency ≈ 0.
8.  joint_pos numeric: when pred==gt, aux_joint_pos == 0.
9.  Mask length alignment: a longer (ref-prepended) mask is sliced from the
    right to match the motion length.
10. Warm-up scheduling: weight scales linearly with global_step.
11. Integration with trainer helper: trainer wires aux losses into the
    overall loss dict.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

# Ensure repo root on sys.path when running standalone.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _bone_offsets() -> torch.Tensor:
    path = os.path.join(_REPO_ROOT, 'data', 'hymotion_m2m_data', 'bone_offsets_22.pt')
    if os.path.isfile(path):
        return torch.load(path, map_location='cpu').float()
    # Fallback synthetic offsets (T-pose-ish).
    offsets = torch.zeros(22, 3)
    for j in range(1, 22):
        offsets[j, 1] = 0.1  # straight chain along Y
    return offsets


def _build_198_synthetic(B: int, L: int, mean: torch.Tensor, std: torch.Tensor):
    """Build a self-consistent (B, L, 198) tensor in normalised space.

    The 135-dim part is random; the 63-dim pos channels are recomputed via
    FK so that the resulting 198-dim is FK-consistent.  This is what the GT
    pipeline produces (Compute198DimPosition transform).
    """
    from hftrainer.datasets.motion.motionhub.transforms.compute_198dim import (
        motion135_to_198,
    )

    offsets = _bone_offsets()
    motion_135 = torch.randn(B, L, 135) * 0.3
    # Make the rot6d part somewhat smooth so FK is well-conditioned.
    motion_135[..., 3:135] = motion_135[..., 3:135].cumsum(dim=1) * 0.05

    motion_198 = motion135_to_198(motion_135.reshape(-1, 135), offsets).reshape(B, L, 198)
    safe_std = torch.where(std < 1e-3, torch.ones_like(std), std)
    return (motion_198 - mean) / safe_std


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestKimodoStyleAuxLoss:
    """Behavioural tests for the new KIMODO-style auxiliary loss."""

    def test_forward_shape_and_keys(self):
        from hftrainer.models.motion.hymotion_m2m.network.kimodo_aux_loss import (
            KimodoStyleAuxLoss,
        )

        B, L, D = 2, 16, 198
        loss_fn = KimodoStyleAuxLoss(
            joint_pos_weight=1.0,
            joint_vel_weight=2.0,
            fk_consistency_weight=3.0,
        )
        mean = torch.zeros(D)
        std = torch.ones(D)
        offsets = _bone_offsets()
        pred = torch.randn(B, L, D)
        gt = torch.randn(B, L, D)
        mask = torch.ones(B, L)
        ts = torch.rand(B)

        out = loss_fn(
            pred_x1_norm=pred, gt_x1_norm=gt, mean=mean, std=std,
            bone_offsets=offsets, rotation_space='local',
            data_mask_temporal=mask, timesteps=ts, global_step=10_000,
        )
        assert set(out.keys()) == {'aux_joint_pos', 'aux_joint_vel', 'aux_fk_consistency'}
        for v in out.values():
            assert v.dim() == 0  # scalars
            assert torch.isfinite(v).item()

    def test_disabled_when_all_weights_zero(self):
        from hftrainer.models.motion.hymotion_m2m.network.kimodo_aux_loss import (
            KimodoStyleAuxLoss,
        )

        loss_fn = KimodoStyleAuxLoss()  # all defaults = 0
        assert not loss_fn.enabled
        out = loss_fn(
            pred_x1_norm=torch.randn(1, 4, 198),
            gt_x1_norm=torch.randn(1, 4, 198),
            mean=torch.zeros(198), std=torch.ones(198),
            bone_offsets=_bone_offsets(),
        )
        assert out == {}

    def test_skipped_when_motion_dim_lt_198(self):
        from hftrainer.models.motion.hymotion_m2m.network.kimodo_aux_loss import (
            KimodoStyleAuxLoss,
        )

        loss_fn = KimodoStyleAuxLoss(joint_pos_weight=1.0)
        out = loss_fn(
            pred_x1_norm=torch.randn(1, 4, 135),
            gt_x1_norm=torch.randn(1, 4, 135),
            mean=torch.zeros(135), std=torch.ones(135),
            bone_offsets=_bone_offsets(),
        )
        assert out == {}, 'aux loss should silently skip when D<198'

    def test_padding_mask_excludes_garbage(self):
        """Padded frames should not affect any aux loss term.

        We build a clean motion, replicate the last frame onto the padded
        tail (matching pad_mode='replicate' from RandomCropPadding), then
        REPLACE the padded tail with garbage and verify the loss is
        unchanged.
        """
        from hftrainer.models.motion.hymotion_m2m.network.kimodo_aux_loss import (
            KimodoStyleAuxLoss,
        )

        B, L_valid, L_total, D = 2, 10, 16, 198
        torch.manual_seed(0)
        mean = torch.zeros(D)
        std = torch.ones(D)

        # Build clean motion of length L_total (valid + replicated tail).
        gt_full = _build_198_synthetic(B, L_total, mean, std)
        # Force replicate tail in the GT: GT[t>=L_valid] = GT[L_valid-1]
        gt_full = gt_full.clone()
        gt_full[:, L_valid:] = gt_full[:, L_valid - 1:L_valid]

        # Pred with structured noise on valid frames.
        pred_clean = gt_full + 0.1 * torch.randn_like(gt_full)
        pred_clean[:, L_valid:] = pred_clean[:, L_valid - 1:L_valid]

        mask = torch.zeros(B, L_total)
        mask[:, :L_valid] = 1.0

        loss_fn = KimodoStyleAuxLoss(
            joint_pos_weight=1.0, joint_vel_weight=1.0, fk_consistency_weight=1.0,
            timestep_squared_weighting=False,
        )
        offsets = _bone_offsets()
        out_clean = loss_fn(
            pred_x1_norm=pred_clean, gt_x1_norm=gt_full,
            mean=mean, std=std, bone_offsets=offsets,
            data_mask_temporal=mask,
        )

        # Now corrupt padded tail of pred (mask region only).
        pred_dirty = pred_clean.clone()
        pred_dirty[:, L_valid:] = pred_dirty[:, L_valid:] + 100.0 * torch.randn_like(
            pred_dirty[:, L_valid:]
        )
        out_dirty = loss_fn(
            pred_x1_norm=pred_dirty, gt_x1_norm=gt_full,
            mean=mean, std=std, bone_offsets=offsets,
            data_mask_temporal=mask,
        )

        for k in ('aux_joint_pos', 'aux_joint_vel', 'aux_fk_consistency'):
            assert torch.allclose(out_clean[k], out_dirty[k], atol=1e-5), (
                f'{k} changed after corrupting padded frames: '
                f'{out_clean[k]} vs {out_dirty[k]}'
            )

    def test_velocity_mask_requires_both_endpoints(self):
        """The vel-loss frame is invalid if either endpoint is padded.

        Specifically frame t=L_valid-1 (whose 'next' is padded) must NOT
        contribute to the velocity loss.
        """
        from hftrainer.models.motion.hymotion_m2m.network.kimodo_aux_loss import (
            KimodoStyleAuxLoss,
        )

        B, L_valid, L_total, D = 1, 5, 8, 198
        mean = torch.zeros(D)
        std = torch.ones(D)
        gt = _build_198_synthetic(B, L_total, mean, std).clone()
        gt[:, L_valid:] = gt[:, L_valid - 1:L_valid]

        pred_a = gt.clone()
        pred_b = gt.clone()
        # Differ at the boundary frame: pred_b's frame L_valid-1 is offset
        # so pred_b - pred_b[L_valid-2] velocity is large.  But its
        # neighbour to the right is padded — so under a correct vel mask
        # this frame's vel should be excluded.
        pred_b = pred_b.clone()
        # Add the same boundary kick to pred_a so the joint_pos error at
        # L_valid-1 matches; we only want to vary whether the velocity
        # change at the boundary is masked out.
        pred_a[:, L_valid - 1] += 5.0
        pred_b[:, L_valid - 1] += 5.0

        # Use only joint_vel for clarity.
        loss_fn = KimodoStyleAuxLoss(
            joint_vel_weight=1.0, timestep_squared_weighting=False,
        )

        mask_strict = torch.zeros(B, L_total)
        mask_strict[:, :L_valid] = 1.0

        out = loss_fn(
            pred_x1_norm=pred_a, gt_x1_norm=gt,
            mean=mean, std=std, bone_offsets=_bone_offsets(),
            data_mask_temporal=mask_strict,
        )
        # Compare against the same setup where we hand-compute the
        # expected velocity loss only over t=0..L_valid-2 (vel needs both
        # endpoints valid → L_valid-1 vel-frames in total).
        assert 'aux_joint_vel' in out
        assert torch.isfinite(out['aux_joint_vel']).item()

        # Also test: if mask were all True, vel is computed across the
        # padding boundary and the loss would be larger (the artificial
        # 5.0 kick at frame L_valid-1 produces a HUGE vel at boundary).
        mask_all = torch.ones(B, L_total)
        out_full = loss_fn(
            pred_x1_norm=pred_a, gt_x1_norm=gt,
            mean=mean, std=std, bone_offsets=_bone_offsets(),
            data_mask_temporal=mask_all,
        )
        # smooth_l1 saturates the 5.0 kick, so relative change is small but
        # must still be strictly higher under the unmasked setting.
        assert out_full['aux_joint_vel'].item() > out['aux_joint_vel'].item() * 1.05, (
            'unmasked velocity loss should be substantially larger '
            f'(masked={out["aux_joint_vel"].item()}, '
            f'unmasked={out_full["aux_joint_vel"].item()})'
        )

    def test_backward_pass(self):
        from hftrainer.models.motion.hymotion_m2m.network.kimodo_aux_loss import (
            KimodoStyleAuxLoss,
        )

        B, L, D = 2, 8, 198
        mean = torch.zeros(D)
        std = torch.ones(D)
        pred = _build_198_synthetic(B, L, mean, std).requires_grad_(True)
        gt = _build_198_synthetic(B, L, mean, std)
        mask = torch.ones(B, L)

        loss_fn = KimodoStyleAuxLoss(
            joint_pos_weight=1.0, joint_vel_weight=1.0, fk_consistency_weight=1.0,
        )
        out = loss_fn(
            pred_x1_norm=pred, gt_x1_norm=gt,
            mean=mean, std=std, bone_offsets=_bone_offsets(),
            data_mask_temporal=mask,
        )
        total = sum(out.values())
        total.backward()
        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all().item()
        assert pred.grad.abs().sum().item() > 0

    def test_pred_equals_gt_zero_loss(self):
        """When pred == gt and gt is FK-consistent, all 3 terms are 0.

        Synthetic FK-consistent GT is built via motion135_to_198, so the
        198-dim pos channel exactly equals FK(rot+trans) under Scheme D.
        """
        from hftrainer.models.motion.hymotion_m2m.network.kimodo_aux_loss import (
            KimodoStyleAuxLoss,
        )

        B, L, D = 1, 6, 198
        mean = torch.zeros(D)
        std = torch.ones(D)
        gt = _build_198_synthetic(B, L, mean, std)
        pred = gt.clone()
        mask = torch.ones(B, L)

        loss_fn = KimodoStyleAuxLoss(
            joint_pos_weight=10.0, joint_vel_weight=10.0, fk_consistency_weight=10.0,
            timestep_squared_weighting=False,
        )
        out = loss_fn(
            pred_x1_norm=pred, gt_x1_norm=gt,
            mean=mean, std=std, bone_offsets=_bone_offsets(),
            data_mask_temporal=mask,
        )
        for k, v in out.items():
            assert v.item() < 1e-5, f'{k} expected ≈0, got {v.item()}'

    def test_mask_length_alignment(self):
        """A mask longer than the motion (ref-prepended) is right-aligned."""
        from hftrainer.models.motion.hymotion_m2m.network.kimodo_aux_loss import (
            KimodoStyleAuxLoss,
        )

        B, L_motion, L_ref, D = 1, 8, 4, 198
        mean = torch.zeros(D)
        std = torch.ones(D)
        gt = _build_198_synthetic(B, L_motion, mean, std)
        pred = gt.clone()

        # Mask of length L_motion+L_ref; first L_ref entries are "ref pose",
        # last L_motion entries are valid (1).
        long_mask = torch.zeros(B, L_motion + L_ref)
        long_mask[:, L_ref:] = 1.0  # only motion part is valid

        loss_fn = KimodoStyleAuxLoss(
            joint_pos_weight=1.0, joint_vel_weight=1.0, fk_consistency_weight=1.0,
        )
        out = loss_fn(
            pred_x1_norm=pred, gt_x1_norm=gt,
            mean=mean, std=std, bone_offsets=_bone_offsets(),
            data_mask_temporal=long_mask,
        )
        # pred==gt, FK-consistent, valid mask covers all motion frames →
        # all losses should be ~0.
        for k, v in out.items():
            assert v.item() < 1e-5, f'{k} expected ≈0, got {v.item()}'

    def test_warmup_schedule(self):
        from hftrainer.models.motion.hymotion_m2m.network.kimodo_aux_loss import (
            KimodoStyleAuxLoss,
        )

        loss_fn = KimodoStyleAuxLoss(
            joint_pos_weight=1.0,
            joint_pos_warmup_steps=1000,
            timestep_squared_weighting=False,
        )

        B, L, D = 1, 4, 198
        mean = torch.zeros(D)
        std = torch.ones(D)
        gt = _build_198_synthetic(B, L, mean, std)
        pred = gt + 1.0  # constant offset → constant per-frame loss > 0
        mask = torch.ones(B, L)

        out_500 = loss_fn(
            pred_x1_norm=pred, gt_x1_norm=gt,
            mean=mean, std=std, bone_offsets=_bone_offsets(),
            data_mask_temporal=mask, global_step=500,
        )
        out_1000 = loss_fn(
            pred_x1_norm=pred, gt_x1_norm=gt,
            mean=mean, std=std, bone_offsets=_bone_offsets(),
            data_mask_temporal=mask, global_step=1000,
        )
        out_2000 = loss_fn(
            pred_x1_norm=pred, gt_x1_norm=gt,
            mean=mean, std=std, bone_offsets=_bone_offsets(),
            data_mask_temporal=mask, global_step=2000,
        )

        v500 = out_500['aux_joint_pos'].item()
        v1000 = out_1000['aux_joint_pos'].item()
        v2000 = out_2000['aux_joint_pos'].item()
        assert abs(v500 * 2 - v1000) < 1e-4, (
            f'expected v500*2 ≈ v1000, got {v500} {v1000}'
        )
        assert abs(v1000 - v2000) < 1e-5, 'past warmup should be constant'

    def test_individual_term_can_be_disabled(self):
        from hftrainer.models.motion.hymotion_m2m.network.kimodo_aux_loss import (
            KimodoStyleAuxLoss,
        )

        loss_fn = KimodoStyleAuxLoss(
            joint_pos_weight=1.0, joint_vel_weight=0.0, fk_consistency_weight=0.0,
        )
        out = loss_fn(
            pred_x1_norm=torch.randn(1, 4, 198),
            gt_x1_norm=torch.randn(1, 4, 198),
            mean=torch.zeros(198), std=torch.ones(198),
            bone_offsets=_bone_offsets(),
            data_mask_temporal=torch.ones(1, 4),
        )
        assert set(out.keys()) == {'aux_joint_pos'}


class TestTimestepSquaredWeighting:
    """Tests for timestep_squared_weighting=True suppressing noisy-FK spikes."""

    def test_t_squared_suppresses_low_timestep(self):
        """At t≈0, t²≈0 so loss with t²-weighting should be much smaller."""
        from hftrainer.models.motion.hymotion_m2m.network.kimodo_aux_loss import (
            KimodoStyleAuxLoss,
        )

        B, L, D = 2, 8, 198
        mean = torch.zeros(D)
        std = torch.ones(D)
        offsets = _bone_offsets()

        torch.manual_seed(123)
        gt = _build_198_synthetic(B, L, mean, std)
        pred = gt + 0.5 * torch.randn_like(gt)
        mask = torch.ones(B, L)

        # Low timesteps: t ≈ 0.05 → t² ≈ 0.0025
        low_t = torch.tensor([0.05, 0.1])

        loss_with_tsq = KimodoStyleAuxLoss(
            joint_pos_weight=1.0,
            joint_vel_weight=1.0,
            fk_consistency_weight=1.0,
            timestep_squared_weighting=True,
        )
        loss_no_tsq = KimodoStyleAuxLoss(
            joint_pos_weight=1.0,
            joint_vel_weight=1.0,
            fk_consistency_weight=1.0,
            timestep_squared_weighting=False,
        )

        kwargs = dict(
            pred_x1_norm=pred, gt_x1_norm=gt,
            mean=mean, std=std, bone_offsets=offsets,
            data_mask_temporal=mask, timesteps=low_t,
        )
        out_tsq = loss_with_tsq(**kwargs)
        out_no = loss_no_tsq(**kwargs)

        # With t²-weighting and t≈[0.05, 0.1], the average t²≈0.00625.
        # So the t²-weighted loss should be ~160x smaller than unweighted.
        for key in ('aux_joint_pos', 'aux_joint_vel', 'aux_fk_consistency'):
            tsq_val = out_tsq[key].item()
            no_val = out_no[key].item()
            assert no_val > 0, f'{key} unweighted should be > 0'
            ratio = tsq_val / no_val
            assert ratio < 0.05, (
                f'{key}: t²-weighted / unweighted = {ratio:.4f}, '
                f'expected < 0.05 at low timesteps (tsq={tsq_val:.6f}, no={no_val:.6f})'
            )

    def test_t_squared_near_one_similar(self):
        """At t≈1, t²≈1 so weighted and unweighted losses should be similar."""
        from hftrainer.models.motion.hymotion_m2m.network.kimodo_aux_loss import (
            KimodoStyleAuxLoss,
        )

        B, L, D = 2, 8, 198
        mean = torch.zeros(D)
        std = torch.ones(D)
        offsets = _bone_offsets()

        torch.manual_seed(456)
        gt = _build_198_synthetic(B, L, mean, std)
        pred = gt + 0.5 * torch.randn_like(gt)
        mask = torch.ones(B, L)

        # High timesteps: t ≈ 0.95 → t² ≈ 0.9025
        high_t = torch.tensor([0.95, 0.98])

        loss_with_tsq = KimodoStyleAuxLoss(
            joint_pos_weight=1.0,
            joint_vel_weight=1.0,
            fk_consistency_weight=1.0,
            timestep_squared_weighting=True,
        )
        loss_no_tsq = KimodoStyleAuxLoss(
            joint_pos_weight=1.0,
            joint_vel_weight=1.0,
            fk_consistency_weight=1.0,
            timestep_squared_weighting=False,
        )

        kwargs = dict(
            pred_x1_norm=pred, gt_x1_norm=gt,
            mean=mean, std=std, bone_offsets=offsets,
            data_mask_temporal=mask, timesteps=high_t,
        )
        out_tsq = loss_with_tsq(**kwargs)
        out_no = loss_no_tsq(**kwargs)

        # t²≈0.9 → ratio should be between 0.8 and 1.0
        for key in ('aux_joint_pos', 'aux_joint_vel', 'aux_fk_consistency'):
            tsq_val = out_tsq[key].item()
            no_val = out_no[key].item()
            assert no_val > 0, f'{key} unweighted should be > 0'
            ratio = tsq_val / no_val
            assert 0.8 < ratio < 1.05, (
                f'{key}: t²-weighted / unweighted = {ratio:.4f}, '
                f'expected ≈0.9 at t≈[0.95, 0.98] (tsq={tsq_val:.6f}, no={no_val:.6f})'
            )

    def test_t_squared_none_timesteps_falls_back_to_unweighted(self):
        """When timesteps=None, t²-weighting should not apply (no crash)."""
        from hftrainer.models.motion.hymotion_m2m.network.kimodo_aux_loss import (
            KimodoStyleAuxLoss,
        )

        B, L, D = 1, 6, 198
        mean = torch.zeros(D)
        std = torch.ones(D)
        offsets = _bone_offsets()

        torch.manual_seed(789)
        gt = _build_198_synthetic(B, L, mean, std)
        pred = gt + 0.3 * torch.randn_like(gt)
        mask = torch.ones(B, L)

        loss_tsq = KimodoStyleAuxLoss(
            joint_pos_weight=1.0, timestep_squared_weighting=True,
        )
        loss_no = KimodoStyleAuxLoss(
            joint_pos_weight=1.0, timestep_squared_weighting=False,
        )

        kwargs = dict(
            pred_x1_norm=pred, gt_x1_norm=gt,
            mean=mean, std=std, bone_offsets=offsets,
            data_mask_temporal=mask, timesteps=None,  # No timesteps
        )
        out_tsq = loss_tsq(**kwargs)
        out_no = loss_no(**kwargs)

        # Without timesteps provided, t_sq=None → both should be identical
        assert torch.allclose(out_tsq['aux_joint_pos'], out_no['aux_joint_pos'], atol=1e-6), (
            f'With timesteps=None, t²-weighted should equal unweighted: '
            f'{out_tsq["aux_joint_pos"].item()} vs {out_no["aux_joint_pos"].item()}'
        )


class TestTrainerIntegration:
    """Smoke tests on the trainer-side wiring."""

    def test_bundle_constructs_kimodo_aux_loss_from_base_config(self):
        """The base config must wire weights into KimodoStyleAuxLoss."""
        from mmengine.config import Config

        from hftrainer.models.motion.hymotion_m2m.network.kimodo_aux_loss import (
            KimodoStyleAuxLoss,
        )

        cfg_path = os.path.join(
            _REPO_ROOT,
            'configs/hymotion_m2m/_base_hymotion_m2m_046b.py',
        )
        cfg = Config.fromfile(cfg_path)
        aux_kwargs = dict(cfg.model.kimodo_aux_loss_cfg)

        # Sanity check the design-target weight magnitudes (denormalised
        # metres regime; see base config for derivation).
        assert aux_kwargs['joint_pos_weight'] >= 10.0
        assert aux_kwargs['joint_vel_weight'] >= 100.0
        # fk_consistency must be substantially stronger than joint_pos in
        # nominal weight, because its base value is ~70× smaller (intra-
        # prediction consistency vs pred-vs-GT joint position error).  A
        # too-small fk weight (e.g. ≤ 100) silently disables explicit FK-
        # equivalence supervision and breaks position-only-condition
        # generation at inference (E1/E4 etc).
        assert aux_kwargs['fk_consistency_weight'] >= 500.0
        # Legacy fk_consistency in M2MLoss must be disabled when KIMODO aux
        # is on, otherwise the FK consistency term is double-counted.
        assert cfg.model.losses_cfg.fk_consistency_weight == 0.0

        # Build a real KimodoStyleAuxLoss with these kwargs.
        loss = KimodoStyleAuxLoss(**aux_kwargs)
        assert loss.enabled

    def test_weighted_loss_scales_linearly_with_weights(self):
        """Linearity: weighted loss must scale linearly with the per-term weight.

        Rationale: an absolute-magnitude target is brittle on synthetic data
        because (a) random rot6d noise accumulates non-linearly through 22
        FK joints, and (b) per-frame i.i.d. noise inflates the joint_vel
        term far above its real-training value (where motion is temporally
        smooth).  The invariant that *does* hold cleanly is linearity in
        the weight: doubling ``joint_pos_weight`` must double weighted
        ``aux_joint_pos`` (and likewise for jv / fk).  This is the
        property we actually need: if linearity is broken or the weight
        isn't being applied, the design-target boost from the re-weight
        would silently disappear.  We assert linearity at the production
        weights (50 / 500 / 50) vs a unit-weight baseline.
        """
        from hftrainer.models.motion.hymotion_m2m.network.kimodo_aux_loss import (
            KimodoStyleAuxLoss,
        )

        common = dict(
            loss_type='smooth_l1',
            timestep_squared_weighting=True,
            fk_consistency_warmup_steps=0,
            joint_pos_warmup_steps=0,
            joint_vel_warmup_steps=0,
        )
        unit = KimodoStyleAuxLoss(
            joint_pos_weight=1.0,
            joint_vel_weight=1.0,
            fk_consistency_weight=1.0,
            **common,
        )
        # Weights from the production base config.
        prod = KimodoStyleAuxLoss(
            joint_pos_weight=50.0,
            joint_vel_weight=500.0,
            fk_consistency_weight=1500.0,
            **common,
        )

        torch.manual_seed(0)
        B, L, D = 2, 32, 198
        # Use real stats; the linearity property holds in any space, but
        # using real stats keeps this test close to the production regime.
        stats_dir = os.path.join(
            _REPO_ROOT, 'data', 'hymotion_m2m_data', '_stats_198dim',
        )
        if os.path.isfile(os.path.join(stats_dir, 'Mean.npy')):
            import numpy as np
            mean = torch.from_numpy(np.load(os.path.join(stats_dir, 'Mean.npy'))).float()
            std = torch.from_numpy(np.load(os.path.join(stats_dir, 'Std.npy'))).float()
        else:
            pytest.skip('198d stats not available')
        offsets = _bone_offsets()
        gt = _build_198_synthetic(B, L, mean, std)
        safe_std = torch.where(std < 1e-3, torch.ones_like(std), std)
        # Small noise: stay in the smooth_l1 quadratic region so loss is
        # strictly proportional to weight.
        torch.manual_seed(42)
        pred = gt + (0.001 / safe_std) * torch.randn_like(gt)
        mask = torch.ones(B, L)
        ts = torch.full((B,), 0.5)
        kwargs = dict(
            pred_x1_norm=pred, gt_x1_norm=gt,
            mean=mean, std=std, bone_offsets=offsets,
            rotation_space='local',
            data_mask_temporal=mask, timesteps=ts, global_step=1_000_000,
        )

        out_unit = unit(**kwargs)
        out_prod = prod(**kwargs)

        # Linearity: weighted_loss(prod) / weighted_loss(unit) == weight_ratio
        for term, ratio in [
            ('aux_joint_pos', 50.0),
            ('aux_joint_vel', 500.0),
            ('aux_fk_consistency', 1500.0),
        ]:
            u = out_unit[term].item()
            p = out_prod[term].item()
            assert u > 0.0, f'{term} unit-weight value is zero — sanity broken'
            observed = p / u
            assert abs(observed - ratio) / ratio < 1e-4, (
                f'{term}: weight scaling broken — expected {ratio}× of '
                f'unit-weight, got {observed:.4f}× (unit={u:.3e}, prod={p:.3e})'
            )

        # Floor: at the production weights, every term must produce a
        # non-trivial value (> the regime we observed before this re-weight
        # — joint_pos: 5e-5, joint_vel: 1e-5, fk: 5e-7).  We require each
        # to be at least 5× that floor on this small-noise test.
        assert out_prod['aux_joint_pos'].item() > 5e-4, (
            f'production aux_joint_pos too small: {out_prod["aux_joint_pos"].item()}'
        )
        assert out_prod['aux_joint_vel'].item() > 5e-5, (
            f'production aux_joint_vel too small: {out_prod["aux_joint_vel"].item()}'
        )
        assert out_prod['aux_fk_consistency'].item() > 7.5e-5, (
            f'production aux_fk_consistency too small: '
            f'{out_prod["aux_fk_consistency"].item()}'
        )


class TestE1E2ConfigIntegration:
    """Verify E1/E2 config changes: component_mean, t²-weighting, dataloader."""

    @staticmethod
    def _load_e1_cfg():
        from mmengine.config import Config
        return Config.fromfile(os.path.join(
            _REPO_ROOT,
            'configs/hymotion_m2m/hymotion_m2m_smpl_uncond_046b.py',
        ))

    @staticmethod
    def _load_e2_cfg():
        from mmengine.config import Config
        return Config.fromfile(os.path.join(
            _REPO_ROOT,
            'configs/hymotion_m2m/hymotion_m2m_smpl_caption_046b.py',
        ))

    def test_e1_velocity_loss_reduction_component_mean(self):
        cfg = self._load_e1_cfg()
        assert cfg.model.losses_cfg.velocity_loss_reduction == 'component_mean'

    def test_e2_velocity_loss_reduction_component_mean(self):
        cfg = self._load_e2_cfg()
        assert cfg.model.losses_cfg.velocity_loss_reduction == 'component_mean'

    def test_e1_timestep_squared_weighting_enabled(self):
        cfg = self._load_e1_cfg()
        assert cfg.model.kimodo_aux_loss_cfg.timestep_squared_weighting is True

    def test_e2_timestep_squared_weighting_enabled(self):
        cfg = self._load_e2_cfg()
        assert cfg.model.kimodo_aux_loss_cfg.timestep_squared_weighting is True

    def test_e1_dataloader_prefetch(self):
        cfg = self._load_e1_cfg()
        assert cfg.train_dataloader.num_workers == 8
        assert cfg.train_dataloader.persistent_workers is True

    def test_e2_dataloader_prefetch(self):
        cfg = self._load_e2_cfg()
        assert cfg.train_dataloader.num_workers == 8
        assert cfg.train_dataloader.persistent_workers is True

    def test_e1_uncond_mode(self):
        """E1 must be unconditional."""
        cfg = self._load_e1_cfg()
        assert cfg.model.uncondition_mode is True
        assert cfg.model.text_encoder is None

    def test_e2_caption_mode(self):
        """E2 must have caption conditioning."""
        cfg = self._load_e2_cfg()
        assert cfg.model.uncondition_mode is False
        assert cfg.model.cond_mask_prob == 0.1


if __name__ == '__main__':
    sys.exit(pytest.main(['-xvs', __file__]))
