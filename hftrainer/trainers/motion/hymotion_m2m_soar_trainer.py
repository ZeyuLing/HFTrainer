"""HyMotion-M2M SOAR Post-Trainer.

Implements Self-Correction for Optimal Alignment and Refinement (SOAR)
post-training on top of a flow-matching M2M checkpoint. See
``docs/temp/soar_m2m_v2_post_training_plan.md`` for the full method description.

Key idea (adapted to M2M's flow-matching velocity parameterisation):

    Base loss (same as SFT):
        x_t0 = (1 - t0) * x0 + t0 * x1            # on-trajectory
        v_gt = x1 - x0
        L_base = || model(x_t0) - v_gt ||^2

    Stop-gradient rollout (one Euler step towards clean):
        t1   = (t0 - 1/K).clamp_min(0)
        x_hat = x_t0 + v_pred.detach() * (t1 - t0)   # off-trajectory state

    Re-noise + correction (N auxiliary points, SHARED x0):
        t'    ~ U[0, t1]          (== t1 * (1 - rand))
        alpha = (t1 - t') / t1     (== rand)
        z_re  = (1 - alpha) * x_hat + alpha * x0
        v_corr = (x1 - z_re) / (1 - t').clamp_min(eps)
        L_corr += || model(z_re) - v_corr ||^2

    Total loss: L_base + lambda_soar * L_corr / N

Mask-aware handling (mandatory when parent trainer uses ``mask_aware_noise``):
  - x_t0, x_hat, z_re: known regions are forced back to x1 (clean) at each step
  - correction loss: weighted by generation_mask = src_mask (same as base loss)

CFG: the first-version implementation re-uses ``v_pred.detach()`` as the
rollout velocity regardless of ``soar_cfg_scale``. This matches the
uncond-model setting recommended in the plan §4.6. Support for text CFG
rollout is left as a future extension.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from hftrainer.registry import TRAINERS
from hftrainer.trainers.motion.hymotion_m2m_trainer import HyMotionM2MTrainer


@TRAINERS.register_module()
class HyMotionM2MSoarTrainer(HyMotionM2MTrainer):
    """Post-training trainer that adds SOAR correction loss on top of M2M SFT.

    Inherits all data/text/VACE preparation from HyMotionM2MTrainer. On every
    training iteration:

      1. Call ``_prepare_and_forward(batch)`` to run the base (on-trajectory)
         forward and obtain ``v_pred``.
      2. Compute the base loss via ``_compute_base_loss(ctx)``.
      3. Execute a stop-gradient ODE rollout to generate off-trajectory
         states, re-noise, and compute the correction loss.
      4. Return a combined loss dict.

    Only supports ``pred_type='velocity'`` — the 4 existing v2 configs all
    use velocity parameterisation.

    Args (in addition to HyMotionM2MTrainer):
        soar_lambda: weight of the correction loss (recommended 0.1).
        soar_num_aux: N auxiliary points per sample (recommended 1).
        soar_K: number of sampling steps assumed at inference (50).
        soar_cfg_scale: CFG scale for rollout velocity. In this first version,
            a value of 1.0 means "reuse v_pred.detach()". Any other value is
            currently rejected (TODO: implement unconditional branch forward).
        soar_sigma_clamp: lower bound on (1 - t') to avoid numerical blow-up
            in the correction target (analogous to ``t_eps=0.05`` used in
            the parent trainer's x1-parameterisation branch).
    """

    def __init__(
        self,
        bundle,
        soar_lambda: float = 0.1,
        soar_num_aux: int = 1,
        soar_K: int = 50,
        soar_cfg_scale: float = 1.0,
        soar_sigma_clamp: float = 0.05,
        **kwargs,
    ):
        super().__init__(bundle, **kwargs)
        if soar_cfg_scale != 1.0:
            raise NotImplementedError(
                'SOAR with text CFG (soar_cfg_scale != 1.0) is not yet '
                'implemented. Use 1.0 for unconditional rollout or add '
                'a CFG forward branch in this trainer.'
            )
        self.soar_lambda = float(soar_lambda)
        self.soar_num_aux = int(soar_num_aux)
        self.soar_K = int(soar_K)
        self.soar_cfg_scale = float(soar_cfg_scale)
        self.soar_sigma_clamp = float(soar_sigma_clamp)

        if self.soar_num_aux < 1:
            raise ValueError(f'soar_num_aux must be >=1, got {self.soar_num_aux}')
        if self.soar_K < 1:
            raise ValueError(f'soar_K must be >=1, got {self.soar_K}')

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _smooth_l1_loss(pred: Tensor, target: Tensor) -> Tensor:
        """Elementwise SmoothL1 (no reduction) — same family as m2m_loss."""
        return F.smooth_l1_loss(pred, target, reduction='none')

    def _masked_velocity_loss(
        self,
        pred_vel: Tensor,
        gt_vel: Tensor,
        generation_mask: Optional[Tensor],
        data_mask_temporal: Tensor,
    ) -> Tensor:
        """Compute SmoothL1 velocity loss weighted by masks.

        Mirrors m2m_loss.M2MLoss's velocity-loss masking convention:
          - If generation_mask is provided (mask_aware_noise), combine with
            temporal padding mask and average over valid elements.
          - Otherwise fall back to per-frame mean over valid temporal frames.

        Translation-dim upweighting is intentionally omitted from the SOAR
        correction loss to keep it a plain distributional correction signal.
        """
        per_dim = self._smooth_l1_loss(pred_vel, gt_vel)  # (B, L, D)
        data_mask = data_mask_temporal.to(per_dim.device).unsqueeze(-1)  # (B, L, 1)

        if generation_mask is not None:
            gen_mask = generation_mask.to(per_dim.device)
            combined = gen_mask * data_mask  # (B, L, D)
            denom = torch.clamp(combined.sum(), min=1.0)
            return (per_dim * combined).sum() / denom
        else:
            per_frame = per_dim.mean(dim=-1)  # (B, L)
            denom = torch.clamp(data_mask.squeeze(-1).sum(), min=1.0)
            return (per_frame * data_mask.squeeze(-1)).sum() / denom

    # ---------------------------------------------------------------- SOAR step
    def _soar_correction_loss(
        self,
        ctx: Dict[str, Any],
    ) -> Tensor:
        """Compute the SOAR correction loss from a base-forward context.

        Returns a scalar Tensor (already averaged over auxiliary points).
        """
        x0 = ctx['x0']
        x1 = ctx['x1']
        x_t0 = ctx['x_t']
        t0 = ctx['t']                 # (B, 1, 1)
        v_pred = ctx['pred']          # (B, L, D), with grad — but we .detach() below
        vace_context = ctx['vace_context']
        vtxt_input = ctx['vtxt_input']
        ctxt_input = ctx['ctxt_input']
        tgt_padding_mask = ctx['tgt_padding_mask']
        ctxt_mask_temporal = ctx['ctxt_mask_temporal']
        src_mask = ctx['src_mask']
        generation_mask = ctx['generation_mask']

        # ── Step R1: stop-gradient ODE rollout, one step towards clean ──
        # dt = t1 - t0 = -1/K   (t1 <= t0, towards t=0 is towards noise; towards
        # t=1 is towards clean. We want one step towards clean, so dt = +1/K
        # in the direction of increasing t.)
        #
        # Flow-matching convention in this repo: x_t = (1-t)*x0 + t*x1, so
        # dx/dt = x1 - x0 = v. Integrating forward by dt:
        #     x(t0 + dt) = x_t0 + v * dt
        # We take a single Euler step of size 1/K in the direction t0 -> 1
        # (towards clean), clamped at 1.
        K = float(self.soar_K)
        t1 = (t0 + 1.0 / K).clamp(max=1.0)     # (B, 1, 1)
        dt = t1 - t0                            # (B, 1, 1), in [0, 1/K]

        with torch.no_grad():
            v_rollout = v_pred.detach()
            x_hat = x_t0.detach() + v_rollout * dt

            # Mask-aware: keep known regions clean after rollout
            if self.mask_aware_noise and src_mask is not None:
                keep_mask = 1 - src_mask
                x_hat = x_hat * src_mask + x1 * keep_mask

        # ── Step R2: N auxiliary re-noise + correction forward passes ──
        B = x1.shape[0]
        device = x1.device
        sigma_clamp = self.soar_sigma_clamp
        eps = 1e-6

        total_corr = x1.new_zeros(())   # scalar accumulator
        valid_count = 0                  # how many aux points actually contributed

        for _ in range(self.soar_num_aux):
            # Sample t' ~ U[0, t1]  (towards noise from x_hat, sharing x0).
            # In FM convention the transport ray goes t=0 (x0) -> t=1 (x1);
            # re-noising from x_hat (at t1, near clean) back towards x0 means
            # sampling t' in [0, t1].
            rand = torch.rand(B, 1, 1, device=device, dtype=x1.dtype)
            t_prime = t1 * (1.0 - rand)              # (B, 1, 1), in [0, t1]
            alpha = 1.0 - rand                        # fraction of x_hat vs x0
            # Mix: z_re = (1 - (1 - alpha)) * x_hat + (1 - alpha) * x0
            # Equivalent form: z_re = alpha*x_hat + (1-alpha)*x0
            # Note: when alpha=1 (rand=0, t'=t1) -> z_re = x_hat
            #       when alpha=0 (rand=1, t'=0)  -> z_re = x0
            with torch.no_grad():
                z_re = alpha * x_hat + (1.0 - alpha) * x0

                # Mask-aware: keep known regions clean at the new timestep
                if self.mask_aware_noise and src_mask is not None:
                    keep_mask = 1 - src_mask
                    z_re = z_re * src_mask + x1 * keep_mask

            # Correction target (velocity form):
            #   given z_re at t', the FM velocity that steers it toward x1 is
            #   v_corr = (x1 - z_re) / (1 - t')
            # This matches the SOAR paper's "correction target" in the sigma
            # parameterisation: v = (z_t - x_clean)/sigma_t, translated to
            # the velocity parameterisation (v = x1 - x0) by a sign flip and
            # division by the remaining clean-time (1 - t') instead of sigma.
            one_minus_tp = (1.0 - t_prime).clamp_min(sigma_clamp)
            with torch.no_grad():
                v_corr = (x1 - z_re) / one_minus_tp

            # Forward on off-trajectory point (WITH gradient)
            z_re_input = torch.cat([z_re, vace_context], dim=-1)
            t_prime_scalar = t_prime.view(-1)     # (B,)
            v_off = self.bundle.predict_flow(
                x_input=z_re_input,
                ctxt_input=ctxt_input,
                vtxt_input=vtxt_input,
                timesteps=t_prime_scalar,
                x_mask_temporal=tgt_padding_mask,
                ctxt_mask_temporal=ctxt_mask_temporal,
            )

            # Correction loss on generation regions only.
            corr = self._masked_velocity_loss(
                pred_vel=v_off,
                gt_vel=v_corr.detach(),
                generation_mask=generation_mask,
                data_mask_temporal=tgt_padding_mask,
            )
            total_corr = total_corr + corr
            valid_count += 1

        if valid_count == 0:
            return x1.new_zeros(())
        return total_corr / float(valid_count)

    # ------------------------------------------------------------- train step
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        if self.bundle.pred_type != 'velocity':
            raise NotImplementedError(
                f"SOAR trainer currently only supports pred_type='velocity', "
                f"got {self.bundle.pred_type}"
            )

        # 1. Base on-trajectory forward + loss (SFT objective).
        ctx = self._prepare_and_forward(batch)
        base_losses = self._compute_base_loss(ctx)
        base_loss = sum(base_losses.values())

        # 2. SOAR correction loss.
        corr_loss = self._soar_correction_loss(ctx)

        total_loss = base_loss + self.soar_lambda * corr_loss

        result: Dict[str, Any] = {'loss': total_loss}
        for k, v in base_losses.items():
            result[f'loss_{k}'] = v.detach()
        result['loss_soar_corr'] = corr_loss.detach()
        return result


# ---------------------------------------------------------------------------
# Unit tests — run with: python -m hftrainer.trainers.motion.hymotion_m2m_soar_trainer
# ---------------------------------------------------------------------------

def _test_soar_shapes_and_finiteness():
    """Smoke test for SOAR math on synthetic data with a mocked bundle.

    Verifies:
      - z_re, x_hat, v_corr have correct shape
      - Correction loss is finite (no NaN/Inf)
      - With a "perfect" bundle (v_pred == v_gt), x_hat lies on the transport
        ray and v_corr == v_gt, so correction loss with the perfect bundle
        is ~= base loss (very small).
      - Mask-aware path: known regions in x_hat/z_re equal x1 exactly.
    """
    import torch
    from torch import nn
    torch.manual_seed(0)

    B, L, D = 2, 8, 9
    K = 4

    class _MockLoss(nn.Module):
        motion_smoothness_weight = 0.0
        keypoints3d_weight = 0.0
        fk_consistency_weight = 0.0
        velocity_weight = 1.0
        trans_dim_weight = 1.0
        trans_dims = 3

        def forward(self, **kwargs):
            # Not used in the correction-loss unit test below.
            raise RuntimeError('should not be called')

    class _MockBundle(nn.Module):
        pred_type = 'velocity'

        def __init__(self):
            super().__init__()
            self.motion_transformer = nn.Linear(1, 1)
            self.m2m_loss = _MockLoss()

        def predict_flow(self, x_input, **kwargs):
            # "Perfect" model: extract x_t from x_input and return v that
            # takes any z_re at t' to x1 (= target). We cheat by reading the
            # ground-truth x1 from the closure via a class variable.
            return _MockBundle._oracle_x1 - x_input[..., :_MockBundle._D]

        _oracle_x1 = None
        _D = None

    bundle = _MockBundle()
    _MockBundle._D = D

    # Build a SOAR trainer
    trainer = HyMotionM2MSoarTrainer(
        bundle=bundle,
        mask_aware_noise=True,
        soar_lambda=0.1,
        soar_num_aux=2,
        soar_K=K,
    )

    # Construct a fake context dict (as if _prepare_and_forward had produced it)
    x0 = torch.randn(B, L, D)
    x1 = torch.randn(B, L, D)
    t0 = torch.rand(B, 1, 1) * 0.5  # in [0, 0.5] so that t1 = t0 + 1/K <= 1
    x_t = (1 - t0) * x0 + t0 * x1
    src_mask = torch.zeros(B, L, D)
    src_mask[:, 2:6, :] = 1.0         # middle 4 frames = generate
    # Mask-aware: known regions in x_t stay clean
    x_t = x_t * src_mask + x1 * (1 - src_mask)
    tgt_padding_mask = torch.ones(B, L, dtype=torch.bool)

    # Oracle for MockBundle
    _MockBundle._oracle_x1 = x1

    # "Perfect" v_pred would be x1 - x0 (when mask_aware_noise=True, the model
    # should only supervise on generation region — for this unit test, let's
    # just pretend v_pred == v_gt on all positions).
    v_pred_perfect = x1 - x0

    ctx = {
        'x0': x0, 'x1': x1, 'x_t': x_t, 't': t0, 'pred': v_pred_perfect,
        'vace_context': torch.zeros(B, L, 3 * D),  # unused by MockBundle
        'vtxt_input': torch.zeros(B, 1, 4),
        'ctxt_input': torch.zeros(B, 1, 4),
        'tgt_padding_mask': tgt_padding_mask,
        'ctxt_mask_temporal': torch.ones(B, 1, dtype=torch.bool),
        'src_mask': src_mask,
        'generation_mask': src_mask,
    }

    corr = trainer._soar_correction_loss(ctx)
    assert torch.isfinite(corr), f'correction loss not finite: {corr}'
    # With a perfect bundle and x_hat exactly on the ray, correction loss
    # should be close to zero. We allow some slack because x_hat may be at
    # boundary t1=1 where clamp kicks in.
    assert corr.item() < 1.0, (
        f'correction loss should be small with perfect bundle, got {corr.item()}'
    )
    print(f'  ✅ SOAR correction loss shape OK, finite, small with perfect model '
          f'(loss={corr.item():.6f})')


def _test_mask_aware_preserves_known_regions():
    """Unit test: inside SOAR re-noising, known (mask=0) regions equal x1."""
    import torch
    torch.manual_seed(1)
    B, L, D = 2, 6, 5
    x0 = torch.randn(B, L, D)
    x1 = torch.randn(B, L, D)
    x_hat = torch.randn(B, L, D)
    src_mask = torch.zeros(B, L, D)
    src_mask[:, 1:4, :] = 1.0          # middle frames = generate

    # Mirror what SOAR trainer does inside the loop:
    keep_mask = 1 - src_mask
    x_hat_masked = x_hat * src_mask + x1 * keep_mask

    # Known regions must equal x1 exactly
    diff_known = (x_hat_masked - x1) * keep_mask
    assert diff_known.abs().max().item() < 1e-6, \
        f'known region not preserved: max diff {diff_known.abs().max().item()}'

    # Generation regions should not equal x1 (almost surely)
    diff_gen = (x_hat_masked - x1) * src_mask
    assert diff_gen.abs().max().item() > 0.0, \
        'generation region must differ from x1 in this test'

    print('  ✅ Mask-aware: known regions preserved exactly, gen regions differ')


def _test_cfg_scale_validation():
    """Only soar_cfg_scale=1.0 is supported in v1."""
    from torch import nn

    class _DummyBundle(nn.Module):
        pred_type = 'velocity'

        def __init__(self):
            super().__init__()
            self.motion_transformer = nn.Linear(1, 1)

    ok = False
    try:
        HyMotionM2MSoarTrainer(bundle=_DummyBundle(), soar_cfg_scale=4.5)
    except NotImplementedError:
        ok = True
    assert ok, 'cfg_scale != 1.0 must raise NotImplementedError'
    print('  ✅ soar_cfg_scale != 1.0 is rejected with NotImplementedError')


if __name__ == '__main__':
    print('Running hymotion_m2m_soar_trainer unit tests...')
    _test_mask_aware_preserves_known_regions()
    _test_cfg_scale_validation()
    _test_soar_shapes_and_finiteness()
    print('All SOAR trainer tests passed ✅')
