"""HyMotion-M2M CRFM Trainer: Condition-Routed Flow Matching.

Extends HyMotionM2MTrainer with:
1. Condition Density Embedding (CDE) passed to predict_flow
2. Text Attention Preservation (TAP) gradient scaling
3. Text-Awareness Loss (TAL) regularization

Design document: docs/temp/m2m_v3_crfm_implementation_plan.md
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from hftrainer.registry import TRAINERS
from hftrainer.trainers.motion.hymotion_m2m_trainer import HyMotionM2MTrainer


@TRAINERS.register_module()
class HyMotionM2MCRFMTrainer(HyMotionM2MTrainer):
    """CRFM trainer for text-conditioned motion completion.

    Inherits all base training logic from HyMotionM2MTrainer and adds:
    - TAP (Text Attention Preservation): gradient scaling on text-related
      parameters to prevent text cross-attention atrophy.
    - TAL (Text-Awareness Loss): regularization that ensures text conditioning
      always affects generated regions, even when motion condition is strong.
    - CDE (Condition Density Embedding): passes mask_density to the model's
      forward so the timestep modulation is density-aware.

    Args:
        bundle: HyMotionM2MBundle instance.
        tal_weight: Weight for Text-Awareness Loss.
        tal_interval: Compute TAL every N steps (extra forward pass).
        tal_min_effect: Minimum expected text effect (hinge threshold).
        tal_density_threshold: Only apply TAL when mask_density < this.
        text_grad_scale: Gradient scale for text-related parameters (TAP).
            1.0 = no scaling (disabled). 0.01 = near-frozen.
        **kwargs: Additional args passed to HyMotionM2MTrainer.
    """

    def __init__(
        self,
        bundle,
        tal_weight: float = 0.01,
        tal_interval: int = 4,
        tal_min_effect: float = 0.005,
        tal_density_threshold: float = 0.7,
        text_grad_scale: float = 0.01,
        **kwargs,
    ):
        super().__init__(bundle, **kwargs)
        self.tal_weight = tal_weight
        self.tal_interval = tal_interval
        self.tal_min_effect = tal_min_effect
        self.tal_density_threshold = tal_density_threshold
        self.text_grad_scale = text_grad_scale

        # Apply TAP gradient scaling
        self._tap = None
        if text_grad_scale < 1.0:
            from hftrainer.models.motion.hymotion_m2m.network.condition_routing import (
                TextAttentionPreservation,
            )
            self._tap = TextAttentionPreservation(
                text_grad_scale=text_grad_scale,
                refiner_grad_scale=min(text_grad_scale * 10, 1.0),
            )
            self._tap.apply(self.bundle.motion_transformer)

    def _prepare_and_forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Override: compute mask_density and pass to predict_flow via CDE.

        We override _prepare_and_forward to inject mask_density into the
        model's forward call. The base class builds all tensors; we intercept
        the predict_flow call to add the density signal.
        """
        # Run base preparation (steps 1-4: padding, text, flow matching, VACE)
        # We need to reimplement the forward step to inject mask_density.
        # Rather than duplicating the entire method, we call super() and then
        # re-do only the forward pass if CDE is enabled.

        ctx = super()._prepare_and_forward(batch)

        # If CDE is enabled and model supports it, re-do the forward pass
        # with mask_density. This is slightly wasteful (double forward) but
        # keeps the code clean and works for the validation phase.
        # In production, we'd modify super() directly.
        if not getattr(self.bundle, 'enable_cde', False):
            return ctx

        # Compute mask_density from src_mask
        src_mask = ctx['src_mask']
        if src_mask is None:
            return ctx

        mask_density = src_mask.mean(dim=(-1, -2))  # (B,)

        # Re-do forward pass with mask_density
        x_input = torch.cat([ctx['x_t'], ctx['vace_context']], dim=-1)
        pred = self.bundle.predict_flow(
            x_input=x_input,
            ctxt_input=ctx['ctxt_input'],
            vtxt_input=ctx['vtxt_input'],
            timesteps=ctx['timesteps'],
            x_mask_temporal=ctx['tgt_padding_mask'],
            ctxt_mask_temporal=ctx['ctxt_mask_temporal'],
            mask_density=mask_density,
        )
        ctx['pred'] = pred
        ctx['mask_density'] = mask_density

        return ctx

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Train step with base loss + TAL.

        Steps:
        1. Base forward and loss (inherited from HyMotionM2MTrainer)
        2. TAL loss (every tal_interval steps, requires extra forward)
        """
        ctx = self._prepare_and_forward(batch)
        losses = self._compute_base_loss(ctx)

        # TAL loss (every N steps)
        global_step = self.get_global_step()
        if (self.tal_weight > 0
                and global_step % self.tal_interval == 0
                and ctx.get('src_mask') is not None):
            tal = self._compute_tal_loss(ctx)
            if tal is not None:
                losses['tal'] = tal

        loss = sum(losses.values())
        result = {'loss': loss}
        for k, v in losses.items():
            result[f'loss_{k}'] = v.detach()
        return result

    def _compute_tal_loss(self, ctx: Dict[str, Any]) -> Optional[Tensor]:
        """Compute Text-Awareness Loss via extra null-text forward.

        Returns:
            Scalar TAL loss, or None if conditions aren't met.
        """
        src_mask = ctx['src_mask']
        if src_mask is None or src_mask.sum() == 0:
            return None

        mask_density = ctx.get('mask_density')
        if mask_density is None:
            mask_density = src_mask.mean(dim=(-1, -2))

        # Skip if all samples are pure T2M or all known
        if (mask_density > 0.9).all():
            return None
        if (mask_density < 0.01).all():
            return None

        # Prepare null text embeddings (no gradient for this branch)
        B = ctx['x_t'].shape[0]
        ctxt_tokens = ctx['ctxt_input'].shape[1]

        null_vtxt = self.bundle.null_vtxt_feat.detach().expand(B, 1, -1)
        null_ctxt = self.bundle.null_ctxt_input.detach().expand(B, ctxt_tokens, -1)

        # Forward with null text
        x_input = torch.cat([ctx['x_t'], ctx['vace_context']], dim=-1)

        # Use no_grad for the null forward to save memory
        with torch.no_grad():
            pred_null = self.bundle.predict_flow(
                x_input=x_input,
                ctxt_input=null_ctxt,
                vtxt_input=null_vtxt,
                timesteps=ctx['timesteps'],
                x_mask_temporal=ctx['tgt_padding_mask'],
                ctxt_mask_temporal=ctx['ctxt_mask_temporal'],
                mask_density=mask_density if ctx.get('mask_density') is not None else None,
            )

        # TAL: penalize when text has no effect on generated regions
        from hftrainer.models.motion.hymotion_m2m.network.condition_routing import (
            text_awareness_loss,
        )
        tal = text_awareness_loss(
            pred_with_text=ctx['pred'],
            pred_without_text=pred_null.detach(),
            src_mask=src_mask,
            mask_density=mask_density,
            min_effect=self.tal_min_effect,
            density_threshold=self.tal_density_threshold,
        )

        return tal * self.tal_weight


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def _test_crfm_trainer_inherits_base():
    """CRFM trainer is a proper subclass of HyMotionM2MTrainer."""
    assert issubclass(HyMotionM2MCRFMTrainer, HyMotionM2MTrainer)
    print("  OK test_crfm_trainer_inherits_base")


def _test_crfm_trainer_tap_applied():
    """TAP hooks are applied when text_grad_scale < 1.0."""
    import torch.nn as nn

    # Create a minimal mock bundle
    class MockBundle(nn.Module):
        def __init__(self):
            super().__init__()
            self.motion_transformer = nn.ModuleDict({
                'text_mod_block': nn.Linear(10, 10),
                'motion_block': nn.Linear(10, 10),
            })
            self.null_vtxt_feat = nn.Parameter(torch.zeros(1, 1, 768), requires_grad=False)
            self.null_ctxt_input = nn.Parameter(torch.zeros(1, 1, 4096), requires_grad=False)
            self.pred_type = 'velocity'
            self.enable_cde = False

        def normalize_motion(self, x):
            return x

        def trainable_parameters(self):
            return list(self.parameters())

    bundle = MockBundle()
    trainer = HyMotionM2MCRFMTrainer(
        bundle=bundle,
        text_grad_scale=0.01,
        tal_weight=0.01,
    )
    assert trainer._tap is not None
    assert trainer._tap.num_scaled_params > 0
    print("  OK test_crfm_trainer_tap_applied")


def _test_crfm_trainer_no_tap_when_scale_1():
    """TAP is disabled when text_grad_scale=1.0."""
    import torch.nn as nn

    class MockBundle(nn.Module):
        def __init__(self):
            super().__init__()
            self.motion_transformer = nn.Linear(10, 10)
            self.null_vtxt_feat = nn.Parameter(torch.zeros(1, 1, 768), requires_grad=False)
            self.null_ctxt_input = nn.Parameter(torch.zeros(1, 1, 4096), requires_grad=False)
            self.pred_type = 'velocity'
            self.enable_cde = False

        def normalize_motion(self, x):
            return x

        def trainable_parameters(self):
            return list(self.parameters())

    bundle = MockBundle()
    trainer = HyMotionM2MCRFMTrainer(
        bundle=bundle,
        text_grad_scale=1.0,
    )
    assert trainer._tap is None
    print("  OK test_crfm_trainer_no_tap_when_scale_1")


if __name__ == '__main__':
    print("Running CRFM trainer unit tests...")
    _test_crfm_trainer_inherits_base()
    _test_crfm_trainer_tap_applied()
    _test_crfm_trainer_no_tap_when_scale_1()
    print("All CRFM trainer tests passed!")
