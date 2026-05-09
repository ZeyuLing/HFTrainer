"""Condition Routing modules for CRFM v3.

Implements:
- ConditionDensityEmbedding (CDE): encodes mask density as a continuous
  embedding via sinusoidal positional encoding + MLP. Similar to timestep
  embedding but for the generation ratio [0, 1].
- TextAttentionPreservation (TAP): applies gradient scaling to text-related
  parameters to prevent text attention atrophy during mixed training.
- text_awareness_loss (TAL): regularization loss that ensures text conditioning
  always affects the model's output in generated regions.

Design document: docs/temp/m2m_v3_crfm_implementation_plan.md
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ConditionDensityEmbedding(nn.Module):
    """Encode mask density as a continuous embedding.

    mask_density = 1.0 -> pure generation (T2M), model needs 100% text
    mask_density = 0.0 -> identity (all known), model needs 0% text
    mask_density = 0.3 -> partial completion, model needs moderate text

    The output is zero-initialized so that CDE has no effect at the
    start of training (gradual introduction).

    Args:
        dim: Output embedding dimension (should match feat_dim of MMDiT).
        max_period: Maximum period for sinusoidal encoding.
        mlp_ratio: Width multiplier for the hidden layer.
    """

    def __init__(self, dim: int = 1024, max_period: int = 10000, mlp_ratio: int = 4):
        super().__init__()
        self.dim = dim
        half = dim // 2
        # Frequency buffer (not trainable)
        freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32) / half)
        self.register_buffer('freqs', freqs)

        hidden = dim * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )
        # Zero-init output layer so CDE starts with no effect
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, density: Tensor) -> Tensor:
        """
        Args:
            density: (B,) float in [0, 1], mask density per sample.

        Returns:
            (B, dim) embedding tensor.
        """
        # density: (B,) -> (B, 1) for broadcasting
        args = density.unsqueeze(-1).float() * self.freqs.unsqueeze(0)  # (B, dim//2)
        emb = torch.cat([args.cos(), args.sin()], dim=-1)  # (B, dim)
        return self.mlp(emb)


class TextAttentionPreservation:
    """Utility class for applying gradient scaling to text-related parameters.

    Strategy:
    - Text cross-attention layers in double-stream blocks get scaled gradients
      (default 0.01x, i.e., near-frozen but not completely frozen).
    - This prevents text attention from atrophying while still allowing
      slow adaptation to the VACE context.

    Usage::

        tap = TextAttentionPreservation(text_grad_scale=0.01)
        tap.apply(model.motion_transformer)
        # Now text-related params have gradient hooks attached.
    """

    # Parameter name patterns that belong to the text pathway in MMDiT.
    # Only double-stream text parameters are scaled; single-stream blocks
    # mix text and motion tokens and cannot be isolated.
    TEXT_PARAM_PATTERNS = (
        'text_mod',       # Text modulation (shift/scale/gate)
        'text_norm',      # Text LayerNorm
        'text_qkv',       # Text Q/K/V projection
        'text_proj',      # Text output projection
        'text_mlp',       # Text MLP in double-stream
    )

    # Text refiner gets a slightly higher scale (can adapt more).
    REFINER_PATTERNS = (
        'text_refiner',
    )

    def __init__(
        self,
        text_grad_scale: float = 0.01,
        refiner_grad_scale: float = 0.1,
    ):
        self.text_grad_scale = text_grad_scale
        self.refiner_grad_scale = refiner_grad_scale
        self._hooks: List[torch.utils.hooks.RemovableHook] = []

    def apply(self, model: nn.Module) -> None:
        """Attach gradient scaling hooks to text-related parameters."""
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            scale = None
            # Check text patterns
            for pattern in self.TEXT_PARAM_PATTERNS:
                if pattern in name:
                    scale = self.text_grad_scale
                    break
            # Check refiner patterns (higher priority if both match)
            if scale is None:
                for pattern in self.REFINER_PATTERNS:
                    if pattern in name:
                        scale = self.refiner_grad_scale
                        break

            if scale is not None and scale < 1.0:
                hook = param.register_hook(lambda grad, s=scale: grad * s)
                self._hooks.append(hook)

    def remove(self) -> None:
        """Remove all gradient hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    @property
    def num_scaled_params(self) -> int:
        return len(self._hooks)


def text_awareness_loss(
    pred_with_text: Tensor,
    pred_without_text: Tensor,
    src_mask: Tensor,
    mask_density: Tensor,
    min_effect: float = 0.005,
    density_threshold: float = 0.7,
) -> Tensor:
    """Text-Awareness Loss (TAL): ensures text always affects generated regions.

    Penalizes when the model's output is identical regardless of whether
    text conditioning is present or null. Only active when motion condition
    is strong (mask_density < density_threshold), because when mask_density
    is near 1.0 (pure T2M), text effect is naturally high.

    Args:
        pred_with_text: (B, L, D) model prediction with real text.
        pred_without_text: (B, L, D) model prediction with null text.
            Should be .detach()-ed (no gradient through the null branch).
        src_mask: (B, L, D) mask, 1=generate, 0=known.
        mask_density: (B,) per-sample mask density in [0, 1].
        min_effect: Minimum expected absolute difference in generated
            regions. If actual difference < min_effect, loss is positive.
        density_threshold: Only apply loss when mask_density < this value.

    Returns:
        Scalar loss tensor.
    """
    # Compute per-sample mean absolute difference in generated regions only
    gen_count = src_mask.sum(dim=(-1, -2))  # (B,)
    diff = ((pred_with_text - pred_without_text) * src_mask).abs()
    diff_per_sample = diff.sum(dim=(-1, -2)) / (gen_count + 1e-6)  # (B,)

    # Only active when motion condition is strong
    apply_weight = (mask_density < density_threshold).float()  # (B,)

    # Hinge loss: penalize when text effect < min_effect
    loss = F.relu(min_effect - diff_per_sample) * apply_weight

    return loss.mean()


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def _test_cde_shape_and_gradient():
    """CDE output shape matches dim; gradients flow."""
    cde = ConditionDensityEmbedding(dim=1024)
    density = torch.rand(4, requires_grad=False)
    out = cde(density)
    assert out.shape == (4, 1024), f"Expected (4, 1024), got {out.shape}"
    out.sum().backward()
    assert cde.mlp[0].weight.grad is not None, "Gradient not flowing"
    print("  OK test_cde_shape_and_gradient")


def _test_cde_zero_init():
    """CDE output is near-zero at initialization."""
    cde = ConditionDensityEmbedding(dim=1024)
    density = torch.rand(4)
    with torch.no_grad():
        out = cde(density)
    assert out.abs().max() < 1e-5, f"CDE output not zero-init: max={out.abs().max()}"
    print("  OK test_cde_zero_init")


def _test_cde_different_densities():
    """Different densities produce different sinusoidal bases."""
    cde = ConditionDensityEmbedding(dim=1024)
    d0 = torch.tensor([0.0])
    d1 = torch.tensor([1.0])
    # Even with zero-init MLP, the sinusoidal base differs
    # After the MLP, both map to zero (due to zero-init), so we check the
    # intermediate sinusoidal embedding
    args0 = d0.unsqueeze(-1) * cde.freqs.unsqueeze(0)
    args1 = d1.unsqueeze(-1) * cde.freqs.unsqueeze(0)
    emb0 = torch.cat([args0.cos(), args0.sin()], dim=-1)
    emb1 = torch.cat([args1.cos(), args1.sin()], dim=-1)
    # d0=0 -> cos(0)=1 for all, sin(0)=0 for all
    # d1=1 -> cos(freq), sin(freq) varies
    # The difference should be nonzero
    assert (emb0 - emb1).abs().max() > 0.01, "Sinusoidal bases should differ"
    print("  OK test_cde_different_densities")


def _test_tap_gradient_scaling():
    """TAP correctly scales gradients of text-named parameters."""
    model = nn.ModuleDict({
        'text_mod_layer': nn.Linear(10, 10),
        'motion_layer': nn.Linear(10, 10),
    })
    tap = TextAttentionPreservation(text_grad_scale=0.01)
    tap.apply(model)
    assert tap.num_scaled_params > 0, "No params were hooked"

    x = torch.randn(2, 10)
    out = model['text_mod_layer'](x) + model['motion_layer'](x)
    out.sum().backward()

    text_grad_norm = model['text_mod_layer'].weight.grad.norm().item()
    motion_grad_norm = model['motion_layer'].weight.grad.norm().item()
    # text grad should be ~100x smaller
    ratio = text_grad_norm / (motion_grad_norm + 1e-10)
    assert ratio < 0.05, f"TAP ratio too high: {ratio:.4f}"
    tap.remove()
    print("  OK test_tap_gradient_scaling")


def _test_tal_penalizes_zero_effect():
    """TAL loss > 0 when text has zero effect on generated regions."""
    B, L, D = 2, 100, 198
    pred_with = torch.randn(B, L, D)
    pred_without = pred_with.clone()  # identical → zero text effect
    src_mask = torch.ones(B, L, D)
    src_mask[:, :50, :] = 0  # first 50 frames known
    mask_density = src_mask.mean(dim=(-1, -2))  # ~0.5

    loss = text_awareness_loss(pred_with, pred_without, src_mask,
                               mask_density, min_effect=0.005)
    assert loss.item() > 0, f"TAL should penalize zero text effect, got {loss.item()}"
    print("  OK test_tal_penalizes_zero_effect")


def _test_tal_zero_for_pure_t2m():
    """TAL loss = 0 for pure T2M (mask_density=1.0, above threshold)."""
    B, L, D = 2, 100, 198
    pred_with = torch.randn(B, L, D)
    pred_without = pred_with.clone()  # even zero effect is OK for pure T2M
    src_mask = torch.ones(B, L, D)  # all generate
    mask_density = torch.ones(B)  # density = 1.0

    loss = text_awareness_loss(pred_with, pred_without, src_mask,
                               mask_density, min_effect=0.005,
                               density_threshold=0.7)
    assert loss.item() == 0, f"TAL should be 0 for pure T2M, got {loss.item()}"
    print("  OK test_tal_zero_for_pure_t2m")


def _test_tal_zero_when_text_active():
    """TAL loss = 0 when text produces sufficient effect."""
    B, L, D = 2, 100, 198
    pred_with = torch.randn(B, L, D)
    # Large difference in generated regions
    pred_without = pred_with + torch.randn_like(pred_with) * 0.1
    src_mask = torch.ones(B, L, D)
    src_mask[:, :50, :] = 0
    mask_density = src_mask.mean(dim=(-1, -2))  # ~0.5

    loss = text_awareness_loss(pred_with, pred_without, src_mask,
                               mask_density, min_effect=0.005)
    assert loss.item() == 0, f"TAL should be 0 when text is active, got {loss.item()}"
    print("  OK test_tal_zero_when_text_active")


if __name__ == '__main__':
    print("Running condition_routing unit tests...")
    _test_cde_shape_and_gradient()
    _test_cde_zero_init()
    _test_cde_different_densities()
    _test_tap_gradient_scaling()
    _test_tal_penalizes_zero_effect()
    _test_tal_zero_for_pure_t2m()
    _test_tal_zero_when_text_active()
    print("All condition_routing tests passed!")
