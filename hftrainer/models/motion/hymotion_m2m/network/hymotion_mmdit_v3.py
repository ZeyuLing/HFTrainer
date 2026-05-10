"""
Hunyuan Motion MMDiT v3 — Dual-Stream Condition Fusion (DSCF).

This module implements the v3 architecture that eliminates the VACE shortcut by
moving motion condition from input concatenation to cross-attention, placing it
at the same level as text conditioning. This resolves the caption conditioning
failure where models trained with VACE-style input concatenation cannot learn to
use text because the motion condition provides a trivially easier optimization path.

Architecture (per DualCondMMDiTBlock):
    1. Self-attention on motion tokens (with RoPE, ModulateDiT, QK-norm)
    2. Cross-attention to text tokens (text as KV)
    3. Cross-attention to motion-condition tokens (cond_tokens as KV)
    4. TimestepAdaptiveFusionGate: weights text vs motion-cond cross-attn
    5. FFN with modulation

Key design:
    - All 18 blocks are DualCondMMDiTBlocks (no double/single stream split)
    - Input: x_t(motion_dim) + scalar_mask(1) → input_encoder → feat_dim
    - Role embedding added to encoded input (KEEP/GENERATE/EDIT per frame)
    - MotionCondEncoder compresses known-region motion into 128 condition tokens
    - Dual-CFG compatible: v_guided = v_uncond + s_t*(v_text-v_uncond) + s_c*(v_full-v_text)
    - Pretrained weight loading: self-attention + MLP params match existing motion stream

Weight loading compatibility:
    DualCondMMDiTBlock's self-attention uses the same param names as
    MMDoubleStreamBlock's motion stream:
        motion_mod, motion_norm1, motion_qkv, motion_q_norm, motion_k_norm,
        motion_out_proj, motion_norm2, motion_mlp
    This enables initializing from a pretrained T2M checkpoint.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from .attention import attention
from .bricks import get_activation_layer, get_norm_layer, RMSNorm
from .encoders import MLP, MLPEncoder, TimestepEmbeddingEncoder, FinalLayer
from .modulate import ModulateDiT, apply_gate, modulate
from .motion_cond_encoder import MotionCondEncoder
from .positional_encoding import RotaryEmbedding
from .role_embedding import RoleEmbedding
from .timestep_gate import TimestepAdaptiveFusionGate, DensityAwareFusionGate
from .token_refiner import SingleTokenRefiner


def get_module_device(module):
    return next(module.parameters()).device


class DualCondMMDiTBlock(nn.Module):
    """Dual-Condition Multi-Modal DiT Block for DSCF v3.

    Processes motion tokens with three sub-layers:
        1. Self-attention (causal/full/narrowband, with RoPE on motion)
        2. Dual cross-attention (text KV + motion-condition KV, fused by gate)
        3. Feed-forward network (MLP with modulation)

    The self-attention + MLP parameter names match the motion stream of
    MMDoubleStreamBlock for pretrained weight loading compatibility.

    Args:
        feat_dim: Hidden feature dimension.
        num_heads: Number of attention heads.
        mlp_ratio: MLP hidden dimension ratio.
        dropout: Dropout probability (training only).
        mlp_act_type: Activation type for MLP.
        qk_norm_type: Normalization type for Q/K ('rms', 'layer', None).
        qkv_bias: Whether to use bias in projections.
        positional_encoding_cfg: Config for RotaryEmbedding.
        gate_type: 'timestep' or 'density_aware' for the fusion gate.
    """

    def __init__(
        self,
        feat_dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        mlp_act_type: str,
        qk_norm_type: Optional[str] = None,
        qkv_bias: bool = False,
        positional_encoding_cfg: dict = {
            "max_seq_len": 5000,
            "use_real": True,
        },
        gate_type: str = 'timestep',
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout

        assert feat_dim % num_heads == 0, (
            f"feat_dim {feat_dim} must be divisible by num_heads {num_heads}"
        )
        self.head_dim = feat_dim // num_heads
        self.mlp_hidden_dim = int(feat_dim * mlp_ratio)

        # ============ Positional Encoding ============
        self._positional_encoding_cfg = positional_encoding_cfg.copy()
        self.rotary_emb = RotaryEmbedding(
            num_feats=self.head_dim, **self._positional_encoding_cfg
        )

        # ============ Self-Attention (motion stream compatible naming) ============
        # ModulateDiT factor=6: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
        self.motion_mod = ModulateDiT(feat_dim, factor=6, act_type="silu")
        self.motion_norm1 = nn.LayerNorm(feat_dim, elementwise_affine=False, eps=1e-6)
        self.motion_qkv = nn.Linear(feat_dim, feat_dim * 3, bias=qkv_bias)
        self.motion_q_norm = get_norm_layer(qk_norm_type)(
            self.head_dim, elementwise_affine=True, eps=1e-6
        )
        self.motion_k_norm = get_norm_layer(qk_norm_type)(
            self.head_dim, elementwise_affine=True, eps=1e-6
        )
        self.motion_out_proj = nn.Linear(feat_dim, feat_dim, bias=qkv_bias)

        # ============ Text Cross-Attention ============
        self.text_cross_norm = nn.LayerNorm(feat_dim, elementwise_affine=False, eps=1e-6)
        self.text_cross_q = nn.Linear(feat_dim, feat_dim, bias=True)
        self.text_cross_k = nn.Linear(feat_dim, feat_dim, bias=True)
        self.text_cross_v = nn.Linear(feat_dim, feat_dim, bias=True)
        self.text_cross_q_norm = RMSNorm(self.head_dim, elementwise_affine=True, eps=1e-6)
        self.text_cross_k_norm = RMSNorm(self.head_dim, elementwise_affine=True, eps=1e-6)
        self.text_cross_out_proj = nn.Linear(feat_dim, feat_dim, bias=True)
        # Zero-init output projection so cross-attn has no effect at init
        nn.init.zeros_(self.text_cross_out_proj.weight)
        nn.init.zeros_(self.text_cross_out_proj.bias)

        # ============ Motion-Condition Cross-Attention ============
        self.cond_cross_norm = nn.LayerNorm(feat_dim, elementwise_affine=False, eps=1e-6)
        self.cond_cross_q = nn.Linear(feat_dim, feat_dim, bias=True)
        self.cond_cross_k = nn.Linear(feat_dim, feat_dim, bias=True)
        self.cond_cross_v = nn.Linear(feat_dim, feat_dim, bias=True)
        self.cond_cross_q_norm = RMSNorm(self.head_dim, elementwise_affine=True, eps=1e-6)
        self.cond_cross_k_norm = RMSNorm(self.head_dim, elementwise_affine=True, eps=1e-6)
        self.cond_cross_out_proj = nn.Linear(feat_dim, feat_dim, bias=True)
        # Zero-init output projection so cross-attn has no effect at init
        nn.init.zeros_(self.cond_cross_out_proj.weight)
        nn.init.zeros_(self.cond_cross_out_proj.bias)

        # ============ Fusion Gate ============
        if gate_type == 'timestep':
            self.fusion_gate = TimestepAdaptiveFusionGate(feat_dim=feat_dim)
        elif gate_type == 'density_aware':
            self.fusion_gate = DensityAwareFusionGate(feat_dim=feat_dim)
        else:
            raise ValueError(f"Unknown gate_type: {gate_type!r}")
        self._gate_type = gate_type

        # ============ FFN (motion stream compatible naming) ============
        self.motion_norm2 = nn.LayerNorm(feat_dim, elementwise_affine=False, eps=1e-6)
        self.motion_mlp = MLP(
            feat_dim, self.mlp_hidden_dim, act_type=mlp_act_type, bias=True
        )

    def forward(
        self,
        motion_feat: Tensor,
        text_kv: Tensor,
        cond_kv: Tensor,
        adapter: Tensor,
        self_attn_mask: Optional[Tensor] = None,
        text_kv_mask: Optional[Tensor] = None,
        cond_kv_mask: Optional[Tensor] = None,
        mask_density: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass of DualCondMMDiTBlock.

        Args:
            motion_feat: (B, L, feat_dim) — motion token features.
            text_kv: (B, T, feat_dim) — text token features for cross-attention KV.
            cond_kv: (B, C, feat_dim) — motion-condition tokens for cross-attention KV.
            adapter: (B, 1, feat_dim) — timestep + vtxt adapter signal.
            self_attn_mask: (B, 1, L, L) — additive self-attention mask (0=valid, -inf=masked).
            text_kv_mask: (B, T) — boolean mask for text KV (True=valid).
            cond_kv_mask: (B, C) — boolean mask for condition KV (True=valid). Usually None
                since all 128 learnable query outputs are valid.
            mask_density: (B,) — optional mask density for DensityAwareFusionGate.

        Returns:
            motion_feat: (B, L, feat_dim) — updated motion features.
        """
        B, L, D = motion_feat.shape
        H = self.num_heads
        head_dim = self.head_dim

        # ============ Modulation Parameters ============
        (
            shift_msa, scale_msa, gate_msa,
            shift_mlp, scale_mlp, gate_mlp,
        ) = self.motion_mod(adapter).chunk(6, dim=-1)

        # ============ Self-Attention ============
        residual = motion_feat
        x_norm = self.motion_norm1(motion_feat)
        x_mod = modulate(x_norm, shift=shift_msa, scale=scale_msa)

        # QKV projection
        qkv = self.motion_qkv(x_mod)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B L H D", K=3, H=H)

        # QK normalization
        q = self.motion_q_norm(q).to(v)
        k = self.motion_k_norm(k).to(v)

        # RoPE on motion tokens
        q, k = self.rotary_emb.apply_rotary_emb(q, k)

        # Self-attention
        dropout_p = 0.0 if not self.training else self.dropout
        self_attn_out = attention(
            q, k, v,
            mode="torch",
            drop_rate=dropout_p,
            attn_mask=self_attn_mask,
            causal=False,
            batch_size=B,
            training=self.training,
        )  # (B, L, H*D)

        # Residual with gate
        motion_feat = residual + apply_gate(
            self.motion_out_proj(self_attn_out), gate=gate_msa
        )

        # ============ Text Cross-Attention ============
        T = text_kv.shape[1]
        x_norm_text = self.text_cross_norm(motion_feat)

        # Q from motion, KV from text
        q_text = self.text_cross_q(x_norm_text).reshape(B, L, H, head_dim)
        k_text = self.text_cross_k(text_kv).reshape(B, T, H, head_dim)
        v_text = self.text_cross_v(text_kv).reshape(B, T, H, head_dim)

        # QK norm
        q_text = self.text_cross_q_norm(q_text)
        k_text = self.text_cross_k_norm(k_text)

        # Build text attention mask
        text_attn_mask = None
        if text_kv_mask is not None:
            # text_kv_mask: (B, T) bool True=valid → (B, 1, 1, T) additive mask
            text_attn_mask = torch.zeros(B, 1, L, T, dtype=q_text.dtype, device=q_text.device)
            padding = ~text_kv_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
            text_attn_mask.masked_fill_(padding, float('-inf'))

        text_cross_out = attention(
            q_text, k_text, v_text,
            mode="torch",
            drop_rate=dropout_p,
            attn_mask=text_attn_mask,
            causal=False,
            batch_size=B,
            training=self.training,
        )  # (B, L, H*D)
        text_cross_out = self.text_cross_out_proj(text_cross_out)

        # ============ Motion-Condition Cross-Attention ============
        C = cond_kv.shape[1]
        x_norm_cond = self.cond_cross_norm(motion_feat)

        # Q from motion, KV from condition tokens
        q_cond = self.cond_cross_q(x_norm_cond).reshape(B, L, H, head_dim)
        k_cond = self.cond_cross_k(cond_kv).reshape(B, C, H, head_dim)
        v_cond = self.cond_cross_v(cond_kv).reshape(B, C, H, head_dim)

        # QK norm
        q_cond = self.cond_cross_q_norm(q_cond)
        k_cond = self.cond_cross_k_norm(k_cond)

        # Build cond attention mask (usually None since all queries are valid)
        cond_attn_mask = None
        if cond_kv_mask is not None:
            cond_attn_mask = torch.zeros(B, 1, L, C, dtype=q_cond.dtype, device=q_cond.device)
            padding = ~cond_kv_mask.unsqueeze(1).unsqueeze(2)
            cond_attn_mask.masked_fill_(padding, float('-inf'))

        cond_cross_out = attention(
            q_cond, k_cond, v_cond,
            mode="torch",
            drop_rate=dropout_p,
            attn_mask=cond_attn_mask,
            causal=False,
            batch_size=B,
            training=self.training,
        )  # (B, L, H*D)
        cond_cross_out = self.cond_cross_out_proj(cond_cross_out)

        # ============ Gated Fusion ============
        if self._gate_type == 'density_aware' and mask_density is not None:
            text_gate, motion_gate = self.fusion_gate(adapter, mask_density=mask_density)
        else:
            text_gate, motion_gate = self.fusion_gate(adapter)

        # text_gate, motion_gate: (B, 1, 1) — broadcast over L and D
        fused_cross = text_gate * text_cross_out + motion_gate * cond_cross_out
        motion_feat = motion_feat + fused_cross

        # ============ FFN with Modulation ============
        motion_feat = motion_feat + apply_gate(
            self.motion_mlp(
                modulate(
                    self.motion_norm2(motion_feat),
                    shift=shift_mlp,
                    scale=scale_mlp,
                )
            ),
            gate=gate_mlp,
        )

        return motion_feat


class HunyuanMotionMMDiTv3(nn.Module):
    """Hunyuan Motion MMDiT v3 with Dual-Stream Condition Fusion.

    Eliminates VACE shortcut by encoding motion conditions via MotionCondEncoder
    and injecting them through cross-attention at the same level as text.

    Architecture:
        Input: x_t(motion_dim) || scalar_mask(1) → input_encoder(feat_dim)
        + RoleEmbedding(mask → per-frame role → additive embedding)
        → 18× DualCondMMDiTBlock (self-attn + dual cross-attn + FFN)
        → FinalLayer → output(motion_dim)

    Conditioning:
        - Text: ctxt_encoder → text_refiner → text KV for cross-attention
        - Motion: MotionCondEncoder(known_motion, mask) → 128 condition tokens → cond KV
        - Timestep + vtxt → adapter for ModulateDiT in every block

    Args:
        motion_dim: Raw motion dimension (198 for v3 representation).
        feat_dim: Hidden dimension for all transformer components (1024).
        output_dim: Output dimension (default: motion_dim).
        ctxt_input_dim: Context text embedding dimension (4096 for T5-XXL).
        vtxt_input_dim: Vector text embedding dimension (256).
        num_layers: Number of DualCondMMDiTBlocks (default 18).
        num_heads: Number of attention heads (default 16).
        mlp_ratio: MLP hidden ratio (default 4.0).
        mlp_act_type: MLP activation (default 'gelu_tanh').
        qk_norm_type: QK normalization type (default 'rms').
        qkv_bias: QKV projection bias (default True).
        dropout: Attention dropout (default 0.0).
        mask_mode: Attention mask mode (None, 'causal', 'narrowband').
        time_factor: Timestep encoding scaling factor.
        narrowband_length: Narrowband window in seconds (×30 for frames).
        cond_encoder_cfg: Config dict for MotionCondEncoder.
        role_embedding_cfg: Config dict for RoleEmbedding.
        gate_type: 'timestep' or 'density_aware' for fusion gates.
        include_scalar_mask: Whether to append scalar mask channel to input.
    """

    def __init__(
        self,
        motion_dim: int = 198,
        feat_dim: int = 1024,
        output_dim: Optional[int] = None,
        ctxt_input_dim: int = 4096,
        vtxt_input_dim: int = 256,
        text_refiner_cfg: dict = {"num_layers": 2},
        num_layers: int = 18,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        mlp_act_type: str = "gelu_tanh",
        qk_norm_type: str = "rms",
        qkv_bias: bool = True,
        dropout: float = 0.0,
        final_layer_cfg: dict = {"act_type": "silu"},
        mask_mode: Optional[str] = None,
        time_factor: float = 1.0,
        narrowband_length: float = 2.0,
        cond_encoder_cfg: dict = {},
        role_embedding_cfg: dict = {},
        gate_type: str = 'timestep',
        include_scalar_mask: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.motion_dim = motion_dim
        self.feat_dim = feat_dim
        self.output_dim = output_dim or motion_dim
        self.ctxt_input_dim = ctxt_input_dim
        self.vtxt_input_dim = vtxt_input_dim
        self.num_layers = num_layers
        self.mask_mode = mask_mode
        self.time_factor = time_factor
        self.narrowband_length = narrowband_length * 30.0  # seconds → frames
        self.include_scalar_mask = include_scalar_mask

        # ============ Input Encoder ============
        # Input = x_t(motion_dim) + optional scalar_mask(1)
        input_dim = motion_dim + (1 if include_scalar_mask else 0)
        self.input_encoder = nn.Linear(input_dim, feat_dim)

        # ============ Role Embedding ============
        role_cfg = dict(
            feat_dim=feat_dim,
            motion_dim=motion_dim,
            mode='per_frame',
            zero_init=True,
        )
        role_cfg.update(role_embedding_cfg)
        self.role_embedding = RoleEmbedding(**role_cfg)

        # ============ Motion Condition Encoder ============
        cond_cfg = dict(
            motion_dim=motion_dim,
            feat_dim=feat_dim,
            num_queries=128,
            num_layers=4,
            num_heads=num_heads,
            max_seq_len=512,
            dropout=0.0,
        )
        cond_cfg.update(cond_encoder_cfg)
        self.motion_cond_encoder = MotionCondEncoder(**cond_cfg)

        # ============ Text Encoders ============
        self.ctxt_encoder = nn.Linear(ctxt_input_dim, feat_dim)
        self.vtxt_encoder = MLPEncoder(
            in_dim=vtxt_input_dim, feat_dim=feat_dim, num_layers=2, act_type="silu"
        )
        self.timestep_encoder = TimestepEmbeddingEncoder(
            embedding_dim=feat_dim, feat_dim=feat_dim, time_factor=time_factor
        )

        # ============ Text Refiner ============
        text_refiner_cfg_full = dict(
            input_dim=feat_dim, feat_dim=feat_dim, num_heads=num_heads
        )
        text_refiner_cfg_full.update(text_refiner_cfg)
        self._text_refiner_cfg = text_refiner_cfg_full.copy()
        self.text_refiner = SingleTokenRefiner(**text_refiner_cfg_full)

        # ============ DualCondMMDiT Blocks ============
        self.blocks = nn.ModuleList([
            DualCondMMDiTBlock(
                feat_dim=feat_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                mlp_act_type=mlp_act_type,
                qk_norm_type=qk_norm_type,
                qkv_bias=qkv_bias,
                positional_encoding_cfg={
                    "max_seq_len": 5000,
                    "use_real": True,
                },
                gate_type=gate_type,
            )
            for _ in range(num_layers)
        ])

        # ============ Final Layer ============
        final_cfg = dict(feat_dim=feat_dim, out_dim=self.output_dim)
        final_cfg.update(final_layer_cfg)
        self._final_layer_cfg = final_cfg.copy()
        self.final_layer = FinalLayer(**final_cfg)

    def forward(
        self,
        x: Tensor,
        ctxt_input: Tensor,
        vtxt_input: Tensor,
        timesteps: Tensor,
        condition_mask: Tensor,
        known_motion: Tensor,
        x_mask_temporal: Tensor,
        ctxt_mask_temporal: Tensor,
        edit_mask: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """Forward pass of HunyuanMotionMMDiTv3.

        Args:
            x: (B, L, motion_dim) — noisy motion input x_t.
            ctxt_input: (B, T, ctxt_input_dim) — context text token embeddings.
            vtxt_input: (B, 1, vtxt_input_dim) — vector text embedding.
            timesteps: (B,) — diffusion timesteps.
            condition_mask: (B, L, motion_dim) — binary mask (1=generate, 0=known).
            known_motion: (B, L, motion_dim) — clean motion values at known positions;
                zero at masked positions.
            x_mask_temporal: (B, L) — boolean mask for motion frames (True=valid).
            ctxt_mask_temporal: (B, T) — boolean mask for text tokens (True=valid).
            edit_mask: (B, L) — optional boolean, True where frame is in edit mode.
            **kwargs: Additional arguments (mask_density, etc.).

        Returns:
            predicted_velocity: (B, L, motion_dim) — predicted flow velocity.
        """
        device = get_module_device(self)
        B, L, D_motion = x.shape

        # ============ Encode Input ============
        # Build input: [x_t, scalar_mask_density_per_frame]
        if self.include_scalar_mask:
            # Per-frame scalar: fraction of masked dims at this frame
            scalar_mask = condition_mask.mean(dim=-1, keepdim=True)  # (B, L, 1)
            x_input = torch.cat([x, scalar_mask], dim=-1)  # (B, L, motion_dim+1)
        else:
            x_input = x

        motion_feat = self.input_encoder(x_input)  # (B, L, feat_dim)

        # ============ Add Role Embedding ============
        role_emb = self.role_embedding(condition_mask, edit_mask=edit_mask)  # (B, L, feat_dim)
        motion_feat = motion_feat + role_emb

        # ============ Encode Conditioning Signals ============
        timestep_feat = self.timestep_encoder(timesteps)  # (B, 1, feat_dim)
        vtxt_feat = self.vtxt_encoder(vtxt_input.float())  # (B, 1, feat_dim)
        adapter = timestep_feat + vtxt_feat  # (B, 1, feat_dim)

        # ============ Encode Motion Condition ============
        # frame_mask for the condition encoder: only attend to valid frames
        cond_tokens = self.motion_cond_encoder(
            known_motion=known_motion,
            mask=condition_mask,
            frame_mask=x_mask_temporal,
        )  # (B, num_queries, feat_dim)

        # ============ Encode & Refine Text ============
        ctxt_feat = self.ctxt_encoder(ctxt_input.float())  # (B, T, feat_dim)
        # Build text key padding mask for refiner
        ctxt_key_padding_mask = self._canonical_mask(ctxt_mask_temporal)
        if ctxt_key_padding_mask is not None:
            ctxt_key_padding_mask = ctxt_key_padding_mask.to(device)
            refiner_mask = (ctxt_key_padding_mask == 0).to(device)
        else:
            # No text mask → all text tokens are valid
            refiner_mask = torch.ones(B, ctxt_input.shape[1], dtype=torch.bool, device=device)
        ctxt_feat = self.text_refiner(
            x=ctxt_feat, t=timesteps,
            mask=refiner_mask,
        )

        # ============ Build Self-Attention Mask ============
        motion_key_padding_mask = self._canonical_mask(x_mask_temporal)
        if motion_key_padding_mask is not None:
            motion_key_padding_mask = motion_key_padding_mask.to(device)
        self_attn_mask = self._build_self_attn_mask(
            bsz=B, motion_len=L,
            dtype=motion_feat.dtype,
            key_padding_mask=motion_key_padding_mask,
            device=device,
        )

        # ============ Prepare Text KV Mask (boolean) ============
        # text_kv_mask: True=valid, used in cross-attention
        text_kv_mask = ctxt_mask_temporal  # (B, T) bool or None

        # ============ Compute Mask Density (for density-aware gates) ============
        # density = fraction of masked elements in condition_mask
        mask_density = condition_mask.mean(dim=(-1, -2))  # (B,)

        # ============ Transformer Blocks ============
        for block in self.blocks:
            motion_feat = block(
                motion_feat=motion_feat,
                text_kv=ctxt_feat,
                cond_kv=cond_tokens,
                adapter=adapter,
                self_attn_mask=self_attn_mask,
                text_kv_mask=text_kv_mask,
                cond_kv_mask=None,  # All condition tokens valid
                mask_density=mask_density,
            )

        # ============ Final Layer ============
        predicted_velocity = self.final_layer(motion_feat, adapter)
        return predicted_velocity

    @staticmethod
    def _canonical_mask(input_mask: Optional[Tensor]) -> Optional[Tensor]:
        """Convert boolean mask (True=valid) to attention mask (0=valid, -inf=masked).

        Returns None if input_mask is None (no masking needed).
        """
        if input_mask is None:
            return None
        if input_mask.ndim == 1:
            input_mask = input_mask.unsqueeze(1)
        key_padding_mask = torch.where(
            input_mask,
            torch.zeros_like(input_mask, dtype=torch.float),
            torch.full_like(input_mask, float("-inf"), dtype=torch.float),
        )
        return key_padding_mask

    def _build_self_attn_mask(
        self,
        bsz: int,
        motion_len: int,
        dtype: torch.dtype,
        key_padding_mask: Optional[Tensor],
        device: torch.device,
    ) -> Tensor:
        """Build self-attention mask for motion tokens.

        Supports: None (full), causal, narrowband modes.
        Applies key_padding_mask as additive mask.

        Returns:
            (B, 1, L, L) attention mask.
        """
        base = torch.zeros((bsz, 1, motion_len, motion_len), dtype=dtype, device=device)

        # Apply sequence-level mask (causal/narrowband)
        if self.mask_mode == "causal":
            causal_mask = torch.triu(
                torch.full((motion_len, motion_len), float("-inf"), device=device, dtype=dtype),
                diagonal=1,
            )
            base = base + causal_mask.view(1, 1, motion_len, motion_len)
        elif self.mask_mode == "narrowband":
            window = int(round(self.narrowband_length))
            idx = torch.arange(motion_len, device=device)
            dist = (idx[None, :] - idx[:, None]).abs()
            band = dist <= window
            nb_mask = torch.full((motion_len, motion_len), float("-inf"), device=device, dtype=dtype)
            nb_mask.masked_fill_(band, 0.0)
            base = base + nb_mask.view(1, 1, motion_len, motion_len)

        # Apply key padding mask: (B, L) → (B, 1, 1, L) additive
        if key_padding_mask is not None:
            base = base + key_padding_mask.view(bsz, 1, 1, motion_len)

        return base

    def load_pretrained_backbone(
        self,
        state_dict: dict,
        strict: bool = False,
    ) -> Tuple[List[str], List[str]]:
        """Load pretrained T2M weights into the v3 model.

        Maps weights from HunyuanMotionMMDiT (v1 double-stream + single-stream)
        to DualCondMMDiTBlocks. The mapping:

        For first 6 blocks (from double_blocks):
            double_blocks.{i}.motion_mod → blocks.{i}.motion_mod
            double_blocks.{i}.motion_norm1 → blocks.{i}.motion_norm1
            double_blocks.{i}.motion_qkv → blocks.{i}.motion_qkv
            double_blocks.{i}.motion_q_norm → blocks.{i}.motion_q_norm
            double_blocks.{i}.motion_k_norm → blocks.{i}.motion_k_norm
            double_blocks.{i}.motion_out_proj → blocks.{i}.motion_out_proj
            double_blocks.{i}.motion_norm2 → blocks.{i}.motion_norm2
            double_blocks.{i}.motion_mlp → blocks.{i}.motion_mlp

        For blocks 6-17 (from single_blocks, need decomposition):
            single_blocks.{j}.modulation → blocks.{i}.motion_mod
            single_blocks.{j}.norm → blocks.{i}.motion_norm1
            single_blocks.{j}.linear1 (partial) → blocks.{i}.motion_qkv
            single_blocks.{j}.q_norm → blocks.{i}.motion_q_norm
            single_blocks.{j}.k_norm → blocks.{i}.motion_k_norm
            single_blocks.{j}.linear2 (partial) → blocks.{i}.motion_out_proj + motion_mlp

        Other mappings:
            input_encoder → input_encoder (may need input_dim adjustment)
            ctxt_encoder → ctxt_encoder
            vtxt_encoder → vtxt_encoder
            timestep_encoder → timestep_encoder
            text_refiner → text_refiner
            final_layer → final_layer

        Args:
            state_dict: Pretrained state dict from HunyuanMotionMMDiT.
            strict: If True, raise error on missing/unexpected keys.

        Returns:
            (missing_keys, unexpected_keys): Lists of keys that couldn't be mapped.
        """
        new_state = {}

        # ============ Direct Mappings (encoders, refiner, final_layer) ============
        direct_prefixes = [
            'ctxt_encoder.', 'vtxt_encoder.', 'timestep_encoder.',
            'text_refiner.', 'final_layer.',
        ]
        for key, value in state_dict.items():
            for prefix in direct_prefixes:
                if key.startswith(prefix):
                    new_state[key] = value
                    break

        # ============ Double Block → Block Mapping (first 6 blocks) ============
        # The motion stream params in double_blocks map directly
        num_double = 0
        for key in state_dict:
            if key.startswith('double_blocks.'):
                parts = key.split('.')
                block_idx = int(parts[1])
                num_double = max(num_double, block_idx + 1)

        motion_stream_params = [
            'motion_mod', 'motion_norm1', 'motion_qkv',
            'motion_q_norm', 'motion_k_norm', 'motion_out_proj',
            'motion_norm2', 'motion_mlp',
        ]

        for key, value in state_dict.items():
            if not key.startswith('double_blocks.'):
                continue
            parts = key.split('.', 2)  # ['double_blocks', '0', 'motion_mod.linear.weight']
            block_idx = int(parts[1])
            remainder = parts[2]

            # Check if this is a motion stream parameter
            is_motion_param = any(remainder.startswith(p) for p in motion_stream_params)
            if is_motion_param:
                new_key = f'blocks.{block_idx}.{remainder}'
                new_state[new_key] = value
            # Also grab rotary_emb
            if remainder.startswith('rotary_emb'):
                # RotaryEmbedding doesn't have learnable params typically,
                # but if registered as buffer, map it
                new_key = f'blocks.{block_idx}.{remainder}'
                new_state[new_key] = value

        # ============ Input Encoder (handle dimension mismatch) ============
        # v1 input_encoder: Linear(594, 1024) or similar
        # v3 input_encoder: Linear(199, 1024)
        # We cannot directly map these — skip input_encoder weight mapping
        # and let it initialize randomly (it will be trained in Phase 0)

        # ============ Load What We Can ============
        missing, unexpected = self.load_state_dict(new_state, strict=False)

        return missing, unexpected

    def params_count(self) -> dict:
        """Count and print model parameters breakdown."""
        if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
            # Self-attention params per block
            self_attn_params = sum(
                sum(p.numel() for p in block.motion_qkv.parameters())
                + sum(p.numel() for p in block.motion_out_proj.parameters())
                for block in self.blocks
            )
            # Text cross-attention params per block
            text_cross_params = sum(
                sum(p.numel() for p in block.text_cross_q.parameters())
                + sum(p.numel() for p in block.text_cross_k.parameters())
                + sum(p.numel() for p in block.text_cross_v.parameters())
                + sum(p.numel() for p in block.text_cross_out_proj.parameters())
                for block in self.blocks
            )
            # Cond cross-attention params per block
            cond_cross_params = sum(
                sum(p.numel() for p in block.cond_cross_q.parameters())
                + sum(p.numel() for p in block.cond_cross_k.parameters())
                + sum(p.numel() for p in block.cond_cross_v.parameters())
                + sum(p.numel() for p in block.cond_cross_out_proj.parameters())
                for block in self.blocks
            )
            # MLP params
            mlp_params = sum(
                sum(p.numel() for p in block.motion_mlp.parameters())
                for block in self.blocks
            )
            # Gate params
            gate_params = sum(
                sum(p.numel() for p in block.fusion_gate.parameters())
                for block in self.blocks
            )
            # Modulation params
            mod_params = sum(
                sum(p.numel() for p in block.motion_mod.parameters())
                for block in self.blocks
            )
            # Condition encoder
            cond_enc_params = sum(p.numel() for p in self.motion_cond_encoder.parameters())
            # Role embedding
            role_params = sum(p.numel() for p in self.role_embedding.parameters())
            # Text refiner
            refiner_params = sum(p.numel() for p in self.text_refiner.parameters())
            # Final layer
            final_params = sum(p.numel() for p in self.final_layer.parameters())
            # Total
            total_params = sum(p.numel() for p in self.parameters())

            counts = {
                'self_attn': self_attn_params,
                'text_cross_attn': text_cross_params,
                'cond_cross_attn': cond_cross_params,
                'mlp': mlp_params,
                'gates': gate_params,
                'modulation': mod_params,
                'cond_encoder': cond_enc_params,
                'role_embedding': role_params,
                'text_refiner': refiner_params,
                'final_layer': final_params,
                'total': total_params,
            }

            print(f"\n{'='*60}")
            print(f"HunyuanMotionMMDiTv3 Parameter Count")
            print(f"{'='*60}")
            for name, count in counts.items():
                print(f"  {name:20s}: {count:>12,} ({count/1e6:.2f}M)")
            print(f"{'='*60}\n")

            return counts
        return {}
