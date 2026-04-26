"""WanVACE-style ControlNet branch for PrismTransformerMotionModel.

Implements a sparse control branch following the WanVACE architecture:
- Only creates a subset of transformer blocks at evenly-spaced layer indices
- Each VACE block has a proj_out (all blocks) and optionally proj_in (first block)
- Audio features are projected and added to the control branch's hidden states
- Zero-initialized proj_out ensures zero initial contribution (like bridge modules)
- Uses U-skip pattern: VACE block outputs are reversed before injection
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.normalization import FP32LayerNorm
from diffusers.models.transformers.transformer_wan import (
    WanAttention,
    WanAttnProcessor,
    FeedForward,
)
from diffusers.utils.torch_utils import maybe_allow_in_graph

from hftrainer.models.motion.prism.network.embedding import (
    WanTimeTextEmbedding,
)
from hftrainer.models.motion.prism.network.motion_rope import (
    MotionWanRotaryPosEmbed,
)
from hftrainer.registry import HF_MODELS


@maybe_allow_in_graph
class PrismVACEControlBlock(nn.Module):
    """A single VACE-style control block with mask support.

    Mirrors :class:`WanTransformerBlockWithMask` but adds:
    - ``proj_in`` (first block only): projects control hidden states and adds
      the main branch hidden states as a residual.
    - ``proj_out`` (all blocks): projects output to produce conditioning states
      for injection into the main branch. Zero-initialized for training stability.

    Args:
        dim: Hidden dimension.
        ffn_dim: Feed-forward inner dimension.
        num_heads: Number of attention heads.
        qk_norm: Query-key normalization type.
        cross_attn_norm: Whether to use LayerNorm before cross-attention.
        eps: LayerNorm epsilon.
        added_kv_proj_dim: Dimension for additional KV projections (for I2V).
        apply_input_projection: Whether to add proj_in (only first block).
        apply_output_projection: Whether to add proj_out (all blocks).
    """

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = 'rms_norm_across_heads',
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: Optional[int] = None,
        apply_input_projection: bool = False,
        apply_output_projection: bool = False,
    ):
        super().__init__()

        # Input projection (first block only)
        self.proj_in = nn.Linear(dim, dim) if apply_input_projection else None

        # Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            cross_attention_dim_head=None,
            processor=WanAttnProcessor(),
        )

        # Cross-attention
        self.attn2 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            added_kv_proj_dim=added_kv_proj_dim,
            cross_attention_dim_head=dim // num_heads,
            processor=WanAttnProcessor(),
        )
        self.norm2 = (
            FP32LayerNorm(dim, eps, elementwise_affine=True)
            if cross_attn_norm
            else nn.Identity()
        )

        # Feed-forward
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn='gelu-approximate')

        # Output projection (zero-initialized for training stability)
        self.proj_out = None
        if apply_output_projection:
            self.proj_out = nn.Linear(dim, dim)
            nn.init.zeros_(self.proj_out.weight)
            nn.init.zeros_(self.proj_out.bias)

        # Adaptive modulation parameters (independent per block)
        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        control_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]],
        hidden_states_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states_mask: Optional[torch.Tensor] = None,
        causal_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """Forward pass.

        Args:
            hidden_states: Main branch hidden states ``[B, N, D]`` (read-only).
            encoder_hidden_states: Text embeddings ``[B, N_text, D]``.
            control_hidden_states: Control branch's running hidden states ``[B, N, D]``.
            temb: Timestep projection ``[B, 6, D]`` or ``[B, N, 6, D]``.
            rotary_emb: Rotary position embeddings.
            hidden_states_mask: Self-attention mask ``[B, 1, 1, N]``.
            encoder_hidden_states_mask: Cross-attention mask ``[B, 1, 1, N_text]``.
            causal_mask: Causal attention mask ``[1, 1, N, N]``.

        Returns:
            (conditioning_states, control_hidden_states):
            - conditioning_states: ``[B, N, D]`` output for injection into main branch
              (None if proj_out is absent).
            - control_hidden_states: Updated control hidden states for next VACE block.
        """
        # Input projection (first block only): inject main branch features
        if self.proj_in is not None:
            control_hidden_states = self.proj_in(control_hidden_states)
            control_hidden_states = control_hidden_states + hidden_states

        # Compute adaptive modulation parameters
        if temb.ndim == 4:
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table.unsqueeze(0) + temb.float()
            ).chunk(6, dim=2)
            shift_msa = shift_msa.squeeze(2)
            scale_msa = scale_msa.squeeze(2)
            gate_msa = gate_msa.squeeze(2)
            c_shift_msa = c_shift_msa.squeeze(2)
            c_scale_msa = c_scale_msa.squeeze(2)
            c_gate_msa = c_gate_msa.squeeze(2)
        else:
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table + temb.float()
            ).chunk(6, dim=1)

        # 1. Self-attention
        norm_hs = (
            self.norm1(control_hidden_states.float()) * (1 + scale_msa) + shift_msa
        ).type_as(control_hidden_states)

        combined_self_attn_mask = hidden_states_mask
        if causal_mask is not None:
            if combined_self_attn_mask is not None:
                combined_self_attn_mask = combined_self_attn_mask + causal_mask
            else:
                combined_self_attn_mask = causal_mask

        attn_output = self.attn1(
            hidden_states=norm_hs,
            encoder_hidden_states=None,
            attention_mask=combined_self_attn_mask,
            rotary_emb=rotary_emb,
        )
        control_hidden_states = (
            control_hidden_states.float() + attn_output * gate_msa
        ).type_as(control_hidden_states)

        # 2. Cross-attention
        norm_hs = self.norm2(control_hidden_states.float()).type_as(control_hidden_states)
        attn_output = self.attn2(
            hidden_states=norm_hs,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=encoder_hidden_states_mask,
            rotary_emb=None,
        )
        control_hidden_states = control_hidden_states + attn_output

        # 3. Feed-forward
        norm_hs = (
            self.norm3(control_hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa
        ).type_as(control_hidden_states)
        ff_output = self.ffn(norm_hs)
        control_hidden_states = (
            control_hidden_states.float() + ff_output.float() * c_gate_msa
        ).type_as(control_hidden_states)

        # Output projection
        conditioning_states = None
        if self.proj_out is not None:
            conditioning_states = self.proj_out(control_hidden_states)

        return conditioning_states, control_hidden_states


@HF_MODELS.register_module()
class PrismVACEControlTransformer(nn.Module):
    """WanVACE-style sparse control branch for PrismTransformerMotionModel.

    Instead of duplicating ALL main-branch transformer blocks (as in MCM),
    this creates only a small subset at evenly-spaced layer indices, following
    the WanVACE architecture. This reduces trainable parameters by ~72%.

    Architecture:
        - ``len(vace_layers)`` VACE blocks (default 8) vs ``num_layers`` main blocks
        - Each VACE block has its own self-attention, cross-attention, FFN
        - First VACE block has ``proj_in`` to inject main branch features
        - All VACE blocks have ``proj_out`` (zero-initialized) for injection
        - Audio features are added to the control branch's patch-embedded tokens
        - U-skip pattern: VACE outputs are reversed before injection into main branch

    Args:
        patch_size: Temporal and joint patch sizes ``(p_t, p_j)``.
        num_attention_heads: Number of attention heads.
        attention_head_dim: Dimension per head.
        in_channels: VAE latent channels.
        out_channels: Output channels (kept for config parity).
        text_dim: Text encoder hidden dim.
        freq_dim: Timestep frequency dim.
        ffn_dim: Feed-forward inner dim.
        num_layers: Number of main-branch blocks (used for vace_layers default).
        vace_layers: Indices of main-branch blocks to inject at.
            Default: one per ~4 layers, evenly spaced.
        audio_feature_dim: Dimension of incoming audio features.
        cross_attn_norm: Whether to use LayerNorm before cross-attention.
        qk_norm: Query-key normalisation type.
        eps: LayerNorm epsilon.
        rope_max_seq_len: Max sequence length for RoPE.
    """

    def __init__(
        self,
        patch_size: Tuple[int, ...] = (1, 1),
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: int = 16,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        vace_layers: Optional[List[int]] = None,
        audio_feature_dim: int = 768,
        cross_attn_norm: bool = True,
        qk_norm: Optional[str] = 'rms_norm_across_heads',
        eps: float = 1e-6,
        added_kv_proj_dim: Optional[int] = None,
        rope_max_seq_len: int = 1024,
        pos_embed_seq_len: Optional[int] = None,
    ):
        super().__init__()

        assert patch_size[-1] == 1, 'Joint patchification is not supported'

        inner_dim = num_attention_heads * attention_head_dim
        self.inner_dim = inner_dim
        self.num_layers = num_layers
        self.patch_size = patch_size

        # Default vace_layers: one per ~4 layers, evenly spaced
        if vace_layers is None:
            if num_layers <= 4:
                vace_layers = list(range(num_layers))
            else:
                step = max(num_layers // 8, 1)
                vace_layers = list(range(0, num_layers, step))
        self.vace_layers = vace_layers
        num_vace_blocks = len(vace_layers)

        # Shared embeddings (will be copied from main branch)
        self.rope = MotionWanRotaryPosEmbed(
            attention_head_dim, patch_size, rope_max_seq_len,
        )
        self.patch_embedding = nn.Conv2d(
            in_channels, inner_dim, kernel_size=patch_size, stride=patch_size,
        )
        self.condition_embedder = WanTimeTextEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
            pos_embed_seq_len=pos_embed_seq_len,
        )

        # VACE blocks (sparse subset)
        self.vace_blocks = nn.ModuleList([
            PrismVACEControlBlock(
                inner_dim, ffn_dim, num_attention_heads,
                qk_norm, cross_attn_norm, eps, added_kv_proj_dim,
                apply_input_projection=(i == 0),
                apply_output_projection=True,
            )
            for i in range(num_vace_blocks)
        ])

        # Audio projection MLP
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_feature_dim, inner_dim),
            nn.SiLU(),
            nn.Linear(inner_dim, inner_dim),
        )

        self.gradient_checkpointing = True

    def enable_gradient_checkpointing(self, value: bool = True) -> None:
        self.gradient_checkpointing = value

    @classmethod
    def init_from_main_branch(
        cls,
        instance: 'PrismVACEControlTransformer',
        main_transformer: nn.Module,
    ) -> None:
        """Copy shared components from the main (frozen) transformer.

        Copies rope, patch_embedding, condition_embedder from the main branch.
        For VACE blocks, copies weights from the corresponding main-branch blocks
        (at the indices specified by vace_layers), but only the shared components
        (norm, attention, ffn, scale_shift_table).

        proj_in, proj_out, and audio_proj retain their initial values.

        Shape mismatches are skipped with a warning instead of raising, so
        that this method remains safe to call after FSDP wrapping or when
        main/control architectures differ slightly.
        """
        import logging
        _logger = logging.getLogger(__name__)

        def _safe_copy(target: nn.Module, source: nn.Module, name: str):
            """Copy state_dict from source to target, skipping shape mismatches."""
            src_sd = source.state_dict()
            tgt_sd = target.state_dict()
            filtered = {}
            skipped = []
            for k, v in src_sd.items():
                if k in tgt_sd and v.shape == tgt_sd[k].shape:
                    filtered[k] = v
                elif k in tgt_sd:
                    skipped.append(
                        f"{k}: src {tuple(v.shape)} vs tgt {tuple(tgt_sd[k].shape)}"
                    )
                # keys not in target are silently ignored
            if skipped:
                _logger.warning(
                    "init_from_main_branch: skipped %d shape-mismatched "
                    "params in '%s': %s",
                    len(skipped), name, skipped[:5],
                )
            if filtered:
                target.load_state_dict(filtered, strict=False)

        # Copy shared embeddings
        _safe_copy(instance.rope, main_transformer.rope, 'rope')
        _safe_copy(
            instance.patch_embedding,
            main_transformer.patch_embedding,
            'patch_embedding',
        )
        _safe_copy(
            instance.condition_embedder,
            main_transformer.condition_embedder,
            'condition_embedder',
        )

        # Copy VACE block weights from corresponding main blocks
        for vace_idx, main_layer_idx in enumerate(instance.vace_layers):
            if main_layer_idx >= len(main_transformer.blocks):
                continue
            main_block = main_transformer.blocks[main_layer_idx]
            vace_block = instance.vace_blocks[vace_idx]

            # Copy shared components: norm, attention, ffn, scale_shift_table
            _copy_block_weights(main_block, vace_block)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        audio_features: Optional[torch.Tensor] = None,
        hidden_states_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> List[Tuple[torch.Tensor, int]]:
        """Run VACE control branch.

        Args:
            hidden_states: ``[B, C, T, J]`` motion latent.
            timestep: ``[B]`` or ``[B, N]`` diffusion timesteps.
            encoder_hidden_states: ``[B, N_text, text_dim]`` text embeddings.
            audio_features: ``[B, N_audio, audio_feature_dim]`` (optional).
            hidden_states_mask: ``[B, T, J]`` padding mask (1=valid).
            encoder_hidden_states_mask: ``[B, N_text]`` text mask (1=valid).
            is_causal: Whether to apply causal attention.

        Returns:
            List of ``(conditioning_states, layer_idx)`` tuples, in U-skip
            reversed order (last VACE block's output first). Each
            ``conditioning_states`` is ``[B, N, inner_dim]``.
        """
        batch_size, _, num_frames, num_joints = hidden_states.shape
        p_t, p_j = self.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_num_joints = num_joints // p_j

        # 1. RoPE
        rotary_emb = self.rope(hidden_states)

        # 2. Patch embedding (for control branch input)
        control_hidden_states = self.patch_embedding(hidden_states)
        control_hidden_states = control_hidden_states.flatten(2).transpose(1, 2)

        # 3. Inject audio features via addition to control hidden states
        if audio_features is not None:
            audio_features = audio_features.to(dtype=control_hidden_states.dtype)
            audio_proj = self.audio_proj(audio_features)
            seq_len = control_hidden_states.shape[1]
            if audio_proj.shape[1] != seq_len:
                audio_proj = audio_proj.transpose(1, 2)
                audio_proj = F.interpolate(
                    audio_proj, size=seq_len, mode='linear', align_corners=False,
                )
                audio_proj = audio_proj.transpose(1, 2)
            control_hidden_states = control_hidden_states + audio_proj

        # 4. Also compute main branch patch embedding (read-only, for proj_in)
        # This is the main branch's hidden states that the first VACE block reads
        main_hidden_states = control_hidden_states.detach().clone()
        # Note: we use the control's patch_embedding output (initialized from main)
        # as the "main_hidden_states" input to proj_in. This matches WanVACE where
        # hidden_states is the main backbone's patch-embedded output.

        # 5. Process masks
        if hidden_states_mask is not None:
            hidden_states_mask = hidden_states_mask.reshape(
                batch_size, post_patch_num_frames, p_t,
                post_patch_num_joints, p_j,
            )
            hidden_states_mask = hidden_states_mask.amin(dim=(2, 4))
            hidden_states_mask = hidden_states_mask.flatten(1)
            hidden_states_mask = (
                (1.0 - hidden_states_mask.float())
                * torch.finfo(control_hidden_states.dtype).min
            ).unsqueeze(1).unsqueeze(2)

        if encoder_hidden_states_mask is not None:
            encoder_hidden_states_mask = (
                (1.0 - encoder_hidden_states_mask.float())
                * torch.finfo(control_hidden_states.dtype).min
            ).unsqueeze(1).unsqueeze(2)

        # 6. Causal mask
        causal_mask = None
        if is_causal:
            seq_len = control_hidden_states.shape[1]
            frame_idx = torch.arange(seq_len, device=control_hidden_states.device) // post_patch_num_joints
            causal_mask = (
                (frame_idx.unsqueeze(0) > frame_idx.unsqueeze(1))
                .to(control_hidden_states.dtype)
                * torch.finfo(control_hidden_states.dtype).min
            ).unsqueeze(0).unsqueeze(0)

        # 7. Timestep + text conditioning
        if timestep.ndim == 2:
            ts_seq_len = timestep.shape[1]
            timestep = timestep.flatten()
        else:
            ts_seq_len = None

        _temb, timestep_proj, encoder_hidden_states = self.condition_embedder(
            timestep, encoder_hidden_states, timestep_seq_len=ts_seq_len,
        )

        if ts_seq_len is not None:
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
        else:
            timestep_proj = timestep_proj.unflatten(1, (6, -1))

        # 8. Run VACE blocks sequentially, collect conditioning outputs
        conditioning_list: List[Tuple[torch.Tensor, int]] = []
        for i, vace_block in enumerate(self.vace_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                conditioning_states, control_hidden_states = (
                    torch.utils.checkpoint.checkpoint(
                        vace_block,
                        main_hidden_states,
                        encoder_hidden_states,
                        control_hidden_states,
                        timestep_proj,
                        rotary_emb,
                        hidden_states_mask,
                        encoder_hidden_states_mask,
                        causal_mask,
                        use_reentrant=False,
                    )
                )
            else:
                conditioning_states, control_hidden_states = vace_block(
                    hidden_states=main_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    control_hidden_states=control_hidden_states,
                    temb=timestep_proj,
                    rotary_emb=rotary_emb,
                    hidden_states_mask=hidden_states_mask,
                    encoder_hidden_states_mask=encoder_hidden_states_mask,
                    causal_mask=causal_mask,
                )

            conditioning_list.append((conditioning_states, self.vace_layers[i]))

        # U-skip: reverse the list so that the last VACE block's output
        # is injected first (at the highest layer index)
        conditioning_list = conditioning_list[::-1]

        return conditioning_list


def _copy_block_weights(
    main_block: nn.Module,
    vace_block: PrismVACEControlBlock,
) -> None:
    """Copy shared component weights from a main-branch block to a VACE block.

    Copies: norm1, attn1, norm2, attn2, norm3, ffn, scale_shift_table.
    Does NOT copy: proj_in, proj_out (they keep their initialization).

    Shape mismatches within any component are skipped with a warning.
    """
    import logging
    _logger = logging.getLogger(__name__)

    def _safe_load(target, source, comp_name):
        src_sd = source.state_dict()
        tgt_sd = target.state_dict()
        filtered = {}
        for k, v in src_sd.items():
            if k in tgt_sd and v.shape == tgt_sd[k].shape:
                filtered[k] = v
            elif k in tgt_sd:
                _logger.warning(
                    "_copy_block_weights: shape mismatch in %s.%s: "
                    "src %s vs tgt %s — skipped",
                    comp_name, k, tuple(v.shape), tuple(tgt_sd[k].shape),
                )
        if filtered:
            target.load_state_dict(filtered, strict=False)

    # Self-attention
    _safe_load(vace_block.norm1, main_block.norm1, 'norm1')
    _safe_load(vace_block.attn1, main_block.attn1, 'attn1')

    # Cross-attention
    _safe_load(vace_block.attn2, main_block.attn2, 'attn2')
    # norm2 may be nn.Identity in main block but LayerNorm in VACE block (or vice versa)
    if not isinstance(main_block.norm2, nn.Identity) and not isinstance(vace_block.norm2, nn.Identity):
        _safe_load(vace_block.norm2, main_block.norm2, 'norm2')

    # FFN
    _safe_load(vace_block.norm3, main_block.norm3, 'norm3')
    _safe_load(vace_block.ffn, main_block.ffn, 'ffn')

    # Scale-shift table
    if main_block.scale_shift_table.shape == vace_block.scale_shift_table.shape:
        vace_block.scale_shift_table.data.copy_(main_block.scale_shift_table.data)
    else:
        _logger.warning(
            "_copy_block_weights: scale_shift_table shape mismatch: "
            "src %s vs tgt %s — skipped",
            tuple(main_block.scale_shift_table.shape),
            tuple(vace_block.scale_shift_table.shape),
        )
