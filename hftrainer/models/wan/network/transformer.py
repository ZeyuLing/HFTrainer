"""Repository-local PyTorch implementation of the Wan 3D diffusion transformer."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from .common import (
    LocalWanModelMixin,
    Transformer3DModelOutput,
    WanConfig,
    make_sinusoidal_embedding,
)


class WanRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if elementwise_affine else None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.float().square().mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        if self.weight is not None:
            hidden_states = hidden_states * self.weight
        return hidden_states


def _rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
    paired = hidden_states.reshape(*hidden_states.shape[:-1], -1, 2)
    first, second = paired.unbind(dim=-1)
    return torch.stack((-second, first), dim=-1).flatten(-2)


def _apply_rotary(
    query: torch.Tensor,
    key: torch.Tensor,
    rotary_embedding: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    cos, sin = rotary_embedding
    cos = cos.to(device=query.device, dtype=query.dtype)
    sin = sin.to(device=query.device, dtype=query.dtype)
    cos = cos[None, None]
    sin = sin[None, None]
    query = query * cos + _rotate_half(query) * sin
    key = key * cos + _rotate_half(key) * sin
    return query, key


class WanRotaryPosEmbed(nn.Module):
    """Three-axis rotary frequencies matching Wan's T/H/W head split."""

    def __init__(
        self, attention_head_dim: int, max_seq_len: int, theta: float = 10000.0
    ):
        super().__init__()
        h_dim = w_dim = 2 * (attention_head_dim // 6)
        self.split_dims = (attention_head_dim - h_dim - w_dim, h_dim, w_dim)
        self.max_seq_len = int(max_seq_len)
        self.theta = float(theta)

    def _axis(self, length: int, dim: int, device: torch.device):
        if dim == 0:
            empty = torch.empty(length, 0, device=device)
            return empty, empty
        frequencies = torch.exp(
            -math.log(self.theta)
            * torch.arange(0, dim, 2, device=device, dtype=torch.float32)
            / dim
        )
        angles = (
            torch.arange(length, device=device, dtype=torch.float32)[:, None]
            * frequencies[None]
        )
        return (
            torch.repeat_interleave(torch.cos(angles), 2, dim=-1),
            torch.repeat_interleave(torch.sin(angles), 2, dim=-1),
        )

    def forward(
        self, grid: tuple[int, int, int], device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        frames, height, width = grid
        if max(grid) > self.max_seq_len:
            raise ValueError(
                f"Patch grid {grid} exceeds rope_max_seq_len={self.max_seq_len}"
            )
        axis_values = [
            self._axis(length, dim, device)
            for length, dim in zip(grid, self.split_dims, strict=True)
        ]
        cos_t, sin_t = axis_values[0]
        cos_h, sin_h = axis_values[1]
        cos_w, sin_w = axis_values[2]
        cos = torch.cat(
            (
                cos_t[:, None, None].expand(frames, height, width, -1),
                cos_h[None, :, None].expand(frames, height, width, -1),
                cos_w[None, None, :].expand(frames, height, width, -1),
            ),
            dim=-1,
        ).reshape(-1, sum(self.split_dims))
        sin = torch.cat(
            (
                sin_t[:, None, None].expand(frames, height, width, -1),
                sin_h[None, :, None].expand(frames, height, width, -1),
                sin_w[None, None, :].expand(frames, height, width, -1),
            ),
            dim=-1,
        ).reshape(-1, sum(self.split_dims))
        return cos, sin


class WanAttention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        heads: int,
        dim_head: int,
        cross_attention_dim: int | None = None,
        eps: float = 1e-6,
        qk_norm: str | None = "rms_norm_across_heads",
    ):
        super().__init__()
        cross_attention_dim = cross_attention_dim or query_dim
        inner_dim = heads * dim_head
        self.heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(query_dim, inner_dim, bias=True)
        self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias=True)
        self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias=True)
        self.to_out = nn.ModuleList(
            [nn.Linear(inner_dim, query_dim, bias=True), nn.Dropout(0.0)]
        )
        use_qk_norm = qk_norm not in (None, "none", False)
        self.norm_q = WanRMSNorm(inner_dim, eps=eps) if use_qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(inner_dim, eps=eps) if use_qk_norm else nn.Identity()

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        rotary_embedding: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        context = (
            hidden_states if encoder_hidden_states is None else encoder_hidden_states
        )
        batch, query_length, _ = hidden_states.shape
        key_length = context.shape[1]
        query = (
            self.norm_q(self.to_q(hidden_states))
            .view(batch, query_length, self.heads, self.dim_head)
            .transpose(1, 2)
        )
        key = (
            self.norm_k(self.to_k(context))
            .view(batch, key_length, self.heads, self.dim_head)
            .transpose(1, 2)
        )
        value = (
            self.to_v(context)
            .view(batch, key_length, self.heads, self.dim_head)
            .transpose(1, 2)
        )
        if rotary_embedding is not None and encoder_hidden_states is None:
            query, key = _apply_rotary(query, key, rotary_embedding)

        additive_mask = None
        if attention_mask is not None:
            mask = attention_mask[:, None, None, :].to(torch.bool)
            additive_mask = torch.zeros(
                mask.shape, device=query.device, dtype=query.dtype
            ).masked_fill(~mask, torch.finfo(query.dtype).min)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=additive_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).reshape(batch, query_length, -1)
        return self.to_out[1](self.to_out[0](attended))


class WanGELU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.proj(hidden_states), approximate="tanh")


class WanFeedForward(nn.Module):
    def __init__(self, dim: int, inner_dim: int):
        super().__init__()
        self.net = nn.ModuleList(
            [WanGELU(dim, inner_dim), nn.Dropout(0.0), nn.Linear(inner_dim, dim)]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.net:
            hidden_states = layer(hidden_states)
        return hidden_states


class WanTimestepEmbedding(nn.Module):
    def __init__(self, freq_dim: int, dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(freq_dim, dim)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(dim, dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.act(self.linear_1(hidden_states)))


class WanTextProjection(nn.Module):
    def __init__(self, text_dim: int, dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(text_dim, dim)
        self.act = nn.GELU(approximate="tanh")
        self.linear_2 = nn.Linear(dim, dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.act(self.linear_1(hidden_states)))


class WanTimeTextConditioning(nn.Module):
    def __init__(self, freq_dim: int, dim: int, text_dim: int):
        super().__init__()
        self.freq_dim = freq_dim
        self.time_embedder = WanTimestepEmbedding(freq_dim, dim)
        self.time_proj = nn.Linear(dim, dim * 6)
        self.text_embedder = WanTextProjection(text_dim, dim)

    def forward(
        self,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        time_features = make_sinusoidal_embedding(timestep, self.freq_dim).to(
            dtype=dtype
        )
        temb = self.time_embedder(time_features)
        return (
            temb,
            self.time_proj(F.silu(temb)),
            self.text_embedder(encoder_hidden_states),
        )


class WanTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        eps: float,
        qk_norm: str | None,
        cross_attn_norm: bool,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.attn1 = WanAttention(
            dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            eps=eps,
            qk_norm=qk_norm,
        )
        self.norm2 = (
            nn.LayerNorm(dim, eps=eps, elementwise_affine=True)
            if cross_attn_norm
            else nn.Identity()
        )
        self.attn2 = WanAttention(
            dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            cross_attention_dim=dim,
            eps=eps,
            qk_norm=qk_norm,
        )
        self.norm3 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.ffn = WanFeedForward(dim, ffn_dim)
        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_embedding: tuple[torch.Tensor, torch.Tensor],
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        modulation = self.scale_shift_table + temb.reshape(
            hidden_states.shape[0], 6, hidden_states.shape[-1]
        )
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            modulation.unbind(dim=1)
        )
        normalized = self.norm1(hidden_states.float()).to(hidden_states.dtype)
        normalized = normalized * (1 + scale_msa[:, None]) + shift_msa[:, None]
        hidden_states = hidden_states + gate_msa[:, None] * self.attn1(
            normalized, rotary_embedding=rotary_embedding
        )
        hidden_states = hidden_states + self.attn2(
            self.norm2(hidden_states.float()).to(hidden_states.dtype),
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
        )
        normalized = self.norm3(hidden_states.float()).to(hidden_states.dtype)
        normalized = normalized * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        return hidden_states + gate_mlp[:, None] * self.ffn(normalized)


class WanTransformer3DModel(LocalWanModelMixin, nn.Module):
    """Patchified video flow transformer with Wan-compatible public fields."""

    component_name = "transformer"

    def __init__(
        self,
        patch_size: Sequence[int] = (1, 2, 2),
        num_attention_heads: int = 12,
        attention_head_dim: int = 128,
        in_channels: int = 16,
        out_channels: int | None = None,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 8960,
        num_layers: int = 30,
        cross_attn_norm: bool = True,
        qk_norm: str | None = "rms_norm_across_heads",
        eps: float = 1e-6,
        image_dim: int | None = None,
        added_kv_proj_dim: int | None = None,
        rope_max_seq_len: int = 1024,
        pos_embed_seq_len: int | None = None,
        **kwargs,
    ):
        super().__init__()
        if len(patch_size) != 3:
            raise ValueError(
                "patch_size must contain temporal, height, and width values"
            )
        patch_size = tuple(int(value) for value in patch_size)
        if any(value <= 0 for value in patch_size):
            raise ValueError("patch_size values must be positive")
        out_channels = int(out_channels or in_channels)
        inner_dim = int(num_attention_heads) * int(attention_head_dim)
        if inner_dim <= 0 or num_layers <= 0:
            raise ValueError("Transformer dimensions and layer counts must be positive")
        self.config = WanConfig(
            patch_size=list(patch_size),
            num_attention_heads=int(num_attention_heads),
            attention_head_dim=int(attention_head_dim),
            in_channels=int(in_channels),
            out_channels=out_channels,
            text_dim=int(text_dim),
            freq_dim=int(freq_dim),
            ffn_dim=int(ffn_dim),
            num_layers=int(num_layers),
            cross_attn_norm=bool(cross_attn_norm),
            qk_norm=qk_norm,
            eps=float(eps),
            image_dim=image_dim,
            added_kv_proj_dim=added_kv_proj_dim,
            rope_max_seq_len=int(rope_max_seq_len),
            pos_embed_seq_len=pos_embed_seq_len,
            **kwargs,
        )
        self.inner_dim = inner_dim
        self.rope = WanRotaryPosEmbed(attention_head_dim, rope_max_seq_len)
        self.patch_embedding = nn.Conv3d(
            in_channels,
            inner_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.condition_embedder = WanTimeTextConditioning(freq_dim, inner_dim, text_dim)
        self.blocks = nn.ModuleList(
            [
                WanTransformerBlock(
                    inner_dim,
                    ffn_dim,
                    num_attention_heads,
                    attention_head_dim,
                    eps,
                    qk_norm,
                    cross_attn_norm,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_out = nn.LayerNorm(inner_dim, eps=eps, elementwise_affine=False)
        self.scale_shift_table = nn.Parameter(
            torch.randn(1, 2, inner_dim) / inner_dim**0.5
        )
        patch_volume = patch_size[0] * patch_size[1] * patch_size[2]
        self.proj_out = nn.Linear(inner_dim, out_channels * patch_volume)
        self.gradient_checkpointing = False

    def gradient_checkpointing_enable(self, **kwargs) -> None:
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        self.gradient_checkpointing = False

    def enable_gradient_checkpointing(self, **kwargs) -> None:
        self.gradient_checkpointing_enable(**kwargs)

    def _patchify(self, hidden_states: torch.Tensor):
        patch_t, patch_h, patch_w = self.config.patch_size
        frames, height, width = hidden_states.shape[-3:]
        pad_t = (-frames) % patch_t
        pad_h = (-height) % patch_h
        pad_w = (-width) % patch_w
        if pad_t or pad_h or pad_w:
            hidden_states = F.pad(hidden_states, (0, pad_w, 0, pad_h, 0, pad_t))
        patches = self.patch_embedding(hidden_states)
        grid = patches.shape[-3:]
        tokens = patches.flatten(2).transpose(1, 2)
        return tokens, grid, (frames, height, width)

    def _unpatchify(
        self,
        tokens: torch.Tensor,
        grid: tuple[int, int, int],
        original_shape: tuple[int, int, int],
    ) -> torch.Tensor:
        batch = tokens.shape[0]
        grid_t, grid_h, grid_w = grid
        patch_t, patch_h, patch_w = self.config.patch_size
        channels = self.config.out_channels
        tokens = tokens.view(
            batch,
            grid_t,
            grid_h,
            grid_w,
            patch_t,
            patch_h,
            patch_w,
            channels,
        )
        tokens = tokens.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
        sample = tokens.view(
            batch,
            channels,
            grid_t * patch_t,
            grid_h * patch_h,
            grid_w * patch_w,
        )
        frames, height, width = original_shape
        return sample[:, :, :frames, :height, :width]

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None = None,
        return_dict: bool = True,
        **kwargs,
    ):
        if kwargs:
            ignored = {
                key: value
                for key, value in kwargs.items()
                if key not in {"image_embeds", "attention_kwargs"} and value is not None
            }
            if ignored:
                raise TypeError(
                    f"Unsupported Wan transformer kwargs: {sorted(ignored)}"
                )
        if hidden_states.ndim != 5:
            raise ValueError(
                f"Expected latent input [B,C,T,H,W], got {tuple(hidden_states.shape)}"
            )
        batch = hidden_states.shape[0]
        if timestep.ndim == 0:
            timestep = timestep.expand(batch)
        elif timestep.numel() == 1 and batch > 1:
            timestep = timestep.reshape(1).expand(batch)
        elif timestep.shape[0] != batch:
            raise ValueError("timestep batch dimension must match hidden_states")

        tokens, grid, original_shape = self._patchify(hidden_states)
        temb, timestep_projection, text = self.condition_embedder(
            timestep.to(tokens.device),
            encoder_hidden_states.to(device=tokens.device, dtype=tokens.dtype),
            tokens.dtype,
        )
        temb = temb.to(tokens.dtype)
        timestep_projection = timestep_projection.to(tokens.dtype).view(
            batch, 6, self.inner_dim
        )
        rotary_embedding = self.rope(grid, tokens.device)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training and tokens.requires_grad:

                def custom_forward(states, context, embedding, module=block):
                    return module(
                        states,
                        context,
                        embedding,
                        rotary_embedding,
                        encoder_attention_mask,
                    )

                tokens = checkpoint(
                    custom_forward,
                    tokens,
                    text,
                    timestep_projection,
                    use_reentrant=False,
                )
            else:
                tokens = block(
                    tokens,
                    text,
                    timestep_projection,
                    rotary_embedding,
                    encoder_attention_mask,
                )
        output_modulation = self.scale_shift_table + temb.unsqueeze(1)
        shift, scale = output_modulation.unbind(dim=1)
        tokens = (
            self.norm_out(tokens.float()).to(tokens.dtype) * (1 + scale[:, None])
            + shift[:, None]
        )
        tokens = self.proj_out(tokens)
        sample = self._unpatchify(tokens, grid, original_shape)
        if not return_dict:
            return (sample,)
        return Transformer3DModelOutput(sample=sample)
