"""
Custom Attention Processor with FP32 Upscaling for Softmax Stability

This module implements an extended version of WanAttnProcessor that automatically
upcasts query, key, and value tensors to fp32 during attention computation to prevent
softmax overflow in mixed-precision (fp16) training.

Problem: 
  In fp16 training, attention scores can exceed the safe range for exp(), causing overflow.
  fp16 exp() overflows at x > 11.09, and even softmax(x) = exp(x - max(x)) can overflow
  when attention scores are large.

Solution:
  Upcast Q, K, V to fp32 only during scaled_dot_product_attention, then cast back.
  This prevents softmax overflow while minimizing performance impact.

References:
  - PyTorch mixed precision docs: https://pytorch.org/docs/stable/notes/amp_examples.html
  - Flash Attention: https://github.com/dao-ailab/flash-attention (handles fp32 internally)
  - Diffusers attention dispatch: diffusers/models/attention_dispatch.py
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models.transformers.transformer_wan import (
    WanAttnProcessor,
    _get_qkv_projections,
    _get_added_kv_projections,
)
from diffusers.models.attention_dispatch import dispatch_attention_fn


class WanAttnProcessorFP32Upcast(WanAttnProcessor):
    """
    WanAttnProcessor with automatic fp32 upscaling for softmax stability.

    This processor extends WanAttnProcessor to automatically upcast query, key, and value
    tensors to fp32 during attention computation when the input dtype is fp16. This prevents
    softmax numerical overflow while maintaining the efficiency benefits of fp16 for
    other computations.

    Behavior:
      - If input is fp16: Upcasts Q, K, V to fp32 for attention, then casts output back
      - If input is fp32 or higher: Uses input dtype directly (no overhead)
      - Automatically detects dtype from input tensors

    Example:
        >>> processor = WanAttnProcessorFP32Upcast()
        >>> attn_module = WanAttention(..., processor=processor)
        >>> output = attn_module(hidden_states, encoder_hidden_states)  # Handles fp16 automatically

    Attributes:
        _use_fp32_upcast (bool): If True, automatically upcast to fp32. Default: True.
        _supported_precisions (tuple): Precisions for which upcast is helpful. Default: (torch.float16, torch.bfloat16).
    """

    _use_fp32_upcast = True
    _supported_precisions = (torch.float16, torch.bfloat16)

    def __init__(self, use_fp32_upcast: bool = True):
        """
        Initialize the FP32 upcast attention processor.

        Args:
            use_fp32_upcast (bool): Whether to enable fp32 upscaling. Default: True.
        """
        super().__init__()
        self._use_fp32_upcast = use_fp32_upcast

    def __call__(
        self,
        attn: "WanAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Forward pass with optional fp32 upscaling for softmax stability.

        The attention computation is performed with upcast precision if enabled and
        the input dtype is fp16. Otherwise, it behaves identically to WanAttnProcessor.

        Args:
            attn (WanAttention): The attention module.
            hidden_states (torch.Tensor): Input hidden states.
                Shape: [batch_size, seq_len, hidden_dim]
            encoder_hidden_states (torch.Tensor, optional): Cross-attention key/value states.
                Shape: [batch_size, encoder_seq_len, hidden_dim]. Default: None (self-attention)
            attention_mask (torch.Tensor, optional): Attention bias mask.
                Shape: [batch_size, 1, 1, seq_len] or [batch_size, seq_len]
                Typical values: 0 for valid, -inf for masked positions.
            rotary_emb (Tuple[torch.Tensor, torch.Tensor], optional): Rotary embeddings.
                (cos_embed, sin_embed) for RoPE. Default: None

        Returns:
            torch.Tensor: Output with same shape and dtype as hidden_states.
        """
        # Determine if we should upcast
        # Case 1: hidden_states are already fp16 (direct fp16 parameters)
        # Case 2: autocast(fp16) is active — norm layers output fp32 but downstream
        #         linear/SDPA ops will be cast to fp16 by autocast, causing softmax
        #         overflow. We must intercept and run SDPA in fp32 with autocast disabled.
        autocast_fp16_active = (
            torch.is_autocast_enabled()
            and torch.get_autocast_gpu_dtype() == torch.float16
        )
        should_upcast = (
            self._use_fp32_upcast
            and (
                hidden_states.dtype in self._supported_precisions
                or autocast_fp16_active
            )
        )

        if not should_upcast:
            # No upcast needed - use parent implementation
            return super().__call__(
                attn, hidden_states, encoder_hidden_states, attention_mask, rotary_emb
            )

        # ====================================================================
        # FP32 Upcast Path - Softmax overflow prevention
        # ====================================================================
        original_dtype = hidden_states.dtype

        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            # 512 is the context length of the text encoder, hardcoded for now
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

        # Get QKV projections (remains in original dtype)
        query, key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)

        # Apply RMSNorm (remains in original dtype)
        query = attn.norm_q(query)
        key = attn.norm_k(key)

        # Reshape to multi-head format (remains in original dtype)
        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        # Apply rotary embeddings (remains in original dtype)
        if rotary_emb is not None:

            def apply_rotary_emb(
                hidden_states: torch.Tensor,
                freqs_cos: torch.Tensor,
                freqs_sin: torch.Tensor,
            ):
                x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
                cos = freqs_cos[..., 0::2]
                sin = freqs_sin[..., 1::2]
                out = torch.empty_like(hidden_states)
                out[..., 0::2] = x1 * cos - x2 * sin
                out[..., 1::2] = x1 * sin + x2 * cos
                return out.type_as(hidden_states)

            query = apply_rotary_emb(query, *rotary_emb)
            key = apply_rotary_emb(key, *rotary_emb)

        # I2V task (image-to-video additional key/value)
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img, value_img = _get_added_kv_projections(attn, encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)
            key_img = key_img.unflatten(2, (attn.heads, -1))
            value_img = value_img.unflatten(2, (attn.heads, -1))

            # ============ UPCAST FOR SOFTMAX ============
            # Upcast to fp32 for attention computation (image branch)
            # NOTE: We disable autocast here because F.scaled_dot_product_attention
            # is an autocast-eligible op and would silently cast fp32 inputs back to
            # fp16, defeating the entire purpose of the manual upcast.
            query_fp32 = query.to(torch.float32)
            key_img_fp32 = key_img.to(torch.float32)
            value_img_fp32 = value_img.to(torch.float32)

            with torch.cuda.amp.autocast(enabled=False):
                hidden_states_img = dispatch_attention_fn(
                    query_fp32,
                    key_img_fp32,
                    value_img_fp32,
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=False,
                    backend=self._attention_backend,
                    parallel_config=self._parallel_config,
                )
            # Cast back to original dtype
            hidden_states_img = hidden_states_img.to(original_dtype)
            # ==========================================

            hidden_states_img = hidden_states_img.flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        # ============ UPCAST FOR SOFTMAX ============
        # Upcast to fp32 for attention computation (main branch)
        # NOTE: We disable autocast here because F.scaled_dot_product_attention
        # is an autocast-eligible op and would silently cast fp32 inputs back to
        # fp16, defeating the entire purpose of the manual upcast.
        query_fp32 = query.to(torch.float32)
        key_fp32 = key.to(torch.float32)
        value_fp32 = value.to(torch.float32)

        # Also upcast attention_mask to fp32 if present (it may contain -65504 fp16 values)
        attn_mask_fp32 = attention_mask.to(torch.float32) if attention_mask is not None else None

        with torch.cuda.amp.autocast(enabled=False):
            hidden_states = dispatch_attention_fn(
                query_fp32,
                key_fp32,
                value_fp32,
                attn_mask=attn_mask_fp32,
                dropout_p=0.0,
                is_causal=False,
                backend=self._attention_backend,
                parallel_config=self._parallel_config,
            )
        # Cast back to original dtype
        hidden_states = hidden_states.to(original_dtype)
        # ==========================================

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states

    @classmethod
    def enable_fp32_upcast(cls, enable: bool = True) -> None:
        """
        Class method to globally enable/disable fp32 upcast.

        Args:
            enable (bool): Whether to enable fp32 upscaling. Default: True.
        """
        cls._use_fp32_upcast = enable

    @classmethod
    def get_fp32_upcast_enabled(cls) -> bool:
        """
        Check if fp32 upcast is globally enabled.

        Returns:
            bool: Whether fp32 upcast is enabled.
        """
        return cls._use_fp32_upcast
