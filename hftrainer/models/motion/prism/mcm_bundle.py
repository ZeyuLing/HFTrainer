"""MCM audio-conditioned PRISM bundle (WanVACE-style).

Extends PrismBundle with a WanVACE-style sparse control branch for
audio-conditioned motion generation.  The main transformer is frozen;
only the control branch (a small subset of blocks) is trained.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers
from einops import rearrange

from hftrainer.models.base_model_bundle import ModelBundle
from hftrainer.models.motion.prism.gaussian_distribution import (
    DiagonalGaussianDistributionNd,
)
from hftrainer.models.motion.prism.bundle import PrismBundle, _get_sigmas
from hftrainer.models.motion.prism.control_transformer import PrismVACEControlTransformer
from hftrainer.registry import MODEL_BUNDLES


@MODEL_BUNDLES.register_module()
class PrismMCMBundle(PrismBundle):
    """Bundle for MCM audio-conditioned PRISM (WanVACE-style).

    Uses a sparse control branch that creates only a subset of transformer
    blocks at evenly-spaced layer indices, following the WanVACE architecture.
    This reduces trainable parameters by ~72% compared to the original MCM
    full-copy approach.

    Architecture:
        Phase 1 — Run all VACE blocks sequentially (control branch).
            Audio features are added to patch-embedded motion tokens.
            Each VACE block outputs conditioning_states via proj_out.

        Phase 2 — Run all main-branch blocks, injecting VACE outputs
            at the layer indices specified by ``vace_layers``.
            U-skip pattern: last VACE block → highest injection layer.

    Modules:
        transformer          -- PrismTransformerMotionModel (FROZEN)
        control_transformer  -- PrismVACEControlTransformer (TRAINABLE)
        vae                  -- AutoencoderKLPrism2DTK (frozen)
        tokenizer            -- T5/AutoTokenizer (frozen)
        text_encoder         -- T5/UMT5 encoder (frozen)
        scheduler            -- FlowMatchEulerDiscreteScheduler (frozen)
        smpl_pose_processor  -- SMPLPoseProcessor (frozen)
        audio_encoder        -- AudioEncoderWrapper (frozen, optional)
    """

    def __init__(
        self,
        transformer: dict,
        control_transformer: dict,
        vae: dict,
        tokenizer: dict,
        text_encoder: dict,
        scheduler: dict,
        smpl_pose_processor: dict,
        audio_encoder: Optional[dict] = None,
        init_control_from_main: bool = True,
    ):
        # Skip PrismBundle.__init__ — call ModelBundle.__init__ directly
        ModelBundle.__init__(self)

        modules = {
            'transformer': transformer,
            'control_transformer': control_transformer,
            'vae': vae,
            'tokenizer': tokenizer,
            'text_encoder': text_encoder,
            'scheduler': scheduler,
            'smpl_pose_processor': smpl_pose_processor,
        }
        if audio_encoder is not None:
            modules['audio_encoder'] = audio_encoder

        self._build_modules(modules)

        # Copy main branch weights into control branch
        if init_control_from_main:
            PrismVACEControlTransformer.init_from_main_branch(
                self.control_transformer, self.transformer,
            )

        # Setup scheduler timesteps
        if hasattr(self.scheduler, 'set_timesteps'):
            self.scheduler.set_timesteps(self.scheduler.config.num_train_timesteps)

        # Copy latent normalisation stats from VAE
        self.use_static = bool(getattr(self.vae.config, 'use_static', False))
        self.register_buffer(
            'latents_mean',
            torch.tensor(self.vae.config.latents_mean).view(1, self.vae.config.z_dim, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            'latents_std',
            torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1),
            persistent=False,
        )

    # ------------------------------------------------------------------
    # Checkpoint loading: re-init control from main after weight load
    # ------------------------------------------------------------------

    def load_state_dict_selective(self, state_dict, strict=False):
        """Override to re-copy transformer → control_transformer after load.

        When loading pretrained PRISM weights into the ``transformer`` module
        (e.g. from a converged PRISM checkpoint via ``load_from``), the
        control branch must be re-initialized from the newly loaded weights
        so that the zero-init proj_out guarantee holds.
        """
        # Check if transformer weights are being loaded
        has_transformer_weights = False
        if isinstance(state_dict, dict):
            first_val = next(iter(state_dict.values()), None)
            if isinstance(first_val, dict) and 'transformer' in state_dict:
                has_transformer_weights = True
            elif isinstance(first_val, torch.Tensor):
                has_transformer_weights = any(
                    k.startswith('transformer.') for k in state_dict
                )

        # Load weights normally
        super().load_state_dict_selective(state_dict, strict=strict)

        # If transformer weights were loaded, re-init control branch from them
        if has_transformer_weights:
            from hftrainer.utils.logger import get_logger
            logger = get_logger()
            logger.info(
                "Transformer weights loaded — re-initializing control branch "
                "from main branch (preserving proj_out zeros and audio_proj)."
            )
            PrismVACEControlTransformer.init_from_main_branch(
                self.control_transformer, self.transformer,
            )

    # ------------------------------------------------------------------
    # Audio encoding helper
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_audio(self, waveform: torch.Tensor, sr: int = 16000) -> torch.Tensor:
        """``[B, T_samples]`` -> ``[B, N_frames, audio_feature_dim]``."""
        return self.audio_encoder(waveform, sr=sr)

    # ------------------------------------------------------------------
    # Main forward: VACE control branch → inject into main branch
    # ------------------------------------------------------------------

    def predict_with_control(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        text_states: torch.Tensor,
        audio_features: Optional[torch.Tensor] = None,
        hidden_states_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """Run VACE control branch, then main branch with sparse injection.

        Phase 1: All VACE blocks run sequentially, producing conditioning
        outputs at each block. Audio features are injected into the control
        branch via additive projection.

        Phase 2: Main branch blocks run sequentially. At layer indices
        specified by ``vace_layers``, the corresponding VACE conditioning
        output is added to the main hidden states.

        Returns:
            Predicted velocity / noise, same shape as ``noisy_latents``.
        """
        # --- Phase 1: Control branch (VACE blocks) ---
        # Cast inputs to control_transformer's parameter dtype.  During FSDP
        # training the mixed-precision wrapper handles this automatically, but
        # during inference (no FSDP) the control branch may be fp32 while the
        # inputs arrive in bf16 from the frozen transformer's dtype.
        ctrl_dtype = next(self.control_transformer.parameters()).dtype
        conditioning_list = self.control_transformer(
            hidden_states=noisy_latents.to(dtype=ctrl_dtype),
            timestep=timesteps,
            encoder_hidden_states=text_states.to(dtype=ctrl_dtype),
            audio_features=(
                audio_features.to(dtype=ctrl_dtype)
                if audio_features is not None else None
            ),
            hidden_states_mask=hidden_states_mask,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            is_causal=is_causal,
        )

        # Build a dict for O(1) lookup: layer_idx -> conditioning_states
        conditioning_dict = {
            layer_idx: cond_states
            for cond_states, layer_idx in conditioning_list
        }

        # --- Phase 2: Inline main branch forward with sparse injection ---
        main = self.transformer
        main_dtype = next(main.parameters()).dtype
        noisy_latents = noisy_latents.to(dtype=main_dtype)
        text_states = text_states.to(dtype=main_dtype)

        batch_size, num_channels, num_frames, num_joints = noisy_latents.shape
        p_t, p_j = main.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_num_joints = num_joints // p_j

        # RoPE
        rotary_emb = main.rope(noisy_latents)

        # Patch embedding
        hidden_states = main.patch_embedding(noisy_latents)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        # Process hidden_states_mask
        hs_mask_processed = None
        if hidden_states_mask is not None:
            m = hidden_states_mask.reshape(
                batch_size, post_patch_num_frames, p_t,
                post_patch_num_joints, p_j,
            )
            m = m.amin(dim=(2, 4)).flatten(1)
            hs_mask_processed = (
                (1.0 - m.float()) * torch.finfo(hidden_states.dtype).min
            ).to(dtype=hidden_states.dtype).unsqueeze(1).unsqueeze(2)

        # Process encoder mask
        enc_mask_processed = None
        if encoder_hidden_states_mask is not None:
            enc_mask_processed = (
                (1.0 - encoder_hidden_states_mask.float())
                * torch.finfo(hidden_states.dtype).min
            ).to(dtype=hidden_states.dtype).unsqueeze(1).unsqueeze(2)

        # Causal mask
        causal_mask = None
        if is_causal:
            seq_len = hidden_states.shape[1]
            frame_idx = torch.arange(seq_len, device=hidden_states.device) // post_patch_num_joints
            causal_mask = (
                (frame_idx.unsqueeze(0) > frame_idx.unsqueeze(1))
                .to(hidden_states.dtype)
                * torch.finfo(hidden_states.dtype).min
            ).unsqueeze(0).unsqueeze(0)

        # Timestep + text conditioning
        timestep_input = timesteps
        if timestep_input.ndim == 2:
            ts_seq_len = timestep_input.shape[1]
            timestep_input = timestep_input.flatten()
        else:
            ts_seq_len = None

        temb, timestep_proj, encoder_hs = main.condition_embedder(
            timestep_input, text_states, timestep_seq_len=ts_seq_len,
        )

        if ts_seq_len is not None:
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
        else:
            timestep_proj = timestep_proj.unflatten(1, (6, -1))

        # Transformer blocks with sparse VACE injection
        for i, block in enumerate(main.blocks):
            # VACE injection: add conditioning at designated layers
            if i in conditioning_dict:
                hidden_states = hidden_states + conditioning_dict[i].to(dtype=hidden_states.dtype)

            if torch.is_grad_enabled() and getattr(main, 'gradient_checkpointing', False):
                hidden_states = torch.utils.checkpoint.checkpoint(
                    block,
                    hidden_states,
                    encoder_hs,
                    timestep_proj,
                    rotary_emb,
                    hs_mask_processed,
                    enc_mask_processed,
                    causal_mask,
                    use_reentrant=False,
                )
            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hs,
                    temb=timestep_proj,
                    rotary_emb=rotary_emb,
                    hidden_states_mask=hs_mask_processed,
                    encoder_hidden_states_mask=enc_mask_processed,
                    causal_mask=causal_mask,
                )

        # Output normalization
        if temb.ndim == 3:
            shift, scale = (
                main.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)
            ).chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
        else:
            shift, scale = (
                main.scale_shift_table.to(temb.device) + temb.unsqueeze(1)
            ).chunk(2, dim=1)

        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        hidden_states = (
            main.norm_out(hidden_states.float()) * (1 + scale) + shift
        ).type_as(hidden_states)

        # Output projection
        hidden_states = main.proj_out(hidden_states)

        # Unpatchify
        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_num_joints, p_t, p_j, -1,
        )
        hidden_states = hidden_states.permute(0, 5, 1, 3, 2, 4)
        output = hidden_states.flatten(4, 5).flatten(2, 3)

        return output
