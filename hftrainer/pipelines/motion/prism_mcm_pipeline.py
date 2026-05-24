"""MCM audio-conditioned PRISM inference pipeline with autoregressive support.

Supports two modes:
1. **Single-segment**: Generate one fixed-length motion clip.
2. **Autoregressive**: Generate arbitrarily long motion by chaining segments,
   where each segment's first frame(s) are noise-free condition from the
   previous segment's last frame(s). This is the key feature inherited from
   PRISM's autoregressive generation mechanism.

The autoregressive mechanism uses per-token timesteps (``expand_timesteps``):
- Condition frames receive t=0 (noise-free) throughout denoising.
- Non-condition frames receive the current denoising timestep.
- After each scheduler step, condition frames are force-restored to their
  clean latent values.

Audio conditioning can be provided as:
- Raw waveform (``audio`` parameter, encoded on-the-fly)
- Pre-computed features (``audio_features`` parameter)
- Per-segment audio chunks for autoregressive generation

For dance/speech tasks, the pipeline automatically slices audio to match
each segment's temporal span.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import torch
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange

from hftrainer.models.motion.prism.gaussian_distribution import (
    DiagonalGaussianDistributionNd,
)
from hftrainer.pipelines.base_pipeline import BasePipeline
from hftrainer.registry import PIPELINES


@PIPELINES.register_module()
class PrismMCMPipeline(BasePipeline):
    """Inference pipeline for MCM audio-conditioned PRISM.

    Supports autoregressive infinite-length motion generation via noise-free
    condition injection, following PRISM's per-token timestep mechanism.

    In autoregressive mode:
    1. The first segment is generated normally (or with an optional first-frame
       condition from a reference motion file).
    2. The last frame of each segment is encoded to VAE latent space and used
       as the noise-free first-frame condition for the next segment.
    3. Audio is automatically sliced per segment when a full waveform is provided.
    """

    def __init__(self, bundle, **kwargs):
        super().__init__(bundle)

    # ------------------------------------------------------------------
    # Single-segment generation (low-level)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_single_segment(
        self,
        text_states: torch.Tensor,
        audio_feat: Optional[torch.Tensor] = None,
        negative_text_states: Optional[torch.Tensor] = None,
        first_frame_latents: Optional[torch.Tensor] = None,
        num_frames: int = 129,
        num_joints: int = 23,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        expand_timesteps: bool = True,
    ) -> torch.Tensor:
        """Generate a single motion segment with optional first-frame conditioning.

        Args:
            text_states: Encoded text embeddings ``[B, N_text, D]``.
            audio_feat: Audio features ``[B, N_audio, D]`` (optional).
            negative_text_states: Negative text embeddings for CFG (optional).
            first_frame_latents: Clean first-frame latents ``[B, C, 1, J]`` (optional).
            num_frames: Number of motion frames to generate.
            num_joints: Number of body joints.
            num_inference_steps: Denoising steps.
            guidance_scale: CFG scale (>1 enables guidance).
            expand_timesteps: Use per-token timesteps for noise-free conditioning.

        Returns:
            Output latents ``[B, C, T_latent, J]``.
        """
        bundle = self.bundle
        device = text_states.device
        dtype = text_states.dtype
        batch_size = text_states.shape[0]

        # Prepare latent shape
        vae_temporal = bundle.vae.config.scale_factor_temporal
        num_latent_frames = (num_frames - 1) // vae_temporal + 1
        num_channels = bundle.transformer.config.in_channels

        # Random noise
        latents = randn_tensor(
            (batch_size, num_channels, num_latent_frames, num_joints),
            device=device, dtype=dtype,
        )

        # Condition mask: 0 = condition (noise-free), 1 = denoise
        condition = torch.zeros_like(latents)
        first_frame_mask = torch.ones_like(latents)

        if first_frame_latents is not None:
            if first_frame_latents.shape[0] == 1 and batch_size > 1:
                first_frame_latents = first_frame_latents.expand(batch_size, -1, -1, -1)
            condition[:, :, :1, :] = first_frame_latents.to(device=device, dtype=dtype)
            first_frame_mask[:, :, :1, :] = 0.0

        # Scheduler timesteps
        bundle.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = bundle.scheduler.timesteps

        do_cfg = guidance_scale > 1.0 and negative_text_states is not None

        # Denoising loop
        for t in timesteps:
            if expand_timesteps and first_frame_latents is not None:
                # Replace condition frames with clean latents
                latent_model_input = (
                    (1 - first_frame_mask) * condition + first_frame_mask * latents
                ).to(dtype)

                # Per-token timesteps: condition frames get t=0
                temp_ts = (first_frame_mask[0][0] * t).flatten()
                t_batch = temp_ts.unsqueeze(0).expand(batch_size, -1)
            else:
                latent_model_input = latents.to(dtype)
                t_batch = t.unsqueeze(0).expand(batch_size)

            # Forward with control branch
            model_pred = bundle.predict_with_control(
                noisy_latents=latent_model_input,
                timesteps=t_batch,
                text_states=text_states,
                audio_features=audio_feat,
            )

            # Classifier-free guidance
            if do_cfg:
                noise_uncond = bundle.predict_with_control(
                    noisy_latents=latent_model_input,
                    timesteps=t_batch,
                    text_states=negative_text_states,
                    audio_features=None,  # No audio for unconditional
                )
                model_pred = noise_uncond + guidance_scale * (model_pred - noise_uncond)

            latents = bundle.scheduler.step(model_pred, t, latents).prev_sample

            # Force-restore condition frames after each step
            if first_frame_latents is not None:
                latents = (1 - first_frame_mask) * condition + first_frame_mask * latents

        # Final merge (redundant but safe)
        if expand_timesteps and first_frame_latents is not None:
            latents = (1 - first_frame_mask) * condition + first_frame_mask * latents

        return latents

    # ------------------------------------------------------------------
    # Decode latents → motion (VAE space → SMPL motion space)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to motion.

        Args:
            latents: ``[B, C, T_latent, J]``.

        Returns:
            Motion tensor ``[B, T, J, D]`` in normalised 6D rotation format.
        """
        bundle = self.bundle
        latents = latents * bundle.latents_std.to(latents) + bundle.latents_mean.to(latents)
        # VAE must run in fp32 — override any global AMP autocast.
        device_type = latents.device.type
        with torch.autocast(device_type, enabled=False):
            motion = bundle.vae.decode(latents.float())
        return motion  # [B, T, J, D]

    @torch.no_grad()
    def encode_motion_to_latent(self, motion: torch.Tensor) -> torch.Tensor:
        """Encode motion to VAE latent space.

        Args:
            motion: ``[B, T, J, D]`` in normalised 6D rotation format.

        Returns:
            Latents ``[B, C, T_latent, J]``.
        """
        bundle = self.bundle
        # VAE must run in fp32 — override any global AMP autocast.
        device_type = motion.device.type
        with torch.autocast(device_type, enabled=False):
            z = bundle.vae.encode(motion.float())
        lat = DiagonalGaussianDistributionNd(z)
        z = lat.mode()
        z = (z - bundle.latents_mean.to(z)) / bundle.latents_std.to(z)
        return z

    # ------------------------------------------------------------------
    # Public entry: supports both single and autoregressive modes
    # ------------------------------------------------------------------

    @torch.no_grad()
    def __call__(
        self,
        prompts: Union[str, List[str]],
        audio: Optional[torch.Tensor] = None,
        audio_features: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        first_frame_motion_path: Optional[str] = None,
        num_frames_per_segment: Union[int, List[int]] = 129,
        num_joints: int = 23,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        max_sequence_length: int = 256,
        expand_timesteps: bool = True,
        overlap_frames: int = 1,
        audio_sr: int = 16000,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate motion conditioned on text and optional audio.

        Supports autoregressive multi-segment generation when ``prompts``
        is a list with multiple entries, or when the requested duration
        exceeds a single segment.

        Args:
            prompts: Text prompt(s). If a list, each entry generates one segment
                autoregressively. Use ``;`` separated string for convenience.
            audio: Raw audio waveform ``[1, T_samples]`` at ``audio_sr`` Hz.
                For multi-segment, automatically sliced per segment.
            audio_features: Pre-computed audio features ``[B, N, D]``.
            negative_prompt: Negative prompt for classifier-free guidance.
            first_frame_motion_path: Path to ``.npz`` file for first-frame
                condition of the first segment.
            num_frames_per_segment: Motion frames per segment (int or per-segment list).
            num_joints: Number of body joints.
            num_inference_steps: Denoising steps per segment.
            guidance_scale: CFG scale (>1 enables guidance).
            max_sequence_length: Max text token length.
            expand_timesteps: Use per-token timesteps for AR conditioning.
            overlap_frames: Overlap between segments (default 1 = last frame).
            audio_sr: Sample rate of raw audio waveform.

        Returns:
            Dict with:
            - ``'motion'``: Raw latent output ``[1, C, T_total, J]`` (all segments concatenated).
            - ``'motion_decoded'``: Decoded motion ``[1, T_total, J, D]`` if VAE decode succeeds.
            - ``'prompts'``: The input prompt list.
            - ``'num_frames'``: Total motion frames generated.
        """
        bundle = self.bundle
        device = next(bundle.transformer.parameters()).device
        dtype = next(bundle.transformer.parameters()).dtype

        # Parse prompts
        if isinstance(prompts, str):
            # Support semicolon-separated multi-prompt
            if ';' in prompts:
                prompts = [p.strip() for p in prompts.split(';') if p.strip()]
            else:
                prompts = [prompts]

        num_segments = len(prompts)
        vae_temporal = bundle.vae.config.scale_factor_temporal

        # Per-segment frame counts
        def _round_frames(n: int) -> int:
            if (n - 1) % vae_temporal != 0:
                return (n // vae_temporal) * vae_temporal + 1
            return max(1, n)

        if isinstance(num_frames_per_segment, list):
            if len(num_frames_per_segment) != num_segments:
                single = _round_frames(num_frames_per_segment[0] if num_frames_per_segment else 129)
                frame_counts = [single] * num_segments
            else:
                frame_counts = [_round_frames(n) for n in num_frames_per_segment]
        else:
            single = _round_frames(num_frames_per_segment)
            frame_counts = [single] * num_segments

        # Encode negative prompt for CFG
        negative_text_states = None
        if guidance_scale > 1.0:
            neg_prompt = negative_prompt or ''
            negative_text_states = bundle.encode_prompt(
                [neg_prompt],
                max_sequence_length=max_sequence_length,
                prompt_drop_rate=0.0,
                dtype=dtype,
            ).to(device)

        # Load first-frame condition if provided
        first_frame_latents = None
        if first_frame_motion_path is not None:
            first_frame_latents = self._load_first_frame(first_frame_motion_path, device, dtype)

        # Audio handling: compute total audio and segment boundaries for slicing
        full_audio = audio  # [1, T_samples] or None
        full_audio_features = audio_features  # [1, N, D] or None

        # Generate segments autoregressively
        all_latent_segments = []
        all_decoded_segments = []

        for seg_idx in range(num_segments):
            prompt = prompts[seg_idx]
            num_frames_this = frame_counts[seg_idx]

            # Encode this segment's text
            text_states = bundle.encode_prompt(
                [prompt],
                max_sequence_length=max_sequence_length,
                prompt_drop_rate=0.0,
                dtype=dtype,
            ).to(device)

            # Prepare audio features for this segment
            seg_audio_feat = self._get_segment_audio(
                seg_idx=seg_idx,
                num_segments=num_segments,
                num_frames_this=num_frames_this,
                frame_counts=frame_counts,
                full_audio=full_audio,
                full_audio_features=full_audio_features,
                audio_sr=audio_sr,
                device=device,
                dtype=dtype,
            )

            # Generate single segment
            seg_latents = self.generate_single_segment(
                text_states=text_states,
                audio_feat=seg_audio_feat,
                negative_text_states=negative_text_states,
                first_frame_latents=first_frame_latents,
                num_frames=num_frames_this,
                num_joints=num_joints,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                expand_timesteps=expand_timesteps,
            )

            # Decode this segment for extracting last frame
            seg_decoded = self.decode_latents(seg_latents)  # [1, T, J, D]

            # Store segment (skip overlapping frames for non-first segments)
            if seg_idx == 0:
                all_latent_segments.append(seg_latents)
                all_decoded_segments.append(seg_decoded)
            else:
                # Skip first `overlap_frames` in latent space
                latent_overlap = max(1, (overlap_frames - 1) // vae_temporal + 1)
                all_latent_segments.append(seg_latents[:, :, latent_overlap:, :])
                # Skip first `overlap_frames` in decoded motion
                all_decoded_segments.append(seg_decoded[:, overlap_frames:, :, :])

            # Extract last frame as condition for next segment
            last_frame_motion = seg_decoded[:, -1:, :, :]  # [1, 1, J, D]
            first_frame_latents = self.encode_motion_to_latent(last_frame_motion)

        # Concatenate all segments
        full_latents = torch.cat(all_latent_segments, dim=2)  # [1, C, T_total, J]
        full_decoded = torch.cat(all_decoded_segments, dim=1)  # [1, T_total, J, D]

        return {
            'motion': full_latents,
            'motion_decoded': full_decoded,
            'prompts': prompts,
            'num_frames': full_decoded.shape[1],
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_first_frame(
        self,
        motion_path: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Load first-frame condition from an npz file.

        Returns:
            First-frame latents ``[1, C, 1, J]``.
        """
        bundle = self.bundle
        smplx_dict = bundle.smpl_pose_processor.load_smplx_dict_from_npz(motion_path)
        motion = (
            bundle.smpl_pose_processor.smplx_dict_to_motion_vector(smplx_dict)
            .unsqueeze(0)
            .to(device=device, dtype=torch.float32)
        )
        motion = bundle.smpl_pose_processor.normalize(motion)
        motion = rearrange(motion, 'b t (j d) -> b t j d', d=6)
        motion = motion[:, :1]  # First frame only [1, 1, J, 6]
        return self.encode_motion_to_latent(motion)

    def _get_segment_audio(
        self,
        seg_idx: int,
        num_segments: int,
        num_frames_this: int,
        frame_counts: List[int],
        full_audio: Optional[torch.Tensor],
        full_audio_features: Optional[torch.Tensor],
        audio_sr: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """Get audio features for a specific segment.

        If full audio waveform is provided, slices it according to the
        segment's temporal position. Assumes 30fps motion.
        """
        bundle = self.bundle

        # Pre-computed features: return as-is for single segment, or None
        if full_audio_features is not None:
            if num_segments == 1:
                return full_audio_features.to(device=device, dtype=dtype)
            # For multi-segment with pre-computed features, we'd need temporal
            # info to slice. Just return full features for now.
            return full_audio_features.to(device=device, dtype=dtype)

        if full_audio is None:
            return None

        if not hasattr(bundle, 'audio_encoder'):
            return None

        # Slice audio for this segment based on frame counts
        # Assume 30fps motion
        motion_fps = 30.0
        total_frames_before = sum(frame_counts[:seg_idx])
        start_time = total_frames_before / motion_fps
        end_time = (total_frames_before + num_frames_this) / motion_fps

        start_sample = int(start_time * audio_sr)
        end_sample = int(end_time * audio_sr)

        # Clamp to audio length
        audio_len = full_audio.shape[-1]
        start_sample = min(start_sample, audio_len)
        end_sample = min(end_sample, audio_len)

        if start_sample >= end_sample:
            return None

        seg_audio = full_audio[..., start_sample:end_sample].to(device)
        return bundle.encode_audio(seg_audio, sr=audio_sr).to(dtype=dtype)

    # ------------------------------------------------------------------
    # Post-processing (optional: convert to SMPL-X dict)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def post_process_to_smplx(
        self,
        result: Dict[str, Any],
        normalize: bool = True,
        mocap_framerate: float = 30.0,
        gender: str = 'neutral',
    ) -> Dict[str, Any]:
        """Convert pipeline output to SMPL-X parameter dict.

        Args:
            result: Pipeline output dict containing ``'motion_decoded'``.
            normalize: Whether to normalize facing direction.
            mocap_framerate: Output frame rate.
            gender: SMPL gender.

        Returns:
            SMPL-X parameter dict (transl, body_pose, etc.).
        """
        from hftrainer.models.motion.components.utils.geometry.rotation_convert import (
            rotation_6d_to_axis_angle,
        )

        bundle = self.bundle
        motion_decoded = result['motion_decoded']  # [1, T, J, D]

        # De-normalise and convert to axis-angle
        x_dec = rearrange(motion_decoded, 'b t j d -> b t (j d)')
        x_dec = bundle.smpl_pose_processor.denormalize(x_dec)
        transl_abs_rel = x_dec[..., :6]
        transl = bundle.smpl_pose_processor.inv_convert_transl(transl_abs_rel)
        pred_poses = x_dec[..., 6:]
        pred_poses = rearrange(pred_poses, 'b t (j d) -> (b t) j d', d=6)
        # Training data already uses column-major 6D convention [R00,R10,R20,R01,R11,R21]
        # (matrix_to_rotation_6d uses _stack_cols01 → columns of rotation matrix).
        # rotation_6d_to_axis_angle expects column-major input — no permutation needed.
        pred_poses = rotation_6d_to_axis_angle(pred_poses)
        pred_poses = rearrange(pred_poses, '(b t) j d -> b t (j d)', b=1)

        smplx_dict = bundle.smpl_pose_processor.transl_pose_to_smplx_dict(
            transl.squeeze(0),
            pred_poses.squeeze(0),
            mocap_framerate=mocap_framerate,
            gender=gender,
            rot_type='axis_angle',
        )

        if normalize:
            smplx_dict = bundle.smpl_pose_processor.normalize_smplx_dict(smplx_dict)

        return smplx_dict
