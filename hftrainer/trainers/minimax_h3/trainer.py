"""Experimental local fine-tuning objective for MiniMax-H3.

MiniMax released complete weights for further development but did not publish
its training recipe.  This trainer therefore implements the rectified-flow
objective implied by the public scheduler, without claiming recipe parity.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F

from hftrainer.models.minimax_h3.network.layout import (
    AUDIO_CHANNELS,
    MiniMaxH3PackedLayout,
    MiniMaxH3ReferenceGeometry,
    build_row_timesteps,
    patchify_video_latents,
)
from hftrainer.registry import TRAINERS
from hftrainer.trainers.base_trainer import BaseTrainer


@TRAINERS.register_module()
class MiniMaxH3Trainer(BaseTrainer):
    """Fine-tune the local H3 transformer on cached synchronized A/V features.

    The preferred batch contract avoids loading the 32B conditioner and both
    VAEs while the 33B denoiser is resident:

    ``video_latents``: ``[B,24,T,H,W]`` normalized clean latents
    ``audio_latents``: ``[B,2,32,L]`` normalized clean stereo latents
    ``prompt_embeds``: ``[B,N,5120]`` Qwen hidden state 50
    ``text_token_tags``: ``[N]`` H3 tags (text=1, vision rows=0)

    Optional conditioning rows are fixed throughout the forward and excluded
    from both losses.
    """

    def __init__(
        self,
        bundle,
        *,
        video_loss_weight: float = 1.0,
        audio_loss_weight: float = 1.0,
        timestep_distribution: str = "uniform",
        min_timestep: float = 0.0,
        max_timestep: float = 1.0,
        keyframe_noise_aug: float = 0.999,
        mode: str = "t2va",
        **kwargs,
    ) -> None:
        super().__init__(bundle, **kwargs)
        if video_loss_weight < 0 or audio_loss_weight < 0:
            raise ValueError("Loss weights must be non-negative.")
        if video_loss_weight == 0 and audio_loss_weight == 0:
            raise ValueError("At least one MiniMax-H3 loss weight must be positive.")
        if not 0 <= min_timestep < max_timestep <= 1:
            raise ValueError(
                "Training timestep bounds must satisfy 0 <= min < max <= 1."
            )
        if timestep_distribution not in {"uniform", "logit_normal"}:
            raise ValueError("timestep_distribution must be uniform or logit_normal.")
        if mode not in {"t2va", "fl2va", "ref2va"}:
            raise ValueError("mode must be t2va, fl2va, or ref2va.")
        required_variant = "ref2va" if mode == "ref2va" else "fl2va"
        if bundle.variant != required_variant:
            raise ValueError(
                f"Training mode {mode!r} requires bundle variant {required_variant!r}."
            )
        self.video_loss_weight = float(video_loss_weight)
        self.audio_loss_weight = float(audio_loss_weight)
        self.timestep_distribution = timestep_distribution
        self.min_timestep = float(min_timestep)
        self.max_timestep = float(max_timestep)
        self.keyframe_noise_aug = float(keyframe_noise_aug)
        self.mode = mode

    def _sample_timestep(self, device: torch.device) -> torch.Tensor:
        if self.timestep_distribution == "uniform":
            value = torch.rand((), device=device)
        else:
            value = torch.sigmoid(torch.randn((), device=device))
        return self.min_timestep + value * (self.max_timestep - self.min_timestep)

    @staticmethod
    def _plain_references(values: Any) -> tuple[MiniMaxH3ReferenceGeometry, ...]:
        if values is None:
            return ()
        result = []
        for value in values:
            if isinstance(value, MiniMaxH3ReferenceGeometry):
                result.append(value)
            elif isinstance(value, dict):
                result.append(MiniMaxH3ReferenceGeometry(**value))
            else:
                raise TypeError(
                    "reference_geometries entries must be dataclasses/dicts."
                )
        return tuple(result)

    def _build_layout(
        self,
        batch: dict[str, Any],
        video_latents: torch.Tensor,
        audio_latents: torch.Tensor,
        text_token_tags: torch.Tensor,
    ) -> MiniMaxH3PackedLayout:
        layout = batch.get("layout")
        if layout is not None:
            if not isinstance(layout, MiniMaxH3PackedLayout):
                raise TypeError("batch['layout'] must be MiniMaxH3PackedLayout.")
            return layout.to(video_latents.device)
        anchors: Sequence[str] = tuple(batch.get("keyframe_anchors", ()))
        references = self._plain_references(batch.get("reference_geometries"))
        return self.bundle.build_layout(
            text_token_tags,
            num_latent_frames=video_latents.shape[2],
            latent_height=video_latents.shape[3],
            latent_width=video_latents.shape[4],
            num_audio_latents=audio_latents.shape[-1],
            keyframe_anchors=anchors,
            references=references,
        ).to(video_latents.device)

    def train_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        required = {
            "video_latents",
            "audio_latents",
            "prompt_embeds",
            "text_token_tags",
        }
        missing = sorted(required - set(batch))
        if missing:
            raise KeyError(
                "MiniMaxH3Trainer expects cached features; missing: "
                + ", ".join(missing)
            )
        device = self.bundle.device
        video_clean = batch["video_latents"].to(device=device, dtype=torch.float32)
        audio_clean = batch["audio_latents"].to(device=device, dtype=torch.float32)
        prompt_embeds = batch["prompt_embeds"].to(
            device=device, dtype=self.bundle.dtype
        )
        text_token_tags = torch.as_tensor(
            batch["text_token_tags"], device=device, dtype=torch.long
        )
        if video_clean.ndim != 5:
            raise ValueError("video_latents must be [B,C,T,H,W].")
        if audio_clean.ndim != 4 or audio_clean.shape[1] != AUDIO_CHANNELS:
            raise ValueError("audio_latents must be [B,2,C,L].")
        batch_size = video_clean.shape[0]
        if audio_clean.shape[0] != batch_size or prompt_embeds.shape[0] != batch_size:
            raise ValueError("Cached feature batch sizes do not agree.")
        if prompt_embeds.shape[1] != text_token_tags.numel():
            raise ValueError("text_token_tags must address every prompt embedding row.")

        layout = self._build_layout(batch, video_clean, audio_clean, text_token_tags)
        patch_size = tuple(self.bundle.transformer.config.patch_size)
        video_clean_rows = patchify_video_latents(video_clean, patch_size)
        audio_clean_rows = audio_clean.permute(0, 1, 3, 2).reshape(
            batch_size, -1, audio_clean.shape[2]
        )
        video_noise = torch.randn_like(video_clean_rows)
        audio_noise = torch.randn_like(audio_clean_rows)
        video_t = self._sample_timestep(device)
        audio_t = self._sample_timestep(device)
        video_noisy = video_t * video_clean_rows + (1 - video_t) * video_noise
        audio_noisy = audio_t * audio_clean_rows + (1 - audio_t) * audio_noise

        condition_video_rows = batch.get("condition_video_rows")
        condition_audio_rows = batch.get("condition_audio_rows")
        if layout.num_condition_video_rows:
            if condition_video_rows is None:
                raise KeyError("This layout requires condition_video_rows.")
            condition_video_rows = condition_video_rows.to(device, torch.float32)
            if condition_video_rows.ndim == 2:
                condition_video_rows = condition_video_rows[None].expand(
                    batch_size, -1, -1
                )
            if condition_video_rows.shape[:2] != (
                batch_size,
                layout.num_condition_video_rows,
            ):
                raise ValueError("condition_video_rows and the packed layout disagree.")
            condition_video_t = max(float(video_t.detach()), self.keyframe_noise_aug)
            condition_video_noise = torch.randn_like(condition_video_rows)
            condition_video_input = self.bundle.scheduler.scale_noise(
                condition_video_rows,
                condition_video_t,
                condition_video_noise,
            )
            video_input = torch.cat((condition_video_input, video_noisy), dim=1)
        else:
            video_input = video_noisy
        if layout.num_condition_audio_rows:
            if condition_audio_rows is None:
                raise KeyError("This layout requires condition_audio_rows.")
            condition_audio_rows = condition_audio_rows.to(device, torch.float32)
            if condition_audio_rows.ndim == 2:
                condition_audio_rows = condition_audio_rows[None].expand(
                    batch_size, -1, -1
                )
            if condition_audio_rows.shape[:2] != (
                batch_size,
                layout.num_condition_audio_rows,
            ):
                raise ValueError("condition_audio_rows and the packed layout disagree.")
            audio_input = torch.cat((condition_audio_rows, audio_noisy), dim=1)
        else:
            audio_input = audio_noisy
        if video_input.shape[1] != layout.video_indices.numel():
            raise ValueError("Video rows and packed layout disagree.")
        if audio_input.shape[1] != layout.audio_indices.numel():
            raise ValueError("Audio rows and packed layout disagree.")

        unique_t, inverse = build_row_timesteps(
            layout,
            video_timestep=float(video_t.detach()),
            audio_timestep=float(audio_t.detach()),
            condition_video_timestep=max(
                float(video_t.detach()), self.keyframe_noise_aug
            ),
            condition_audio_timestep=1.0,
        )
        video_prediction, audio_prediction = self.bundle.predict_velocity(
            video_input,
            audio_input,
            prompt_embeds,
            layout,
            unique_t.to(device),
            inverse.to(device),
        )
        # H3 predicts data-ward velocity: x_t=t*x0+(1-t)*eps => v=x0-eps.
        video_target = video_clean_rows - video_noise
        audio_target = audio_clean_rows - audio_noise
        video_prediction = video_prediction[:, layout.num_condition_video_rows :]
        audio_prediction = audio_prediction[:, layout.num_condition_audio_rows :]
        loss_video = F.mse_loss(video_prediction.float(), video_target.float())
        loss_audio = F.mse_loss(audio_prediction.float(), audio_target.float())
        loss = self.video_loss_weight * loss_video + self.audio_loss_weight * loss_audio
        return {
            "loss": loss,
            "loss_video": loss_video.detach(),
            "loss_audio": loss_audio.detach(),
            "video_timestep": video_t.detach(),
            "audio_timestep": audio_t.detach(),
        }

    @torch.no_grad()
    def val_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        result = self.train_step(batch)
        return {
            "loss": result["loss"].detach(),
            "loss_video": result["loss_video"],
            "loss_audio": result["loss_audio"],
        }


__all__ = ["MiniMaxH3Trainer"]
