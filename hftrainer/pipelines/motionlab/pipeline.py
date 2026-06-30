"""MotionLab text-to-motion pipeline."""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import torch

from hftrainer.pipelines.base_pipeline import BasePipeline
from hftrainer.registry import PIPELINES


@PIPELINES.register_module()
class MotionLabPipeline(BasePipeline):
    """Inference pipeline for the MotionLab bundle."""

    BUNDLE_CLS = "hftrainer.models.motion.motionlab.MotionLabBundle"

    def __init__(self, bundle, device: Optional[str] = None, **kwargs):
        super().__init__(bundle, **kwargs)
        if device is not None:
            self.to(device)

    def to(self, device):
        self.bundle.to_device(device)
        return self

    @property
    def device(self) -> torch.device:
        return self.bundle.device

    @staticmethod
    def clamp_length(n_frames: int, min_length: int = 40, max_length: int = 196) -> int:
        length = (int(n_frames) // 4) * 4
        return max(min_length, min(max_length, length))

    def _encode_text(self, texts):
        text_encoder = self.bundle.text_encoder
        device = self.device
        if hasattr(text_encoder, "tokenizer") and hasattr(text_encoder, "text_model"):
            tokenizer = text_encoder.tokenizer
            max_length = getattr(text_encoder, "max_length", tokenizer.model_max_length)
            text_inputs = tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            input_ids = text_inputs.input_ids.to(device)
            text_model = text_encoder.text_model
            model_device = getattr(text_model, "device", device)
            input_ids = input_ids.to(model_device)
            name = getattr(text_encoder, "name", "")
            if hasattr(text_model, "get_text_features"):
                pooled = text_model.get_text_features(input_ids=input_ids)
                if not torch.is_tensor(pooled):
                    pooled = getattr(pooled, "pooler_output", None)
                if pooled is None:
                    raise TypeError("CLIP get_text_features did not return a tensor")
                pooled = pooled.to(device).unsqueeze(1)
                if name == "clip_hidden":
                    if hasattr(text_model, "text_model"):
                        hidden = text_model.text_model(input_ids).last_hidden_state
                    else:
                        hidden = text_model(input_ids).last_hidden_state
                    return (pooled, hidden.to(device))
                return (pooled,)
            encoded = text_model(input_ids)
            hidden = getattr(encoded, "last_hidden_state", None)
            pooler = getattr(encoded, "pooler_output", None)
            if hidden is not None and pooler is not None and name == "clip_hidden":
                return (pooler.to(device).unsqueeze(1), hidden.to(device))
            if hidden is not None:
                return (hidden.to(device),)

        encoded = text_encoder(texts)
        if isinstance(encoded, (tuple, list)):
            return tuple(encoded)
        last_hidden = getattr(encoded, "last_hidden_state", None)
        pooler = getattr(encoded, "pooler_output", None)
        if last_hidden is not None and pooler is not None:
            return (pooler.unsqueeze(1), last_hidden)
        if last_hidden is not None:
            return (last_hidden,)
        if torch.is_tensor(encoded):
            return (encoded.unsqueeze(1) if encoded.ndim == 2 else encoded,)
        raise TypeError(f"Unsupported text encoder output type: {type(encoded)!r}")

    @torch.no_grad()
    def infer_t2m(
        self,
        captions: Sequence[str],
        lengths: Sequence[int],
        stage: str = "demo",
        num_steps: Optional[int] = None,
    ) -> List[np.ndarray]:
        if len(captions) != len(lengths):
            raise ValueError("captions and lengths must have equal length")
        cfg = self.bundle.cfg
        denoiser = self.bundle.denoiser
        scheduler = self.bundle.scheduler
        device = self.device

        lengths = [self.clamp_length(x) for x in lengths]
        bsz = len(captions)
        max_len = max(lengths)
        noisy_latents = torch.randn((bsz, max_len, 263), device=device, dtype=torch.float32)

        if num_steps is not None:
            steps = int(num_steps)
        elif stage == "demo":
            steps = int(cfg.model.scheduler.num_demo_steps)
        else:
            steps = int(cfg.model.scheduler.num_eval_steps)
        scheduler.set_timesteps(num_inference_steps=steps, device=device)

        model_type = str(getattr(cfg.model, "model_type", ""))
        guidance_scale = float(getattr(cfg.model, "text_guidance_scale", 1.0))
        text_lengths = [0] * bsz + [77] * bsz
        if model_type == "rfmotion_seperate":
            instructions = None
        else:
            uncond_instruction = self._encode_text(["reconstruct given masked source motion."])[0][0]
            text_instruction = self._encode_text(["generate motion by given text."])[0][0]
            instructions = torch.cat(
                [uncond_instruction.repeat(bsz, 1), text_instruction.repeat(bsz, 1)],
                dim=0,
            ).to(device)

        text = self._encode_text([""] * bsz + list(captions))
        for t in scheduler.timesteps.to(torch.int32):
            if int(t.item()) == 0:
                continue
            latent_model_input = torch.cat([noisy_latents] * 2)
            v_pred = denoiser(
                instructions=instructions,
                hidden_states=latent_model_input,
                timestep=t,
                text=text,
                text_lengths=text_lengths,
                hint=None,
                hint_lengths=None,
                style=None,
                style_lengths=None,
                content=None,
                content_lengths=None,
                source_motion=None,
                source_lengths=None,
                source_lengths_z=None,
                target_lengths=list(lengths) + list(lengths),
                target_lengths_z=list(lengths) + list(lengths),
                return_dict=False,
            )[0]
            v_uncond, v_cond = v_pred.chunk(2)
            v_pred = v_uncond + guidance_scale * (v_cond - v_uncond)
            noisy_latents = scheduler.step(v_pred, t, noisy_latents, return_dict=False)[0]

        pred = self.bundle.denormalize(noisy_latents).detach().cpu().numpy().astype(np.float32)
        return [pred[i, : lengths[i]] for i in range(bsz)]

    @torch.no_grad()
    def infer_tp2m(
        self,
        captions: Sequence[str],
        lengths: Sequence[int],
        gt_features: Sequence[np.ndarray],
        condition_num_frames: int,
        stage: str = "eval",
        num_steps: Optional[int] = None,
        motionlab_condition_type: str = "text_hint",
    ) -> List[np.ndarray]:
        """Generate MotionLab TP2M samples from text plus a GT HML263 prefix."""
        if not (len(captions) == len(lengths) == len(gt_features)):
            raise ValueError("captions, lengths, and gt_features must have equal length")
        if int(condition_num_frames) < 1:
            raise ValueError("condition_num_frames must be >= 1")

        from scripts.eval.motionlab_infer_hml3d263 import _sample_text_batch

        target_lengths = [self.clamp_length(x) for x in lengths]
        gt_batch = [
            np.asarray(arr[:length], dtype=np.float32)
            for arr, length in zip(gt_features, target_lengths)
        ]
        pred_norm = _sample_text_batch(
            self.bundle.cfg,
            self.bundle.text_encoder,
            self.bundle.denoiser,
            self.bundle.scheduler,
            list(captions),
            target_lengths,
            self.device,
            stage,
            num_steps,
            int(condition_num_frames),
            gt_batch,
            self.bundle.mean_motion,
            self.bundle.std_motion,
            motionlab_condition_type,
        )
        pred = self.bundle.denormalize(pred_norm).detach().cpu().numpy().astype(np.float32)
        return [pred[i, : target_lengths[i]] for i in range(len(target_lengths))]

    def __call__(self, captions, lengths, **kwargs):
        if kwargs.pop("tp2m", False):
            return self.infer_tp2m(captions, lengths, **kwargs)
        return self.infer_t2m(captions, lengths, **kwargs)
