# coding=utf-8
"""MotionCLIPTrainer: contrastive training for the MotionCLIP evaluator.

Mirrors versatilemotion's ``MotionCLIPTrainer`` but exposes the standard
hftrainer Trainer API (``train_step`` / ``val_step``).  The contrastive
loss is computed inside :class:`MotionCLIPModel.forward` itself, including
the all-gather across DDP ranks (preserved from the original
implementation).
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch
from torch import Tensor

from hftrainer.registry import TRAINERS
from hftrainer.trainers.base_trainer import BaseTrainer


def _ensure_list_of_int(value, default_b: int) -> List[int]:
    if value is None:
        return [default_b] * 1
    if isinstance(value, Tensor):
        return value.long().cpu().tolist()
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    return [int(value)]


@TRAINERS.register_module()
class MotionCLIPTrainer(BaseTrainer):
    """Trainer for MotionCLIP contrastive (text <-> motion) learning."""

    def __init__(self, bundle, **kwargs):
        super().__init__(bundle)
        self._extra_kwargs = kwargs  # reserved for future use

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _unpack_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        device = next(self.bundle.motionclip_model.parameters()).device

        motion = batch.get('motion')
        if motion is None:
            raise KeyError(
                "MotionCLIPTrainer expects 'motion' in batch (B, T, D) raw motion."
            )
        if not isinstance(motion, Tensor):
            motion = torch.as_tensor(motion)
        motion = motion.to(device, dtype=torch.float32)

        captions = batch.get('caption') or batch.get('text')
        if captions is None:
            raise KeyError(
                "MotionCLIPTrainer expects 'caption' (or 'text') in batch."
            )
        if isinstance(captions, str):
            captions = [captions]

        num_frames = batch.get('num_frames')
        num_frames = _ensure_list_of_int(num_frames, motion.shape[1])
        if len(num_frames) != motion.shape[0]:
            num_frames = [motion.shape[1]] * motion.shape[0]

        return {
            'motion': motion,
            'captions': list(captions),
            'num_frames': num_frames,
        }

    # ------------------------------------------------------------------
    # Train / val
    # ------------------------------------------------------------------

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        data = self._unpack_batch(batch)
        outputs = self.bundle(
            motion=data['motion'],
            captions=data['captions'],
            num_frames=data['num_frames'],
            return_loss=True,
        )
        loss = outputs.loss
        return {
            'loss': loss,
            'loss_clip': loss.detach() if loss is not None else None,
        }

    @torch.no_grad()
    def val_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        data = self._unpack_batch(batch)
        outputs = self.bundle(
            motion=data['motion'],
            captions=data['captions'],
            num_frames=data['num_frames'],
            return_loss=True,
        )
        return {
            'loss': outputs.loss,
            'text_embeds': outputs.text_embeds,
            'motion_embeds': outputs.motion_embeds,
            'logits_per_text': outputs.logits_per_text,
            'logits_per_motion': outputs.logits_per_motion,
        }
