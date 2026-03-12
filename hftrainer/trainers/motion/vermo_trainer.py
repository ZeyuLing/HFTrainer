"""VerMo trainer."""

from __future__ import annotations

from typing import Any, Dict

from hftrainer.registry import TRAINERS
from hftrainer.trainers.base_trainer import BaseTrainer


@TRAINERS.register_module()
class VermoTrainer(BaseTrainer):
    """Trainer for VerMo causal multimodal generation."""

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        outputs = self.bundle.forward_lm(batch)
        return {'loss': outputs.loss, 'loss_lm': outputs.loss.detach()}
