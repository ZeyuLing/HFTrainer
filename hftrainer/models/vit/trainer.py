"""Classification trainer."""

import torch
import torch.nn.functional as F
from typing import Dict, Any

from hftrainer.trainers.base_trainer import BaseTrainer
from hftrainer.registry import TRAINERS


@TRAINERS.register_module()
class ClassificationTrainer(BaseTrainer):
    """
    Trainer for image classification tasks.

    train_step: pixel_values → logits → cross-entropy loss
    val_step: pixel_values → preds, scores, gts
    """

    def __init__(self, bundle, label_smoothing: float = 0.0, **kwargs):
        super().__init__(bundle)
        self.label_smoothing = label_smoothing

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Args:
            batch: dict with 'pixel_values' (Tensor[B,3,H,W]) and 'labels' (Tensor[B])

        Returns:
            {'loss': Tensor, 'loss_ce': Tensor}
        """
        pixel_values = batch['pixel_values']
        labels = batch['labels']

        logits = self.bundle.forward_features(pixel_values)
        loss = F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)

        return {'loss': loss, 'loss_ce': loss.detach()}

    def val_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Args:
            batch: dict with 'pixel_values' and 'labels'

        Returns:
            {
                'preds': Tensor[B],       # predicted class ids
                'scores': Tensor[B, C],   # softmax probabilities
                'gts': Tensor[B],         # ground truth class ids
                'metas': list[dict]       # optional metadata
            }
        """
        pixel_values = batch['pixel_values']
        labels = batch['labels']

        pred_ids, scores = self.bundle.classify(pixel_values)

        return {
            'preds': pred_ids,
            'scores': scores,
            'gts': labels,
            'metas': batch.get('metas', []),
        }
