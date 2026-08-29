"""Reusable image-classification training and inference contracts."""

from hftrainer.tasks.image_classification.pipeline import ClassificationPipeline
from hftrainer.tasks.image_classification.trainer import ClassificationTrainer

__all__ = ['ClassificationPipeline', 'ClassificationTrainer']
