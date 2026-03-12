"""Compatibility package for motion trainer imports."""

from hftrainer.models.prism.trainer import PrismTrainer
from hftrainer.models.vermo.trainer import VermoTrainer

__all__ = ['PrismTrainer', 'VermoTrainer']
