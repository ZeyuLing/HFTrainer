"""
BaseTrainer: abstract base class for all task trainers.

Trainers are responsible for:
  - Assembling the training/validation forward graph
  - Computing the loss
  - Returning structured output dicts

Trainers do NOT handle:
  - Optimizer creation / step (done by AccelerateRunner)
  - Checkpoint saving/loading (done by CheckpointHook)
  - Distributed communication (done by Accelerator)
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

import torch.nn as nn

from hftrainer.models.base_model_bundle import ModelBundle


class BaseTrainer(nn.Module, ABC):
    """
    Abstract base class for all trainers.

    Subclasses must implement:
      - train_step(batch) -> dict  with at least {'loss': Tensor}
      - val_step(batch) -> dict    with task-specific keys

    The trainer holds a reference to the ModelBundle and calls its
    atomic forward methods (encode_text, predict_noise, etc.).
    The accelerator instance is injected by AccelerateRunner after prepare().
    """

    def __init__(self, bundle: ModelBundle, **kwargs):
        super().__init__()
        self.bundle = bundle
        self.accelerator = None  # injected by AccelerateRunner

    @abstractmethod
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform one training step.

        Args:
            batch: dict from DataLoader

        Returns:
            dict with at least {'loss': Tensor}. May include additional
            loss components for logging (e.g. 'loss_mse', 'loss_kl', ...).
        """

    def val_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform one validation step.

        Args:
            batch: dict from DataLoader

        Returns:
            Task-specific dict. See CLAUDE.md for per-task key conventions.
            Default implementation raises NotImplementedError.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement val_step(). "
            "Override this method to enable validation."
        )

    def get_bundle(self) -> ModelBundle:
        """Return the ModelBundle held by this trainer."""
        return self.bundle

    def forward(self, *args, **kwargs):
        """Redirect forward() to train_step() for compatibility."""
        return self.train_step(*args, **kwargs)
