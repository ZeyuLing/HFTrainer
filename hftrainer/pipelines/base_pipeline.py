"""Base pipeline class."""

from abc import ABC, abstractmethod
from typing import Any

import torch

from hftrainer.models.base_model_bundle import ModelBundle


class BasePipeline(ABC):
    """
    Abstract base class for all inference pipelines.

    Pipelines hold a ModelBundle and assemble the inference forward graph.
    All atomic forward functions are called from the bundle (shared with Trainer).
    """

    def __init__(self, bundle: ModelBundle, **kwargs):
        self.bundle = bundle
        self.bundle.eval()

    @classmethod
    def from_checkpoint(cls, bundle_cls, bundle_cfg: dict, ckpt_path: str, **kwargs):
        """
        Build pipeline by loading a checkpoint into a ModelBundle.

        Args:
            bundle_cls: the ModelBundle subclass to instantiate
            bundle_cfg: config dict for the bundle
            ckpt_path: path to checkpoint file or directory
            **kwargs: additional args passed to cls.__init__
        """
        import torch
        bundle = bundle_cls(bundle_cfg)
        state_dict = torch.load(ckpt_path, map_location='cpu')
        bundle.load_state_dict_selective(state_dict)
        bundle.eval()
        return cls(bundle=bundle, **kwargs)

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Any:
        """Run inference."""
