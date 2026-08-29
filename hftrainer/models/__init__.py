"""Model bundle primitives and repository-local adapter utilities."""

from hftrainer.models.base_model_bundle import ModelBundle
from hftrainer.models.lora import apply_lora

__all__ = ['ModelBundle', 'apply_lora']
