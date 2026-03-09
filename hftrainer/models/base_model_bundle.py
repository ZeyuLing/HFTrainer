"""
ModelBundle base class.

ModelBundle holds all sub-modules for a task and serves as the shared core
between Trainer and Pipeline. It handles:
  - Module instantiation via HF_MODELS registry
  - Per-module trainable / save_ckpt control
  - LoRA injection via peft
  - Selective checkpoint save / load
"""

import copy
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn


class ModelBundle(nn.Module):
    """
    Base class for all task-specific ModelBundles.

    Subclasses should call self._build_modules(modules_cfg) in __init__
    to instantiate and configure all sub-modules.

    Sub-modules declared with trainable=False are kept in eval mode during
    training (overridden in train()). Sub-modules with save_ckpt=True are
    included in state_dict_to_save() / load_state_dict_selective().
    """

    def __init__(self):
        super().__init__()
        self._save_ckpt_modules: List[str] = []
        self._trainable_modules: List[str] = []
        self._frozen_modules: List[str] = []
        # Non-nn.Module attributes (e.g. tokenizers)
        self._extra_attributes: Dict[str, Any] = {}

    def _build_modules(self, modules_cfg: dict):
        """
        Instantiate all sub-modules from config dict.

        Each entry in modules_cfg is a sub-module config with optional keys:
          - trainable: bool | 'lora' (default True)
          - save_ckpt: bool (default: True if trainable else False)
          - All other keys are passed to HF_MODELS.build()
        """
        from hftrainer.registry import HF_MODELS
        from hftrainer.models.peft_utils import apply_lora

        self._save_ckpt_modules = []
        self._trainable_modules = []
        self._frozen_modules = []

        for name, sub_cfg in modules_cfg.items():
            sub_cfg = copy.deepcopy(sub_cfg)
            trainable = sub_cfg.pop('trainable', True)
            lora_cfg = sub_cfg.pop('lora_cfg', None)
            save_ckpt = sub_cfg.pop('save_ckpt', True if trainable else False)

            # Build the module
            module = HF_MODELS.build(sub_cfg)

            # Apply LoRA if requested
            if trainable == 'lora':
                module = apply_lora(module, lora_cfg or {})
                trainable = True
                save_ckpt = True  # always save lora weights

            if isinstance(module, nn.Module):
                if not trainable:
                    module.requires_grad_(False)
                    self._frozen_modules.append(name)
                else:
                    self._trainable_modules.append(name)

                setattr(self, name, module)

                if save_ckpt:
                    self._save_ckpt_modules.append(name)
            else:
                # Non-nn.Module (e.g. tokenizer, scheduler with no params)
                # Store as plain attribute
                self._extra_attributes[name] = module
                object.__setattr__(self, name, module)

    def trainable_parameters(self) -> List[torch.nn.Parameter]:
        """Return all trainable parameters for optimizer construction."""
        params = []
        for name in self._trainable_modules:
            module = getattr(self, name)
            if isinstance(module, nn.Module):
                params.extend(module.parameters())
        return params

    def trainable_named_parameters(self):
        """Yield (name, param) for all trainable parameters."""
        for mod_name in self._trainable_modules:
            module = getattr(self, mod_name)
            if isinstance(module, nn.Module):
                for param_name, param in module.named_parameters():
                    yield f"{mod_name}.{param_name}", param

    def state_dict_to_save(self) -> Dict[str, dict]:
        """
        Return a nested state dict containing only save_ckpt=True modules.
        Format: {module_name: state_dict}
        """
        sd = {}
        for name in self._save_ckpt_modules:
            module = getattr(self, name)
            if isinstance(module, nn.Module):
                sd[name] = module.state_dict()
        return sd

    def load_state_dict_selective(self, state_dict: Dict[str, dict], strict: bool = False):
        """
        Load only modules that are present in state_dict.
        Modules not present in state_dict are left unchanged.

        Args:
            state_dict: {module_name: module_state_dict} or flat state dict
            strict: whether to enforce strict key matching per module
        """
        if not state_dict:
            return

        # Detect format: nested {module_name: {weight_name: tensor}} or flat
        first_val = next(iter(state_dict.values()))
        if isinstance(first_val, torch.Tensor):
            # Flat state dict — try to split by module name
            nested = {}
            for key, val in state_dict.items():
                parts = key.split('.', 1)
                if len(parts) == 2 and hasattr(self, parts[0]):
                    mod_name, param_name = parts
                    if mod_name not in nested:
                        nested[mod_name] = {}
                    nested[mod_name][param_name] = val
                else:
                    # Try direct load
                    nested[key] = {key: val}
            state_dict = nested

        for name, sd in state_dict.items():
            if hasattr(self, name):
                module = getattr(self, name)
                if isinstance(module, nn.Module):
                    missing, unexpected = module.load_state_dict(sd, strict=strict)
                    if missing:
                        from hftrainer.utils.logger import get_logger
                        logger = get_logger()
                        logger.warning(f"Missing keys in module '{name}': {missing[:5]}...")
                    if unexpected:
                        from hftrainer.utils.logger import get_logger
                        logger = get_logger()
                        logger.warning(f"Unexpected keys in module '{name}': {unexpected[:5]}...")

    def train(self, mode: bool = True):
        """
        Override train() to keep frozen modules always in eval mode.
        This ensures BatchNorm / Dropout in frozen modules behave correctly during training.
        """
        super().train(mode)
        if mode:
            for name in self._frozen_modules:
                module = getattr(self, name, None)
                if isinstance(module, nn.Module):
                    module.eval()
        return self

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "ModelBundle.forward() is not implemented directly. "
            "Use the atomic forward methods (encode_text, predict_noise, etc.) instead."
        )
