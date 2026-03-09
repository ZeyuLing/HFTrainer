"""PEFT (LoRA / QLoRA) utilities for ModelBundle."""

from typing import Dict, Any, Optional

import torch.nn as nn


def apply_lora(module: nn.Module, lora_cfg: Optional[Dict[str, Any]] = None) -> nn.Module:
    """
    Apply LoRA to a module using peft.

    Args:
        module: The nn.Module to apply LoRA to.
        lora_cfg: Configuration dict for LoraConfig. Supports:
            - r: int (LoRA rank, default 16)
            - lora_alpha: float (default 32)
            - target_modules: list[str] | str (default 'all-linear')
            - lora_dropout: float (default 0.1)
            - bias: str (default 'none')
            - task_type: str (optional, e.g. 'CAUSAL_LM')

    Returns:
        The module wrapped with LoRA (peft.PeftModel).
    """
    try:
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError:
        raise ImportError("peft is required for LoRA. Install with: pip install peft")

    if lora_cfg is None:
        lora_cfg = {}

    # Map task_type string to peft TaskType enum
    task_type_str = lora_cfg.pop('task_type', None)
    task_type = None
    if task_type_str is not None:
        task_type = getattr(TaskType, task_type_str, None)

    config = LoraConfig(
        r=lora_cfg.get('r', 16),
        lora_alpha=lora_cfg.get('lora_alpha', 32),
        target_modules=lora_cfg.get('target_modules', 'all-linear'),
        lora_dropout=lora_cfg.get('lora_dropout', 0.1),
        bias=lora_cfg.get('bias', 'none'),
        task_type=task_type,
    )

    return get_peft_model(module, config)


def apply_qlora(module: nn.Module, lora_cfg: Optional[Dict[str, Any]] = None) -> nn.Module:
    """
    Apply QLoRA (4-bit quantization + LoRA) to a module.
    Requires bitsandbytes to be installed.
    """
    try:
        import bitsandbytes as bnb
        from peft import prepare_model_for_kbit_training
    except ImportError:
        raise ImportError(
            "bitsandbytes and peft are required for QLoRA. "
            "Install with: pip install bitsandbytes peft"
        )

    module = prepare_model_for_kbit_training(module)
    return apply_lora(module, lora_cfg)
