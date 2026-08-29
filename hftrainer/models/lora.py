"""Repository-local low-rank adaptation layers.

The implementation deliberately operates on ordinary :class:`torch.nn.Linear`
modules.  Model packages therefore keep one execution graph for full fine
tuning, adapter training, checkpointing, and merged inference without handing
ownership of the model to a second model framework.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence

import torch
import torch.nn as nn


@dataclass(frozen=True)
class LoRAConfig:
    """Configuration shared by every local LoRA-injected linear layer."""

    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.0
    target_modules: str | Sequence[str] = 'all-linear'
    bias: str = 'none'

    @classmethod
    def from_dict(cls, value: Optional[Dict[str, Any]]) -> 'LoRAConfig':
        cfg = dict(value or {})
        # ``task_type`` used to be consumed by the old adapter package.  It
        # never changed the matrix update itself, so accepting it keeps old
        # recipes readable without coupling the implementation to that API.
        cfg.pop('task_type', None)
        rank = int(cfg.pop('r', cfg.pop('rank', 16)))
        alpha = float(cfg.pop('lora_alpha', cfg.pop('alpha', 32.0)))
        dropout = float(cfg.pop('lora_dropout', cfg.pop('dropout', 0.0)))
        target_modules = cfg.pop('target_modules', 'all-linear')
        bias = str(cfg.pop('bias', 'none'))
        if cfg:
            unknown = ', '.join(sorted(cfg))
            raise ValueError(f'Unsupported local LoRA options: {unknown}')
        if rank <= 0:
            raise ValueError('LoRA rank must be positive.')
        if alpha <= 0:
            raise ValueError('LoRA alpha must be positive.')
        if not 0.0 <= dropout < 1.0:
            raise ValueError('LoRA dropout must be in [0, 1).')
        if bias not in {'none', 'all', 'lora_only'}:
            raise ValueError("LoRA bias must be 'none', 'all', or 'lora_only'.")
        return cls(rank, alpha, dropout, target_modules, bias)


class LoRALinear(nn.Module):
    """A linear layer plus a trainable low-rank residual."""

    def __init__(self, base_layer: nn.Linear, config: LoRAConfig):
        super().__init__()
        self.base_layer = base_layer
        self.rank = config.rank
        self.alpha = config.alpha
        self.scaling = config.alpha / config.rank
        self.dropout = nn.Dropout(config.dropout) if config.dropout else nn.Identity()
        self.lora_A = nn.Linear(base_layer.in_features, config.rank, bias=False)
        self.lora_B = nn.Linear(config.rank, base_layer.out_features, bias=False)
        self.merged = False

        nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B.weight)
        self.base_layer.requires_grad_(False)
        if base_layer.bias is not None and config.bias in {'all', 'lora_only'}:
            base_layer.bias.requires_grad_(True)

        # Match the existing layer immediately; callers can still cast the
        # complete model afterwards through ModelBundle.module_dtype.
        self.lora_A.to(device=base_layer.weight.device, dtype=base_layer.weight.dtype)
        self.lora_B.to(device=base_layer.weight.device, dtype=base_layer.weight.dtype)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        result = self.base_layer(inputs)
        if self.merged:
            return result
        update = self.lora_B(self.lora_A(self.dropout(inputs)))
        return result + update * self.scaling

    def merged_weight(self) -> torch.Tensor:
        update = self.lora_B.weight @ self.lora_A.weight
        return self.base_layer.weight + update.to(self.base_layer.weight) * self.scaling

    def merge(self) -> nn.Linear:
        if not self.merged:
            self.base_layer.weight.data.copy_(self.merged_weight().data)
            self.merged = True
        return self.base_layer


def _matches_target(name: str, target_modules: str | Sequence[str]) -> bool:
    if target_modules == 'all-linear':
        return True
    if isinstance(target_modules, str):
        target_modules = (target_modules,)
    return any(name == target or name.endswith(f'.{target}') for target in target_modules)


def _iter_named_linears(module: nn.Module) -> Iterable[tuple[str, nn.Linear]]:
    for name, child in module.named_modules():
        if isinstance(child, nn.Linear):
            yield name, child


def _parent_and_child(module: nn.Module, qualified_name: str) -> tuple[nn.Module, str]:
    parts = qualified_name.split('.')
    parent = module
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def apply_lora(module: nn.Module, lora_cfg: Optional[Dict[str, Any]] = None) -> nn.Module:
    """Inject local LoRA layers and freeze non-adapter parameters.

    The function mutates and returns ``module`` so its public model type and
    state-dict prefix remain stable.  This is important for existing HFTrainer
    artifacts and avoids a wrapper-owned checkpoint namespace.
    """

    config = LoRAConfig.from_dict(lora_cfg)
    if is_lora_model(module):
        raise ValueError(f'{type(module).__name__} already contains LoRA layers.')

    candidates = [
        (name, layer)
        for name, layer in _iter_named_linears(module)
        if _matches_target(name, config.target_modules)
    ]
    if not candidates:
        raise ValueError(
            f'No linear layers matched target_modules={config.target_modules!r} '
            f'in {type(module).__name__}.'
        )

    module.requires_grad_(False)
    for name, layer in candidates:
        parent, child_name = _parent_and_child(module, name)
        setattr(parent, child_name, LoRALinear(layer, config))

    if config.bias == 'all':
        for name, parameter in module.named_parameters():
            if name.endswith('.bias'):
                parameter.requires_grad_(True)

    # Plain attributes intentionally avoid becoming persistent checkpoint
    # tensors while still making ownership and configuration inspectable.
    object.__setattr__(module, '_hftrainer_lora_config', copy.deepcopy(config))
    return module


def is_lora_model(module: nn.Module) -> bool:
    module = _unwrap_module(module)
    return any(isinstance(child, LoRALinear) for child in module.modules())


def _unwrap_module(module: nn.Module) -> nn.Module:
    while hasattr(module, 'module') and isinstance(module.module, nn.Module):
        module = module.module
    return module


def get_lora_state_dict(
    module: nn.Module,
    state_dict: Optional[Dict[str, Any]] = None,
    adapter_name: str = 'default',
    *,
    fold_scaling: bool = False,
) -> Dict[str, Any]:
    """Return adapter tensors using stable, framework-owned key names.

    When ``fold_scaling`` is true, ``alpha / rank`` is multiplied into every
    ``lora_B`` tensor. The resulting A/B pair can be consumed by inference
    loaders whose fusion contract is simply ``B @ A``. Callers that later
    resume local training must pass ``scaling_folded=True`` to
    :func:`set_lora_state_dict`.
    """

    del adapter_name
    module = _unwrap_module(module)
    full = state_dict if state_dict is not None else module.state_dict()
    result = {
        key: value
        for key, value in full.items()
        if '.lora_A.' in key or '.lora_B.' in key
    }
    config = getattr(module, '_hftrainer_lora_config', None)
    if isinstance(config, LoRAConfig):
        if config.bias == 'all':
            result.update({key: value for key, value in full.items() if key.endswith('.bias')})
        elif config.bias == 'lora_only':
            for name, child in module.named_modules():
                if isinstance(child, LoRALinear) and child.base_layer.bias is not None:
                    result[f'{name}.base_layer.bias'] = child.base_layer.bias.detach()
        if fold_scaling:
            result = {
                key: value * (config.alpha / config.rank)
                if key.endswith('.lora_B.weight')
                else value
                for key, value in result.items()
            }
    elif fold_scaling:
        raise RuntimeError(
            'Cannot fold LoRA scaling because the local adapter configuration '
            'is missing from the model.'
        )
    return result


def set_lora_state_dict(
    module: nn.Module,
    state_dict: Dict[str, Any],
    adapter_name: str = 'default',
    *,
    scaling_folded: bool = False,
):
    """Load a local adapter checkpoint and reject silent key loss.

    ``scaling_folded`` reverses the portable inference representation emitted
    by :func:`get_lora_state_dict(fold_scaling=True)` before copying tensors
    into trainable :class:`LoRALinear` modules.
    """

    del adapter_name
    module = _unwrap_module(module)
    if not is_lora_model(module):
        raise TypeError(f'{type(module).__name__} has no local LoRA layers.')
    expected = set(get_lora_state_dict(module))
    supplied = set(state_dict)
    missing = sorted(expected - supplied)
    unexpected = sorted(supplied - set(module.state_dict()))
    if missing or unexpected:
        raise RuntimeError(
            f'Invalid LoRA state dict: missing={missing[:8]}, unexpected={unexpected[:8]}'
        )
    values = state_dict
    if scaling_folded:
        config = getattr(module, '_hftrainer_lora_config', None)
        if not isinstance(config, LoRAConfig):
            raise RuntimeError(
                'Cannot unfold LoRA scaling because the local adapter '
                'configuration is missing from the model.'
            )
        scaling = config.alpha / config.rank
        values = {
            key: value / scaling if key.endswith('.lora_B.weight') else value
            for key, value in state_dict.items()
        }
    return module.load_state_dict(values, strict=False)


def looks_like_lora_state_dict(state_dict: Dict[str, Any]) -> bool:
    return bool(state_dict) and any(
        '.lora_A.' in key or '.lora_B.' in key for key in state_dict
    )


def _merge_children(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, LoRALinear):
            setattr(module, name, child.merge())
        else:
            _merge_children(child)


def merge_lora(module: nn.Module) -> nn.Module:
    """Merge every local adapter into its base weight in place."""

    module = _unwrap_module(module)
    if not is_lora_model(module):
        raise TypeError(f'{type(module).__name__} has no local LoRA layers.')
    _merge_children(module)
    if hasattr(module, '_hftrainer_lora_config'):
        delattr(module, '_hftrainer_lora_config')
    return module


def apply_qlora(module: nn.Module, lora_cfg: Optional[Dict[str, Any]] = None) -> nn.Module:
    """Reject implicit 4-bit backends; quantization must be locally implemented."""

    del module, lora_cfg
    raise NotImplementedError(
        'QLoRA is not exposed until HFTrainer owns and validates a local 4-bit '
        'linear kernel. Use local LoRA or full fine tuning.'
    )


__all__ = [
    'LoRAConfig',
    'LoRALinear',
    'apply_lora',
    'apply_qlora',
    'get_lora_state_dict',
    'is_lora_model',
    'looks_like_lora_state_dict',
    'merge_lora',
    'set_lora_state_dict',
]
