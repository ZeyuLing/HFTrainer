"""Construct optimizers from PyTorch or an explicitly selected external class."""

from __future__ import annotations

import importlib
import re

import torch
from torch import nn


def build_optimizer(config, parameters):
    kwargs = dict(config)
    name = kwargs.pop('type')
    if not isinstance(name, str) or not name:
        raise ValueError('optimizer.type must be a PyTorch class name or a qualified import path.')
    module_name, separator, class_name = name.rpartition('.')
    if separator:
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            raise ImportError(f"Cannot import optimizer {name}: {exc}") from exc
    else:
        module, class_name = torch.optim, name

    cls = getattr(module, class_name, None)
    if not isinstance(cls, type) or not issubclass(cls, torch.optim.Optimizer):
        hint = ' Install PyTorch 2.9 or newer.' if name in {'Muon', 'torch.optim.Muon'} else ''
        raise ValueError(f"Unknown optimizer class: {name}.{hint}")
    return cls(parameters, **kwargs)


def uses_muon(config):
    return config.get('type') in {'Muon', 'torch.optim.Muon'}


def build_optimizers(config, parameters, model):
    """Apply native Muon to hidden matrices and AdamW to the remaining weights."""
    if not uses_muon(config):
        return {'default': build_optimizer(config, parameters)}

    muon_config = dict(config)
    adamw_config = dict(muon_config.pop('adamw', None) or {})
    for key in ('lr', 'weight_decay'):
        if key in muon_config:
            adamw_config.setdefault(key, muon_config[key])
    if 'betas' in muon_config:
        adamw_config.setdefault('betas', muon_config.pop('betas'))
    excludes = muon_config.pop('exclude', None) or ()
    if isinstance(excludes, str):
        excludes = [excludes]
    patterns = [re.compile(pattern) for pattern in excludes]
    boundary = re.compile(
        r'(?:^|\.)(?:embedding\w*|\w*_embed(?:ding)?s?|embed_\w+|embed|embd|'
        r'wte|wpe|\w*norm\w*|lm_head|head|classifier|output(?:_proj|_projection)?|'
        r'proj_out|\w+_proj_out|null_\w+|pos_tokens|cls_token)(?:\.|$)',
        re.IGNORECASE,
    )

    excluded_ids = set()
    for name, param in model.named_parameters(remove_duplicate=False):
        if boundary.search(name) or any(pattern.search(name) for pattern in patterns):
            excluded_ids.add(id(param))
    for module in model.modules():
        if isinstance(module, (nn.Embedding, nn.EmbeddingBag)):
            excluded_ids.update(id(param) for param in module.parameters())
        output_embeddings = getattr(module, 'get_output_embeddings', None)
        if callable(output_embeddings):
            output = output_embeddings()
            if isinstance(output, nn.Module):
                excluded_ids.update(id(param) for param in output.parameters())

    matrices = []
    remaining = []
    for param in dict.fromkeys(parameters):
        group = matrices if param.ndim == 2 and id(param) not in excluded_ids else remaining
        group.append(param)
    if not matrices:
        raise ValueError('Muon needs at least one trainable, two-dimensional hidden-layer weight.')

    optimizers = {'default': build_optimizer(muon_config, matrices)}
    if remaining:
        optimizers['adamw'] = build_optimizer(dict(adamw_config, type='AdamW'), remaining)
    return optimizers


def optimizer_classes(optimizers):
    """Identify the underlying optimizer even after Accelerate wraps it."""
    classes = {}
    for name, optimizer in optimizers.items():
        implementation = type(getattr(optimizer, 'optimizer', optimizer))
        classes[name] = f'{implementation.__module__}.{implementation.__qualname__}'
    return classes
