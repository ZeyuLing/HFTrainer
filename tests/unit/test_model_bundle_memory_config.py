import torch
import torch.nn as nn
import pytest

from hftrainer.models.base_model_bundle import ModelBundle
from hftrainer.registry import HF_MODELS


class _DummyBundle(ModelBundle):
    def __init__(self, modules_cfg):
        super().__init__()
        self._build_modules(modules_cfg)


class DummyGradientCheckpointModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4)
        self.gc_enabled = False
        self.gc_kwargs = None

    def gradient_checkpointing_enable(self, **kwargs):
        self.gc_enabled = True
        self.gc_kwargs = kwargs


class DummyEnableGradientCheckpointModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4)
        self.gc_enabled = False
        self.gc_kwargs = None

    def enable_gradient_checkpointing(self, **kwargs):
        self.gc_enabled = True
        self.gc_kwargs = kwargs


class DummyPlainModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4)


HF_MODELS.register_module(
    name='TestGradientCheckpointModule',
    module=DummyGradientCheckpointModule,
    force=True,
)
HF_MODELS.register_module(
    name='TestEnableGradientCheckpointModule',
    module=DummyEnableGradientCheckpointModule,
    force=True,
)
HF_MODELS.register_module(
    name='TestPlainModule',
    module=DummyPlainModule,
    force=True,
)


def test_module_dtype_cast_from_config():
    bundle = _DummyBundle(
        {
            'model': dict(
                type='TestGradientCheckpointModule',
                module_dtype='bf16',
            ),
        }
    )

    assert next(bundle.model.parameters()).dtype == torch.bfloat16
    assert bundle.get_module_build_cfg('model')['module_dtype'] == 'bf16'


def test_gradient_checkpointing_enable_hook():
    bundle = _DummyBundle(
        {
            'model': dict(
                type='TestGradientCheckpointModule',
                gradient_checkpointing=True,
            ),
        }
    )

    assert bundle.model.gc_enabled is True
    assert bundle.model.gc_kwargs == {}


def test_gradient_checkpointing_enable_hook_with_kwargs():
    bundle = _DummyBundle(
        {
            'model': dict(
                type='TestEnableGradientCheckpointModule',
                gradient_checkpointing=dict(use_reentrant=False),
            ),
        }
    )

    assert bundle.model.gc_enabled is True
    assert bundle.model.gc_kwargs == {'use_reentrant': False}


def test_gradient_checkpointing_unsupported_module_raises():
    with pytest.raises(ValueError, match='does not expose gradient checkpointing hooks'):
        _DummyBundle(
            {
                'model': dict(
                    type='TestPlainModule',
                    gradient_checkpointing=True,
                ),
            }
        )
