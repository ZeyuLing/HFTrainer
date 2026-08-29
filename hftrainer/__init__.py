"""HFTrainer's lightweight public API.

Importing :mod:`hftrainer` only creates the framework registries. Components
that pull in PyTorch or Accelerate are loaded when their public attributes are
first accessed, while concrete models, trainers, datasets, and pipelines are
registered explicitly through :func:`register_all_modules` or a config's
``custom_imports`` entry.
"""

from importlib import import_module

from hftrainer.registry import (
    DATASETS,
    EVALUATORS,
    HF_MODELS,
    HOOKS,
    MODEL_BUNDLES,
    MODELS,
    PIPELINES,
    TRAINERS,
    TRANSFORMS,
    VISUALIZERS,
    build_hf_model_from_cfg,
)


_LAZY_IMPORTS = {
    'AccelerateRunner': ('hftrainer.runner.accelerate_runner', 'AccelerateRunner'),
    'BasePipeline': ('hftrainer.pipelines.base_pipeline', 'BasePipeline'),
    'BaseTrainer': ('hftrainer.trainers.base_trainer', 'BaseTrainer'),
    'ModelBundle': ('hftrainer.models.base_model_bundle', 'ModelBundle'),
    'build_pipeline_from_cfg': (
        'hftrainer.pipelines.builder',
        'build_pipeline_from_cfg',
    ),
    'build_runner_from_cfg': ('hftrainer.runner.builder', 'build_runner_from_cfg'),
}


def __getattr__(name):
    """Load compatibility exports without making the package import heavy."""
    target = _LAZY_IMPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attribute_name = target
    value = getattr(import_module(module_name), attribute_name)
    # Cache the resolved object so subsequent accesses have normal module
    # attribute semantics and do not repeat import-system work.
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY_IMPORTS))


def register_all_modules() -> None:
    """Register framework components and the built-in method implementations.

    The operation is idempotent. Applications that only need one custom
    method can instead import that method from ``custom_imports`` and avoid
    loading the complete built-in catalogue.
    """
    from hftrainer.utils.setup_env import register_all_modules as _register

    _register()


__version__ = '0.1.0'

__all__ = [
    'HF_MODELS',
    'MODELS',
    'MODEL_BUNDLES',
    'TRAINERS',
    'PIPELINES',
    'DATASETS',
    'TRANSFORMS',
    'HOOKS',
    'EVALUATORS',
    'VISUALIZERS',
    'build_hf_model_from_cfg',
    'ModelBundle',
    'BasePipeline',
    'BaseTrainer',
    'AccelerateRunner',
    'build_pipeline_from_cfg',
    'build_runner_from_cfg',
    'register_all_modules',
]
