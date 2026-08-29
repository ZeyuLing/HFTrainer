"""
Registry system for hftrainer.

Registries:
  MODEL_COMPONENTS — repository-owned model and scheduler classes
  MODEL_BUNDLES — ModelBundle subclasses
  TRAINERS      — Trainer subclasses
  PIPELINES     — Pipeline subclasses
  DATASETS      — Dataset subclasses
  TRANSFORMS    — Transform classes
  HOOKS         — Hook classes
  EVALUATORS    — Evaluator classes
  VISUALIZERS   — Visualizer classes
"""

import copy
from collections.abc import Mapping
from mmengine.registry import Registry


class RepositoryRegistry(Registry):
    """Registry whose string lookup is limited to explicitly registered names.

    MMEngine's default :meth:`Registry.get` treats an unknown dotted string as
    an import path.  That is convenient for a general plugin system, but it
    violates HFTrainer's ownership boundary: a config such as
    ``types.SimpleNamespace`` or ``some_model_package.Model`` must not turn
    into executable code merely because that package is installed.  Extension
    modules remain supported, but they must explicitly register their classes
    (for example through ``custom_imports``) before a config can build them.
    """

    def get(self, key: str):
        if not isinstance(key, str):
            raise TypeError(
                'The key argument of RepositoryRegistry.get must be a string, '
                f'got {type(key).__name__}.'
            )
        return self.module_dict.get(key)

    def build(self, cfg, *args, **kwargs):
        """Build only classes that were explicitly registered by identity."""

        entries = cfg if isinstance(cfg, (list, tuple)) else (cfg,)
        for entry in entries:
            if not isinstance(entry, Mapping) or 'type' not in entry:
                continue
            obj_type = entry['type']
            if isinstance(obj_type, str):
                continue
            if not any(obj_type is value for value in self.module_dict.values()):
                raise KeyError(
                    f"{obj_type!r} is not explicitly registered in '{self.name}'. "
                    'Register the local class before using it in a config.'
                )
        return super().build(cfg, *args, **kwargs)

# Map string shorthand to torch.dtype
_DTYPE_MAP = {
    'fp32': 'torch.float32',
    'fp16': 'torch.float16',
    'bf16': 'torch.bfloat16',
    'float32': 'torch.float32',
    'float16': 'torch.float16',
    'bfloat16': 'torch.bfloat16',
}


def _resolve_dtype(kwargs: dict) -> dict:
    """Convert 'torch_dtype' string shortcuts to actual torch.dtype objects."""
    if 'torch_dtype' in kwargs:
        val = kwargs['torch_dtype']
        if isinstance(val, str):
            import torch
            resolved = _DTYPE_MAP.get(val, val)
            # Support 'torch.bfloat16' style strings
            if isinstance(resolved, str) and resolved.startswith('torch.'):
                attr = resolved.split('.', 1)[1]
                kwargs['torch_dtype'] = getattr(torch, attr)
            else:
                kwargs['torch_dtype'] = resolved
    # Also handle the generic ``dtype`` spelling used by local components.
    if 'dtype' in kwargs:
        val = kwargs['dtype']
        if isinstance(val, str):
            import torch
            resolved = _DTYPE_MAP.get(val, val)
            if isinstance(resolved, str) and resolved.startswith('torch.'):
                attr = resolved.split('.', 1)[1]
                kwargs['dtype'] = getattr(torch, attr)
            else:
                kwargs['dtype'] = resolved
    return kwargs


def build_model_component_from_cfg(cfg, registry):
    """
    Build a repository-local model component from a config dictionary.

    Handles three loading patterns:
      - from_pretrained: cls.from_pretrained(**from_pretrained_kwargs)
      - from_config:     cls.from_config(**from_config_kwargs)
      - from_single_file: cls.from_single_file(**from_single_file_kwargs)
      - fallback:        cls(**remaining_kwargs)

    The 'type' key is used to look up the class in the registry.
    Supports 'torch_dtype' as string shorthand: 'fp32', 'fp16', 'bf16'.
    """
    cfg = copy.deepcopy(cfg)
    obj_type = cfg.pop('type')

    # Model execution must have one visible, repository-owned implementation.
    # Falling back to arbitrary import discovery made a config appear local
    # while silently handing execution to whichever package happened to be
    # installed in the environment.
    if isinstance(obj_type, str):
        cls = registry.get(obj_type)
        if cls is None:
            available = sorted(registry.module_dict)
            raise KeyError(
                f"Model component '{obj_type}' is not registered in "
                f"'{registry.name}'. Import its hftrainer.models.<implementation> "
                f"package through config.custom_imports. Available: {available}"
            )
    else:
        cls = obj_type
        if not any(cls is value for value in registry.module_dict.values()):
            raise KeyError(
                f"{cls!r} is not explicitly registered in '{registry.name}'."
            )

    # Check for special loading patterns
    if 'from_pretrained' in cfg:
        kwargs = _resolve_dtype(cfg.pop('from_pretrained'))
        return cls.from_pretrained(**kwargs)
    elif 'from_config' in cfg:
        kwargs = _resolve_dtype(cfg.pop('from_config'))
        return cls.from_config(**kwargs)
    elif 'from_single_file' in cfg:
        kwargs = _resolve_dtype(cfg.pop('from_single_file'))
        return cls.from_single_file(**kwargs)
    else:
        return cls(**cfg)


# Core registries
MODEL_COMPONENTS = RepositoryRegistry(
    'model_component', build_func=build_model_component_from_cfg
)
MODELS = MODEL_COMPONENTS
MODEL_BUNDLES = RepositoryRegistry('model_bundle')
TRAINERS = RepositoryRegistry('trainer')
PIPELINES = RepositoryRegistry('pipeline')
DATASETS = RepositoryRegistry('dataset')
TRANSFORMS = RepositoryRegistry('transform')
HOOKS = RepositoryRegistry('hook')
EVALUATORS = RepositoryRegistry('evaluator')
VISUALIZERS = RepositoryRegistry('visualizer')
