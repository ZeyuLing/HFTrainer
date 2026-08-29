"""Explicit module registration for HFTrainer.

Keeping this catalogue out of :mod:`hftrainer.__init__` makes importing the
core package cheap and prevents optional training dependencies from becoming
an import-time requirement.
"""

from importlib import import_module
from threading import RLock

# Framework-level components shared by all methods.
_FRAMEWORK_MODULES = (
    'hftrainer.hooks.checkpoint_hook',
    'hftrainer.hooks.logger_hook',
    'hftrainer.hooks.ema_hook',
    'hftrainer.hooks.lr_scheduler_hook',
    'hftrainer.evaluation.image_classification.accuracy',
    'hftrainer.evaluation.causal_language_modeling.perplexity',
    'hftrainer.visualization.tensorboard_visualizer',
    'hftrainer.visualization.file_visualizer',
    'hftrainer.datasets.transforms',
)

# Built-in vertical slices. Third-party and separately packaged methods stay
# opt-in through config ``custom_imports`` and therefore do not belong here.
_BUILTIN_METHOD_MODULES = (
    # Classification / ViT
    'hftrainer.models.vit.bundle',
    'hftrainer.tasks.image_classification.trainer',
    'hftrainer.tasks.image_classification.pipeline',
    'hftrainer.datasets.image_classification.hf_dataset',
    'hftrainer.datasets.image_classification.image_folder',
    # Text-to-image / Stable Diffusion 1.5
    'hftrainer.models.sd15.bundle',
    'hftrainer.trainers.sd15.trainer',
    'hftrainer.pipelines.sd15.pipeline',
    'hftrainer.datasets.text_to_image.hf_image_folder',
    # Causal language modelling
    'hftrainer.models.llama.bundle',
    'hftrainer.tasks.causal_language_modeling.trainer',
    'hftrainer.tasks.causal_language_modeling.pipeline',
    'hftrainer.datasets.instruction_sft.alpaca',
    # Text-to-video / Wan
    'hftrainer.models.wan.bundle',
    'hftrainer.trainers.wan.trainer',
    'hftrainer.pipelines.wan.pipeline',
    'hftrainer.datasets.text_to_video.hf_video',
    # StyleGAN2
    'hftrainer.models.stylegan2.network',
    'hftrainer.models.stylegan2.bundle',
    'hftrainer.trainers.stylegan2.trainer',
    'hftrainer.pipelines.stylegan2.pipeline',
    'hftrainer.datasets.unconditional_image.image_folder',
    # Distribution Matching Distillation
    'hftrainer.models.dmd.bundle',
    'hftrainer.trainers.dmd.trainer',
    'hftrainer.pipelines.dmd.pipeline',
    'hftrainer.datasets.dmd.image_pair',
    # LTX-Video 2.5 (heavy numerical modules remain lazy behind wrappers)
    'hftrainer.models.ltx_video.bundle',
    'hftrainer.trainers.ltx_video.trainer',
    'hftrainer.pipelines.ltx_video.pipeline',
)

_REGISTRATION_LOCK = RLock()
_REGISTERED = False


def import_custom_modules(cfg):
    """Import extension modules declared by an MMEngine-style config.

    Args:
        cfg: A config object (or mapping) whose optional ``custom_imports``
            value follows ``mmengine.utils.import_modules_from_strings``.

    Returns:
        The module object(s) returned by MMEngine, or ``None`` when the config
        declares no custom imports.
    """
    if isinstance(cfg, dict):
        custom_imports = cfg.get('custom_imports')
    else:
        custom_imports = getattr(cfg, 'custom_imports', None)
    if custom_imports is None:
        return None
    if hasattr(custom_imports, 'to_dict'):
        custom_imports = custom_imports.to_dict()

    from mmengine.utils import import_modules_from_strings

    return import_modules_from_strings(**dict(custom_imports))


def register_all_modules() -> None:
    """Import and register every framework and built-in method component."""
    global _REGISTERED

    if _REGISTERED:
        return

    with _REGISTRATION_LOCK:
        if _REGISTERED:
            return

        for module_name in (*_FRAMEWORK_MODULES, *_BUILTIN_METHOD_MODULES):
            import_module(module_name)

        _REGISTERED = True


__all__ = ['import_custom_modules', 'register_all_modules']
