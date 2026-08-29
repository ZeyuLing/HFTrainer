"""Explicit module registration for HFTrainer.

Keeping this catalogue out of :mod:`hftrainer.__init__` makes importing the
core package cheap and prevents optional training dependencies from becoming
an import-time requirement.
"""

from importlib import import_module
from threading import RLock

from hftrainer.registry import HF_MODELS

# Framework-level components shared by all methods.
_FRAMEWORK_MODULES = (
    'hftrainer.hooks.checkpoint_hook',
    'hftrainer.hooks.logger_hook',
    'hftrainer.hooks.ema_hook',
    'hftrainer.hooks.lr_scheduler_hook',
    'hftrainer.evaluation.classification.accuracy_evaluator',
    'hftrainer.evaluation.llm.perplexity_evaluator',
    'hftrainer.visualization.tensorboard_visualizer',
    'hftrainer.visualization.file_visualizer',
    'hftrainer.datasets.transforms',
)

# Built-in vertical slices. Third-party and separately packaged methods stay
# opt-in through config ``custom_imports`` and therefore do not belong here.
_BUILTIN_METHOD_MODULES = (
    # Classification / ViT
    'hftrainer.models.vit.bundle',
    'hftrainer.trainers.classification.classification_trainer',
    'hftrainer.pipelines.classification.classification_pipeline',
    'hftrainer.datasets.classification.hf_image_classification_dataset',
    'hftrainer.datasets.classification.imagefolder_dataset',
    # Text-to-image / Stable Diffusion 1.5
    'hftrainer.models.sd15.bundle',
    'hftrainer.trainers.text2image.sd15_trainer',
    'hftrainer.pipelines.text2image.sd15_pipeline',
    'hftrainer.datasets.text2image.hf_imagefolder_dataset',
    # Causal language modelling
    'hftrainer.models.causal_lm.bundle',
    'hftrainer.trainers.llm.causal_lm_trainer',
    'hftrainer.pipelines.llm.causal_lm_pipeline',
    'hftrainer.datasets.llm.alpaca_dataset',
    # Text-to-video / Wan
    'hftrainer.models.wan.bundle',
    'hftrainer.trainers.text2video.wan_trainer',
    'hftrainer.pipelines.text2video.wan_pipeline',
    'hftrainer.datasets.text2video.hf_video_dataset',
    # StyleGAN2
    'hftrainer.models.stylegan2.model',
    'hftrainer.models.stylegan2.bundle',
    'hftrainer.trainers.gan.gan_trainer',
    'hftrainer.pipelines.gan.stylegan2_pipeline',
    'hftrainer.datasets.gan.image_folder_gan_dataset',
    # Distribution Matching Distillation
    'hftrainer.models.dmd.bundle',
    'hftrainer.trainers.distillation.dmd_trainer',
    'hftrainer.pipelines.text2image.dmd_pipeline',
    'hftrainer.datasets.distillation.dmd_image_pair_dataset',
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


def _register_hf_classes() -> None:
    """Register common Hugging Face classes when their packages are present."""
    classes_to_register = []

    try:
        from transformers import (
            AutoModelForCausalLM,
            CLIPTextModel,
            T5EncoderModel,
            UMT5EncoderModel,
            ViTForImageClassification,
        )

        classes_to_register.extend(
            [
                ('ViTForImageClassification', ViTForImageClassification),
                ('CLIPTextModel', CLIPTextModel),
                ('AutoModelForCausalLM', AutoModelForCausalLM),
                ('UMT5EncoderModel', UMT5EncoderModel),
                ('T5EncoderModel', T5EncoderModel),
            ]
        )
    except ImportError:
        pass

    try:
        from diffusers import (
            AutoencoderKL,
            DDIMScheduler,
            DDPMScheduler,
            FlowMatchEulerDiscreteScheduler,
            PNDMScheduler,
            UNet2DConditionModel,
        )

        classes_to_register.extend(
            [
                ('AutoencoderKL', AutoencoderKL),
                ('UNet2DConditionModel', UNet2DConditionModel),
                ('DDPMScheduler', DDPMScheduler),
                ('DDIMScheduler', DDIMScheduler),
                ('PNDMScheduler', PNDMScheduler),
                ('FlowMatchEulerDiscreteScheduler', FlowMatchEulerDiscreteScheduler),
            ]
        )
    except ImportError:
        pass

    try:
        from diffusers import AutoencoderKLWan, WanTransformer3DModel

        classes_to_register.extend(
            [
                ('AutoencoderKLWan', AutoencoderKLWan),
                ('WanTransformer3DModel', WanTransformer3DModel),
            ]
        )
    except ImportError:
        pass

    for name, cls in classes_to_register:
        if HF_MODELS.get(name) is None:
            HF_MODELS.register_module(name=name, module=cls)


def register_all_modules() -> None:
    """Import and register every framework and built-in method component."""
    global _REGISTERED

    if _REGISTERED:
        return

    with _REGISTRATION_LOCK:
        if _REGISTERED:
            return

        _register_hf_classes()
        for module_name in (*_FRAMEWORK_MODULES, *_BUILTIN_METHOD_MODULES):
            import_module(module_name)

        _REGISTERED = True


__all__ = ['import_custom_modules', 'register_all_modules']
