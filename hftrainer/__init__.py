"""
hftrainer full package init.
Imports all sub-modules to trigger registry registrations.
"""

# Core infrastructure
from hftrainer.registry import (
    HF_MODELS, MODELS, MODEL_BUNDLES, TRAINERS, PIPELINES,
    DATASETS, TRANSFORMS, HOOKS, EVALUATORS, VISUALIZERS,
    build_hf_model_from_cfg,
)
from hftrainer.models.base_model_bundle import ModelBundle
from hftrainer.trainers.base_trainer import BaseTrainer
from hftrainer.runner.accelerate_runner import AccelerateRunner

# ── Register HF model classes in HF_MODELS registry ──
# (These are loaded on-demand via _import_hf_class, but explicit registration
#  allows configs to reference them by name)
def _register_hf_classes():
    """Register common HF classes so they're available in HF_MODELS registry."""
    _classes_to_register = []

    # transformers
    try:
        from transformers import (
            ViTForImageClassification,
            CLIPTextModel,
            CLIPTokenizer,
            AutoModelForCausalLM,
            AutoTokenizer,
            UMT5EncoderModel,
            T5EncoderModel,
        )
        _classes_to_register.extend([
            ('ViTForImageClassification', ViTForImageClassification),
            ('CLIPTextModel', CLIPTextModel),
            ('AutoModelForCausalLM', AutoModelForCausalLM),
            ('UMT5EncoderModel', UMT5EncoderModel),
            ('T5EncoderModel', T5EncoderModel),
        ])
    except ImportError:
        pass

    # diffusers
    try:
        from diffusers import (
            AutoencoderKL,
            UNet2DConditionModel,
            DDPMScheduler,
            DDIMScheduler,
            PNDMScheduler,
            FlowMatchEulerDiscreteScheduler,
        )
        _classes_to_register.extend([
            ('AutoencoderKL', AutoencoderKL),
            ('UNet2DConditionModel', UNet2DConditionModel),
            ('DDPMScheduler', DDPMScheduler),
            ('DDIMScheduler', DDIMScheduler),
            ('PNDMScheduler', PNDMScheduler),
            ('FlowMatchEulerDiscreteScheduler', FlowMatchEulerDiscreteScheduler),
        ])
    except ImportError:
        pass

    # WAN-specific
    try:
        from diffusers import AutoencoderKLWan, WanTransformer3DModel
        _classes_to_register.extend([
            ('AutoencoderKLWan', AutoencoderKLWan),
            ('WanTransformer3DModel', WanTransformer3DModel),
        ])
    except ImportError:
        pass

    for name, cls in _classes_to_register:
        if not HF_MODELS.get(name):
            HF_MODELS.register_module(name=name, module=cls)


_register_hf_classes()

# ── Import task-specific modules to trigger @register_module decorators ──
def _import_task_modules():
    import importlib, warnings

    modules_to_import = [
        # Hooks
        'hftrainer.hooks.checkpoint_hook',
        'hftrainer.hooks.logger_hook',
        'hftrainer.hooks.ema_hook',
        'hftrainer.hooks.lr_scheduler_hook',
        # Evaluation
        'hftrainer.evaluation.classification.accuracy_evaluator',
        # Visualization
        'hftrainer.visualization.tensorboard_visualizer',
        'hftrainer.visualization.file_visualizer',
        # ViT
        'hftrainer.models.vit.bundle',
        'hftrainer.models.vit.trainer',
        'hftrainer.models.vit.pipeline',
        'hftrainer.datasets.classification.hf_image_classification_dataset',
        'hftrainer.datasets.classification.imagefolder_dataset',
    ]

    # Conditionally import task modules (may not all exist yet)
    optional_modules = [
        'hftrainer.models.sd15.bundle',
        'hftrainer.models.sd15.trainer',
        'hftrainer.models.sd15.pipeline',
        'hftrainer.models.dmd.pipeline',
        'hftrainer.datasets.text2image.hf_imagefolder_dataset',
        'hftrainer.models.causal_lm.bundle',
        'hftrainer.models.causal_lm.trainer',
        'hftrainer.models.causal_lm.pipeline',
        'hftrainer.datasets.llm.alpaca_dataset',
        'hftrainer.evaluation.llm.perplexity_evaluator',
        'hftrainer.models.wan.bundle',
        'hftrainer.models.wan.trainer',
        'hftrainer.models.wan.pipeline',
        'hftrainer.datasets.text2video.hf_video_dataset',
        # StyleGAN2
        'hftrainer.models.stylegan2.model',
        'hftrainer.models.stylegan2.bundle',
        'hftrainer.models.stylegan2.trainer',
        'hftrainer.models.stylegan2.pipeline',
        'hftrainer.datasets.gan.image_folder_gan_dataset',
        # DMD
        'hftrainer.models.dmd.bundle',
        'hftrainer.models.dmd.trainer',
        'hftrainer.datasets.distillation.dmd_image_pair_dataset',
        # Motion
        'hftrainer.models.motion.components.autoencoder_prism.autoencoder_kl_prism_2d',
        'hftrainer.models.motion.components.autoencoder_prism.autoencoder_kl_prism_1d',
        'hftrainer.models.motion.components.autoencoder_prism.vqvae_prism_2d',
        'hftrainer.models.motion.components.autoencoder_prism.vqvae_prism_1d',
        'hftrainer.models.motion.components.fs_quantizer',
        'hftrainer.models.motion.components.motion_prism.transformer_prism',
        'hftrainer.models.motion.components.motion_prism.transformer_prism_notext',
        'hftrainer.models.motion.components.body_models.smplx_lite',
        'hftrainer.models.motion.components.motion_processor.smpl_processor',
        'hftrainer.models.motion.components.wavtokenizer.wavtokenizer',
        'hftrainer.models.vermo.llama',
        'hftrainer.models.vermo.qwen3',
        'hftrainer.models.vermo.processor',
        'hftrainer.models.prism.bundle',
        'hftrainer.models.prism.trainer',
        'hftrainer.models.prism.pipeline',
        'hftrainer.models.vermo.bundle',
        'hftrainer.models.vermo.trainer',
        'hftrainer.models.vermo.pipeline',
        'hftrainer.datasets.motion.random_motion_text_dataset',
        'hftrainer.datasets.motion.vermo_toy_dataset',
        'hftrainer.datasets.motionhub.transforms',
        'hftrainer.datasets.motionhub',
    ]

    for mod_name in modules_to_import:
        try:
            importlib.import_module(mod_name)
        except ImportError as e:
            warnings.warn(f"Could not import {mod_name}: {e}")

    for mod_name in optional_modules:
        try:
            importlib.import_module(mod_name)
        except (ImportError, ModuleNotFoundError):
            pass


_import_task_modules()

__version__ = '0.1.0'

__all__ = [
    'HF_MODELS', 'MODELS', 'MODEL_BUNDLES', 'TRAINERS', 'PIPELINES',
    'DATASETS', 'TRANSFORMS', 'HOOKS', 'EVALUATORS', 'VISUALIZERS',
    'build_hf_model_from_cfg',
    'ModelBundle',
    'BaseTrainer',
    'AccelerateRunner',
]
