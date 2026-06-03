"""
hftrainer full package init (motion branch).

Imports all motion sub-modules to trigger registry registrations.
Non-motion task stacks (classification / GAN / LLM / SD15 / Wan / DMD)
have been removed from this branch; the corresponding loaders live
on the main branch.
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

    # transformers — only motion-relevant text encoders / language backbones
    try:
        from transformers import (
            CLIPTextModel,
            CLIPTokenizer,
            AutoModelForCausalLM,
            AutoTokenizer,
            UMT5EncoderModel,
            T5EncoderModel,
        )
        _classes_to_register.extend([
            ('CLIPTextModel', CLIPTextModel),
            ('AutoModelForCausalLM', AutoModelForCausalLM),
            ('UMT5EncoderModel', UMT5EncoderModel),
            ('T5EncoderModel', T5EncoderModel),
        ])
    except ImportError:
        pass

    # diffusers — only the schedulers used by motion flow-matching pipelines
    try:
        from diffusers import (
            DDPMScheduler,
            DDIMScheduler,
            PNDMScheduler,
            FlowMatchEulerDiscreteScheduler,
        )
        _classes_to_register.extend([
            ('DDPMScheduler', DDPMScheduler),
            ('DDIMScheduler', DDIMScheduler),
            ('PNDMScheduler', PNDMScheduler),
            ('FlowMatchEulerDiscreteScheduler', FlowMatchEulerDiscreteScheduler),
        ])
    except ImportError:
        pass

    for name, cls in _classes_to_register:
        if not HF_MODELS.get(name):
            HF_MODELS.register_module(name=name, module=cls)


_register_hf_classes()

# ── Import motion task modules to trigger @register_module decorators ──
def _import_task_modules():
    import importlib, warnings

    modules_to_import = [
        # Hooks
        'hftrainer.hooks.checkpoint_hook',
        'hftrainer.hooks.logger_hook',
        'hftrainer.hooks.ema_hook',
        'hftrainer.hooks.lr_scheduler_hook',
        # Visualization
        'hftrainer.visualization.tensorboard_visualizer',
        'hftrainer.visualization.file_visualizer',
        # Generic dataset transforms
        'hftrainer.datasets.transforms',
    ]

    # Motion-specific task modules.  Wrapped in optional-import so that
    # missing optional deps in some sub-trees do not break package import.
    optional_modules = [
        # Motion
        'hftrainer.models.motion.prism.autoencoder_kl_2d',
        'hftrainer.models.motion.prism.autoencoder_kl_1d',
        'hftrainer.models.motion.vermo.vqvae_2d',
        'hftrainer.models.motion.vermo.vqvae_1d',
        'hftrainer.models.motion.vermo.fs_quantizer',
        'hftrainer.models.motion.prism.network.transformer_prism',
        'hftrainer.models.motion.prism.network.transformer_prism_notext',
        'hftrainer.models.motion.components.body_models.smplx_lite',
        'hftrainer.models.motion.components.motion_processor.smpl_processor',
        'hftrainer.models.motion.vermo.wavtokenizer.wavtokenizer',
        'hftrainer.models.motion.vermo.llama',
        'hftrainer.models.motion.vermo.qwen3',
        'hftrainer.models.motion.vermo.processor',
        'hftrainer.models.motion.prism.bundle',
        'hftrainer.models.motion.prism.audio_encoder',
        'hftrainer.models.motion.prism.control_transformer',
        'hftrainer.models.motion.prism.mcm_bundle',
        'hftrainer.trainers.motion.prism_trainer',
        'hftrainer.trainers.motion.prism_mcm_trainer',
        'hftrainer.pipelines.motion.prism_pipeline',
        'hftrainer.pipelines.motion.prism_mcm_pipeline',
        'hftrainer.models.motion.vermo.bundle',
        'hftrainer.trainers.motion.vermo_trainer',
        'hftrainer.pipelines.motion.vermo_pipeline',
        'hftrainer.datasets.motion.random_motion_text_dataset',
        'hftrainer.datasets.motion.random_motion_audio_dataset',
        'hftrainer.datasets.motion.vermo_toy_dataset',
        # HyMotion-M2M
        'hftrainer.models.motion.hymotion_m2m.network',
        'hftrainer.models.motion.hymotion_m2m.bundle',
        'hftrainer.trainers.motion.hymotion_m2m_trainer',
        'hftrainer.pipelines.motion.hymotion_m2m_pipeline',
        'hftrainer.datasets.motion.hymotion_m2m_dataset',
        # HyMotion-T2M
        'hftrainer.models.motion.hymotion_t2m.bundle',
        'hftrainer.trainers.motion.hymotion_t2m_trainer',
        'hftrainer.pipelines.motion.hymotion_t2m_pipeline',
        'hftrainer.datasets.motion.hymotion_t2m_dataset',
        'hftrainer.datasets.motion.motionhub.transforms',
        'hftrainer.datasets.motion.motionhub',
        # MotionCLIP evaluator
        'hftrainer.models.motion.motion_clip',
        'hftrainer.trainers.motion.motion_clip_trainer',
        'hftrainer.pipelines.motion.motion_clip_pipeline',
        'hftrainer.datasets.motion.motionclip_synthetic_dataset',
        # PhysFlow (KIMODO-G1 online adversarial)
        'hftrainer.models.motion.physflow.bundle',
        'hftrainer.models.motion.physflow.dataset',
        'hftrainer.trainers.motion.physflow_trainer',
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
