"""
hftrainer full package init (motion branch).

Imports all motion sub-modules to trigger registry registrations.
Non-motion task stacks (classification / GAN / LLM / SD15 / Wan / DMD)
have been removed from this branch; the corresponding loaders live
on the main branch.

Import-light escape hatch
-------------------------
By default, importing ``hftrainer`` eagerly imports the full motion training
stack (DeepSpeed, transformers, every bundle/trainer/pipeline) to populate the
registries. This is expensive. The public motion library
(``hftrainer.motion.*``) does NOT need any of that.

Set ``HFTRAINER_SKIP_AUTOREGISTER=1`` to skip auto-registration so that
``import hftrainer.motion.representation.rotation`` (etc.) stays cheap. Training
and inference entry points (``tools/train.py`` / ``tools/infer.py``) should call
``hftrainer.register_all_modules()`` explicitly when this is set. Default
behavior is unchanged.
"""

import os as _os

# Core infrastructure
from hftrainer.registry import (
    HF_MODELS, MODELS, MODEL_BUNDLES, TRAINERS, PIPELINES,
    DATASETS, TRANSFORMS, HOOKS, EVALUATORS, VISUALIZERS,
    build_hf_model_from_cfg,
)

_SKIP_AUTOREGISTER = _os.environ.get("HFTRAINER_SKIP_AUTOREGISTER", "").lower() in (
    "1", "true", "yes", "on",
)

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
        'hftrainer.motion.body_models.smplx_lite',
        'hftrainer.motion.processing.smpl_processor',
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
        # HyMotion-V2M (video/feature -> motion; vendored self-contained source)
        'hftrainer.models.motion.hymotion_v2m.bundle',
        'hftrainer.pipelines.motion.hymotion_v2m_pipeline',
        'hftrainer.datasets.motion.hymotion_v2m_dataset',
        # MotionCLIP evaluator
        'hftrainer.models.motion.motion_clip',
        'hftrainer.trainers.motion.motion_clip_trainer',
        'hftrainer.pipelines.motion.motion_clip_pipeline',
        'hftrainer.datasets.motion.motionclip_synthetic_dataset',
        # PhysFlow (KIMODO-G1 / HyMotion-G1 online adversarial)
        'hftrainer.models.motion.physflow.bundle',
        'hftrainer.models.motion.physflow.dataset',
        'hftrainer.models.motion.physflow.g1_dataset',
        'hftrainer.trainers.motion.physflow_trainer',
        'hftrainer.trainers.motion.physflow_g1_trainer',
        # MDM (open-source T2M baseline; vendored, ref_repo-independent)
        'hftrainer.models.motion.mdm.bundle',
        'hftrainer.pipelines.mdm.pipeline',
        # MotionStreamer (open-source T2M baseline; vendored, ref_repo-independent)
        'hftrainer.models.motion.motionstreamer.bundle',
        'hftrainer.pipelines.motionstreamer.pipeline',
        # MotionMillion / "Go to Zero" (open-source T2M baseline; vendored)
        'hftrainer.models.motion.motionmillion.bundle',
        'hftrainer.pipelines.motionmillion.pipeline',
        # T2M-GPT (VQ-VAE + GPT; open-source T2M baseline; vendored, ref_repo-free)
        'hftrainer.models.motion.t2mgpt.bundle',
        'hftrainer.pipelines.t2mgpt.pipeline',
        # MoMask (RVQ + masked/residual transformer; vendored, ref_repo-independent)
        'hftrainer.models.motion.momask.bundle',
        'hftrainer.pipelines.momask.pipeline',
        # MoGenTS (dual spatial-temporal RVQ + masked/residual transformers)
        'hftrainer.models.motion.mogents.bundle',
        'hftrainer.pipelines.mogents.pipeline',
        # MotionLCM (latent consistency model; vendored, ref_repo-independent)
        'hftrainer.models.motion.motionlcm.bundle',
        'hftrainer.pipelines.motionlcm.pipeline',
        # KIMODO (official NVIDIA runtime wrapper; self-contained hftrainer artifacts)
        'hftrainer.models.motion.kimodo.bundle',
        'hftrainer.pipelines.motion.kimodo_pipeline',
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


def register_all_modules():
    """Eagerly import the full motion stack to populate all registries.

    Call this explicitly when ``HFTRAINER_SKIP_AUTOREGISTER`` is set (e.g. at the
    top of training/inference entry points). It also binds the core classes
    (``ModelBundle`` / ``BaseTrainer`` / ``AccelerateRunner``) onto this module.
    Idempotent and cheap to call more than once (Python import caching).
    """
    global ModelBundle, BaseTrainer, AccelerateRunner
    from hftrainer.models.base_model_bundle import ModelBundle  # noqa: F401
    from hftrainer.trainers.base_trainer import BaseTrainer  # noqa: F401
    from hftrainer.runner.accelerate_runner import AccelerateRunner  # noqa: F401
    _register_hf_classes()
    _import_task_modules()


if not _SKIP_AUTOREGISTER:
    register_all_modules()
else:
    # Import-light mode: provide lazy access to the core classes so that
    # ``from hftrainer import ModelBundle`` still works on demand.
    def __getattr__(name):  # PEP 562
        if name in ("ModelBundle", "BaseTrainer", "AccelerateRunner"):
            register_all_modules()
            return globals()[name]
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__version__ = '0.1.0'

__all__ = [
    'HF_MODELS', 'MODELS', 'MODEL_BUNDLES', 'TRAINERS', 'PIPELINES',
    'DATASETS', 'TRANSFORMS', 'HOOKS', 'EVALUATORS', 'VISUALIZERS',
    'build_hf_model_from_cfg',
    'ModelBundle',
    'BaseTrainer',
    'AccelerateRunner',
    'register_all_modules',
]
