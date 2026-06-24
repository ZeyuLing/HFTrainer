from hftrainer.models.motion.vermo.bundle import VermoBundle
from hftrainer.models.motion.vermo.fs_quantizer import FSQuantizer
from hftrainer.models.motion.vermo.llama import VermoLlamaForCausalLM
from hftrainer.models.motion.vermo.qwen3 import VermoQwen3ForCausalLM
from hftrainer.models.motion.vermo.processor import VermoProcessor
from hftrainer.models.motion.vermo.vqvae_2d import VQVAEVermo2DTK
from hftrainer.models.motion.vermo.vqvae_1d import VQVAEVermo1D
from hftrainer.registry import HF_MODELS

# Backward-compatible aliases used in existing VerMo configs/code paths.
VQVAEWanMotion1D = VQVAEVermo1D
VQVAEWanMotion2DTK = VQVAEVermo2DTK

if not HF_MODELS.get('VQVAEWanMotion1D'):
    HF_MODELS.register_module(name='VQVAEWanMotion1D', module=VQVAEWanMotion1D, force=True)
if not HF_MODELS.get('VQVAEWanMotion2DTK'):
    HF_MODELS.register_module(name='VQVAEWanMotion2DTK', module=VQVAEWanMotion2DTK, force=True)

__all__ = [
    'VermoBundle',
    'VermoLlamaForCausalLM',
    'VermoQwen3ForCausalLM',
    'VermoProcessor',
    'FSQuantizer',
    'VQVAEVermo1D',
    'VQVAEVermo2DTK',
    'VQVAEWanMotion1D',
    'VQVAEWanMotion2DTK',
]
