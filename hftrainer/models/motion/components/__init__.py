"""Vendored motion components shared by PRISM and VerMo."""

from hftrainer.models.motion.components.autoencoder_prism import (
    AutoencoderKLPrism1D,
    AutoencoderKLPrism2DTK,
    VQVAEPrism1D,
    VQVAEPrism2DTK,
)
from hftrainer.models.motion.components.fs_quantizer import FSQuantizer
from hftrainer.models.motion.components.gaussian_distribution import (
    DiagonalGaussianDistributionNd,
)
from hftrainer.models.motion.components.motion_prism import PrismTransformerMotionModel
from hftrainer.models.motion.components.wavtokenizer.wavtokenizer import WavTokenizer
from hftrainer.registry import HF_MODELS

# Keep the original class names used inside VerMo configs/code paths.
VQVAEWanMotion1D = VQVAEPrism1D
VQVAEWanMotion2DTK = VQVAEPrism2DTK

if not HF_MODELS.get('VQVAEWanMotion1D'):
    HF_MODELS.register_module(name='VQVAEWanMotion1D', module=VQVAEWanMotion1D, force=True)
if not HF_MODELS.get('VQVAEWanMotion2DTK'):
    HF_MODELS.register_module(name='VQVAEWanMotion2DTK', module=VQVAEWanMotion2DTK, force=True)

__all__ = [
    'AutoencoderKLPrism1D',
    'AutoencoderKLPrism2DTK',
    'DiagonalGaussianDistributionNd',
    'FSQuantizer',
    'PrismTransformerMotionModel',
    'VQVAEPrism1D',
    'VQVAEPrism2DTK',
    'VQVAEWanMotion1D',
    'VQVAEWanMotion2DTK',
    'WavTokenizer',
]
