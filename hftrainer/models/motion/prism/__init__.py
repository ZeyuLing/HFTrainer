from hftrainer.models.motion.prism.bundle import PrismBundle
from hftrainer.models.motion.prism.audio_encoder import AudioEncoderWrapper
from hftrainer.models.motion.prism.autoencoder_kl_2d import AutoencoderKLPrism2DTK
from hftrainer.models.motion.prism.autoencoder_kl_1d import AutoencoderKLPrism1D
from hftrainer.models.motion.prism.gaussian_distribution import (
    DiagonalGaussianDistributionNd,
)

try:
    from hftrainer.models.motion.prism.control_transformer import (
        PrismVACEControlBlock,
        PrismVACEControlTransformer,
    )
except ImportError:
    PrismVACEControlBlock = None
    PrismVACEControlTransformer = None

try:
    from hftrainer.models.motion.prism.mcm_bundle import PrismMCMBundle
except ImportError:
    PrismMCMBundle = None

__all__ = [name for name in [
    'PrismBundle',
    'PrismMCMBundle',
    'PrismVACEControlTransformer',
    'PrismVACEControlBlock',
    'AudioEncoderWrapper',
    'AutoencoderKLPrism2DTK',
    'AutoencoderKLPrism1D',
    'DiagonalGaussianDistributionNd',
] if globals().get(name) is not None]
