"""Self-contained vendored MoMask model.

RVQ-VAE tokenizer (6 quantizers) + masked generative transformer
(``MaskTransformer``) + residual transformer (``ResidualTransformer``) +
length estimator, ported into hftrainer so the bundle/pipeline are fully
independent of the original ``momask-codes`` repository while preserving
parity with the released HumanML3D checkpoints. Only the T2M inference path is
exercised here. The CLIP ViT-B/32 text encoder lives inside the two
transformers (frozen, reloaded by name) and is **not** part of the artifact.
"""

from .inference import estimate_token_lengths, generate_motion
from .mask_transformer import MaskTransformer, ResidualTransformer
from .vq import LengthEstimator, RVQVAE

__all__ = [
    "RVQVAE",
    "LengthEstimator",
    "MaskTransformer",
    "ResidualTransformer",
    "generate_motion",
    "estimate_token_lengths",
]
