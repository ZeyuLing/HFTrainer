"""MoMask (RVQ + Masked Transformer + Residual Transformer) bundle.

Open-source baseline integrated into the hftrainer zoo. The RVQ-VAE tokenizer,
masked / residual transformers and length estimator live in
``hftrainer.models.motion.momask.network``. Runtime loading is artifact-based;
raw upstream checkpoints are handled by converter/debug code.
"""

from hftrainer.models.motion.momask.bundle import MoMaskBundle

__all__ = ["MoMaskBundle"]
