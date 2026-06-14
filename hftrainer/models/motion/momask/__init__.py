"""MoMask (RVQ + Masked Transformer + Residual Transformer) bundle.

Open-source baseline integrated into the hftrainer zoo. The RVQ-VAE tokenizer,
masked / residual transformers and length estimator are **vendored** into
``hftrainer.models.motion.momask._momask`` (fully independent of ``ref_repo``)
to guarantee numerical parity with the released HumanML3D checkpoints; this
module only exposes a clean hftrainer-native ``ModelBundle`` facade.
"""

from hftrainer.models.motion.momask.bundle import MoMaskBundle

__all__ = ["MoMaskBundle"]
