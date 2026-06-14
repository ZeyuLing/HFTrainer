"""T2M-GPT (VQ-VAE + GPT) bundle.

CVPR'23 open-source text-to-motion model integrated into the hftrainer Model
Zoo. The VQ-VAE motion tokenizer and the cross-conditional GPT are **vendored**
into ``hftrainer.models.motion.t2mgpt._t2mgpt`` so the reproduction is fully
independent of the original repository while preserving parity with the released
HumanML3D checkpoint.
"""

from hftrainer.models.motion.t2mgpt.bundle import T2MGPTBundle

__all__ = ["T2MGPTBundle"]
