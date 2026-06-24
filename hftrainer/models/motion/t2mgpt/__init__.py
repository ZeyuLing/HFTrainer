"""T2M-GPT (VQ-VAE + GPT) bundle.

CVPR'23 open-source text-to-motion model integrated into the hftrainer Model
Zoo. The VQ-VAE motion tokenizer and the cross-conditional GPT live in the
local ``hftrainer.models.motion.t2mgpt.network`` package; runtime loading is
artifact-based, while raw upstream checkpoints are handled by converter scripts.
"""

from hftrainer.models.motion.t2mgpt.bundle import T2MGPTBundle

__all__ = ["T2MGPTBundle"]
