"""MoGenTS (spatial-temporal T2M) bundle.

NeurIPS'24 open-source text-to-motion model integrated into the hftrainer Model
Zoo. Runtime components live in ``hftrainer.models.motion.mogents.network``;
raw upstream checkpoints are handled by converter/debug scripts.
"""

from hftrainer.models.motion.mogents.bundle import MoGenTSBundle

__all__ = ["MoGenTSBundle"]
