"""MDM (Motion Diffusion Model) bundle.

Open-source baseline integrated into the hftrainer zoo. The neural network and
Gaussian diffusion live in ``hftrainer.models.motion.mdm.network``. Runtime
loading is artifact-based; raw upstream checkpoints are handled by converter
code.
"""

from hftrainer.models.motion.mdm.bundle import MDMBundle

__all__ = ["MDMBundle"]
