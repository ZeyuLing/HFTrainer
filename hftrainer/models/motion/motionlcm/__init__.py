"""MotionLCM (latent consistency model) bundle.

Open-source baseline integrated into the hftrainer zoo. The MLD motion VAE,
latent consistency denoiser and text encoder live in
``hftrainer.models.motion.motionlcm.network``. Runtime loading is
artifact-based; raw upstream checkpoints are handled by converter/debug code.
"""

from hftrainer.models.motion.motionlcm.bundle import MotionLCMBundle

__all__ = ["MotionLCMBundle"]
