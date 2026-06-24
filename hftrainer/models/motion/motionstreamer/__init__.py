"""MotionStreamer (causal TAE + LLaMA AR + diffusion head) bundle.

Open-source text-to-motion baseline integrated into the hftrainer zoo. The TAE,
the LLaMA autoregressive transformer, the per-token diffusion head and the
Gaussian-diffusion sampler live in
``hftrainer.models.motion.motionstreamer.network``. Runtime loading is
artifact-based; raw upstream checkpoints are handled by converter/debug code.
"""

from hftrainer.models.motion.motionstreamer.bundle import MotionStreamerBundle

__all__ = ["MotionStreamerBundle"]
