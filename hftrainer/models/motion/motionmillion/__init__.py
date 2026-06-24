"""MotionMillion / "Go to Zero" (FSQ tokenizer + LLaMA AR) bundle.

ICCV'25 open-source text-to-motion model integrated into the hftrainer zoo. The
HumanVQVAE (FSQ) tokenizer and the LLaMA autoregressive transformer live in the
local ``hftrainer.models.motion.motionmillion.network`` package; runtime loading
is artifact-based, while raw upstream checkpoints are handled by converter code.
"""

from hftrainer.models.motion.motionmillion.bundle import MotionMillionBundle

__all__ = ["MotionMillionBundle"]
