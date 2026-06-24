"""HyMotion-V2M: video(feature)-to-motion migration into hf_trainer.

Stage 1 exposes pre-extracted-feature -> motion inference through the standard
Bundle / Pipeline / Registry surface, wrapping the vendored, self-contained
``MotionGenerationV2M`` source so the original checkpoints load numerically
unchanged.
"""

from .bundle import HyMotionV2MBundle

__all__ = ["HyMotionV2MBundle"]
