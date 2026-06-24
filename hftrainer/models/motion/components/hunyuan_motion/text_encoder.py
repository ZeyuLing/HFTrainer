"""Compatibility imports for shared HunyuanMotion text encoders."""

from hftrainer.models.motion.hymotion_m2m.network.text_encoder import (
    HYTextModel,
    LLM_ENCODER_LAYOUT,
    SENTENCE_EMB_LAYOUT,
)

__all__ = ["HYTextModel", "LLM_ENCODER_LAYOUT", "SENTENCE_EMB_LAYOUT"]
