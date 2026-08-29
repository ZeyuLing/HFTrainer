"""Public repository-local MiniMax-H3 components and registry entries."""

from hftrainer.registry import MODEL_COMPONENTS

from .audio_vae import (
    AutoencoderKLMiniMaxH3Audio,
    MiniMaxH3AudioDiagonalGaussianDistribution,
)
from .processor import (
    MiniMaxH3Presentation,
    MiniMaxH3Processor,
    Qwen3VLProcessor,
)
from .qwen3_vl import (
    MiniMaxH3Qwen3VLEncoder,
    Qwen3VLConfig,
    Qwen3VLForConditionalGeneration,
    Qwen3VLTextConfig,
    Qwen3VLVisionConfig,
)
from .scheduler import MiniMaxH3Scheduler
from .tokenizer import MiniMaxH3Tokenizer, Qwen2Tokenizer
from .transformer import MiniMaxH3Transformer3DModel
from .video_vae import AutoencoderKLMiniMaxH3

_LOCAL_COMPONENTS = (
    AutoencoderKLMiniMaxH3,
    AutoencoderKLMiniMaxH3Audio,
    MiniMaxH3Qwen3VLEncoder,
    MiniMaxH3Scheduler,
    MiniMaxH3Transformer3DModel,
    Qwen3VLForConditionalGeneration,
)

for _component in _LOCAL_COMPONENTS:
    MODEL_COMPONENTS.register_module(
        name=_component.__name__, module=_component, force=True
    )


__all__ = [
    "AutoencoderKLMiniMaxH3",
    "AutoencoderKLMiniMaxH3Audio",
    "MiniMaxH3AudioDiagonalGaussianDistribution",
    "MiniMaxH3Presentation",
    "MiniMaxH3Processor",
    "MiniMaxH3Qwen3VLEncoder",
    "MiniMaxH3Scheduler",
    "MiniMaxH3Tokenizer",
    "MiniMaxH3Transformer3DModel",
    "Qwen2Tokenizer",
    "Qwen3VLConfig",
    "Qwen3VLForConditionalGeneration",
    "Qwen3VLProcessor",
    "Qwen3VLTextConfig",
    "Qwen3VLVisionConfig",
]
