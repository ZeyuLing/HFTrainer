"""Public local Wan model components and their repository registry entries."""

from hftrainer.registry import MODEL_COMPONENTS

from .scheduler import FlowMatchEulerDiscreteScheduler
from .text_encoder import UMT5EncoderModel
from .tokenizer import WanTokenizer
from .transformer import WanTransformer3DModel
from .vae import AutoencoderKLWan

_LOCAL_COMPONENTS = (
    UMT5EncoderModel,
    AutoencoderKLWan,
    WanTransformer3DModel,
    FlowMatchEulerDiscreteScheduler,
    WanTokenizer,
)

# Imperative registration keeps these reusable by direct component configs
# without making them model-bundle exports in the package taxonomy.
for _component in _LOCAL_COMPONENTS:
    MODEL_COMPONENTS.register_module(
        name=_component.__name__, module=_component, force=True
    )


__all__ = [
    "AutoencoderKLWan",
    "FlowMatchEulerDiscreteScheduler",
    "UMT5EncoderModel",
    "WanTokenizer",
    "WanTransformer3DModel",
]
