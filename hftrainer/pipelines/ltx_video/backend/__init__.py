# MODIFIED BY HFTRAINER: relocated into repository-owned layers and adapted for local execution.
# See hftrainer/models/ltx_video/UPSTREAM.md and LICENSE.ltx-2.x.
"""
LTX-2 Pipelines: High-level video generation pipelines and utilities.
This package provides ready-to-use pipelines for video generation:
- TI2VidOneStagePipeline: Text/image-to-video in a single stage
- T2AOneStagePipeline: Text-to-audio in a single stage (audio-only output)
- TI2VidTwoStagesPipeline: Two-stage generation with upsampling
- DistilledPipeline: Fast distilled two-stage generation
- DubItPipeline: Dub-It with IC-LoRA and audio conditioning
- ICLoraPipeline: Image/video conditioning with distilled LoRA
- KeyframeInterpolationPipeline: Keyframe-based video interpolation
- DFRPipeline: keyframe slots → spatial detailing → optional tiled temporal x2 (Diffusion Fidelity Rendering)
- RetakePipeline: Regenerate a time region (retake) of an existing video
For more detailed components and utilities, import from specific submodules
like `hftrainer.pipelines.ltx_video.backend.utils.media_io` or `hftrainer.pipelines.ltx_video.backend.utils.constants`.
Pipeline classes are imported lazily (PEP 562). Importing this package therefore
does not eagerly pull in every pipeline module, which keeps `import hftrainer.pipelines.ltx_video.backend`
light and avoids the runpy double-import warning when a pipeline is run as a module
(e.g. `python -m hftrainer.pipelines.ltx_video.backend.distilled`).
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hftrainer.pipelines.ltx_video.backend.a2vid_two_stage import A2VidPipelineTwoStage
    from hftrainer.pipelines.ltx_video.backend.dfr_pipeline import DFRPipeline
    from hftrainer.pipelines.ltx_video.backend.distilled import DistilledPipeline
    from hftrainer.pipelines.ltx_video.backend.dubit import DubItPipeline
    from hftrainer.pipelines.ltx_video.backend.ic_lora import ICLoraPipeline
    from hftrainer.pipelines.ltx_video.backend.keyframe_interpolation import KeyframeInterpolationPipeline
    from hftrainer.pipelines.ltx_video.backend.retake import RetakePipeline
    from hftrainer.pipelines.ltx_video.backend.t2a_one_stage import T2AOneStagePipeline
    from hftrainer.pipelines.ltx_video.backend.ti2vid_one_stage import TI2VidOneStagePipeline
    from hftrainer.pipelines.ltx_video.backend.ti2vid_two_stages import TI2VidTwoStagesPipeline

__all__ = [
    "A2VidPipelineTwoStage",
    "DFRPipeline",
    "DistilledPipeline",
    "DubItPipeline",
    "ICLoraPipeline",
    "KeyframeInterpolationPipeline",
    "RetakePipeline",
    "T2AOneStagePipeline",
    "TI2VidOneStagePipeline",
    "TI2VidTwoStagesPipeline",
]


def __getattr__(name: str) -> object:
    # Keep package import light while making every executable target explicit.
    # String-driven module resolution is deliberately avoided: repository-owned
    # pipelines must be visible to static review and dependency-boundary tests.
    if name == "A2VidPipelineTwoStage":
        from .a2vid_two_stage import A2VidPipelineTwoStage as value
    elif name == "DFRPipeline":
        from .dfr_pipeline import DFRPipeline as value
    elif name == "DistilledPipeline":
        from .distilled import DistilledPipeline as value
    elif name == "DubItPipeline":
        from .dubit import DubItPipeline as value
    elif name == "ICLoraPipeline":
        from .ic_lora import ICLoraPipeline as value
    elif name == "KeyframeInterpolationPipeline":
        from .keyframe_interpolation import KeyframeInterpolationPipeline as value
    elif name == "RetakePipeline":
        from .retake import RetakePipeline as value
    elif name == "T2AOneStagePipeline":
        from .t2a_one_stage import T2AOneStagePipeline as value
    elif name == "TI2VidOneStagePipeline":
        from .ti2vid_one_stage import TI2VidOneStagePipeline as value
    elif name == "TI2VidTwoStagesPipeline":
        from .ti2vid_two_stages import TI2VidTwoStagesPipeline as value
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value  # cache so later lookups skip __getattr__
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
