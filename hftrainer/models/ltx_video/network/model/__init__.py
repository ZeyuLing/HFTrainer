# MODIFIED BY HFTRAINER: relocated into repository-owned layers and adapted for local execution.
# See hftrainer/models/ltx_video/UPSTREAM.md and LICENSE.ltx-2.x.
"""Model definitions for LTX-2."""

from hftrainer.models.ltx_video.network.model.disposable import Disposable, DisposableProtocol
from hftrainer.models.ltx_video.network.model.model_protocol import ModelConfigurator, ModelType

__all__ = [
    "Disposable",
    "DisposableProtocol",
    "ModelConfigurator",
    "ModelType",
]
