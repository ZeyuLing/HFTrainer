# MODIFIED BY HFTRAINER: relocated into repository-owned layers and adapted for local execution.
# See hftrainer/models/ltx_video/UPSTREAM.md and LICENSE.ltx-2.x.
from dataclasses import dataclass

from hftrainer.models.ltx_video.network.loader.fuse_loras import FuseRule, bf16_fuse_rule
from hftrainer.models.ltx_video.network.loader.module_ops import ModuleOps
from hftrainer.models.ltx_video.network.loader.sd_ops import SDOps
from hftrainer.models.ltx_video.network.model.model_protocol import ModelConfigurator
from hftrainer.models.ltx_video.network.model.transformer.model import LTXModel


@dataclass(frozen=True)
class QuantizationPolicy:
    """Configuration for model quantization during loading.
    Attributes:
        sd_ops: State-dict operations applied to each tensor during load.
        module_ops: Post-load module transformations applied to the meta model.
        model_configurator: Configurator class to use when constructing the transformer.
        fuse_rule: How LoRA deltas merge into this policy's weight layout.
            Default ``bf16_fuse_rule`` is used when no policy is configured.
    """

    sd_ops: SDOps | None = None
    module_ops: tuple[ModuleOps, ...] = ()
    model_configurator: type[ModelConfigurator[LTXModel]] | None = None
    fuse_rule: FuseRule = bf16_fuse_rule
