# MODIFIED BY HFTRAINER: relocated into repository-owned layers and adapted for local execution.
# See hftrainer/models/ltx_video/UPSTREAM.md and LICENSE.ltx-2.x.
from typing import Callable, NamedTuple

import torch


class ModuleOps(NamedTuple):
    """
    Defines a named operation for matching and mutating PyTorch modules.
    Used to selectively transform modules in a model (e.g., replacing layers with quantized versions).
    """

    name: str
    matcher: Callable[[torch.nn.Module], bool]
    mutator: Callable[[torch.nn.Module], torch.nn.Module]
