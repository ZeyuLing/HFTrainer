# MODIFIED BY HFTRAINER: relocated into repository-owned layers and adapted for local execution.
# See hftrainer/models/ltx_video/UPSTREAM.md and LICENSE.ltx-2.x.
from typing import Protocol

from hftrainer.models.ltx_video.network.tools import LatentTools
from hftrainer.models.ltx_video.network.types import LatentState


class ConditioningItem(Protocol):
    """Protocol for conditioning items that modify latent state during diffusion."""

    def apply_to(self, latent_state: LatentState, latent_tools: LatentTools) -> LatentState:
        """
        Apply the conditioning to the latent state.
        Args:
            latent_state: The latent state to apply the conditioning to. This is state always patchified.
        Returns:
            The latent state after the conditioning has been applied.
        IMPORTANT: If the conditioning needs to add extra tokens to the latent, it should add them to the end of the
        latent.
        """
        ...
