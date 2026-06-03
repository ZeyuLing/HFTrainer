"""PhysFlow task stack: KIMODO-G1 generator + frozen-judge physics reward."""

from hftrainer.models.motion.physflow.bundle import PhysFlowBundle
from hftrainer.models.motion.physflow.dataset import PhysFlowPromptDataset

__all__ = ["PhysFlowBundle", "PhysFlowPromptDataset"]
