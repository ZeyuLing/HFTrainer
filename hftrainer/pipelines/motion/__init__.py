"""Compatibility package for motion pipeline imports."""

from hftrainer.pipelines.motion.prism_pipeline import PrismPipeline
from hftrainer.pipelines.motion.vermo_pipeline import VermoPipeline
from hftrainer.pipelines.motion.hymotion_m2m_pipeline import HyMotionM2MPipeline
from hftrainer.pipelines.motion.hymotion_t2m_pipeline import HyMotionT2MPipeline
from hftrainer.pipelines.motion.mogendit_pipeline import MoGenDITRepairPipeline

__all__ = [
    'PrismPipeline',
    'VermoPipeline',
    'HyMotionM2MPipeline',
    'HyMotionT2MPipeline',
    'MoGenDITRepairPipeline',
]
