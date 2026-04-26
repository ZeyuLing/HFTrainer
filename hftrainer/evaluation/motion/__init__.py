"""Motion evaluation metrics for hftrainer.

This module provides physics-based quality metrics for evaluating motion data,
including joint-level metrics (jerk, pop, twist, velocity, bone length) and
vertex-level metrics (penetration, floating, skating).

Main entry point:
    compute_phys_metrics(file_path, ...) -> dict
"""

from hftrainer.evaluation.motion.phys_metrics import (
    compute_phys_metrics,
    load_motion_data,
    PHYS_METRICS_CACHE,
)

__all__ = [
    'compute_phys_metrics',
    'load_motion_data',
    'PHYS_METRICS_CACHE',
]
