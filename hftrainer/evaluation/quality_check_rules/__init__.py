"""Quality check rules for motion data.

These checkers detect motion quality issues (jitter, foot skating,
arm penetration, joint jumps, etc.) and produce per-frame per-joint
invalid masks for targeted repair.
"""
from .motion_quality_checker import MotionQualityChecker
from .mask_utils import empty_invalid_mask, mask_to_sparse_dict, merge_invalid_masks
