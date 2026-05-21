"""
Rot6D Alignment Validation Package

Tools for validating rot6d convention consistency in PRISM/VERMO pipelines.

Classes:
  - Rot6DValidator: Low-level rot6d orthonormality checks
  - PrismPipelineValidator: End-to-end pipeline validation
  - Rot6DAlignmentTests: Test suite for alignment verification

Usage:
  from scripts.debug.rot6d_validation import Rot6DValidator, PrismPipelineValidator
"""

from .rot6d_validator import Rot6DValidator, PrismPipelineValidator
from .test_alignment import Rot6DAlignmentTests

__all__ = [
    "Rot6DValidator",
    "PrismPipelineValidator", 
    "Rot6DAlignmentTests",
]

__version__ = "1.0.0"
__date__ = "2026-05-21"
