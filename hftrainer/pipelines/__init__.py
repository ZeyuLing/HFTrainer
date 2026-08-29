"""Inference pipeline APIs."""

from hftrainer.pipelines.base_pipeline import BasePipeline
from hftrainer.pipelines.builder import build_pipeline_from_cfg

__all__ = ['BasePipeline', 'build_pipeline_from_cfg']
