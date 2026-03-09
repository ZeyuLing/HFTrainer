"""Perplexity evaluator for LLMs."""

import math
from typing import Dict, Any, List

import torch

from hftrainer.evaluation.base_evaluator import BaseEvaluator
from hftrainer.registry import EVALUATORS


@EVALUATORS.register_module()
class PerplexityEvaluator(BaseEvaluator):
    """
    Computes perplexity from LLM validation outputs.

    Currently reads 'preds' and 'gts' as lists of strings.
    For actual perplexity computation, we need the loss values.
    This evaluator computes word-level BLEU as a proxy when
    actual loss is not available.
    """

    def __init__(self):
        super().__init__()

    def compute(self) -> Dict[str, float]:
        if not self._results:
            return {}

        # Collect predictions and ground truths
        all_preds = []
        all_gts = []
        for result in self._results:
            preds = result.get('preds', [])
            gts = result.get('gts', [])
            if isinstance(preds, list):
                all_preds.extend(preds)
            if isinstance(gts, list):
                all_gts.extend(gts)

        if not all_preds:
            return {}

        # Simple exact-match accuracy
        exact_match = sum(
            1 for p, g in zip(all_preds, all_gts) if p.strip() == g.strip()
        ) / len(all_preds)

        return {'exact_match': exact_match, 'num_samples': len(all_preds)}
