"""Perplexity evaluator for LLMs."""

import math
from typing import Dict

import torch

from hftrainer.evaluation.base_evaluator import BaseEvaluator
from hftrainer.registry import EVALUATORS


@EVALUATORS.register_module()
class PerplexityEvaluator(BaseEvaluator):
    """
    Computes perplexity from LLM validation outputs.

    Expected inputs:
      - 'loss_lm': scalar tensor with the batch LM loss
      - 'preds': generated texts (optional, for exact match)
      - 'gts': reference texts (optional, for exact match)
    """

    def __init__(self):
        super().__init__()

    def compute(self) -> Dict[str, float]:
        if not self._results:
            return {}

        losses = []
        all_preds = []
        all_gts = []
        for result in self._results:
            loss = result.get('loss_lm')
            if isinstance(loss, torch.Tensor):
                losses.append(loss.detach().float().reshape(-1).cpu())
            preds = result.get('preds', [])
            gts = result.get('gts', [])
            if isinstance(preds, list):
                all_preds.extend(preds)
            if isinstance(gts, list):
                all_gts.extend(gts)

        metrics = {}

        if losses:
            mean_loss = torch.cat(losses).mean().item()
            metrics['loss_lm'] = mean_loss
            metrics['perplexity'] = math.exp(mean_loss)

        if all_preds and all_gts:
            exact_match = sum(
                1 for p, g in zip(all_preds, all_gts) if p.strip() == g.strip()
            ) / len(all_preds)
            metrics['exact_match'] = exact_match
            metrics['num_samples'] = len(all_preds)

        return metrics
