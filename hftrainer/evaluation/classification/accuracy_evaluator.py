"""Accuracy evaluator for classification."""

from typing import Dict, Any, List

import torch

from hftrainer.evaluation.base_evaluator import BaseEvaluator
from hftrainer.registry import EVALUATORS


@EVALUATORS.register_module()
class AccuracyEvaluator(BaseEvaluator):
    """
    Computes top-1 and top-5 accuracy for classification.

    Consumes output dict with keys:
      - 'preds': Tensor[B] — predicted class indices
      - 'scores': Tensor[B, num_classes] — softmax probabilities
      - 'gts': Tensor[B] — ground truth class indices
    """

    def __init__(self, topk: tuple = (1, 5)):
        super().__init__()
        self.topk = topk

    def compute(self) -> Dict[str, float]:
        if not self._results:
            return {}

        all_scores = []
        all_gts = []

        for result in self._results:
            scores = result.get('scores')
            gts = result.get('gts')
            if scores is None or gts is None:
                continue
            if isinstance(scores, torch.Tensor):
                all_scores.append(scores.cpu())
            if isinstance(gts, torch.Tensor):
                all_gts.append(gts.cpu())

        if not all_scores or not all_gts:
            return {}

        all_scores = torch.cat(all_scores, dim=0)  # [N, C]
        all_gts = torch.cat(all_gts, dim=0)         # [N]

        metrics = {}
        maxk = max(self.topk)
        n = all_gts.size(0)

        _, pred = all_scores.topk(min(maxk, all_scores.size(-1)), dim=1, largest=True, sorted=True)
        pred = pred.t()  # [maxk, N]
        correct = pred.eq(all_gts.view(1, -1).expand_as(pred))  # [maxk, N]

        for k in self.topk:
            if k <= all_scores.size(-1):
                correct_k = correct[:k].reshape(-1).float().sum(0)
                metrics[f'top{k}_acc'] = (correct_k / n).item()

        return metrics
