"""Validation-only mean loss collector for the single-clip smoke experiment."""
import torch
from hftrainer.evaluation.base_evaluator import BaseEvaluator
from hftrainer.registry import EVALUATORS


@EVALUATORS.register_module()
class H3SmokeLossEvaluator(BaseEvaluator):
    def compute(self):
        if not self._results:
            raise ValueError('Validation produced no batches')
        metrics = {}
        for key in ('loss', 'loss_video', 'loss_audio'):
            values = torch.cat([torch.as_tensor(x[key]).detach().float().cpu().reshape(-1) for x in self._results])
            if not torch.isfinite(values).all():
                raise ValueError(f'Nonfinite validation {key}')
            metrics[f'val/{key}'] = values.mean().item()
        return metrics
