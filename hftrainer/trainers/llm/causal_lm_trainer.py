"""CausalLM SFT trainer."""

import torch
from typing import Dict, Any, Optional, List

from hftrainer.trainers.base_trainer import BaseTrainer
from hftrainer.registry import TRAINERS


@TRAINERS.register_module()
class CausalLMTrainer(BaseTrainer):
    """
    Trainer for causal language model SFT (instruction tuning, etc.).

    train_step:
      1. Get input_ids, attention_mask, labels from batch
      2. Run forward_logits → uses HF built-in CE loss
      3. Return loss

    val_step:
      1. Generate completions for input prompts
      2. Return {'preds', 'gts', 'input_prompts'}
    """

    def __init__(
        self,
        bundle,
        val_max_new_tokens: int = 64,
        do_sample: bool = False,
        **kwargs,
    ):
        super().__init__(bundle)
        self.val_max_new_tokens = val_max_new_tokens
        self.do_sample = do_sample

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Args:
            batch: dict with 'input_ids', 'attention_mask', 'labels'

        Returns:
            {'loss': Tensor}
        """
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        outputs = self.bundle.forward_logits(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss

        return {'loss': loss}

    def val_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Args:
            batch: dict with 'input_ids', 'attention_mask', 'labels',
                   optionally 'input_prompts', 'output_texts'

        Returns:
            {
                'preds': list[str],           # generated completions
                'gts': list[str],             # ground truth texts
                'input_prompts': list[str],   # input prompts
            }
        """
        input_prompts = batch.get('input_prompts', [''] * batch['input_ids'].shape[0])
        gt_texts = batch.get('output_texts', [''] * batch['input_ids'].shape[0])

        outputs = self.bundle.forward_logits(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
        )

        # Generate
        preds = self.bundle.generate(
            input_prompts,
            max_new_tokens=self.val_max_new_tokens,
            do_sample=self.do_sample,
        )

        return {
            'preds': preds,
            'gts': gt_texts,
            'input_prompts': input_prompts,
            'loss_lm': outputs.loss.detach(),
        }
