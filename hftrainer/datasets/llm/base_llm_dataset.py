"""Base LLM dataset with MMEngine-style pipelines."""

from abc import ABC
from typing import Dict, Any, List, Optional, Sequence, Union, Callable

import torch

from hftrainer.datasets.base_dataset import PipelineDataset


class BaseLLMDataset(PipelineDataset, ABC):
    """
    Abstract base class for LLM datasets.

    Subclasses should emit raw text fields and let transforms handle prompt
    formatting / tokenization.
    """

    def __init__(
        self,
        pipeline: Optional[Sequence[Union[dict, Callable]]] = None,
        serialize_data: bool = False,
    ):
        super().__init__(pipeline=pipeline, serialize_data=serialize_data)

    @classmethod
    def collate_fn(cls, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        result = {
            'input_ids': torch.stack([item['input_ids'] for item in batch]),
            'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
            'labels': torch.stack([item['labels'] for item in batch]),
        }
        if 'input_prompts' in batch[0]:
            result['input_prompts'] = [item['input_prompts'] for item in batch]
        if 'output_texts' in batch[0]:
            result['output_texts'] = [item['output_texts'] for item in batch]
        return result
