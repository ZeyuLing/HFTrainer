"""Base LLM dataset interface."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List

import torch
from torch.utils.data import Dataset


class BaseLLMDataset(Dataset, ABC):
    """
    Abstract base class for LLM datasets.

    __getitem__ must return:
        {
            'input_ids': Tensor[L],
            'attention_mask': Tensor[L],
            'labels': Tensor[L],  # -100 for masked positions
            'input_prompts': str (optional, for val_step)
            'output_texts': str (optional, for val_step)
        }
    """

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pass

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
